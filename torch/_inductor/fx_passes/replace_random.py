# mypy: allow-untyped-defs
# [MODULE DOC] Inductor FX pass replacing aten random ops (rand, randn, randint) with inductor_prims random ops
# driven by explicit seeds, and fusing the per-device seed/offset creation nodes into single nodes.
# Registers graph patterns into a module-level PatternMatcherPass (joint_graph_passes) at import time.
import collections
import logging

import torch
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import _extract_tensor_metadata

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunctionVarArgs,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from ..virtualized import V


log = logging.getLogger(__name__)
patterns = PatternMatcherPass(subsystem="joint_graph_passes")
aten = torch.ops.aten


def _shape_to_offset(shape, device: torch.device):
    # [FUNCTION DOC] HELPER: Computes the Philox RNG offset consumed by generating a tensor of the given shape on device.
    # Returns 0 for empty tensors (statically or via guard_or_false) and for non-cuda devices; device may be None
    # (cpu), a string, or a torch.device. For cuda it mirrors the philox offset formula: grid size is clamped
    # between 1 and the max resident grid from device properties, then offset is a ceil division times 4.
    # Queries CUDA device properties; may return a symbolic value for symbolic shapes.
    # Modified from torch/_prims/rng_prims.py:philox_rand_offset
    nelem = 1
    for s in shape:
        nelem *= s

    # Empty tensor: no random numbers are generated/consumed.
    is_empty = nelem == 0
    if statically_known_true(is_empty) or guard_or_false(is_empty):
        return 0

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        return 0

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4

    device_property = torch.cuda.get_device_properties(device)

    blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
    max_grid = device_property.multi_processor_count * blocks_per_sm
    grid_size = (nelem + block_size - 1) // block_size
    grid_size = -torch.sym_min(-grid_size, -1)
    grid_size = torch.sym_min(grid_size, max_grid)

    return ((nelem - 1) // (block_size * grid_size * unroll) + 1) * curand4_engine_calls


def replace_random_passes(gm: torch.fx.GraphModule):
    # [FUNCTION DOC] Entry point: runs the random-replacement patterns, then fuses seed creation, and (if
    # config.align_random_eager) fuses eager-offset creation, each under a GraphTransformObserver.
    # Returns total number of rewrites, or 0 immediately if config.fallback_random is set. Mutates gm.
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    count = patterns.apply(gm)
    with GraphTransformObserver(gm, "fuse_seed_creation_pass", "joint_graph_passes"):
        count += fuse_seed_creation_pass(gm.graph)
    if config.align_random_eager:
        with GraphTransformObserver(gm, "fuse_offset_creation_pass"):
            count += fuse_offset_creation_pass(gm.graph)

    return count


def fuse_offset_creation_pass(graph: torch.fx.Graph) -> int:
    # [FUNCTION DOC] Groups all rand_eager_offset call nodes by device (second arg) and replaces each group with one
    # rand_eager_offsets node placed before the first of the group, plus a select node per original offset.
    # The combined node gets fake-tensor metadata (int64, shape count x 2) and the original uses are rewired
    # and old nodes erased. Mutates graph in place; returns the number of devices fused (0 if none).
    """
    Here offset node means seed << 32 + offset, will unpacked in lowering.py:inductor_random()
    Horizontally fuse all the seed generation on each device
        a = inductor_prims.rand_eager_offset(offset, dev)
        b = inductor_prims.rand_eager_offset(offset, dev)
    Becomes:
        offsets = inductor_prims.rand_eager_offsets([offset1, offset2...], dev)
        a = torch.ops.aten.select.int(offsets, 0, 0)
        b = torch.ops.aten.select.int(offsets, 0, 1)
    We do this because seed creation is entirely launch overhead bound.
    """
    device_offsets = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.rand_eager_offset).match(node):
            device_offsets[node.args[1]].append(node)

    if not device_offsets:
        return 0

    for device, offsets in device_offsets.items():
        with graph.inserting_before(offsets[0]):
            offs = [n.args[0] for n in offsets]
            combined = graph.call_function(
                inductor_prims.rand_eager_offsets, (offs, device)
            )
            combined.meta.update(offsets[0].meta)
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(offsets), 2], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, offset in enumerate(offsets):
            with graph.inserting_before(offset):
                new_state = graph.call_function(
                    torch.ops.aten.select.int, (combined, 0, idx)
                )
            offset.replace_all_uses_with(new_state)
            new_state.meta.update(offset.meta)
            graph.erase_node(offset)

    return len(device_offsets)


def fuse_seed_creation_pass(graph: torch.fx.Graph):
    # [FUNCTION DOC] Groups all inductor_prims.seed nodes by device (first arg) and replaces each group with a single
    # seeds node (count, device) placed before the first one, plus a lookup_seed node per original seed.
    # Combined node gets fake metadata (int64 vector of length count); old seed nodes are replaced and erased.
    # Mutates graph in place; returns number of devices fused (0 if none). Exists because seed creation is launch bound.
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)

    if not device_seeds:
        return 0

    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))
            combined.meta.update(seeds[0].meta)
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(seeds)], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )
            seed.replace_all_uses_with(new_seed)
            new_seed.meta.update(seed.meta)
            graph.erase_node(seed)

    return len(device_seeds)


def default_kwargs(device):
    # [FUNCTION DOC] HELPER: Returns extra keyword arguments for inductor_prims.random for a device; currently always empty.
    return {}


def get_device(device):
    # [FUNCTION DOC] HELPER: Returns the given device if not None, else the device of a default-created tensor (the current default device).
    if device is not None:
        return device
    return torch.empty([]).device  # default device


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.generator), pass_dict=patterns)
def replace_random(
    match: Match,
    size,
    *,
    generator=None,
    dtype=None,
    device=None,
    layout=None,
    pin_memory=None,
):
    # [FUNCTION DOC] Pattern handler for aten rand/randn (default and generator overloads). Skips (no change) if a generator
    # is given or pin_memory is set. Otherwise replaces the call with inductor_prims.random using a fresh seed node
    # and mode 'rand' or 'randn', casting to dtype if given. With config.align_random_eager on cuda and mode rand,
    # uses an eager-offset node (from _shape_to_offset) instead of a seed, passing align_dtype.
    # Replacement is done via match.replace_by_example; layout is ignored.
    if generator is not None:
        return
    if pin_memory:
        return

    def replacement(size):
        # [FUNCTION DOC] HELPER: Replacement body traced by replace_by_example: builds the random tensor from a seed, mode and size, casting to dtype if set.
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        if dtype is not None:
            result = result.to(dtype)
        return result

    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]
    device = get_device(device)
    replacement_fn = replacement

    if mode == "rand" and config.align_random_eager and device.type == "cuda":
        # Only enable when align_random_eager is on.
        def replacement_align(size):
            # [FUNCTION DOC] HELPER: Replacement body for eager-aligned rand on cuda: computes the philox offset for the size, creates an
            # eager-offset node, calls inductor_prims.random with align_dtype (first element if dtype is a sequence), then casts to dtype if set.
            offset = _shape_to_offset(size, device)

            align_dtype = dtype
            if isinstance(align_dtype, (tuple, list)):
                align_dtype = align_dtype[0] if len(align_dtype) else None

            result = inductor_prims.random(
                size,
                inductor_prims.rand_eager_offset(offset, device),
                mode,
                **default_kwargs(device),
                align_dtype=align_dtype,
            )
            if dtype is not None:
                result = result.to(dtype)
            return result

        replacement_fn = replacement_align

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement_fn, [size])


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(
    match: Match,
    low,
    high,
    size,
    *,
    dtype=torch.int64,
    device=None,
    layout=None,
    pin_memory=None,
):
    # [FUNCTION DOC] Pattern handler for aten.randint.low. Skips if pin_memory is set; otherwise replaces the call with
    # inductor_prims.randint(low, high, size, seed) cast to dtype (default int64), resolving device via get_device.
    # layout is ignored.
    if pin_memory:
        return

    def replacement(low, high, size):
        # [FUNCTION DOC] HELPER: Replacement body for randint: builds the integer random tensor from low, high, size and a fresh seed, then casts to dtype.
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        return result.to(dtype)

    device = get_device(device)
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [low, high, size])
