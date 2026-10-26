# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

# [MODULE DOC]
# Defines the local_map_hop higher order operator used by the AutoParallel flow:
# a graph body executed on local (per-shard) shapes while the surrounding graph
# sees global shapes. Contains placement-based redistribution helpers that
# compute local/global fake tensors, a joint-graph forward/backward partitioner
# (create_hop_fw_bw), an autograd Function, and autograd / functionalize / fake /
# proxy / eager implementations. A module-level flag toggled by the defer_inlining
# context manager chooses between inlining the body and proxying the HOP.
# Import-time side effects: creates the singleton local_map_hop and registers it
# with the activation-checkpointing caching dispatch modes via redirect_to_mode.
import contextlib
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, TypeAlias

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    clone_outputs_aliasing_inputs,
    redirect_to_mode,
    register_fake,
    save_values_for_backward,
    saved_values,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, is_fake_tensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


# Proxy the HOP instead of inlining into it
# And trace it with local shapes for AP
_DEFER_INLINING = False

GraphArg: TypeAlias = tuple[torch.Tensor, int, torch.SymInt, None]


@contextmanager
def defer_inlining() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that sets the module-global defer-inlining flag
    # to True for its body (so local_map is proxied and traced with local shapes
    # rather than inlined) and restores the prior value on exit, even on error.
    global _DEFER_INLINING
    prior = _DEFER_INLINING
    try:
        _DEFER_INLINING = True
        yield
    finally:
        _DEFER_INLINING = prior


# Used to unwrap tensors classes like FunctionalTensor and Parameter
def _new_tensor(
    t: Any,
    new_shape: Sequence[int] | None = None,
    new_stride: Sequence[int] | None = None,
) -> Any:
    # [FUNCTION DOC] HELPER: For a tensor, returns a fresh empty strided tensor with
    # the same device, dtype and requires_grad, using the given shape/stride when
    # provided and the original otherwise (this strips wrapper classes like
    # FunctionalTensor or Parameter). Non-tensors are returned unchanged. Raises
    # AssertionError for tensor subclasses other than FunctionalTensor, FakeTensor
    # and plain Tensor.
    if isinstance(t, torch.Tensor):
        if type(t) not in (FunctionalTensor, FakeTensor, torch.Tensor):
            raise AssertionError(f"No subclasses support for now, found {type(t)}")
        return torch.empty_strided(
            t.size() if new_shape is None else new_shape,
            t.stride() if new_stride is None else new_stride,
            device=t.device,
            dtype=t.dtype,
            requires_grad=t.requires_grad,
        )
    return t


# Autoparallel specific, we want to treat plain tensors as DTensors
def _redistribute(
    args: Any,
    all_placements: tuple[Any],
    mesh: Any,
    shape_stride_fn: Callable[[torch.Tensor, Any, Any], tuple[list[int], list[int]]],
) -> GraphArg:
    # [FUNCTION DOC] HELPER: Creates new fake tensors for args with shapes and strides
    # recomputed per entry via shape_stride_fn(tensor, mesh, placements), under the
    # detected fake mode with functionalization and proxy tracing disabled. None
    # entries are left as they are. Raises AssertionError if no fake mode is found
    # or an output is not a fake tensor / int / SymInt / None. Returns a tuple.
    from torch._dispatch.python import suspend_functionalization
    from torch._guards import detect_fake_mode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    with (
        suspend_functionalization(),
        disable_functional_mode(),
        disable_proxy_modes_tracing(),
    ):
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            raise AssertionError("defer_inlining() is only supported for FakeTensors")

        with fake_mode:
            new_args = list(pytree.tree_map(_new_tensor, args))
            for i, (tensor, placements) in enumerate(zip(new_args, all_placements)):
                if tensor is None:
                    # Sometimes gradients can be None
                    continue

                new_shape, new_stride = shape_stride_fn(
                    tensor,
                    mesh,
                    placements,
                )
                new_args[i] = _new_tensor(
                    tensor, new_shape=new_shape, new_stride=new_stride
                )

            new_args = tuple(new_args)
            if not all(
                (is_fake_tensor(t) or isinstance(t, (int, torch.SymInt, type(None))))
                for t in new_args
            ):
                raise AssertionError(f"Unexpected element in {args=}")

    return new_args


def redistribute_fw_inputs(
    global_args: Any, all_placements: Any, mesh: Any, _: int | None = None
) -> GraphArg:
    # [FUNCTION DOC] HELPER: Converts global-shaped forward inputs into local-shaped
    # fakes using the input placements on the mesh. Requires one placement per
    # input (AssertionError otherwise). The last parameter is unused.
    if len(global_args) != len(all_placements):
        raise AssertionError(
            f"global_args length ({len(global_args)}) != all_placements length ({len(all_placements)})"
        )
    return _redistribute(
        global_args,
        all_placements,
        mesh,
        torch.distributed.tensor._utils.compute_local_tensor_info,
    )


def redistribute_fw_outputs(
    local_outs: Any, all_placements: Any, mesh: Any, num_activations: int
) -> GraphArg:
    # [FUNCTION DOC] HELPER: Converts local forward outputs to global-shaped fakes.
    # local_outs holds the forward outputs followed by num_activations saved
    # activations; only the outputs are redistributed (by out placements), the
    # activations are appended unchanged. AssertionError if lengths mismatch or
    # there are no forward outputs.
    if len(local_outs) != len(all_placements) + num_activations:
        raise AssertionError(
            f"local_outs length ({len(local_outs)}) != "
            f"all_placements length ({len(all_placements)}) + num_activations ({num_activations})"
        )
    num_fw_outs = len(local_outs) - num_activations
    if num_fw_outs <= 0:
        raise AssertionError(f"num_fw_outs must be > 0, got {num_fw_outs}")
    outs, activations = local_outs[:num_fw_outs], local_outs[num_fw_outs:]
    return (
        *_redistribute(
            outs,
            all_placements,
            mesh,
            torch.distributed.tensor._utils.compute_global_tensor_info,
        ),
        *activations,
    )


def redistribute_bw_inputs(
    global_args: Any, all_placements: Any, mesh: Any, num_activations: int
) -> GraphArg:
    # [FUNCTION DOC] HELPER: Backward-input counterpart. global_args holds
    # num_activations activations first, then the incoming gradients; the
    # gradients are converted to local shapes by placements and the activations
    # pass through unchanged. AssertionError on length mismatch or no gradients.
    if len(global_args) != len(all_placements) + num_activations:
        raise AssertionError(
            f"global_args length ({len(global_args)}) != "
            f"all_placements length ({len(all_placements)}) + num_activations ({num_activations})"
        )
    activations, inputs = global_args[:num_activations], global_args[num_activations:]
    if len(inputs) <= 0:
        raise AssertionError("inputs must not be empty")
    local_inputs = _redistribute(
        inputs,
        all_placements,
        mesh,
        torch.distributed.tensor._utils.compute_local_tensor_info,
    )
    return (
        *activations,
        *local_inputs,
    )


def redistribute_bw_outputs(
    local_outs: Any, all_placements: Any, mesh: Any, _: int | None = None
) -> GraphArg:
    # [FUNCTION DOC] HELPER: Converts local backward outputs (input gradients) to
    # global-shaped fakes using the placements. One placement per output is
    # required (AssertionError otherwise). The last parameter is unused.
    if len(local_outs) != len(all_placements):
        raise AssertionError(
            f"local_outs length ({len(local_outs)}) != all_placements length ({len(all_placements)})"
        )
    return _redistribute(
        local_outs,
        all_placements,
        mesh,
        torch.distributed.tensor._utils.compute_global_tensor_info,
    )


class LocalMapHOP(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator "local_map_hop": runs a GraphModule body (whose
    # meta carries local_map_kwargs: in/out placements and device mesh) on local
    # shards. Stateless.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the operator under the name "local_map_hop".
        super().__init__("local_map_hop")

    def __call__(self, gm: GraphModule, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Dispatches the body graph module gm with args and kwargs
        # through the HOP dispatcher and returns the outputs.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(gm, *args, **kwargs)


local_map_hop = LocalMapHOP()

# Registers dispatches for SAC
redirect_to_mode(local_map_hop, _CachingTorchDispatchMode)
redirect_to_mode(local_map_hop, _CachedTorchDispatchMode)


def create_hop_fw_bw(
    fw_gm: GraphModule,
    *_args: Any,
) -> tuple[GraphModule, GraphModule, int, int, set[int]]:
    # [FUNCTION DOC] Builds forward and backward graph modules for a local_map body.
    # Reads in_placements, out_placements and device_mesh from fw_gm.meta
    # (AssertionError if absent or in_placements length differs from the args).
    # Under a fake mode (detected, or created if missing) converts args to local
    # shapes, runs fw_gm to get example outputs/gradients, and records indices of
    # outputs that are tensors requiring grad. Traces a joint function (via
    # AOTAutograd create_joint, which rejects unbacked symints) with make_fx,
    # copies forward metadata to backward nodes, partitions it with the Inductor
    # partition function, tags nodes as forward/backward, copies local_map_kwargs
    # into both graph metas (backward in/out placements derived from forward
    # out/in placements), validates input/output/activation counts (AssertionError
    # on any mismatch), and stores num_activations and is_backward in metas.
    # Returns (fw graph, bw graph, number of forward inputs, number of forward
    # outputs, set of output indices that receive gradients).
    """
    Traces a joint, applies passes and partitions it
    """
    # Keeping these imports here
    # Avoid circular dependencies once we upstream with dynamo frontend
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, make_fx

    local_map_kwargs = fw_gm.meta["local_map_kwargs"]  # type: ignore[attr-defined]
    if "in_placements" not in local_map_kwargs:
        raise AssertionError("'in_placements' not found in local_map_kwargs")
    if "out_placements" not in local_map_kwargs:
        raise AssertionError("'out_placements' not found in local_map_kwargs")
    if "device_mesh" not in local_map_kwargs:
        raise AssertionError("'device_mesh' not found in local_map_kwargs")
    if len(local_map_kwargs["in_placements"]) != len(_args):
        raise AssertionError(
            f"in_placements length ({len(local_map_kwargs['in_placements'])}) != _args length ({len(_args)})"
        )

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.

            fake_mode = detect_fake_mode(_args)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                fw_inputs = redistribute_fw_inputs(
                    _args,
                    local_map_kwargs["in_placements"],
                    local_map_kwargs["device_mesh"],
                )
                if len(fw_inputs) != len(local_map_kwargs["in_placements"]):
                    raise AssertionError(
                        f"fw_inputs length ({len(fw_inputs)}) != "
                        f"in_placements length ({len(local_map_kwargs['in_placements'])})"
                    )

            if not all(
                (is_fake_tensor(t) or isinstance(t, (int, torch.SymInt)))
                for t in fw_inputs
            ):
                raise AssertionError(f"Unexpected element in {fw_inputs=}")

            ctx = (
                fake_mode.shape_env.ignore_fresh_unbacked_symbols
                if fake_mode.shape_env is not None
                else contextlib.nullcontext
            )
            with ctx():
                fw_outs = fw_gm(*fw_inputs)

            example_grads = pytree.tree_map(
                _new_tensor,
                fw_outs,
            )
            if not isinstance(example_grads, (list, tuple)):
                example_grads = [example_grads]

            num_fw_inputs = len(fw_inputs)
            num_fw_outputs = len(example_grads)

        def joint_f(
            *primals_and_tangents: list[torch.Tensor],
        ) -> Any:
            # [FUNCTION DOC] HELPER: Joint forward+backward function traced by
            # make_fx. Splits inputs into primals and tangents, runs the body
            # through create_joint, raises AssertionError if free unbacked symbols
            # appear in the outputs or grads, clones outputs that alias inputs,
            # and returns the gradients followed by the forward outputs.
            primals = primals_and_tangents[:num_fw_inputs]
            tangents = primals_and_tangents[num_fw_inputs:]

            def prepare_fw_with_masks(
                fw_gm: torch.fx.GraphModule,
            ) -> Callable[..., Any]:
                # [FUNCTION DOC] HELPER: Returns a function that runs fw_gm and also
                # reports, per output, whether it is a tensor requiring grad.
                def fw_with_masks(*args: Any) -> tuple[tuple[Any], list[bool]]:
                    # [FUNCTION DOC] HELPER: Runs fw_gm via an FX Interpreter (to
                    # carry node metadata through); raises AssertionError if the
                    # result is not a tuple. Returns (outputs, requires-grad masks).
                    # The Interpreter here is required to propagate metadata
                    # from the dynamo graph body to the local_map graph body.
                    # This is required for fx_traceback.annotate for work.
                    fw_out = torch.fx.Interpreter(fw_gm).run(*args)
                    if not isinstance(fw_out, tuple):
                        raise AssertionError(
                            "Dynamo traced submodule should return tuple"
                        )
                    return fw_out, [
                        bool(isinstance(ret, torch.Tensor) and ret.requires_grad)
                        for ret in fw_out
                    ]

                return fw_with_masks

            fw_outs, grads = create_joint(
                prepare_fw_with_masks(fw_gm), aot_config=dummy_aot_config
            )(primals, tangents)
            from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

            if has_free_unbacked_symbols((*fw_outs, *grads)):
                raise AssertionError(
                    "Unbacked symints leaking outside of the joint graph is not yet supported."
                )

            maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)
            # put grads first to work with existing hop utils
            return pytree.tree_map(maybe_clone, (*grads, *fw_outs))

        filtered_grads_idx = set()
        for i, example_grad in enumerate(example_grads):
            # Filter out grads that are None or do not require_grad.
            # The AOTAutograd utils we rely on force this assumption.
            # We must also filter the runtime tangents too.
            if example_grad is not None and (
                isinstance(example_grad, torch.Tensor) and example_grad.requires_grad
            ):
                filtered_grads_idx.add(i)

        primals_and_tangents = [
            *fw_inputs,
            *[example_grads[i] for i in filtered_grads_idx],
        ]
        joint_hop_gm = make_fx(joint_f)(*primals_and_tangents)
        from torch._functorch._aot_autograd.graph_capture import (
            copy_fwd_metadata_to_bw_nodes,
        )

        copy_fwd_metadata_to_bw_nodes(joint_hop_gm)

        from torch._functorch._aot_autograd.graph_compile import prepare_for_partitioner
        from torch._inductor.compile_fx import partition_fn

        # Match partitioner convention
        prepped_joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )
        with disable_proxy_modes_tracing():
            # Also runs joint passes
            new_fw_gm, new_bw_gm = partition_fn(
                prepped_joint_hop_gm,
                [],
                num_fwd_outputs=num_fw_outputs,
                static_lifetime_input_indices=[],
            )

        # Fix tags because min-cut does not respect fw/bw boundary, breaking
        # default partitioner's assumptions.
        for node in new_fw_gm.graph.nodes:
            node.meta["partitioner_tag"] = "is_forward"
            node.meta.pop("autograd_backward", None)
        for node in new_bw_gm.graph.nodes:
            node.meta["partitioner_tag"] = "is_backward"
            node.meta["autograd_backward"] = True

        # Propagate meta onto fw/bw graphs, later will be set on proxied nodes
        new_fw_gm.meta["local_map_kwargs"] = local_map_kwargs
        new_bw_gm.meta["local_map_kwargs"] = {**local_map_kwargs}
        # Okay because Autoparallel assumes same sharding between param and grads
        new_bw_gm.meta["local_map_kwargs"]["in_placements"] = tuple(
            [local_map_kwargs["out_placements"][i] for i in filtered_grads_idx]
        )
        new_bw_gm.meta["local_map_kwargs"]["out_placements"] = local_map_kwargs[
            "in_placements"
        ]

        # Validate Forward
        fw_kwargs = new_fw_gm.meta["local_map_kwargs"]
        expected_fw_inputs = len(fw_kwargs["in_placements"])
        expected_fw_outputs = len(fw_kwargs["out_placements"])
        actual_fw_inputs = len(new_fw_gm.graph.find_nodes(op="placeholder"))
        actual_fw_outputs = num_fw_outputs
        if expected_fw_inputs != actual_fw_inputs:
            raise AssertionError(
                f"expected_fw_inputs ({expected_fw_inputs}) != actual_fw_inputs ({actual_fw_inputs})"
            )
        if expected_fw_outputs != actual_fw_outputs:
            raise AssertionError(
                f"expected_fw_outputs ({expected_fw_outputs}) != actual_fw_outputs ({actual_fw_outputs})"
            )

        # Validate Activations
        if len(new_fw_gm.graph.find_nodes(op="output")) != 1:
            raise AssertionError(
                f"Expected exactly 1 output node, got {len(new_fw_gm.graph.find_nodes(op='output'))}"
            )
        num_activations = (
            len(new_fw_gm.graph.find_nodes(op="output")[0].args[0]) - num_fw_outputs
        )
        # tensors first, then symints
        if num_activations < 0:
            raise AssertionError(f"num_activations must be >= 0, got {num_activations}")

        # Validate Backward
        bw_kwargs = new_bw_gm.meta["local_map_kwargs"]
        expected_bw_inputs = len(bw_kwargs["in_placements"])
        expected_bw_outputs = len(bw_kwargs["out_placements"])
        actual_bw_inputs = (
            len(new_bw_gm.graph.find_nodes(op="placeholder")) - num_activations
        )
        if actual_bw_inputs <= 0:
            raise AssertionError(
                f"actual_bw_inputs must be > 0, got {actual_bw_inputs}"
            )
        if expected_fw_inputs + expected_bw_inputs != len(primals_and_tangents):
            raise AssertionError(
                f"expected_fw_inputs ({expected_fw_inputs}) + expected_bw_inputs ({expected_bw_inputs}) "
                f"!= primals_and_tangents length ({len(primals_and_tangents)})"
            )
        if actual_fw_inputs + actual_bw_inputs != len(primals_and_tangents):
            raise AssertionError(
                f"actual_fw_inputs ({actual_fw_inputs}) + actual_bw_inputs ({actual_bw_inputs}) "
                f"!= primals_and_tangents length ({len(primals_and_tangents)})"
            )
        if len(new_bw_gm.graph.find_nodes(op="output")) != 1:
            raise AssertionError(
                f"Expected exactly 1 bw output node, got {len(new_bw_gm.graph.find_nodes(op='output'))}"
            )
        actual_bw_outputs = len(new_bw_gm.graph.find_nodes(op="output")[0].args[0])
        if expected_bw_inputs != actual_bw_inputs:
            raise AssertionError(
                f"expected_bw_inputs ({expected_bw_inputs}) != actual_bw_inputs ({actual_bw_inputs})"
            )
        if expected_bw_outputs != actual_bw_outputs:
            raise AssertionError(
                f"expected_bw_outputs ({expected_bw_outputs}) != actual_bw_outputs ({actual_bw_outputs})"
            )

        new_fw_gm.meta["num_activations"] = num_activations
        new_fw_gm.meta["is_backward"] = False
        new_bw_gm.meta["num_activations"] = num_activations
        new_bw_gm.meta["is_backward"] = True

        return new_fw_gm, new_bw_gm, num_fw_inputs, num_fw_outputs, filtered_grads_idx


class LocalMapAutogradOp(torch.autograd.Function):
    # [CLASS DOC] Autograd Function for local_map_hop. Forward runs the partitioned
    # forward graph (outputs plus saved activations) below autograd; backward runs
    # the partitioned backward graph on saved activations and incoming grads.
    # State on ctx: bw_gm, num_fw_ins, filtered_grads_idx, expected tangent metadata.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        num_fw_ins: int,
        num_fw_outs: int,
        filtered_grads_idx: set[int],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, ...]:
        # [FUNCTION DOC] Stores backward graph, input count and gradient indices on
        # ctx, runs local_map_hop on fw_gm below autograd, saves the trailing
        # activations for backward, records memory-format expectations of each
        # gradient-receiving output (forced memory-format mode because backward
        # tangents are global-shaped), and returns only the forward outputs.
        from torch._functorch._aot_autograd.schemas import MemoryFormatMeta

        ctx.bw_gm = bw_gm
        ctx.num_fw_ins = num_fw_ins
        ctx.filtered_grads_idx = filtered_grads_idx

        with torch._C._AutoDispatchBelowAutograd():
            fw_outs_with_saved_activations = local_map_hop(fw_gm, *args, **kwargs)

        fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
        saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
        save_values_for_backward(ctx, saved_activations)

        # Force memory_format path (not exact size/stride) because local_map forward
        # operates on local shapes but backward receives global-shaped tangents.
        # TODO(ivankobzarev): Support exact size/stride by converting between local/global shapes.
        ctx.expected_tangent_metadata = {
            i: MemoryFormatMeta.from_tensor(fw_outs[i], force_use_memory_format=True)
            for i in filtered_grads_idx
        }
        return fw_outs

    @staticmethod
    def backward(
        ctx: Any, *_grads: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor | None, ...]:
        # [FUNCTION DOC] Computes input gradients. Requires saved-value positions to
        # be sorted (AssertionError otherwise, i.e. no interleaving of tensors and
        # symints), reverses them so symints come first, selects incoming grads at
        # filtered indices, coerces each to the expected memory format, runs
        # local_map_hop on the backward graph with activations and grads, and
        # raises RuntimeError if the number of gradients differs from the number
        # of forward inputs. Returns None for the five non-tensor forward args
        # followed by the input gradients.
        # KNOWN ISSUE: coercion indexes the filtered grad list by the original
        # output index (keys of expected_tangent_metadata), which is only right
        # when filtered indices are contiguous from zero.
        from torch._functorch._aot_autograd.runtime_wrappers import (
            coerce_to_expected_memory_format,
        )

        if ctx.pos != sorted(ctx.pos):
            raise AssertionError(
                "Interleaving saved tensor activations and symints is not expected from min-cut partitioner."
            )
        ctx.pos = list(reversed(ctx.pos))  # make saved_values return symints first
        saved_activations = saved_values(ctx)
        with torch._C._AutoDispatchBelowAutograd():
            # Filter out grads that are None or do not require_grad.
            # The AOTAutograd utils we rely on force this assumption.
            grads = [_grads[i] for i in ctx.filtered_grads_idx]
            if len(grads) != len(ctx.expected_tangent_metadata):
                raise AssertionError(
                    f"{len(grads)=} vs {len(ctx.expected_tangent_metadata)}"
                )

            for i, meta in ctx.expected_tangent_metadata.items():
                grads[i] = coerce_to_expected_memory_format(grads[i], meta)

            grad_ins = local_map_hop(ctx.bw_gm, *saved_activations, *grads)
            if len(grad_ins) != ctx.num_fw_ins:
                raise RuntimeError(
                    f"Expected {ctx.num_fw_ins} grad_ins, got {len(grad_ins)}"
                )
        return None, None, None, None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> Any:
    # [FUNCTION DOC] Autograd-key implementation. Raises AssertionError if
    # in_grad_placements is set (unsupported). When deferred inlining is on,
    # builds forward/backward graphs with create_hop_fw_bw and applies
    # LocalMapAutogradOp; otherwise simply interprets fw_gm on the args (inlined).
    local_map_kwargs = fw_gm.meta["local_map_kwargs"]  # type: ignore[attr-defined]
    if local_map_kwargs.get("in_grad_placements", None) is not None:
        raise AssertionError("local_map in_grad_placements are not yet supported.")
    if _DEFER_INLINING:
        fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx = create_hop_fw_bw(
            fw_gm, *args
        )
        return LocalMapAutogradOp.apply(
            fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx, *args, **kwargs
        )

    # TODO: get rid of this when we can install as a subgraph
    return torch.fx.Interpreter(fw_gm).run(*args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(
    ctx: Any, gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    # [FUNCTION DOC] Functionalization implementation: requires no kwargs
    # (AssertionError), unwraps tensor args, redispatches local_map_hop to the
    # next key and wraps the outputs.
    if kwargs:
        raise AssertionError(f"kwargs must be empty, got {kwargs}")

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        out = local_map_hop(gm, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@register_fake(local_map_hop, skip_cache=True)
def fake_mode_key(
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> GraphArg:
    # [FUNCTION DOC] Fake-tensor implementation (not cached). Without deferred
    # inlining, just runs gm on the args. Otherwise redistributes the global-shaped
    # args to local shapes (backward or forward variant per gm.meta is_backward,
    # using num_activations), runs gm, and converts the local outputs back to
    # global shapes using out placements. Returns the global outputs.
    if not _DEFER_INLINING:
        return gm(*args, **kwargs)

    # otherwise, we need to convert to local shapes for AP
    # invoke_subgraph runs its body below Autograd, so local_map_hop's Autograd
    # key may not have partitioned the body or set fw/bw metadata.
    is_backward = gm.meta.get("is_backward", False)
    num_activations = gm.meta.get("num_activations", 0)
    redistribute_inputs = (
        redistribute_bw_inputs if is_backward else redistribute_fw_inputs
    )
    local_args = redistribute_inputs(
        args,
        gm.meta["local_map_kwargs"]["in_placements"],
        gm.meta["local_map_kwargs"]["device_mesh"],
        num_activations,
    )
    local_outs = gm(*local_args)
    redistribute_outputs = (
        redistribute_bw_outputs if is_backward else redistribute_fw_outputs
    )
    global_outs = redistribute_outputs(
        local_outs,
        gm.meta["local_map_kwargs"]["out_placements"],
        gm.meta["local_map_kwargs"]["device_mesh"],
        num_activations,
    )
    return global_outs


def _proxy_mode_arg(
    proxy_mode: ProxyTorchDispatchMode,
    arg: Any,
) -> Any:
    # [FUNCTION DOC] HELPER: Converts an argument to its proxy. A GraphModule not
    # already a submodule of the tracer root is first registered there under a
    # fresh name local_map_body so generated code can reference it as an
    # attribute. Returns the unwrapped proxy for the argument.
    if isinstance(arg, torch.fx.GraphModule):
        # Install local_map bodies as real submodules so FX source emission can
        # reload them through self instead of closing over a local Python target.
        registered = any(
            arg is submod
            for _, submod in proxy_mode.tracer.root.named_modules()  # type: ignore[union-attr]
        )
        if not registered:
            qualname = proxy_mode.tracer.get_fresh_qualname("local_map_body")  # type: ignore[union-attr]
            proxy_mode.tracer.root.register_module(qualname, arg)  # type: ignore[union-attr]
    return proxy_mode.tracer.unwrap_proxy(arg)  # type: ignore[union-attr]


def proxy_mode_key_common(
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # [FUNCTION DOC] Shared proxy-tracing logic. Requires an active proxy mode and no
    # kwargs (AssertionError), computes example outputs by calling local_map_hop,
    # records a call_function node for the HOP with proxied args, requires
    # gm.meta to hold non-empty local_map_kwargs (AssertionError otherwise) and
    # copies them onto the new node's meta. Returns the example outputs tracked to
    # the new proxy. Side effects: may register submodules and adds a graph node.
    if proxy_mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    if len(kwargs) != 0:
        raise AssertionError(f"kwargs must be empty, got {kwargs}")

    example_out = local_map_hop(gm, *args, **kwargs)
    proxy_args = pytree.tree_map(
        lambda arg: _proxy_mode_arg(proxy_mode, arg),
        (gm, *args),
    )

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", local_map_hop, proxy_args, {}
    )

    # extract local_map args, post-dispatch operates on GraphModules
    if not gm.meta["local_map_kwargs"]:
        raise AssertionError("gm.meta['local_map_kwargs'] must be set")
    local_map_kwargs = gm.meta["local_map_kwargs"]

    # propagate local_map args to the call_function node
    out_proxy.node.meta["local_map_kwargs"] = local_map_kwargs

    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@local_map_hop.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # [FUNCTION DOC] HELPER: ProxyTorchDispatchMode implementation; delegates to
    # proxy_mode_key_common.
    return proxy_mode_key_common(proxy_mode, gm, *args, **kwargs)


# Running HOP in eager with real tensors
@local_map_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def real_impl(
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Eager implementation with real tensors; calls the body
    # graph module directly and returns its outputs.
    return gm(*args, **kwargs)
