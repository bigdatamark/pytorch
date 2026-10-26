# mypy: allow-untyped-defs
# [MODULE DOC] Pre-grad FX graph passes for Inductor, run on the Torch IR (or predispatch ATen IR) graph before autograd.
# Defines PatternMatcherPass instances (gumbel max trick, efficient conv/bn eval, split/linear/add, chunk/squeeze/cat, reshape
# removal, and predispatch ATen variants), no-op stubs for passes that only exist in internal builds, the pass driver
# pre_grad_passes, conv/bn fusion, and permute/linear/matmul/cat fusion helpers.
# Import-time side effects: creates the module-level pass objects and a logger; no graph is modified until a function is called.
import copy
import functools
import itertools
import logging
import types
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch._dynamo.utils import counters, detect_fake_mode
from torch._logging import trace_structured
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.passes.graph_transform_observer import (
    GraphTransformObserver as GraphTransformObserverBase,
)
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..custom_graph_pass import get_custom_graph_passes
from ..fx_utils import matches_module_function_pattern
from ..pattern_matcher import (
    init_once_fakemode,
    PatternMatcherPass as PatternMatcherPassBase,
    stable_topological_sort,
)
from ..utils import is_cpu_device, pass_execution_and_save
from .group_batch_fusion import group_batch_fusion_passes, PRE_GRAD_FUSIONS
from .misc_patterns import numpy_compat_normalization
from .split_cat import PRE_GRAD_PATTERNS


PatternMatcherPass = functools.partial(
    PatternMatcherPassBase, subsystem="pre_grad_passes"
)
GraphTransformObserver = functools.partial(
    GraphTransformObserverBase, subsystem="pre_grad_passes"
)

log = logging.getLogger(__name__)

apply_gumbel_max_trick_pass = PatternMatcherPass(
    pass_name="apply_gumbel_max_trick_pass"
)

efficient_conv_bn_eval_pass = PatternMatcherPass(
    pass_name="efficient_conv_bn_eval_pass"
)

fuse_split_linear_add_pass = PatternMatcherPass(
    pass_name="fuse_split_linear_add_pass",
)
fuse_chunk_squeeze_cat_pass = PatternMatcherPass(
    pass_name="fuse_chunk_squeeze_cat_pass",
)
remove_reshape_pass = PatternMatcherPass(
    pass_name="remove_reshape_pass",
)

# based on predispatch aten IR
normalization_pass_aten = PatternMatcherPass(pass_name="normalization_pass_aten")
merge_splits_pass_aten = PatternMatcherPass(pass_name="merge_splits_pass_aten")
split_cat_pass_aten = PatternMatcherPass(pass_name="split_cat_pass_aten")
unbind_stack_pass_aten = PatternMatcherPass(pass_name="unbind_stack_pass_aten")
merge_getitem_cat_pass_aten = PatternMatcherPass(
    pass_name="merge_getitem_cat_pass_aten"
)
merge_stack_tahn_unbind_pass_aten = PatternMatcherPass(
    pass_name="merge_stack_tahn_unbind_pass_aten"
)
mutate_cat_pass_aten = PatternMatcherPass(pass_name="mutate_cat_pass_aten")
remove_split_with_size_one_pass_aten = PatternMatcherPass(
    pass_name="remove_split_with_size_one_pass_aten"
)


def save_inductor_dict(pass_to_compare=None):
    # [FUNCTION DOC] HELPER: Snapshot Inductor pass counters. Takes an optional list of pass names (default: all pre-grad and post-grad
    # fusion option names from config) and returns a mapping from each name to its current count in the global inductor counters (0 if absent).
    if not pass_to_compare:
        pass_to_compare = list(config.pre_grad_fusion_options.keys()) + list(
            config.post_grad_fusion_options.keys()
        )
    return {p: dict(counters["inductor"]).get(p, 0) for p in pass_to_compare}


def is_same_dict(inductor_dict, optimus_dict):
    # [FUNCTION DOC] HELPER: Compare counters to a baseline. Returns False if any pass in optimus_dict has a count in inductor_dict that differs
    # from the baseline count (missing counts treated as 0); otherwise True.
    for pass_name, count in optimus_dict.items():
        if count != dict(inductor_dict).get(pass_name, 0):
            return False
    return True


def shape_prop(mod) -> None:
    # [FUNCTION DOC] HELPER: No-op placeholder for shape propagation; accepts a module and returns None.
    return None


def normalize_node_kwargs_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the normalize_node_kwargs_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def fuse_parallel_linear_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the fuse_parallel_linear_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def remove_split_ops(graph, shape_prop):
    # [FUNCTION DOC] HELPER: No-op placeholder for the split-removal helper; takes a graph and a shape propagation callable, returns None.
    return None


def remove_split_ops_pass(graph):
    # [FUNCTION DOC] HELPER: Run the (stub) split-removal helper on the graph's owning module using the module-level stub shape_prop. Returns None.
    remove_split_ops(graph.owning_module, shape_prop)


def fuse_chunk_reshape_unsqueeze_concat_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the fuse_chunk_reshape_unsqueeze_concat_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def fuse_chunk_reshape_concat_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the fuse_chunk_reshape_concat_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def remove_noop_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the remove_noop_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def stack_to_unsqueeze_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the stack_to_unsqueeze_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def merge_concats_pass(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the merge_concats_pass pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def relu_nan_to_num(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the relu_nan_to_num pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def fuse_split_getitem_squeeze_cat(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the fuse_split_getitem_squeeze_cat pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def use_triton_dot_compress(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the use_triton_dot_compress pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def use_triton_lce_replace_simple_LCE_helper(gm, shape_prop):
    # [FUNCTION DOC] HELPER: No-op placeholder taking a graph module and shape propagation callable; returns None.
    return None


def use_triton_lce_replace_simple_LCE(graph):
    # [FUNCTION DOC] HELPER: Pass wrapper: calls the simple-LCE helper with the graph's owning module and the module-level stub shape_prop, and returns its result (None).
    return use_triton_lce_replace_simple_LCE_helper(graph.owning_module, shape_prop)


def use_triton_lce_replace_normal_LCE_helper(gm, shape_prop):
    # [FUNCTION DOC] HELPER: No-op placeholder taking a graph module and shape propagation callable; returns None.
    return None


def use_triton_lce_replace_normal_LCE(graph):
    # [FUNCTION DOC] HELPER: Pass wrapper for the normal-LCE replacement.
    # KNOWN ISSUE: it calls the simple-LCE helper instead of the normal-LCE helper defined just above (likely copy-paste bug).
    return use_triton_lce_replace_simple_LCE_helper(graph.owning_module, shape_prop)


def use_matmul_lce_replace_normal_LCE(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the use_matmul_lce_replace_normal_LCE pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


def use_matmul_fuse_lce_replace_first_LCE(graph):
    # [FUNCTION DOC] HELPER: No-op placeholder for the use_matmul_fuse_lce_replace_first_LCE pass (real implementation exists only in internal builds); takes a graph, does nothing, returns None.
    return None


@init_once_fakemode
def lazy_init(input_device: torch.device | None = None):
    # [FUNCTION DOC] Once-only (and fake-mode) lazy initialization: imports the modules that register pre-grad patterns
    # (apply_gumbel_max_trick, efficient_conv_bn_eval, split_cat) and, in fbcode builds, the internal fb module.
    # Takes an optional input device (unused). Side effect: pattern registration on import.
    from . import (  # noqa: F401  # noqa: F401
        apply_gumbel_max_trick,
        efficient_conv_bn_eval,
        split_cat,
    )

    if config.is_fbcode():
        from . import fb  # type: ignore[attr-defined]  # noqa: F401


def _get_pass_name_func(p):
    # [FUNCTION DOC] HELPER: Map a pass entry to a (name, callable) pair. For a PatternMatcherPass it uses the pass_name and its apply method;
    # for a plain function it uses the function name with leading underscores stripped; otherwise returns (None, None).
    if isinstance(p, PatternMatcherPassBase):
        pass_name = p.pass_name
        pass_func = p.apply
    elif isinstance(p, types.FunctionType):
        pass_name = p.__name__.lstrip("_")
        pass_func = p
    else:
        pass_name = None
        pass_func = None

    return pass_name, pass_func


def _run_pre_dispatch_passes(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object] = (),
    add_passes: str | None = None,
    remove_passes: str | None = None,
) -> None:
    # [FUNCTION DOC] Run the predispatch-IR pre-grad pass pipeline in place on gm. Builds an ordered default pass list (normalization first,
    # then fusions, concat/split/reshape cleanups) and an extended list of optional passes. add_passes and remove_passes are
    # comma separated pass-name strings: default passes named in remove_passes are skipped, and optional passes named in add_passes
    # are run. Each pass is executed via pass_execution_and_save. Finally remove_noop runs again unless removed, and shape
    # propagation is run over gm using the fake mode detected from example_inputs. Returns None; mutates gm; logs the options.
    # Wrapper that resolves pre_grad_fusion_options by device before
    # calling group_batch_fusion_passes, so the predispatch path
    # gets the same device filtering as the non-predispatch path.
    def _group_batch_fusion_passes_wrapper(graph: torch.fx.Graph) -> None:
        # [FUNCTION DOC] HELPER: Run group batch fusion on the graph with fusion options resolved by the example inputs' device types, so the predispatch
        # path filters by device like the non-predispatch path. Returns None.
        group_batch_fusion_passes(
            graph,
            pre_grad=True,
            fusion_options=_resolve_pre_grad_fusion_options(
                config.pre_grad_fusion_options, example_inputs
            ),
        )

    _group_batch_fusion_passes_wrapper.__name__ = "group_batch_fusion_passes"

    # order matters
    default_pass_list = [
        # normalize passes, must be called as the first passes
        normalization_pass_aten,
        normalize_node_kwargs_pass,
        remove_noop_pass,
        relu_nan_to_num,
        fuse_chunk_reshape_concat_pass,
        _group_batch_fusion_passes_wrapper,
        normalize_node_kwargs_pass,
        fuse_chunk_squeeze_cat_pass,
        merge_concats_pass,
        fuse_split_linear_add_pass,
        remove_reshape_pass,
        fuse_parallel_linear_pass,
        remove_split_ops_pass,
        stack_to_unsqueeze_pass,  # run before fuse_chunk_reshape_unsqueeze_concat_pass
        fuse_chunk_reshape_unsqueeze_concat_pass,
    ]

    full_pass_list = default_pass_list + [
        fuse_split_getitem_squeeze_cat,
        use_triton_dot_compress,
        use_triton_lce_replace_simple_LCE,
        use_triton_lce_replace_normal_LCE,
        use_matmul_fuse_lce_replace_first_LCE,
        use_matmul_lce_replace_normal_LCE,
    ]

    log.info(
        f"pre_grad_passes: add_passes: {add_passes}, remove_pass: {remove_passes}"  # noqa: G004
    )
    add_passes_list = []
    remove_passes_list = []
    if add_passes:
        add_passes_list = add_passes.split(",")
    if remove_passes:
        remove_passes_list = remove_passes.split(",")

    shape_prop = lambda mod: ShapeProp(  # noqa: E731
        gm=mod,
        # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
        fake_mode=detect_fake_mode(example_inputs),
    ).propagate(*tuple(example_inputs))

    for p in default_pass_list:
        pass_name, pass_func = _get_pass_name_func(p)
        # should not happen
        if pass_name is None or pass_func is None:
            continue
        if pass_name in remove_passes_list:
            continue
        pass_execution_and_save(
            pass_func,
            gm,
            example_inputs,
            f"[Pre grad(predispatch IR)] Apply {pass_name} pass",
        )

    for p in full_pass_list:
        pass_name, pass_func = _get_pass_name_func(p)
        if pass_name is None or pass_func is None:
            continue
        if pass_name in add_passes_list:
            pass_execution_and_save(
                pass_func,
                gm,
                example_inputs,
                f"[Pre grad(predispatch IR)] Apply {pass_name} pass",
            )

    if "remove_noop" not in remove_passes_list:
        # Remove noops at the end, which may be generated by other passes.
        pass_execution_and_save(
            remove_noop_pass,
            gm,
            example_inputs,
            "[Pre grad(predispatch IR)]Apply remove_noop pass",
        )
    shape_prop(gm)


def _resolve_pre_grad_fusion_options(
    fusion_options: dict[str, dict[str, object]],
    example_inputs: Sequence[object] | None,
) -> dict[str, dict[str, object]]:
    # [FUNCTION DOC] Return a copy of the fusion options filtered by device. An option with a devices entry is kept only if one of the
    # example input tensors has a device type listed; options without devices always pass. The devices key is stripped from
    # every returned option. If example_inputs is None nothing is filtered (only devices keys are stripped). Does not mutate the input.
    """Filter pre_grad_fusion_options by device compatibility.

    If an option has a "devices" key, only include it when at least one
    example input tensor's device type matches. Fusions without a "devices"
    key always pass through (user explicitly opted in). The "devices" key
    is stripped from the returned options.
    """
    if example_inputs is None:
        resolved: dict[str, dict[str, object]] = {}
        for pass_name, options in fusion_options.items():
            resolved[pass_name] = {k: v for k, v in options.items() if k != "devices"}
        return resolved

    example_device_types = OrderedSet(
        t.device.type for t in example_inputs if isinstance(t, torch.Tensor)
    )

    resolved = {}
    for pass_name, options in fusion_options.items():
        devices = options.get("devices")
        if (
            devices is not None
            and isinstance(devices, (list, tuple, OrderedSet))
            and example_device_types.isdisjoint(OrderedSet(devices))
        ):
            continue
        resolved[pass_name] = {k: v for k, v in options.items() if k != "devices"}
    return resolved


def pre_grad_passes(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[object] = (),
    add_passes: str | None = None,
    remove_passes: str | None = None,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Entry point for pre-grad passes. Applies pattern matcher based passes to gm and returns the (possibly replaced) module.
    # Only when config.pattern_matcher is set: lazily initializes patterns; optionally keeps a copy for a numeric check; runs the
    # predispatch pipeline if config.is_predispatch, otherwise numpy-compat normalization, fuse_fx, device-filtered normalization,
    # group/batch fusions, per-option pattern passes (repeated by the counter option, with a structured trace when counters change),
    # conv/bn eval and gumbel max passes. Always afterwards runs the configured custom pre-grad passes, topologically sorts,
    # lifts quantization ops, lints and recompiles gm, and optionally runs the numeric check comparing before and after.
    # Raises an error if a counter option is not an integer. Side effects: mutates the graph, updates global counters, emits traces.
    """
    Apply passes on the input FX graph using Torch IR.

    WARNING:
    The IR before grad is not functional or normalized, so it is harder
    to write passes on this IR.  Passes must be safe with respect to
    aliasing and mutation and need to handle all possible arg schemas.

    Consider adding a new pass to post_grad.py or joint_graph.py which
    are after functionalization and normalization.
    """
    if config.pattern_matcher:
        lazy_init()
        if hasattr(
            config, "fx_passes_numeric_check"
        ) and config.fx_passes_numeric_check.get("pre_grad", False):
            gm_before_fx_passes = gm.__copy__()
        # explicitly run with predispatch atenIR based passes
        if config.is_predispatch:
            _run_pre_dispatch_passes(gm, example_inputs, add_passes, remove_passes)
        else:
            # We only log the graph with changes to avoid the excessive compilation time
            # https://fb.workplace.com/groups/257735836456307/permalink/633533465543207/
            numpy_compat_normalization(gm.graph)
            if example_inputs is not None:
                gm = fuse_fx(gm, example_inputs)
            # Resolve pre_grad_fusion_options by filtering on device type
            # declared in each fusion's "devices" config key. Fusions without
            # a "devices" key always pass through. Uses a local copy to avoid
            # mutating global config.
            fusion_options = _resolve_pre_grad_fusion_options(
                config.pre_grad_fusion_options, example_inputs
            )
            # We should always do the normalization_pass first
            if "normalization_pass" in fusion_options:
                pattern_matcher_pass = PRE_GRAD_PATTERNS["normalization_pass"]
                pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
            GraphTransformObserver(gm, "group_batch_fusion_passes").apply_graph_pass(
                lambda graph: group_batch_fusion_passes(
                    graph, pre_grad=True, fusion_options=fusion_options
                )
            )
            for pass_name in fusion_options:
                # skip all patterns for group batch fusions
                if pass_name in PRE_GRAD_FUSIONS or pass_name == "normalization_pass":
                    continue
                # skip auto-enabled fusion options that are not registered
                # as pattern matcher passes (e.g., batch_linear_lhs when
                # PRE_GRAD_FUSIONS is mocked by tests)
                if pass_name not in PRE_GRAD_PATTERNS:
                    continue
                pattern_matcher_pass = PRE_GRAD_PATTERNS[pass_name]
                inductor_before_change = save_inductor_dict(
                    [pattern_matcher_pass.pass_name]
                )
                # we support run same pattern multiple times, the default is to run only once
                counter_raw = fusion_options[pass_name].get("counter", 1)
                if not isinstance(counter_raw, int):
                    raise AssertionError(f"expected int, got {type(counter_raw)}")
                for _ in range(counter_raw):
                    pattern_matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
                if not is_same_dict(counters["inductor"], inductor_before_change):
                    trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": f"{pattern_matcher_pass.pass_name}_pre_grad",
                            "encoding": "string",
                        },
                        payload_fn=lambda: gm.print_readable(
                            print_output=False, include_stride=True, include_device=True
                        ),
                    )
            # TODO: move efficient_conv_bn_eval_pass to the fusions dict too.
            GraphTransformObserver(gm, "efficient_conv_bn_eval_pass").apply_graph_pass(
                efficient_conv_bn_eval_pass.apply
            )
            GraphTransformObserver(gm, "apply_gumbel_max_trick_pass").apply_graph_pass(
                apply_gumbel_max_trick_pass.apply
            )

    for pre_grad_custom_pass in get_custom_graph_passes(config.pre_grad_custom_pass):
        GraphTransformObserver(gm, "pre_grad_custom_pass").apply_graph_pass(
            pre_grad_custom_pass
        )

    stable_topological_sort(gm.graph)

    from .quantization import quant_lift_up

    quant_lift_up(gm)

    gm.graph.lint()
    gm.recompile()

    if (
        config.pattern_matcher
        and hasattr(config, "fx_passes_numeric_check")
        and config.fx_passes_numeric_check.get("pre_grad", False)
        and example_inputs is not None
    ):
        from .numeric_utils import numeric_check_if_enabled

        gm_after_fx_passes = gm.__copy__()
        numeric_check_if_enabled(
            gm_before_fx_passes,  # type: ignore[possibly-undefined]
            gm_after_fx_passes,
            example_inputs,
            config.fx_passes_numeric_check.get("num_iterations", 1),
            config.fx_passes_numeric_check.get("precision", 1e-4),
        )

    return gm


def fuse_fx(gm: torch.fx.GraphModule, example_inputs) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Apply inference-oriented module-level fusions. Sinks cat after pointwise ops; for non-CPU inputs with permute_fusion
    # enabled, runs shape propagation and the linear/permute and matmul/permute fusions. Returns early if autograd is enabled or
    # inputs are not on CPU; otherwise, when config.freezing is set, removes identity modules and fuses conv with batch norm.
    # Returns the resulting graph module.
    is_cpu = is_cpu_device(example_inputs)
    # pyre-fixme[16]: Module `torch._dynamo.utils` has no attribute `detect_fake_mode`
    fake_mode = detect_fake_mode(example_inputs)

    gm = sink_cat_after_pointwise(gm)
    if config.permute_fusion and not is_cpu:
        # For linear permute fusion, we need to check input info to identify
        # and perform proper permutation/transpose
        ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        with GraphTransformObserver(gm, "linear_permute_fusion"):
            gm = linear_permute_fusion(gm)
        with GraphTransformObserver(gm, "permute_linear_fusion"):
            gm = permute_linear_fusion(gm)
        with GraphTransformObserver(gm, "permute_matmul_fusion"):
            gm = permute_matmul_fusion(gm)

    # make sure the autograd is disabled.
    if torch.is_grad_enabled() or not is_cpu:
        return gm
    if config.freezing:
        with GraphTransformObserver(gm, "remove_identity"):
            gm = remove_identity(gm)
        with GraphTransformObserver(gm, "fuse_conv_bn"):
            gm = fuse_conv_bn(gm)
    return gm


def fetch_attr(target: str, mod):
    # [FUNCTION DOC] HELPER: Resolve a dotted attribute path on mod and return the attribute. Raises RuntimeError naming the prefix path
    # if any component does not exist.
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def remove_identity(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Remove nn.Identity call_module nodes from gm by redirecting their users to the input and erasing the node.
    # Raises an error if an identity node does not have exactly one argument. Lints and recompiles only if something changed. Returns gm.
    """
    Removes all identity layers from the module.
    """
    graph = gm.graph
    work_done = False
    for module_name, module in gm.named_modules():
        if type(module) is nn.Identity:
            for node in list(graph.find_nodes(op="call_module", target=module_name)):
                if len(node.args) != 1:
                    raise AssertionError(
                        f"expected 1 arg for identity node, got {len(node.args)}"
                    )
                input_node = node.args[0]
                node.replace_all_uses_with(input_node)
                graph.erase_node(node)
                work_done = True
    if work_done:
        graph.lint()
        gm.recompile()
    return gm


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Fuse convolution with batch norm for inference, in place on gm (inplace argument is unused). Pass 1 handles
    # Conv1d/2d/3d followed by the matching BatchNorm module; pass 2 handles conv modules followed by functional batch_norm with
    # constant get_attr stats (only 8 positional args, float eps, eval mode). Candidates are grouped by conv module name; a conv
    # whose output has other users, in training mode, or shared across different batch norms is skipped/disabled. Fused conv is
    # created via fuse_conv_bn_eval or by deep-copying the conv and fusing weights; the bn nodes are erased. Lints, recompiles, returns gm.
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    modules_patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    module_function_patterns = [
        (torch.nn.Conv1d, F.batch_norm),
        (torch.nn.Conv2d, F.batch_norm),
        (torch.nn.Conv3d, F.batch_norm),
    ]
    modules = dict(gm.named_modules())

    class ConvBNFusion:
        # [CLASS DOC] HELPER: Record of one conv module that is candidate for folding with its batch norm node(s).
        # Holds the list of bn nodes, the conv module, either a bn module or functional bn constants (mean, var, eps, weight, bias),
        # and a fusion_enabled flag that starts True and is cleared when the conv is shared by different batch norms.
        def __init__(
            self,
            bn_node,
            conv_module,
            bn_module=None,  # For BN Module
            bn_running_mean=None,  # For Functional BN
            bn_running_var=None,
            bn_eps=None,
            bn_weight=None,
            bn_bias=None,
        ) -> None:
            # [FUNCTION DOC] HELPER: Store the first bn node (as a one-item list), conv module, bn module or functional bn parameters; enable fusion.
            self.bn_nodes = [
                bn_node,
            ]
            self.conv_module = conv_module
            self.bn_module = bn_module
            self.bn_running_mean = bn_running_mean
            self.bn_running_var = bn_running_var
            self.bn_eps = bn_eps
            self.bn_weight = bn_weight
            self.bn_bias = bn_bias
            self.fusion_enabled = True

        def add_bn_node(self, bn_node):
            # [FUNCTION DOC] HELPER: Append another bn node that uses the same conv and same bn.
            self.bn_nodes.append(bn_node)

        def disable_fusion(self):
            # [FUNCTION DOC] HELPER: Mark this conv/bn group as not fusable.
            self.fusion_enabled = False

        def is_fusion_enabled(self):
            # [FUNCTION DOC] HELPER: Return True if fusion is still enabled for this group.
            return self.fusion_enabled

    conv_bn_to_fuse: dict[int, ConvBNFusion] = {}
    for pattern in modules_patterns:
        conv_bn_to_fuse.clear()
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                if not bn.track_running_stats:
                    continue

                # Do hash based on the module name of conv
                hash_id = hash(node.args[0].target)
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(node, conv, bn)
                else:
                    if bn == conv_bn_to_fuse[hash_id].bn_module:
                        # Do fusion if same bn module
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # Disable the conv bn folding if conv shared by different bn
                        conv_bn_to_fuse[hash_id].disable_fusion()

        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn = conv_bn_fusion.bn_module

                # pyrefly: ignore [bad-argument-type]
                fused_conv = fuse_conv_bn_eval(conv, bn)
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    gm.graph.erase_node(bn_node)

    gm.graph.lint()
    for pattern in module_function_patterns:
        conv_bn_to_fuse.clear()
        for node in gm.graph.nodes:
            if matches_module_function_pattern(pattern, node, modules):
                # TODO: support kwargs.
                if len(node.args) != 8:
                    continue
                conv = modules[node.args[0].target]
                bn_training = node.args[5]
                bn_eps = node.args[7]
                if conv.training or bn_training:
                    continue
                if type(bn_eps) is not float:
                    continue

                def _used_by_same_conv_module(users):
                    # [FUNCTION DOC] HELPER: Given the users of a bn constant, return True if all users take the same conv module (same target of first arg) as the first user.
                    conv_module_name = users[0].args[0].target
                    return all(
                        conv_module_name == user.args[0].target for user in users
                    )

                bn_args_is_constant = all(
                    n.op == "get_attr"
                    and (len(n.users) == 1 or _used_by_same_conv_module(list(n.users)))
                    for n in node.args[1:5]
                )
                if not bn_args_is_constant:
                    continue
                bn_running_mean = fetch_attr(node.args[1].target, gm)
                bn_running_var = fetch_attr(node.args[2].target, gm)
                bn_weight = fetch_attr(node.args[3].target, gm)
                bn_bias = fetch_attr(node.args[4].target, gm)
                if bn_running_mean is None or bn_running_var is None:
                    continue

                # Do hash based on the module name of conv
                hash_id = hash(node.args[0].target)
                if hash_id not in conv_bn_to_fuse:
                    conv_bn_to_fuse[hash_id] = ConvBNFusion(
                        node,
                        conv,
                        bn_running_mean=bn_running_mean,
                        bn_running_var=bn_running_var,
                        bn_eps=bn_eps,
                        bn_weight=bn_weight,
                        bn_bias=bn_bias,
                    )
                else:
                    if (
                        hash(bn_running_mean)
                        == hash(conv_bn_to_fuse[hash_id].bn_running_mean)
                        and hash(bn_running_var)
                        == hash(conv_bn_to_fuse[hash_id].bn_running_var)
                        and torch.allclose(
                            torch.tensor(bn_eps),
                            torch.tensor(conv_bn_to_fuse[hash_id].bn_eps),
                        )
                        and hash(bn_weight) == hash(conv_bn_to_fuse[hash_id].bn_weight)
                        and hash(bn_bias) == hash(conv_bn_to_fuse[hash_id].bn_bias)
                    ):
                        # Do fusion if same functional bn
                        conv_bn_to_fuse[hash_id].add_bn_node(node)
                    else:
                        # Disable the conv bn folding if conv shared by different bn
                        conv_bn_to_fuse[hash_id].disable_fusion()

        for conv_bn_fusion in conv_bn_to_fuse.values():
            if conv_bn_fusion.is_fusion_enabled():
                bn_nodes = conv_bn_fusion.bn_nodes
                conv = conv_bn_fusion.conv_module
                bn_running_mean = conv_bn_fusion.bn_running_mean
                bn_running_var = conv_bn_fusion.bn_running_var
                bn_eps = conv_bn_fusion.bn_eps
                bn_weight = conv_bn_fusion.bn_weight
                bn_bias = conv_bn_fusion.bn_bias

                fused_conv = copy.deepcopy(conv)
                fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
                    fused_conv.weight,
                    fused_conv.bias,
                    # pyrefly: ignore [bad-argument-type]
                    bn_running_mean,
                    # pyrefly: ignore [bad-argument-type]
                    bn_running_var,
                    # pyrefly: ignore [bad-argument-type]
                    bn_eps,
                    bn_weight,
                    bn_bias,
                )
                for bn_node in bn_nodes:
                    replace_node_module(bn_node.args[0], modules, fused_conv)
                    bn_node.replace_all_uses_with(bn_node.args[0])
                    gm.graph.erase_node(bn_node)
    gm.graph.lint()
    gm.recompile()

    return gm


class NormalizedLinearNode:
    # [CLASS DOC] HELPER: View over an FX call_function node targeting the functional linear op, providing accessors for
    # input, weight and bias whether passed positionally or by keyword. Construction raises an error for other nodes.
    def __init__(self, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] HELPER: Validate that node is a call_function of the functional linear op (raises an error otherwise) and store it.
        if node.op != "call_function":
            raise AssertionError(f"expected call_function node, got {node.op}")
        if node.target is not torch.nn.functional.linear:
            raise AssertionError(
                f"expected torch.nn.functional.linear target, got {node.target}"
            )
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Return the input argument node (positional index 0 or keyword input).
        if len(self.node.args) > 0:
            return self.node.args[0]  # type: ignore[return-value]
        else:
            return self.node.kwargs["input"]  # type: ignore[return-value]

    def get_weight(self) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Return the weight argument node (positional index 1 or keyword weight).
        if len(self.node.args) > 1:
            return self.node.args[1]  # type: ignore[return-value]
        else:
            return self.node.kwargs["weight"]  # type: ignore[return-value]

    def get_bias(self) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Return the bias argument node (positional index 2 or keyword bias); None if absent.
        if len(self.node.args) > 2:
            return self.node.args[2]  # type: ignore[return-value]
        else:
            return self.node.kwargs.get("bias", None)  # type: ignore[return-value]


class NormalizedMatmulNode:
    # [CLASS DOC] HELPER: View over an FX call_function node targeting bmm or matmul, giving accessors for the two operands
    # whether passed positionally or by keyword. Construction raises an error for other nodes.
    def __init__(self, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] HELPER: Validate that node is a call_function of bmm or matmul (raises an error otherwise) and store it.
        if node.op != "call_function":
            raise AssertionError(f"expected call_function node, got {node.op}")
        if node.target not in [torch.bmm, torch.matmul]:
            raise AssertionError(
                f"expected torch.bmm or torch.matmul target, got {node.target}"
            )
        self.node: torch.fx.Node = node

    def get_input(self) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Return the first operand node (positional 0 or keyword input).
        if len(self.node.args) > 0:
            return self.node.args[0]  # type: ignore[return-value]
        else:
            return self.node.kwargs["input"]  # type: ignore[return-value]

    def get_other(self) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Return the second operand node (positional 1 or keyword other).
        if len(self.node.args) > 1:
            return self.node.args[1]  # type: ignore[return-value]
        else:
            return self.node.kwargs["other"]  # type: ignore[return-value]


def check_permute(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] HELPER: Return True if node (a permute call with tensor_meta) swaps only the last two dimensions of its tensor and keeps
    # all others, with at least 3 dims in the permutation; negative dims normalized by rank. Otherwise False.
    ranks = len(node.meta["tensor_meta"].shape)
    if len(node.args) > 3:
        permutation = [node.args[i] % ranks for i in range(1, ranks + 1)]  # type: ignore[operator]
    elif (
        "permutation" in node.kwargs
        and node.kwargs["permutation"] is not None
        and len(node.kwargs["permutation"]) > 2  # type: ignore[arg-type]
    ):
        permutation = [i % ranks for i in node.kwargs["permutation"]]  # type: ignore[operator, union-attr]
    else:
        return False
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def sink_cat_after_pointwise(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Rewrite a cat whose sole consumer chain (through single-user view calls) ends in a unary relu/tanh by applying the
    # pointwise op to each cat input and concatenating afterwards. Skips cats using the out argument or a None dim.
    # Mutates the graph (creates and erases nodes), lints and recompiles, returns the module.
    def one_user(node):
        # [FUNCTION DOC] HELPER: Return the single user of node, or None if the node does not have exactly one user.
        users = list(node.users)
        return users[0] if len(users) == 1 else None

    def is_view(node):
        # [FUNCTION DOC] HELPER: Return True if node is a call_method named view.
        return node.op == "call_method" and node.target == "view"

    def is_pointwise_unary(node):
        # [FUNCTION DOC] HELPER: Return True if node is a call to relu or tanh (function or method form).
        ops = "call_function", "call_method"
        pointwise = torch.relu, torch.tanh, "relu", "tanh"
        return node.op in ops and node.target in pointwise

    g = module.graph
    for node in g.nodes:
        if node.op != "call_function" or node.target != torch.cat:
            continue

        cat_or_view = node
        while True:
            user = one_user(cat_or_view)
            if not user or not is_view(user):
                break
            cat_or_view = user

        if user and is_pointwise_unary(user):
            with g.inserting_before(node):
                arg_not_provided = object()

                def cat_args(
                    tensors,
                    dim=arg_not_provided,
                    axis=arg_not_provided,
                    out=None,
                ):
                    # [FUNCTION DOC] HELPER: Normalize cat arguments: returns a tuple of tensors, dim (default 0, axis overrides) and out.
                    if dim is arg_not_provided:
                        dim = 0
                    if axis is not arg_not_provided:
                        dim = axis
                    return tensors, dim, out

                tensors, dim, out = cat_args(*node.args, **node.kwargs)
                if out is not None or dim is None:
                    continue
                new_kwargs = {
                    name: val for name, val in user.kwargs.items() if name != "input"
                }
                new_tensors = [
                    g.create_node(user.op, user.target, args=(arg,), kwargs=new_kwargs)
                    for arg in tensors
                ]
                new_cat = g.create_node(
                    "call_function", torch.cat, args=(new_tensors, dim)
                )
                user.replace_all_uses_with(cat_or_view)
                node.replace_all_uses_with(new_cat)
                g.erase_node(user)
                g.erase_node(node)
    g.lint()
    module.recompile()
    return module


def linear_permute_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Fuse a linear op followed by a last-two-dims permute into a single call of linear_transpose. Erases the permute and
    # the linear node if it has no other users. Lints and recompiles; returns the module.
    for node in module.graph.find_nodes(op="call_method", target="permute"):
        if check_permute(node):
            if len(node.args) > 0:
                input_node = node.args[0]
            else:
                input_node = node.kwargs["input"]
            if (
                input_node.op == "call_function"
                and input_node.target is torch.nn.functional.linear
            ):
                normalized = NormalizedLinearNode(input_node)
                input = normalized.get_input()
                weight = normalized.get_weight()
                bias = normalized.get_bias()
                with module.graph.inserting_before(node):
                    fused_node = module.graph.call_function(
                        linear_transpose, args=(input, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)
                    module.graph.erase_node(node)
                    if len(input_node.users) == 0:
                        module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


# Y1 = X * W^T + bias
# Y2 = Y1.permute(0, 2, 1)
# ---->
# Y2 = (W * X^T + bias.unsqueeze(-1))^T
def linear_transpose(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    # [FUNCTION DOC] Compute the transpose of linear output directly: weight times transposed input, plus bias expanded on the last dim when bias is given. Returns a tensor.
    if bias is None:
        return torch.matmul(weight, input.transpose(-1, -2))
    return torch.matmul(weight, input.transpose(-1, -2)) + bias.unsqueeze(-1)


def permute_linear_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in module.graph.find_nodes(
        op="call_function", target=torch.nn.functional.linear
    ):
    # [FUNCTION DOC] Fuse a last-two-dims permute feeding a linear op into transpose_linear, erasing the replaced nodes when unused.
    # Lints and recompiles; returns the module.
        if len(node.args) > 0:
            input_node = node.args[0]
        else:
            input_node = node.kwargs["input"]
        if (
            input_node.op == "call_method"
            and input_node.target == "permute"
            and check_permute(input_node)
        ):
            normalized = NormalizedLinearNode(node)
            if len(input_node.args) > 0:
                input = input_node.args[0]
            else:
                input = input_node.kwargs["input"]
            weight = normalized.get_weight()
            bias = normalized.get_bias()
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_linear, args=(input, weight, bias)
                )
                node.replace_all_uses_with(fused_node)
                module.graph.erase_node(node)
                if len(input_node.users) == 0:
                    module.graph.erase_node(input_node)

    module.graph.lint()
    module.recompile()
    return module


def permute_matmul_fusion(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Fuse permute (last two dims) operands of bmm/matmul nodes into a single transpose_matmul call with flags telling which
    # operand is transposed. Erases replaced nodes that become unused. Lints and recompiles; returns the module.
    for node in itertools.chain(
        module.graph.find_nodes(op="call_function", target=torch.bmm),
        module.graph.find_nodes(op="call_function", target=torch.matmul),
    ):
        normalized = NormalizedMatmulNode(node)
        input_A_node = normalized.get_input()
        input_B_node = normalized.get_other()
        input_A = input_A_node
        input_B = input_B_node
        Atrans = Btrans = False
        if (
            input_A_node.op == "call_method"
            and input_A_node.target == "permute"
            and check_permute(input_A_node)
        ):
            Atrans = True
            if len(input_A_node.args) > 0:
                input_A = input_A_node.args[0]  # type: ignore[assignment]
            else:
                input_A = input_A_node.kwargs["input"]  # type: ignore[assignment]

        if (
            input_B_node.op == "call_method"
            and input_B_node.target == "permute"
            and check_permute(input_B_node)
        ):
            Btrans = True
            if len(input_B_node.args) > 0:
                input_B = input_B_node.args[0]  # type: ignore[assignment]
            else:
                input_B = input_B_node.kwargs["input"]  # type: ignore[assignment]

        if Atrans or Btrans:
            with module.graph.inserting_before(node):
                fused_node = module.graph.call_function(
                    transpose_matmul,
                    args=(input_A, input_B, Atrans, Btrans),
                )
            node.replace_all_uses_with(fused_node)
            module.graph.erase_node(node)
            if Atrans and len(input_A_node.users) == 0:
                module.graph.erase_node(input_A_node)
            if Btrans and len(input_B_node.users) == 0:
                module.graph.erase_node(input_B_node)

    module.graph.lint()
    module.recompile()
    return module


# X1 = X.permute(0, 2, 1)
# Y1 = X1 * W1^T + bias1
# ---->
# Y2 = X1.transpose(-1, -2) * W1^T + bias1
def transpose_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    # [FUNCTION DOC] Compute linear on an input given in permuted layout: matmul of the transposed input with the transposed weight, plus bias if provided. Returns a tensor.
    if bias is None:
        return torch.matmul(input.transpose(-1, -2), weight.t())
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def transpose_matmul(
    A: torch.Tensor, B: torch.Tensor, Atrans: bool, Btrans: bool
) -> torch.Tensor:
    # [FUNCTION DOC] Matrix multiply A and B after optionally transposing the last two dims of A (Atrans) and/or B (Btrans). Returns a tensor.
    if Atrans:
        A = A.transpose(-1, -2)
    if Btrans:
        B = B.transpose(-1, -2)
    return torch.matmul(A, B)
