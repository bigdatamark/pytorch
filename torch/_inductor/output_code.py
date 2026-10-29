# [MODULE DOC] Defines the OutputCode abstraction: the serializable, cacheable callable Inductor produces,
# plus implementations CompiledFxGraph (Python wrapper), CompiledAOTI (AOTInductor .so), MockFXGraphCacheOutput
# and RegionalOutputCode, and cudagraph/input-alignment post-compile helpers. No import-time side effects
# beyond creating the module logger.
"""
This provides an abstract class which parametrizes over an "output code" concept
for Inductor.  Intuitively, this represents the compiled callable which Inductor
produces which you can call to get optimized code.  However, this callable
has some other capabilities:

- It is serializable, so you can save/load this product from disk without
  having to do compilation again.

- (When using remote cache) it is addressable, so you can save just a key
  which you can use to load this product from remote cache later.

This class is abstract because we have several different implementations of
serialized format:

- Python wrapper (the default)

- AOTInductor (this produces ABI stable binaries which work across PyTorch
  versions)

"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from functools import partial
from typing import Any, cast, Protocol, TYPE_CHECKING, TypeAlias

import torch
from torch._custom_class_base import CustomClassBase
from torch._dynamo.utils import counters, get_runtime_metrics_context
from torch._guards import compile_context, CompileContext
from torch._higher_order_ops.wrap import inductor_compiled_code
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    cudagraph_trees_clone_live_user_visible_outputs,
    CudagraphCachedInfo,
    CudagraphMetadata,
    get_input_storage_mutation_info,
    get_input_storage_mutation_reason,
    get_partition_cudagraph_metadata,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.utils import (
    _unstable_customized_partition_wrapper,
    align_inputs_from_check_idxs,
    BoxedBool,
    CUDAGraphWrapperMetadata,
    GraphPartitionMap,
    InputType,
    is_gpu,
    output_node,
    set_tracing_context_output_strides,
)
from torch.fx._graph_pickler import _node_metadata_key_filter_safe, _ops_filter_safe
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from . import config
from .runtime.autotune_cache import AutotuneCacheBundler


if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Sequence

    from torch._inductor import metrics
    from torch._inductor.graph import GraphLowering
    from torch._library.fake_class_registry import FakeScriptObject
    from torch.export.pt2_archive._package_weights import Weights

    from .compile_fx import _CompileFxKwargs
    from .triton_bundler import TritonBundle

    # Boxed calling convention: a single list of inputs in, graph outputs out.
    # Inductor's compiled-graph entry points and cudagraph wrappers all take
    # exactly this shape, so we name it rather than repeating Callable[..., Any].
    _BoxedCallable: TypeAlias = Callable[[Sequence[InputType]], object]

    class CompiledFnRunner(Protocol):
        # [CLASS DOC] HELPER: Typing-only protocol (under TYPE_CHECKING) for the runner object emitted by the Python
        # wrapper under graph_partition: exposes the list of partition callables and recursively_apply_fns, which
        # rewrites every partition in place.
        """Runner emitted by the Python wrapper when graph_partition is on.

        codegen writes a Runner class (see codegen/wrapper.py) whose partitions
        hold the per-partition callables and whose recursively_apply_fns rewrites
        each partition in place (e.g. wrapping it in a cudagraph).
        """

        partitions: list[Callable[..., Any]]

        def recursively_apply_fns(self, fns: Sequence[Callable[..., Any]]) -> None: ...


log = logging.getLogger(__name__)


@dataclasses.dataclass
class OutputCode:
    # [CLASS DOC] Abstract base for Inductor compiled output. Holds optional remote-cache key, cache debug lines and
    # end-to-end compile time (all set after construction). Subclasses implement calling with a boxed input list,
    # prepare_for_serialization, post_compile and set_triton_bundle; base versions raise NotImplementedError.
    # TODO: Remove underscores here

    # None if the output is not remote cacheable
    _fx_graph_cache_key: str | None = dataclasses.field(default=None, init=False)
    _fx_graph_cache_debug_lines: list[str] | None = dataclasses.field(
        default=None, init=False
    )

    # How long it took to compile this OutputCode, end to end
    _time_taken_ns: int | None = dataclasses.field(default=None, init=False)

    def __call__(self, inputs: Sequence[Any]) -> Any:
        # [FUNCTION DOC] Run the compiled code on a sequence of inputs. Abstract: raises NotImplementedError.
        raise NotImplementedError(type(self))

    def prepare_for_serialization(self) -> None:
        # [FUNCTION DOC] Make the object picklable (drop unserializable callables). Abstract: raises NotImplementedError.
        raise NotImplementedError(type(self))

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        # [FUNCTION DOC] Post-processing after compile or cache load (cudagraphs, alignment, etc.). Abstract: raises NotImplementedError.
        raise NotImplementedError(type(self))

    # TODO: Get rid of this
    def set_triton_bundle(self, triton_bundle: Any) -> None:
        # [FUNCTION DOC] Attach a triton bundle to this output. Abstract: raises NotImplementedError.
        raise NotImplementedError(type(self))


_StrideExprStr: TypeAlias = str


# copy_ fails when trying to write to tensors with memory overlap,
# for expanded dimensions (a dimension which used to have size 1 -> ?)
# we can select one element from that dimension and write to it
# to achieve writing to all values of that dimension of the input tensor
def get_expanded_dims(t: torch.Tensor) -> list[int]:
    # [FUNCTION DOC] Returns the indices of dimensions of tensor t that are expanded (stride 0 with size not 1),
    # or None when t is not a tensor. Used to avoid overlapping-memory writes in copy_.
    if not isinstance(t, torch.Tensor):
        # pyrefly: ignore [bad-return]
        return None
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t: torch.Tensor, expanded_dims: list[int]) -> torch.Tensor:
    # [FUNCTION DOC] Returns t narrowed to a single element along each listed expanded dimension (via slice),
    # so writing to the result covers every value of those dims exactly once.
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t


# Bytewise copy of src's reachable strided extent into dst, starting from each
# tensor's data_ptr. Requires matching sizes/strides/dtype. Use when dst
# self-overlaps and a regular strided copy_ would be ambiguous: the overlap
# pattern reproduces in dst because the underlying bytes are copied verbatim.
def copy_strided_storage_(dst: torch.Tensor, src: torch.Tensor) -> None:
    # [FUNCTION DOC] Bytewise copy of src's reachable strided storage extent into dst, starting at each tensor's
    # offset, so a self-overlapping dst reproduces src's overlap pattern. Requires equal dtype, size and stride and
    # non-negative strides, else raises AssertionError. Returns early for empty tensors. Mutates dst storage.
    if dst.dtype != src.dtype:
        raise AssertionError(f"dtype mismatch: dst={dst.dtype} src={src.dtype}")
    if tuple(dst.size()) != tuple(src.size()):
        raise AssertionError(
            f"size mismatch: dst={tuple(dst.size())} src={tuple(src.size())}"
        )
    if tuple(dst.stride()) != tuple(src.stride()):
        raise AssertionError(
            f"stride mismatch: dst={tuple(dst.stride())} src={tuple(src.stride())}"
        )
    if dst.numel() == 0:
        return
    if not (all(st >= 0 for st in src.stride())):
        raise AssertionError("copy_strided_storage_ requires non-negative strides")
    elem = src.element_size()
    nbytes = (sum((s - 1) * st for s, st in zip(src.size(), src.stride())) + 1) * elem
    src_off = src.storage_offset() * elem
    dst_off = dst.storage_offset() * elem
    dst.untyped_storage()[dst_off : dst_off + nbytes].copy_(
        src.untyped_storage()[src_off : src_off + nbytes]
    )


def maybe_handle_backward_generation(
    compiled_graph: CompiledFxGraph,
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    # [FUNCTION DOC] For a backward graph under cudagraph trees, wraps compiled_graph.current_callable so the
    # cudagraph manager (looked up from the forward device index) is told backward is running before each call.
    # Does nothing for forward graphs. Raises AssertionError if the callable, device index or manager is missing.
    # Mutates compiled_graph.current_callable.
    if compiled_graph.current_callable is None:
        raise AssertionError("compiled_graph.current_callable must not be None")
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # See [Backward Generation Handling]
    # if cudagraph'd the forward and set the device, we need to let the cudagraph manager
    # know we are running the backward even if we will not run it in cudagraphs
    if is_backward and config.triton.cudagraph_trees:
        if boxed_forward_device_index is None:
            raise AssertionError("boxed_forward_device_index must not be None")
        if boxed_forward_device_index.value is None:
            raise AssertionError("boxed_forward_device_index.value must not be None")
        compiled_graph_callable = compiled_graph.current_callable

        manager = torch._inductor.cudagraph_trees.get_manager(
            boxed_forward_device_index.value, create_if_none_exists=False
        )
        # should already exist from forward
        if manager is None:
            raise AssertionError("CUDAGraph manager must not be None")

        def compiled_artifact(new_inputs: Sequence[InputType]) -> object:
        # [FUNCTION DOC] HELPER: Marks the cudagraph manager as running backward, then calls the original callable.
            manager.set_to_running_backward()  # type: ignore[union-attr]
            return compiled_graph_callable(new_inputs)

        compiled_graph.current_callable = compiled_artifact


def prepare_cudagraph_post_compile(
    compiled_graph: CompiledFxGraph,
    example_inputs: Sequence[InputType],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    # [FUNCTION DOC] Pre-cudagraph setup: when cudagraph trees are off, forces specialization of symbolic int
    # inputs (so graphs are valid); for non-inference forward graphs records the first device index into the boxed
    # forward device index. Mutates the boxed index; no return value.
    if not config.triton.cudagraph_trees:
        # Force specialize all inputs so that CUDA graphs will work
        for t in example_inputs:
            if isinstance(t, torch.SymInt):
                int(t)  # guard

    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]
    if boxed_forward_device_index is not None and not is_inference and not is_backward:
        boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))


def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor | type],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    # [FUNCTION DOC] Applies cudagraphs to a whole compiled graph if allowed. If the compiler bisector disables it,
    # or cached failure reasons exist, disables the boxed cudagraphs flag, handles backward generation and logs a
    # skip counter. Otherwise gathers tensor constants and metadata and replaces current_callable with the result of
    # the configured cudagraph policy or the default cudagraphify. Raises AssertionError on missing state.
    """
    Checks for any reasons not to run cudagraphs and then
    runs it on compiled_graph.
    Mutates the `compiled_graph.current_callable` and `cudagraphs`
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    if compiled_graph.current_callable is None:
        raise AssertionError("compiled_graph.current_callable must not be None")
    if compiled_graph.cudagraph_info is None:
        raise AssertionError("compiled_graph.cudagraph_info must not be None")
    cached_info = compiled_graph.cudagraph_info
    cudagraph_fail_reasons = cached_info.cudagraph_fail_reasons
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # Check if bisector wants to disable cudagraphs for this graph
    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    if not cudagraph_fail_reasons:
        fx_kwargs = compiled_graph.fx_kwargs
        static_input_idxs = fx_kwargs["static_input_idxs"]

        placeholders = cached_info.placeholders
        stack_traces = cached_info.stack_traces
        if stack_traces is None:
            raise AssertionError(
                "stack_traces should not be None in cudagraph_post_compile"
            )
        user_visible_output_idxs = getattr(cached_info, "user_visible_output_idxs", ())

        prepare_cudagraph_post_compile(
            compiled_graph, example_inputs, boxed_forward_device_index
        )

        current_callable = compiled_graph.current_callable
        if current_callable is None:
            raise AssertionError("current_callable must not be None")
        # Filter to only tensor constants (exclude opaque value type classes)
        tensor_constants = {
            k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
        }

        device_index = next(iter(compiled_graph.device_idxs))
        cudagraphify_kwargs = dict(
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(tensor_constants.values()),
            placeholders=placeholders,
            mutated_input_idxs=tuple(compiled_graph.mutated_input_idxs),
            kernel_free_cudagraph=compiled_graph.kernel_free_cudagraph,
            user_visible_output_idxs=tuple(user_visible_output_idxs),
        )

        policy = config.cudagraph_policy
        if policy is not None:
            compiled_graph.current_callable = policy.cudagraphify(
                current_callable,
                example_inputs,
                static_input_idxs or (),
                **cudagraphify_kwargs,
            )
        else:
            from .compile_fx import cudagraphify

            compiled_graph.current_callable = cudagraphify(
                current_callable,
                static_input_idxs=static_input_idxs or (),
                **cudagraphify_kwargs,
            )

    else:
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)

        if any(is_gpu(device) for device in compiled_graph.device_types):
            # prefer better disable_cudagraphs_reason bc stack trace
            # TODO: migrate all disable reasons to stack trace, refactor
            if compiled_graph.disabled_cudagraphs_reason:
                log_cudagraph_skip_and_bump_counter(
                    compiled_graph.disabled_cudagraphs_reason
                )
            else:
                log_cudagraph_skip_and_bump_counter(
                    f"skipping cudagraphs due to {cudagraph_fail_reasons}"
                )


def cudagraph_partition_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, torch.Tensor | type],
    boxed_forward_device_index: BoxedDeviceIndex | None,
) -> None:
    # [FUNCTION DOC] Graph-partition variant of cudagraph application. Disables cudagraphs (and handles backward
    # generation) on bisector request, cached failure reasons, or no partitions. Otherwise builds per-partition
    # metadata from partition_maps, creates one cudagraphify function per partition and applies them all through
    # compiled_graph.recursively_apply_fns. Raises AssertionError if required state is missing.
    """
    Cudagraphify each partition function, which first prepares the necessary
    metadata and then applies the cudagraphify function to each partition.

    Assuming all partition functions are cudagraphified and share the same order
    as `compiled_graph.partition_maps`. See [Note: Graph Partition Map for CUDAGraph].
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    if compiled_graph.cudagraph_info is None:
        raise AssertionError("compiled_graph.cudagraph_info must not be None")
    cudagraph_fail_reasons = compiled_graph.cudagraph_info.cudagraph_fail_reasons

    if (
        cudagraph_fail_reasons
        or compiled_graph.partition_maps is None
        or len(compiled_graph.partition_maps) == 0
    ):
        # cudagraphify is not called if there are no partitions
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        return

    if compiled_graph.current_callable is None:
        raise AssertionError("compiled_graph.current_callable must not be None")
    if compiled_graph.recursively_apply_fns is None:
        raise AssertionError("compiled_graph.recursively_apply_fns must not be None")
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]
    static_input_idxs = OrderedSet(compiled_graph.fx_kwargs["static_input_idxs"] or ())
    mutated_input_idxs = compiled_graph.mutated_input_idxs
    device_index = next(iter(compiled_graph.device_idxs))

    # Filter to only tensor constants (exclude opaque value type classes)
    tensor_constants = {
        k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
    }

    if compiled_graph.cudagraph_info.stack_traces is None:
        raise AssertionError(
            "stack_traces should not be None in cudagraph_partition_post_compile"
        )
    graph_metadata = CudagraphMetadata(
        compiled_graph.cudagraph_info.placeholders,
        static_input_idxs,
        mutated_input_idxs,
        compiled_graph.cudagraph_info.stack_traces,
        OrderedSet(
            getattr(compiled_graph.cudagraph_info, "user_visible_output_idxs", ())
        ),
        tensor_constants,
    )

    prepare_cudagraph_post_compile(
        compiled_graph, example_inputs, boxed_forward_device_index
    )

    from .compile_fx import cudagraphify

    # cudagraphify each partition function, assuming every graph partition function
    # is cudagraphable. Non-cudagraphable ops (e.g., cpu ops) are inlined into
    # `call` function and not included in partition functions.
    cudagraphify_fns = []
    for partition_map in compiled_graph.partition_maps:
        partition_metadata = get_partition_cudagraph_metadata(
            partition_map,
            graph_metadata,
        )

        cudagraphify_fn = partial(
            cudagraphify,
            static_input_idxs=tuple(partition_metadata.static_input_idxs),
            device_index=device_index,
            stack_traces=partition_metadata.stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(partition_metadata.constants.values()),
            placeholders=partition_metadata.placeholders,
            mutated_input_idxs=tuple(partition_metadata.mutated_input_idxs),
            kernel_free_cudagraph=compiled_graph.kernel_free_cudagraph,
            user_visible_output_idxs=tuple(partition_metadata.user_visible_output_idxs),
        )
        cudagraphify_fns.append(cudagraphify_fn)

    compiled_graph.recursively_apply_fns(cudagraphify_fns)


def maybe_realign_inputs(
    ran_cudagraphs: BoxedBool,
    compiled_graph: CompiledFxGraph,
    inputs_to_check: Sequence[int],
    mutated_inputs_idxs: OrderedSet[int],
) -> None:
    # [FUNCTION DOC] If cudagraphs did not run, wraps current_callable so inputs in inputs_to_check are realigned.
    # When alignment is deferred into generated code, only mutated inputs are wrapped (for writeback). Mutates
    # compiled_graph.current_callable; raises AssertionError if it is None while a wrapper is needed.
    """
    Realigns input strides from inputs_to_check if
    we didn't end up running cudagraphs. Mutates
    `compiled_graph.current_callable` if cudagraphs
    was run. Otherwise, does nothing.

    Non-mutated inputs are handled by deferred alignment copies
    in the generated code. Only mutated inputs need the wrapper
    for writeback.
    """
    if not ran_cudagraphs:
        check_idxs = inputs_to_check
        if compiled_graph._defers_input_alignment:
            # Non-mutated inputs are handled by deferred alignment copies
            # in the generated Python code. Only mutated inputs need the wrapper
            # for writeback. Backends that don't emit deferred copies (cpp_wrapper,
            # FXIR) need the full wrapper.
            check_idxs = [i for i in inputs_to_check if i in mutated_inputs_idxs]
        if check_idxs:
            if compiled_graph.current_callable is None:
                raise AssertionError("compiled_graph.current_callable must not be None")
            new_callable = align_inputs_from_check_idxs(
                compiled_graph.current_callable,
                check_idxs,
                mutated_inputs_idxs,
            )
            if new_callable is not compiled_graph.current_callable:
                # align_inputs_from_check_idxs preserves the boxed convention but
                # its wrapper is typed with a list[InputType] parameter; the field
                # uses the Sequence[InputType] boxed alias, so re-tag it here.
                compiled_graph.current_callable = cast("_BoxedCallable", new_callable)


class CompiledFxGraphConstants:
    # [CLASS DOC] Strategy object that returns a compiled graph's constants by reading them directly off the saved
    # CompiledFxGraph, merged with opaque value type classes. Counterpart of CompiledFxGraphConstantsWithGm.
    """Wrapper class that unwraps constants from a compiled fx graph. This
    version of the class only supports directly grabbing the saved constants off of
    a CompiledFxGraph.

    With freezing, FxGraphCache doesn't store the constants of the input
    GraphModule it gets from AOTAutograd. Instead, it saves just the **names**
    of those constants, and grabs the constant values directly from the graph module
    passed in at runtime.

    Thing is, we don't always *have* the graph module available at runtime, hence
    the existence of this class and its CompiledFxGraphConstantsWithGm counterpart.

    To support freezing, FXGraphCache gets passed a CompiledFxGraphConstantsWithGm during
    post compile. Otherwise, CompiledFxGraphConstants supports the basic case of loading
    the value of constants directly off of the original saved object.
    """

    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor | type]:
        # [FUNCTION DOC] Returns a mapping of constants name to tensor or class: the graph's saved constants plus its
        # opaque value type classes. Raises AssertionError if g.constants is None.
        if g.constants is None:
            raise AssertionError("g.constants must not be None")
        return {**g.constants, **g.opaque_value_type_classes}


class CompiledFxGraphConstantsWithGm(CompiledFxGraphConstants):
    # [CLASS DOC] Constants strategy for freezing: looks up frozen parameters by their original names on a supplied
    # GraphModule (stored in attribute gm) and merges them with saved constants and opaque value type classes.
    """
    This version of CompiledFxGraphConstants, instead of grabbing constants
    directly saved on CompiledFxGraphs, will just grab their names. Then, it takes
    a second GraphModule to grab the corresponding constant values out of.

    This is necessary for supporting freezing in FxGraphCache.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] Stores the GraphModule used later to look up frozen parameter values.
        self.gm = gm

    def unwrap(self, g: CompiledFxGraph) -> dict[str, torch.Tensor | type]:
        # [FUNCTION DOC] Returns a mapping combining the graph's own constants, frozen parameters fetched from self.gm
        # by original attribute name, and opaque value type classes.
        frozen_params = {
            name: getattr(self.gm, orig_name)
            for name, orig_name in g.frozen_param_names.items()
        }
        constants = g.constants or {}
        return {**constants, **frozen_params, **g.opaque_value_type_classes}


@dataclasses.dataclass
class CompiledFxGraph(OutputCode):
    # [CLASS DOC] OutputCode for a compiled FX graph via the Python wrapper; the object serialized to disk for the
    # FX graph cache. Holds the callable (cleared on serialization), generated source, cache key, device info,
    # mutation info, constants (or frozen param names), output strides, metrics and counter deltas, guards
    # expression, cudagraph info, partition maps and a metadata-stripped copy of the graph. Key methods:
    # post_compile, prepare_for_serialization, write_to_disk, after_deserialization.
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: _BoxedCallable | None
    recursively_apply_fns: Callable[[Sequence[Callable[..., Any]]], None] | None
    compiled_fn_runner: CompiledFnRunner | None
    cache_key: str
    source_code: str = dataclasses.field(repr=False)  # Do not display source_code
    runnable_graph_str: str = dataclasses.field(repr=False)  # Do not display graph
    inductor_post_grad_graph_str: str = dataclasses.field(
        repr=False
    )  # Do not display graph
    cache_linemap: list[tuple[int, str]] | None
    device_types: OrderedSet[str]
    device_idxs: OrderedSet[int]
    mutated_inputs: OrderedSet[str]
    mutated_input_idxs: OrderedSet[int]
    constants: dict[str, torch.Tensor] | None
    frozen_param_names: dict[str, str]
    torchbind_constants: dict[str, torch._C.ScriptObject | FakeScriptObject]
    opaque_value_type_classes: dict[str, type]
    output_strides: list[tuple[_StrideExprStr, ...] | None] | None
    disabled_cudagraphs_reason: str | None
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: str | None
    extern_libs_key: str | None
    inductor_provenance_mapping_str: str | None
    inductor_provenance_stack_traces_str: str | None
    kernel_free_cudagraph: bool

    cudagraph_info: CudagraphCachedInfo | None
    partition_maps: list[GraphPartitionMap] | None
    compile_region_name: str | None
    fx_kwargs: _CompileFxKwargs
    inputs_to_check: Sequence[int]

    _boxed_call: bool | None = None
    _triton_bundle: TritonBundle | None = None
    _wrap_compiled_regions: bool = False
    _defers_input_alignment: bool = False
    _compile_context: CompileContext | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )
    # Metadata-stripped copy of the FX graph for fake tensor propagation.
    # Running this graph under FakeTensorMode re-derives output shapes
    # (including aliasing) from the input shapes.
    _original_gm: torch.fx.GraphModule | None = None
    _serialized_original_gm: bytes | None = None

    def __init__(
        self,
        current_callable: _BoxedCallable | None,
        graph: GraphLowering,
        gm: torch.fx.GraphModule,
        output_strides: list[tuple[_StrideExprStr, ...] | None],
        disabled_cudagraphs_reason: str | None,
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
        cudagraphs: BoxedBool,
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        compile_region_name: str | None,
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        runnable_graph_str: str,
        inductor_post_grad_graph_str: str,
        compiled_fn_runner: CompiledFnRunner | None = None,
        inductor_provenance_mapping_str: str | None = None,
        inductor_provenance_stack_traces_str: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds the record from a finished GraphLowering and the source GraphModule. Copies cache key,
        # source code (read from graph.cache_path if set), device and mutation sets, constants (splitting frozen
        # params into a name mapping), strides and metrics. If cudagraphs are requested, evaluates disabling reasons
        # (lowering reason, input mutation, storage mutation, forced disable, non-tensor inputs), records stack
        # traces and placeholder info in cudagraph_info, and may disable the boxed cudagraphs flag. Sets the boxed call
        # flag, captures autotune compile context, and optionally deep-copies the graph with metadata cleared. Reads
        # the cache file; mutates the cudagraphs flag and counters.
        self.current_callable = current_callable
        self.compiled_fn_runner = compiled_fn_runner
        self.recursively_apply_fns = (
            compiled_fn_runner.recursively_apply_fns
            if compiled_fn_runner is not None
            else None
        )
        self.cache_key = graph.cache_key
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()
        self.runnable_graph_str = runnable_graph_str
        self.inductor_post_grad_graph_str = inductor_post_grad_graph_str
        self.inductor_provenance_mapping_str = inductor_provenance_mapping_str
        self.inductor_provenance_stack_traces_str = inductor_provenance_stack_traces_str
        self.cache_linemap = graph.cache_linemap
        # TODO - ordered set
        self.device_types = OrderedSet(graph.device_types)
        self.device_idxs = OrderedSet(graph.device_idxs)
        self.mutated_inputs = OrderedSet(graph.mutated_inputs)
        self.mutated_input_idxs = OrderedSet(graph.mutated_input_idxs)

        # We store the constant attributes in the cache entry and re-attach them
        # to the module created in PyCodeCache.load_by_key_path. In the case that
        # the graph has frozen parameters, we save the mapping from the attribute
        # names in the GraphLowering to the original name of the attribute in the
        # GraphModule. When we create the module from the cache entry, we then
        # look up the constants from the current GraphModule. This scheme allows
        # us to support caching with freezing.
        if not has_frozen_params(gm):
            self.constants = graph.constants
            self.frozen_param_names = {}
        else:
            self.constants = {}
            self.frozen_param_names = {}
            for k, v in graph.constants.items():
                if is_frozen_param(v):
                    self.frozen_param_names[k] = graph.allocated_constant_name[k]
                else:
                    self.constants[k] = v

        self.torchbind_constants = graph.torchbind_constants
        self.opaque_value_type_classes = graph.opaque_value_type_classes
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.kernel_free_cudagraph = graph.kernel_free_cudagraph
        self.metrics_deltas = metrics_deltas
        self.counter_deltas = counter_deltas
        self.guards_expr = None
        self.extern_libs_key = None
        self.cudagraph_info = None
        self.partition_maps = graph.partition_maps
        self._defers_input_alignment = getattr(graph, "_defers_input_alignment", False)
        storage_mutation_info = get_input_storage_mutation_info(gm)
        self.fx_kwargs = {}
        self.inputs_to_check = ()

        cudagraph_info = None
        if cudagraphs:
            # check cudagraph disabling reasons from inductor lowering
            if self.disabled_cudagraphs_reason:
                if any(is_gpu(device) for device in self.device_types):
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                if not config.triton.cudagraph_support_input_mutation:
                    # Skip supports for cudagraph-managed tensors
                    from torch._inductor.cudagraph_utils import (
                        check_for_mutation_ignore_cuda_graph_managed_tensor,
                    )

                    has_mutation_str = (
                        check_for_mutation_ignore_cuda_graph_managed_tensor(
                            gm,
                            self.mutated_inputs,
                            self.mutated_input_idxs,
                            static_input_idxs,
                        )
                    )
                    has_mutation = has_mutation_str is not None

                    if has_mutation:
                        self.disabled_cudagraphs_reason = has_mutation_str
                else:
                    # Check mutation later to support cudagraph-managed tensors
                    has_mutation = None

                storage_mutation_reason = get_input_storage_mutation_reason(
                    storage_mutation_info,
                )
                if storage_mutation_reason is not None:
                    self.disabled_cudagraphs_reason = storage_mutation_reason

                cudagraph_tests = [
                    (storage_mutation_reason is None, "input storage mutation"),
                    (not has_mutation, "mutated inputs"),
                    (
                        not config.force_disable_cudagraph_TESTING_ONLY,
                        "force_disable_cudagraph_TESTING_ONLY",
                    ),
                    (
                        all(
                            isinstance(
                                t,
                                (
                                    torch.Tensor,
                                    torch.SymInt,
                                    torch.Generator,
                                    CustomClassBase,
                                ),
                            )
                            for t in example_inputs
                        ),
                        "non-Tensor inputs",
                    ),
                ]
                output = output_node(gm)
                # output args are tuple of first argument
                if len(output.args) != 1:
                    raise AssertionError(
                        f"Expected output.args to have length 1, got {len(output.args)}"
                    )
                # Use stack traces captured on the output node before
                # post-grad passes, which may strip stack_trace from
                # individual arg nodes.
                stack_traces = output.meta.get("output_stack_traces") or [
                    (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
                    for arg in output.args[0]  # type: ignore[union-attr]
                ]
                user_visible_output_idxs = (
                    output.meta.get("user_visible_output_idxs", ())
                    if cudagraph_trees_clone_live_user_visible_outputs()
                    else ()
                )
                cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]
                placeholders = tuple(get_placeholder_info(gm.graph))
                cudagraph_info = CudagraphCachedInfo(
                    placeholders,
                    stack_traces,
                    tuple(user_visible_output_idxs),
                    cudagraph_fail_reasons,
                )

        self.cudagraph_info = cudagraph_info
        self.compile_region_name = compile_region_name
        self.inputs_to_check = inputs_to_check
        self.fx_kwargs = fx_kwargs
        self._set_compile_context_for_autotune_cache()

        # aot autograd needs to know to pass in inputs as a list
        self._boxed_call = True

        # Store whether to wrap compiled regions in inductor_compiled_code HOP
        # This is set at compile time to avoid runtime overhead
        self._wrap_compiled_regions = config.wrap_inductor_compiled_regions

        if self._wrap_compiled_regions:
            # Store a metadata-stripped copy of the FX graph. Running this
            # under FakeTensorMode re-derives output shapes and aliasing
            # from the input fake tensors.
            import copy

            gm_copy = copy.deepcopy(gm)
            for node in gm_copy.graph.nodes:
                node.meta.clear()
            self._original_gm = gm_copy

    def __del__(self) -> None:
        # [FUNCTION DOC] Destructor: when a compiled fn runner exists, deletes its partitions attribute so
        # cudagraphified partition functions can be freed.
        if self.compiled_fn_runner is not None:
            # For torch._inductor.config.graph_partition = True,
            # self.compiled_fn_runner.partitions hold cudagraphified functions
            # which prevents deallocation. When CompiledFxGraph is deleted,
            # self.compiled_fn_runner will not be called in the future so we
            # should also delete these partitions.
            del self.compiled_fn_runner.partitions

    def _set_compile_context_for_autotune_cache(self) -> None:
        # [FUNCTION DOC] HELPER: Saves the current compile context only if an autotune cache bundler has an active
        # compile for it; otherwise stores None.
        compile_context_for_autotune_cache = CompileContext.try_get()
        self._compile_context = (
            compile_context_for_autotune_cache
            if AutotuneCacheBundler.has_active_compile(
                compile_context_for_autotune_cache
            )
            else None
        )

    def __call__(self, inputs: Sequence[Any]) -> Any:
        # [FUNCTION DOC] Executes the compiled graph on a boxed list of inputs and returns its outputs. Optionally
        # records graph execution order for debugging, restores the saved compile context during the call, wraps in a
        # profiler record function when profiling, and always finishes runtime metrics and ends the autotune cache
        # bundler compile afterward, clearing the saved context if the compile is no longer active. Raises
        # AssertionError if current_callable is None.
        if self.current_callable is None:
            raise AssertionError("self.current_callable must not be None")

        if (
            torch._inductor.debug.RECORD_GRAPH_EXECUTION
            and torch._inductor.debug.GRAPH_EXECUTION_ORDER is not None
        ):
            graph_id = self.fx_kwargs.get("graph_id")
            compile_id = (
                torch._inductor.debug.GRAPH_COMPILE_IDS.get(graph_id)
                if graph_id is not None
                and torch._inductor.debug.GRAPH_COMPILE_IDS is not None
                else None
            )
            torch._inductor.debug.GRAPH_EXECUTION_ORDER.append(
                {
                    "compile_id": compile_id,
                }
            )
        compile_context_for_autotune_cache = self._compile_context
        has_active_autotune_cache_bundler = AutotuneCacheBundler.has_active_compile(
            compile_context_for_autotune_cache
        )
        autotune_cache_context = (
            compile_context(compile_context_for_autotune_cache)
            if has_active_autotune_cache_bundler
            else contextlib.nullcontext()
        )
        # Autotune cache writes happen during the first call. End the bundler
        # while its saved compile context is restored, then clear the saved
        # context after leaving the compile_context manager.
        try:
            with autotune_cache_context:
                try:
                    # Checking the profiler directly is faster than nullcontext
                    if torch.autograd.profiler._is_profiler_enabled:
                        with torch._C._profiler._RecordFunctionFast(
                            f"## Call CompiledFxGraph {self._fx_graph_cache_key} ##",
                            keyword_values={"scope": "user_scope"},
                        ):
                            return self.current_callable(inputs)
                    else:
                        return self.current_callable(inputs)
                finally:
                    get_runtime_metrics_context().finish()
                    if has_active_autotune_cache_bundler:
                        AutotuneCacheBundler.end_compile()
        finally:
            if (
                has_active_autotune_cache_bundler
                and not AutotuneCacheBundler.has_active_compile(
                    compile_context_for_autotune_cache
                )
            ):
                self._compile_context = None

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        # [FUNCTION DOC] Post-processing run after compile or cache load (not saved in cache). If a user partition
        # wrapper is configured under graph_partition, applies it to each partition and returns. Otherwise sets
        # tracing-context output strides, optionally disables cudagraphs per policy or cached reason, runs the
        # whole-graph or partition cudagraph post-compile, realigns inputs, lazily deserializes the stored original
        # graph, and optionally wraps current_callable to dispatch via the inductor_compiled_code higher-order op when
        # in torch dispatch mode. Raises AssertionError on missing required kwargs. Mutates self and the cudagraphs flag.
        """
        Run a set of post processing steps after loading from the cache. These involve:
         - Setting the tracing context output strides
         - Running cudagraphs if enabled
         - Realigning inputs

        This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
        The results of this function are *not* saved in the cache itself.
        """
        if config.graph_partition and _unstable_customized_partition_wrapper.wrapper:
            # Mechanically apply user-specified cudagraph wrappers without modification
            if self.recursively_apply_fns is None:
                raise AssertionError("self.recursively_apply_fns must not be None")
            if self.compiled_fn_runner is None:
                raise AssertionError("self.compiled_fn_runner must not be None")
            num_partitions = len(self.compiled_fn_runner.partitions)
            wrapper_metadatas = [
                CUDAGraphWrapperMetadata(num_partitions, i)
                for i in range(num_partitions)
            ]
            customized_wrapper = _unstable_customized_partition_wrapper.wrapper
            customized_wrappers_with_metadata = [
                lambda f, m=metadata: customized_wrapper(f, m)
                for metadata in wrapper_metadatas
            ]
            self.recursively_apply_fns(customized_wrappers_with_metadata)
            return

        set_tracing_context_output_strides(example_inputs, self)
        if graph_kwargs["cudagraphs"] is None:
            raise AssertionError("graph_kwargs['cudagraphs'] must not be None")
        if graph_kwargs["is_backward"] is None:
            raise AssertionError("graph_kwargs['is_backward'] must not be None")
        is_backward = graph_kwargs["is_backward"]
        cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]

        # When a CUDAGraphPolicy is set and it says not to wrap this
        # inner CompiledFxGraph (e.g. because wrapping happens at the
        # outer level via policy.wrap_output), disable cudagraphs for
        # this graph so the rest of post_compile (input realignment,
        # _wrap_compiled_regions) still runs normally.
        policy = config.cudagraph_policy
        if policy is not None and not policy.should_wrap(self):
            counters["inductor"]["cudagraph_skips"] += 1
            BoxedBool.disable(cudagraphs)

        if cudagraphs:
            # It's possible that cudagraphs is enabled, but was disabled
            # during a previous compilation we're loading from the cache.
            # If so, we need to disable it on this new process too.
            if self.disabled_cudagraphs_reason:
                if any(is_gpu(device) for device in self.device_types):
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                if is_backward:
                    if "boxed_forward_device_index" not in graph_kwargs:
                        raise AssertionError(
                            "Expected 'boxed_forward_device_index' in graph_kwargs"
                        )
                    boxed_forward_device_index = graph_kwargs[
                        "boxed_forward_device_index"
                    ]
                else:
                    # On the forward we don't know whether or not
                    # boxed_forward_device_index is set yet
                    boxed_forward_device_index = graph_kwargs.get(
                        "boxed_forward_device_index", None
                    )

                if config.graph_partition and policy is None:
                    # With graph_partition=True, we skip some cudagraph checks
                    # if it's supported with partition, so we use
                    # cudagraph_partition_post_compile.  When a CUDAGraphPolicy
                    # is active, we use cudagraph_post_compile instead so the
                    # policy controls wrapping via policy.cudagraphify().
                    cudagraph_partition_post_compile(
                        example_inputs,
                        self,
                        cudagraphs,
                        constants.unwrap(self),
                        boxed_forward_device_index,
                    )
                else:
                    cudagraph_post_compile(
                        example_inputs,
                        self,
                        cudagraphs,
                        constants.unwrap(self),
                        boxed_forward_device_index,
                    )
        inputs_to_check = self.inputs_to_check
        # cudagraphs could have been disabled from the earlier conditions
        # so we still need to realign inputs if that happens
        maybe_realign_inputs(
            cudagraphs,
            self,
            inputs_to_check,
            self.mutated_input_idxs,
        )

        if self._original_gm is None and self._serialized_original_gm is not None:
            from torch._subclasses import FakeTensorMode
            from torch.fx._graph_pickler import GraphPickler
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            fake_mode = FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(),
            )
            original_gm = cast(
                torch.fx.GraphModule,
                GraphPickler.loads(self._serialized_original_gm, fake_mode),
            )
            original_gm.recompile()
            self._original_gm = original_gm
            self._serialized_original_gm = None

        # Apply inductor_compiled_code HOP wrapper if configured
        # This is done in post_compile to ensure it works with cached artifacts
        if self._wrap_compiled_regions and self.current_callable is not None:
            from torch._higher_order_ops.wrap import InductorCompiledCallable

            original_callable = self.current_callable

            inductor_callable = InductorCompiledCallable(
                original_callable,
                self._original_gm,
                compile_region_name=self.compile_region_name,
            )

            def wrapped_callable(inputs):
            # [FUNCTION DOC] HELPER: Calls the compiled-code higher-order op (with region name if any) when in torch
            # dispatch mode, else calls the original callable with the boxed inputs.
                if is_in_torch_dispatch_mode():
                    kwargs = (
                        {"name": self.compile_region_name}
                        if self.compile_region_name is not None
                        else {}
                    )
                    return inductor_compiled_code(inductor_callable, inputs, **kwargs)
                else:
                    return original_callable(inputs)

            self.current_callable = wrapped_callable

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        # [FUNCTION DOC] Stores the triton bundle on this object.
        self._triton_bundle = triton_bundle

    def prepare_for_serialization(self) -> None:
        # [FUNCTION DOC] Prepares for pickling: clears the compile context, callable, recursion function and runner, and
        # replaces the original graph module with its pickled bytes (GraphPickler, no ops filter). Mutates self.
        # We can't really serialize callables that may be C++/Triton/etc.,
        # so we serialize their PyCodeCache disk cache location instead.
        # TODO: This could be better if we're ever able to serialize compiled
        # models to disk.
        self._compile_context = None
        self.current_callable = None
        self.recursively_apply_fns = None
        self.compiled_fn_runner = None
        if self._original_gm is not None:
            from torch.fx._graph_pickler import GraphPickler, Options

            self._serialized_original_gm = GraphPickler.dumps(
                self._original_gm, Options(ops_filter=None)
            )
            self._original_gm = None
        # Note: _serialized_fx_graph is already in serializable form (SerializedGraphModule)
        # so it doesn't need to be cleared

    def write_to_disk(self) -> str:
        # [FUNCTION DOC] Ensures the generated Python source exists on disk at the cache path for cache_key (written
        # atomically, creating dirs, and bumping a counter if it was missing). Returns the artifact path.
        from torch._dynamo.utils import counters
        from torch._inductor.codecache import get_path, write_atomic

        # See _save_graph(); we don't store the callable in the cache entry so
        # recreate it here from the PyCodeCache disk cache.
        artifact_path = get_path(self.cache_key, "py")[2]
        code = self.source_code
        if not os.path.exists(artifact_path):
            counters["inductor"]["fxgraph_lookup_write_file"] += 1
            write_atomic(artifact_path, code, make_dirs=True)
        return artifact_path

    def after_deserialization(self, constants: CompiledFxGraphConstants) -> str:
        # [FUNCTION DOC] Restores the callable after loading from cache: writes source to disk, loads it through
        # PyCodeCache with constants from the given strategy, and sets current_callable, recursively_apply_fns and the
        # runner. Logs and re-raises OSError on load failure. Returns the artifact path.
        from torch._dynamo.utils import dynamo_timed
        from torch._inductor.codecache import PyCodeCache

        artifact_path = self.write_to_disk()

        try:
            with dynamo_timed(
                "PyCodeCache.load_by_key_path",
                log_pt2_compile_event=True,
            ):
                code_cache = PyCodeCache.load_by_key_path(
                    self.cache_key,
                    artifact_path,
                    self.cache_linemap,
                    constants.unwrap(self),
                )
                self.current_callable = code_cache.call
                self.recursively_apply_fns = getattr(
                    code_cache, "recursively_apply_fns", None
                )
                self.compiled_fn_runner = getattr(code_cache, "runner", None)
        except OSError:
            log.error("Failed to load artifact: %s", artifact_path)
            raise

        return artifact_path


@dataclasses.dataclass
class CompiledAOTI(OutputCode):
    # [CLASS DOC] OutputCode for an AOTInductor compiled artifact. Holds filename (a path, list of files/weights, or a
    # GraphModule), device_type, the loaded callable and cached file bytes for serialization. The callable is
    # created in __post_init__ when libtorch linking and autograd-for-aot are enabled.
    """
    Class holding an AOTInductor compiled so.
    """

    filename: str | list[str | Weights] | torch.fx.GraphModule
    device_type: str
    current_callable: Callable[..., Any] | None = None
    _cached_files: dict[str, bytes] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # [FUNCTION DOC] Creates the model runner callable for the .so when linking libtorch is enabled. Returns early
        # (no callable) on macOS/Windows, windows cross-target, cpp-only packaging, or when autograd for AOT is off. A
        # GraphModule filename is used directly. Otherwise builds a CUDA, XPU or CPU container runner and uses its run
        # method; raises RuntimeError for unsupported device types. Also rewrites missing cached files to disk.
        if not config.aot_inductor.link_libtorch:
            return

        if (
            torch._inductor.cpp_builder._IS_MACOS
            or torch._inductor.cpp_builder._IS_WINDOWS
        ):
            return

        if config.aot_inductor.cross_target_platform == "windows":
            return

        if config.aot_inductor.package_cpp_only:
            return

        if not config.enable_autograd_for_aot:
            return

        if isinstance(self.filename, list):
            current_callable = next(
                fn for fn in self.filename if isinstance(fn, str) and fn.endswith(".so")
            )
        else:
            current_callable = self.filename

        if isinstance(current_callable, torch.fx.GraphModule):
            # pyrefly: ignore [bad-assignment]
            self.current_callable = current_callable
            return

        if self.device_type.startswith("cuda"):
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerCuda(  # type: ignore[call-arg]
                    current_callable,
                    1,
                    self.device_type,
                    "",
                    True,
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        elif self.device_type.startswith("xpu"):
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerXpu(  # type: ignore[call-arg]
                    current_callable,
                    1,
                    self.device_type,
                    "",
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        elif self.device_type == "cpu":
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerCpu(  # type: ignore[call-arg]
                    current_callable, 1
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"unsupported device type {self.device_type}")
        self.current_callable = current_callable
        self._boxed_call = True
        for file in self._cached_files:
            if not os.path.exists(file):
                with open(file, "wb") as f:
                    f.write(self._cached_files[file])

    def __call__(self, inputs: Sequence[Any]) -> Any:
        # [FUNCTION DOC] Calls the loaded AOTI callable with the inputs; raises RuntimeError if not loaded.
        if self.current_callable is None:
            raise RuntimeError("AOTInductor compiled so is not loaded")
        return self.current_callable(inputs)

    def prepare_for_serialization(self) -> None:
        # [FUNCTION DOC] Clears the callable and stores the bytes of each file named by filename (a string or list)
        # into the cached files mapping. Reads files from disk.
        self.current_callable = None
        self._cached_files = {}
        filenames: list[str] = []
        if isinstance(self.filename, list):
            filenames = self.filename  # type: ignore[assignment]
        elif isinstance(self.filename, str):
            filenames = [self.filename]
        for name in filenames:
            with open(name, "rb") as f:
                self._cached_files[name] = f.read()

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: returns a copy of the instance state with the callable removed.
        state = self.__dict__.copy()
        state["current_callable"] = None
        return state

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        # [FUNCTION DOC] Reloads the callable via __post_init__ if it is not loaded yet; arguments are unused.
        if self.current_callable is None:
            self.__post_init__()

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        # [FUNCTION DOC] No-op: AOTI output does not use triton bundles.
        pass


@dataclasses.dataclass
class MockFXGraphCacheOutput(OutputCode):
    # [CLASS DOC] HELPER: Test/mock OutputCode that simply wraps a callable gm and calls it with the boxed inputs.
    gm: Any = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the boxed call flag.
        self._boxed_call = True

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        pass

    def __call__(self, inputs: Sequence[Any]) -> Any:
        # [FUNCTION DOC] HELPER: Calls the wrapped gm with the inputs and returns the result.
        return self.gm(inputs)

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        pass


@dataclasses.dataclass
class RegionalOutputCode(OutputCode):
    # [CLASS DOC] OutputCode for regional inductor results: holds a GraphModule (mixing compiled and eager regions),
    # or after serialization its GraphPickler bytes plus recorded dynamo-disable wrappers and the op/metadata
    # filters. Deserialization requires a fake mode found in the example inputs at post_compile.
    """
    OutputCode for regional inductor compilation results.

    Regional inductor returns a torch.fx.GraphModule that contains both
    compiled regions (via standalone_compile) and eager regions. This needs
    special serialization using GraphPickler instead of standard pickle.

    The serialization strategy stores the GraphModule as bytes using
    GraphPickler.dumps(), which handles FakeTensors, AOTCompiledArtifacts,
    and other special objects that standard pickle cannot handle.
    """

    # The serialized graph module as bytes (using GraphPickler)
    _serialized_graph_module: bytes | None = None

    # The actual graph module (cleared during serialization)
    _graph_module: torch.nn.Module | None = None

    # Optional filter for ops during serialization
    _ops_filter: Callable[[str], bool] | None = None

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        ops_filter: Callable[[str], bool] = _ops_filter_safe,
        node_metadata_key_filter: Callable[[str], bool] | None = (
            _node_metadata_key_filter_safe
        ),
    ):
        # [FUNCTION DOC] Stores the graph module and serialization filters, marks boxed calling, and detects whether the
        # inner module uses a boxed code generator. Raises AssertionError if the module is not a GraphModule.
        """
        Args:
            graph_module: The torch.fx.GraphModule returned by regional_inductor
            ops_filter: Optional filter function for op names during serialization.
                If provided, only ops whose name passes the filter will be serialized.
        """
        super().__init__()
        self._graph_module = graph_module
        self._serialized_graph_module = None
        self._serialized_wrappers = []
        self._boxed_call = True
        _, module = self._unwrap_graph_module()
        self._inner_boxed_call = isinstance(
            module.graph._codegen, torch.fx.graph._BoxedCodeGen
        )
        self._ops_filter = ops_filter
        self._node_metadata_key_filter = node_metadata_key_filter

    def __call__(self, inputs: list[Any]) -> Any:
        # [FUNCTION DOC] Runs the graph module on the input list (passed whole if boxed, else unpacked). Raises
        # RuntimeError if the module is not loaded.
        """
        Execute the regional compiled graph.

        Args:
            inputs: A mutable list of inputs. Must be a list (not an arbitrary
                Sequence) because the boxed calling convention allows the callee
                to mutably clear the list when inputs become dead, enabling early
                memory deallocation.
        """
        if self._graph_module is None:
            raise RuntimeError(
                "RegionalOutputCode has no graph module loaded. "
                "Did you forget to call post_compile()?"
            )

        if self._inner_boxed_call:
            return self._graph_module(inputs)
        return self._graph_module(*inputs)

    @property
    def graph(self):
        # [FUNCTION DOC] Property: the fx graph of the underlying (unwrapped) GraphModule.
        _, module = self._unwrap_graph_module()
        return module.graph

    def _unwrap_graph_module(
        self,
    ) -> tuple[list[tuple[Callable, dict[str, Any]]], torch.fx.GraphModule]:
        # [FUNCTION DOC] HELPER: Unwraps an optional dynamo OptimizedModule (recording a dynamo disable wrapper entry
        # with its reason and wrapping flag) and returns the list of wrappers and the inner GraphModule. Raises
        # AssertionError if the dynamo context is not a disable context or the inner module is not a GraphModule.
        module = self._graph_module
        serialized_wrappers = []
        if isinstance(module, torch._dynamo.OptimizedModule):
            dynamo_ctx = module.dynamo_ctx
            if not isinstance(dynamo_ctx, torch._dynamo.eval_frame.DisableContext):
                raise AssertionError(f"Expected DisableContext, got {type(dynamo_ctx)}")
            serialized_wrappers.append(
                (
                    torch._dynamo.disable,
                    {"reason": dynamo_ctx.msg, "wrapping": dynamo_ctx.wrapping},
                )
            )
            module = module._orig_mod
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected torch.fx.GraphModule, got {type(module)}")
        return serialized_wrappers, module

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        # [FUNCTION DOC] If the module is already loaded, does nothing. Otherwise deserializes the stored bytes with
        # GraphPickler using the fake mode detected from example_inputs, recompiles, reapplies recorded wrappers in
        # reverse order and stores the module. Raises AssertionError if nothing was serialized and RuntimeError if no
        # fake mode is found.
        """
        Post-compile processing for regional inductor.

        This deserializes the GraphModule from bytes using GraphPickler,
        extracting the fake_mode from example_inputs.
        """
        if self._graph_module is not None:
            return
        if self._serialized_graph_module is None:
            raise AssertionError("self._serialized_graph_module must not be None")
        # Get fake mode from example inputs
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(example_inputs)
        if fake_mode is None:
            raise RuntimeError(
                "Could not detect fake mode from example inputs. "
                "Regional inductor requires fake mode for deserialization."
            )

        # Deserialize the graph module
        from torch.fx._graph_pickler import GraphPickler

        gm = GraphPickler.loads(self._serialized_graph_module, fake_mode)
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"Expected torch.fx.GraphModule, got {type(gm)}")
        gm.recompile()
        for fn, kwargs in reversed(self._serialized_wrappers):
            gm = fn(gm, **kwargs)
        self._graph_module = gm

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        # [FUNCTION DOC] No-op: regional inductor does not use triton bundles.
        """Regional inductor doesn't use triton bundles directly."""

    def prepare_for_serialization(self) -> None:
        # [FUNCTION DOC] If a module is loaded, unwraps it, serializes it to bytes via GraphPickler with the configured
        # filters, and clears the live module so standard pickle works.
        """
        Prepare for serialization by converting the GraphModule to bytes.

        This uses GraphPickler to serialize the graph module since it contains
        special objects like FakeTensors and AOTCompiledArtifacts that need
        custom pickling.
        """
        if self._graph_module is not None:
            from torch.fx._graph_pickler import GraphPickler, Options

            self._serialized_graph_module = None
            self._serialized_wrappers, graph_module = self._unwrap_graph_module()

            self._serialized_graph_module = GraphPickler.dumps(
                graph_module,
                options=Options(
                    ops_filter=self._ops_filter,
                    node_metadata_key_filter=self._node_metadata_key_filter,
                ),
            )
            # Clear the graph module to avoid pickling it with standard pickle
            self._graph_module = None
