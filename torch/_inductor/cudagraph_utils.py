# [MODULE DOC] Utilities for Inductor's CUDA graph support: skip-reason checks (mutation, devices, allocator), placeholder
# metadata, partition metadata remapping, and a pluggable CUDAGraphPolicy.
# Import-time side effects: creates two artifact loggers (cudagraphs, cudagraph_static_inputs) and defines
# type aliases OutputType and ModelType.
# mypy: disallow-untyped-defs
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TYPE_CHECKING, TypeVar

import torch
from torch._dynamo.utils import counters, get_metrics_context
from torch._inductor.utils import GraphPartitionMap, InputType
from torch._subclasses.fake_tensor import get_plain_tensors, is_fake
from torch.utils._ordered_set import OrderedSet

from . import config
from .utils import is_using_cudagraph_partition


if TYPE_CHECKING:
    from collections.abc import Sequence, Set as AbstractSet

    from torch._inductor.output_code import OutputCode

_OC = TypeVar("_OC", bound="OutputCode")


cudagraphs_log = torch._logging.getArtifactLogger(__name__, "cudagraphs")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


OutputType = list[int | torch.Tensor | None]
ModelType = Callable[[list[InputType]], OutputType]


def cudagraph_trees_generation_cloning() -> Literal["user_visible"] | None:
    # [FUNCTION DOC] Read the triton cudagraph_trees_generation_cloning config value and validate it.
    # Returns None or the string user_visible; raises AssertionError for any other value.
    mode = config.triton.cudagraph_trees_generation_cloning
    if mode not in (None, "user_visible"):
        raise AssertionError(
            "Expected torch._inductor.config.triton."
            "cudagraph_trees_generation_cloning to be None or 'user_visible', "
            f"got {mode!r}"
        )
    return mode


def cudagraph_trees_clone_live_user_visible_outputs() -> bool:
    # [FUNCTION DOC] Returns True when the generation cloning config mode is user_visible, otherwise False.
    return cudagraph_trees_generation_cloning() == "user_visible"


INPUT_STORAGE_MUTATION_TARGETS = (
    torch.ops.aten.set_.default,
    torch.ops.aten.set_.source_Storage,
    torch.ops.aten.set_.source_Storage_storage_offset,
    torch.ops.aten.set_.source_Tensor,
    torch.ops.aten.set_.source_Tensor_storage_offset,
    torch.ops.aten.shallow_copy_data_.default,
)


class CUDAGraphPolicy:
    # [CLASS DOC] Pluggable policy hooking into Inductor's post_compile CUDA graph wrapping, selected via the cudagraph_policy
    # config (None means built-in behavior). Subclasses override cudagraphify (how to wrap), should_wrap (whether to
    # wrap an inner compiled graph) and wrap_output (optional outer wrapping). Defaults wrap everything via
    # compile_fx.cudagraphify and leave outputs unchanged. Stateless.
    """Pluggable policy controlling CUDA graph wrapping in Inductor's post_compile.

    Override methods to customize:
      - HOW compiled functions are cudagraph-wrapped (cudagraphify)
      - WHETHER inner CompiledFxGraphs should be wrapped (should_wrap)
      - OUTER wrapping of compound outputs like RegionalOutputCode (wrap_output)

    Set via ``torch._inductor.config.cudagraph_policy``.  When ``None``
    (the default), the existing built-in behaviour is used unchanged.

    Example usage::

        class MyCUDAGraphPolicy(CUDAGraphPolicy):
            def cudagraphify(self, model, example_inputs, static_input_idxs, **kwargs):
                return my_custom_wrapper(model, example_inputs, static_input_idxs)


        with torch._inductor.config.patch("cudagraph_policy", MyCUDAGraphPolicy()):
            compiled_fn = deserialize_artifacts(...)
    """

    def cudagraphify(
        self,
        model: Callable[..., Any],
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        *,
        device_index: int,
        is_backward: bool,
        is_inference: bool,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Wrap a compiled callable with CUDA graph capture/replay by delegating to compile_fx.cudagraphify.
        # Args: model callable, example inputs (unused by the default), static input indices, device index and
        # is_backward/is_inference flags; extra keyword args are forwarded. Returns the wrapped callable.
        # The import of compile_fx happens lazily at call time.
        """Wrap a single compiled callable with CUDA graph capture/replay.

        Called by ``cudagraph_post_compile`` for each ``CompiledFxGraph``.
        The default delegates to ``compile_fx.cudagraphify`` (cudagraph_trees).

        ``example_inputs`` are the example inputs at post_compile time.
        The default implementation does not forward them because
        ``compile_fx.cudagraphify`` defers graph recording to the first
        real call via an inner closure.  Subclasses that need the
        example inputs for warmup or static-input detection may use them.

        When ``config.graph_partition=True``, setting a CUDAGraphPolicy
        bypasses ``cudagraph_partition_post_compile`` (which wraps each
        partition individually) and routes through ``cudagraph_post_compile``
        instead, so this method wraps the *entire* callable, not individual
        partitions.  Subclasses that need per-partition control should
        handle partitioning internally.
        """
        from torch._inductor.compile_fx import cudagraphify

        return cudagraphify(
            model,
            static_input_idxs,
            device_index=device_index,
            is_backward=is_backward,
            is_inference=is_inference,
            **kwargs,
        )

    def should_wrap(self, compiled_graph: OutputCode) -> bool:
        # [FUNCTION DOC] Decide whether a given inner compiled graph should get CUDA graph wrapping. Default always returns True.
        """Whether to apply cudagraph wrapping to this CompiledFxGraph.

        Called for each inner ``CompiledFxGraph`` during ``post_compile``.
        Return ``False`` to skip wrapping (e.g. when wrapping at the outer
        level via ``wrap_output`` instead).

        Default: ``True`` (wrap everything, same as current behaviour).
        """
        return True

    def wrap_output(self, output_code: _OC) -> _OC:
        # [FUNCTION DOC] Optional outer-level wrapping of an OutputCode after post_compile. Default returns the input unchanged.
        """Optional outer-level wrapping after inner post_compile completes.

        Called by ``_compile_fx_inner``, ``BundledOutputCodeLoadable.post_compile``,
        and ``FxGraphCacheLoadable.post_compile`` on the ``OutputCode`` returned
        from ``post_compile``.  Subclasses that only want to wrap specific
        output types should check ``isinstance`` and return the input
        unchanged for types they don't handle.

        Default: identity (no outer wrapping).
        """
        return output_code


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionID:
    # [CLASS DOC] HELPER: frozen record holding an integer id that uniquely identifies a function wrapped for cudagraph recording.
    "Unique counter of a function wrapped in cudagraphify_impl"

    id: int


@dataclasses.dataclass(frozen=True, slots=True)
class PlaceholderInfo:
    # [CLASS DOC] Frozen, serializable summary of an fx placeholder node: name, stack trace, user placeholder infos (recursive,
    # one level deep in practice) and the stack trace of its mutating use. Used in cudagraph logging/error messages.
    """
    A serializable version of torch.fx.Node that contains information
    pertinent to placeholder stack traces. We use these in logging and error messages
    related to cudagraphs, and will cache these results.
    """

    name: str
    stack_trace: str | None
    # This field is recursive, but never cyclic (since a node never uses itself)
    users: list[PlaceholderInfo]
    mutating_use_stack_trace: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class InputStorageMutationInfo:
    # [CLASS DOC] Frozen record of graph inputs whose storage is rebound by set-like ops (input_idxs) plus the first
    # stack trace found for such an op (or None).
    input_idxs: OrderedSet[int]
    stack_trace: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class WrappedFunction:
    # [CLASS DOC] Frozen description of a function to record for CUDA graph replay: the callable, static input indices, a
    # FunctionID, constant tensors, placeholder infos, mutated input indices, a kernel_free_cudagraph flag and the
    # set of user visible output indices. Used to find an applicable graph in the cudagraph tree.
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]
    kernel_free_cudagraph: bool = False
    user_visible_output_idxs: frozenset[int] = dataclasses.field(
        default_factory=frozenset
    )


def get_mutating_use_stack_trace_from_node(
    placeholder_node: torch.fx.Node,
) -> str | None:
    # [FUNCTION DOC] HELPER: Find a stack trace for the mutating use of a placeholder node. If the node has exactly one user,
    # return that user's stack_trace metadata; otherwise return the first stack trace on a copy_ user; else None.
    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target is torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the stored mutating-use stack trace of a PlaceholderInfo (may be None).
    return placeholder_info.mutating_use_stack_trace


def get_input_storage_mutation_info(
    gm: torch.fx.GraphModule,
) -> InputStorageMutationInfo:
    # [FUNCTION DOC] Scan a graph module for storage-rebinding ops (set_, shallow_copy_data_ variants) whose first argument is
    # directly a graph placeholder. Returns InputStorageMutationInfo with the placeholder indices (in placeholder
    # order) and the stack trace of the first such op. Ops on temporary views are ignored. No side effects.
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    placeholder_indices = {node: idx for idx, node in enumerate(placeholders)}
    storage_mutation_input_idxs: OrderedSet[int] = OrderedSet()
    stack_trace: str | None = None

    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target not in INPUT_STORAGE_MUTATION_TARGETS
        ):
            continue

        mutated_arg = node.args[0] if node.args else None
        # Only direct graph-input set_ rebinds the tensor object that lives
        # outside this graph. set_ on a temporary view/alias does not rebind the
        # input TensorImpl and is handled by the usual mutation checks.
        if (
            isinstance(mutated_arg, torch.fx.Node)
            and mutated_arg in placeholder_indices
        ):
            storage_mutation_input_idxs.add(placeholder_indices[mutated_arg])
            if stack_trace is None:
                stack_trace = node.meta.get("stack_trace", None)

    return InputStorageMutationInfo(storage_mutation_input_idxs, stack_trace)


def get_input_storage_mutation_reason(
    storage_mutation_info: InputStorageMutationInfo,
) -> str | None:
    # [FUNCTION DOC] Build a human readable skip reason from storage mutation info: None if no inputs were mutated, otherwise
    # a message with the count of instances, with the stack trace appended when present.
    storage_mutation_input_idxs = storage_mutation_info.input_idxs
    if not storage_mutation_input_idxs:
        return None

    msg = f"input storage mutation ({len(storage_mutation_input_idxs)} instances)"
    if storage_mutation_info.stack_trace:
        return f"{msg}. Found from:\n {storage_mutation_info.stack_trace}"

    return msg


def to_placeholder_info(placeholder_node: torch.fx.Node) -> PlaceholderInfo:
    # [FUNCTION DOC] Convert an fx node into a PlaceholderInfo. For placeholder nodes, recursively converts the users and finds the
    # mutating use stack trace; for other nodes (the users) no recursion is done, so recursion depth is one.
    name = placeholder_node.name
    stack_trace = placeholder_node.meta.get("stack_trace", None)
    users = []
    mutating_use_stack_trace = None
    # Only recurse to users once, since we only care about user's stack traces
    if placeholder_node.op == "placeholder":
        users = [to_placeholder_info(i) for i in placeholder_node.users]
        mutating_use_stack_trace = get_mutating_use_stack_trace_from_node(
            placeholder_node
        )

    return PlaceholderInfo(name, stack_trace, users, mutating_use_stack_trace)


def get_placeholder_info(graph: torch.fx.Graph) -> list[PlaceholderInfo]:
    # [FUNCTION DOC] Returns a list of PlaceholderInfo for every placeholder node of the graph, in graph order.
    return [
        to_placeholder_info(node) for node in graph.nodes if node.op == "placeholder"
    ]


def format_default_skip_message(reason: str) -> str:
    # [FUNCTION DOC] Returns the standard message text saying cudagraphs are skipped due to the given reason.
    return f"skipping cudagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: Sequence[PlaceholderInfo],
    mutation_indices: AbstractSet[int] | Sequence[int],
) -> str:
    # [FUNCTION DOC] Build the skip message for mutated inputs. Uses the first non-empty mutating-use stack trace among the placeholders
    # at the mutation indices; appends it to the message when found. Returns the message string.
    stack_trace: str | None = ""

    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    msg = format_default_skip_message(
        f"mutated inputs ({len(mutation_indices)} instances)"
    )
    if stack_trace:
        return f"{msg}. Found from : \n {stack_trace}"

    return msg


def check_for_mutation(
    func: WrappedFunction,
    inputs: list[InputType],
    is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> str | None:
    # [FUNCTION DOC] Check whether a wrapped function mutates inputs in a way that prevents cudagraphs. With cudagraph_trees on,
    # mutations of static inputs or tensors already recorded by a cuda graph are ignored; otherwise all mutated indices count.
    # Logs the indices at debug level. Returns None if no relevant mutation, else a skip-reason message.
    # doesn't work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices: Sequence[int] = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_cuda_graph_recorded_tensor(inputs[idx])  # type: ignore[arg-type]
            )
        ]
    else:
        mutation_indices = func.mutated_input_idxs

    static_inputs_log.debug(
        "check mutation static input indices: %s", func.static_input_idxs
    )
    static_inputs_log.debug("check mutation indices: %s", mutation_indices)

    return (
        get_mutation_stack_trace(func.placeholders, mutation_indices)
        if mutation_indices
        else None
    )


def _get_use_stack_trace(node: torch.fx.Node) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the first non-empty stack_trace metadata among the node's users, or None.
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None:
    # [FUNCTION DOC] Check device constraints for cudagraphs. Mutates the given mapping: removes meta entries, and cpu entries when cudagraph
    # partition is in use. Returns a skip message if a cpu node remains (with a user stack trace if known) or if there are
    # multiple devices or no cuda device; returns None when exactly one cuda device remains.
    # meta tensors are supported since there is no compute
    device_node_mapping.pop(torch.device("meta"), None)

    # dynamo cudagraph does not support graph partition
    if is_using_cudagraph_partition():
        # graph partition supports splitting on cpu op. So we can ignore cpu nodes.
        device_node_mapping.pop(torch.device("cpu"), None)

    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        msg = f"cpu device ({cpu_node.name})"
        if stack_trace := _get_use_stack_trace(cpu_node):
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        return format_default_skip_message(msg)

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "cuda"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping)
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_caching_allocator_for_cudagraphs() -> str | None:
    # [FUNCTION DOC] Returns a skip message if CUDA is available but the caching allocator is disabled, else None.
    """Skip cudagraphs when the CUDA/HIP caching allocator is disabled
    (via ``torch.cuda.caching_allocator_enable(False)`` or the env-var
    bypass ``PYTORCH_NO_(CUDA|HIP)_MEMORY_CACHING``). Cudagraph capture
    pools allocations through the caching allocator; with it bypassed,
    capture appears to succeed but pool tracking diverges (see
    check_memory_pool in cudagraph_trees.py), surfacing as 'storage data
    ptrs not allocated in pool ...' at replay time."""
    if (
        torch.cuda.is_available()
        # pyrefly: ignore [missing-attribute]
        and not torch._C._cuda_cudaCachingAllocator_is_enabled()
    ):
        return format_default_skip_message(
            "cudagraph capture requires the caching allocator; "
            "current allocator is uncached"
        )
    return None


def check_lowering_disable_cudagraph(
    device_node_mapping: dict[torch.device, torch.fx.Node],
) -> str | None:
    # [FUNCTION DOC] Lowering-time check: returns the allocator skip reason if any, else the device check result (None when OK).
    return (
        check_caching_allocator_for_cudagraphs()
        or check_multiple_devices_or_any_cpu_nodes(device_node_mapping)
    )


def log_cudagraph_skip_and_bump_counter(msg: str) -> None:
    # [FUNCTION DOC] Log a cudagraph skip warning and increment the inductor cudagraph_skips counter. Raises RuntimeError with the
    # message if the cudagraph_or_error config is set. Otherwise records cudagraph_skip_reason in the metrics context
    # when one is in progress.
    cudagraphs_log.warning(msg)
    counters["inductor"]["cudagraph_skips"] += 1

    if torch._inductor.config.triton.cudagraph_or_error:
        raise RuntimeError(msg)

    metrics_context = get_metrics_context()
    if metrics_context.in_progress():
        metrics_context.set("cudagraph_skip_reason", msg, overwrite=True)


@dataclasses.dataclass
class BoxedDeviceIndex:
    # [CLASS DOC] Mutable box holding an optional device index (value), shared between holders so it can be set later.
    value: int | None

    def set(self, device_idx: int | None) -> None:
        # [FUNCTION DOC] Set the boxed device index; raises AssertionError unless the argument is None or an int.
        if not (device_idx is None or isinstance(device_idx, int)):
            raise AssertionError(
                f"expected device_idx to be None or int, got {device_idx!r}"
            )
        self.value = device_idx


def check_for_mutation_ignore_cuda_graph_managed_tensor(
    gm: torch.fx.GraphModule,
    mutated_inputs: OrderedSet[str],
    mutated_input_idxs: OrderedSet[int],
    static_input_idxs: Sequence[int],
) -> str | None:
    # [FUNCTION DOC] Check mutated inputs for the graph-module path. With cudagraph_trees, mutations of static inputs are ignored and
    # a stack-trace based message is returned if others remain; otherwise any mutated input yields the default message.
    # Returns None when no relevant mutation.
    # KNOWN ISSUE: the mutated_inputs set is only used for its size in the non-tree path.
    default_msg = format_default_skip_message("mutated inputs")

    # doesn't work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        unique_idxs = OrderedSet(static_input_idxs)
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [idx for idx in mutated_input_idxs if idx not in unique_idxs]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None
        placeholders = get_placeholder_info(gm.graph)
        return get_mutation_stack_trace(placeholders, mutation_indices)

    else:
        has_mutation = len(mutated_inputs) != 0
        return None if not has_mutation else default_msg


def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> str | None:
    # [FUNCTION DOC] Returns the placeholder's own stack trace if non-empty, else the first non-empty stack trace of its users, else None.
    """
    Gets the first non-empty stack trace of a placeholder or its users.
    """
    if placeholder.stack_trace:
        return placeholder.stack_trace

    for user in placeholder.users:
        if user.stack_trace:
            return user.stack_trace

    return None


class CheckInvariantStatus(Enum):
    # [CLASS DOC] Enum of outcomes for the cudagraph invariant check: SUCCESS and three mismatch kinds (managed tensor pointers,
    # static input pointers, expected dead indices). str() gives a human readable description.
    # Check invariant succeeded
    SUCCESS = 1

    # Previously managed data pointers are not stable
    CudagraphManagedIdxMismatch = 2

    # Static tensor input addresses are not stable
    StaticInputIdxMismatch = 3

    # Expected dead indices before graph are live
    ExpectedDeadIndicesBeforeGraphMismatch = 4

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns a readable description for mismatch statuses; other members give the name and value.
        if self.name == "CudagraphManagedIdxMismatch":
            return "cudagraph managed tensor data pointer changed"
        elif self.name == "StaticInputIdxMismatch":
            return "static input data pointer changed"
        elif self.name == "ExpectedDeadIndicesBeforeGraphMismatch":
            return "expected dead indices before graph are live"
        else:
            return f"{self.name}: {self.value}"


def log_data_ptr_mismatch(
    placeholders: Sequence[PlaceholderInfo],
    inputs: list[InputType],
    recorded_data_ptr: Sequence[int | None],
    target_idxs: Sequence[int],
    mismatch: CheckInvariantStatus,
) -> str:
    # [FUNCTION DOC] Build an error message listing, for the target indices, inputs whose current data pointer differs from the recorded one,
    # with input name and stack trace. Raises AssertionError on length mismatch among inputs, recorded pointers and
    # placeholders, or if a target input is not a tensor. Returns the message string, starting with the mismatch text.
    """
    Logs the mismatch between input data pointers and recorded data pointers.
    This checks only idxs in target_idxs.
    """
    if not (len(inputs) == len(recorded_data_ptr) and len(inputs) == len(placeholders)):
        raise AssertionError(
            "length mismatch between inputs, recorded_data_ptr, and placeholders"
        )

    t_tensors = [inputs[i] for i in target_idxs]
    t_data_ptrs = [recorded_data_ptr[i] for i in target_idxs]
    error_msg = f"{mismatch}.\n"
    for i, (tensor, data_ptr) in enumerate(zip(t_tensors, t_data_ptrs)):
        if not isinstance(tensor, torch.Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(tensor)}")
        index = target_idxs[i]
        if tensor.data_ptr() != data_ptr:
            placeholder = placeholders[index]
            error_msg = (
                f"{error_msg}input name: {placeholder.name}. "
                f"data pointer changed from {data_ptr} to {tensor.data_ptr()}. "
                f"input stack trace: {get_placeholder_stack_trace(placeholder)}\n"
            )
    return error_msg


def maybe_warning_due_to_dynamic_shape(
    fn_cache: dict[tuple[int, ...], Callable[..., Any]],
    new_int_key: Any,
) -> bool:
    # [FUNCTION DOC] Warn if the number of recorded cudagraphs (cache size plus one) exceeds the dynamic shape warn limit config.
    # Returns True if a warning was logged, else False. new_int_key is unused.
    num_cudagraphs = len(fn_cache.keys()) + 1

    def warn_msg() -> str:
        # [FUNCTION DOC] HELPER: Builds the dynamic shape warning text including the distinct size count.
        return (
            "CUDAGraph supports dynamic shapes by recording a new graph for each "
            "distinct input size. Recording too many CUDAGraphs may lead to "
            f"extra overhead. We have observed {num_cudagraphs} distinct sizes. "
            "Please consider the following options for better performance: "
            "a) padding inputs to a few fixed number of shapes; or b) set "
            "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
            "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
            "to silence this warning."
        )

    if (
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
        and num_cudagraphs
        > torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
    ):
        cudagraphs_log.warning(warn_msg())
        return True

    return False


@dataclasses.dataclass(frozen=True)
class CudagraphCachedInfo:
    # [CLASS DOC] Frozen info needed to realign inputs: placeholders, stack traces, user visible output indices and
    # cudagraph failure reasons.
    """
    Info needed to realign inputs
    """

    placeholders: Sequence[PlaceholderInfo]
    stack_traces: list[str | None]
    user_visible_output_idxs: Sequence[int]
    cudagraph_fail_reasons: list[str]


@dataclasses.dataclass(frozen=True)
class CudagraphMetadata:
    # [CLASS DOC] Frozen metadata for recording a CUDA graph: placeholders, static and mutated input index sets, stack traces,
    # user visible output indices and a mapping of constant names to tensors.
    """
    Metadata for recording a CUDA graph.
    """

    placeholders: Sequence[PlaceholderInfo]
    static_input_idxs: OrderedSet[int]
    mutated_input_idxs: OrderedSet[int]
    stack_traces: list[str | None]
    user_visible_output_idxs: OrderedSet[int]
    constants: dict[str, torch.Tensor]


def get_partition_cudagraph_metadata(
    partition_map: GraphPartitionMap,
    metadata: CudagraphMetadata,
) -> CudagraphMetadata:
    # [FUNCTION DOC] Convert graph level cudagraph metadata to a partition's metadata using the partition's input and output index
    # mappings and constant names. Inputs not coming from graph inputs get dummy placeholders. For each output the stack
    # trace comes from a user visible graph output if any, else the first mapped one; None if unmapped. Returns new CudagraphMetadata.
    """
    Convert the cudagraph metadata at the graph level to the graph partition level,
    given the graph partition info (i.e., mapping from partition input/output index
    to graph input/output index).
    """

    partition_placeholders = []
    partition_static_input_idxs: OrderedSet[int] = OrderedSet()
    partition_mutated_input_idxs: OrderedSet[int] = OrderedSet()
    for partition_input_idx, graph_input_idx in enumerate(
        partition_map.input_index_mapping
    ):
        if graph_input_idx in metadata.static_input_idxs:
            partition_static_input_idxs.add(partition_input_idx)

        if graph_input_idx in metadata.mutated_input_idxs:
            partition_mutated_input_idxs.add(partition_input_idx)

        if graph_input_idx is not None:
            placeholder = metadata.placeholders[graph_input_idx]
        else:
            # create a dummy placeholder info since this partition input is not a graph input
            placeholder = PlaceholderInfo(
                name=f"partition_{partition_map.id}_placeholder_{partition_input_idx}",
                stack_trace=None,
                users=[],
                mutating_use_stack_trace=None,
            )
        partition_placeholders.append(placeholder)

    partition_stack_traces = []
    partition_user_visible_output_idxs: OrderedSet[int] = OrderedSet()
    # Graph output metadata must be remapped to the partition output indices
    # passed to cudagraphify.
    for partition_output_idx, graph_output_idxs in enumerate(
        partition_map.output_index_mapping
    ):
        if not graph_output_idxs:
            partition_stack_traces.append(None)
            continue

        user_visible_graph_output_idxs = [
            graph_output_idx
            for graph_output_idx in graph_output_idxs
            if graph_output_idx in metadata.user_visible_output_idxs
        ]
        stack_trace_idx = (
            user_visible_graph_output_idxs[0]
            if user_visible_graph_output_idxs
            else graph_output_idxs[0]
        )
        partition_stack_traces.append(metadata.stack_traces[stack_trace_idx])
        if user_visible_graph_output_idxs:
            partition_user_visible_output_idxs.add(partition_output_idx)

    partition_constants = {
        name: metadata.constants[name] for name in partition_map.constant_names
    }

    return CudagraphMetadata(
        partition_placeholders,
        partition_static_input_idxs,
        partition_mutated_input_idxs,
        partition_stack_traces,
        partition_user_visible_output_idxs,
        partition_constants,
    )


def collect_cuda_data_ptrs(obj: object) -> OrderedSet[int]:
    # [FUNCTION DOC] HELPER: Debug helper returning the set of data pointers of CUDA tensors in the object (including subclass inner
    # tensors, skipping fake, meta and non-cuda tensors). Non-tensors give an empty set; pointer errors are ignored.
    """Debug helper that collects the data pointers of all CUDA tensors in the object."""
    if not isinstance(obj, torch.Tensor):
        return OrderedSet()

    ptrs: OrderedSet[int] = OrderedSet()
    for base in get_plain_tensors(obj, out=[]):
        if type(base) is not torch.Tensor:
            continue
        if is_fake(base) or base.is_meta or base.device.type != "cuda":
            continue
        try:
            ptrs.add(base.data_ptr())
        except Exception:
            pass
    return ptrs
