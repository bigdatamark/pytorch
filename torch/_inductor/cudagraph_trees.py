# [MODULE DOC] Implements CUDA graph trees for Inductor: a manager that records, replays and warms
# up CUDA graphs that all share one private memory pool, organized as a tree of recordings keyed by
# the liveness pattern of earlier outputs. Main contents: cudagraphify_impl and cudagraphify (entry
# points that wrap a compiled model), CUDAGraphTreeManager (per-device tree controller and state
# machine), CUDAGraphNode (one recorded graph), CUDAWarmupNode (a non-recorded eager run in the
# pool), StorageWeakRefWrapper (storage liveness tracking), and helpers for allocator checkpointing,
# memory pool validation, and generation tracking. Import-time side effects: creates the module
# logger, a thread-local holder plus a process-wide capture lock, registers per-device containers in
# autograd thread-local storage, and defines stub AllocatorState and cached-tensor toggle when CUDA
# is not built.
"""
CUDA graph trees are a safety abstraction over CUDAGraphs, similar to make_graph_callables,
which share the same memory pool.  Sharing a memory pool is an extremely
important optimization when chaining multiple CUDA graphs together, as it
prevents you from needing to copy intermediate tensors from one graph to the
next, and reduces overall memory usage by allowing dead memory from the first
pool to be reused in the second.

The standard graph/make_graph_callables support sharing memory pool, but
with a lot of caveats.  CUDA graph trees remove these restrictions:

* Previously, if you recorded graphs A, B, you had to replay A, B in that
  order.  With CUDA graph trees, after replaying A, you can change your
  mind and record/replay a different graph B'; we will support efficient
  execution of both A, B and A, B', using only max(mem(A, B), mem(A, B')).  In
  other words: we support arbitrary trees of CUDA graph operations, not just
  sequences (this is why this feature is called CUDA graph trees.)

* Previously, if you executed graph A, some non-CUDA graph code, and then
  graph B, after executing graph B, it was not safe to retain any references
  to intermediates produced by A.  With CUDA graph trees, we track if any
outputs of graph A are still live by the time graph B is run, and make
  sure graph B doesn't clobber there memory when reusing the CUDA graphs
  pool.  You'll get a separate recording of B depending on what tensors
  stay live or dead.

CUDA graph trees are flexible enough to be used in Dynamo across graph breaks,
which is their primary use case.

The ability to switch from replay to record is fairly nontrivial: remember that
when you replay a CUDA graph, you only replay CUDA operations; no CPU side state
is updated.  In particular, the CPU-side book-keeping for the allocator is not
reconstructed.  However, to record a new child CUDA graph, we must restore this
book-keeping.  This is what checkpoint pool state is used for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Any, cast, TYPE_CHECKING, TypeVar

import torch.fx
from torch import Tensor
from torch._custom_class_base import CustomClassBase
from torch._dynamo.callback import CallbackTrigger
from torch._dynamo.graph_bytecode_inputs import (
    CURRENT_STREAM_INDEX,
    set_external_object_by_index,
)
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import counters, dynamo_timed, preserve_rng_state
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    ControlFlowOpWarmupDispatchMode,
    CUDAGraphCaptureControlFlowOpDispatchMode,
)
from torch._inductor.compile_fx import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    copy_strided_storage_,
    get_expanded_dims,
    get_input_idxs_to_check,
    index_expanded_dims,
    remove_unaligned_input_idxs,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    check_for_mutation,
    CheckInvariantStatus,
    collect_cuda_data_ptrs,
    FunctionID,
    log_cudagraph_skip_and_bump_counter,
    log_data_ptr_mismatch,
    maybe_warning_due_to_dynamic_shape,
    ModelType,
    OutputType,
    PlaceholderInfo,
    WrappedFunction,
)
from torch._library.opaque_object import is_custom_class_obj
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils.weak import TensorWeakRef


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator, Sequence

    from torch._guards import CompileId
    from torch._inductor.utils import InputType
    from torch.cuda import _POOL_HANDLE
    from torch.types import _bool

StorageWeakRefPointer = int
StorageDataPtr = int
NBytes = int
S = TypeVar("S", bound="StorageWeakRefWrapper")


if torch.backends.cuda.is_built():
    from torch._C import (
        _cuda_CUDAAllocator_AllocatorState as AllocatorState,
        _set_cached_tensors_enabled,
    )
else:

    class AllocatorState:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Placeholder AllocatorState used only when PyTorch is built without
        # CUDA; has no state or behavior.
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        # [FUNCTION DOC] HELPER: No-op stand-in for the C++ cached-tensors toggle when CUDA is not
        # built. Takes an enabled flag, ignores it, returns nothing.
        pass


log = torch._logging.getArtifactLogger(__name__, "cudagraphs")


def format_inputs_log(inputs: list[Any]) -> str:
    # [FUNCTION DOC] Builds a one-line human readable summary of a list of graph inputs for debug
    # logging. Tensors are shown with size, stride, dtype and hex data pointer, None as None, other
    # values by type name and value. Only the first 10 items are shown, followed by a count of the
    # omitted items. Returns the string for an empty list as empty brackets. No side effects.
    def format_item(i: int, inp: Any) -> str:
        # [FUNCTION DOC] HELPER: Formats one input (index and value) as a string: tensor details,
        # None, or type name plus value.
        if isinstance(inp, torch.Tensor):
            return (
                f"[{i}]: Tensor(size={list(inp.size())}, stride={inp.stride()}, "
                f"dtype={inp.dtype}, data_ptr=0x{inp.data_ptr():X})"
            )
        elif inp is None:
            return f"[{i}]: None"
        else:
            return f"[{i}]: {type(inp).__name__}({inp})"

    n = len(inputs)
    if n == 0:
        return "[]"

    # Only display the first 10 inputs if the input list is too long
    max_items = 10
    parts = [format_item(i, inp) for i, inp in enumerate(inputs[:max_items])]
    if n > max_items:
        parts.append(f"... ({n - max_items} more items)")

    return ", ".join(parts)


from . import config


@dataclasses.dataclass(frozen=True)
class GraphID:
    # [CLASS DOC] Immutable (frozen dataclass) identifier for one CUDA graph recording; holds an
    # integer id taken from a counter. Warmup nodes use negative ids (see the warmup counter in the
    # tree manager) so they never collide with recorded graph ids.
    "Unique counter of a cuda graph recording"

    id: int


def clear_cublass_cache() -> None:
    # [FUNCTION DOC] Frees the cuBLAS workspaces held by PyTorch so later matmul workspace
    # allocations land in the currently active (possibly private graph) pool instead of persisting
    # from a previous run. Takes no arguments, returns nothing; side effect is clearing global
    # cuBLAS workspace state.
    """
    Cublas keeps a persistent workspace allocation for running matmuls. This poses a problem for
    doing warmup within a CUDAGraph private pool because we do not want persistent allocations from
    one one run to the next. When we begin a new run of a cudagraphs path (generation), all tensors
    from the previous generation are freed. This frees them the memory pool, but not elsewhere.
    A tensor in the cublas workspace would continue to be in use the workspace but would also get allocated
    in the next run. The memory would be in use in two places.

    To solve this, we clear cublas caches before and after warming up or recording. If a workspace is required
    it will be allocated to the cudagraph private pool and accounted for in the allocator for the duration of the
    program. There is no overhead to this on replay since cudagraphs removes allocation overhead.
    """
    torch._C._cuda_clearCublasWorkspaces()


@contextlib.contextmanager
def clear_cublas_manager() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that clears the cuBLAS workspace cache on entry and again on
    # exit (even if the body raises). Used around warmup and recording. Yields nothing.
    "Context manager around clearing cublas caches that will clear on enter and exit"
    clear_cublass_cache()
    try:
        yield
    finally:
        clear_cublass_cache()


@contextlib.contextmanager
def disable_conv_cache_emptying() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that turns off the cuDNN convolution benchmark behavior of
    # emptying the cache, restoring the previous setting on exit (even on error). Mutates a global
    # cuDNN flag while active. Yields nothing.
    prev = torch._C._cuda_get_conv_benchmark_empty_cache()
    torch._C._cudnn_set_conv_benchmark_empty_cache(False)
    try:
        yield
    finally:
        torch._C._cudnn_set_conv_benchmark_empty_cache(prev)


@contextlib.contextmanager
def enable_history_recording() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that enables CUDA caching allocator memory history recording if
    # it is not already on, and turns it back off on exit only if this manager enabled it. Used to
    # get allocation stack traces when debugging pool leaks. Yields nothing.
    "Turns on history recording in the CUDA Caching Allocator"
    enabled = torch._C._cuda_isHistoryEnabled()
    try:
        if not enabled:
            torch.cuda.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.cuda.memory._record_memory_history(None)


def get_history_recording() -> AbstractContextManager[None]:
    # [FUNCTION DOC] Returns a context manager for allocator history recording: the real recorder
    # when the cudagraph_trees_history_recording config is on, otherwise a null context. KNOWN
    # ISSUE: the in-code TODO says this should be removed because it prevents cleanup.
    # TODO - remove, prevents cleanup
    if not config.triton.cudagraph_trees_history_recording:
        return contextlib.nullcontext()
    return enable_history_recording()


class TreeManagerContainer:
    # [CLASS DOC] Owns the lifetime of one CUDAGraphTreeManager for a single device, one container
    # per device. Keeps a strong reference to the manager while any wrapped function (cudagraphify
    # result) is alive and drops it when the last such function is garbage collected. State:
    # tree_manager (the manager or None), live_cudagraphify_fns (count of live wrapped functions),
    # device_index, live_storages_count and graph (only meant for keeping the pool alive while
    # outputs outlive functions; currently unused because the code that sets them is commented out),
    # and a lock guarding all of them. Key methods: get_tree_manager (lazy create),
    # add_strong_reference, finalize_cudagraphify_fn.
    """
    Manages the lifetime of the tree manager. Like `PrivatePool` in cuda caching allocator,
    the tree and its corresponding memory pool should be kept alive as long as any outstanding
    graph or tensor which is an output of a graph remains alive.

    There is a single tree manager container per device.

    The lifecycle of a tree_manager is:
    -  Is constructed, no graph, no fns, no tensors
    -  Tree manager is fetched, resulting in tree manager being allocated
    -  We generate a bunch of functions, calling add_strong_reference
    -  These functions die, calling finalize_reference
    -  When all the functions die, we finalize_tree_manager.

    TODO: in the future, we would like to do the following once storage weak refs land
    -  We look for all the live storages and add references to THOSE
    -  We count as storages die
    -  All the storages are dead, we deallocate the tree manager
    """

    def __init__(self, device_index: int) -> None:
        # [FUNCTION DOC] Initializes an empty container for the given device index: no tree manager,
        # zero live function count, zero live storage count, no graph, and a fresh lock.
        # This class keeps a strong reference to tree_manager,
        # but upon all other strong references to the tree_manager will reset it to None.
        # We need a strong reference so that we can still access its attributes upon cleanup.
        self.tree_manager: CUDAGraphTreeManager | None = None

        # Number of outstanding references to the current tree manager
        self.live_cudagraphify_fns = 0

        self.device_index = device_index

        # Following two objects are only set in the case that Tensor outputs outlive
        # the cudagraphify_fns. Reference to the Graph is needed to keep the private pool from
        # deallocation.
        self.live_storages_count = 0
        self.graph: torch.cuda.CUDAGraph | None = None

        self.lock = threading.Lock()

    def _finalize_tensor(self) -> None:
        # [FUNCTION DOC] HELPER: Called when a tracked output storage dies. Under the lock,
        # decrements the live storage count; when it reaches zero, releases the pinned graph and, if
        # no wrapped functions are alive either, drops the tree manager. KNOWN ISSUE: nothing
        # currently registers this callback (the registering code is commented out), so it is
        # effectively dead code.
        with self.lock:
            self.live_storages_count -= 1
            if self.live_storages_count == 0:
                self.graph = None

                # manager was used again after existing cleanup,
                # we shouldn't set it to None
                if self.live_cudagraphify_fns == 0:
                    self.tree_manager = None

    def finalize_cudagraphify_fn(self) -> None:
        # [FUNCTION DOC] Weakref finalizer callback run when a wrapped function dies. Under the
        # lock, decrements the live function count and, if it hits zero, drops the tree manager via
        # the internal finalize helper.
        with self.lock:
            self.live_cudagraphify_fns -= 1
            if self.live_cudagraphify_fns == 0:
                self._finalize_tree_manager()

    def _finalize_tree_manager(self) -> None:
        # [FUNCTION DOC] HELPER: Drops the strong reference to the tree manager. Must be called with
        # the container lock held; raises an assertion error otherwise. The remainder is commented-
        # out future logic about pinning live storages.
        if not self.lock.locked():
            raise AssertionError("expected lock to be held")
        self.tree_manager = None

        # TODO - when issue #91395 is landed, we can set a weakref on
        # storages and trigger a deallocation when all outputs of the
        # cudagraph are dead.

        # live_storages = list(
        #     tree_manager.live_cudagraph_pool_storages_in_curr_execution()
        # )

        # # Maintain reference to graph to keep tensors alive
        # assert len(tree_manager.roots) > 0, "expected at least one use"
        # root = next(tree_manager.get_roots())
        # self.graph = root.graph
        # seen_storages = set()
        # for stor in live_storages:
        #     if stor in seen_storages:
        #         continue
        #     seen_storages.add(stor)
        #     self.live_storages_count += 1
        # .   weakref.finalize(stor, self._finalize_tensor)

    def add_strong_reference(self, fn: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Registers a wrapped function with this container: increments the live
        # function count under the lock and installs a weakref finalizer on the function that will
        # decrement it. Takes the function; returns nothing.
        with self.lock:
            self.live_cudagraphify_fns += 1

        weakref.finalize(fn, self.finalize_cudagraphify_fn)

    def get_tree_manager(self) -> CUDAGraphTreeManager:
        # [FUNCTION DOC] Returns the tree manager for this device, creating a CUDAGraphTreeManager
        # (which performs CUDA setup) under the lock if none exists. Never returns None.
        with self.lock:
            if self.tree_manager is None:
                self.tree_manager = CUDAGraphTreeManager(self.device_index)
            return self.tree_manager


def _initialize_tls(local: threading.local) -> None:
    # [FUNCTION DOC] HELPER: Initializes thread-local storage for tree managers: sets an empty
    # mapping of containers and a mapping of per-device locks (created on demand) on the given
    # thread-local object, and stashes both objects in C++ autograd thread-local storage so threads
    # spawned by autograd can see the same objects. Side effects: mutates the argument and global
    # TLS registry.
    # one tree manager per device
    local.tree_manager_containers = {}
    local.tree_manager_locks = defaultdict(threading.Lock)

    # We need to register this as an object that will be copied over as TLS when new
    # threads are created in autograd
    torch._C._stash_obj_in_tls("tree_manager_containers", local.tree_manager_containers)
    torch._C._stash_obj_in_tls("tree_manager_locks", local.tree_manager_locks)


local = threading.local()
_initialize_tls(local)


# CUDA graph capture and private-pool warmup are process-wide CUDA states.  Keep
# them serialized across Python threads so a synchronize or allocator-pool switch
# in one thread does not invalidate another thread's capture.
graph_capture_lock = threading.Lock()


# only incremented by user call of mark_step_begin
class MarkStepBox:
    # [CLASS DOC] Holder for the process-wide step counter that is changed only by an explicit user
    # call to mark a step beginning. Class attribute mark_step_counter starts at zero and is
    # decremented (negative direction) so it cannot be confused with the dynamo generation counter,
    # which counts upward.
    mark_step_counter = 0


def mark_step_begin() -> None:
    # [FUNCTION DOC] Public API: signals that a new inference or training iteration is about to
    # begin, by decrementing the global mark step counter. This makes the current generation differ,
    # allowing earlier outputs to be overwritten. No arguments, no return value.
    "Indicates that a new iteration of inference or training is about to begin."

    # iterate down to distinguish from GenerationTracking counter
    MarkStepBox.mark_step_counter -= 1


def mark_warmup_incomplete() -> None:
    # [FUNCTION DOC] Public API: asks that the currently warming-up CUDA graph tree function be
    # warmed up again. Looks up the existing manager for the current device without creating one and
    # forwards the request; does nothing if there is no manager or no function is currently warming
    # up.
    """Request another warmup for the active CUDA Graph Trees function.

    This is a no-op unless called synchronously from a function currently running
    in CUDA Graph Trees warmup.
    """
    manager = get_manager(create_if_none_exists=False)
    if manager is not None:
        manager.mark_warmup_incomplete()


def reset_cudagraph_trees() -> None:
    # [FUNCTION DOC] Shuts down and clears all cudagraph trees for the calling thread: for every
    # device whose container holds a manager, takes its lock and calls shutdown; then disables
    # cached tensors, clears the container mapping and resets the mark step counter to zero. Returns
    # early if the thread-local storage was never set up for this thread. Side effects: global TLS
    # and counter state.
    "Clear all cudagraph trees"
    # see shutdown below for why this is necessary
    # Note: get_obj may fail if called from a thread that wasn't spawned by
    # autograd (e.g., test threads). In that case, there's nothing to reset.
    if not torch._C._is_key_in_tls("tree_manager_containers"):
        # TLS not set up for this thread, nothing to reset
        return
    container_dict = get_obj(local, "tree_manager_containers")
    locks_dict = get_obj(local, "tree_manager_locks")
    for device, lock in locks_dict.items():
        with lock:
            container = container_dict.get(device)
            if not container or not container.tree_manager:
                continue

            container.tree_manager.shutdown()

    _set_cached_tensors_enabled(False)
    container_dict.clear()

    MarkStepBox.mark_step_counter = 0


def get_obj(local: Any, attr_name: str) -> Any:
    # [FUNCTION DOC] Fetches a thread-local object by attribute name. Returns the attribute from the
    # given thread-local if present; otherwise, if the key is not in C++ TLS, initializes the
    # thread-local storage and returns the new attribute; otherwise returns the object stashed in
    # C++ TLS (for threads created by autograd). May initialize TLS as a side effect.
    if hasattr(local, attr_name):
        return getattr(local, attr_name)
    if not torch._C._is_key_in_tls(attr_name):
        _initialize_tls(local)
        return getattr(local, attr_name)
    return torch._C._get_obj_in_tls(attr_name)


def get_container(device_index: int) -> TreeManagerContainer:
    # [FUNCTION DOC] Returns the TreeManagerContainer for the device index, creating and registering
    # it under the per-device lock if it does not exist yet.
    container_dict = get_obj(local, "tree_manager_containers")
    lock = get_obj(local, "tree_manager_locks")[device_index]

    with lock:
        if device_index not in container_dict:
            container_dict[device_index] = TreeManagerContainer(device_index)

        return container_dict[device_index]


def get_manager(
    device_index: int | None = None, create_if_none_exists: bool = True
) -> CUDAGraphTreeManager | None:
    # [FUNCTION DOC] Returns the CUDAGraphTreeManager for a device, or None. If device_index is
    # None, uses the current device, returning None when CUDA is not initialized. If
    # create_if_none_exists is true, creates the manager as needed; otherwise returns the existing
    # manager or None without creating a container.
    if device_index is None:
        if not torch.cuda.is_initialized():
            return None
        device_index = torch.cuda.current_device()

    if create_if_none_exists:
        return get_container(device_index).get_tree_manager()

    container = get_obj(local, "tree_manager_containers").get(device_index)
    return None if container is None else container.tree_manager


def is_cudagraph_capture_sizes(int_key: int | tuple[int, ...]) -> bool:
    # [FUNCTION DOC] Returns true if cudagraph capture is allowed for the given dynamic shape key
    # (an int or tuple of ints): true when no capture size restriction is configured or the key is
    # in the configured sizes.
    """
    Returns true if all dynamic shapes should be captured or the dynamic shape
    int_key should be captured.
    """
    return (
        config.triton.cudagraph_capture_sizes is None
        or int_key in config.triton.cudagraph_capture_sizes
    )


def cudagraphify_impl(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int],
    *args: Any,
    **kwargs: Any,
) -> ModelType:
    # [FUNCTION DOC] Inductor entry point that wraps a compiled model so it is cudagraphified lazily
    # per distinct combination of integer (symint) inputs. Takes the model, example inputs (used
    # only to find which positions are ints, then deleted), the static input indices, and extra args
    # forwarded to cudagraphify. Returns a callable which, on each call, computes the int key, calls
    # the plain model if the key is not in the configured capture sizes, reuses a cached
    # cudagraphified function for the key, or otherwise records one. On first recording it fixes up
    # input alignment (copying misaligned inputs, adjusting static indices), logs, may warn about
    # dynamic shapes, and caches the function wrapped with alignment handling, returning the
    # recorded outputs of that first call.
    fn_cache: dict[tuple[int, ...], Callable[..., Any]] = {}

    # Detect int inputs: we need to index on these
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    has_warn = False

    del inputs

    def deferred_cudagraphify(inputs: list[InputType]) -> OutputType:
        # [FUNCTION DOC] The returned wrapper: given the input list, derives the integer key,
        # dispatches to the plain model, a cached cudagraph function, or records a new one as
        # described for the enclosing function. Returns the model outputs. Side effects: updates the
        # function cache and the nonlocal warning flag, logs, and may consume or copy inputs.
        nonlocal has_warn

        int_key = get_ints(inputs)

        if not is_cudagraph_capture_sizes(int_key):
            return model(inputs)

        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)
        compile_id = kwargs.get("compile_id", "")
        if int_key is None:
            log.info(
                "[%s] Recording cudagraph tree for graph without symints", compile_id
            )
        else:
            log.info(
                "[%s] Recording cudagraph tree for symint key %s", compile_id, int_key
            )

        if not has_warn:
            has_warn = maybe_warning_due_to_dynamic_shape(fn_cache, int_key)

        # first get indices we need to check to align, then update our static inputs,
        # and finally copy
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        copy_misaligned_inputs(inputs, check_input_idxs)

        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
        # cudagraph will already clones input locally, no need to copy back
        mutated_input_idxs: OrderedSet[int] = OrderedSet()
        fn = align_inputs_from_check_idxs(
            fn, inputs_to_check=check_input_idxs, mutated_input_idxs=mutated_input_idxs
        )

        fn_cache[int_key] = fn

        return out

    return deferred_cudagraphify


@contextlib.contextmanager
def dynamo_timed_cudagraph(
    name: str,
    compile_id: CompileId | None,
    mode: CompilationMode | None,
) -> Generator[Any, None, None]:
    # [FUNCTION DOC] HELPER: Context manager wrapping dynamo_timed so cudagraph runtime work is
    # logged as a compile event and accumulated into the runtime cudagraphify time column; the
    # backward flag is derived from the compilation mode. Takes a name, compile id and mode; yields
    # nothing. Use only for time that counts as compiler caused runtime overhead.
    """
    Makes usages of dynamo_timed in this file less verbose. NOTE: This CM sums
    all durations into a single column in the dynamo_compile table. Use only if
    you consider the timed region to be part of the runtime overhead associated
    with the compiler.
    """
    with dynamo_timed(
        name,
        log_pt2_compile_event=True,
        compile_id=compile_id,
        is_backward=mode == CompilationMode.BACKWARD,
        dynamo_compile_column_us="runtime_cudagraphify_time_us",
    ):
        yield


def cudagraphify(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    stack_traces: StackTraces | None = None,
    constants: tuple[torch.Tensor, ...] = (),
    placeholders: tuple[PlaceholderInfo, ...] = (),
    mutated_input_idxs: tuple[int, ...] = (),
    kernel_free_cudagraph: bool = False,
    user_visible_output_idxs: tuple[int, ...] = (),
    compile_id: CompileId | None = None,
) -> tuple[ModelType, OutputType]:
    # [FUNCTION DOC] Records or warms up a model under CUDA graph trees for a device and returns
    # (callable, outputs). Determines the compilation mode from the backward and inference flags
    # (raises an assertion error if both are set), fetches the device tree manager, and delegates to
    # its add_function with all metadata (static input indices, stack traces, constants,
    # placeholders, mutated inputs, kernel-free flag, user-visible output indices, compile id). The
    # first run happens inside add_function, so the returned outputs are those of that first
    # invocation.
    if is_backward and is_inference:
        raise AssertionError("expected not (is_backward and is_inference)")
    mode = (
        CompilationMode.BACKWARD
        if is_backward
        else (CompilationMode.INFERENCE if is_inference else CompilationMode.FORWARD)
    )

    with dynamo_timed_cudagraph("cudagraphify.get_container", compile_id, mode):
        manager = get_container(device_index).get_tree_manager()

    return manager.add_function(
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
        constants,
        placeholders,
        mutated_input_idxs,
        kernel_free_cudagraph,
        user_visible_output_idxs,
        compile_id,
    )


class StorageWeakRefWrapper:
    # [CLASS DOC] Weak reference to a storage that also remembers the storage data pointer and knows
    # when the storage is effectively dead. Uses slots; state: ref (a storage weak ref), the
    # recorded data pointer, and an optional extra_ref_check callable used for cached output tensors
    # whose own references must be discounted when counting uses. Calling the object returns the
    # storage cdata pointer if alive, else None. Key methods: expired, swap_weakref, data_ptr (valid
    # even after expiry), remove_extra_reference.
    """
    Wrapper around a storage weak ref. Will deallocate it upon expiration if invoked.
    """

    __slots__ = ["ref", "_data_ptr", "extra_ref_check"]

    storage_ref: StorageWeakRef | None

    def __init__(
        self,
        inp: Tensor | UntypedStorage,
        extra_ref_check: Callable[[], bool] | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a wrapper from a tensor or an untyped storage (raises an assertion
        # error for any other type): takes the untyped storage, creates a weak reference to it,
        # records its data pointer, and stores the optional extra reference check.
        """
        extra_ref_check is an additional check we need to run to check if the
        weak ref has expired. in checking storage use count we assume extra_ref_check
        will hold an additional reference to the storage.
        """
        if isinstance(inp, Tensor):
            stor = inp.untyped_storage()
        else:
            if not isinstance(inp, UntypedStorage):
                raise AssertionError(f"expected UntypedStorage, got {type(inp)}")
            stor = inp
        self.ref = StorageWeakRef(stor)
        self._data_ptr = stor.data_ptr()
        self.extra_ref_check = extra_ref_check

    @classmethod
    def from_weakref_and_data_ptr(
        cls: type[StorageWeakRefWrapper],
        cdata: Any,
        data_ptr: int,
        extra_ref_check: Callable[[], bool] | None = None,
    ) -> StorageWeakRefWrapper:
        # [FUNCTION DOC] Alternate constructor: creates an instance without running the normal
        # initializer, from a raw weak-ref handle, a known data pointer and an optional extra check.
        # Returns the new wrapper.
        instance = cls.__new__(cls)
        instance._data_ptr = data_ptr
        instance.ref = StorageWeakRef.from_weakref(cdata)
        instance.extra_ref_check = extra_ref_check
        return instance

    def __call__(self) -> StorageWeakRefPointer | None:
        # [FUNCTION DOC] Returns the underlying storage weak pointer (cdata) if the storage is not
        # expired, else None.
        if self.expired():
            return None

        return self.ref.cdata

    def swap_weakref(self, cdata: Any) -> None:
        # [FUNCTION DOC] Replaces the tracked storage pointer with a new weak-ref handle: releases
        # the old weak reference and installs the new one. Used when outputs are reconstructed or
        # storages cloned. Mutates the wrapper in place.
        self.ref.__del__()
        self.ref.cdata = cdata

    def data_ptr(self) -> int:
        # [FUNCTION DOC] Returns the data pointer recorded at construction, even if the storage has
        # already expired.
        "NB: returns the data ptr even if the storage has expired"
        return self._data_ptr

    def remove_extra_reference(self) -> None:
        # [FUNCTION DOC] Clears the extra reference check so expiry is determined purely by the
        # storage use count.
        self.extra_ref_check = None

    def expired(self) -> bool:
        # [FUNCTION DOC] Returns whether the tracked storage is dead. If an extra reference check
        # exists and reports that outside references remain, returns false. Otherwise reads the
        # storage use count, subtracts two when an extra check exists (the Python storage object and
        # the cached tensor), raises an assertion error if the count would be negative, and returns
        # true when it is zero.
        if self.extra_ref_check is not None and not self.extra_ref_check():
            return False

        stor_count = torch._C._storage_Use_Count(self.ref.cdata)
        if self.extra_ref_check is not None:
            # if extra_ref_check is not None we expect two additional references:
            #  - one from the Python storage object
            #  - one from the cached Tensor
            stor_count -= 2
        if stor_count < 0:
            raise AssertionError(f"expected stor_count >= 0, got {stor_count}")
        return stor_count == 0

    def __repr__(self) -> str:
        # [FUNCTION DOC] Debug string with the data pointer and whether the weak reference is alive
        # or dead.
        if self.ref is None or self.ref.expired():
            return f"StorageWeakRefWrapper to {self.data_ptr()}; dead"
        else:
            return f"StorageWeakRefWrapper to {self.data_ptr()}; alive"


def is_live(weak_ref: StorageWeakRefWrapper | None) -> bool:
    # [FUNCTION DOC] Returns true if the given wrapper is not None and its storage is still alive.
    return maybe_deref(weak_ref) is not None


def maybe_deref(
    weak_ref: StorageWeakRefWrapper | None,
) -> tuple[StorageWeakRefPointer, int] | None:
    # [FUNCTION DOC] Returns a pair of (storage weak pointer, recorded data pointer) when the
    # wrapper exists and its storage is alive, otherwise None. The recorded data pointer need not
    # equal the data pointer of the live storage object.
    if weak_ref is None:
        return None
    r = weak_ref()
    if r is None:
        return None
    # NB: r.data_ptr() does not necessarily equal weak_ref.data_ptr()
    return r, weak_ref.data_ptr()


@contextlib.contextmanager
def _use_cuda_memory_pool_manager(
    device: int, mem_pool: tuple[int, int], stream: torch.cuda.Stream
) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that routes all allocations on the current thread to
    # the given CUDA graph private memory pool on the given device and stream. On entry it
    # synchronizes the device, makes the stream wait for the current stream, switches stream and
    # device, and begins allocating the current thread to the pool; on exit it ends the pool
    # allocation, releases the pool reference, and makes the original stream wait for the work. The
    # pool must already exist; no reference to it is held. Yields nothing.
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.device(device):
        # Begin allocate to mem pool for all memory allocation on the current thread.
        # This is thread safe since a thread can only warmup or record 1 cudagraph
        # at the same time.
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, mem_pool)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateToPool(device, mem_pool)
            torch._C._cuda_releasePool(device, mem_pool)

    torch.cuda.current_stream().wait_stream(stream)


@contextlib.contextmanager
def _update_current_stream_external_object() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the dynamo external object at the current
    # stream slot to the actual current CUDA stream, so custom ops that record or wait on events
    # during capture or warmup use the capture stream. Yields nothing; does not restore the previous
    # object.
    """Update the external object registry so custom ops see the capture stream.

    During cudagraph recording/warmup the current stream differs from the
    trace-time default stream.  The external object at CURRENT_STREAM_INDEX
    must reflect the actual current stream so that custom ops (e.g. event
    record/wait) executed during capture use the right stream.
    """
    set_external_object_by_index(CURRENT_STREAM_INDEX, torch.cuda.current_stream())
    yield


def map_to_ref(t: Tensor | None) -> StorageWeakRefWrapper | None:
    # [FUNCTION DOC] HELPER: Maps a tensor to a StorageWeakRefWrapper, or None for None. Raises an
    # assertion error if given anything other than a tensor or None.
    if not isinstance(t, torch.Tensor):
        if t is not None:
            raise AssertionError(f"expected t to be None, got {t}")
        return None
    return StorageWeakRefWrapper(t)


# A path index of (depth, offset) indices into a graph that is `depth`` number of nodes from the root
# at graph output offset
PathOutputIndex = tuple[int, int]
# Cached per-node lists and output index to check for user-visible output cloning.
PathUserVisibleOutputEntry = tuple[
    list[StorageWeakRefWrapper | None],
    list[TensorWeakRef | None],
    list[Tensor | None] | None,
    int,
]
PathUserVisibleStorageGroup = tuple[PathUserVisibleOutputEntry, ...]

# For each node in the path, for each output, is the output alive
PathLiveness = list[list[bool]]

StackTraces = list[str | None]


class CUDAWarmupNode:
    # [CLASS DOC] A graph node representing one eager warmup run of a function inside the CUDA graph
    # private pool, with no recording. May have either a CUDAGraphNode or CUDAWarmupNode parent, but
    # only warmup children, because tensors without stable addresses cannot be recorded. State:
    # wrapped function, parent, pool handle, existing graph, device index, stack traces, stream,
    # already_warm flag, id, and the weakrefs of outputs (outputs_weakrefs, tensor_weakrefs). Can be
    # run once only. Exposes the same interface as CUDAGraphNode where the tree manager needs it:
    # path_live_weakrefs, all_outputs_are_dead, outputs_weakrefs, stack_traces, tensor_weakrefs.
    """
    Simplified Wrapper around A CUDA Model that wraps outputs in storage refs and exposes
    apis to get the live storages in the current chain of warmup.

    A CUDAWarmupNode may have either CUDAGraphNode or CUDAWarmupNode as a parent, but may only have
    CUDAWarmupNode as children, because we cannot record or execute with tensors which do not have stable
    memory addresses.

    CUDAWarmupNode and CUDAGraphNode have a number of differences that make it easier to use separate classes.
    - Much of the CUDAGraphNode logic & initialization is based on the tensor properties of first recording. In the
    first instance of warmup, these are not finalized yet.
    - All Inputs to the RecordedFunction must be copied over to the cuda graph memory pool, this is unnecessary in warmup.
    - CUDAWarmup is only used once and so does not need to optimize as much bookkeeping. It is much simpler.

    NB: this class and CUDAGraphNode need to expose `path_live_weakrefs`, `all_outputs_are_dead`, and
    `self.outputs_weakrefs`, `stack_traces`, and `tensor_weakrefs` for compatibility.
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        parent: CUDAGraphNode | CUDAWarmupNode | None,
        cuda_graphs_pool: tuple[int, int],
        existing_cuda_graph: torch.cuda.CUDAGraph | None,
        device_index: int,
        stack_traces: StackTraces | None,
        stream: torch.cuda.Stream,
        already_warm: bool,
        id: GraphID,
    ) -> None:
        # [FUNCTION DOC] Stores the wrapped function, its user-visible output indices, the parent
        # node, pool, optional existing CUDA graph, device index, stack traces, stream, warm flag
        # and id; starts with empty output weakref lists and has_run false.
        self.wrapped_function = wrapped_function
        self.user_visible_output_idxs = wrapped_function.user_visible_output_idxs
        self.parent: CUDAGraphNode | CUDAWarmupNode | None = parent
        self.cuda_graphs_pool = cuda_graphs_pool
        self.outputs_weakrefs: list[StorageWeakRefWrapper | None] = []
        self.tensor_weakrefs: list[TensorWeakRef | None] = []
        self.existing_cuda_graph = existing_cuda_graph
        self.has_run = False
        self.device_index = device_index
        self.stack_traces = stack_traces
        self.stream = stream
        self.already_warm = already_warm
        self.id = id

    @functools.cached_property
    def path_user_visible_storage_groups(
        self,
    ) -> tuple[PathUserVisibleStorageGroup, ...]:
        # [FUNCTION DOC] Cached property: returns the grouped path storage aliases that must be
        # cloned for user-visible outputs, computed over the path from root to this node by the
        # module level collection function. Computed once per node.
        return collect_path_user_visible_storage_groups(tuple(self._path_from_root))

    def run(self, new_inputs: Any) -> OutputType:
        # [FUNCTION DOC] Runs the wrapped function eagerly once inside the private pool and returns
        # its outputs. Raises an assertion error if run twice. Before running, notes weakrefs to
        # input and constant tensor storages that are not already pool outputs on the path;
        # optionally (slow path asserts, not already warm) validates the pool against live storages.
        # Runs the model on the pool stream with cuBLAS cache clearing, conv cache emptying
        # disabled, the stream external object updated, and the control flow warmup dispatch mode.
        # Afterwards the inputs list must be empty (assertion error otherwise); it records weakrefs
        # for CUDA output tensors with allocated storage that are not inputs, and None for others;
        # may validate the pool again.
        if self.has_run:
            raise AssertionError("Wrapped function should never be run twice")

        # See: output_is_alias_of_persistent_static_inputs below. We should only be returning freshly created
        # storages in path_live_weakrefs.
        existing_path_data_ptrs = OrderedSet(
            [t.data_ptr() for t in self.path_live_weakrefs() if t()]
        )

        def get_non_cudagraph_inps() -> list[weakref.ReferenceType[UntypedStorage]]:
            # [FUNCTION DOC] HELPER: Returns weak references to untyped storages of tensor inputs
            # and constants whose data pointer is not already among the live path outputs. Used to
            # avoid treating inputs as pool outputs.
            non_cudagraph_inps = [
                weakref.ref(t.untyped_storage())
                for t in itertools.chain(new_inputs, self.wrapped_function.constants)
                if isinstance(t, torch.Tensor)
                and t.untyped_storage().data_ptr() not in existing_path_data_ptrs
            ]
            return non_cudagraph_inps

        non_cudagraph_inps_storages = get_non_cudagraph_inps()

        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)

        with (
            torch.cuda.device(self.device_index),
            disable_conv_cache_emptying(),
            clear_cublas_manager(),
            _use_cuda_memory_pool_manager(
                self.device_index, self.cuda_graphs_pool, self.stream
            ),
            # NB: must go after _use_cuda_memory_pool_manager which switches the stream
            _update_current_stream_external_object(),
            ControlFlowOpWarmupDispatchMode(),
            get_history_recording(),
        ):
            out = self.wrapped_function.model(new_inputs)

        # We need to know which outputs are allocated within the cudagraph pool
        # so that we can deallocate them at the beginning of the next cudagraph step,
        # and set their access to error.
        # We use a weakref to the inputs storage, in case a block which was previously
        # allocated to the general caching allocator pool gets reallocated to a private pool.

        non_cudagraph_inps_storage_ptrs = OrderedSet[Any]()
        for storage in non_cudagraph_inps_storages:
            s = storage()
            if s is not None:
                non_cudagraph_inps_storage_ptrs.add(s._cdata)

        if len(new_inputs) != 0:
            raise AssertionError(
                f"expected new_inputs to be empty, got {len(new_inputs)}"
            )

        # sdpa returns cpu tensors when not recording cuda graph
        def add_ref(o: Any) -> bool:
            # [FUNCTION DOC] HELPER: Predicate: true for CUDA tensors with a non-empty storage that
            # is not one of the recorded non-pool input storages, meaning it should be tracked as a
            # pool output. (CPU results such as from sdpa outside recording are excluded.)
            return (
                isinstance(o, torch.Tensor)
                and o.is_cuda
                and o.untyped_storage()._cdata not in non_cudagraph_inps_storage_ptrs
                and o.untyped_storage().data_ptr() != 0
            )

        self.outputs_weakrefs.extend(
            [map_to_ref(o) if add_ref(o) else None for o in out]
        )
        self.tensor_weakrefs.extend(
            [TensorWeakRef(o) if add_ref(o) else None for o in out]
        )

        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            out_refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, out_refs)

        return out

    @property
    def _path_from_root(
        self,
    ) -> Generator[CUDAGraphNode | CUDAWarmupNode, None, None]:
        # [FUNCTION DOC] Property generator yielding the nodes from the root of the tree down to
        # this node, found by following parent links and reversing.
        nodes = []
        node: CUDAGraphNode | CUDAWarmupNode = self
        while node:
            nodes.append(node)
            node = node.parent  # type: ignore[assignment]

        yield from reversed(nodes)

    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        # [FUNCTION DOC] Generator over storage weakrefs from all nodes on the path from the root to
        # this node whose storages are still alive.
        "Returns all live storages weakrefs that created by nodes in this path"
        for node in self._path_from_root:
            for output in node.outputs_weakrefs:
                if is_live(output):
                    yield output  # type: ignore[misc]

    def all_outputs_are_dead(self) -> bool:
        # [FUNCTION DOC] Returns true when no output storage on the path is alive.
        return not list(self.path_live_weakrefs())

    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the tensor's storage data pointer matches a live
        # output weakref on this path.
        for storage_weak_ref in self.path_live_weakrefs():
            if t.untyped_storage().data_ptr() == storage_weak_ref.data_ptr():
                return True
        return False


# Aliases for List that say what the indices denote
InputList = list  # input indexes
OutputList = list  # output indexes
LevelList = list  # levels (distance from root of tree)


class OutputAliasInfo:
    # [CLASS DOC] HELPER: Base class (no slots, no state) for markers describing how a graph output
    # relates to other storages.
    __slots__ = []


class _UnaliasedStorage(OutputAliasInfo):
    # [CLASS DOC] HELPER: Marker type for outputs that are a new unaliased storage or are None. A
    # single module instance (UnaliasedStorage) is used for identity comparisons.
    "Singleton to mark that the graph output constructs a new alias or is None"


UnaliasedStorage = _UnaliasedStorage()


class AliasesPriorGraphOutput(OutputAliasInfo):
    # [CLASS DOC] HELPER: Marker for a graph output that aliases an output of a prior graph on the
    # path. State: index, a pair (depth, output offset) into the path.
    "Marks that the graph output aliases an output of a prior graph"

    __slots__ = ["index"]

    index: PathOutputIndex

    def __init__(self, index: PathOutputIndex) -> None:
        # [FUNCTION DOC] HELPER: Stores the path index; raises an assertion error if it is not a
        # tuple.
        if not isinstance(index, tuple):
            raise AssertionError(f"expected index to be a tuple, got {type(index)}")
        self.index = index


class AliasesNewOutput(OutputAliasInfo):
    # [CLASS DOC] HELPER: Marker for a graph output that aliases an earlier output of the same
    # graph. State: index of that output in the new outputs list.
    "Marks that the graph output aliases an index in the new, returned outputs"

    __slots__ = ["index"]

    index: int

    def __init__(self, index: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the output index; raises an assertion error if it is not an
        # int.
        if not isinstance(index, int):
            raise AssertionError(f"expected index to be an int, got {type(index)}")
        self.index = index


class CUDAGraphNode:
    # [CLASS DOC] One recording of a function into a CUDA graph, forming a node in the tree. All
    # nodes share a single memory pool; a node has at most one parent (None for the root) and any
    # number of children grouped by function id. The graph is recorded inside the constructor, and
    # run_first_inputs must be called immediately afterward. State highlights: wrapped_function, id,
    # device, stream, weakly held parent, children, graph, recorded input metadata (static input
    # indices, cudagraph managed indices, data pointers, expanded dims), output metadata and alias
    # info (output_storage_alias, outputs_metadata, unaliased_in_all_paths, cached_tensor_outputs,
    # static_output_tensors), path weakrefs for liveness of all outputs on the path, recorded
    # liveness and expected dead indices before and after the graph, and the checkpointed allocator
    # state used to resume recording after replay. Key methods: check_invariants, run,
    # run_first_inputs, reconstruct_outputs, debug checks.
    """
    A single recording of a function into a CUDA Graph. Recordings of CUDA Graphs share a single memory pool
    and are structured into a tree, where there is a single recording that can precede it (parent) and multiple
    subsequent recordings that may follow (children). A node will have no parent if it is the first recording
    in a tree; i.e., when it is first recorded, there are no live tensors from a previous recording which
    would force a dependency.

    On first recording, all of the live tensors in the current CUDA Graph Node path will be
    reflected in the corresponding private pool. On subsequent executions, the caching allocator
    is unaffected when the graph is replayed.

    In order to support recording a subsequent cuda graph recording after execution of this graph,
    we checkpoint the state of the memory pool so that it may later be resumed.

    WrappedFunction should have already been warmed up prior to invocation.

    See [setCheckpointPoolState] for further explanation, as well as
    https://user-images.githubusercontent.com/13564/222815509-374f3400-f83d-4f7d-8fa6-4a092b3250bb.png
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        id: GraphID,
        parent: CUDAGraphNode | None,
        inputs: list[InputType],
        cuda_graphs_pool: _POOL_HANDLE,
        device_index: int,
        stack_traces: StackTraces | None,
        stream: torch.cuda.Stream,
        mode: CompilationMode | None,
        compile_id: CompileId | None,
    ) -> None:
        # [FUNCTION DOC] Records the function into a CUDA graph. Raises an assertion error if inputs
        # is not a list or tuple. Sets up identity, weak parent link, empty children, per-path
        # output weakrefs and stack traces, and classifies inputs: those that are outputs of earlier
        # graphs on the path (cudagraph managed), opaque custom objects, and static ones; computes
        # non-static indices, expanded dims, data pointers and tracked counters. If it has a parent,
        # computes the liveness expected before the graph. Copies non-static inputs into pool
        # allocated recording inputs (clearing the caller's list), creates the CUDA graph unless
        # kernel free, reconstructs non-owning input tensors from metadata, records the model
        # (timed), and saves output metadata (tensors as metadata, ints and None as is; anything
        # else raises an assertion error). Finally replays the graph once if one exists. Side
        # effects: allocator pool, CUDA capture, counters.
        if not isinstance(inputs, (list, tuple)):
            raise AssertionError(
                f"expected inputs to be list or tuple, got {type(inputs)}"
            )

        self.wrapped_function = wrapped_function
        self.user_visible_output_idxs = wrapped_function.user_visible_output_idxs
        self.id = id
        self.device = device_index
        self.stack_traces = stack_traces
        self.stream = stream

        # Enable re-record a cudagraph when static tensor address changed.
        # if not we should error when it changed.
        self.rerecord_if_static_inputs_change = True

        # if this is a root parent will be None. use weakref to prevent reference cycle
        self._parent = weakref.ref(parent) if parent is not None else None
        # reference to the shared memory pool for the entire cuda graphs tree
        self.cuda_graphs_pool = cuda_graphs_pool

        # A single wrapped function may be recorded multiple times if memory patterns or
        # invariants change from one execution to the next
        self.children: dict[FunctionID, list[CUDAGraphNode]] = defaultdict(list)

        # StorageWeakRef maintains whether the Storage C++ object remains allocated,
        # not whether the corresponding memory has been deallocated. In order
        # to use them to track memory deallocations we must maintain a single StorageWeakRef
        # for all Storages that reference that memory (even if we are constructing Storages
        # that do not have a deallocator function). We maintain one single storage_cache
        # as we execute any tree path. When we retrieve a storage from the cache we
        # check that it is still alive, and we hash based on observed recording data ptr
        # and storage cdata.

        # we preserve a single reference to executed outputs that is then referenced
        # in children to avoid children having to chase parent pointers in the hot path
        # DO NOT reassign output_weakrefs, only call `clear()`
        # Path is a series of nodes from root to the current node
        self.outputs_weakrefs: OutputList[StorageWeakRefWrapper | None] = []
        self.path_weakrefs: LevelList[OutputList[StorageWeakRefWrapper | None]] = [
            node.outputs_weakrefs for node in self._path_from_root
        ]
        self.path_stacktraces: LevelList[StackTraces | None] = [
            node.stack_traces for node in self._path_from_root
        ]
        self.tensor_weakrefs: OutputList[TensorWeakRef | None] = []

        # tensors which are outputs of previous graphs in the tree
        self.cudagraph_managed_idxs: list[int] = [
            idx
            for idx, t in enumerate(inputs)
            if isinstance(t, torch.Tensor) and self._is_cuda_graph_recorded_tensor(t)
        ]

        # (depth, offset) of live tensors which are alias of previous graph outputs
        self.live_cudagraph_managed_path_refs: InputList[PathOutputIndex | None] = [
            (
                self._is_alias_of_live_recorded_tensor(t)
                if isinstance(t, torch.Tensor)
                else None
            )
            for t in inputs
        ]

        # when replay, preserve the liveness of an input if it AliasesPriorGraphOutput
        # and also aliases an output of the current CUDAGraphNode
        self.preserved_aliased_inputs: InputList[bool] = [False] * len(inputs)

        # Opaque values (e.g. DeviceMesh, ProcessGroup) are non-tensor
        # inputs that cannot be copied like tensors. We include them in
        # static_input_idxs to keep them out of non_static_input_idx
        # (which drives the tensor-copy path during replay). "Static"
        # here just means "don't try to copy this as a tensor" — it
        # does NOT mean the object is semantically immutable.
        #
        # Opaque indices must also be excluded from any list passed to
        # _tensors_data_ptrs_at_indices_equal (the C++ data-pointer
        # stability check), because opaque objects have no data_ptr.
        # That is why tensor_static_input_idxs and
        # non_managed_static_input_idxs filter them out below.
        opaque_input_idxs = OrderedSet(
            i for i, inp in enumerate(inputs) if is_custom_class_obj(inp)
        )
        static_input_idxs = OrderedSet(wrapped_function.static_input_idxs)
        cudagraph_managed_idxs = OrderedSet(self.cudagraph_managed_idxs)

        self.static_input_idxs: list[int] = list(
            static_input_idxs | cudagraph_managed_idxs | opaque_input_idxs
        )

        self.non_static_input_idx: LevelList[int] = [
            i for i in range(len(inputs)) if i not in self.static_input_idxs
        ]

        counters["inductor"]["cudagraph_recorded_non_static_inputs"] += len(
            self.non_static_input_idx
        )

        self.non_managed_static_input_idxs: LevelList[int] = LevelList(
            static_input_idxs - cudagraph_managed_idxs - opaque_input_idxs
        )

        self.tensor_static_input_idxs: list[int] = list(
            static_input_idxs | cudagraph_managed_idxs
        )

        def maybe_get_static_data_ptr(
            idx: int,
            inputs: list[InputType],
            static_input_idxs: list[int],
        ) -> int | None:
        # [FUNCTION DOC] HELPER: Returns the data pointer of the input at the index if it is a
        # tensor whose index is in the given static list, otherwise None.
            inp = inputs[idx]
            if isinstance(inp, torch.Tensor) and idx in static_input_idxs:
                return inp.data_ptr()
            return None

        self.static_input_data_ptrs: InputList[int | None] = [
            maybe_get_static_data_ptr(i, inputs, self.static_input_idxs)
            for i in range(len(inputs))
        ]

        # When we checkpoint, and free generations, we will be manually freeing the outputs
        # of CUDAGraphNodes. We should not be freeing parameters, not do we need to account for
        # their liveness (they are static), so we need to compute which outputs are aliases of
        # parameters. Some static inputs are saved tensors from the forward that die in the backward.
        # Their locations are static but lifetimes are not. We only include the persistent static
        # data ptrs below because the non persistent data ptrs may be outputs of this record and
        # fresh allocations.

        # precompute expanded dims to avoid computing in the hot path
        self.expanded_dims: list[list[int]] = [
            get_expanded_dims(x)
            if isinstance(x, torch.Tensor) and idx not in self.static_input_idxs
            else []
            for idx, x in enumerate(inputs)
        ]

        # For each node in path, which outputs were observed to be live
        # before invoking graph recording, and after graph recording
        self.recorded_liveness_before_graph: LevelList[OutputList[bool]] = []
        self.recorded_liveness_after_graph: LevelList[OutputList[bool]] = []

        # List of tuples of (depth, output_index) that index into node at depth
        # number of nodes from root and output_index of outputs. Will index into
        # path_weakrefs.
        self.expected_dead_indices_before_graph: list[PathOutputIndex] = []
        self.expected_dead_indices_after_graph: list[PathOutputIndex] = []

        # all live indices after graph recording
        self.live_indices_after_graph: list[PathOutputIndex] = []

        if self.parent is not None:
            previous_liveness = self.parent.recorded_liveness_after_graph
            curr_liveness = self._get_liveness(self.path_weakrefs)

            different_indices = self._get_different_indices(
                previous_liveness, curr_liveness
            )

            self.recorded_liveness_before_graph = curr_liveness
            self.expected_dead_indices_before_graph = different_indices

        recording_inputs = self._allocate_and_copy_recording_inputs(inputs)
        # recording inputs will copy over memory, so we can free non recording inputs

        inputs.clear()
        del inputs

        self.graph: torch.cuda.CUDAGraph | None = (
            None if wrapped_function.kernel_free_cudagraph else torch.cuda.CUDAGraph()
        )

        # we allocate non-static inputs within the same memory pool as the CUDAGraph
        # which we will record the model with. For memory efficiency, it is important
        # to reclaim the input memory when the inputs are no longer live. To accomplish this,
        # we reconstruct tensors at the correct data pointers of our inputs which are
        # non owning and do not prevent deallocation. On subsequent executions, input values
        # will be copied over to these tensors.
        self.reconstructed_inputs: list[InputType] = [
            self._reconstruct_from_tensor_metadata(self._tensor_metadata(x))
            if isinstance(x, torch.Tensor)
            else x
            for x in recording_inputs
        ]

        # DO THE RECORDING!!!
        # We record the CUDA graph in the constructor of CUDAGraphNode, which
        # gives you what the CPU side compute of the function would do.  We
        # don't throw the recording outputs away: their memory is
        # correctly accounted for in the CUDAGraphs caching allocator.  This
        # means on the very FIRST run of the CUDA graph node, we can directly
        # do more recording, because we have a valid caching allocator state.
        # NB: This relies on run() being called immediately after the
        # constructor, otherwise this optimization would not be valid.

        # initialized below in _record

        self.checkpointed_caching_state: AllocatorState | None = None

        # Output Storage Alias information, can be:
        # - A new, unaliased storage, or the output is None
        # - An alias of an output of a prior graph
        # - An alias of an output already created in the reconstructed outputs
        # This is None if the output in question is an int
        self.output_storage_alias: OutputList[OutputAliasInfo | None] = []

        # is the output Storage unaliased in subsequent outputs, of all subsequent paths
        # If a descendant node discovers it has an alias of a prior output, then the output
        # will no longer be cached in the ancestor.
        # The large majority of tensors are unaliased, and preserving aliased output tensors would add
        # significant additional complexity with marginal gains
        # The cached tensor outputs are added on the first execution, and cleared whenever we need
        # to do subsequent recording
        self.unaliased_in_all_paths: OutputList[bool] = []
        self.cached_tensor_outputs: OutputList[Tensor | None] = []

        # if an output aliases a static, persistent input then the corresponding Tensor will
        # be set here. These are different than cached tensors, because they are tensors that
        # are aliases of parameters that are always live.
        self.static_output_tensors: OutputList[Tensor | None] = []

        # Cleared after recording
        with dynamo_timed_cudagraph("CUDAGraphNode.record", compile_id, mode):
            self.recording_outputs: OutputType | None = self._record(
                wrapped_function.model, recording_inputs
            )
        self.outputs_metadata: OutputList[dict[str, Any] | int | None] = []

        # As with inputs, we do not want to keep the outputs permanently alive because that would prevent
        # their memory being reclaimed in subsequent cuda graph recordings. We record the tensor metadata
        # needed to reconstruct instead.
        if self.recording_outputs is None:
            raise AssertionError("expected recording_outputs to not be None")
        for out in self.recording_outputs:
            if isinstance(out, torch.Tensor):
                self.outputs_metadata.append(
                    self._tensor_metadata(out, ignore_storage_offset=False)
                )
            else:
                if not isinstance(out, (int, type(None))):
                    raise AssertionError(type(out))
                self.outputs_metadata.append(out)

        if self.graph is not None:
            self.graph.replay()

    @functools.cached_property
    def path_user_visible_storage_groups(
        self,
    ) -> tuple[PathUserVisibleStorageGroup, ...]:
        # [FUNCTION DOC] Cached property: returns the user-visible storage alias groups for the path
        # from the root to this node via the shared collection function.
        return collect_path_user_visible_storage_groups(tuple(self._path_from_root))

    def _copy_inputs_and_remove_from_src(
        self, dsts: list[InputType], srcs: list[InputType]
    ) -> None:
        # [FUNCTION DOC] HELPER: Copies values from source inputs into destination tensors for all
        # non-static tensor input positions, then sets each copied source slot to None to drop the
        # reference so memory can be reclaimed. Expanded dims are removed before copying;
        # destinations that still self-overlap fall back to a strided storage copy. All other copies
        # are done in one batch. Mutates both lists.
        dst_tensors = []
        src_tensors = []
        for idx in self.non_static_input_idx:
            if not isinstance(srcs[idx], torch.Tensor):
                continue
            expanded_dims = self.expanded_dims[idx]
            indexed_dst = index_expanded_dims(dsts[idx], expanded_dims)  # type: ignore[arg-type]
            if torch._debug_has_internal_overlap(indexed_dst) != 0:
                # rare path: dst still self-overlaps after dropping expanded dims
                copy_strided_storage_(dsts[idx], srcs[idx])  # type: ignore[arg-type]
                srcs[idx] = None  # type: ignore[call-overload]
                continue
            dst_tensors.append(indexed_dst)
            src_tensors.append(index_expanded_dims(srcs[idx], expanded_dims))  # type: ignore[arg-type]
            srcs[idx] = None  # type: ignore[call-overload]
        # Fails on empty lists
        if dst_tensors:
            torch._foreach_copy_(dst_tensors, src_tensors)

    def check_static_inputs_are_stable(self, new_inputs: list[InputType]) -> None:
        # [FUNCTION DOC] If re-recording on static input change is disabled and the data pointers of
        # non-managed static inputs differ from the recorded ones, builds a mismatch message and
        # raises through the check mechanism. Otherwise does nothing.
        # avoid checking managed tensor static points since we already checked those in check_invariants
        if (
            not self.rerecord_if_static_inputs_change
            and not torch._C._tensors_data_ptrs_at_indices_equal(
                new_inputs,  # type: ignore[arg-type]
                self.static_input_data_ptrs,
                self.non_managed_static_input_idxs,
            )
        ):
            # this should error
            error_msg = log_data_ptr_mismatch(
                self.wrapped_function.placeholders,
                new_inputs,
                self.static_input_data_ptrs,
                self.non_managed_static_input_idxs,
                CheckInvariantStatus.StaticInputIdxMismatch,
            )
            torch._check(False, lambda: error_msg)

    def run_first_inputs(self, new_inputs: list[InputType]) -> OutputType:
        # [FUNCTION DOC] Called right after construction: optionally checks invariants, asserts the
        # incoming input list was already emptied, and returns the outputs produced during
        # recording, clearing the stored reference. Raises an assertion error if the inputs list is
        # not empty or outputs are missing.
        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_before_invocation()

        # graph is already invoked in the __init__
        # inputs are copied over in _allocate_recording_inputs and subsequently cleared
        if len(new_inputs) != 0:
            raise AssertionError(
                f"expected new_inputs to be empty, got {len(new_inputs)}"
            )
        outputs = self.recording_outputs
        self.recording_outputs = None
        if outputs is None:
            raise AssertionError("expected outputs to not be None")
        return outputs

    def run(self, new_inputs: list[InputType]) -> OutputType:
        # [FUNCTION DOC] Replays this recorded node on new inputs: checks static input stability,
        # copies non-static inputs into the recorded buffers and drops them from the argument list,
        # replays the graph, reconstructs output tensors, clears the input list, optionally checks
        # invariants and synchronizes the device. Returns the outputs. KNOWN ISSUE: sets a
        # static_inputs_stable attribute that is never read anywhere in this file.
        self.check_static_inputs_are_stable(new_inputs)

        self._copy_inputs_and_remove_from_src(self.reconstructed_inputs, new_inputs)

        self.run_graph()

        outputs = self.reconstruct_outputs()
        new_inputs.clear()

        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_after_invocation()

        if config.triton.force_cudagraph_sync:
            torch.cuda.synchronize()

        # Reset this to run the check in the future
        self.static_inputs_stable = False

        return outputs

    def reconstruct_outputs(self) -> OutputType:
        # [FUNCTION DOC] Rebuilds output values from stored metadata and alias info after a replay.
        # Ints and None pass through. Outputs that are cached tensors are returned as is (clearing
        # backward hooks). Static outputs are returned directly. Others are reconstructed on the
        # right storage: a prior path output, a new output of this graph, or fresh storage; the
        # output weakref is repointed to the new storage and user-visible tensors get a new tensor
        # weakref. Populates cached tensors first if empty. Raises assertion errors on inconsistent
        # state. Returns the output list.
        "Reconstruct output tensors according to their saved metadata and alias information"

        # Cached tensors will not yet be set on the first execution
        # They are also cleared in checkpointing, so if we checkpoint this node
        # and then execute it again we will need to repopulate cached tensors
        if not self.cached_tensor_outputs:
            self._initialize_cached_tensors()

        outputs: OutputType = []
        user_visible_output_idxs = self.user_visible_output_idxs

        for i, (storage_info, metadata) in enumerate(
            zip(self.output_storage_alias, self.outputs_metadata)
        ):
            if not isinstance(metadata, dict):  # tensor metadata
                if not isinstance(metadata, (int, type(None))):
                    raise AssertionError(f"expected int or None, got {type(metadata)}")
                outputs.append(metadata)
                continue

            cached_t = self.cached_tensor_outputs[i]
            if cached_t is not None:
                # this output represents a fresh allocated tensor.
                # We return the same TensorImpl from run to run to avoid overhead.
                # autograd.Function will reset the Autograd meta of output tensors
                # as part of aot_autograd, but _backward_hooks are stored on tensors separately,
                # so we need to manually reset hooks.
                if cached_t._backward_hooks is not None:
                    cached_t._backward_hooks = None

                # No need to update weakrefs, already correctly initialized
                outputs.append(cached_t)
                continue

            static_t = self.static_output_tensors[i]
            if static_t is not None:
                if self.outputs_weakrefs[i] is not None:
                    raise AssertionError(f"expected outputs_weakrefs[{i}] to be None")
                outputs.append(static_t)
                continue

            storage = self.prepare_alias_info_for_tensor_construction(
                storage_info, metadata
            )

            if isinstance(storage, UntypedStorage) or storage is None:
                out = self._reconstruct_from_tensor_metadata(metadata, storage)
            else:
                if not isinstance(storage, int):
                    raise AssertionError(
                        f"expected storage to be int, got {type(storage)}"
                    )
                out = self._reconstruct_from_tensor_metadata(
                    metadata, cast(torch.Tensor, outputs[storage]).untyped_storage()
                )

            outputs.append(out)
            w = self.outputs_weakrefs[i]
            if w is None:
                raise AssertionError("expected w to not be None")
            w.swap_weakref(out.untyped_storage()._weak_ref())
            if user_visible_output_idxs and i in user_visible_output_idxs:
                self.tensor_weakrefs[i] = TensorWeakRef(out)

        return outputs

    def prepare_alias_info_for_tensor_construction(
        self,
        out_alias_info: OutputAliasInfo | None,
        metadata: dict[str, Any] | int | None,
    ) -> UntypedStorage | None | int:
        # [FUNCTION DOC] Determines what storage to use when rebuilding one output: None for ints,
        # None metadata or unaliased outputs; a new weak-pointer storage for outputs aliasing a
        # prior graph output; or the integer index for outputs aliasing a new output of this graph.
        # Raises an assertion error for unexpected alias info types or missing refs.
        if (
            isinstance(metadata, (int, type(None)))
            or out_alias_info is UnaliasedStorage
        ):
            return None

        if isinstance(out_alias_info, AliasesPriorGraphOutput):
            depth, existing_output_index = out_alias_info.index
            ref = self.path_weakrefs[depth][existing_output_index]
            if ref is None:
                raise AssertionError("expected ref to not be None")
            return torch.UntypedStorage._new_with_weak_ptr(ref())

        if not isinstance(out_alias_info, AliasesNewOutput):
            raise AssertionError(
                f"expected AliasesNewOutput, got {type(out_alias_info)}"
            )
        return out_alias_info.index

    def prepare_storages_for_construction(
        self,
    ) -> list[UntypedStorage | None | int]:
        # [FUNCTION DOC] Applies the alias preparation to every output and returns the list of
        # results (storage, None or index).
        output_storages = []
        for output_storage_alias, metadata in zip(
            self.output_storage_alias, self.outputs_metadata
        ):
            output_storages.append(
                self.prepare_alias_info_for_tensor_construction(
                    output_storage_alias, metadata
                )
            )

        return output_storages

    def run_graph(self) -> None:
        # [FUNCTION DOC] Replays the CUDA graph unless the function is kernel free; raises an
        # assertion error if a graph is needed but missing.
        if self.wrapped_function.kernel_free_cudagraph:
            return
        if self.graph is None:
            raise AssertionError("expected graph to not be None")
        self.graph.replay()

    def all_outputs_are_dead(self) -> bool:
        # [FUNCTION DOC] Returns true if none of the outputs recorded as live after this graph,
        # across the path to the root, is still alive.
        "All outputs of the path from this node to its root are dead"
        for depth, output_index in self.live_indices_after_graph:
            if is_live(self.path_weakrefs[depth][output_index]):
                return False
        return True

    def _record(self, model: ModelType, inputs: list[InputType]) -> OutputType:
        # [FUNCTION DOC] HELPER: Records the model into the graph and returns its outputs. Builds a
        # mapping from data pointers of persistent static inputs and constants to weakrefs so
        # outputs that alias them are recognized. For kernel free functions it runs the model under
        # the pool manager and warmup dispatch mode without capture. Otherwise (optionally
        # validating the pool first) it captures the model with the CUDA graph capture context on
        # the shared pool and stream, with RNG state preserved. Requires the inputs list to be
        # emptied by the model (assertion error otherwise). Wraps non-list results into a sequence,
        # registers the outputs through _add_first_outputs and returns them.
        "Record the model"

        def static_input_iter() -> Generator[torch.Tensor, None, None]:
        # [FUNCTION DOC] HELPER: Generator yielding the tensor inputs at static indices that are not
        # already outputs of an earlier graph on the path.
            for i in self.wrapped_function.static_input_idxs:
                _inp = inputs[i]
                if isinstance(
                    _inp, torch.Tensor
                ) and not self._is_cuda_graph_recorded_tensor(_inp):
                    yield _inp

        # see: output_is_alias_of_persistent_static_inputs above
        static_input_persistent_storage_ptrs: dict[int, StorageWeakRefWrapper] = {
            inp.untyped_storage().data_ptr(): StorageWeakRefWrapper(inp)
            for inp in itertools.chain(
                static_input_iter(), self.wrapped_function.constants
            )
        }

        if self.wrapped_function.kernel_free_cudagraph:
            with (
                preserve_rng_state(),
                torch.cuda.device(self.device),
                clear_cublas_manager(),
                _use_cuda_memory_pool_manager(
                    self.device, self.cuda_graphs_pool, self.stream
                ),
                # NB: must go after _use_cuda_memory_pool_manager which switches the stream
                _update_current_stream_external_object(),
                ControlFlowOpWarmupDispatchMode(),
                get_history_recording(),
            ):
                static_outputs = model(inputs)
            if len(inputs) != 0:
                raise AssertionError(f"expected inputs to be empty, got {len(inputs)}")

            if not isinstance(static_outputs, (list, tuple)):
                static_outputs = [static_outputs]
            elif isinstance(static_outputs, tuple):
                static_outputs = list(static_outputs)
            static_outputs = cast(OutputType, static_outputs)
            self._add_first_outputs(
                static_outputs, static_input_persistent_storage_ptrs
            )
            return static_outputs

        if self.graph is None:
            raise AssertionError("expected graph to not be None")

        if config.triton.slow_path_cudagraph_asserts:
            # need to use parent live weakrefs because live_indices isn't set yet
            memory = (
                [] if self.parent is None else list(self.parent.path_live_weakrefs())
            )
            memory += [
                StorageWeakRefWrapper(elem)
                for i, elem in enumerate(inputs)
                if isinstance(elem, torch.Tensor)
                and i not in self.wrapped_function.static_input_idxs
                and elem.untyped_storage().data_ptr() != 0
            ]
            check_memory_pool(self.device, self.cuda_graphs_pool, memory)

        with (
            preserve_rng_state(),
            torch.cuda.device(self.device),
            clear_cublas_manager(),
            torch.cuda.graph(
                self.graph,
                stream=self.stream,
                pool=self.cuda_graphs_pool,
                capture_error_mode="thread_local",
            ),
            # NB: must go after torch.cuda.graph which switches the stream
            _update_current_stream_external_object(),
            CUDAGraphCaptureControlFlowOpDispatchMode(),
            get_history_recording(),
        ):
            static_outputs = model(inputs)

        # running model should reclaim memory
        if len(inputs) != 0:
            raise AssertionError(f"expected inputs to be empty, got {len(inputs)}")

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

        # pyrefly: ignore [bad-argument-type]
        self._add_first_outputs(static_outputs, static_input_persistent_storage_ptrs)

        # pyrefly: ignore [bad-return]
        return static_outputs

    def _add_first_outputs(
        self,
        outputs: OutputType,
        static_input_persistent_storage_ptrs: dict[int, StorageWeakRefWrapper],
    ) -> None:
        # [FUNCTION DOC] HELPER: Registers outputs of the first run: computes which path outputs
        # died during recording, then classifies each output as unaliased, aliasing a prior graph
        # output (marking the prior one as aliased and preserving aliased inputs), aliasing an
        # earlier new output, or static (aliasing persistent inputs or empty storage). Raises on
        # non-CUDA outputs or a wrong number of stack traces. Creates storage and tensor weakrefs
        # for owned outputs, records liveness after the graph, checkpoints the allocator state, and
        # fills live_indices_after_graph. Runs invariant checks and optional pool validation.
        "Add the outputs from the first invocation of the node and set up metadata"

        # getting liveness before we have added the outputs to path, so the length
        # of the two lists is equal
        prev_liveness = self.recorded_liveness_before_graph
        curr_liveness = self._get_liveness(self.path_weakrefs)

        delta = self._get_different_indices(prev_liveness, curr_liveness)
        self.expected_dead_indices_after_graph = delta

        if len(self.outputs_weakrefs) != 0:
            raise AssertionError(
                f"expected outputs_weakrefs to be empty, got {len(self.outputs_weakrefs)}"
            )
        # index from data pointer to index in outputs
        output_new_storages_index: dict[StorageDataPtr, int] = {}

        self.unaliased_in_all_paths = [False for _ in range(len(outputs))]
        self.static_output_tensors = [None for _ in range(len(outputs))]

        for i, o in enumerate(outputs):
            if o is None or not isinstance(o, torch.Tensor):
                self.output_storage_alias.append(UnaliasedStorage)
                continue

            torch._check(
                o.is_cuda or o.untyped_storage().data_ptr() == 0,
                lambda: (
                    "Expected all cuda outputs in cuda graph recording. Non cuda output "
                    f"from {self.stack_traces[i] if self.stack_traces else '(unknown)'}"
                ),
            )

            ref = static_input_persistent_storage_ptrs.get(
                o.untyped_storage().data_ptr()
            )
            # also treat empty storages as static outputs because we do not need to manage their lifetime
            # and they should not participate in checkpointing
            is_empty_storage = o.untyped_storage().data_ptr() == 0
            if (ref and ref() is not None) or is_empty_storage:
                self.output_storage_alias.append(None)
                self.static_output_tensors[i] = o
                continue

            path_ref = self._is_alias_of_live_recorded_tensor(o)
            if path_ref is not None:
                self._mark_prior_graph_output_as_aliased(path_ref)

                for idx, inp_path_ref in enumerate(
                    self.live_cudagraph_managed_path_refs
                ):
                    if path_ref == inp_path_ref:
                        self.preserved_aliased_inputs[idx] = True
                self.output_storage_alias.append(AliasesPriorGraphOutput(path_ref))
                continue

            if o.untyped_storage().data_ptr() in output_new_storages_index:
                index = output_new_storages_index[o.untyped_storage().data_ptr()]
                self.unaliased_in_all_paths[index] = False
                self.output_storage_alias.append(AliasesNewOutput(index))
                continue

            output_new_storages_index[o.untyped_storage().data_ptr()] = i
            self.output_storage_alias.append(UnaliasedStorage)
            self.unaliased_in_all_paths[i] = True

        if self.stack_traces is None:
            self.stack_traces = [None for _ in range(len(outputs))]
        else:
            if len(self.stack_traces) != len(outputs):
                raise AssertionError("Wrong number of stack traces passed in")

        if self.outputs_weakrefs:
            raise AssertionError("expected outputs_weakrefs to be empty")
        for out, static_output_tensor in zip(outputs, self.static_output_tensors):
            if not isinstance(out, torch.Tensor) or static_output_tensor is not None:
                self.outputs_weakrefs.append(None)
                self.tensor_weakrefs.append(None)
            else:
                self.outputs_weakrefs.append(StorageWeakRefWrapper(out))
                self.tensor_weakrefs.append(TensorWeakRef(out))

        self.recorded_liveness_after_graph = self._get_liveness(self.path_weakrefs)
        self.checkpointed_caching_state = torch._C._cuda_getCheckpointState(
            self.device, self.cuda_graphs_pool
        )

        # now, get liveness with outputs added
        for depth in range(len(self.path_weakrefs)):
            for output_index in range(len(self.path_weakrefs[depth])):
                if is_live(self.path_weakrefs[depth][output_index]):
                    self.live_indices_after_graph.append((depth, output_index))

        self.debug_check_invariants_after_invocation()
        if config.triton.slow_path_cudagraph_asserts:
            check_memory_pool(
                self.device, self.cuda_graphs_pool, list(self.path_live_weakrefs())
            )

    def _mark_prior_graph_output_as_aliased(self, index: PathOutputIndex) -> None:
        # [FUNCTION DOC] HELPER: Marks an output of an ancestor node (given by depth and offset) as
        # aliased: unsets its unaliased flag so it is no longer cached, and removes the extra
        # reference from its weakref. Raises an assertion error if the weakref is missing.
        "Remove a graph output from the unaliased, cached tensors in an ancestor node"
        depth, output_index = index
        node = list(self._path_from_root)[depth]
        node.unaliased_in_all_paths[output_index] = False
        x = self.path_weakrefs[depth][output_index]
        if x is None:
            raise AssertionError("expected x to not be None")
        x.remove_extra_reference()

    def _initialize_cached_tensors(self) -> None:
        # [FUNCTION DOC] HELPER: Creates cached output tensors for outputs unaliased in all paths:
        # for each, rebuilds a tensor on a storage from the metadata, registers it as a cached
        # tensor with autograd, and installs a weakref wrapper whose extra check verifies, using
        # reference counts, that nobody else holds the tensor. Appends None for outputs not cached.
        # Raises assertion errors on length mismatch or inconsistent alias info.
        # we should not be clearing output_weakrefs, and they should be set in the first
        # record run
        if len(self.outputs_weakrefs) != len(self.outputs_metadata):
            raise AssertionError(
                "expected len(outputs_weakrefs) == len(outputs_metadata)"
            )
        for i, (storage_info, metadata, make_cached) in enumerate(
            zip(
                self.output_storage_alias,
                self.outputs_metadata,
                self.unaliased_in_all_paths,
            )
        ):
            if not make_cached:
                self.cached_tensor_outputs.append(None)
                continue

            if storage_info is not UnaliasedStorage:
                raise AssertionError(
                    f"expected storage_info to be UnaliasedStorage, got {storage_info}"
                )
            if not isinstance(metadata, dict):
                raise AssertionError(
                    f"expected metadata to be dict, got {type(metadata)}"
                )
            s = self.create_storage(metadata)
            out = self._reconstruct_from_tensor_metadata(metadata, storage=s)  # type: ignore[arg-type]

            # XXX: let autograd know that there will be an additional reference to the tensor
            # that can be ignored when deciding whether to do gradient buffer inplacing.
            # Otherwise, inplacing could differ between tracing and subsequent execution.
            # For some models we tested this led to inputs no longer being in cudagraph pools,
            # leading to spurious re-recordings.
            # It also tells AMP cache that even though the tensor impls cannot be cached
            # in dtype conversions.

            torch._C._add_cached_tensor(out)

            self_ref = weakref.ref(self)

            # one reference in our array, and calling sys.getrefcount bumps the refcount by one
            def check_refcount(i: int) -> bool:
            # [FUNCTION DOC] HELPER: Extra reference check for a cached output at an index: false if
            # the node is gone; otherwise compares the Python reference count of the cached tensor
            # (adjusted for an extra C++ reference) to the expected baseline and returns true when
            # no additional users exist. Raises an assertion error if the count is lower than
            # expected.
                self_loc = self_ref()
                if self_loc is None:
                    return False
                refcount = self_loc.get_output_refcount(i)
                # pyrefly: ignore
                if self_loc.cached_tensor_outputs[i]._use_count() > 1:
                    # c10::Tensor may also holds one reference count
                    if refcount < 3:
                        raise AssertionError(f"expected refcount >= 3, got {refcount}")
                    return refcount == 3
                else:
                    if refcount < 2:
                        raise AssertionError(f"expected refcount >= 2, got {refcount}")
                    return refcount == 2

            check = functools.partial(check_refcount, i=i)

            self.outputs_weakrefs[i] = StorageWeakRefWrapper(out, extra_ref_check=check)
            self.cached_tensor_outputs.append(out)

    def get_output_refcount(self, index: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the Python reference count of the cached output tensor at
        # the index.
        return sys.getrefcount(self.cached_tensor_outputs[index])

    @property
    def parent(self) -> CUDAGraphNode | None:
        # [FUNCTION DOC] Property: returns the parent node by dereferencing the weak reference, or
        # None for roots or if the parent is gone.
        "unwraps the weakref to _parent"
        return self._parent() if self._parent is not None else None

    @property
    def _path_to_root(self) -> Generator[CUDAGraphNode, None, None]:
        # [FUNCTION DOC] HELPER: Property generator yielding this node and its ancestors up to the
        # root.
        "Returns all nodes in the path starting at self and ending at root"
        node = self
        # pyrefly: ignore [bad-assignment]
        while node:
            yield node
            node = node.parent  # type: ignore[assignment]

    @property
    def _path_from_root(self) -> Generator[CUDAGraphNode, None, None]:
        # [FUNCTION DOC] HELPER: Property generator yielding nodes from the root down to this node.
        "Returns all nodes in the path starting at the root and ending at self"
        nodes = reversed(list(self._path_to_root))
        yield from nodes

    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the tensor's storage data pointer matches any
        # output weakref on the path, without checking liveness (pool memory is never released).
        "Is this tensor an output of a node in this path"
        for output_refs in self.path_weakrefs:
            for storage_weak_ref in output_refs:
                if storage_weak_ref is None:
                    continue
                # don't need to check liveness of storage since the cuda graph managed
                # memory is never released.
                data_ptr = storage_weak_ref.data_ptr()
                if t.untyped_storage().data_ptr() == data_ptr:
                    return True

        return False

    def _is_alias_of_live_recorded_tensor(
        self, t: torch.Tensor
    ) -> PathOutputIndex | None:
        # [FUNCTION DOC] HELPER: Looks for a live output on the path whose data pointer equals the
        # tensor's storage data pointer and returns its (depth, output index) pair, or None.
        for depth, output_refs in enumerate(self.path_weakrefs):
            for output_index, storage_ref in enumerate(output_refs):
                if (storage_and_ptr := maybe_deref(storage_ref)) is not None:
                    _storage, ptr = storage_and_ptr
                    if ptr == t.untyped_storage().data_ptr():
                        return (depth, output_index)

        return None

    @staticmethod
    def _check_liveness(
        indices: list[PathOutputIndex],
        output_refs: list[list[StorageWeakRefWrapper | None]],
    ) -> bool:
        # [FUNCTION DOC] HELPER: Static helper: returns true only if every referenced (depth,
        # offset) output in the given path weakref lists is dead. Raises an assertion error if a
        # referenced weakref is None.
        "Check that all of the indices specified are dead references"
        for depth, output_index in indices:
            w = output_refs[depth][output_index]
            if w is None:
                raise AssertionError("expected w to not be None")
            if w() is not None:
                return False
        return True

    def add_child(self, function_id: FunctionID, node: CUDAGraphNode) -> None:
        # [FUNCTION DOC] Registers a node as a child of this node under the function id.
        "Adds node as a child of self"
        self.children[function_id].append(node)

    @staticmethod
    def _get_different_indices(
        prev: list[list[bool]], curr: list[list[bool]]
    ) -> list[PathOutputIndex]:
        # [FUNCTION DOC] HELPER: Static helper: compares two per-node boolean liveness tables and
        # returns the (node, output) positions where they differ. Raises an assertion error if the
        # first has more nodes than the second or if row lengths differ.
        "Find indices where the two lists differ."
        dead_indices = []
        if len(prev) > len(curr):
            raise AssertionError(
                f"expected len(prev) <= len(curr), got {len(prev)} > {len(curr)}"
            )
        for i, (outputs1, outputs2) in enumerate(zip(prev, curr)):
            if len(outputs1) != len(outputs2):
                raise AssertionError(
                    f"expected len(outputs1) == len(outputs2), got {len(outputs1)} != {len(outputs2)}"
                )
            for j, (output1, output2) in enumerate(zip(outputs1, outputs2)):
                if output1 != output2:
                    dead_indices.append((i, j))

        return dead_indices

    @staticmethod
    def _get_liveness(
        weakrefs: list[list[StorageWeakRefWrapper | None]],
    ) -> list[list[bool]]:
        # [FUNCTION DOC] HELPER: Static helper: converts per-node lists of weakrefs into per-node
        # lists of booleans telling whether each is alive; empty input gives an empty list.
        "Maps weakrefs to true if the reference is alive and false otherwise"
        if len(weakrefs) == 0:
            return []

        return [pytree.tree_map(is_live, outputs) for outputs in weakrefs]

    def debug_assert_invariants(
        self, expected_liveness: list[list[bool]], newly_dead: list[PathOutputIndex]
    ) -> None:
        # [FUNCTION DOC] HELPER: When fast path asserts are enabled, verifies that path weakrefs are
        # the nodes' own lists, that no output expected to be dead is alive, that live storages have
        # consistent data pointer and weak pointer sets, and that persistent alias outputs are not
        # in the live pool blocks. Raises assertion errors on violations; no-op otherwise.
        if not config.triton.fast_path_cudagraph_asserts:
            return

        for i, node in enumerate(self._path_from_root):
            if self.path_weakrefs[i] is not node.outputs_weakrefs:
                raise AssertionError(
                    "expected path_weakrefs[i] to be node.outputs_weakrefs"
                )

        nodes = list(self._path_from_root)

        live_blocks = get_block_addrs(self.cuda_graphs_pool)

        live_storage_data_ptrs = OrderedSet[Any]()
        live_storage_weak_ptrs = OrderedSet[Any]()

        for depth, outputs_liveness in enumerate(expected_liveness):
            for output_idx, output_liveness in enumerate(outputs_liveness):
                # tensor can die early, but it can't be alive when it should be dead
                w = self.path_weakrefs[depth][output_idx]
                if (stor_weak_ptr_and_data_ptr := maybe_deref(w)) is not None:
                    if not output_liveness:
                        raise AssertionError("expected output_liveness to be truthy")
                    stor_weak_ptr, stor_data_ptr = stor_weak_ptr_and_data_ptr
                    if not (
                        (stor_data_ptr in live_storage_data_ptrs)
                        == (stor_weak_ptr in live_storage_weak_ptrs)
                    ):
                        raise AssertionError(
                            "expected data_ptr and weak_ptr liveness to agree"
                        )
                    live_storage_data_ptrs.add(stor_data_ptr)
                    live_storage_weak_ptrs.add(stor_weak_ptr)

                    is_persistent_alias = (
                        nodes[depth].static_output_tensors[output_idx] is not None
                    )

                    if is_persistent_alias:
                        if stor_data_ptr in live_blocks:
                            raise AssertionError(
                                "expected stor_data_ptr not in live_blocks"
                            )

        for depth, output_index in newly_dead:
            if is_live(self.path_weakrefs[depth][output_index]):
                raise AssertionError(
                    "expected path_weakrefs[depth][output_index] to not be live"
                )

    def debug_check_invariants_before_invocation(self) -> None:
        # [FUNCTION DOC] HELPER: Checks invariants against the liveness recorded before the graph
        # and the indices expected dead before it.
        self.debug_assert_invariants(
            self.recorded_liveness_before_graph, self.expected_dead_indices_before_graph
        )

    def debug_check_invariants_after_invocation(self) -> None:
        # [FUNCTION DOC] HELPER: Checks invariants after invocation using the indices expected dead
        # after the graph. KNOWN ISSUE: passes the liveness recorded before the graph, not the one
        # recorded after the graph, which looks inconsistent with the name.
        self.debug_assert_invariants(
            self.recorded_liveness_before_graph, self.expected_dead_indices_after_graph
        )

    def data_ptrs_dead_since_invocation(self) -> list[int]:
        # [FUNCTION DOC] Returns data pointers of path outputs that died since this node was
        # recorded, found by comparing recorded liveness after the graph with current liveness. Used
        # to free those blocks when checkpointing the allocator.
        """
        Since this node was invoked, return data ptrs of all tensor outputs that have died
        in the current executing tree path.
        """
        curr_liveness = self._get_liveness(self.path_weakrefs)
        _get_different_indices = self._get_different_indices(
            self.recorded_liveness_after_graph, curr_liveness
        )

        path = list(self._path_from_root)
        ptrs_to_deallocate = []
        for depth, output_index in _get_different_indices:
            ptrs_to_deallocate.append(
                path[depth].outputs_metadata[output_index]["data_ptr"]  # type: ignore[index]
            )

        return ptrs_to_deallocate

    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        # [FUNCTION DOC] Generator over weakrefs of outputs that were live after this graph recorded
        # and are still alive now.
        for i, j in self.live_indices_after_graph:
            out = self.path_weakrefs[i][j]
            if out is not None and is_live(out):
                yield out

    def remove_node_cached_tensors(self) -> None:
        # [FUNCTION DOC] Removes this node's cached output tensors from the global cached tensor
        # registry, clears the list, and drops the extra reference from weakrefs of unaliased
        # outputs. Raises an assertion error if such a weakref is missing.
        for t in self.cached_tensor_outputs:
            if t is not None:
                torch._C._remove_cached_tensor(t)
        self.cached_tensor_outputs.clear()

        for i, unaliased in enumerate(self.unaliased_in_all_paths):
            if unaliased:
                n = self.outputs_weakrefs[i]
                if n is None:
                    raise AssertionError("expected n to not be None")
                n.remove_extra_reference()

    def remove_path_cached_tensors(self) -> None:
        # [FUNCTION DOC] Removes cached tensors for every node on the path from the root to this
        # node.
        for node in self._path_from_root:
            node.remove_node_cached_tensors()

    def clear_path_state(self) -> None:
        # [FUNCTION DOC] Placeholder for clearing per-path state when execution ends; currently does
        # nothing.
        "Clear the path state in this current executing node"
        # this doesn't actually do anything right now, leaving it as placeholder

    @staticmethod
    def _tensor_metadata(
        x: torch.Tensor, ignore_storage_offset: bool = True
    ) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Static helper: builds a metadata mapping for a tensor (storage
        # nbytes, data pointer, size, stride, dtype, device, storage offset). The offset is zero
        # when ignored, which is the default for inputs. Raises an assertion error if given a non-
        # tensor.
        if not isinstance(x, torch.Tensor):
            raise AssertionError(f"expected x to be a torch.Tensor, got {type(x)}")
        # We ignore the storage offset for inputs, but not for outputs
        # TODO: - should we make the storage resizable ?
        return {
            "nbytes": x.untyped_storage().nbytes(),
            "data_ptr": x.untyped_storage().data_ptr(),
            "size": x.shape,
            "stride": x.stride(),
            "dtype": x.dtype,
            "device": x.device,
            "storage_offset": x.storage_offset() if not ignore_storage_offset else 0,
        }

    def _reconstruct_from_tensor_metadata(
        self, metadata: dict[str, Any], storage: UntypedStorage | None = None
    ) -> Tensor:
        # [FUNCTION DOC] HELPER: Rebuilds a CUDA tensor from metadata, on the given storage or on a
        # new storage created from the metadata data pointer.
        s = self.create_storage(metadata) if storage is None else storage
        return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, s)  # type: ignore[arg-type]

    def create_storage(self, metadata: dict[str, Any]) -> torch.types.Storage:
        # [FUNCTION DOC] Creates a non-owning storage at the data pointer, device and size given in
        # the metadata.
        return torch._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )

    def _allocate_and_copy_recording_inputs(
        self, inputs: list[InputType]
    ) -> list[InputType]:
        # [FUNCTION DOC] HELPER: Allocates, inside the private pool and on the node stream, copies
        # of non-static tensor inputs and copies values into them, while static tensors and allowed
        # non-tensor values (int, Generator, custom class object) are passed through unchanged.
        # Raises an assertion error for other non-tensor types. Synchronizes first; suppresses
        # warnings. Returns the list of recording inputs; emptied source slots are cleared by the
        # copy step.
        """
        Allocate inputs for non static, non cudagraph managed tensors in the memory pool
        and copy over the tensor values.
        """

        torch.cuda.synchronize()
        self.stream.wait_stream(torch.cuda.current_stream())
        recording_inputs: list[InputType] = []

        with (
            warnings.catch_warnings(record=True),
            torch.cuda.device(self.device),
            _use_cuda_memory_pool_manager(
                self.device,
                mem_pool=self.cuda_graphs_pool,
                stream=self.stream,
            ),
        ):
            for i, inp in enumerate(inputs):
                if not isinstance(inp, torch.Tensor):
                    if not isinstance(inp, (int, torch.Generator, CustomClassBase)):
                        raise AssertionError(
                            f"expected int, Generator, or CustomClassBase, got {type(inp)}"
                        )

                    recording_inputs.append(inp)
                elif i not in self.static_input_idxs:
                    # static_input does an allocation!
                    recording_inputs.append(static_input(inp))
                else:
                    recording_inputs.append(inp)

            self._copy_inputs_and_remove_from_src(recording_inputs, inputs)

        return recording_inputs

    def check_invariants(
        self, inputs: list[InputType]
    ) -> tuple[CheckInvariantStatus, Callable[..., str]]:
        # [FUNCTION DOC] Decides whether this node can be replayed for the given inputs. Returns a
        # pair of a status value and a callable producing a mismatch explanation. Checks in order:
        # data pointers of cudagraph managed inputs, that outputs expected dead before the graph are
        # dead, data pointers of static inputs (when re-recording is allowed), then drops references
        # to managed inputs not preserved and verifies the outputs expected dead after the graph are
        # dead (raises through the check mechanism otherwise). Returns SUCCESS when all hold.
        # Mutates the input list by clearing managed slots.
        """
        Checks if this node can be run. The same pattern of tensor liveness, static inputs,
        and tensors managed in the cudagraph private pool must remain stable.
        """

        _logger = functools.partial(
            log_data_ptr_mismatch,
            self.wrapped_function.placeholders,
            inputs,
            self.static_input_data_ptrs,
        )

        # previously managed data pointers remain stable
        # this is on the hot path so moved to C++. equivalent to:
        # return all(t.data_ptr() == data_ptr for (t, data_ptr) in zip(tensors, data_ptrs))
        if not torch._C._tensors_data_ptrs_at_indices_equal(
            inputs,  # type: ignore[arg-type]
            self.static_input_data_ptrs,
            self.cudagraph_managed_idxs,
        ):
            status = CheckInvariantStatus.CudagraphManagedIdxMismatch
            _logger = functools.partial(
                _logger,
                self.cudagraph_managed_idxs,
                status,
            )
            return status, _logger

        if not self._check_liveness(
            self.expected_dead_indices_before_graph, self.path_weakrefs
        ):
            status = CheckInvariantStatus.ExpectedDeadIndicesBeforeGraphMismatch
            return status, lambda: f"{status}"

        # static input data pointers should remain stable
        # if we are inlining builtin nn modules we re-record in this case
        # if we are not inlining builtin nn modules, we check this in check_static_inputs_are_stable
        # and error if they are not stable
        if (
            self.rerecord_if_static_inputs_change
            and not torch._C._tensors_data_ptrs_at_indices_equal(
                inputs,  # type: ignore[arg-type]
                self.static_input_data_ptrs,
                self.tensor_static_input_idxs,
            )
        ):
            status = CheckInvariantStatus.StaticInputIdxMismatch
            _logger = functools.partial(
                _logger,
                self.tensor_static_input_idxs,
                status,
            )
            return status, _logger

        # the cudagraph managed tensors which died upon recording must also die upon
        # this invocation. it is too late to check after we've replayed the graph,
        # because we would have already written over their memory.
        for idx in self.cudagraph_managed_idxs:
            if not self.preserved_aliased_inputs[idx]:
                inputs[idx] = None  # type: ignore[call-overload]

        torch._check(
            self._check_liveness(
                self.expected_dead_indices_after_graph, self.path_weakrefs
            ),
            lambda: "TODO: graph recording observed an input tensor deallocate during graph "
            " recording that did not occur during replay. Please file an issue.",
        )
        return CheckInvariantStatus.SUCCESS, lambda: f"{CheckInvariantStatus.SUCCESS}"

    def num_descendants(self) -> int:
        # [FUNCTION DOC] Returns the total count of descendants (children, grandchildren, etc.) of
        # this node.
        "Total number of descendents of this node"
        num_desc = 0
        for children in self.children.values():
            for child in children:
                num_desc += 1
                num_desc += child.num_descendants()
        return num_desc


def canonical_path_output_index(
    node: CUDAGraphNode | CUDAWarmupNode, depth: int, idx: int
) -> PathOutputIndex:
    # [FUNCTION DOC] HELPER: Returns the canonical (depth, index) identity of an output: for graph
    # nodes whose output aliases a prior graph output, that prior location; for aliases of a new
    # output, the earlier index at the same depth; otherwise the output's own location.
    if isinstance(node, CUDAGraphNode):
        alias_info = node.output_storage_alias[idx]
        if isinstance(alias_info, AliasesPriorGraphOutput):
            return alias_info.index
        if isinstance(alias_info, AliasesNewOutput):
            return (depth, alias_info.index)
    return (depth, idx)


def collect_path_user_visible_storage_groups(
    path: tuple[CUDAGraphNode | CUDAWarmupNode, ...],
) -> tuple[PathUserVisibleStorageGroup, ...]:
    # [FUNCTION DOC] Computes, for a path of nodes, groups of output positions that share storage
    # with a user-visible output. First collects data pointers and canonical locations of user-
    # visible outputs (raises an assertion error if an index is out of range), then adds other
    # outputs on the path that alias them by data pointer or canonical location. Returns a tuple of
    # groups, one per data pointer; each entry holds the node's output weakrefs, tensor weakrefs,
    # cached tensors (or None for warmup nodes) and the index. Skips outputs without a weakref. Pure
    # function.
    """Return path storage-alias groups for user-visible output cloning.

    Starts from user-visible outputs, then includes outputs on the same path
    that may alias those outputs by recorded storage pointer or alias metadata.
    This is cached per node path so aliases discovered on a different tree path
    do not affect the current generation transition.
    """
    selected_outputs: OrderedSet[PathOutputIndex] = OrderedSet()
    selected_alias_outputs: OrderedSet[PathOutputIndex] = OrderedSet()
    selected_data_ptrs: OrderedSet[int] = OrderedSet()

    # First, collect storage pointers and canonical outputs referenced by
    # user-visible outputs.
    for depth, node in enumerate(path):
        for idx in node.user_visible_output_idxs:
            if idx >= len(node.outputs_weakrefs):
                raise AssertionError(
                    "expected user_visible_output_idxs to index outputs_weakrefs"
                )
            selected_outputs.add((depth, idx))
            selected_alias_outputs.add(canonical_path_output_index(node, depth, idx))

            storage_ref = node.outputs_weakrefs[idx]
            if storage_ref is not None:
                selected_data_ptrs.add(storage_ref.data_ptr())

    # Then, collect outputs that may alias a user-visible output.
    for depth, node in enumerate(path):
        for idx, storage_ref in enumerate(node.outputs_weakrefs):
            output_index = (depth, idx)

            alias_index = canonical_path_output_index(node, depth, idx)
            if (
                output_index in selected_outputs
                or alias_index in selected_alias_outputs
                or storage_ref is not None
                and storage_ref.data_ptr() in selected_data_ptrs
            ):
                selected_outputs.add(output_index)

    entries_by_data_ptr: dict[int, list[PathUserVisibleOutputEntry]] = {}
    for depth, idx in selected_outputs:
        node = path[depth]
        output_weakrefs = node.outputs_weakrefs
        storage_ref = output_weakrefs[idx]
        if storage_ref is None:
            continue

        cached_tensor_outputs = (
            node.cached_tensor_outputs if isinstance(node, CUDAGraphNode) else None
        )
        data_ptr = storage_ref.data_ptr()
        entries_by_data_ptr.setdefault(data_ptr, []).append(
            (
                output_weakrefs,
                node.tensor_weakrefs,
                cached_tensor_outputs,
                idx,
            )
        )

    return tuple(tuple(entries) for entries in entries_by_data_ptr.values())


def get_cudagraph_segments(pool_id: tuple[int, int]) -> Any:
    # [FUNCTION DOC] HELPER: Returns the allocator memory snapshot segments that belong to the given
    # pool id.
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


def get_block_addrs(pool_id: tuple[int, int], live_only: bool = True) -> list[int]:
    # [FUNCTION DOC] HELPER: Returns start addresses of blocks in the pool's segments; by default
    # only allocated (live) blocks, or all blocks when live_only is false.
    blocks = []

    for segment in get_cudagraph_segments(pool_id):
        addr = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated" or not live_only:
                blocks.append(addr)

            addr += block["size"]

    return blocks


def format_tb(frames: list[Any]) -> str:
    # [FUNCTION DOC] HELPER: Formats a list of frame records (filename, line, name) from allocator
    # history into a traceback string.
    formatted_traceback = [
        traceback.FrameSummary(entry["filename"], entry["line"], entry["name"])
        for entry in frames
    ]

    return "".join(traceback.format_list(formatted_traceback))


def check_memory_pool(
    device: int,
    pool_id: tuple[int, int],
    live_storages_ptrs: list[StorageWeakRefWrapper],
) -> None:
    # [FUNCTION DOC] Validates that the live allocations in a cudagraph pool exactly match the
    # tracked live storages. Raises an assertion error if any argument is not a
    # StorageWeakRefWrapper. Fast path asks the C++ allocator; if consistent, returns. Otherwise it
    # runs garbage collection, synchronizes, scans the snapshot, raises through the check mechanism
    # if expected storages are not allocated, and raises RuntimeError listing untracked allocations
    # with history, optional objgraph backref images (written to files in the working directory when
    # configured) and debugging hints. Side effects: gc, synchronize, possible file output.
    """Validate cudagraph pool allocations against tracked live storages and surface leaks."""
    if not all(isinstance(elem, StorageWeakRefWrapper) for elem in live_storages_ptrs):
        raise AssertionError(
            "expected all live_storages_ptrs to be StorageWeakRefWrapper"
        )
    unique_storages = {stor.data_ptr() for stor in live_storages_ptrs if stor()}  # noqa: set_linter

    # check if there is a divergence first, then do the expensive snapshot call after
    # we know it will error
    if torch._C._cuda_checkPoolLiveAllocations(device, pool_id, unique_storages):
        return

    # at this point we are past the fast-path. we have seen rare cases where a dead tensor is dead,
    # but hasn't been gc'd yet, and gives false positive for allocated_not_in_live_storages
    gc.collect()
    torch.cuda.synchronize()

    segments = get_cudagraph_segments(pool_id)

    allocated_not_in_live_storages = {}

    for segment in segments:
        addr = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                if addr not in unique_storages:
                    allocated_not_in_live_storages[addr] = block
                else:
                    unique_storages.remove(addr)

            addr += block["size"]

    torch._check(
        len(unique_storages) == 0,
        lambda: (
            f"These storage data ptrs are not allocated in pool {pool_id} but should be: {unique_storages}. "
            f"This could be a bug in inductor aliasing tracking or in a custom op's meta function. Please file an issue."
        ),
    )

    if len(allocated_not_in_live_storages) != 0:
        formatted = []
        for dp, block in allocated_not_in_live_storages.items():
            trace = format_tb(block.get("frames", []))
            # pyrefly: ignore [bad-argument-type]
            formatted.append(f"Data Pointer: {dp}, history: \n{trace}")
        formatted_s = "\n".join(formatted)

        history_hint = ""
        if not config.triton.cudagraph_trees_history_recording:
            history_hint = "- Set torch._inductor.config.triton.cudagraph_trees_history_recording = True for allocation origins\n"

        objgraph_hint = (
            (
                "- Objgraph backrefs disabled; set torch._inductor.config.triton.cudagraph_trees_objgraph = True "
                "(refs_live_tensor_{index}.svg)\n"
            )
            if not config.triton.cudagraph_trees_objgraph
            else ""
        )
        objgraph_files_hint = ""

        if config.triton.cudagraph_trees_objgraph:
            # pyrefly: ignore  # import-error
            import objgraph

            generated_files: list[str] = []

            tensors = objgraph.by_type("torch.Tensor")
            for index, bad_dp in enumerate(allocated_not_in_live_storages):
                bad_tensor = next(
                    (t for t in tensors if bad_dp in collect_cuda_data_ptrs(t)), None
                )
                if bad_tensor is None:
                    continue
                filename = f"refs_live_tensor_{index}.svg"
                objgraph.show_backrefs(bad_tensor, filename=filename, max_depth=5)
                generated_files.append(filename)
            if generated_files:
                objgraph_files_hint = (
                    f"- Objgraph backrefs written: {', '.join(generated_files)}\n"
                )

        dangling_addrs = OrderedSet(allocated_not_in_live_storages.keys())
        msg = (
            f"Detected {len(allocated_not_in_live_storages)} tensor(s) in the cudagraph pool not tracked as outputs. "
            f"All live allocations must be tracked for correctness.\n"
            f"Debugging:\n"
            f"{history_hint}"
            f"{objgraph_hint}"
            f"- Search gc.get_objects() for tensors with data_ptr() in {dangling_addrs}\n"
            f"- Use refcycle to find what is preventing cleanup\n"
            f"{objgraph_files_hint}"
            f"Allocations:\n{formatted_s}"
        )
        raise RuntimeError(msg)


class ExecutionState(Enum):
    # [CLASS DOC] Enum of the state of the tree manager path: NONE (no live memory in the pool),
    # WARMUP, RECORDING, EXECUTION, reflecting the most recently executed node.
    """
    Represents the state of the CUDAGraph Tree. Will be None if there is no live current memory allocated
    in the cuda graph pool. Otherwise will reflect the state of the most recently executed node.
    """

    NONE = auto()
    WARMUP = auto()
    RECORDING = auto()
    EXECUTION = auto()


class CompilationMode(Enum):
    # [CLASS DOC] Enum of what a function is compiled for: FORWARD, BACKWARD or INFERENCE.
    FORWARD = auto()
    BACKWARD = auto()
    INFERENCE = auto()


class CUDAGraphTreeManager:
    # [CLASS DOC] Per-device controller of CUDA graph trees. Groups recordings into a tree, enforces
    # that later executions follow the same order and output liveness, ends the current path when it
    # is valid (all outputs dead or a new generation), and manages warmups before recording. State:
    # roots by function id, id to function, stack trace and mode tables, warmed up and warned
    # function sets, a dedicated stream, a graph that keeps the memory pool alive plus the pool
    # handle, counters for graph, function and warmup ids, the current node and path state, current
    # generation, rerecord counts and mutation hints per node and function, flags for pending
    # backwards and user visible output cloning. Key methods: add_function, run, _run (decision
    # logic), record_function, run_eager, try_end_curr_* , dealloc_current_path_weakrefs,
    # apply_checkpoint_execution_state_in_allocator, shutdown.
    """
    Groups individual recordings or executions of cuda graphs into a tree of recordings,
    and checks required invariants, and manages warmups of graphs.

    When graphs are recorded in the same tree, it enforces subsequent execution
    to follow the same order and have the same output tensor livespans. To remove
    unnecessary coupling of cuda graphs (and additional imposed invariants),
    the tree manager will end a currently recording tree whenever it is valid - when
    the memory pool no longer has any live allocations.

    We ignore outputs from a previous generation that correspond to prior model outputs.
    Currently this is hardcoded `GenerationTracker.generation` tracked in torch dynamo.
    # TODO: make generation increment configurable, warn on overwrite.

    We run graph warmups in the cudagraph memory pool and return the result on the first invocation
    of a function. For many models it is important to reclaim activations as you run the backward.
    If we were to warm up the model and keep an extra copy of the inputs around to subsequently
    use for recording, we would incur a memory penalty. Additionally, if we are part way through training
    your model and need to recompile, memory will be allocated to the cuda graph pool, so we run this
    warmup run in the cuda graph memory pool. As for recording, warm up needs the state of live tensors
    to be accurately reflected so we checkpoint the allocator state if we need to warm up following graph
    replay.
    """

    def __init__(self, device_index: int) -> None:
        # [FUNCTION DOC] Sets up an empty manager for the device: empty tables and counters, enables
        # cached tensors globally, and under the global capture lock on that device creates a
        # dedicated stream, an empty captured graph and a pool handle to keep the pool alive. Path
        # state starts as NONE with no current node and generation -1; also decides whether alias
        # invalidation is disabled (only via a justknob in fbcode). KNOWN ISSUE: compile_id is not
        # initialized here and only set in run; reading it earlier would fail.
        # roots are functions which have no dependencies on an other node. I.e.,
        # when they are first invoked, none of their inputs are outputs are outputs
        # of another node, nor are there any live outputs of another node whose
        # liveness would create a dependency.
        self.roots: dict[FunctionID, list[CUDAGraphNode]] = defaultdict(list)

        # mapping from function id to wrapped function
        self.ids_to_funcs: dict[FunctionID, WrappedFunction] = {}

        self.ids_to_stack_traces: dict[FunctionID, StackTraces | None] = {}

        self.warmed_up_functions: OrderedSet[FunctionID] = OrderedSet()
        # if we fail to increment generation, and are stuck warming up,
        # only warn on each function once
        self.warned_functions: OrderedSet[FunctionID] = OrderedSet()
        torch._C._set_cached_tensors_enabled(True)

        # warn only once if a function mutates inputs
        self.warned_mutation: OrderedSet[FunctionID] = OrderedSet()

        # NB: cuda caching allocator will remember the stream a segment is allocated to
        # and only allocate that segment to the same stream. we need to use a single stream
        # for all allocations to the memory pool, otherwise the allocations to separate streams
        # will not be reused; separate recordings would have use the same memory pool, but not
        # the same memory.

        with graph_capture_lock, torch.cuda.device(device_index):
            torch.cuda.synchronize()
            self.stream = torch.cuda.Stream()
            self.stream.wait_stream(torch.cuda.current_stream())

            # Keeps Memory Pool Alive
            self.graph: torch.cuda.CUDAGraph | None = torch.cuda.CUDAGraph()
            self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()

            with (
                warnings.catch_warnings(record=True),
                torch.cuda.graph(
                    self.graph,
                    pool=self.cuda_graphs_thread_pool,
                    stream=self.stream,
                    capture_error_mode="thread_local",
                ),
            ):
                pass

        self.graph_counter = itertools.count(0)
        self.func_counter = itertools.count(0)

        # mapping from graph_id to (function id to mutation type hint) since we are
        # specializing on a particular combination of Parent Node -> Function ID.
        self.non_cudagraph_managed_mutation_hint: dict[
            GraphID | None, dict[FunctionID, bool]
        ] = defaultdict(dict)
        self.warmup_node_counter = itertools.count(start=-1, step=-1)

        # mapping from graph_id to (function id to re-record count). We fall back to
        # eager function if a function is re-recorded frequently on a node.
        self.num_rerecord: dict[GraphID | None, dict[FunctionID, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )

        # whether we the current node is in a state of warmup, recording, execution. If
        # there is no current node the state will be ExecutionState.None.
        self.path_state = ExecutionState.NONE
        self.active_warmup_function: FunctionID | None = None
        self.device_index = device_index

        # the most recently invoked cudagraph wrapping of a function. Will be None
        # when there is no output from a previous recording or execution whose memory
        # we need to respect in the cuda caching allocation. If you incremented generation,
        # this will also be none, as ignore those allocations.
        self.current_node: CUDAGraphNode | CUDAWarmupNode | None = None

        # current generation of cudagraph invocations. when torch.compile is run
        # we increment the current generation. are willing to ignore live outputs
        # of a previous generation in checking liveness.
        self.current_gen: int = -1

        # number of instances we are in execution and failed to match to an
        # existing child
        self.debug_fail_counter = 0
        # number of instances we had to checkpoint the function
        self.debug_checkpointing_counter = 0

        self.id_to_mode: dict[FunctionID, CompilationMode] = {}
        self.id_to_compile_id: dict[FunctionID, CompileId | None] = {}
        self.has_live_user_visible_output_cloning = False

        # Note: [Backward Generation Handling]
        # We generally perform a sequence of forward executions followed by backward executions.
        # If multiple torch.compile wrapped forwards are executed with their backwards pending,
        # we should not disregard the outputs from a prior torch.compile since the entire training
        # loop hasn't completed.  Occasionally, a backward pass corresponding to a forward pass may
        # not be executed, so we cannot wait for all pending forward pass backward completions, so
        # we cannot wait for all backwards to have been invoked. Instead we wait for a single backward
        # invocation. Triggering a backward pass typically doesn't lead to another torch.compile
        # invocation, making it less likely for the generation to increase between multiple
        # backward calls. The following use case is covered by this approach:
        # mod1 = torch.compile(...)
        # mod2 = torch.compile(...)
        # mod2(mod1(x)).sum().backward()

        self.running_forwards_with_pending_backwards = False
        self.mode: CompilationMode | None = None

        self.disable_invalidate_aliases = (
            False
            if not torch._environment.is_fbcode()
            else torch._utils_internal.justknobs_check(
                "pytorch/inductor:disable_cudagraph_alias_invalidation"
            )
        )

    def run(self, new_inputs: list[InputType], function_id: FunctionID) -> OutputType:
        # [FUNCTION DOC] Entry point for a wrapped function call: raises an assertion error if the
        # manager was shut down; sets mode and compile id for the function, runs the internal logic,
        # and afterwards updates the pending backwards flag (set after a forward, cleared after a
        # backward). Returns the outputs.
        if self.graph is None:
            raise AssertionError("Running CUDAGraph after shutdown")
        self.mode = self.id_to_mode[function_id]
        self.compile_id = self.id_to_compile_id[function_id]
        out = self._run(new_inputs, function_id)

        # The forwards are only pending following invocation, not before
        if self.mode == CompilationMode.FORWARD:
            self.running_forwards_with_pending_backwards = True
        elif self.mode == CompilationMode.BACKWARD:
            self.running_forwards_with_pending_backwards = False

        return out

    def set_to_running_backward(self) -> None:
        # [FUNCTION DOC] Marks that backward is starting: clears the pending backwards flag and sets
        # mode to backward.
        self.running_forwards_with_pending_backwards = False
        self.mode = CompilationMode.BACKWARD

    def _get_cuda_graph_recorded_tensor_checker(self) -> Callable[[Tensor], bool]:
        # [FUNCTION DOC] HELPER: Returns the recorded tensor membership test of the current node, or
        # a test that always returns false when there is no current node.
        return (
            self.current_node._is_cuda_graph_recorded_tensor
            if isinstance(self.current_node, (CUDAGraphNode, CUDAWarmupNode))
            else lambda _: False
        )

    def new_warmup_node_id(self) -> GraphID:
        # [FUNCTION DOC] Returns a new warmup node id from a decreasing counter (negative ids).
        return GraphID(next(self.warmup_node_counter))

    def _update_non_cudagraph_managed_mutation(
        self, function_id: FunctionID, inputs: list[InputType]
    ) -> None:
        # [FUNCTION DOC] HELPER: Determines and caches (per current node and function) whether the
        # function mutates inputs that are neither parameters or buffers nor cudagraph recorded
        # tensors. Records the hint and, once per function, logs a skip reason and bumps the skip
        # counter when mutation is found.
        node_id = self._get_node_id()
        if maybe_mutation_str := check_for_mutation(
            self.ids_to_funcs[function_id],
            inputs,
            self._get_cuda_graph_recorded_tensor_checker(),
        ):
            self.non_cudagraph_managed_mutation_hint[node_id][function_id] = True
            # warn once per function_id
            if function_id in self.warned_mutation:
                return
            self.warned_mutation.add(function_id)
            log_cudagraph_skip_and_bump_counter(maybe_mutation_str)
        else:
            self.non_cudagraph_managed_mutation_hint[node_id][function_id] = False

    def _get_node_id(self) -> GraphID | None:
        # [FUNCTION DOC] HELPER: Returns the id of the current node, or None if there is none;
        # raises RuntimeError for an unknown node type.
        if self.current_node is None:
            return None
        elif isinstance(self.current_node, (CUDAGraphNode, CUDAWarmupNode)):
            return self.current_node.id
        else:
            raise RuntimeError(f"Unknown node type {type(self.current_node)}")

    def exceed_rerecord_limit(
        self, node_id: GraphID | None, function_id: FunctionID
    ) -> bool:
        # [FUNCTION DOC] Returns true if the function was re-recorded at this node more times than
        # the configured unexpected re-record limit.
        return (
            self.num_rerecord[node_id][function_id]
            > torch._inductor.config.triton.cudagraph_unexpected_rerecord_limit
        )

    def _run(self, new_inputs: list[InputType], function_id: FunctionID) -> OutputType:
        # [FUNCTION DOC] Core dispatch for one call. Lazily ends the current recording or warmup if
        # possible; at generation boundaries with user visible output cloning ends the current
        # execution. Falls back to the plain model if the function mutates uncontrolled inputs or
        # exceeded the re-record limit. Runs eagerly (warmup, with allocator checkpoint if in
        # execution) if the function is not warmed up yet, if already warming up, or if forced.
        # Otherwise tries each child node of the current node (or root) with matching function id,
        # replaying the first that passes its invariants; if none pass it may end the current
        # execution and retry, fall back to the model for mutation or re-record limit, or checkpoint
        # and record a new node. Counts only unexpected mismatches (not static input mismatches)
        # toward the re-record limit. Returns outputs.
        # we will try to end the current execution lazily, since
        # we don't want to do unnecessary checking of the existing outputs
        # on the hot path, but both recording and warmup only happen once
        # so we check up front
        if self.in_recording:
            self.try_end_curr_recording(function_id)

        if self.in_warmup:
            self.try_end_curr_warmup(function_id)

        if (
            self.has_live_user_visible_output_cloning
            and self.path_state == ExecutionState.EXECUTION
        ):
            curr_generation = self.get_curr_generation()
            # current_node is cleared lazily, so the first invocation in a new
            # generation may still see children from the previous generation.
            # Enforce the existing generation boundary before matching children.
            if self.in_new_torch_compile_invocation(curr_generation):
                self.try_end_curr_execution(
                    curr_generation,
                    skip_dead_output_cleanup=True,
                )

        node_id = self._get_node_id()
        if function_id not in self.non_cudagraph_managed_mutation_hint[node_id]:
            self._update_non_cudagraph_managed_mutation(function_id, new_inputs)

        # Early exit if the function mutates inputs which are neither parameters/buffers nor
        # cudagraph recorded tensors. This check should happen after `try_end_curr_recording`
        # and `try_end_curr_warmup` which may change self.current_node.
        if self.non_cudagraph_managed_mutation_hint[node_id][
            function_id
        ] or self.exceed_rerecord_limit(node_id, function_id):
            return self.ids_to_funcs[function_id].model(new_inputs)

        # warming up a function and subsequentally recording may use different memory addresses
        # because both depend on the state of the caching allocator. if we warm up graph A,
        # then warm up graph B and make more allocations, the subsequent recording of A will not
        # necessarily use the same addresses as in the warm up. Thus any warm up of a node can only
        # be followed by warm up runs.
        if (
            (
                not (
                    function_id in self.warmed_up_functions
                    or config.triton.skip_cudagraph_warmup
                )
            )
            or self.in_warmup
            or config.triton.force_cudagraphs_warmup
        ):
            # If we are in the middle of executing cuda graphs, then we need to checkpoint memory state.
            # Both Recording and Warmup will be reflected in the allocator and don't need changes
            if self.path_state == ExecutionState.EXECUTION:
                self.apply_checkpoint_execution_state_in_allocator()

            with graph_capture_lock:
                return self.run_eager(new_inputs, function_id)

        if isinstance(self.current_node, CUDAWarmupNode):
            raise AssertionError("expected current_node to not be a CUDAWarmupNode")
        child_nodes = (
            self.roots if self.current_node is None else self.current_node.children
        )

        if not self.in_recording:
            unexpected_rerecord = False
            unexpected_rerecord_reason = None
            for child in child_nodes[function_id]:
                # here we are checking memory consistency between recording and execution,
                # as well as things like stability of tensor locations, etc
                # and other
                status, status_logger = child.check_invariants(new_inputs)
                if status == CheckInvariantStatus.SUCCESS:
                    return self.execute_node(child, new_inputs)

                if status != CheckInvariantStatus.SUCCESS:
                    # StaticInputIdxMismatch fires on non-cudagraph-managed
                    # static inputs (nn parameters and mark_static_address
                    # tensors) whose identity can churn under
                    # inline_inbuilt_nn_modules. We re-record but don't count
                    # those toward cudagraph_unexpected_rerecord_limit. All
                    # other mismatch reasons do count.
                    if status != CheckInvariantStatus.StaticInputIdxMismatch:
                        unexpected_rerecord = True
                    # Only compute detailed reason when debug logging is enabled
                    if log.isEnabledFor(logging.DEBUG):
                        unexpected_rerecord_reason = status_logger()
                        log.debug(
                            "[%s] Re-recording function=%s, mode=%s, reason=%s",
                            self.compile_id,
                            self.get_func_name(function_id),
                            self.id_to_mode[function_id].name,
                            unexpected_rerecord_reason,
                        )
                    else:
                        # Defer reason computation until needed (for exceed_rerecord_limit)
                        unexpected_rerecord_reason = status_logger

            # now that we know the new function can't be run as a child of the
            # current node, if it is a root, try to end the current execution.
            # as noted above, we want to do this lazily to avoid having to
            # check all existing outputs
            if self.current_node is not None and function_id in self.roots:
                self.try_end_curr_execution()

                # run again to hit the root matching case which must succeed
                if self.current_node is None:
                    return self.run(new_inputs, function_id)

            if len(self.ids_to_funcs[function_id].mutated_input_idxs) > 0:
                self._update_non_cudagraph_managed_mutation(function_id, new_inputs)
                if self.non_cudagraph_managed_mutation_hint[self._get_node_id()][
                    function_id
                ]:
                    return self.ids_to_funcs[function_id].model(new_inputs)

            # nb: run before checkpointing because checkpointing is slow, and we will
            # be using the eager caching allocator pool which does not require live
            # accounting of tensors in cudagraph allocator
            if unexpected_rerecord:
                curr_node_id = self._get_node_id()
                self.num_rerecord[curr_node_id][function_id] += 1
                if self.exceed_rerecord_limit(curr_node_id, function_id):
                    _id = curr_node_id.id if curr_node_id else None
                    # unexpected_rerecord_reason is either a string (if debug was enabled)
                    # or a callable (if debug was disabled)
                    if unexpected_rerecord_reason is None:
                        raise AssertionError(
                            "expected unexpected_rerecord_reason to not be None"
                        )
                    reason = (
                        unexpected_rerecord_reason
                        if isinstance(unexpected_rerecord_reason, str)
                        else unexpected_rerecord_reason()
                    )
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraph due to function {function_id.id} exceeding max "
                        f"re-recording limit "
                        f"(={torch._inductor.config.triton.cudagraph_unexpected_rerecord_limit}) "
                        f"on cudagraph node {_id} due to {reason}."
                    )
                    return self.ids_to_funcs[function_id].model(new_inputs)

            # at this point, we necessarily will do a new recording
            self.debug_fail_counter += 1

            self.try_end_curr_execution()
            if self.current_node is not None:
                self.apply_checkpoint_execution_state_in_allocator()

        # now, we are in a recording state !
        with graph_capture_lock:
            return self.record_function(new_inputs, function_id)

    def shutdown(self) -> None:
        # [FUNCTION DOC] Tears the manager down: walks all nodes of all trees, removes their cached
        # tensors and graph references to break reference cycles through gradients, then drops the
        # manager's graph, roots and current node. The manager cannot be used afterward.
        """
        Remove all cached tensors in all nodes. Because cached tensors can hold gradients which in turn
        might reference a backward which invokes a CUDA Graph Node, we have to manually clear them on shutdown
        to avoid a reference cycle.
        """
        nodes = []
        for roots in self.roots.values():
            nodes.extend(roots)

        while nodes:
            node = nodes.pop()
            for children in node.children.values():
                nodes.extend(children)
            node.remove_node_cached_tensors()
            node.graph = None

        self.graph = None
        self.roots = None  # type: ignore[assignment]
        self.current_node = None

    def record_function(
        self, new_inputs: list[InputType], function_id: FunctionID
    ) -> OutputType:
        # [FUNCTION DOC] Records a new node for the function under the current node (or as a root):
        # installs the recording callbacks, allocates a graph id, synchronizes, creates the
        # CUDAGraphNode (which records), registers it as root or child, makes it current, sets state
        # to RECORDING, updates the generation, synchronizes again and returns the recorded outputs
        # from run_first_inputs. Raises an assertion error if the current node is a warmup node.
        if isinstance(self.current_node, CUDAWarmupNode):
            raise AssertionError("expected current_node to not be a CUDAWarmupNode")
        with torch._dynamo.callback_handler.install_callbacks(
            CallbackTrigger.CUDAGRAPH_RECORDING, str(self.compile_id)
        ):
            graph_id = self.new_graph_id()
            log.debug(
                "[%s] Recording function=%s, mode=%s, cuda_graph_id=%d, inputs: %s",
                self.compile_id,
                self.get_func_name(function_id),
                self.id_to_mode[function_id].name,
                graph_id.id,
                format_inputs_log(new_inputs),
            )
            torch.cuda.synchronize()
            node = CUDAGraphNode(
                self.ids_to_funcs[function_id],
                graph_id,
                self.current_node,
                new_inputs,
                self.cuda_graphs_thread_pool,
                self.device_index,
                self.ids_to_stack_traces[function_id],
                self.stream,
                self.mode,
                self.compile_id,
            )
            if self.current_node is None:
                self.roots[function_id].append(node)
            else:
                self.current_node.add_child(function_id, node)
            self.current_node = node
            self.path_state = ExecutionState.RECORDING
            self.update_generation()
            torch.cuda.synchronize()
            return node.run_first_inputs(new_inputs)

    def execute_node(
        self, node: CUDAGraphNode, new_inputs: list[InputType]
    ) -> OutputType:
        # [FUNCTION DOC] Replays an existing node: makes it current, sets state to EXECUTION,
        # updates generation and runs the node on the inputs. Returns outputs.
        self.current_node = node
        self.path_state = ExecutionState.EXECUTION
        self.update_generation()
        return node.run(new_inputs)

    def run_eager(
        self, new_inputs: list[InputType], function_id: FunctionID
    ) -> OutputType:
        # [FUNCTION DOC] Runs the function eagerly as warmup in the pool: marks it warmed up,
        # creates a CUDAWarmupNode child of the current node, makes it current with state WARMUP,
        # updates the generation, tracks the active warmup function during the run and returns the
        # outputs. Logs whether it is warmup or an already warm eager run.
        # this is only stored on current node, because when we start a new path,
        # we will deallocate it
        already_warm = function_id in self.warmed_up_functions
        func_name = self.get_func_name(function_id)
        if not already_warm:
            log.debug(
                "[%s] Running warmup function=%s, mode=%s",
                self.compile_id,
                func_name,
                self.id_to_mode[function_id].name,
            )
        else:
            log.debug(
                "[%s] Running eager function=%s",
                self.compile_id,
                func_name,
            )
        self.warmed_up_functions.add(function_id)
        node = CUDAWarmupNode(
            self.ids_to_funcs[function_id],
            self.current_node,
            self.cuda_graphs_thread_pool,
            self.graph,
            self.device_index,
            self.ids_to_stack_traces[function_id],
            self.stream,
            already_warm,
            self.new_warmup_node_id(),
        )
        self.current_node = node
        self.path_state = ExecutionState.WARMUP
        self.update_generation()
        self.active_warmup_function = function_id
        result = node.run(new_inputs)
        self.active_warmup_function = None
        return result

    def mark_warmup_incomplete(self) -> None:
        # [FUNCTION DOC] If a function is currently warming up, removes it from the warmed up set so
        # it will be warmed again.
        if self.active_warmup_function is not None:
            self.warmed_up_functions.discard(self.active_warmup_function)

    def new_graph_id(self) -> GraphID:
        # [FUNCTION DOC] Returns the next graph id from the graph counter.
        return GraphID(next(self.graph_counter))

    def new_func_id(self) -> FunctionID:
        # [FUNCTION DOC] Returns the next function id from the function counter.
        return FunctionID(next(self.func_counter))

    def get_func_name(self, function_id: FunctionID) -> str:
        # [FUNCTION DOC] Returns the name of the model of the function id, or the word unknown if it
        # has none.
        return getattr(self.ids_to_funcs[function_id].model, "__name__", "unknown")

    def add_function(
        self,
        model: ModelType,
        inputs: list[InputType],
        static_input_idxs: Sequence[int],
        stack_traces: StackTraces | None,
        mode: CompilationMode,
        constants: tuple[torch.Tensor, ...],
        placeholders: tuple[PlaceholderInfo, ...],
        mutated_input_idxs: tuple[int, ...],
        kernel_free_cudagraph: bool,
        user_visible_output_idxs: tuple[int, ...],
        compile_id: CompileId | None,
    ) -> tuple[
        ModelType,
        OutputType,
    ]:
        # [FUNCTION DOC] Registers a function with the manager and runs it once. Allocates a
        # function id; backward mode clears user-visible output indices; if any remain, enables user
        # visible output cloning. Stores stack traces, a WrappedFunction (with only CUDA tensor
        # constants kept), mode and compile id; creates a callable bound to run with that id and
        # registers it with the container so the manager lives as long as it does. Returns the
        # callable and the outputs of the first call.
        id = self.new_func_id()
        if mode == CompilationMode.BACKWARD:
            user_visible_output_idxs = ()
        user_visible_output_idxs_set = frozenset(user_visible_output_idxs)
        if user_visible_output_idxs_set:
            self.has_live_user_visible_output_cloning = True
        self.ids_to_stack_traces[id] = stack_traces
        self.ids_to_funcs[id] = WrappedFunction(
            model,
            list(static_input_idxs),
            id,
            tuple(t for t in constants if isinstance(t, torch.Tensor) and t.is_cuda),
            placeholders,
            mutated_input_idxs,
            kernel_free_cudagraph,
            user_visible_output_idxs_set,
        )
        self.id_to_mode[id] = mode
        self.id_to_compile_id[id] = compile_id
        fn = functools.partial(self.run, function_id=id)

        # container needs to set clean up when fn dies
        get_container(self.device_index).add_strong_reference(fn)
        return fn, fn(inputs)

    @property
    def in_recording(self) -> bool:
        # [FUNCTION DOC] Property: true if the path state is RECORDING.
        return self.path_state == ExecutionState.RECORDING

    @property
    def in_warmup(self) -> bool:
        # [FUNCTION DOC] Property: true if the path state is WARMUP.
        return self.path_state == ExecutionState.WARMUP

    def get_roots(self) -> Iterator[CUDAGraphNode]:
        # [FUNCTION DOC] Generator over all root nodes of all trees.
        for nodes in self.roots.values():
            yield from nodes

    @property
    def current_node(self) -> CUDAGraphNode | CUDAWarmupNode | None:
        # [FUNCTION DOC] Property getter for the current node (CUDAGraphNode, CUDAWarmupNode or
        # None).
        return self._current_node

    @current_node.setter
    def current_node(self, value: CUDAGraphNode | CUDAWarmupNode | None) -> None:
        # [FUNCTION DOC] Setter for the current node; setting None also resets the path state to
        # NONE.
        self._current_node = value
        if value is None:
            self.path_state = ExecutionState.NONE

    def update_generation(self) -> None:
        # [FUNCTION DOC] Stores the current generation (user mark step counter or dynamo generation)
        # as the manager current generation.
        self.current_gen = self.get_curr_generation()

    @staticmethod
    def get_curr_generation() -> int:
        # [FUNCTION DOC] Static method: returns the user marked step counter if it was ever changed
        # (non zero), otherwise the dynamo generation counter.
        if MarkStepBox.mark_step_counter != 0:
            return MarkStepBox.mark_step_counter

        return GenerationTracker.generation

    @staticmethod
    def user_invoked_mark_step() -> bool:
        # [FUNCTION DOC] Static method: true if the user has called the mark step API (counter non
        # zero).
        return MarkStepBox.mark_step_counter != 0

    def can_start_new_generation(self, curr_generation: int | None = None) -> bool:
        # [FUNCTION DOC] Returns true if a new generation of invocations can begin: requires a
        # different generation than the recorded one, and either the user marked a step or no
        # forwards are still waiting for their backward.
        if not self.in_new_torch_compile_invocation(curr_generation):
            return False

        if self.user_invoked_mark_step():
            return True

        return not self.running_forwards_with_pending_backwards

    def in_new_torch_compile_invocation(
        self, curr_generation: int | None = None
    ) -> bool:
        # [FUNCTION DOC] Returns true if the given (or current) generation differs from the
        # manager's recorded generation.
        if curr_generation is None:
            curr_generation = self.get_curr_generation()
        return self.current_gen != curr_generation

    def _clone_current_path_live_user_visible_outputs(self) -> None:
        # [FUNCTION DOC] HELPER: Clones live user visible outputs of the current path before they
        # would be overwritten: gathers, per alias group, the first live storage and clones those
        # groups. Does nothing if no groups. Raises an assertion error if there is no current node.
        if self.current_node is None:
            raise AssertionError("expected current_node to not be None")
        path_user_visible_storage_groups = (
            self.current_node.path_user_visible_storage_groups
        )
        if not path_user_visible_storage_groups:
            return

        storages_to_clone: list[
            tuple[StorageWeakRefPointer, PathUserVisibleStorageGroup]
        ] = []

        for output_entries in path_user_visible_storage_groups:
            for (
                output_weakrefs,
                _tensor_weakrefs,
                _cached_tensor_outputs,
                idx,
            ) in output_entries:
                storage_and_ptr = maybe_deref(output_weakrefs[idx])
                if storage_and_ptr is not None:
                    storage_ptr, _data_ptr = storage_and_ptr
                    storages_to_clone.append((storage_ptr, output_entries))
                    break

        if storages_to_clone:
            self._clone_live_user_visible_storages(storages_to_clone)

    def _clone_live_user_visible_storages(
        self,
        storages_to_clone: Iterable[
            tuple[
                StorageWeakRefPointer,
                PathUserVisibleStorageGroup,
            ]
        ],
    ) -> None:
        # [FUNCTION DOC] HELPER: Replaces pool storages of live user-visible outputs by freshly
        # allocated regular memory: copies bytes into new buffers in one batch, swaps the data
        # pointers so existing tensors see the new memory, strips deleters from the swapped pool
        # block where it owns it, sets an error message on the old storage, then updates every alias
        # entry: clears tensor weakrefs, drops extra references, points the weakref at the
        # replacement and removes cached tensors. Side effects: allocations, mutation of tensors and
        # caches.
        replacements: list[
            tuple[
                UntypedStorage,
                UntypedStorage,
                PathUserVisibleStorageGroup,
            ]
        ] = []

        with torch.cuda.device(self.device_index):
            source_tensors = []
            replacement_tensors = []

            for storage_ptr, entries in storages_to_clone:
                source_storage = torch.UntypedStorage._new_with_weak_ptr(storage_ptr)
                replacement = torch.empty(
                    (source_storage.nbytes(),),
                    dtype=torch.uint8,
                    device=source_storage.device,
                )
                replacement_storage = replacement.untyped_storage()
                source_tensors.append(
                    torch.empty(
                        (), dtype=torch.uint8, device=source_storage.device
                    ).set_(source_storage, 0, (source_storage.nbytes(),), (1,))
                )
                replacement_tensors.append(replacement)
                replacements.append((source_storage, replacement_storage, entries))

            torch._foreach_copy_(replacement_tensors, source_tensors)

            for source_storage, replacement_storage, entries in replacements:
                # pyrefly: ignore[missing-attribute]
                source_storage._swap_data_ptr_(replacement_storage)
                placeholder_weakrefs = [
                    replacement_storage._weak_ref() for _ in entries
                ]

                # Warmup/recording storages own the graph-pool block after
                # the swap. Reconstructed replay storages are non-owning.
                if torch._C._has_Standard_Deleter(replacement_storage._cdata):
                    torch._C._free_And_Remove_DeleterFn(replacement_storage._cdata)
                torch._C._set_storage_data_ptr_access_error_msg(
                    replacement_storage._cdata,
                    "CUDAGraph output storage was cloned before a new generation.",
                )

                for (
                    output_weakrefs,
                    tensor_weakrefs,
                    cached_tensor_outputs,
                    idx,
                ), placeholder_weakref in zip(
                    entries, placeholder_weakrefs, strict=True
                ):
                    cached_t = (
                        cached_tensor_outputs[idx] if cached_tensor_outputs else None
                    )
                    output_ref = cast(StorageWeakRefWrapper, output_weakrefs[idx])
                    tensor_weakrefs[idx] = None
                    output_ref.remove_extra_reference()
                    output_ref.swap_weakref(placeholder_weakref)

                    if cached_tensor_outputs is not None and cached_t is not None:
                        torch._C._remove_cached_tensor(cached_t)
                        cached_tensor_outputs[idx] = None

    def try_end_curr_recording(self, function_id: FunctionID) -> None:
        """
        Check if the current recording can be terminated, either because all outputs of the
        previously recorded node are dead or because it was executed in a different
        generation. Will set current_node to None and in_recording to False if successful.
        """
        if not self.in_recording:
            raise AssertionError("expected in_recording to be True")
        if self.current_node is None:
            raise AssertionError("expected current_node to not be None")

        # multiple invocations, allow overwriting the previous generation
        if self.can_start_new_generation():
            self.clear_current_path_state_and_set_to_none(
                clone_live_user_visible_outputs=True,
                dealloc_current_path_weakrefs=True,
            )
            return

        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()
            return

        self.check_warn_on_unable_to_start_executing(function_id)

    def try_end_curr_execution(
        self,
        curr_generation: int | None = None,
        *,
        skip_dead_output_cleanup: bool = False,
    ) -> None:
        # [FUNCTION DOC] Tries to end the current recording: raises an assertion error unless
        # recording with a current node. If a new generation may start, clones user visible outputs,
        # deallocates path weakrefs and clears the current node; else if all outputs are dead,
        # clears the node; otherwise may warn about being unable to hit the fast path.
        """
        Check if the current executing node can be terminated, either because all outputs of the
        previously executed node are dead or because it was executed in a different generation.
        Will set current_node to None if successful.
        """

        if self.in_recording:
            raise AssertionError("expected in_recording to be False")
        if self.current_node is None:
            return

        if self.can_start_new_generation(curr_generation):
            self.clear_current_path_state_and_set_to_none(
                clone_live_user_visible_outputs=True
            )
            return

        if skip_dead_output_cleanup:
            return

        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()

    def try_end_curr_warmup(self, function_id: FunctionID) -> None:
        # [FUNCTION DOC] Tries to end the current warmup (raises an assertion error if no current
        # node). New generation: clone user visible outputs, deallocate path weakrefs, clear current
        # node. All outputs dead: clear current node. Otherwise may warn.
        if self.current_node is None:
            raise AssertionError("expected current_node to not be None")
        if self.can_start_new_generation():
            self._clone_current_path_live_user_visible_outputs()
            self.dealloc_current_path_weakrefs()
            self.current_node = None
            return

        if self.current_node.all_outputs_are_dead():
            self.current_node = None
            return

        self.check_warn_on_unable_to_start_executing(function_id)

    def check_warn_on_unable_to_start_executing(self, function_id: FunctionID) -> None:
        # [FUNCTION DOC] Emits a one time (per function) warning when in a new compile invocation
        # and the path already contains repeated nodes for this function with a repeated parent
        # pattern, which suggests outputs from a previous step still require backward and the fast
        # path cannot be hit. Raises an assertion error if there is no current node.
        "Warn if we in a potential loop where we are unable to hit fast path"
        if (
            function_id in self.warned_functions
            or not self.in_new_torch_compile_invocation()
        ):
            return

        if self.current_node is None:
            raise AssertionError("expected current_node to not be None")
        existing_nodes = [
            node
            for node in self.current_node._path_from_root
            if node.wrapped_function.id == function_id
        ]

        if len(existing_nodes) <= 1:
            return

        # repeated same pattern
        parents = OrderedSet(
            [
                n.parent.wrapped_function.id
                for n in itertools.chain(existing_nodes, (self.current_node,))
                if n.parent is not None
            ]
        )
        if len(parents) == len(existing_nodes):
            return

        self.warned_functions.add(function_id)
        warnings.warn(
            "Unable to hit fast path of CUDAGraphs because outputs from a previous step "
            "still require backward. Ensure backward() is invoked or detach outputs. "
            "You may also call torch.compiler.cudagraph_mark_step_begin() before each model invocation."
        )

    @staticmethod
    def format_dealloc_msg(
        stack_trace: str | None, *, is_grad_output: bool = False
    ) -> str:
        # [FUNCTION DOC] Static method: builds the error message stored on invalidated tensors,
        # including the stack trace of where the output was created (or a placeholder) and advice; a
        # special gradient accumulation message when the output is a gradient. Returns the string.
        stack_trace = (
            stack_trace.strip() if stack_trace else "[Could not find stack trace]"
        )
        if is_grad_output:
            return (
                "Error: accessing gradient tensor output of CUDAGraphs that has been overwritten "
                f"by a subsequent run. Stack trace: {stack_trace}. "
                "This can happen with torch.compile(mode='reduce-overhead') and gradient "
                "accumulation when a .grad tensor is allocated during CUDAGraph capture. "
                "If you need gradient accumulation, allocate stable .grad buffers outside "
                "CUDAGraph capture before the compiled backward runs, for example by running "
                "an eager warmup iteration or by preallocating zeroed .grad tensors. "
                "If you do not need gradient accumulation, set .grad to None before each backward."
            )
        return (
            "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. "
            f"Stack trace: {stack_trace}. "
            "To prevent overwriting, call torch.compiler.cudagraph_mark_step_begin() "
            "before each model invocation. If a tensor is a user-visible output "
            "that must remain live across generations, manually clone the tensor "
            "outside torch.compile(), or set torch._inductor.config.triton."
            "cudagraph_trees_generation_cloning = 'user_visible' to clone all "
            "user-visible tensors that persist across generations."
        )

    def dealloc_current_path_weakrefs(self) -> None:
        # [FUNCTION DOC] Invalidates output tensors of the current path before a new generation
        # overwrites them: sets an access error message on live tensors (with creation stack trace),
        # and, unless alias invalidation is disabled, collects stack traces per data pointer, frees
        # deleters of live storages (once per data pointer) and sets the access error on their data.
        # Raises an assertion error if there is no current node or stack traces are missing.
        if self.current_node is None:
            raise AssertionError("expected current_node to not be None")
        # TODO: we could also allow the these weak refs to continue to be allocated,
        # but that adds some complications.

        stor_dealloc_info: dict[int, tuple[str | None, bool]] = {}
        for node in self.current_node._path_from_root:
            if node.stack_traces is None:
                raise AssertionError("expected node.stack_traces to not be None")
            # tensor_weakrefs and stack_traces are paired below with zip(),
            # matching the defensive handling for outputs_weakrefs and
            # stack_traces.
            if len(node.tensor_weakrefs) != len(node.stack_traces):
                log.warning(
                    "tensor_weakrefs length (%d) != stack_traces length (%d)",
                    len(node.tensor_weakrefs),
                    len(node.stack_traces),
                )
            is_grad_output = (
                self.id_to_mode[node.wrapped_function.id] == CompilationMode.BACKWARD
            )
            for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
                ten = None if t is None else t()
                if ten is None:
                    continue

                torch._C._set_storage_access_error_msg(
                    ten,
                    self.format_dealloc_msg(stack_trace, is_grad_output=is_grad_output),
                )

            # we would to enable the following assertion, but an internal model failed with a command
            # that does not repro. len(node.outputs_weakrefs) == len(node.stack_traces)
            # so, pessimistically assume that they might differ by doing the debug info
            # loop separately from the dealloc loop
            if self.disable_invalidate_aliases:
                continue

            for storage_ref, stack_trace in zip(
                node.outputs_weakrefs, node.stack_traces
            ):
                if not storage_ref:
                    continue

                stor_dealloc_info[storage_ref.data_ptr()] = (
                    stack_trace,
                    is_grad_output,
                )

        deleted = OrderedSet[Any]()
        for storage_ref in self.current_node.path_live_weakrefs():
            _storage_deref = storage_ref()
            if _storage_deref and storage_ref.data_ptr() not in deleted:
                deleted.add(storage_ref.data_ptr())

                stack_trace, is_grad_output = stor_dealloc_info.get(
                    storage_ref.data_ptr(), (None, False)
                )
                msg = self.format_dealloc_msg(
                    stack_trace, is_grad_output=is_grad_output
                )
                torch._C._free_And_Remove_DeleterFn(_storage_deref)

                if self.disable_invalidate_aliases:
                    continue

                torch._C._set_storage_data_ptr_access_error_msg(_storage_deref, msg)

    def clear_current_path_state_and_set_to_none(
        self,
        *,
        clone_live_user_visible_outputs: bool = False,
        dealloc_current_path_weakrefs: bool = False,
    ) -> None:
        # [FUNCTION DOC] Ends the current path: requires a graph node as current node (assertion
        # error otherwise); optionally clones user visible outputs and deallocates path weakrefs,
        # calls the node path state clearing and sets current node to None.
        if not isinstance(self.current_node, CUDAGraphNode):
            raise AssertionError(
                f"expected current_node to be a CUDAGraphNode, got {type(self.current_node)}"
            )
        if clone_live_user_visible_outputs:
            self._clone_current_path_live_user_visible_outputs()
        if dealloc_current_path_weakrefs:
            self.dealloc_current_path_weakrefs()
        self.current_node.clear_path_state()
        self.current_node = None

    def apply_checkpoint_execution_state_in_allocator(self) -> None:
        # [FUNCTION DOC] Restores the caching allocator bookkeeping to the state checkpointed at the
        # current node so new recordings or warmups respect live pool tensors: counts and logs the
        # checkpoint, removes cached tensors on the path, sets the pool state with live storages,
        # then frees blocks of outputs that died since recording (deduplicated). Optionally
        # validates the pool and checks standard deleters. Raises assertion errors if the current
        # node is not a graph node or state is missing.
        """
        Checkpoint the current execution state in the caching allocator so that
        additional cudagraph recordings can be made respecting existent live storages.
        """
        if not isinstance(self.current_node, CUDAGraphNode):
            raise AssertionError(
                f"expected current_node to be a CUDAGraphNode, got {type(self.current_node)}"
            )
        self.debug_checkpointing_counter += 1
        log.debug(
            "Checkpointing cuda caching allocator state. Number of checkpoints %d",
            self.debug_checkpointing_counter,
        )

        state = self.current_node.checkpointed_caching_state
        device = self.current_node.device
        if not (state is not None and device is not None):
            raise AssertionError("expected state and device to not be None")

        # currently we deallocate on instead of allowing stale recordings
        stale_storages: list[int] = []

        # remove cached tensors, otherwise they would prevent memory from being
        # reclaimed in subsequent recordings
        self.current_node.remove_path_cached_tensors()
        live_storages_wrappers = list(self.current_node.path_live_weakrefs())

        # path_live_weakrefs guarantees that t() will not be None
        live_storages_weak_refs: list[int] = [t() for t in live_storages_wrappers]  # type: ignore[misc]
        ptrs_to_deallocate = self.current_node.data_ptrs_dead_since_invocation()
        torch._C._cuda_setCheckpointPoolState(
            device,
            # pyrefly: ignore [bad-argument-type]
            state,
            stale_storages,
            live_storages_weak_refs,
        )

        # NB: deduplicate aliased outputs
        for ptr in OrderedSet(ptrs_to_deallocate):
            torch._C._cuda_cudaCachingAllocator_raw_delete(ptr)

        # Now the live blocks should be exactly equal to the live storages in private pool
        if config.triton.slow_path_cudagraph_asserts:
            check_memory_pool(
                self.device_index, self.cuda_graphs_thread_pool, live_storages_wrappers
            )
            for wrapper in live_storages_wrappers:
                storage_ptr = wrapper()
                if storage_ptr is None:
                    raise AssertionError("expected storage_ptr to not be None")
                if not torch._C._has_Standard_Deleter(storage_ptr):
                    raise AssertionError(
                        "expected storage_ptr to have standard deleter"
                    )
                if wrapper.data_ptr() in ptrs_to_deallocate:
                    raise AssertionError(
                        "expected wrapper.data_ptr() not in ptrs_to_deallocate"
                    )

    def live_cudagraph_pool_storages_in_curr_execution(
        self,
    ) -> list[StorageWeakRefPointer]:
        # [FUNCTION DOC] Returns the storage pointers of currently live outputs on the current path,
        # or an empty list with no current node.
        if self.current_node is None:
            return []
        # explicitly ignoring previous recorded outputs from past path
        # path_live_weakrefs() guarantees that t() will not be None
        return [t() for t in self.current_node.path_live_weakrefs()]  # type: ignore[misc]
