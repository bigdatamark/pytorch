# mypy: allow-untyped-defs
# [MODULE DOC] Staging of state dicts to CPU for async checkpointing. StateDictStager performs a custom deep copy that
# offloads tensors into cached CPU storages, optionally in shared memory (multi-process) and pinned memory (faster
# transfers), reusing cached storages across calls and preserving storage sharing between tensors.
# No import-time side effects.
import types
import warnings
import weakref
from copyreg import dispatch_table
from typing import Any

import torch
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.storage import UntypedStorage
from torch.utils.weak import WeakIdKeyDictionary


class StateDictStager:
    # [CLASS DOC] Deep-copies state dicts to CPU. Attributes: pin_memory (forced off with a warning when CUDA is unavailable),
    # share_memory, pin_memory_min_bytes (skip pinning for smaller storages), a weak-keyed cache from original storage to staged
    # CPU storage, and a dispatch table of per-type copy functions built in the constructor. Call close to clear caches and break
    # the reference cycle created by closures capturing self.
    """
    A class for optimizing storage objects during staging for async checkpointing.

    StateDictStager stages the state_dict to CPU DRAM while applying optimizations
    like memory sharing and pinning to improve performance. It caches storage objects
    to avoid redundant copies and can be configured to automatically share memory
    (for multi-process usage) and pin memory (for faster CPU-GPU transfers).

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        pin_memory_min_bytes (int): Minimum tensor size in bytes to pin memory (default: 5)
        _cached_storage_mapping (WeakIdKeyDictionary): Maps storage objects to optimized CPU storages using weak references
    """

    def __init__(
        self,
        pin_memory: bool = False,
        share_memory: bool = False,
        pin_memory_min_bytes: int = 5,
    ):
        # [FUNCTION DOC] Sets the flags (disabling pinning with a warning if CUDA is not available), creates the storage cache, and
        # builds the type-to-copier dispatch table with atomic, list, tuple, dict and bound method copiers defined as closures.
        # KNOWN ISSUE: the warning message string lacks a space between the concatenated sentences.
        if pin_memory and not torch.cuda.is_available():
            warnings.warn(
                "Ignoring pin_memory flag for checkpoint staging as pinning memory"
                "requires CUDA, but CUDA is not available. ",
                stacklevel=2,
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory
        self.share_memory = share_memory
        # Mapping from original storage objects to CPU storages using weak references
        self._cached_storage_mapping = WeakIdKeyDictionary()
        self.pin_memory_min_bytes = pin_memory_min_bytes

        def _deepcopy_atomic(x, _):
        # [FUNCTION DOC] HELPER: Copier for immutable types; returns the object itself.
            return x

        def _deepcopy_list(x, memo, non_blocking=False):
        # [FUNCTION DOC] HELPER: Copier for lists; registers the new list in the memo and fills it with offloaded deep copies of the elements.
            y: list = []
            memo[id(x)] = y
            append = y.append
            for a in x:
                append(
                    self.deepcopy_with_tensor_offload(
                        a, memo, non_blocking=non_blocking
                    )
                )
            return y

        def _deepcopy_tuple(x, memo, non_blocking=False):
        # [FUNCTION DOC] HELPER: Copier for tuples; copies the elements, returns a memoized result if one exists, returns the original
        # tuple if no element changed, otherwise a new tuple.
            y = [
                self.deepcopy_with_tensor_offload(a, memo, non_blocking=non_blocking)
                for a in x
            ]
            # We're not going to put the tuple in the memo, but it's still important we
            # check for it, in case the tuple contains recursive mutable structures.
            try:
                return memo[id(x)]
            except KeyError:
                pass

            # Check if any elements changed during deepcopy
            for k, j in zip(x, y):
                if k is not j:
                    # At least one element changed, create new tuple
                    return tuple(y)

            # No elements changed, return original tuple
            return x

        def _deepcopy_dict(x, memo, non_blocking=False):
        # [FUNCTION DOC] HELPER: Copier for dicts; registers the new dict in the memo and fills it with deep copies of keys and values.
            y: dict = {}
            memo[id(x)] = y
            for key, value in x.items():
                y[
                    self.deepcopy_with_tensor_offload(
                        key, memo, non_blocking=non_blocking
                    )
                ] = self.deepcopy_with_tensor_offload(
                    value, memo, non_blocking=non_blocking
                )
            return y

        def _deepcopy_method(x, memo, non_blocking=False):  # Copy instance methods
        # [FUNCTION DOC] HELPER: Copier for bound methods; rebuilds the method around a deep copy of its instance.
            return type(x)(
                x.__func__,
                self.deepcopy_with_tensor_offload(
                    x.__self__, memo, non_blocking=non_blocking
                ),
            )

        d: dict[Any, Any] = {}
        self._deepcopy_dispatch = d
        d[type(None)] = _deepcopy_atomic
        d[int] = _deepcopy_atomic
        d[float] = _deepcopy_atomic
        d[bool] = _deepcopy_atomic
        d[complex] = _deepcopy_atomic
        d[bytes] = _deepcopy_atomic
        d[str] = _deepcopy_atomic
        d[types.CodeType] = _deepcopy_atomic
        d[type] = _deepcopy_atomic
        d[range] = _deepcopy_atomic
        d[types.BuiltinFunctionType] = _deepcopy_atomic
        d[types.FunctionType] = _deepcopy_atomic
        d[weakref.ref] = _deepcopy_atomic
        d[property] = _deepcopy_atomic
        d[types.MethodType] = _deepcopy_method
        d[dict] = _deepcopy_dict
        d[tuple] = _deepcopy_tuple
        d[list] = _deepcopy_list

    def _stage_untyped_storage(
        self,
        storage: UntypedStorage,
        non_blocking: bool = False,
    ):
        # [FUNCTION DOC] HELPER: Returns a CPU storage holding the data of the given storage. If the original was staged before, copies
        # new data into the cached CPU storage (optionally non-blocking) and returns it. Otherwise allocates a CPU storage (shared
        # if share_memory), pins it when pinning is on and it is at least pin_memory_min_bytes (unpinned by a finalizer that does not
        # run at exit), copies the data, caches it and returns it. Raises AssertionError if the cached storage size differs.
        # KNOWN ISSUE: docstring refers to a StagingStateDict class that does not exist.
        """
        Called from the hooked storage_deepcopy function in torch.Tensor.__deepcopy__.

        This method handles the storage optimization logic for the StagingStateDict class.
        It checks if the storage has already been cached, and if so, reuses it.
        Otherwise, it creates a new CPU storage and applies memory optimizations.

        Args:
            storage: The storage to optimize

        Returns:
            The optimized storage
        """
        # Check if we've already cached this storage
        if storage in self._cached_storage_mapping:
            cached_storage = self._cached_storage_mapping[storage]
            if cached_storage.size() != storage.size():
                raise AssertionError(
                    "For async checkpointing,  We cache storages in DRAM and reuse them."
                    "Cached storage size does not match original storage size."
                    "This should never happen as we track the original storage weakref "
                    "and clean up the cache storage. Please report this to PyTorch Distributed Checkpointing."
                )
            # Reuse cached storage but update with new data
            cached_storage.copy_(storage, non_blocking=non_blocking)
            return cached_storage

        # Create new CPU storage
        if self.share_memory:
            new_storage = type(storage)._new_shared(storage.size(), device="cpu")
        else:
            new_storage = type(storage)(storage.size(), device="cpu")

        # Skip pinning for tensors below the minimum size threshold
        # Small tensors (e.g., optimizer step counters, scalars) have negligible
        # transfer time improvement from pinning, but pinning overhead is significant
        if self.pin_memory and new_storage.nbytes() >= self.pin_memory_min_bytes:
            pin_memory_utils.pin_memory(new_storage.data_ptr(), new_storage.nbytes())
            # Set up a weak reference to unpin when cpu storage is garbage collected
            f = weakref.finalize(
                new_storage, pin_memory_utils.unpin_memory, new_storage.data_ptr()
            )
            # This makes sure that the finalizer is not called after
            # cuda context is destroyed.
            f.atexit = False

        new_storage.copy_(storage, non_blocking=non_blocking)

        # Cache the storage - WeakIdKeyDictionary will automatically clean up when storage is garbage collected
        self._cached_storage_mapping[storage] = new_storage
        return new_storage

    @torch.no_grad()
    def stage(
        self,
        state_dict: Any,
        non_blocking: bool = False,
    ) -> Any:
        # [FUNCTION DOC] Stages state_dict under no-grad: deep copies it with tensors offloaded to CPU storages (see
        # deepcopy_with_tensor_offload) and returns the copy. non_blocking requests asynchronous device-to-host copies, so the
        # caller must synchronize before using the result.
        return self.deepcopy_with_tensor_offload(state_dict, None, [], non_blocking)

    def _offload_tensor(self, x, memo, non_blocking=False):
        # [FUNCTION DOC] HELPER: Copies one tensor (or subclass) to CPU. Creates an empty CPU tensor, registers it in the memo, stages
        # its untyped storage once per call (a memo keyed by storage id keeps tensors sharing a storage sharing the copy), sets the
        # copy's storage, offset, size and stride, and deep copies instance dict attributes and slots. Returns the new tensor.
        """
        Deep copy a PyTorch tensor with optimized storage handling.

        This method creates a CPU copy of a tensor while applying memory optimizations
        like sharing and pinning based on the StateDictStager configuration.

        Args:
            x: The tensor to copy
            memo: Memo dictionary for tracking already copied objects
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A CPU copy of the tensor with optimized storage
        """
        # if data_ptr is not 0, we allocate a new storage below. so we can skip
        # memory allocation by using [] for size.
        y = x.new_empty([] if x.data_ptr() != 0 else x.size(), device="cpu")

        # Store in memo dict early to handle recursive references
        d = id(x)
        memo[d] = y

        if type(x) is torch.Tensor or x.data_ptr() != 0:
            # Get the untyped storage
            untyped_storage = x.untyped_storage()
            storage_id = id(untyped_storage)

            # Check if this storage has already been staged in this deepcopy operation
            # This handles the case where different tensors share the same storage
            # (e.g., FSDP state_dict where norm.weight and norm_weight reference same storage)
            # PyTorch caches untyped_storage() calls, so same storage -> same id
            if storage_id in memo:
                copied_storage = memo[storage_id]
            else:
                # Storage not seen before in this operation, stage it
                copied_storage = self._stage_untyped_storage(
                    untyped_storage, non_blocking=non_blocking
                )
                # Add to memo to avoid re-staging if we see this storage again
                memo[storage_id] = copied_storage

            # Set the tensor data using the staged storage
            y.set_(copied_storage, x.storage_offset(), x.size(), x.stride())

        # Copy any attributes the tensor might have
        if hasattr(x, "__dict__"):
            for attr_name, attr_value in x.__dict__.items():
                setattr(
                    y,
                    attr_name,
                    self.deepcopy_with_tensor_offload(
                        attr_value, memo, non_blocking=non_blocking
                    ),
                )

        if hasattr(x, "__slots__"):
            for slot in x.__slots__:
                if hasattr(x, slot):
                    setattr(
                        y,
                        slot,
                        self.deepcopy_with_tensor_offload(
                            getattr(x, slot), memo, non_blocking=non_blocking
                        ),
                    )

        return y

    def close(self):
        # [FUNCTION DOC] Clears the storage cache (pinned memory is released by finalizers when storages are collected) and the dispatch
        # table to break reference cycles. The stager cannot copy standard types after closing.
        """
        Clean up all cached storages and release associated resources.

        This method clears the internal storage cache, allowing garbage collection
        of cached CPU storages. Any pinned memory associated with cached storages
        will be automatically unpinned through weak reference finalizers.

        It also clears the _deepcopy_dispatch dict to break the reference cycle
        created by closures that capture self. Without this, it may
        cause memory leaks.
        """
        self._cached_storage_mapping.clear()
        self._deepcopy_dispatch.clear()

    @torch.no_grad()
    def deepcopy_with_tensor_offload(self, x, memo=None, _nil=[], non_blocking=False):  # noqa: B006
        # [FUNCTION DOC] Deep copy variant of the standard library algorithm that sends tensors to the offload path. Returns a memoized
        # copy when available; tensors go to the tensor offloader; types in the dispatch table use their copier; type objects are atomic;
        # objects defining a deep copy method use it; others are rebuilt from their pickle reduction (reduce_ex with protocol 4, or reduce),
        # via the reconstruct helper. Raises RuntimeError for uncopyable objects or an unexpected reduction tuple length.
        # Memoizes results that differ from the input and keeps inputs alive. Runs under no-grad.
        # KNOWN ISSUE: the mutable default for the sentinel argument is intentional but stage passes a distinct list; callers need not provide it.
        """Deep copy operation on arbitrary Python objects with special handling for PyTorch tensors.

        This implementation extends the standard deepcopy functionality to handle PyTorch tensors
        and their storages in a way that optimizes memory usage and performance, similar to the
        stage method. It applies memory sharing and pinning optimizations based on the StateDictStager
        configuration.

        Args:
            x: The object to deep copy
            memo: Memo dictionary for tracking already copied objects
            _nil: Sentinel value for memo dictionary
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A deep copy of the input object with optimized tensor storage handling
        """
        if memo is None:
            memo = {}

        d = id(x)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        cls = type(x)

        # tensors and subclasses of tensors are handled separately
        if isinstance(x, torch.Tensor):
            y = self._offload_tensor(x, memo, non_blocking=non_blocking)
        else:
            # Use the dispatch table for standard types
            copier = self._deepcopy_dispatch.get(cls)
            if copier is not None:
                # Check if this is an atomic copier (only accepts x and memo)
                if copier.__name__ == "_deepcopy_atomic":
                    y = copier(x, memo)
                else:
                    y = copier(x, memo, non_blocking=non_blocking)
            else:
                if issubclass(cls, type):
                    # type copier is also atomic
                    y = self._deepcopy_dispatch[type](x, memo)
                else:
                    copier = getattr(x, "__deepcopy__", None)
                    if copier is not None:
                        y = copier(memo)
                    else:
                        reductor = dispatch_table.get(cls)
                        if reductor:
                            rv = reductor(x)
                        else:
                            reductor = getattr(x, "__reduce_ex__", None)
                            if reductor is not None:
                                rv = reductor(4)
                            else:
                                reductor = getattr(x, "__reduce__", None)
                                if reductor:
                                    rv = reductor()
                                else:
                                    raise RuntimeError(
                                        f"un(deep)copyable object of type {cls}"
                                    )
                        if isinstance(rv, str):
                            y = x
                        else:
                            # Unpack rv tuple elements (up to 5 from pickle protocol)
                            # and explicitly pass non_blocking as keyword arg
                            if len(rv) == 2:
                                func, args = rv
                                y = self._reconstruct(
                                    x, memo, func, args, non_blocking=non_blocking
                                )
                            elif len(rv) == 3:
                                func, args, state = rv
                                y = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    non_blocking=non_blocking,
                                )
                            elif len(rv) == 4:
                                func, args, state, listiter = rv
                                y = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    listiter,
                                    non_blocking=non_blocking,
                                )
                            elif len(rv) == 5:
                                func, args, state, listiter, dictiter = rv
                                y = self._reconstruct(
                                    x,
                                    memo,
                                    func,
                                    args,
                                    state,
                                    listiter,
                                    dictiter,
                                    non_blocking=non_blocking,
                                )
                            else:
                                raise RuntimeError(
                                    f"Unexpected pickle protocol return value length: {len(rv)}"
                                )

        # If it is its own copy, don't memoize.
        if y is not x:
            memo[d] = y
            self._keep_alive(x, memo)  # Make sure x lives at least as long as d
        return y

    def _keep_alive(self, x, memo):
        # [FUNCTION DOC] HELPER: Keeps x alive by appending it to a list stored in the memo under the memo's own id, so object ids stay unique.
        """Keeps a reference to the object x in the memo.

        Because we remember objects by their id, we have
        to assure that possibly temporary objects are kept
        alive by referencing them.
        We store a reference at the id of the memo, which should
        normally not be used unless someone tries to deepcopy
        the memo itself...
        """
        try:
            memo[id(memo)].append(x)
        except KeyError:
            # aha, this is the first one :-)
            memo[id(memo)] = [x]

    def _reconstruct(
        self,
        x,
        memo,
        func,
        args,
        state=None,
        listiter=None,
        dictiter=None,
        non_blocking=False,
    ):
        # [FUNCTION DOC] HELPER: Rebuilds an object from a pickle reduction tuple (callable, args, optional state, list iterator, dict iterator),
        # deep copying args, state and items via the offload copier when a memo is provided. Applies state through the set-state method or by
        # updating the instance dict and slots, appends list items and sets dict items. Returns the new object.
        deep = memo is not None
        if deep and args:
            args = tuple(
                self.deepcopy_with_tensor_offload(arg, memo, non_blocking=non_blocking)
                for arg in args
            )
        y = func(*args)
        if deep:
            memo[id(x)] = y

        if state is not None:
            if deep:
                state = self.deepcopy_with_tensor_offload(
                    state, memo, non_blocking=non_blocking
                )
            if hasattr(y, "__setstate__"):
                y.__setstate__(state)
            else:
                if isinstance(state, tuple) and len(state) == 2:
                    state, slotstate = state
                else:
                    slotstate = None
                if state is not None:
                    y.__dict__.update(state)
                if slotstate is not None:
                    for key, value in slotstate.items():
                        setattr(y, key, value)

        if listiter is not None:
            if deep:
                for item in listiter:
                    item = self.deepcopy_with_tensor_offload(
                        item, memo, non_blocking=non_blocking
                    )
                    y.append(item)
            else:
                for item in listiter:
                    y.append(item)
        if dictiter is not None:
            if deep:
                for key, value in dictiter:
                    key = self.deepcopy_with_tensor_offload(
                        key, memo, non_blocking=non_blocking
                    )
                    value = self.deepcopy_with_tensor_offload(
                        value, memo, non_blocking=non_blocking
                    )
                    y[key] = value
            else:
                for key, value in dictiter:
                    y[key] = value
        return y
