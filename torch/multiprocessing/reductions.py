# mypy: allow-untyped-defs
# [MODULE DOC] Pickling support for sharing tensors, storages and CUDA events between processes.
# Defines a weak-reference wrapper for storages, a process-wide cache of already-opened shared storages,
# and reduce/rebuild function pairs (CUDA IPC, meta, CPU shared fd/filename, nested, sparse, typed storages).
# Import-time side effect: creates the global shared_cache. init_reductions registers the reducers with
# the multiprocessing ForkingPickler registry.
import multiprocessing
import os
import threading
from multiprocessing import reduction
from multiprocessing.util import register_after_fork

import torch


try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


class StorageWeakRef:
    # [CLASS DOC] Holds a weak (non-owning) reference to a storage, stored as an integer pointer in cdata.
    # Uses slots. Frees the weak reference when destroyed; equality and hashing are based on cdata.
    # Methods: from_weakref builds one from a raw cdata, expired reports whether the storage is gone.
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer.
    """

    __slots__ = ["cdata", "_free_weak_ref"]

    def __init__(self, storage):
        # [FUNCTION DOC] Takes a storage, records its weak pointer in cdata, and keeps a direct reference to the
        # weak-ref freeing function so destruction still works during interpreter shutdown.
        self.cdata = storage._weak_ref()
        # Save a direct reference to _free_weak_ref because the `torch` module
        # might be cleared during Python shutdown before this module is cleared.
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    @classmethod
    def from_weakref(cls, cdata):
        # [FUNCTION DOC] Class method. Builds an instance from an existing weak pointer without calling the normal
        # constructor. Takes cdata, returns the new instance.
        instance = cls.__new__(cls)
        instance.cdata = cdata
        instance._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]
        return instance

    def expired(self):
        # [FUNCTION DOC] Returns True if the referenced storage has already been freed, else False.
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        # [FUNCTION DOC] Releases the weak reference held in cdata.
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        # [FUNCTION DOC] Hash is the integer weak pointer itself.
        return self.cdata

    def __eq__(self, other):
        # [FUNCTION DOC] True if the same object or if both hold the same cdata.
        # KNOWN ISSUE: no type check on other, so comparing with an object lacking cdata raises an attribute error.
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata


class SharedCache(dict):
    # [CLASS DOC] Dictionary mapping sharing handles (fd identity, filename handle, or CUDA handle keys)
    # to StorageWeakRef objects, used to reuse storages already opened in this process.
    # State: limit (size above which dead entries are purged) and lock (guards get and set).
    # The lock is replaced after fork to avoid inheriting a held lock.
    """Dictionary from multiprocessing handles to StorageWeakRef."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Sets the purge limit to 128, creates the lock, and registers an after-fork hook that recreates it.
        # free_dead_references() is called if the len exceeds the current
        # limit. The limit scales with the number of remaining live objects.
        self.limit = 128
        # `fork` inherits lock state, so in case we fork when the lock is held,
        # we register a function to reset the lock to a new object to avoid
        # possible deadlocks, following python multiprocessing library design.
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        # [FUNCTION DOC] HELPER: Replaces the lock with a fresh one; run in a forked child.
        self.lock = threading.Lock()

    def get(self, key):  # type: ignore[override]
        # [FUNCTION DOC] Thread-safe lookup of key; returns the stored value or None.
        # KNOWN ISSUE: the override takes no default argument, unlike the base dict method.
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        # [FUNCTION DOC] Thread-safe store of key/value. If the cache now exceeds limit, purges dead references.
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            if len(self) > self.limit:
                self.free_dead_references()

    def free_dead_references(self):
        # [FUNCTION DOC] Deletes entries whose storage has expired, then sets limit to the larger of 128 and twice
        # the number of live entries. Called with the lock held from __setitem__.
        live = 0
        for key, storage_ref in list(self.items()):
            if storage_ref.expired():
                del self[key]
            else:
                live += 1
        self.limit = max(128, live * 2)


# mapping from handles to StorageWeakRef objects
shared_cache = SharedCache()


def rebuild_event(device, handle):
    # [FUNCTION DOC] Rebuilds a CUDA event in the receiving process from its device and IPC handle; returns the event.
    return torch.cuda.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    # [FUNCTION DOC] Pickle reducer for a CUDA event. Returns the rebuild function and its args (device, IPC handle).
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def rebuild_tensor(cls, storage, metadata):
    # [FUNCTION DOC] Rebuilds a CPU tensor over the given storage using metadata (storage offset, size, stride, requires_grad).
    # Parameters are rebuilt through the Parameter constructor so requires_grad is set at creation;
    # other classes get requires_grad assigned afterwards. Returns the tensor.
    storage_offset, size, stride, requires_grad = metadata
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        # we have to pass requires_grad into constructor, rather than set it as an
        # attribute later, because it's an important check for Integer Tensors to
        # have requires_grad=False (or else they raise an error)
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t


def rebuild_meta_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    dtype,
    storage_size_bytes,
    requires_grad,
):
    # [FUNCTION DOC] Rebuilds a tensor of the meta device: creates an empty meta storage of the given byte size,
    # wraps it as typed storage with dtype, builds the strided tensor, applies requires_grad
    # (via the constructor for Parameter). Returns the tensor.
    untyped_storage = torch.UntypedStorage(storage_size_bytes, device="meta")

    typed_storage = torch.TypedStorage(
        wrap_storage=untyped_storage, dtype=dtype, _internal=True
    )

    t = torch._utils._rebuild_tensor(
        typed_storage,
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # [FUNCTION DOC] Receiver-side rebuild of a tensor from CUDA IPC data. If the handle is null or the size is zero an
    # empty storage is made. Otherwise looks the storage up in shared_cache keyed by handle and byte offset;
    # on a miss opens the shared CUDA storage (initializing CUDA) and caches a weak ref; on a hit
    # releases the sender's extra IPC ref counter. Then builds the tensor on the storage and applies requires_grad.
    # Returns the tensor. Side effects: updates shared_cache, may initialize CUDA.
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(
            storage_cls, (storage_handle, storage_offset_bytes)
        )
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(
                storage
            )
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter(
                ref_counter_handle, ref_counter_offset, device=storage_device
            )

    _storage = (
        storage
        if isinstance(storage, torch.UntypedStorage)
        else storage._untyped_storage
    )

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def reduce_tensor(tensor):
    # [FUNCTION DOC] Pickle reducer for any tensor. Raises a runtime error for non-leaf tensors requiring grad; warns
    # if hooks are present. Dispatches nested (non-subclass) and sparse tensors to their own reducers.
    # For CUDA storage it shares via IPC, records the storage in shared_cache and returns the CUDA rebuild
    # function with full metadata; for meta storage returns the meta rebuild; otherwise returns the CPU
    # rebuild with the (shared) storage and metadata. Returns a pair of rebuild function and args.
    # KNOWN ISSUE: the CUDA branch caches under the bare handle while rebuild_cuda_tensor looks up (handle, offset).
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    torch.utils.hooks.warn_if_has_hooks(tensor)

    # Note [CUDA IPC and the caching allocator]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # When you send a CUDA tensor over IPC, you might expect that you will
    # get out the same storage from the other end.  However, the CUDA caching
    # allocator makes it difficult to preserve this invariant.  Consider
    # the following situation: a tensor of size 0x100 points to offset 0x20 of
    # a storage at 0xA100 of size 0x100.  (For simplicity, all of these
    # sizes are given in bytes).  HOWEVER, with the caching allocator, this storage
    # might be part of a larger cudaMalloc allocation 0xA000 of size 0x4000.
    #
    # When we want to send this CUDA tensor over IPC, we must send the
    # *entire* cudaMalloc allocation, i.e., the 0xA000 region, not just
    # the storage 0xA100 (because that is what CUDA supports).  So, on the
    # other end, there simply isn't any way to say, "Wait, you gave me
    # a bigger region (0xA000) than the one I wanted (0xA100)".
    #
    # OK, so if you sent the cudaMalloc allocation, can you just wrap that up as
    # one storage itself? No, because this cudaMalloc allocation might contain
    # storages of mixed types: float, bytes, double... If you make the entire
    # allocation a single storage of a type A, we'll hit an error when constructing
    # a tensor of type B on the storage.
    #
    # cudaIpcMemHandle is an identifier to access the sender cudaMalloc allocation on the
    # receiver side. However, cudaIpcMemHandles from each device in a given process may
    # only be opened by one context per device per other process.
    # If we open and close a memory handle multiples times in a process, CUDA is allowed
    # to give it a different address; similarly, once we close the memory, we're not
    # allowed to access it(and the storage/tensor built on top of it), even if it is
    # still live in the original process. As we cannot make a cudaMalloc allocation
    # to a single storage in one go, this requires us to cache the device pointer for
    # each cudaIpcMemHandle on C++ side to reconstruct types of storages, while keep
    # the old ones alives.
    # See [https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html]
    #
    # This is fine, because all we need to do is to save our position in the allocation,
    # and reconstruct storage and tensor from it.
    # 0xA000 ->  -------CUDA Allocation------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA100 ->  --------storage1 begin------
    #           |                            |
    # 0xA120 ->  --------tensor1 begin ------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA160 ->  --------tensor1 end---------
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA200 ->  --------storage1 end--------
    #           |                            |
    # 0xE000 ->  --------CUDA allocation-----
    #
    # To send tensor1, the following info are required from sender to receiver for
    # storage reconstruction.
    #   1. cudaIpcMemHandle of 0xA000(which can be mapped to a basePtr in receiver process).
    #      basePtr may not be exactly 0xA000 since it's a different process.
    #   2. offset(0xA100) of storage1 in the CUDA allocation.
    #   3. size of storage1(0x100).
    #
    # On receiver side:
    #   1. Get the devPtr of the MemHandle to access the memory, reconstruct a storage
    #      of the same type using (basePtr, offset, size).
    #   2. we can reconstruct the tensor on top of the reconstructed storage
    #   Tensor(size=0x040, offset=0x020, storage=Storage(data=basePtr+0xA100, size=0x0100))
    #
    # This strategy has a few implications:
    #
    # 1. When we serialize a CUDA tensor for IPC, we cannot do it all in one
    #    go (non-compositionally), and this requires to have a global map
    #    memHandle -> devPtr for each process.
    #
    # 2. We MUST NOT let the new IPC tensor be resizable.  Originally, a resize
    #    of the storage beyond 0x100 would merely have caused us to do a
    #    reallocation.  You don't really want to do this, but if you did,
    #    all that would happen is that you would lose IPC sharing.  But if
    #    you do this in the new world, we will happily let you write out of
    #    bounds of your "allocation", clobbering unrelated data in the cached
    #    allocator block.  BAD!
    #
    # By the way, in old versions of PyTorch, we supported this situation
    # natively using a "storage view", which permitted multiple storages to be
    # views on each other.  But this was the *only* use of storage views, so we
    # eliminated it so that we could just use tensor views to implement the same
    # thing.
    #

    # TODO: Handle distinguishing between subclass and non-subclass versions of NT better
    # https://github.com/pytorch/pytorch/issues/110543
    from torch.nested._internal.nested_tensor import NestedTensor

    if tensor.is_nested and not isinstance(tensor, NestedTensor):
        return reduce_nested_tensor(tensor)

    if tensor.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        return reduce_sparse_tensor(tensor)

    storage = tensor._typed_storage()

    if storage._untyped_storage.device.type == "cuda":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which CUDA allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )
    elif storage._untyped_storage.device.type == "meta":
        return (
            rebuild_meta_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.dtype,
                tensor.untyped_storage().size(),
                tensor.requires_grad,
            ),
        )

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))


def rebuild_nested_tensor(
    rebuild_buffer_func,
    rebuild_buffer_args,
    rebuild_sizes_func,
    rebuild_sizes_args,
    rebuild_strides_func,
    rebuild_strides_args,
    rebuild_offsets_func,
    rebuild_offsets_args,
):
    # [FUNCTION DOC] Rebuilds a strided nested tensor from four sub-tensors (buffer, sizes, strides, offsets), each given
    # as a rebuild function plus args, then makes the nested view by copying the buffer. Returns it.
    buffer = rebuild_buffer_func(*rebuild_buffer_args)
    sizes = rebuild_sizes_func(*rebuild_sizes_args)
    strides = rebuild_strides_func(*rebuild_strides_args)
    offsets = rebuild_offsets_func(*rebuild_offsets_args)
    return torch._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)


def reduce_nested_tensor(nt):
    # [FUNCTION DOC] Pickle reducer for a strided nested tensor: reduces its values buffer, size, stride and
    # storage-offset tensors separately. Returns the nested rebuild function and the combined args.
    rebuild_buffer_func, rebuild_buffer_args = reduce_tensor(nt.values())
    rebuild_sizes_func, rebuild_sizes_args = reduce_tensor(nt._nested_tensor_size())
    rebuild_strides_func, rebuild_strides_args = reduce_tensor(
        nt._nested_tensor_strides()
    )
    rebuild_offsets_func, rebuild_offsets_args = reduce_tensor(
        nt._nested_tensor_storage_offsets()
    )

    return (
        rebuild_nested_tensor,
        (
            rebuild_buffer_func,
            rebuild_buffer_args,
            rebuild_sizes_func,
            rebuild_sizes_args,
            rebuild_strides_func,
            rebuild_strides_args,
            rebuild_offsets_func,
            rebuild_offsets_args,
        ),
    )


def rebuild_sparse_coo_tensor(
    rebuild_indices_func,
    rebuild_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    is_coalesced,
):
    # [FUNCTION DOC] Rebuilds a sparse COO tensor from reduced indices and values plus shape and coalesced flag. Returns it.
    indices = rebuild_indices_func(*rebuild_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_coo_tensor(indices, values, shape, is_coalesced=is_coalesced)


def rebuild_sparse_compressed_tensor(
    rebuild_compressed_indices_func,
    rebuild_compressed_indices_args,
    rebuild_plain_indices_func,
    rebuild_plain_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    layout,
):
    # [FUNCTION DOC] Rebuilds a sparse compressed tensor (CSR, BSR, CSC or BSC) from reduced compressed indices,
    # plain indices and values, with shape and layout. Returns it.
    compressed_indices = rebuild_compressed_indices_func(
        *rebuild_compressed_indices_args
    )
    plain_indices = rebuild_plain_indices_func(*rebuild_plain_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_compressed_tensor(
        compressed_indices, plain_indices, values, shape, layout=layout
    )


def reduce_sparse_tensor(sparse):
    # [FUNCTION DOC] Pickle reducer for sparse tensors. COO: reduces indices and values. Compressed layouts: picks the
    # row/column pointer and index tensors per layout and reduces them with the values.
    # Raises NotImplementedError for an unknown layout. Returns rebuild function and args.
    if sparse.layout is torch.sparse_coo:
        rebuild_indices_func, rebuild_indices_args = reduce_tensor(sparse._indices())
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse._values())
        return (
            rebuild_sparse_coo_tensor,
            (
                rebuild_indices_func,
                rebuild_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                sparse.shape,
                sparse.is_coalesced(),
            ),
        )
    else:
        if sparse.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices = sparse.crow_indices()
            plain_indices = sparse.col_indices()
        elif sparse.layout in {torch.sparse_csc, torch.sparse_bsc}:
            compressed_indices = sparse.ccol_indices()
            plain_indices = sparse.row_indices()
        else:
            raise NotImplementedError(sparse.layout)
        (
            rebuild_compressed_indices_func,
            rebuild_compressed_indices_args,
        ) = reduce_tensor(compressed_indices)
        rebuild_plain_indices_func, rebuild_plain_indices_args = reduce_tensor(
            plain_indices
        )
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse.values())
        return (
            rebuild_sparse_compressed_tensor,
            (
                rebuild_compressed_indices_func,
                rebuild_compressed_indices_args,
                rebuild_plain_indices_func,
                rebuild_plain_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                sparse.shape,
                sparse.layout,
            ),
        )


def fd_id(fd):
    # [FUNCTION DOC] Returns a pair (inode, device) from fstat that uniquely identifies an open file descriptor.
    # Used as the cache key for fd-shared storages.
    # Returns a tuple which uniquely identifies a file descriptor. In Mac OS,
    # this doesn't work with shared memory handles, which is why we don't
    # support the "file_descriptor" sharing method on that platform.
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def storage_from_cache(cls, key):
    # [FUNCTION DOC] Looks key up in shared_cache; returns None on a miss, else a new storage object created from the weak pointer.
    # KNOWN ISSUE: the cls argument is unused and an untyped storage is always returned.
    storage_ref = shared_cache.get(key)
    if storage_ref is None:
        return None
    return torch.UntypedStorage._new_with_weak_ptr(storage_ref.cdata)


def rebuild_storage_fd(cls, df, size):
    # [FUNCTION DOC] Rebuilds a CPU storage from a duplicated file descriptor. Detaches the descriptor, reuses a cached
    # storage when the fd identity is known, else maps a new shared storage and caches a weak ref.
    # Always closes the descriptor afterwards. Returns the storage.
    fd = df.detach()
    try:
        storage = storage_from_cache(cls, fd_id(fd))
        if storage is not None:
            return storage
        storage = cls._new_shared_fd_cpu(fd, size)
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)


def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    # [FUNCTION DOC] Rebuilds a CPU storage shared by filename. Reuses a cached storage if present; otherwise opens it
    # through the manager and handle (wrapping as typed storage when a dtype is given, converting the
    # element count to bytes) and caches a weak ref. Decrements the share count and returns the storage.
    # KNOWN ISSUE: a cache hit returns an untyped storage even when dtype was given.
    storage: torch.TypedStorage | torch.UntypedStorage = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    if dtype is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
    else:
        byte_size = size * torch._utils._element_size(dtype)
        untyped_storage: torch.UntypedStorage = (
            torch.UntypedStorage._new_shared_filename_cpu(manager, handle, byte_size)
        )
        storage = torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=dtype, _internal=True
        )
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()


def rebuild_storage_empty(cls):
    # [FUNCTION DOC] Returns a new empty storage of the given class (zero-size storages cannot be mmapped).
    return cls()


def rebuild_typed_storage(storage, dtype):
    # [FUNCTION DOC] Wraps an untyped storage as TypedStorage with the given dtype and returns it.
    return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)


# Use for torch.storage.TypedStorage
def reduce_typed_storage(storage):
    # [FUNCTION DOC] Pickle reducer for TypedStorage: rebuild function with the untyped storage and dtype.
    return (rebuild_typed_storage, (storage._untyped_storage, storage.dtype))


def rebuild_typed_storage_child(storage, storage_type):
    # [FUNCTION DOC] Wraps an untyped storage in the given typed storage subclass and returns it.
    return storage_type(wrap_storage=storage, _internal=True)


# Use for child classes of torch.storage.TypedStorage, like torch.FloatStorage
def reduce_typed_storage_child(storage):
    # [FUNCTION DOC] Pickle reducer for typed storage subclasses: rebuild function with the untyped storage and its class.
    return (rebuild_typed_storage_child, (storage._untyped_storage, type(storage)))


def reduce_storage(storage):
    # [FUNCTION DOC] Pickle reducer for untyped CPU storage. Raises a runtime error for CUDA and meta storage. With the
    # file_system sharing strategy it shares by filename (adds dtype for typed storage and bumps the share count);
    # empty storages return the empty rebuild; otherwise shares via a duplicated file descriptor.
    # Records a weak ref in shared_cache and returns the rebuild function with the class and metadata.
    from . import get_sharing_strategy

    if storage.is_cuda:
        raise RuntimeError(
            "Cannot pickle CUDA storage; try pickling a CUDA tensor instead"
        )
    elif storage.device.type == "meta":
        raise RuntimeError(
            "Cannot pickle meta storage; try pickling a meta tensor instead"
        )
    elif get_sharing_strategy() == "file_system":
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)


def init_reductions():
    # [FUNCTION DOC] Registers the reducers with the multiprocessing reduction registry for CUDA events, untyped and
    # typed storages, all tensor classes, the Tensor base class and Parameter. Global side effect; no return value.
    reduction.register(torch.cuda.Event, reduce_event)

    for t in torch._storage_classes:
        if t.__name__ == "UntypedStorage":
            reduction.register(t, reduce_storage)
        else:
            reduction.register(t, reduce_typed_storage_child)

    reduction.register(torch.storage.TypedStorage, reduce_typed_storage)

    for t in torch._tensor_classes:
        reduction.register(t, reduce_tensor)

    # TODO: Maybe this should be in tensor_classes? :)
    reduction.register(torch.Tensor, reduce_tensor)

    from torch.nn.parameter import Parameter

    reduction.register(Parameter, reduce_tensor)
