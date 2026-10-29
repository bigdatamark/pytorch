# mypy: allow-untyped-defs

# [MODULE DOC] Python-level storage classes: UntypedStorage (raw byte buffer wrapping the C
# storage), the deprecated TypedStorage (an UntypedStorage plus a dtype), and the legacy per-dtype
# storage class machinery used for pickling backward compatibility. Exports TypedStorage and
# UntypedStorage via __all__. Import-time side effects: tries to import numpy (sets HAS_NUMPY),
# creates the module-level lock and map used to serialize share_memory_ calls, and patches
# docstrings onto several TypedStorage methods after the class is defined.

from __future__ import annotations

import collections
import copy
import functools
import io
import threading
import warnings
from typing import Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self

import torch
from torch._utils import _to, _type
from torch.types import _bool, _int, Storage


if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


__all__ = ["TypedStorage", "UntypedStorage"]


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


_share_memory_lock = threading.Lock()
_share_memory_map: dict[int, threading.RLock] = {}

T = TypeVar("T", bound="_StorageBase | TypedStorage")


def _throws_on_data_ptr_access(storage: _StorageBase | TypedStorage) -> _bool:
    # [FUNCTION DOC] HELPER: Returns True if reading the data pointer of the storage would raise
    # (fake tensor, functional tensor, or cudagraph-invalidated storage). Queries the two C-level
    # throw flags (immutable and mutable data_ptr) on the storage's C handle and ORs them; never
    # raises itself.
    # Whether storage.data_ptr() would raise (e.g. FakeTensor,
    # FunctionalTensor, or a cudagraph-invalidated storage). Non-throwing
    # replacement for wrapping data_ptr() in a try/except RuntimeError.
    return torch._C._storage_throw_on_immutable_data_ptr(
        storage._cdata
    ) or torch._C._storage_throw_on_mutable_data_ptr(storage._cdata)


class _StorageBase:
    # [CLASS DOC] HELPER: Python base class mixed into UntypedStorage. Most methods are abstract
    # stubs that raise NotImplementedError and are filled in by the C base class
    # torch._C.StorageBase; the concrete methods here (device moves, dtype casts, pinning,
    # sharing, pickling, copy) are implemented on top of those. Class attributes: is_sparse and
    # is_sparse_csr (always False), and _fake_device and _checkpoint_offset (optional metadata
    # stashed on a storage when saving or loading under fake tensor mode). The C handle is exposed
    # as _cdata.
    _cdata: Any
    is_sparse: _bool = False
    is_sparse_csr: _bool = False
    device: torch.device
    # Used when
    # (1) stashing FakeTensor device onto storage in torch.serialization.skip_data
    # (2) stashing device onto storage to propagate to FakeTensor when torch.load under FakeTensorMode
    _fake_device: torch.device | None = None
    # Used when loading with FakeTensorMode to give information about offset of storage in torch.saved-file
    _checkpoint_offset: int | None = None

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: No-op constructor that ignores all arguments; real construction
        # happens in the C base class.
        pass

    def __len__(self) -> _int:
        # [FUNCTION DOC] HELPER: Abstract: number of elements in the storage; raises
        # NotImplementedError here.
        raise NotImplementedError

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: Abstract: element or slice access; raises
        # NotImplementedError here.
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: element or slice assignment; raises
        # NotImplementedError here.
        raise NotImplementedError

    def copy_(self, source: T, non_blocking: _bool | None = None) -> T:
        # [FUNCTION DOC] HELPER: Abstract: copies data from the source storage in place,
        # optionally asynchronously; raises NotImplementedError here.
        raise NotImplementedError

    def new(self) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] HELPER: Abstract: creates a new empty storage of the same kind; raises
        # NotImplementedError here.
        raise NotImplementedError

    def nbytes(self) -> _int:
        # [FUNCTION DOC] HELPER: Abstract: size of the storage in bytes; raises
        # NotImplementedError here.
        raise NotImplementedError

    def size(self) -> _int:
        # [FUNCTION DOC] Returns the size of the storage, which for this class is the byte count
        # from nbytes.
        return self.nbytes()

    def type(
        self, dtype: str | None = None, non_blocking: _bool = False
    ) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] Converts the storage to another type name (or, with no name, returns the
        # type name string) by delegating to the shared _type helper in torch._utils. Takes an
        # optional type string and a non_blocking flag; returns a storage or a string.
        return _type(self, dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] Returns a copy of the storage on a CUDA device (the original if already
        # there) by calling to. device is an optional GPU index; non_blocking makes the copy
        # asynchronous for pinned sources.
        # KNOWN ISSUE: the device index is tested for truthiness, so index 0 is treated as
        # unspecified and falls back to the current CUDA device.
        """Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination GPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("cuda", device) if device else torch.device("cuda")
        return self.to(device=device2, non_blocking=non_blocking)

    def hpu(self, device=None, non_blocking=False) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] Returns a copy of the storage on an HPU device (the original if already
        # there) by calling to. device is an optional HPU index; non_blocking makes the copy
        # asynchronous for pinned sources.
        # KNOWN ISSUE: the device index is tested for truthiness, so index 0 is treated as
        # unspecified and falls back to the current device.
        """Returns a copy of this object in HPU memory.

        If this object is already in HPU memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination HPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("hpu", device) if device else torch.device("hpu")
        return self.to(device=device2, non_blocking=non_blocking)

    def element_size(self) -> _int:
        # [FUNCTION DOC] HELPER: Abstract: size in bytes of one element; raises
        # NotImplementedError here.
        raise NotImplementedError

    def get_device(self) -> _int:
        # [FUNCTION DOC] Returns the index of the device the storage lives on (None for devices
        # without an index, such as CPU).
        return self.device.index

    def data_ptr(self) -> _int:
        # [FUNCTION DOC] HELPER: Abstract: address of the first byte of the data; raises
        # NotImplementedError here.
        raise NotImplementedError

    def resizable(self) -> _bool:
        # [FUNCTION DOC] HELPER: Abstract: whether the storage can be resized; raises
        # NotImplementedError here.
        raise NotImplementedError

    # Defined in torch/csrc/generic/StorageSharing.cpp
    def _share_filename_cpu_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract stub for sharing a CPU storage by file name (implemented
        # in C++ StorageSharing); raises NotImplementedError here.
        raise NotImplementedError

    def _share_fd_cpu_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract stub for sharing a CPU storage by file descriptor
        # (implemented in C++ StorageSharing); raises NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _new_using_filename_cpu(cls, size: _int) -> Self:
        # [FUNCTION DOC] HELPER: Abstract class-level constructor for a CPU storage of the given
        # size backed by a shared file name; raises NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _new_using_fd_cpu(cls, size: _int) -> Self:
        # [FUNCTION DOC] HELPER: Abstract class-level constructor for a CPU storage of the given
        # size backed by a shared file descriptor; raises NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def from_buffer(cls, *args, **kwargs) -> Self:
        # [FUNCTION DOC] HELPER: Abstract class-level constructor creating a storage from a buffer
        # object; raises NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _new_shared_filename_cpu(
        cls,
        manager,
        obj,
        size,
        *,
        device=None,
        dtype=None,
    ) -> Self:
        # [FUNCTION DOC] HELPER: Abstract class-level constructor that rebuilds a CPU storage
        # shared by file name from a manager handle, object handle and size; raises
        # NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        # [FUNCTION DOC] HELPER: Releases an IPC reference counter by forwarding the arguments to
        # the CUDA variant. The device keyword is accepted and ignored.
        return cls._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _release_ipc_counter_cuda(cls, *args, **kwargs) -> Self:
        # [FUNCTION DOC] HELPER: Abstract: releases a CUDA IPC reference counter; raises
        # NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs) -> Self:
        # [FUNCTION DOC] HELPER: Abstract class-level constructor that makes a storage from a weak
        # pointer; raises NotImplementedError here.
        raise NotImplementedError

    def _shared_decref(self) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] HELPER: Abstract: decrements the shared reference count; raises
        # NotImplementedError here.
        raise NotImplementedError

    def _write_file(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: writes the storage to a file; raises
        # NotImplementedError here.
        raise NotImplementedError

    def resize_(self, size: _int):
        # [FUNCTION DOC] HELPER: Abstract: resizes the storage to the given number of bytes;
        # raises NotImplementedError here.
        raise NotImplementedError

    def _weak_ref(self, *args, **kwargs) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] HELPER: Abstract: creates a weak reference to the storage; raises
        # NotImplementedError here.
        raise NotImplementedError

    def _set_from_file(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: fills the storage from a file; raises
        # NotImplementedError here.
        raise NotImplementedError

    def _set_cdata(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: replaces the underlying C handle; raises
        # NotImplementedError here.
        raise NotImplementedError

    def _share_cuda_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: exports the storage for CUDA IPC sharing; raises
        # NotImplementedError here.
        raise NotImplementedError

    def is_shared(self) -> _bool:
        # [FUNCTION DOC] Abstract: whether the storage is in shared memory; raises
        # NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs) -> Self:
        # [FUNCTION DOC] HELPER: Abstract class-level constructor that rebuilds a CUDA storage
        # shared across processes; raises NotImplementedError here.
        raise NotImplementedError

    def _shared_incref(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: increments the shared reference count; raises
        # NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract class-level hook that frees a weak reference; raises
        # NotImplementedError here.
        raise NotImplementedError

    @property
    def is_cuda(self):
        # [FUNCTION DOC] Abstract property: whether the storage is on a CUDA device; raises
        # NotImplementedError here (overridden in UntypedStorage).
        raise NotImplementedError

    @property
    def is_hpu(self):
        # [FUNCTION DOC] Abstract property: whether the storage is on an HPU device; raises
        # NotImplementedError here (overridden in UntypedStorage).
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename, shared, nbytes) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] Abstract class-level constructor mapping a file into a storage given file
        # name, shared flag and byte count; raises NotImplementedError here.
        raise NotImplementedError

    @classmethod
    def _expired(cls, *args, **kwargs) -> _StorageBase | TypedStorage:
        # [FUNCTION DOC] HELPER: Abstract class-level check for expired weak references; raises
        # NotImplementedError here.
        raise NotImplementedError

    def _byteswap(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Abstract: swaps bytes in place for the given element size; raises
        # NotImplementedError here.
        raise NotImplementedError

    def _get_filename(self, *args, **kwargs) -> str | None:
        # [FUNCTION DOC] HELPER: Abstract: returns the backing file name or None; raises
        # NotImplementedError here.
        raise NotImplementedError

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Builds a text description: one line per element followed by a
        # header line with the type name, device and length. For meta-device storages the data
        # lines are replaced by an ellipsis because values do not exist.
        info_str = f"[{torch.typename(self)}(device={self.device}) of size {len(self)}]"
        if self.device.type == "meta":
            return "...\n" + info_str
        data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
        return data_str + "\n" + info_str

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over the elements, reading them one index at
        # a time.
        return iter(self[i] for i in range(self.size()))

    def __copy__(self):
        # [FUNCTION DOC] HELPER: Shallow copy; returns a full clone of the storage data.
        return self.clone()

    def __deepcopy__(self, memo):
        # [FUNCTION DOC] HELPER: Deep copy that uses a per-torch section of the memo mapping keyed
        # by the C handle, so the same storage copied twice within one deep copy returns the same
        # clone. Stores the new clone in the memo.
        memo = memo.setdefault("torch", {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        # [FUNCTION DOC] HELPER: Pickle support: serializes the storage with torch.save (legacy
        # non-zipfile format) into an in-memory buffer and returns the loader function with the
        # bytes as its argument.
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def __sizeof__(self):
        # [FUNCTION DOC] HELPER: Returns the Python object size plus the storage size reported
        # by size.
        return super().__sizeof__() + self.size()

    def clone(self):
        # [FUNCTION DOC] Returns a new storage of the same type and device holding a copy of the
        # data. Allocates new memory.
        """Return a copy of this storage."""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    def tolist(self):
        # [FUNCTION DOC] Returns the elements as a Python list.
        """Return a list containing the elements of this storage."""
        return list(self)

    def cpu(self):
        # [FUNCTION DOC] Returns the storage itself if on CPU, otherwise a new CPU UntypedStorage
        # copy (blocking copy).
        """Return a CPU copy of this storage if it's not already on the CPU."""
        if self.device.type != "cpu":
            return torch.UntypedStorage(self.size()).copy_(self, False)
        return self

    def mps(self):
        # [FUNCTION DOC] Returns the storage itself if on MPS, otherwise a new MPS UntypedStorage
        # copy (blocking copy).
        """Return a MPS copy of this storage if it's not already on the MPS."""
        if self.device.type != "mps":
            return torch.UntypedStorage(self.size(), device="mps").copy_(self, False)
        return self

    def _to(self, dtype):
        # [FUNCTION DOC] HELPER: Casts the byte storage to the given dtype by viewing it as
        # a uint8 tensor, converting the tensor and extracting the typed storage; clones if
        # the result aliases the original memory. Raises TypeError if the argument is not a
        # torch dtype.
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def to(self, *, device: DeviceLikeType, non_blocking: _bool = False):
        # [FUNCTION DOC] Moves the storage to the given device (string or torch.device) by
        # delegating to the shared _to helper; non_blocking requests an asynchronous copy. Returns
        # the new or same storage.
        if not isinstance(device, torch.device):
            device = torch.device(device)
        return _to(self, device, non_blocking)

    def double(self):
        # [FUNCTION DOC] Returns a copy cast to double (float64) via the dtype conversion helper.
        """Casts this storage to double type."""
        return self._to(torch.double)

    def float(self):
        # [FUNCTION DOC] Returns a copy cast to float32 via the dtype conversion helper.
        """Casts this storage to float type."""
        return self._to(torch.float)

    def half(self):
        # [FUNCTION DOC] Returns a copy cast to float16 via the dtype conversion helper.
        """Casts this storage to half type."""
        return self._to(torch.half)

    def long(self):
        # [FUNCTION DOC] Returns a copy cast to int64 via the dtype conversion helper.
        """Casts this storage to long type."""
        return self._to(torch.long)

    def int(self):
        # [FUNCTION DOC] Returns a copy cast to int32 via the dtype conversion helper.
        """Casts this storage to int type."""
        return self._to(torch.int)

    def short(self):
        # [FUNCTION DOC] Returns a copy cast to int16 via the dtype conversion helper.
        """Casts this storage to short type."""
        return self._to(torch.short)

    def char(self):
        # [FUNCTION DOC] Returns a copy cast to int8 via the dtype conversion helper.
        """Casts this storage to char type."""
        return self._to(torch.int8)

    def byte(self):
        # [FUNCTION DOC] Returns a copy cast to uint8 via the dtype conversion helper.
        """Casts this storage to byte type."""
        return self._to(torch.uint8)

    def bool(self):
        # [FUNCTION DOC] Returns a copy cast to bool via the dtype conversion helper.
        """Casts this storage to bool type."""
        return self._to(torch.bool)

    def bfloat16(self):
        # [FUNCTION DOC] Returns a copy cast to bfloat16 via the dtype conversion helper.
        """Casts this storage to bfloat16 type."""
        return self._to(torch.bfloat16)

    def complex_double(self):
        # [FUNCTION DOC] Returns a copy cast to complex128 via the dtype conversion helper.
        """Casts this storage to complex double type."""
        return self._to(torch.cdouble)

    def complex_float(self):
        # [FUNCTION DOC] Returns a copy cast to complex64 via the dtype conversion helper.
        """Casts this storage to complex float type."""
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        # [FUNCTION DOC] Returns a copy cast to float8_e5m2 via the dtype conversion helper.
        """Casts this storage to float8_e5m2 type"""
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        # [FUNCTION DOC] Returns a copy cast to float8_e4m3fn via the dtype conversion helper.
        """Casts this storage to float8_e4m3fn type"""
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        # [FUNCTION DOC] Returns a copy cast to float8_e5m2fnuz via the dtype conversion helper.
        """Casts this storage to float8_e5m2fnuz type"""
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        # [FUNCTION DOC] Returns a copy cast to float8_e4m3fnuz via the dtype conversion helper.
        """Casts this storage to float8_e4m3fnuz type"""
        return self._to(torch.float8_e4m3fnuz)

    def is_pinned(self, device: DeviceLikeType | None = None):
        # [FUNCTION DOC] Returns whether the storage is pinned, by wrapping it in an empty uint8
        # tensor and asking the tensor. The device argument is discouraged.
        r"""Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
        return (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .is_pinned(device)
        )

    def pin_memory(self, device: DeviceLikeType | None = None):
        # [FUNCTION DOC] Returns a pinned copy of a CPU storage (pinning is done through a
        # wrapping uint8 tensor and its untyped storage). Raises TypeError if the storage is not
        # on CPU. The device argument is discouraged.
        r"""Copy the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
        if self.device.type != "cpu":
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")

        pinned_tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .pin_memory(device)
        )
        return pinned_tensor.untyped_storage()

    def share_memory_(self):
        # [FUNCTION DOC] Moves a CPU or meta storage into shared memory using file-system or
        # file-descriptor sharing depending on the global multiprocessing sharing strategy;
        # storages on other devices are left unchanged. Returns self. Side effect: may create
        # shared memory objects.
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        from torch.multiprocessing import get_sharing_strategy

        if self.device.type not in ("cpu", "meta"):
            pass  # only CPU uses POSIX shared memory
        elif get_sharing_strategy() == "file_system":
            self._share_filename_cpu_()
        else:
            self._share_fd_cpu_()
        return self

    @classmethod
    def _new_shared(cls, size, *, device="cpu"):
        # [FUNCTION DOC] HELPER: Class-level constructor for a storage in shared memory. Non-CPU
        # devices just get an ordinary storage of the requested size; CPU storages use file-system
        # or file-descriptor sharing based on the current sharing strategy.
        """Create a new storage in shared memory with the same data type."""
        from torch.multiprocessing import get_sharing_strategy

        device = torch.device(device)
        if device.type != "cpu":
            return cls(size, device=device)
        elif get_sharing_strategy() == "file_system":
            return cls._new_using_filename_cpu(size)
        else:
            return cls._new_using_fd_cpu(size)

    def untyped(self):
        # [FUNCTION DOC] Returns self; a storage is already untyped.
        return self

    def byteswap(self, dtype):
        # [FUNCTION DOC] Swaps the bytes of the underlying data in place for the given dtype. The
        # swap unit is the dtype element size, halved (minimum 1) for complex dtypes so the real
        # and imaginary parts are swapped separately.
        """Swap bytes in underlying data."""
        elem_size = torch._utils._element_size(dtype)
        # for complex types, don't swap first and second numbers
        if dtype.is_complex:
            elem_size = max(int(elem_size / 2), 1)
        self._byteswap(elem_size)


def _share_memory_lock_protected(fn):
    # [FUNCTION DOC] HELPER: Decorator that makes a sharing method safe against concurrent calls
    # on the same storage. It records an in-progress lock per C handle in a global mapping under a
    # global lock; a second thread waits for the first to finish, then still runs the wrapped
    # function. Returns the wrapper.
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Registers or waits on the per-storage lock, runs the wrapped
        # method, and finally releases and removes the lock entry. Raises AssertionError if the
        # storage C handle changed while the lock was held.
        to_free = None
        to_wait = None
        with _share_memory_lock:
            key = self._cdata
            if key in _share_memory_map:
                to_wait = _share_memory_map[key]
            else:
                _share_memory_map[key] = threading.RLock()
                _share_memory_map[key].acquire()
                to_free = key

        # If we're already in the process of sharing the storage, wait
        # for it to be done.
        if to_wait is not None:
            with to_wait:
                pass

        try:
            return fn(self, *args, **kwargs)
        finally:
            # If we acquired the storage lock here and we're done working on it
            # we can now release it and free the entry.
            if to_free is not None:
                # Ensure that the cdata from the storage didn't change and only
                # the data_ptr did.
                if self._cdata != to_free:
                    raise AssertionError(
                        f"storage cdata changed unexpectedly: expected {to_free}, got {self._cdata}"
                    )
                with _share_memory_lock:
                    _share_memory_map[to_free].release()
                    del _share_memory_map[to_free]

    return wrapper


class UntypedStorage(torch._C.StorageBase, _StorageBase):
    # [CLASS DOC] Public raw byte storage: combines the C base class with the Python base class.
    # Adds a meta-device guard on indexing, device-type properties, the filename property and
    # lock-protected sharing methods. State lives in the C object; the device is exposed as the
    # device attribute.
    def __getitem__(self, *args, **kwargs):
        # [FUNCTION DOC] Indexes the storage; raises NotImplementedError for meta-device storages,
        # otherwise defers to the C implementation.
        if self.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")
        return super().__getitem__(*args, **kwargs)

    @property
    def is_cuda(self):
        # [FUNCTION DOC] Returns True when the storage device type is cuda.
        return self.device.type == "cuda"

    @property
    def is_hpu(self):
        # [FUNCTION DOC] Returns True when the storage device type is hpu.
        return self.device.type == "hpu"

    @property
    def filename(self) -> str | None:
        # [FUNCTION DOC] Returns the file name the storage is mapped from, or None if it is
        # not file-backed.
        """Returns the file name associated with this storage.

        The file name will be a string if the storage is on CPU and was created via
        :meth:`~torch.from_file()` with ``shared`` as ``True``. This attribute is ``None`` otherwise.
        """
        return self._get_filename()

    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        # [FUNCTION DOC] Moves the storage into shared memory, serialized per storage by the lock
        # decorator; delegates to the base implementation and returns self. No-op for already
        # shared and non-CPU storages.
        """
        Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        .. note::
            When all references to a storage in shared memory are deleted, the associated shared memory
            object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
            even if the current process exits unexpectedly.

            It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

            #. ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
               POSIX shared memory object while :meth:`from_file` uses
               `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
            #. Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
               to map the file/object into the current virtual address space
            #. ``share_memory_`` will call ``shm_unlink(3)`` on the object after mapping it to make sure the shared memory
               object is freed when no process has the object open. ``torch.from_file(shared=True)`` does not unlink the
               file. This file is persistent and will remain until it is deleted by the user.

        Returns:
            ``self``
        """
        return super().share_memory_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lock-protected wrapper that shares the CPU storage through a file
        # descriptor; returns what the C implementation returns.
        return super()._share_fd_cpu_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Lock-protected wrapper that shares the CPU storage through a file
        # name; returns what the C implementation returns.
        return super()._share_filename_cpu_(*args, **kwargs)


def _load_from_bytes(b):
    # [FUNCTION DOC] HELPER: Unpickling helper: loads an object saved by torch.save from raw
    # bytes using full (not weights-only) unpickling. Executes pickled code, so only use with
    # trusted data.
    return torch.load(io.BytesIO(b), weights_only=False)


@functools.cache
def _new_dtypes():
    # [FUNCTION DOC] HELPER: Cached. Returns the set of newer dtypes (float8 variants, bit types,
    # complex32, wider unsigned ints and similar) that are serialized as an UntypedStorage plus a
    # dtype rather than as a legacy typed storage class.
    # These are dtypes serialized as UntypedStorage unlike those in
    # _dtype_to_storage_type_map
    return {
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.float8_e8m0fnu,
        torch.float4_e2m1fn_x2,
        torch.bits8,
        torch.bits16,
        torch.bits1x8,
        torch.bits2x4,
        torch.bits4x2,
        torch.complex32,
        torch.bcomplex32,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }


@functools.cache
def _dtype_to_storage_type_map():
    # [FUNCTION DOC] HELPER: Cached. Returns the mapping from dtype to legacy storage class
    # name used only for backward and forward compatibility of old pickles; no new dtypes
    # should be added.
    # NOTE: We should no longer add dtypes to this map. This map
    # is only used for BC/FC with older PyTorch versions. Going forward,
    # new dtypes of TypedStorage should not translate to a legacy
    # <type>Storage class. Instead, new dtypes of TypedStorage should
    # be serialized as an UntypedStorage paired with a torch.dtype
    return {
        torch.double: "DoubleStorage",
        torch.float: "FloatStorage",
        torch.half: "HalfStorage",
        torch.long: "LongStorage",
        torch.int: "IntStorage",
        torch.int16: "ShortStorage",
        torch.int8: "CharStorage",
        torch.uint8: "ByteStorage",
        torch.bool: "BoolStorage",
        torch.bfloat16: "BFloat16Storage",
        torch.cdouble: "ComplexDoubleStorage",
        torch.cfloat: "ComplexFloatStorage",
        torch.qint8: "QInt8Storage",
        torch.qint32: "QInt32Storage",
        torch.quint8: "QUInt8Storage",
        torch.quint4x2: "QUInt4x2Storage",
        torch.quint2x4: "QUInt2x4Storage",
    }


@functools.cache
def _storage_type_to_dtype_map():
    # [FUNCTION DOC] HELPER: Cached. Returns the inverse of the dtype to legacy storage class
    # name mapping.
    dtype_map = {val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map


def _get_storage_from_sequence(sequence, dtype, device):
    # [FUNCTION DOC] HELPER: Builds an untyped storage from a Python sequence. Quantized dtypes
    # are built through the matching plain integer dtype; other dtypes build the tensor directly.
    # Returns the untyped storage of the temporary tensor.
    if dtype in [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]:
        interpret_dtypes = {
            torch.quint8: torch.uint8,
            torch.quint4x2: torch.uint8,
            torch.quint2x4: torch.uint8,
            torch.qint32: torch.int32,
            torch.qint8: torch.int8,
        }
        tmp_tensor = torch.tensor(
            sequence, dtype=interpret_dtypes[dtype], device=device
        )

    else:
        tmp_tensor = torch.tensor(sequence, dtype=dtype, device=device)

    return tmp_tensor._typed_storage()._untyped_storage


def _isint(x):
    # [FUNCTION DOC] HELPER: Returns True if the value is a Python integer (or a numpy integer
    # when numpy is available).
    if HAS_NUMPY:
        return isinstance(x, (int, np.integer))  # pyrefly: ignore [missing-attribute]
    else:
        return isinstance(x, int)


_always_warn_typed_storage_removal = False


def _get_always_warn_typed_storage_removal():
    # [FUNCTION DOC] HELPER: Returns the global flag that forces the TypedStorage deprecation
    # warning on every use.
    return _always_warn_typed_storage_removal


def _set_always_warn_typed_storage_removal(always_warn):
    # [FUNCTION DOC] HELPER: Sets the global flag that forces the TypedStorage deprecation warning
    # on every use. Raises AssertionError if the argument is not a bool.
    global _always_warn_typed_storage_removal
    if not isinstance(always_warn, bool):
        raise AssertionError(
            f"always_warn must be bool, got {type(always_warn).__name__}"
        )
    _always_warn_typed_storage_removal = always_warn


def _warn_typed_storage_removal(stacklevel=2):
    # [FUNCTION DOC] HELPER: Emits a UserWarning that TypedStorage is deprecated, the first time
    # only unless the always-warn flag is set. Records the fact it has warned as an attribute on
    # the function. stacklevel is adjusted so the warning points at the caller.
    global _always_warn_typed_storage_removal

    def is_first_time():
        # [FUNCTION DOC] HELPER: Returns True if the warning has not been emitted yet.
        if not hasattr(_warn_typed_storage_removal, "has_warned"):
            return True
        else:
            return not _warn_typed_storage_removal.__dict__["has_warned"]

    if _get_always_warn_typed_storage_removal() or is_first_time():
        message = (
            "TypedStorage is deprecated. It will be removed in the future and "
            "UntypedStorage will be the only storage class. This should only matter "
            "to you if you are using storages directly.  To access UntypedStorage "
            "directly, use tensor.untyped_storage() instead of tensor.storage()"
        )
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        _warn_typed_storage_removal.__dict__["has_warned"] = True


def _reset_warn_typed_storage_removal():
    # [FUNCTION DOC] HELPER: Clears the recorded warned state so the deprecation warning can
    # fire again.
    _warn_typed_storage_removal.__dict__["has_warned"] = False


def _get_device_from_module(module: str):
    # [FUNCTION DOC] HELPER: Maps a module name to a device name: returns the last dotted part if
    # it is cuda, hpu or the private-use backend name, otherwise cpu.
    last_part = module.rsplit(".", 1)[-1]
    if last_part in ["cuda", torch._C._get_privateuse1_backend_name(), "hpu"]:
        return last_part
    else:
        return "cpu"


class TypedStorage:
    # [CLASS DOC] Deprecated storage with a dtype, implemented as a wrapper around an
    # UntypedStorage. State: dtype (element type), _untyped_storage (the wrapped raw bytes). Sizes
    # are in elements, byte counts come from the wrapped storage. Nearly every public method emits
    # a deprecation warning and then delegates to a private warning-free twin. Subclasses of the
    # legacy storage classes are only usable through the constructor redirect.
    is_sparse: _bool = False
    # Used when stashing FakeTensor device onto storage in torch.save(metadata_only=True)
    _fake_device: torch.device | None = None

    dtype: torch.dtype

    @property
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Returns the dtype; kept so subclasses can define it as a class
        # attribute and internals avoid the public attribute.
        # KNOWN ISSUE: the constructor redirect for legacy subclasses reads it via the class,
        # which yields the property object rather than a dtype.
        return self.dtype

    @property
    def filename(self) -> str | None:
        # [FUNCTION DOC] Returns the file name the storage is memory mapped from, or None.
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
        or ``None`` if the storage was not created by memory mapping a file."""
        return self._untyped_storage.filename

    def fill_(self, value):
        # [FUNCTION DOC] Sets every element to the given value by slice assignment over the whole
        # storage. Warns of deprecation. Returns self.
        _warn_typed_storage_removal()
        self._setitem(slice(0, self._size()), value)
        return self

    def __new__(
        cls,
        *args,
        wrap_storage=None,
        dtype=None,
        device=None,
        _internal=False,
    ):
        # [FUNCTION DOC] HELPER: Controls construction. For TypedStorage itself, creates a plain
        # instance. For legacy subclasses it validates arguments (no device, no dtype, at most one
        # positional argument that is an int or sequence, or only a wrap_storage UntypedStorage on
        # the matching device) and returns a new TypedStorage with the class dtype, rather than an
        # instance of the subclass. Raises RuntimeError or TypeError for invalid combinations, and
        # for the legacy base class itself.
        # KNOWN ISSUE: the positional path passes the class attribute _dtype, which is a property
        # on the class, not a dtype.
        if not _internal:
            _warn_typed_storage_removal()

        if cls == torch.storage._LegacyStorage:
            raise RuntimeError(
                "Only child classes of _LegacyStorage can be instantiated"
            )

        if cls == TypedStorage:
            return super().__new__(cls)

        else:
            arg_error_msg = (
                f"{cls}.__new__ received an invalid combination "
                f"of arguments. Expected one of:\n"
                " * no arguments\n"
                " * (int size)\n"
                " * (Sequence data)\n"
                " * (*, UntypedStorage wrap_storage)"
            )

            if device is not None:
                raise RuntimeError(
                    arg_error_msg + "\nKeyword argument 'device' cannot be specified"
                )

            if dtype is not None:
                raise RuntimeError(
                    arg_error_msg + "\nKeyword argument 'dtype' cannot be specified"
                )

            if wrap_storage is None:
                if len(args) > 1:
                    raise RuntimeError(
                        arg_error_msg + "\nToo many positional arguments"
                    )

                if (
                    len(args) == 1
                    and not _isint(args[0])
                    and not isinstance(args[0], collections.abc.Sequence)
                ):
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument type not recognized: {type(args[0])}"
                    )

                return TypedStorage(
                    *args,
                    dtype=cls._dtype,
                    device=_get_device_from_module(cls.__module__),
                    _internal=True,
                )

            else:
                if len(args) != 0:
                    raise RuntimeError(
                        arg_error_msg
                        + "\nNo positional arguments should be given when using "
                        "'wrap_storage'"
                    )

                if not isinstance(wrap_storage, torch.UntypedStorage):
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                    )

                cls_device = _get_device_from_module(cls.__module__)

                if wrap_storage.device.type != cls_device:
                    raise RuntimeError(
                        arg_error_msg
                        + f"\nDevice of 'wrap_storage' must be {cls_device}"
                        f", but got {wrap_storage.device.type}"
                    )

                return TypedStorage(
                    *args,
                    wrap_storage=wrap_storage,
                    dtype=cls.dtype,
                    _internal=True,
                )

    def __init__(
        self,
        *args,
        device=None,
        dtype=None,
        wrap_storage=None,
        _internal=False,
    ):
        # [FUNCTION DOC] Builds a TypedStorage either by wrapping an existing UntypedStorage
        # (dtype required, device and positional arguments forbidden) or by allocating a new one
        # from nothing, an element count or a Python sequence on the given device (default CPU,
        # default dtype is the global default dtype). Raises RuntimeError or TypeError for invalid
        # argument combinations, and RuntimeError for quantized dtypes on CUDA. Warns of
        # deprecation unless internal.
        if not _internal:
            _warn_typed_storage_removal()
        arg_error_msg = (
            "TypedStorage.__init__ received an invalid combination "
            "of arguments. Expected one of:\n"
            " * (*, torch.device device, torch.dtype dtype)\n"
            " * (int size, *, torch.device device, torch.dtype dtype)\n"
            " * (Sequence data, *, torch.device device, torch.dtype dtype)\n"
            " * (*, UntypedStorage wrap_storage, torch.dtype dtype)"
        )

        if wrap_storage is not None:
            if len(args) != 0:
                raise RuntimeError(
                    arg_error_msg
                    + "\nNo positional arguments should be given when using "
                    "'wrap_storage'"
                )

            if dtype is None:
                raise RuntimeError(
                    arg_error_msg + "\nArgument 'dtype' must be specified"
                )

            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    arg_error_msg
                    + f"\nArgument 'dtype' must be torch.dtype, not {type(dtype)}"
                )

            if device is not None:
                raise RuntimeError(
                    arg_error_msg
                    + "\nArgument 'device' should not be specified when 'wrap_storage' is given"
                )

            self.dtype = dtype

            if not isinstance(wrap_storage, torch.UntypedStorage):
                raise TypeError(
                    arg_error_msg
                    + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                )

            self._untyped_storage = wrap_storage

        else:
            self.dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device("cpu" if device is None else device)

            if self.dtype in [
                torch.quint8,
                torch.quint4x2,
                torch.quint2x4,
                torch.qint32,
                torch.qint8,
            ]:
                if device.type == "cuda":
                    raise RuntimeError(
                        "Cannot create CUDA storage with quantized dtype"
                    )

            if len(args) == 0:
                self._untyped_storage = torch.UntypedStorage(device=device)

            elif len(args) == 1:
                if _isint(args[0]):
                    self._untyped_storage = torch.UntypedStorage(
                        int(args[0]) * self._element_size(), device=device
                    )
                elif isinstance(args[0], collections.abc.Sequence):
                    self._untyped_storage = _get_storage_from_sequence(
                        args[0], self.dtype, device
                    )
                else:
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument type not recognized: {type(args[0])}"
                    )

            else:
                raise RuntimeError(arg_error_msg + "\nToo many positional arguments")

    @property
    def is_cuda(self):
        # [FUNCTION DOC] Warns of deprecation; returns True if the storage is on CUDA.
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "cuda"

    @property
    def is_hpu(self):
        # [FUNCTION DOC] Warns of deprecation; returns True if the storage is on HPU.
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "hpu"

    def untyped(self):
        # [FUNCTION DOC] Warns of deprecation; returns the wrapped UntypedStorage.
        """Return the internal :class:`torch.UntypedStorage`."""
        _warn_typed_storage_removal()
        return self._untyped_storage

    def _new_wrapped_storage(self, untyped_storage) -> Self:
        # [FUNCTION DOC] HELPER: Wraps a new UntypedStorage in an object of the same storage kind
        # (a TypedStorage with the same dtype, or the same subclass). Raises AssertionError if the
        # argument is not exactly UntypedStorage.
        if type(untyped_storage) is not torch.UntypedStorage:
            raise AssertionError(
                f"expected UntypedStorage, got {type(untyped_storage).__name__}"
            )

        if type(self) is TypedStorage:
            return cast(
                Self,
                TypedStorage(
                    wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
                ),
            )
        else:
            return type(self)(wrap_storage=untyped_storage)

    def __len__(self):
        # [FUNCTION DOC] Warns of deprecation; returns the number of elements.
        _warn_typed_storage_removal()
        return self._size()

    def _maybe_wrap_index(self, idx, is_stop=False):
        # [FUNCTION DOC] HELPER: Normalizes an index to a nonnegative position: None becomes 0 (or
        # the size for a stop index), negative values wrap around, out-of-range values raise
        # IndexError, and a non-int raises TypeError. A stop index may equal the size.
        # KNOWN ISSUE: the error message uses the public size method, which emits an extra
        # deprecation warning.
        if idx is None:
            if is_stop:
                return self._size()
            else:
                return 0

        else:
            if type(idx) is not int:
                raise TypeError(f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                if (idx > self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                if idx > 0:
                    return idx
                else:
                    return idx % self._size()
            else:
                if (idx >= self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                return idx % self._size()

    def __setitem__(self, idx, value):
        # [FUNCTION DOC] Warns of deprecation; sets one element or a slice via the private setter.
        _warn_typed_storage_removal()
        return self._setitem(idx, value)

    def _setitem(self, idx, value):
        # [FUNCTION DOC] HELPER: Assigns a value at an int or slice index by viewing the storage
        # as a tensor (quantized dtypes are viewed through the matching integer dtype) and doing
        # tensor assignment. Raises RuntimeError for an unsupported index type or if the value is
        # a storage.
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if torch.is_storage(value):
            raise RuntimeError(f"cannot set item with value type {type(value)}")
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            tmp_dtype = interpret_dtypes[self.dtype]
            tmp_tensor = torch.tensor(
                [], dtype=tmp_dtype, device=self._untyped_storage.device
            )
            tmp_tensor.set_(
                TypedStorage(
                    wrap_storage=self._untyped_storage, dtype=tmp_dtype, _internal=True
                )
            )
        else:
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)

        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        # [FUNCTION DOC] Warns of deprecation; returns the element at an index via the
        # private getter.
        _warn_typed_storage_removal()
        return self._getitem(idx)

    def _getitem(self, idx):
        # [FUNCTION DOC] HELPER: Returns the Python scalar at an int index by viewing the storage
        # as a tensor, with fake mode temporarily disabled. Quantized dtypes are read as their
        # integer representation. Raises NotImplementedError for meta storages and RuntimeError
        # for slices or non-int indexes.
        if self._untyped_storage.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")

        # NOTE: Before TypedStorage existed, indexing with a slice used to be
        # possible for <type>Storage objects. However, it would return
        # a storage view, which would be a hassle to implement in TypedStorage,
        # so it was disabled
        if isinstance(idx, slice):
            raise RuntimeError(
                "slices are only supported in UntypedStorage.__getitem__"
            )
        elif not isinstance(idx, int):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")

        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            return TypedStorage(
                wrap_storage=self._untyped_storage,
                dtype=interpret_dtypes[self.dtype],
                _internal=True,
            )._getitem(idx)

        idx_wrapped = self._maybe_wrap_index(idx)
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)
            return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking: bool | None = None):
        _warn_typed_storage_removal()
        if isinstance(source, TypedStorage):
            self._untyped_storage.copy_(source._untyped_storage, non_blocking)
        else:
            self._untyped_storage.copy_(source, non_blocking)
        # [FUNCTION DOC] Warns of deprecation; copies data in place from another storage (typed or
        # untyped) into this one and returns self.
        return self

    def nbytes(self):
        # [FUNCTION DOC] Warns of deprecation; returns the size in bytes.
        _warn_typed_storage_removal()
        return self._nbytes()

    # For internal use only, to avoid deprecation warning
    def _nbytes(self):
        # [FUNCTION DOC] HELPER: Returns the byte size without a deprecation warning.
        return self._untyped_storage.nbytes()

    def type(
        self,
        dtype: str | None = None,
        non_blocking: bool = False,
    ) -> _StorageBase | TypedStorage | str:
        # [FUNCTION DOC] Warns of deprecation; with no dtype, returns the dotted type name of
        # the legacy storage class (or of this class); with a dtype, converts through the
        # wrapped storage.
        _warn_typed_storage_removal()
        if dtype is None:
            legacy_class = self._get_legacy_storage_class()

            if legacy_class is not None:
                return legacy_class.__module__ + "." + legacy_class.__name__

            return ".".join([self.__module__, type(self).__name__])

        else:
            return self._untyped_storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False) -> Self:
        # [FUNCTION DOC] Warns of deprecation; returns a copy on CUDA wrapped in a storage of the
        # same kind. Raises RuntimeError for quantized dtypes.
        _warn_typed_storage_removal()
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create CUDA storage with quantized dtype")
        cuda_storage = self._untyped_storage.cuda(device, non_blocking)
        return self._new_wrapped_storage(cuda_storage)

    def hpu(self, device=None, non_blocking=False) -> Self:
        # [FUNCTION DOC] Warns of deprecation; returns a copy on HPU wrapped in a storage of the
        # same kind. Raises RuntimeError for quantized dtypes.
        _warn_typed_storage_removal()
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create HPU storage with quantized dtype")
        hpu_storage = self._untyped_storage.hpu(device, non_blocking)
        return self._new_wrapped_storage(hpu_storage)

    def to(self, *, device: DeviceLikeType, non_blocking: bool = False) -> Self:
        # [FUNCTION DOC] Warns of deprecation; moves the storage to a device and returns the
        # wrapped result. Raises RuntimeError for quantized dtypes.
        _warn_typed_storage_removal()
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError(
                f"Cannot create {device.type.upper()} storage with quantized dtype"
            )
        to_storage = self._untyped_storage.to(device=device, non_blocking=non_blocking)
        return self._new_wrapped_storage(to_storage)

    def element_size(self):
        # [FUNCTION DOC] Warns of deprecation; returns the element size in bytes.
        _warn_typed_storage_removal()
        return self._element_size()

    # For internal use only, to avoid deprecation warning
    def _element_size(self):
        # [FUNCTION DOC] HELPER: Returns the element size in bytes of the dtype without a warning.
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> _int:
        # [FUNCTION DOC] Warns of deprecation; returns the device index of the wrapped storage.
        _warn_typed_storage_removal()
        return self._untyped_storage.get_device()

    def __str__(self):
        # [FUNCTION DOC] Warns of deprecation; builds a text description with each element on
        # its own line and a header with type, dtype, device and length; meta storages show only
        # an ellipsis.
        _warn_typed_storage_removal()
        info_str = (
            f"[{torch.typename(self)}(dtype={self.dtype}, "
            f"device={self.device}) of size {len(self)}]"
        )
        if self.device.type == "meta":
            return "...\n" + info_str
        else:
            data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
            return data_str + "\n" + info_str

    def __repr__(self):
        # [FUNCTION DOC] Warns of deprecation; returns the same text as the string conversion.
        _warn_typed_storage_removal()
        return str(self)

    def __iter__(self):
        # [FUNCTION DOC] Warns of deprecation; returns an iterator over the elements.
        _warn_typed_storage_removal()
        return iter(self[i] for i in range(self.size()))

    def __copy__(self):
        # [FUNCTION DOC] Warns of deprecation; shallow-copies the wrapped storage and wraps
        # the result.
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(copy.copy(self._untyped_storage))

    def __deepcopy__(self, memo):
        # [FUNCTION DOC] Warns of deprecation; deep copies via the private helper.
        _warn_typed_storage_removal()
        return self._deepcopy(memo)

    # For internal use only, to avoid deprecation warning
    def _deepcopy(self, memo):
        # [FUNCTION DOC] HELPER: Deep copies the wrapped storage (using the memo) and wraps the
        # result, without a warning.
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))

    def __sizeof__(self):
        # [FUNCTION DOC] Warns of deprecation; returns the Python object size plus the byte size.
        _warn_typed_storage_removal()
        return super().__sizeof__() + self.nbytes()

    def clone(self):
        # [FUNCTION DOC] Warns of deprecation; returns a new storage holding a copy of the data.
        """Return a copy of this storage."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.clone())

    def tolist(self):
        # [FUNCTION DOC] Warns of deprecation; returns the elements as a Python list.
        """Return a list containing the elements of this storage."""
        _warn_typed_storage_removal()
        return list(self)

    def cpu(self):
        # [FUNCTION DOC] Warns of deprecation; returns a CPU copy (or self-equivalent) wrapped in
        # a storage of the same kind.
        """Return a CPU copy of this storage if it's not already on the CPU."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.cpu())

    def is_pinned(self, device: DeviceLikeType | None = None):
        # [FUNCTION DOC] Warns of deprecation; returns whether the wrapped storage is pinned.
        r"""Determine whether the CPU TypedStorage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
        _warn_typed_storage_removal()
        return self._untyped_storage.is_pinned(device)

    def pin_memory(self, device: DeviceLikeType | None = None):
        # [FUNCTION DOC] Warns of deprecation; returns a pinned copy of the data wrapped in a
        # storage of the same kind.
        r"""Copy the CPU TypedStorage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``None``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(
            self._untyped_storage.pin_memory(device=device)
        )

    def share_memory_(self):
        # [FUNCTION DOC] Warns of deprecation; moves the data to shared memory via the private
        # helper and returns self.
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        _warn_typed_storage_removal()
        return self._share_memory_()

    # For internal use only, to avoid deprecation warning
    def _share_memory_(self):
        # [FUNCTION DOC] HELPER: Moves the wrapped storage into shared memory and returns self,
        # without a warning.
        self._untyped_storage.share_memory_()
        return self

    def _new_shared(self, size, *, device=None):
        # [FUNCTION DOC] HELPER: Creates a new shared-memory storage with room for the given
        # number of elements and the same dtype. device defaults to CPU.
        """Create a new storage in shared memory with the same data type."""
        if device is None:
            device = "cpu"
        device = torch.device(device)
        untyped_storage = torch.UntypedStorage._new_shared(
            size * self._element_size(), device=device
        )
        return TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    @property
    def _cdata(self):
        # [FUNCTION DOC] HELPER: Property returning the C handle of the wrapped storage.
        return self._untyped_storage._cdata

    @property
    def device(self):
        # [FUNCTION DOC] Warns of deprecation; returns the device of the wrapped storage.
        _warn_typed_storage_removal()
        return self._untyped_storage.device

    def size(self):
        # [FUNCTION DOC] Warns of deprecation; returns the number of elements.
        _warn_typed_storage_removal()
        return self._size()

    # For internal use only, to avoid deprecation warning
    def _size(self):
        # [FUNCTION DOC] HELPER: Returns the number of elements as the byte size divided by the
        # element size, avoiding the length dunder that would require an int result.
        # NB: don't indirect through __len__, as that requires
        # an int to be returned
        return self._untyped_storage.nbytes() // self._element_size()

    def pickle_storage_type(self):
        # [FUNCTION DOC] Warns of deprecation; returns the legacy storage class name for
        # the dtype.
        _warn_typed_storage_removal()
        return self._pickle_storage_type()

    # For internal use only, to avoid deprecation warning
    def _pickle_storage_type(self):
        # [FUNCTION DOC] HELPER: Looks up the legacy storage class name of the dtype. Raises
        # KeyError if the dtype is not in the mapping.
        try:
            return _dtype_to_storage_type_map()[self.dtype]
        except KeyError as e:
            raise KeyError(f"dtype {self.dtype} is not recognized") from e

    def __reduce__(self):
        # [FUNCTION DOC] HELPER: Pickle support: serializes with torch.save (legacy format) into
        # memory and returns the bytes loader and its argument.
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def data_ptr(self):
        # [FUNCTION DOC] Warns of deprecation; returns the data pointer address.
        _warn_typed_storage_removal()
        return self._data_ptr()

    # For internal use only, to avoid deprecation warning
    def _data_ptr(self):
        # [FUNCTION DOC] HELPER: Returns the data pointer of the wrapped storage without
        # a warning.
        return self._untyped_storage.data_ptr()

    def resizable(self):
        # [FUNCTION DOC] Warns of deprecation; returns whether the wrapped storage is resizable.
        _warn_typed_storage_removal()
        return self._untyped_storage.resizable()

    def resize_(self, size):
        # [FUNCTION DOC] Warns of deprecation; resizes the storage to the given element count.
        # Returns nothing.
        _warn_typed_storage_removal()
        self._resize_(size)

    # For internal use only, to avoid deprecation warning
    def _resize_(self, size):
        # [FUNCTION DOC] HELPER: Resizes the wrapped storage to the element count converted to
        # bytes, without a warning.
        self._untyped_storage.resize_(size * self._element_size())

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Class-level passthrough to the weak-reference freeing hook
        # of UntypedStorage.
        return UntypedStorage._free_weak_ref(*args, **kwargs)

    def _weak_ref(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passthrough creating a weak reference of the wrapped storage.
        return self._untyped_storage._weak_ref(*args, **kwargs)

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        # [FUNCTION DOC] Warns of deprecation; class-level constructor from a buffer through the
        # private helper.
        _warn_typed_storage_removal()
        return cls._from_buffer(*args, **kwargs)

    @classmethod
    def _from_buffer(cls, *args, dtype=None, device=None, **kwargs):
        # [FUNCTION DOC] HELPER: Builds a TypedStorage from a buffer. On TypedStorage itself,
        # dtype (default global dtype) and device (CPU only) may be given; on legacy subclasses
        # they may not. Raises RuntimeError for non-CPU devices or forbidden keywords.
        # KNOWN ISSUE: the subclass branch rejects a call with exactly five positional arguments,
        # a magic number that does not correspond to anything obvious, and reads the class dtype
        # through the property attribute.
        if cls == TypedStorage:
            dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device("cpu" if device is None else device)
            if device.type != "cpu":
                raise RuntimeError(
                    f"TypedStorage.from_buffer: Not available for device {device.type}"
                )
            untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        else:
            if dtype is not None or len(args) == 5:
                raise RuntimeError(
                    "from_buffer: 'dtype' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )
            if device is not None:
                raise RuntimeError(
                    "from_buffer: 'device' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )

            dtype = cls._dtype
            untyped_storage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        return TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)

    def _to(self, dtype):
        # [FUNCTION DOC] HELPER: Casts to the given dtype by converting a tensor view and
        # returning the typed storage, cloning if memory is shared with the original. Raises
        # TypeError if the argument is not a dtype.
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=self.dtype, device=self.device)
            .set_(self)
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to double.
        """Casts this storage to double type."""
        _warn_typed_storage_removal()
        return self._to(torch.double)

    def float(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to float.
        """Casts this storage to float type."""
        _warn_typed_storage_removal()
        return self._to(torch.float)

    def half(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to half.
        """Casts this storage to half type."""
        _warn_typed_storage_removal()
        return self._to(torch.half)

    def long(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to long.
        """Casts this storage to long type."""
        _warn_typed_storage_removal()
        return self._to(torch.long)

    def int(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to int.
        """Casts this storage to int type."""
        _warn_typed_storage_removal()
        return self._to(torch.int)

    def short(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to short.
        """Casts this storage to short type."""
        _warn_typed_storage_removal()
        return self._to(torch.short)

    def char(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to int8.
        """Casts this storage to char type."""
        _warn_typed_storage_removal()
        return self._to(torch.int8)

    def byte(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to uint8.
        """Casts this storage to byte type."""
        _warn_typed_storage_removal()
        return self._to(torch.uint8)

    def bool(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to bool.
        """Casts this storage to bool type."""
        _warn_typed_storage_removal()
        return self._to(torch.bool)

    def bfloat16(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to bfloat16.
        """Casts this storage to bfloat16 type."""
        _warn_typed_storage_removal()
        return self._to(torch.bfloat16)

    def complex_double(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to complex double.
        """Casts this storage to complex double type."""
        _warn_typed_storage_removal()
        return self._to(torch.cdouble)

    def complex_float(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to complex float.
        """Casts this storage to complex float type."""
        _warn_typed_storage_removal()
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to float8_e5m2.
        """Casts this storage to float8_e5m2 type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to float8_e4m3fn.
        """Casts this storage to float8_e4m3fn type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to float8_e5m2fnuz.
        """Casts this storage to float8_e5m2fnuz type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        # [FUNCTION DOC] Warns of deprecation; returns a copy cast to float8_e4m3fnuz.
        """Casts this storage to float8_e4m3fnuz type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fnuz)

    @classmethod
    def from_file(cls, filename, shared, size):
        # [FUNCTION DOC] Class-level constructor: creates a CPU storage backed by a memory-mapped
        # file with the given name, shared flag and size in elements, and wraps it in this class.
        # Warns of deprecation. Raises RuntimeError when called on TypedStorage itself instead of
        # a legacy subclass.
        """from_file(filename, shared=False, size=0) -> Storage

        Creates a CPU storage backed by a memory-mapped file.

        If ``shared`` is ``True``, then memory is shared between all processes.
        All changes are written to the file. If ``shared`` is ``False``, then the changes on
        the storage do not affect the file.

        ``size`` is the number of elements in the storage. If ``shared`` is ``False``,
        then the file must contain at least ``size * sizeof(Type)`` bytes
        (``Type`` is the type of storage). If ``shared`` is ``True`` the file will be created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                            underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
            size (int): number of elements in the storage
        """
        _warn_typed_storage_removal()
        if cls == TypedStorage:
            raise RuntimeError("from_file can only be called on derived classes")
        untyped_storage = UntypedStorage.from_file(
            filename, shared, size * torch._utils._element_size(cls.dtype)
        )
        storage = cls(wrap_storage=untyped_storage)
        return storage

    @classmethod
    def _expired(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Class-level passthrough to the expired-weak-reference check
        # of UntypedStorage.
        return UntypedStorage._expired(*args, **kwargs)

    def _write_file(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passthrough that writes the wrapped storage to a file.
        return self._untyped_storage._write_file(*args, **kwargs)

    def _set_from_file(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passthrough that fills the wrapped storage from a file.
        return self._untyped_storage._set_from_file(*args, **kwargs)

    def _set_cdata(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passthrough that replaces the C handle of the wrapped storage.
        return self._untyped_storage._set_cdata(*args, **kwargs)

    def _share_cuda_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passthrough that exports the wrapped storage for CUDA
        # IPC sharing.
        return self._untyped_storage._share_cuda_(*args, **kwargs)

    def is_shared(self):
        # [FUNCTION DOC] Warns of deprecation; returns whether the storage is in shared memory.
        _warn_typed_storage_removal()
        return self._is_shared()

    # For internal use only, to avoid deprecation warning
    def _is_shared(self):
        # [FUNCTION DOC] HELPER: Returns whether the wrapped storage is in shared memory, without
        # a warning.
        return self._untyped_storage.is_shared()

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Class-level passthrough that rebuilds a shared CUDA storage
        # via UntypedStorage.
        return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)

    def _share_filename_cpu_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Shares the wrapped storage by file name and returns the manager
        # handle, storage handle and a size converted from bytes to elements.
        (
            manager_handle,
            storage_handle,
            size,
        ) = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
        return manager_handle, storage_handle, size // self._element_size()

    def _shared_decref(self):
        # [FUNCTION DOC] HELPER: Decrements the shared reference count of the wrapped storage;
        # returns self.
        self._untyped_storage._shared_decref()
        return self

    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        # [FUNCTION DOC] HELPER: Class-level passthrough that releases the CUDA IPC counter via
        # UntypedStorage; device is ignored.
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    def _shared_incref(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Passthrough that increments the shared reference count of the
        # wrapped storage.
        return self._untyped_storage._shared_incref(*args, **kwargs)

    def _share_fd_cpu_(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Shares the wrapped storage by file descriptor and returns the
        # descriptor and a size converted from bytes to elements.
        fd, size = self._untyped_storage._share_fd_cpu_(*args, **kwargs)
        return fd, size // self._element_size()

    def _get_legacy_storage_class(self):
        # [FUNCTION DOC] HELPER: Finds the legacy storage class (for example FloatStorage in torch
        # or torch.cuda) matching the dtype and device. Returns None if the dtype has no legacy
        # name, the device type has no legacy module, or the class attribute is missing.
        if self.dtype not in _dtype_to_storage_type_map():
            return None

        storage_name = _dtype_to_storage_type_map()[self.dtype]

        if self.device.type not in [
            "cpu",
            "cuda",
            "hpu",
            torch._C._get_privateuse1_backend_name(),
        ]:
            return None

        module = (
            torch if self.device.type == "cpu" else getattr(torch, self.device.type)
        )

        try:
            return getattr(module, storage_name)
        except AttributeError:
            return None


TypedStorage.type.__doc__ = _type.__doc__
TypedStorage.cuda.__doc__ = _StorageBase.cuda.__doc__
TypedStorage.hpu.__doc__ = _StorageBase.hpu.__doc__
TypedStorage.to.__doc__ = _to.__doc__


class _LegacyStorageMeta(type):
    # [CLASS DOC] HELPER: Metaclass of the legacy per-dtype storage classes. It customizes
    # isinstance checks so a plain TypedStorage with matching device and dtype counts as
    # an instance.
    dtype: torch.dtype

    def __instancecheck__(cls, instance):
        # [FUNCTION DOC] HELPER: Returns True only for exact TypedStorage instances whose device
        # type (derived from the legacy class module) and dtype equal those of the class; False
        # for anything else.
        if type(instance) is TypedStorage:
            cls_device = _get_device_from_module(cls.__module__)
            return (cls_device == instance.device.type) and (
                cls.dtype == instance.dtype
            )
        return False


class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):
    # [CLASS DOC] HELPER: Base for legacy per-dtype storage classes such as FloatStorage.
    # Subclasses carry a dtype class attribute and construct TypedStorage objects; only the
    # classmethod helpers for shared-memory and IPC reconstruction are defined here.
    @classmethod
    def _new_shared(cls, size):  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Class-level constructor for a shared-memory storage with room for
        # the given number of elements, wrapped in this legacy class.
        """Create a new storage in shared memory with the same data type."""
        untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
        return cls(wrap_storage=untyped_storage)

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Class-level passthrough that releases the CUDA IPC counter
        # via UntypedStorage.
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        # [FUNCTION DOC] HELPER: Class-level constructor that rebuilds a CPU storage shared by
        # file name from a manager handle, object handle and element count (converted to bytes),
        # wrapped in this legacy class.
        bytes_size = size * torch._utils._element_size(cls.dtype)
        return cls(
            wrap_storage=torch.UntypedStorage._new_shared_filename_cpu(
                manager, obj, bytes_size
            )
        )


def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    # [FUNCTION DOC] HELPER: Maps a pickled legacy storage type name back to a dtype. Raises
    # KeyError with a descriptive message if the name is not recognized.
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError as e:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized'
        ) from e
