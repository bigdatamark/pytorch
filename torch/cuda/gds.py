# [MODULE DOC] GPUDirect Storage (GDS) bindings for torch.cuda: buffer registration helpers and the
# GdsFile wrapper around cuFile for direct storage<->GPU transfers.
# Import-time side effect: if GDS support is not built in, installs dummy stub functions into the
# torch._C namespace that raise a RuntimeError when called; raises an error if a partial set of
# real GDS functions is found in that case.
import os
import sys
from collections.abc import Callable

import torch
from torch.types import Storage


__all__: list[str] = [
    "is_available",
    "gds_register_buffer",
    "gds_deregister_buffer",
    "GdsFile",
]


def _dummy_fn(name: str) -> Callable:
    # [FUNCTION DOC] HELPER: Build a stub callable for the named torch._C function that accepts any
    # arguments and always raises a RuntimeError saying the function is unsupported on this platform.
    def fn(*args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Stub body; ignores arguments and raises RuntimeError.
        raise RuntimeError(f"torch._C.{name} is not supported on this platform")

    return fn


def is_available() -> bool:
    # [FUNCTION DOC] Return the boolean flag recording whether PyTorch was built with GDS support
    # (cuFile on CUDA or hipFile on ROCm). No side effects.
    r"""Return ``True`` if GDS (GPUDirect Storage) support is built in.

    This requires PyTorch to be built with cuFile (CUDA) or hipFile (ROCm).
    """
    return torch._C._has_gds


# Without GDS built in, install dummy stubs so the API raises a clear error.
if not is_available():
    if hasattr(torch._C, "_gds_deregister_buffer"):
        raise AssertionError(
            "_gds_deregister_buffer exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_register_handle"):
        raise AssertionError(
            "_gds_register_handle exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_deregister_handle"):
        raise AssertionError(
            "_gds_deregister_handle exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_load_storage"):
        raise AssertionError(
            "_gds_load_storage exists but _gds_register_buffer does not"
        )
    if hasattr(torch._C, "_gds_save_storage"):
        raise AssertionError(
            "_gds_save_storage exists but _gds_register_buffer does not"
        )
    # Define functions
    torch._C.__dict__["_gds_register_buffer"] = _dummy_fn("_gds_register_buffer")
    torch._C.__dict__["_gds_deregister_buffer"] = _dummy_fn("_gds_deregister_buffer")
    torch._C.__dict__["_gds_register_handle"] = _dummy_fn("_gds_register_handle")
    torch._C.__dict__["_gds_deregister_handle"] = _dummy_fn("_gds_deregister_handle")
    torch._C.__dict__["_gds_load_storage"] = _dummy_fn("_gds_load_storage")
    torch._C.__dict__["_gds_save_storage"] = _dummy_fn("_gds_save_storage")


def gds_register_buffer(s: Storage) -> None:
    # [FUNCTION DOC] Register the given CUDA storage as a cuFile buffer by delegating to the native
    # binding. Returns nothing; the native call (or the unsupported-platform stub) raises on failure.
    """Registers a storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_register_buffer(s)


def gds_deregister_buffer(s: Storage) -> None:
    # [FUNCTION DOC] Deregister a previously registered cuFile buffer storage by delegating to the
    # native binding. Returns nothing; errors come from the native call or the unsupported stub.
    """Deregisters a previously registered storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)
        >>> gds_deregister_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_deregister_buffer(s)


class GdsFile:
    # [CLASS DOC] File-like wrapper over a cuFile handle for GPUDirect Storage reads/writes of storages.
    # State: filename, flags, fd (OS file descriptor opened with direct I/O added to the flags) and
    # handle (the registered cuFile handle, or None when not registered). The handle is registered at
    # construction and deregistered/closed on garbage collection. Key methods: register_handle,
    # deregister_handle, load_storage, save_storage.
    r"""Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    See the `cufile docs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
    for more details.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open`` when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src1 = torch.randn(1024, device="cuda")
        >>> src2 = torch.randn(2, 1024, device="cuda")
        >>> file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
        >>> file.save_storage(src1.untyped_storage(), offset=0)
        >>> file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
        >>> dest1 = torch.empty(1024, device="cuda")
        >>> dest2 = torch.empty(2, 1024, device="cuda")
        >>> file.load_storage(dest1.untyped_storage(), offset=0)
        >>> file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        >>> torch.equal(src1, dest1)
        True
        >>> torch.equal(src2, dest2)
        True

    """

    def __init__(self, filename: str, flags: int):
        # [FUNCTION DOC] Open the file with the given flags plus direct I/O and register it with cuFile.
        # Args: filename, flags (os.open flags). Raises RuntimeError on Windows; OS errors from opening.
        # Side effects: opens a file descriptor and registers a cuFile handle.
        if sys.platform == "win32":
            raise RuntimeError("GdsFile is not supported on this platform.")
        self.filename = filename
        self.flags = flags
        self.fd = os.open(filename, flags | os.O_DIRECT)  # type: ignore[attr-defined]
        self.handle: int | None = None
        self.register_handle()

    def __del__(self) -> None:
        # [FUNCTION DOC] Finalizer: deregister the cuFile handle if one is registered, then close the
        # file descriptor.
        if self.handle is not None:
            self.deregister_handle()
        os.close(self.fd)

    def register_handle(self) -> None:
        # [FUNCTION DOC] Register the file descriptor with the cuFile driver and store the resulting
        # handle. Raises an AssertionError if a handle is already registered.
        """Registers file descriptor to cuFile Driver.

        This is a wrapper around ``cuFileHandleRegister``.
        """
        if self.handle is not None:
            raise AssertionError("Cannot register a handle that is already registered.")
        self.handle = torch._C._gds_register_handle(self.fd)

    def deregister_handle(self) -> None:
        # [FUNCTION DOC] Deregister the stored cuFile handle and reset it to None. Raises an
        # AssertionError if no handle is registered.
        """Deregisters file descriptor from cuFile Driver.

        This is a wrapper around ``cuFileHandleDeregister``.
        """
        if self.handle is None:
            raise AssertionError("Cannot deregister a handle that is not registered.")
        torch._C._gds_deregister_handle(self.handle)
        self.handle = None

    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        # [FUNCTION DOC] Read the storage's full byte size from the file at the given offset into the
        # storage via cuFile. Raises an AssertionError if the handle is not registered. Mutates storage.
        """Loads data from the file into the storage.

        This is a wrapper around ``cuFileRead``. ``storage.nbytes()`` of data
        will be loaded from the file at ``offset`` into the storage.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot load data from a file that is not registered.")
        torch._C._gds_load_storage(self.handle, storage, offset)

    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        # [FUNCTION DOC] Write all bytes of the storage to the file at the given offset via cuFile.
        # Raises an AssertionError if the handle is not registered. Mutates the file.
        """Saves data from the storage into the file.

        This is a wrapper around ``cuFileWrite``. All bytes of the storage
        will be written to the file at ``offset``.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot save data to a file that is not registered.")
        torch._C._gds_save_storage(self.handle, storage, offset)
