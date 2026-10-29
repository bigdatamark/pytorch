# [PACKAGE DOC] torch_openreg: reference out-of-tree PrivateUse1 backend named "openreg".
# Importing it (directly or via the autoload entry point) loads Windows DLLs when needed,
# imports the native _C module, the compiler backend and the openreg device module, renames
# the PrivateUse1 backend to "openreg", registers the device module with torch, generates
# tensor/storage methods for the backend, and registers the "occl" distributed backend when
# torch.distributed is available. Exports the _autoload entry-point placeholder.
import sys

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import torch_openreg._C  # type: ignore[misc]
import torch_openreg.compiler
import torch_openreg.openreg


torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", torch_openreg.openreg)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

# LITERALINCLUDE START: OCCL BACKEND REGISTRATION
if torch.distributed.is_available():
    try:
        from torch_openreg._C import ProcessGroupOCCL

        def _create_occl_backend(store, rank, size, timeout):
            # [FUNCTION DOC] HELPER: Backend factory for "occl": builds a native
            # ProcessGroupOCCL from the rendezvous store, rank, world size and timeout.
            return ProcessGroupOCCL(store, rank, size, timeout)

        torch.distributed.Backend.register_backend(
            "occl", _create_occl_backend, devices=["openreg"]
        )
    except Exception as e:
        raise RuntimeError("Failed to register 'occl' process group backend.") from e
# LITERALINCLUDE END: OCCL BACKEND REGISTRATION


# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # [FUNCTION DOC] HELPER: Placeholder entry-point target for backend autoloading; the
    # real work happens at import time of this package. Does nothing.
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD
