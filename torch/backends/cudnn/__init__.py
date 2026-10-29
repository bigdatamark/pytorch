# mypy: allow-untyped-defs
# [PACKAGE DOC] Exposes torch.backends.cudnn: cuDNN (or MIOpen on ROCm) availability, version, tensor acceptability checks, and runtime flags (enabled, benchmark, benchmark_limit, deterministic, allow_tf32, fp32_precision, conv sub-setting, depthwise_kernel).
# Provides set_flags and a flags context manager, and imports the rnn subpackage module.
# Import-time side effects: tries to import the compiled cudnn extension (None if absent); replaces this module in the module registry with a property-intercepting wrapper. Setting enabled to False disables cuDNN/MIOpen globally.
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Optional

import torch
from torch.backends import (
    __allow_nonbracketed_mutation,
    _FP32Precision,
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    ContextProp,
    PropModule,
)

from . import rnn


try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None  # type: ignore[assignment]

# Write:
#
#   torch.backends.cudnn.enabled = False
#
# to globally disable CuDNN/MIOpen

__cudnn_version: int | None = None

if _cudnn is not None:

    def _init():
        # [FUNCTION DOC] HELPER: Lazily initializes and caches the cuDNN version; returns True on success.
        # On first call checks that runtime and compile-time versions are compatible (major must match; minor must be equal for old versions or non-CUDA builds, otherwise runtime minor must be at least compile minor), raising RuntimeError with advice about library paths unless the environment variable PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK equals 1 (then it returns True immediately without caching further checks).
        # Also raises RuntimeError if cuDNN is version 9.11 or newer and any visible CUDA device has compute capability below 7.5 (non-ROCm only). Mutates the module-level version cache.
        # KNOWN ISSUE: with the skip variable set, the early return happens after the version cache is assigned, so the device capability check is skipped.
        global __cudnn_version
        if __cudnn_version is None:
            # pyrefly: ignore [missing-attribute]
            __cudnn_version = _cudnn.getVersionInt()
            # pyrefly: ignore [missing-attribute]
            runtime_version = _cudnn.getRuntimeVersion()
            # pyrefly: ignore [missing-attribute]
            compile_version = _cudnn.getCompileVersion()
            runtime_major, runtime_minor, _ = runtime_version
            compile_major, compile_minor, _ = compile_version
            # Different major versions are always incompatible
            # Starting with cuDNN 7, minor versions are backwards-compatible
            # Not sure about MIOpen (ROCm), so always do a strict check
            if runtime_major != compile_major:
                cudnn_compatible = False
            # pyrefly: ignore [missing-attribute]
            elif runtime_major < 7 or not _cudnn.is_cuda:
                cudnn_compatible = runtime_minor == compile_minor
            else:
                cudnn_compatible = runtime_minor >= compile_minor
            if not cudnn_compatible:
                if os.environ.get("PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK", "0") == "1":
                    return True
                base_error_msg = (
                    f"cuDNN version incompatibility: "
                    f"PyTorch was compiled  against {compile_version} "
                    f"but found runtime version {runtime_version}. "
                    f"PyTorch already comes bundled with cuDNN. "
                    f"One option to resolving this error is to ensure PyTorch "
                    f"can find the bundled cuDNN. "
                )

                if "LD_LIBRARY_PATH" in os.environ:
                    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if any(
                        substring in ld_library_path for substring in ["cuda", "cudnn"]
                    ):
                        raise RuntimeError(
                            f"{base_error_msg}"
                            f"Looks like your LD_LIBRARY_PATH contains incompatible version of cudnn. "
                            f"Please either remove it from the path or install cudnn {compile_version}"
                        )
                    else:
                        raise RuntimeError(
                            f"{base_error_msg}"
                            f"one possibility is that there is a "
                            f"conflicting cuDNN in LD_LIBRARY_PATH."
                        )
                else:
                    raise RuntimeError(base_error_msg)
            # Check if cuDNN version is compatible with available CUDA devices
            if torch.cuda.is_available() and not torch.version.hip:
                min_cc = min(
                    [
                        torch.cuda.get_device_capability(i)
                        for i in range(torch.cuda.device_count())
                    ]
                )
                if __cudnn_version >= 91100 and min_cc < (7, 5):
                    raise RuntimeError(
                        f"cuDNN version {__cudnn_version} is not compatible with devices with SM < 7.5. "
                        f"Please install a version of PyTorch with a compatible cuDNN version. "
                        f"https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix"
                    )

        return True

else:

    def _init():
        # [FUNCTION DOC] HELPER: Fallback used when the extension is missing; always returns False.
        return False


def version():
    # [FUNCTION DOC] Returns the cuDNN version integer, or None when cuDNN cannot be initialized.
    """Return the version of cuDNN."""
    if not _init():
        return None
    return __cudnn_version


CUDNN_TENSOR_DTYPES = {
    torch.half,
    torch.float,
    torch.double,
}


def is_available():
    # [FUNCTION DOC] Returns the compiled-in flag telling whether cuDNN support is present.
    r"""Return a bool indicating if CUDNN is currently available."""
    return torch._C._has_cudnn


def is_acceptable(tensor):
    # [FUNCTION DOC] Returns True if cuDNN may be used for the given tensor: cuDNN must be globally enabled, the tensor must be on a CUDA device with half, float or double dtype, support must be built in, and initialization must succeed.
    # Emits a warning (and returns False) when support is not built in or the library cannot be initialized; returns False silently for the other failing checks. May raise RuntimeError from initialization.
    if not torch._C._get_cudnn_enabled():
        return False
    if tensor.device.type != "cuda" or tensor.dtype not in CUDNN_TENSOR_DTYPES:
        return False
    if not is_available():
        warnings.warn(
            "PyTorch was compiled without cuDNN/MIOpen support. To use cuDNN/MIOpen, rebuild "
            "PyTorch making sure the library is visible to the build system.",
            stacklevel=2,
        )
        return False
    if not _init():
        warnings.warn(
            "cuDNN/MIOpen library not found. Check your {libpath}".format(
                libpath={"darwin": "DYLD_LIBRARY_PATH", "win32": "PATH"}.get(
                    sys.platform, "LD_LIBRARY_PATH"
                )
            ),
            stacklevel=2,
        )
        return False
    return True


def set_flags(
    _enabled=None,
    _benchmark=None,
    _benchmark_limit=None,
    _deterministic=None,
    _allow_tf32=None,
    _fp32_precision="none",
    _depthwise_kernel=None,
):
    # [FUNCTION DOC] HELPER: Applies the given cuDNN settings to global state and returns a tuple of the previous values (for restoring).
    # A None argument leaves that setting unchanged, except fp32 precision whose default is the string none, so calling with defaults resets it. The benchmark limit is only applied when cuDNN is built in, and its previous value is None otherwise.
    orig_flags = (
        torch._C._get_cudnn_enabled(),
        torch._C._get_cudnn_benchmark(),
        None if not is_available() else torch._C._cuda_get_cudnn_benchmark_limit(),
        torch._C._get_cudnn_deterministic(),
        torch._C._get_cudnn_allow_tf32(),
        torch._C._get_fp32_precision_getter("cuda", "all"),
        torch._C._get_cudnn_depthwise_kernel(),
    )
    if _enabled is not None:
        torch._C._set_cudnn_enabled(_enabled)
    if _benchmark is not None:
        torch._C._set_cudnn_benchmark(_benchmark)
    if _benchmark_limit is not None and is_available():
        torch._C._cuda_set_cudnn_benchmark_limit(_benchmark_limit)
    if _deterministic is not None:
        torch._C._set_cudnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_cudnn_allow_tf32(_allow_tf32)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("cuda", "all", _fp32_precision)
    if _depthwise_kernel is not None:
        torch._C._set_cudnn_depthwise_kernel(_depthwise_kernel)
    return orig_flags


@contextmanager
def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
    fp32_precision="none",
    depthwise_kernel="auto",
):
    # [FUNCTION DOC] Context manager that sets cuDNN enabled (default False), benchmark (False), benchmark_limit (10), deterministic (False), allow_tf32 (True), fp32_precision (none) and depthwise_kernel (auto) for the body, then restores the previous values on exit even on error.
    # Note the defaults override current global settings when arguments are omitted.
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            enabled,
            benchmark,
            benchmark_limit,
            deterministic,
            allow_tf32,
            fp32_precision,
            depthwise_kernel,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<cudnn|mkldnn>.enabled = True


class CudnnModule(PropModule):
    # [CLASS DOC] Module wrapper installed in place of this module; exposes enabled, deterministic, benchmark, benchmark_limit (only if cuDNN is built in, else None), allow_tf32, fp32_precision, conv and depthwise_kernel as context-aware properties over C++ getters/setters.
    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    deterministic = ContextProp(
        torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic
    )
    benchmark = ContextProp(
        torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark
    )
    benchmark_limit = None
    if is_available():
        benchmark_limit = ContextProp(
            torch._C._cuda_get_cudnn_benchmark_limit,
            torch._C._cuda_set_cudnn_benchmark_limit,
        )
    allow_tf32 = ContextProp(
        torch._C._get_cudnn_allow_tf32, torch._C._set_cudnn_allow_tf32
    )
    conv = _FP32Precision("cuda", "conv")
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("cuda", "all"),
        _set_fp32_precision_setter("cuda", "all"),
    )
    depthwise_kernel = ContextProp(
        torch._C._get_cudnn_depthwise_kernel,
        torch._C._set_cudnn_depthwise_kernel,
    )


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
enabled: bool
deterministic: bool
benchmark: bool
allow_tf32: bool
fp32_precision: str
benchmark_limit: int
depthwise_kernel: str
