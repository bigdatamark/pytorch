# mypy: allow-untyped-defs
# [MODULE DOC] Main entry of the torch.cuda package: lazy CUDA initialization, device/stream management, memory,
# RNG, properties, capability checks and re-exports from submodules (graphs, streams, green contexts, gds, memory,
# random, profiler, etc). Import-time side effects: probes optional pynvml/amdsmi (on ROCm temporarily hooks
# ctypes CDLL to avoid loading two copies of libamd_smi), records import errors in module globals, defines
# fallback stubs when built without CUDA, creates the thread-local state, init lock and queued-call list,
# and queues the compute-capability and cubin compatibility checks to run at CUDA initialization.
r"""
This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

import glob
import importlib
import importlib.util
import os
import platform
import threading
import traceback
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast, NewType, Optional, TYPE_CHECKING

import torch
import torch._C
from torch._utils import _dummy_type, _LazySeedTracker, classproperty
from torch.types import Device

from . import _device_limits, gds
from ._utils import _get_device_index
from .graphs import (
    CUDAGraph,
    export_dot,
    export_graph_data,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)
from .green_contexts import GreenContext
from .streams import Event, ExternalStream, Stream


try:
    from torch._C import _cudart  # type: ignore[attr-defined]
except ImportError:
    _cudart = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)

_HAS_PYNVML = False
_HAS_AMDSMI = False
_PYNVML_ERR = None
_AMDSMI_ERR = None
try:
    from torch import version as _version

    try:
        if not _version.hip:
            import pynvml  # type: ignore[import]
        else:
            import ctypes
            from pathlib import Path

            # In ROCm (at least up through 6.3.2) there're 2 copies of libamd_smi.so:
            # - One at lib/libamd_smi.so
            # - One at share/amd_smi/amdsmi/libamd_smi.so
            #
            # The amdsmi python module hardcodes loading the second one in share-
            # https://github.com/ROCm/amdsmi/blob/1d305dc9708e87080f64f668402887794cd46584/py-interface/amdsmi_wrapper.py#L174
            #
            # See also https://github.com/ROCm/amdsmi/issues/72.
            #
            # This creates an ODR violation if the copy of libamd_smi.so from lib
            # is also loaded (via `ld` linking, `LD_LIBRARY_PATH` or `rpath`).
            #
            # In order to avoid the violation we hook CDLL and try using the
            # already loaded version of amdsmi, or any version in the processes
            # rpath/LD_LIBRARY_PATH first, so that we only load a single copy
            # of the .so.
            class _amdsmi_cdll_hook:
            # [CLASS DOC] HELPER: Context manager (ROCm only) that temporarily replaces ctypes.CDLL so that any request to load
            # libamd_smi is first tried against ROCM_HOME/ROCM_PATH, the loader path and rocm sdk core wheel locations, avoiding
            # loading a second copy of the library. State: original_CDLL and the candidate path list. Restores CDLL on exit.
                def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Save the original CDLL and build the candidate path list (ROCM home copy first when set,
                # then the bare name, then on Linux the versioned files found in the rocm sdk core wheel).
                    self.original_CDLL = ctypes.CDLL  # type: ignore[misc,assignment]
                    paths = ["libamd_smi.so"]
                    if rocm_home := os.getenv("ROCM_HOME", os.getenv("ROCM_PATH")):
                        paths = [os.path.join(rocm_home, "lib/libamd_smi.so")] + paths
                    # TheRock ROCm wheels install the amdsmi python package in
                    # site-packages/amdsmi and ship the native library under the
                    # _rocm_sdk_core wheel (site-packages/_rocm_sdk_core/lib)
                    # rather than on the loader path or under $ROCM_HOME. The
                    # file there is a versioned soname (e.g. libamd_smi.so.26),
                    # so amdsmi's bare CDLL("libamd_smi.so") can't find it. Locate
                    # the package and append the concrete versioned files as a
                    # last-resort fallback so the hook can redirect to them.
                    if platform.system() == "Linux":
                        paths = paths + self._rocm_sdk_core_amdsmi_paths()
                    self.paths: list[str] = paths

                @staticmethod
                def _rocm_sdk_core_amdsmi_paths() -> list[str]:
                # [FUNCTION DOC] HELPER: Static. Locate the _rocm_sdk_core package and return a sorted list of its libamd_smi shared
                # library files; returns an empty list if the package is missing or has no search locations.
                    try:
                        spec = importlib.util.find_spec("_rocm_sdk_core")
                    except (ImportError, ValueError):
                        return []
                    if spec is None or not spec.submodule_search_locations:
                        return []
                    found: list[str] = []
                    for location in spec.submodule_search_locations:
                        found += glob.glob(
                            os.path.join(location, "lib", "libamd_smi.so*")
                        )
                    return sorted(found)

                def hooked_CDLL(
                    self, name: str | Path | None, *args: Any, **kwargs: Any
                ) -> ctypes.CDLL:
                # [FUNCTION DOC] HELPER: Replacement for CDLL. For library names starting with libamd_smi.so, try each candidate path
                # and return the first that loads (ignoring OS errors); otherwise fall back to the original loader with the given name.
                    if name and Path(name).name.startswith("libamd_smi.so"):
                        for path in self.paths:
                            try:
                                return self.original_CDLL(path, *args, **kwargs)
                            except OSError:
                                pass
                    return self.original_CDLL(name, *args, **kwargs)  # type: ignore[arg-type]

                def __enter__(self) -> None:
                # [FUNCTION DOC] HELPER: Install the hooked loader as ctypes.CDLL (global side effect).
                    ctypes.CDLL = self.hooked_CDLL  # type: ignore[misc,assignment]

                def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
                # [FUNCTION DOC] HELPER: Restore the original ctypes.CDLL; does not suppress exceptions.
                    ctypes.CDLL = self.original_CDLL  # type: ignore[misc]

            try:
                with _amdsmi_cdll_hook():
                    import amdsmi  # type: ignore[import]
                _HAS_AMDSMI = True
            except ModuleNotFoundError as err:
                _AMDSMI_ERR = err
                raise
            except (KeyError, OSError) as err:
                # The amdsmi python package is installed but its native library
                # (libamd_smi.so) could not be discovered/loaded -- e.g. TheRock
                # ROCm wheels lay amdsmi out so that its own find_smi_library()
                # misses the versioned libamd_smi.so.* and raises KeyError. Treat
                # this like a missing optional dependency (degrade to
                # _HAS_AMDSMI=False) instead of aborting `import torch`.
                _AMDSMI_ERR = err
                raise ModuleNotFoundError(
                    "amdsmi is installed but libamd_smi.so could not be loaded"
                ) from err

        _HAS_PYNVML = True
    except ModuleNotFoundError:
        pass
    finally:
        del _version
except ImportError as err:
    _PYNVML_ERR = err  # sometimes a lib is installed but the import fails for some other reason, so we log the error for later

_lazy_seed_tracker = _LazySeedTracker()

# Define dummy _CudaDeviceProperties type if PyTorch was compiled without CUDA
if hasattr(torch._C, "_CudaDeviceProperties"):
    _CudaDeviceProperties = torch._C._CudaDeviceProperties
else:
    _CudaDeviceProperties = _dummy_type("_CudaDeviceProperties")  # type: ignore[assignment, misc]

if hasattr(torch._C, "_cuda_exchangeDevice"):
    _exchange_device = torch._C._cuda_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
    # [FUNCTION DOC] HELPER: Fallback when built without CUDA: returns -1 for a negative device, otherwise raises RuntimeError.
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")


if hasattr(torch._C, "_cuda_maybeExchangeDevice"):
    _maybe_exchange_device = torch._C._cuda_maybeExchangeDevice
else:

    def _maybe_exchange_device(device: int) -> int:
    # [FUNCTION DOC] HELPER: Fallback when built without CUDA: returns -1 for a negative device, otherwise raises RuntimeError.
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without CUDA support")


has_half: bool = True
has_magma: bool = torch._C._has_magma

default_generators: tuple[torch._C.Generator] = ()  # type: ignore[assignment]


def _is_compiled() -> bool:
    # [FUNCTION DOC] HELPER: True if the build contains CUDA bindings (detected by the presence of the device count binding).
    r"""Return true if compile with CUDA support."""
    return hasattr(torch._C, "_cuda_getDeviceCount")


def _nvml_based_avail() -> bool:
    # [FUNCTION DOC] HELPER: True if the environment variable requesting NVML based availability checks is set to 1.
    return os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"


def is_available() -> bool:
    # [FUNCTION DOC] Return whether CUDA is usable: False if not compiled with CUDA; if the NVML check env var is set,
    # use device_count (NVML based, avoids poisoning fork); otherwise ask the CUDA runtime whether at least one device exists.
    r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return torch._C._cuda_getDeviceCount() > 0


def is_bf16_supported(including_emulation: bool = True):
    # [FUNCTION DOC] Return whether the current device supports bfloat16. Always True on ROCm, False when CUDA is unavailable,
    # True for compute capability major 8 or higher; otherwise, when including_emulation is set, probes by trying to create a
    # bfloat16 tensor on the current device.
    r"""Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16."""
    # Check for ROCm, if true return true, no ROCM_VERSION check required,
    # since it is supported on AMD GPU archs.
    if torch.version.hip:
        return True

    # If CUDA is not available, then it does not support bf16 either
    if not is_available():
        return False

    device = torch.cuda.current_device()

    if torch.cuda.get_device_properties(device).major >= 8:
        return True

    if not including_emulation:
        return False

    # Finally try to create a bfloat16 device.
    return _check_bf16_tensor_supported(device)


@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device: Device):
    # [FUNCTION DOC] HELPER: Cached (16 entries) probe that tries creating a bfloat16 tensor on the device; True on success, False on any exception.
    try:
        torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        return True
    except Exception:
        return False


def is_tf32_supported() -> bool:
    # [FUNCTION DOC] Return whether the current device supports tf32. On ROCm true only for gfx94 or gfx95 architectures;
    # otherwise equals native (non-emulated) bfloat16 support.
    r"""Return a bool indicating if the current CUDA/ROCm device supports dtype tf32."""
    if torch.version.hip:
        prop_name = torch.cuda.get_device_properties().gcnArchName
        archs = ("gfx94", "gfx95")
        for arch in archs:
            if arch in prop_name:
                return True
        return False

    # Otherwise, tf32 is supported on CUDA platforms that natively (i.e. no emulation)
    # support bfloat16.
    return is_bf16_supported(including_emulation=False)


def _sleep(cycles):
    # [FUNCTION DOC] HELPER: Launch a kernel that spins the GPU for the given number of cycles (testing aid).
    torch._C._cuda_sleep(cycles)


def _extract_arch_version(arch_string: str) -> int:
    # [FUNCTION DOC] HELPER: Parse an architecture string like sm_90a (second underscore-separated token, with a trailing a or f removed) into an integer.
    """Extracts the architecture string from a CUDA version"""
    base = arch_string.split("_", maxsplit=2)[1]
    base = base.removesuffix("a").removesuffix("f")
    return int(base)


class _CompatInterval:
    # [CLASS DOC] HELPER: Membership test for a range of compute capabilities (integers like 90): all versions with the same major
    # as the start and a minor at or above the start's minor, minus an optional exclusion set. Provides printing of the range.
    """
    Defines a range of compute capabilities starting at a given
    version and going up to the end of that major version. This
    also allows excluding specific versions from the range.
    """

    def __init__(self, start, exclude: set[int] | None = None):
        # [FUNCTION DOC] HELPER: Split start into major and minor digits and store the exclusion set (empty by default).
        self.major, self.minor = start // 10, start % 10
        self.exclude = set() if exclude is None else exclude

    def __contains__(self, x):
        # [FUNCTION DOC] HELPER: True if x is not excluded, has the same major and a minor at or above the start minor.
        if x in self.exclude:
            return False
        x_major, x_minor = x // 10, x % 10
        return x_major == self.major and x_minor >= self.minor

    def __str__(self):
        # [FUNCTION DOC] HELPER: Human readable range text, with excluded versions appended.
        result = f">={self.major}.{self.minor},<{self.major + 1}.0"
        if len(self.exclude) > 0:
            exceptions = ", ".join(f"{x // 10}.{x % 10}" for x in self.exclude)
            result += f" except {{{exceptions}}}"
        return result


class _CompatSet:
    # [CLASS DOC] HELPER: Plain set of compute capabilities with membership test and custom printing as major.minor values.
    """
    A set of compute capabilities. It exists primarily to support custom
    printing logic and is otherwise equivalent to a plain python set().
    """

    def __init__(self, values: set[int]):
        # [FUNCTION DOC] HELPER: Store the set of integer capabilities.
        self.values = values

    def __contains__(self, x):
        # [FUNCTION DOC] HELPER: Membership test against the stored set.
        return x in self.values

    def __str__(self):
        # [FUNCTION DOC] HELPER: Format members as major.minor values in braces (set iteration order).
        return "{" + ", ".join(f"{v // 10}.{v % 10}" for v in self.values) + "}"


# (code SM)->(device SM required to execute the code)
#
# Developer Notes:
# - This dict should be kept up to date with keys corresponding
#   to SM versions that PyTorch can be built for. An out of date
#   mapping will lead to false warnings.
# - The keys in dict correspond to known sm versions but the values
#   are merely rules based on sm compatibility guarantees for NVIDIA
#   devices while accounting for incompatibility of iGPU and dGPU.
DEVICE_REQUIREMENT: dict[int, _CompatSet | _CompatInterval] = {
    50: _CompatInterval(start=50, exclude={53}),
    52: _CompatInterval(start=52, exclude={53}),
    53: _CompatSet({53}),
    60: _CompatInterval(start=60, exclude={62}),
    61: _CompatInterval(start=61, exclude={62}),
    62: _CompatSet({62}),
    70: _CompatInterval(start=70, exclude={72}),
    72: _CompatSet({72}),
    75: _CompatInterval(start=75),
    80: _CompatInterval(start=80, exclude={87}),
    86: _CompatInterval(start=86, exclude={87}),
    87: _CompatSet({87}),
    89: _CompatInterval(start=89),
    90: _CompatInterval(start=90),
    100: _CompatInterval(start=100, exclude={101}),
    101: _CompatSet({101, 110}),  # 101 was renamed to 110
    103: _CompatInterval(start=103),
    110: _CompatSet({101, 110}),  # 101 was renamed to 110
    120: _CompatInterval(start=120),
    121: _CompatInterval(start=121),
}

# CUDA 13.2 allows SBSA binaries to run on Jetson devices.
# This dict can be combined with DEVICE_REQUIREMENT once
# the minimum supported CUDA version is 13.2.
DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION: dict[
    int, _CompatSet | _CompatInterval
] = DEVICE_REQUIREMENT | {
    70: _CompatInterval(start=70),
    80: _CompatInterval(start=80),
    86: _CompatInterval(start=86),
}

# TORCH_CUDA_ARCH_LIST for PyTorch releases, keyed by host arch.
# Kept in sync with .ci/manywheel/build_cuda.sh by the validator in
# .github/scripts/generate_binary_build_matrix.py.
PYTORCH_RELEASES_CODE_CC: dict[str, dict[str, set[int]]] = {
    "12.6": {
        "x86_64": {50, 60, 70, 75, 80, 86, 90},
        "aarch64": {80, 90},
    },
    "13.0": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
    "13.2": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
    "13.4": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
}


def _device_requirement(code_cc):
    # [FUNCTION DOC] HELPER: Return the device capability rule (interval or set) able to run code built for code_cc, choosing the post-Jetson-SBSA
    # table for CUDA 13.2 or later; None if torch is not a CUDA build or the capability is unknown.
    if torch.version.cuda is None:
        return None
    requirement = (
        DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION
        if tuple(int(x) for x in torch.version.cuda.split(".")) >= (13, 2)
        else DEVICE_REQUIREMENT
    )
    return requirement.get(code_cc, None)


def _host_arch_key() -> str:
    # [FUNCTION DOC] HELPER: Return aarch64 if the host machine is aarch64, otherwise x86_64.
    machine = platform.machine().lower()
    return "aarch64" if machine == "aarch64" else "x86_64"


def _code_compatible_with_device(device_cc: int, code_cc: int):
    # [FUNCTION DOC] HELPER: True if code built for code_cc can run on a device of capability device_cc. Warns (and falls back to a
    # same-major, minor-or-higher rule) when code_cc is not a known capability.
    compatible_devices = _device_requirement(code_cc)
    if compatible_devices is None:
        warnings.warn(
            f"PyTorch was compiled with an unknown compute capability {code_cc // 10}.{code_cc % 10}. "
            + " Please create an issue on Github if this is a valid compute capability.",
            stacklevel=2,
        )
        return device_cc in _CompatInterval(start=code_cc)
    return device_cc in compatible_devices


def _warn_unsupported_code(device_index: int, device_cc: int, code_ccs: list[int]):
    # [FUNCTION DOC] HELPER: Emit a warning that the GPU is not supported by this build: lists the build capabilities and the hardware
    # each supports, then suggests pip install commands for published CUDA builds (nightly index for dev versions) that support
    # the device, or points to the install page if none does. Side effect: calls warnings.warn.
    name = get_device_name(device_index)

    arch = _host_arch_key()
    compatible_releases: list[str] = []
    for cuda, by_arch in PYTORCH_RELEASES_CODE_CC.items():
        build_ccs = by_arch.get(arch, set())
        if any(_code_compatible_with_device(device_cc, cc) for cc in build_ccs):
            compatible_releases.append(cuda)

    lines = [
        f"Found GPU{device_index} {name} which is of compute capability (CC) {device_cc // 10}.{device_cc % 10}.",
        "The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:",
    ] + [
        f"- {cc // 10}.{cc % 10} which supports hardware CC {_device_requirement(cc)}"
        for cc in code_ccs
    ]

    if len(compatible_releases) > 0:
        version = torch.__version__
        base_version = version.split("+")[0]
        is_nightly = "dev" in base_version
        index_root = (
            "https://download.pytorch.org/whl/nightly"
            if is_nightly
            else "https://download.pytorch.org/whl"
        )
        lines.append(
            f"Your installed torch=={version} does not include kernels for this GPU. "
            "Reinstall the same version against a CUDA build that does, e.g.:"
        )
        for cuda in compatible_releases:
            cu_tag = "cu" + cuda.replace(".", "")
            lines.append(
                f"  For CUDA {cuda} use pip install torch=={base_version} --index-url {index_root}/{cu_tag}"
            )
    else:
        lines.append(
            f"No published PyTorch CUDA builds for release {torch.__version__} support this GPU. "
            "Visit https://pytorch.org/get-started/locally/ to find a compatible release."
        )

    warnings.warn("\n".join(lines), stacklevel=2)


def _check_capability():
    # [FUNCTION DOC] HELPER: Lazy startup check (CUDA builds only): for each visible device, warn if no compiled architecture
    # in this build is compatible. Does nothing on ROCm or when the arch list is empty.
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return

    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return

    code_ccs = [_extract_arch_version(cc) for cc in arch_list]
    for d in range(device_count()):
        major, minor = get_device_capability(d)
        device_cc = 10 * major + minor
        if not any(
            _code_compatible_with_device(device_cc, code_cc) for code_cc in code_ccs
        ):
            _warn_unsupported_code(d, device_cc, code_ccs)


def _check_cubins():
    # [FUNCTION DOC] HELPER: Lazy startup check (CUDA builds only): warn for each device whose compute major version matches no
    # compiled sm architecture. Does nothing on ROCm or an empty arch list.
    # KNOWN ISSUE: overlaps with the newer capability check and uses only major version matching.
    incompatible_device_warn = """
{} with CUDA capability sm_{} is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities {}.
If you want to use the {} GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
"""
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return
    supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        # NVIDIA GPU compute architectures are backward compatible within major version
        supported = any(sm // 10 == cap_major for sm in supported_sm)
        if not supported:
            device_name = get_device_name(idx)
            capability = cap_major * 10 + cap_minor
            warnings.warn(
                incompatible_device_warn.format(
                    device_name, capability, " ".join(arch_list), device_name
                ),
                stacklevel=2,
            )


def is_initialized():
    # [FUNCTION DOC] Return True if CUDA state is initialized and the process is not in a bad forked child.
    r"""Return whether PyTorch's CUDA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    # [FUNCTION DOC] HELPER: Run callable now if CUDA is initialized; otherwise queue it under the init lock. Keyword seed_all or seed
    # route it to the lazy seed tracker; other calls are appended to the queued calls list with the stack trace text.
    with _initialization_lock:
        if is_initialized():
            callable()
        else:
            # TODO(torch_deploy): this accesses linecache, which attempts to read the
            # file system to get traceback info. Patch linecache or do something
            # else here if this ends up being important.
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                # Don't store the actual traceback to avoid memory cycle
                _queued_calls.append((callable, traceback.format_stack()))


_lazy_call(_check_capability)
_lazy_call(_check_cubins)


class DeferredCudaCallError(Exception):
# [CLASS DOC] Exception raised when a call queued before CUDA initialization fails once initialization runs (carries the original stack).
    pass


AcceleratorError = torch._C.AcceleratorError
OutOfMemoryError = torch._C.OutOfMemoryError


def init():
    # [FUNCTION DOC] Initialize CUDA state explicitly by calling the lazy initializer; no-op if already initialized.
    r"""Initialize PyTorch's CUDA state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for CUDA functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    # [FUNCTION DOC] HELPER: Lazily initialize CUDA exactly once (double-checked locking). Returns immediately if initialized
    # or if the current thread is already mid-initialization. Raises RuntimeError in a forked child, AssertionError if torch lacks
    # CUDA or the cudart module. Calls the native init, then runs all queued calls (plus seed calls from the seed tracker),
    # wrapping any failure in DeferredCudaCallError that includes the original call stack. Sets the initialized flag last.
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize CUDA in forked subprocess. To use CUDA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            raise AssertionError("Torch not compiled with CUDA enabled")
        if _cudart is None:
            raise AssertionError(
                "libcudart functions unavailable. It looks like you have a broken build?"
            )
        # This function throws if there's a driver initialization error, no GPUs
        # are found or any other error occurs
        torch._C._cuda_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"CUDA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredCudaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def cudart():
    # [FUNCTION DOC] Initialize CUDA if needed and return the native CUDA runtime module (None-valued if unavailable).
    # Raises the same errors as lazy initialization.
    r"""Retrieves the CUDA runtime API module.


    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart). The CUDA
    runtime API module provides access to various CUDA runtime functions.

    Args:
        ``None``

    Returns:
        module: The CUDA runtime API module (_cudart).

    Raises:
        RuntimeError: If CUDA cannot be re-initialized in a forked subprocess.
        AssertionError: If PyTorch is not compiled with CUDA support or if libcudart functions are unavailable.

    Example of CUDA operations with profiling:
        >>> import torch
        >>> from torch.cuda import cudart, check_error
        >>> import os
        >>>
        >>> os.environ["CUDA_PROFILE"] = "1"
        >>>
        >>> def perform_cuda_operations_with_streams():
        >>>     stream = torch.cuda.Stream()
        >>>     with torch.cuda.stream(stream):
        >>>         x = torch.randn(100, 100, device='cuda')
        >>>         y = torch.randn(100, 100, device='cuda')
        >>>         z = torch.mul(x, y)
        >>>     return z
        >>>
        >>> torch.cuda.synchronize()
        >>> print("====== Start nsys profiling ======")
        >>> check_error(cudart().cudaProfilerStart())
        >>> with torch.autograd.profiler.emit_nvtx():
        >>>     result = perform_cuda_operations_with_streams()
        >>>     print("CUDA operations completed.")
        >>> check_error(torch.cuda.cudart().cudaProfilerStop())
        >>> print("====== End nsys profiling ======")

    To run this example and save the profiling information, execute:
        >>> $ nvprof --profile-from-start off --csv --print-summary -o trace_name.prof -f -- python cudart_test.py

    This command profiles the CUDA operations in the provided script and saves
    the profiling information to a file named `trace_name.prof`.
    The `--profile-from-start off` option ensures that profiling starts only
    after the `cudaProfilerStart` call in the script.
    The `--csv` and `--print-summary` options format the profiling output as a
    CSV file and print a summary, respectively.
    The `-o` option specifies the output file name, and the `-f` option forces the
    overwrite of the output file if it already exists.
    """
    _lazy_init()
    return _cudart


class cudaStatus:
    # [CLASS DOC] HELPER: Constants for CUDA runtime status codes: SUCCESS is 0 and ERROR_NOT_READY is 34.
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34


class CudaError(RuntimeError):
    # [CLASS DOC] RuntimeError subclass for CUDA runtime failures; message holds the runtime error string and numeric code.
    def __init__(self, code: int) -> None:
        # [FUNCTION DOC] Build the message from the runtime error string for code plus the code in parentheses.
        # pyrefly: ignore [missing-attribute]
        msg = _cudart.cudaGetErrorString(_cudart.cudaError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    # [FUNCTION DOC] Raise CudaError if res differs from the runtime success code; otherwise return nothing.
    r"""Raise an error if the result of a CUDA runtime API call is not success."""
    # pyrefly: ignore [missing-attribute]
    if res != _cudart.cudaError.success:
        raise CudaError(res)


class _DeviceGuard:
    # [CLASS DOC] HELPER: Lightweight device context manager without index normalization: on enter switches the current device
    # to idx (saving the previous one), on exit switches back. State: idx and prev_idx.
    def __init__(self, index: int):
        # [FUNCTION DOC] HELPER: Store the device index and initialize the previous index to -1.
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Exchange the current device with idx and remember the previous device.
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # [FUNCTION DOC] HELPER: Restore the previous device; never suppresses exceptions (returns False).
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device:
    # [CLASS DOC] Context manager that selects a CUDA device for its body and restores the previous device on exit.
    # No-op when the index is negative or None. State: idx (normalized device index) and prev_idx.
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        # [FUNCTION DOC] Normalize the given device (int, string, device object or None) into an index; previous index starts at -1.
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        # [FUNCTION DOC] Switch to the selected device, recording the previously current device.
        self.prev_idx = torch.cuda._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # [FUNCTION DOC] Switch back to the recorded device; returns False so exceptions propagate.
        self.idx = torch.cuda._maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    # [CLASS DOC] Device context manager that selects the device holding a given tensor or storage; no-op if the object is not on a GPU.
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        # [FUNCTION DOC] Pick the object's device index if it is a CUDA object, else -1, and delegate to the base constructor.
        idx = obj.get_device() if obj.is_cuda else -1
        super().__init__(idx)


def set_device(device: Device) -> None:
    # [FUNCTION DOC] Set the current CUDA device (index, string or device object). Negative values are ignored.
    # Side effect: changes the thread's current device via the native call.
    r"""Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._cuda_setDevice(device)


def get_device_name(device: Device = None) -> str:
    # [FUNCTION DOC] Return the name string of the given device (current device if None) from its properties.
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Device = None) -> tuple[int, int]:
    # [FUNCTION DOC] Return the (major, minor) compute capability of the given device (current if None).
    r"""Get the cuda capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


# pyrefly: ignore [not-a-type]
def get_device_properties(device: Device = None) -> _CudaDeviceProperties:
    # [FUNCTION DOC] Return the properties object of a device (current if None). Initializes CUDA lazily.
    # Raises AssertionError for an out of range device index.
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            properties of the device.  It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)  # type: ignore[name-defined]


def can_device_access_peer(device: Device, peer_device: Device) -> bool:
    # [FUNCTION DOC] Return whether the first device can directly access memory of peer_device. Initializes CUDA lazily;
    # raises AssertionError for out of range indices.
    r"""Check if peer access between two devices is possible."""
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch._C._cuda_canDeviceAccessPeer(device, peer_device)


class StreamContext:
    # [CLASS DOC] Context manager that makes a stream current for kernels launched in its body and restores the previous
    # streams on exit; a no-op if the stream is None or no CUDA device exists. State: stream, idx (device index, -1 if none),
    # src_prev_stream and dst_prev_stream (streams to restore on the original and the stream's own device).
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.cuda.Stream"]

    def __init__(self, stream: Optional["torch.cuda.Stream"]):
        # [FUNCTION DOC] Store the stream and the current device index (-1 if unavailable outside scripting); initialize
        # the saved previous streams (default streams under scripting, otherwise None).
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                # pyrefly: ignore [bad-assignment]
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        )

    def __enter__(self):
        # [FUNCTION DOC] Save the current stream; if the target stream lives on another device also save that device's current
        # stream; then set the target stream current. Does nothing for a None stream or when no device exists.
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or CUDA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.cuda.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream.device)
        torch.cuda.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # [FUNCTION DOC] Restore the saved streams (destination device first when devices differ, then the original one).
        # Does nothing for a None stream or when no device exists. Does not suppress exceptions.
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.cuda.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.cuda.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.cuda.Stream"]) -> StreamContext:
    # [FUNCTION DOC] Return a StreamContext for the given stream (may be None, which makes the context a no-op).
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        In eager mode stream is of type Stream class while in JIT it is
        an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type):
    # [FUNCTION DOC] HELPER: Set the current stream through the native call using stream id, device index and device type enum.
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    torch._C._cuda_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream):
    # [FUNCTION DOC] Set the current stream from a Stream object (via its id, device index and device type); does nothing if None.
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def _parse_visible_devices() -> list[int] | list[str]:
    # [FUNCTION DOC] HELPER: Parse visible device environment variables into either a list of integer ordinals or a list of UUID strings.
    # On ROCm, combines ROCR and HIP variables (HIP preferred; raises RuntimeError if HIP lists more devices than ROCR).
    # Unset gives ordinals 0..63. UUID-style lists (GPU- or MIG- prefix) keep matching prefixed entries until the first non-prefixed;
    # integer lists are parsed like strtoul, a duplicate yields an empty list and a negative value ends parsing.
    r"""Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")

    if torch.version.hip:
        hip_devices = os.getenv("HIP_VISIBLE_DEVICES")
        rocr_devices = os.getenv("ROCR_VISIBLE_DEVICES")

        # You must take care if both HIP and ROCR env vars are set as they have
        # different meanings. Both env vars accept either a list of ints or a
        # list of UUIDs. The ROCR env var is processed first which then reduces
        # the number of GPUs that HIP can select from.
        if rocr_devices is not None:
            rocr_count = len(rocr_devices.split(","))
            if hip_devices is not None:
                # sanity check if both env vars are set
                if len(hip_devices.split(",")) > rocr_count:
                    raise RuntimeError(
                        "HIP_VISIBLE_DEVICES contains more devices than ROCR_VISIBLE_DEVICES"
                    )
                # HIP_VISIBLE_DEVICES is preferred over ROCR_VISIBLE_DEVICES
                var = hip_devices
            else:
                return list(range(rocr_count))
        elif hip_devices is not None:
            var = hip_devices

    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        # [FUNCTION DOC] HELPER: Parse the leading integer (optional sign plus digits) of a string like strtoul; -1 if empty or no digits.
        # KNOWN ISSUE: a lone sign character makes the final integer conversion raise ValueError instead of returning -1.
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
        # [FUNCTION DOC] HELPER: Split a comma list, collecting elements until one lacks the prefix; a repeated element yields an empty list.
        rcs: list[str] = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return cast(list[str], [])
            # Anything other but prefix is ignored
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    if var.startswith("GPU-"):
        return parse_list_with_prefix(var, "GPU-")
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: list[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # Repeated ordinal results in empty set
        if x in rc:
            return cast(list[int], [])
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc


def _raw_device_count_amdsmi() -> int:
    # [FUNCTION DOC] HELPER: Device count via amdsmi: -1 if amdsmi is unavailable or fails to initialize (with a warning),
    # otherwise the number of processor handles.
    if not _HAS_AMDSMI:
        return -1
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        warnings.warn(
            f"Can't initialize amdsmi - Error code: {e.err_code}", stacklevel=2
        )
        return -1
    socket_handles = amdsmi.amdsmi_get_processor_handles()
    return len(socket_handles)


def _raw_device_count_nvml() -> int:
    # [FUNCTION DOC] HELPER: Device count via the NVML shared library loaded with ctypes. Returns -1 with a warning if init or count fails.
    # May raise OSError if the NVML library cannot be loaded.
    r"""Return number of devices as reported by NVML or negative value if NVML discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return -1
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return -1
    del nvml_h
    return dev_count.value


def _raw_device_uuid_amdsmi() -> list[str] | None:
    # [FUNCTION DOC] HELPER: List of lower-cased device serial identifiers (without 0x prefix) from amdsmi. Returns None with a warning on
    # any amdsmi failure or when amdsmi is unavailable.
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    if not _HAS_AMDSMI:
        return None
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException:
        warnings.warn("Can't initialize amdsmi", stacklevel=2)
        return None
    try:
        socket_handles = amdsmi.amdsmi_get_processor_handles()
        dev_count = len(socket_handles)
    except amdsmi.AmdSmiException:
        warnings.warn("Can't get amdsmi device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count):
        try:
            handler = amdsmi.amdsmi_get_processor_handles()[idx]
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get amd device handler", stacklevel=2)
            return None
        try:
            uuid = amdsmi.amdsmi_get_gpu_asic_info(handler)["asic_serial"][
                2:
            ]  # Removes 0x prefix from serial
        except amdsmi.AmdSmiException:
            warnings.warn("Cannot get uuid for amd device", stacklevel=2)
            return None
        uuids.append(
            str(uuid).lower()
        )  # Lower-case to match expected HIP_VISIBLE_DEVICES uuid input
    return uuids


def _raw_device_uuid_nvml() -> list[str] | None:
    # [FUNCTION DOC] HELPER: List of device UUID strings from NVML via ctypes; returns None with a warning if init, count, handle or UUID queries fail.
    r"""Return list of device UUID as reported by NVML or None if NVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return None
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle", stacklevel=2)
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID", stacklevel=2)
            return None
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del nvml_h
    return uuids


def _transform_uuid_to_ordinals(candidates: list[str], uuids: list[str]) -> list[int]:
    # [FUNCTION DOC] HELPER: Convert partial UUID strings into device ordinals by prefix match against the known UUID list.
    # Parsing stops at the first unmatched or ambiguous candidate; a duplicate ordinal yields an empty list.
    # On ROCm the GPU- prefix is stripped from candidates first.
    r"""Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_ordinal(candidate: str, uuids: list[str]) -> int:
    # [FUNCTION DOC] HELPER: Index of the single known UUID starting with candidate; -1 if none or more than one matches.
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            # Ambiguous candidate
            if best_match != -1:
                return -1
            best_match = idx
        return best_match

    rc: list[int] = []
    for candidate in candidates:
        if torch.version.hip:
            candidate = candidate.replace(
                "GPU-", "", 1
            )  # Remove GPU-prefix to match amdsmi asic serial
        idx = uuid_to_ordinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return cast(list[int], [])
        rc.append(idx)
    return rc


def _device_count_amdsmi() -> int:
    # [FUNCTION DOC] HELPER: Device count seen through amdsmi honoring visible-device variables. Returns 0 if none visible, -1 on failures
    # (missing UUIDs, OS or attribute errors); for ordinal lists, truncates at the first ordinal beyond the real count.
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            uuids = _raw_device_uuid_amdsmi()
            if uuids is None:
                return -1
            # Create string version of visible devices to avoid mypy warnings
            visible_device_str = cast(list[str], visible_devices)
            visible_devices = _transform_uuid_to_ordinals(visible_device_str, uuids)
        else:
            raw_cnt = _raw_device_count_amdsmi()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            # pyrefly: ignore [bad-argument-type]
            for idx, val in enumerate(visible_devices):
                # pyrefly: ignore [redundant-cast]
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _device_count_nvml() -> int:
    # [FUNCTION DOC] HELPER: Device count seen through NVML honoring the visible-devices variable. 0 if none visible, negative on NVML failure;
    # MIG identifiers are not supported (returns -1); UUID lists are mapped to ordinals; ordinal lists are cut at the first ordinal
    # at or beyond the real count.
    r"""Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.

    Negative value is returned if NVML discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            # Skip MIG parsing
            if visible_devices[0].startswith("MIG-"):
                return -1
            uuids = _raw_device_uuid_nvml()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(
                cast(list[str], visible_devices), uuids
            )
        else:
            raw_cnt = _raw_device_count_nvml()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            # pyrefly: ignore [bad-argument-type]
            for idx, val in enumerate(visible_devices):
                # pyrefly: ignore [redundant-cast]
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _get_nvml_device_index(device: Device) -> int:
    # [FUNCTION DOC] HELPER: Map a logical device (None meaning current) to its NVML index using the visible-devices variable
    # (UUID entries are converted to ordinals). Raises RuntimeError if UUIDs cannot be obtained or the index is not visible.
    # KNOWN ISSUE: assumes the visible device list is nonempty; an empty list would raise IndexError.
    r"""Return the NVML index of the device, taking CUDA_VISIBLE_DEVICES into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if type(visible_devices[0]) is str:
        uuids = _raw_device_uuid_nvml()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices = _transform_uuid_to_ordinals(
            cast(list[str], visible_devices), uuids
        )
    visible_devices = cast(list[int], visible_devices)
    if idx < 0 or idx >= len(visible_devices):
        raise RuntimeError(
            f"device {idx} is not visible (CUDA_VISIBLE_DEVICES={visible_devices})"
        )
    return visible_devices[idx]


_cached_device_count: int | None = None


def device_count() -> int:
    # [FUNCTION DOC] Return the number of visible GPUs; 0 if built without CUDA. Returns the cached value when set. Before CUDA init
    # it asks NVML (or amdsmi on ROCm) to avoid poisoning fork, falling back to the runtime when that fails; once initialized
    # it uses the runtime and caches the result in a module global (never cached before init, since visibility may change).
    r"""
    Return the number of GPUs available.

    .. note:: This API will NOT poison fork if NVML discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    if _initialized or hasattr(_tls, "is_initializing"):
        r = torch._C._cuda_getDeviceCount()
    else:
        # bypass _device_count_nvml() if rocm (not supported)
        if torch.version.hip:
            nvml_count = _device_count_amdsmi()
        else:
            nvml_count = _device_count_nvml()
        r = torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count
    # NB: Do not cache the device count prior to CUDA initialization, because
    # the number of devices can change due to changes to CUDA_VISIBLE_DEVICES
    # setting prior to CUDA initialization.
    if _initialized:
        _cached_device_count = r
    return r


def get_arch_list() -> list[str]:
    # [FUNCTION DOC] Return the list of architecture strings this build was compiled for (empty if no CUDA or none reported).
    r"""Return list CUDA architectures this library was compiled for."""
    if not _is_compiled():
        return []
    arch_flags = torch._C._cuda_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    # [FUNCTION DOC] Return the NVCC gencode flag string for each compiled architecture, or an empty string if none.
    r"""Return NVCC gencode flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.split("_") for arch in arch_list]
    return " ".join(
        [
            f"-gencode compute=compute_{arch},code={kind}_{arch}"
            for (kind, arch) in arch_list_
        ]
    )


def current_device() -> int:
    # [FUNCTION DOC] Initialize CUDA lazily and return the index of the current device.
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize(device: Device = None) -> None:
    # [FUNCTION DOC] Block until all kernels in all streams on the device (current if None) finish. Initializes CUDA lazily.
    r"""Wait for all kernels in all streams on a CUDA device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with torch.cuda.device(device):
        return torch._C._cuda_synchronize()


def ipc_collect():
    # [FUNCTION DOC] Force-collect GPU memory released through CUDA IPC (closes unused shared reference-counting files) via the native call.
    r"""Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
    _lazy_init()
    return torch._C._cuda_ipc_collect()


def current_stream(device: Device = None) -> Stream:
    # [FUNCTION DOC] Return a Stream object for the current stream of the device (current device if None).
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def default_stream(device: Device = None) -> Stream:
    # [FUNCTION DOC] Return a Stream object for the default stream of the device (current device if None).
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._cuda_getDefaultStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def get_stream_from_external(data_ptr: int, device: Device = None) -> Stream:
    # [FUNCTION DOC] Wrap an externally created CUDA stream (given as an integer pointer) in a Stream object for the device.
    # Does not own or keep the stream alive; the caller must.
    r"""Return a :class:`Stream` from an externally allocated CUDA stream.

    This function is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this returned
       stream is being used.

    Args:
        data_ptr(int): Integer representation of the `cudaStream_t` value that
            is allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """
    _lazy_init()
    streamdata = torch._C._cuda_getStreamFromExternal(
        data_ptr, _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def current_blas_handle():
    # [FUNCTION DOC] Initialize CUDA and return the pointer to the current cuBLAS handle.
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentBlasHandle()


def current_solver_handle():
    # [FUNCTION DOC] Initialize CUDA and return the pointer to the current cuSOLVER dense handle.
    r"""Return cusolverDnHandle_t pointer to current cuSOLVER handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentSolverHandle()


def set_sync_debug_mode(debug_mode: int | str) -> None:
    # [FUNCTION DOC] Set the sync debug mode (0 or default: nothing, 1 or warn: warn, 2 or error: raise) for synchronizing operations.
    # Raises RuntimeError for an unrecognized string. Initializes CUDA lazily.
    r"""Set the debug mode for cuda synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """
    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )

    torch._C._cuda_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode() -> int:
    # [FUNCTION DOC] Return the current sync debug mode integer.
    r"""Return current value of debug mode for cuda synchronizing operations."""
    _lazy_init()
    return torch._C._cuda_get_sync_debug_mode()


def _get_pynvml_handler(device: Device = None):
    # [FUNCTION DOC] HELPER: Initialize NVML through pynvml and return the handle for the device (current if None). Raises ModuleNotFoundError
    # if pynvml is missing, RuntimeError if the driver is not loaded or the device is invisible.
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "nvidia-ml-py does not seem to be installed or it can't be imported."
            # pyrefly: ignore [invalid-inheritance]
        ) from _PYNVML_ERR
    # pyrefly: ignore [import-error, missing-import, missing-module-attribute]
    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded as e:
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?") from e

    device = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    return handle


def _get_amdsmi_handler(device: Device = None):
    # [FUNCTION DOC] HELPER: Initialize amdsmi and return the processor handle for the device (current if None). Raises ModuleNotFoundError if
    # amdsmi is missing and RuntimeError if the driver cannot be loaded.
    if not _HAS_AMDSMI:
        raise ModuleNotFoundError(
            "amdsmi does not seem to be installed or it can't be imported."
        ) from _AMDSMI_ERR
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        raise RuntimeError(
            "amdsmi driver can't be loaded, requires >=ROCm6.0 installation"
        ) from e
    device = _get_amdsmi_device_index(device)
    handle = amdsmi.amdsmi_get_processor_handles()[device]
    return handle


_cached_hip_to_amdsmi: dict[int, int] | None = None


def _get_amdsmi_device_index_from_hip_index(device: int) -> int:
    # [FUNCTION DOC] HELPER: Translate a HIP device id to an amdsmi index using a lazily built cached mapping from the enumeration info.
    # Warns when the mapping is unavailable (older ROCm with several devices) or the id is absent, then returns the id unchanged.
    # Assumes amdsmi is already initialized.
    r"""Return amdsmi index from HIP device index. They are not always the same.

    Assume amdsmi_init() already completes successfully."""
    global _cached_hip_to_amdsmi
    if _cached_hip_to_amdsmi is None:
        amdsmi_handles = amdsmi.amdsmi_get_processor_handles()

        def gen():
        # [FUNCTION DOC] HELPER: Generator yielding pairs of HIP id and amdsmi index for handles that report a HIP id.
            for amdsmi_idx, handle in enumerate(amdsmi_handles):
                info = amdsmi.amdsmi_get_gpu_enumeration_info(handle)
                if "hip_id" in info:
                    yield info["hip_id"], amdsmi_idx

        _cached_hip_to_amdsmi = dict(gen())
        if not _cached_hip_to_amdsmi and len(amdsmi_handles) > 1:
            warnings.warn(
                "Cannot translate HIP ID to AMD SMI ID due to"
                " lack of translation information prior to ROCm 6.4."
                " Functions that rely on amdsmi"
                " (e.g. temperature()) may operate on wrong devices."
            )
    if device not in _cached_hip_to_amdsmi:
        warnings.warn(
            f"Cannot translate HIP ID {device} to AMD SMI ID due to"
            " undetected HIP ID from amdsmi."
            " amdsmi_get_gpu_enumeration_info() only report these HIP IDs"
            f" {list(_cached_hip_to_amdsmi.keys())}."
            " Functions that rely on amdsmi"
            " (e.g. temperature()) may operate on wrong devices."
        )
    return _cached_hip_to_amdsmi.get(device, device)


def _get_amdsmi_device_index(device: Device) -> int:
    # [FUNCTION DOC] HELPER: Map a logical device to its amdsmi index honoring visible-device settings (UUIDs map directly to ordinals,
    # ordinal lists go through the HIP-to-amdsmi translation). Raises RuntimeError if UUIDs are unavailable or the device is not visible.
    r"""Return the amdsmi index of the device, taking visible_devices into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    visible_device_is_str = type(visible_devices[0]) is str
    if visible_device_is_str:
        uuids = _raw_device_uuid_amdsmi()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices_str = cast(
            list[str], visible_devices
        )  # Create str variable for mypy
        visible_devices = _transform_uuid_to_ordinals(visible_devices_str, uuids)
    idx_map = dict(enumerate(cast(list[int], visible_devices)))
    if idx not in idx_map:
        raise RuntimeError(
            f"device {idx} is not visible (HIP_VISIBLE_DEVICES={visible_devices})"
        )
    if visible_device_is_str:
        return idx_map[idx]
    else:
        return _get_amdsmi_device_index_from_hip_index(idx_map[idx])


def _get_amdsmi_device_memory_used(device: Device = None) -> int:
    # [FUNCTION DOC] HELPER: Used VRAM in bytes for the device via amdsmi (reported in megabytes, converted).
    handle = _get_amdsmi_handler(device)
    # amdsmi_get_gpu_vram_usage returns mem usage in megabytes
    mem_mega_bytes = amdsmi.amdsmi_get_gpu_vram_usage(handle)["vram_used"]
    mem_bytes = mem_mega_bytes * 1024 * 1024
    return mem_bytes


def _get_amdsmi_memory_usage(device: Device = None) -> int:
    # [FUNCTION DOC] HELPER: Memory controller activity percentage for the device via amdsmi.
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_gpu_activity(handle)["umc_activity"]


def _get_amdsmi_utilization(device: Device = None) -> int:
    # [FUNCTION DOC] HELPER: Graphics activity percentage for the device via amdsmi.
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]


def _get_amdsmi_temperature(device: Device = None) -> int:
    # [FUNCTION DOC] HELPER: Current junction temperature of the device via amdsmi.
    handle = _get_amdsmi_handler(device)
    return amdsmi.amdsmi_get_temp_metric(
        handle,
        amdsmi.AmdSmiTemperatureType.JUNCTION,
        amdsmi.AmdSmiTemperatureMetric.CURRENT,
    )


def _get_amdsmi_power_draw(device: Device = None) -> int:
    # [FUNCTION DOC] HELPER: Average socket power via amdsmi, falling back to current socket power, then 0 if both are reported unavailable.
    handle = _get_amdsmi_handler(device)
    socket_power = amdsmi.amdsmi_get_power_info(handle)["average_socket_power"]
    if socket_power != "N/A":
        return socket_power
    else:
        socket_power = amdsmi.amdsmi_get_power_info(handle)["current_socket_power"]
        if socket_power != "N/A":
            return socket_power
        else:
            return 0


def _get_amdsmi_clock_rate(device: Device = None) -> int:
    # [FUNCTION DOC] HELPER: Current graphics clock via amdsmi (handles the older and newer key names); 0 if reported unavailable.
    handle = _get_amdsmi_handler(device)
    clock_info = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
    if "cur_clk" in clock_info:  # ROCm 6.2 deprecation
        clock_rate = clock_info["cur_clk"]
    else:
        clock_rate = clock_info["clk"]
    if clock_rate != "N/A":
        return clock_rate
    else:
        return 0


def device_memory_used(device: Device = None) -> int:
    # [FUNCTION DOC] Return used device memory in bytes from NVML (or amdsmi on ROCm) for the device (current if None).
    # Raises ModuleNotFoundError if the monitoring library is missing.
    r"""Return used global (device) memory in bytes as given by `nvidia-smi` or `amd-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    """
    if not torch.version.hip:
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    else:
        return _get_amdsmi_device_memory_used(device)


def memory_usage(device: Device = None) -> int:
    # [FUNCTION DOC] Return the percent of the last sample period in which device memory was read or written (NVML or amdsmi).
    # KNOWN ISSUE: the NVML branch acquires a handle for the default device first, then overwrites it, so the device argument is only used for the final handle.
    r"""Return the percent of time over the past sample period during which global (device)
    memory was being read or written as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler()
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).memory
    else:
        return _get_amdsmi_memory_usage(device)


def utilization(device: Device = None) -> int:
    # [FUNCTION DOC] Return the percent of the last sample period in which any kernel was executing on the GPU (NVML or amdsmi).
    r"""Return the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    else:
        return _get_amdsmi_utilization(device)


def temperature(device: Device = None) -> int:
    # [FUNCTION DOC] Return the GPU die temperature in degrees Celsius (NVML sensor 0, or amdsmi junction temperature).
    r"""Return the average temperature of the GPU sensor in Degrees C (Centigrades).

    The average temperature is computed based on past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        # 0 refers to the temperature sensor for the GPU die.
        return pynvml.nvmlDeviceGetTemperature(handle, 0)
    else:
        return _get_amdsmi_temperature(device)


def power_draw(device: Device = None) -> int:
    # [FUNCTION DOC] Return GPU power draw in milliwatts from NVML or amdsmi on ROCm.
    r"""Return the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        return pynvml.nvmlDeviceGetPowerUsage(handle)
    else:
        return _get_amdsmi_power_draw(device)


def clock_rate(device: Device = None) -> int:
    # [FUNCTION DOC] Return the SM clock in MHz from NVML or the graphics clock from amdsmi on ROCm.
    r"""Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    if not torch.version.hip:
        handle = _get_pynvml_handler(device)
        return pynvml.nvmlDeviceGetClockInfo(handle, 1)
    else:
        return _get_amdsmi_clock_rate(device)


def _get_device(device: int | str | torch.device) -> torch.device:
    # [FUNCTION DOC] HELPER: Normalize a string or integer into a torch device (integers become CUDA devices); device objects pass through.
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return device


def _get_generator(device: torch.device) -> torch._C.Generator:
    # [FUNCTION DOC] HELPER: Return the default CUDA generator for the device (current index if the device has none).
    r"""Return the CUDA Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """
    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.cuda.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: int | str | torch.device = "cuda"
) -> None:
    # [FUNCTION DOC] HELPER: Set the RNG state offset of the device's default generator, deferring the work until CUDA is initialized.
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    final_device = _get_device(device)

    def cb():
    # [FUNCTION DOC] HELPER: Deferred callback that sets the offset on the default generator of the resolved device.
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: int | str | torch.device = "cuda") -> int:
    # [FUNCTION DOC] HELPER: Return the RNG state offset of the device's default generator; eagerly initializes CUDA.
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


# pyrefly: ignore [deprecated]
from .memory import *  # noqa: F403
from .memory import _use_uvm
from .random import *  # noqa: F403


################################################################################
# Define Storage and Tensor classes
################################################################################


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for object construction on CUDA storage classes: initializes CUDA lazily then defers to the parent new.
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _CudaBase.__new__
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase:
    # [CLASS DOC] HELPER: Mixin for legacy CUDA storage classes: marks them as CUDA (not sparse), runs type conversion under the storage's
    # device, and routes construction through the lazy initializer.
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Call the parent type method with the storage's device selected as current.
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super().type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new


from torch.storage import _LegacyStorage, _warn_typed_storage_removal


class _CudaLegacyStorage(_LegacyStorage):
    # [CLASS DOC] HELPER: Base for legacy typed CUDA storage classes; disables from_buffer, weak pointer and shared filename constructors.
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Warn about typed storage removal then always raise RuntimeError (unsupported for CUDA storage).
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for CUDA storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises RuntimeError (unsupported for CUDA storage).
        raise RuntimeError("_new_with_weak_ptr: Not available for CUDA storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        # [FUNCTION DOC] HELPER: Always raises RuntimeError (unsupported for CUDA storage).
        raise RuntimeError("_new_shared_filename: Not available for CUDA storage")


class ByteStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for uint8 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype uint8.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype uint8 without warning.
        return torch.uint8


class DoubleStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for float64 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype float64.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype float64 without warning.
        return torch.double


class FloatStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for float32 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype float32.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype float32 without warning.
        return torch.float


class HalfStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for float16 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype float16.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype float16 without warning.
        return torch.half


class LongStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for int64 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype int64.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype int64 without warning.
        return torch.long


class IntStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for int32 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype int32.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype int32 without warning.
        return torch.int


class ShortStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for int16 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype int16.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype int16 without warning.
        return torch.short


class CharStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for int8 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype int8.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype int8 without warning.
        return torch.int8


class BoolStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for bool elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype bool.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype bool without warning.
        return torch.bool


class BFloat16Storage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for bfloat16 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype bfloat16.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype bfloat16 without warning.
        return torch.bfloat16


class ComplexDoubleStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for complex128 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype complex128.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype complex128 without warning.
        return torch.cdouble


class ComplexFloatStorage(_CudaLegacyStorage):
    # [CLASS DOC] Legacy CUDA typed storage class for complex64 elements; exposes the element type via class properties (dtype warns about typed storage removal).
    @classproperty
    def dtype(self):
        # [FUNCTION DOC] HELPER: Class property: warn that typed storage is deprecated, then return the element dtype complex64.
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        # [FUNCTION DOC] HELPER: Class property returning the element dtype complex64 without warning.
        return torch.cfloat


del _LegacyStorage
del _CudaLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)


class _WrappedTritonKernel:
    # [CLASS DOC] HELPER: Callable wrapper around a kernel function that records whether it was ever invoked (kernel_invoked), for tests.
    """Just a simple wrapper to store some metadata for testing purposes."""

    def __init__(self, kernel):
        # [FUNCTION DOC] HELPER: Store the wrapped kernel and set the invoked flag to False.
        self.kernel = kernel
        self.kernel_invoked = False

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forward all arguments to the wrapped kernel, mark it invoked, and return its result.
        res = self.kernel(*args, **kwargs)
        self.kernel_invoked = True
        return res


def _register_triton_kernels():
    # [FUNCTION DOC] HELPER: Queued at CUDA init: if triton is importable, register the BSR dense matmul and addmm sparse CUDA ops with the
    # triton library registry, each wrapped to record invocation. Side effect: global operator registration.
    @_WrappedTritonKernel
    def kernel_impl(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapped kernel calling the sparse BSR dense matmul with argument checks skipped.
        from torch.sparse._triton_ops import bsr_dense_mm

        # pyrefly: ignore [not-callable]
        return bsr_dense_mm(*args, skip_checks=True, **kwargs)

    @_WrappedTritonKernel
    def addmm_kernel_impl(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapped kernel calling the sparse BSR dense addmm with argument checks skipped.
        from torch.sparse._triton_ops import bsr_dense_addmm

        return bsr_dense_addmm(*args, skip_checks=True, **kwargs)

    has_triton = importlib.util.find_spec("triton") is not None
    if has_triton:
        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_mm_out",
            "_triton_bsr_dense_mm_out(Tensor bsr, Tensor dense, *, Tensor(a!) out) -> Tensor(a!)",
            kernel_impl,
            "SparseCsrCUDA",
        )

        torch._TritonLibrary.registerOp(
            "_triton_bsr_dense_addmm_out",
            (
                "_triton_bsr_dense_addmm_out(Tensor input, Tensor bsr, Tensor dense,"
                " *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)"
            ),
            addmm_kernel_impl,
            "SparseCsrCUDA",
        )


_lazy_call(_register_triton_kernels)


def _compile_kernel(
    kernel_source: str,
    kernel_name: str,
    compute_capability: str | None = None,
    cuda_include_dirs: list | None = None,
    nvcc_options: list | None = None,
):
    # [FUNCTION DOC] HELPER: Compile CUDA source with NVRTC (target capability defaults to the current device) and load the resulting PTX,
    # returning the callable for the named kernel; it works on raw pointers. Extra include dirs and compiler options are optional.
    # KNOWN ISSUE: the docstring example refers to a public compile_kernel name, but the function defined here is private.
    """
    Compiles a CUDA kernel using NVRTC and returns a callable function.

    This function is a wrapper for NVRTC that enables runtime compilation of CUDA kernels.
    Note that this returns a raw CUDA kernel that operates on raw memory pointers.
    To use this kernel as a proper PyTorch operator, you should wrap it following the guide at:
    pytorch.org/tutorials/advanced/python_custom_ops.html

    Args:
        kernel_source (str): The CUDA kernel source code as a string
        kernel_name (str): The name of the kernel function to compile
        compute_capability (str, optional): The compute capability to target (e.g., "86").
                                           If None, will detect from current device.
        cuda_include_dirs (list, optional): List of directories containing CUDA headers
        nvcc_options (list, optional): Additional options to pass to NVRTC

    Returns:
        callable: A Python function that can be called with PyTorch tensor arguments to execute the kernel

    Example:
        >>> # xdoctest: +SKIP
        >>> kernel_code = '''
        extern "C"
        __global__ void add_tensors(const float* a, const float* b, float* c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n)
                c[i] = a[i] + b[i];
        }
        '''
        >>> add_kernel = torch.cuda.compile_kernel(kernel_code, "add_tensors")
        >>> a = torch.randn(1024, device="cuda")
        >>> b = torch.randn(1024, device="cuda")
        >>> c = torch.empty_like(a)
        >>> add_kernel(grid=(4, 1, 1), block=(256, 1, 1), args=[a, b, c, a.numel()])
    """
    from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

    # Compile the kernel to PTX
    ptx, mangled_name = _nvrtc_compile(
        kernel_source,
        kernel_name,
        compute_capability,
        cuda_include_dirs,
        nvcc_options,
    )

    # Load the module and get the kernel
    result = _cuda_load_module(ptx, [mangled_name])

    if isinstance(result, dict):
        return result[mangled_name]
    else:
        # This branch shouldn't be executed if kernel_names is provided,
        # but MyPy needs this to understand type narrowing
        return getattr(result, mangled_name)


from . import amp, graph_annotations, jiterator, nvtx, profiler, sparse, tunable


_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])


__all__ = [
    # Typed storage and tensors
    "BFloat16Storage",
    # pyrefly: ignore [bad-dunder-all]
    "BFloat16Tensor",
    "BoolStorage",
    # pyrefly: ignore [bad-dunder-all]
    "BoolTensor",
    "ByteStorage",
    # pyrefly: ignore [bad-dunder-all]
    "ByteTensor",
    "CharStorage",
    # pyrefly: ignore [bad-dunder-all]
    "CharTensor",
    "ComplexDoubleStorage",
    "ComplexFloatStorage",
    "DoubleStorage",
    # pyrefly: ignore [bad-dunder-all]
    "DoubleTensor",
    "FloatStorage",
    # pyrefly: ignore [bad-dunder-all]
    "FloatTensor",
    "HalfStorage",
    # pyrefly: ignore [bad-dunder-all]
    "HalfTensor",
    "IntStorage",
    # pyrefly: ignore [bad-dunder-all]
    "IntTensor",
    "LongStorage",
    # pyrefly: ignore [bad-dunder-all]
    "LongTensor",
    "ShortStorage",
    # pyrefly: ignore [bad-dunder-all]
    "ShortTensor",
    "CUDAGraph",
    "CudaError",
    "DeferredCudaCallError",
    "Event",
    "ExternalStream",
    "Stream",
    "StreamContext",
    "GreenContext",
    "amp",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_disabled",
    "caching_allocator_enable",
    "can_device_access_peer",
    "check_error",
    "cudaStatus",
    "cudart",
    "current_blas_handle",
    "current_solver_handle",
    "current_device",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_memory_used",
    "device_of",
    "empty_cache",
    "export_dot",
    "export_graph_data",
    "get_allocator_backend",
    "CUDAPluggableAllocator",
    "change_current_allocator",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "get_sync_debug_mode",
    "graph",
    "graph_annotations",
    "graph_pool_handle",
    "graphs",
    "has_half",
    "has_magma",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "init",
    "initial_seed",
    "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "is_tf32_supported",
    "jiterator",
    "list_gpu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "memory_usage",
    "MemPool",
    "use_mem_pool",
    "temperature",
    "power_draw",
    "clock_rate",
    "nccl",
    "nvtx",
    "profiler",
    "random",
    "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "set_sync_debug_mode",
    "sparse",
    "stream",
    "streams",
    "synchronize",
    "tunable",
    "utilization",
]
