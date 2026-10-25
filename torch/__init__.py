# [PACKAGE DOC]
# Top-level entry point of the torch package. Importing it loads the compiled
# torch._C extension (optionally preloading CUDA/global-dependency shared
# libraries first, or Windows DLL directories), re-exports the public C
# bindings, and defines the symbolic scalar types (SymInt, SymFloat, SymBool)
# plus sym_* helpers, basic utilities (typename, is_tensor, is_storage),
# global default device/dtype and deterministic-algorithm settings, and further
# global configuration helpers. Later in the file it imports many submodules
# and re-exports names such as Tensor, storage classes, autograd, compile and
# export. Import-time side effects: platform-specific library loading,
# patching __module__ of C-extension objects to "torch", registering C
# submodules in sys.modules, and growing __all__ with names from torch._C.
"""
The torch package contains data structures for multi-dimensional
tensors and defines mathematical operations over these tensors.
Additionally, it provides many utilities for efficient serialization of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import builtins
import ctypes
import functools
import glob
import importlib
import inspect
import math
import os
import platform
import sys
import textwrap
import threading
import typing
import typing_extensions
import warnings
from collections.abc import Callable as _Callable, Sequence as _Sequence
from types import ModuleType as _ModuleType
from typing import (
    Any as _Any,
    get_origin as _get_origin,
    overload as _overload,
    TYPE_CHECKING,
    TypeAlias as _TypeAlias,
    TypeGuard as _TypeGuard,
    TypeVar as _TypeVar,
)
from typing_extensions import (
    deprecated as _deprecated,
    LiteralString as _LiteralString,
    Never as _Never,
    ParamSpec as _ParamSpec,
    Self as _Self,
    TypeIs as _TypeIs,
)


# As a bunch of torch.packages internally still have this check
# we need to keep this. @todo: Remove tests that rely on this check as
# they are likely stale.
def _running_with_deploy() -> builtins.bool:
    # [FUNCTION DOC] HELPER: Legacy check kept for old torch.package code; always returns False.
    return False


from torch._utils import (
    _functionalize_sync as _sync,
    _import_dotted_name,
    classproperty,
)
from torch._utils_internal import (
    get_file_path,
    prepare_multiprocessing_environment,
    profiler_allow_cudagraph_cupti_lazy_reinit_cuda12,
    USE_GLOBAL_DEPS,
    USE_RTLD_GLOBAL_WITH_LIBTORCH,
)
from torch.torch_version import __version__ as __version__


if TYPE_CHECKING:
    import sympy

    from torch.distributed._local_tensor import LocalIntNode
    from torch.fx.experimental._constant_symnode import ConstantIntNode
    from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec
    from torch.fx.experimental.sym_node import SymNode
    from torch.nested._internal.nested_int import NestedIntNode
    from torch.types import BoolLikeType, Device, FloatLikeType, IntLikeType, PySymType

    _SymNodeLike = SymNode | NestedIntNode | ConstantIntNode | LocalIntNode


__all__ = [
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "GradScaler",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "SymBool",
    "SymFloat",
    "SymInt",
    "Tensor",
    "TypedStorage",
    "UntypedStorage",
    "are_deterministic_algorithms_enabled",
    "autocast",
    "chunk",
    "compile",
    "cond",
    "enable_grad",
    "export",
    "get_default_device",
    "get_deterministic_debug_mode",
    "get_device_module",
    "get_float32_matmul_precision",
    "get_rng_state",
    "inference_mode",
    "initial_seed",
    "is_deterministic_algorithms_warn_only_enabled",
    "is_storage",
    "is_tensor",
    "is_warn_always_enabled",
    "load",
    "lobpcg",
    "manual_seed",
    "matmul",
    "no_grad",
    "rand",
    "randn",
    "save",
    "seed",
    "set_default_device",
    "set_default_tensor_type",
    "set_deterministic_debug_mode",
    "set_float32_matmul_precision",
    "set_printoptions",
    "set_rng_state",
    "set_warn_always",
    "split",
    "stack",
    "sym_float",
    "sym_fresh_size",
    "sym_int",
    "sym_ite",
    "sym_max",
    "sym_min",
    "sym_not",
    "sym_sum",
    "thread_safe_generator",
    "typename",
    "unravel_index",
    "use_deterministic_algorithms",
    "vmap",
]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ must be kept sorted")

################################################################################
# Load the extension module
################################################################################

# If PyTorch was built against the ROCm runtime wheels, then there will be
# a _rocm_init module and it will define an initialize() function which can
# prepare ROCm for use. See general documentation on ROCm runtime wheels:
# https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
# Since this module is only ever added to the wheel if built for such a
# deployment, it is always safe to attempt.
try:
    from . import _rocm_init  # type: ignore[attr-defined]
except ImportError:
    pass
else:
    _rocm_init.initialize()
    del _rocm_init


if sys.platform == "win32":

    def _load_dll_libraries() -> None:
        # [FUNCTION DOC] HELPER: Windows only. Builds the list of DLL search directories (torch lib dir
        # located via the torch._C extension, Python prefix dirs, user base, optional NvToolsExt and CUDA
        # bin dirs), registers them with the loader, warns if the VC++ runtime is missing, then loads each
        # DLL in the torch lib dir. Raises the Windows error (annotated with the DLL name) if a load fails.
        # Side effects: may prepend to the PATH environment variable, prints a warning, changes error mode
        # temporarily. Called once at import and then deleted.
        import importlib.util
        import sysconfig

        from torch.version import cuda as cuda_version

        pfiles_path = os.getenv("ProgramFiles", r"C:\Program Files")
        py_dll_path = os.path.join(sys.exec_prefix, "Library", "bin")
        th_dll_path = os.path.join(os.path.dirname(__file__), "lib")
        # Anchor the DLL dir on the compiled _C extension rather than __file__:
        # editable installs (e.g. scikit-build-core) run this __init__ from the
        # source tree, whose lib/ is empty, while _C and its dependent DLLs live
        # in the installed tree. In a wheel install both resolve to the same dir.
        _spec = importlib.util.find_spec("torch._C")
        if _spec is not None and _spec.origin:
            th_dll_path = os.path.join(os.path.dirname(_spec.origin), "lib")
        usebase_path = os.path.join(
            sysconfig.get_config_var("userbase"), "Library", "bin"
        )
        py_root_bin_path = os.path.join(sys.exec_prefix, "bin")

        # When users create a virtualenv that inherits the base environment,
        # we will need to add the corresponding library directory into
        # DLL search directories. Otherwise, it will rely on `PATH` which
        # is dependent on user settings.
        if sys.exec_prefix != sys.base_exec_prefix:
            base_py_dll_path = os.path.join(sys.base_exec_prefix, "Library", "bin")
        else:
            base_py_dll_path = ""

        dll_paths = [
            p
            for p in (
                th_dll_path,
                py_dll_path,
                base_py_dll_path,
                usebase_path,
                py_root_bin_path,
            )
            if os.path.exists(p)
        ]

        if not builtins.any(
            os.path.exists(os.path.join(p, "nvToolsExt64_1.dll")) for p in dll_paths
        ):
            nvtoolsext_dll_path = os.path.join(
                os.getenv(
                    "NVTOOLSEXT_PATH",
                    os.path.join(pfiles_path, "NVIDIA Corporation", "NvToolsExt"),
                ),
                "bin",
                "x64",
            )
        else:
            nvtoolsext_dll_path = ""

        if cuda_version and builtins.all(
            not glob.glob(os.path.join(p, "cudart64*.dll")) for p in dll_paths
        ):
            cuda_version_1 = cuda_version.replace(".", "_")
            cuda_path_var = "CUDA_PATH_V" + cuda_version_1
            default_path = os.path.join(
                pfiles_path, "NVIDIA GPU Computing Toolkit", "CUDA", f"v{cuda_version}"
            )
            cuda_path = os.path.join(os.getenv(cuda_path_var, default_path), "bin")
        else:
            cuda_path = ""

        dll_paths.extend(
            p for p in (nvtoolsext_dll_path, cuda_path) if os.path.exists(p)
        )

        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
        prev_error_mode = kernel32.SetErrorMode(0x0001)

        kernel32.LoadLibraryW.restype = ctypes.c_void_p
        if with_load_library_flags:
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

        for dll_path in dll_paths:
            os.add_dll_directory(dll_path)

        try:
            ctypes.CDLL("vcruntime140.dll")
            ctypes.CDLL("msvcp140.dll")
            if platform.machine() != "ARM64":
                ctypes.CDLL("vcruntime140_1.dll")
        except OSError:
            print(
                textwrap.dedent(
                    """
                    Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                    It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe
                    """
                ).strip()
            )

        dlls = glob.glob(os.path.join(th_dll_path, "*.dll"))
        path_patched = False
        for dll in dlls:
            is_loaded = False
            if with_load_library_flags:
                res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
                last_error = ctypes.get_last_error()
                if res is None and last_error != 126:
                    err = ctypes.WinError(last_error)
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err
                elif res is not None:
                    is_loaded = True
            if not is_loaded:
                if not path_patched:
                    os.environ["PATH"] = ";".join(dll_paths + [os.environ["PATH"]])
                    path_patched = True
                res = kernel32.LoadLibraryW(dll)
                if res is None:
                    err = ctypes.WinError(ctypes.get_last_error())
                    err.strerror += (
                        f' Error loading "{dll}" or one of its dependencies.'
                    )
                    raise err

        kernel32.SetErrorMode(prev_error_mode)

    _load_dll_libraries()
    del _load_dll_libraries


def _get_cuda_dep_paths(path: str, lib_folder: str, lib_name: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns candidate file paths for a CUDA shared library under one search
    # root, checking the nvidia vendor folder (both per-library and per-CUDA-major layouts) first and
    # then the plain library folder. Pure file-system lookup; returns a possibly empty list.
    # Libraries can either be in
    # path/nvidia/lib_folder/lib or
    # path/nvidia/cuXX/lib (since CUDA 13.0) or
    # path/lib_folder/lib
    from torch.version import cuda as cuda_version

    nvidia_lib_paths = glob.glob(
        os.path.join(path, "nvidia", lib_folder, "lib", lib_name)
    )
    if cuda_version is not None:
        maj_cuda_version = cuda_version.split(".")[0]
        nvidia_lib_paths += glob.glob(
            os.path.join(path, "nvidia", f"cu{maj_cuda_version}", "lib", lib_name)
        )
    lib_paths = glob.glob(os.path.join(path, lib_folder, "lib", lib_name))

    return nvidia_lib_paths + lib_paths


def _preload_cuda_lib(lib_folder: str, lib_name: str, required: bool = True) -> None:  # type: ignore[valid-type]
    # [FUNCTION DOC] HELPER: Linux only. Scans every entry of the module search path for the named CUDA
    # library and loads the first match into the process via the dynamic loader. Raises AssertionError
    # on non-Linux platforms and ValueError if the library is not found while required is true.
    # Side effect: loads a shared library.
    """Preloads cuda library if it could not be found otherwise."""
    # Should only be called on Linux if default path resolution have failed
    if platform.system() != "Linux":
        raise AssertionError(f"Should only be called on Linux, got {platform.system()}")

    lib_path = None
    for path in sys.path:
        candidate_lib_paths = _get_cuda_dep_paths(path, lib_folder, lib_name)
        if candidate_lib_paths:
            lib_path = candidate_lib_paths[0]
            break
    if not lib_path and required:
        raise ValueError(f"{lib_name} not found in the system path {sys.path}")
    if lib_path:
        ctypes.CDLL(lib_path)


def _preload_cuda_deps(err: OSError | None = None, required: bool = True) -> None:
    # [FUNCTION DOC] HELPER: Preloads the ordered list of CUDA wheel libraries (cublasLt before cublas
    # on purpose) plus the optional NVTX library. If an error is given and it does not mention any of
    # the known libraries, that error is re-raised unchanged. With required false, missing libraries
    # are skipped instead of raising ValueError. Side effect: loads shared libraries.
    cuda_libs: list[tuple[str, str]] = [
        # NOTE: Order matters! We must preload libcublasLt BEFORE libcublas to prevent
        # libcublas from loading a mismatched system-wide libcublasLt via its RUNPATH.
        # Without this, if a different CUDA Toolkit version exists in the system PATH,
        # libcublas may load the wrong libcublasLt, causing symbol errors or runtime failures.
        ("cublas", "libcublasLt.so.*[0-9]"),
        ("cublas", "libcublas.so.*[0-9]"),
        ("cudnn", "libcudnn.so.*[0-9]"),
        ("cuda_nvrtc", "libnvrtc.so.*[0-9]"),
        ("cuda_nvrtc", "libnvrtc-builtins.so.*[0-9]"),
        ("cuda_runtime", "libcudart.so.*[0-9]"),
        ("cuda_cupti", "libcupti.so.*[0-9]"),
        ("cufft", "libcufft.so.*[0-9]"),
        ("curand", "libcurand.so.*[0-9]"),
        ("nvjitlink", "libnvJitLink.so.*[0-9]"),
        ("cusparse", "libcusparse.so.*[0-9]"),
        ("cusparselt", "libcusparseLt.so.*[0-9]"),
        ("cusolver", "libcusolver.so.*[0-9]"),
        ("nccl", "libnccl.so.*[0-9]"),
        ("nvshmem", "libnvshmem_host.so.*[0-9]"),
        ("cufile", "libcufile.so.*[0-9]"),
    ]
    # If error is passed, re-raise it if it's not about one of the abovementioned
    # libraries
    if err is not None and not [
        lib for _, lib in cuda_libs if lib.split(".", 1)[0] in err.args[0]
    ]:
        raise err

    # Otherwise, try to preload dependencies from site-packages. With
    # required=False each lib is best-effort: a partial install that only ships
    # some of these wheels (e.g. just a cupti wheel alongside a system CUDA)
    # preloads the ones present instead of aborting on the first missing one.
    for lib_folder, lib_name in cuda_libs:
        _preload_cuda_lib(lib_folder, lib_name, required=required)

    # libnvToolsExt is Optional Dependency
    _preload_cuda_lib("nvtx", "libnvToolsExt.so.*[0-9]", required=False)


# See Note [Global dependencies]
def _load_global_deps() -> None:
    # [FUNCTION DOC] HELPER: Loads libtorch_global_deps with global symbol visibility (no-op on Windows).
    # Falls back to the installed distribution location for editable installs. After a successful load,
    # if libcudart is mapped into the process it best-effort preloads CUDA wheel dependencies; if the
    # first load fails with OSError it preloads CUDA deps for that error and retries. Side effects:
    # loads shared libraries, reads the process memory map file.
    if platform.system() == "Windows":
        return

    # Determine the file extension based on the platform
    lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"
    lib_name = f"libtorch_global_deps{lib_ext}"
    here = os.path.abspath(__file__)
    global_deps_lib_path = os.path.join(os.path.dirname(here), "lib", lib_name)

    # In scikit-build-core editable installs with redirect mode, native libs are
    # installed to the dist package location rather than relative to __file__.
    if not os.path.exists(global_deps_lib_path):
        try:
            from importlib.metadata import distribution

            installed = distribution("torch").locate_file(
                os.path.join("torch", "lib", lib_name)
            )
            # The importlib metadata SimplePath protocol was missing the exists
            # method in older versions; however, the actual Path implementation
            # has it and newer versions of importlib metadata have added it to
            # the protocol, making the following ignore unnecessary from
            # importlib_metadata 7.0.1 and Python 3.13 onwards.
            # pyrefly: ignore[missing-attribute]
            if installed.exists():
                global_deps_lib_path = str(installed)
        except Exception:
            pass

    try:
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
        # Workaround slim-wheel CUDA dependency bugs in cusparse and cudnn by preloading nvjitlink
        # and nvrtc. In CUDA-12.4+ cusparse depends on nvjitlink, but does not have rpath when
        # shipped as wheel, which results in OS picking wrong/older version of nvjitlink library
        # if `LD_LIBRARY_PATH` is defined, see https://github.com/pytorch/pytorch/issues/138460
        # Similar issue exist in cudnn that dynamically loads nvrtc, unaware of its relative path.
        # See https://github.com/pytorch/pytorch/issues/145580
        try:
            with open("/proc/self/maps") as f:
                _maps = f.read()

            # libtorch_global_deps.so always depends in cudart, check if its installed and loaded
            if "libcudart.so" not in _maps:
                return
            # If all above-mentioned conditions are met, preload CUDA dependencies.
            # Only libtorch_global_deps is loaded so far (it pulls libcudart, not
            # the rest); libtorch_cpu, which DT_NEEDEDs the other CUDA libs, is
            # imported below via torch._C. So be best-effort and front-load any
            # component wheels that ARE present now, before that import, so they
            # win libtorch_cpu's DT_NEEDED soname lookups (e.g. a newer cupti
            # wheel over a system copy). A missing wheel must not abort -- rpath
            # will satisfy libtorch_cpu's DT_NEEDED for whatever isn't preloaded.
            _preload_cuda_deps(required=False)
        except Exception:
            pass

    except OSError as err:
        # Can happen for wheel with cuda libs as PYPI deps
        # As PyTorch is not purelib, but nvidia-*-cu12 is
        _preload_cuda_deps(err)
        ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)


if (USE_RTLD_GLOBAL_WITH_LIBTORCH or os.getenv("TORCH_USE_RTLD_GLOBAL")) and (
    platform.system() != "Windows"
):
    # Do it the hard way.  You might want to load libtorch with RTLD_GLOBAL in a
    # few circumstances:
    #
    #   1. You're in a build environment (e.g., fbcode) where
    #      libtorch_global_deps is not available, but you still need
    #      to get mkl to link in with RTLD_GLOBAL or it will just
    #      not work.
    #
    #   2. You're trying to run PyTorch under UBSAN and you need
    #      to ensure that only one copy of libtorch is loaded, so
    #      vptr checks work properly
    #
    # If you're using this setting, you must verify that all the libraries
    # you load consistently use the same libstdc++, or you may have
    # mysterious segfaults.
    #
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

    from torch._C import *  # noqa: F403

    sys.setdlopenflags(old_flags)
    del old_flags

else:
    # Easy way.  You want this most of the time, because it will prevent
    # C++ symbols from libtorch clobbering C++ symbols from other
    # libraries, leading to mysterious segfaults.
    #
    # If building in an environment where libtorch_global_deps isn't available
    # like parts of fbsource, but where RTLD_GLOBAL causes segfaults, you will
    # want USE_RTLD_GLOBAL_WITH_LIBTORCH = False and USE_GLOBAL_DEPS = False
    #
    # See Note [Global dependencies]
    if USE_GLOBAL_DEPS:
        _load_global_deps()
    from torch._C import *  # noqa: F403


_PrimType = _TypeVar("_PrimType")
_SymType = _TypeVar("_SymType")
_SymIteT = _TypeVar("_SymIteT")
_BecomesIntPrimType = _TypeVar("_BecomesIntPrimType")
_BecomesIntSymType = _TypeVar("_BecomesIntSymType")
# Operand type for bitwise ops: IntLikeType for SymInt; BoolLikeType for SymBool.
_BitwiseLikeType = _TypeVar("_BitwiseLikeType")
# Operand types that promote the result to SymFloat. _Never for SymFloat
# (no further promotion); float | SymFloat for SymInt and SymBool.
_FloatPromotionType = _TypeVar("_FloatPromotionType")
_LikeNumber: _TypeAlias = "torch.types.Number | torch.types.PySymType"


class _SymTypingMagicAlsoBool(
    typing.Generic[
        _PrimType, _BecomesIntPrimType, _BecomesIntSymType, _FloatPromotionType
    ]
):
    # [CLASS DOC] HELPER: Typing-only generic base declaring the magic methods shared by SymInt,
    # SymFloat and SymBool (equality, add, sub, mul and their reflected forms) with overloads that
    # describe result types. Every real method body raises TypeError because the actual
    # implementations are installed at runtime by torch.fx.experimental.sym_node.
    # Magic methods installed by torch.fx.experimental.sym_node
    # From sym_node: also_bool_magic_methods

    __LikeType: _TypeAlias = (
        _PrimType | _Self | _BecomesIntPrimType | _BecomesIntSymType
    )
    _BecomesIntLikeType: _TypeAlias = _BecomesIntPrimType | _BecomesIntSymType

    @_overload
    def __eq__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __eq__(self, other: object) -> builtins.bool: ...

    def __eq__(self, other: object) -> "builtins.bool | SymBool":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __ne__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __ne__(self, other: object) -> builtins.bool: ...

    def __ne__(self, other: object) -> "builtins.bool | SymBool":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    # bool_becomes_int_magic_methods

    @_overload
    def __add__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __add__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __add__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    # Tensor + SymT -> Tensor
    # prim|SymT + SymT -> SymT
    @_overload
    def __radd__(self, other: "torch.Tensor") -> "torch.Tensor": ...

    @_overload
    def __radd__(
        self, other: _PrimType | _BecomesIntPrimType | _BecomesIntSymType
    ) -> _BecomesIntSymType: ...

    @_overload
    def __radd__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __radd__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __sub__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __sub__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __sub__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    # Tensor - SymT = Tensor
    # prim|SymT - SymT = SymT
    @_overload
    def __rsub__(self, other: "torch.Tensor") -> "torch.Tensor": ...

    @_overload
    def __rsub__(
        self, other: _PrimType | _BecomesIntPrimType | _BecomesIntSymType
    ) -> _BecomesIntSymType: ...

    @_overload
    def __rsub__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __rsub__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __mul__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __mul__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __mul__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __rmul__(self, other: __LikeType) -> _BecomesIntSymType: ...

    @_overload
    def __rmul__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __rmul__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")


class _SymTypingMagic(
    _SymTypingMagicAlsoBool[_PrimType, _PrimType, _SymType, _FloatPromotionType],
    typing.Generic[_PrimType, _SymType, _FloatPromotionType],
):
    # [CLASS DOC] HELPER: Typing-only extension of the shared symbolic base adding ordering
    # comparisons, unary ops, shifts, modulo, power helpers, rounding helpers and the sym_* math hooks
    # (sym_sqrt, sym_sin, sym_max, ...) used by SymInt and SymFloat. All real bodies raise TypeError
    # because sym_node installs the real methods at runtime.
    # Magic methods installed by torch.fx.experimental.sym_node
    # From sym_node:
    #   magic_methods
    #   - only_float_magic_methods
    #   - bitwise_ops
    #   - only_bool_magic_methods
    #   - also_bool_magic_methods
    #   - bool_becomes_int_magic_methods

    __LikeType: _TypeAlias = _PrimType | _Self

    # Ordering comparisons: only on ordered types (SymInt/SymFloat), not SymBool.
    @_overload
    def __ge__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __ge__(self, other: object) -> builtins.bool: ...

    def __ge__(self, other: object) -> "builtins.bool | SymBool":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __gt__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __gt__(self, other: object) -> builtins.bool: ...

    def __gt__(self, other: object) -> "builtins.bool | SymBool":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __le__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __le__(self, other: object) -> builtins.bool: ...

    def __le__(self, other: object) -> "builtins.bool | SymBool":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __lt__(self, other: __LikeType) -> "SymBool": ...

    @_overload
    def __lt__(self, other: object) -> builtins.bool: ...

    def __lt__(self, other: object) -> "builtins.bool | SymBool":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __abs__(self) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __ceil__(self) -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __float_pow__(self, other: _LikeNumber) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rfloat_pow__(self, other: _LikeNumber) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __float_truediv__(self, other: _LikeNumber) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rfloat_truediv__(self, other: _LikeNumber) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __floor__(self) -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __int_floordiv__(self, other: _LikeNumber) -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rint_floordiv__(self, other: _LikeNumber) -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __int_truediv__(self, other: _LikeNumber) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rint_truediv__(self, other: _LikeNumber) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __lshift__(self, other: "IntLikeType") -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rlshift__(self, other: "IntLikeType") -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __mod__(self, other: __LikeType) -> _SymType: ...

    @_overload
    def __mod__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __mod__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    @_overload
    def __rmod__(self, other: __LikeType) -> _SymType: ...

    @_overload
    def __rmod__(self, other: _FloatPromotionType) -> "SymFloat": ...

    def __rmod__(self, other: object) -> object:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __neg__(self) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __pos__(self) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __pow_by_natural__(self, other: "IntLikeType") -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rpow_by_natural__(self, other: "IntLikeType") -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rshift__(self, other: "IntLikeType") -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rrshift__(self, other: "IntLikeType") -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_acos__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_asin__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_atan__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_cos__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_cosh__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_float__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_max__(self, other: _LikeNumber) -> _LikeNumber:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_min__(self, other: _LikeNumber) -> _LikeNumber:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_sin__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_sinh__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_sqrt__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_tan__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __sym_tanh__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __trunc__(self) -> "SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")


class _SymTypingMagicBitwise(typing.Generic[_BitwiseLikeType]):
    # [CLASS DOC] HELPER: Typing-only generic base declaring bitwise and/or/xor (and reflected forms)
    # for SymInt and SymBool. Bodies raise TypeError; sym_node installs the real methods at runtime.
    # Magic methods installed by torch.fx.experimental.sym_node
    # From sym_node: bitwise_ops

    def __and__(self, other: _BitwiseLikeType) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rand__(self, other: _BitwiseLikeType) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __or__(self, other: _BitwiseLikeType) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __ror__(self, other: _BitwiseLikeType) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __xor__(self, other: _BitwiseLikeType) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __rxor__(self, other: _BitwiseLikeType) -> _Self:
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")


class SymInt(
    _SymTypingMagic[builtins.int, "SymInt", "FloatLikeType"],
    _SymTypingMagicBitwise["IntLikeType"],
):
    # [CLASS DOC] Symbolic integer: a Python-int-like value that wraps a node object (attribute node,
    # name fixed because C++ bindings read it) and forwards arithmetic to it so operations are recorded
    # symbolically for dynamic shapes. Arithmetic magic methods come from the stub bases and are
    # installed by sym_node; this class adds conversion (bool, int, index), true/floor division, power
    # with guard on exponent sign, repr, hashing (only for nested ints) and hint/constant accessors.
    # Converting to a concrete int or bool forces a guard.
    """
    Like an int (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node: "_SymNodeLike") -> None:
        # [FUNCTION DOC] Stores the given node object (any of the symbolic node kinds) in the attribute
        # named node. No validation or copying is done.
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = typing.cast("SymNode", node)

    def __bool__(self) -> builtins.bool:
        # [FUNCTION DOC] Truthiness: compares against zero (producing a symbolic bool) and converts it to
        # a concrete bool, which installs a guard.
        return builtins.bool(self != 0)

    def __int__(self) -> builtins.int:
        # [FUNCTION DOC] Returns the concrete Python int by asking the node to evaluate (guarding).
        # pyrefly: ignore[missing-attribute]  # duck-typed: not on NestedIntNode/ConstantIntNode/LocalIntNode
        return self.node.int_()

    def __index__(self) -> builtins.int:
        # [FUNCTION DOC] Allows use as a sequence index; returns the concrete int from the node (guarding).
        # pyrefly: ignore[missing-attribute]  # duck-typed: not on NestedIntNode/ConstantIntNode/LocalIntNode
        return self.node.int_()

    # Magic methods installed by torch.fx.experimental.sym_node

    def __round__(self, ndigits: builtins.int | None = None) -> "SymInt":
        # [FUNCTION DOC] Rounding an integer is the identity; returns self and ignores ndigits.
        return self

    def __truediv__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] True division. A float/SymFloat operand converts self to SymFloat first; an
        # int/SymInt operand uses integer true division; any other type returns NotImplemented.
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__float_truediv__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__int_truediv__(other)

    def __rtruediv__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] Reflected true division (other divided by self); same dispatch rules as
        # __truediv__, returning NotImplemented for unsupported operand types.
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__rfloat_truediv__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__rint_truediv__(other)

    @_overload
    def __floordiv__(self, other: "builtins.int | SymInt") -> "SymInt": ...

    @_overload
    def __floordiv__(self, other: "builtins.float | SymFloat") -> "SymFloat": ...

    def __floordiv__(self, other: object) -> "SymInt | SymFloat":
        # [FUNCTION DOC] Floor division. With a float operand, divides as floats, floors, and returns a
        # SymFloat; with an int operand uses integer floor division; otherwise returns NotImplemented.
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(math.floor(sym_float(self) / other))
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__int_floordiv__(other)

    @_overload
    def __rfloordiv__(self, other: "builtins.int | SymInt") -> "SymInt": ...

    @_overload
    def __rfloordiv__(self, other: "builtins.float | SymFloat") -> "SymFloat": ...

    def __rfloordiv__(self, other: object) -> "SymInt | SymFloat":
        # [FUNCTION DOC] Reflected floor division (other floor-divided by self); same dispatch rules as
        # __floordiv__.
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(math.floor(other / sym_float(self)))
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        return self.__rint_floordiv__(other)

    # nb: complex is impossible to handle correctly lol, with
    # negative base and integral float need to diverge semantics and
    # just always return complex.  Neener neener pretend this problem
    # doesn't exist
    def __pow__(self, other: object) -> "SymInt | SymFloat":
        # [FUNCTION DOC] Exponentiation. Float exponent: float power. Int exponent: guards on whether it
        # is non-negative; non-negative gives an integer power, negative promotes both to float and does
        # a float power (matching Python). Other types return NotImplemented.
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__pow__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        # Guards!  This guard is necessary because we need to know it to
        # determine the output type of this operation
        if other >= 0:
            return self.__pow_by_natural__(other)
        else:
            # Mercifully, when the exponent is negative, Python just promotes
            # to doubles and does a float pow:
            #
            #   if (Py_SIZE(b) < 0 && c == NULL) {
            #       /* if exponent is negative and there's no modulus:
            #              return a float.  This works because we know
            #              that this calls float_pow() which converts its
            #              arguments to double. */
            #       Py_DECREF(a);
            #       Py_DECREF(b);
            #       return PyFloat_Type.tp_as_number->nb_power(v, w, x);
            #   }
            return sym_float(self).__pow__(sym_float(other))

    def __rpow__(self, other: object) -> "SymInt | SymFloat":
        # [FUNCTION DOC] Reflected exponentiation where self is the exponent and other the base. Guards
        # on self being non-negative: integer power if so, otherwise float power. Float base uses float
        # power; other types return NotImplemented.
        if isinstance(other, (builtins.float, SymFloat)):
            return sym_float(self).__rpow__(other)
        if not isinstance(other, (builtins.int, SymInt)):
            return NotImplemented  # type: ignore[return-value]
        if self >= 0:  # self is exponent
            return self.__rpow_by_natural__(other)
        else:
            return sym_float(self).__rpow__(sym_float(other))

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the node's graph-style string form of the symbolic expression.
        return self.node._graph_repr()

    def _sympy_(self) -> "sympy.Basic":
        # [FUNCTION DOC] HELPER: Sympy conversion hook; returns the node's underlying sympy expression.
        # pyrefly: ignore[missing-attribute]  # duck-typed: not on NestedIntNode/ConstantIntNode/LocalIntNode
        return self.node.expr

    def __hash__(self) -> builtins.int:
        # [FUNCTION DOC] Hashes only nested ints (using the nested int identity); raises TypeError
        # (unhashable) for every other SymInt on purpose, because libraries like einops rely on that.
        if self.node.is_nested_int():
            # pyrefly: ignore[missing-attribute]  # narrowed by is_nested_int(); only NestedIntNode/SymNode reach here
            return hash(self.node.nested_int())
        else:
            # We could support constant SymInts as well, but not doing it for now
            raise TypeError("unhashable type: non-nested SymInt")
            # TODO: Force specialization
            # This can't be done because the TypeError here is load bearing
            # for einops
            # https://github.com/arogozhnikov/einops/blob/6181e1e95dc58c00a3143c1726da1c6ee0463164/einops/einops.py#L237
            # return hash(builtins.int(self))

    def as_integer_ratio(self) -> tuple["SymInt", builtins.int]:
        # [FUNCTION DOC] Returns the pair (self, 1) as the exact integer ratio.
        """Represent this int as an exact integer ratio"""
        return self, 1

    def bit_length(self) -> builtins.int:
        # [FUNCTION DOC] Returns the bit length of the concrete int value (forces specialization/guard).
        # TODO: A more relaxed guard is possible here, where you guard to
        # allow all integer quantities which would result in the same bit
        # length.  We can also just make a dedicated Sympy function for
        # computing this quantity and represent it symbolically.
        return builtins.int(self).bit_length()

    def conjugate(self) -> "SymInt":
        # [FUNCTION DOC] Complex conjugate of an integer is itself; returns self.
        return self

    def has_hint(self) -> builtins.bool:
        # [FUNCTION DOC] Returns whether the node carries a concrete hint value.
        return self.node.has_hint()

    @property
    def hint(self) -> builtins.int | None:
        # [FUNCTION DOC] Property: the node's concrete hint value as an int, or None when unavailable.
        return typing.cast(builtins.int | None, self.node.hint)

    @property
    def constant(self) -> builtins.int | None:
        # [FUNCTION DOC] Property: the constant int value if the node is a constant, else None.
        return typing.cast(builtins.int | None, self.node.constant)


class SymFloat(_SymTypingMagic[builtins.float, "SymFloat", _Never]):
    # [CLASS DOC] Symbolic float: wraps a node (attribute node, name fixed for C++ bindings) and
    # forwards operations to it for symbolic shape tracing. Arithmetic magic methods are installed by
    # sym_node; this class defines division, floor division, power (requires non-negative base via a
    # check), conversions, repr, hashing via the concrete float, and hint/constant accessors.
    """
    Like a float (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.
    """

    def __init__(self, node: "SymNode") -> None:
        # [FUNCTION DOC] Stores the given SymNode in the attribute named node (name required by C++).
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __truediv__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] True division by an int/float/Sym operand (converted to SymFloat); returns
        # NotImplemented for other types.
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return self.__float_truediv__(sym_float(other))

    def __rtruediv__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] Reflected true division (other divided by self); NotImplemented for other types.
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return self.__rfloat_truediv__(sym_float(other))

    def __floordiv__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] Floor division implemented as float division followed by floor, returned as a
        # SymFloat; NotImplemented for unsupported operand types.
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return sym_float(math.floor(self / sym_float(other)))

    def __rfloordiv__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] Reflected floor division (other floor-divided by self), returned as SymFloat.
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        return sym_float(math.floor(sym_float(other) / self))

    def __bool__(self) -> builtins.bool:
        # [FUNCTION DOC] Truthiness via the node's concrete boolean evaluation (installs a guard).
        return self.node.bool_()

    def __float__(self) -> builtins.float:
        # [FUNCTION DOC] Returns the concrete Python float by guarding on the node's value.
        return self.node.guard_float("", 0)

    def __int__(self) -> builtins.int:
        # [FUNCTION DOC] Truncates toward zero symbolically, then converts the result to a concrete int.
        return self.__trunc__().__int__()

    # Symbolic power does NOT work with negative base, this is to avoid
    # potential complex outputs
    def __pow__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] Power with a numeric exponent. Asserts via a check that the base is non-negative
        # (to avoid complex results), then uses float power. NotImplemented for other operand types.
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        torch._check(self >= 0)
        return self.__float_pow__(other)

    def __rpow__(self, other: object) -> "SymFloat":
        # [FUNCTION DOC] Reflected power where other is the base; checks that the base is non-negative,
        # then uses reflected float power. NotImplemented for other operand types.
        if not isinstance(other, (builtins.int, builtins.float, SymInt, SymFloat)):
            return NotImplemented  # type: ignore[return-value]
        torch._check(other >= 0)
        return self.__rfloat_pow__(other)

    # Magic methods installed by torch.fx.experimental.sym_node

    @_overload
    def __round__(self) -> "SymInt": ...

    @_overload
    def __round__(self, ndigits: "IntLikeType") -> "SymFloat": ...

    def __round__(self, ndigits: "IntLikeType | None" = None) -> "SymFloat | SymInt":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    # def __sym_int__(self) -> "SymInt":
    #     raise TypeError("type stub not overridden")

    def __sym_log2__(self) -> "SymFloat":
        # [FUNCTION DOC] HELPER: Typing stub; raises TypeError until overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def is_integer(self) -> builtins.bool:
        # [FUNCTION DOC] HELPER: Typing stub (docstring says it returns whether the float is integral);
        # raises TypeError until overridden by sym_node at runtime.
        """Return True if the float is an integer."""
        raise TypeError("type stub not overridden")

    def as_integer_ratio(self) -> tuple[builtins.int, builtins.int]:
        # [FUNCTION DOC] Returns the exact (numerator, denominator) ratio of the concrete float value
        # (forces a guard through float conversion).
        """Represent this float as an exact integer ratio"""
        return builtins.float(self).as_integer_ratio()

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the node's graph-style string form of the symbolic expression.
        return self.node._graph_repr()

    def _sympy_(self) -> "sympy.Basic":
        # [FUNCTION DOC] HELPER: Sympy conversion hook; returns the node's sympy expression.
        return self.node.expr

    def __hash__(self) -> builtins.int:
        # [FUNCTION DOC] Hashes the concrete float value (forces a guard).
        return hash(builtins.float(self))

    def conjugate(self) -> "SymFloat":
        # [FUNCTION DOC] Complex conjugate of a real float is itself; returns self.
        """Returns the complex conjugate of the float."""
        return self

    def hex(self) -> str:
        # [FUNCTION DOC] Returns the hexadecimal string of the concrete float value (guards).
        """Returns the hexadecimal representation of the float."""
        return self.node.guard_float("", 0).hex()

    def has_hint(self) -> builtins.bool:
        # [FUNCTION DOC] Returns whether the node carries a concrete hint value.
        return self.node.has_hint()

    @property
    def hint(self) -> builtins.float | None:
        # [FUNCTION DOC] Property: the node's concrete hint as a float, or None when unavailable.
        return typing.cast(builtins.float | None, self.node.hint)

    @property
    def constant(self) -> builtins.float | None:
        # [FUNCTION DOC] Property: the constant float value if the node is constant, else None.
        return typing.cast(builtins.float | None, self.node.constant)


class SymBool(
    _SymTypingMagicAlsoBool[builtins.bool, builtins.int, "SymInt", "FloatLikeType"],
    _SymTypingMagicBitwise["BoolLikeType"],
):
    # [CLASS DOC] Symbolic boolean: wraps a node (attribute node, name fixed for C++ bindings) and
    # forwards operations to it. Python truthiness converts to a concrete bool and installs a guard;
    # use bitwise operators and sym_not to stay symbolic. Provides sym_not/sym_ite stubs, repr, hashing,
    # float conversion and hint/constant accessors.
    """
    Like a bool (including magic methods), but redirects all operations on the
    wrapped node. This is used in particular to symbolically record operations
    in the symbolic shape workflow.

    Unlike regular bools, regular boolean operators will force extra guards instead
    of symbolically evaluate.  Use the bitwise operators instead to handle this.
    """

    def __init__(self, node: "SymNode") -> None:
        # [FUNCTION DOC] Stores the given SymNode in the attribute named node (name required by C++).
        # This field MUST be named node; C++ binding code assumes that this
        # class has a field named node that stores SymNode
        self.node = node

    def __bool__(self) -> builtins.bool:
        # [FUNCTION DOC] Concrete truth value from the node (installs a guard).
        return self.node.bool_()

    def __int__(self) -> builtins.int:
        # [FUNCTION DOC] Returns 1 or 0 from the concrete truth value (installs a guard).
        return builtins.int(self.node.bool_())

    # Magic methods installed by torch.fx.experimental.sym_node

    # We very carefully define __sym_not__, and not a number of other
    # plausible alternatives:
    #
    #   - We do not override __not__ because this is not a real magic
    #     method; you cannot override the meaning of the not builtin in
    #     Python.  We use the name 'sym_not' to clarify that in user code you
    #     cannot use the builtin not or operator.not_ or operator.__not__ and
    #     hit this magic method; you must use our custom sym_not operator.
    #
    #   - We do not override the __invert__ method because SymBool is
    #     meant to be usable in situations where bool is expected.  However,
    #     bitwise negation ~a does the wrong thing with booleans (because
    #     bool is a subclass of int, so ~1 = -2 which is not falseish.)
    #     This would be a giant footgun, so we get around it by defining
    #     our own operator.  Note that bitwise and/or do the right thing,
    #     so we reuse the conventional operators there for readability.
    #
    def __sym_not__(self) -> "SymBool":
        # [FUNCTION DOC] HELPER: Typing stub for symbolic logical negation; raises TypeError until
        # overridden by sym_node. Deliberately not named after the invert operator, which would
        # give wrong results on booleans.
        raise TypeError("type stub not overridden")

    def __sym_ite__(self, then_val: _SymIteT, else_val: _SymIteT) -> _SymIteT:
        # [FUNCTION DOC] HELPER: Typing stub for symbolic if-then-else; raises TypeError until
        # overridden by sym_node at runtime.
        raise TypeError("type stub not overridden")

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the node's graph-style string form of the symbolic expression.
        return self.node._graph_repr()

    def _sympy_(self) -> "sympy.Basic":
        # [FUNCTION DOC] HELPER: Sympy conversion hook; returns the node's sympy expression.
        return self.node.expr

    def __hash__(self) -> builtins.int:
        # [FUNCTION DOC] Hash of the constant truth value if the node is constant; otherwise forces
        # specialization by hashing the concrete bool (installs a guard).
        if self.node.is_constant():
            return hash(self.node.bool_())
        else:
            # Force specialization
            return hash(builtins.bool(self))

    def __sym_float__(self) -> "SymFloat":
        # [FUNCTION DOC] Returns a SymFloat (0.0 or 1.0) for this symbolic bool by wrapping the node's
        # float conversion; used by sym_float when casting.
        """
        Provides a SymFloat representation (0.0 or 1.0) for this SymBool.
        Called by torch.sym_float() when casting SymBool to float.
        """
        from torch.fx.experimental.sym_node import wrap_node

        # pyrefly: ignore[bad-return]
        return wrap_node(self.node.sym_float())

    @property
    def hint(self) -> builtins.bool | None:
        # [FUNCTION DOC] Property: the node's concrete hint as a bool, or None when unavailable.
        return typing.cast(builtins.bool | None, self.node.hint)

    @property
    def constant(self) -> builtins.bool | None:
        # [FUNCTION DOC] Property: the constant bool value if the node is constant, else None.
        return typing.cast(builtins.bool | None, self.node.constant)


@_overload
def sym_not(a: "BoolLikeType") -> "BoolLikeType": ...


@_overload
def sym_not(a: object) -> object: ...


def sym_not(a: object) -> object:
    # [FUNCTION DOC] Logical negation that works on SymBool. Dispatches to torch-function overrides if
    # present, then to the object's symbolic-not method, then to bitwise invert for sympy expressions,
    # and otherwise uses plain boolean not. Returns the negated value (symbolic when possible).
    # Raises ImportError if sympy is missing.
    r"""SymInt-aware utility for logical negation.

    Args:
        a (SymBool or bool): Object to negate
    """
    import sympy

    if overrides.has_torch_function_unary(a):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_not, (a,), a)
    if hasattr(a, "__sym_not__"):
        return a.__sym_not__()  # type: ignore[attr-defined]
    if isinstance(a, sympy.Basic):
        return ~a  # type: ignore[operator]
    return not a


@_overload
def sym_float(a: "PySymType") -> "SymFloat": ...


@_overload
def sym_float(a: object) -> "FloatLikeType": ...


def sym_float(a: object) -> "FloatLikeType":
    # [FUNCTION DOC] Casts to float while preserving symbolic values: honors torch-function overrides,
    # returns a SymFloat unchanged, uses an object's symbolic float-cast method if it has one, else
    # falls back to the builtin float conversion (which may raise TypeError/ValueError).
    r"""SymInt-aware utility for float casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if overrides.has_torch_function_unary(a):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_float, (a,), a)
    if isinstance(a, SymFloat):
        return a
    elif hasattr(a, "__sym_float__"):
        return a.__sym_float__()  # type: ignore[attr-defined]
    return builtins.float(a)  # type: ignore[arg-type]


@_overload
def sym_int(a: "PySymType") -> "IntLikeType": ...


@_overload
def sym_int(a: object) -> builtins.int: ...


def sym_int(a: object) -> "IntLikeType":
    # [FUNCTION DOC] Casts to int while preserving symbolic values: honors torch-function overrides,
    # returns a SymInt unchanged, truncates a SymFloat toward zero, else uses the builtin int
    # conversion (which may raise TypeError/ValueError).
    r"""SymInt-aware utility for int casting.

    Args:
        a (SymInt, SymFloat, or object): Object to cast
    """
    if overrides.has_torch_function_unary(a):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_int, (a,), a)
    if isinstance(a, SymInt):
        return a
    elif isinstance(a, SymFloat):
        return math.trunc(a)
    return builtins.int(a)  # type: ignore[arg-type]


@_overload
def sym_max(a: "IntLikeType", b: "IntLikeType") -> "IntLikeType": ...


@_overload
def sym_max(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "FloatLikeType": ...


def sym_max(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "IntLikeType | FloatLikeType":
    # [FUNCTION DOC] Maximum of two int/float values without branching on symbolic comparisons.
    # Honors torch-function overrides; if either side is symbolic, delegates to its symbolic max
    # method. Otherwise requires both to be Python or numpy int/float (raises AssertionError
    # otherwise) and returns a float if either is a float, else the builtin max.
    """
    SymInt-aware utility for max which avoids branching on a < b.
    Unlike builtins.max(), this only works for int/float, and it always
    promotes to float if any argument is float (unlike builtins.max, which
    will faithfully preserve the type of the input argument).
    """
    if overrides.has_torch_function((a, b)):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_max, (a, b), a, b)
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_max__(b)  # type: ignore[return-value]
    elif isinstance(b, (SymInt, SymFloat)):
        # Due to promotion semantics, this is operator is commutative:
        # max(1, 1.0) === max(1.0, 1) === 1.0
        return b.__sym_max__(a)  # type: ignore[return-value]
    # TODO: Probably can make bool work too, just lazy

    all_types, float_types = __all_and_float_types()

    if not isinstance(a, all_types):
        raise AssertionError(f"expected {all_types}, got {type(a)}")
    if not isinstance(b, all_types):
        raise AssertionError(f"expected {all_types}, got {type(b)}")
    if isinstance(a, float_types) or isinstance(b, float_types):
        return builtins.float(builtins.max(a, b))  # type: ignore[call-overload]
    else:
        return builtins.max(a, b)  # type: ignore[call-overload]


def __all_and_float_types() -> tuple[tuple[type, ...], tuple[type, ...]]:
    # [FUNCTION DOC] HELPER: Returns a pair of type tuples (all accepted numeric types, float types):
    # Python int/float, plus numpy integer/floating types when numpy is importable.
    try:
        import numpy as np

        all_types: tuple[type, ...] = (
            np.integer,
            np.floating,
            builtins.int,
            builtins.float,
        )
        float_types: tuple[type, ...] = (np.floating, builtins.float)
    except ModuleNotFoundError:
        all_types = (builtins.int, builtins.float)
        float_types = (builtins.float,)

    return all_types, float_types


@_overload
def sym_min(a: "IntLikeType", b: "IntLikeType") -> "IntLikeType": ...


@_overload
def sym_min(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "FloatLikeType": ...


def sym_min(
    a: "IntLikeType | FloatLikeType", b: "IntLikeType | FloatLikeType"
) -> "IntLikeType | FloatLikeType":
    # [FUNCTION DOC] Minimum of two int/float values without branching on symbolic comparisons.
    # Same dispatch and type rules as sym_max: overrides first, then symbolic min method, then an
    # AssertionError for non-numeric inputs, float result if either input is a float.
    """SymInt-aware utility for min()."""
    if overrides.has_torch_function((a, b)):
        # pyrefly: ignore[bad-argument-type]
        return overrides.handle_torch_function(sym_min, (a, b), a, b)
    if isinstance(a, (SymInt, SymFloat)):
        return a.__sym_min__(b)  # type: ignore[return-value]
    elif isinstance(b, (SymInt, SymFloat)):
        return b.__sym_min__(a)  # type: ignore[return-value]

    all_types, float_types = __all_and_float_types()

    if not isinstance(a, all_types):
        raise AssertionError(f"expected {all_types}, got {type(a)}")
    if not isinstance(b, all_types):
        raise AssertionError(f"expected {all_types}, got {type(b)}")
    if isinstance(a, float_types) or isinstance(b, float_types):
        return builtins.float(builtins.min(a, b))  # type: ignore[call-overload]
    else:
        return builtins.min(a, b)  # type: ignore[call-overload]


def sym_sum(
    *args: "IntLikeType | _Sequence[IntLikeType]",
) -> "IntLikeType":
    # [FUNCTION DOC] Sums many ints in one n-ary symbolic add. Accepts either varargs or a single
    # list/tuple. Honors torch-function overrides. If any item is not an int/SymInt, or none is a
    # SymInt, returns the builtin sum; otherwise converts items to nodes and returns a wrapped
    # symbolic sum.
    """
    N-ary add which is faster to compute for long lists than iterated binary
    addition.  Only does something special for integers.

    Accepts both ``sym_sum([a, b, c])`` and ``sym_sum(a, b, c)``.
    """
    # Normalise: accept both sym_sum([a, b, c]) and sym_sum(a, b, c).
    items: _Sequence[IntLikeType] = args  # type: ignore[assignment]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        items = args[0]

    if overrides.has_torch_function(items):
        return overrides.handle_torch_function(sym_sum, items, items)

    found = None
    for a in items:
        if not isinstance(a, (SymInt, builtins.int)):
            return builtins.sum(items)
        if isinstance(a, SymInt):
            found = a.node
    if found is None:
        return builtins.sum(items)

    from torch.fx.experimental.sym_node import to_node, wrap_node

    # pyrefly: ignore[missing-attribute, bad-argument-type, bad-return]  # found is a SymNode in practice (LocalIntNode also has sym_sum)
    return wrap_node(found.sym_sum([to_node(found, a) for a in items]))


# Drop in replacement for math.sqrt, math.sin, math.cos etc
def _get_sym_math_fn(name: str) -> _Callable[[_Any], _Any]:
    # [FUNCTION DOC] HELPER: Factory returning a replacement for the math module function with the
    # given name that understands symbolic numbers. Used at import time to create the module-level
    # sym math functions; deleted afterwards.
    def fn(a: _Any) -> _Any:
        # [FUNCTION DOC] HELPER: Applies the named math function: honors torch-function overrides,
        # casts SymInt to SymFloat, uses the argument's symbolic method for that function if present,
        # otherwise calls the standard math function on the plain value.
        if overrides.has_torch_function_unary(a):
            return overrides.handle_torch_function(fn, (a,), a)
        if isinstance(a, SymInt):
            a = torch.sym_float(a)
        if hasattr(a, f"__sym_{name}__"):
            return getattr(a, f"__sym_{name}__")()
        return getattr(math, name)(a)

    return fn


__fn, __name, __sym_name = None, "", ""
for __name in (
    "sqrt",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "asin",
    "acos",
    "atan",
    "log2",
):
    __sym_name = f"_sym_{__name}"
    __fn = _get_sym_math_fn(__name)
    __fn.__qualname__ = __fn.__name__ = __sym_name
    globals()[__sym_name] = __fn


del __fn, __name, __sym_name, _get_sym_math_fn

# Adding temporary shortcut
sym_sqrt = globals()["_sym_sqrt"]
__all__.append("sym_sqrt")


def sym_ite(b: "BoolLikeType", t: _SymIteT, f: _SymIteT) -> _SymIteT:
    # [FUNCTION DOC] Symbolic ternary: returns t if b is true else f. Honors torch-function overrides.
    # Raises AssertionError if b is not a SymBool or bool, or if t and f have different types. A
    # SymBool condition delegates to its symbolic if-then-else; a plain bool selects directly.
    """SymInt-aware utility for ternary operator (``t if b else f``.)"""
    if overrides.has_torch_function((b, t, f)):
        return overrides.handle_torch_function(sym_ite, (b, t, f), b, t, f)
    if not isinstance(b, (SymBool, builtins.bool)):
        raise AssertionError(f"expected SymBool or bool, got {type(b)}")
    if type(t) is not type(f):
        raise AssertionError(f"type mismatch: {type(t)} vs {type(f)}")
    if isinstance(b, SymBool):
        return b.__sym_ite__(t, f)
    return t if b else f


# Create a fresh unbacked int, from an (possibly unbacked int) expression.
def sym_fresh_size(expr: "IntLikeType") -> "IntLikeType":
    # [FUNCTION DOC] Creates a fresh unbacked symbolic int from an expression by building a tensor
    # from it and reading back its item. Returns the resulting int-like value.
    return typing.cast("IntLikeType", torch.tensor(expr).item())


# Check to see if we can load C extensions, and if not provide some guidance
# on what the problem might be.
try:
    # _initExtension is chosen (arbitrarily) as a sentinel.
    from torch._C import _initExtension
except ImportError:
    import torch._C as _C_for_compiled_check

    if _C_for_compiled_check.__file__ is None:
        raise ImportError(
            textwrap.dedent(
                """
                Failed to load PyTorch C extensions:
                    It appears that PyTorch has loaded the `torch/_C` folder
                    of the PyTorch repository rather than the C extensions which
                    are expected in the `torch._C` namespace. This can occur when
                    using the `install` workflow. e.g.
                        $ python -m pip install --no-build-isolation -v . && python -c "import torch"

                    This error can generally be solved using the `develop` workflow
                        $ python -m pip install --no-build-isolation -v -e . && python -c "import torch"  # This should succeed
                    or by running Python from a different directory.
                """
            ).strip()
        ) from None
    raise  # If __file__ is not None the cause is unknown, so just re-raise.

# The torch._C submodule is already loaded via `from torch._C import *` above
# Make an explicit reference to the _C submodule to appease linters
from torch import _C as _C


__name, __obj = "", None
for __name in dir(_C):
    if __name[0] != "_" and not __name.endswith("Base"):
        __all__.append(__name)
        __obj = getattr(_C, __name)
        if callable(__obj) or inspect.isclass(__obj):
            if __obj.__module__ != __name__:  # "torch"
                # TODO: fix their module from C++ side
                if __name not in {
                    "DisableTorchFunctionSubclass",
                    "DisableTorchFunction",
                    "Generator",
                }:
                    __obj.__module__ = __name__  # "torch"
    elif __name == "TensorBase":
        # issue 109438 / pr 109940. Prevent TensorBase from being copied into torch.
        delattr(sys.modules[__name__], __name)

del __name, __obj

if not TYPE_CHECKING:
    # issue 38137 and python issue 43367. Submodules of a C extension are
    # non-standard, and attributes of those submodules cannot be pickled since
    # pickle expect to be able to import them as "from _C.sub import attr"
    # which fails with "_C is not a package
    def _import_extension_to_sys_modules(module, memo=None):
        # [FUNCTION DOC] HELPER: Recursively walks a C extension module's attributes and registers
        # every submodule whose name starts with the parent's name in sys.modules (without
        # overriding existing entries), so attributes of those submodules can be pickled. The memo
        # set prevents revisiting modules. Side effect: mutates sys.modules.
        if memo is None:
            memo = set()
        if module in memo:
            return
        memo.add(module)
        module_name = module.__name__
        for name in dir(module):
            member = getattr(module, name)
            member_name = getattr(member, "__name__", "")
            if inspect.ismodule(member) and member_name.startswith(module_name):
                sys.modules.setdefault(member_name, member)
                # Recurse for submodules (e.g., `_C._dynamo.eval_frame`)
                _import_extension_to_sys_modules(member, memo)

    _import_extension_to_sys_modules(_C)
    del _import_extension_to_sys_modules

################################################################################
# Define basic utilities
################################################################################


def typename(obj: object, /) -> str:
    # [FUNCTION DOC] Returns a fully qualified type name string. For a tensor returns its tensor type
    # string; otherwise uses the object's module and qualified (or plain) name, falling back to its
    # class when it has neither. Builtins and empty modules yield the bare name.
    """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """
    if isinstance(obj, torch.Tensor):
        return obj.type()

    module = getattr(obj, "__module__", "") or ""
    qualname = ""

    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__  # type: ignore[attr-defined]
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__  # type: ignore[attr-defined]
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__

    if module in {"", "builtins"}:
        return qualname
    return f"{module}.{qualname}"


def is_tensor(obj: object, /) -> _TypeIs["torch.Tensor"]:
    # [FUNCTION DOC] Returns whether obj is an instance of Tensor (type-narrowing predicate).
    r"""Returns True if `obj` is a PyTorch tensor.

    Args:
        obj (object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True

    """
    return isinstance(obj, torch.Tensor)


def is_storage(obj: object, /) -> _TypeGuard["TypedStorage | UntypedStorage"]:
    # [FUNCTION DOC] Returns whether the exact type of obj is one of the registered storage classes
    # (module-level collection defined later). Subclasses of storage classes are not matched.
    r"""Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    Example::

        >>> import torch
        >>> # UntypedStorage (recommended)
        >>> tensor = torch.tensor([1, 2, 3])
        >>> storage = tensor.untyped_storage()
        >>> torch.is_storage(storage)
        True
        >>>
        >>> # TypedStorage (legacy)
        >>> warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated")  # docs: hide
        >>> typed_storage = torch.TypedStorage(5, dtype=torch.float32)
        >>> torch.is_storage(typed_storage)
        True
        >>>
        >>> # regular tensor (should return False)
        >>> torch.is_storage(tensor)
        False
        >>>
        >>> # non-storage object
        >>> torch.is_storage([1, 2, 3])
        False
    """
    return type(obj) in _storage_classes


_GLOBAL_DEVICE_CONTEXT = threading.local()


def get_default_device() -> "torch.device":
    # [FUNCTION DOC] Returns the current default device for new tensors. The innermost active device
    # context manager wins; otherwise the global device set by set_default_device; otherwise cpu.
    # If the chosen device has no index, the index is resolved by creating an empty tensor.
    r"""Gets the default ``torch.Tensor`` to be allocated on ``device``"""
    global _GLOBAL_DEVICE_CONTEXT

    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext

    def _get_device_with_index(device: "torch.device") -> "torch.device":
        # [FUNCTION DOC] HELPER: Returns the device unchanged if it has an index, else the device of a
        # freshly created empty tensor (which fills in the current index).
        if device.index is not None:
            return device
        else:
            # TODO: Call like get_device_index() method corresponding to
            # each device type
            return torch.tensor([]).device

    # Get device from any active DeviceContext.
    device_mode = next(
        filter(
            lambda mode: isinstance(mode, DeviceContext),
            reversed(_get_current_function_mode_stack()),
        ),
        None,
    )
    if device_mode:
        device = device_mode.device
        return _get_device_with_index(device)

    device_context = getattr(_GLOBAL_DEVICE_CONTEXT, "device_context", None)
    if device_context is not None:
        return _get_device_with_index(device_context.device)
    return torch.device("cpu")


def set_default_device(device: "Device") -> None:
    # [FUNCTION DOC] Sets (or, when None, clears) the process-wide default device. Exits any previously
    # installed global device context, then enters a new one for the given device and stores it in the
    # thread-local global slot. Side effects: global state mutation, adds per-call overhead to torch API.
    # KNOWN ISSUE: the stored context lives in thread-local storage, so the default only applies to the
    # calling thread despite the docstring calling it global.
    """Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    does not affect factory function calls which are called with an explicit
    ``device`` argument.  Factory calls will be performed as if they
    were passed ``device`` as an argument.

    To only temporarily change the default device instead of setting it
    globally, use ``with torch.device(device):`` instead.

    The default device is initially ``cpu``.  If you set the default tensor
    device to another device (e.g., ``cuda``) without a device index, tensors
    will be allocated on whatever the current device for the device type,
    even after :func:`torch.cuda.set_device` is called.

    .. warning::

        This function imposes a slight performance cost on every Python
        call to the torch API (not just factory functions).  If this
        is causing problems for you, please comment on
        https://github.com/pytorch/pytorch/issues/92701

    .. note::

        This doesn't affect functions that create tensors that share the same memory as the input, like:
        :func:`torch.from_numpy` and :func:`torch.frombuffer`

    Args:
        device (:class:`torch.device`, str, int, or None): the device to set as
            default, or ``None`` to clear the override. An integer is
            interpreted as an index for the current accelerator.

    Example::

        >>> # xdoctest: +SKIP("requires cuda, changes global state")
        >>> torch.get_default_device()
        device(type='cpu')
        >>> torch.set_default_device('cuda')  # current device is 0
        >>> torch.get_default_device()
        device(type='cuda', index=0)
        >>> torch.set_default_device('cuda')
        >>> torch.cuda.set_device('cuda:1')  # current device is 1
        >>> torch.get_default_device()
        device(type='cuda', index=1)
        >>> torch.set_default_device('cuda:1')
        >>> torch.get_default_device()
        device(type='cuda', index=1)

    """
    global _GLOBAL_DEVICE_CONTEXT
    if hasattr(_GLOBAL_DEVICE_CONTEXT, "device_context"):
        device_context = _GLOBAL_DEVICE_CONTEXT.device_context
        if device_context is not None:
            device_context.__exit__(None, None, None)

    if device is None:
        device_context = None
    else:
        from torch.utils._device import DeviceContext

        device_context = DeviceContext(device)
        device_context.__enter__()
    _GLOBAL_DEVICE_CONTEXT.device_context = device_context


def set_default_tensor_type(t: type["torch.Tensor"] | str, /) -> None:
    # [FUNCTION DOC] Deprecated. Sets the default floating-point tensor type; a string is first resolved
    # to a type by dotted-name import. Delegates to the C extension. Side effect: global default dtype
    # and type change. Raises on unresolvable names or invalid types.
    r"""
    .. warning::

        This function is deprecated as of PyTorch 2.1, please use :func:`torch.set_default_dtype()` and
        :func:`torch.set_default_device()` as alternatives.

    Sets the default ``torch.Tensor`` type to floating point tensor type
    ``t``. This type will also be used as default floating point type for
    type inference in :func:`torch.tensor`.

    The default floating point tensor type is initially ``torch.FloatTensor``.

    Args:
        t (type or string): the floating point tensor type or its name

    Example::

        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
        torch.float32
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
        torch.float64

    """
    if isinstance(t, str):
        t = _import_dotted_name(t)
    _C._set_default_tensor_type(t)


def set_default_dtype(d: "torch.dtype", /) -> None:
    # [FUNCTION DOC] Sets the global default floating-point dtype (and implied complex dtype) via the
    # C extension. Non-floating dtypes raise an error. Side effect: changes global type inference.
    r"""

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)
        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex128

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex32

    """
    _C._set_default_dtype(d)


def use_deterministic_algorithms(
    mode: builtins.bool,
    *,
    warn_only: builtins.bool = False,
) -> None:
    # [FUNCTION DOC] Globally enables or disables deterministic-algorithm mode. Also sets the inductor
    # config deterministic flag to the same value, then forwards the mode and warn_only flag to the C
    # extension. With warn_only, nondeterministic ops warn instead of raising RuntimeError.
    # Side effects: global state (core and inductor config).
    r"""Sets whether PyTorch operations must use "deterministic"
    algorithms. That is, algorithms which, given the same input, and when
    run on the same software and hardware, always produce the same output.
    When enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    :class:`RuntimeError` when called.

    .. note:: This setting alone is not always enough to make an application
        reproducible. Refer to :ref:`reproducibility` for more information.

    .. note:: :func:`torch.set_deterministic_debug_mode` offers an alternative
        interface for this feature.

    The following normally-nondeterministic operations will act
    deterministically when ``mode=True``:

        * :class:`torch.nn.Conv1d` when called on CUDA tensor
        * :class:`torch.nn.Conv2d` when called on CUDA tensor
        * :class:`torch.nn.Conv3d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose1d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose2d` when called on CUDA tensor
        * :class:`torch.nn.ConvTranspose3d` when called on CUDA tensor
        * :class:`torch.nn.ReplicationPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReplicationPad3d` when attempting to differentiate a CUDA tensor
        * :func:`torch.bmm` when called on sparse-dense CUDA tensors
        * :func:`torch.cumsum` when called on a CUDA tensor when dtype is floating point or complex
        * :func:`torch.Tensor.__getitem__` when attempting to differentiate a CPU tensor
          and the index is a list of tensors
        * :func:`torch.Tensor.index_put` with ``accumulate=False``
        * :func:`torch.Tensor.index_put` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.put_` with ``accumulate=True`` when called on a CPU
          tensor
        * :func:`torch.Tensor.scatter_add_` when called on a CUDA tensor
        * :func:`torch.gather` when called on a CUDA tensor that requires grad
        * :func:`torch.index_add` when called on CUDA tensor
        * :func:`torch.index_select` when attempting to differentiate a CUDA tensor
        * :func:`torch.repeat_interleave` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.index_copy` when called on a CPU or CUDA tensor
        * :func:`torch.Tensor.scatter` when `src` type is Tensor and called on CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='sum'`` or ``reduce='mean'`` and called on CUDA tensor
        * :class:`torch.nn.MaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.Embedding` when attempting to differentiate a CUDA tensor

    The following normally-nondeterministic operations will throw a
    :class:`RuntimeError` when ``mode=True``:

        * :class:`torch.nn.AvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveAvgPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.AdaptiveMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.FractionalMaxPool3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.MaxUnpool1d`
        * :class:`torch.nn.MaxUnpool2d`
        * :class:`torch.nn.MaxUnpool3d`
        * :func:`torch.nn.functional.interpolate` when attempting to differentiate a CUDA tensor
          and one of the following modes is used:

          - ``linear``
          - ``bilinear``
          - ``bicubic``
          - ``trilinear``

        * :class:`torch.nn.ReflectionPad1d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad2d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.ReflectionPad3d` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.NLLLoss` when called on a CUDA tensor
        * :class:`torch.nn.CTCLoss` when attempting to differentiate a CUDA tensor
        * :class:`torch.nn.EmbeddingBag` when attempting to differentiate a CUDA tensor when
          ``mode='max'``
        * :func:`torch.Tensor.put_` when ``accumulate=False``
        * :func:`torch.Tensor.put_` when ``accumulate=True`` and called on a CUDA tensor
        * :func:`torch.histc` when called on a CUDA tensor
        * :func:`torch.bincount` when called on a CUDA tensor and ``weights``
          tensor is given
        * :func:`torch.median` with indices output when called on a CUDA tensor
        * :func:`torch.nn.functional.grid_sample` when attempting to differentiate a CUDA tensor
        * :func:`torch.Tensor.scatter_reduce` when ``reduce='prod'`` and called on CUDA tensor
        * :func:`torch.Tensor.resize_` when called with a quantized tensor

    In addition, several operations fill uninitialized memory when this setting
    is turned on and when
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` is turned on.
    See the documentation for that attribute for more information.

    Note that deterministic operations tend to have worse performance than
    nondeterministic operations.


    When this setting is turned on, the Inductor deterministic mode is also tuned on
    automatically. In deterministic mode, Inductor would avoid doing on device benchmarking
    that affect numerics. This includes:

      - don't pad matmul input shapes. Without enabling deterministic mode, Inductor would do
        benchmarking to check if padding matmul shape is beneficial.
      - don't autotune templates. Inductor has templates for kernels like matmul/conv/attention.
        Without enabling deterministic mode, Inductor would do autotuning to
        pick the best configs for those templates and adopt it if it's faster
        than the kernel in eager mode. In deterministic mode, we pick the eager kernel.
      - don't autotune triton configs for reduction. Reduction numerics are
        very sensitive to triton configs. In deterministic mode, Inductor
        will use some heuristics to pick the most promising configs rather
        than do autotuning.
      - Skip autotuning for reduction in coordinate descent tuning.
      - Don't benchmark for the computation/communication reordering pass
      - Disable the feature that dynamically scale down RBLOCK triton config for higher
        occupancy.


    .. note::

        This flag does not detect or prevent nondeterministic behavior caused
        by calling an inplace operation on a tensor with an internal memory
        overlap or by giving such a tensor as the :attr:`out` argument for an
        operation. In these cases, multiple writes of different data may target
        a single memory location, and the order of writes is not guaranteed.

    Args:
        mode (:class:`bool`): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw a runtime
            error. If False, allows nondeterministic operations.

    Keyword args:
        warn_only (:class:`bool`, optional): If True, operations that do not
            have a deterministic implementation will throw a warning instead of
            an error. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> torch.use_deterministic_algorithms(True)

        # Backward mode nondeterministic error
        >>> torch.nn.AvgPool3d(1)(torch.randn(3, 4, 5, 6, requires_grad=True).cuda()).sum().backward()
        ...
        RuntimeError: avg_pool3d_backward_cuda does not have a deterministic implementation...
    """
    import torch._inductor.config as inductor_config

    inductor_config.deterministic = mode
    _C._set_deterministic_algorithms(mode, warn_only=warn_only)


def are_deterministic_algorithms_enabled() -> builtins.bool:
    # [FUNCTION DOC] Returns whether deterministic algorithms mode is currently on (reads C extension state).
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return _C._get_deterministic_algorithms()


def is_deterministic_algorithms_warn_only_enabled() -> builtins.bool:
    # [FUNCTION DOC] Returns whether deterministic mode is in warn-only mode (reads C extension state).
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """
    return _C._get_deterministic_algorithms_warn_only()


def set_deterministic_debug_mode(debug_mode: builtins.int | str) -> None:
    # [FUNCTION DOC] Sets deterministic mode via a level: default/0 = off, warn/1 = on with warn-only,
    # error/2 = on with errors. Raises TypeError if the argument is neither int nor str, RuntimeError for
    # unknown strings or ints outside 0-2. Side effect: sets the C extension global flags.
    # KNOWN ISSUE: unlike use_deterministic_algorithms, this does not update the inductor deterministic
    # config flag.
    r"""Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """

    # NOTE: builtins.int is used here because int in this scope resolves
    # to torch.int
    if not isinstance(debug_mode, (builtins.int, str)):
        raise TypeError(f"debug_mode must be str or int, but got {type(debug_mode)}")

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, "
                f"`warn`, `error`, but got {debug_mode}"
            )

    if debug_mode == 0:
        _C._set_deterministic_algorithms(False)
    elif debug_mode == 1:
        _C._set_deterministic_algorithms(True, warn_only=True)
    elif debug_mode == 2:
        _C._set_deterministic_algorithms(True)
    else:
        raise RuntimeError(
            f"invalid value of debug_mode, expected 0, 1, or 2, but got {debug_mode}"
        )


def get_deterministic_debug_mode() -> builtins.int:
    # [FUNCTION DOC] Returns the current deterministic level: 0 when off, 1 when on in warn-only mode,
    # 2 when on and raising errors.
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """

    if _C._get_deterministic_algorithms():
        if _C._get_deterministic_algorithms_warn_only():
            return 1
        else:
            return 2
    else:
        return 0


def get_float32_matmul_precision() -> str:
    # [FUNCTION DOC] Returns the current float32 matmul precision string (highest, high or medium).
    r"""Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """
    return _C._get_float32_matmul_precision()


def set_float32_matmul_precision(precision: str) -> None:
    # [FUNCTION DOC] Sets the internal precision used for float32 matrix multiplication (highest, high
    # or medium) by forwarding to the C extension, which validates the string. Side effect: global state.
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """
    _C._set_float32_matmul_precision(precision)


def set_warn_always(b: builtins.bool, /) -> None:
    # [FUNCTION DOC] Sets the global flag that forces once-per-process warnings to be emitted every
    # time when true. Side effect: global state in the C extension.
    r"""When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """
    _C._set_warnAlways(b)


def is_warn_always_enabled() -> builtins.bool:
    # [FUNCTION DOC] Returns the current value of the global warn-always flag.
    r"""Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
    return _C._get_warnAlways()


################################################################################
# Define error checking functions
################################################################################

# These error checking functions must be kept consistent with their C++
# equivalents. Their C++ equivalents are mentioned where applicable.


def _check_with(
    error_type: type[BaseException],
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None,
) -> None:
    # [FUNCTION DOC] HELPER: Core of the _check family. Validates cond is a bool or SymBool (raises
    # TypeError for tensors with a hint to use the tensor variant, or any other type). A condition
    # that is true, or that symbolic shape machinery can assume true, returns silently. Otherwise
    # builds the message (default text, or the string of the message, calling it first if callable)
    # and raises error_type with it. Raises AssertionError if error_type is not an Exception subclass
    # or is a Warning subclass.
    if not isinstance(cond, (builtins.bool, SymBool)):
        if isinstance(cond, torch.Tensor):
            raise TypeError(
                "cond must be a bool, but got a Tensor. "
                "For tensor conditions, use torch._check_tensor_all() "
                "to assert that all elements are true."
            )
        raise TypeError(f"cond must be a bool, but got {type(cond)}")

    from torch.fx.experimental.symbolic_shapes import expect_true

    if expect_true(cond):
        return

    # error_type must be a subclass of Exception and not subclass of Warning
    if not issubclass(error_type, Exception) or issubclass(error_type, Warning):
        raise AssertionError(
            f"error_type must be a subclass of Exception but not Warning, got {error_type}"
        )

    if message is None:
        message_evaluated = (
            "Expected cond to be True, but got False. (Could this error "
            "message be improved? If so, please report an enhancement request "
            "to PyTorch.)"
        )

    else:
        # message may be a callable (deferred construction) or a plain
        # str/object; both are accepted.
        message_evaluated = str(message() if callable(message) else message)

    raise error_type(message_evaluated)


def _check(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Raises RuntimeError with the optional message if cond is false (mirrors the
    # C++ check macro). Delegates to _check_with.
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``RuntimeError``

    C++ equivalent: ``TORCH_CHECK``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(RuntimeError, cond, message)


@_deprecated(
    "_check_is_size will be removed in a future PyTorch release along with guard_size_oblivious. \
    Use _check(i >= 0) instead.",
    category=FutureWarning,
)
def _check_is_size(
    i: "IntLikeType",
    message: _LiteralString | _Callable[[], object] | None = None,
    *,
    max: "IntLikeType | None" = None,
) -> None:
    # [FUNCTION DOC] HELPER: Deprecated (emits FutureWarning). Checks i is non-negative (raising
    # RuntimeError with message otherwise), tells the symbolic shape system that i is a size, and if max
    # is given also checks i <= max and records the upper bound. Side effect: records symbolic range
    # facts for unbacked symbols.
    """Checks that a given integer is a valid size (i.e., is non-negative).
    You should use this over ``_check(i >= 0)`` because it can prevent
    ``GuardOnDataDependentSymNode`` exceptions by opting yourself into alternate
    semantics for ``guard_size_oblivious`` tests that treat values 0 and 1
    equivalently to all other values.

    When max is not None, this specifies an upper bound equivalent to
    ``_check(i <= max)``.  This bound is also subject to alternate semantics:
    in ``guard_size_oblivious`` tests, we assume that a constant max bound is
    treated equivalently to all other values.  Symbolic max bounds are not yet
    supported.

    NB: Do NOT use this in contexts where a -1 size would be valid (indicating
    to infer the size from context, or if you should wrap-around or truncate).
    Only use this if the only valid value is an honest to goodness size.
    """
    # This is responsible for the expect_true
    _check(i >= 0, message)
    from torch.fx.experimental.symbolic_shapes import _advise_is_size

    _advise_is_size(i)  # type: ignore[arg-type]

    if max is not None:
        _check(i <= max, message)

        from torch.fx.experimental.symbolic_shapes import _advise_is_bounded

        _advise_is_bounded(i, max)  # type: ignore[arg-type]


def _check_index(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Raises IndexError with the optional message if cond is false (mirrors the
    # C++ index check macro). Delegates to _check_with.
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``IndexError``

    C++ equivalent: ``TORCH_CHECK_INDEX``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(IndexError, cond, message)


def _check_value(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Raises ValueError with the optional message if cond is false (mirrors the
    # C++ value check macro). Delegates to _check_with.
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``ValueError``

    C++ equivalent: ``TORCH_CHECK_VALUE``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(ValueError, cond, message)


def _check_type(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Raises TypeError with the optional message if cond is false (mirrors the
    # C++ type check macro). Delegates to _check_with.
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``TypeError``

    C++ equivalent: ``TORCH_CHECK_TYPE``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(TypeError, cond, message)


def _check_not_implemented(
    cond: builtins.bool | SymBool,
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Raises NotImplementedError with the optional message if cond is false
    # (mirrors the C++ not-implemented check macro). Delegates to _check_with.
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``NotImplementedError``

    C++ equivalent: ``TORCH_CHECK_NOT_IMPLEMENTED``

    Args:
        cond (:class:`bool`): If False, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_with(NotImplementedError, cond, message)


def _check_tensor_all_with(
    error_type: type[BaseException],
    cond: "torch.Tensor",
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Requires cond to be a boolean tensor (raises TypeError if not a tensor or
    # wrong dtype), reduces it to a single all-true value (a device sync via item), and raises
    # error_type with the message if any element is false.
    if not is_tensor(cond):
        raise TypeError(f"cond must be a tensor, but got {type(cond)}")

    if not cond.dtype == torch.bool:
        raise TypeError(f"cond tensor must have dtype torch.bool, but got {cond.dtype}")

    _check_with(error_type, cond._is_all_true().item(), message)  # type: ignore[arg-type]


# C++ equivalent: `TORCH_CHECK_TENSOR_ALL`
def _check_tensor_all(
    cond: "torch.Tensor",
    message: _LiteralString | _Callable[[], object] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Raises RuntimeError with the optional message unless every element of the
    # boolean tensor cond is true (mirrors the C++ tensor-all check macro).
    r"""Throws error containing an optional message if the specified condition
    is False.

    Error type: ``RuntimeError``

    C++ equivalent: ``TORCH_CHECK_TENSOR_ALL``

    Args:
        cond (:class:`torch.Tensor`): Tensor of dtype ``torch.bool``. If any
            element is ``False``, throw error

        message (LiteralString or Callable, optional): A literal string, or a
            callable that returns a string or an object with a ``__str__()``
            method, to be used as the error message. Use the callable form for a
            dynamically constructed message. Default: ``None``
    """
    _check_tensor_all_with(RuntimeError, cond, message)


################################################################################
# Define numeric constants
################################################################################

# For Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html) and
# NumPy consistency (https://numpy.org/devdocs/reference/constants.html)
from math import e, inf, nan, pi


newaxis: None = None

__all__.extend(["e", "pi", "nan", "inf", "newaxis"])

################################################################################
# Define Storage and Tensor classes
################################################################################

from torch._tensor import Tensor  # usort: skip

# needs to be after torch.Tensor is defined to avoid circular dependencies
from torch import storage as storage  # usort: skip
from torch.storage import (
    _LegacyStorage,
    _StorageBase,
    _warn_typed_storage_removal,
    TypedStorage,
    UntypedStorage,
)


# NOTE: New <type>Storage classes should never be added. When adding a new
# dtype, use torch.storage.TypedStorage directly.
class ByteStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the uint8 element type. No new storage classes should be added.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (uint8) without warning.
        return torch.uint8


class DoubleStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the float64 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (float64) without warning.
        return torch.double


class FloatStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the float32 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (float32) without warning.
        return torch.float


class HalfStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the float16 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (float16) without warning.
        return torch.half


class LongStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the int64 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (int64) without warning.
        return torch.long


class IntStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the int32 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (int32) without warning.
        return torch.int


class ShortStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the int16 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (int16) without warning.
        return torch.short


class CharStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the int8 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (int8) without warning.
        return torch.int8


class BoolStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the bool element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (bool) without warning.
        return torch.bool


class BFloat16Storage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the bfloat16 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (bfloat16) without warning.
        return torch.bfloat16


class ComplexDoubleStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the complex128 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (complex128) without warning.
        return torch.cdouble


class ComplexFloatStorage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the complex64 element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (complex64) without warning.
        return torch.cfloat


class QUInt8Storage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the quantized unsigned 8-bit element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (quint8) without warning.
        return torch.quint8


class QInt8Storage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the quantized signed 8-bit element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (qint8) without warning.
        return torch.qint8


class QInt32Storage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the quantized signed 32-bit element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (qint32) without warning.
        return torch.qint32


class QUInt4x2Storage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the packed quantized unsigned 4-bit (two per byte) element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (quint4x2) without warning.
        return torch.quint4x2


class QUInt2x4Storage(_LegacyStorage):
    # [CLASS DOC] Legacy typed storage fixed to the packed quantized unsigned 2-bit (four per byte) element type.
    @classproperty
    def dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] Class-level property: emits the typed-storage deprecation warning, then returns the element dtype.
        _warn_typed_storage_removal(stacklevel=3)
        return self._dtype

    @classproperty
    def _dtype(self) -> "torch.dtype":
        # [FUNCTION DOC] HELPER: Class-level property returning the fixed element dtype (quint2x4) without warning.
        return torch.quint2x4


_storage_classes: set[type[TypedStorage | UntypedStorage]] = {
    UntypedStorage,
    DoubleStorage,
    FloatStorage,
    LongStorage,
    IntStorage,
    ShortStorage,
    CharStorage,
    ByteStorage,
    HalfStorage,
    BoolStorage,
    QUInt8Storage,
    QInt8Storage,
    QInt32Storage,
    BFloat16Storage,
    ComplexFloatStorage,
    ComplexDoubleStorage,
    QUInt4x2Storage,
    QUInt2x4Storage,
    TypedStorage,
}

# The _tensor_classes set is initialized by the call to initialize_python_bindings.
_tensor_classes: set[type["torch.Tensor"]] = set()

# If you edit these imports, please update torch/__init__.py.in as well
from torch import amp as amp, random as random, serialization as serialization
from torch._tensor_str import set_printoptions
from torch.amp import autocast, GradScaler
from torch.random import (
    get_rng_state,
    initial_seed,
    manual_seed,
    seed,
    set_rng_state,
    thread_safe_generator,
)
from torch.serialization import load, save


################################################################################
# Initialize extension
################################################################################


# Shared memory manager needs to know the exact location of manager executable
def _manager_path() -> bytes:
    # [FUNCTION DOC] HELPER: Returns the UTF-8 encoded path of the shared-memory manager executable
    # (empty bytes on Windows). Prepares the multiprocessing environment as a side effect and raises
    # RuntimeError if the executable is missing.
    if platform.system() == "Windows":
        return b""
    path = get_file_path("torch", "bin", "torch_shm_manager")
    prepare_multiprocessing_environment(get_file_path("torch"))
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode("utf-8")


_C._initExtension(_manager_path())  # pyrefly: ignore[bad-argument-type]

del _manager_path

# Appease the type checker: it can't deal with direct setting of globals().
# Note that we will see "too many" functions when reexporting this way; there
# is not a good way to fix this problem.  Perhaps, try to redesign VariableFunctions
# so that this import is good enough
if TYPE_CHECKING:
    # Some type signatures pulled in from _VariableFunctions here clash with
    # signatures already imported. For now these clashes are ignored; see
    # PR #43339 for details.
    from torch._C._VariableFunctions import *  # type: ignore[assignment, misc] # noqa: F403

    # Fixup segment_reduce visibility
    _segment_reduce = segment_reduce
    del segment_reduce  # noqa: F821

# Ops not to be exposed in `torch` namespace,
# mostly helper ops.
PRIVATE_OPS = ("unique_dim",)

__name, __obj = "", None
for __name in dir(_C._VariableFunctions):
    if __name.startswith("__") or __name in PRIVATE_OPS:
        continue
    __obj = getattr(_C._VariableFunctions, __name)
    __obj.__module__ = __name__  # "torch"
    # Hide some APIs that should not be public
    if __name == "segment_reduce":
        # TODO: Once the undocumented FC window is passed, remove the line below
        globals()[__name] = __obj
        __name = "_" + __name
    globals()[__name] = __obj
    if not __name.startswith("_"):
        __all__.append(__name)

del __name, __obj

################################################################################
# Add torch.dtype instances to the public API
################################################################################

import torch


__all__.extend(
    # pyrefly: ignore [unresolvable-dunder-all]
    name
    for name in dir(torch)
    if isinstance(getattr(torch, name), torch.dtype)
)

################################################################################
# Import TorchDynamo's lazy APIs to avoid circular dependencies
################################################################################

# needs to be before from torch.functional import * to avoid circular dependencies
from torch._compile import _disable_dynamo  # usort: skip

################################################################################
# Import interface functions defined in Python
################################################################################

# needs to be after the above ATen bindings so we can overwrite from Python side
from torch import _VF as _VF, functional as functional  # usort: skip
from torch.functional import *  # usort: skip # noqa: F403

################################################################################
# Remove unnecessary members
################################################################################

del _StorageBase
del _LegacyStorage

################################################################################
# Define _assert
################################################################################


# needs to be before the submodule imports to avoid circular dependencies
def _assert(condition: object, message: object) -> None:
    # [FUNCTION DOC] HELPER: Symbolically traceable assert. For non-tensor conditions with torch-function
    # overrides, dispatches to the override. Otherwise raises AssertionError with the message if the
    # condition is falsy (evaluating truthiness, which may guard).
    r"""A wrapper around Python's assert which is symbolically traceable."""
    if type(condition) is not torch.Tensor and overrides.has_torch_function(
        (condition,)
    ):
        return overrides.handle_torch_function(
            _assert, (condition,), condition, message
        )
    if not condition:
        raise AssertionError(message)


################################################################################
# Import most common subpackages
################################################################################

# Use the redundant form so that type checkers know that these are a part of
# the public API. The "regular" import lines are there solely for the runtime
# side effect of adding to the imported module's members for other users.

# needs to be before import torch.nn as nn to avoid circular dependencies
from torch.autograd import (  # usort: skip
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
)

from torch import (
    __config__ as __config__,
    __future__ as __future__,
    _awaits as _awaits,
    accelerator as accelerator,
    autograd as autograd,
    backends as backends,
    cpu as cpu,
    cuda as cuda,
    distributed as distributed,
    distributions as distributions,
    fft as fft,
    futures as futures,
    hub as hub,
    jit as jit,
    linalg as linalg,
    mps as mps,
    mtia as mtia,
    multiprocessing as multiprocessing,
    nested as nested,
    nn as nn,
    optim as optim,
    overrides as overrides,
    profiler as profiler,
    sparse as sparse,
    special as special,
    testing as testing,
    types as types,
    utils as utils,
    version as version,
    xpu as xpu,
)
from torch.signal import windows as windows


# Quantized, sparse, AO, etc. should be last to get imported, as nothing
# is expected to depend on them.
from torch import ao as ao  # usort: skip

# nn.quant* depends on ao -- so should be after those.
import torch.nn.intrinsic
import torch.nn.qat
import torch.nn.quantizable
import torch.nn.quantized


_C._init_names(list(_storage_classes))

# attach docstrings to torch and tensor functions
from torch import _size_docs, _storage_docs, _tensor_docs, _torch_docs


del _torch_docs, _tensor_docs, _storage_docs, _size_docs


def compiled_with_cxx11_abi() -> builtins.bool:
    # [FUNCTION DOC] Returns True always (hardcoded) to report the build uses the C++11 ABI.
    # KNOWN ISSUE: the value is a constant, not queried from the build.
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return True


from torch import _library as _library, _ops as _ops


# Import the ops and classes "namespace"
from torch._ops import ops as ops  # usort: skip
from torch._classes import classes as classes  # usort: skip

sys.modules.setdefault(f"{__name__}.ops", ops)
sys.modules.setdefault(f"{__name__}.classes", classes)

if hasattr(torch._C, "_c10d_init"):
    from torch.distributed.distributed_c10d import _register_process_group_opaque_type

    _register_process_group_opaque_type()
    del _register_process_group_opaque_type

# quantization depends on torch.fx and torch.ops
# Import quantization
from torch import quantization as quantization  # usort: skip

# Import the quasi random sampler
from torch import quasirandom as quasirandom  # usort: skip

# If you are seeing this, it means that this call site was not checked if
# the memory format could be preserved, and it was switched to old default
# behaviour of contiguous
legacy_contiguous_format = contiguous_format  # defined by _C._initExtension()

# Register fork handler to initialize OpenMP in child processes (see gh-28389)
from torch.multiprocessing._atfork import register_after_fork


register_after_fork(torch.get_num_threads)
del register_after_fork

# Import tools that require fully imported torch (for applying
# torch.jit.script as a decorator, for instance):
from torch._lobpcg import lobpcg as lobpcg


# These were previously defined in native_functions.yaml and appeared on the
# `torch` namespace, but we moved them to c10 dispatch to facilitate custom
# class usage. We add these lines here to preserve backward compatibility.
quantized_lstm = ops.aten.quantized_lstm
quantized_gru = ops.aten.quantized_gru

# Import experimental masked operations support. See
# [RFC-0016](https://github.com/pytorch/rfcs/pull/27) for more
# information.
from torch import masked as masked

# Import removed ops with error message about removal
from torch._linalg_utils import (  # type: ignore[misc]
    _symeig as symeig,
    cholesky,
    eig,
    lstsq,
    matrix_rank,
    qr,
    solve,
)
from torch.utils.dlpack import from_dlpack, to_dlpack


class _TorchCompileInductorWrapper:
    # [CLASS DOC] HELPER: Callable backend object that compiles an FX graph with Inductor. Holds a
    # config mapping (dotted option names to values built from mode, user options and compiler
    # bisector overrides), the dynamic-shape flag and an optional region name. Compared by value.
    # Key methods: apply_mode, apply_options, call (compile), get_compiler_config, reset.
    compiler_name = "inductor"

    def __init__(
        self,
        mode: str | None,
        options: dict[str, _Any] | None,
        dynamic: builtins.bool | None,
        name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores dynamic and name, then builds the config by applying the mode
        # preset, the user options and any bisector config change. When CUDA graphs are enabled with an
        # older CUDA version or no lazy CUPTI support, sets environment variables disabling lazy CUPTI
        # re-init and CUPTI teardown. Raises RuntimeError via apply_options for bad options.
        from torch._inductor.compiler_bisector import CompilerBisector

        self.config: dict[str, _Any] = {}
        self.dynamic = dynamic
        self.name = name
        self.apply_mode(mode)
        self.apply_options(options)
        self.apply_options(CompilerBisector.get_config_change("inductor"))

        cuda_version = None
        if hasattr(torch, "version"):
            from torch.torch_version import TorchVersion

            cuda_version = TorchVersion(getattr(torch.version, "cuda", "0.0"))

        if self.config.get("triton.cudagraphs", False) and (
            (cuda_version and cuda_version < "12.6")
            or not profiler_allow_cudagraph_cupti_lazy_reinit_cuda12()
        ):
            os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
            # FIXME: CUDA Graph does not work well with CUPTI teardown.
            #   1) crashes on 1st lazy CUPTI re-init after teardown (CUDA 11)
            #   2) crashes on 2nd non-lazy CUPTI re-init after teardown (CUDA 12)
            # Workaround: turn off CUPTI teardown when using CUDA Graphs.
            os.environ["TEARDOWN_CUPTI"] = "0"

    def __eq__(self, other: object) -> builtins.bool:
        # [FUNCTION DOC] HELPER: Equal when other is also an inductor wrapper with the same config,
        # dynamic flag and name. KNOWN ISSUE: defines equality without hash, so instances are unhashable.
        return (
            isinstance(other, _TorchCompileInductorWrapper)
            and self.config == other.config
            and self.dynamic == other.dynamic
            and self.name == other.name
        )

    def apply_mode(self, mode: str | None) -> None:
        # [FUNCTION DOC] HELPER: For a non-empty mode other than default, looks up that mode's option
        # preset (depending on the dynamic flag) and applies it to the config. No-op otherwise.
        if mode and mode != "default":
            from torch._inductor import list_mode_options

            self.apply_options(list_mode_options(mode, self.dynamic))

    def apply_options(self, options: dict[str, _Any] | None) -> None:
        # [FUNCTION DOC] HELPER: Merges options into the config. Keys have dashes turned into
        # underscores and must name a known inductor config entry, otherwise raises RuntimeError listing
        # known options. Values are type-checked against the config entry type when that type is a plain
        # class (generic types are skipped); a mismatch raises RuntimeError. Mutates self.config.
        if not options:
            return

        from torch._inductor import config

        current_config: dict[str, _Any] = config.get_config_copy()

        for key, val in options.items():
            attr_name = key.replace("-", "_")
            if attr_name not in current_config:
                raise RuntimeError(
                    f"Unexpected optimization option {key}, known options are {list(current_config.keys())}"
                )
            attr_type = config.get_type(attr_name)  # type: ignore[attr-defined]
            # Subscriptable generic types don't support isinstance so skip the type
            # check. There doesn't seem to be a good way of checking membership without
            # 3rd party libraries.
            if _get_origin(attr_type) is None:
                if not isinstance(val, attr_type):
                    val_type_str = type(val).__name__
                    expected_type_str = type(current_config[attr_name]).__name__
                    raise RuntimeError(
                        f"Unexpected type of attr {key}, got {val_type_str} should be {expected_type_str}"
                    )
            self.config[attr_name] = val

    def __call__(
        self,
        model_: _Any,
        inputs_: _Any,
        *,
        config_patches: dict[str, _Any] | None = None,
    ) -> _Any:
        # [FUNCTION DOC] HELPER: Compiles the given graph module and example inputs with Inductor,
        # using the stored config overlaid with any extra config patches, and the region name. Returns
        # the compiled callable.
        from torch._inductor.compile_fx import compile_fx

        all_patches = {**self.config, **(config_patches or {})}
        return compile_fx(
            model_,
            inputs_,
            config_patches=all_patches,
            compile_region_name=self.name,
        )

    def get_compiler_config(self) -> dict[str, _Any]:
        # [FUNCTION DOC] HELPER: Returns the full inductor config dictionary with this wrapper's config applied.
        from torch._inductor.compile_fx import get_patched_config_dict

        return get_patched_config_dict(config_patches=self.config)

    def reset(self) -> None:
        # [FUNCTION DOC] HELPER: If CUDA graphs are enabled (by this config, or globally unless this
        # config disables it), resets the global CUDA graph trees state. Side effect: global state reset.
        from torch._inductor import config

        if "triton.cudagraphs" in self.config or config.triton.cudagraphs:
            if self.config.get("triton.cudagraphs", True):
                from torch._inductor.cudagraph_trees import reset_cudagraph_trees

                reset_cudagraph_trees()


class _TorchCompileAOTInductorWrapper(_TorchCompileInductorWrapper):
    # [CLASS DOC] HELPER: Inductor wrapper variant for ahead-of-time compilation: forces the C++
    # wrapper and AOT package options on, and compiles under AOT compilation mode.
    compiler_name = "aotinductor"

    def __init__(
        self,
        mode: str | None,
        options: dict[str, _Any] | None,
        dynamic: builtins.bool | None,
        name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs the base initializer, then enables the cpp_wrapper and
        # aot_inductor.package options.
        super().__init__(mode, options, dynamic, name)
        self.apply_options({"cpp_wrapper": True})
        self.apply_options({"aot_inductor.package": True})

    def __call__(
        self,
        model_: _Any,
        inputs_: _Any,
        *,
        config_patches: dict[str, _Any] | None = None,
    ) -> _Any:
        # [FUNCTION DOC] HELPER: Compiles like the base class but inside AOT compilation mode, with
        # autograd for AOT enabled in the inductor config and, if a fake mode is detected from the
        # inputs, temporarily allowing non-fake inputs. Returns the compiled result.
        from contextlib import nullcontext
        from unittest import mock

        from torch._guards import detect_fake_mode
        from torch._inductor.virtualized import V

        fake_mode = detect_fake_mode(inputs_)
        ctx = (
            mock.patch.object(fake_mode, "allow_non_fake_inputs", True)
            if fake_mode
            else nullcontext()
        )
        with (
            V.set_aot_compilation(True),
            ctx,
            torch._inductor.config.patch("enable_autograd_for_aot", True),
        ):
            return super().__call__(model_, inputs_, config_patches=config_patches)


class _TorchCompileWrapper:
    # [CLASS DOC] HELPER: Callable adapter around any non-inductor dynamo backend. Holds the resolved
    # backend function, a display name, the dynamic flag and keyword arguments (mode and options, only
    # when non-default). Compared by value.
    def __init__(
        self,
        backend: str | _Callable[..., _Any],
        mode: str | None,
        options: dict[str, _Any] | None,
        dynamic: builtins.bool | None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Resolves the backend (name string or callable) via the backend
        # registry, derives compiler_name from the string, the callable name, or its string form, and
        # records mode and options as keyword arguments only if non-empty and non-default.
        from torch._dynamo.backends.registry import lookup_backend

        if isinstance(backend, str):
            self.compiler_name = backend
        elif hasattr(backend, "__name__"):
            self.compiler_name = backend.__name__
        else:
            self.compiler_name = str(backend)
        self.dynamic = dynamic
        self.compiler_fn = lookup_backend(backend)
        self.kwargs: dict[str, _Any] = {}
        # only pass the args if they non-empty
        if mode and mode != "default":
            self.kwargs["mode"] = mode
        if options:
            self.kwargs["options"] = options

    def __eq__(self, other: object) -> builtins.bool:
        # [FUNCTION DOC] HELPER: Equal when other is also a generic wrapper with the same backend
        # function, kwargs and dynamic flag. KNOWN ISSUE: defines equality without hash, so
        # instances are unhashable.
        return (
            isinstance(other, _TorchCompileWrapper)
            and self.compiler_fn == other.compiler_fn
            and self.kwargs == other.kwargs
            and self.dynamic == other.dynamic
        )

    def __call__(self, model_: _Any, inputs_: _Any) -> _Any:
        # [FUNCTION DOC] HELPER: Invokes the backend function on the graph module and example inputs
        # with the stored keyword arguments; returns its compiled callable.
        return self.compiler_fn(model_, inputs_, **self.kwargs)

    def reset(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the backend's own reset method if it has one; otherwise no-op.
        if hasattr(self.compiler_fn, "reset"):
            self.compiler_fn.reset()


_InputT = _ParamSpec("_InputT")
_RetT = _TypeVar("_RetT")


@_overload
def compile(
    model: _Callable[_InputT, _RetT],
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | _Callable[..., _Any] = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = None,
    name: str | None = None,
    disable: builtins.bool = False,
    dynamic_shapes: _Any = None,
) -> _Callable[_InputT, _RetT]: ...


@_overload
def compile(
    model: None = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | _Callable[..., _Any] = "inductor",
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = None,
    name: str | None = None,
    disable: builtins.bool = False,
    dynamic_shapes: _Any = None,
) -> _Callable[[_Callable[_InputT, _RetT]], _Callable[_InputT, _RetT]]: ...


def compile(
    model: _Callable[_InputT, _RetT] | None = None,
    *,
    fullgraph: builtins.bool = False,
    dynamic: builtins.bool | None = None,
    backend: str | _Callable[..., _Any] | None = None,
    mode: str | None = None,
    options: dict[str, str | builtins.int | builtins.bool | _Callable[..., _Any]]
    | None = None,
    name: str | None = None,
    disable: builtins.bool = False,
    recompile_limit: builtins.int | None = None,
    isolate_recompiles: builtins.bool = False,
    dynamic_shapes: _Any = None,
) -> (
    _Callable[[_Callable[_InputT, _RetT]], _Callable[_InputT, _RetT]]
    | _Callable[_InputT, _RetT]
):
    # [FUNCTION DOC] Public torch.compile entry. Wraps a function or module with TorchDynamo plus the
    # chosen backend; with no model it returns a decorator that applies the same settings.
    # Behavior: logs API usage; raises RuntimeError on Python 3.15+ or on free-threaded builds below
    # 3.13.3; defaults the backend; wraps a parameter spec into a shapes spec and resolves a spec
    # attached to the model; raises RuntimeError if both mode and options are given (mode defaults to
    # default when neither is); lets the compiler bisector override the backend; pops guard_filter_fn
    # and use_aoti out of options (mutating the caller's dict); inside an export region warns and
    # returns the model unchanged; builds an inductor, AOT inductor or generic backend wrapper; and
    # returns the dynamo-optimized callable. Args mirror the docstring.
    # KNOWN ISSUE: the two typing overloads omit recompile_limit and isolate_recompiles and default
    # backend to inductor, while the implementation defaults backend to None.
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    If you are compiling an :class:`torch.nn.Module`, you can also use :meth:`torch.nn.Module.compile`
    to compile the module inplace without changing its structure.

    Concretely, for every frame executed within the compiled region, we will attempt
    to compile it and cache the compiled result on the code object for future
    use.  A single frame may be compiled multiple times if previous compiled
    results are not applicable for subsequent calls (this is called a "guard
    failure"), you can use TORCH_LOGS=guards to debug these situations.
    Multiple compiled results can be associated with a frame up to
    ``torch._dynamo.config.recompile_limit``, which defaults to 8; at which
    point we will fall back to eager.  Note that compile caches are per
    *code object*, not frame; if you dynamically create multiple copies of a
    function, they will all share the same code cache.

    Args:
       model (Callable or None): Module/function to optimize
       fullgraph (bool): If False (default), torch.compile attempts to discover compilable regions
        in the function that it will optimize. If True, then we require that the entire function be
        capturable into a single graph. If this is not possible (that is, if there are graph breaks),
        then this will raise an error. This also opts into unbacked semantics, notably it will turn on
        capture_scalar_outputs and capture_dynamic_output_shape_ops on by default.
       dynamic (bool or None): Use dynamic shape tracing.  When this is True, we will up-front attempt
        to generate a kernel that is as dynamic as possible to avoid recompilations when
        sizes change.  This may not always work as some operations/optimizations will
        force specialization; use TORCH_LOGS=dynamic to debug overspecialization.
        When this is False, we will NEVER generate dynamic kernels, we will always specialize.
        By default (None), we automatically detect if dynamism has occurred and compile a more
        dynamic kernel upon recompile.
       backend (str or Callable): backend to be used

        - "inductor" is the default backend, which is a good balance between performance and overhead

        - Non experimental in-tree backends can be seen with `torch._dynamo.list_backends()`

        - Experimental or debug in-tree backends can be seen with `torch._dynamo.list_backends(None)`

        - To register an out-of-tree custom backend:
          https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html#registering-custom-backends
       mode (str): Can be either "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"

        - "default" is the default mode, which is a good balance between performance and overhead

        - "reduce-overhead" is a mode that reduces the overhead of python with CUDA graphs,
          useful for small batches.  Reduction of overhead can come at the cost of more memory
          usage, as we will cache the workspace memory required for the invocation so that we
          do not have to reallocate it on subsequent runs.  Reduction of overhead is not guaranteed
          to work; today, we only reduce overhead for CUDA only graphs which do not mutate inputs.
          There are other circumstances where CUDA graphs are not applicable; use TORCH_LOGS=perf_hints
          to debug.

        - "max-autotune" is a mode that leverages Triton or template based matrix multiplications
          on supported devices and Triton based convolutions on GPU.
          It enables CUDA graphs by default on GPU.

        - "max-autotune-no-cudagraphs" is a mode similar to "max-autotune" but without CUDA graphs

        - To see the exact configs that each mode sets you can call `torch._inductor.list_mode_options()`

       options (dict): A dictionary of options to pass to the backend. Some notable ones to try out are

        - `epilogue_fusion` which fuses pointwise ops into templates. Requires `max_autotune` to also be set

        - `max_autotune` which will profile to pick the best matmul configuration

        - `fallback_random` which is useful when debugging accuracy issues

        - `shape_padding` which pads matrix shapes to better align loads on GPUs especially for tensor cores

        - `triton.cudagraphs` which will reduce the overhead of python with CUDA graphs

        - `trace.enabled` which is the most useful debugging flag to turn on

        - `trace.graph_diagram` which will show you a picture of your graph after fusion

        - `guard_filter_fn` that controls which dynamo guards are saved with compilations.
          This is an unsafe feature and there is no backward compatibility guarantee provided
          for dynamo guards as data types.
          For stable helper functions to use, see the documentation in `torch.compiler`, for example:
          - `torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe`
          - `torch.compiler.skip_guard_on_all_nn_modules_unsafe`
          - `torch.compiler.keep_tensor_guards_unsafe`

        - For inductor you can see the full list of configs that it supports by calling `torch._inductor.list_options()`
       name (str or None): Optional identifier for the compiled region. When supported by downstream
        tooling, this is surfaced on wrapped compiled-region higher-order operators and other debug metadata.
       disable (bool): Turn torch.compile() into a no-op for testing
       recompile_limit (int or None): Maximum number of recompilations allowed for this
        ``torch.compile()`` call before falling back to eager. If None (default), uses
        the global ``torch._dynamo.config.recompile_limit`` (default 8). With
        ``fullgraph=True``, exceeding the limit raises ``FailOnRecompileLimitHit``.
       isolate_recompiles (bool): If True, this ``torch.compile()`` call tracks
        recompilations independently. By default, all ``torch.compile()`` calls on the
        same function share a single set of compiled entries, so one call's
        recompilations count against every other call's limit. With
        ``isolate_recompiles=True``, each call gets its own isolated set of entries.
        Lookups for an isolated compile call will still fall back to entries from
        non-isolated compile calls (BC-friendly reuse), but new compilations are
        stored separately. ``recompile_limit`` is checked per-region;
        ``accumulated_recompile_limit`` is a global cap across all regions.
        Default-strategy behavior is asymmetric: SKIP decisions (from
        ``skip_code``, ``@torch._dynamo.skip``, FX-generated code, etc.) are
        inherited by isolated regions — those remain skipped. RUN_ONLY persisted
        by a non-isolated region hitting its recompile limit does NOT bleed
        into isolated regions — each region manages its own RUN_ONLY state.
        Default False.

    Example::

        @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
        def foo(x):
            return torch.sin(x) + torch.cos(x)

    """
    import sysconfig

    _C._log_api_usage_once("torch.compile")
    if sys.version_info >= (3, 15):
        raise RuntimeError("torch.compile is not supported on Python 3.15+")
    elif sysconfig.get_config_var("Py_GIL_DISABLED") == 1 and sys.version_info < (
        3,
        13,
        3,
    ):
        raise RuntimeError(
            "torch.compile is not supported on Python < 3.13.3 built with GIL disabled. "
            "Please use Python 3.13.3+."
        )

    if backend is None:
        from torch._dynamo.backends.registry import get_default_backend

        backend = get_default_backend()

    # Auto-wrap ParamsSpec → ShapesSpec for convenience
    if dynamic_shapes is not None:
        from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec

        if isinstance(dynamic_shapes, ParamsSpec):
            dynamic_shapes = ShapesSpec(dynamic_shapes)

    # If ``model`` carries an ``@dynamic_spec(...)`` decorator, the attached
    # ``ShapesSpec`` is used as ``dynamic_shapes``. Passing both raises.
    if model is not None:
        from torch.fx.experimental.dynamic_spec import _resolve_dynamic_shapes

        dynamic_shapes = _resolve_dynamic_shapes(model, dynamic_shapes)

    # Decorator mode
    if model is None:

        def fn(model: _Callable[_InputT, _RetT]) -> _Callable[_InputT, _RetT]:
            # [FUNCTION DOC] HELPER: Decorator returned when compile is called without a model. Raises
            # RuntimeError if given None; otherwise calls compile with the captured settings.
            if model is None:
                raise RuntimeError("Model can't be None")
            return compile(  # pyrefly: ignore  # no-matching-overload
                model,
                fullgraph=fullgraph,
                dynamic=dynamic,
                backend=backend,
                mode=mode,
                options=options,
                name=name,
                disable=disable,
                recompile_limit=recompile_limit,
                isolate_recompiles=isolate_recompiles,
                dynamic_shapes=dynamic_shapes,
            )

        return fn

    if mode is not None and options is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be specified at the same time."
        )
    if mode is None and options is None:
        mode = "default"

    from torch._inductor.compiler_bisector import CompilerBisector

    if bisect_backend := CompilerBisector.get_backend():
        import torch._inductor.config as inductor_config

        # don't override the backend for use cases like vllm
        # which leverages their custom backend.
        if not (
            inductor_config.test_configs.bisect_keep_custom_backend_for_inductor
            and bisect_backend == "inductor"
            and not isinstance(backend, str)
        ):
            backend = bisect_backend

    guard_filter_fn = None
    use_aoti = False
    if options and isinstance(options, dict):
        guard_filter_fn = options.pop("guard_filter_fn", None)
        use_aoti = options.pop("use_aoti", False)

    if torch.compiler.is_exporting():
        from torch._higher_order_ops.utils import _in_hop_compile

        if not _in_hop_compile():
            warnings.warn(
                "torch.compile is ignored when called inside torch.export region",
                stacklevel=2,
            )
            # torch.compile is a no-op when inside torch.export region
            return model

    if backend == "inductor":
        if use_aoti:
            backend = _TorchCompileAOTInductorWrapper(mode, options, dynamic, name)
        else:
            backend = _TorchCompileInductorWrapper(mode, options, dynamic, name)
    else:
        backend = _TorchCompileWrapper(backend, mode, options, dynamic)

    return torch._dynamo.optimize(
        backend=backend,
        nopython=fullgraph,
        dynamic=dynamic,
        disable=disable,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        isolate_recompiles=isolate_recompiles,
        dynamic_shapes=dynamic_shapes,
    )(model)  # type: ignore[return-value]


def _register_device_module(device_type: str, module: _ModuleType) -> None:
    # [FUNCTION DOC] HELPER: Registers an external device runtime module as an attribute of torch and
    # in sys.modules under the dotted name. The device type is normalized through the device class;
    # raises RuntimeError if an attribute for that device type already exists. Side effects: mutates
    # the torch module and sys.modules.
    r"""Register an external runtime module of the specific :attr:`device_type`
    supported by torch.

    After the :attr:`module` is registered correctly, the user can refer
    the external runtime module as part of torch with attribute torch.xxx.
    """
    # Make sure the device_type represent a supported device type for torch.
    device_type = torch.device(device_type).type
    m = sys.modules[__name__]
    if hasattr(m, device_type):
        raise RuntimeError(
            f"The runtime module of '{device_type}' has already "
            f"been registered with '{getattr(m, device_type)}'"
        )
    setattr(m, device_type, module)
    torch_module_name = ".".join([__name__, device_type])
    sys.modules[torch_module_name] = module


from torch import (
    export as export,
    func as func,
    library as library,
    return_types as return_types,
)
from torch._higher_order_ops import cond as cond, while_loop as while_loop
from torch.func import vmap as vmap


if not TYPE_CHECKING:
    # register python metas for distributed ops
    # Only import if distributed is available (USE_DISTRIBUTED=1)
    if hasattr(torch._C, "_c10d_init"):
        import torch.distributed._meta_registrations as coll_meta_registrations

        del coll_meta_registrations
    from torch import _meta_registrations

# Enable CUDA Sanitizer
if "TORCH_CUDA_SANITIZER" in os.environ:
    import torch.cuda._sanitizer as csan

    csan.enable_cuda_sanitizer()

# Populate magic methods on SymInt and SymFloat
import torch.fx.experimental.sym_node
from torch import compiler as compiler, fx as fx


class _TritonLibrary:
    # [CLASS DOC] HELPER: Namespace holder (class-level state only) for custom operators in the triton
    # namespace. lib is the shared operator library; ops_table maps (operator key, dispatch key) to the
    # registered implementation so each pair registers at most once.
    lib = torch.library.Library("triton", "DEF")
    ops_table: dict[tuple[str, str], _Callable[..., _Any]] = {}

    @classmethod
    def registerOp(
        cls,
        op_key: str,
        full_schema: str,
        op_impl: _Callable[..., _Any],
        dispatch_key: str,
    ) -> _Callable[..., _Any]:
        # [FUNCTION DOC] HELPER: Idempotently registers an operator: if the (op_key, dispatch_key)
        # pair is new, defines the schema and registers the implementation for that dispatch key on the
        # triton library and records it. Returns the implementation stored for the pair (the first one
        # registered; later different implementations are ignored).
        if (op_key, dispatch_key) not in cls.ops_table:
            cls.lib.define(full_schema)
            cls.lib.impl("triton::" + op_key, op_impl, dispatch_key)
            cls.ops_table[(op_key, dispatch_key)] = op_impl

        return cls.ops_table[(op_key, dispatch_key)]


# Deprecated attributes
_deprecated_attrs = {
    "has_mps": torch.backends.mps.is_built,
    "has_cuda": torch.backends.cuda.is_built,
    "has_cudnn": torch.backends.cudnn.is_available,
    "has_mkldnn": torch.backends.mkldnn.is_available,
}

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.
    from torch import (
        _dynamo as _dynamo,
        _inductor as _inductor,
        _subclasses as _subclasses,
        onnx as onnx,
    )

else:
    _lazy_modules = {
        "_dynamo",
        "_inductor",
        "_export",
        # ONNX must be imported after _dynamo, _ops, _subclasses, fx, func and jit
        "onnx",
    }

    def __getattr__(name: str) -> _Any:
        # [FUNCTION DOC] Module-level lazy attribute hook, called only when normal lookup fails. Deprecated
        # has_* backend flags warn and return the result of the matching availability function; lazy
        # submodule names (dynamo, inductor, export, onnx) are imported on demand and returned;
        # set_vital warns and returns a no-op function; anything else raises AttributeError.
        # Deprecated attrs
        replacement = _deprecated_attrs.get(name)
        if replacement is not None:
            import warnings

            warnings.warn(
                f"'{name}' is deprecated, please use '{replacement.__module__}.{replacement.__name__}()'",
                stacklevel=2,
            )
            return replacement()

        # Lazy modules
        if name in _lazy_modules:
            return importlib.import_module(f".{name}", __name__)

        # set_vital
        if name == "set_vital":
            import warnings

            warnings.warn(f"'{name}' is deprecated, please do not call", stacklevel=2)
            return lambda *args: None

        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


@functools.cache
def get_device_module(device: "torch.device | str | None" = None) -> _ModuleType:
    # [FUNCTION DOC] Returns the torch submodule for a device (device object, device string, or None for
    # the current accelerator, falling back to cpu). Raises RuntimeError for other argument types or if
    # no torch attribute exists for that device type. Results are cached per argument.
    """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """
    if isinstance(device, torch.device):
        device_module_name = device.type
    elif isinstance(device, str):
        device_module_name = torch.device(device).type
    elif device is None:
        # Use the current accelerator's type, falling back to CPU when none is available.
        acc = torch._C._accelerator_getAccelerator()
        device_module_name = acc.type if acc is not None else "cpu"
    else:
        raise RuntimeError(
            f"Invalid value of device '{device}', expect torch.device, str, or None"
        )
    device_module = getattr(torch, device_module_name, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_module_name}' does not have a corresponding module registered as 'torch.{device_module_name}'."
        )
    return device_module


def _constrain_as_size(
    symbol: "IntLikeType",
    min: builtins.int | None = None,
    max: builtins.int | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Marks a symbolic (typically unbacked) int as size-like with optional
    # min/max bounds by forwarding to the symbolic range-for-size constraint function.
    # KNOWN ISSUE: the docstring ends with a stray code-fence marker.
    """
    This indicates that a given int is size-like, and can be used in any context where a size is expected.
    You will typically use this when reading out integers from Tensors, e.g., max.item() or lengths.tolist()
    which then need to be used as tensor constructors. Providing these assertions to PyTorch can help resolve
      GuardOnDataDependentSymNode errors upon export, since we cannot guard on unbacked SymInts.

    This function has unusual semantics in some circumstances in framework
    code, we will treat this int as >= 2 (when we do a size-oblivious guard).
    This makes it easier to use the unbacked int in size contexts,
    as we will often attempt to guard on a size being zero/one
    (e.g., when computing the contiguity of a tensor, or testing if
    broadcasting can occur), which will not work on unbacked SymInts.
    However, if we conservatively assume that the size is not zero/one, we will
    end up with a graph that will still work even if the size is zero/one.

    For more details, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit
    ```
    """
    # pyrefly: ignore[bad-argument-type]
    torch.sym_constrain_range_for_size(symbol, min=min, max=max)


def _import_device_backends() -> None:
    # [FUNCTION DOC] HELPER: Discovers entry points in the torch.backends group, loads each and calls
    # it, to autoload out-of-tree device extensions. Any failure is re-raised as RuntimeError naming
    # the extension and mentioning how to disable autoload. Side effects: imports and runs plugin code.
    """
    Leverage the Python plugin mechanism to load out-of-the-tree device extensions.
    See this RFC: https://github.com/pytorch/pytorch/issues/122468
    """
    from importlib.metadata import entry_points

    group_name = "torch.backends"
    backend_extensions = entry_points(group=group_name)

    for backend_extension in backend_extensions:
        try:
            # Load the extension
            entrypoint = backend_extension.load()
            # Call the entrypoint
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the backend extension: {backend_extension.name}. "
                f"You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0."
            ) from err


def _is_device_backend_autoload_enabled() -> builtins.bool:
    # [FUNCTION DOC] HELPER: Returns True unless the environment variable TORCH_DEVICE_BACKEND_AUTOLOAD
    # is set to something other than 1.
    """
    Whether autoloading out-of-the-tree device extensions is enabled.
    The switch depends on the value of the environment variable
    `TORCH_DEVICE_BACKEND_AUTOLOAD`.

    Returns:
        bool: Whether to enable autoloading the extensions. Enabled by default.

    Examples:
        >>> torch._is_device_backend_autoload_enabled()
        True
    """
    # enabled by default
    return os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1") == "1"


def _as_tensor_fullprec(t: object) -> "torch.Tensor":
    # [FUNCTION DOC] HELPER: Converts to a tensor keeping Python scalars at full precision: exact float
    # becomes float64 and exact int becomes int64, both on cpu; any other input goes through the
    # ordinary as_tensor conversion.
    """
    Like torch.as_tensor, but when given Python data types it will keep
    them in full precision.  Used for calling convention for Dynamo.
    Python scalars (float, int) are always created on CPU to avoid being
    affected by DeviceContext.
    """
    ty = type(t)
    if ty is builtins.float:
        return torch.as_tensor(t, dtype=torch.float64, device="cpu")
    elif ty is builtins.int:
        return torch.as_tensor(t, dtype=torch.int64, device="cpu")
    else:
        return torch.as_tensor(t)


# `_import_device_backends` should run after the definitions above to ensure
# all the other functions in this module that may be accessed by an autoloaded
# backend are defined.
if _is_device_backend_autoload_enabled():
    _import_device_backends()

from torch import _logging


# Keep `TORCH_LOGS` initialization after backend autoload so backends can
# register their log names before `TORCH_LOGS` is parsed and validated.
_logging._init_logs()

# Register all registered custom / override ops in torch/_native
import torch._native
