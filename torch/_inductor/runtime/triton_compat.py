# [MODULE DOC] Compatibility layer over the optional triton package. If triton imports, re-exports its
# Config, CompiledKernel, OutOfResources, JITFunction, KernelInterface, tl, libdevice, math,
# knobs, triton_key and related names, with fallbacks (stub exception classes, GPUTarget
# builder, log2 stub) for older/newer versions; sets HAS_TRITON, HAS_WARP_SPEC (always false) and
# builtins_use_semantic_kwarg. If triton is missing, defines placeholder objects whose use
# raises RuntimeError. Also exposes autograd_profiler (mock for compile workers).
# Import-time side effects: attempts to import triton and probes its API via introspection.
from __future__ import annotations

import inspect
from typing import Any

import torch


try:
    import triton
except ImportError:
    triton = None


if triton is not None:
    import triton.language as tl
    from triton import Config
    from triton.compiler import CompiledKernel
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import JITFunction, KernelInterface

    try:
        from triton.runtime.autotuner import PTXASError
    except ImportError:

        class PTXASError(Exception):  # type: ignore[no-redef]
            # [CLASS DOC] HELPER: Fallback exception type used when triton does not provide PTXASError.
            pass

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:

        def GPUTarget(
            backend: str,
            arch: int | str,
            warp_size: int,
        ) -> Any:
            # [FUNCTION DOC] HELPER: Fallback GPUTarget builder for triton versions lacking it: returns a list of
            # backend, arch and warp size on ROCm builds, otherwise a tuple of backend and arch.
            if torch.version.hip:
                return [backend, arch, warp_size]
            return (backend, arch)

    # In the latest triton, math functions were shuffled around into different modules:
    # https://github.com/triton-lang/triton/pull/3172
    try:
        from triton.language.extra import libdevice

        libdevice = tl.extra.libdevice  # noqa: F811
        math = tl.math
    except ImportError:
        if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
            libdevice = tl.extra.cuda.libdevice
            math = tl.math
        elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
            libdevice = tl.extra.intel.libdevice
            math = tl.math
        else:
            libdevice = tl.math
            math = tl

    try:
        from triton.language.standard import _log2
    except ImportError:

        def _log2(x: Any) -> Any:
            # [FUNCTION DOC] HELPER: Fallback stub when triton has no log2 helper; always raises NotImplementedError.
            raise NotImplementedError

    def _triton_config_has(param_name: str) -> bool:
        # [FUNCTION DOC] HELPER: True if triton's Config constructor accepts a parameter with the given name
        # (checked by signature inspection); false if Config or its constructor is missing.
        if not hasattr(triton, "Config"):
            return False
        if not hasattr(triton.Config, "__init__"):
            return False
        return param_name in inspect.signature(triton.Config.__init__).parameters

    # Drop the legacy support of autoWS
    HAS_WARP_SPEC = False

    try:
        from triton import knobs
    except ImportError:
        knobs = None

    try:
        from triton.runtime.cache import triton_key  # type: ignore[attr-defined]
    except ImportError:
        from triton.compiler.compiler import (
            triton_key,  # type: ignore[attr-defined,no-redef]
        )

    try:
        from triton.runtime.errors import IntelGPUError
    except ImportError:

        class IntelGPUError(Exception):  # type: ignore[no-redef]
            # [CLASS DOC] HELPER: Fallback exception type used when triton does not provide IntelGPUError.
            pass

    builtins_use_semantic_kwarg = (
        "_semantic" in inspect.signature(triton.language.core.view).parameters
    )
    HAS_TRITON = True
else:

    def _raise_error(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Placeholder for triton-dependent callables; accepts anything and always raises
        # RuntimeError saying triton is not installed.
        raise RuntimeError("triton package is not installed")

    class OutOfResources(Exception):  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Stand-in exception type when triton is not installed.
        pass

    class PTXASError(Exception):  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Stand-in exception type when triton is not installed.
        pass

    class IntelGPUError(Exception):  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Stand-in exception type when triton is not installed.
        pass

    Config = object
    CompiledKernel = object
    KernelInterface = object
    ASTSource = None
    GPUTarget = None
    _log2 = _raise_error
    libdevice = None
    math = None
    knobs = None
    builtins_use_semantic_kwarg = False

    class triton:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Stand-in for the triton module when it is not installed; only jit is provided.
        @staticmethod
        def jit(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Stand-in decorator factory: ignores its arguments and returns the
            # raising placeholder, so using a jit-decorated kernel raises RuntimeError.
            return _raise_error

    class tl:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Stand-in for triton.language when triton is not installed: constexpr returns
        # its value, and tensor and dtype are aliases of the generic Any type.
        @staticmethod
        def constexpr(val: Any) -> Any:
            # [FUNCTION DOC] HELPER: Identity; returns val unchanged.
            return val

        tensor = Any
        dtype = Any

    class JITFunction:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Empty stand-in for triton's JITFunction when triton is not installed.
        pass

    HAS_WARP_SPEC = False
    triton_key = _raise_error
    HAS_TRITON = False


try:
    autograd_profiler = torch.autograd.profiler
except AttributeError:  # Compile workers only have a mock version of torch

    class autograd_profiler:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Mock of the autograd profiler module for compile workers with a mock torch;
        # only reports the profiler as disabled.
        _is_profiler_enabled = False


__all__ = [
    "Config",
    "CompiledKernel",
    "OutOfResources",
    "KernelInterface",
    "PTXASError",
    "IntelGPUError",
    "ASTSource",
    "GPUTarget",
    "tl",
    "_log2",
    "libdevice",
    "math",
    "triton",
    "knobs",
    "triton_key",
]
