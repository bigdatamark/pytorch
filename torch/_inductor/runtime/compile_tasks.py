# [MODULE DOC] Helpers used by inductor compile workers (subprocesses): loading generated python modules, pointing
# triton at ptxas and libdevice, syncing cache-related environment variables, and worker entry points that compile
# PyCodeCache-based kernels or warm the triton cache for a kernel. Module-level state: last applied cache env.
from __future__ import annotations

import functools
import linecache
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, TYPE_CHECKING

from torch._utils_internal import log_triton_builds

from ..utils import apply_subprocess_env, clear_caches


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
    # [FUNCTION DOC] HELPER: Compiles the source file at path into a new module named after this module plus the key, sets its
    # file and key attributes, runs the code in the module namespace and optionally registers it in the module table.
    # Returns the module. Raises RuntimeError if compilation of the source fails (original error suppressed); errors while
    # executing the module code propagate. Side effects: file read, module registration.
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
def _set_triton_ptxas_path() -> None:
    # [FUNCTION DOC] HELPER: Cached (runs once). Unless TRITON_PTXAS_PATH is already set, looks for a ptxas binary in the torch bin
    # directory and, if it is an executable file, exports the environment variable. Warns if it exists but is not executable.
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[2] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    # [FUNCTION DOC] HELPER: If the eager-numerics libdevice option or the emulate-precision-casts option is enabled in the inductor
    # config, points triton at the CUDA toolkit libdevice (via the implementation helper); otherwise does nothing.
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's libdevice calls match CUDA eager numerics for bitwise
    precision.  Gated by config.eager_numerics.use_pytorch_libdevice and by
    config.emulate_precision_casts, which also requests eager-like numerics.
    """
    from torch._inductor import config

    if not (
        config.eager_numerics.use_pytorch_libdevice or config.emulate_precision_casts
    ):
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    # [FUNCTION DOC] HELPER: Does nothing for non-CUDA torch builds or when triton knobs are unavailable. Uses TRITON_LIBDEVICE_PATH if
    # set; keeps an existing triton setting; otherwise derives libdevice from CUDA_HOME, setting the triton knob and exporting the
    # environment variable for subprocesses. Emits warnings (not errors) when CUDA_HOME or the file or cpp_extension is missing.
    import torch

    if torch.version.cuda is None:
        return

    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torch.utils.cpp_extension not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


_WORKER_CACHE_ENV_VARS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
_last_applied_cache_env: dict[str, str | None] | None = None


def _apply_subprocess_env_and_clear_caches(extra_env: dict[str, str | None]) -> None:
    # [FUNCTION DOC] HELPER: Applies the given environment mapping in the worker. If the cache-directory variables (inductor or triton
    # cache dir) in the mapping changed since the last call, clears the in-process caches first and remembers the new values.
    # Mutates a module-level global and the process environment.
    global _last_applied_cache_env

    cache_env = {
        key: extra_env.get(key) for key in _WORKER_CACHE_ENV_VARS if key in extra_env
    }
    if cache_env and cache_env != _last_applied_cache_env:
        clear_caches()
        _last_applied_cache_env = cache_env.copy()
    apply_subprocess_env(extra_env)


def _worker_compile_pycodecache_kernel(
    kernel_name: str,
    source_code: str,
    main_suffix: str,
    extra_env: dict[str, str | None],
    precompile_metadata: dict[str, Any] | None = None,
) -> tuple[str, str, int]:
    # [FUNCTION DOC] HELPER: Worker entry point. Applies the environment, writes the source to the PyCodeCache, loads the module and checks it
    # has a function named kernel name plus main suffix (else raises RuntimeError listing callables). If precompile metadata is
    # given, calls the module's precompile function with it, or logs a warning when missing. Clears the line cache. Returns the cache
    # key, the module path and the elapsed time in microseconds.
    """
    Subprocess worker for PyCodeCache-based kernel compilation.

    Writes source to PyCodeCache, loads the module, validates the entry point,
    and optionally triggers real GPU compilation (MLIR -> PTX -> CUBIN) via a
    _precompile entry point. Compiled artifacts are persisted to disk cache so
    the parent process can load them without recompilation.

    Used by both CuteDSL and NV Universal GEMM backends.
    """
    _apply_subprocess_env_and_clear_caches(extra_env)

    start_ns = time.time_ns()

    import torch._inductor.codecache as codecache

    key, path = codecache.PyCodeCache.write(source_code)
    mod = codecache.PyCodeCache.load_by_key_path(key, path)

    main_func_name = f"{kernel_name}_{main_suffix}"
    if not hasattr(mod, main_func_name):
        available = [name for name in dir(mod) if callable(getattr(mod, name))]
        raise RuntimeError(
            f"Could not find kernel function '{main_func_name}'. "
            f"Available callables: {available}"
        )

    if precompile_metadata is not None:
        precompile_fn_name = f"{kernel_name}_precompile"
        precompile_fn = getattr(mod, precompile_fn_name, None)
        if precompile_fn is not None:
            precompile_fn(**precompile_metadata)
        else:
            import logging

            logging.getLogger(__name__).warning(
                "Precompile metadata was provided but module has no %s "
                "— the scheduling layer expected this template to support "
                "subprocess precompilation. Kernel will compile lazily on "
                "first call instead.",
                precompile_fn_name,
            )

    elapsed_ns = time.time_ns() - start_ns
    linecache.clearcache()
    return key, path, elapsed_ns // 1000


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str | None],
    extra_config: dict[str, Any],
) -> tuple[CachingAutotuner, int]:
    # [FUNCTION DOC] HELPER: Worker entry point for triton kernels. Sets the ptxas path, applies the environment (and syncs the triton libdevice
    # knob if provided), then under a patched inductor config loads the kernel, reports the compiling phase to the watchdog and
    # precompiles it with warm-cache-only while skipping GPU driver setup. Prepares the kernel for pickling and returns it with the
    # elapsed microseconds. Logs triton builds (with the failure message) in all cases; exceptions propagate.
    _set_triton_ptxas_path()
    _apply_subprocess_env_and_clear_caches(extra_env)
    # Keep Triton's in-process knob in sync with the parent environment, including
    # clearing stale worker state when the parent no longer has this variable.
    if "TRITON_LIBDEVICE_PATH" in extra_env:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = extra_env["TRITON_LIBDEVICE_PATH"]
        except ImportError:
            pass
    from torch._inductor import config
    from torch._inductor.compile_worker import watchdog
    from torch._inductor.runtime import triton_helpers

    with config.patch(extra_config):
        fail = None
        try:
            start_ns = time.time_ns()
            # Generated Triton modules set up the GPU driver at import time,
            # but compile workers only need to warm the compile cache.
            with triton_helpers.skip_gpu_driver_setup():
                kernel = load_kernel()
                watchdog.report_phase(watchdog.Phase.COMPILING)
                kernel.precompile(warm_cache_only=True)
            elapsed_ns = time.time_ns() - start_ns
            kernel.prepare_for_pickle()
            # We can release this memory in the compile subprocesses:
            linecache.clearcache()
            return kernel, elapsed_ns // 1000
        except Exception as e:
            fail = str(e)
            raise
        finally:
            log_triton_builds(fail=fail)
