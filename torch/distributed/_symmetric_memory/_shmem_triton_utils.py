# [MODULE DOC] Shared helpers for the NVSHMEM and rocSHMEM Triton backends: a kernel-
# name registry, a common post-compile init hook, and (only when Triton is installed)
# a kernel wrapper that injects extern libraries plus a builder for the requires-
# backend decorator. Leaf module: imports no backend or dispatch module, avoiding
# cycles. Import-time side effect: conditional import of Triton classes if has_triton
# is true.
"""
Shared utilities for SHMEM Triton backends (_nvshmem_triton, _rocshmem_triton).

This is a leaf module — it does NOT import any backend or dispatch module,
so backend modules can safely import from here without circular dependencies.
"""

from collections.abc import Callable
from typing import Any

from torch.utils._triton import has_triton


class ShmemKernelRegistry:
    # [CLASS DOC] Registry of Triton kernel function names that need SHMEM module
    # initialization after compilation. State is a class-level mapping shared by the class
    # and, unless a subclass overrides it, all subclasses. Classmethods register,
    # deregister (silent if absent) and has (membership test) manage it.
    """Base registry tracking Triton kernels that need SHMEM module init."""

    _to_init: dict[str, Any] = {}

    @classmethod
        # [FUNCTION DOC] HELPER: Adds the kernel name to the registry if not already present
        # (value is left unset).
    def register(cls, name: str) -> None:
        cls._to_init.setdefault(name)

    @classmethod
        # [FUNCTION DOC] HELPER: Removes the kernel name from the registry; no error if it was
        # not registered.
    def deregister(cls, name: str) -> None:
        cls._to_init.pop(name, None)

    @classmethod
        # [FUNCTION DOC] HELPER: Returns True if the kernel name is registered.
    def has(cls, name: str) -> bool:
        return name in cls._to_init


def run_shmem_init_hook(
    *,
    kwargs: dict[str, Any],
    registry: type[ShmemKernelRegistry],
    module_init: Callable[[Any], None],
    logger: Any,
) -> None:
    # [FUNCTION DOC] Post-compile hook body shared by both backends. Takes the Triton hook
    # keyword arguments (function wrapper, cache key, compile info with device), the
    # registry, a module-init callable and a logger. If the compiled function name is not
    # registered, returns doing nothing. Otherwise looks up the compiled kernel in the
    # per-device cache by key; if found, forces its lazy loading and calls module_init
    # with the kernel module; if missing, logs a warning asking to report to Triton.
    # Returns nothing.
    """Common post-compile hook logic shared by NVSHMEM and rocSHMEM backends."""
    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    if not registry.has(fn_name):
        return

    key = kwargs["key"]
    device = kwargs["compile"]["device"]
    kernel_cache = jit_function.device_caches[device][0]
    kernel = kernel_cache.get(key, None)
    if kernel is not None:
        _ = kernel.run
        module_init(kernel.module)
    else:
        logger.warning(
            "It seems Triton hasn't created a kernel for function %s. "
            "Please report this issue to Triton.",
            fn_name,
        )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    class GridCallableWithExtern(KernelInterface):
    # [CLASS DOC] HELPER: Triton KernelInterface wrapper around a JIT function. Stores the
    # JIT function and an extern-libs mapping (library name to path); its run method
    # forwards all arguments to the JIT function with extern_libs added, so callers using
    # the launch-grid indexing syntax need not pass the library.
        """
        KernelInterface invokes self.run in __getitem__, i.e. [].
        We direct the call to JITFunction.run with extern_libs injected,
        so that users don't have to pass it.
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
        # [FUNCTION DOC] HELPER: Stores the JIT function and the extern libraries mapping.
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Forwards positional and keyword arguments to the wrapped JIT
        # function run, injecting the stored extern libraries; returns its result.
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def build_requires_shmem_decorator(  # type: ignore[no-untyped-def]
        *,
        jit_func,
        find_device_library: Callable[[], str],
        extern_libs_key: str,
        registry: type[ShmemKernelRegistry],
        init_hook: Callable[..., None],
        error_prefix: str,
    ):
        # [FUNCTION DOC] Builds the requires-backend decorator result for a Triton JIT
        # function. Keyword args: the function, a callable locating the device library path,
        # the extern-libs key name, the kernel registry, the post-compile init hook and an
        # error message prefix. Raises TypeError if the input is not a JIT function. Side
        # effects: registers the kernel name and installs the init hook as the global Triton
        # post-compile hook (overwriting any previous one). Returns a GridCallableWithExtern
        # wrapping the function with the library injected.
        """Build the @requires_<backend> decorator for a Triton JIT function."""
        import triton
        from triton.runtime.jit import JITFunction

        if not isinstance(jit_func, JITFunction):
            raise TypeError(
                f"{error_prefix} must be applied to a @triton.jit function, "
                f"got {type(jit_func)}"
            )

        lib_path = find_device_library()
        extern_libs = {extern_libs_key: lib_path}
        registry.register(jit_func.fn.__name__)
        triton.knobs.runtime.jit_post_compile_hook = init_hook
        return GridCallableWithExtern(jit_func, extern_libs)
