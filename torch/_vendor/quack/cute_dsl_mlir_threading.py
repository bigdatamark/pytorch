# Copyright (c) 2025-2026, QuACK team.
# [MODULE DOC] Workaround for MLIR context thread-pool leaks in CuTeDSL (cutlass#3062): patch() swaps ir.Context for a subclass
# that disables multithreading. Can be skipped via env var QUACK_DISABLE_MLIR_THREADING_PATCH=1. No import-time side effects.
"""Work around CuTeDSL MLIR context thread-pool leaks.

NVIDIA/cutlass#3062 documents that MLIR contexts keep LLVM worker thread pools
alive when context threading is enabled. Large pytest sweeps compile many CuTe
kernels in one Python process, so these idle workers accumulate until LLVM
eventually fails to create another pthread.

Remove this workaround once QuACK updates to cutlass-dsl 4.5.
"""

import os

_PATCHED = False


def patch() -> None:
    # [FUNCTION DOC] Replace cutlass._mlir.ir.Context with a subclass whose instances call enable_multithreading(False) after
    # construction. Idempotent (module flag plus marker attribute on the class); no-op if the env var disables it or cutlass is
    # not importable. Side effect: global replacement of ir.Context.
    """Disable MLIR context threading for new CuTeDSL contexts. Idempotent."""
    global _PATCHED
    if _PATCHED or os.getenv("QUACK_DISABLE_MLIR_THREADING_PATCH", "0") == "1":
        return
    try:
        from cutlass._mlir import ir
    except Exception:
        return  # CuTeDSL not importable; nothing to patch

    orig_context = ir.Context
    if getattr(orig_context, "_quack_mlir_threading_patch", False):
        _PATCHED = True
        return

    class SingleThreadedContext(orig_context):
    # [CLASS DOC] HELPER: Subclass of the original MLIR Context that carries a marker attribute and turns off context
    # multithreading on construction; name, qualname and module are copied from the original class.
        _quack_mlir_threading_patch = True

        def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the parent constructor with the given arguments, then disable MLIR multithreading.
            super().__init__(*args, **kwargs)
            self.enable_multithreading(False)

    SingleThreadedContext.__name__ = orig_context.__name__
    SingleThreadedContext.__qualname__ = orig_context.__qualname__
    SingleThreadedContext.__module__ = orig_context.__module__
    ir.Context = SingleThreadedContext
    _PATCHED = True
