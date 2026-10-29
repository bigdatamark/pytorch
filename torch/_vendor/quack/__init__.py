# [PACKAGE DOC] Vendored subset of the quack CuTeDSL kernel library. On import it applies
# two CuTeDSL workaround patches (ELF section fix and MLIR threading fix) before any compile.
# Exports lazily (via module attribute lookup) the rmsnorm function; unknown names raise
# AttributeError. Also defines __version__ and __upstream_sha__.
"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

The pinned upstream commit is recorded in ``__upstream_sha__`` below and is
sourced from ``PINNED_SHA`` in tools/vendoring/quack/vendor.sh. The
vendoring script verifies that commit is reachable from Dao-AILab/quack main
before applying the local FlexGEMM patchset. Only the modules required by
torch._native.ops.norm.rmsnorm_impl and selected GEMM epilogue implementation
paths are vendored. Imports are rewritten to be package-relative
so this copy is independent of any ``quack`` top-level package that may be
installed via pip. Custom op namespaces are renamed from ``quack::`` to
``torch_vendor_quack::`` for the same reason.
"""
__version__ = "0.5.0"
__upstream_sha__ = "99bd7973bf3dc6db40961e413d4bdfea6c6fee3e"

# Two CuTeDSL workarounds, both must run before the first cute.compile call:
#   - cutlass#3161: duplicate .text section flags break MCJIT in multi-process
#     loads (see cute_dsl_elf_fix).
#   - cutlass#3062: ir.Context spawns LLVM thread pools that leak across
#     compiles, eventually exhausting pthreads (see cute_dsl_mlir_threading).
from . import cute_dsl_elf_fix
from . import cute_dsl_mlir_threading

cute_dsl_elf_fix.patch()
cute_dsl_mlir_threading.patch()

def __getattr__(name):
    # [FUNCTION DOC] Lazy module attribute hook: for the name rmsnorm, imports and returns the
    # rmsnorm function from the rmsnorm submodule (deferring heavy imports); any other name
    # raises AttributeError.
    if name == "rmsnorm":
        from .rmsnorm import rmsnorm

        return rmsnorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "rmsnorm",
]
