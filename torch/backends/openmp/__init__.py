# mypy: allow-untyped-defs
# [MODULE DOC] Exposes torch.backends.openmp, which reports whether this PyTorch build includes OpenMP threading support.
# Contains a single query function; no import-time side effects.
import torch


def is_available():
    # [FUNCTION DOC] Returns True if the build was compiled with OpenMP support, read from a compiled-extension flag. Takes no arguments, no side effects.
    r"""Return whether PyTorch is built with OpenMP support."""
    return torch._C.has_openmp
