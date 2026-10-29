# mypy: allow-untyped-defs
# [MODULE DOC] Exposes torch.backends.kleidiai, which reports whether this PyTorch build includes KleidiAI (Arm optimized kernels) support.
# Contains a single query function; no import-time side effects.
import torch


def is_available():
    # [FUNCTION DOC] Returns True if the build was compiled with KleidiAI support, read from a compiled-extension flag. Takes no arguments, no side effects.
    r"""Return whether PyTorch is built with KleidiAI support."""
    return torch._C._has_kleidiai
