# mypy: allow-untyped-defs
# [PACKAGE DOC] Exposes torch.backends.nnpack: availability query, a function to globally enable or disable NNPACK, and a flags context manager.
# Exports is_available, flags and set_flags. No import-time side effects.
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


__all__ = ["is_available", "flags", "set_flags"]


def is_available():
    # [FUNCTION DOC] Returns whether this build includes NNPACK support, by asking the torch core module.
    r"""Return whether PyTorch is built with NNPACK support."""
    return torch._nnpack_available()


def set_flags(_enabled):
    # [FUNCTION DOC] Globally enables or disables NNPACK using the given boolean and returns a one-element tuple holding the previous setting.
    # Side effect: mutates global backend state.
    r"""Set if nnpack is enabled globally"""
    orig_flags = (torch._C._get_nnpack_enabled(),)
    torch._C._set_nnpack_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=False):
    # [FUNCTION DOC] Context manager that sets the NNPACK enabled flag (default False) for the body and restores the prior value on exit, including on exceptions.
    r"""Context manager for setting if nnpack is enabled globally"""
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])
