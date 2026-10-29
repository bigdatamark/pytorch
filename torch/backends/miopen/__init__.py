# mypy: allow-untyped-defs
# [PACKAGE DOC] Exposes torch.backends.miopen: runtime configuration of the MIOpen (AMD GPU) backend.
# Provides a flags context manager and set_flags for the 'immediate' mode setting, plus a module attribute 'immediate' backed by the C++ getter/setter.
# Import-time side effect: replaces this module in the module registry with a property-intercepting module wrapper.
import sys
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def set_flags(
    _immediate=None,
):
    # [FUNCTION DOC] Sets the MIOpen immediate-mode flag if a value is given (None leaves it unchanged) and returns a one-element tuple of the previous value so callers can restore it.
    orig_flags = (torch._C._get_miopen_immediate(),)
    if _immediate is not None:
        torch._C._set_miopen_immediate(_immediate)
    return orig_flags


@contextmanager
def flags(
    immediate=False,
):
    # [FUNCTION DOC] Context manager that temporarily sets MIOpen immediate mode (default False) and restores the previous value on exit, even if the body raises.
    # Uses the unbracketed-mutation guard while changing the setting.
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            immediate,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<miopen|mkldnn>.immediate = True


class MiopenModule(PropModule):
    # [CLASS DOC] Module wrapper class installed in place of this module; exposes the 'immediate' attribute as a context-aware property that reads/writes the C++ MIOpen immediate flag.
    immediate = ContextProp(
        torch._C._get_miopen_immediate, torch._C._set_miopen_immediate
    )


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MiopenModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
immediate: bool
