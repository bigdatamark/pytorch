# mypy: allow-untyped-defs
# [MODULE DOC] Exposes torch.backends.mkl: an availability query and an on-demand oneMKL verbose-logging context manager.
# Defines VERBOSE_OFF (0) and VERBOSE_ON (1) level constants. No import-time side effects.
import torch


def is_available():
    # [FUNCTION DOC] Returns True if the build was compiled with MKL support, read from a compiled-extension flag.
    r"""Return whether PyTorch is built with MKL support."""
    return torch._C.has_mkl


VERBOSE_OFF = 0
VERBOSE_ON = 1


class verbose:
    # [CLASS DOC] Context manager that turns oneMKL verbose output on for the duration of a with block, so verbose messages can be limited to a chosen scope.
    # State: the requested verbose level in the attribute enable. Entering raises an error if MKL cannot be put into verbose mode; exiting always switches verbosing off.
    """
    On-demand oneMKL verbosing functionality.

    To make it easier to debug performance issues, oneMKL can dump verbose
    messages containing execution information like duration while executing
    the kernel. The verbosing functionality can be invoked via an environment
    variable named `MKL_VERBOSE`. However, this methodology dumps messages in
    all steps. Those are a large amount of verbose messages. Moreover, for
    investigating the performance issues, generally taking verbose messages
    for one single iteration is enough. This on-demand verbosing functionality
    makes it possible to control scope for verbose message dumping. In the
    following example, verbose messages will be dumped out for the second
    inference only.

    .. highlight:: python
    .. code-block:: python

        import torch

        model(data)
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
    """

    def __init__(self, enable):
        # [FUNCTION DOC] Stores the requested verbose level (VERBOSE_OFF or VERBOSE_ON) in the attribute enable.
        self.enable = enable

    def __enter__(self):
        # [FUNCTION DOC] On entry, does nothing when the level is off (returns None); otherwise asks MKL to enable verbose mode and raises AssertionError if that fails; returns the manager itself.
        # Side effect: changes process-wide MKL verbose state.
        if self.enable == VERBOSE_OFF:
            return
        st = torch._C._verbose.mkl_set_verbose(self.enable)
        if not st:
            raise AssertionError(
                "Failed to set MKL into verbose mode. Please consider to disable this verbose scope."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] On exit, unconditionally switches MKL verbose mode off (even if it was never enabled) and returns False so exceptions propagate.
        torch._C._verbose.mkl_set_verbose(VERBOSE_OFF)
        return False
