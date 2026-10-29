# [MODULE DOC] Public helpers describing the PyTorch build: a configuration summary, the C++
# compiler flags and the parallelization settings, each returned as text from the C
# extension. No import-time side effects.

import torch


def show() -> str:
    # [FUNCTION DOC] Returns the human readable build configuration string.
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    return torch._C._show_config()


# TODO: In principle, we could provide more structured version/config
# information here. For now only CXX_FLAGS is exposed, as Timer
# uses them.
def _cxx_flags() -> str:
    # [FUNCTION DOC] HELPER: Returns the C++ compiler flags used to build PyTorch.
    """Returns the CXX_FLAGS used when building PyTorch."""
    return torch._C._cxx_flags()


def parallel_info() -> str:
    # [FUNCTION DOC] Returns a string describing the parallelization settings.
    r"""Returns detailed string with parallelization settings"""
    return torch._C._parallel_info()
