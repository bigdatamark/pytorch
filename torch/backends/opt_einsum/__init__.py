# mypy: allow-untyped-defs
# [MODULE DOC] Exposes torch.backends.opt_einsum: controls whether einsum uses the optional opt_einsum package to find an optimal contraction path.
# Import-time side effects: tries to import opt_einsum (falls back to None), sets module state enabled (True if available) and strategy (auto if available, else None), and replaces this module in the module registry with a property-intercepting wrapper.
# Exports availability query, package accessor, set_flags and a flags context manager.
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from typing import Any

from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


try:
    import opt_einsum as _opt_einsum  # type: ignore[import]
except ImportError:
    _opt_einsum = None


@_lru_cache
def is_available() -> bool:
    # [FUNCTION DOC] Returns True if the opt_einsum package was imported successfully; result is cached.
    r"""Return a bool indicating if opt_einsum is currently available.

    You must install opt-einsum in order for torch to automatically optimize einsum. To
    make opt-einsum available, you can install it along with torch: ``pip install torch[opt-einsum]``
    or by itself: ``pip install opt-einsum``. If the package is installed, torch will import
    it automatically and use it accordingly. Use this function to check whether opt-einsum
    was installed and properly imported by torch.
    """
    return _opt_einsum is not None


def get_opt_einsum() -> Any:
    # [FUNCTION DOC] Returns the imported opt_einsum package or None when unavailable.
    r"""Return the opt_einsum package if opt_einsum is currently available, else None."""
    return _opt_einsum


def _set_enabled(_enabled: bool) -> None:
    # [FUNCTION DOC] HELPER: Sets the module-level enabled flag; raises ValueError when asked to enable while opt_einsum is not installed.
    if not is_available() and _enabled:
        raise ValueError(
            f"opt_einsum is not available, so setting `enabled` to {_enabled} will not reap "
            "the benefits of calculating an optimal path for einsum. torch.einsum will "
            "fall back to contracting from left to right. To enable this optimal path "
            "calculation, please install opt-einsum."
        )
    global enabled
    enabled = _enabled


def _get_enabled() -> bool:
    # [FUNCTION DOC] HELPER: Returns the module-level enabled flag.
    return enabled


def _set_strategy(_strategy: str) -> None:
    # [FUNCTION DOC] HELPER: Sets the module-level contraction strategy. Raises ValueError if opt_einsum is unavailable, if it is not enabled, or if the value is not one of auto, greedy, optimal.
    if not is_available():
        raise ValueError(
            f"opt_einsum is not available, so setting `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please install opt_einsum or unset `strategy`."
        )
    if not enabled:
        raise ValueError(
            f"opt_einsum is not enabled, so setting a `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please set `enabled` to `True` as well or unset `strategy`."
        )
    if _strategy not in ["auto", "greedy", "optimal"]:
        raise ValueError(
            f"`strategy` must be one of the following: [auto, greedy, optimal] but is {_strategy}"
        )
    global strategy
    strategy = _strategy


def _get_strategy() -> str:
    # [FUNCTION DOC] HELPER: Returns the module-level strategy string.
    # pyrefly: ignore [bad-return]
    return strategy


def set_flags(_enabled=None, _strategy=None):
    # [FUNCTION DOC] HELPER: Applies the given enabled and strategy values (None means leave unchanged) through the validating setters and returns a tuple of the previous enabled flag and previous strategy (None if opt_einsum unavailable). Raises ValueError from the setters.
    orig_flags = (enabled, None if not is_available() else strategy)
    if _enabled is not None:
        _set_enabled(_enabled)
    if _strategy is not None:
        _set_strategy(_strategy)
    return orig_flags


@contextmanager
def flags(enabled=None, strategy=None):
    # [FUNCTION DOC] Context manager that temporarily sets enabled and/or strategy (None leaves each unchanged) and restores the previous values on exit, even on error.
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, strategy)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.opt_einsum.enabled = True


class OptEinsumModule(PropModule):
    # [CLASS DOC] Module wrapper installed in place of this module; exposes enabled (always) and strategy (only when opt_einsum is available; otherwise it is a plain None attribute) as context-aware properties.
    global enabled
    enabled = ContextProp(_get_enabled, _set_enabled)
    global strategy
    strategy = None
    if is_available():
        strategy = ContextProp(_get_strategy, _set_strategy)


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = OptEinsumModule(sys.modules[__name__], __name__)

enabled = bool(is_available())
strategy = "auto" if is_available() else None
