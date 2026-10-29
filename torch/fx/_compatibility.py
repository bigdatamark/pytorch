# [MODULE DOC]
# Backward-compatibility markers for FX APIs. Holds two registries (dicts used as
# ordered sets) of marked objects and the compatibility decorator factory.
# Import time: creates the empty registries and a type variable.
import textwrap
from collections.abc import Callable
from typing import Any, TypeVar


_BACK_COMPAT_OBJECTS: dict[Any, None] = {}
_MARKED_WITH_COMPATIBILITY: dict[Any, None] = {}


_T = TypeVar("_T")


def compatibility(is_backward_compatible: bool) -> Callable[[_T], _T]:
    # [FUNCTION DOC] Decorator factory. Returns a decorator that appends a backward-compatible note
    # (if True) or an experimental/not-compatible warning (if False) to the decorated object's
    # docstring, records it in the registries, and returns the same object. Mutates the
    # object's __doc__ and module-level state.
    if is_backward_compatible:

        def mark_back_compat(fn: _T) -> _T:
            # [FUNCTION DOC] HELPER: dedent fn's docstring, append the compatibility-guaranteed note, set it
            # back, register fn in both registries, and return fn unchanged otherwise.
            docstring = textwrap.dedent(getattr(fn, "__doc__", None) or "")
            docstring += """

.. note::
    Backwards-compatibility for this API is guaranteed.
"""
            fn.__doc__ = docstring
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn

        return mark_back_compat
    else:

        def mark_not_back_compat(fn: _T) -> _T:
            # [FUNCTION DOC] HELPER: dedent fn's docstring, append the experimental warning, set it back,
            # register fn only in the marked-with-compatibility registry, and return fn.
            docstring = textwrap.dedent(getattr(fn, "__doc__", None) or "")
            docstring += """

.. warning::
    This API is experimental and is *NOT* backward-compatible.
"""
            fn.__doc__ = docstring
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn

        return mark_not_back_compat
