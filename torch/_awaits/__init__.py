# [PACKAGE DOC] Python-side definition of the Await type, a generic wrapper
# around the native torch._C._Await that represents delayed execution of a
# callable (used by the JIT awaitable primitives). Exports: Await (listed in the
# export list; note the class actually defined here is named _Await).
# KNOWN ISSUE: the export list names Await but no such name is defined in this
# module, so a star import would fail; only _Await exists.
from __future__ import annotations

from typing import Generic, TypeVar

import torch

# pyrefly: ignore [bad-dunder-all]
__all__ = ['Await']

W = TypeVar("W")

class _PyAwaitMeta(type(torch._C._Await), type(Generic)):  # type: ignore[misc, no-redef]
    # [CLASS DOC] HELPER: Metaclass combining the native Await metaclass with the
    # Generic metaclass so that the Python Await class can be both a subclass of
    # the native type and subscriptable. No state or methods of its own.
    pass

class _Await(torch._C._Await, Generic[W], metaclass=_PyAwaitMeta):
    # [CLASS DOC] Generic Python wrapper over the native Await object, holding a
    # callable plus its args for deferred execution, with result type W. All
    # behavior (fn, args, is_nowait, waiting) comes from the native base class;
    # in eager mode attribute access is forwarded to the awaited result. This
    # class adds only typing support and defines no methods itself.
    r"""
    Wrapper around a ``torch._C.Await`` which encapsulates delayed execution
    of a callable. All manipulations happen with functions ``torch.jit._awaitable``,
    ``torch.jit._awaitable_wait``, ``torch.jit._awaitable_nowait``.

    Torch scriptable manipulations:
    ``torch.jit._awaitable(func, *args)``
    Creates ``Await[W]`` object, where W is return type of func.

    Returns:
    ``torch.jit._awaitable_wait(Await[W])``
    Returns the result of the function, specified at ``_awaitable``,  with specified arguments.

    Returns:
        The result of type ``W`` of the function call. The result is owned by ``Await[W]``
        and returned on all following ``_awaitable_wait`` calls.


    ``torch.jit._awaitable_nowait(W)``
    Returns:
        Trivial ``Await[W]`` with specified result.


    Only in eager mode:
    ``fn() -> Callable[Tuple[Any], W]``
    Returns:
        Specified at ``_awaitable`` python function ``func``.

    ``args() -> Tuple[Any]``
    Returns:
        Specified at ``_awaitable`` python args.

    ``is_nowait() -> _bool``
    Returns:
        ``True`` if this object was created via ``_awaitable_nowait`` call (trivial `Await[W]`).

    In eager mode ``Await[W]`` can be used as ``W`` i.e. attributes of W can be called on ``Await[W]``,
    ``_awaitable_wait()`` call will be transparently added.
    """
