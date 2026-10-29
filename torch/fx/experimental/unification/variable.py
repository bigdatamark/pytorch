# [MODULE DOC] Logic variables for unification. Defines the Var class, the
# var/vars factories, the type-dispatched isvar predicate, and the variables
# context manager that temporarily treats arbitrary hashable values as variables.
# Module state: a global set of such "logic variable" values (_global_logic_variables,
# alias _glv). Import-time side effect: registers isvar in the shared dispatch
# namespace.
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable
    from typing import Literal

from .dispatch import dispatch
from .utils import hashable


_global_logic_variables: set[Hashable] = set()
_glv = _global_logic_variables


class Var:
    # [CLASS DOC] A logic variable identified by its token. Equality and hash use
    # the class and token. Class attribute _id is a counter used to generate
    # unique names like _1, _2 for anonymous variables. Prints as ~token.
    """Logic Variable"""

    _id = 1

    def __new__(cls, *token: Hashable) -> Var:  # noqa: PYI034
        # [FUNCTION DOC] Create a variable. With no token, generate a fresh name
        # from the class counter (mutating it). With one token, use it directly;
        # with several, the token is the tuple of them. Returns the new object.
        if len(token) == 0:
            token = f"_{Var._id}"  # type: ignore[assignment]
            Var._id += 1
        elif len(token) == 1:
            token = token[0]  # pyrefly: ignore[bad-assignment]

        obj = object.__new__(cls)
        obj.token = token  # type: ignore[attr-defined]
        return obj

    def __str__(self) -> str:
        # [FUNCTION DOC] Return the token's text prefixed with a tilde.
        return "~" + str(self.token)  # type: ignore[attr-defined]

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when other has exactly the same type and an equal token.
        return type(self) is type(other) and self.token == other.token  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash of the type and token, consistent with equality.
        return hash((type(self), self.token))  # type: ignore[attr-defined]


def var() -> Callable[..., Var]:
    # [FUNCTION DOC] Takes no arguments and returns a function that builds a Var
    # from whatever arguments it is given.
    # KNOWN ISSUE: this does not create a variable itself, and calling it with a
    # name (as the docstrings in this package do) raises TypeError.
    return lambda *args: Var(*args)


def vars() -> Callable[[int], list[Callable[..., Var]]]:
    # [FUNCTION DOC] Takes no arguments and returns a function that, given n,
    # returns a list of n variable-factory functions (not Var objects).
    # KNOWN ISSUE: same shape problem as var; it does not directly create Vars.
    return lambda n: [var() for i in range(n)]


@dispatch(Var)
def isvar(v: Var) -> Literal[True]:
    # [FUNCTION DOC] Dispatch case for Var instances: always True.
    return True


@dispatch(object)  # type: ignore[no-redef]
# The noqa suppresses F811 for this multipledispatch overload redefinition
# (it replaces the bare-name-statement hack that previously silenced it).
def isvar(o: object) -> bool:  # noqa: F811
    # [FUNCTION DOC] Fallback dispatch case: True only if some values are
    # currently registered as logic variables and o is hashable and in that set.
    return bool(_glv and hashable(o) and o in _glv)


@contextmanager
def variables(*variables: Hashable) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that adds the given hashable values to the
    # global logic-variable set for the duration of the block, then restores the
    # previous set contents (even on exceptions). Mutates global state; not
    # thread-safe.
    """
    Context manager for logic variables

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> from __future__ import with_statement
        >>> with variables(1):
        ...     print(isvar(1))
        True
        >>> print(isvar(1))
        False
        >>> # Normal approach
        >>> from unification import unify
        >>> x = var("x")
        >>> unify(x, 1)
        {~x: 1}
        >>> # Context Manager approach
        >>> with variables("x"):
        ...     print(unify("x", 1))
        {'x': 1}
    """
    old_global_logic_variables = _global_logic_variables.copy()
    _global_logic_variables.update(set(variables))
    try:
        yield
    finally:
        _global_logic_variables.clear()
        _global_logic_variables.update(old_global_logic_variables)
