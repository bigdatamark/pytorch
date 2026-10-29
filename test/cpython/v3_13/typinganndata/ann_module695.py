# [MODULE DOC] HELPER: fixture for PEP 695 generic syntax with postponed annotations: classes A and
# B, and a generic function, each with a type variable, a type-variable tuple and a ParamSpec.
from __future__ import annotations
from typing import Callable


class A[T, *Ts, **P]:
    # [CLASS DOC] HELPER: generic class annotating attributes with its type parameters T, Ts, P.
    x: T
    y: tuple[*Ts]
    z: Callable[P, str]


class B[T, *Ts, **P]:
    # [CLASS DOC] HELPER: generic class whose body rebinds T, Ts and P to ordinary values (int,
    # str, bytes) that shadow the type parameters when annotations are resolved.
    T = int
    Ts = str
    P = bytes
    x: T
    y: Ts
    z: P


# [FUNCTION DOC] HELPER: stub generic function (doc placed above because the body is on the closing
# line) taking x of type T, extra positional args of types Ts, and ParamSpec args and kwargs;
# does nothing and returns None.
def generic_function[T, *Ts, **P](
    x: T, *y: *Ts, z: P.args, zz: P.kwargs
) -> None: ...
