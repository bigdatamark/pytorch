# [MODULE DOC] HELPER: fixture for generics across modules: optional forward-reference defaults,
# generic classes A and B (B nests another A), plus a PEP 695 alias and a TypeAliasType alias.
"""Module for testing the behavior of generics across different modules."""

from typing import TypeVar, Generic, Optional, TypeAliasType

default_a: Optional['A'] = None
default_b: Optional['B'] = None

T = TypeVar('T')


class A(Generic[T]):
    # [CLASS DOC] HELPER: generic class with a forward-reference attribute some_b naming B.
    some_b: 'B'


class B(Generic[T]):
    # [CLASS DOC] HELPER: generic class holding a nested generic A and attributes that refer to
    # inner A, the nested name A, and the outer A through string annotations.
    class A(Generic[T]):
        # [CLASS DOC] HELPER: empty nested generic class shadowing outer A inside B.
        pass

    my_inner_a1: 'B.A'
    my_inner_a2: A
    my_outer_a: 'A'  # unless somebody calls get_type_hints with localns=B.__dict__

type Alias = int
OldStyle = TypeAliasType("OldStyle", int)
