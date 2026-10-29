

"""
The module for testing variable annotations.
Empty lines above are for good reason (testing for correct line numbers)
"""

from typing import Optional
from functools import wraps

__annotations__[1] = 2

class C:
    # [CLASS DOC] HELPER: plain class with a class attribute x and an optional self-referencing
    # (string forward reference) annotated attribute y defaulting to None.

    x = 5; y: Optional['C'] = None

from typing import Tuple
x: int = 5; y: str = x; f: Tuple[int, int]

class M(type):
    # [CLASS DOC] HELPER: metaclass that injects extra entries into its own annotations mapping
    # at class creation (a key "123") and annotates a class attribute o.

    __annotations__['123'] = 123
    o: type = object

(pars): bool = True

class D(C):
    # [CLASS DOC] HELPER: subclass of C adding two annotated string attributes j and k.
    j: str = 'hi'; k: str= 'bye'

from types import new_class
h_class = new_class('H', (C,))
j_class = new_class('J')

class F():
    # [CLASS DOC] HELPER: class with one annotated class attribute z and a trivial initializer.
    z: int = 5
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: accepts one argument and does nothing.
        pass

class Y(F):
    # [CLASS DOC] HELPER: subclass of F whose initializer skips F and calls the next class in
    # the resolution order after F.
    def __init__(self):
        # [FUNCTION DOC] HELPER: calls the parent initializer found after F in the MRO with the
        # value 123 (super called with F as the start class); takes no other arguments.
        super(F, self).__init__(123)

class Meta(type):
    # [CLASS DOC] HELPER: metaclass that simply delegates class creation to type.
    def __new__(meta, name, bases, namespace):
        # [FUNCTION DOC] HELPER: creates and returns the new class by delegating to type.__new__
        # with the same name, bases and namespace.
        return super().__new__(meta, name, bases, namespace)

class S(metaclass = Meta):
    # [CLASS DOC] HELPER: class built through Meta with two annotated string attributes x and y.
    x: str = 'something'
    y: str = 'something else'

def foo(x: int = 10):
    # [FUNCTION DOC] HELPER: defines a nested function bar then calls it with no arguments.
    # KNOWN ISSUE: bar requires one argument, so calling foo raises a TypeError; the List name is
    # also not imported (only matters if the annotation is evaluated). Never meant to be run.
    def bar(y: List[str]):
        # [FUNCTION DOC] HELPER: body only performs a local annotated assignment of a string to x;
        # the parameter y is unused.
        x: str = 'yes'
    bar()

def dec(func):
    # [FUNCTION DOC] HELPER: decorator returning a wrapper that forwards all arguments to func and
    # returns its result; functools.wraps copies the metadata of func.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: calls the wrapped function with the given arguments and returns
        # its result.
        return func(*args, **kwargs)
    return wrapper

u: int | float

# [MODULE DOC] HELPER: fixture for variable-annotation tests; placed at the END of the file because
# the original relies on leading blank lines for line-number checks. Defines classes C, D, F, Y, S,
# metaclasses M and Meta, module-level annotated names, and functions foo and dec. Import-time side
# effects: assigns an entry into the module annotations mapping and builds classes H and J.
