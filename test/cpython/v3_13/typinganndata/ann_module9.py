# [MODULE DOC] HELPER: fixture for formatting annotations: builds Union annotations ann and ann1,
# where ann1 uses class A whose module and qualified name are overwritten at import to mimic a
# class from a module named testModule.typing.
# Test ``inspect.formatannotation``
# https://github.com/python/cpython/issues/96073

from typing import Union, List

ann = Union[List[str], int]

# mock typing._type_repr behaviour
# [CLASS DOC] HELPER: next line is an empty class A (single-line body, doc placed above);
# __module__ and __qualname__ are overwritten right after.
class A: ...

A.__module__ = 'testModule.typing'
A.__qualname__ = 'A'

ann1 = Union[List[A], int]
