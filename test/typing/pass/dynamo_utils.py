# [MODULE DOC] Static typing check for the exact-type predicate from torch._dynamo.utils. It
# verifies that, when the predicate is given a tuple of candidate classes, the type checker
# narrows the tested value to the union of those classes, and that generic container parameters
# of the original declared type are preserved. Only defines two functions; nothing runs at
# import time.
from typing import Any
from typing_extensions import assert_type

from torch._dynamo.utils import istype


def check_istype_tuple_narrowing(x: object) -> None:
    # [FUNCTION DOC] Typing sample: takes an arbitrary object and, inside two conditional
    # branches guarded by the exact-type predicate, asserts the narrowed type is the union of
    # str and int for the first, and the union of list of Any and tuple of Any for the second.
    # Returns nothing; not meant to be called.
    if istype(x, (str, int)):
        assert_type(x, str | int)

    if istype(x, (list, tuple)):
        assert_type(x, list[Any] | tuple[Any, ...])


def check_istype_preserves_container_parameters(
    x: list[int] | tuple[int, ...] | str,
) -> None:
    # [FUNCTION DOC] Typing sample: takes a value declared as a list of int, tuple of int, or
    # str, and in the branch guarded by the exact-type predicate for list or tuple asserts the
    # narrowed type keeps the element types (list of int or tuple of int), i.e. the str
    # alternative is dropped and the parameters are preserved. Returns nothing; not meant to be
    # called.
    if istype(x, (list, tuple)):
        assert_type(x, list[int] | tuple[int, ...])
