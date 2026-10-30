# mypy: enable-error-code=unused-ignore
# [MODULE DOC] Static typing check for torch.Size, which is a tuple subclass. It builds tuples
# of various lengths (including empty and a never-typed variadic one) and a Size, then asserts:
# Size is assignable to a tuple of ints; indexing with an int or an object providing the index
# protocol gives int; slicing gives Size; addition with Size or any tuple gives Size;
# multiplication by an int or index-protocol object (either side) gives Size. The reflected-
# addition (tuple plus Size) assertions carry type-ignore markers because the current mypy
# infers a different type (linked mypy issue); the mypy flag in the first line makes unused
# ignores an error so they must stay necessary. Executes the expressions at import time.

from typing_extensions import assert_type, Never

from torch import Size


class ZeroIndex:
    # [CLASS DOC] HELPER: Minimal class whose only member provides the integer-index protocol
    # and always yields zero. It is used to check that Size indexing and multiplication accept
    # index-like objects, not only int.
    def __index__(self) -> int:
        # [FUNCTION DOC] HELPER: Index-protocol method; always returns the integer 0.
        return 0


tup0: tuple[()] = ()
tup1: tuple[int] = (1,)
tup2: tuple[int, int] = (1, 2)
tupN: tuple[int, int, int] = (1, 2, 3)
tupX: tuple[Never, ...] = tuple()
s = Size([1, 2, 3])

# assignability to tuple
t: tuple[int, ...] = s

# __getitem__
assert_type(s[0], int)
assert_type(s[ZeroIndex()], int)
assert_type(s[:2], Size)
# __add__
assert_type(s + s, Size)
assert_type(s + tup0, Size)
assert_type(s + tup1, Size)
assert_type(s + tup2, Size)
assert_type(s + tupN, Size)
assert_type(s + tupX, Size)
# __radd__
# NOTE: currently incorrect inference, see: https://github.com/python/mypy/issues/19006
assert_type(tup0 + s, Size)  # type: ignore[assert-type]
assert_type(tup1 + s, Size)  # type: ignore[assert-type]
assert_type(tup2 + s, Size)  # type: ignore[assert-type]
assert_type(tupN + s, Size)  # type: ignore[assert-type]
assert_type(tupX + s, Size)  # type: ignore[assert-type]
# __mul__
assert_type(s * 3, Size)
assert_type(s * ZeroIndex(), Size)
assert_type(3 * s, Size)
assert_type(ZeroIndex() * s, Size)
