# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Augmented-assignment tests copied from CPython 3.13
# Lib/test/test_augassign.py, run under the Dynamo CPython test harness (CPythonTestCase).
# Contains one test class covering in-place operators on names, list items, dict items,
# sequences, and custom __iadd__/__add__/__radd__ dispatch. Runs the tests via run_tests
# when executed as a script; no import-time side effects beyond imports.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_augassign.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# Augmented assignment test.

import unittest


class AugAssignTest(CPythonTestCase):
    # [CLASS DOC] Test case for augmented assignment semantics (+=, *=, **=, -=, //=,
    # %=, &=, |=, ^=, /=, @=, >>=, <<=) on plain variables, containers and user classes.
    def testBasic(self):
        # [FUNCTION DOC] Applies a chain of augmented operators to
        # an integer variable and asserts the final result is 3.0.
        x = 2
        x += 1
        x *= 2
        x **= 2
        x -= 8
        x //= 5
        x %= 3
        x &= 2
        x |= 5
        x ^= 1
        x /= 2
        self.assertEqual(x, 3.0)

    def test_with_unpacking(self):
        # [FUNCTION DOC] Checks that an augmented assignment with a tuple
        # target is rejected: compiling such source raises SyntaxError.
        self.assertRaises(SyntaxError, compile, "x, b += 3", "<test>", "exec")

    def testInList(self):
        # [FUNCTION DOC] Applies the same operator chain to
        # an element of a list and asserts the result is 3.0.
        x = [2]
        x[0] += 1
        x[0] *= 2
        x[0] **= 2
        x[0] -= 8
        x[0] //= 5
        x[0] %= 3
        x[0] &= 2
        x[0] |= 5
        x[0] ^= 1
        x[0] /= 2
        self.assertEqual(x[0], 3.0)

    def testInDict(self):
        # [FUNCTION DOC] Applies the same operator chain to a
        # value stored in a dict and asserts the result is 3.0.
        x = {0: 2}
        x[0] += 1
        x[0] *= 2
        x[0] **= 2
        x[0] -= 8
        x[0] //= 5
        x[0] %= 3
        x[0] &= 2
        x[0] |= 5
        x[0] ^= 1
        x[0] /= 2
        self.assertEqual(x[0], 3.0)

    def testSequences(self):
        # [FUNCTION DOC] Checks in-place add and multiply on lists, including slice
        # targets, and that the list object keeps its identity (an alias sees the changes).
        x = [1,2]
        x += [3,4]
        x *= 2

        self.assertEqual(x, [1, 2, 3, 4, 1, 2, 3, 4])

        x = [1, 2, 3]
        y = x
        x[1:2] *= 2
        y[1:2] += [1]

        self.assertEqual(x, [1, 2, 1, 2, 3])
        self.assertTrue(x is y)

    def testCustomMethods1(self):
        # [FUNCTION DOC] Checks dispatch of augmented addition on user classes:
        # with only __add__ a new object is produced; __iadd__ returning self
        # keeps identity; __iadd__ returning a new object rebinds; setting
        # __iadd__ to None in a subclass blocks it and raises TypeError.

        class aug_test:
            # [CLASS DOC] HELPER: Class with __add__ and __radd__
            # (no in-place hook) wrapping a number in val.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the wrapped value.
                self.val = value
            def __radd__(self, val):
                # [FUNCTION DOC] HELPER: Returns the plain sum
                # of the other operand and the wrapped value.
                return self.val + val
            def __add__(self, val):
                # [FUNCTION DOC] HELPER: Returns a new aug_test holding the sum.
                return aug_test(self.val + val)

        class aug_test2(aug_test):
            # [CLASS DOC] HELPER: Subclass whose __iadd__ mutates val in place and returns self.
            def __iadd__(self, val):
                # [FUNCTION DOC] HELPER: Adds to val and returns self.
                self.val = self.val + val
                return self

        class aug_test3(aug_test):
            # [CLASS DOC] HELPER: Subclass whose __iadd__ returns a new instance of aug_test3.
            def __iadd__(self, val):
                # [FUNCTION DOC] HELPER: Returns a new aug_test3 holding the sum.
                return aug_test3(self.val + val)

        class aug_test4(aug_test3):
            # [CLASS DOC] HELPER: Subclass that sets __iadd__ to None to block
            # both the inherited in-place hook and the fallback to __add__.
            """Blocks inheritance, and fallback to __add__"""
            __iadd__ = None

        x = aug_test(1)
        y = x
        x += 10

        self.assertIsInstance(x, aug_test)
        self.assertTrue(y is not x)
        self.assertEqual(x.val, 11)

        x = aug_test2(2)
        y = x
        x += 10

        self.assertTrue(y is x)
        self.assertEqual(x.val, 12)

        x = aug_test3(3)
        y = x
        x += 10

        self.assertIsInstance(x, aug_test3)
        self.assertTrue(y is not x)
        self.assertEqual(x.val, 13)

        x = aug_test4(4)
        with self.assertRaises(TypeError):
            x += 10


    def testCustomMethods2(test_self):
        # [FUNCTION DOC] Defines a class implementing every binary, reflected and in-place
        # operator hook so each appends a marker string to a shared list, applies each operator
        # form to an instance (including reversed operand forms with an int on the left), and
        # asserts the list of recorded markers equals the expected sequence of hook names.
        output = []

        class testall:
            # [CLASS DOC] HELPER: Operator-recording helper: each hook appends a
            # message such as "__add__ called" to the enclosing test's output list;
            # most return None, floordiv, truediv and the in-place hooks return self.
            def __add__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the add
                # operator hook in the enclosing output list and returns None.
                output.append("__add__ called")
            def __radd__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the radd
                # operator hook in the enclosing output list and returns None.
                output.append("__radd__ called")
            def __iadd__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the iadd operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__iadd__ called")
                return self

            def __sub__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the sub
                # operator hook in the enclosing output list and returns None.
                output.append("__sub__ called")
            def __rsub__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rsub
                # operator hook in the enclosing output list and returns None.
                output.append("__rsub__ called")
            def __isub__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the isub operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__isub__ called")
                return self

            def __mul__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the mul
                # operator hook in the enclosing output list and returns None.
                output.append("__mul__ called")
            def __rmul__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rmul
                # operator hook in the enclosing output list and returns None.
                output.append("__rmul__ called")
            def __imul__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the imul operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__imul__ called")
                return self

            def __matmul__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the matmul
                # operator hook in the enclosing output list and returns None.
                output.append("__matmul__ called")
            def __rmatmul__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rmatmul
                # operator hook in the enclosing output list and returns None.
                output.append("__rmatmul__ called")
            def __imatmul__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the imatmul operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__imatmul__ called")
                return self

            def __floordiv__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the floordiv operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__floordiv__ called")
                return self
            def __ifloordiv__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the ifloordiv operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__ifloordiv__ called")
                return self
            def __rfloordiv__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rfloordiv operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__rfloordiv__ called")
                return self

            def __truediv__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the truediv operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__truediv__ called")
                return self
            def __rtruediv__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rtruediv operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__rtruediv__ called")
                return self
            def __itruediv__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the itruediv operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__itruediv__ called")
                return self

            def __mod__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the mod
                # operator hook in the enclosing output list and returns None.
                output.append("__mod__ called")
            def __rmod__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rmod
                # operator hook in the enclosing output list and returns None.
                output.append("__rmod__ called")
            def __imod__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the imod operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__imod__ called")
                return self

            def __pow__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the pow
                # operator hook in the enclosing output list and returns None.
                output.append("__pow__ called")
            def __rpow__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rpow
                # operator hook in the enclosing output list and returns None.
                output.append("__rpow__ called")
            def __ipow__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the ipow operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__ipow__ called")
                return self

            def __or__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the or
                # operator hook in the enclosing output list and returns None.
                output.append("__or__ called")
            def __ror__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the ror
                # operator hook in the enclosing output list and returns None.
                output.append("__ror__ called")
            def __ior__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the ior operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__ior__ called")
                return self

            def __and__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the and
                # operator hook in the enclosing output list and returns None.
                output.append("__and__ called")
            def __rand__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rand
                # operator hook in the enclosing output list and returns None.
                output.append("__rand__ called")
            def __iand__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the iand operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__iand__ called")
                return self

            def __xor__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the xor
                # operator hook in the enclosing output list and returns None.
                output.append("__xor__ called")
            def __rxor__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rxor
                # operator hook in the enclosing output list and returns None.
                output.append("__rxor__ called")
            def __ixor__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the ixor operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__ixor__ called")
                return self

            def __rshift__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rshift
                # operator hook in the enclosing output list and returns None.
                output.append("__rshift__ called")
            def __rrshift__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rrshift
                # operator hook in the enclosing output list and returns None.
                output.append("__rrshift__ called")
            def __irshift__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the irshift operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__irshift__ called")
                return self

            def __lshift__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the lshift
                # operator hook in the enclosing output list and returns None.
                output.append("__lshift__ called")
            def __rlshift__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the rlshift
                # operator hook in the enclosing output list and returns None.
                output.append("__rlshift__ called")
            def __ilshift__(self, val):
                # [FUNCTION DOC] HELPER: Records the marker string for the ilshift operator
                # hook in the enclosing output list and returns the instance itself.
                output.append("__ilshift__ called")
                return self

        x = testall()
        x + 1
        1 + x
        x += 1

        x - 1
        1 - x
        x -= 1

        x * 1
        1 * x
        x *= 1

        x @ 1
        1 @ x
        x @= 1

        x / 1
        1 / x
        x /= 1

        x // 1
        1 // x
        x //= 1

        x % 1
        1 % x
        x %= 1

        x ** 1
        1 ** x
        x **= 1

        x | 1
        1 | x
        x |= 1

        x & 1
        1 & x
        x &= 1

        x ^ 1
        1 ^ x
        x ^= 1

        x >> 1
        1 >> x
        x >>= 1

        x << 1
        1 << x
        x <<= 1

        test_self.assertEqual(output, '''\
__add__ called
__radd__ called
__iadd__ called
__sub__ called
__rsub__ called
__isub__ called
__mul__ called
__rmul__ called
__imul__ called
__matmul__ called
__rmatmul__ called
__imatmul__ called
__truediv__ called
__rtruediv__ called
__itruediv__ called
__floordiv__ called
__rfloordiv__ called
__ifloordiv__ called
__mod__ called
__rmod__ called
__imod__ called
__pow__ called
__rpow__ called
__ipow__ called
__or__ called
__ror__ called
__ior__ called
__and__ called
__rand__ called
__iand__ called
__xor__ called
__rxor__ called
__ixor__ called
__rshift__ called
__rrshift__ called
__irshift__ called
__lshift__ called
__rlshift__ called
__ilshift__ called
'''.splitlines())

if __name__ == '__main__':
    run_tests()
