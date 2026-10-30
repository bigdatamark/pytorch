# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that scalar types (torch._numpy float64) can be subclassed and combined
# with plain Python classes in various base orders. Defines helper classes A, B, C, D, B0, C0,
# HasNew and B1 at import time; the test class is skipped because scalar repr differs from
# numpy.

"""Test printing of scalar types."""

import functools
from unittest import skipIf as skipif

import pytest

import torch._numpy as np
from torch._numpy.testing import assert_
from torch.testing._internal.common_utils import run_tests, TestCase


skip = functools.partial(skipif, True)


class A:
    # [CLASS DOC] HELPER: plain empty Python base class used as a mixin in inheritance tests.
    pass


class B(A, np.float64):
    # [CLASS DOC] HELPER: subclass of A listed before the float64 scalar type; empty body.
    pass


class C(B):
    # [CLASS DOC] HELPER: empty subclass of B, a second level of inheritance.
    pass


class D(C, B):
    # [CLASS DOC] HELPER: empty class inheriting from C and B together (diamond style
    # hierarchy).
    pass


class B0(np.float64, A):
    # [CLASS DOC] HELPER: same as B but with the float64 scalar type listed before A; empty
    # body.
    pass


class C0(B0):
    # [CLASS DOC] HELPER: empty subclass of B0.
    pass


class HasNew:
    # [CLASS DOC] HELPER: plain class whose constructor hook is overridden to return a tuple,
    # used to check that the scalar base class takes priority when listed first.
    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: constructor hook that ignores normal instance creation and
        # returns a tuple of the class, positional arguments and keyword arguments.
        return cls, args, kwargs


class B1(np.float64, HasNew):
    # [CLASS DOC] HELPER: empty class inheriting from float64 first and HasNew second.
    pass


@skip(reason="scalar repr: numpy plans to make it more explicit")
class TestInherit(TestCase):
    # [CLASS DOC] Skipped tests (scalar repr may become more explicit) that subclass instances
    # built from floats print the same as plain floats.
    def test_init(self):
        # [FUNCTION DOC] Builds instances of B, C and D from floats and asserts the string form
        # of each equals the plain float text.
        x = B(1.0)
        assert_(str(x) == "1.0")
        y = C(2.0)
        assert_(str(y) == "2.0")
        z = D(3.0)
        assert_(str(z) == "3.0")

    def test_init2(self):
        # [FUNCTION DOC] Builds instances of B0 and C0 from floats and asserts their string form
        # equals the plain float text.
        x = B0(1.0)
        assert_(str(x) == "1.0")
        y = C0(2.0)
        assert_(str(y) == "2.0")

    def test_gh_15395(self):
        # [FUNCTION DOC] Regression test for numpy issue 15395: with the scalar type listed
        # first, B1 built from one float prints as a float, and building it with two arguments
        # raises TypeError.
        # HasNew is the second base, so `np.float64` should have priority
        x = B1(1.0)
        assert_(str(x) == "1.0")

        # previously caused RecursionError!?
        with pytest.raises(TypeError):
            B1(1.0, 2.0)


if __name__ == "__main__":
    run_tests()
