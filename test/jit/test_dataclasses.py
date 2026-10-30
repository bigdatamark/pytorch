# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for scripting Python dataclasses: InitVar fields, post-init hooks with
# optional fields, generated ordering comparators, custom equality, unsupported default
# factories, and error cases. Defines the Point, mixup parameter dataclasses and enum fixtures
# at import time and builds a shared hypothesis strategy of bounded floats. Running the file
# directly is refused in favor of test/test_jit.py.
from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import List, Optional

from hypothesis import given, settings, strategies as st

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Example jittable dataclass
@dataclass(order=True)
class Point:
    # [CLASS DOC] Example jittable ordered dataclass with float fields x and y and an optional
    # tensor norm that is computed in the post-init hook; comparisons order by fields in
    # declaration order.
    x: float
    y: float
    norm: Optional[torch.Tensor] = None

    def __post_init__(self):
        # [FUNCTION DOC] Sets norm to the Euclidean norm of x and y, computed as a tensor.
        self.norm = (torch.tensor(self.x) ** 2 + torch.tensor(self.y) ** 2) ** 0.5


class MixupScheme(Enum):
    # [CLASS DOC] HELPER: Enum fixture with two members whose values are lists of strings;
    # list-valued enums are not supported by the script compiler.
    INPUT = ["input"]

    MANIFOLD = [
        "input",
        "before_fusion_projection",
        "after_fusion_projection",
        "after_classifier_projection",
    ]


@dataclass
class MixupParams:
    # [CLASS DOC] HELPER: Dataclass fixture with a hand-written init taking a float alpha and an
    # enum scheme (list-valued enum), which stores both; used to show scripting fails for
    # list-valued enums.
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme = MixupScheme.INPUT):
        # [FUNCTION DOC] HELPER: Stores alpha (default 0.125) and scheme (default INPUT member)
        # as attributes.
        self.alpha = alpha
        self.scheme = scheme


class MixupScheme2(Enum):
    # [CLASS DOC] HELPER: Enum fixture with two integer-valued members A and B, which the script
    # compiler supports.
    A = 1
    B = 2


@dataclass
class MixupParams2:
    # [CLASS DOC] HELPER: Dataclass fixture like the first mixup parameter class but with an
    # integer-valued enum default; scripting it must succeed.
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme2 = MixupScheme2.A):
        # [FUNCTION DOC] HELPER: Stores alpha (default 0.125) and scheme (default member A)
        # as attributes.
        self.alpha = alpha
        self.scheme = scheme


@dataclass
class MixupParams3:
    # [CLASS DOC] HELPER: Dataclass fixture with the same shape as the second mixup parameter
    # class, used only as a type annotation and never scripted directly.
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme2 = MixupScheme2.A):
        # [FUNCTION DOC] HELPER: Stores alpha (default 0.125) and scheme (default member A)
        # as attributes.
        self.alpha = alpha
        self.scheme = scheme


# Make sure the Meta internal tooling doesn't raise an overflow error
NonHugeFloats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False)


class TestDataclasses(JitTestCase):
    # [CLASS DOC] Test case for scripting dataclasses; the class registry is cleared once after
    # all tests.
    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class-level teardown that clears the JIT class registry so classes
        # registered by these tests do not leak into other tests.
        torch._C._jit_clear_class_registry()

    def test_init_vars(self):
        # [FUNCTION DOC] Scripts an ordered dataclass with an InitVar field (norm_p, default 2)
        # whose post-init hook computes a generalized p-norm of x and y into an optional tensor
        # field, then checks a function constructing it with p equal to 3 and returning the norm
        # via checkScript.
        @torch.jit.script
        @dataclass(order=True)
        class Point2:
            # [CLASS DOC] HELPER: Ordered dataclass with float fields x and y, an init-only
            # integer norm_p and an optional tensor norm; scripted inside the test.
            x: float
            y: float
            norm_p: InitVar[int] = 2
            norm: Optional[torch.Tensor] = None

            def __post_init__(self, norm_p: int):
                # [FUNCTION DOC] HELPER: Post-init hook receiving the init-only norm_p and
                # storing the p-norm of x and y, a tensor, in norm.
                self.norm = (
                    torch.tensor(self.x) ** norm_p + torch.tensor(self.y) ** norm_p
                ) ** (1 / norm_p)

        def fn(x: float, y: float, p: int):
            # [FUNCTION DOC] HELPER: Builds the scripted dataclass from two floats and an
            # integer and returns its norm field.
            pt = Point2(x, y, p)
            return pt.norm

        self.checkScript(fn, (1.0, 2.0, 3))

    # Sort of tests both __post_init__ and optional fields
    @settings(deadline=None)
    @given(NonHugeFloats, NonHugeFloats)
    def test__post_init__(self, x, y):
        # [FUNCTION DOC] Property test (hypothesis, bounded floats, no deadline) covering the
        # post-init hook and the optional norm field: scripts Point and checks via checkScript
        # that constructing it from two floats and returning the norm matches eager.
        P = torch.jit.script(Point)

        def fn(x: float, y: float):
            # [FUNCTION DOC] HELPER: Constructs the scripted Point from two floats and returns
            # its norm field.
            pt = P(x, y)
            return pt.norm

        self.checkScript(fn, [x, y])

    @settings(deadline=None)
    @given(
        st.tuples(NonHugeFloats, NonHugeFloats), st.tuples(NonHugeFloats, NonHugeFloats)
    )
    def test_comparators(self, pt1, pt2):
        # [FUNCTION DOC] Property test (hypothesis, pairs of bounded floats, no deadline)
        # checking the comparison operators generated for the ordered dataclass. Scripts Point
        # and uses checkScript on a function returning equality, less-than, less-equal,
        # greater-than and greater-equal results for two instances; not-equal is omitted (see
        # the TODO in the code about the interpreter).
        x1, y1 = pt1
        x2, y2 = pt2
        P = torch.jit.script(Point)

        def compare(x1: float, y1: float, x2: float, y2: float):
            # [FUNCTION DOC] HELPER: Builds two scripted Point values from coordinates and
            # returns a tuple of five comparison results.
            pt1 = P(x1, y1)
            pt2 = P(x2, y2)
            return (
                pt1 == pt2,
                # pt1 != pt2,   # TODO: Modify interpreter to auto-resolve (a != b) to not (a == b) when there's no __ne__
                pt1 < pt2,
                pt1 <= pt2,
                pt1 > pt2,
                pt1 >= pt2,
            )

        self.checkScript(compare, [x1, y1, x2, y2])

    def test_default_factories(self):
        # [FUNCTION DOC] Checks that a dataclass using a default factory field is unsupported:
        # scripting it must raise NotImplementedError. Code after the scripting call in the same
        # block is never reached.
        @dataclass
        class Foo(object):
            # [CLASS DOC] HELPER: Dataclass with one list-of-int field whose default comes from
            # a factory.
            x: List[int] = field(default_factory=list)

        with self.assertRaises(NotImplementedError):
            torch.jit.script(Foo)

            def fn():
                # [FUNCTION DOC] HELPER: Would construct Foo and return its list field, but is
                # never executed because the preceding script call raises first.
                foo = Foo()
                return foo.x

            torch.jit.script(fn)()

    # The user should be able to write their own __eq__ implementation
    # without us overriding it.
    def test_custom__eq__(self):
        # [FUNCTION DOC] Checks a user-defined equality method is respected rather than
        # overridden by the dataclass machinery: scripts a dataclass whose equality compares
        # only field a, then uses checkScript on a function comparing two instances with equal a
        # but different b, expecting equality to be true.
        @torch.jit.script
        @dataclass
        class CustomEq:
            # [CLASS DOC] HELPER: Scripted dataclass with int fields a and b and a custom
            # equality that compares only a.
            a: int
            b: int

            def __eq__(self, other: "CustomEq") -> bool:
                # [FUNCTION DOC] HELPER: Custom equality comparing only field a with the other
                # instance; b is ignored.
                return self.a == other.a  # ignore the b field

        def fn(a: int, b1: int, b2: int):
            # [FUNCTION DOC] HELPER: Builds two instances sharing a but with different b values
            # and returns whether they are equal.
            pt1 = CustomEq(a, b1)
            pt2 = CustomEq(a, b2)
            return pt1 == pt2

        self.checkScript(fn, [1, 2, 3])

    def test_no_source(self):
        # [FUNCTION DOC] Scripting the dataclass with the list-valued enum must raise
        # RuntimeError, while scripting the one with the integer-valued enum must succeed
        # without error.
        with self.assertRaises(RuntimeError):
            # uses list in Enum is not supported
            torch.jit.script(MixupParams)

        torch.jit.script(MixupParams2)  # don't throw

    def test_use_unregistered_dataclass_raises(self):
        # [FUNCTION DOC] Scripting a function whose argument is annotated with a dataclass that
        # was never registered via scripting must raise OSError.
        def f(a: MixupParams3):
            # [FUNCTION DOC] HELPER: Function annotated with the never-scripted dataclass type;
            # body just returns zero.
            return 0

        with self.assertRaises(OSError):
            torch.jit.script(f)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
