# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for true division (nb_true_divide slot, the slash operator) run through
# make_dynamo_test. Covers ints (always promoting to float), floats, complex, bools, error cases,
# in-place division, user-defined forward, reflected and in-place methods, int subclass priority,
# NotImplemented fallbacks, and compiled symbolic-size and tensor division. No import-time side
# effects beyond decorator application.

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbTrueDivide(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for true division under Dynamo. Class decorators enable unittest
    # tracing and class construction tracing for all tests, so classes defined in test bodies are
    # traced.
    # --- Integer truediv (always promotes to float) ---
    @make_dynamo_test
    def test_truediv_integers(self):
        # [FUNCTION DOC] Integer true division always returns floats, including exact quotients and
        # zero numerator.
        self.assertEqual(7 / 2, 3.5)
        self.assertEqual(8 / 2, 4.0)
        self.assertEqual(0 / 5, 0.0)

    @make_dynamo_test
    def test_truediv_negative(self):
        # [FUNCTION DOC] True division with negative operands.
        self.assertEqual(-7 / 2, -3.5)
        self.assertEqual(7 / -2, -3.5)

    @make_dynamo_test
    def test_truediv_chained(self):
        # [FUNCTION DOC] Chained true division evaluates left to right.
        self.assertEqual(100 / 5 / 2, 10.0)

    # --- Floats ---

    @make_dynamo_test
    def test_truediv_floats(self):
        # [FUNCTION DOC] Float true division.
        self.assertEqual(7.0 / 2.0, 3.5)
        self.assertEqual(1.0 / 4.0, 0.25)

    @make_dynamo_test
    def test_truediv_mixed_int_float(self):
        # [FUNCTION DOC] Mixed int and float true division.
        self.assertEqual(7 / 2.0, 3.5)
        self.assertEqual(7.0 / 2, 3.5)

    # --- Complex ---

    @make_dynamo_test
    def test_truediv_complex(self):
        # [FUNCTION DOC] Complex divided by an int gives a complex result.
        self.assertEqual((4 + 2j) / 2, (2 + 1j))

    # --- Booleans ---

    @make_dynamo_test
    def test_truediv_bools(self):
        # [FUNCTION DOC] True division with bool operands.
        self.assertEqual(True / True, 1.0)
        self.assertEqual(False / True, 0.0)

    @make_dynamo_test
    def test_truediv_int_and_bool(self):
        # [FUNCTION DOC] True division mixing an int and a bool.
        self.assertEqual(5 / True, 5.0)
        self.assertEqual(True / 2, 0.5)

    # --- Errors ---

    @make_dynamo_test
    def test_truediv_by_zero_raises(self):
        # [FUNCTION DOC] Integer and float division by zero both raise ZeroDivisionError.
        with self.assertRaises(ZeroDivisionError):
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            1.0 / 0.0

    @make_dynamo_test
    def test_truediv_str_raises(self):
        # [FUNCTION DOC] Dividing an int by a string, or a string by an int, raises TypeError.
        with self.assertRaises(TypeError):
            1 / "a"
        with self.assertRaises(TypeError):
            "a" / 1

    # --- Inplace /= ---

    @make_dynamo_test
    def test_inplace_truediv_integers(self):
        # [FUNCTION DOC] In-place true division on an int variable gives a float.
        x = 10
        x /= 4
        self.assertEqual(x, 2.5)

    @make_dynamo_test
    def test_inplace_truediv_chained(self):
        # [FUNCTION DOC] Two successive in-place true divisions on an int variable.
        x = 100
        x /= 5
        x /= 2
        self.assertEqual(x, 10.0)

    # --- User-defined __truediv__ ---

    @make_dynamo_test
    def test_user_defined_truediv_basic(self):
        # [FUNCTION DOC] Local class with forward true division on another instance; checks
        # instance divided by instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward true division and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Forward true division: new instance with own value divided
                # by the other instance's value.
                return type(self)(self.value / other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) / C(2), C(3.5))

    @make_dynamo_test
    def test_user_defined_truediv_with_integer(self):
        # [FUNCTION DOC] Local class with forward true division taking a raw int; checks instance
        # divided by int.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward true division and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Forward true division: new instance with own value divided
                # by the raw operand.
                return type(self)(self.value / other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) / 2, C(3.5))

    @make_dynamo_test
    def test_reversed_truediv_with_integer(self):
        # [FUNCTION DOC] Local class with only a reflected true division; checks int divided by
        # instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with reflected true division and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rtruediv__(self, other):
                # [FUNCTION DOC] HELPER: Reflected true division: new instance with the raw operand
                # divided by own value.
                return type(self)(other / self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(7 / C(2), C(3.5))

    @make_dynamo_test
    def test_inplace_user_defined_truediv(self):
        # [FUNCTION DOC] Local class with an in-place true division that mutates and returns self;
        # checks the in-place operator.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with in-place true division and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __itruediv__(self, other):
                # [FUNCTION DOC] HELPER: In-place true division: divides own value by the other
                # instance's value and returns self.
                self.value /= other.value
                return self

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        x = C(10)
        x /= C(4)
        self.assertEqual(x, C(2.5))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Int subclass overriding forward and reflected true division to return
        # markers; override used on both sides.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose division methods return marker strings.
            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Forward true division: returns the forward marker string.
                return "IntSub.__truediv__"

            def __rtruediv__(self, other):
                # [FUNCTION DOC] HELPER: Reflected true division: returns the reflected marker
                # string.
                return "IntSub.__rtruediv__"

        self.assertEqual(IntSub(7) / 2, "IntSub.__truediv__")
        self.assertEqual(7 / IntSub(2), "IntSub.__rtruediv__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline plain class with forward and reflected true division markers used
        # with an int on either side.
        class Base:
            # [CLASS DOC] HELPER: Local class whose division methods return marker strings.
            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Forward true division: returns the forward marker string.
                return "Base.__truediv__"

            def __rtruediv__(self, other):
                # [FUNCTION DOC] HELPER: Reflected true division: returns the reflected marker
                # string.
                return "Base.__rtruediv__"

        self.assertEqual(Base() / 1, "Base.__truediv__")
        self.assertEqual(1 / Base(), "Base.__rtruediv__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_truediv_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] A class whose forward and reflected true division return NotImplemented
        # raises TypeError when divided by itself.
        class C:
            # [CLASS DOC] HELPER: Local class with both division methods declining.
            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Forward true division: always NotImplemented.
                return NotImplemented

            def __rtruediv__(self, other):
                # [FUNCTION DOC] HELPER: Reflected true division: always NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a / a

    @make_dynamo_test
    def test_truediv_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Left forward division declines, so the right operand's reflected division
        # marker is returned.
        class A:
            # [CLASS DOC] HELPER: Local class whose forward true division declines.
            def __truediv__(self, other):
                # [FUNCTION DOC] HELPER: Forward true division: always NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Local class defining only a reflected true division returning a
            # marker.
            def __rtruediv__(self, other):
                # [FUNCTION DOC] HELPER: Reflected true division: returns the marker string.
                return "B.__rtruediv__ called"

        result = A() / B()
        self.assertEqual(result, "B.__rtruediv__ called")

    # --- SymNode truediv ---

    def test_truediv_symnode_and_int(self):
        # [FUNCTION DOC] Compiles a nested function with fullgraph and dynamic first dimension that
        # adds half the symbolic size to the tensor; compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function adding the size divided by two to its tensor
            # argument.
            s = x.shape[0]
            return x + (s / 2)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_truediv_int_and_symnode(self):
        # [FUNCTION DOC] Same but with sixteen divided by the symbolic size added to the tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function adding sixteen divided by the size to its
            # tensor argument.
            s = x.shape[0]
            return x + (16 / s)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_truediv_tensor(self):
        # [FUNCTION DOC] Compiles tensor divided by tensor with fullgraph, with positive divisors,
        # and compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the elementwise quotient of its two
            # tensor arguments.
            return x / y

        x = torch.randn(4)
        y = torch.randn(4).abs() + 1
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
