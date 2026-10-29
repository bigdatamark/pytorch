# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the builtin divmod function (nb_divmod slot): ints, floats, bools,
# error cases, user-defined __divmod__/__rdivmod__, subclass priority, NotImplemented fallback and
# a tensor error case. The test class is patched to trace unittest and class building. Import has
# no side effects beyond registering tests.

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbDivmod(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for divmod in Dynamo; patched with the dynamo configs
    # enable_trace_unittest and enable_trace_load_build_class so assert methods and class
    # definitions inside make_dynamo_test bodies can be traced.
    # --- Integer divmod ---
    @make_dynamo_test
    def test_divmod_integers(self):
        # [FUNCTION DOC] Checks divmod over positive ints gives the quotient and remainder pair.
        self.assertEqual(divmod(7, 2), (3, 1))
        self.assertEqual(divmod(8, 4), (2, 0))
        self.assertEqual(divmod(5, 3), (1, 2))

    @make_dynamo_test
    def test_divmod_negative(self):
        # [FUNCTION DOC] Checks divmod with a negative dividend or divisor follows floor semantics
        # (quotient rounds down, remainder takes the divisor sign).
        self.assertEqual(divmod(-7, 2), (-4, 1))
        self.assertEqual(divmod(7, -2), (-4, -1))

    # --- Floats ---

    @make_dynamo_test
    def test_divmod_floats(self):
        # [FUNCTION DOC] Checks divmod over two floats gives float quotient and remainder.
        self.assertEqual(divmod(7.5, 2.0), (3.0, 1.5))
        self.assertEqual(divmod(7.0, 2.0), (3.0, 1.0))

    @make_dynamo_test
    def test_divmod_mixed_int_float(self):
        # [FUNCTION DOC] Checks divmod mixing an int and a float on either side gives float
        # results.
        self.assertEqual(divmod(7, 2.0), (3.0, 1.0))
        self.assertEqual(divmod(7.5, 2), (3.0, 1.5))

    # --- Booleans ---

    @make_dynamo_test
    def test_divmod_bools(self):
        # [FUNCTION DOC] Checks divmod over bools and an int with True gives int results.
        self.assertEqual(divmod(True, True), (1, 0))
        self.assertEqual(divmod(5, True), (5, 0))

    # --- Errors ---

    @make_dynamo_test
    def test_divmod_by_zero_raises(self):
        # [FUNCTION DOC] Checks divmod by integer zero and by float zero both raise
        # ZeroDivisionError.
        with self.assertRaises(ZeroDivisionError):
            divmod(1, 0)
        with self.assertRaises(ZeroDivisionError):
            divmod(1.0, 0.0)

    @make_dynamo_test
    def test_divmod_str_raises(self):
        # [FUNCTION DOC] Checks divmod between an int and a str, in both orders, raises TypeError.
        with self.assertRaises(TypeError):
            divmod(1, "a")
        with self.assertRaises(TypeError):
            divmod("a", 1)

    # --- User-defined __divmod__ ---

    @make_dynamo_test
    def test_user_defined_divmod_basic(self):
        # [FUNCTION DOC] Checks a user-defined class with __divmod__ taking another instance and
        # delegating to divmod on the stored values.
        class C:
            # [CLASS DOC] HELPER: Test class holding a value; defines __divmod__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __divmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns divmod of the stored value and the other instance
                # value.
                return divmod(self.value, other.value)

        self.assertEqual(divmod(C(7), C(2)), (3, 1))

    @make_dynamo_test
    def test_user_defined_divmod_with_integer(self):
        # [FUNCTION DOC] Checks user-defined __divmod__ where the right operand is a plain int.
        class C:
            # [CLASS DOC] HELPER: Test class holding a value; defines __divmod__ taking a raw
            # number.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __divmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns divmod of the stored value and the raw other
                # operand.
                return divmod(self.value, other)

        self.assertEqual(divmod(C(7), 2), (3, 1))

    @make_dynamo_test
    def test_reversed_divmod_with_integer(self):
        # [FUNCTION DOC] Checks that an int on the left and an object with only __rdivmod__ on the
        # right dispatches to the reflected method.
        class C:
            # [CLASS DOC] HELPER: Test class holding a value; defines only __rdivmod__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __rdivmod__(self, other):
                # [FUNCTION DOC] HELPER: Reflected divmod: returns divmod of the left operand and
                # its own stored value.
                return divmod(other, self.value)

        self.assertEqual(divmod(7, C(2)), (3, 1))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Checks an int subclass overriding __divmod__ and __rdivmod__ takes
        # priority on the left and, as the right operand of an int, uses the reflected method.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose divmod methods return marker strings.
            def __divmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "IntSub.__divmod__"

            def __rdivmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "IntSub.__rdivmod__"

        self.assertEqual(divmod(IntSub(7), 2), "IntSub.__divmod__")
        self.assertEqual(divmod(7, IntSub(2)), "IntSub.__rdivmod__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline check with a plain object subclass: forward method used on the
        # left, reflected one used when the other operand is an int.
        class Base:
            # [CLASS DOC] HELPER: Plain class whose divmod methods return marker strings.
            def __divmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "Base.__divmod__"

            def __rdivmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "Base.__rdivmod__"

        self.assertEqual(divmod(Base(), 1), "Base.__divmod__")
        self.assertEqual(divmod(1, Base()), "Base.__rdivmod__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_divmod_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] Checks that when both __divmod__ and __rdivmod__ return NotImplemented,
        # divmod raises TypeError.
        class C:
            # [CLASS DOC] HELPER: Class whose forward and reflected divmod both return
            # NotImplemented.
            def __divmod__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

            def __rdivmod__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            divmod(a, a)

    @make_dynamo_test
    def test_divmod_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Checks fallback: the left operand declines with NotImplemented so the
        # right operand reflected method result is returned.
        class A:
            # [CLASS DOC] HELPER: Class whose __divmod__ returns NotImplemented.
            def __divmod__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Class with a reflected divmod returning a marker string.
            def __rdivmod__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string showing the reflected method
                # ran.
                return "B.__rdivmod__ called"

        result = divmod(A(), B())
        self.assertEqual(result, "B.__rdivmod__ called")

    # --- Tensor ---

    def test_divmod_tensor_raises_type_error(self):
        # [FUNCTION DOC] Not decorated with make_dynamo_test: compiles a function applying divmod
        # to two float tensors with the eager backend (no fullgraph) and asserts that calling it
        # raises TypeError, since tensors do not support divmod.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns divmod of the two arguments.
            return divmod(x, y)

        x = torch.randn(4)
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaises(TypeError):
            opt_fn(x, y)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
