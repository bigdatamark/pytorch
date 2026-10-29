# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the remainder operator (nb_remainder slot, the percent operator)
# run through make_dynamo_test. Covers ints, floats, bools, string and bytes formatting, error
# cases, in-place remainder, user-defined forward, reflected and in-place methods, int subclass
# priority, NotImplemented fallbacks, and compiled symbolic-size and tensor remainder. No import-
# time side effects beyond decorator application.

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbRemainder(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the remainder operator under Dynamo. Class decorators enable
    # tracing of unittest and class construction for all tests, so classes defined in test bodies
    # are traced.
    # --- Integer remainder ---
    @make_dynamo_test
    def test_mod_integers(self):
        # [FUNCTION DOC] Integer remainder for several operand pairs.
        self.assertEqual(7 % 2, 1)
        self.assertEqual(8 % 4, 0)
        self.assertEqual(5 % 3, 2)

    @make_dynamo_test
    def test_mod_negative(self):
        # [FUNCTION DOC] Negative operands: the result takes the sign of the divisor.
        # Python remainder follows the sign of the divisor.
        self.assertEqual(-7 % 3, 2)
        self.assertEqual(7 % -3, -2)

    @make_dynamo_test
    def test_mod_chained(self):
        # [FUNCTION DOC] Chained remainder evaluates left to right.
        self.assertEqual(100 % 7 % 3, 2)

    # --- Floats ---

    @make_dynamo_test
    def test_mod_floats(self):
        # [FUNCTION DOC] Float remainder results.
        self.assertEqual(7.5 % 2.0, 1.5)
        self.assertEqual(7.0 % 2.0, 1.0)

    @make_dynamo_test
    def test_mod_mixed_int_float(self):
        # [FUNCTION DOC] Mixed int and float remainder gives floats.
        self.assertEqual(7 % 2.0, 1.0)
        self.assertEqual(7.5 % 2, 1.5)

    # --- Booleans ---

    @make_dynamo_test
    def test_mod_bools(self):
        # [FUNCTION DOC] Remainder with bool operands.
        self.assertEqual(True % True, 0)
        self.assertEqual(False % True, 0)

    @make_dynamo_test
    def test_mod_int_and_bool(self):
        # [FUNCTION DOC] Remainder mixing an int and a bool.
        self.assertEqual(5 % True, 0)
        self.assertEqual(True % 2, 1)

    # --- String %-formatting (str/bytes also own nb_remainder) ---

    @make_dynamo_test
    def test_mod_str_formatting(self):
        # [FUNCTION DOC] String percent formatting with an int, a string and a tuple of two ints.
        self.assertEqual("%d" % 5, "5")  # noqa: UP031
        self.assertEqual("%s world" % "hello", "hello world")  # noqa: UP031
        self.assertEqual("%d-%d" % (1, 2), "1-2")  # noqa: UP031

    @make_dynamo_test
    def test_mod_bytes_formatting(self):
        # [FUNCTION DOC] Bytes percent formatting with an int.
        self.assertEqual(b"%d" % 5, b"5")

    # --- Errors ---

    @make_dynamo_test
    def test_mod_by_zero_raises(self):
        # [FUNCTION DOC] Integer and float remainder by zero both raise ZeroDivisionError.
        with self.assertRaises(ZeroDivisionError):
            1 % 0
        with self.assertRaises(ZeroDivisionError):
            1.0 % 0.0

    @make_dynamo_test
    def test_mod_int_and_str_raises(self):
        # [FUNCTION DOC] Int remainder by a string raises TypeError.
        with self.assertRaises(TypeError):
            1 % "a"

    # --- Inplace %= ---

    @make_dynamo_test
    def test_inplace_mod_integers(self):
        # [FUNCTION DOC] In-place remainder on an int variable.
        x = 17
        x %= 5
        self.assertEqual(x, 2)

    @make_dynamo_test
    def test_inplace_mod_str(self):
        # [FUNCTION DOC] In-place remainder on a format string variable substitutes the integer.
        x = "%d"
        x %= 7
        self.assertEqual(x, "7")

    # --- User-defined __mod__ ---

    @make_dynamo_test
    def test_user_defined_mod_basic(self):
        # [FUNCTION DOC] Local class with a forward remainder method on another instance; checks
        # instance remainder instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward remainder and equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __mod__(self, other):
                # [FUNCTION DOC] HELPER: Forward remainder: returns a new instance with own value
                # modulo the other instance's value.
                return type(self)(self.value % other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) % C(3), C(1))

    @make_dynamo_test
    def test_user_defined_mod_with_integer(self):
        # [FUNCTION DOC] Local class with a forward remainder method accepting a raw int; checks
        # instance remainder int.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward remainder and equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __mod__(self, other):
                # [FUNCTION DOC] HELPER: Forward remainder: new instance with own value modulo the
                # raw operand.
                return type(self)(self.value % other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) % 3, C(1))

    @make_dynamo_test
    def test_reversed_mod_with_integer(self):
        # [FUNCTION DOC] Local class with only a reflected remainder; checks int remainder
        # instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with reflected remainder and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rmod__(self, other):
                # [FUNCTION DOC] HELPER: Reflected remainder: new instance with the raw operand
                # modulo own value.
                return type(self)(other % self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(7 % C(3), C(1))

    @make_dynamo_test
    def test_inplace_user_defined_mod(self):
        # [FUNCTION DOC] Local class with an in-place remainder that mutates and returns self;
        # checks the in-place operator.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with in-place remainder and equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __imod__(self, other):
                # [FUNCTION DOC] HELPER: In-place remainder: reduces own value modulo the other
                # instance's value and returns self.
                self.value %= other.value
                return self

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        x = C(17)
        x %= C(5)
        self.assertEqual(x, C(2))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Int subclass overriding forward and reflected remainder to return markers;
        # the override is used on both sides.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose remainder methods return marker strings.
            def __mod__(self, other):
                # [FUNCTION DOC] HELPER: Forward remainder: returns the forward marker string.
                return "IntSub.__mod__"

            def __rmod__(self, other):
                # [FUNCTION DOC] HELPER: Reflected remainder: returns the reflected marker string.
                return "IntSub.__rmod__"

        self.assertEqual(IntSub(7) % 3, "IntSub.__mod__")
        self.assertEqual(7 % IntSub(3), "IntSub.__rmod__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline plain class with forward and reflected remainder markers used
        # with an int on either side.
        class Base:
            # [CLASS DOC] HELPER: Local class whose remainder methods return marker strings.
            def __mod__(self, other):
                # [FUNCTION DOC] HELPER: Forward remainder: returns the forward marker string.
                return "Base.__mod__"

            def __rmod__(self, other):
                # [FUNCTION DOC] HELPER: Reflected remainder: returns the reflected marker string.
                return "Base.__rmod__"

        self.assertEqual(Base() % 1, "Base.__mod__")
        self.assertEqual(1 % Base(), "Base.__rmod__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_mod_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] A class whose forward and reflected remainder both return NotImplemented
        # raises TypeError when applied to itself.
        class C:
            # [CLASS DOC] HELPER: Local class with both remainder methods declining.
            def __mod__(self, other):
                # [FUNCTION DOC] HELPER: Forward remainder: always NotImplemented.
                return NotImplemented

            def __rmod__(self, other):
                # [FUNCTION DOC] HELPER: Reflected remainder: always NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a % a

    @make_dynamo_test
    def test_mod_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Left forward remainder declines, so the right operand's reflected
        # remainder marker is returned.
        class A:
            # [CLASS DOC] HELPER: Local class whose forward remainder declines.
            def __mod__(self, other):
                # [FUNCTION DOC] HELPER: Forward remainder: always NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Local class defining only a reflected remainder returning a
            # marker.
            def __rmod__(self, other):
                # [FUNCTION DOC] HELPER: Reflected remainder: returns the marker string.
                return "B.__rmod__ called"

        result = A() % B()
        self.assertEqual(result, "B.__rmod__ called")

    # --- SymNode remainder ---

    def test_mod_symnode_and_int(self):
        # [FUNCTION DOC] Compiles a nested function with fullgraph and dynamic first dimension,
        # creating zeros with size equal to the symbolic size modulo three; compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor with the length given
            # by size modulo three.
            s = x.shape[0]
            return x.new_zeros(s % 3)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_mod_int_and_symnode(self):
        # [FUNCTION DOC] Same but with an int modulo the symbolic size on the left.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor with length seventeen
            # modulo the size.
            s = x.shape[0]
            return x.new_zeros(17 % s)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_mod_tensor(self):
        # [FUNCTION DOC] Compiles tensor remainder tensor with fullgraph over positive inputs and
        # compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the elementwise remainder of its two
            # tensor arguments.
            return x % y

        x = torch.randn(4).abs() + 1
        y = torch.randn(4).abs() + 1
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
