# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the floor division operator (nb_floor_divide slot) and its inplace
# form: ints, floats, bools, errors, user-defined __floordiv__/__rfloordiv__/__ifloordiv__,
# subclass priority, NotImplemented fallback, symbolic sizes and tensors. Test class is patched to
# trace unittest and class building. No import-time side effects.

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbFloorDivide(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for floor division in Dynamo; patched with enable_trace_unittest and
    # enable_trace_load_build_class so asserts and local classes can be traced in make_dynamo_test
    # bodies.
    # --- Integer floordiv ---
    @make_dynamo_test
    def test_floordiv_integers(self):
        # [FUNCTION DOC] Checks floor division over positive ints, including zero dividend.
        self.assertEqual(7 // 2, 3)
        self.assertEqual(8 // 2, 4)
        self.assertEqual(1 // 1, 1)
        self.assertEqual(0 // 5, 0)

    @make_dynamo_test
    def test_floordiv_negative(self):
        # [FUNCTION DOC] Checks floor division with negative operands rounds toward negative
        # infinity.
        self.assertEqual(-7 // 2, -4)
        self.assertEqual(7 // -2, -4)
        self.assertEqual(-7 // -2, 3)

    @make_dynamo_test
    def test_floordiv_chained(self):
        # [FUNCTION DOC] Checks chained floor division left to right.
        self.assertEqual(100 // 5 // 2, 10)

    # --- Floats ---

    @make_dynamo_test
    def test_floordiv_floats(self):
        # [FUNCTION DOC] Checks floor division over floats gives float floor results.
        self.assertEqual(7.0 // 2.0, 3.0)
        self.assertEqual(7.5 // 2.0, 3.0)

    @make_dynamo_test
    def test_floordiv_mixed_int_float(self):
        # [FUNCTION DOC] Checks floor division mixing int and float in either order gives float
        # results.
        self.assertEqual(7 // 2.0, 3.0)
        self.assertEqual(7.0 // 2, 3.0)

    # --- Booleans ---

    @make_dynamo_test
    def test_floordiv_bools(self):
        # [FUNCTION DOC] Checks floor division over bools gives int results.
        self.assertEqual(True // True, 1)
        self.assertEqual(False // True, 0)

    @make_dynamo_test
    def test_floordiv_int_and_bool(self):
        # [FUNCTION DOC] Checks floor division of an int by True or True by one.
        self.assertEqual(5 // True, 5)
        self.assertEqual(True // 1, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_floordiv_by_zero_raises(self):
        # [FUNCTION DOC] Checks floor division by int zero and by float zero raises
        # ZeroDivisionError.
        with self.assertRaises(ZeroDivisionError):
            1 // 0
        with self.assertRaises(ZeroDivisionError):
            1.0 // 0.0

    @make_dynamo_test
    def test_floordiv_str_raises(self):
        # [FUNCTION DOC] Checks floor division between int and str in either order raises
        # TypeError.
        with self.assertRaises(TypeError):
            1 // "a"
        with self.assertRaises(TypeError):
            "a" // 1

    # --- Inplace //= ---

    @make_dynamo_test
    def test_inplace_floordiv_integers(self):
        # [FUNCTION DOC] Checks the inplace floor-divide-assign on a local int.
        x = 17
        x //= 4
        self.assertEqual(x, 4)

    @make_dynamo_test
    def test_inplace_floordiv_chained(self):
        # [FUNCTION DOC] Checks two consecutive inplace floor-divide-assign operations.
        x = 100
        x //= 5
        x //= 2
        self.assertEqual(x, 10)

    # --- User-defined __floordiv__ ---

    @make_dynamo_test
    def test_user_defined_floordiv_basic(self):
        # [FUNCTION DOC] Checks a user-defined class whose __floordiv__ takes another instance and
        # returns a new instance; compared by value.
        class C:
            # [CLASS DOC] HELPER: Test class holding a value; defines __floordiv__ and value-based
            # __eq__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance with the floor division of both
                # values.
                return type(self)(self.value // other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other has the same exact type and equal
                # value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) // C(2), C(3))

    @make_dynamo_test
    def test_user_defined_floordiv_with_integer(self):
        # [FUNCTION DOC] Checks user-defined __floordiv__ where the right operand is a plain int.
        class C:
            # [CLASS DOC] HELPER: Test class holding a value; __floordiv__ takes a raw number.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance with the stored value floor-
                # divided by the raw operand.
                return type(self)(self.value // other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other has the same exact type and equal
                # value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) // 2, C(3))

    @make_dynamo_test
    def test_reversed_floordiv_with_integer(self):
        # [FUNCTION DOC] Checks an int on the left and an object with only __rfloordiv__ on the
        # right dispatches to the reflected method.
        class C:
            # [CLASS DOC] HELPER: Test class holding a value; defines only reflected floor
            # division.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: Reflected: returns a new instance with the left operand
                # floor-divided by the stored value.
                return type(self)(other // self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other has the same exact type and equal
                # value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(7 // C(2), C(3))

    @make_dynamo_test
    def test_inplace_user_defined_floordiv(self):
        # [FUNCTION DOC] Checks the inplace floor-divide-assign operator calls a user-defined
        # __ifloordiv__ that mutates and returns the same object.
        class C:
            # [CLASS DOC] HELPER: Test class holding a mutable value; defines __ifloordiv__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __ifloordiv__(self, other):
                # [FUNCTION DOC] HELPER: Inplace: floor-divides its own value by the other value
                # and returns itself.
                self.value //= other.value
                return self

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other has the same exact type and equal
                # value.
                return type(other) is type(self) and self.value == other.value

        x = C(17)
        x //= C(4)
        self.assertEqual(x, C(4))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Checks an int subclass overriding the forward and reflected floor
        # division takes priority on both sides, returning marker strings.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose floor-division methods return marker strings.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "IntSub.__floordiv__"

            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "IntSub.__rfloordiv__"

        self.assertEqual(IntSub(7) // 2, "IntSub.__floordiv__")
        self.assertEqual(7 // IntSub(2), "IntSub.__rfloordiv__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline check with a plain class: forward method on the left, reflected
        # when the other operand is an int.
        class Base:
            # [CLASS DOC] HELPER: Plain class whose floor-division methods return marker strings.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "Base.__floordiv__"

            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "Base.__rfloordiv__"

        self.assertEqual(Base() // 1, "Base.__floordiv__")
        self.assertEqual(1 // Base(), "Base.__rfloordiv__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_floordiv_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] Checks that when both forward and reflected methods return NotImplemented
        # for the same object, floor division raises TypeError.
        class C:
            # [CLASS DOC] HELPER: Class whose forward and reflected floor division both decline.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a // a

    @make_dynamo_test
    def test_floordiv_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Checks fallback to the right operand reflected method when the left
        # returns NotImplemented.
        class A:
            # [CLASS DOC] HELPER: Class whose __floordiv__ returns NotImplemented.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Class with a reflected floor division returning a marker string.
            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string showing the reflected method
                # ran.
                return "B.__rfloordiv__ called"

        result = A() // B()
        self.assertEqual(result, "B.__rfloordiv__ called")

    # --- SymNode floordiv ---

    def test_floordiv_symnode_and_int(self):
        # [FUNCTION DOC] Compiles floor division of a dynamic (symbolic) size by two and uses it
        # as the size of a new zero tensor; input dimension zero is marked dynamic; compared with
        # eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the first dimension size, halves it by floor division
            # and returns a zero tensor of that size.
            s = x.shape[0]
            return x.new_zeros(s // 2)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_floordiv_int_and_symnode(self):
        # [FUNCTION DOC] Compiles floor division of sixteen by a dynamic size, used as a new_zeros
        # size; compared with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Floor-divides sixteen by the first dimension size and returns
            # a zero tensor of that size.
            s = x.shape[0]
            return x.new_zeros(16 // s)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_floordiv_tensor(self):
        # [FUNCTION DOC] Compiles floor division of two float tensors (divisor is the absolute
        # value plus one, at least one) and compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the first argument floor-divided by the second.
            return x // y

        x = torch.randn(4)
        y = torch.randn(4).abs() + 1
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
