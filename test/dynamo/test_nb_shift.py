# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the shift operators (nb_lshift and nb_rshift slots) run through
# make_dynamo_test. Two TestCase classes, TestNbLshift and TestNbRshift, cover ints, bools, error
# cases, in-place shifts, user-defined forward and reflected shift methods, int subclass priority,
# NotImplemented fallbacks, and compiled tensor and symbolic-size shifts. No import-time side
# effects beyond decorator application.

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbLshift(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for left shift under Dynamo. Class decorators enable unittest tracing
    # and class construction tracing for all tests, so classes defined in test bodies are traced.
    # --- Integer lshift ---
    @make_dynamo_test
    def test_lshift_integers(self):
        # [FUNCTION DOC] Left shift of ints by several counts gives the expected powers and
        # products.
        self.assertEqual(1 << 0, 1)
        self.assertEqual(1 << 1, 2)
        self.assertEqual(1 << 4, 16)
        self.assertEqual(3 << 2, 12)

    @make_dynamo_test
    def test_lshift_zero(self):
        # [FUNCTION DOC] Shifting zero, and shifting by zero, leave the value unchanged.
        self.assertEqual(0 << 5, 0)
        self.assertEqual(5 << 0, 5)

    @make_dynamo_test
    def test_lshift_large(self):
        # [FUNCTION DOC] Large shifts (31 and 62 bits) equal the matching powers of two.
        self.assertEqual(1 << 31, 2**31)
        self.assertEqual(1 << 62, 2**62)

    @make_dynamo_test
    def test_lshift_negative_value(self):
        # [FUNCTION DOC] Left shift of negative values keeps the sign and multiplies by powers of
        # two.
        self.assertEqual(-1 << 1, -2)
        self.assertEqual(-3 << 2, -12)

    @make_dynamo_test
    def test_lshift_chained(self):
        # [FUNCTION DOC] Chained left shifts evaluate left to right.
        self.assertEqual(1 << 2 << 3, 32)

    # --- Booleans ---

    @make_dynamo_test
    def test_lshift_bools(self):
        # [FUNCTION DOC] Left shift with bool operands behaves like int shifts.
        self.assertEqual(True << True, 2)
        self.assertEqual(True << False, 1)
        self.assertEqual(False << True, 0)

    @make_dynamo_test
    def test_lshift_int_and_bool(self):
        # [FUNCTION DOC] Left shift mixing an int and a bool.
        self.assertEqual(5 << True, 10)
        self.assertEqual(True << 3, 8)

    # --- Errors ---

    @make_dynamo_test
    def test_lshift_negative_count_raises(self):
        # [FUNCTION DOC] A negative shift count raises ValueError.
        with self.assertRaises(ValueError):
            1 << -1

    @make_dynamo_test
    def test_lshift_float_raises(self):
        # [FUNCTION DOC] Left shift with a float on either side raises TypeError.
        with self.assertRaises(TypeError):
            1 << 1.0
        with self.assertRaises(TypeError):
            1.0 << 1

    # --- Inplace <<= ---

    @make_dynamo_test
    def test_inplace_lshift_integers(self):
        # [FUNCTION DOC] In-place left shift on an int variable.
        x = 1
        x <<= 4
        self.assertEqual(x, 16)

    @make_dynamo_test
    def test_inplace_lshift_chained(self):
        # [FUNCTION DOC] Two successive in-place left shifts on an int variable.
        x = 1
        x <<= 2
        x <<= 3
        self.assertEqual(x, 32)

    # --- User-defined __lshift__ ---

    @make_dynamo_test
    def test_user_defined_lshift_basic(self):
        # [FUNCTION DOC] Local class with forward left shift on another instance; checks instance
        # shift instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward left shift and equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: returns a new instance with own value
                # shifted by the other instance's value.
                return type(self)(self.value << other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(1) << C(4), C(16))

    @make_dynamo_test
    def test_user_defined_lshift_with_integer(self):
        # [FUNCTION DOC] Local class with forward left shift taking a raw int; checks instance
        # shift int.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward left shift and equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: new instance with own value shifted by
                # the raw count.
                return type(self)(self.value << other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(1) << 4, C(16))

    @make_dynamo_test
    def test_reversed_lshift_with_integer(self):
        # [FUNCTION DOC] Local class with only a reflected left shift; checks int shift instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with reflected left shift and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rlshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected left shift: new instance with the raw operand
                # shifted by own value.
                return type(self)(other << self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(1 << C(2), C(4))

    @make_dynamo_test
    def test_user_defined_lshift_chained(self):
        # [FUNCTION DOC] Chained left shift of three user-defined instances evaluates left to
        # right.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward left shift and equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: new instance with own value shifted by
                # the other instance's value.
                return type(self)(self.value << other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(1) << C(2) << C(1), C(8))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Int subclass overriding forward and reflected left shift to return
        # markers; override used on both sides.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose shift methods return marker strings.
            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: returns the forward marker string.
                return "IntSub.__lshift__"

            def __rlshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected left shift: returns the reflected marker string.
                return "IntSub.__rlshift__"

        self.assertEqual(IntSub(1) << 1, "IntSub.__lshift__")
        self.assertEqual(1 << IntSub(1), "IntSub.__rlshift__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline plain class with forward and reflected left shift markers used
        # with an int on either side.
        class Base:
            # [CLASS DOC] HELPER: Local class whose shift methods return marker strings.
            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: returns the forward marker string.
                return "Base.__lshift__"

            def __rlshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected left shift: returns the reflected marker string.
                return "Base.__rlshift__"

        self.assertEqual(Base() << 1, "Base.__lshift__")
        self.assertEqual(1 << Base(), "Base.__rlshift__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_lshift_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] A class whose forward and reflected left shift return NotImplemented
        # raises TypeError when shifted by itself.
        class C:
            # [CLASS DOC] HELPER: Local class with both shift methods declining.
            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: always NotImplemented.
                return NotImplemented

            def __rlshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected left shift: always NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a << a

    @make_dynamo_test
    def test_lshift_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Left forward shift declines, so the right operand's reflected shift marker
        # is returned.
        class A:
            # [CLASS DOC] HELPER: Local class whose forward left shift declines.
            def __lshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward left shift: always NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Local class defining only a reflected left shift returning a
            # marker.
            def __rlshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected left shift: returns the marker string.
                return "B.__rlshift__ called"

        result = A() << B()
        self.assertEqual(result, "B.__rlshift__ called")

    # --- SymNode lshift ---

    def test_lshift_tensor_and_int(self):
        # [FUNCTION DOC] Compiles tensor shifted left by an int with fullgraph and compares with
        # eager on an int64 tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning its tensor argument shifted left by
            # two.
            return x << 2

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_lshift_int_and_tensor(self):
        # [FUNCTION DOC] Compiles int shifted left by a tensor with fullgraph and compares with
        # eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning two shifted left by its tensor
            # argument.
            return 2 << x

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_lshift_tensor_and_tensor(self):
        # [FUNCTION DOC] Compiles tensor shifted left by a tensor with fullgraph and compares with
        # eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the first tensor shifted left by the
            # second.
            return x << y

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        y = torch.tensor([1, 0, 2], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_lshift_symnode_and_int(self):
        # [FUNCTION DOC] Compiles a symbolic size shifted left by one, used as the length of a new
        # zeros tensor, with a dynamic first dimension.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor whose length is the
            # size shifted left by one.
            s = x.shape[0]
            return x.new_zeros(s << 1)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_lshift_int_and_symnode(self):
        # [FUNCTION DOC] Compiles one shifted left by the symbolic size, used as a tensor length,
        # with a dynamic first dimension.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor whose length is one
            # shifted left by the size.
            s = x.shape[0]
            return x.new_zeros(1 << s)

        x = torch.randn(3)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbRshift(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for right shift under Dynamo. Class decorators enable unittest tracing
    # and class construction tracing for all tests.
    # --- Integer rshift ---
    @make_dynamo_test
    def test_rshift_integers(self):
        # [FUNCTION DOC] Right shift of ints by several counts gives the expected quotients.
        self.assertEqual(16 >> 0, 16)
        self.assertEqual(16 >> 1, 8)
        self.assertEqual(16 >> 4, 1)
        self.assertEqual(12 >> 2, 3)

    @make_dynamo_test
    def test_rshift_zero(self):
        # [FUNCTION DOC] Shifting zero, and shifting by zero, leave the value unchanged.
        self.assertEqual(0 >> 5, 0)
        self.assertEqual(5 >> 0, 5)

    @make_dynamo_test
    def test_rshift_large(self):
        # [FUNCTION DOC] Right shift of a large power of two.
        self.assertEqual((2**31) >> 1, 2**30)

    @make_dynamo_test
    def test_rshift_to_zero(self):
        # [FUNCTION DOC] Shifting a small value by more than its bit length gives zero.
        self.assertEqual(1 >> 10, 0)

    @make_dynamo_test
    def test_rshift_negative_value(self):
        # [FUNCTION DOC] Right shift of negative values is an arithmetic shift rounding toward
        # negative infinity.
        # arithmetic shift on negatives
        self.assertEqual(-1 >> 1, -1)
        self.assertEqual(-8 >> 1, -4)

    @make_dynamo_test
    def test_rshift_chained(self):
        # [FUNCTION DOC] Chained right shifts evaluate left to right.
        self.assertEqual(32 >> 1 >> 2, 4)

    # --- Booleans ---

    @make_dynamo_test
    def test_rshift_bools(self):
        # [FUNCTION DOC] Right shift with bool operands behaves like int shifts.
        self.assertEqual(True >> True, 0)
        self.assertEqual(True >> False, 1)
        self.assertEqual(False >> True, 0)

    @make_dynamo_test
    def test_rshift_int_and_bool(self):
        # [FUNCTION DOC] Right shift mixing an int and a bool.
        self.assertEqual(10 >> True, 5)
        self.assertEqual(True >> 0, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_rshift_negative_count_raises(self):
        # [FUNCTION DOC] A negative shift count raises ValueError.
        with self.assertRaises(ValueError):
            1 >> -1

    @make_dynamo_test
    def test_rshift_float_raises(self):
        # [FUNCTION DOC] Right shift with a float on either side raises TypeError.
        with self.assertRaises(TypeError):
            1 >> 1.0
        with self.assertRaises(TypeError):
            1.0 >> 1

    # --- Inplace >>= ---

    @make_dynamo_test
    def test_inplace_rshift_integers(self):
        # [FUNCTION DOC] In-place right shift on an int variable.
        x = 16
        x >>= 2
        self.assertEqual(x, 4)

    @make_dynamo_test
    def test_inplace_rshift_chained(self):
        # [FUNCTION DOC] Two successive in-place right shifts on an int variable.
        x = 32
        x >>= 1
        x >>= 2
        self.assertEqual(x, 4)

    # --- User-defined __rshift__ ---

    @make_dynamo_test
    def test_user_defined_rshift_basic(self):
        # [FUNCTION DOC] Local class with forward right shift on another instance; checks instance
        # shift instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward right shift and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: returns a new instance with own value
                # shifted right by the other instance's value.
                return type(self)(self.value >> other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(16) >> C(2), C(4))

    @make_dynamo_test
    def test_user_defined_rshift_with_integer(self):
        # [FUNCTION DOC] Local class with forward right shift taking a raw int; checks instance
        # shift int.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward right shift and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: new instance with own value shifted
                # right by the raw count.
                return type(self)(self.value >> other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(16) >> 2, C(4))

    @make_dynamo_test
    def test_reversed_rshift_with_integer(self):
        # [FUNCTION DOC] Local class with only a reflected right shift; checks int shift instance.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with reflected right shift and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rrshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected right shift: new instance with the raw operand
                # shifted right by own value.
                return type(self)(other >> self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(16 >> C(1), C(8))

    @make_dynamo_test
    def test_user_defined_rshift_chained(self):
        # [FUNCTION DOC] Chained right shift of three user-defined instances evaluates left to
        # right.
        class C:
            # [CLASS DOC] HELPER: Local class holding a value with forward right shift and
            # equality.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value on the instance attribute value.
                self.value = v

            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: new instance with own value shifted
                # right by the other instance's value.
                return type(self)(self.value >> other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: same exact type and equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(32) >> C(1) >> C(2), C(4))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Int subclass overriding forward and reflected right shift to return
        # markers; override used on both sides.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose shift methods return marker strings.
            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: returns the forward marker string.
                return "IntSub.__rshift__"

            def __rrshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected right shift: returns the reflected marker
                # string.
                return "IntSub.__rrshift__"

        self.assertEqual(IntSub(1) >> 1, "IntSub.__rshift__")
        self.assertEqual(1 >> IntSub(1), "IntSub.__rrshift__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline plain class with forward and reflected right shift markers used
        # with an int on either side.
        class Base:
            # [CLASS DOC] HELPER: Local class whose shift methods return marker strings.
            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: returns the forward marker string.
                return "Base.__rshift__"

            def __rrshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected right shift: returns the reflected marker
                # string.
                return "Base.__rrshift__"

        self.assertEqual(Base() >> 1, "Base.__rshift__")
        self.assertEqual(1 >> Base(), "Base.__rrshift__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_rshift_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] A class whose forward and reflected right shift return NotImplemented
        # raises TypeError when shifted by itself.
        class C:
            # [CLASS DOC] HELPER: Local class with both shift methods declining.
            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: always NotImplemented.
                return NotImplemented

            def __rrshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected right shift: always NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a >> a

    @make_dynamo_test
    def test_rshift_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Right forward shift declines, so the right operand's reflected shift
        # marker is returned.
        class A:
            # [CLASS DOC] HELPER: Local class whose forward right shift declines.
            def __rshift__(self, other):
                # [FUNCTION DOC] HELPER: Forward right shift: always NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Local class defining only a reflected right shift returning a
            # marker.
            def __rrshift__(self, other):
                # [FUNCTION DOC] HELPER: Reflected right shift: returns the marker string.
                return "B.__rrshift__ called"

        result = A() >> B()
        self.assertEqual(result, "B.__rrshift__ called")

    # --- SymNode rshift ---

    def test_rshift_tensor_and_int(self):
        # [FUNCTION DOC] Compiles tensor shifted right by an int with fullgraph and compares with
        # eager on an int64 tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning its tensor argument shifted right by
            # one.
            return x >> 1

        x = torch.tensor([16, 8, 4], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rshift_int_and_tensor(self):
        # [FUNCTION DOC] Compiles int shifted right by a tensor with fullgraph and compares with
        # eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning thirty-two shifted right by its
            # tensor argument.
            return 32 >> x

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rshift_tensor_and_tensor(self):
        # [FUNCTION DOC] Compiles tensor shifted right by a tensor with fullgraph and compares with
        # eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the first tensor shifted right by
            # the second.
            return x >> y

        x = torch.tensor([16, 8, 4], dtype=torch.int64)
        y = torch.tensor([1, 2, 0], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_rshift_symnode_and_int(self):
        # [FUNCTION DOC] Compiles a symbolic size shifted right by one, used as the length of a new
        # zeros tensor, with a dynamic first dimension.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor whose length is the
            # size shifted right by one.
            s = x.shape[0]
            return x.new_zeros(s >> 1)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rshift_int_and_symnode(self):
        # [FUNCTION DOC] Compiles thirty-two shifted right by the symbolic size, used as a tensor
        # length, with a dynamic first dimension.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a zeros tensor whose length is
            # thirty-two shifted right by the size.
            s = x.shape[0]
            return x.new_zeros(32 >> s)

        x = torch.randn(3)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
