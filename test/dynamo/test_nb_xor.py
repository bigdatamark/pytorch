# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the bitwise xor binary and in-place operators under dynamo. Covers
# integers, booleans, sets, frozensets, dict views, OrderedSet, user-defined dunder methods,
# reflected dispatch, subclass priority and NotImplemented fallback (traced via make_dynamo_test),
# plus compiled tensor and symbolic-int xor. Calls instantiate_parametrized_tests on the class at
# import time.

"""Tests for nb_xor (^) operator in PyTorch Dynamo."""

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
)
from torch.utils._ordered_set import OrderedSet


class TestNbXor(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for xor semantics. setUp enables two dynamo config flags (tracing of
    # unittest and of class building) and tearDown restores them; stores the previous values on
    # the instance.
    def setUp(self):
        # [FUNCTION DOC] Saves the current values of the trace_unittest and trace_load_build_class
        # dynamo configs on the instance, then turns both on.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        # [FUNCTION DOC] Restores the two dynamo config flags saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # --- Integer xor ---

    @make_dynamo_test
    def test_xor_integers(self):
        # [FUNCTION DOC] Checks xor on several integer pairs gives the expected bit patterns.
        self.assertEqual(0b1100 ^ 0b1010, 0b0110)
        self.assertEqual(0xFF ^ 0x0F, 0xF0)
        self.assertEqual(7 ^ 3, 4)

    @make_dynamo_test
    def test_xor_zero(self):
        # [FUNCTION DOC] Checks xor with zero is identity and a value xor itself is zero.
        self.assertEqual(0 ^ 0xFFFF, 0xFFFF)
        self.assertEqual(0xFFFF ^ 0, 0xFFFF)
        self.assertEqual(0xABCD ^ 0xABCD, 0)

    @make_dynamo_test
    def test_xor_negative_integers(self):
        # [FUNCTION DOC] Checks xor with negative integers follows two-complement semantics.
        self.assertEqual(-1 ^ 0xFF, ~0xFF)
        self.assertEqual(-2 ^ 3, -3)

    @make_dynamo_test
    def test_xor_chained(self):
        # [FUNCTION DOC] Checks chaining three xor operations on integers.
        self.assertEqual(0b1110 ^ 0b1100 ^ 0b1000, 0b1010)

    # --- Booleans ---

    @make_dynamo_test
    def test_xor_bools(self):
        # [FUNCTION DOC] Checks the full truth table of xor on booleans.
        self.assertEqual(True ^ True, False)
        self.assertEqual(True ^ False, True)
        self.assertEqual(False ^ True, True)
        self.assertEqual(False ^ False, False)

    @make_dynamo_test
    def test_xor_int_and_bool(self):
        # [FUNCTION DOC] Checks xor mixing int and bool operands gives integer results.
        self.assertEqual(5 ^ True, 4)
        self.assertEqual(True ^ 3, 2)

    # --- Errors ---

    @make_dynamo_test
    def test_xor_float_raises(self):
        # [FUNCTION DOC] Checks xor between int and float in either order raises TypeError.
        with self.assertRaises(TypeError):
            1 ^ 1.0
        with self.assertRaises(TypeError):
            1.0 ^ 1

    # --- Inplace ^= ---

    @make_dynamo_test
    def test_inplace_xor_integers(self):
        # [FUNCTION DOC] Checks the in-place xor assignment on an integer.
        x = 0b1110
        x ^= 0b1100
        self.assertEqual(x, 0b0010)

    @make_dynamo_test
    def test_inplace_xor_chained(self):
        # [FUNCTION DOC] Checks two successive in-place xor assignments on an integer.
        x = 0b1111
        x ^= 0b1110
        x ^= 0b1100
        self.assertEqual(x, 0b1101)

    # --- Sets ---

    @make_dynamo_test
    def test_set_xor_set(self):
        # [FUNCTION DOC] Checks symmetric difference of two sets via xor.
        self.assertEqual({1, 2, 3} ^ {2, 3, 4}, {1, 4})

    @make_dynamo_test
    def test_set_ixor_set(self):
        # [FUNCTION DOC] Checks in-place xor of a set with another set mutates it to the
        # symmetric difference.
        s = {1, 2, 3}
        s ^= {2, 3, 4}
        self.assertEqual(s, {1, 4})

    @make_dynamo_test
    def test_frozenset_xor(self):
        # [FUNCTION DOC] Checks xor of two frozensets gives their symmetric difference.
        self.assertEqual(frozenset({1, 2, 3}) ^ frozenset({2, 3, 4}), frozenset({1, 4}))

    # --- Dict views ---

    @make_dynamo_test
    def test_dict_keys_xor_set(self):
        # [FUNCTION DOC] Checks dict keys view xor a set gives the symmetric difference as a set.
        self.assertEqual({1: 0, 2: 0, 3: 0}.keys() ^ {2, 3, 4}, {1, 4})

    @make_dynamo_test
    def test_set_xor_dict_keys(self):
        # [FUNCTION DOC] Checks a set xor a dict keys view (reflected path) gives the
        # symmetric difference.
        self.assertEqual({2, 3, 4} ^ {1: 0, 2: 0, 3: 0}.keys(), {1, 4})

    @make_dynamo_test
    def test_dict_items_xor_set(self):
        # [FUNCTION DOC] Checks dict items view xor a set of pairs yields the symmetric difference
        # of the pairs.
        d = {1: "a", 2: "b", 3: "c"}
        self.assertEqual(
            d.items() ^ {(2, "b"), (3, "x")}, {(1, "a"), (3, "c"), (3, "x")}
        )

    # --- OrderedSet ---

    @make_dynamo_test
    def test_ordered_set_xor(self):
        # [FUNCTION DOC] Checks xor of two OrderedSet objects gives their symmetric difference.
        self.assertEqual(
            OrderedSet([1, 2, 3]) ^ OrderedSet([2, 3, 4]), OrderedSet([1, 4])
        )

    # --- User-defined __xor__ ---

    @make_dynamo_test
    def test_user_defined_xor_basic(self):
        # [FUNCTION DOC] Defines a class with a xor method combining the value attributes of two
        # instances and checks the result equals the expected instance.
        class C:
            # [CLASS DOC] HELPER: Holds a value; xor builds a new instance from the xor of
            # both values.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument in the value attribute.
                self.value = v

            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance of the same type holding the xor
                # of the two values.
                return type(self)(self.value ^ other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: true only when the other object has exactly the
                # same type and an equal value attribute.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) ^ C(0b1100), C(0b0010))

    @make_dynamo_test
    def test_user_defined_xor_with_integer(self):
        # [FUNCTION DOC] Like the basic user-defined test, but the right operand is a plain
        # integer rather than an instance.
        class C:
            # [CLASS DOC] HELPER: Holds a value; xor builds a new instance from the value xor a
            # plain integer.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument in the value attribute.
                self.value = v

            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance of the same type holding the value
                # xor the raw other operand.
                return type(self)(self.value ^ other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: true only when the other object has exactly the
                # same type and an equal value attribute.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) ^ 0b1100, C(0b0010))

    @make_dynamo_test
    def test_reversed_xor_with_integer(self):
        # [FUNCTION DOC] Checks the reflected xor method is used for an integer left operand and
        # an instance right operand.
        class C:
            # [CLASS DOC] HELPER: Holds a value; only the reflected xor is defined.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument in the value attribute.
                self.value = v

            def __rxor__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance holding the other operand xor the
                # stored value.
                return type(self)(other ^ self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: true only when the other object has exactly the
                # same type and an equal value attribute.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(0b1110 ^ C(0b1100), C(0b0010))

    @make_dynamo_test
    def test_user_defined_xor_chained(self):
        # [FUNCTION DOC] Checks chaining two xor operations on instances of a user-defined class.
        class C:
            # [CLASS DOC] HELPER: Holds a value; xor builds a new instance from both values.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument in the value attribute.
                self.value = v

            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance of the same type holding the xor
                # of the two values.
                return type(self)(self.value ^ other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: true only when the other object has exactly the
                # same type and an equal value attribute.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) ^ C(0b1100) ^ C(0b1000), C(0b1010))

    @make_dynamo_test
    def test_user_defined_ixor(self):
        # [FUNCTION DOC] Checks the in-place xor assignment dispatches to a user-defined in-
        # place method.
        class C:
            # [CLASS DOC] HELPER: Holds a value; defines only the in-place xor.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument in the value attribute.
                self.value = v

            def __ixor__(self, other):
                # [FUNCTION DOC] HELPER: Mutates the stored value by xor with the other value and
                # returns self.
                self.value ^= other.value
                return self

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality: true only when the other object has exactly the
                # same type and an equal value attribute.
                return type(other) is type(self) and self.value == other.value

        a = C(0b1110)
        a ^= C(0b1100)
        self.assertEqual(a, C(0b0010))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Checks an int subclass overriding xor and reflected xor wins in both
        # operand orders, returning marker strings.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose xor methods return marker strings
            # instead of numbers.
            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "IntSub.__xor__"

            def __rxor__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "IntSub.__rxor__"

        self.assertEqual(IntSub(1) ^ 1, "IntSub.__xor__")
        self.assertEqual(1 ^ IntSub(1), "IntSub.__rxor__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline: a plain object class with forward and reflected xor is
        # dispatched correctly on either side of an int operand.
        class Base:
            # [CLASS DOC] HELPER: Plain class whose xor methods return marker strings.
            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "Base.__xor__"

            def __rxor__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "Base.__rxor__"

        self.assertEqual(Base() ^ 1, "Base.__xor__")
        self.assertEqual(1 ^ Base(), "Base.__rxor__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_xor_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] Checks that when both forward and reflected xor return NotImplemented,
        # the operation raises TypeError.
        class C:
            # [CLASS DOC] HELPER: Both xor methods decline by returning NotImplemented.
            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

            def __rxor__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a ^ a

    @make_dynamo_test
    def test_xor_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Checks that if the left operand declines with NotImplemented, the right
        # operand reflected method is used.
        class A:
            # [CLASS DOC] HELPER: Defines only a forward xor that declines.
            def __xor__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Defines only a reflected xor that returns a marker string.
            def __rxor__(self, other):
                # [FUNCTION DOC] HELPER: Returns a marker string.
                return "B.__rxor__ called"

        result = A() ^ B()
        self.assertEqual(result, "B.__rxor__ called")

    # --- Tensor xor ---

    def test_xor_tensor_and_tensor(self):
        # [FUNCTION DOC] Compiles (eager, fullgraph) xor of two boolean tensors and compares
        # with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the xor of the two inputs.
            return x ^ y

        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_xor_tensor_and_tensor_int(self):
        # [FUNCTION DOC] Same as the boolean tensor test but with int64 tensors.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the xor of the two inputs.
            return x ^ y

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        y = torch.tensor([0b1100, 0b1010], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_xor_tensor_and_int(self):
        # [FUNCTION DOC] Compiles xor of an int64 tensor with an integer constant and compares
        # with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input xor the constant 7.
            return x ^ 7

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_xor_int_and_tensor(self):
        # [FUNCTION DOC] Compiles xor with the integer on the left and a tensor on the right.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the constant 7 xor the input.
            return 7 ^ x

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_xor_tensor_and_bool(self):
        # [FUNCTION DOC] Compiles xor of a boolean tensor with a boolean constant.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input xor True.
            return x ^ True

        x = torch.tensor([True, False, True, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_xor_bool_and_tensor(self):
        # [FUNCTION DOC] Compiles xor with a boolean constant on the left and a boolean tensor on
        # the right.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns True xor the input.
            return True ^ x

        x = torch.tensor([True, False, True, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_inplace_xor_tensor(self):
        # [FUNCTION DOC] Compiles an in-place xor on a tensor, feeding clones so eager and
        # compiled see the same input.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies in-place xor of the second input into the first and
            # returns the first.
            x ^= y
            return x

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        y = torch.tensor([0b1100, 0b1010], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x.clone(), y), fn(x.clone(), y))

    # --- SymNode xor ---

    def test_xor_symnode_and_int(self):
        # [FUNCTION DOC] Marks the input dimension dynamic and compiles a function xoring the
        # symbolic size with an integer to size a new zeros tensor; compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the first dimension size, xors it with 7 and returns a
            # zero tensor of that size.
            s = x.shape[0]
            return x.new_zeros(s ^ 7)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_xor_int_and_symnode(self):
        # [FUNCTION DOC] Same as the previous test with the integer on the left of the
        # symbolic size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the first dimension size, xors 7 with it and returns a
            # zero tensor of that size.
            s = x.shape[0]
            return x.new_zeros(7 ^ s)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


instantiate_parametrized_tests(TestNbXor)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
