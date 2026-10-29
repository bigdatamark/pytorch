# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the binary and operator (nb_and slot) and its inplace form. Covers
# ints, bools, sets, frozensets, dict views, OrderedSet, user-defined __and__/__rand__/__iand__,
# subclass priority, NotImplemented fallback, tensors and symbolic ints. Most cases use
# make_dynamo_test (eager run then compiled run with no graph break); tensor and symnode cases use
# torch.compile with the eager backend and fullgraph. No import-time side effects besides test
# registration.

"""Tests for nb_and (&) operator in PyTorch Dynamo."""

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
)
from torch.utils._ordered_set import OrderedSet


class TestNbAnd(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the and operator in Dynamo. Its setUp turns on tracing of unittest
    # and of class building (load_build_class) in the dynamo config and tearDown restores the
    # saved values, so tests may define classes and use assert methods inside traced code.
    def setUp(self):
        # [FUNCTION DOC] Saves the current values of the two dynamo config flags
        # (enable_trace_unittest, enable_trace_load_build_class) on the instance, then sets both
        # to True. Calls the parent setUp first.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and then restores the two dynamo config flags
        # saved by setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # --- Integer and ---

    @make_dynamo_test
    def test_and_integers(self):
        # [FUNCTION DOC] Checks integer and results for several binary literals, hex masks and
        # small decimals.
        self.assertEqual(0b1100 & 0b1010, 0b1000)
        self.assertEqual(0xFF & 0x0F, 0x0F)
        self.assertEqual(7 & 3, 3)

    @make_dynamo_test
    def test_and_zero(self):
        # [FUNCTION DOC] Checks that anding with zero on either side gives zero.
        self.assertEqual(0 & 0xFFFF, 0)
        self.assertEqual(0xFFFF & 0, 0)

    @make_dynamo_test
    def test_and_negative_integers(self):
        # [FUNCTION DOC] Checks and with negative integers (two's complement semantics): minus one
        # and 0xFF gives 0xFF, minus two and three gives two.
        self.assertEqual(-1 & 0xFF, 0xFF)
        self.assertEqual(-2 & 3, 2)

    @make_dynamo_test
    def test_and_chained(self):
        # [FUNCTION DOC] Checks that chained and operators over three ints evaluate left to right
        # to the expected bit pattern.
        self.assertEqual(0b1110 & 0b1100 & 0b1000, 0b1000)

    # --- Booleans ---

    @make_dynamo_test
    def test_and_bools(self):
        # [FUNCTION DOC] Checks the full truth table of and over booleans; results compare equal
        # to bools.
        self.assertEqual(True & True, True)
        self.assertEqual(True & False, False)
        self.assertEqual(False & True, False)
        self.assertEqual(False & False, False)

    @make_dynamo_test
    def test_and_int_and_bool(self):
        # [FUNCTION DOC] Checks that an int combined with a bool on either side behaves as an int
        # and (bit one survives).
        self.assertEqual(5 & True, 1)
        self.assertEqual(True & 3, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_and_float_raises(self):
        # [FUNCTION DOC] Checks that the and operator between an int and a float, in either order,
        # raises TypeError.
        with self.assertRaises(TypeError):
            1 & 1.0
        with self.assertRaises(TypeError):
            1.0 & 1

    # --- Inplace &= ---

    @make_dynamo_test
    def test_inplace_and_integers(self):
        # [FUNCTION DOC] Checks the inplace and-assign operator on a local int gives the masked
        # value.
        x = 0b1110
        x &= 0b1100
        self.assertEqual(x, 0b1100)

    @make_dynamo_test
    def test_inplace_and_chained(self):
        # [FUNCTION DOC] Checks two consecutive inplace and-assign operations on a local int
        # narrow the mask step by step.
        x = 0b1111
        x &= 0b1110
        x &= 0b1100
        self.assertEqual(x, 0b1100)

    # --- Sets ---

    @make_dynamo_test
    def test_set_and_set(self):
        # [FUNCTION DOC] Checks the and operator on two sets gives their intersection.
        self.assertEqual({1, 2, 3} & {2, 3, 4}, {2, 3})

    @make_dynamo_test
    def test_set_iand_set(self):
        # [FUNCTION DOC] Checks the inplace and-assign on a set leaves the intersection in the
        # same variable.
        s = {1, 2, 3}
        s &= {2, 3, 4}
        self.assertEqual(s, {2, 3})

    @make_dynamo_test
    def test_frozenset_and(self):
        # [FUNCTION DOC] Checks the and operator on two frozensets gives the frozenset
        # intersection.
        self.assertEqual(frozenset({1, 2, 3}) & frozenset({2, 3, 4}), frozenset({2, 3}))

    # --- Dict views ---

    @make_dynamo_test
    def test_dict_keys_and_set(self):
        # [FUNCTION DOC] Checks a dict keys view and a set gives the set of common keys.
        self.assertEqual({1: 0, 2: 0, 3: 0}.keys() & {2, 3, 4}, {2, 3})

    @make_dynamo_test
    def test_set_and_dict_keys(self):
        # [FUNCTION DOC] Checks a set and a dict keys view (reversed operand order) gives the
        # common elements.
        self.assertEqual({2, 3, 4} & {1: 0, 2: 0, 3: 0}.keys(), {2, 3})

    @make_dynamo_test
    def test_dict_items_and_set(self):
        # [FUNCTION DOC] Checks a dict items view and a set of pairs gives the set of matching
        # key/value pairs; only the pair that matches in both key and value survives.
        d = {1: "a", 2: "b", 3: "c"}
        self.assertEqual(d.items() & {(2, "b"), (3, "x")}, {(2, "b")})

    # --- OrderedSet ---

    @make_dynamo_test
    def test_ordered_set_and(self):
        # [FUNCTION DOC] Checks the and operator on two OrderedSet objects gives an OrderedSet of
        # the common elements.
        self.assertEqual(
            OrderedSet([1, 2, 3]) & OrderedSet([2, 3, 4]), OrderedSet([2, 3])
        )

    # --- User-defined __and__ ---

    @make_dynamo_test
    def test_user_defined_and_basic(self):
        # [FUNCTION DOC] Checks a user-defined class defining __and__ taking another instance; the
        # result is a new instance and equality is by value.
        class C:
            # [CLASS DOC] HELPER: Test class wrapping an int value; defines __and__ and value-
            # based __eq__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance of the same type holding the
                # bitwise and of both values.
                return type(self)(self.value & other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other object has exactly the same type and
                # an equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) & C(0b1100), C(0b1100))

    @make_dynamo_test
    def test_user_defined_and_with_integer(self):
        # [FUNCTION DOC] Checks user-defined __and__ where the right operand is a plain int rather
        # than an instance.
        class C:
            # [CLASS DOC] HELPER: Test class wrapping an int value; defines __and__ accepting a
            # plain int, and value-based __eq__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance of the same type holding the value
                # anded with the raw other operand.
                return type(self)(self.value & other)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other object has exactly the same type and
                # an equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) & 0b1100, C(0b1100))

    @make_dynamo_test
    def test_reversed_and_with_integer(self):
        # [FUNCTION DOC] Checks that an int on the left and an object defining only __rand__ on
        # the right dispatches to the reflected method.
        class C:
            # [CLASS DOC] HELPER: Test class wrapping an int value; defines only the reflected
            # __rand__ and value-based __eq__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __rand__(self, other):
                # [FUNCTION DOC] HELPER: Reflected and: returns a new instance holding the left
                # operand anded with its own value.
                return type(self)(other & self.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other object has exactly the same type and
                # an equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(0b1110 & C(0b1100), C(0b1100))

    @make_dynamo_test
    def test_user_defined_and_chained(self):
        # [FUNCTION DOC] Checks chained and over three user-defined instances using __and__ twice,
        # left to right.
        class C:
            # [CLASS DOC] HELPER: Test class wrapping an int value; defines __and__ for instances
            # and value-based __eq__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Returns a new instance holding the and of both values.
                return type(self)(self.value & other.value)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other object has exactly the same type and
                # an equal value.
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) & C(0b1100) & C(0b1000), C(0b1000))

    @make_dynamo_test
    def test_user_defined_iand(self):
        # [FUNCTION DOC] Checks the inplace and-assign operator dispatches to a user-defined
        # __iand__ which mutates and returns the same object.
        class C:
            # [CLASS DOC] HELPER: Test class with a mutable value; defines __iand__ and value-
            # based __eq__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
                self.value = v

            def __iand__(self, other):
                # [FUNCTION DOC] HELPER: Inplace and: updates its own value by anding with the
                # other value and returns itself.
                self.value &= other.value
                return self

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal when the other object has exactly the same type and
                # an equal value.
                return type(other) is type(self) and self.value == other.value

        a = C(0b1110)
        a &= C(0b1100)
        self.assertEqual(a, C(0b1100))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Checks that an int subclass overriding __and__ and __rand__ takes
        # priority on both sides: left operand uses __and__, and when it is on the right the
        # subclass reflected method is used even though int is the left type.
        class IntSub(int):
            # [CLASS DOC] HELPER: Int subclass whose __and__ and __rand__ return marker strings
            # naming the method.
            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "IntSub.__and__"

            def __rand__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "IntSub.__rand__"

        self.assertEqual(IntSub(1) & 1, "IntSub.__and__")
        self.assertEqual(1 & IntSub(1), "IntSub.__rand__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline check with a plain object subclass defining both methods:
        # forward is used on the left and reflected is used when the other operand is an int.
        class Base:
            # [CLASS DOC] HELPER: Plain class whose __and__ and __rand__ return marker strings.
            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
                return "Base.__and__"

            def __rand__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
                return "Base.__rand__"

        self.assertEqual(Base() & 1, "Base.__and__")
        self.assertEqual(1 & Base(), "Base.__rand__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_and_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] Checks that when both __and__ and __rand__ return NotImplemented for an
        # object and itself, the and operator raises TypeError.
        class C:
            # [CLASS DOC] HELPER: Class whose forward and reflected and both decline by returning
            # NotImplemented.
            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

            def __rand__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a & a

    @make_dynamo_test
    def test_and_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Checks fallback: the left operand __and__ returns NotImplemented, so the
        # right operand reflected method is called and its result is returned.
        class A:
            # [CLASS DOC] HELPER: Class whose __and__ returns NotImplemented.
            def __and__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        class B:
            # [CLASS DOC] HELPER: Class defining only a reflected and that returns a marker
            # string.
            def __rand__(self, other):
                # [FUNCTION DOC] HELPER: Returns the marker string showing the reflected method
                # ran.
                return "B.__rand__ called"

        result = A() & B()
        self.assertEqual(result, "B.__rand__ called")

    # --- Tensor and ---

    def test_and_tensor_and_tensor(self):
        # [FUNCTION DOC] Compiles a function applying the and operator to two boolean tensors with
        # the eager backend and fullgraph and compares to eager results.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the and of the two tensor arguments.
            return x & y

        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_inplace_and_tensor(self):
        # [FUNCTION DOC] Compiles a function doing inplace and-assign on an int64 tensor argument;
        # runs with clones of the input so the in-place mutation is compared fairly.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies inplace and-assign of the second argument onto the
            # first and returns the first.
            x &= y
            return x

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        y = torch.tensor([0b1100, 0b1010], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x.clone(), y), fn(x.clone(), y))

    def test_and_tensor_and_tensor_int(self):
        # [FUNCTION DOC] Compiles the and operator over two int64 tensors and compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the and of the two tensor arguments.
            return x & y

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        y = torch.tensor([0b1100, 0b1010], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_and_tensor_and_int(self):
        # [FUNCTION DOC] Compiles tensor and python int on the right and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the tensor anded with the int seven.
            return x & 7

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_int_and_tensor(self):
        # [FUNCTION DOC] Compiles python int on the left and tensor on the right and compares with
        # eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns seven anded with the tensor argument.
            return 7 & x

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_tensor_and_bool(self):
        # [FUNCTION DOC] Compiles a boolean tensor and the Python bool True on the right and
        # compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the tensor anded with True.
            return x & True

        x = torch.tensor([True, False, True, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_bool_and_tensor(self):
        # [FUNCTION DOC] Compiles the Python bool True on the left and a boolean tensor on the
        # right and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns True anded with the tensor argument.
            return True & x

        x = torch.tensor([True, False, True, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    # --- SymNode and ---

    def test_and_symnode_and_int(self):
        # [FUNCTION DOC] Compiles a function that ands a dynamic (symbolic) size with the int
        # seven and uses it as a size for new_zeros; the input has dimension zero marked dynamic.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the first dimension size, ands it with seven and
            # returns a zero tensor of that size.
            s = x.shape[0]
            return x.new_zeros(s & 7)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_int_and_symnode(self):
        # [FUNCTION DOC] Same as the previous symbolic test but with the int on the left of the
        # and operator.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the first dimension size, ands seven with it and
            # returns a zero tensor of that size.
            s = x.shape[0]
            return x.new_zeros(7 & s)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


instantiate_parametrized_tests(TestNbAnd)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
