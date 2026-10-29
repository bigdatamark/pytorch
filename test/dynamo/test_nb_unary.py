# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for unary operators on many value kinds under dynamo: negate, positive,
# absolute and invert, via both operator functions and dunder methods. Defines the lists
# UNARY_OPS (without invert) and ALL_UNARY_OPS used to parametrize one test class, instantiated
# at import time.
"""Tests for nb unary: unary ops via PyNumber_slot."""

import operator
import sys

import torch
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
    subtest,
)


# Ops that support float and complex (neg, pos, abs).
UNARY_OPS = [
    subtest((operator.neg, "__neg__"), name="neg"),
    subtest((operator.pos, "__pos__"), name="pos"),
    subtest((operator.abs, "__abs__"), name="abs"),
]

# All unary ops including invert (int/bool only, no float/complex).
ALL_UNARY_OPS = UNARY_OPS + [
    subtest((operator.invert, "__invert__"), name="inv"),
]


class NbUnaryTests(TestCase):
    # [CLASS DOC] Parametrized tests for unary operators (negate, positive, absolute, invert) in
    # dynamo. Exercises constants, tensors, symbolic sizes, user-defined objects, modules,
    # metaclasses, enums, unbound dunder calls and error cases. Stateless; the class is
    # parametrized at import time with instantiate_parametrized_tests.
    # --- int (ConstantVariable) ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_int(self, op, dunder):
        # [FUNCTION DOC] The test runs eagerly and then under dynamo (make_dynamo_test), for each
        # parametrized unary operator on a small positive int; the assertion compares the
        # operator with itself so it checks the traced result is consistent and graph-break free.
        self.assertEqual(op(42), op(42))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_int_negative_value(self, op, dunder):
        # [FUNCTION DOC] Same as the int test but with a negative int argument.
        self.assertEqual(op(-7), op(-7))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_int_zero(self, op, dunder):
        # [FUNCTION DOC] Same as the int test but with zero.
        self.assertEqual(op(0), op(0))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_int_large(self, op, dunder):
        # [FUNCTION DOC] Same as the int test but with a very large int (a hundred bits) to check
        # big-int handling.
        self.assertEqual(op(2**100), op(2**100))

    # --- float (ConstantVariable) ---
    # invert does not apply to float; these use UNARY_OPS only.

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float(self, op, dunder):
        # [FUNCTION DOC] Applies each of the negate, positive and absolute operators to a float
        # constant under make_dynamo_test.
        self.assertEqual(op(3.14), op(3.14))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float_negative_value(self, op, dunder):
        # [FUNCTION DOC] Same as the float test with a negative float constant.
        self.assertEqual(op(-2.5), op(-2.5))

    @parametrize("op,dunder", UNARY_OPS)
    def test_float_zero_copysign(self, op, dunder):
        # [FUNCTION DOC] Compiles (eager backend, fullgraph) a function applying the operator to
        # positive zero and checks the sign of the result (via copysign) matches eager, so
        # negative zero is preserved.
        import math

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the operator
            # applied to 0.0.
            return op(0.0)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        expected = op(0.0)
        self.assertEqual(math.copysign(1.0, result), math.copysign(1.0, expected))

    @parametrize("op,dunder", UNARY_OPS)
    def test_float_double_apply_zero(self, op, dunder):
        # [FUNCTION DOC] Like the zero-sign test but applies the operator twice, comparing the
        # sign of the result against eager.
        import math

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the operator applied twice
            # to 0.0.
            return op(op(0.0))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        expected = op(op(0.0))
        self.assertEqual(math.copysign(1.0, result), math.copysign(1.0, expected))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float_inf(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to positive infinity under make_dynamo_test.
        import math

        self.assertEqual(op(math.inf), op(math.inf))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float_neg_inf(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to negative infinity under make_dynamo_test.
        import math

        self.assertEqual(op(-math.inf), op(-math.inf))

    @parametrize("op,dunder", UNARY_OPS)
    def test_float_nan(self, op, dunder):
        # [FUNCTION DOC] Compiles a function applying the operator to NaN and checks the compiled
        # result is NaN.
        import math

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the operator applied to a
            # float NaN.
            return op(float("nan"))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(math.isnan(result))

    # --- complex (ConstantVariable) ---
    # invert does not apply to complex; these use UNARY_OPS only.

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a complex constant under make_dynamo_test.
        self.assertEqual(op(3 + 4j), op(3 + 4j))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex_zero(self, op, dunder):
        # [FUNCTION DOC] Same as the complex test with complex zero.
        self.assertEqual(op(0 + 0j), op(0 + 0j))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex_real_only(self, op, dunder):
        # [FUNCTION DOC] Same as the complex test with a purely real complex value.
        self.assertEqual(op(5 + 0j), op(5 + 0j))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex_imag_only(self, op, dunder):
        # [FUNCTION DOC] Same as the complex test with a purely imaginary complex value.
        self.assertEqual(op(0 + 3j), op(0 + 3j))

    # --- bool (ConstantVariable, inherits slot from int) ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_bool_true(self, op, dunder):
        # [FUNCTION DOC] Applies the operator (including invert) to True under make_dynamo_test.
        self.assertEqual(op(True), op(True))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_bool_false(self, op, dunder):
        # [FUNCTION DOC] Applies the operator (including invert) to False under make_dynamo_test.
        self.assertEqual(op(False), op(False))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_bool_result_type(self, op, dunder):
        # [FUNCTION DOC] Compiles a function applying the operator to True and checks the
        # result is an int and not a bool, matching CPython where these operators return
        # plain ints for booleans.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the operator
            # applied to True.
            return op(True)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertNotIsInstance(result, bool)

    # --- operator.X on constants ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_operator_int(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to an int constant under make_dynamo_test.
        self.assertEqual(op(10), op(10))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_operator_float(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a float constant under make_dynamo_test.
        self.assertEqual(op(2.5), op(2.5))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_operator_bool(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a bool constant under make_dynamo_test.
        self.assertEqual(op(True), op(True))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_operator_complex(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a complex constant under make_dynamo_test.
        self.assertEqual(op(1 + 2j), op(1 + 2j))

    # --- dunder method on constants ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_dunder_int(self, op, dunder):
        # [FUNCTION DOC] Calls the operator dunder method looked up by name on an int and
        # compares with the operator function, under make_dynamo_test.
        self.assertEqual(getattr(42, dunder)(), op(42))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_dunder_float(self, op, dunder):
        # [FUNCTION DOC] Same as the int dunder test but on a float.
        self.assertEqual(getattr(3.14, dunder)(), op(3.14))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_dunder_bool(self, op, dunder):
        # [FUNCTION DOC] Same as the int dunder test but on a bool.
        self.assertEqual(getattr(True, dunder)(), op(True))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_dunder_complex(self, op, dunder):
        # [FUNCTION DOC] Same as the int dunder test but on a complex number.
        self.assertEqual(getattr(1 + 2j, dunder)(), op(1 + 2j))

    # --- TypeError for types without the slot ---
    # CPython: "bad operand type for unary <symbol>: '<type>'"

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_str_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a string inside a function that catches
        # TypeError and returns its message. Compiles with fullgraph and compares the captured
        # error message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to a
            # string, or the TypeError message string if it raises.
            try:
                return op("hello")
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_list_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a list inside a function that catches TypeError
        # and returns its message. Compiles with fullgraph and compares the captured error
        # message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to a
            # list, or the TypeError message string if it raises.
            try:
                return op([1, 2, 3])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_dict_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a dict inside a function that catches TypeError
        # and returns its message. Compiles with fullgraph and compares the captured error
        # message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to a
            # dict, or the TypeError message string if it raises.
            try:
                return op({"a": 1})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_set_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a set inside a function that catches TypeError
        # and returns its message. Compiles with fullgraph and compares the captured error
        # message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to a set,
            # or the TypeError message string if it raises.
            try:
                return op({1, 2})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_tuple_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to an empty tuple inside a function that catches
        # TypeError and returns its message. Compiles with fullgraph and compares the captured
        # error message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to an
            # empty tuple, or the TypeError message string if it raises.
            try:
                return op(())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_none_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to None inside a function that catches TypeError
        # and returns its message. Compiles with fullgraph and compares the captured error
        # message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to None,
            # or the TypeError message string if it raises.
            try:
                return op(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_bytes_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a bytes object inside a function that catches
        # TypeError and returns its message. Compiles with fullgraph and compares the captured
        # error message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to a
            # bytes object, or the TypeError message string if it raises.
            try:
                return op(b"hello")
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_operator_none_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to None (second variant of the none test) inside a
        # function that catches TypeError and returns its message. Compiles with fullgraph and
        # compares the captured error message string with the eager one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to None
            # (second variant of the none test), or the TypeError message string if it raises.
            try:
                return op(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Tensor (TensorVariable, proxy-based) ---
    # Float/complex tensor tests use UNARY_OPS only (invert doesn't support them).

    @parametrize("op,dunder", UNARY_OPS)
    def test_tensor(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a random float tensor. Compiles with fullgraph
        # and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied to the input.
            return op(x)

        x = torch.randn(3, 4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_tensor_integer_dtype(self, op, dunder):
        # [FUNCTION DOC] Applies the operator (including invert) to an int tensor with mixed
        # signs. Compiles with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied to the input.
            return op(x)

        x = torch.tensor([1, -2, 3])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", UNARY_OPS)
    def test_tensor_complex_dtype(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a complex tensor. Compiles with fullgraph and
        # compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied to the input.
            return op(x)

        x = torch.tensor([1 + 2j, -3 + 4j])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_tensor_creates_graph_node(self, op, dunder):
        # [FUNCTION DOC] Compiles with a graph-recording backend and fullgraph, runs on an int
        # tensor, and checks exactly one graph was captured whose readable text contains the
        # operator function name in lowercase.
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that returns the operator applied to the
            # input tensor.
            return op(x)

        fn(torch.randint(0, 5, (3,)))
        self.assertEqual(len(backend.graphs), 1)
        graph_str = backend.graphs[0].print_readable(False)
        self.assertIn(op.__name__, graph_str.lower())

    @parametrize("op,dunder", UNARY_OPS)
    def test_tensor_operator(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a random float tensor (operator-function
        # variant). Compiles with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied to the input.
            return op(x)

        x = torch.randn(5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", UNARY_OPS)
    def test_tensor_dunder(self, op, dunder):
        # [FUNCTION DOC] Calls the tensor method named by the dunder on a float tensor. Compiles
        # with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the tensor dunder method and returns the result.
            return getattr(x, dunder)()

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_tensor_integer_dtype_dunder(self, op, dunder):
        # [FUNCTION DOC] Calls the tensor dunder method (including invert) on an int tensor.
        # Compiles with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the tensor dunder method and returns the result.
            return getattr(x, dunder)()

        x = torch.tensor([1, -2, 3])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- SymNodeVariable ---
    # SymInt does not support invert; these use UNARY_OPS only.

    @parametrize("op,dunder", UNARY_OPS)
    def test_symnode(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to the symbolic-capable first dimension size of a
        # tensor and slices up to that bound, comparing compiled and eager results.
        def fn(x):
            # [FUNCTION DOC] HELPER: Slices the input along the first dimension using the
            # operator applied to its size.
            s = x.size(0)
            return x[: op(s)]

        x = torch.randn(10, 5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", UNARY_OPS)
    def test_symnode_operator(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to the symbolic-capable first dimension size of a
        # tensor and slices up to that bound, comparing compiled and eager results.
        def fn(x):
            # [FUNCTION DOC] HELPER: Slices the input along the first dimension using the
            # operator applied to its size.
            s = x.size(0)
            return x[: op(s)]

        x = torch.randn(8, 4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", UNARY_OPS)
    def test_symnode_dunder(self, op, dunder):
        # [FUNCTION DOC] Like the symnode test but calls the dunder method by name on the
        # size value.
        def fn(x):
            # [FUNCTION DOC] HELPER: Slices the input along dimension zero using the dunder
            # applied to its size.
            s = x.size(0)
            return x[: getattr(s, dunder)()]

        x = torch.randn(6, 3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- UserDefinedObjectVariable ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined(self, op, dunder):
        # [FUNCTION DOC] Attaches the unary dunder to a local class that returns a negated copy,
        # compiles a function applying the operator to an instance, and checks the resulting
        # value attribute is negative ten.
        class MyNum:
            # [CLASS DOC] HELPER: Simple class holding one value attribute; the unary dunder
            # is attached afterwards with a function that returns a new instance with the
            # negated value.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the constructor argument on the instance.
                self.val = val

        setattr(MyNum, dunder, lambda self: MyNum(-self.val))
        obj = MyNum(10)

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the value attribute of the
            # operator applied to the captured instance.
            return op(obj).val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -10)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined_operator(self, op, dunder):
        # [FUNCTION DOC] Same as the user-defined test with value seven, checking negative seven.
        class MyNum:
            # [CLASS DOC] HELPER: Simple class holding one value attribute; the unary dunder
            # is attached afterwards with a function that returns a new instance with the
            # negated value.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the constructor argument on the instance.
                self.val = val

        setattr(MyNum, dunder, lambda self: MyNum(-self.val))
        obj = MyNum(7)

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the value attribute of the
            # operator applied to the captured instance.
            return op(obj).val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -7)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined_dunder(self, op, dunder):
        # [FUNCTION DOC] Same as the user-defined test with value three but invokes the dunder
        # method directly by name, checking negative three.
        class MyNum:
            # [CLASS DOC] HELPER: Simple class holding one value attribute; the unary dunder
            # is attached afterwards with a function that returns a new instance with the
            # negated value.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the constructor argument on the instance.
                self.val = val

        setattr(MyNum, dunder, lambda self: MyNum(-self.val))
        obj = MyNum(3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the value attribute of calling
            # the dunder method on the captured instance.
            return getattr(obj, dunder)().val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -3)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined_no_slot_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to an instance of a class defining no such dunder,
        # in a function catching TypeError, and checks compiled and eager messages match.
        class NoSlot:
            # [CLASS DOC] HELPER: Empty class with no unary methods.
            pass

        obj = NoSlot()

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # captured instance or the TypeError message string.
            try:
                return op(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined_returns_arbitrary_type(self, op, dunder):
        # [FUNCTION DOC] Attaches a dunder returning a string to a local class and checks the
        # compiled result of the operator is that string (non-instance return types are allowed).
        class Weird:
            # [CLASS DOC] HELPER: Empty class; the dunder is attached later.
            pass

        setattr(Weird, dunder, lambda self: "applied")
        obj = Weird()

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # captured instance.
            return op(obj)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "applied")

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined_raising_exception_propagates(self, op, dunder):
        # [FUNCTION DOC] Attaches a dunder that raises ValueError to a local class and checks
        # the exception propagates into the compiled function where it is caught and its
        # message returned.
        class Raising:
            # [CLASS DOC] HELPER: Empty class; the raising dunder is attached later.
            pass

        def raise_error(self):
            # [FUNCTION DOC] HELPER: Dunder replacement that always raises ValueError with a
            # custom message.
            raise ValueError("custom error")

        setattr(Raising, dunder, raise_error)
        obj = Raising()

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # captured instance, or the ValueError message string.
            try:
                return op(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_user_defined_subclass(self, op, dunder):
        # [FUNCTION DOC] Attaches the dunder to a base class and applies the operator to an
        # instance of a subclass, checking inherited dispatch gives the negated value.
        class Base:
            # [CLASS DOC] HELPER: Base class holding one value attribute; the dunder is attached
            # later returning a negated instance of the same type.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument on the instance.
                self.v = v

        setattr(Base, dunder, lambda self: type(self)(-self.v))

        class Sub(Base):
            # [CLASS DOC] HELPER: Empty subclass of the base used to test inherited
            # dunder dispatch.
            pass

        obj = Sub(5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the value attribute of the
            # operator applied to the captured instance.
            return op(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -5)

    # --- UserDefinedObjectVariable wrapping a Python constant (enum) ---

    def test_invert_flag_enum(self):
        # [FUNCTION DOC] Checks inverting a flag enum member inside a compiled function gives the
        # other member (identity comparison is true).
        import enum

        class Flags(enum.Flag):
            # [CLASS DOC] HELPER: Enum flag type with two members of values one and two.
            A = 1
            B = 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns whether inverting the first
            # flag member is the second member.
            return ~Flags.A is Flags.B

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_enum_constant(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a member of an integer enum and checks the
        # compiled result equals the eager one.
        import enum

        class MyEnum(enum.IntEnum):
            # [CLASS DOC] HELPER: Integer enum with one member of value forty-two.
            X = 42

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # enum member.
            return op(MyEnum.X)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, op(MyEnum.X))

    # --- nn.Module ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_nn_module(self, op, dunder):
        # [FUNCTION DOC] Attaches the dunder to an nn.Module subclass, returning a constant
        # tensor, then compiles a function adding the operator result on the module to the input
        # and compares with eager.
        class OpModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns its input unchanged; the unary
            # dunder is attached afterwards.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        setattr(OpModule, dunder, lambda self: torch.tensor(-1.0))
        mod = OpModule()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied to the captured module plus
            # the input.
            return op(mod) + x

        x = torch.tensor(5.0)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- Metaclass (UserDefinedClassVariable path) ---
    # In CPython, op(SomeClass) calls type(SomeClass)->nb_slot,
    # i.e. the metaclass's dunder, not the class's.

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_metaclass(self, op, dunder):
        # [FUNCTION DOC] Defines a metaclass with the dunder attached, a class using it, and
        # checks the operator on the class (not an instance) calls the metaclass dunder,
        # producing a string with the class name.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass; the dunder is attached afterwards returning a string
            # with the class name.
            pass

        setattr(Meta, dunder, lambda cls: f"applied {cls.__name__}")

        class A(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using the metaclass under test.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # class object.
            return op(A)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "applied A")

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_metaclass_no_slot_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to a class whose metaclass defines no such dunder,
        # catching TypeError, and checks compiled and eager messages match.
        class Bar(type):
            # [CLASS DOC] HELPER: Metaclass without unary methods.
            pass

        class B(metaclass=Bar):
            # [CLASS DOC] HELPER: Class using the metaclass without unary methods.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # class or the TypeError message.
            try:
                return op(B)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_metaclass_inherited(self, op, dunder):
        # [FUNCTION DOC] Like the metaclass test but the dunder is on a parent metaclass of the
        # one used by the class, checking inheritance of the metaclass slot.
        class Meta(type):
            # [CLASS DOC] HELPER: Base metaclass; the dunder is attached afterwards.
            pass

        setattr(Meta, dunder, lambda cls: f"applied {cls.__name__}")

        class SubMeta(Meta):
            # [CLASS DOC] HELPER: Metaclass subclass inheriting the dunder.
            pass

        class D(metaclass=SubMeta):
            # [CLASS DOC] HELPER: Class using the derived metaclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # class object.
            return op(D)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "applied D")

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_default_metaclass_raises(self, op, dunder):
        # [FUNCTION DOC] Applies the operator to an ordinary class (default metaclass) catching
        # TypeError, and checks compiled and eager messages match.
        class Plain:
            # [CLASS DOC] HELPER: Plain empty class.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the operator applied to the
            # class or the TypeError message.
            try:
                return op(Plain)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Unbound method calls: Type.dunder(instance) ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_unbound_user_defined(self, op, dunder):
        # [FUNCTION DOC] Calls the dunder function through the class (unbound) with an instance,
        # checking the negated value.
        class MyNum:
            # [CLASS DOC] HELPER: Holds one value attribute; dunder attached afterwards returning
            # a negated instance.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument on the instance.
                self.v = v

        setattr(MyNum, dunder, lambda self: MyNum(-self.v))
        obj = MyNum(5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the value attribute of the
            # class-level dunder applied to the captured instance.
            return getattr(MyNum, dunder)(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -5)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_unbound_parent_class(self, op, dunder):
        # [FUNCTION DOC] Like the unbound test but fetches the dunder from the base class and
        # passes a subclass instance.
        class Base:
            # [CLASS DOC] HELPER: Base class with one value attribute; dunder
            # attached afterwards.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the constructor argument on the instance.
                self.v = v

        setattr(Base, dunder, lambda self: type(self)(-self.v))

        class Sub(Base):
            # [CLASS DOC] HELPER: Empty subclass of the base.
            pass

        obj = Sub(10)

        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the value attribute of the
            # base class dunder applied to the captured instance.
            return getattr(Base, dunder)(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -10)

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_int(self, op, dunder):
        # [FUNCTION DOC] Calls the unbound dunder on the int type with a number and compares with
        # the operator, under make_dynamo_test.
        self.assertEqual(getattr(int, dunder)(42), op(42))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_float(self, op, dunder):
        # [FUNCTION DOC] Same as the unbound int test for the float type.
        self.assertEqual(getattr(float, dunder)(3.14), op(3.14))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_complex(self, op, dunder):
        # [FUNCTION DOC] Same as the unbound int test for the complex type.
        self.assertEqual(getattr(complex, dunder)(1 + 2j), op(1 + 2j))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_bool(self, op, dunder):
        # [FUNCTION DOC] Same as the unbound int test for the bool type.
        self.assertEqual(getattr(bool, dunder)(True), op(True))

    # --- Double application ---

    @parametrize("op,dunder", ALL_UNARY_OPS)
    @make_dynamo_test
    def test_double_apply_int(self, op, dunder):
        # [FUNCTION DOC] Applies the operator twice to an int under make_dynamo_test.
        self.assertEqual(op(op(42)), op(op(42)))

    @parametrize("op,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_double_apply_float(self, op, dunder):
        # [FUNCTION DOC] Applies the operator twice to a float under make_dynamo_test.
        self.assertEqual(op(op(3.14)), op(op(3.14)))

    @parametrize("op,dunder", UNARY_OPS)
    def test_double_apply_tensor(self, op, dunder):
        # [FUNCTION DOC] Applies the operator twice to a random float tensor. Compiles with
        # fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied twice to the input.
            return op(op(x))

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_double_apply_tensor_integer_dtype(self, op, dunder):
        # [FUNCTION DOC] Applies the operator (including invert) twice to an int tensor. Compiles
        # with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the operator applied twice to the input.
            return op(op(x))

        x = torch.tensor([1, 2, 3])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- Used in expressions ---

    @parametrize("op,dunder", UNARY_OPS)
    def test_in_arithmetic(self, op, dunder):
        # [FUNCTION DOC] Adds a float tensor to the operator applied to itself. Compiles with
        # fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus the operator applied to the input.
            return x + op(x)

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_in_arithmetic_integer_dtype(self, op, dunder):
        # [FUNCTION DOC] Adds an int tensor to the operator applied to itself (including invert).
        # Compiles with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus the operator applied to the input.
            return x + op(x)

        x = torch.tensor([1, 2, 3])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", UNARY_OPS)
    def test_constant_in_tensor_op(self, op, dunder):
        # [FUNCTION DOC] Adds a float tensor of ones to the operator applied to the constant five
        # (folded at trace time). Compiles with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the operator to the constant five and adds it to
            # the input.
            val = op(5)
            return x + val

        x = torch.ones(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,dunder", ALL_UNARY_OPS)
    def test_constant_in_tensor_op_integer_dtype(self, op, dunder):
        # [FUNCTION DOC] Same as the previous test with an int64 tensor and invert included.
        # Compiles with fullgraph and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the operator to the constant five and adds it to
            # the input.
            val = op(5)
            return x + val

        x = torch.ones(3, dtype=torch.int64)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- op specific tests ---

    # --- complex abs overflow (OverflowError from _Py_c_abs) ---
    # CPython's complex_abs calls _Py_c_abs which sets errno=ERANGE when
    # hypot(real, imag) overflows a double, then raises OverflowError.

    def test_complex_abs_overflow(self):
        # [FUNCTION DOC] Checks that the absolute value of a complex number with huge components
        # raises OverflowError the same way in compiled code as in eager, comparing returned
        # message strings.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns the absolute value of the
            # complex with maximum float parts, or the OverflowError message.
            try:
                return abs(complex(sys.float_info.max, sys.float_info.max))
            except OverflowError as e:
                return str(e)

        eager_result = fn(0)
        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_complex_abs_return_type(self):
        # [FUNCTION DOC] Compares the dtype of the absolute value of a complex float32 tensor
        # between compiled and eager runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the absolute value of the input.
            return abs(x)

        real = torch.rand(1, dtype=torch.float32)
        imag = torch.rand(1, dtype=torch.float32)
        complex_val = torch.complex(real, imag)
        eager_result = fn(complex_val)
        result = torch.compile(fn, backend="eager", fullgraph=True)(complex_val)
        self.assertEqual(result.dtype, eager_result.dtype)

    # --- nb_invert specific tests ---
    # Tests for behavior unique to invert: float/complex TypeError,
    # bool tensor support, float tensor TypeError, int tensor graph ops.

    def test_invert_float_raises(self):
        # [FUNCTION DOC] Inverts a float constant in a function catching TypeError and checks
        # compiled and eager messages match.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns bitwise inversion of 1.5 or
            # the TypeError message string.
            try:
                return ~1.5
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_invert_complex_raises(self):
        # [FUNCTION DOC] Inverts a complex constant in a function catching TypeError and checks
        # compiled and eager messages match.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its argument; returns bitwise inversion of a complex
            # literal or the TypeError message string.
            try:
                return ~(1 + 2j)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_invert_tensor_bool_dtype(self):
        # [FUNCTION DOC] Compiles bitwise inversion of a boolean tensor and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the bitwise inversion of the input.
            return ~x

        x = torch.tensor([True, False, True])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_invert_tensor_float_raises(self):
        # [FUNCTION DOC] Checks that compiling bitwise inversion of a float tensor
        # raises RuntimeError.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the bitwise inversion of the input.
            return ~x

        x = torch.randn(3)
        with self.assertRaises(RuntimeError):
            torch.compile(fn, backend="eager", fullgraph=True)(x)


instantiate_parametrized_tests(NbUnaryTests)


if __name__ == "__main__":
    run_tests()
