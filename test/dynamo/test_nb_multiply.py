# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the multiplication operator and its inplace form (nb_multiply
# slot, with the sequence repeat fallback): ints, bools, floats, complex, sequence repetition for
# list, tuple, str and bytes, inplace forms, type errors, operator module functions, direct dunder
# calls, user-defined __mul__/__rmul__, subclass priority, tensors and symbolic ints and floats.
# Defines module-level helper classes. The test class is wrapped by instantiate_parametrized_tests
# at import time; it is patched to trace unittest.

"""Tests for the * and *= operators in PyTorch Dynamo (nb_multiply slot)."""

import operator
import sys

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
)


class UserDefinedClassWithMul:
    # [CLASS DOC] HELPER: Helper class wrapping a value; multiplication and reflected
    # multiplication multiply the stored value by another instance value or a raw number and
    # return a new instance.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
        self.value = value

    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance with the stored value times the other
        # instance value or raw operand.
        if isinstance(other, UserDefinedClassWithMul):
            return UserDefinedClassWithMul(self.value * other.value)
        return UserDefinedClassWithMul(self.value * other)

    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: Reflected: returns a new instance with the left operand (instance
        # value or raw number) times the stored value.
        if isinstance(other, UserDefinedClassWithMul):
            return UserDefinedClassWithMul(other.value * self.value)
        return UserDefinedClassWithMul(other * self.value)

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is an instance of the same class and values
        # are equal.
        return isinstance(other, UserDefinedClassWithMul) and self.value == other.value

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string showing the class name and value.
        return f"UserDefinedClassWithMul({self.value})"


class LeftMulClass:
    # [CLASS DOC] HELPER: Helper class whose multiplication works only with the same class and
    # otherwise returns NotImplemented, for operand-fallback tests.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
        self.value = value

    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance with the product when the other is the
        # same class; otherwise NotImplemented.
        if isinstance(other, LeftMulClass):
            return LeftMulClass(self.value * other.value)
        return NotImplemented

    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: Reflected: returns a new instance with the product when the other
        # is the same class; otherwise NotImplemented.
        if isinstance(other, LeftMulClass):
            return LeftMulClass(other.value * self.value)
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is the same class and values are equal.
        return isinstance(other, LeftMulClass) and self.value == other.value


class RightMulClass:
    # [CLASS DOC] HELPER: Helper class holding a string value; its multiplication joins two values
    # with a star only for the same class, and its reflected multiplication accepts only the left
    # helper class and returns a formatted string.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
        self.value = value

    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance whose value joins both stored strings with
        # a star when the other is the same class; otherwise NotImplemented.
        if isinstance(other, RightMulClass):
            return RightMulClass(self.value + "*" + other.value)
        return NotImplemented

    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: Reflected: when the other is the left helper class returns a
        # formatted string naming both values; otherwise NotImplemented.
        if isinstance(other, LeftMulClass):
            return f"LeftMulClass({other.value})*RightMulClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is the same class and values are equal.
        return isinstance(other, RightMulClass) and self.value == other.value


class _IntSubWithMul(int):
    # [CLASS DOC] HELPER: Int subclass whose forward and reflected multiplication return marker
    # strings, to test priority of a right operand subclass.
    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Returns the marker string for the forward method.
        return "_IntSubWithMul.__mul__"

    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: Returns the marker string for the reflected method.
        return "_IntSubWithMul.__rmul__"


class _BaseWithMul:
    # [CLASS DOC] HELPER: Plain base class whose forward and reflected multiplication return
    # marker strings.
    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Returns the marker string for the base forward method.
        return "_BaseWithMul.__mul__"

    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: Returns the marker string for the base reflected method.
        return "_BaseWithMul.__rmul__"


class _SubWithMul(_BaseWithMul):
    # [CLASS DOC] HELPER: Subclass of the base class overriding both methods with its own marker
    # strings, so as a right operand it gets priority.
    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Returns the marker string for the subclass forward method.
        return "_SubWithMul.__mul__"

    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: Returns the marker string for the subclass reflected method.
        return "_SubWithMul.__rmul__"


class _InheritedSub(_BaseWithMul):
    # [CLASS DOC] HELPER: Subclass of the base class that overrides nothing, used to show an
    # inherited reflected method gives no priority.
    pass


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestNbMultiply(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for multiplication in Dynamo; patched with enable_trace_unittest. Mix
    # of make_dynamo_test cases and compiled tensor or symbolic cases; parametrized-test
    # instantiation runs after the class definition.
    # --- Integer multiply ---

    @make_dynamo_test
    def test_int_mul(self):
        # [FUNCTION DOC] Checks int multiplication including zero and negatives.
        self.assertEqual(2 * 3, 6)
        self.assertEqual(0 * 5, 0)
        self.assertEqual(-2 * 3, -6)
        self.assertEqual(-2 * -3, 6)

    @make_dynamo_test
    def test_int_mul_large(self):
        # [FUNCTION DOC] Checks multiplication of larger ints, including a power-of-two result that
        # exceeds machine word size, to show arbitrary precision.
        self.assertEqual(100 * 200, 20000)
        # Python ints have arbitrary precision.
        self.assertEqual(2**60 * 2, 2**61)

    @make_dynamo_test
    def test_bool_mul(self):
        # [FUNCTION DOC] Checks multiplication involving bools gives int results.
        self.assertEqual(True * True, 1)
        self.assertEqual(True * 5, 5)
        self.assertEqual(False * 5, 0)
        self.assertEqual(5 * True, 5)

    # --- Float multiply ---

    @make_dynamo_test
    def test_float_mul(self):
        # [FUNCTION DOC] Checks float multiplication, including int times float.
        self.assertEqual(2.5 * 4, 10.0)
        self.assertEqual(2 * 2.5, 5.0)
        self.assertEqual(0.5 * 0.5, 0.25)

    @make_dynamo_test
    def test_float_int_mixed(self):
        # [FUNCTION DOC] Checks float times int in both orders.
        self.assertEqual(1.5 * 3, 4.5)
        self.assertEqual(3 * 1.5, 4.5)

    # --- Complex multiply ---

    @make_dynamo_test
    def test_complex_mul(self):
        # [FUNCTION DOC] Checks complex times complex, complex times int and int times complex.
        self.assertEqual(complex(1, 2) * complex(3, 4), complex(-5, 10))
        self.assertEqual(complex(1, 2) * 3, complex(3, 6))
        self.assertEqual(3 * complex(1, 2), complex(3, 6))

    # --- Sequence repetition (sq_repeat fallback) ---

    @make_dynamo_test
    def test_list_mul_int(self):
        # [FUNCTION DOC] Checks list repetition by a positive int, zero and a negative int (the
        # last two give an empty list).
        self.assertEqual([1, 2] * 3, [1, 2, 1, 2, 1, 2])
        self.assertEqual([1, 2] * 0, [])
        self.assertEqual([1, 2] * -1, [])

    @make_dynamo_test
    def test_int_mul_list(self):
        # [FUNCTION DOC] Checks an int on the left repeating a list on the right.
        self.assertEqual(3 * [1, 2], [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_list_mul_bool(self):
        # [FUNCTION DOC] Checks repeating a list by a bool: True gives a copy of the contents,
        # False gives an empty list.
        # bool is index-like, so list * True == list
        self.assertEqual([1, 2] * True, [1, 2])
        self.assertEqual([1, 2] * False, [])

    @make_dynamo_test
    def test_tuple_mul_int(self):
        # [FUNCTION DOC] Checks tuple repetition by an int.
        self.assertEqual((1, 2) * 3, (1, 2, 1, 2, 1, 2))
        self.assertEqual((1,) * 5, (1, 1, 1, 1, 1))

    @make_dynamo_test
    def test_tuple_mul_identity(self):
        # [FUNCTION DOC] Checks tuple identity rules: an empty tuple repeated any number of times
        # is the same object, a tuple repeated once is the same object, and repeating by zero or
        # two gives a different object.
        empty = ()
        self.assertEqual(id(empty), id(empty * -3))
        self.assertEqual(id(empty), id(empty * 0))
        self.assertEqual(id(empty), id(empty * 3))

        values = (1, 2)
        self.assertEqual(id(values), id(values * 1))
        self.assertFalse(id(values) == id(values * 0))
        self.assertFalse(id(values) == id(values * 2))

    @make_dynamo_test
    def test_empty_tuple_mul_overflow_raises(self):
        # [FUNCTION DOC] Checks that repeating the empty tuple by a value just above the maximum
        # index-sized int, or below its negative bound, raises OverflowError with the message
        # about fitting an int into an index-sized integer.
        with self.assertRaisesRegex(
            OverflowError, "cannot fit 'int' into an index-sized integer"
        ):
            () * (sys.maxsize + 1)
        with self.assertRaisesRegex(
            OverflowError, "cannot fit 'int' into an index-sized integer"
        ):
            () * (-sys.maxsize - 2)

    @make_dynamo_test
    def test_list_mul_one_returns_new_list(self):
        # [FUNCTION DOC] Checks repeating a list once gives an equal list that is a different
        # object.
        values = [1, 2]
        self.assertEqual(values * 1, values)
        self.assertFalse(id(values) == id(values * 1))

    @make_dynamo_test
    def test_str_mul_int(self):
        # [FUNCTION DOC] Checks string repetition by int on either side, empty string and negative
        # counts.
        self.assertEqual("ab" * 3, "ababab")
        self.assertEqual(3 * "ab", "ababab")
        self.assertEqual("" * 5, "")
        self.assertEqual("a" * -3, "")

    @make_dynamo_test
    def test_bytes_mul_int(self):
        # [FUNCTION DOC] Checks bytes repetition by an int on either side.
        self.assertEqual(b"ab" * 3, b"ababab")
        self.assertEqual(3 * b"ab", b"ababab")

    # --- Inplace multiply ---

    @make_dynamo_test
    def test_int_imul(self):
        # [FUNCTION DOC] Checks inplace multiply-assign on an int.
        x = 5
        x *= 3
        self.assertEqual(x, 15)

    @make_dynamo_test
    def test_float_imul(self):
        # [FUNCTION DOC] Checks inplace multiply-assign on a float.
        x = 1.5
        x *= 2
        self.assertEqual(x, 3.0)

    @make_dynamo_test
    def test_list_imul(self):
        # [FUNCTION DOC] Checks inplace multiply-assign on a list repeats its contents.
        a = [1, 2]
        a *= 3
        self.assertEqual(a, [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_tuple_imul(self):
        # [FUNCTION DOC] Checks inplace multiply-assign on a tuple, which has no inplace repeat
        # and so rebinds to a new repeated tuple.
        # tuple lacks sq_inplace_repeat; *= falls back to non-inplace via
        # pynumber_inplace_multiply -> sq_repeat.
        t = (1, 2)
        t *= 3
        self.assertEqual(t, (1, 2, 1, 2, 1, 2))

    @make_dynamo_test
    def test_str_imul(self):
        # [FUNCTION DOC] Checks inplace multiply-assign on a string.
        s = "ab"
        s *= 3
        self.assertEqual(s, "ababab")

    # --- TypeError cases ---

    @make_dynamo_test
    def test_list_mul_str_raises(self):
        # [FUNCTION DOC] Checks list times string raises TypeError saying it cannot multiply a
        # sequence by a non-int of type str.
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'str'"
        ):
            [1, 2] * "foo"

    @make_dynamo_test
    def test_tuple_mul_str_raises(self):
        # [FUNCTION DOC] Checks tuple times string raises the same kind of TypeError.
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'str'"
        ):
            (1, 2) * "foo"

    @make_dynamo_test
    def test_str_mul_str_raises(self):
        # [FUNCTION DOC] Checks string times string raises the same kind of TypeError.
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'str'"
        ):
            "ab" * "cd"

    @make_dynamo_test
    def test_str_mul_float_raises(self):
        # [FUNCTION DOC] Checks string times float raises TypeError saying it cannot multiply a
        # sequence by a non-int of type float.
        with self.assertRaisesRegex(
            TypeError, r"can't multiply sequence by non-int of type 'float'"
        ):
            "ab" * 3.0

    @make_dynamo_test
    def test_range_mul_raises(self):
        # [FUNCTION DOC] Checks range times int raises TypeError with an unsupported operand
        # message naming range and int.
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'range' and 'int'",
        ):
            range(5) * 3

    @make_dynamo_test
    def test_int_mul_range_raises(self):
        # [FUNCTION DOC] Checks int times range raises TypeError with an unsupported operand
        # message naming int and range.
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'int' and 'range'",
        ):
            3 * range(5)

    @make_dynamo_test
    def test_dict_mul_raises(self):
        # [FUNCTION DOC] Checks dict times int raises TypeError with an unsupported operand
        # message.
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'dict' and 'int'",
        ):
            {1: 2} * 3

    @make_dynamo_test
    def test_set_mul_raises(self):
        # [FUNCTION DOC] Checks set times int raises TypeError with an unsupported operand
        # message.
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'set' and 'int'",
        ):
            {1, 2} * 3

    # --- operator.mul / operator.imul function-call form ---
    # BuiltinVariable.call_mul / call_imul are hit by the BINARY_OP /
    # INPLACE_MULTIPLY opcodes above, but also by an explicit
    # ``operator.mul(a, b)`` / ``operator.imul(a, b)`` call; verify both routes.

    @make_dynamo_test
    def test_operator_mul_int(self):
        # [FUNCTION DOC] Checks the operator module multiplication function on two ints.
        self.assertEqual(operator.mul(5, 7), 35)

    @make_dynamo_test
    def test_operator_mul_list_int(self):
        # [FUNCTION DOC] Checks the operator module multiplication function repeating a list.
        self.assertEqual(operator.mul([1, 2], 3), [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_operator_mul_int_str(self):
        # [FUNCTION DOC] Checks the operator module multiplication function with int then string
        # uses the sequence repeat fallback.
        # int * str via sq_repeat fallback in pynumber_multiply.
        self.assertEqual(operator.mul(3, "ab"), "ababab")

    @make_dynamo_test
    def test_operator_imul_int(self):
        # [FUNCTION DOC] Checks the operator module inplace multiply on two ints falls back to
        # plain multiplication.
        # int has no nb_inplace_multiply — pynumber_inplace_multiply falls
        # back to nb_multiply via binary_iop1.
        self.assertEqual(operator.imul(5, 3), 15)

    @make_dynamo_test
    def test_operator_imul_list(self):
        # [FUNCTION DOC] Checks the operator module inplace multiply on a list mutates the list in
        # place.
        # list has sq_inplace_repeat; operator.imul mutates a.
        a = [1, 2]
        operator.imul(a, 3)
        self.assertEqual(a, [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_operator_imul_tuple(self):
        # [FUNCTION DOC] Checks the operator module inplace multiply on a tuple returns a new
        # repeated tuple.
        # tuple has no sq_inplace_repeat — falls back to sq_repeat, returns
        # a new tuple.
        t = (1, 2)
        self.assertEqual(operator.imul(t, 3), (1, 2, 1, 2, 1, 2))

    @make_dynamo_test
    def test_operator_imul_str(self):
        # [FUNCTION DOC] Checks the operator module inplace multiply on a string returns the
        # repeated string.
        self.assertEqual(operator.imul("ab", 3), "ababab")

    @make_dynamo_test
    def test_operator_imul_unsupported_raises(self):
        # [FUNCTION DOC] Checks the operator module inplace multiply on a set and int raises
        # TypeError with the unsupported operand message for the inplace star form.
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*=: 'set' and 'int'",
        ):
            operator.imul({1, 2}, 3)

    # --- Direct method calls ---

    @make_dynamo_test
    def test_int_dunder_mul(self):
        # [FUNCTION DOC] Checks calling __mul__ directly on an int.
        self.assertEqual((2).__mul__(3), 6)

    @make_dynamo_test
    def test_list_dunder_mul(self):
        # [FUNCTION DOC] Checks list __mul__ with an int works and with a string raises TypeError;
        # the regex accepts both the CPython message and the sequence-repeat message used by
        # Dynamo.
        # list.__mul__ wraps sq_repeat — TypeError on non-int.  CPython's
        # slot wrapper raises "'str' object cannot be interpreted as an
        # integer"; Dynamo's slot_wrapper_mul currently delegates to
        # pysequence_repeat and raises "can't multiply sequence by non-int of
        # type 'str'". Both are TypeErrors, so use a permissive regex.
        self.assertEqual([1, 2].__mul__(3), [1, 2, 1, 2, 1, 2])
        with self.assertRaisesRegex(
            TypeError,
            r"(can't multiply sequence by non-int of type 'str'"
            r"|'str' object cannot be interpreted as an integer)",
        ):
            [1, 2].__mul__("foo")

    @make_dynamo_test
    def test_list_dunder_imul(self):
        # [FUNCTION DOC] Checks calling __imul__ directly on a list mutates it in place.
        a = [1, 2]
        a.__imul__(3)
        self.assertEqual(a, [1, 2, 1, 2, 1, 2])

    # --- User-defined __mul__ ---

    @make_dynamo_test
    def test_user_defined_mul(self):
        # [FUNCTION DOC] Checks a user-defined class multiplied with another instance.
        a = UserDefinedClassWithMul(5)
        b = UserDefinedClassWithMul(3)
        self.assertEqual(a * b, UserDefinedClassWithMul(15))

    @make_dynamo_test
    def test_user_defined_mul_with_int(self):
        # [FUNCTION DOC] Checks a user-defined instance multiplied by a plain int on the right.
        a = UserDefinedClassWithMul(5)
        self.assertEqual(a * 3, UserDefinedClassWithMul(15))

    @make_dynamo_test
    def test_user_defined_rmul(self):
        # [FUNCTION DOC] Checks a plain int on the left times a user-defined instance using the
        # reflected method.
        a = UserDefinedClassWithMul(5)
        self.assertEqual(3 * a, UserDefinedClassWithMul(15))

    @make_dynamo_test
    def test_left_mul_left_uses_mul(self):
        # [FUNCTION DOC] Checks the same-class-only helper with two instances uses the forward
        # method.
        a = LeftMulClass(5)
        b = LeftMulClass(3)
        self.assertEqual(a * b, LeftMulClass(15))

    @make_dynamo_test
    def test_left_mul_right_falls_back_to_rmul(self):
        # [FUNCTION DOC] Checks that the left helper times the right helper falls back to the
        # right reflected method and returns its formatted string.
        a = LeftMulClass(7)
        b = RightMulClass("x")
        self.assertEqual(a * b, "LeftMulClass(7)*RightMulClass(x)")

    @make_dynamo_test
    def test_right_mul_left_raises(self):
        # [FUNCTION DOC] Checks that the right helper times the left helper raises TypeError with
        # an unsupported operand message naming both classes, because neither side handles it.
        a = RightMulClass("x")
        b = LeftMulClass(7)
        with self.assertRaisesRegex(
            TypeError,
            r"unsupported operand type\(s\) for \*: 'RightMulClass' and 'LeftMulClass'",
        ):
            a * b

    # --- Subclass right-op dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Checks an int subclass with its own multiplication methods takes
        # priority: forward on the left and reflected on the right of an int.
        self.assertEqual(_IntSubWithMul(1) * 1, "_IntSubWithMul.__mul__")
        self.assertEqual(1 * _IntSubWithMul(1), "_IntSubWithMul.__rmul__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Checks baseline marker results for a plain class in both operand orders.
        self.assertEqual(_BaseWithMul() * 1, "_BaseWithMul.__mul__")
        self.assertEqual(1 * _BaseWithMul(), "_BaseWithMul.__rmul__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        # [FUNCTION DOC] Checks that a subclass overriding the methods as the right operand takes
        # priority over the base class forward method, and as left uses its own forward.
        self.assertEqual(_SubWithMul() * _BaseWithMul(), "_SubWithMul.__mul__")
        self.assertEqual(_BaseWithMul() * _SubWithMul(), "_SubWithMul.__rmul__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        # [FUNCTION DOC] Checks a subclass that inherits the methods gets no priority: the
        # reflected attribute is identical to the base one and both operand orders call the base
        # methods.
        self.assertIs(_InheritedSub.__rmul__, _BaseWithMul.__rmul__)
        self.assertEqual(_InheritedSub() * 1, "_BaseWithMul.__mul__")
        self.assertEqual(1 * _InheritedSub(), "_BaseWithMul.__rmul__")

    # --- Tensors ---

    def test_tensor_mul_scalar(self):
        # [FUNCTION DOC] Compiles a tensor times the int two with fullgraph and compares the list
        # of values.
        @torch.compile(backend="eager", fullgraph=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns the tensor argument times two.
            return t * 2

        self.assertEqual(f(torch.tensor([1.0, 2.0, 3.0])).tolist(), [2.0, 4.0, 6.0])

    def test_scalar_mul_tensor(self):
        # [FUNCTION DOC] Compiles the int two times a tensor and compares the values.
        @torch.compile(backend="eager", fullgraph=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns two times the tensor argument.
            return 2 * t

        self.assertEqual(f(torch.tensor([1.0, 2.0, 3.0])).tolist(), [2.0, 4.0, 6.0])

    def test_tensor_mul_tensor(self):
        # [FUNCTION DOC] Compiles tensor times tensor and compares values.
        @torch.compile(backend="eager", fullgraph=True)
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of the two arguments.
            return a * b

        out = f(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        self.assertEqual(out.tolist(), [3.0, 8.0])

    def test_symint_mul(self):
        # [FUNCTION DOC] Compiles a dynamic-shape function multiplying a symbolic size by three;
        # expects fifteen.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first dimension size times three.
            return x.shape[0] * 3

        self.assertEqual(f(torch.randn(5, 4)), 15)

    def test_int_mul_symint(self):
        # [FUNCTION DOC] Compiles a dynamic-shape function multiplying three by a symbolic size;
        # expects fifteen.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns three times the first dimension size.
            return 3 * x.shape[0]

        self.assertEqual(f(torch.randn(5, 4)), 15)

    def test_symint_mul_float(self):
        # [FUNCTION DOC] Checks a symbolic size times a float constant gives 7.5, so the symbolic
        # implementation accepts any Python constant, not only symbolic-like ones.
        # SymNodeVariable.nb_multiply_impl must accept any python constant, not
        # just symnode-like ones — float constants aren't is_symnode_like().
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first dimension size times 1.5.
            return x.shape[0] * 1.5

        self.assertEqual(f(torch.randn(5, 4)), 7.5)

    def test_float_mul_symint(self):
        # [FUNCTION DOC] Checks a float constant times a symbolic size gives 7.5.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns 1.5 times the first dimension size.
            return 1.5 * x.shape[0]

        self.assertEqual(f(torch.randn(5, 4)), 7.5)

    def test_symint_mul_str(self):
        # [FUNCTION DOC] Checks a symbolic size times a string repeats the string (the sequence
        # repeat fallback); expects the string repeated three times.
        # int * str works in CPython via PyNumber_Multiply's sq_repeat fallback.
        # SymNode operand must allow str on the other side just like a constant int.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first dimension size times the string ab.
            return x.shape[0] * "ab"

        self.assertEqual(f(torch.randn(3, 4)), "ababab")

    def test_symfloat_mul_int(self):
        # [FUNCTION DOC] Checks a symbolic float (from the item method of a float tensor) times an
        # int; expects 7.0.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the extracted scalar times two.
            return x.item() * 2

        self.assertEqual(f(torch.tensor(3.5)), 7.0)

    def test_int_mul_symfloat(self):
        # [FUNCTION DOC] Checks an int times a symbolic float; expects 7.0.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns two times the extracted scalar.
            return 2 * x.item()

        self.assertEqual(f(torch.tensor(3.5)), 7.0)

    def test_symfloat_mul_float(self):
        # [FUNCTION DOC] Checks a symbolic float times a float constant; expects 4.5.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the extracted scalar times 1.5.
            return x.item() * 1.5

        self.assertEqual(f(torch.tensor(3.0)), 4.5)

    def test_float_mul_symfloat(self):
        # [FUNCTION DOC] Checks a float constant times a symbolic float; expects 4.5.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns 1.5 times the extracted scalar.
            return 1.5 * x.item()

        self.assertEqual(f(torch.tensor(3.0)), 4.5)

    def test_list_mul_symnode(self):
        # [FUNCTION DOC] Checks repeating a one-element list by a symbolic int argument: two
        # elements for count three; a bool count of True gives one element.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns a list holding the first argument repeated as many
            # times as the second argument.
            return [x] * y

        self.assertEqual(f(2, 3), [2, 2, 2])
        self.assertEqual(f(2, True), [2])

    # --- Sequence * SymNode must route through sq_repeat, not nb_multiply ---

    def test_list_mul_symint_iter(self):
        # [FUNCTION DOC] Checks that a two-element list times a symbolic size routes through the
        # sequence repeat path and yields the elements repeated three times when listed.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns the list built from a two-element list repeated by
            # the first dimension size.
            return list([1, 2] * t.shape[0])

        self.assertEqual(f(torch.randn(3, 4)), [1, 2, 1, 2, 1, 2])

    def test_symint_mul_list_iter(self):
        # [FUNCTION DOC] Checks a symbolic size times a list (reversed order) gives the same
        # repeated elements.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns the list built from the first dimension size
            # repeating a two-element list.
            return list(t.shape[0] * [1, 2])

        self.assertEqual(f(torch.randn(3, 4)), [1, 2, 1, 2, 1, 2])

    def test_tuple_mul_symint_iter(self):
        # [FUNCTION DOC] Checks a tuple repeated by a symbolic size gives the repeated elements.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns the list built from a two-element tuple repeated by
            # the first dimension size.
            return list((1, 2) * t.shape[0])

        self.assertEqual(f(torch.randn(3, 4)), [1, 2, 1, 2, 1, 2])

    def test_str_mul_symint_iter(self):
        # [FUNCTION DOC] Checks a string repeated by a symbolic size; listing gives repeated
        # characters.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns the list of characters of the string ab repeated by
            # the first dimension size.
            return list("ab" * t.shape[0])

        self.assertEqual(f(torch.randn(3, 4)), ["a", "b", "a", "b", "a", "b"])

    def test_list_imul_symint_iter(self):
        # [FUNCTION DOC] Checks inplace multiply-assign of a list by a symbolic size repeats its
        # contents.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Repeats a local two-element list in place by the first
            # dimension size and returns it as a list.
            xs = [1, 2]
            xs *= t.shape[0]
            return list(xs)

        self.assertEqual(f(torch.randn(3, 4)), [1, 2, 1, 2, 1, 2])

    def test_list_mul_symint_sum(self):
        # [FUNCTION DOC] Checks that the sum of a list repeated by a symbolic size is nine,
        # exercising a different consumer of the sequence (summing rather than listing).
        # Forces iteration via sum() rather than list() to exercise a
        # different consumer of the resulting sequence VT.
        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(t):
            # [FUNCTION DOC] HELPER: Returns the sum of a two-element list repeated by the first
            # dimension size.
            return sum([1, 2] * t.shape[0])

        self.assertEqual(f(torch.randn(3, 4)), 9)

    # --- Error message accuracy (matches CPython exactly) ---

    @make_dynamo_test
    def test_error_message_list_mul_str(self):
        # [FUNCTION DOC] Checks the exact TypeError message for list times string matches CPython.
        # KNOWN ISSUE: the assertion is inside an except block only, so if no exception is raised
        # the test passes without checking anything.
        try:
            [1, 2] * "foo"
        except TypeError as e:
            self.assertEqual(str(e), "can't multiply sequence by non-int of type 'str'")

    @make_dynamo_test
    def test_error_message_range_mul(self):
        # [FUNCTION DOC] Checks the exact TypeError message for range times int matches CPython.
        # KNOWN ISSUE: the assertion is inside an except block only, so if no exception is raised
        # the test passes without checking anything.
        try:
            range(5) * 3
        except TypeError as e:
            self.assertEqual(
                str(e), "unsupported operand type(s) for *: 'range' and 'int'"
            )


instantiate_parametrized_tests(TestNbMultiply)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
