# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the matrix multiplication operator (nb_matrix_multiply slot) and
# its inplace form. Defines module-level helper classes: a fully user-defined class that
# multiplies values, and two classes that return NotImplemented to exercise operand fallback.
# Tests cover operator module functions, dunder calls, inplace use, unsupported scalar operands,
# and compiled tensor matmul. Import has no side effects besides class definitions.

import operator

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class UserDefinedClassWithMatMul:
    # [CLASS DOC] HELPER: Helper class wrapping a value that implements matmul, inplace matmul and
    # reflected matmul by multiplying the stored value with another instance value or a raw
    # number; supports equality and repr.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
        self.value = value

    def __matmul__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance holding the product of the stored value
        # with the other instance value, or with the raw other operand.
        if isinstance(other, UserDefinedClassWithMatMul):
            return UserDefinedClassWithMatMul(self.value * other.value)
        return UserDefinedClassWithMatMul(self.value * other)

    def __imatmul__(self, other):
        # [FUNCTION DOC] HELPER: Multiplies the stored value in place by the other instance value
        # or raw operand and returns itself.
        if isinstance(other, UserDefinedClassWithMatMul):
            self.value *= other.value
        else:
            self.value *= other
        return self

    def __rmatmul__(self, other):
        # [FUNCTION DOC] HELPER: Reflected: returns a new instance holding the product of the left
        # operand (instance value or raw number) with the stored value.
        if isinstance(other, UserDefinedClassWithMatMul):
            return UserDefinedClassWithMatMul(other.value * self.value)
        return UserDefinedClassWithMatMul(other * self.value)

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is an instance of the same class and values
        # are equal.
        return (
            isinstance(other, UserDefinedClassWithMatMul) and self.value == other.value
        )

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string showing the class name and value.
        return f"UserDefinedClassWithMatMul({self.value})"


class LeftMatMulClass:
    # [CLASS DOC] HELPER: Helper class whose matmul and reflected matmul work only with another
    # instance of the same class, otherwise return NotImplemented.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
        self.value = value

    def __matmul__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance with the product when the other is the
        # same class; otherwise NotImplemented.
        if isinstance(other, LeftMatMulClass):
            return LeftMatMulClass(self.value * other.value)
        return NotImplemented

    def __rmatmul__(self, other):
        # [FUNCTION DOC] HELPER: Reflected: returns a new instance with the product when the other
        # is the same class; otherwise NotImplemented.
        if isinstance(other, LeftMatMulClass):
            return LeftMatMulClass(other.value * self.value)
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is the same class and values are equal.
        return isinstance(other, LeftMatMulClass) and self.value == other.value


class RightMatMulClass:
    # [CLASS DOC] HELPER: Helper class wrapping a value whose matmul works only with the same
    # class (product of the values); its reflected matmul accepts only the left helper class and
    # returns a formatted string, otherwise NotImplemented.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value argument in the value attribute.
        self.value = value

    def __matmul__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance holding the product of both stored values
        # when the other operand is the same class; otherwise NotImplemented.
        if isinstance(other, RightMatMulClass):
            return RightMatMulClass(self.value * other.value)
        return NotImplemented

    def __rmatmul__(self, other):
        # [FUNCTION DOC] HELPER: Reflected: when the other is the left helper class returns a
        # formatted string naming both values; otherwise NotImplemented.
        if isinstance(other, LeftMatMulClass):
            return f"LeftMatMulClass({other.value})@RightMatMulClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal when the other is the same class and values are equal.
        return isinstance(other, RightMatMulClass) and self.value == other.value


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestNbMatrixMultiply(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for matrix multiplication in Dynamo; patched with
    # enable_trace_unittest so assert methods can be traced in make_dynamo_test bodies. Uses the
    # helper classes above and compiled tensor functions.
    # --- using operator ---

    @make_dynamo_test
    def test_operator_matmul(self):
        # [FUNCTION DOC] Checks operator matmul on two helper instances gives an instance with the
        # product.
        self.assertEqual(
            operator.matmul(
                UserDefinedClassWithMatMul(2), UserDefinedClassWithMatMul(3)
            ),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_operator_imatmul(self):
        # [FUNCTION DOC] Checks operator imatmul mutates the left instance, returns that same
        # object and leaves the product in it.
        x = UserDefinedClassWithMatMul(2)
        out = operator.imatmul(x, UserDefinedClassWithMatMul(3))
        self.assertIs(out, x)
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    # --- user defined matrix multiply ---

    @make_dynamo_test
    def test_user_defined_matmul(self):
        # [FUNCTION DOC] Checks the matmul operator between two helper instances.
        self.assertEqual(
            UserDefinedClassWithMatMul(2) @ UserDefinedClassWithMatMul(3),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_user_defined_matmul_with_int(self):
        # [FUNCTION DOC] Checks the matmul operator between a helper instance and a plain int on
        # the right.
        self.assertEqual(
            UserDefinedClassWithMatMul(2) @ 3, UserDefinedClassWithMatMul(6)
        )

    @make_dynamo_test
    def test_user_defined_rmatmul_with_int(self):
        # [FUNCTION DOC] Checks the matmul operator with a plain int on the left, using the
        # reflected method.
        self.assertEqual(
            3 @ UserDefinedClassWithMatMul(2), UserDefinedClassWithMatMul(6)
        )

    # --- using left matrix multiply ---

    @make_dynamo_test
    def test_left_matmul_left_uses_matmul(self):
        # [FUNCTION DOC] Checks that the left-only helper class handles two instances of itself
        # with the forward method.
        a = LeftMatMulClass(5)
        b = LeftMatMulClass(3)
        self.assertEqual(a @ b, LeftMatMulClass(15))

    @make_dynamo_test
    def test_left_matmul_direct_dunder(self):
        # [FUNCTION DOC] Checks that calling __matmul__ directly on the left helper with an
        # instance of the other helper class returns the NotImplemented singleton.
        self.assertIs(
            LeftMatMulClass(5).__matmul__(RightMatMulClass(7)),
            NotImplemented,
        )

    @make_dynamo_test
    def test_left_matmul_right_fallback_rmatmul(self):
        # [FUNCTION DOC] Checks that the matmul operator between the left helper and the right
        # helper falls back to the right helper reflected method and returns its formatted string.
        a = LeftMatMulClass(5)
        b = RightMatMulClass(3)
        self.assertEqual(a @ b, "LeftMatMulClass(5)@RightMatMulClass(3)")

    # --- using right matrix multiply ---

    @make_dynamo_test
    def test_right_matmul_direct_dunder(self):
        # [FUNCTION DOC] Checks that calling __matmul__ directly on the right helper with an
        # instance of the left helper returns NotImplemented.
        self.assertIs(
            RightMatMulClass(5).__matmul__(LeftMatMulClass(7)),
            NotImplemented,
        )

    # --- using dunder ---

    @make_dynamo_test
    def test_user_defined_matmul_dunder(self):
        # [FUNCTION DOC] Checks calling __matmul__ directly on a user-defined instance with
        # another instance.
        self.assertEqual(
            UserDefinedClassWithMatMul(2).__matmul__(UserDefinedClassWithMatMul(3)),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_user_defined_rmatmul_dunder(self):
        # [FUNCTION DOC] Checks calling __rmatmul__ directly with an int on the full helper.
        self.assertEqual(
            UserDefinedClassWithMatMul(2).__rmatmul__(3),
            UserDefinedClassWithMatMul(6),
        )

    @make_dynamo_test
    def test_user_defined_imatmul_dunder(self):
        # [FUNCTION DOC] Checks calling __imatmul__ directly mutates and returns the same
        # instance.
        x = UserDefinedClassWithMatMul(2)
        out = x.__imatmul__(UserDefinedClassWithMatMul(3))
        self.assertIs(out, x)
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    # --- Inplace matrix multiply ---

    @make_dynamo_test
    def test_user_defined_imatmul(self):
        # [FUNCTION DOC] Checks the inplace matmul-assign operator with a helper instance mutates
        # the value.
        x = UserDefinedClassWithMatMul(2)
        x @= UserDefinedClassWithMatMul(3)
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    @make_dynamo_test
    def test_user_defined_imatmul_with_int(self):
        # [FUNCTION DOC] Checks the inplace matmul-assign operator with a plain int.
        x = UserDefinedClassWithMatMul(2)
        x @= 3
        self.assertEqual(x, UserDefinedClassWithMatMul(6))

    # --- unsupported operations ---

    @make_dynamo_test
    def test_scalar_matmul_scalar(self):
        # [FUNCTION DOC] Checks the matmul operator between two ints raises TypeError with an
        # unsupported operand type message.
        with self.assertRaisesRegex(TypeError, r"unsupported operand type"):
            1 @ 2

    @make_dynamo_test
    def test_scalar_matmul_scalar_operator(self):
        # [FUNCTION DOC] Checks operator matmul on two floats raises TypeError with an unsupported
        # operand type message.
        with self.assertRaisesRegex(TypeError, r"unsupported operand type"):
            operator.matmul(1.0, 2.0)

    # --- using torch.compile ---

    def test_compile_matmul_tensor(self):
        # [FUNCTION DOC] Compiles the matmul operator over two 4 by 4 float tensors (eager
        # backend, fullgraph) and compares to eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two arguments.
            return x @ y

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_compile_matmul_tensor2(self):
        # [FUNCTION DOC] Compiles an inplace matmul-assign on a tensor; checks the compiled result
        # equals the expected product from a clone and that the original input tensor is unchanged
        # afterwards (not mutated in place by the compiled function), as asserted by the test.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies inplace matmul-assign of the second argument to the
            # first and returns the first.
            x @= y
            return x

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        x_clone = x.clone()
        expected = x_clone @ y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), expected)
        self.assertEqual(x, x_clone)

    def test_compile_matmul_dunder(self):
        # [FUNCTION DOC] Compiles a direct call of the tensor __matmul__ method on a 4 by 3 and a
        # 3 by 5 tensor and compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the result of calling __matmul__ on the first
            # argument.
            return x.__matmul__(y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 5)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_compile_int_tensor_matmul(self):
        # [FUNCTION DOC] Compiles the matmul operator over two int64 matrices and compares with
        # eager.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two arguments.
            return a @ b

        a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(a, b), fn(a, b))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
