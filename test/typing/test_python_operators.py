# mypy: ignore-errors
# Owner(s): ["module: unknown"]
# [MODULE DOC] Runtime tests that keep the operator typing sample (pass/arithmetic_ops.py)
# honest. It defines tables of operator strings (comparison, float-or-int, int-only, matrix
# multiply, unary, punctuation) and sample operands (Python and tensor floats and ints). Tests
# verify the tables against the tokenizer's operator list, that the typing sample covers every
# operator with the same operand shapes, and the runtime result types of every operator and
# operand combination. Importing the module only builds the constants; the test class is
# parametrized at import time and run_tests is invoked when run as a script.
import token
from itertools import product
from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


MM = "@"

BINARY_RETURNS_BOOL = "!=", "<", "<=", "==", ">", ">="
BINARY_ACCEPTS_FLOAT_OR_INT = "%", "*", "**", "+", "-", "/", "//"
BINARY_ACCEPTS_INT_ONLY = "&", "<<", ">>", "^", "|"
BINARY_OPS = (
    *BINARY_RETURNS_BOOL,
    *BINARY_ACCEPTS_FLOAT_OR_INT,
    *BINARY_ACCEPTS_INT_ONLY,
    MM,
)

BINARY_RETURNS_FLOAT = ("/",)

UNARY_ACCEPTS_FLOAT_OR_INT = "+", "-"
UNARY_ACCEPTS_INT_ONLY = ("~",)
UNARY_OPS = *UNARY_ACCEPTS_FLOAT_OR_INT, *UNARY_ACCEPTS_INT_ONLY

PUNCTUATION = ",", ";"

OPERATORS = *UNARY_OPS, *BINARY_OPS, *PUNCTUATION

FLOATS = 1.5, torch.tensor((2.5, 3.5))
INTS = 3, torch.tensor((1, 2))
ALL = *FLOATS, *INTS

TYPE_TEST_FILE = Path(__file__).parent / "pass/arithmetic_ops.py"


class TestPythonOperators(TestCase):
    # [CLASS DOC] TestCase holding four checks: operator tables versus the tokenizer, coverage
    # of the typing sample file, and parametrized runtime result checks for binary and unary
    # operators over the combinations of the module-level sample operands. Parametrization is
    # expanded by instantiate_parametrized_tests below the class.
    # Prove that UNARY_OPS, BINARY_OPS, and OPERATORS are correct and complete
    def test_operators_are_correct_and_complete(self):
        # [FUNCTION DOC] Checks that the module-level operator tables are correct and complete.
        # Asserts every listed operator is a known exact token, then for each exact token that
        # is not punctuation tries to compile a unary and a binary expression using it,
        # collecting those that compile. The collected unary, binary and punctuation sets must
        # equal the unary, binary and punctuation tables. Fails if Python gains an operator the
        # tables lack.
        self.assertFalse(set(OPERATORS).difference(token.EXACT_TOKEN_TYPES))

        unary, binary, punctuation = {}, {}, {}

        for op in token.EXACT_TOKEN_TYPES:
            if op in PUNCTUATION:
                punctuation[op] = True
            else:
                try:
                    unary[op] = compile(f"{op}1 ; {op}a", op, "single")
                except SyntaxError:
                    pass
                try:
                    binary[op] = compile(f"2 {op} 3 ; a {op} b", op, "single")
                except SyntaxError:
                    pass

        self.assertEqual(sorted(unary), sorted(UNARY_OPS))
        self.assertEqual(sorted(binary), sorted(BINARY_OPS))
        self.assertEqual(sorted(punctuation), sorted(PUNCTUATION))

    def test_type_tests_are_complete(self):
        # [FUNCTION DOC] Checks that the typing sample file has assertions for every operator.
        # Reads the sample file line by line, keeps lines mentioning the tensor constant, and
        # parses the first argument of the assert_type call: a leading letter means a binary
        # expression, split into left operand, operator and right operand (the power operator is
        # padded with spaces first), recorded per operator; otherwise the first character is a
        # unary operator. Lines without an assert_type call are ignored. The unary list must
        # equal the unary table, the binary operators must equal the binary table, and every
        # binary operator must list exactly the same operand pairs as the others.
        binary, unary = {}, []

        with TYPE_TEST_FILE.open() as fp:
            # Looking for lines like:  assert_type(TENSOR ^ BOOL, Tensor)
            # But not:                 assert_type(BOOL ^ BINARY, Binary)
            lines = (i for i in fp if "TENSOR" in i)
            for line in lines:
                if expr := line.partition("assert_type(")[2].partition(",")[0]:
                    if expr[0].isalpha():
                        # ** formats differently from all other operators
                        a, op, b = expr.replace("**", " ** ").split()
                        binary.setdefault(op, []).append((a, b))
                    else:
                        unary.append(expr[0])

        self.assertEqual(sorted(unary), sorted(UNARY_OPS))
        self.assertEqual(sorted(binary), sorted(BINARY_OPS))
        value, *values = binary.values()
        self.assertEqual(values, [value] * len(values))

    @parametrize("a, op, b", product(ALL, BINARY_OPS, ALL))
    def test_binary(self, a, op, b):
        # [FUNCTION DOC] Parametrized over every combination of binary operator and two operands
        # from the sample operand list. Evaluates the expression and captures any raised
        # exception as the result, then checks the outcome category. Matrix multiply: a
        # TypeError unless both are tensors; a tensor result when both are the identical tensor;
        # otherwise a RuntimeError. With a tensor operand: int-only operators with any float
        # operand give a NotImplementedError (linked issue), else a tensor whose dtype is bool
        # for comparisons, non-floating for int-only operators, and floating exactly when an
        # operand is a float or the operation is true division. Without tensors: bool for
        # comparisons, TypeError for int-only operators with a float, int for those without,
        # float if any float or true division, else int. Reaching the default branch fails the
        # test.
        try:
            r = eval(f"a {op} b")
        except Exception as e:
            r = e

        any_tensor = isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
        any_float = _any_float(a, b)
        returns_float = any_float or op in BINARY_RETURNS_FLOAT

        if op == MM:
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                self.assertIsInstance(r, TypeError)
            elif a is b:
                self.assertIsInstance(r, torch.Tensor)
            else:
                self.assertIsInstance(r, RuntimeError)

        elif any_tensor:
            if op in BINARY_ACCEPTS_INT_ONLY and any_float:
                # See https://github.com/pytorch/pytorch/issues/15754
                self.assertIsInstance(r, NotImplementedError)
            else:
                self.assertIsInstance(r, torch.Tensor)

                if op in BINARY_RETURNS_BOOL:
                    self.assertEqual(r.dtype, torch.bool)
                elif op in BINARY_ACCEPTS_INT_ONLY:
                    self.assertFalse(r.dtype.is_floating_point)
                elif op in BINARY_ACCEPTS_FLOAT_OR_INT:
                    self.assertEqual(r.dtype.is_floating_point, returns_float)
                else:
                    self.assertFalse("Logic error")

        elif op in BINARY_RETURNS_BOOL:
            self.assertIsInstance(r, bool)

        elif op in BINARY_ACCEPTS_INT_ONLY:
            if any_float:
                self.assertIsInstance(r, TypeError)
            else:
                self.assertIsInstance(r, int)

        elif returns_float:
            self.assertIsInstance(r, float)

        else:
            self.assertIsInstance(r, int)

    @parametrize("op, a", product(UNARY_OPS, ALL))
    def test_unary(self, op, a):
        # [FUNCTION DOC] Parametrized over every unary operator and sample operand. Evaluates
        # the operator on the operand, capturing any exception as the result. Expected:
        # TypeError for bitwise invert on a float or float tensor; a tensor for tensor operands;
        # an int for invert on a Python int; a float for a Python float; an int otherwise.
        try:
            r = eval(f"{op} a")
        except Exception as e:
            r = e

        if op in UNARY_ACCEPTS_INT_ONLY and _any_float(a):
            self.assertIsInstance(r, TypeError)
        elif isinstance(a, torch.Tensor):
            self.assertIsInstance(r, torch.Tensor)
        elif op in UNARY_ACCEPTS_INT_ONLY:
            self.assertIsInstance(r, int)
        elif isinstance(a, float):
            self.assertIsInstance(r, float)
        else:
            self.assertIsInstance(r, int)


def _any_float(*x):
    # [FUNCTION DOC] HELPER: Takes any number of values and returns True if any is a Python
    # float or a tensor with a floating-point dtype; returns False if none is (including no
    # arguments).
    for i in x:
        if isinstance(i, float) or (
            isinstance(i, torch.Tensor) and i.dtype.is_floating_point
        ):
            return True
    return False


instantiate_parametrized_tests(TestPythonOperators)


if __name__ == "__main__":
    run_tests()
