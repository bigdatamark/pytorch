# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Tests for the SymPy-based utilities under torch.utils._sympy and related inductor
# analyses: int infinity numbers, the C++ printer for min/max, value range analysis checked
# against the reference analysis, sympy_interp over several analyses (sympy, python/fx, tensor),
# the relational solver try_solve, custom sympy functions, SingletonInt, Identity, TypedExpr and
# C code printing. Defines shared operator-name lists and constant pools at import, and
# instantiates the parametrized test classes at the bottom of the file.

import functools
import itertools
import math
import pickle
import sys
from collections.abc import Callable

import sympy

import torch
import torch.fx as fx
from sympy.core.relational import is_ge, is_gt, is_le, is_lt
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_Z3,
    TestCase,
)
from torch.utils._sympy.functions import (
    FloorDiv,
    Identity,
    Max as TorchSymMax,
    Min as TorchSymMin,
    LShift,
    Mod,
    ModularIndexing,
    OpaqueUnaryFn_cos,
    PythonMod,
    RShift,
    BitwiseFn_bitwise_and,
    simple_floordiv_gcd,
)
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.numbers import int_oo, IntInfinity, NegativeIntInfinity
from torch.utils._sympy.printers import CppPrinter
from torch.utils._sympy.reference import (
    PythonReferenceAnalysis,
    ReferenceAnalysis,
    TensorReferenceAnalysis,
)
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.solve import INEQUALITY_TYPES, mirror_rel_op, try_solve
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from torch._inductor.bounds import ValueRangeAnalysis
from torch._inductor.index_propagation import TypedExpr


UNARY_OPS = [
    "reciprocal",
    "square",
    "abs",
    "neg",
    "exp",
    "log",
    "sqrt",
    "floor",
    "ceil",
]
BINARY_OPS = [
    "truediv",
    "floordiv",
    # "truncdiv",  # TODO
    # NB: pow is float_pow
    "add",
    "mul",
    "sub",
    "pow",
    "pow_by_natural",
    "minimum",
    "maximum",
    "mod",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
]
BITWISE_OPS = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
]

UNARY_BOOL_OPS = ["not_"]
BINARY_BOOL_OPS = ["or_", "and_"]
COMPARE_OPS = ["eq", "ne", "lt", "gt", "le", "ge"]

# a mix of constants, powers of two, primes
CONSTANTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    8,
    16,
    32,
    64,
    100,
    101,
    2**24,
    2**32,
    2**37 - 1,
    sys.maxsize - 1,
    sys.maxsize,
]
# less constants for N^2 situations
LESS_CONSTANTS = [-1, 0, 1, 2, 100]
# SymPy relational types.
RELATIONAL_TYPES = [sympy.Eq, sympy.Ne, sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le]


def valid_unary(fn, v):
    # [FUNCTION DOC] HELPER: Returns whether the named unary op is defined for the numeric
    # value v: false for log of non-positive, reciprocal of zero, and sqrt of negative
    # values; true otherwise.
    if fn == "log" and v <= 0:
        return False
    elif fn == "reciprocal" and v == 0:
        return False
    elif fn == "sqrt" and v < 0:
        return False
    return True


def valid_binary(fn, a, b):
    # [FUNCTION DOC] HELPER: Returns whether the named binary op is defined and cheap for
    # operands a and b. Rejects pow with exponent above 4, non-positive base or zero to
    # the zero; the natural power with exponent above 4, negative exponent or zero to the
    # zero; mod with negative a or non-positive b; and division-like ops with a zero
    # divisor. Otherwise true.
    if fn == "pow" and (
        # sympy will expand to x*x*... for integral b; don't do it if it's big
        b > 4
        # no imaginary numbers
        or a <= 0
        # 0**0 is undefined
        or (a == b == 0)
    ):
        return False
    elif fn == "pow_by_natural" and (
        # sympy will expand to x*x*... for integral b; don't do it if it's big
        b > 4
        or b < 0
        or (a == b == 0)
    ):
        return False
    elif fn == "mod" and (a < 0 or b <= 0):
        return False
    elif (fn in ["div", "truediv", "floordiv"]) and b == 0:
        return False
    return True


def generate_range(vals):
    # [FUNCTION DOC] HELPER: Generator yielding a ValueRanges for every ordered pair of values
    # taken from vals (cartesian product). For boolean values skips only the pair true then
    # false; for numbers skips pairs where the lower bound exceeds the upper. Also skips ranges
    # whose lower bound is positive infinity or whose upper bound is negative infinity.
    for a1, a2 in itertools.product(vals, repeat=2):
        if a1 in [sympy.true, sympy.false]:
            if a1 == sympy.true and a2 == sympy.false:
                continue
        else:
            if a1 > a2:
                continue
        # ranges that only admit infinite values are not interesting
        if a1 == sympy.oo or a2 == -sympy.oo:
            continue
        yield ValueRanges(a1, a2)


class TestNumbers(TestCase):
    # [CLASS DOC] Tests of the integer infinity singletons (positive and negative) arithmetic,
    # comparisons and interaction with the ordinary sympy infinity.
    def test_int_infinity(self):
        # [FUNCTION DOC] Exhaustively checks that int infinity has the right types, stays a
        # singleton under addition, subtraction, multiplication, absolute value and powers (a
        # negative power gives zero), and compares equal or ordered correctly against itself,
        # plain integers and its negation.
        self.assertIsInstance(int_oo, IntInfinity)
        self.assertIsInstance(-int_oo, NegativeIntInfinity)
        self.assertTrue(int_oo.is_integer)
        # is tests here are for singleton-ness, don't use it for comparisons
        # against numbers
        self.assertIs(int_oo + int_oo, int_oo)
        self.assertIs(int_oo + 1, int_oo)
        self.assertIs(int_oo - 1, int_oo)
        self.assertIs(-int_oo - 1, -int_oo)
        self.assertIs(-int_oo + 1, -int_oo)
        self.assertIs(-int_oo + (-int_oo), -int_oo)
        self.assertIs(-int_oo - int_oo, -int_oo)
        self.assertIs(1 + int_oo, int_oo)
        self.assertIs(1 - int_oo, -int_oo)
        self.assertIs(int_oo * int_oo, int_oo)
        self.assertIs(2 * int_oo, int_oo)
        self.assertIs(int_oo * 2, int_oo)
        self.assertIs(-1 * int_oo, -int_oo)
        self.assertIs(-int_oo * int_oo, -int_oo)
        self.assertIs(2 * -int_oo, -int_oo)
        self.assertIs(-int_oo * 2, -int_oo)
        self.assertIs(-1 * -int_oo, int_oo)
        self.assertIs(int_oo / 2, sympy.oo)
        self.assertIs(-(-int_oo), int_oo)  # noqa: B002
        self.assertIs(abs(int_oo), int_oo)
        self.assertIs(abs(-int_oo), int_oo)
        self.assertIs(int_oo**2, int_oo)
        self.assertIs((-int_oo) ** 2, int_oo)
        self.assertIs((-int_oo) ** 3, -int_oo)
        self.assertEqual(int_oo**-1, 0)
        self.assertEqual((-int_oo) ** -1, 0)
        self.assertIs(int_oo**int_oo, int_oo)
        self.assertTrue(int_oo == int_oo)
        self.assertFalse(int_oo != int_oo)
        self.assertTrue(-int_oo == -int_oo)
        self.assertFalse(int_oo == 2)
        self.assertTrue(int_oo != 2)
        self.assertFalse(int_oo == sys.maxsize)
        self.assertTrue(int_oo >= sys.maxsize)
        self.assertTrue(int_oo >= 2)
        self.assertTrue(int_oo >= -int_oo)

    def test_relation(self):
        # [FUNCTION DOC] Checks adding int infinity to a number gives int infinity and negative
        # int infinity is not greater than 2.
        self.assertIs(sympy.Add(2, int_oo), int_oo)
        self.assertFalse(-int_oo > 2)

    def test_lt_self(self):
        # [FUNCTION DOC] Checks int infinity is not less than itself and min over negative int
        # infinity returns the negative infinity singleton.
        self.assertFalse(int_oo < int_oo)
        self.assertIs(min(-int_oo, -4), -int_oo)
        self.assertIs(min(-int_oo, -int_oo), -int_oo)

    def test_float_cast(self):
        # [FUNCTION DOC] Checks converting positive and negative int infinity to float gives the
        # corresponding float infinities.
        self.assertEqual(float(int_oo), math.inf)
        self.assertEqual(float(-int_oo), -math.inf)

    def test_mixed_oo_int_oo(self):
        # [FUNCTION DOC] Checks the (arbitrary) ordering convention between int infinity and
        # sympy infinity: int infinity is smaller than sympy infinity and negative int infinity
        # is larger than negative sympy infinity, and max/min follow that order.
        # Arbitrary choice
        self.assertTrue(int_oo < sympy.oo)
        self.assertFalse(int_oo > sympy.oo)
        self.assertTrue(sympy.oo > int_oo)
        self.assertFalse(sympy.oo < int_oo)
        self.assertIs(max(int_oo, sympy.oo), sympy.oo)
        self.assertTrue(-int_oo > -sympy.oo)
        self.assertIs(min(-int_oo, -sympy.oo), -sympy.oo)


class TestCppPrinter(TestCase):
    # [CLASS DOC] Tests of the C++ printer for min and max expressions.
    def test_min_max_preserve_float_expressions(self):
        # [FUNCTION DOC] Prints minimum and maximum of float expressions containing a symbol and
        # checks the output uses the standard min or max with a cast to double and no cast to a
        # 64-bit integer.
        s0 = sympy.Symbol("s0")
        expr = TorchSymMin(
            sympy.Float(4.0), sympy.Float(0.5) + sympy.Float(1.5) * s0
        )
        printed = CppPrinter().doprint(expr)

        self.assertIn("std::min", printed)
        self.assertIn("static_cast<double>", printed)
        self.assertNotIn("static_cast<int64_t>", printed)

        expr = TorchSymMax(
            sympy.Float(4.0), sympy.Float(0.5) + sympy.Float(1.5) * s0
        )
        printed = CppPrinter().doprint(expr)

        self.assertIn("std::max", printed)
        self.assertIn("static_cast<double>", printed)
        self.assertNotIn("static_cast<int64_t>", printed)

    def test_min_max_preserve_integer_expressions(self):
        # [FUNCTION DOC] Prints minimum and maximum of an integer constant and an integer symbol
        # and checks the output uses the standard min or max with a cast to a 64-bit integer.
        s0 = sympy.Symbol("s0", integer=True)
        printed = CppPrinter().doprint(TorchSymMin(sympy.Integer(4), s0))
        self.assertIn("std::min", printed)
        self.assertIn("static_cast<int64_t>", printed)

        printed = CppPrinter().doprint(TorchSymMax(sympy.Integer(4), s0))
        self.assertIn("std::max", printed)
        self.assertIn("static_cast<int64_t>", printed)


class TestValueRanges(TestCase):
    # [CLASS DOC] Tests of value range analysis (ValueRangeAnalysis) and bound_sympy, checking
    # the range results against the reference analysis on concrete values. Parametrized over
    # operator names; instantiated by the parametrized-tests helper at the end of the file.
    @parametrize("fn", UNARY_OPS)
    @parametrize("dtype", ("int", "float"))
    def test_unary_ref(self, fn, dtype):
        # [FUNCTION DOC] For every valid constant and for integer and float sympy types, checks
        # the unary op on a single-point range gives a single-point result that matches the
        # reference analysis value, including integer-ness of the bounds.
        dtype = {"int": sympy.Integer, "float": sympy.Float}[dtype]
        for v in CONSTANTS:
            if not valid_unary(fn, v):
                continue
            with self.subTest(v=v):
                v = dtype(v)
                ref_r = getattr(ReferenceAnalysis, fn)(v)
                r = getattr(ValueRangeAnalysis, fn)(v)
                self.assertEqual(r.lower.is_integer, r.upper.is_integer)
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r.is_integer, r.upper.is_integer)
                self.assertEqual(ref_r, r.lower)

    def test_pow_half(self):
        # [FUNCTION DOC] Smoke test: range analysis power with an unknown base and exponent one
        # half must not raise. Nothing about the result is asserted.
        ValueRangeAnalysis.pow(ValueRanges.unknown(), ValueRanges.wrap(0.5))

    def test_bound_sympy_negative_pow(self):
        # [FUNCTION DOC] Checks that bounding a reciprocal of a positive integer symbol
        # with range 3 to 4 gives the unknown range, since the integer reciprocal range is
        # not representable.
        x = sympy.Symbol("x", integer=True, positive=True)

        self.assertEqual(
            bound_sympy(x**-1, {x: ValueRanges(3, 4)}), ValueRanges.unknown()
        )

    @parametrize("fn", BINARY_OPS)
    @parametrize("dtype", ("int", "float"))
    def test_binary_ref(self, fn, dtype):
        # [FUNCTION DOC] For each pair of valid constants and integer or float types, checks the
        # binary op on single-point ranges either returns unknown or equals the reference result
        # with matching integer-ness. Returns early (no test) for bitwise ops, and for float
        # with natural power or mod.
        to_dtype = {"int": sympy.Integer, "float": sympy.Float}
        # Don't test bitwise methods since value range analysis on a singleton
        # range may not return a singleton result.
        if fn in BITWISE_OPS:
            return
        # Don't test float on int only methods
        if dtype == "float" and fn in ["pow_by_natural", "mod"]:
            return
        dtype = to_dtype[dtype]
        for a, b in itertools.product(CONSTANTS, repeat=2):
            if not valid_binary(fn, a, b):
                continue
            a = dtype(a)
            b = dtype(b)
            with self.subTest(a=a, b=b):
                r = getattr(ValueRangeAnalysis, fn)(a, b)
                if r == ValueRanges.unknown():
                    continue
                ref_r = getattr(ReferenceAnalysis, fn)(a, b)

                self.assertEqual(r.lower.is_integer, r.upper.is_integer)
                self.assertEqual(ref_r.is_integer, r.upper.is_integer)
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r, r.lower)

    def test_mul_zero_unknown(self):
        # [FUNCTION DOC] Checks that multiplying the range of zero (integer or float) by the
        # unknown range gives the zero range.
        self.assertEqual(
            ValueRangeAnalysis.mul(ValueRanges.wrap(0), ValueRanges.unknown()),
            ValueRanges.wrap(0),
        )
        self.assertEqual(
            ValueRangeAnalysis.mul(ValueRanges.wrap(0.0), ValueRanges.unknown()),
            ValueRanges.wrap(0.0),
        )

    def test_trunc_infinite(self):
        # [FUNCTION DOC] Checks truncation of ranges wrapping positive or negative float
        # infinity returns the same infinite range.
        self.assertEqual(
            ValueRangeAnalysis.trunc(ValueRanges.wrap(sympy.Float('inf'))),
            ValueRanges.wrap(sympy.Float('inf')),
        )
        self.assertEqual(
            ValueRangeAnalysis.trunc(ValueRanges.wrap(sympy.Float('-inf'))),
            ValueRanges.wrap(sympy.Float('-inf')),
        )

    @parametrize("fn", UNARY_BOOL_OPS)
    def test_unary_bool_ref_range(self, fn):
        # [FUNCTION DOC] For every boolean range, checks the boolean unary op range contains the
        # reference result of each member value, and has one distinct result when the range is a
        # single point or two distinct results otherwise.
        vals = [sympy.false, sympy.true]
        for a in generate_range(vals):
            with self.subTest(a=a):
                ref_r = getattr(ValueRangeAnalysis, fn)(a)
                unique = set()
                for a0 in vals:
                    if a0 not in a:
                        continue
                    with self.subTest(a0=a0):
                        r = getattr(ReferenceAnalysis, fn)(a0)
                        self.assertIn(r, ref_r)
                        unique.add(r)
                if ref_r.lower == ref_r.upper:
                    self.assertEqual(len(unique), 1)
                else:
                    self.assertEqual(len(unique), 2)

    @parametrize("fn", BINARY_BOOL_OPS + BITWISE_OPS)
    def test_binary_bool_ref_range(self, fn):
        # [FUNCTION DOC] Same as the unary boolean check but over all pairs of boolean ranges
        # for the binary boolean and bitwise ops: each reference result lies in the range result
        # and the number of distinct results is one for a point range or two otherwise.
        vals = [sympy.false, sympy.true]
        for a, b in itertools.product(generate_range(vals), repeat=2):
            with self.subTest(a=a, b=b):
                ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                unique = set()
                for a0, b0 in itertools.product(vals, repeat=2):
                    if a0 not in a or b0 not in b:
                        continue
                    with self.subTest(a0=a0, b0=b0):
                        r = getattr(ReferenceAnalysis, fn)(a0, b0)
                        self.assertIn(r, ref_r)
                        unique.add(r)
                if ref_r.lower == ref_r.upper:
                    self.assertEqual(len(unique), 1)
                else:
                    self.assertEqual(len(unique), 2)

    def test_bool_minimum_maximum(self):
        # [FUNCTION DOC] For minimum and maximum over all pairs of boolean ranges, checks the
        # result range is boolean, contains the logical and (for minimum) or or (for maximum) of
        # each member pair, and has the right count of distinct results.
        vals = [sympy.false, sympy.true]
        for fn in ("minimum", "maximum"):
            for a, b in itertools.product(generate_range(vals), repeat=2):
                with self.subTest(fn=fn, a=a, b=b):
                    ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                    self.assertTrue(ref_r.is_bool)
                    unique = set()
                    for a0, b0 in itertools.product(vals, repeat=2):
                        if a0 not in a or b0 not in b:
                            continue
                        if fn == "minimum":
                            expected = bool(a0) and bool(b0)
                        else:
                            expected = bool(a0) or bool(b0)
                        r = sympy.true if expected else sympy.false
                        self.assertIn(r, ref_r)
                        unique.add(r)
                    if ref_r.lower == ref_r.upper:
                        self.assertEqual(len(unique), 1)
                    else:
                        self.assertEqual(len(unique), 2)

    def test_bool_minimum_maximum_mixed_raises(self):
        # [FUNCTION DOC] Checks minimum and maximum of a boolean range mixed with an integer
        # range (in either order) raise an AssertionError saying operands must both be boolean.
        # min/max require both operands to be boolean or both non-boolean;
        # a mixed boolean/non-boolean pair must raise.
        bool_range = ValueRanges(sympy.false, sympy.true)
        int_range = ValueRanges(sympy.Integer(0), sympy.Integer(4))
        for fn in ("minimum", "maximum"):
            for a, b in ((bool_range, int_range), (int_range, bool_range)):
                with self.subTest(fn=fn, a=a, b=b):
                    with self.assertRaisesRegex(
                        AssertionError, "operands must both be boolean"
                    ):
                        getattr(ValueRangeAnalysis, fn)(a, b)

    def test_eq_ne_bool_ranges(self):
        # [FUNCTION DOC] Regression test that eq and ne accept boolean ranges without raising:
        # results are boolean ranges with boolean bounds, and for single-point true/false ranges
        # the results are the exact expected boolean ranges.
        # eq/ne must accept boolean ranges. sympy booleans do not support
        # ordered comparison (bool >/< raises), which the disjoint-range test in
        # eq would otherwise hit. Regression test for eq/ne raising TypeError on
        # a boolean range, which test_binary_bool_ref_range does not cover (it
        # omits COMPARE_OPS).
        full = ValueRanges(sympy.false, sympy.true)
        T = ValueRanges.wrap(sympy.true)
        F = ValueRanges.wrap(sympy.false)
        for a, b in [(full, full), (full, T), (F, full), (T, F), (T, T), (F, F)]:
            with self.subTest(a=a, b=b):
                for fn in ("eq", "ne"):
                    r = getattr(ValueRangeAnalysis, fn)(a, b)
                    self.assertTrue(r.is_bool)
                    self.assertIn(r.lower, (sympy.true, sympy.false))
                    self.assertIn(r.upper, (sympy.true, sympy.false))
        # singletons resolve precisely: equal -> definitely eq, disjoint -> not
        self.assertEqual(ValueRangeAnalysis.eq(T, T), T)
        self.assertEqual(ValueRangeAnalysis.ne(T, T), F)
        self.assertEqual(ValueRangeAnalysis.eq(F, F), T)
        self.assertEqual(ValueRangeAnalysis.ne(F, F), F)
        self.assertEqual(ValueRangeAnalysis.eq(T, F), F)
        self.assertEqual(ValueRangeAnalysis.ne(T, F), T)
        self.assertEqual(ValueRangeAnalysis.eq(F, T), F)
        self.assertEqual(ValueRangeAnalysis.ne(F, T), T)

    @parametrize("fn", UNARY_OPS)
    def test_unary_ref_range(self, fn):
        # [FUNCTION DOC] For every range built from the constants, checks that the reference
        # unary result of each valid contained constant lies inside the range analysis result.
        # TODO: bring back sympy.oo testing for float unary fns
        vals = CONSTANTS
        for a in generate_range(vals):
            with self.subTest(a=a):
                ref_r = getattr(ValueRangeAnalysis, fn)(a)
                for a0 in CONSTANTS:
                    if a0 not in a:
                        continue
                    if not valid_unary(fn, a0):
                        continue
                    with self.subTest(a0=a0):
                        r = getattr(ReferenceAnalysis, fn)(sympy.Integer(a0))
                        self.assertIn(r, ref_r)

    # This takes about 4s for all the variants
    @parametrize("fn", BINARY_OPS + COMPARE_OPS)
    def test_binary_ref_range(self, fn):
        # [FUNCTION DOC] Over small ranges built from the reduced constant list, checks that for
        # each valid contained pair the finite reference binary or comparison result lies inside
        # the range result. Skips power with large finite exponent ranges.
        # TODO: bring back sympy.oo testing for float unary fns
        vals = LESS_CONSTANTS
        for a, b in itertools.product(generate_range(vals), repeat=2):
            # don't attempt pow on exponents that are too large (but oo is OK)
            if fn == "pow" and b.upper > 4 and b.upper != sympy.oo:
                continue
            with self.subTest(a=a, b=b):
                for a0, b0 in itertools.product(LESS_CONSTANTS, repeat=2):
                    if a0 not in a or b0 not in b:
                        continue
                    if not valid_binary(fn, a0, b0):
                        continue
                    with self.subTest(a0=a0, b0=b0):
                        ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                        r = getattr(ReferenceAnalysis, fn)(
                            sympy.Integer(a0), sympy.Integer(b0)
                        )
                        if r.is_finite:
                            self.assertIn(r, ref_r)

    # stronger test specially for bitwise ops
    @parametrize("fn", BITWISE_OPS)
    def test_bitwise_ref_range(self, fn):
        # [FUNCTION DOC] Checks bitwise op range results contain the reference results for all
        # integer pairs drawn from the range of minus four to four, then checks the same for
        # mixed integer and boolean arguments.
        # N^4 complexity
        vals = range(-4, 5)
        for a, b in itertools.product(generate_range(vals), repeat=2):
            with self.subTest(a=a, b=b):
                for a0, b0 in itertools.product(vals, repeat=2):
                    if a0 not in a or b0 not in b:
                        continue
                    with self.subTest(a0=a0, b0=b0):
                        ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                        r = getattr(ReferenceAnalysis, fn)(a0, b0)
                        self.assertIn(r, ref_r)

        # test that bitwise ops can take bool arguments
        bool_vals = [
            (3, sympy.true),
            (3, sympy.false),
            (sympy.true, 3),
            (sympy.false, 3),
            (sympy.true, sympy.true),
            (sympy.true, sympy.false),
            (sympy.false, sympy.true),
            (sympy.false, sympy.false),
        ]
        for a, b in bool_vals:
            with self.subTest(a=a, b=b):
                ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                r = getattr(ReferenceAnalysis, fn)(a, b)
                self.assertIn(r, ref_r)


    def test_python_mod(self):
        # [FUNCTION DOC] Checks range analysis of python mod against hand-derived bounds for
        # positive, negative, mixed and infinite divisor ranges and point divisors of one and
        # minus one. Then, for many integer x and nonzero y, checks the true result lies in the
        # computed range.
        """Test python_mod value range analysis."""
        # Test with positive divisor
        # x % 4 should be in [0, 3]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(4, 4))
        self.assertEqual(r, ValueRanges(0, 3))

        # Test with range of positive divisors
        # x % [2, 10] should be in [0, 9]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(2, 10))
        self.assertEqual(r, ValueRanges(0, 9))

        # Test with negative divisor
        # x % -4 should be in [-3, 0]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(-4, -4))
        self.assertEqual(r, ValueRanges(-3, 0))

        # Test with range of negative divisors
        # x % [-10, -2] should be in [-9, 0]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(-10, -2))
        self.assertEqual(r, ValueRanges(-9, 0))

        # Test with mixed divisor range (assuming y != 0)
        # x % [-5, 5] should be in [-4, 4]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(-5, 5))
        self.assertEqual(r, ValueRanges(-4, 4))

        # Test with infinite bounds
        # x % [1, int_oo] should be in [0, int_oo - 1] = [0, int_oo]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(1, int_oo))
        self.assertEqual(r.lower, 0)
        self.assertEqual(r.upper, int_oo)  # int_oo - 1 == int_oo

        # x % [-int_oo, -1] should be in [-int_oo + 1, 0] = [-int_oo, 0]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(-int_oo, -1))
        self.assertEqual(r.lower, -int_oo)  # -int_oo + 1 == -int_oo
        self.assertEqual(r.upper, 0)

        # x % [-int_oo, int_oo] should be in [-int_oo + 1, int_oo - 1] = [-int_oo, int_oo]
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(-int_oo, int_oo))
        self.assertEqual(r.lower, -int_oo)  # -int_oo + 1 == -int_oo
        self.assertEqual(r.upper, int_oo)  # int_oo - 1 == int_oo

        # x % [0, int_oo] should be in [0, int_oo - 1] = [0, int_oo]
        # (assuming y != 0 at runtime, lower bound should be 0, not 1)
        r = ValueRangeAnalysis.python_mod(ValueRanges(-int_oo, int_oo), ValueRanges(0, int_oo))
        self.assertEqual(r.lower, 0)
        self.assertEqual(r.upper, int_oo)  # int_oo - 1 == int_oo

        # Test edge case: y = [1, 1] -> result is always 0
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(1, 1))
        self.assertEqual(r, ValueRanges(0, 0))

        # Test edge case: y = [-1, -1] -> result is always 0
        r = ValueRangeAnalysis.python_mod(ValueRanges(-100, 100), ValueRanges(-1, -1))
        self.assertEqual(r, ValueRanges(0, 0))

        # Test that actual values fall within computed range
        for x in range(-20, 21):
            for y in [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:
                result = x % y
                y_range = ValueRanges(y, y)
                r = ValueRangeAnalysis.python_mod(ValueRanges(x, x), y_range)
                self.assertIn(result, r, lambda msg: f"{msg}\nx={x}, y={y}, result={result}, range={r}")

    def test_bound_sympy_mod_subtraction(self):
        # [FUNCTION DOC] For each of the three mod function flavors, checks bound_sympy of
        # expressions that subtract a mod from its base (including scaled, offset and symbolic-
        # divisor variants) yields the expected non-negative or non-positive infinite range.
        s0 = sympy.Symbol("s0", integer=True)
        s1 = sympy.Symbol("s1", integer=True)
        ranges = {s0: ValueRanges(2, int_oo), s1: ValueRanges(1, int_oo)}

        for mod in (Mod, PythonMod, sympy.Mod):
            with self.subTest(mod=mod.__name__):
                self.assertEqual(
                    bound_sympy(s0 - mod(s0, 8), ranges), ValueRanges(0, int_oo)
                )
                self.assertEqual(
                    bound_sympy(1 + s0 - mod(s0, 8), ranges),
                    ValueRanges(1, int_oo),
                )
                self.assertEqual(
                    bound_sympy(2 * s0 - 2 * mod(s0, 8), ranges),
                    ValueRanges(0, int_oo),
                )
                self.assertEqual(
                    bound_sympy(s0 - mod(s0, s1), ranges), ValueRanges(0, int_oo)
                )
                self.assertEqual(
                    bound_sympy(
                        (2 * s0 + 1) - mod(2 * s0 + 1, 8),
                        ranges,
                    ),
                    ValueRanges(0, int_oo),
                )
                self.assertEqual(
                    bound_sympy(mod(s0, 8) - s0, ranges), ValueRanges(-int_oo, 0)
                )

    def test_bound_sympy_python_mod_subtraction_negative_base(self):
        # [FUNCTION DOC] Checks bounds of subtracting a python mod from a base whose range
        # includes negative values: the lower bounds are the base lower bound minus the divisor,
        # and the upper bound is infinite.
        s0 = sympy.Symbol("s0", integer=True)
        ranges = {s0: ValueRanges(-2, int_oo)}

        self.assertEqual(
            bound_sympy(s0 - PythonMod(s0, 8), ranges), ValueRanges(-8, int_oo)
        )
        self.assertEqual(
            bound_sympy(1 + s0 - PythonMod(s0, 8), ranges),
            ValueRanges(-7, int_oo),
        )


class TestSympyInterp(TestCase):
    # [CLASS DOC] Tests of sympy_interp evaluating expressions with different analysis backends
    # (reference sympy, python with fx tracing, tensors). Parametrized over operator names.
    @parametrize(
        "fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS
    )
    def test_interp(self, fn):
        # [FUNCTION DOC] For each op (skipping truncating ones) and each valid argument
        # combination, checks that applying the reference analysis to concrete values
        # equals interpreting the symbolic expression with the values substituted for the
        # dummy symbols.
        # SymPy does not implement truncation for Expressions
        if fn in ("div", "truncdiv", "minimum", "maximum", "mod"):
            return

        is_integer = None
        if fn == "pow_by_natural":
            is_integer = True

        x = sympy.Dummy("x", integer=is_integer)
        y = sympy.Dummy("y", integer=is_integer)

        vals = CONSTANTS
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]
        elif fn in BITWISE_OPS:
            vals = vals + [True, False]
        arity = 1
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2
        symbols = [x]
        if arity == 2:
            symbols = [x, y]
        for args in itertools.product(vals, repeat=arity):
            if arity == 1 and not valid_unary(fn, *args):
                continue
            elif arity == 2 and not valid_binary(fn, *args):
                continue
            with self.subTest(args=args):
                sargs = [sympy.sympify(a) for a in args]
                sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)
                ref_r = getattr(ReferenceAnalysis, fn)(*sargs)
                # Yes, I know this is a long-winded way of saying xreplace; the
                # point is to test sympy_interp
                r = sympy_interp(
                    ReferenceAnalysis, dict(zip(symbols, sargs)), sympy_expr
                )
                self.assertEqual(ref_r, r)

    @parametrize(
        "fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS
    )
    def test_python_interp_fx(self, fn):
        # [FUNCTION DOC] For each op (skipping log, exp, truncdiv and mod) and each valid
        # argument set, builds a symbolic expression, interprets it under the python analysis
        # inside an fx symbolic trace, and checks the traced graph module result equals direct
        # interpretation with the values.
        # These never show up from symbolic_shapes
        if fn in ("log", "exp"):
            return

        # Sympy does not support truncation on symbolic shapes
        if fn in ("truncdiv", "mod"):
            return

        vals = CONSTANTS
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]
        elif fn in BITWISE_OPS:
            vals = vals + [True, False]

        arity = 1
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2

        is_integer = None
        if fn == "pow_by_natural":
            is_integer = True

        x = sympy.Dummy("x", integer=is_integer)
        y = sympy.Dummy("y", integer=is_integer)

        symbols = [x]
        if arity == 2:
            symbols = [x, y]

        for args in itertools.product(vals, repeat=arity):
            if arity == 1 and not valid_unary(fn, *args):
                continue
            elif arity == 2 and not valid_binary(fn, *args):
                continue
            if fn == "truncdiv" and args[1] == 0:
                continue
            elif fn in ("pow", "pow_by_natural") and (args[0] == 0 and args[1] <= 0):
                continue
            elif fn == "floordiv" and args[1] == 0:
                continue
            with self.subTest(args=args):
                # Workaround mpf from symbol error
                if fn == "minimum":
                    sympy_expr = sympy.Min(x, y)
                elif fn == "maximum":
                    sympy_expr = sympy.Max(x, y)
                else:
                    sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)

                if arity == 1:

                    def trace_f(px):
                        # [FUNCTION DOC] HELPER: One-argument function that interprets the
                        # expression under the python analysis with the symbol bound to the
                        # argument. Traced by fx.
                        return sympy_interp(
                            PythonReferenceAnalysis, {x: px}, sympy_expr
                        )

                else:

                    def trace_f(px, py):
                        # [FUNCTION DOC] HELPER: Two-argument version that binds both symbols to
                        # the arguments and interprets the expression under the python analysis.
                        # Traced by fx.
                        return sympy_interp(
                            PythonReferenceAnalysis, {x: px, y: py}, sympy_expr
                        )

                gm = fx.symbolic_trace(trace_f)

                self.assertEqual(
                    sympy_interp(
                        PythonReferenceAnalysis, dict(zip(symbols, args)), sympy_expr
                    ),
                    gm(*args),
                )

    @parametrize(
        "fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS
    )
    def test_tensor_interp(self, fn):
        # [FUNCTION DOC] For each applicable op and valid arguments, compares the tensor
        # analysis applied directly to tensors with sympy_interp using the tensor analysis,
        # after aligning dtypes, within tolerance; also checks boolean ops give bool dtype and
        # integer-conversion ops give int64. Unimplemented ops print a message; any other
        # exception fails the test.
        # Skip operations not implemented or not applicable for tensors
        if fn in ("div", "truncdiv", "int_truediv", "mod", "round_decimal"):
            return

        is_integer = None
        if fn == "pow_by_natural":
            is_integer = True

        x = sympy.Symbol("x", integer=is_integer)
        y = sympy.Symbol("y", integer=is_integer)

        vals = CONSTANTS
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]
        elif fn in BITWISE_OPS:
            vals = vals + [True, False]

        arity = 1
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2

        symbols = [x]
        if arity == 2:
            symbols = [x, y]

        for args in itertools.product(vals, repeat=arity):
            if arity == 1 and not valid_unary(fn, *args):
                continue
            elif arity == 2 and not valid_binary(fn, *args):
                continue

            with self.subTest(args=args):
                tensor_args = [
                    torch.tensor(
                        a, dtype=torch.double if isinstance(a, float) else torch.int64
                    )
                    for a in args
                ]

                try:
                    tensor_fn = getattr(TensorReferenceAnalysis, fn)
                    sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)
                    direct_result = tensor_fn(*tensor_args)
                    interp_result = sympy_interp(
                        TensorReferenceAnalysis,
                        dict(zip(symbols, tensor_args)),
                        sympy_expr,
                    )

                    # Ensure both results are of the same dtype for comparison
                    if direct_result.dtype != interp_result.dtype:
                        if (
                            direct_result.dtype == torch.bool
                            or interp_result.dtype == torch.bool
                        ):
                            direct_result = direct_result.to(torch.bool)
                            interp_result = interp_result.to(torch.bool)
                        else:
                            direct_result = direct_result.to(torch.double)
                            interp_result = interp_result.to(torch.double)

                    self.assertTrue(
                        torch.allclose(
                            direct_result, interp_result, rtol=1e-5, atol=1e-8
                        ),
                        lambda msg: f"{msg}\nMismatch for {fn}{args}: direct={direct_result}, interp={interp_result}",
                    )

                    if fn in UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS:
                        self.assertEqual(direct_result.dtype, torch.bool)
                        self.assertEqual(interp_result.dtype, torch.bool)

                    if fn in (
                        "floor_to_int",
                        "ceil_to_int",
                        "round_to_int",
                        "trunc_to_int",
                    ):
                        self.assertEqual(direct_result.dtype, torch.int64)
                        self.assertEqual(interp_result.dtype, torch.int64)

                except NotImplementedError:
                    print(f"Operation {fn} not implemented for TensorReferenceAnalysis")
                except Exception as e:
                    self.fail(f"Unexpected error for {fn}{args}: {str(e)}")


def type_name_fn(type: type) -> str:
    # [FUNCTION DOC] HELPER: Returns the class name of a type; used to name parametrized
    # test cases.
    return type.__name__


def parametrize_relational_types(*types):
    # [FUNCTION DOC] HELPER: Returns a decorator that parametrizes a test over the operator
    # argument using the given relational types, or all relational types if none are given,
    # named by class.
    def wrapper(f: Callable):
        # [FUNCTION DOC] HELPER: Applies the parametrization to the test function and returns
        # the result.
        return parametrize("op", types or RELATIONAL_TYPES, name_fn=type_name_fn)(f)

    return wrapper


class TestSympySolve(TestCase):
    # [CLASS DOC] Tests of the relational solver that isolates a symbol on one side of a
    # relation, including giving up, multiplication and division, and floor division handling.
    def _create_integer_symbols(self) -> list[sympy.Symbol]:
        # [FUNCTION DOC] HELPER: Returns three integer sympy symbols named a, b and c.
        return sympy.symbols("a b c", integer=True)

    def test_give_up(self):
        # [FUNCTION DOC] Checks the solver returns nothing for non-relations, relations where
        # the symbol appears on both sides or neither side, and results that would be a logical
        # and or or.
        from sympy import Eq, Ne

        a, b, c = self._create_integer_symbols()

        cases = [
            # Not a relational operation.
            a + b,
            # 'a' appears on both sides.
            Eq(a, a + 1),
            # 'a' doesn't appear on neither side.
            Eq(b, c + 1),
            # Result is a 'sympy.And'.
            Eq(FloorDiv(a, b), c),
            # Result is a 'sympy.Or'.
            Ne(FloorDiv(a, b), c),
        ]

        for case in cases:
            e = try_solve(case, a)
            self.assertEqual(e, None)

    @parametrize_relational_types()
    def test_noop(self, op):
        # [FUNCTION DOC] Checks that when the symbol is already alone on the left, the
        # solver returns the original relation and the right-hand side unchanged, for
        # every relational type.
        a, b, _ = self._create_integer_symbols()

        lhs, rhs = a, 42 * b
        expr = op(lhs, rhs)

        r = try_solve(expr, a)
        self.assertNotEqual(r, None)

        r_expr, r_rhs = r
        self.assertEqual(r_expr, expr)
        self.assertEqual(r_rhs, rhs)

    @parametrize_relational_types()
    def test_noop_rhs(self, op):
        # [FUNCTION DOC] Checks that when the symbol is on the right, the solver mirrors
        # the relation operator and swaps the sides, returning the other side as the
        # right-hand side.
        a, b, _ = self._create_integer_symbols()

        lhs, rhs = 42 * b, a

        mirror = mirror_rel_op(op)
        self.assertNotEqual(mirror, None)

        expr = op(lhs, rhs)

        r = try_solve(expr, a)
        self.assertNotEqual(r, None)

        r_expr, r_rhs = r
        self.assertEqual(r_expr, mirror(rhs, lhs))
        self.assertEqual(r_rhs, lhs)

    def _test_cases(
        self,
        cases: list[tuple[sympy.Basic, sympy.Basic]],
        thing: sympy.Basic,
        op: type[sympy.Rel],
        **kwargs,
    ):
        # [FUNCTION DOC] HELPER: For each source and expected pair, solves for the given thing
        # with extra keyword arguments, checks the solver result is nothing exactly when nothing
        # is expected, and otherwise that the right-hand side equals expected and the relation
        # equals the op applied to the thing and expected.
        for source, expected in cases:
            r = try_solve(source, thing, **kwargs)

            self.assertTrue(
                (r is None and expected is None)
                or (r is not None and expected is not None)
            )

            if r is not None:
                r_expr, r_rhs = r
                self.assertEqual(r_rhs, expected)
                self.assertEqual(r_expr, op(thing, expected))

    def test_addition(self):
        # [FUNCTION DOC] Checks solving equality relations with an added term isolates the
        # symbol.
        from sympy import Eq

        a, b, c = self._create_integer_symbols()

        cases = [
            (Eq(a + b, 0), -b),
            (Eq(a + 5, b - 5), b - 10),
            (Eq(a + c * b, 1), 1 - c * b),
        ]

        self._test_cases(cases, a, Eq)

    @parametrize_relational_types(sympy.Eq, sympy.Ne)
    def test_multiplication_division(self, op):
        # [FUNCTION DOC] For equality and inequality-not-equal, checks solving a product for the
        # symbol divides both sides by the factor.
        a, b, c = self._create_integer_symbols()

        cases = [
            (op(a * b, 1), 1 / b),
            (op(a * 5, b - 5), (b - 5) / 5),
            (op(a * b, c), c / b),
        ]

        self._test_cases(cases, a, op)

    @parametrize_relational_types(*INEQUALITY_TYPES)
    def test_multiplication_division_inequality(self, op):
        # [FUNCTION DOC] For ordering relations, checks multiplication and division by positive
        # factors keep the operator, by negative factors flip it, and factors of unknown sign
        # give no solution.
        a, b, _ = self._create_integer_symbols()
        intneg = sympy.Symbol("neg", integer=True, negative=True)
        intpos = sympy.Symbol("pos", integer=True, positive=True)

        cases = [
            # Divide/multiply both sides by positive number.
            (op(a * intpos, 1), 1 / intpos),
            (op(a / (5 * intpos), 1), 5 * intpos),
            (op(a * 5, b - 5), (b - 5) / 5),
            # 'b' is not strictly positive nor negative, so we can't
            # divide/multiply both sides by 'b'.
            (op(a * b, 1), None),
            (op(a / b, 1), None),
            (op(a * b * intpos, 1), None),
        ]

        mirror_cases = [
            # Divide/multiply both sides by negative number.
            (op(a * intneg, 1), 1 / intneg),
            (op(a / (5 * intneg), 1), 5 * intneg),
            (op(a * -5, b - 5), -(b - 5) / 5),
        ]
        mirror_op = mirror_rel_op(op)
        if mirror_op is None:
            raise AssertionError("mirror_op should not be None")

        self._test_cases(cases, a, op)
        self._test_cases(mirror_cases, a, mirror_op)

    @parametrize_relational_types()
    def test_floordiv(self, op):
        # [FUNCTION DOC] For each relational type, checks solving relations on floor division by
        # a positive symbol gives the expected bound and operator (greater and less variants
        # normalize to at-least and less-than), and that it gives nothing when the divisor or
        # other side is not known positive or integer. With floor division inequalities disabled
        # the special case gives nothing.
        from sympy import Eq, Ge, Gt, Le, Lt, Ne

        a, b, c = sympy.symbols("a b c")
        pos = sympy.Symbol("pos", positive=True)
        integer = sympy.Symbol("integer", integer=True)

        # (Eq(FloorDiv(a, pos), integer), And(Ge(a, integer * pos), Lt(a, (integer + 1) * pos))),
        # (Eq(FloorDiv(a + 5, pos), integer), And(Ge(a, integer * pos), Lt(a, (integer + 1) * pos))),
        # (Ne(FloorDiv(a, pos), integer), Or(Lt(a, integer * pos), Ge(a, (integer + 1) * pos))),

        special_case = {
            # 'FloorDiv' turns into 'And', which can't be simplified any further.
            Eq: (Eq(FloorDiv(a, pos), integer), None),
            # 'FloorDiv' turns into 'Or', which can't be simplified any further.
            Ne: (Ne(FloorDiv(a, pos), integer), None),
            Gt: (Gt(FloorDiv(a, pos), integer), (integer + 1) * pos),
            Ge: (Ge(FloorDiv(a, pos), integer), integer * pos),
            Lt: (Lt(FloorDiv(a, pos), integer), integer * pos),
            Le: (Le(FloorDiv(a, pos), integer), (integer + 1) * pos),
        }[op]

        cases: list[tuple[sympy.Basic, sympy.Basic]] = [
            # 'b' is not strictly positive
            (op(FloorDiv(a, b), integer), None),
            # 'c' is not strictly positive
            (op(FloorDiv(a, pos), c), None),
        ]

        # The result might change after 'FloorDiv' transformation.
        # Specifically:
        #   - [Ge, Gt] => Ge
        #   - [Le, Lt] => Lt
        if op in (sympy.Gt, sympy.Ge):
            r_op = sympy.Ge
        elif op in (sympy.Lt, sympy.Le):
            r_op = sympy.Lt
        else:
            r_op = op

        self._test_cases([special_case, *cases], a, r_op)
        self._test_cases(
            [(special_case[0], None), *cases], a, r_op, floordiv_inequality=False
        )

    def test_floordiv_eq_simplify(self):
        # [FUNCTION DOC] Checks equality on floor division by a constant simplifies to a single
        # strict or non-strict inequality on the symbol for two specific examples.
        from sympy import Eq, Le, Lt

        a = sympy.Symbol("a", positive=True, integer=True)

        def check(expr, expected):
            # [FUNCTION DOC] HELPER: Solves the expression for the symbol, requires a result,
            # and checks the returned relation equals the expected one.
            r = try_solve(expr, a)
            self.assertNotEqual(r, None)
            r_expr, _ = r
            self.assertEqual(r_expr, expected)

        # (a + 10) // 3 == 3
        # =====================================
        # 3 * 3 <= a + 10         (always true)
        #          a + 10 < 4 * 3 (not sure)
        check(Eq(FloorDiv(a + 10, 3), 3), Lt(a, (3 + 1) * 3 - 10))

        # (a + 10) // 2 == 4
        # =====================================
        # 4 * 2 <= 10 - a         (not sure)
        #          10 - a < 5 * 2 (always true)
        check(Eq(FloorDiv(10 - a, 2), 4), Le(a, -(4 * 2 - 10)))

    @skipIf(not TEST_Z3, "Z3 not installed")
    def test_z3_proof_floordiv_eq_simplify(self):
        # [FUNCTION DOC] Uses the Z3 solver to prove the floor division simplification is sound:
        # checks the solver output equals the expected inequality, then asks Z3 whether any
        # positive integer makes the original and simplified conditions differ and asserts none
        # exists. Skipped when Z3 is not installed.
        import z3
        from sympy import Eq, Lt

        a = sympy.Symbol("a", positive=True, integer=True)
        a_ = z3.Int("a")

        # (a + 10) // 3 == 3
        # =====================================
        # 3 * 3 <= a + 10         (always true)
        #          a + 10 < 4 * 3 (not sure)
        solver = z3.SolverFor("QF_NRA")

        # Add assertions for 'a_'.
        solver.add(a_ > 0)

        expr = Eq(FloorDiv(a + 10, 3), 3)
        r_expr, _ = try_solve(expr, a)

        # Check 'try_solve' really returns the 'expected' below.
        expected = Lt(a, (3 + 1) * 3 - 10)
        self.assertEqual(r_expr, expected)

        # Check whether there is an integer 'a_' such that the
        # equation below is satisfied.
        solver.add(
            # expr
            (z3.ToInt((a_ + 10) / 3.0) == 3)
            !=
            # expected
            (a_ < (3 + 1) * 3 - 10)
        )

        # Assert that there's no such an integer.
        # i.e. the transformation is sound.
        r = solver.check()
        self.assertEqual(r, z3.unsat)

    def test_simple_floordiv_gcd(self):
        # [FUNCTION DOC] Checks the simple gcd helper for floor division returns the expected
        # common factor for several monomial and polynomial cases, and one for a case with no
        # common factor.
        x, y, z = sympy.symbols("x y z")

        # positive tests
        self.assertEqual(simple_floordiv_gcd(x, x), x)
        self.assertEqual(simple_floordiv_gcd(128 * x, 2304), 128)
        self.assertEqual(simple_floordiv_gcd(128 * x + 128 * y, 2304), 128)
        self.assertEqual(simple_floordiv_gcd(128 * x + 128 * y + 8192 * z, 9216), 128)
        self.assertEqual(simple_floordiv_gcd(49152 * x, 96 * x), 96 * x)
        self.assertEqual(simple_floordiv_gcd(96 * x, 96 * x), 96 * x)
        self.assertEqual(simple_floordiv_gcd(x * y, x), x)
        self.assertEqual(simple_floordiv_gcd(384 * x * y, x * y), x * y)
        self.assertEqual(simple_floordiv_gcd(256 * x * y, 8 * x), 8 * x)

        # negative tests
        self.assertEqual(simple_floordiv_gcd(x * y + x + y + 1, x + 1), 1)


class TestSympyFunctions(TestCase):
    # [CLASS DOC] Tests of custom sympy function classes (shift, pickling, min/max,
    # modular indexing).
    def test_shift_edges(self):
        # [FUNCTION DOC] Checks shifting by zero returns the operand and a negative shift count
        # raises a ValueError for both left and right shift.
        x = sympy.Symbol("x", integer=True)
        self.assertEqual(LShift(x, 0), x)
        self.assertEqual(RShift(x, 0), x)
        with self.assertRaisesRegex(ValueError, "negative shift count"):
            LShift(x, -1)
        with self.assertRaisesRegex(ValueError, "negative shift count"):
            RShift(x, -1)

    def test_pickle(self):
        # [FUNCTION DOC] Checks that opaque unary and bitwise function instances survive a
        # pickle round trip and compare equal to the original.
        x = OpaqueUnaryFn_cos(sympy.Symbol("a"))
        r = pickle.loads(pickle.dumps(x))
        self.assertEqual(x, r)

        x = BitwiseFn_bitwise_and(sympy.Symbol("a"), sympy.Symbol("b"))
        r = pickle.loads(pickle.dumps(x))
        self.assertEqual(x, r)

    def test_min_max_scaled_known_sign_term(self):
        # [FUNCTION DOC] Checks min and max of scaled copies of one symbol simplify to the right
        # scaled symbol when its sign is known, and stay unevaluated min/max instances when the
        # sign is unknown.
        s = sympy.Symbol("s", positive=True, integer=True)
        self.assertEqual(TorchSymMin(128 * s, 512 * s), 128 * s)
        self.assertEqual(TorchSymMax(128 * s, 512 * s), 512 * s)

        z = sympy.Symbol("z", nonpositive=True, integer=True)
        self.assertEqual(TorchSymMin(128 * z, 512 * z), 512 * z)
        self.assertEqual(TorchSymMax(128 * z, 512 * z), 128 * z)

        x = sympy.Symbol("x", integer=True)
        self.assertIsInstance(TorchSymMin(128 * x, 512 * x), TorchSymMin)
        self.assertIsInstance(TorchSymMax(128 * x, 512 * x), TorchSymMax)

    def test_modular_indexing_does_not_strip_unproven_nonnegative_term(self):
        # [FUNCTION DOC] Checks modular indexing is not simplified by stripping a term whose
        # non-negativity is unproven, and numerically verifies the expression equals the
        # mathematical modulus for a grid of values and stays within the modulus range.
        q0 = sympy.Symbol("q0", integer=True, nonnegative=True)
        n = sympy.Symbol("n", integer=True, positive=True)
        poisoned_base = n**2 + FloorDiv(-1 - q0, 2)
        expr = ModularIndexing(poisoned_base, 1, n**2)

        self.assertNotEqual(expr, ModularIndexing(-1 - q0, 2, n**2))
        for nv in range(2, 9):
            for q0v in range(4 * nv * nv):
                actual = int(expr.subs({n: nv, q0: q0v}))
                expected = (nv * nv + (-1 - q0v) // 2) % (nv * nv)
                self.assertEqual(actual, expected)
                self.assertTrue(0 <= actual < nv * nv)


class TestSingletonInt(TestCase):
    # [CLASS DOC] Tests of the singleton integer symbolic type used for nested tensor sizes.
    def test_basic(self):
        # [FUNCTION DOC] Checks equality, ordering and multiplication semantics of singleton
        # ints against each other and integers: equal only with the same id and coefficient,
        # comparisons that cannot be decided raise a ValueError mentioning indeterminate,
        # multiplying by an int scales the coefficient, multiplying two singletons raises, and
        # the free symbol set is empty.
        j1 = SingletonInt(1, coeff=1)
        j1_copy = SingletonInt(1, coeff=1)
        j2 = SingletonInt(2, coeff=1)
        j1x2 = SingletonInt(1, coeff=2)

        def test_eq(a, b, expected):
            # [FUNCTION DOC] HELPER: Checks the equality relation of two values equals the
            # expected boolean and the inequality relation with swapped operands equals
            # its negation.
            self.assertEqual(sympy.Eq(a, b), expected)
            self.assertEqual(sympy.Ne(b, a), not expected)

        # eq, ne
        test_eq(j1, j1, True)
        test_eq(j1, j1_copy, True)
        test_eq(j1, j2, False)
        test_eq(j1, j1x2, False)
        test_eq(j1, sympy.Integer(1), False)
        test_eq(j1, sympy.Integer(3), False)

        def test_ineq(a, b, expected, *, strict=True):
            # [FUNCTION DOC] HELPER: Checks the strict or non-strict greater and less
            # comparisons (sympy and predicate versions, both operand orders). If the expected
            # value is a boolean, results match it and its negation; otherwise it is a message
            # pattern and each comparison must raise a ValueError.
            greater = (sympy.Gt, is_gt) if strict else (sympy.Ge, is_ge)
            less = (sympy.Lt, is_lt) if strict else (sympy.Le, is_le)

            if isinstance(expected, bool):
                # expected is always True
                for fn in greater:
                    self.assertEqual(fn(a, b), expected)
                    self.assertEqual(fn(b, a), not expected)
                for fn in less:
                    self.assertEqual(fn(b, a), expected)
                    self.assertEqual(fn(a, b), not expected)
            else:
                for fn in greater:
                    with self.assertRaisesRegex(ValueError, expected):
                        fn(a, b)
                for fn in less:
                    with self.assertRaisesRegex(ValueError, expected):
                        fn(b, a)

        # ge, le, gt, lt
        for strict in (True, False):
            _test_ineq = functools.partial(test_ineq, strict=strict)
            _test_ineq(j1, sympy.Integer(0), True)
            _test_ineq(j1, sympy.Integer(3), "indeterminate")
            _test_ineq(j1, j2, "indeterminate")
            _test_ineq(j1x2, j1, True)

        # Special cases for ge, le, gt, lt:
        for ge in (sympy.Ge, is_ge):
            self.assertTrue(ge(j1, j1))
            self.assertTrue(ge(j1, sympy.Integer(2)))
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                ge(sympy.Integer(2), j1)
        for le in (sympy.Le, is_le):
            self.assertTrue(le(j1, j1))
            self.assertTrue(le(sympy.Integer(2), j1))
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                le(j1, sympy.Integer(2))

        for gt in (sympy.Gt, is_gt):
            self.assertFalse(gt(j1, j1))
            self.assertFalse(gt(sympy.Integer(2), j1))
            # it is only known to be that j1 >= 2, j1 > 2 is indeterminate
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                gt(j1, sympy.Integer(2))

        for lt in (sympy.Lt, is_lt):
            self.assertFalse(lt(j1, j1))
            self.assertFalse(lt(j1, sympy.Integer(2)))
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                lt(sympy.Integer(2), j1)

        # mul
        self.assertEqual(j1 * 2, j1x2)
        # Unfortunately, this doesn't not automatically simplify to 2*j1
        # since sympy.Mul doesn't trigger __mul__ unlike the above.
        self.assertIsInstance(sympy.Mul(j1, 2), sympy.core.mul.Mul)

        with self.assertRaisesRegex(ValueError, "cannot be multiplied"):
            j1 * j2

        self.assertEqual(j1.free_symbols, set())

class TestIdentity(TestCase):
    # [CLASS DOC] Tests of the Identity sympy wrapper function.
    def test_expand_identity(self):
        # [FUNCTION DOC] Checks expanding with the identity option removes all Identity nodes
        # and gives the wrapped argument.
        """
        Test removing an identity via expansion.
        """
        x = sympy.Symbol("x")
        arg = x + sympy.S.One
        expr = Identity(arg)
        expanded = expr.expand(identity=True)
        self.assertEqual(expanded.count(Identity), 0)
        self.assertEqual(expanded, arg)

    def test_cast_identity_int(self):
        # [FUNCTION DOC] Checks converting an Identity wrapping an integer to int gives
        # the integer.
        num = 1
        expr = Identity(num)
        self.assertEqual(num, int(expr))

    def test_cast_identity_float(self):
        # [FUNCTION DOC] Checks converting an Identity wrapping a float to float gives the
        # float.
        num = 1.1
        expr = Identity(num)
        self.assertEqual(num, float(expr))

    def test_cast_identity_illegal(self):
        # [FUNCTION DOC] Checks int and float conversion of an Identity wrapping a symbol or a
        # tuple raises a TypeError.
        sym = Identity(sympy.Symbol("x"))
        self.assertRaises(TypeError, int, sym)
        self.assertRaises(TypeError, float, sym)

        tup = (0, 1, 2)
        tup_I = Identity(tup)
        self.assertRaises(TypeError, int, tup_I)
        self.assertRaises(TypeError, float, tup_I)

class TestTypedExpr(TestCase):
    # [CLASS DOC] Tests of the typed expression wrapper from index propagation.
    def test_typed_expr(self):
        # [FUNCTION DOC] Wraps an Identity of one with a 32-bit integer dtype and checks the
        # stored expression equals one.
        I = Identity(1)
        typed_I = TypedExpr(I, torch.int32)
        self.assertEqual(typed_I.expr, 1)


class TestCCodePrinting(TestCase):
    # [CLASS DOC] Tests that custom sympy functions print to the expected C code strings.
    """Test _ccode methods on sympy function classes."""

    def test_floor_to_int_ccode(self):
        # [FUNCTION DOC] Checks the C code for floor-to-int of a symbol casts the floor call to
        # a 64-bit integer.
        from torch.utils._sympy.functions import FloorToInt

        x = sympy.Symbol("x")
        expr = FloorToInt(x)
        self.assertEqual(sympy.ccode(expr), "(int64_t)(floor(x))")

    def test_floor_to_int_ccode_compound(self):
        # [FUNCTION DOC] Checks the C code for floor-to-int of a sum casts the floor of the sum.
        from torch.utils._sympy.functions import FloorToInt

        x, y = sympy.symbols("x y")
        expr = FloorToInt(x + y)
        self.assertEqual(sympy.ccode(expr), "(int64_t)(floor(x + y))")

    def test_ceil_to_int_ccode(self):
        # [FUNCTION DOC] Checks the C code for ceil-to-int of a symbol casts the ceil call to a
        # 64-bit integer.
        from torch.utils._sympy.functions import CeilToInt

        x = sympy.Symbol("x")
        expr = CeilToInt(x)
        self.assertEqual(sympy.ccode(expr), "(int64_t)(ceil(x))")

    def test_trunc_to_int_ccode(self):
        # [FUNCTION DOC] Checks the C code for trunc-to-int of a symbol casts the trunc call to
        # a 64-bit integer.
        from torch.utils._sympy.functions import TruncToInt

        x = sympy.Symbol("x")
        expr = TruncToInt(x)
        self.assertEqual(sympy.ccode(expr), "(int64_t)(trunc(x))")

    def test_to_float_ccode(self):
        # [FUNCTION DOC] Checks the C code for converting an integer symbol to float casts
        # it to double.
        from torch.utils._sympy.functions import ToFloat

        x = sympy.Symbol("x", integer=True)
        expr = ToFloat(x)
        self.assertEqual(sympy.ccode(expr), "(double)(x)")

    def test_ccode_in_compound_expr(self):
        # [FUNCTION DOC] Checks a compound expression of floor-to-int of a to-float plus one
        # prints text containing the 64-bit integer type, floor and double.
        from torch.utils._sympy.functions import FloorToInt, ToFloat

        x = sympy.Symbol("x")
        expr = FloorToInt(ToFloat(x) + 1)
        result = sympy.ccode(expr)
        self.assertIn("int64_t", result)
        self.assertIn("floor", result)
        self.assertIn("double", result)


instantiate_parametrized_tests(TestValueRanges)
instantiate_parametrized_tests(TestSympyInterp)
instantiate_parametrized_tests(TestSympySolve)


if __name__ == "__main__":
    run_tests()
