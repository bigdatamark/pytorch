# mypy: allow-untyped-defs
# [MODULE DOC] Reference implementations of the symbolic-shape operator interface (the 'ops'
# handler API) over three value domains: sympy expressions, plain Python numbers (FX traceable
# via torch sym helpers) and tensors (aten OpOverloads). Imports sympy and the custom sympy
# functions; no import-time side effects beyond defining classes.
import math
import operator
from typing import NoReturn

import sympy

import torch
from torch.utils._sympy.functions import (
    _keep_float,
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    IntTrueDiv,
    Max,
    Min,
    Mod,
    OpaqueUnaryFn_exp,
    OpaqueUnaryFn_log,
    OpaqueUnaryFn_log2,
    OpaqueUnaryFn_sqrt,
    PowByNatural,
    RoundDecimal,
    RoundToInt,
    ToFloat,
    TruncToInt,
)


# The sympy interpretation of operators.  It will also sometimes work with
# plain int/float, but if you do certain operations you will get out a
# sympy.Basic in the end.  If you want the Python/FX traceable interpretation,
# check PythonReferenceAnalysis.
# NB: For magic methods this needs to use normal magic methods
# so that test_magic_methods works
class ReferenceAnalysis:
    # [CLASS DOC] Stateless namespace of static methods interpreting each operator on sympy
    # expressions (or plain numbers, which are sympified or promoted to sympy results). Used
    # to evaluate shape expressions; the Python and Tensor variants below mirror the same
    # method names.
    @staticmethod
    def constant(c, dtype):
        # [FUNCTION DOC] Convert the constant c to a sympy value with sympify. The dtype
        # argument is ignored.
        return sympy.sympify(c)

    @staticmethod
    def or_(a, b):
        # [FUNCTION DOC] Logical or of a and b via the bitwise-or operator; works on sympy
        # booleans.
        return a | b

    @staticmethod
    def and_(a, b):
        # [FUNCTION DOC] Logical and of a and b via the bitwise-and operator; works on sympy
        # booleans.
        return a & b

    @staticmethod
    def eq(a, b):
        # [FUNCTION DOC] Equality. Returns a symbolic sympy Eq if either side is a sympy
        # expression, otherwise a plain Python bool.
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        # [FUNCTION DOC] Not-equal, computed as the logical negation of eq.
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        # [FUNCTION DOC] Return a less-than b using the native operator (symbolic for sympy
        # inputs).
        return a < b

    @staticmethod
    def gt(a, b):
        # [FUNCTION DOC] Return a greater-than b using the native operator (symbolic for sympy
        # inputs).
        return a > b

    @staticmethod
    def le(a, b):
        # [FUNCTION DOC] Return a less-or-equal b using the native operator (symbolic for
        # sympy inputs).
        return a <= b

    @staticmethod
    def ge(a, b):
        # [FUNCTION DOC] Return a greater-or-equal b using the native operator (symbolic for
        # sympy inputs).
        return a >= b

    @staticmethod
    def not_(a):
        # [FUNCTION DOC] Logical negation of a sympy boolean via the invert operator. Raises
        # an AssertionError if a is a plain Python bool, since the result would not be
        # symbolic.
        if isinstance(a, bool):
            raise AssertionError("not_ needs sympy expr")
        return ~a

    @staticmethod
    def reciprocal(x):
        # [FUNCTION DOC] Reciprocal of x as a float true division of 1.0 by x.
        return FloatTrueDiv(1.0, x)

    @staticmethod
    def square(x):
        # [FUNCTION DOC] Square of x, as a power-by-natural with exponent 2.
        return PowByNatural(x, 2)

    @staticmethod
    def trunc_to_int(x, dtype):
        # [FUNCTION DOC] Truncate x toward zero to an integer (TruncToInt). The dtype argument
        # is ignored.
        return TruncToInt(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        # [FUNCTION DOC] Ceiling of x as an integer-valued sympy expression. The dtype
        # argument is ignored.
        return sympy.ceiling(x)

    @staticmethod
    def floor_to_int(x, dtype):
        # [FUNCTION DOC] Floor of x as an integer-valued sympy expression. The dtype argument
        # is ignored.
        return sympy.floor(x)

    @staticmethod
    def floor(x):
        # [FUNCTION DOC] Floor of x that keeps float-ness of the result (via the keep-float
        # wrapper).
        return _keep_float(sympy.floor)(x)

    @staticmethod
    def ceil(x):
        # [FUNCTION DOC] Ceiling of x that keeps float-ness of the result (via the keep-float
        # wrapper).
        return _keep_float(sympy.ceiling)(x)

    @staticmethod
    def to_dtype(x, dtype):
        # [FUNCTION DOC] Convert x to the given dtype. Only float64 is supported (wraps in
        # ToFloat); any other dtype raises NotImplementedError.
        if dtype == torch.float64:
            return ToFloat(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def mod(x, y):
        # [FUNCTION DOC] Modulus of x by y using the symbolic Mod function.
        return Mod(x, y)

    @staticmethod
    def abs(x):
        # [FUNCTION DOC] Absolute value of x.
        return abs(x)

    @staticmethod
    def neg(x):
        # [FUNCTION DOC] Negation of x.
        return -x

    @staticmethod
    def truediv(a, b):
        # [FUNCTION DOC] Float true division of a by b (symbolic FloatTrueDiv).
        return FloatTrueDiv(a, b)

    @staticmethod
    def int_truediv(a, b):
        # [FUNCTION DOC] True division of two integers producing a float (symbolic
        # IntTrueDiv).
        return IntTrueDiv(a, b)

    @staticmethod
    def floordiv(a, b):
        # [FUNCTION DOC] Floor division of a by b (symbolic FloorDiv).
        return FloorDiv(a, b)

    @staticmethod
    def truncdiv(a, b) -> NoReturn:
        # [FUNCTION DOC] Not implemented; always raises NotImplementedError (truncating
        # division is a TODO).
        raise NotImplementedError("TODO: truncdiv")

    @staticmethod
    def add(a, b):
        # [FUNCTION DOC] Sum of a and b, keeping float-ness of the result.
        return _keep_float(operator.add)(a, b)

    @classmethod
    def sym_sum(cls, args):
        # [FUNCTION DOC] Sum of all values in the args sequence as one sympy Add. Classmethod,
        # cls unused.
        return sympy.Add(*args)

    @staticmethod
    def mul(a, b):
        # [FUNCTION DOC] Product of a and b, keeping float-ness of the result.
        return _keep_float(operator.mul)(a, b)

    @staticmethod
    def sub(a, b):
        # [FUNCTION DOC] Difference a minus b, keeping float-ness of the result.
        return _keep_float(operator.sub)(a, b)

    @staticmethod
    def exp(x):
        # [FUNCTION DOC] Opaque (unevaluated) exponential function of x.
        return OpaqueUnaryFn_exp(x)

    @staticmethod
    def log(x):
        # [FUNCTION DOC] Opaque (unevaluated) natural logarithm of x.
        return OpaqueUnaryFn_log(x)

    @staticmethod
    def log2(x):
        # [FUNCTION DOC] Opaque (unevaluated) base-2 logarithm of x.
        return OpaqueUnaryFn_log2(x)

    @staticmethod
    def sqrt(x):
        # [FUNCTION DOC] Opaque (unevaluated) square root of x.
        return OpaqueUnaryFn_sqrt(x)

    @staticmethod
    def pow(a, b):
        # [FUNCTION DOC] Float power a to the b via FloatPow, keeping float-ness of the
        # result.
        # pyrefly: ignore [bad-argument-count, bad-argument-type]
        return _keep_float(FloatPow)(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        # [FUNCTION DOC] Integer power of a by a natural-number exponent b (PowByNatural).
        return PowByNatural(a, b)

    @staticmethod
    def minimum(a, b):
        # [FUNCTION DOC] Symbolic minimum of a and b.
        return Min(a, b)

    @staticmethod
    def maximum(a, b):
        # [FUNCTION DOC] Symbolic maximum of a and b.
        return Max(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        # [FUNCTION DOC] Round a to the nearest integer (RoundToInt). The dtype argument is
        # ignored.
        return RoundToInt(a)

    @staticmethod
    def round_decimal(a, b):
        # [FUNCTION DOC] Round a to b decimal digits (RoundDecimal).
        return RoundDecimal(a, b)

    @staticmethod
    def bitwise_and(a, b):
        # [FUNCTION DOC] Symbolic bitwise and of a and b.
        return BitwiseFn_bitwise_and(a, b)

    @staticmethod
    def bitwise_or(a, b):
        # [FUNCTION DOC] Symbolic bitwise or of a and b.
        return BitwiseFn_bitwise_or(a, b)

    @staticmethod
    def bitwise_xor(a, b):
        # [FUNCTION DOC] Symbolic bitwise xor of a and b.
        return BitwiseFn_bitwise_xor(a, b)


# Unlike ReferenceAnalysis, does NOT sympyify, instead, works with plain
# Python types and is FX traceable.  Inheritance here is purely for code
# sharing (TODO: considering splitting out a BaseReferenceAnalysis).
class PythonReferenceAnalysis(ReferenceAnalysis):
    # [CLASS DOC] Variant of ReferenceAnalysis working on plain Python numbers and torch
    # symbolic helpers instead of sympy, so it is FX traceable. Inherits unchanged operators
    # and overrides those that need native Python semantics. Stateless.
    @staticmethod
    def constant(c, dtype):
        # [FUNCTION DOC] Convert c to a Python int, float or bool according to dtype (int64,
        # double, bool). Raises AssertionError for any other dtype.
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")

    @staticmethod
    def not_(a):
        # [FUNCTION DOC] Logical negation through torch.sym_not.
        return torch.sym_not(a)

    @classmethod
    def sym_sum(cls, args):
        # [FUNCTION DOC] Sum of the args sequence by folding add left to right; returns 0 for
        # an empty sequence and the single element for length one.
        if len(args) == 0:
            return 0
        if len(args) == 1:
            return args[0]
        acc = cls.add(args[0], args[1])
        for i in range(2, len(args)):
            acc = cls.add(acc, args[i])
        return acc

    @staticmethod
    def floordiv(a, b):
        # [FUNCTION DOC] Floor division using the native operator.
        return a // b

    @staticmethod
    def mod(x, y):
        # [FUNCTION DOC] Modulus using the native percent operator (Python sign semantics).
        return x % y

    @staticmethod
    def python_mod(x, y):
        # [FUNCTION DOC] Python-semantics modulus of x by y using the native percent operator
        # (same as mod).
        return x % y

    @staticmethod
    def truncdiv(a, b):
        # [FUNCTION DOC] Division a by b using the native true-division operator.
        # KNOWN ISSUE: despite the name this is true division, not truncating division, so the
        #   result is a float and not truncated.
        return a / b

    @staticmethod
    def to_dtype(x, dtype):
        # [FUNCTION DOC] Convert x to float64 with torch.sym_float; any other dtype raises
        # NotImplementedError.
        if dtype == torch.float64:
            return torch.sym_float(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    def exp(x) -> NoReturn:
        # [FUNCTION DOC] Always raises AssertionError: exp is not a valid shape expression
        # operator here.
        raise AssertionError("exp is not valid shape sympy expr")

    @staticmethod
    def log(x) -> NoReturn:
        # [FUNCTION DOC] Always raises AssertionError: log is not a valid shape expression
        # operator here.
        raise AssertionError("log is not valid shape sympy expr")

    @staticmethod
    def log2(x):
        # [FUNCTION DOC] Base-2 logarithm via torch._sym_log2.
        return torch._sym_log2(x)  # type: ignore[attr-defined]

    @staticmethod
    def sqrt(x):
        # [FUNCTION DOC] Square root via torch._sym_sqrt.
        return torch._sym_sqrt(x)  # type: ignore[attr-defined]

    @staticmethod
    def minimum(a, b):
        # [FUNCTION DOC] Minimum of a and b via torch.sym_min.
        return torch.sym_min(a, b)

    @staticmethod
    def maximum(a, b):
        # [FUNCTION DOC] Maximum of a and b via torch.sym_max.
        return torch.sym_max(a, b)

    @staticmethod
    def floor_to_int(x, dtype):
        # [FUNCTION DOC] Floor of x as a Python int using math.floor. The dtype argument is
        # ignored.
        return math.floor(x)

    @staticmethod
    def ceil_to_int(x, dtype):
        # [FUNCTION DOC] Ceiling of x as a Python int using math.ceil. The dtype argument is
        # ignored.
        return math.ceil(x)

    @staticmethod
    def floor(x):
        # [FUNCTION DOC] Floor of x returned as a Python float.
        return float(math.floor(x))

    @staticmethod
    def ceil(x):
        # [FUNCTION DOC] Ceiling of x returned as a Python float.
        return float(math.ceil(x))

    @staticmethod
    def truediv(a, b):
        # [FUNCTION DOC] True division a by b using the native operator.
        return a / b

    @staticmethod
    def pow(a, b):
        # [FUNCTION DOC] Power a to the b using the native operator.
        return a**b

    @staticmethod
    def pow_by_natural(a, b):
        # [FUNCTION DOC] Power a to the b using the native operator, with no overflow-safe
        # power helper.
        # Pray that safe_pow is not needed here lol.  In particular, this
        # never participates in VR low/high ranges, so overflow should be
        # unlikely
        return a**b

    @staticmethod
    def round_to_int(a, dtype):
        # [FUNCTION DOC] Round a to the nearest integer using the built-in round. The dtype
        # argument is ignored.
        return round(a)

    @staticmethod
    def round_decimal(a, b):
        # [FUNCTION DOC] Round a to b decimal digits using the built-in round.
        return round(a, ndigits=b)

    @staticmethod
    def bitwise_and(a, b):
        # [FUNCTION DOC] Bitwise and of a and b using the native operator.
        return a & b

    @staticmethod
    def bitwise_or(a, b):
        # [FUNCTION DOC] Bitwise or of a and b using the native operator.
        return a | b

    @staticmethod
    def bitwise_xor(a, b):
        # [FUNCTION DOC] Bitwise xor of a and b using the native operator.
        return a ^ b

    @staticmethod
    def expr_cond_pair(expr, cond):
        # [FUNCTION DOC] Package an expression and its condition as a pair (tuple), for use by
        # piecewise.
        return (expr, cond)

    @staticmethod
    def piecewise(*pairs):
        # [FUNCTION DOC] Build a conditional value from (expr, cond) pairs; the last pair acts
        # as the default branch and its condition is ignored. Folds from the right into nested
        # torch.sym_ite calls. Requires at least one pair.
        # Build nested sym_ite from right to left.
        # Piecewise((e1, c1), (e2, c2), ..., (en, True)) becomes
        # sym_ite(c1, e1, sym_ite(c2, e2, ... en))
        result = pairs[-1][0]
        for expr, cond in reversed(pairs[:-1]):
            result = torch.sym_ite(cond, expr, result)
        return result


# Like PythonReferenceAnalysis, but some export-unfriendly choices of
# operators to make things faster
class OptimizedPythonReferenceAnalysis(PythonReferenceAnalysis):
    # [CLASS DOC] Faster but export-unfriendly variant of PythonReferenceAnalysis: uses
    # torch.sym_sum and skips rounding on symbolic values (returning them unchanged).
    # Stateless.
    @staticmethod
    def sym_sum(args):
        # [FUNCTION DOC] Sum args with the single torch.sym_sum call instead of a left fold.
        return torch.sym_sum(args)

    @staticmethod
    def floor_to_int(x, dtype):
        # [FUNCTION DOC] Floor of x for concrete int or float inputs; any other (symbolic) x
        # is returned unchanged.
        if isinstance(x, (int, float)):
            return math.floor(x)
        return x

    @staticmethod
    def ceil_to_int(x, dtype):
        # [FUNCTION DOC] Ceiling of x for concrete int or float inputs; any other (symbolic) x
        # is returned unchanged.
        if isinstance(x, (int, float)):
            return math.ceil(x)
        return x

    @staticmethod
    def trunc_to_int(x, dtype):
        # [FUNCTION DOC] Truncate x toward zero for concrete int or float inputs; any other
        # (symbolic) x is returned unchanged.
        if isinstance(x, (int, float)):
            return math.trunc(x)
        return x


def _to_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Convert tensor x to dtype through the prims convert_element_type
    # op.
    return torch.ops.prims.convert_element_type.default(x, dtype)


# Suppose we have some int/float arguments.  This diagram commutes:
#
#   int/float  -- PythonReferenceAnalysis.op -->  int/float
#       |                                           |
#       |                                           |
#      torch.tensor(..., dtype=torch.int64/torch.float64)
#       |                                           |
#       V                                           V
#    Tensor    -- TensorReferenceAnalysis.op -->  Tensor
#
# NB: int before and after must be representable in int64 (we will
# insert guards accordingly.)
#
# This is guaranteed to be FX traceable with OpOverloads only.
class TensorReferenceAnalysis:
    # [CLASS DOC] Stateless namespace of static methods interpreting each operator on tensors
    # using aten OpOverloads only, so it is FX traceable. Intended to commute with the Python
    # variant when ints and floats are wrapped as int64 and float64 tensors (values must fit
    # int64). Several operators are unsupported and raise NotImplementedError.
    # NB: This is actually dead, because with Proxy tracing the factory
    # function isn't traced correctly.  Here for completeness.
    @staticmethod
    def constant(c, dtype):
        # [FUNCTION DOC] Build a scalar tensor of dtype from c, converted to int, float or
        # bool for int64, double or bool. Raises AssertionError for other dtypes.
        # KNOWN ISSUE: per its own comment it is dead code because proxy tracing does not
        #   trace the factory call.
        d: int | float | bool
        if dtype is torch.int64:
            d = int(c)
        elif dtype is torch.double:
            d = float(c)
        elif dtype is torch.bool:
            d = bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")
        return torch.ops.aten.scalar_tensor.default(d, dtype=dtype)

    @staticmethod
    def or_(a, b):
        # [FUNCTION DOC] Elementwise logical or of tensors a and b.
        return torch.ops.aten.logical_or.default(a, b)

    @staticmethod
    def and_(a, b):
        # [FUNCTION DOC] Elementwise logical and of tensors a and b.
        return torch.ops.aten.logical_and.default(a, b)

    @staticmethod
    def bitwise_and(a, b):
        # [FUNCTION DOC] Elementwise bitwise and of tensors a and b.
        return torch.ops.aten.bitwise_and(a, b)

    @staticmethod
    def bitwise_or(a, b):
        # [FUNCTION DOC] Elementwise bitwise or of tensors a and b.
        return torch.ops.aten.bitwise_or(a, b)

    @staticmethod
    def bitwise_xor(a, b):
        # [FUNCTION DOC] Elementwise bitwise xor of tensors a and b.
        return torch.ops.aten.bitwise_xor(a, b)

    @staticmethod
    def eq(a, b):
        # [FUNCTION DOC] Elementwise equality of tensors a and b.
        return torch.ops.aten.eq.Tensor(a, b)

    @classmethod
    def ne(cls, a, b):
        # [FUNCTION DOC] Elementwise inequality of tensors a and b. Classmethod, cls unused.
        return torch.ops.aten.ne.Tensor(a, b)

    @staticmethod
    def lt(a, b):
        # [FUNCTION DOC] Elementwise less-than of tensors a and b.
        return torch.ops.aten.lt.Tensor(a, b)

    @staticmethod
    def gt(a, b):
        # [FUNCTION DOC] Elementwise greater-than of tensors a and b.
        return torch.ops.aten.gt.Tensor(a, b)

    @staticmethod
    def le(a, b):
        # [FUNCTION DOC] Elementwise less-or-equal of tensors a and b.
        return torch.ops.aten.le.Tensor(a, b)

    @staticmethod
    def ge(a, b):
        # [FUNCTION DOC] Elementwise greater-or-equal of tensors a and b.
        return torch.ops.aten.ge.Tensor(a, b)

    @staticmethod
    def not_(a):
        # [FUNCTION DOC] Elementwise logical not of tensor a.
        return torch.ops.aten.logical_not.default(a)

    @staticmethod
    def reciprocal(x):
        # [FUNCTION DOC] Elementwise reciprocal of tensor x.
        return torch.ops.aten.reciprocal.default(x)

    @staticmethod
    def square(x):
        # [FUNCTION DOC] Elementwise square of tensor x.
        # TODO: maybe composite implicit autograd doesn't work here?
        return torch.ops.aten.square.default(x)

    @staticmethod
    def trunc_to_int(x, dtype):
        # [FUNCTION DOC] Truncate tensor x toward zero, then convert to dtype.
        return _to_dtype(torch.ops.aten.trunc.default(x), dtype)

    @staticmethod
    def ceil_to_int(x, dtype):
        # [FUNCTION DOC] Ceiling of tensor x, then convert to dtype.
        return _to_dtype(torch.ops.aten.ceil.default(x), dtype)

    @staticmethod
    def floor_to_int(x, dtype):
        # [FUNCTION DOC] Floor of tensor x, then convert to dtype.
        return _to_dtype(torch.ops.aten.floor.default(x), dtype)

    @staticmethod
    def floor(x):
        # [FUNCTION DOC] Elementwise floor of tensor x (dtype unchanged).
        return torch.ops.aten.floor.default(x)

    @staticmethod
    def ceil(x):
        # [FUNCTION DOC] Elementwise ceiling of tensor x (dtype unchanged).
        return torch.ops.aten.ceil.default(x)

    @staticmethod
    def to_dtype(x, dtype):
        # [FUNCTION DOC] Convert tensor x to dtype.
        return _to_dtype(x, dtype)

    @staticmethod
    def mod(x, y) -> NoReturn:
        # [FUNCTION DOC] Always raises NotImplementedError: no C-style modulus is available
        # from the frontend.
        # TODO: https://github.com/pytorch/pytorch/pull/133654
        raise NotImplementedError(
            "no C-style modulus operation available from frontend atm"
        )

    @staticmethod
    def abs(x):
        # [FUNCTION DOC] Elementwise absolute value of tensor x.
        return torch.ops.aten.abs.default(x)

    @staticmethod
    def neg(x):
        # [FUNCTION DOC] Elementwise negation of tensor x.
        return torch.ops.aten.neg.default(x)

    @staticmethod
    def truediv(a, b):
        # [FUNCTION DOC] Elementwise true division of tensors a and b.
        return torch.ops.aten.true_divide.Tensor(a, b)

    @staticmethod
    def int_truediv(a, b):
        # [FUNCTION DOC] Always raises NotImplementedError.
        # KNOWN ISSUE: unreachable code after the raise would be wrong anyway, since CPython
        #   true division of large ints is more precise than float64 division.
        raise NotImplementedError(
            "Python int truediv difficult to implement in PyTorch atm"
        )

        # TODO: This is wrong, CPython has a custom implementation of true
        # division that results in higher precision when the floats are
        # sufficiently large.  Short term fix: add a guard here
        # pyrefly: ignore [unreachable]
        return torch.ops.aten.true_divide.default(
            _to_dtype(a, torch.float64), _to_dtype(b, torch.float64)
        )

    @staticmethod
    def floordiv(a, b):
        # [FUNCTION DOC] Floor division of tensors a and b using div with floor rounding mode.
        return torch.ops.aten.div.Tensor_mode(a, b, rounding_mode="floor")

    @staticmethod
    def truncdiv(a, b) -> NoReturn:
        # [FUNCTION DOC] Always raises NotImplementedError: no C-style truncating division is
        # available.
        raise NotImplementedError(
            "no C-style truncdiv operation available from frontend atm"
        )

    @staticmethod
    def add(a, b):
        # [FUNCTION DOC] Elementwise sum of tensors a and b.
        return torch.ops.aten.add.Tensor(a, b)

    @staticmethod
    def mul(a, b):
        # [FUNCTION DOC] Elementwise product of tensors a and b.
        return torch.ops.aten.mul.Tensor(a, b)

    @staticmethod
    def sub(a, b):
        # [FUNCTION DOC] Elementwise difference of tensors a and b.
        return torch.ops.aten.sub.Tensor(a, b)

    @staticmethod
    def exp(x):
        # [FUNCTION DOC] Elementwise exponential of tensor x.
        return torch.ops.aten.exp.default(x)

    @staticmethod
    def log(x):
        # [FUNCTION DOC] Elementwise natural logarithm of tensor x.
        return torch.ops.aten.log.default(x)

    @staticmethod
    def log2(x):
        # [FUNCTION DOC] Elementwise base-2 logarithm of tensor x.
        return torch.ops.aten.log2.default(x)

    @staticmethod
    def sqrt(x):
        # [FUNCTION DOC] Elementwise square root of tensor x.
        return torch.ops.aten.sqrt.default(x)

    @staticmethod
    def sin(x):
        # [FUNCTION DOC] Elementwise sine of tensor x.
        return torch.ops.aten.sin.default(x)

    @staticmethod
    def cos(x):
        # [FUNCTION DOC] Elementwise cosine of tensor x.
        return torch.ops.aten.cos.default(x)

    @staticmethod
    def tanh(x):
        # [FUNCTION DOC] Elementwise hyperbolic tangent of tensor x.
        return torch.ops.aten.tanh.default(x)

    @staticmethod
    def sinh(x):
        # [FUNCTION DOC] Elementwise hyperbolic sine of tensor x.
        return torch.ops.aten.sinh.default(x)

    @staticmethod
    def cosh(x):
        # [FUNCTION DOC] Elementwise hyperbolic cosine of tensor x.
        return torch.ops.aten.cosh.default(x)

    @staticmethod
    def tan(x):
        # [FUNCTION DOC] Elementwise tangent of tensor x.
        return torch.ops.aten.tan.default(x)

    @staticmethod
    def acos(x):
        # [FUNCTION DOC] Elementwise arc cosine of tensor x.
        return torch.ops.aten.acos.default(x)

    @staticmethod
    def atan(x):
        # [FUNCTION DOC] Elementwise arc tangent of tensor x.
        return torch.ops.aten.atan.default(x)

    @staticmethod
    def asin(x):
        # [FUNCTION DOC] Elementwise arc sine of tensor x.
        return torch.ops.aten.asin.default(x)

    @staticmethod
    def pow(a, b):
        # [FUNCTION DOC] Elementwise power of tensor a raised to tensor b.
        return torch.ops.aten.pow.Tensor_Tensor(a, b)

    @staticmethod
    def pow_by_natural(a, b):
        # [FUNCTION DOC] Elementwise power of tensor a raised to tensor b; the same op handles
        # integer inputs.
        # NB: pow handles int x int fine
        return torch.ops.aten.pow.Tensor_Tensor(a, b)

    @staticmethod
    def minimum(a, b):
        # [FUNCTION DOC] Elementwise minimum of tensors a and b.
        return torch.ops.aten.minimum.default(a, b)

    @staticmethod
    def maximum(a, b):
        # [FUNCTION DOC] Elementwise maximum of tensors a and b.
        return torch.ops.aten.maximum.default(a, b)

    @staticmethod
    def round_to_int(a, dtype):
        # [FUNCTION DOC] Round tensor a to the nearest integer value. The dtype argument is
        # ignored.
        return torch.ops.aten.round.default(a)

    @staticmethod
    def round_decimal(a, b) -> NoReturn:
        # [FUNCTION DOC] Always raises NotImplementedError: rounding to decimals with a tensor
        # argument is unsupported.
        raise NotImplementedError(
            "round decimal doesn't support Tensor second argument atm"
        )

        # return torch.ops.aten.round.decimals(a, b)
