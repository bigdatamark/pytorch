# [MODULE DOC] Implements IndexPropagation, an ops handler wrapper that propagates constants and sympy index expressions (from index_expr) through supported ops,
# so that indirect indexing derived from them can become direct indexing. Contains TypedExpr, SymPyOps (sympy-based op semantics), IndexPropVar and IndexPropagation.
# No import-time side effects.
# mypy: allow-untyped-defs
"""This file implements the IndexPropagation ops handler, which wraps an
underlying handler to add a limited form of constant propagation, as well as
propagation of sympy expressions downstream of ops.index_expr calls.

For example, say we have the IR:

    tmp0 = ops.index_expr(x, torch.int32)
    tmp1 = ops.constant(2, torch.int32)
    tmp2 = ops.mul(tmp0, tmp1)
    tmp3 = ops.indirect_indexing(tmp2, x_size)
    tmp4 = ops.load("buf0", tmp3)

The underlying handler would just see:

    ops.load("buf0", x * 2)

This is limited by the set of operators handled in the sympy expression
printers.

"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload, TypeAlias

import sympy

import torch
from torch._prims_common import dtype_to_type, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, Max, Min, ModularIndexing, Where
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from .ops_handler import DefaultHandler
from .sizevars import statically_known_true
from .utils import generate_assert
from .virtualized import V


_ExprType = sympy.Expr | float | int | bool


def _is_constant(val: _ExprType):
    # [FUNCTION DOC] HELPER: Return True if val is a numeric sympy value or a plain int, float or bool.
    if isinstance(val, sympy.Basic):
        return val.is_number
    return isinstance(val, (int, float, bool))


def upper_bound(val: _ExprType):
    # [FUNCTION DOC] HELPER: Return the upper bound of val's value range if it is a sympy expression, else val itself.
    return bound_sympy(val).upper if isinstance(val, sympy.Expr) else val


@dataclass
class TypedExpr:
    # [CLASS DOC] HELPER: A value (sympy expression or Python scalar) paired with a torch dtype. Constant values are wrapped to the dtype in post-init (integer overflow wraparound).
    """A SymPy expression with associated type"""

    expr: _ExprType
    dtype: torch.dtype

    def is_constant(self):
        # [FUNCTION DOC] HELPER: Whether the stored expression is a constant.
        return _is_constant(self.expr)

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: For constants, cast the expression to the dtype's Python type and, for integer dtypes, wrap it into the representable range (two's complement for signed); stores the result.
        if _is_constant(self.expr):
            expr = self.expr
            if isinstance(expr, sympy.Expr):
                expr = expr.expand(identity=True)
            expr = dtype_to_type(self.dtype)(expr)
            if is_integer_dtype(self.dtype):
                bits = torch.iinfo(self.dtype).bits
                if self.dtype.is_signed:
                    expr = expr + 2 ** (bits - 1)
                expr = expr % 2**bits
                if self.dtype.is_signed:
                    expr = expr - 2 ** (bits - 1)
            self.expr = expr


class SymPyOps:
    # [CLASS DOC] Ops handler in which values are TypedExpr. Static methods implement ops as sympy expressions; unsupported cases (non-integer floordiv/mod, bool min/max, uncertain remainder) return NotImplemented. Stateless.
    """An ops handler where all IR values are SymPy expressions

    When a value cannot be represented as a SymPy expression, the method is
    either not defined, or returns NotImplemented

    """

    @staticmethod
    def identity(value: Any) -> Any:
        # [FUNCTION DOC] Return the value unchanged.
        return value

    @staticmethod
    def constant(value: int | float | bool, dtype: torch.dtype) -> TypedExpr:
        # [FUNCTION DOC] Wrap a constant value and dtype as a TypedExpr.
        return TypedExpr(value, dtype)

    @staticmethod
    def index_expr(value: sympy.Expr | int, dtype: torch.dtype) -> TypedExpr:
        # [FUNCTION DOC] Wrap an index expression and dtype as a TypedExpr.
        return TypedExpr(value, dtype)

    @staticmethod
    def value_expr(value: sympy.Expr | int, dtype: torch.dtype) -> TypedExpr:
        # [FUNCTION DOC] Wrap a value expression and dtype as a TypedExpr.
        return TypedExpr(value, dtype)

    @staticmethod
    def to_dtype(
        value: TypedExpr,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = False,
    ) -> TypedExpr:
        # [FUNCTION DOC] Return a TypedExpr with the same expression but the new dtype (no actual conversion of the expression; src_dtype and use_compute_types are ignored).
        return TypedExpr(value.expr, dtype)

    @staticmethod
    def abs(x: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Absolute value of the expression, same dtype.
        return TypedExpr(abs(x.expr), x.dtype)  # type: ignore[arg-type]

    @staticmethod
    def square(x: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Square of the expression, same dtype.
        return TypedExpr(x.expr * x.expr, x.dtype)

    @staticmethod
    def add(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Sum of the expressions with the promoted dtype.
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr + y.expr, result_type)

    @staticmethod
    def sub(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Difference of the expressions with the promoted dtype.
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr - y.expr, result_type)

    @staticmethod
    def mul(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Product of the expressions with the promoted dtype.
        result_type = torch.promote_types(x.dtype, y.dtype)
        return TypedExpr(x.expr * y.expr, result_type)

    @staticmethod
    def neg(x: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Negated expression, same dtype.
        return TypedExpr(-x.expr, x.dtype)

    @staticmethod
    def floordiv(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Floor division as a FloorDiv expression; returns NotImplemented if the promoted dtype is not an integer type.
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        return TypedExpr(FloorDiv(x.expr, y.expr), result_type)

    @staticmethod
    def mod(x: TypedExpr, y: TypedExpr) -> TypedExpr | None:
        # [FUNCTION DOC] Modulo as a ModularIndexing expression; returns NotImplemented for non-integer promoted dtype.
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        result_expr = ModularIndexing(x.expr, sympy.S.One, y.expr)
        return TypedExpr(result_expr, result_type)

    @staticmethod
    def remainder(x: TypedExpr, y: TypedExpr) -> TypedExpr | None:
        # [FUNCTION DOC] C-style remainder. Returns NotImplemented for non-integer dtype; converts to ModularIndexing only when the sign condition between operands is provable, otherwise NotImplemented.
        # KNOWN ISSUE: the check compares nonnegativity of x with positivity of y for equality, which also holds when both are known false (negative x with non-positive y), where Python modulo and C remainder differ.
        result_type = torch.promote_types(x.dtype, y.dtype)
        if not is_integer_dtype(result_type):
            return NotImplemented

        x_expr = sympy.sympify(x.expr)
        y_expr = sympy.sympify(y.expr)
        # In these cases, remainder in Python == remainder in C++, so this transformation
        # is sound
        if (
            x_expr.is_nonnegative is not None
            and x_expr.is_nonnegative == y_expr.is_positive
        ):
            result_expr = ModularIndexing(x.expr, sympy.S.One, y.expr)
            return TypedExpr(result_expr, result_type)
        return NotImplemented

    @staticmethod
    def minimum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Min of the expressions; NotImplemented for bool promoted dtype.
        result_type = torch.promote_types(x.dtype, y.dtype)
        if result_type == torch.bool:
            return NotImplemented
        return TypedExpr(Min(x.expr, y.expr), result_type)

    @staticmethod
    def maximum(x: TypedExpr, y: TypedExpr) -> TypedExpr:
        # [FUNCTION DOC] Max of the expressions; NotImplemented for bool promoted dtype.
        result_type = torch.promote_types(x.dtype, y.dtype)
        if result_type == torch.bool:
            return NotImplemented
        return TypedExpr(Max(x.expr, y.expr), result_type)


@dataclass
class IndexPropVar:
    # [CLASS DOC] A value in the index propagation handler: either an IR value from the inner handler, or (when is_symbolic is true) a TypedExpr that can still be simplified symbolically.
    value: Any  # Either an IR value, or TypedExpr if is_symbolic is true
    is_symbolic: bool = False

    @staticmethod
    def new_symbolic(expr: TypedExpr) -> "IndexPropVar":
        # [FUNCTION DOC] Create a symbolic IndexPropVar holding the given TypedExpr.
        return IndexPropVar(expr, is_symbolic=True)

    def __post_init__(self):
        # [FUNCTION DOC] Validate the invariant that a symbolic IndexPropVar holds a TypedExpr; raises AssertionError otherwise.
        if not (not self.is_symbolic or isinstance(self.value, TypedExpr)):
            raise AssertionError("Symbolic IndexPropVar must contain a TypedExpr")


IndexPropResult: TypeAlias = IndexPropVar | tuple["IndexPropResult", ...]


class IndexPropagation(DefaultHandler):
    # [CLASS DOC] Ops handler wrapper around an inner handler. Ops on symbolic values are folded using SymPyOps when possible; otherwise symbolic arguments are materialized and the op is forwarded to the inner handler.
    # State: inner handler, shape env, variable ranges (var_to_range), indirect_var_ranges (kept as a reference so the caller can update it) and axioms. Key methods: indirect_indexing, fallback, propagate_sympy.
    """Ops wrapper that tries to propagate constant and index_expr values through the computation.

    This aims to maximize the compile time simplification possible, and convert
    indirect indexing from arange into normal static indexing.

    """

    def __init__(
        self,
        inner: Any,
        iter_ranges: dict[sympy.Symbol, sympy.Expr],
        indirect_var_ranges: dict[sympy.Symbol, sympy.Expr],
    ) -> None:
        # [FUNCTION DOC] Store the inner handler, shape env from the current graph, value ranges for iteration variables (zero up to size minus one) combined with shape env ranges, the indirect var range mapping (by reference), and axioms (bounds of iter vars plus shape env axioms).
        self._inner = inner
        self.shape_env = V.graph.sizevars.shape_env

        var_to_range = {
            k: ValueRanges(0, upper_bound(v) - 1) for k, v in iter_ranges.items()
        }
        self.var_to_range = tuple(
            itertools.chain(self.shape_env.var_to_range.items(), var_to_range.items())
        )
        # NOTE: this is intentionally kept as a reference so the caller can
        # update it in-place
        self.indirect_var_ranges = indirect_var_ranges

        axioms = []
        for x, s in iter_ranges.items():
            axioms.append(0 <= x)
            axioms.append(x < s)
        self.axioms = tuple(axioms) + self.shape_env.get_axioms()

    def materialize_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> Any:
        # [FUNCTION DOC] Turn a sympy expression into an inner-handler value: a constant if it is numeric, otherwise an index_expr.
        # Construct a new constant/index_expr from the SymPy expression
        if _is_constant(expr):
            val = dtype_to_type(dtype)(expr)
            return self._inner.constant(val, dtype)
        return self._inner.index_expr(expr, dtype)

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> IndexPropResult:
        # [FUNCTION DOC] Create a value_expr in the inner handler and wrap the result.
        return self.wrap(self._inner.value_expr(expr, dtype))

    def unwrap(self, a: Any | IndexPropVar) -> Any:
        # [FUNCTION DOC] Convert wrapped values (recursing into sequences) back to inner-handler values; symbolic ones are materialized from their expression, non-IndexPropVar values pass through.
        if isinstance(a, (list, tuple)):
            return tuple(self.unwrap(v) for v in a)

        if not isinstance(a, IndexPropVar):
            return a

        # Prefer the sympy representation if possible
        if a.is_symbolic:
            return self.materialize_expr(a.value.expr, a.value.dtype)

        return a.value

    def wrap(self, a) -> IndexPropResult:
        # [FUNCTION DOC] Wrap inner-handler values (recursing into sequences) as non-symbolic IndexPropVar objects.
        if isinstance(a, (list, tuple)):
            return tuple(self.wrap(v) for v in a)
        return IndexPropVar(a)

    @overload
    def fallback(
        self,
        name: Literal["indirect_indexing"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> IndexPropVar: ...

    @overload
    def fallback(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult: ...

    def fallback(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult:
        # [FUNCTION DOC] Call the named op on the inner handler with unwrapped arguments and wrap the result.
        # Fallback to the wrapped handler
        new_args = [self.unwrap(a) for a in args]
        new_kwargs = {k: self.unwrap(v) for k, v in kwargs.items()}
        return self.wrap(getattr(self._inner, name)(*new_args, **new_kwargs))

    def propagate_sympy(
        self, name: str, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> IndexPropResult:
        # [FUNCTION DOC] Try to compute the op with SymPyOps on the symbolic values. Falls back to the inner handler on overflow/value errors, NotImplemented results, or non-integer non-constant expressions;
        # otherwise returns a new symbolic IndexPropVar.
        # Build a new SymPy expression from this ops call
        def unwrap(a: Any | IndexPropVar) -> Any:
        # [FUNCTION DOC] HELPER: Return the underlying value of an IndexPropVar, or the argument itself.
            if not isinstance(a, IndexPropVar):
                return a
            return a.value

        new_args = [unwrap(a) for a in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        try:
            new_expr = getattr(SymPyOps, name)(*new_args, **new_kwargs)
        except (OverflowError, ValueError):
            # e.g. int(inf) raises OverflowError, int(nan) raises ValueError
            return self.fallback(name, args, kwargs)
        is_valid_expr = new_expr is not NotImplemented and (
            # Inductor doesn't expect floating point in sympy expressions, but
            # allow floating point constants to be propagated
            new_expr.is_constant() or new_expr.expr.is_integer
        )
        if not is_valid_expr:
            return self.fallback(name, args, kwargs)
        return IndexPropVar.new_symbolic(new_expr)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Default handler for any op: fall back to the inner handler if SymPyOps has no such op or any IndexPropVar argument is non-symbolic; otherwise propagate symbolically.
        if not hasattr(SymPyOps, name):
            return self.fallback(name, args, kwargs)

        var_arguments = [
            a
            for a in itertools.chain(args, kwargs.values())
            if isinstance(a, IndexPropVar)
        ]
        if not all(v.is_symbolic for v in var_arguments):
            return self.fallback(name, args, kwargs)

        return self.propagate_sympy(name, args, kwargs)

    def statically_true(self, e):
        # [FUNCTION DOC] Return whether the expression is provably true using the value ranges, indirect var ranges, axioms and shape env.
        """
        Given some iter_ranges, return a function that given an expression, returns whether
        it is true or false using value ranges, guard knowledge and runtime_asserts.

        FIXME I think this may not be entirely right, as we may not be able to use all runtime_asserts
              If this is an issue, just use guards in `self.axioms`.

              The proper way of handling this would be to have a global shape_env that adds
              runtime_asserts as they happen in the code. Then, it should be used in SimplifyIndexing
              to perform wrap_expr and in CSEProxy.check_bounds to elide upper / lower bounds also
              for indirect_indexing
        """
        var_to_range = (
            *self.var_to_range,
            *(
                (k, ValueRanges(0, upper_bound(v) - 1))
                for k, v in self.indirect_var_ranges.items()
            ),
        )
        # pyrefly: ignore [bad-argument-type]
        return statically_known_true(self.shape_env, e, self.axioms, var_to_range)

    def indirect_indexing(
        self,
        index: Any | IndexPropVar,
        size: Any,
        check: bool = True,
        wrap_neg=True,
    ) -> Any:
        # [FUNCTION DOC] Handle indirect indexing. For a symbolic index, returns the sympy expression (wrapped for negative values if wrap_neg, using Where when sign is unknown)
        # and emits a bounds check through the inner handler for bounds that cannot be proven (when asserts are enabled). For other indices, delegates to the inner handler and returns its value.
        if isinstance(index, IndexPropVar) and index.is_symbolic:
            # If we find something we can convert into a direct indexing we do so
            # We still need to (perhaps) wrap the expression and add bound checks
            # We want to do this "constant folding", as we don't allow to fuse
            # kernels into indirect indexing

            expr = sympy.sympify(index.value.expr)

            # TODO Perhaps move this logic to the simplify indexing pass
            def wrap_expr(expr):
            # [FUNCTION DOC] HELPER: Wrap a negative-capable index into range: unchanged if provably nonnegative, plus size if provably negative, otherwise a conditional expression.
                # Positive, negative, mixed
                if self.statically_true(0 <= expr):
                    return expr
                elif self.statically_true(expr < 0):
                    return expr + size
                else:
                    return Where(expr < 0, expr + size, expr)

            # -size <= expr only proves the lower bound after negative wrapping.
            can_prove_lower = self.statically_true(0 <= expr) or (
                wrap_neg and self.statically_true(-size <= expr)
            )
            can_prove_upper = self.statically_true(expr < size)
            if wrap_neg:
                expr = wrap_expr(expr)
            if generate_assert(check):
                self.fallback(
                    "check_bounds",
                    (expr, size),
                    dict(lower=not can_prove_lower, upper=not can_prove_upper),
                )
            return expr

        indirect_var = self.fallback(
            "indirect_indexing", (index, size, check, wrap_neg), {}
        ).value
        return indirect_var
