# mypy: allow-untyped-defs
# [MODULE DOC] Size-variable reasoning for Inductor: SizeVarAllocator wraps a ShapeEnv to simplify
# sympy index/size expressions, answer statically-known queries without guarding, add guards,
# compute strides, evaluate size hints and decide vectorization/contiguity facts. Also provides
# LaneContiguity, stride helpers, and index simplification inside vectorized ranges. Import-time
# side effects: creates the module logger and a few module constants.
import dataclasses
import functools
import itertools
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import sympy
from sympy import Expr

from torch import SymInt
from torch.fx.experimental._size_hinting import (
    _guarding_hint_or_throw_base,
    _maybe_realize_expr,
    _optimization_hint_base,
)
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    IterateExprs,
    ShapeEnv,
    SymNode,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    FloorDiv,
    Max,
    Min,
    Mod,
    ModularIndexing,
    safe_gcd,
    simple_floordiv_gcd,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._sympy.value_ranges import IntInfinity, ValueRanges

from . import config
from .runtime.runtime_utils import is_power_of_2
from .utils import (
    has_free_symbols,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_subs,
    VarRanges,
)
from .virtualized import V


log = logging.getLogger(__name__)


Width = int | IntInfinity


@dataclasses.dataclass(frozen=True)
class LaneContiguity:
    # [CLASS DOC] Frozen record describing how an index expression varies across vector lanes: contiguous_width
    # and uniform_width (maximum widths, None unknown, int_oo unbounded), stride, and unknown flag. It does
    # not prove memory safety; callers must still check dimension, stride and alignment.
    """How an index expression varies across lanes.

    This describes only the per-lane relationship, not full memory safety.
    Callers still need to prove the relevant tensor dimension, stride, and base
    alignment make a vector load valid. ``int_oo`` means the expression does not
    impose a finite limit on that width.
    """

    contiguous_width: Width | None = None
    uniform_width: Width | None = None
    stride: int | Expr | None = None
    unknown: bool = False

    @property
    def uniform(self) -> bool:
        # [FUNCTION DOC] True if a uniform width is known (not None).
        return self.uniform_width is not None

    def is_contiguous_for(self, width: int) -> bool:
        # [FUNCTION DOC] True if the contiguous width covers the requested width.
        return _width_covers(self.contiguous_width, width)

    def is_uniform_for(self, width: int) -> bool:
        # [FUNCTION DOC] True if the uniform width covers the requested width.
        return _width_covers(self.uniform_width, width)


def _width_covers(max_width: Width | None, width: int) -> bool:
    # [FUNCTION DOC] HELPER: True if max_width is known and is unbounded or at least width.
    return max_width is not None and (max_width == int_oo or max_width >= width)


def _min_width(lhs: Width | None, rhs: Width | None) -> Width | None:
    # [FUNCTION DOC] HELPER: Minimum of two optional widths; None if either is None; an unbounded width
    # yields the other.
    if lhs is None or rhs is None:
        return None
    if lhs == int_oo:
        return rhs
    if rhs == int_oo:
        return lhs
    return min(lhs, rhs)


def _largest_power_of_2_factor(n: int) -> int:
    # [FUNCTION DOC] HELPER: Largest power of two dividing n (the lowest set bit).
    return n & -n


# Symbols created by CppTemplateKernel.slice_nd → parse_expr_with_index_symbols
# and are internal to C++ GEMM codegen (not tracked by ShapeEnv).
_GEMM_TEMPLATE_SYMBOL_NAMES = OrderedSet(
    [
        "m_start",
        "m_end",
        "n_start",
        "n_end",
        "k_start",
        "k_end",
        "m_size",
        "n_size",
        "m_offset",
        "m_start_unsliced",
        "m_end_unsliced",
        "m_size_unsliced",
    ]
)


# Threshold above which we skip sympy reasoning that scales poorly in the
# number of symbols. Past this point, polynomial-domain conversions inside
# sympy (e.g. gcd, Mod) dominate compile time on wide concat/sum expressions
# (see https://github.com/sympy/sympy/issues/28200). Chosen empirically from
# AOT-partitioned bwd graphs with ~60-variable shape expressions.
_MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS = 20


@functools.lru_cache
def stride_at(index: sympy.Expr, var: sympy.Symbol):
    # [FUNCTION DOC] Cached. Stride of index with respect to var: zero if index does not depend on var,
    # otherwise the simplified difference of index with var replaced by var plus one, minus index.
    if not index.has(var):
        # see test_torchinductor_dynamic_shapes.py::test_full_boolean_dynamic_shapes_cpu
        # which has tmp0 = ops.index_expr(s0 >= 1024, torch.bool) and fails below calculation.
        # in this case, there is no dependencies between index and var.
        return sympy.S.Zero
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)  # type: ignore[arg-type]
    return sympy.simplify(new_index - index)


@functools.lru_cache
def simplify_index_in_vec_range(index: sympy.Expr, var: sympy.Expr, vec_length: int):
    # [FUNCTION DOC] Cached. Simplify an index for analysis (not codegen) inside a vectorized loop of
    # vec_length lanes: floor-division and modular-indexing terms of var whose divisor is a multiple of
    # vec_length become fresh free symbols; modular indexing with divisor 1 and modulus a multiple
    # becomes var plus a fresh symbol. Applies simplification and recurses until a fixed point.
    # Returns the new expression.
    """
    Simplifies the index expression within the range of a vectorized loop.
    Given a vectorized loop variable `var` in the range of a loop with `vec_length`,
    this function transforms the `index` into an equivalent form. It handles
    simplifications for cases where `var` can be expressed as `vec_length * a + b`,
    where `b` ranges from 0 to `vec_length - 1`. The function reduces occurrences
    of `FloorDiv` and `ModularIndexing` in the `index` with best-effort optimizations.

    NOTE:
    The simplified index expression is intended for analysis purposes only, not
    for code generation. It replaces `FloorDiv` and `ModularIndexing` with free variables
    which are not dependent on the loop variable `var` in the vectorized range. Check
    https://github.com/pytorch/pytorch/pull/117221#discussion_r1449746217 for more details.

    Examples:
    1. If `var` is `x3` and `vec_length` is 16, and `x3 = 16*a + b`, then
       `FloorDiv(x3, div)` or `ModularIndexing(x3, div, mod)` becomes a free variable
       when `div` is divisible by 16.
    2. `ModularIndexing(x3, 1, mod)` can be simplified to `x3 + c` where `c` is a free
       variable when `mod` is divisible by 16.
    """

    div_freevar_id = 0
    mod_freevar_id = 0

    def visit_indexing_div(divisor):
        # [FUNCTION DOC] HELPER: Replacement for floor division of var by divisor: a fresh symbol if divisor is a
        # multiple of vec_length, otherwise the original term. Updates a counter.
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        # [FUNCTION DOC] HELPER: Replacement for modular indexing of var: fresh symbol if divisor is a multiple of
        # vec_length; var plus fresh symbol if divisor is 1 and modulus is a multiple; else unchanged.
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div)

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing)

    if not index.has(sympy.Rel):
        index = sympy.simplify(index)
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index


@functools.lru_cache
def stride_at_vec_range(
    index: sympy.Expr, var: sympy.Symbol, vec_length: int | None = None
):
    # [FUNCTION DOC] Cached. Stride of index with respect to var, first simplifying the index for a
    # vectorized range when vec_length is truthy.
    if vec_length:
        index = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index, var)


def statically_known_true(
    shape_env: ShapeEnv,
    expr: sympy.Basic | bool,
    axioms: tuple[sympy.Expr] | None = None,
    var_to_range: tuple[tuple[sympy.Symbol, ValueRanges[Any]]] | None = None,
) -> bool:
    # [FUNCTION DOC] Best-effort check that a sympy expression is always true. Plain booleans return as is.
    # If substituting backed hints makes it false, returns False early. Otherwise asks the shape env's
    # static evaluator (with optional axioms and var ranges); any failure is logged at debug level and
    # gives False. Never adds guards.
    if expr in (True, False):
        return bool(expr)

    # Hint fast-path: if the current concrete hints make `expr` evaluate to
    # False, it cannot be universally True, so bail out without running the
    # much more expensive `_maybe_evaluate_static`. (A True hint doesn't let
    # us conclude universal truth, so we still fall through in that case.)
    try:
        hinted = expr.xreplace(shape_env.backed_var_to_val)
        if hinted is sympy.S.false:
            return False
    except Exception:
        pass

    try:
        simplified = shape_env._maybe_evaluate_static(
            expr,
            axioms=axioms,
            var_to_range=var_to_range,
        )
        if simplified is not None:
            return bool(simplified)
    except Exception:
        log.debug("Could not simplify  %s", expr, exc_info=True)

    return False


# This class is a little awkward, because ShapeEnv is doing most of the heavy
# lifting and in some cases we should be directly passing through to ShapeEnv,
# but there is some extra inductor logic that needs to be handled here
class SizeVarAllocator:
    # [CLASS DOC] Inductor-side manager of symbolic sizes on top of a ShapeEnv (created if none given).
    # State: shape_env and aliases to its hints and replacements, unbacked replacements, precomputed
    # replacements (host-precomputed expressions mapped to kernel arg symbols, which cannot be guarded
    # and so are inverted when guarding) plus inverse, and cached stride/simplify callables. Methods
    # simplify expressions, answer statically-known queries without guards, guard, and evaluate hints.
    """
    A class that manages symbolic size variables and their relationships.

    This class works with the ShapeEnv to handle symbolic shape expressions,
    simplify them, and provide utilities for guarding, checking, and evaluating
    symbolic expressions. It also manages precomputed replacements and stride
    calculations for tensor operations.
    """

    def __init__(self, shape_env=None) -> None:
        # [FUNCTION DOC] Set up from a shape env (a new one if None; reasoning needs existing var ranges), alias
        # its hints and replacements, init empty precomputed replacement maps, and build the stride, range
        # simplification and loop simplification caches.
        super().__init__()
        # Note: this can lead to bugs. Reasoning APIs depends on existing information in
        # in the shape_env. For example! var_to_ranges can't be empty!
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.backed_var_to_val = self.shape_env.backed_var_to_val
        self.var_to_hint_override = self.shape_env.var_to_hint_override
        self.replacements: dict[sympy.Symbol, Expr] = self.shape_env.replacements
        self.unbacked_replacements: dict[Expr, Expr] | None = None
        # Maps of dynamic sizes that have to be precomputed on the host to the kernel args.
        # The basic idea is if we have some complicated sympy expression
        # f(s0), we may choose to precompute it on the host and then replace
        # all occurrences of that sympy expression with ps0, so that when we
        # codegen we simply reference ps0 directly without repeating
        # f(s0).  Unlike regular size variables, ps variables cannot be
        # guarded upon; so if we are asked to guard on a Sympy expression
        # which potentially could have already had a precomputed replacement
        # on it, we are obligated to invert the precomputed replacements
        # (inv_precomputed_replacements).
        self.precomputed_replacements: dict[Expr, sympy.Symbol] = {}
        self.inv_precomputed_replacements: dict[sympy.Symbol, Expr] = {}
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def simplify(self, expr: Expr):
        # [FUNCTION DOC] Expand an expression and substitute the shape env replacements.
        return sympy.expand(expr).xreplace(self.replacements)

    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        # [FUNCTION DOC] Return a cached wrapper of the range-based simplifier keyed by expression and ranges;
        # the cache is cleared when the replacement count changes, and the result is also cached as its own
        # fixed point.
        # KNOWN ISSUE: the inherited docstring names a private simplify function but the method builds the
        # range simplification cache.
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: dict[tuple[Any, ...], Expr] = {}
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges) -> Expr:
            # [FUNCTION DOC] HELPER: Cached lookup; invalidates cache on new replacements, else computes and stores.
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (expr, *var_ranges.items())
            result = cache.get(key)
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
                if result != expr:
                    cache[(result, *var_ranges.items())] = result
            return result

        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        # [FUNCTION DOC] Return a cached wrapper of the loop simplifier keyed by index vars, sizes and formulas,
        # invalidated when the number of replacements changes.
        # KNOWN ISSUE: docstring says it caches the range simplifier, but it caches the loop simplifier.
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: dict[tuple[Any, ...], Any] = {}
        replacement_count = len(self.replacements)

        def simplify_loops(index_vars, sizes, index_formulas):
            # [FUNCTION DOC] HELPER: Cached lookup of the loop simplification implementation result.
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (*index_vars, *sizes, *index_formulas)
            result = cache.get(key)
            if result is None:
                result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
                cache[key] = result
            return result

        return simplify_loops

    def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges) -> Expr:
        # [FUNCTION DOC] Simplify an indexing expression knowing iteration variable ranges. Joins dimensions,
        # builds variable ranges and axioms, then rewrites modular-indexing and floor-division terms:
        # dropping zero terms, removing the mod or division when the base is statically within bounds, and
        # merging nested modular indexing. Recurses until the expression stops changing. Returns the result.
        """
        Simplify indexing expression with knowledge of the ranges of
        iteration variables.
        """

        expr = join_dimensions(self.simplify(expr))
        original_expr = expr

        var_to_range = dict(self.shape_env.var_to_range)
        var_to_range.update(
            {
                k: ValueRanges(
                    0, max(0, v - 1) if not has_free_symbols([v]) else IntInfinity()
                )
                for k, v in var_ranges.items()
            }
        )
        for var in expr.free_symbols:
            if var not in var_to_range:
                var_to_range[var] = ValueRanges(0, IntInfinity())

        var_to_range_tuple = cast(
            tuple[tuple[sympy.Symbol, ValueRanges[sympy.Expr]]],
            tuple(var_to_range.items()),
        )

        axioms = []
        for var, upper_bound in var_ranges.items():
            axioms.append(0 <= var)
            axioms.append(var < upper_bound)
        axioms = tuple(axioms) + self.shape_env.get_axioms()

        def statically_known(expr):
            # [FUNCTION DOC] HELPER: True if the shape env statically proves the expression under the ranges/axioms.
            evaluated = self.shape_env._maybe_evaluate_static(
                expr,
                # pyrefly: ignore [bad-argument-type]
                axioms=axioms,
                var_to_range=var_to_range_tuple,
            )
            return bool(evaluated)

        def remove_zero_terms(base, divisor):
            # [FUNCTION DOC] HELPER: Drop additive terms of a non-negative base that are statically smaller than the
            # gcd with the divisor (they cannot affect the quotient/modulus).
            """Symbols smaller than the divisor are zero"""
            if not statically_known(base >= 0):
                return base

            for v in base.free_symbols:
                if v in var_ranges:
                    rest = sympy.Wild("_rest", exclude=[v])
                    m = base.match(v + rest)
                    if m and v not in m[rest].free_symbols:
                        gcd = safe_gcd(m[rest], divisor)
                        if statically_known(v < gcd):
                            base = m[rest]
            return base

        def visit_indexing_div(base, divisor):
            # [FUNCTION DOC] HELPER: Rewrite floor division: zero if the base is within [0, divisor); merge with an
            # inner modular indexing when divisible; otherwise unchanged.
            base = remove_zero_terms(base, divisor)
            if statically_known(base >= 0) and statically_known(base < divisor):
                return sympy.S.Zero
            # FloorDiv(ModularIndexing(b, d1, m), d2) = ModularIndexing(b, d1*d2, m//d2)
            if isinstance(base, ModularIndexing) and isinstance(divisor, sympy.Integer):
                b, d1, m = base.args
                if m % divisor == 0:
                    return ModularIndexing(b, d1 * divisor, FloorDiv(m, divisor))
            return FloorDiv(base, divisor)

        def visit_modular_indexing(base, divisor, modulus):
            # [FUNCTION DOC] HELPER: Rewrite modular indexing: merge with inner modular indexing for constant
            # positive arguments when periods align; floor division if the base is in [0, modulus*divisor);
            # otherwise unchanged.
            base = remove_zero_terms(base, divisor)

            if isinstance(base, ModularIndexing):
                inner_base, inner_divisor, inner_modulus = base.args
                period = divisor * modulus
                # Target use cases have constant indices; avoid symbolic compile-time analysis.
                if all(
                    isinstance(value, sympy.Integer) and value > 0
                    for value in (inner_modulus, divisor, modulus)
                ) and self.statically_known_multiple_of(inner_modulus, period):
                    return ModularIndexing(inner_base, inner_divisor * divisor, modulus)

            can_remove_mod = statically_known(base >= 0) and statically_known(
                base < modulus * divisor
            )

            if can_remove_mod:
                return FloorDiv(base, divisor)
            return ModularIndexing(base, divisor, modulus)

        if expr.has(ModularIndexing):
            expr = expr.replace(
                ModularIndexing(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                    sympy.Wild("modulus", integer=True),
                ),
                visit_modular_indexing,
            )

        if expr.has(FloorDiv):
            expr = expr.replace(
                FloorDiv(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                ),
                visit_indexing_div,
            )

        if expr != original_expr:
            return self._simplify_with_ranges(expr, var_ranges)
        return expr

    def _simplify_loops_impl(
        self, index_vars: list[sympy.Symbol], sizes, index_formulas
    ):
        # [FUNCTION DOC] HELPER: Remove size-1 loop dimensions and fuse mergeable dimensions (checked by stride
        # match plus a sound substitution test). Returns (list of remaining sizes, reindex function
        # expanding new indices to the original ones with zeros for removed dims, prune function keeping
        # only surviving indices).
        # KNOWN ISSUE: docstring mentions a channel_last flag that this function does not take.
        """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
        sizes = list(map(self.simplify, sizes))

        strides = [
            # index_formulas may contain boolean expressions (e.g. s0 < 10),
            # for which "strides" don't make sense so we ignore them here.
            # NOTE: These expressions may still block merging dims in the sound
            # substitution test performed in can_merge_dims.
            (
                self.stride_vars(x, index_vars)
                if isinstance(x, sympy.Expr)
                else [0] * len(index_vars)
            )
            for x in index_formulas
        ]
        if len(sizes) != len(strides[0]):
            raise AssertionError((len(sizes), len(strides[0])))

        for i in range(len(sizes)):
            if sizes[i] == 1:
                # remove dim
                sizes[i] = None

        def can_merge_dims(a, b):
            # [FUNCTION DOC] HELPER: True if dimension a can be fused into b for all index formulas.
            for k in range(len(strides)):
                if self.simplify(strides[k][a] * sizes[a]) == self.simplify(
                    strides[k][b]
                ):
                    # approximate test passed, try sound version
                    va = index_vars[a]
                    vb = index_vars[b]
                    m1 = sympy_index_symbol("_merge_tester1")
                    m2 = sympy_index_symbol("_merge_tester2")
                    # NOTE: can't sub vb=0 here in case va * vb appears in the expression,
                    # in which case both expr1 and expr2 would be zero!
                    expr1 = sympy_subs(index_formulas[k], {va: m1 * sizes[a], vb: m2})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: (m1 + m2)})
                    if self.simplify(expr1) == self.simplify(expr2):
                        continue
                return False
            return True

        changed = True
        while changed:
            changed = False
            for i, j in itertools.product(
                reversed(range(len(sizes))), reversed(range(len(sizes)))
            ):
                if i == j or sizes[i] is None or sizes[j] is None:
                    continue
                if can_merge_dims(i, j):
                    changed = True
                    sizes[i] = sizes[i] * sizes[j]
                    sizes[j] = None

        def reindex(index):
            # [FUNCTION DOC] HELPER: Map indices of the reduced loop to the original dims (zero for removed ones);
            # asserts all given indices are consumed.
            it = list(reversed(index))
            new_index = []
            for size in sizes:
                if size is None:
                    new_index.append(sympy.S.Zero)
                else:
                    new_index.append(it.pop())
            if it:
                raise AssertionError(f"expected all entries consumed, got {it}")
            return new_index

        def prune(index):
            # [FUNCTION DOC] HELPER: Keep only entries of an original-rank index list for surviving dims;
            # asserts the length matches the original sizes.
            if len(index) != len(sizes):
                raise AssertionError(
                    f"expected len(index) == len(sizes), got {len(index)} != {len(sizes)}"
                )
            return [i for i, s in zip(index, sizes) if s is not None]

        return [x for x in sizes if x is not None], reindex, prune

    # Note - [On Statically Known]
    # The statically_known_* family of functions below NEVER guard, they could return True if the
    # asked questions can be answered without guarding otherwise they return False.
    # Those are similar to statically_known_true in symbolic_shapes.py but operate on sympy
    # expressions instead of symnodes.
    def statically_known_true(self, expr: sympy.Basic | bool) -> bool:
        # [FUNCTION DOC] True if the expression is provably true without adding guards; never throws data
        # dependent errors. Delegates to the module-level function with this shape env.
        """
        Returns true if an expression is always true (symbolically or via guards),
        false otherwise. Never add guards, or throw data dependent errors.
        """
        return statically_known_true(self.shape_env, expr)

    def statically_known_equals(self, left: Expr | int, right: Expr | int) -> bool:
        # [FUNCTION DOC] True if it is statically known that left equals right.
        """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
        return self.statically_known_true(sympy.Eq(left, right))  # type: ignore[arg-type]

    def statically_known_list_equals(
        self, left: Sequence[Expr], right: Sequence[Expr]
    ) -> bool:
        # [FUNCTION DOC] True if lists have equal length and each pair is statically known equal.
        """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
        return len(left) == len(right) and all(
            self.statically_known_equals(l, r) for l, r in zip(left, right)
        )

    def statically_known_leq(self, left: Expr, right: Expr | int) -> bool:
        # [FUNCTION DOC] True if left <= right is provable without guards.
        """
        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.
        """
        expr = left <= right
        return self.statically_known_true(expr)

    def statically_known_geq(self, left: Expr, right: Expr | int) -> bool:
        # [FUNCTION DOC] True if left >= right is provable without guards.
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right.
        """
        expr = left >= right
        return self.statically_known_true(expr)

    def statically_known_lt(self, left: Expr, right: Expr | int) -> bool:
        # [FUNCTION DOC] True if left < right is provable without guards.
        """
        Returns a bool indicating if it is sound to optimize as if left is less than right.
        """
        expr = left < right
        return self.statically_known_true(expr)

    def statically_known_gt(self, left: Expr, right: Expr | int) -> bool:
        # [FUNCTION DOC] True if left > right is provable without guards.
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than right.
        """
        expr = left > right
        return self.statically_known_true(expr)

    def _is_multiple_of(self, numerator: Expr, denominator: int) -> bool:
        # [FUNCTION DOC] HELPER: Structural divisibility test: True only if numerator is provably a multiple of
        # the integer denominator. Rules in order: concrete integer; product with a divisible factor or
        # constant; sum of divisible terms; floor division and mod by constants; a cheap gcd check; and,
        # unless the expression has too many symbols, a static check that the mod is zero.
        """
        Structural divisibility check: returns True only if numerator is
        provably a multiple of denominator.  Recurses over sympy expression
        structure before falling back to statically_known_true.
        """
        # Rule 1 — concrete value
        if isinstance(numerator, (int, sympy.Integer)):
            return int(numerator) % denominator == 0

        # Rule 2 — product: any factor divisible → product divisible
        if isinstance(numerator, sympy.Mul):
            for factor in numerator.args:
                if self._is_multiple_of(factor, denominator):
                    return True
            const = 1
            for factor in numerator.args:
                if isinstance(factor, (int, sympy.Integer)):
                    const *= int(factor)
            if const != 1 and const % denominator == 0:
                return True

        # Rule 3 — sum: all terms divisible → sum divisible
        if isinstance(numerator, sympy.Add):
            if all(self._is_multiple_of(term, denominator) for term in numerator.args):
                return True

        # Rule 4 — FloorDiv(a, b): if a is multiple of b*n
        if isinstance(numerator, FloorDiv):
            a, b = numerator.args
            if isinstance(b, (int, sympy.Integer)):
                if self._is_multiple_of(a, int(b) * denominator):
                    return True

        # Rule 5 — Mod(a, b): Mod(a,b) = a - b*floor(a/b), so if both a and b
        # are multiples of n, then Mod(a,b) is too.
        if isinstance(numerator, (Mod, sympy.Mod)):
            a, b = numerator.args
            if self._is_multiple_of(a, denominator) and self._is_multiple_of(
                b, denominator
            ):
                return True

        # Rule 6 — cheap gcd check before expensive sympy fallback.
        gcd = simple_floordiv_gcd(numerator, sympy.Integer(denominator))
        if isinstance(gcd, (int, sympy.Integer)) and int(gcd) % denominator == 0:
            return True

        # Rule 7 — full sympy fallback (expensive on many-variable exprs).
        if len(free_symbols(numerator)) > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS:
            return False
        expr = sympy.Eq(Mod(numerator, denominator), 0)
        return self.statically_known_true(expr)

    def statically_known_multiple_of(
        self, numerator: Expr, denominator: Expr | int
    ) -> bool:
        # [FUNCTION DOC] True if numerator is provably a multiple of denominator. Identical expressions and
        # zero numerators succeed; integer denominators use the structural check; symbolic denominators
        # give up beyond the symbol limit, otherwise try gcd-based cancellation (ignoring polynomial
        # errors) and finally a static check that the mod is zero.
        """
        Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.
        """
        # Use cheap structural equality, not statically_known_equals —
        # the latter constructs sympy.Eq(wide, wide) which has measurable
        # overhead (~30s on 64-variable expressions).
        if numerator is denominator or numerator == denominator:
            return True

        if isinstance(denominator, (int, sympy.Integer)):
            return self._is_multiple_of(numerator, int(denominator))

        if numerator == 0:
            return True

        # Symbolic denominator: Mod-based reasoning can miss factorable Add
        # denominators even for small expressions, so prove cancellation
        # directly under the same limit used for the sympy fallback below.
        if (
            len(free_symbols([numerator, denominator]))
            > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS
        ):
            return False

        def gcd_covers_denominator(gcd: sympy.Basic) -> bool:
            # [FUNCTION DOC] HELPER: True if gcd is not 1 and equals the denominator (directly or after simplify).
            return gcd != 1 and (
                gcd == denominator or self.simplify(gcd - denominator) == 0
            )

        try:
            if gcd_covers_denominator(simple_floordiv_gcd(numerator, denominator)):
                return True
            if gcd_covers_denominator(safe_gcd(numerator, denominator)):
                return True
        except sympy.PolynomialError:
            pass

        expr = sympy.Eq(Mod(numerator, denominator), 0)
        return self.statically_known_true(expr)  # type: ignore[arg-type]

    def statically_known_power_of_2(self, expr: Expr) -> bool:
        # [FUNCTION DOC] True only for a sympy Integer that is a power of two.
        """
        Returns a bool indicating if x is known to be a power of 2.
        """
        return isinstance(expr, sympy.Integer) and is_power_of_2(int(expr))

    def analyze_lane_contiguity(
        self, expr: Expr, lane_var: sympy.Symbol
    ) -> LaneContiguity:
        # [FUNCTION DOC] Conservatively classify an index expression over vector lanes of lane_var, returning
        # a LaneContiguity. Simplifies first; no dependence on the lane var or stride 0 is uniform for any
        # width; stride 1 is contiguous for any width; sums, products, floor division, modular indexing and
        # mod are delegated to specialised analyzers; everything else is unknown.
        """Analyze whether `expr` is uniform or contiguous over vector lanes.

        The analysis is conservative: unknown cases must fall back to scalar or
        gather-style lowering. It only describes the lane pattern of the index
        expression; callers must still prove tensor stride, offset, dtype, and
        alignment before emitting a vector memory load.
        """
        expr = self.simplify(expr)
        if lane_var not in expr.free_symbols:
            return LaneContiguity(stride=0, uniform_width=int_oo)
        stride = stride_at(expr, lane_var)
        match expr:
            case _ if self.statically_known_equals(stride, 1):
                return LaneContiguity(contiguous_width=int_oo, stride=1)
            case _ if self.statically_known_equals(stride, 0):
                return LaneContiguity(stride=0, uniform_width=int_oo)
            case sympy.Add():
                return self._analyze_lane_contiguity_add(expr.args, lane_var)
            case sympy.Mul():
                return self._analyze_lane_contiguity_mul(expr.args, lane_var)
            case FloorDiv():
                return self._analyze_lane_contiguity_floor_div(
                    expr.args[0], expr.args[1], lane_var
                )
            case ModularIndexing():
                return self._analyze_lane_contiguity_modular_indexing(expr, lane_var)
            case _ if isinstance(expr, (Mod, sympy.Mod)):
                return self._analyze_lane_contiguity_mod(
                    expr.args[0], expr.args[1], lane_var
                )
            case _:
                return LaneContiguity(unknown=True)

    def _analyze_lane_contiguity_add(
        self, args: tuple[Expr, ...], lane_var: sympy.Symbol
    ) -> LaneContiguity:
        # [FUNCTION DOC] HELPER: Combine terms of a sum: any unknown term or more than one lane-varying term is
        # unknown; otherwise uniform widths are min-combined and the varying term's contiguity is limited
        # by that width.
        """Combine additive terms when at most one term varies across lanes."""
        uniform_width: Width | None = int_oo
        varying_result: LaneContiguity | None = None
        for arg in args:
            arg_result = self.analyze_lane_contiguity(arg, lane_var)
            if arg_result.unknown:
                return LaneContiguity(unknown=True)
            if arg_result.uniform:
                uniform_width = _min_width(uniform_width, arg_result.uniform_width)
            elif varying_result is None:
                varying_result = arg_result
            else:
                return LaneContiguity(unknown=True)
        if varying_result is None:
            return LaneContiguity(stride=0, uniform_width=uniform_width)
        return LaneContiguity(
            contiguous_width=_min_width(varying_result.contiguous_width, uniform_width),
            stride=varying_result.stride,
        )

    def _analyze_lane_contiguity_mul(
        self, args: tuple[Expr, ...], lane_var: sympy.Symbol
    ) -> LaneContiguity:
        # [FUNCTION DOC] HELPER: Product analysis: exactly one lane-dependent factor allowed. The result stride is
        # the uniform factor times the child stride; stride 0 is uniform, stride 1 keeps contiguity, other
        # strides carry no contiguous width. Unknown otherwise.
        """Propagate lane stride through lane-uniform multiplication."""
        uniform_factor: Expr = sympy.S.One
        lane_factors = []
        for arg in args:
            if lane_var in arg.free_symbols:
                lane_factors.append(arg)
            else:
                uniform_factor *= arg
        if len(lane_factors) != 1:
            return LaneContiguity(unknown=True)
        child_result = self.analyze_lane_contiguity(lane_factors[0], lane_var)
        if child_result.unknown or child_result.stride is None:
            return LaneContiguity(unknown=True)
        stride = self.simplify(uniform_factor * child_result.stride)
        if self.statically_known_equals(stride, 0):
            return LaneContiguity(
                stride=0,
                uniform_width=child_result.uniform_width,
            )
        return LaneContiguity(
            contiguous_width=child_result.contiguous_width
            if self.statically_known_equals(stride, 1)
            else None,
            stride=stride,
        )

    def _analyze_lane_contiguity_modular_indexing(
        self, expr: Expr, lane_var: sympy.Symbol
    ) -> LaneContiguity:
        # [FUNCTION DOC] HELPER: Modular indexing with a constant divisor other than 1 is treated as floor
        # division; divisor 1 is treated as mod; non-constant divisor is unknown.
        """Analyze ModularIndexing(base, divisor, modulus)."""
        base, divisor, modulus = expr.args
        if not isinstance(divisor, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        if int(divisor) != 1:
            return self._analyze_lane_contiguity_floor_div(base, divisor, lane_var)
        return self._analyze_lane_contiguity_mod(base, modulus, lane_var)

    def _analyze_lane_contiguity_floor_div(
        self, base: Expr, divisor: Expr, lane_var: sympy.Symbol
    ) -> LaneContiguity:
        # [FUNCTION DOC] HELPER: Largest width for which base floor-divided by a positive constant divisor is
        # lane-uniform: requires base to be unit-stride contiguous; tries power-of-two widths (bounded by
        # divisor factor and base contiguity), halving until the lane-0 group start is a multiple of the
        # width. Otherwise unknown.
        """Return the largest width where ``base // divisor`` is lane-uniform."""
        if not isinstance(divisor, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        divisor_int = int(divisor)
        if divisor_int <= 0:
            return LaneContiguity(unknown=True)
        base_result = self.analyze_lane_contiguity(base, lane_var)
        if not base_result.is_contiguous_for(2) or not self.statically_known_equals(
            base_result.stride, 1
        ):
            return LaneContiguity(unknown=True)
        group_start = self.simplify(base.xreplace({lane_var: sympy.Integer(0)}))
        width = _min_width(
            base_result.contiguous_width,
            _largest_power_of_2_factor(divisor_int),
        )
        while isinstance(width, int) and width >= 2:
            if self.statically_known_multiple_of(group_start, width):
                return LaneContiguity(stride=0, uniform_width=width)
            width //= 2
        return LaneContiguity(unknown=True)

    def _analyze_lane_contiguity_mod(
        self,
        base: Expr,
        modulus: Expr,
        lane_var: sympy.Symbol,
    ) -> LaneContiguity:
        # [FUNCTION DOC] HELPER: Largest aligned no-wrap width for base mod a positive constant modulus: base must
        # be unit-stride contiguous; tries power-of-two widths bounded by the modulus factor and base
        # contiguity, halving until the group start is a multiple. Returns contiguous with stride 1, or
        # unknown.
        """Return the largest aligned no-wrap modulo span.

        `base % modulus` is contiguous for a vector group only when both the
        group start and modulus are multiples of the chosen width. For example,
        lanes 0..3 under `% 4` are contiguous, but lanes 2..5 wrap to 2,3,0,1.
        """
        if not isinstance(modulus, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        modulus_int = int(modulus)
        if modulus_int <= 0:
            return LaneContiguity(unknown=True)
        base_result = self.analyze_lane_contiguity(base, lane_var)
        if not base_result.is_contiguous_for(2) or not self.statically_known_equals(
            base_result.stride, 1
        ):
            return LaneContiguity(unknown=True)
        group_start = self.simplify(base.xreplace({lane_var: sympy.Integer(0)}))
        width = _min_width(
            base_result.contiguous_width,
            _largest_power_of_2_factor(modulus_int),
        )
        while isinstance(width, int) and width >= 2:
            if self.statically_known_multiple_of(group_start, width):
                return LaneContiguity(contiguous_width=width, stride=1)
            width //= 2
        return LaneContiguity(unknown=True)

    # The expect/check functions require you to ALREADY KNOW that a particular
    # condition holds. They are similar to expect_true in symbolic_shapes.py and
    # torch.check but operates on sympy expressions instead of symnodes.
    def expect_true(self, expr: Expr) -> bool:
        # [FUNCTION DOC] Assert-by-guard: if expr is not statically true, defers a runtime assertion through the
        # shape env (returns its result); returns True if already known. Does not raise on a false expr.
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike check, this WON'T raise an error if expr isn't actually true.
        check Note [expect_true].
        """
        if not self.statically_known_true(expr):
            return self.shape_env.guard_or_defer_runtime_assert(
                expr, "sizevars.expect_true"
            )
        return True

    def check(self, expr: Expr) -> None:
        # [FUNCTION DOC] Substitutes inverse precomputed replacements into expr, then requires expect_true to
        # hold; raises AssertionError otherwise. Side effect: may add guards or runtime asserts.
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike expect_true, this WILL raise an error if expr isn't actually true.
        check Note [expect_true].
        """
        expr = sympy_subs(expr, self.inv_precomputed_replacements)
        if not self.expect_true(expr):
            raise AssertionError(f"expect_true failed for {expr}")

    def check_equals(self, left: Expr, right: Expr) -> None:
        # [FUNCTION DOC] Check that left equals right (see check), then return left.
        # KNOWN ISSUE: annotated as returning None but returns left.
        """
        check(sympy.Eq(left, right)).

        """
        self.check(sympy.Eq(left, right))
        return left

    def check_equals_and_simplify(self, left: Expr, right: Expr) -> Expr:
        # [FUNCTION DOC] Check that left equals right, then return left with inverse precomputed replacements applied.
        """
        check(sympy.Eq(left, right)) and returns left after applying
        inv_precomputed_replacements.
        """
        self.check(sympy.Eq(left, right))
        return sympy_subs(left, self.inv_precomputed_replacements)

    def check_leq(self, left: Expr, right: Expr) -> None:
        # [FUNCTION DOC] Check that left <= right (raises AssertionError if it cannot be asserted).
        self.check(sympy.Le(left, right))

    def check_lt(self, left: Expr, right: Expr) -> None:
        # [FUNCTION DOC] Check that left < right (raises AssertionError if it cannot be asserted).
        self.check(sympy.Lt(left, right))

    # Similar to the functions guard_or_false/guard_or_true in symbolic_shapes.py
    # but operates on sympy expressions instead of symnodes. see Note [guard_or_].
    def guard_or_false(self, left):
        # [FUNCTION DOC] Guard-or-false: under backed-size-oblivious mode returns the statically evaluated value or
        # False; otherwise evaluates the expression with fallback value False. Never specializes on unknowns.
        import torch.fx.experimental._config as exp_config

        if exp_config.backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return False
        return self.evaluate_expr(left, fallback_value=False)

    def guard_or_true(self, left):
        # [FUNCTION DOC] Guard-or-true: like guard_or_false but the fallback value is True.
        import torch.fx.experimental._config as exp_config

        if exp_config.backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return True
        return self.evaluate_expr(left, fallback_value=True)

    # The evaluate functions evaluate some symbolic sympy expression
    # (NB: not necessarily an Expr) and return what the concrete result
    # is, guarding on the expression being that result

    # NB: write evaluate_expr(sympy.Lt(a, b)) rather than evaluate_expr(a < b)
    # as this will ensure that you actually have a sympy'ified expression,
    # and will prevent you from incorrectly writing evaluate_expr(a == b)
    # which does the wrong thing if a or b is a sympy expression
    def evaluate_expr(
        self,
        left: Expr | sympy.logic.boolalg.Boolean,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
    ) -> bool:
        # [FUNCTION DOC] Evaluate a sympy expression (Expr or Boolean) to a concrete bool via the shape env,
        # guarding on the result; optional size-oblivious mode and fallback value. Raises AssertionError
        # on a non-sympy input.
        if not isinstance(left, (Expr, sympy.logic.boolalg.Boolean)):
            raise AssertionError(type(left))
        return self.shape_env.evaluate_expr(
            sympy.sympify(left),
            size_oblivious=size_oblivious,
            fallback_value=fallback_value,
        )

    def is_size_one_or_false(self, size: Expr) -> bool:
        # [FUNCTION DOC] True if size is known to be 1; unbacked sizes return False without guarding.
        """Return True if size equals 1.

        Unbacked symbolic sizes return False without introducing a guard.
        """
        return self.guard_or_false(sympy.Eq(size, 1))

    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        # [FUNCTION DOC] Return the smaller of left and right, guarding the choice. Undoes precomputed
        # replacements, tries guard-or-false comparisons both ways, then a gcd fallback (noted as unsound
        # for possibly negative or zero unbacked factors), then Min/Max structure rules. Raises TypeError
        # if undecidable.
        """Return the smaller of left and right, and guard on that choice."""
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if self.guard_or_false(sympy.Le(left, right)):
            return left
        if self.guard_or_false(sympy.Le(right, left)):
            return right

        # GCD fallback: if gcd(a, b) == a then a divides b, implying a <= b.
        #
        # TODO: This is NOT always sound for unbacked symints.  It can
        # produce wrong results when:
        #   - inputs can be negative: gcd(u0, 10*u0) = u0, returns u0,
        #     but if u0 < 0 then u0 > 10*u0 (e.g. u0=-1: min(-1,-10) = -10)
        #   - a factor can be zero: gcd(u0, u0*u1) = u0, returns u0,
        #     but if u1=0 then u0*u1=0 < u0 (e.g. u0=5,u1=0: min(5,0) = 0)
        # TODO shall we add a runtime assertion at least.
        gcd = sympy.gcd(left, right)
        if left == gcd:
            return left
        if right == gcd:
            return right

        # Min/Max fallback: we can prove Min(a, b) <= c when any arg <= c, but
        # sympy doesn't simplify this yet. So, evaluate it here. Same for Max.
        for lhs, rhs in [(left, right), (right, left)]:

            def le_rhs(a: Expr) -> bool:
                # [FUNCTION DOC] HELPER: Guard-or-false check that a is <= the current right-hand side.
                return self.guard_or_false(sympy.Le(a, rhs))

            # Min(Min(a, b), c) ==> Min(a, b) if (a <= c) or (b <= c).
            if isinstance(lhs, (sympy.Min, Min)) and any(le_rhs(a) for a in lhs.args):
                return lhs
            # Min(Max(a, b), c) ==> Max(a, b) if (a <= c) and (b <= c).
            if isinstance(lhs, (sympy.Max, Max)) and all(le_rhs(a) for a in lhs.args):
                return lhs

        raise TypeError(
            f"evaluate_min({left}, {right}) with unbacked symints"
        ) from None

    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        # [FUNCTION DOC] Return the larger of left and right, defined as the opposite choice of evaluate_min so
        # both produce the same guards.
        """Return the larger of left and right, and guard on that choice."""
        # Always choose the opposite of eval min for consistency
        # This means min(a, b) and max(a, b) produce the same guards
        min_val = self.evaluate_min(left, right)
        return right if min_val is left else left

    def guard_int(self, expr: Expr | int) -> int:
        # [FUNCTION DOC] Concretize expr to an int: ints pass through; otherwise take the guarding hint,
        # check equality (adding a guard) and return the int. Raises on unbacked/unhinted symbols.
        """
        Similar to guard_int in symbolic_shapes.py, except this function works with SymPy
        expressions instead of SymNodes. It extracts the value represented by expr from shapeEnv
        and specialize the compiled graph on it. Raises an error if the result cannot be
        determined due to unhinted or unbacked symbols.
        """
        if isinstance(expr, int):
            return expr
        val = self.guarding_hint_or_throw(expr)
        self.check_equals(expr, sympy.Integer(val))
        return int(val)

    def guard_int_seq(self, left: Sequence[Expr | int]) -> list[int]:
        # [FUNCTION DOC] Apply guard_int to each element, returning a list of ints.
        """
        Apply guard_int on a sequence of inputs.
        """
        return [self.guard_int(x) for x in left]

    def remove_precomputed_replacements(self, expr: Expr) -> Expr:
        # [FUNCTION DOC] If expr contains precomputed-size symbols, substitute the inverse replacements to restore
        # original expressions; otherwise return unchanged.
        if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
            return sympy_subs(expr, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        return expr

    def replace_backed_symbols_with_hints(
        self,
        expr: Expr | int,
    ) -> Expr | int:
        # [FUNCTION DOC] Last-resort helper: replace backed symbols with their hint values (unbacked left alone).
        # Ints pass through; simplifies, undoes precomputed replacements, substitutes hints and expands.
        # Returns an int if no free symbols remain (or the expression itself for inf/nan), else the
        # expression. Raises AssertionError if simplify gives a non-Expr non-int.
        """
        Replace all backed symbols in an expression with their concrete hint values.

        This function substitutes backed symbolic variables with their known concrete
        values from the shape environment, while leaving unbacked symbols (data-dependent
        values) untouched. Not recommended to be used except as last resort!
        was added for existing usages compatibility.
        """
        if isinstance(expr, int):
            return expr
        # Substitute all hints into expr, but leave unbacked symints alone
        expr = self.simplify(expr)
        if not isinstance(expr, Expr):
            if not isinstance(expr, int):
                raise AssertionError(f"expected int, got {type(expr)}")
            return expr

        expr = self.remove_precomputed_replacements(expr)
        expr = sympy_subs(expr, self.backed_var_to_val)
        if isinstance(expr, Expr):
            expr = expr.expand(identity=True)

        free_symbols = expr.free_symbols
        if not free_symbols:
            try:
                return int(expr)
            except TypeError:
                return expr  # inf/nan/I

        return expr

    def to_symint_or_int(self, expr: Expr | int) -> SymInt | int:
        # [FUNCTION DOC] Convert an expression to a SymInt (ints and integer constants returned as int). Undoes
        # precomputed replacements and attaches the guarding hint when available (None if it cannot be
        # computed).
        """Convert a sympy expression to SymInt, or return int as is."""
        if isinstance(expr, int):
            return expr

        from torch.fx.experimental.sym_node import SymNode

        expr = self.simplify(expr)
        if isinstance(expr, (int, sympy.Integer)):
            return int(expr)

        # Remove precomputed replacements to get canonical form
        # (consistent with symbolic_hint behavior)
        expr = self.remove_precomputed_replacements(expr)

        try:
            hint = self.guarding_hint_or_throw(expr)
        except Exception:
            hint = None
        node = SymNode(expr, self.shape_env, int, hint)
        return SymInt(node)

    def to_symints_or_ints(self, exprs: Sequence[Expr | int]) -> list[SymInt | int]:
        # [FUNCTION DOC] Convert each element with to_symint_or_int, returning a list.
        """Convert a sequence of sympy expressions to SymInts, or return ints as is."""
        return [self.to_symint_or_int(e) for e in exprs]

    def guarding_hint_or_throw(self, expr: Expr | int) -> int:
        # [FUNCTION DOC] Return a concrete int hint for guarding using only backed symbol hints; no heuristics for
        # unbacked symbols. Raises GuardOnDataDependentSymNode if unbacked symbols are involved.
        """
        Return a concrete integer hint for an expression that is safe to use for guarding.

        This method evaluates the expression using only backed-symbols hints. Unlike
        optimization_hint(), this method does NOT use heuristics or fallback values
        for unbacked symbols.

        Use this method when you need a hint value that will be used for guarding decision.

        Args:
            expr: A sympy expression or integer to evaluate.

        Returns:
            The concrete integer value of the expression based on backed symbol hints.

        Raises:
            GuardOnDataDependentSymNode: If the expression contains unbacked symbols
            (data-dependent values) that cannot be resolved to concrete values.

        See Also:
            optimization_hint: For cases where fallback/heuristic values are acceptable
                for unbacked symbols.
        """
        if isinstance(expr, SymNode):
            raise TypeError(
                f"guarding_hint_or_throw expects a sympy Expr or int, not {type(expr)}. "
                "Use expr.expr to extract the sympy expression from a SymNode."
            )
        return _guarding_hint_or_throw_base(
            self.shape_env, expr, self.inv_precomputed_replacements
        )

    def optimization_hint(self, expr: Expr | int, fallback: int | None = None) -> int:
        # [FUNCTION DOC] Concrete int hint for non-guarding optimizations; unbacked symbols use user hints or
        # the fallback with heuristics. Delegates to the shared hint routine using inverse precomputed
        # replacements. Complex values raise, infinity gives the max size, NaN gives the fallback.
        """
        Return a concrete integer hint for an expression.

        This function should be used for non-guarding based optimizations. If you
        want a hint that you can guard on, use the guarding_hint API instead.

        This function will hint unbacked symbols using user provided optimization
        hints. If not provided, fallback will be used along with some heuristics
        that try to maximize consistency with the shape environment.

        Special cases:
        - Complex numbers (containing sympy.I): raises an error since tensor
          dimensions cannot be complex.
        - Infinity (int_oo, sympy.oo): returns sys.maxsize.
        - NaN (sympy.nan): returns the fallback value.
        """
        return _optimization_hint_base(
            self.shape_env, expr, self.inv_precomputed_replacements, fallback
        )

    def optimization_hints(
        self,
        exprs: Iterable[Expr | int],
        fallback: int | None = None,
    ) -> tuple[int, ...]:
        # [FUNCTION DOC] Optimization hints for a sequence of expressions as a tuple; the fallback defaults to the
        # unbacked_symint_fallback config value read at call time.
        """
        Like optimization_hint but for a sequence of expressions.
        Returns a tuple of concrete integer hints.
        """
        # Read config at call time to respect runtime patches (e.g., in tests)
        if fallback is None:
            fallback = config.unbacked_symint_fallback
        return tuple(self.optimization_hint(x, fallback=fallback) for x in exprs)

    def all_unbacked_explicitly_hinted(self, exprs: IterateExprs) -> bool:
        # [FUNCTION DOC] True if every unbacked symbol in the expressions has an explicit hint override
        # (vacuously True when none).
        """
        Return True if every unbacked symbol in *exprs* has an explicit
        user-provided hint in var_to_hint_override.  If there are no
        unbacked symbols at all, returns True (vacuously).
        *exprs* can be a single expression or any iterable accepted by
        free_unbacked_symbols (list, tuple, etc.).
        """
        unbacked = free_unbacked_symbols(exprs)
        return unbacked.issubset(self.var_to_hint_override.keys())

    def optimization_hint_with_override(
        self,
        expr: Expr | int,
        hint_override: int | None,
    ) -> int:
        # [FUNCTION DOC] Hint with optional override: a statically known integer wins; else the override if
        # given; else the normal optimization hint.
        r"""Return a concrete integer hint for an expression, with optional override.
        This is used in dynamic dispatch scenarios where callers may want to
        provide a specific hint value rather than computing one from the expression.
        The resolution order is:
        1. If ``expr`` simplifies to a static integer, return that value
           (``hint_override`` is ignored for static shapes).
        2. If ``expr`` is dynamic and ``hint_override`` is not ``None``,
           return ``hint_override``.
        3. Otherwise, compute a hint via :meth:`optimization_hint`
        Args:
            expr (Expr or int): The expression to get a hint for.
            hint_override (int, optional): If provided and ``expr`` is dynamic,
                this value is returned instead of computing a hint.
        Returns:
            int: A concrete integer hint for the expression.
        """
        simplified = _maybe_realize_expr(self.simplify(expr), None)

        if simplified is not None:
            return simplified

        # Dynamic shape: use hint_override if set,
        # else return optimization_hint
        if hint_override is not None:
            return hint_override

        return self.optimization_hint(expr)

    def optimization_hints_with_override(
        self,
        exprs: Iterable[Expr | int],
        hint_override: int | None,
    ) -> tuple[int, ...]:
        # [FUNCTION DOC] Apply optimization_hint_with_override over a sequence, returning a tuple.
        """
        Like optimization_hint_with_override but for a sequence of expressions.
        Returns a tuple of concrete integer hints.
        """
        return tuple(
            self.optimization_hint_with_override(e, hint_override) for e in exprs
        )

    def guarding_hints_or_throw(
        self,
        exprs: Iterable[Expr | int],
    ) -> tuple[int, ...]:
        # [FUNCTION DOC] Guarding hints for a sequence of expressions as a tuple (raises as guarding_hint_or_throw).
        return tuple(self.guarding_hint_or_throw(x) for x in exprs)

    def _lru_cache(self, fn, maxsize=None):
        # [FUNCTION DOC] HELPER: Return an LRU-cached wrapper of fn whose cache is cleared whenever the number of
        # replacements changes.
        """
        Wrapper around functools.lru_cache that clears when replacements
        has been invalidated.
        """
        fn_cache = functools.lru_cache(maxsize)(fn)
        prior_len = len(self.replacements)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Clear the cache if replacements changed, then call the cached function.
            nonlocal prior_len
            if prior_len != len(self.replacements):
                prior_len = len(self.replacements)
                fn_cache.cache_clear()
            return fn_cache(*args, **kwargs)

        return wrapper

    def make_stride_vars_cache(self):
        # [FUNCTION DOC] Build a cached stride-computation function around the stride helper, with cache
        # invalidation on new replacements. Returns the callable.
        cache = self._lru_cache(self._stride_vars)

        def stride_vars(
            index: Expr,
            vars: Sequence[sympy.Symbol],
            support_vars: Sequence[sympy.Symbol] | None = None,
        ) -> list[Expr]:
            # [FUNCTION DOC] HELPER: Compute strides of index for vars; support_vars defaults to vars; arguments are
            # converted to tuples for caching.
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))

        return stride_vars

    def _stride_vars(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
    ) -> list[Expr]:
        # [FUNCTION DOC] HELPER: Convert an index expression into a list of per-variable strides: removes the
        # offset, isolates each variable, and takes the difference of the value at one and at zero. A zero
        # variable gets stride zero. Only valid for plain strided index calculations (wraps give odd
        # strides).
        """Convert an indexing expression back into strides

        NOTE: This is only valid if the index is a standard strided offset
        calculation. e.g. 10 * ModularIndexing(i0 + 1, 1, 2) would give a
        stride of -10 because the index wraps around after the first element

        """
        strides = []
        index = self.simplify(index)
        # remove any offset
        index = index - sympy_subs(
            index, {v: sympy.S.Zero for v in support_vars if v != 0}
        )
        for i in range(len(vars)):
            # drop all the other dims
            index_dim = sympy_subs(
                index,
                {
                    support_vars[j]: sympy.S.Zero
                    for j in range(len(support_vars))
                    if vars[i] != support_vars[j] and support_vars[j] != 0
                },
            )
            v = vars[i]
            if v == 0:
                strides.append(sympy.S.Zero)
            else:
                # TODO(jansel): should we use sympy.diff here?
                strides.append(
                    sympy_subs(index_dim, {v: sympy.S.One})
                    - sympy_subs(index_dim, {v: sympy.S.Zero})
                )
        return strides

    def offset_var(self, index: Expr, vars: Sequence[sympy.Symbol]) -> Expr:
        # [FUNCTION DOC] Return the offset part of an index: simplified expression with all vars set to zero.
        """Extract offset part of an indexing expression"""
        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.S.Zero for v in vars if v != 0})

    # Return stride optimizaitons hints,
    # only used for optimizations.
    def stride_hints(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] Stride hints for optimization only: replaces indirect-index symbols by zero, computes
        # strides, and converts each to an int hint (fallback 0). Returns a list of ints.
        for v in index.free_symbols:
            if symbol_is_type(v, SymT.INDIRECT):  # type: ignore[attr-defined]
                index = sympy_subs(index, {v: 0})  # type: ignore[dict-item]
        result = []
        for s in self.stride_vars(index, vars, support_vars):
            result.append(self.optimization_hint(s, fallback=0))
        return result

    def stride_order(self, index: Expr, vars: list[sympy.Symbol]) -> list[int]:
        # [FUNCTION DOC] Return indices of vars ordered by absolute stride hint ascending, with zero-stride
        # vars last.
        strides = tuple(map(abs, self.stride_hints(index, vars)))
        order = list(range(len(strides)))
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order

    def lookup_precomputed_size(self, expr: Expr) -> Expr:
        # [FUNCTION DOC] Return a precomputed-size symbol standing for expr, creating and recording the mapping
        # (both directions) on first use. Numbers and symbols are returned unchanged. Mutates the
        # precomputed replacement maps.
        if (
            isinstance(expr, (int, sympy.Symbol, sympy.Number))
            or expr.is_number
            or expr.is_symbol
        ):
            return expr
        expr = self.remove_precomputed_replacements(expr)
        if expr not in self.precomputed_replacements:
            sym = sympy_index_symbol_with_prefix(
                SymT.PRECOMPUTED_SIZE, len(self.precomputed_replacements)
            )
            self.precomputed_replacements[expr] = sym
            self.inv_precomputed_replacements[sym] = expr
        return self.precomputed_replacements[expr]

    def free_symbols(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Return the backed symbols that have no replacement, as an ordered set.
        return OrderedSet(self.backed_var_to_val.keys()) - OrderedSet(
            self.replacements.keys()
        )

    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Combine nested modular indexing of a non-negative symbol with both divisors 1 and positive
        # constant moduli where the inner modulus is a multiple of the outer one, into a single modular
        # indexing; otherwise return the index unchanged.
        """
        A pair of special ModularIndexing can be combined.

        E.g. ModularIndexing(ModularIndexing(x, 1, a), 1, b)
        We can simplify this to ModuleIndexing(x, 1, b), if
        1. x is non negative integer
        2. a and b are positive integers
        3. a is a multiple of b.
        """

        def _check_args(x, div, mod, is_first):
            # [FUNCTION DOC] HELPER: Validate one modular indexing layer: constant divisor 1, positive constant
            # modulus; the outer layer must wrap a modular indexing, the inner one a statically non-negative symbol.
            if not isinstance(div, sympy.Integer) or not isinstance(mod, sympy.Integer):
                return False
            if div != 1:
                return False
            if mod <= 0:
                return False

            if is_first:
                # first ModularIndexing should contains a nested ModularIndex
                if not isinstance(x, ModularIndexing):
                    return False
            else:
                # second ModularIndexing should contains a non-negative
                # symbol
                if not isinstance(x, sympy.Symbol) or not self.statically_known_geq(
                    x, 0
                ):
                    return False
            return True

        if isinstance(index, ModularIndexing):
            x, div, mod = index.args

            if not _check_args(x, div, mod, True):
                return index

            x2, div2, mod2 = x.args

            if not _check_args(x2, div2, mod2, False):
                return index

            if Mod(mod2, mod) != 0:
                return index

            return ModularIndexing(x2, 1, mod)

        return index

    def expand_floor_div(
        self,
        index: sympy.Expr,
        candidate_vars: Iterable[sympy.Symbol] | None = None,
    ) -> bool | tuple[sympy.Expr, sympy.Expr]:
        # [FUNCTION DOC] Try to factor a floor division over a sum so loops may merge: requires a sum of terms that
        # are non-negative symbol multiples with integer constant factors, plus exactly one floor division
        # of a symbol by a constant (optionally restricted to candidate vars). Returns False if not
        # applicable, else (new expression, denominator) meaning the original equals new // denominator.
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplified.

        E.g., for a 2D contiguous tensor with shape [a, 2 * b], and index variables
        x1, x2, index expression 'x1 * 2b + x2' can be easily combined.
        But index expression 'x1 * b + x2 // 2' can not.
        By expanding the FloorDiv to the entire expression, we get
        '(x1 * 2b + x2) // 2'. This transformation allows us to merge loops
        for the numerator!

        Return false if this optimization cannot be applied;
        Return the new expression and the denominator otherwise.
        The original expression will be equivalent to 'new_expression // denominator'
        """
        candidate_vars_set = (
            OrderedSet(candidate_vars) if candidate_vars is not None else None
        )

        def parse_static_term(
            term: sympy.Expr,
        ) -> tuple[sympy.Expr, sympy.Symbol] | None:
            # [FUNCTION DOC] HELPER: Parse a product of an integer constant and a symbol; None otherwise.
            if not isinstance(term, sympy.Mul):
                return None
            # For dynamic shape, term like '2*s1*x1' has 3 child nodes.
            # Without candidate loop vars we cannot tell shape and loop
            # symbols apart, so preserve the historical conservative path.
            if len(term.args) != 2:
                return None
            factor, var = term.args
            if not isinstance(factor, sympy.Integer) or not isinstance(
                var, sympy.Symbol
            ):
                return None
            return factor, var

        def parse_term(term: sympy.Expr) -> tuple[sympy.Expr, sympy.Symbol] | None:
            # [FUNCTION DOC] HELPER: Parse a term into (integer factor, non-negative symbol), using candidate vars to
            # find the loop variable when given; None if not of that shape.
            if candidate_vars_set is None:
                parsed = parse_static_term(term)
                if parsed is None:
                    return None
                factor, var = parsed
            else:
                if isinstance(term, sympy.Symbol):
                    factor = sympy.S.One
                    var = term
                elif isinstance(term, sympy.Mul):
                    vars_in_term = [
                        symbol
                        for symbol in term.free_symbols
                        if symbol in candidate_vars_set
                    ]
                    if len(vars_in_term) == 0:
                        parsed = parse_static_term(term)
                        if parsed is None:
                            return None
                        factor, var = parsed
                    elif len(vars_in_term) == 1:
                        var = vars_in_term[0]
                        factor = sympy.cancel(term / var)
                        if factor.has(*candidate_vars_set):
                            return None
                        if factor.is_integer is not True:
                            return None
                    else:
                        return None
                else:
                    return None

                if var not in candidate_vars_set:
                    return None

            # It's easier to reason about the correctness of the transformation
            # for non-negative integers.
            if not self.statically_known_geq(var, 0):
                return None
            return factor, var

        if not isinstance(index, sympy.Add):
            return False
        terms = index.args

        if len(terms) < 2:
            return False
        floor_div_index = -1
        varlist = []
        factorlist = []
        for idx, term in enumerate(terms):
            if parsed := parse_term(term):
                factor, var = parsed
                varlist.append(var)
                factorlist.append(factor)
            elif isinstance(term, FloorDiv):
                var, factor = term.args
                if not isinstance(factor, sympy.Integer) or not isinstance(
                    var, sympy.Symbol
                ):
                    return False
                if candidate_vars_set is not None and var not in candidate_vars_set:
                    return False
                if not self.statically_known_geq(var, 0):
                    return False
                if floor_div_index >= 0:
                    # can not handle multi FloorDiv yet
                    return False

                floor_div_index = idx
                varlist.append(var)
                # this factor is denominator
                factorlist.append(factor)
            else:
                return False

        if floor_div_index < 0:
            return False

        # Construct the new expression and remember the denominator
        denominator = factorlist[floor_div_index]
        new_index = sympy.S.Zero

        for var, factor, idx in zip(varlist, factorlist, itertools.count()):
            if idx == floor_div_index:
                new_index += var
            else:
                new_index += (factor * denominator) * var

        return new_index, denominator


def join_dimensions(expr: Expr) -> Expr:
    # [FUNCTION DOC] Fast-path wrapper: returns the expression unchanged unless it is a sum containing
    # modular indexing, in which case defers to the cached joiner.
    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr  # fast exit path
    return _join_dimensions_cached(expr)


@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    # [FUNCTION DOC] HELPER: Cached. Merge pairs of terms in a sum that come from view operations: two
    # modular-indexing terms with chained scale/modulus become one with the product modulus; a modular
    # indexing plus the matching floor division becomes a floor division. Recurses and returns the
    # simplified expression (unchanged if nothing matches). Raises AssertionError if not a sum.
    """
    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
    becomes
    ModularIndexing(i0, 1, 128)
    ModularIndexing(i0, 1, 32) + 32 * FloorDiv(i0, 32)
    becomes i0


    This type of pattern can come from view operations
    """
    if not isinstance(expr, sympy.Add):
        raise AssertionError(f"expected sympy.Add, got {type(expr)}")

    scale = sympy.Wild("scale", exclude=[0], integer=True)
    base = sympy.Wild("base", integer=True)
    divisor = sympy.Wild("divisor", integer=True)
    mod1 = sympy.Wild("modulus", integer=True)
    mod2 = sympy.Wild("modulus2", integer=True)
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale]
                    * m1[mod1]
                    * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2)
                )
                if m2 and term1 != term2:
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale]
                        * ModularIndexing(m1[base], m1[divisor], m1[mod1] * m2[mod2])
                    )
                    return expr
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1])
                )
                if m2 is not None:  # in case of success we get an empty dict here
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale] * FloorDiv(m1[base], m1[divisor])
                    )
                    return expr
    return expr


class SimplifyIndexing(V.WrapperHandler):  # type: ignore[name-defined]
    # [CLASS DOC] Ops handler wrapper that simplifies the index of load, store, store_reduction, index_expr
    # value_expr and check_bounds using variable range information from the graph size vars before
    # forwarding to the inner handler.
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """

    def __init__(self, inner, var_ranges: VarRanges) -> None:
        # [FUNCTION DOC] Wrap the inner handler and define the index simplifier using the current graph's
        # range-aware simplification with the given var ranges.
        super().__init__(inner)
        self.name = "SimplifyIndexing"
        self._simplify: Callable[[Expr], Expr] = (
            lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        )

    def load(self, name: str, index: sympy.Expr):
        # [FUNCTION DOC] Forward a load with a simplified index.
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        # [FUNCTION DOC] Forward a store with a simplified index.
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def store_reduction(self, name, index, value):
        # [FUNCTION DOC] Forward a store_reduction with a simplified index.
        return self._inner.store_reduction(name, self._simplify(index), value)

    def index_expr(self, index, dtype):
        # [FUNCTION DOC] Forward an index_expr with a simplified index.
        return self._inner.index_expr(self._simplify(index), dtype)

    def value_expr(self, index, dtype):
        # [FUNCTION DOC] Forward a value_expr with a simplified index.
        return self._inner.value_expr(self._simplify(index), dtype)

    def check_bounds(self, index, size, lower, upper):
        # [FUNCTION DOC] Forward a bounds check with a simplified index.
        return self._inner.check_bounds(self._simplify(index), size, lower, upper)
