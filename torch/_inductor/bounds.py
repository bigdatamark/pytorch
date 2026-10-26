# [MODULE DOC] Value range analysis for Inductor LoopBody graphs. BoundVars runs an
# fx interpreter over a loop body with ValueRangeAnalysis as the ops handler to compute
# symbolic ValueRanges for every node (loads, reductions and dependents are unknown);
# ValueRangeAnalysis is the ops handler producing conservative ranges. Only creates a
# module logger at import time.
import logging
import operator
from collections.abc import Callable
from functools import partial
from typing import Any

import sympy
from sympy import Expr

import torch
from torch.utils._sympy.value_ranges import (
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRanges,
)

from ..utils._sympy.functions import PowByNatural
from ..utils._sympy.numbers import int_oo
from .loop_body import InterpreterShim, LoopBody, LoopBodyBlock
from .ops_handler import DefaultHandler, ReductionType, StoreMode
from .utils import cache_on_self, dominated_nodes
from .virtualized import V


log = logging.getLogger(__name__)


class BoundVars:
    # [CLASS DOC] Computes value ranges for the nodes of one LoopBody. State: loop_body,
    # replacement_vals (mapping of symbols or index names to ValueRanges, starting with
    # each loop var in 0..extent-1 and updated during analysis), unbounded_vars (nodes
    # downstream of loads, reductions, getitem and masked subblocks, pessimistically
    # unknown) and _bounds (node to ValueRanges, filled by get_bounds). Analysis is per loop.
    """
    Performs Value Range Analysis on LoopBody's fx graph by calling BoundVars.run()
    It exposes the ranges of the nodes in the `bounds` variable

    Note. A current limitation of this analysis is that it just works on a per-loop basis.
    We should be able to propagate the bounds between across the whole graph. This may benefit
    the case a bounded variable is returned by a kernel and fed into another.
    """

    def __init__(self, loop_body: LoopBody) -> None:
        # [FUNCTION DOC] Stores the loop body, seeds replacement_vals with a range from 0 to
        # (upper bound of the extent minus 1) for each loop variable, and collects the set of
        # nodes that must be treated as unbounded. Does not run the analysis.
        def upper_bound(v: Expr | int) -> int:
            # [FUNCTION DOC] HELPER: Returns the upper bound of a symbolic extent via
            # bound_sympy, or the value itself when it is already an int.
            return bound_sympy(v).upper if isinstance(v, Expr) else v

        self.loop_body = loop_body
        self.replacement_vals = {
            k: ValueRanges[Expr](0, upper_bound(v) - 1)
            for k, v in loop_body.var_ranges.items()
        }
        # avoid computing these values, pessimistically assume that they are unbounded
        self.unbounded_vars = dominated_nodes(
            node
            for node in self.loop_body.get_nodes()
            if node.target in ["load", "reduction", operator.getitem]
            or "masked_subblock" in node.target
        )
        # To access this variable call `get_bounds()`
        self._bounds: dict[torch.fx.Node, ValueRanges[Expr]] = {}

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string listing loop body, replacement values,
        # unbounded vars and computed bounds.
        return (
            f"{self.__class__.__name__}("
            f"loop_body={self.loop_body},\n "
            f"replacement_vals={self.replacement_vals}, \n"
            f"unbounded_vars={self.unbounded_vars}, \n"
            f"_bounds={self._bounds})"
        )

    @cache_on_self
    def get_bounds(self) -> dict[torch.fx.Node, ValueRanges[Expr]]:
        # [FUNCTION DOC] Runs the analysis (cached per instance): marks unbounded nodes as
        # unknown (except masked subblock and set_indirect nodes, which must be evaluated),
        # then interprets the root graph under a ValueRangeAnalysis ops handler with the
        # swapped submodules, filling and returning the node to ValueRanges mapping.
        # Side effects: mutates _bounds and replacement_vals; logs the graph at debug level.
        submodules = self.swap_submodules(self.loop_body.submodules)

        # Initialize the environment with the unbounded variables
        for node in self.unbounded_vars:
            # we need to evaluate masked_subblock to recurse, and we need to set indirect values
            if not isinstance(node.target, str) or (
                "masked_subblock" not in node.target
                and "set_indirect" not in node.target
            ):
                self._bounds[node] = ValueRanges[Expr].unknown()

        with V.set_ops_handler(ValueRangeAnalysis()):
            interpreter = InterpreterShim(self.loop_body.root_block.graph, submodules)
            log.debug("get_bounds:\n%s", self.loop_body.root_block.graph)
            interpreter.run(V.get_ops_handler(), initial_env=self._bounds)
        return self._bounds

    def swap_submodules(
        self, submodules: dict[str, Callable[..., Any]]
    ) -> dict[str, Callable[..., ValueRanges[Expr]]]:
        # [FUNCTION DOC] Builds a mapping of submodule names to range-computing callables:
        # get_index maps to this object get_index, masked subblocks map to closures calling
        # masked_subblock, set_indirect submodules are bound to their indirect variable,
        # and scan submodules pass through unchanged. Raises an error for any other key.
        result: dict[str, Callable[..., ValueRanges[Expr]]] = {}
        for key in submodules:
            if key == "get_index":
                result[key] = self.get_index
            elif "masked_subblock" in key:
                subblock = self.loop_body.subblocks[key]
                # The result within the lambda will reference to the final
                # set of modules at the end of the for-loop as it stores a reference to it

                # bind subblock in a function because python lambdas close over by reference
                # moving the lambda out of make_fn would close over the reference to subblock,
                # so all lambdas would have the same subblock reference that is the final
                # subblock in the loop
                def make_fn(
                    subblock: LoopBodyBlock,
                ) -> Callable[[Any, Any], ValueRanges[Expr]]:
                    # [FUNCTION DOC] HELPER: Returns a callable taking mask and value that
                    # evaluates the given subblock; binds subblock early to avoid late binding.
                    return lambda mask, value: self.masked_subblock(
                        subblock, self._bounds, mask, value, result
                    )

                result[key] = make_fn(subblock)
            elif "set_indirect" in key:
                idx = int(key[len("set_indirect") :])
                var = self.loop_body.indirect_vars[idx]
                indirect = partial(self.set_indirect, var)
                result[key] = indirect
            else:
                if "scan" not in key:
                    raise AssertionError(
                        f"expected 'scan' in submodule key, got {key!r}"
                    )
                result[key] = submodules[key]

        return result

    def masked_subblock(
        self,
        subblock: LoopBodyBlock,
        env: dict[torch.fx.Node, ValueRanges[Expr]],
        mask: Any,
        value: Any,
        submodules: dict[str, Callable[..., Any]],
    ) -> ValueRanges[Expr]:
        # [FUNCTION DOC] Interprets a masked subblock graph with the current ops handler,
        # sharing the given environment, and returns the range of its single output node.
        # The mask and value arguments are unused (value is not unioned in because loads are
        # unbounded anyway). Raises an error if the subblock lacks exactly one output node.
        interp = InterpreterShim(subblock.graph, submodules)
        interp.run(V.get_ops_handler(), initial_env=env)
        output = [node for node in subblock.graph.nodes if node.target == "output"]
        if len(output) != 1:
            raise AssertionError(f"expected exactly 1 output node, got {len(output)}")
        # don't bother unioning with value since the load from buffer will be
        # pessimistically assumed to be inf anyway
        return interp.env[output[0]]

    def set_indirect(self, old: Expr, new: ValueRanges[Expr]) -> ValueRanges[Expr]:
        # [FUNCTION DOC] Records the range of an indirect index variable in replacement_vals
        # and returns it. Raises an error if new is not a ValueRanges.
        if not isinstance(new, ValueRanges):
            raise AssertionError(f"expected ValueRanges, got {type(new)}")
        self.replacement_vals[old] = new
        return new

    def get_index(self, name: str) -> ValueRanges[Expr]:
        # [FUNCTION DOC] Returns the range of a named indexing expression of the loop body,
        # using a cached replacement value if present or else bounding the expression with
        # the known variable ranges; caches the result under the name in replacement_vals.
        expr = self.loop_body.indexing_exprs[name]
        bound = self.replacement_vals.get(expr)
        if bound is None:
            bound = bound_sympy(expr, self.replacement_vals)
        # The following assertion is true at the time of this writing
        # We don't assert is as to not execute bound_sympy when bound is not None
        # assert bound is None or bound == bound_sympy(expr, self.replacement_vals)
        self.replacement_vals[name] = bound
        return bound


class ValueRangeAnalysis(SymPyValueRangeAnalysis, DefaultHandler):
    # [CLASS DOC] Ops handler that evaluates operations on ValueRanges instead of values.
    # Inherits arithmetic range rules from the sympy analysis; logical ops return the full
    # boolean range; loads, reductions and any unlisted op return unknown; stores do nothing.
    def __init__(self) -> None:
        # [FUNCTION DOC] Sets the handler name and routes xor and the logical and, or, not
        # ops to bool_handler on this instance.
        self.name = "ValueRangeAnalysis"
        boolean_operators = (
            "xor",
            "logical_and",
            "logical_or",
            "logical_not",
        )
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args: Any, **kwargs: Any) -> ValueRanges[Any]:
        # [FUNCTION DOC] Ignores its arguments and returns the range covering both false
        # and true.
        # just assuming bools can have both values
        return ValueRanges(sympy.false, sympy.true)  # type: ignore[arg-type]

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Fallback for ops without a specific rule; returns an unknown
        # range regardless of arguments.
        # many ops are unlikely to show up in optimizable indexing compute,
        # so we don't have full coverage
        return ValueRanges.unknown()

    def load(self, name: str, index: sympy.Expr) -> ValueRanges[Any]:
        # [FUNCTION DOC] Buffer loads have unknown values; returns an unknown range.
        return ValueRanges.unknown()

    def store(
        self, name: str, index: sympy.Expr, value: Any, mode: StoreMode = None
    ) -> None:
        # [FUNCTION DOC] Stores are ignored; returns nothing.
        return

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Any,
    ) -> ValueRanges[Any]:
        # [FUNCTION DOC] Reduction results are unknown; returns an unknown range.
        return ValueRanges.unknown()

    @classmethod
    def index_expr(cls, index: Any, dtype: torch.dtype) -> ValueRanges[Any]:
        # [FUNCTION DOC] Casts the range of an index to the dtype via to_dtype. Raises an
        # error if index is not a ValueRanges.
        if not isinstance(index, ValueRanges):
            raise AssertionError(f"expected ValueRanges, got {type(index)}")
        return cls.to_dtype(index, dtype)

    @classmethod
    def value_expr(cls, index: Any, dtype: torch.dtype) -> ValueRanges[Any]:
        # [FUNCTION DOC] Same as index_expr: casts the range to the dtype.
        return cls.index_expr(index, dtype)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def to_dtype(
        x: Any,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> ValueRanges[Any]:
        # [FUNCTION DOC] Converts a range to another dtype. For bool targets: a singleton maps
        # to whether it is nonzero, a bool range stays, a range excluding zero becomes true,
        # else the full bool range. For numeric targets: bool ranges map to 0/1 values, other
        # ranges have both bounds cast to float or integer (infinities kept). src_dtype and
        # use_compute_types are unused.
        x = ValueRanges.wrap(x)

        if dtype == torch.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif x.is_bool:
                return x
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x: Any, dtype: torch.dtype) -> sympy.Expr:
            # [FUNCTION DOC] HELPER: Casts a bound to a sympy Float for floating dtypes or a
            # sympy Integer otherwise, leaving infinite values unchanged.
            # dtype is int or float
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                if x in (int_oo, -int_oo):
                    return x
                try:
                    return sympy.Integer(x)
                except TypeError:
                    # inf cannot be cast to Integer
                    return x

        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            # int to float or float to int
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    # pyrefly: ignore [bad-override]
    def square(x: Any) -> ValueRanges[Any]:
        # [FUNCTION DOC] Range of the square of x (convex map with minimum at zero).
        return ValueRanges.convex_min_zero_map(x, lambda y: PowByNatural(y, 2))

    @staticmethod
    # pyrefly: ignore [bad-override]
    def neg(x: Any) -> ValueRanges[Any]:
        # [FUNCTION DOC] Range of the negation of x (decreasing map).
        return ValueRanges.decreasing_map(x, operator.neg)

    # TODO: this is slightly inaccurate because truncdiv operates at integer
    # precision, but we're going through float truediv which means we can
    # potentially lose precision on the bounds
    @classmethod
    def truncdiv(cls, a: Any, b: Any) -> ValueRanges[Any]:
        # [FUNCTION DOC] Range of truncating division: computes true division range, returns
        # unknown if that is unknown, else truncates the result range.
        # KNOWN ISSUE: goes through float true division so integer bounds may lose precision
        # (acknowledged in the TODO above).
        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        return cls.trunc(x)

    @classmethod
    def sub(cls, a: Any, b: Any) -> ValueRanges[Any]:
        # [FUNCTION DOC] Range of a minus b, computed as a plus the negation of b.
        return cls.add(a, cls.neg(b))
