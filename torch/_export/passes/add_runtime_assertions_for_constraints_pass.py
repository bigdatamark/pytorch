# [MODULE DOC]
# Pass that adds runtime assertions for inline (unbacked symint) constraints to an
# exported graph, plus helpers to convert sympy ranges and to find assertions
# already present. Exports InputDim. No import-time side effects.
# mypy: allow-untyped-defs
import math
import operator
import traceback
from functools import partial
from typing import NamedTuple, TYPE_CHECKING

import sympy

import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["InputDim"]


class InputDim(NamedTuple):
    # [CLASS DOC] Named pair identifying a dimension of a named graph input:
    # input_name (str) and dim (int index). Plain immutable record, no behavior.
    input_name: str
    dim: int


def _convert_to_int(val):
    # [FUNCTION DOC] HELPER: Convert a sympy bound to a Python number. Positive and
    # negative infinity (sympy or int_oo) map to float infinities, sympy Integer maps
    # to int. Raises RuntimeError for anything else (non-integer expressions).
    # Convert simple sympy Integers into concrete int
    if val in (sympy.oo, int_oo):
        return math.inf
    if val in (-sympy.oo, -int_oo):
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError("Export constraints cannot be non-integer expressions")


def _convert_range_to_int(range: ValueRanges):
    # [FUNCTION DOC] HELPER: Convert a ValueRanges into a (min, max) pair of Python
    # numbers via the scalar conversion helper. Raises AssertionError if the argument
    # is not a ValueRanges and RuntimeError for non-integer bounds.
    if not isinstance(range, ValueRanges):
        raise AssertionError(f"expected ValueRanges, got {type(range)}")
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return min_val, max_val


class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    # [CLASS DOC] HELPER: Pass that inserts runtime range assertions (compare, scalar
    # tensor, assert_async chains) for unbacked symbolic ints appearing in node meta
    # values, either directly or as tensor shape dims. State: range_constraints
    # (symbol to ValueRanges), _asserts_generated_unbacked_symbols (symbols already
    # asserted, to avoid duplicates), counter (number of assert chains inserted) and
    # existing_inline_assertions (set during call). Key method: call.
    def __init__(
        self,
        range_constraints: dict[sympy.Symbol, ValueRanges],
    ):
        # [FUNCTION DOC] Store the symbol to range mapping and initialize the set of
        # already-asserted symbols and the insertion counter to empty/zero.
        super().__init__()
        self.range_constraints: dict[sympy.Symbol, ValueRanges] = range_constraints
        self._asserts_generated_unbacked_symbols: set[sympy.Symbol] = set()
        self.counter = 0

    def _assert_range_constraint(self, node, lower, upper, assert_msg):
        # [FUNCTION DOC] HELPER: Insert after node a lower-bound assertion (when lower is
        # finite) and an upper-bound assertion (when upper is finite) on the value of
        # node, each via the assert insertion helper and chained after the previous.
        # Mutates the graph; returns nothing.
        last_node = node
        if lower > -math.inf:
            last_node = self._insert_assert_async(
                last_node, operator.ge, node, lower, assert_msg
            )

        if upper < math.inf:
            last_node = self._insert_assert_async(
                last_node, operator.le, node, upper, assert_msg
            )

    def _insert_assert_async(self, last_node, op, lower, upper, assert_msg):
        # [FUNCTION DOC] HELPER: After last_node, add three call_function nodes: the
        # comparison op applied to the two operands (named lower and upper, which are
        # really just left/right operands), a scalar_tensor of that result, and an
        # assert_async with the message. Increments counter and returns the assert node.
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        self.counter += 1
        graph = last_node.graph
        with graph.inserting_after(last_node):
            cmp = graph.call_function(op, (lower, upper), {})
        with graph.inserting_after(cmp):
            cmp_tensor = graph.call_function(
                torch.ops.aten.scalar_tensor.default, (cmp,), {}
            )
        with graph.inserting_after(cmp_tensor):
            assert_async = graph.call_function(
                torch.ops.aten._assert_async.msg,
                (cmp_tensor, assert_msg),
                {},
            )
        return assert_async

    def call(self, graph_module) -> PassResult:
        # [FUNCTION DOC] Compute existing inline assertions, then for every call_function
        # node with a meta val in every GraphModule, find unbacked symbolic ints in the
        # val (directly, or in tensor shape dims, recursing) not yet asserted and insert
        # range assertions (shape dims go through a created size node). Recompiles each
        # module. If nothing was inserted and the class is exactly this pass, returns the
        # module unmodified. Otherwise fills missing stack_trace on non-placeholder,
        # non-output nodes with a dummy value and returns a modified PassResult.
        # Raises KeyError if an unbacked symbol is missing from range_constraints.
        self.existing_inline_assertions = _get_existing_inline_assertions(
            graph_module, self.range_constraints
        )

        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if "val" not in node.meta:
                    continue

                val = node.meta["val"]
                # In general, we may have to deal the case such as: ret[1].shape[0].
                # We need first find out what symbols require assertion, then we need to follow the path
                # from ret to the symbol, construct the proxies along the way and construct the messages
                # piece-wise at the same time.
                #
                # We use post-order traversal to collect all the proxies callbacks needed, construct
                # the error message callbacks, and at the top-level traversal tree we execute all the callbacks.
                # We need the callbacks because, in order to call the function to create a proxy for shape[0], we
                # need the proxy for shape, which further requires the proxy for ret[1], etc.

                def add_assertions(val):
                    # [FUNCTION DOC] HELPER: Recursively collect parallel lists of
                    # insertion callbacks and error-message suffixes for a value. Symbolic
                    # values that are unbacked, not already asserted, yield a range
                    # assertion callback and message (symbol recorded as asserted); tensors
                    # recurse on each shape dim, wrapping callbacks to first create a size
                    # node. Other values yield empty lists.
                    call_backs: list[Callable] = []
                    messages: list[str] = []
                    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                        symbol = val.node.expr
                        if symbol in self.existing_inline_assertions:
                            return call_backs, messages
                        if isinstance(symbol, sympy.Symbol) and free_unbacked_symbols(
                            symbol
                        ):
                            if symbol in self._asserts_generated_unbacked_symbols:
                                return call_backs, messages
                            # We only care about unbacked symints for these inline
                            # constraints, which are prefixed with 'u'
                            constraint = self.range_constraints[symbol]
                            min_val, max_val = _convert_range_to_int(constraint)
                            assert_msg = f" is outside of inline constraint [{min_val}, {max_val}]."
                            call_backs.append(
                                partial(
                                    self._assert_range_constraint,
                                    lower=min_val,
                                    upper=max_val,
                                )
                            )
                            messages.append(assert_msg)
                            self._asserts_generated_unbacked_symbols.add(symbol)

                    elif isinstance(val, torch.Tensor):
                        for i, sym in enumerate(val.shape):
                            cbs, msgs = add_assertions(sym)
                            for cb, msg in zip(cbs, msgs):

                                def sym_size_cb(node, assert_msg, dim):
                                    # [FUNCTION DOC] HELPER: Create a size node for dim
                                    # after node, then run the wrapped callback on it.
                                    with node.graph.inserting_after(node):
                                        dim_node = module.graph.create_size_node(  # type: ignore[union-attr]
                                            node, dim
                                        )
                                    cb(node=dim_node, assert_msg=assert_msg)

                                call_backs.append(partial(sym_size_cb, dim=i))
                                messages.append(f".shape[{i}]" + msg)
                    return call_backs, messages

                callbacks, messages = add_assertions(val)
                for cb, msg in zip(callbacks, messages):
                    cb(node=node, assert_msg=f"{node}" + msg)

            module.recompile()

        # Sometimes this pass would return a wrong graph where we have mismatched
        # node names in signature. Before we fix it, let's just skip it.
        if (
            self.counter == 0
            and type(self) is _AddRuntimeAssertionsForInlineConstraintsPass
        ):
            return PassResult(graph_module, False)

        # Populate the stack trace with dummy vals to respect IR
        for node in graph_module.graph.nodes:
            if not node.meta.get("stack_trace", None) and node.op not in [
                "placeholder",
                "output",
            ]:
                node.meta["stack_trace"] = "".join(traceback.format_stack(limit=1))
        return PassResult(graph_module, True)


def _get_existing_inline_assertions(
    graph_module: torch.fx.GraphModule,
    range_constraints: dict[sympy.Symbol, ValueRanges],
) -> dict[sympy.Symbol, ValueRanges]:
    # [FUNCTION DOC] HELPER: Scan all GraphModules for assert_scalar nodes whose
    # condition is a le/ge comparison between a symbolic int and an integer constant,
    # and return a mapping from symbol to the intersected ValueRanges those asserts
    # already enforce (starting from unbounded). Unrecognized assert shapes are skipped.
    # Raises RuntimeError if a matched symbol is absent from range_constraints.
    existing_inline_assertions: dict[sympy.Symbol, ValueRanges] = {}

    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        # Find all the existing inline assertions. They will look something like:
        # %_local_scalar_dense = call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%arg1_1,), kwargs = {})
        # %ge = call_function[target=operator.ge](args = (%_local_scalar_dense, 0), kwargs = {})
        # %_assert_scalar = call_function[target=torch.ops.aten._assert_scalar.default](args = (%scalar_tensor, "..."), kwargs = {})
        for node in module.graph.nodes:
            if node.target != torch.ops.aten._assert_scalar.default:
                continue

            compare_arg = node.args[0]
            if not (
                isinstance(compare_arg, torch.fx.Node)
                and compare_arg.op == "call_function"
                and compare_arg.target in (operator.le, operator.ge)
                and len(compare_arg.args) == 2
            ):
                continue

            compare_op = compare_arg.target
            lhs, rhs = compare_arg.args

            def maybe_get_symint(x):
                # [FUNCTION DOC] HELPER: Return the sympy expression of the node's
                # SymInt meta val if x is such a node, else x unchanged.
                if (
                    isinstance(x, torch.fx.Node)
                    and "val" in x.meta
                    and isinstance(x.meta["val"], torch.SymInt)
                ):
                    return x.meta["val"].node.expr
                return x

            lhs = maybe_get_symint(lhs)
            rhs = maybe_get_symint(rhs)

            if compare_op is operator.ge:
                lhs, rhs = rhs, lhs

            if isinstance(lhs, sympy.Symbol) and isinstance(rhs, int):
                symint = lhs
                scalar = rhs
            elif isinstance(rhs, sympy.Symbol) and isinstance(lhs, int):
                symint = rhs
                scalar = lhs
            else:
                continue

            if symint not in range_constraints:
                raise RuntimeError(
                    f"Unable to find symint {symint} in {range_constraints}"
                )

            previous_range = existing_inline_assertions.get(
                symint, ValueRanges(-math.inf, math.inf)
            )

            if symint is lhs:
                bounds = ValueRanges(-math.inf, scalar)
            else:
                bounds = ValueRanges(scalar, math.inf)
            existing_inline_assertions[symint] = previous_range & bounds

    return existing_inline_assertions
