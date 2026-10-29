# [MODULE DOC] Loop-body index dtype optimizations for Inductor. Contains (1) a value range analysis pass that
# lowers int64 to_dtype conversions to int32 when all dominated values provably fit, and (2) a backward
# reachability pass that rewrites index_expr nodes feeding value computation into value_expr nodes.
# No import-time side effects.
import math
from dataclasses import dataclass
from typing import Any

import sympy

import torch
from torch.fx.node import map_arg
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .loop_body import LoopBody
from .utils import dominated_nodes


def val_expressable_in_32_bits(val: Any) -> bool:
    # [FUNCTION DOC] Return whether a constant fits in 32 bits. Booleans always fit; sympy numbers are converted to int or float.
    # Floats must lie within +-2**24 (mantissa bound), ints within the int32 range.
    # Raises an error if a sympy expression is not a number, and a type error for other value types.
    if getattr(val, "is_Boolean", False):
        return True

    if isinstance(val, sympy.Expr):
        if not val.is_number:
            raise AssertionError(f"expected a number, got {val}")
        if val.is_Integer or val.is_Boolean:
            val = int(val)
        else:
            val = float(val)

    # bound within mantissa
    if isinstance(val, float):
        return val <= (2**24) and val >= -(2**24)

    if isinstance(val, int):
        iinfo = torch.iinfo(torch.int32)
        return val <= iinfo.max and val >= iinfo.min

    raise TypeError(f"Unexpected value {val}")


def range_expressable_in_32_bits(range: ValueRanges[sympy.Expr]) -> bool:
    # [FUNCTION DOC] Return true if both the lower and upper bounds of a ValueRanges fit in 32 bits.
    return val_expressable_in_32_bits(range.lower) and val_expressable_in_32_bits(
        range.upper
    )


def try_to_reduce_precision(
    node: Any,
    bounds: dict[Any, Any],
    indirect_vars: list[Any],
    indices: dict[Any, sympy.Expr],
    replacement_vals: dict[Any, ValueRanges[sympy.Expr]],
) -> None:
    # [FUNCTION DOC] Try to retarget a to_dtype(int64) node to int32 in place by changing its dtype argument.
    # Walks nodes dominated by the node (stopping at explicit casts to int32/float32/float64), ignoring store and output.
    # Gives up silently (leaves node unchanged) if any set_indirect-derived index has infinite or non-32-bit bounds,
    # or if any dominated node's bounds do not fit in 32 bits. Mutates node.args on success.
    # if a downstream use of a node explicitly converts to int32, or float16/float32/float64,
    # then its precision is set for that chain of uses, and we don't need to consider those
    # dominated values
    def skip_filter(node: Any) -> bool:
        # [FUNCTION DOC] HELPER: Predicate for dominated-node traversal: true for a to_dtype node casting to int32, float32 or float64.
        return node.target == "to_dtype" and node.args[2] in (
            torch.int32,
            torch.float32,
            torch.float64,
        )

    # TODO - there are dominated uses whose dtype does not depend on whether
    # we reduce the precision here, e.g. add(int64, int64) one of the args can be reduced to
    # int32 without changing the output precision of the node. this case hasn't shown up
    for dominated in dominated_nodes([node], skip_filter):
        if dominated.target in ["store", "output"]:
            continue

        if isinstance(dominated.target, str) and "set_indirect" in dominated.target:
            idx = int(dominated.target[len("set_indirect") :])
            indirect_var = indirect_vars[idx]

            # We check that we can compute all the indices it's involved in with int32
            for index, expr in indices.items():
                if indirect_var in expr.free_symbols:
                    index_val = replacement_vals[index]

                    if math.isinf(index_val.lower) or math.isinf(index_val.upper):
                        return

                    # all indices are integers, so make sure that we
                    # use the bounds of integers instead of floats.
                    # TODO - not sure if we should be doing int/float casts while tracing,
                    # might interfere with sympy.

                    index_val_int = ValueRanges[sympy.Expr](
                        int(index_val.lower), int(index_val.upper)
                    )
                    if not range_expressable_in_32_bits(index_val_int):
                        return

        if not range_expressable_in_32_bits(bounds[dominated]):
            return

    args = list(node.args)
    args[2] = torch.int32
    node.args = tuple(args)


def indexing_dtype_strength_reduction(loop_body: LoopBody) -> None:
    # [FUNCTION DOC] Run value range analysis on the loop body and lower int64 to_dtype nodes with known bounds to int32
    # when safe (see try_to_reduce_precision). Returns nothing; mutates fx graph nodes in place.
    # Does nothing if no bounded int64 to_dtype nodes exist.
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
    bv = loop_body.bounds()

    int64_dtype_nodes = [
        node
        for node in loop_body.get_nodes()
        if (
            node.target == "to_dtype"
            and node.args[2] == torch.int64
            and node not in bv.unbounded_vars
        )
    ]
    if not int64_dtype_nodes:
        return

    bounds = bv.get_bounds()

    # TODO - if dominated node of one to_dtype is not expressible in int32,
    # we should short circuit another to_dtype node if that node also dominates
    for node in int64_dtype_nodes:
        try_to_reduce_precision(
            node,
            bounds,
            loop_body.indirect_vars,
            loop_body.indexing_exprs,
            bv.replacement_vals,
        )


@dataclass(frozen=True)
class _ValueUseRule:
    # [CLASS DOC] HELPER: Immutable description of how an op's arguments are used. value_sinks seed the backward value
    # walk unconditionally, value_inputs propagate demand only if the op result is value-reachable, and
    # indexing_inputs (addresses, masks, bounds) block propagation. All fields are tuples of op arguments.
    # These fields are op arguments, so Any covers FX nodes, SymPy exprs,
    # scalars, and nested tuples/lists. value_sinks seed the backward walk
    # unconditionally. value_inputs propagate value demand only when the op's
    # result is already value-reachable. indexing_inputs block propagation
    # because they only affect addresses, bounds, masks, or other indexing-only
    # state.
    value_inputs: tuple[Any, ...] = ()
    value_sinks: tuple[Any, ...] = ()
    indexing_inputs: tuple[Any, ...] = ()


def _collect_fx_nodes(arg: Any) -> OrderedSet[torch.fx.Node]:
    # [FUNCTION DOC] HELPER: Collect all fx Nodes found anywhere in a (possibly nested) argument structure into an ordered set.
    nodes: OrderedSet[torch.fx.Node] = OrderedSet()

    def add_node(node: torch.fx.Node) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Callback for argument mapping: record the node and return it unchanged.
        nodes.add(node)
        return node

    map_arg(arg, add_node)
    return nodes


def _collect_input_nodes(node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
    # [FUNCTION DOC] HELPER: Return the ordered set of input nodes of a node, excluding the receiver (first argument) of a call_method node.
    inputs = OrderedSet(node.all_input_nodes)
    if (
        node.op == "call_method"
        and node.args
        and isinstance(node.args[0], torch.fx.Node)
    ):
        inputs.discard(node.args[0])
    return inputs


class _ValueUseRules:
    # [CLASS DOC] HELPER: Table of per-op rules, looked up by op name via attribute access. Each method mirrors an ops
    # handler signature and returns a _ValueUseRule classifying its arguments as value sinks, value inputs, or
    # indexing-only inputs. Ops without a method use default_rule (None), meaning all inputs are value inputs.
    # Instances get extra attributes set at runtime for call_module targets (subblocks, scan, set_indirect).
    @staticmethod
    def default_rule(*args: Any, **kwargs: Any) -> _ValueUseRule | None:
        # [FUNCTION DOC] HELPER: Default rule for ops with no specific rule: returns no rule.
        return None

    def load(self, name: str, index: sympy.Expr) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: load: the index is indexing-only.
        return _ValueUseRule(indexing_inputs=(index,))

    def load_seed(self, name: str, offset: int) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: load_seed: the offset is indexing-only.
        return _ValueUseRule(indexing_inputs=(offset,))

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: Any,
        mode: Any = None,
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: store: the stored value is a value sink; the index is indexing-only.
        return _ValueUseRule(
            value_sinks=(value,),
            indexing_inputs=(index,),
        )

    def store_reduction(
        self, name: str, index: sympy.Expr, value: Any
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: store_reduction: the value is a value sink; the index is indexing-only.
        return _ValueUseRule(
            value_sinks=(value,),
            indexing_inputs=(index,),
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: Any,
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: reduction: the reduced value is a value sink.
        return _ValueUseRule(value_sinks=(value,))

    def partial_accumulate(
        self,
        name: str,
        reduction_type: str,
        value: Any,
        extra_meta: dict[str, Any],
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: partial_accumulate: the value is a value sink.
        return _ValueUseRule(value_sinks=(value,))

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[Any, ...],
        stable: bool,
        descending: bool,
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: sort: the tuple of values is a value sink.
        return _ValueUseRule(value_sinks=(values,))

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn_or_values: Any,
        values: tuple[Any, ...] | None = None,
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: scan: values (or the second argument when values is not given) are value sinks.
        if values is None:
            values = combine_fn_or_values
        return _ValueUseRule(value_sinks=(values,))

    def bucketize(
        self,
        values: Any,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: Any,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: Any | None = None,
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: bucketize: values are value inputs; boundaries, sorter and their indices are indexing-only.
        return _ValueUseRule(
            value_inputs=(values,),
            indexing_inputs=(boundaries, boundary_indices, sorter, sorter_indices),
        )

    def indirect_indexing(
        self, x: Any, size: sympy.Expr, check: bool = True, wrap_neg: bool = True
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: indirect_indexing: the index value and size are indexing-only.
        return _ValueUseRule(indexing_inputs=(x, size))

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: check_bounds: expression and size are indexing-only.
        return _ValueUseRule(indexing_inputs=(expr, size))

    def masked(self, mask: Any, body: Any, other: Any) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: masked: other is a value input; the mask is indexing-only.
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def masked_subblock(self, mask: Any, other: Any) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: masked_subblock (installed for subblock call_module nodes): other is a value input; mask is indexing-only.
        return _ValueUseRule(value_inputs=(other,), indexing_inputs=(mask,))

    def set_indirect(self, new_var: Any) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: set_indirect: the new variable is indexing-only.
        return _ValueUseRule(indexing_inputs=(new_var,))

    def device_assert_async(self, cond: Any, msg: str) -> _ValueUseRule:
        # [FUNCTION DOC] HELPER: device_assert_async: the condition is indexing-only.
        return _ValueUseRule(indexing_inputs=(cond,))


class _ValueUseAnalysis:
    # [CLASS DOC] HELPER: Backward reachability analysis over a LoopBody's root and sub-block fx graphs. State: rules table,
    # graphs, a value_reachable set, a worklist of (graph, node) pairs, and a map from indirect symbols to the
    # set_indirect arguments that define them. run() converts index_expr nodes reachable from value sinks.
    def __init__(self, loop_body: LoopBody) -> None:
        # [FUNCTION DOC] HELPER: Store the loop body, create the rules table, gather root and sub-block graphs, install call_module rules and init empty state.
        self.loop_body = loop_body
        self.rules = _ValueUseRules()
        self.root_graph = loop_body.root_block.graph
        self.subblocks = getattr(loop_body, "subblocks", {})
        self.indirect_vars = getattr(loop_body, "indirect_vars", ())
        self.graphs = [
            self.root_graph,
            *(block.graph for block in self.subblocks.values()),
        ]
        self.install_call_module_rules()

        self.value_reachable: OrderedSet[torch.fx.Node] = OrderedSet()
        self.worklist: list[tuple[torch.fx.Graph, torch.fx.Node]] = []
        self.indirect_inputs: dict[sympy.Symbol, tuple[torch.fx.Graph, Any]] = {}

    def run(self) -> bool:
        # [FUNCTION DOC] Run the analysis. Returns false early if no index_expr exists; otherwise seeds sinks and propagates,
        # returning true if any index_expr node was rewritten.
        if not self.has_index_expr():
            return False

        self.seed_value_reachable_nodes()
        return self.propagate_value_reachability()

    def has_index_expr(self) -> bool:
        # [FUNCTION DOC] HELPER: Return true if any graph contains an index_expr call_method node.
        return any(
            graph.find_nodes(op="call_method", target="index_expr", sort=False)
            for graph in self.graphs
        )

    def install_call_module_rules(self) -> None:
        # [FUNCTION DOC] HELPER: For each call_module node, register a rule on the rules table by target name: sub-blocks use the masked-subblock rule,
        # scan* targets the scan rule, set_indirect* targets the set_indirect rule.
        for graph in self.graphs:
            for node in graph.find_nodes(op="call_module", sort=False):
                if not isinstance(node.target, str):
                    continue
                if node.target in self.subblocks:
                    setattr(self.rules, node.target, self.rules.masked_subblock)
                elif node.target.startswith("scan"):
                    setattr(self.rules, node.target, self.rules.scan)
                elif node.target.startswith("set_indirect"):
                    setattr(self.rules, node.target, self.rules.set_indirect)

    def seed_value_reachable_nodes(self) -> None:
        # [FUNCTION DOC] HELPER: Seed the worklist with root graph outputs and every node's value sinks; also records set_indirect nodes' arguments
        # keyed by the corresponding indirect variable.
        self._enqueue_graph_output(self.root_graph)
        for graph in self.graphs:
            for node in graph.nodes:
                if node.op == "output":
                    continue
                if (
                    node.op == "call_module"
                    and isinstance(node.target, str)
                    and node.target.startswith("set_indirect")
                ):
                    idx = int(node.target[len("set_indirect") :])
                    self.indirect_inputs[self.indirect_vars[idx]] = (graph, node.args)
                for sink_node in self.value_sink_nodes(node):
                    self.worklist.append((graph, sink_node))

    def propagate_value_reachability(self) -> bool:
        # [FUNCTION DOC] HELPER: Process the worklist: mark nodes reachable, rewrite index_expr targets to value_expr, enqueue sub-block outputs,
        # follow indirect symbols in value-reachable get_index expressions to their set_indirect inputs, and enqueue value inputs.
        # Returns true if any node was rewritten. Raises an error if a get_index name is not a string.
        changed = False
        while self.worklist:
            graph, node = self.worklist.pop()
            if node in self.value_reachable:
                continue
            self.value_reachable.add(node)

            if node.target == "index_expr":
                node.target = "value_expr"
                changed = True

            if (
                node.op == "call_module"
                and isinstance(node.target, str)
                and node.target in self.subblocks
            ):
                self._enqueue_graph_output(self.subblocks[node.target].graph)

            if node.op == "call_module" and node.target == "get_index":
                index_name = node.args[0]
                if not isinstance(index_name, str):
                    raise AssertionError(f"expected str index name, got {index_name!r}")
                expr = self.loop_body.indexing_exprs[index_name]
                if isinstance(expr, sympy.Expr):
                    for symbol in expr.free_symbols:
                        indirect_input = self.indirect_inputs.get(symbol)
                        if indirect_input is not None:
                            ig, ia = indirect_input
                            map_arg(ia, lambda n: self.worklist.append((ig, n)))

            for input_node in self.value_input_nodes(node):
                self.worklist.append((graph, input_node))

        return changed

    def rule_for_node(self, node: torch.fx.Node) -> _ValueUseRule | None:
        # [FUNCTION DOC] HELPER: Return the rule for a call_method or call_module node by calling the matching rules method with the node's
        # arguments (skipping the receiver for call_method), or None for other nodes or when no rule applies.
        if node.op == "call_method" and isinstance(node.target, str):
            rule_fn = getattr(self.rules, node.target, self.rules.default_rule)
            return rule_fn(*node.args[1:], **node.kwargs)
        if node.op == "call_module" and isinstance(node.target, str):
            rule_fn = getattr(self.rules, node.target, self.rules.default_rule)
            return rule_fn(*node.args, **node.kwargs)
        return None

    def value_input_nodes(self, node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
        # [FUNCTION DOC] HELPER: Nodes whose values flow into this node: all input nodes if no rule, else the rule's value inputs and sinks.
        rule = self.rule_for_node(node)
        if rule is None:
            return _collect_input_nodes(node)
        return _collect_fx_nodes(rule.value_inputs) | _collect_fx_nodes(
            rule.value_sinks
        )

    def value_sink_nodes(self, node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
        # [FUNCTION DOC] HELPER: Nodes that this node unconditionally consumes as values (rule's value sinks), empty if no rule.
        rule = self.rule_for_node(node)
        return _collect_fx_nodes(rule.value_sinks) if rule else OrderedSet()

    def _enqueue_graph_output(self, graph: torch.fx.Graph) -> None:
        # [FUNCTION DOC] HELPER: Enqueue the arguments of a graph's single output node. Raises an error if there is not exactly one output node.
        output_nodes = graph.find_nodes(op="output", sort=False)
        if len(output_nodes) != 1:
            raise AssertionError(
                f"expected exactly 1 output node, got {len(output_nodes)}"
            )
        map_arg(output_nodes[0].args, lambda n: self.worklist.append((graph, n)))


def convert_index_expr_to_value_expr(loop_body: LoopBody) -> None:
    # [FUNCTION DOC] Rewrite index_expr nodes that feed value computation into value_expr, in place. If anything changed, clears the cached
    # get_nodes and bounds on the loop body. See the docstring for the classification and mixed-use policy.
    """
    Rewrite ``index_expr`` nodes that participate in value computation to
    ``value_expr``, so codegen can honor the requested dtype instead of
    narrowing to the kernel's indexing dtype.

    Classification: walk backward from value sinks (store value, reduction,
    output). Indexing inputs to ops such as ``load``, ``store``,
    ``check_bounds``, and ``set_indirect`` do not receive value demand. Any
    ``index_expr`` reachable from a value sink is converted in-place to
    ``value_expr``. Indirect indexing is handled by following value-reachable
    ``get_index`` expressions through any referenced ``indirect*`` symbols back
    to the corresponding ``set_indirect*`` input.

    Mixed-use policy: if the same ``index_expr`` feeds both an indexing use and
    a value use, rewrite it in-place to ``value_expr``. This is conservative and
    correct because the indexing path may compute at a wider dtype than needed,
    but the value path cannot lose precision. Cloning would preserve the
    narrower indexing path, but the mixed-use case is rare and the simpler
    in-place rewrite avoids extra CSE/register pressure.
    """
    if _ValueUseAnalysis(loop_body).run():
        LoopBody.get_nodes.clear_cache(loop_body)
        LoopBody.bounds.clear_cache(loop_body)
