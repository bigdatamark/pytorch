# mypy: allow-untyped-defs
# [MODULE DOC] Defines LoopBody, which traces the scalar body of an inductor loop
# node into FX graphs (a root block plus subblocks) with named indexing expressions,
# memory-usage records and op counts, and offers transforms (merge, reorder,
# reindex, expand) that produce new bodies. Also defines the supporting
# InterpreterShim, LightTracer, MemoryEntry, MemoryUsageType, LoopBodyBlock and
# helper ops handlers. Import-time side effects: none beyond defining a TypeVar.
from __future__ import annotations

import collections
import functools
import itertools
import re
from enum import auto, Enum
from typing import Any, NamedTuple, TYPE_CHECKING, TypeVar

import sympy

import torch.fx
from torch._dynamo.utils import identity
from torch.fx.proxy import Scope, TracerBase
from torch.utils._sympy.functions import Mod
from torch.utils._sympy.symbol import SymT

from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .ops_handler import DefaultHandler, OpsHandler, WrapperHandler
from .utils import (
    cache_on_self,
    decompose_index,
    flatten_index,
    reduction_num_outputs,
    sympy_index_symbol_with_prefix,
    sympy_subs,
)
from .virtualized import ops, V


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


T = TypeVar("T")


class InterpreterShim(torch.fx.Interpreter):
    # [CLASS DOC] FX Interpreter that runs a bare graph with a submodules mapping
    # without building a GraphModule (which would codegen and be slow). Tracks the
    # node being executed in current_node and installs itself as the interpreter
    # handler in V while running.
    @staticmethod
    @functools.cache
    def _dummy_gm():
        # [FUNCTION DOC] HELPER: Cached; returns a placeholder GraphModule produced by
        # symbolically tracing the identity function, used only to satisfy the parent
        # constructor.
        return torch.fx.symbolic_trace(identity)

    def __init__(self, graph, submodules):
        # [FUNCTION DOC] Initializes the parent with the dummy module, then points
        # module to itself and stores the graph and submodules; attribute fetches
        # are served by looking up names in submodules. Garbage collection of values
        # is disabled.
        # call super() with a placeholder to avoid constructing a
        # GraphModule which is very expensive (it does codegen).
        super().__init__(self._dummy_gm(), garbage_collect_values=False)
        self.module = self  # type: ignore[assignment]
        self.graph = graph
        self.submodules = submodules
        self.extra_traceback = False
        self.fetch_attr = submodules.__getitem__  # type: ignore[method-assign]
        self.current_node = None

    def run_node(self, n: torch.fx.Node) -> Any:
        # [FUNCTION DOC] Records n as current_node, then executes it via the parent
        # and returns the node's result.
        # pyrefly: ignore [bad-assignment]
        self.current_node = n
        return super().run_node(n)

    def run(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the graph with this shim registered as the virtualized
        # interpreter handler for the duration; returns the parent run result.
        with V.set_interpreter_handler(self):
            return super().run(*args, **kwargs)


# We don't need the nn.Module and constant handling in Tracer
class LightTracer(TracerBase):
    # [CLASS DOC] Minimal FX tracer that only owns an empty Graph, a root scope and
    # empty module-stack / node-scope mappings; no module or constant handling.
    def __init__(self):
        # [FUNCTION DOC] Creates the empty graph (tracer class set to this class),
        # root scope, module stack and node-name-to-scope mapping.
        super().__init__()
        self.graph = torch.fx.Graph(tracer_cls=self.__class__)  # type: ignore[arg-type]
        self.scope = Scope("", None)
        self.module_stack = {}  # type: ignore[assignment]
        self.node_name_to_scope = {}


class MemoryEntry(NamedTuple):
    # [CLASS DOC] Record of one memory access: the name of its indexing expression
    # in the owning LoopBody, the buffer name (may be None), and the store mode.
    index_name: str  # LoopBody.indexing_exprs[index_name]
    buffer_name: str | None
    mode: str | None  # V.ops.store(..., mode=mode)


class MemoryUsageType(Enum):
    # [CLASS DOC] Kinds of indexed memory usage recorded by LoopBody, each matching
    # the op that creates it (load, load_seed, store, store_reduction, index_expr,
    # check_bounds, bucketize).
    # These are 1:1 with the opcode generating the usage
    LOAD = auto()
    LOAD_SEED = auto()
    STORE = auto()
    STORE_REDUCTION = auto()
    INDEX_EXPR = auto()
    CHECK_BOUNDS = auto()
    BUCKETIZE = auto()


class LoopBody:
    # [CLASS DOC] Callable representation of a loop body. State: indexing_exprs
    # (name to sympy expression), submodules and subblocks (FX graph blocks),
    # indirect vars and their ranges, root_block, memory_usage (per usage type list
    # of MemoryEntry), op_counts, iter/reduce vars, var_ranges and sizes. Built
    # either by tracing a function or by cheaply re-substituting indices of an
    # existing LoopBody. Calling it with index lists sets a temporary indexing
    # mapping and runs the root block. Transform methods return new LoopBody objects.
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    indexing_exprs: dict[str, sympy.Expr]
    submodules: dict[str, Any]
    subblocks: dict[str, LoopBodyBlock]
    indirect_vars: list[sympy.Symbol]
    indirect_var_ranges: dict[sympy.Symbol, sympy.Expr]
    root_block: LoopBodyBlock
    memory_usage: dict[MemoryUsageType, list[MemoryEntry]]
    op_counts: collections.Counter[str]

    # defined only temporarily
    indexing_exprs_name: dict[sympy.Expr, str]

    @staticmethod
    def _wrap_int_to_sympy_integer(expr):
        # [FUNCTION DOC] HELPER: Static. Converts an exact Python int to a sympy
        # Integer; any other value is returned unchanged.
        # Static sizes can enter indexing expressions as Python ints.
        if type(expr) is int:
            return sympy.Integer(expr)
        return expr

    def __init__(
        self,
        fn,
        args,
        var_ranges,
        iter_vars,
        reduce_vars,
        allow_same_symbol_in_index=False,
    ):
        # [FUNCTION DOC] Builds a body for fn called with args over var_ranges. Stores
        # iter/reduce vars, var_ranges and sizes split into iteration and reduction
        # parts by the number of iter vars. If fn is itself a LoopBody uses the fast
        # copy path (re-substituting indices), else traces fn. Leaves indexing unset.
        super().__init__()

        _flat_sizes = tuple(var_ranges.values())
        self.sizes = (
            _flat_sizes[: len(iter_vars)],
            _flat_sizes[len(iter_vars) :],
        )

        self.iter_vars = iter_vars
        self.reduce_vars = reduce_vars
        self.var_ranges = var_ranges

        if isinstance(fn, LoopBody):
            self._init_with_copy(fn, args, allow_same_symbol_in_index)
        else:
            self._init_with_tracing(fn, args)

        self.indexing = None

    def get_original_num_rdims(self) -> int:
        # [FUNCTION DOC] Returns the original number of reduction dims recorded in the
        # metadata (last argument) of the first partial_accumulate call in the root
        # graph. Raises AssertionError if the body has no partial accumulate.
        if not self.has_partial_accumulate:
            raise AssertionError("Expected has_partial_accumulate to be set")
        node = self.root_block.graph.find_nodes(
            op="call_method", target="partial_accumulate"
        )[0]
        meta = node.args[-1]
        return meta["num_reduction_dims"]

    def extract_pw_from_reduction(self):
        # [FUNCTION DOC] Mutates this body in place into a pointwise form: replaces the
        # root block with its extracted pointwise version, marks partial accumulate,
        # folds reduction vars and sizes into the iteration ones, leaves no reduction
        # vars. Returns self.
        self.root_block = self.root_block.extract_pw_from_reduction()
        self.has_partial_accumulate = True
        self.iter_vars = self.iter_vars + self.reduce_vars
        self.reduce_vars = []
        self.sizes = (self.sizes[0] + self.sizes[1], tuple())
        return self

    def _init_with_tracing(self, fn, args):
        # [FUNCTION DOC] HELPER: Slow path. Resets all state containers, registers
        # get_index as a submodule, traces fn(args) into the root block (which fills
        # indexing exprs, memory usage and op counts), records whether a partial
        # accumulate call exists, and drops the temporary expression-to-name mapping.
        """Do an FX trace of an arbitrary callable to construct self"""
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.submodules = {"get_index": self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.indirect_var_ranges: dict[sympy.Symbol, sympy.Expr] = {}
        self.memory_usage = {t: [] for t in MemoryUsageType}
        self.op_counts = collections.Counter()
        self.root_block = LoopBodyBlock(self, fn, args)  # traces
        self.has_partial_accumulate = bool(
            self.root_block.graph.find_nodes(
                op="call_method", target="partial_accumulate"
            )
        )
        del self.indexing_exprs_name  # not used after _init_with_tracing

    def _init_with_copy(self, other: LoopBody, args, allow_same_symbol_in_index):
        # [FUNCTION DOC] HELPER: Fast path. Substitutes args for other's vars in its
        # indexing exprs, simplifies them with the new var ranges, clones subblocks,
        # root block and other submodules into this body, and shares (not copies) the
        # indirect vars, memory usage and op counts of other.
        # Assumes other has a get_index submodule (pops it).
        """
        _init_with_tracing() is slow, so this is a fast path in the case
        where we are just reordering/merging/splitting the args of an
        existing LoopBody.
        """
        indexing_exprs = other.indexing_from_args(args, allow_same_symbol_in_index)
        self.indexing_exprs = {
            name: self._wrap_int_to_sympy_integer(
                V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges)
            )
            for name, expr in indexing_exprs.items()
        }
        self.subblocks = {k: v.clone(self) for k, v in other.subblocks.items()}
        self.indirect_vars = other.indirect_vars
        self.indirect_var_ranges = other.indirect_var_ranges
        self.memory_usage = other.memory_usage
        self.op_counts = other.op_counts
        self.root_block = other.root_block.clone(self)
        self.has_partial_accumulate = other.has_partial_accumulate

        submodules = {**other.submodules}
        submodules.pop("get_index")
        self.submodules = {
            "get_index": self.get_index,
            **{k: v.clone(self) for k, v in submodules.items()},  # type: ignore[attr-defined]
        }

    def has_op(self, name: str):
        # [FUNCTION DOC] Returns true if the named op was seen at least once.
        return self.op_counts.get(name, 0) > 0

    def merge_loops(self) -> LoopBody:
        # [FUNCTION DOC] Simplifies (merges adjacent) iteration and reduction loops
        # separately using the sizevars loop simplifier, guarded so index orderings
        # are preserved. Returns self if neither size list changes, else a new
        # LoopBody over freshly named vars reindexed from the old ones.
        """
        Merge both iteration and reduction loops and return a new LoopBody.
        """
        old_body = self
        old_sizes = self.sizes
        old_iter_vars, old_reduce_vars = old_body.vars
        old_iter_sizes, old_reduce_sizes = old_sizes

        index_exprs = [*old_body.indexing_exprs.values()]

        iter_sizes, iter_reindex, _ = V.graph.sizevars._simplify_loops(
            old_iter_vars,
            old_iter_sizes,
            index_prevent_reordering(index_exprs, old_iter_vars, old_iter_sizes),
        )

        reduce_sizes, reduce_reindex, _ = V.graph.sizevars._simplify_loops(
            old_reduce_vars,
            old_reduce_sizes,
            index_prevent_reordering(index_exprs, old_reduce_vars, old_reduce_sizes),
        )

        if iter_sizes == old_iter_sizes and reduce_sizes == old_reduce_sizes:
            return old_body

        (
            (
                iter_vars,
                reduce_vars,
            ),
            var_ranges,
        ) = dependencies.index_vars_no_squeeze(iter_sizes, reduce_sizes, prefix="p")
        new_body = LoopBody(
            old_body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )

        return new_body

    def with_indexing_exprs(self, replacements: dict[str, sympy.Expr]) -> LoopBody:
        # [FUNCTION DOC] Returns a copy of this body (same vars and ranges) whose
        # indexing expressions named in replacements are overwritten with the given
        # expressions.
        iter_vars, reduce_vars = self.vars
        new_body = LoopBody(
            self,
            (iter_vars, reduce_vars),
            self.var_ranges,
            iter_vars,
            reduce_vars,
            allow_same_symbol_in_index=True,
        )
        new_body.indexing_exprs.update(replacements)
        return new_body

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> LoopBody:
        # [FUNCTION DOC] Returns a new body whose iteration range along dimension is
        # new_range; the old body is evaluated with that index taken modulo the
        # original range so extra iterations stay in bounds. Traces twice (temporary
        # prefix, then the standard prefix) so later reorderings still work.
        """
        Expand node on `dimension` to `new_range` and rely on index modular to avoid
        out-of-boundary access.
        """

        old_body = self
        old_sizes = self.sizes

        iter_size, reduce_size = old_sizes
        original_range = iter_size[dimension]
        new_iter_size = list(iter_size)
        new_iter_size[dimension] = new_range
        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            # [FUNCTION DOC] HELPER: Flattens the index groups, splits them into
            # iteration and reduction parts, applies modulo of the original range on
            # the expanded dimension, and calls the old body. Raises AssertionError on
            # an unexpected index count.
            index = [*itertools.chain.from_iterable(indices)]
            if not len(index) == len(iter_size) + len(reduce_size):
                raise AssertionError(
                    f"Expected index length {len(index)} to equal "
                    f"iter_size + reduce_size ({len(iter_size)} + {len(reduce_size)})"
                )
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]

            new_iter_idx = list(iter_idx)
            new_iter_idx[dimension] = Mod(iter_idx[dimension], original_range)

            return old_body(new_iter_idx, reduce_idx)

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        # use the original symbol prefix so we can do multiple round of reordering
        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )
        new_body = LoopBody(
            loop_body, (iter_vars2, reduce_vars2), var_ranges2, iter_vars2, reduce_vars2
        )
        return new_body

    def reindex_iter_loops(self, new_iter_sizes: Sequence[sympy.Expr]) -> LoopBody:
        # [FUNCTION DOC] Returns a new body iterating over new_iter_sizes (another
        # factorization of the same total element count; not checked) with the same
        # reduction sizes. Old iteration indices are recovered by flattening the new
        # indices and decomposing by the old sizes. Traced twice (temporary then
        # standard prefix).
        """
        Reindex iteration loops into a different factorization of the same
        total numel. For example, [1024, 8192] -> [65536, 128].

        The old iteration vars are expressed as functions of the new vars via
        FloorDiv and ModularIndexing on the flat index.
        """
        old_body = self
        old_iter_sizes = self.sizes[0]
        reduce_sizes = self.sizes[1]

        new_sizes = (list(new_iter_sizes), list(reduce_sizes))

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="t",  # type: ignore[arg-type]
        )

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            # [FUNCTION DOC] HELPER: Splits the flattened indices into new iteration
            # and reduction parts, flattens the iteration part with the new sizes,
            # decomposes it by the old sizes and calls the old body.
            index = [*itertools.chain.from_iterable(indices)]
            new_iter_idx = index[: len(new_iter_sizes)]
            reduce_idx = index[len(new_iter_sizes) :]
            flat = flatten_index(new_iter_idx, new_iter_sizes)
            old_iter_idx = decompose_index(flat, old_iter_sizes)
            return old_body(old_iter_idx, list(reduce_idx))

        loop_body = LoopBody(
            new_body, (iter_vars, reduce_vars), var_ranges, iter_vars, reduce_vars
        )

        (iter_vars2, reduce_vars2), var_ranges2 = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )
        return LoopBody(
            loop_body,
            (iter_vars2, reduce_vars2),
            var_ranges2,
            iter_vars2,
            reduce_vars2,
        )

    def reorder_iter_loops(self, new_order) -> LoopBody:
        # [FUNCTION DOC] Returns a new body whose iteration loops are permuted by
        # new_order (same length as the iteration sizes) and reduction loops are
        # unchanged. Raises AssertionError on length mismatch.
        """
        Reorder iteration loops and return a new LoopBody.
        """
        from .ir import same_reorder

        old_body = self
        old_sizes = self.sizes
        if not len(old_sizes[0]) == len(new_order):
            raise AssertionError(
                f"Expected old_sizes[0] length {len(old_sizes[0])} to equal "
                f"new_order length {len(new_order)}"
            )
        reorder_fn = same_reorder(new_order)

        iter_size, reduce_size = old_sizes
        new_iter_size = reorder_fn(iter_size)

        new_sizes = (new_iter_size, reduce_size)

        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            *new_sizes,
            prefix="p",  # type: ignore[arg-type]
        )

        inverse_order = {b: a for a, b in enumerate(new_order)}
        inverse_order = [inverse_order[i] for i in range(len(new_order))]

        def new_body(*indices: Sequence[sympy.Expr]) -> Any:
            # [FUNCTION DOC] HELPER: Flattens and splits the indices, undoes the
            # permutation of the iteration indices via the inverse order, and calls
            # the old body allowing repeated symbols. Raises AssertionError on an
            # unexpected index count.
            index = [*itertools.chain.from_iterable(indices)]
            if not len(index) == len(iter_size) + len(reduce_size):
                raise AssertionError(
                    f"Expected index length {len(index)} to equal "
                    f"iter_size + reduce_size ({len(iter_size)} + {len(reduce_size)})"
                )
            iter_idx = index[: len(iter_size)]
            reduce_idx = index[len(iter_size) :]
            iter_idx = [iter_idx[i] for i in inverse_order]
            return old_body(iter_idx, reduce_idx, allow_same_symbol_in_index=True)

        return LoopBody(
            new_body,
            (iter_vars, reduce_vars),
            var_ranges,
            iter_vars,
            reduce_vars,
        )

    @property
    def vars(self):
        # [FUNCTION DOC] Property: returns the pair (iter_vars, reduce_vars).
        # Raises AssertionError if either is None.
        if self.iter_vars is None:
            raise AssertionError("iter_vars is None")
        if self.reduce_vars is None:
            raise AssertionError("reduce_vars is None")
        return self.iter_vars, self.reduce_vars

    @cache_on_self
    def get_nodes(self):
        # [FUNCTION DOC] Cached per instance. Returns a flat list of all FX nodes in
        # the root graph followed by those in every subblock graph.
        all_graphs = itertools.chain(
            (self.root_block.graph,),
            (block.graph for block in self.subblocks.values()),
        )
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        # [FUNCTION DOC] Cached per instance. Lazily imports and returns a BoundVars
        # analysis object built for this body.
        # Doing a local import to avoid dumping all the code here
        from .bounds import BoundVars

        return BoundVars(self)

    def get_read_expr(self, buffer_name):
        # [FUNCTION DOC] Returns the indexing expression of the most recent load from
        # buffer_name. Raises KeyError if the buffer is never loaded.
        # reversed to match old behavior
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_write_expr(self, buffer_name):
        # [FUNCTION DOC] Returns the indexing expression of the first store or
        # reduction store to buffer_name. Raises KeyError if never written.
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                return self.indexing_exprs[entry.index_name]
        raise KeyError(buffer_name)

    def get_read_exprs(self):
        # [FUNCTION DOC] Returns the indexing expressions of all loads, in recorded order.
        return [
            self.indexing_exprs[entry.index_name]
            for entry in self.memory_usage[MemoryUsageType.LOAD]
        ]

    def get_all_read_expr(self, buffer_name):
        # [FUNCTION DOC] Returns indexing expressions of every load from buffer_name,
        # newest first; empty list if none.
        # reversed to match old behavior
        out = []
        for entry in reversed(self.memory_usage[MemoryUsageType.LOAD]):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def get_write_exprs(self):
        # [FUNCTION DOC] Returns indexing expressions of all stores followed by all
        # reduction stores, in recorded order.
        return [
            self.indexing_exprs[entry.index_name]
            for entry in itertools.chain(
                self.memory_usage[MemoryUsageType.STORE],
                self.memory_usage[MemoryUsageType.STORE_REDUCTION],
            )
        ]

    def get_all_write_expr(self, buffer_name):
        # [FUNCTION DOC] Returns indexing expressions of every store and reduction
        # store to buffer_name (stores first); empty list if none.
        out = []
        for entry in itertools.chain(
            self.memory_usage[MemoryUsageType.STORE],
            self.memory_usage[MemoryUsageType.STORE_REDUCTION],
        ):
            if entry.buffer_name == buffer_name:
                out.append(self.indexing_exprs[entry.index_name])
        return out

    def debug_str(self):
        # [FUNCTION DOC] Returns a multi-line string: var ranges, each named indexing
        # expression, then the debug text of the root block (named body) and each
        # subblock. Also used as the repr.
        lines = [f"var_ranges = {dict(self.var_ranges)}"]
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)

    def is_memory_copy(self) -> bool:
        # [FUNCTION DOC] True if the body has exactly one load, one store, only the
        # get_index submodule, and the root block contains only load and store ops.
        """
        True if this contains only a single loads and store.
        Note, this could involve a layout change.
        """
        return (
            len(self.memory_usage[MemoryUsageType.LOAD]) == 1
            and len(self.memory_usage[MemoryUsageType.STORE]) == 1
            and len(self.submodules) == 1  # get_index
            and self.root_block.contains_only_ops(("load", "store"))
        )

    __repr__ = debug_str

    def add_index_expr(
        self,
        expr: sympy.Expr,
        mtype: MemoryUsageType,
        buffer_name: str | None = None,
        mode: str | None = None,
    ):
        # [FUNCTION DOC] Registers expr (ints wrapped to sympy Integers) under a
        # deduplicated name (index0, index1, ...) in the indexing exprs, appends a
        # MemoryEntry for the usage type, and returns the name. Only valid during
        # tracing (needs the temporary name mapping).
        expr = self._wrap_int_to_sympy_integer(expr)
        name = self.indexing_exprs_name.get(expr)
        if not name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        self.memory_usage[mtype].append(MemoryEntry(name, buffer_name, mode))
        return name

    def add_submodule(self, block, prefix):
        # [FUNCTION DOC] Stores block in submodules and returns its name: the prefix
        # itself if it ends in a digit and is unused, else the prefix plus the current
        # submodule count.
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        # [FUNCTION DOC] Creates a new indirect index symbol numbered by the current
        # count, records it with its range size, and returns it.
        # Raises AssertionError if the symbol already exists.
        var = sympy_index_symbol_with_prefix(SymT.INDIRECT, len(self.indirect_vars))
        if var in self.indirect_var_ranges:
            raise AssertionError(f"Indirect var {var} already in indirect_var_ranges")
        self.indirect_vars.append(var)
        self.indirect_var_ranges[var] = size
        return var

    def replace_indirect(self, old, new):
        # [FUNCTION DOC] Substitutes new for old in every current indexing expression
        # (mutates the temporary indexing mapping). No-op if both print the same.
        # Raises AssertionError if indexing is not set (outside a call).
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return
        if self.indexing is None:
            raise AssertionError("indexing must be set before replace_indirect")
        # pyrefly: ignore [bad-assignment]
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        # [FUNCTION DOC] Returns the current concrete indexing expression for name.
        # Raises AssertionError if called outside a body call.
        if self.indexing is None:
            raise AssertionError("indexing must be set before get_index")
        return self.indexing[name]

    def indexing_from_args(self, indices, allow_same_symbol_in_index=False):
        # [FUNCTION DOC] Flattens the index groups and substitutes them for the body's
        # variables in every indexing expression; returns the name to expression
        # mapping. Raises AssertionError if the count differs from var_ranges, or (unless
        # allowed) if an argument is itself one of the body's variables.
        index = [*itertools.chain.from_iterable(indices)]
        if not len(index) == len(self.var_ranges):
            raise AssertionError(f"Index length mismatch: {index} vs {self.var_ranges}")
        if not allow_same_symbol_in_index and not all(
            v not in self.var_ranges for v in index
        ):
            raise AssertionError(
                f"Same symbol found in index: {self.var_ranges=}, {indices=}"
            )

        replacements = dict(zip(self.var_ranges.keys(), index))
        return {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    def __call__(self, *indices, allow_same_symbol_in_index=False):
        # [FUNCTION DOC] Evaluates the body for the given index groups: sets the
        # temporary indexing mapping, runs the root block, clears the mapping and
        # returns the result. Not reentrant; the mapping is not cleared on exception.
        self.indexing = self.indexing_from_args(indices, allow_same_symbol_in_index)
        result = self.root_block()
        self.indexing = None
        return result

    def bind_set_indirect_shim(self, var, size, check, wrap_neg):
        # [FUNCTION DOC] Returns a callable set_indirect that, given a new variable,
        # replaces var in the body's indexing with the result of the ops indirect
        # indexing (size, check, wrap_neg). The callable carries a clone attribute to
        # rebind to another body.
        def set_indirect(new_var):
            # [FUNCTION DOC] HELPER: Rewrites var using indirect indexing of new_var.
            self.replace_indirect(
                var, V.ops.indirect_indexing(new_var, size, check, wrap_neg)
            )

        set_indirect.clone = functools.partial(  # type: ignore[attr-defined]
            LoopBody.bind_set_indirect_shim,
            var=var,
            size=size,
            check=check,
            wrap_neg=wrap_neg,
        )
        return set_indirect

    def bind_scan_shim(self, combine_fn):
        # [FUNCTION DOC] Returns a shim callable that forwards (dtypes, values) to the
        # ops scan with combine_fn bound; the shim has a clone attribute for rebinding.
        def shim(dtypes, values):
            # [FUNCTION DOC] HELPER: Calls the ops scan with the bound combine function.
            return V.ops.scan(dtypes, combine_fn, values)

        shim.clone = functools.partial(LoopBody.bind_scan_shim, combine_fn=combine_fn)  # type: ignore[attr-defined]
        return shim

    def bind_masked_shim(self, name):
        # [FUNCTION DOC] Returns a shim callable that evaluates the named masked
        # subblock via the ops masked operation for a given mask and other value; the
        # shim has a clone attribute to rebind to another body.
        def shim(mask, other):
            # [FUNCTION DOC] HELPER: Calls the ops masked op with the named subblock.
            return V.ops.masked(mask, self.subblocks[name], other)

        shim.clone = functools.partial(LoopBody.bind_masked_shim, name=name)  # type: ignore[attr-defined]
        return shim


class LoopBodyBlock:
    # [CLASS DOC] One FX graph of a LoopBody, produced by running a function under an
    # ops handler stack (op counting, index capture, optional index propagation).
    # Holds the owning body and the traced graph; ops.masked creates extra blocks.
    # Callable: interprets the graph with the current ops handler.
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, however in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: list[Any]):
        # [FUNCTION DOC] Traces fn(args) into a new FX graph: wraps a proxy ops object
        # in CaptureIndexing and CountOps (adding to the body's op counts), and in
        # IndexPropagation when config.constant_and_index_propagation is on, installs
        # it as the ops handler while calling fn, and emits the result as output.
        # Side effects: mutates the body's indexing, memory usage, indirect vars.
        self.body = body

        tracer = LightTracer()
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})

        from .index_propagation import IndexPropagation

        handler: Any = CountOps(
            CaptureIndexing(
                # pyrefly: ignore[bad-argument-type]
                proxy_ops,
                body,
                tracer,
            ),
            body.op_counts,
        )
        if config.constant_and_index_propagation:
            handler = IndexPropagation(
                handler, self.body.var_ranges, self.body.indirect_var_ranges
            )

        with V.set_ops_handler(handler):
            # This indirection is just a cute way to get IndexPropagation to
            # unwrap the return value.
            ops.output(fn(*args))
        self.graph = tracer.graph

    def extract_pw_from_reduction(self):
        # [FUNCTION DOC] Mutates the graph in place: finds the single reduction node
        # and single store_reduction node, inserts a partial_accumulate method call
        # (ops, buffer, reduction type, reduction arg, metadata with the number of
        # reduction dims) after the store, then erases both old nodes. Returns self.
        # Raises AssertionError if either node is missing or duplicated.
        red = None
        store = None
        for node in self.graph.nodes:
            if node.target == "reduction":
                if red:
                    raise AssertionError("Found multiple reduction nodes")
                red = node
            if node.target == "store_reduction":
                if store:
                    raise AssertionError("Found multiple store_reduction nodes")
                store = node
        if not red:
            raise AssertionError("No reduction node found in graph")
        if not store:
            raise AssertionError("No store_reduction node found in graph")
        reduction_type = red.args[-2]
        red_arg = red.args[-1]
        buf = store.args[1]
        ops = store.args[0]

        extra_meta = {
            "num_reduction_dims": len(self.body.reduce_vars),
        }
        with self.graph.inserting_after(store):
            self.graph.call_method(
                "partial_accumulate", (ops, buf, reduction_type, red_arg, extra_meta)
            )
        self.graph.erase_node(store)
        self.graph.erase_node(red)
        return self

    def __call__(self):
        # [FUNCTION DOC] Interprets the graph with the body's submodules, passing the
        # currently active ops handler as the input; returns the graph's output.
        graph = self.graph
        submodules = self.body.submodules

        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name="block"):
        # [FUNCTION DOC] Returns generated Python source of the graph as a string,
        # with the function renamed to name and trailing deletion statements removed.
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # strip `; del var0` suffixes to make output prettier
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )

    def contains_only_ops(self, allowed_ops) -> bool:
        # [FUNCTION DOC] True if every method-call node in the graph targets an op
        # listed in allowed_ops.
        return all(
            node.target in allowed_ops
            for node in self.graph.find_nodes(op="call_method")
        )

    def clone(self, body: LoopBody):
        # [FUNCTION DOC] Returns a shallow copy (graph shared) bound to a new body.
        """Shallow copy with a new parent LoopBody"""
        copy = LoopBodyBlock.__new__(LoopBodyBlock)
        copy.__dict__.update({**self.__dict__, "body": body})
        return copy


class CountOps(DefaultHandler):
    # [CLASS DOC] HELPER: Ops handler wrapper that increments a per-op-name counter
    # for every op call and forwards the call unchanged to the inner handler.
    def __init__(self, inner: OpsHandler[Any], counts: collections.Counter[str]):
        # [FUNCTION DOC] HELPER: Stores the inner handler and the shared counter.
        self._inner = inner
        self._counts = counts

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Counts the op name, then calls the same-named method
        # on the inner handler and returns its result.
        self._counts[name] += 1
        return getattr(self._inner, name)(*args, **kwargs)


class CaptureIndexing(WrapperHandler):
    # [CLASS DOC] Ops handler that sits between traced code and the FX proxy ops.
    # It simplifies index expressions with the body's ranges, registers them in the
    # body as named indexing expressions (recording memory usage), replaces them in
    # the call by get_index proxies, and handles masked, scan, sort, reduction and
    # indirect-indexing ops by creating submodule calls or unpacking tuples.
    name = "CaptureIndexing"

    def __init__(
        self,
        inner: OpsHandler[Any],
        body: LoopBody,
        tracer: LightTracer,
    ):
        # [FUNCTION DOC] Wraps the inner handler and keeps the owning body and tracer.
        super().__init__(inner)
        self.body = body
        self.tracer = tracer

    def _add_index(self, expr: sympy.Expr, mtype: MemoryUsageType, **kwargs: Any):
        # [FUNCTION DOC] HELPER: Registers expr in the body (extra keyword args such as
        # buffer name and mode go to the body) and returns a get_index call proxy
        # referring to the assigned name.
        return self.tracer.create_proxy(
            "call_module",
            "get_index",
            (self.body.add_index_expr(expr, mtype, **kwargs),),
            {},
        )

    def _simplify(self, expr: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] HELPER: Simplifies expr using the body's variable ranges.
        return V.graph.sizevars.simplify_with_ranges(expr, self.body.var_ranges)

    def load(self, name: str, index: sympy.Expr):
        # [FUNCTION DOC] Simplifies and registers the index as a load usage for buffer
        # name, then forwards the load with the index proxy to the inner handler.
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.LOAD, buffer_name=name)
        return self._inner.load(name, index)

    def load_seed(self, name: str, index: int):
        # [FUNCTION DOC] Records a load-seed usage with the integer index as a sympy
        # Integer expression, then forwards the original integer to the inner handler.
        # Raises AssertionError if index is not an int.
        if not isinstance(index, int):
            raise AssertionError(f"Expected int, got {type(index)}")
        self.body.add_index_expr(
            sympy.Integer(index), MemoryUsageType.LOAD_SEED, buffer_name=name
        )
        return self._inner.load_seed(name, index)

    def store(self, name, index, value, mode=None):
        # [FUNCTION DOC] Simplifies and registers the index as a store usage (with
        # mode) for buffer name, then forwards the store to the inner handler.
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE, buffer_name=name, mode=mode
        )
        return self._inner.store(name, index, value, mode)

    def store_reduction(self, name, index, value):
        # [FUNCTION DOC] Same as store but records a store-reduction usage and
        # forwards store_reduction to the inner handler.
        index = self._simplify(index)
        index = self._add_index(
            index, MemoryUsageType.STORE_REDUCTION, buffer_name=name
        )
        return self._inner.store_reduction(name, index, value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        # [FUNCTION DOC] Forwards the reduction to the inner handler; if the reduction
        # type yields multiple outputs, unpacks the proxy result into a tuple of that
        # length, else returns the result as is.
        result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
        num_outputs = reduction_num_outputs(reduction_type)
        if num_outputs > 1:
            return tuple(result[i] for i in range(num_outputs))
        return result

    def index_expr(self, index, dtype):
        # [FUNCTION DOC] Simplifies the index; if it is a constant integer emits a
        # constant of dtype instead, otherwise registers it as an index-expression
        # usage and forwards index_expr with the proxy.
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.index_expr(index, dtype)

    def value_expr(self, index, dtype):
        # [FUNCTION DOC] Like index_expr, but forwards value_expr to the inner handler
        # for non-constant expressions (constants still become constants of dtype).
        index = self._simplify(index)
        if isinstance(index, (int, sympy.Integer)):
            return self._inner.constant(int(index), dtype)
        index = self._add_index(index, MemoryUsageType.INDEX_EXPR)
        return self._inner.value_expr(index, dtype)

    def check_bounds(self, index, size, lower, upper):
        # [FUNCTION DOC] Simplifies and registers both the index and the size (ints
        # wrapped to sympy Integers) as check-bounds usages, then forwards the check
        # with proxies and the lower/upper flags.
        index = self._simplify(index)
        index = self._add_index(index, MemoryUsageType.CHECK_BOUNDS)
        size = self.body._wrap_int_to_sympy_integer(size)
        size = self._add_index(size, MemoryUsageType.CHECK_BOUNDS)
        return self._inner.check_bounds(index, size, lower, upper)

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> T:
        # [FUNCTION DOC] Registers the three index expressions of the boundaries
        # descriptor (and the two of the optional sorter descriptor) as bucketize
        # usages tied to their buffer names, replacing them with proxies, then forwards
        # the bucketize op with the rewritten descriptors.
        """
        See [Note: Inductor bucketize op]
        """
        boundaries = (
            boundaries[0],
            self._add_index(
                boundaries[1],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[2],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
            self._add_index(
                boundaries[3],
                MemoryUsageType.BUCKETIZE,
                buffer_name=boundaries[0],
            ),
        )
        if sorter is not None:
            sorter = (
                sorter[0],
                self._add_index(
                    sorter[1], MemoryUsageType.BUCKETIZE, buffer_name=sorter[0]
                ),
            )

        return self._inner.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )

    def masked(self, mask_proxy, masked_body: Callable[..., Any], other_proxy):
        # [FUNCTION DOC] Traces masked_body into a new LoopBodyBlock stored as a
        # masked subblock of the body, registers a masked shim submodule under the
        # same generated name, and returns a call proxy taking the mask and other value.
        """
        Recursively capture the masked out body in another LoopBodyBlock
        """
        name = self.body.add_submodule(None, "masked_subblock")
        self.body.submodules[name] = self.body.bind_masked_shim(name)
        self.body.subblocks[name] = LoopBodyBlock(self.body, masked_body, [])
        return self.tracer.create_proxy(
            "call_module", name, (mask_proxy, other_proxy), {}
        )

    def scan(
        self,
        dtype_proxy,
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        value_proxy,
    ):
        # [FUNCTION DOC] Registers a scan shim for combine_fn as a submodule, emits a
        # call proxy with the dtype and value proxies, and returns the results as a
        # tuple with one element per input value.
        shim = self.body.bind_scan_shim(combine_fn)
        name = self.body.add_submodule(shim, "scan")
        result = self.tracer.create_proxy(
            "call_module",
            name,
            (dtype_proxy, value_proxy),
            {},
        )
        # Proxies are iterable, but some methods expect tuples/lists
        return tuple(result[i] for i in range(len(value_proxy)))

    def sort(self, dtypes, values, stable, descending):
        # [FUNCTION DOC] Forwards the sort to the inner handler and unpacks the proxy
        # result into a tuple with one element per input value.
        result = self._inner.sort(dtypes, values, stable, descending)
        # Proxies are iterable, but some methods expect tuples/lists
        return tuple(result[i] for i in range(len(values)))

    def frexp(self, value_proxy):
        # [FUNCTION DOC] Forwards frexp to the inner handler and returns the
        # (mantissa, exponent) pair as a plain tuple.
        result = self._inner.frexp(value_proxy)
        # Proxies are iterable, but some methods expect tuples/lists
        return (result[0], result[1])

    def indirect_indexing(self, index_proxy, size, check=True, wrap_neg=True):
        # [FUNCTION DOC] Allocates a new indirect symbol with range size in the body,
        # adds a set_indirect shim submodule and a call proxy on index_proxy that will
        # bind the symbol at run time, and returns the symbol for use in index formulas.
        """
        Flow data from tensors into indexing formulas.
        Introduce a call_module to update the indexing.
        """

        var = self.body.add_indirect(size)
        set_indirect = self.body.bind_set_indirect_shim(var, size, check, wrap_neg)
        self.tracer.create_proxy(
            "call_module",
            self.body.add_submodule(set_indirect, f"set_{var}"),
            (index_proxy,),
            {},
        )
        return var

    def output(self, *result):
        # [FUNCTION DOC] Emits the graph output node for the given result values.
        self.tracer.create_proxy("output", "output", result, {})
