# [MODULE DOC] Inductor pattern matcher: search/replace over FX graphs. Defines a DAG of
# PatternExpr node types (CallFunction, CallMethod, KeywordArg, Arg, ListOf, etc.), Match and
# FailedMatch results, PatternMatcherPass (applies registered patterns to a graph),
# replacement helpers (register_replacement, gen_register_replacement, register_graph_pattern,
# register_lowering_pattern), tracing helpers (fwd_only, joint_fwd_bwd) and a pretty printer
# that serializes patterns to python source for precompiled patterns.
# Import-time side effects: reads env var TORCHINDUCTOR_PATTERN_MATCH_BACKEND (default
# "inductor") into the module global backend, creates a logger, and builds the MULTIPLE
# singleton sentinel plus a few global registries/caches.
"""
# Inductor Pattern Matcher

The pattern matcher enables search/replace within an FX graph.

The main entrypoint to the pattern matcher is register_replacement(). Given a
search function and a replacement function this will register a replacement with
a pass (such as torch._inductor.fx_passes.joint_graph.patterns).

Internally the pattern matcher represents patterns as a graph (a DAG). Creating
new patterns manually as a graph is cumbersome and error-prone so the standard
way to create patterns (using register_replacement()) is to provide a search
function and a replacement function which is traced and converted into a graph.

Because the search functions are built somewhat generic (they tend to ignore
tensor sizes, for example) register_replacement() allows you to specify an
`extra_check` function which performs additional checks to verify that the
matched pattern fully matches before returning it.

## Precompiled Patterns

New patterns are added using register_replacement(). Patterns added in this way
can have a compile-time overhead because they need to be traced before
use. Patterns can be precompiled and added using gen_register_replacement()
instead. To do this you call gen_register_replacement() instead of
register_replacement(). The arguments are the same except for an additional
unique name which is used as a lookup key.

## Internals

The match DAG is represented by a graph of `PatternExpr` nodes. Each PatternExpr
implements a `_match` method which returns either a `Match` object for a
successful match or a `FailedMatch` object for a failure to match.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import inspect
import itertools
import logging
import math
import operator
import os
import re
import textwrap
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypeVar
from typing_extensions import Self, TypeIs

import torch
import torch._guards
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import counters
from torch._prims_common import is_integer_dtype
from torch._subclasses.fake_tensor import (
    is_fake_tensor,
    maybe_get_fake_constant,
    unset_fake_temporarily,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true
from torch.fx.graph_module import _get_attr
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.traceback import preserve_node_meta
from torch.utils._ordered_set import OrderedSet

from .._functorch import config as functorch_config
from .._functorch.aot_autograd import aot_function, make_boxed_func
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensor, FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

Constant = Any
NodeOrConstant = Constant | torch.fx.Node

backend = os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_BACKEND", "inductor")


_debug_nodes_cache: bool | OrderedSet[str] | None = None
_debug_nodes_env_value_cache: str | None = None


def _should_debug_node(node_name: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if debug logging is enabled for the node with this
    # name, based on env var TORCHINDUCTOR_PATTERN_MATCH_DEBUG ("all" enables every node, otherwise
    # a comma separated list of node names; unset/empty disables). The parsed value is cached in
    # module globals and re-parsed only when the env var value changes.
    def _get_debug_nodes() -> bool | OrderedSet[str]:
        # [FUNCTION DOC] HELPER: Returns the cached parsed debug setting (bool or set of node
        # names), refreshing the module-level cache if the env var changed.
        global _debug_nodes_cache, _debug_nodes_env_value_cache

        def parse_debug_env(env_value: str | None) -> bool | OrderedSet[str]:
            # [FUNCTION DOC] HELPER: Parses the env string: empty -> False, "all" -> True,
            # otherwise an ordered set of comma separated names.
            if not env_value:
                return False
            if env_value == "all":
                return True
            return OrderedSet(env_value.split(","))

        current_env = os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG")

        # Recompute only if env changed
        if current_env != _debug_nodes_env_value_cache or _debug_nodes_cache is None:
            _debug_nodes_cache = parse_debug_env(current_env)
            _debug_nodes_env_value_cache = current_env

        return _debug_nodes_cache

    debug_nodes = _get_debug_nodes()
    if isinstance(debug_nodes, bool):
        return debug_nodes
    return node_name in debug_nodes


class SearchFn(Protocol):
    # [CLASS DOC] Typing protocol for a pattern-search callable: any callable with a __name__
    # that takes arbitrary arguments and returns anything (traced into a pattern graph).
    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class ReplaceFn(Protocol):
    # [CLASS DOC] Typing protocol for a replacement callable (traced into a replacement graph).
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class TraceFn(Protocol):
    # [CLASS DOC] Typing protocol for a tracer: takes a search/replace function plus example
    # arguments and returns a traced torch.fx.GraphModule.
    def __call__(
        self, fn: SearchFn | ReplaceFn, *args: Any, **kwargs: Any
    ) -> torch.fx.GraphModule: ...


T = TypeVar("T")

# What's a better name for this?
FnsType = torch.fx.node.Target | str


class Multiple:
    # [CLASS DOC] HELPER: Type of the MULTIPLE sentinel meaning "any number of matched values".
    # Must be a singleton.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Raises AssertionError if a MULTIPLE global already exists and
        # this instance is not it, enforcing the singleton property.
        # Ensure we're really a singleton.
        if "MULTIPLE" in globals() and self is not MULTIPLE:
            raise AssertionError("Multiple is not a singleton")


# Sentinel indicating multiple quantities can be matched
MULTIPLE = Multiple()


def _transfer_meta(
    new_meta: dict[str, Any], old_node: torch.fx.Node, pass_name: str = ""
) -> None:
    # [FUNCTION DOC] HELPER: Copies metadata from a replaced node into new_meta (mutated in
    # place): the torch.fx.proxy copy-meta fields, stack_trace, and val / tensor_meta only if
    # missing. At provenance tracking level 1 also appends a REPLACE NodeSource entry to
    # from_node naming pass_name.
    from torch.fx.traceback import NodeSource, NodeSourceAction

    # Transfer metadata after pattern matching occurs.
    # Copies _COPY_META_FIELDS, stack_trace, and (if missing) val/tensor_meta.
    if config.effective_provenance_tracking_level() == 1:
        new_from_node = new_meta.get("from_node", []).copy()
        new_from_node.append(NodeSource(old_node, pass_name, NodeSourceAction.REPLACE))
        new_meta.update(
            (k, v)
            for k, v in old_node.meta.items()
            if k in torch.fx.proxy._COPY_META_FIELDS
        )
        new_meta["from_node"] = new_from_node
    else:
        new_meta.update(
            (k, v)
            for k, v in old_node.meta.items()
            if k in torch.fx.proxy._COPY_META_FIELDS
        )
    if "stack_trace" in old_node.meta:
        new_meta["stack_trace"] = old_node.meta["stack_trace"]
    # Copy val/tensor_meta only when the new node doesn't already have them
    # (e.g. from tracing the replacement graph). Don't overwrite if present
    # since the replacement's own val is more accurate.
    if "val" not in new_meta and "val" in old_node.meta:
        new_meta["val"] = old_node.meta["val"]
    if "tensor_meta" not in new_meta and "tensor_meta" in old_node.meta:
        new_meta["tensor_meta"] = old_node.meta["tensor_meta"]


def _common_custom_context(nodes: Sequence[torch.fx.Node]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the shared custom context (stream, mempool, mempool_device)
    # of the nodes' meta "custom" entries, or an empty mapping if there are no nodes or they
    # disagree. Only includes keys that are set (stream when non-default or explicitly present).
    if not nodes:
        return {}

    stream = nodes[0].meta.get("custom", {}).get("stream", 0)
    mempool = nodes[0].meta.get("custom", {}).get("mempool")
    mempool_device = nodes[0].meta.get("custom", {}).get("mempool_device")
    if any(
        (
            node.meta.get("custom", {}).get("stream", 0),
            node.meta.get("custom", {}).get("mempool"),
            node.meta.get("custom", {}).get("mempool_device"),
        )
        != (stream, mempool, mempool_device)
        for node in nodes[1:]
    ):
        return {}

    context: dict[str, Any] = {}
    if stream != 0 or any("stream" in node.meta.get("custom", {}) for node in nodes):
        context["stream"] = stream
    if mempool is not None:
        context["mempool"] = mempool
        context["mempool_device"] = mempool_device
    return context


def _merge_custom_context(
    new_meta: dict[str, Any], custom_context: dict[str, Any]
) -> None:
    # [FUNCTION DOC] HELPER: Merges custom_context into new_meta["custom"] in place, without
    # overriding keys already present. No-op for an empty context.
    # Replacement nodes inherit user stream/mempool context only at explicit
    # graph transform hooks. Other transforms must preserve meta["custom"] or
    # avoid moving context-tagged values across boundaries.
    if not custom_context:
        return
    custom = new_meta.setdefault("custom", {})
    for key, value in custom_context.items():
        custom.setdefault(key, value)


class Match:
    # [CLASS DOC] Result of a successful pattern match. Attributes: pattern (matched PatternExpr),
    # args / kwargs (captured inputs for the handler), nodes (matched graph nodes),
    # targets (map from call patterns to the actual node target), ctx (MatchContext) and
    # replacement_graph (set once a replacement is built). Methods merge matches, bundle args,
    # erase matched nodes and replace the match with a graph or traced function.
    """
    Represents a successfully matched pattern.

    The `Match` object is returned to represent a successfully matched
    pattern. Included in the Match are the pattern that was matched, the graph
    nodes matched, and any args that were used during the matching.

    The args and kwargs are specific to the type of pattern that was matched and
    provide hints about what was matched.
    """

    pattern: PatternExpr
    args: list[Any]
    kwargs: dict[str, Any]
    nodes: list[torch.fx.Node]
    targets: dict[_TargetExpr, torch.fx.node.Target]
    ctx: MatchContext
    replacement_graph: torch.fx.GraphModule | None

    def __init__(
        self,
        ctx: MatchContext,
        pattern: PatternExpr,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates a Match for pattern within ctx with copies of the given
        # args (list) and kwargs (mapping); nodes, targets are empty and replacement_graph None.
        super().__init__()
        self.pattern = pattern
        # The input nodes that must be passed in to the result
        self.args = list(args or [])
        self.kwargs = kwargs or {}
        # The nodes matched in this expression
        self.nodes = []
        # Mapping CallFunction to the node.target
        self.targets = {}
        self.ctx = ctx
        self.replacement_graph = None

    @property
    def graph(self) -> torch.fx.Graph:
        # [FUNCTION DOC] Property: the FX graph being matched (from the match context).
        return self.ctx.graph

    def extend(self, other: Match) -> None:
        # [FUNCTION DOC] Merges another Match into this one in place: appends args and nodes,
        # updates kwargs and targets. Raises FailedMatch if a kwarg name present in both has
        # differing values.
        if self.kwargs:
            for key in OrderedSet(self.kwargs.keys()) & OrderedSet(other.kwargs.keys()):
                if self.kwargs[key] != other.kwargs[key]:
                    raise FailedMatch("kwarg mismatch: {}", key)
        self.args.extend(other.args)
        self.nodes.extend(other.nodes)
        self.kwargs.update(other.kwargs)
        self.targets.update(other.targets)

    def bundle(self) -> Match:
        # [FUNCTION DOC] Replaces the args list with a single tuple of all args (or an empty
        # list if none), mutating and returning self.
        # Wrap args in an extra list
        self.args = [tuple(self.args)] if self.args else []
        return self

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Short debug string showing args and kwargs.
        return f"Match(..., {self.args}, {self.kwargs})"

    def erase_nodes(self) -> None:
        # [FUNCTION DOC] Erases matched nodes from the graph in reverse order, skipping nodes
        # already erased or that still have users. Mutates the graph.
        graph = self.graph
        for n in reversed(self.nodes):
            if not n._erased and not n.users:
                graph.erase_node(n)

    def output_nodes(self) -> list[torch.fx.Node | None]:
        # [FUNCTION DOC] Returns the graph node matched by each output pattern of the context
        # (None for output slots that are None).
        return [
            (self.ctx.pattern_to_node[p] if p is not None else None)
            for p in self.ctx.outputs
        ]

    def output_node(self) -> torch.fx.Node:
        # [FUNCTION DOC] Returns the first non-None output node of the match; raises
        # StopIteration if there is none.
        return next(p for p in self.output_nodes() if p)

    def replace_with_graph(
        self, replacement_graph: torch.fx.Graph, args: Sequence[Any]
    ) -> list[torch.fx.Node]:
        # [FUNCTION DOC] Replaces the matched subgraph with replacement_graph called on args,
        # delegating to ReplacementPatternEntry.replace_with_graph. Returns the new nodes.
        return ReplacementPatternEntry.replace_with_graph(
            self, self.ctx.graph, replacement_graph, args
        )

    def replace_by_example(
        self,
        replacement_fn: ReplaceFn,
        args: Sequence[Any],
        trace_fn: TraceFn | None = None,
        run_functional_passes: bool = True,
    ) -> list[torch.fx.Node]:
        # [FUNCTION DOC] Traces replacement_fn (default tracer: fwd_only) using example values
        # taken from the meta (val or example_value) of args, then replaces the matched subgraph
        # with the traced graph. Runs under the virtualized fake mode if one is set. Special
        # case: for a single auto_functionalized node with eager_input_vals, traces twice (with
        # eager vals, then with inductor vals) and copies eager_input_vals onto the second graph
        # (raises AssertionError if node counts differ). For single-node matches, copies
        # metadata onto replacement nodes. Returns the new nodes; mutates the graph.
        """Replace with a graph generated by tracing the replacement_fn.

        Args:
            run_functional_passes (bool). If we should run passes that
                assume functional IR (like DCE, remove_noop_ops), on the
                replacement graph.

        """
        from torch._inductor.virtualized import NullHandler, V

        context = (
            V.fake_mode
            if (not isinstance(V.fake_mode, NullHandler) or (V.fake_mode is None))
            else contextlib.nullcontext()
        )

        def should_propagate_eager_input_vals(nodes: list[torch.fx.Node]) -> bool:
            # [FUNCTION DOC] HELPER: True only when exactly one node matched, it has
            # eager_input_vals metadata, and its target is auto_functionalized(_v2).
            if len(nodes) != 1:
                return False
            node = nodes[0]
            if "eager_input_vals" not in node.meta:
                return False
            return node.target in OrderedSet(
                [
                    torch.ops.higher_order.auto_functionalized,
                    torch.ops.higher_order.auto_functionalized_v2,
                ]
            )

        # pyrefly: ignore [bad-context-manager]
        with context:
            if trace_fn is None:
                trace_fn = functools.partial(
                    fwd_only, run_functional_passes=run_functional_passes
                )

            if should_propagate_eager_input_vals(self.nodes):
                # Our strategy is:
                # 1) trace out the graph with eager_input_vals (which have accurate eager-mode metadata)
                # 2) trace out the graph with vals (which have the accurate Inductor metadata)
                # 3) Propagate the eager_input_vals from the first graph to the second.
                # 4) Use the second graph as the replacement graph.

                # Construct a map of node -> FakeTensor val in eager_input_vals
                node_to_val = {}

                fake_args, fake_kwargs = self.nodes[0].meta["eager_input_vals"]
                fake_kwargs = {**fake_kwargs}
                match_args, match_kwargs = tuple(self.args), self.kwargs

                def record(node: torch.fx.Node, val: Any) -> None:
                    # [FUNCTION DOC] HELPER: Records val for node in node_to_val if node is an FX Node.
                    if isinstance(node, torch.fx.Node):
                        node_to_val[node] = val

                torch.utils._pytree.tree_map(
                    record, (match_args, match_kwargs), (fake_args, fake_kwargs)
                )
                # map args to their FakeTensor val in eager_input_vals
                example_vals = torch.fx.map_arg(args, lambda arg: node_to_val[arg])

                # first graph
                graph_with_eager_vals = trace_fn(replacement_fn, example_vals)

                # second graph
                example_vals = torch.fx.map_arg(args, lambda arg: arg.meta["val"])
                # pyrefly: ignore [bad-argument-type]
                replacement = trace_fn(graph_with_eager_vals, example_vals)

                # propagate metadata from first graph to second
                # NB: This assertion might not be true in general, but it is true for
                # the auto_functionalized use cases we have
                if len(graph_with_eager_vals.graph.nodes) != len(
                    replacement.graph.nodes
                ):
                    raise AssertionError(
                        "graph node count mismatch between eager and replacement"
                    )
                for old_node, new_node in zip(
                    graph_with_eager_vals.graph.nodes, replacement.graph.nodes
                ):
                    if "eager_input_vals" in old_node.meta:
                        new_node.meta["eager_input_vals"] = old_node.meta[
                            "eager_input_vals"
                        ]

            else:
                example_vals = torch.fx.map_arg(
                    args,
                    lambda arg: arg.meta["val"]
                    if "val" in arg.meta
                    else arg.meta["example_value"],
                )
                fake_mode = torch._dynamo.utils.detect_fake_mode(example_vals)
                if fake_mode is not None:

                    def _convert_to_fake_mode(it):
                        # [FUNCTION DOC] HELPER: Re-fakes a fake tensor under the detected fake_mode; other values pass through.
                        if is_fake_tensor(it) and fake_mode is not None:
                            return fake_mode.from_tensor(it)
                        return it

                    example_vals = torch.fx.node.map_aggregate(
                        example_vals, _convert_to_fake_mode
                    )
                replacement = trace_fn(replacement_fn, example_vals)
            if len(self.nodes) == 1:
                for n in replacement.graph.nodes:
                    _transfer_meta(
                        new_meta=n.meta,
                        old_node=self.nodes[0],
                        pass_name="replace_by_example",
                    )

            return ReplacementPatternEntry.replace_with_graph(
                self,
                self.ctx.graph,
                replacement,
                args,
            )


class FailedMatch(RuntimeError):
    # [CLASS DOC] Falsy exception/result representing a failed match. Stores a lazy format
    # string plus args/kwargs, formatted only in __str__ to avoid compile-time cost. Can be
    # raised or returned.
    """
    Represents a unsuccessful match.

    The `FailedMatch` object is returned to represent a failure to match a
    pattern.
    """

    format_string: str

    def __init__(self, format_string: str, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the format string and its args. Raises RuntimeError if the
        # format string is longer than 200 chars (to force lazy message construction).
        self.format_string = format_string
        # We want to construct error messages lazily instead of eagerly, as
        # constructing them eagerly can significantly worsen compile times.
        if len(format_string) > 200:
            raise RuntimeError(
                f"Format string too long - use lazy construction of strings instead. Format string is\n {format_string}"
            )
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        # [FUNCTION DOC] Lazily formats the message from format_string, args and kwargs.
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self) -> bool:
        # [FUNCTION DOC] Always False so a FailedMatch is falsy in truth tests.
        return False


MatchResult = Match | FailedMatch


def is_match(m: MatchResult) -> TypeIs[Match]:
    # [FUNCTION DOC] Returns the truthiness of m (True for Match, False for FailedMatch); exists as a type narrowing helper.
    """
    TypeIs cannot act on `self`. Thus this function exists to let mypy
    recognize FailedMatch.__bool__ as a TypeIs.
    """
    return bool(m)


class MatchContext:
    # [CLASS DOC] Mutable state shared during one match attempt: outputs (output patterns),
    # pattern_to_node (map from each visited pattern to its matched node or None on failure),
    # graph, and exclusive_node_set (nodes that must only be used inside the match).
    """
    Internal state needed while running PatternExpr._match().
    """

    outputs: list[PatternExpr | None]
    pattern_to_node: dict[PatternExpr, torch.fx.Node | None]
    graph: torch.fx.Graph
    exclusive_node_set: list[NodeOrConstant]

    def __init__(
        self,
        outputs: list[PatternExpr | None],
        pattern_to_node: dict[PatternExpr, torch.fx.Node] | None = None,
        *,
        graph: torch.fx.Graph,
    ) -> None:
        # [FUNCTION DOC] Stores outputs and graph, copies the optional pattern_to_node mapping
        # (empty if None) and starts with an empty exclusive_node_set.
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else dict(pattern_to_node)
        self.graph = graph
        self.exclusive_node_set = []

    def match(self, pattern: PatternExpr, node: NodeOrConstant) -> MatchResult:
        # [FUNCTION DOC] Matches pattern against node, memoizing in pattern_to_node. If the
        # pattern was already visited: returns an empty Match when it mapped to the same node,
        # else FailedMatch ("repeated pattern differs"). Otherwise calls pattern._match, records
        # the node (None on failure) and returns the result. Raises AssertionError if the
        # pattern got recorded during its own match.
        """wrapper to check reused nodes in patterns"""
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(self, pattern)  # already checked this node
            else:
                return FailedMatch("repeated pattern differs")
        m = pattern._match(node, self)
        if pattern in self.pattern_to_node:
            raise AssertionError("pattern already present in pattern_to_node")
        self.pattern_to_node[pattern] = node if m else None
        return m

    def filter_multi_user_patterns(self) -> dict[PatternExpr, torch.fx.Node]:
        # [FUNCTION DOC] Returns the subset of pattern_to_node for successfully matched patterns
        # that report has_multiple_users().
        return {
            pattern: node
            for pattern, node in self.pattern_to_node.items()
            if pattern.has_multiple_users() and node is not None
        }


class PatternExpr(ABC):
    # [CLASS DOC] Abstract base of all pattern DAG nodes. Subclasses implement _match(node, ctx)
    # returning Match or FailedMatch; the base supplies a public match entry point, multi-user
    # tracking, anchor-node discovery and structural equality for debugging.
    """
    Base class for types of patterns.
    """

    @abstractmethod
    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult: ...

    def match(self, node: torch.fx.Node) -> MatchResult:
        # [FUNCTION DOC] Matches this pattern against node using a fresh MatchContext on the
        # node's graph. Returns a Match, or the FailedMatch if one was raised.
        try:
            return MatchContext([self], graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def has_multiple_users(self) -> bool:
        # [FUNCTION DOC] Whether this pattern is reused by several consumers; False by default.
        return False

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Class name followed by empty parentheses.
        return self.__class__.__name__ + "()"

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None, None, None]:
        # [FUNCTION DOC] Generator yielding candidate anchor nodes for starting a search: here
        # just the node already bound to this pattern in ctx, if any. Subclasses extend this.
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural comparison (debugging only): True if other is an instance of this class.
        """
        Compare two `PatternExpr`s and return true if they are the
        same. Note this is NOT matching a pattern - it is comparing the pattern
        structures (for debugging).
        """
        return isinstance(other, self.__class__)


class Arg(PatternExpr):
    # [CLASS DOC] Wildcard pattern that matches any node or constant and captures it as an
    # input argument of the handler (args are collected in depth-first order).
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Always succeeds, returning a Match whose args list holds the node.
        return Match(ctx, self, args=[node])  # matches anything


class Ignored(PatternExpr):
    # [CLASS DOC] Wildcard pattern that matches anything but does not capture it as an argument.
    """
    Match an arg, but don't pass it to handler
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Always succeeds with an empty Match.
        return Match(ctx, self)  # matches anything

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns an asterisk.
        return "*"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # [FUNCTION DOC] HELPER: Source text for this pattern in generated pattern files; pp unused.
        return "Ignored()"


def _get_fake_tensor_constant(value: torch.Tensor) -> torch.Tensor | None:
    # [FUNCTION DOC] HELPER: For a fake tensor returns its tracked real constant (or None if none); other tensors are returned unchanged.
    if is_fake_tensor(value):
        return maybe_get_fake_constant(value)
    return value


def _tensor_values_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: Value equality of two tensors that treats NaN as equal to NaN
    # (recursing on real and imaginary parts for complex dtypes, using an elementwise check for
    # floating point). Returns a bool.
    if torch.equal(a, b):
        return True
    if a.dtype.is_complex:
        return _tensor_values_equal(a.real, b.real) and _tensor_values_equal(
            a.imag, b.imag
        )
    if a.dtype.is_floating_point:
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all().item())
    return False


def _constant_values_equal(a: Any, b: Any) -> bool:
    # [FUNCTION DOC] HELPER: Equality test for pattern constants. Tensors are equal only if
    # both are tensors with same dtype, shape, device, layout, requires_grad, (strided) stride
    # and offset, and equal values (compared on the real constants behind fake tensors; NaN
    # equal NaN); missing constants or comparison errors give False. Non-tensors use ordinary
    # equality, returning False if comparison raises or does not produce a bool.
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        if (
            a.dtype != b.dtype
            or a.shape != b.shape
            or a.device != b.device
            or a.layout != b.layout
            or a.requires_grad != b.requires_grad
        ):
            return False
        if a.layout == torch.strided and (
            a.stride() != b.stride() or a.storage_offset() != b.storage_offset()
        ):
            return False
        a_constant = _get_fake_tensor_constant(a)
        b_constant = _get_fake_tensor_constant(b)
        if a_constant is None or b_constant is None:
            return False
        try:
            with unset_fake_temporarily():
                return _tensor_values_equal(a_constant, b_constant)
        except (RuntimeError, TypeError, ValueError):
            return False

    try:
        result = a == b
    except (RuntimeError, TypeError, ValueError):
        return False
    return result if isinstance(result, bool) else False


def _python_constant_repr(value: Any) -> str:
    # [FUNCTION DOC] HELPER: Returns source text that reconstructs a constant. Handles NaN and
    # infinities of floats, complex numbers, lists and tuples (recursively, single-element
    # tuple gets a trailing comma); everything else uses the builtin repr.
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "-float('inf')"
    if isinstance(value, complex):
        return f"complex({_python_constant_repr(value.real)}, {_python_constant_repr(value.imag)})"
    if isinstance(value, list):
        return f"[{', '.join(map(_python_constant_repr, value))}]"
    if isinstance(value, tuple):
        trailing_comma = "," if len(value) == 1 else ""
        return f"({', '.join(map(_python_constant_repr, value))}{trailing_comma})"
    return repr(value)


def _tensor_constant_repr(value: torch.Tensor) -> str:
    # [FUNCTION DOC] HELPER: Returns source text creating the tensor from its values, dtype and
    # (if not cpu) device. Raises NotImplementedError for non-strided layouts, fake tensors
    # without a real constant, tensors requiring grad, and non-contiguous or offset tensors.
    # Copies data to cpu to read values (with fake mode disabled).
    if value.layout != torch.strided:
        raise NotImplementedError(
            f"NYI: serializing get_attr tensor with layout {value.layout}"
        )
    dtype = value.dtype
    device = value.device
    if is_fake_tensor(value):
        constant = maybe_get_fake_constant(value)
        if constant is None:
            raise NotImplementedError("NYI: serializing fake get_attr tensor")
        data_value = constant
    else:
        data_value = value
    if value.requires_grad:
        raise NotImplementedError("NYI: serializing tensor that requires grad")
    if not value.is_contiguous() or value.storage_offset() != 0:
        raise NotImplementedError("NYI: serializing non-contiguous get_attr tensor")
    with unset_fake_temporarily():
        if data_value.device.type != "cpu":
            cpu_value = data_value.detach().cpu()
        else:
            cpu_value = data_value.detach()
        value_list = cpu_value.tolist()

    args = [_python_constant_repr(value_list)]
    args.append(f"dtype={dtype}")
    if device.type != "cpu":
        args.append(f"device={str(device)!r}")
    return f"torch.tensor({', '.join(args)})"


class GetAttr(PatternExpr):
    # [CLASS DOC] Pattern matching a get_attr node by the value of the referenced attribute
    # (not its module-local name). Attributes: value (constant to compare) and users (expected
    # user count, or MULTIPLE).
    """
    Match a get_attr node by comparing the attribute value instead of the target
    name, which is local to the traced GraphModule.
    """

    def __init__(self, value: Any, users: Multiple | int = 1) -> None:
        # [FUNCTION DOC] Stores the constant value and expected user count.
        super().__init__()
        self.value = value
        self.users = users

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with value and, if not 1, the users count.
        if self.users is MULTIPLE:
            comma_users = ", MULTIPLE"
        elif self.users != 1:
            comma_users = f", {self.users}"
        else:
            comma_users = ""
        return f"{self.__class__.__name__}({self.value!r}{comma_users})"

    def has_multiple_users(self) -> bool:
        # [FUNCTION DOC] True if users is MULTIPLE or greater than one.
        return isinstance(self.users, Multiple) or self.users > 1

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Succeeds if node is a get_attr node (user count matching unless this
        # pattern is an output or users is MULTIPLE), its graph has an owning module, its target
        # is a string and the attribute fetched from the module equals self.value. Otherwise
        # returns a FailedMatch describing the reason. The match records the node.
        if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
            return FailedMatch("get_attr_mismatch: node={}, pattern={}", node, self)
        if (
            self not in ctx.outputs
            and self.users is not MULTIPLE
            and len(node.users) != self.users
        ):
            return FailedMatch("multiple_users {}", self)
        if node.graph.owning_module is None:
            return FailedMatch("get_attr without owning module: {}", node)
        if not isinstance(node.target, str):
            return FailedMatch("get_attr target is not str: {}", node.target)
        value = _get_attr(node.graph.owning_module, node.target)
        if not _constant_values_equal(self.value, value):
            return FailedMatch("get_attr_value: {} != {}", value, self.value)
        m = Match(ctx, self)
        m.nodes.append(node)
        return m

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # [FUNCTION DOC] HELPER: Source text for this pattern: class name with the printed value and non-default users.
        args = [pp.pretty_print(self.value)]
        if self.users is MULTIPLE:
            args.append("MULTIPLE")
        elif self.users != 1:
            args.append(str(self.users))
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class, equal constant values and equal users.
        other = typing.cast(Self, other)
        return (
            super().pattern_eq(other)
            and _constant_values_equal(self.value, other.value)
            and self.users == other.users
        )


class KeywordArg(PatternExpr):
    # [CLASS DOC] Wildcard pattern capturing the matched node into the handler's keyword
    # arguments under the given name; the same name matched twice must bind equal values.
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Stores the keyword name.
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the name.
        return f"KeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Always succeeds, binding the node to the keyword name.
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class and same name.
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class ExclusiveKeywordArg(PatternExpr):
    # [CLASS DOC] Like KeywordArg but a given node may be bound by only one such pattern in a
    # match (tracked in the context's exclusive_node_set).
    """
    Capture a kwarg which will become an input to the handler.
    """

    name: str

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Stores the keyword name.
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the name.
        return f"ExclusiveKeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Fails if the node is already in the context's exclusive set; otherwise
        # adds it to that set (side effect on ctx) and binds it to the keyword name.
        if node in ctx.exclusive_node_set:
            return FailedMatch("exclusive arg appears twice")

        ctx.exclusive_node_set.append(node)
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class and same name.
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class _TargetExpr(PatternExpr):
    # [CLASS DOC] HELPER: Base for patterns that filter nodes by target. Attributes: fns (list
    # of allowed targets, with overloads of any OpOverloadPacket appended), fns_set (same as an
    # ordered set) and users (expected user count or MULTIPLE). Subclasses define the op kind.
    """
    Base class for filtering match by node.target
    """

    fns: list[FnsType]
    fns_set: OrderedSet[FnsType]

    def __init__(
        self, fns: FnsType | Sequence[FnsType], users: Multiple | int = 1
    ) -> None:
        # [FUNCTION DOC] Normalizes fns to a list (a single callable or string is wrapped),
        # expands operator packets into their overloads, and stores fns, fns_set and users.
        super().__init__()
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        for fn in fns:
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend(fn.op_overloads())  # noqa: B909

        self.fns = fns
        self.fns_set = OrderedSet(fns)
        self.users = users

    @property
    @abstractmethod
    def op(self) -> str: ...

    def fns_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Short text naming the targets: for multiple fns the first
        # followed by an ellipsis; a torch or operator module function gets a module prefix;
        # an OpOverload uses its string form; otherwise the bare name.
        first_repr = self.fns[0]
        if not isinstance(first_repr, str):
            first_repr = first_repr.__name__

        if len(self.fns) > 1:
            return f"[{first_repr}, ...]"
        elif self.fns[0] is getattr(torch, first_repr, None):
            return f"torch.{first_repr}"
        elif self.fns[0] is getattr(operator, first_repr, None):
            return f"operator.{first_repr}"
        elif isinstance(self.fns[0], torch._ops.OpOverload):
            return str(self.fns[0])
        else:
            return first_repr

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with targets and non-default users.
        # KNOWN ISSUE: when users is a number other than 1 the users text has a stray closing
        # parenthesis, so the output has unbalanced parentheses.
        if self.users is MULTIPLE:
            comma_users = ", MULTIPLE"
        elif self.users != 1:
            comma_users = f", {self.users})"
        else:
            comma_users = ""
        return f"{self.__class__.__name__}({self.fns_repr()}{comma_users})"

    def has_multiple_users(self) -> bool:
        # [FUNCTION DOC] True if users is MULTIPLE or greater than one.
        return isinstance(self.users, Multiple) or self.users > 1

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None, None, None]:
        # [FUNCTION DOC] Not implemented in this base class; always raises NotImplementedError.
        raise NotImplementedError

    def _match_fns(self, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: True if node is an FX node whose op equals this pattern's op
        # and whose extracted target is in fns_set.
        return (
            isinstance(node, torch.fx.Node)
            and node.op == self.op
            and extract_target(node) in self.fns_set
        )

    def _match_users(self, node: torch.fx.Node, ctx: MatchContext) -> bool:
        # [FUNCTION DOC] HELPER: True if this pattern is an output of the context, users is
        # MULTIPLE, or the node's number of users equals the expected count.
        return (
            self in ctx.outputs
            or self.users is MULTIPLE
            or len(node.users) == self.users
        )

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class, op, fns list and users.
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.op == other.op
            and self.fns == other.fns
            and self.users == other.users
        )


_SimpleSpec = tuple[Any, ...]


class _TargetArgsExpr(_TargetExpr):
    # [CLASS DOC] Base for patterns matching a node by target plus args and kwargs. Each
    # argument may be a nested PatternExpr or a constant. Args are flattened (simple or via
    # pytree when containers are present) into flat_args_kwargs (values, structure spec) for
    # comparison with node arguments. Subclasses CallFunction/CallMethod/CallModule set op.
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    def __init__(
        self,
        fns: torch.fx.node.Target | str | Sequence[Any],
        *args: Any,
        _users: int | Multiple = 1,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores targets, user count, args (tuple) and kwargs (mapping); chooses
        # the pytree flatten when any argument is a dict, list or tuple, else the simple flatten;
        # precomputes flat_args_kwargs.
        super().__init__(fns, _users)
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        if any(
            isinstance(x, (dict, list, tuple))
            for x in itertools.chain(args, kwargs.values())
        ):
            self.flatten = self.pytree_flatten
        else:
            self.flatten = self.simple_flatten
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    @staticmethod
    def simple_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], _SimpleSpec | pytree.TreeSpec]:
        # [FUNCTION DOC] HELPER: Flattens to the positional values followed by keyword values;
        # the spec is the number of positional args plus the keyword names.
        values = (*args, *kwargs.values())
        spec = (len(args), *kwargs.keys())
        return values, spec

    @staticmethod
    def pytree_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> tuple[Sequence[Any], _SimpleSpec | pytree.TreeSpec]:
        # [FUNCTION DOC] HELPER: Flattens (args, kwargs) with pytree after normalizing container
        # types (immutable lists and lists become tuples, immutable dicts become dicts) so FX
        # and pattern containers compare equal. Returns the flat leaves and the tree spec.
        type_mapping: dict[type, type] = {
            immutable_list: tuple,
            list: tuple,
            immutable_dict: dict,
        }

        def convert_type(x: Any) -> Any:
            # [FUNCTION DOC] HELPER: Converts x via type_mapping (recursively over contents) if its exact type is mapped; else returns x.
            cls = type(x)
            convert_fn = type_mapping.get(cls)
            if convert_fn is not None:
                return pytree.tree_map(
                    convert_type,
                    convert_fn(x),
                    is_leaf=lambda x: type(x) in type_mapping,
                )
            return x

        normalized_args_tree = pytree.tree_map(
            convert_type,
            (args, kwargs),
            is_leaf=lambda x: type(x) in type_mapping,
        )
        flat, spec = pytree.tree_flatten(normalized_args_tree)
        return flat, spec

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string: class name with targets, args, kwargs and non-default users.
        args = [
            self.fns_repr(),
            *map(repr, self.args),
            *[f"{k}={v}" for k, v in self.kwargs.items()],
        ]
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        elif self.users != 1:
            args.append(f"_users={self.users}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # [FUNCTION DOC] HELPER: Source text of this pattern for generated files, printing
        # nested args and kwargs through pp and adding _users when not the default.
        args = [
            self.fns_repr(),
            *(pp.pretty_print(x) for x in self.args),
            *[f"{k}={pp.pretty_print(v)}" for k, v in self.kwargs.items()],
        ]
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        elif self.users != 1:
            args.append(f"_users={self.users}")

        joiner_str = ", "
        return f"{self.__class__.__name__}({joiner_str.join(args)})"

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Matches node against target, user count, and arguments. Requires same
        # number of positional args; if the node has fewer kwargs than the pattern, normalizes
        # the call via the target's signature to fill defaults (raises AssertionError if the
        # target is not callable); keeps only kwargs named by the pattern. Flattens node args and
        # compares structure; child PatternExprs are matched through ctx and merged, constants
        # must equal and not be nodes. On success records the node and target. Returns the
        # first FailedMatch encountered otherwise.
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch("function_mismatch: node={}, pattern={}", node, self)

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users {}", self)

        _args = node.args
        _kwargs = node.kwargs
        if len(_kwargs) < len(self.kwargs):
            from torch.fx.operator_schemas import normalize_function

            if not callable(node.target):
                raise AssertionError(
                    f"expected callable node.target, got {node.target}"
                )
            normalized_args_and_kwargs = normalize_function(
                node.target, node.args, node.kwargs
            )

            if normalized_args_and_kwargs is None:
                return FailedMatch("function_mismatch: node={}, pattern={}", node, self)
            else:
                _args, _kwargs = normalized_args_and_kwargs
                if len(_args) == len(self.args) and len(_kwargs) >= len(self.kwargs):
                    _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
                else:
                    return FailedMatch(
                        "function_mismatch: node={}, pattern={}", node, self
                    )
        else:
            _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}

        node_items, node_spec = self.flatten(_args, _kwargs)
        self_items, self_spec = self.flat_args_kwargs
        if node_spec != self_spec:
            return FailedMatch("args_structure {} {}", node_spec, self_spec)
        if len(node_items) != len(self_items):
            raise AssertionError("node_items and self_items length mismatch")

        m = Match(ctx, self)
        for pattern, child_node in zip(self_items, node_items):
            if isinstance(pattern, PatternExpr):
                child_match = ctx.match(pattern, child_node)
                if not is_match(child_match):
                    return child_match
                m.extend(child_match)
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch(
                    "constant_args: {} {!r}!={pattern!r}",
                    node,
                    child_node,
                    pattern=pattern,
                )
        m.nodes.append(node)
        m.targets[self] = node.target
        return m

    def find_anchor_nodes(
        self, ctx: MatchContext, searched: OrderedSet[torch.fx.Node]
    ) -> Generator[torch.fx.Node | None, None, None]:
        # [FUNCTION DOC] Generator for multi-output matching: yields the node already bound to
        # this pattern if any; otherwise walks child patterns' anchors and yields unsearched
        # users of those anchor nodes whose op/target match, marking them as searched.
        """
        This is used when we are matching a pattern with multiple outputs.
        There is a partial match (stored in ctx) and we want to walk
        this pattern to find a connection to an already-matched node.

        Yields candidate nodes that `self._match` might like.
        """
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]
            return

        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(node):
                                yield node
                                searched.add(node)

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: base checks, same flatten spec, and pairwise equal
        # flattened values (using pattern_eq for sub-patterns).
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.flat_args_kwargs[1] == other.flat_args_kwargs[1]
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.flat_args_kwargs[0], other.flat_args_kwargs[0])
            )
        )


class CallFunction(_TargetArgsExpr):
    # [CLASS DOC] Pattern for call_function nodes (op is call_function).
    """
    Matches a call_function node in the FX graphs: `fns[i](*args, **kwargs)`
    """

    op = "call_function"


class CallMethod(_TargetArgsExpr):
    # [CLASS DOC] Pattern for call_method nodes (op is call_method).
    """
    Matches a call_method node in the FX graphs: `fns[i].method(*args, **kwargs)`
    """

    op = "call_method"


class CallModule(_TargetArgsExpr):
    # [CLASS DOC] Pattern for call_module nodes (op is call_module).
    """
    Matches a call_module node in the FX graphs: `module(*args, **kwargs)`
    """

    op = "call_module"


class _TargetExprVarArgs(_TargetExpr):
    # [CLASS DOC] HELPER: Base for patterns matching by target and user count only, capturing
    # all of the node's args and kwargs as handler inputs. Subclasses set the op.
    """
    Matches a call_function node with any arguments which are passed into the pattern
    """

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Fails on target or user-count mismatch; otherwise returns a Match
        # holding the node, its target, and all its args and kwargs.
        if not self._match_fns(node):
            return FailedMatch("function_mismatch")

        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users")

        m = Match(ctx, self)
        m.nodes.append(node)
        m.targets[self] = node.target
        m.args.extend(node.args)
        m.kwargs.update(node.kwargs)
        return m


class CallFunctionVarArgs(_TargetExprVarArgs):
    # [CLASS DOC] Variable-argument pattern for call_function nodes.
    op = "call_function"


class CallMethodVarArgs(_TargetExprVarArgs):
    # [CLASS DOC] Variable-argument pattern for call_method nodes.
    op = "call_method"


class CallModuleVarArgs(_TargetExprVarArgs):
    # [CLASS DOC] Variable-argument pattern for call_module nodes.
    op = "call_module"


class ListOf(PatternExpr):
    # [CLASS DOC] Pattern matching a list/tuple of nodes where every element matches the inner
    # pattern (or, if partial, at least one does). Attributes: pattern and partial.
    """
    Matches a repeated pattern
    """

    def __init__(self, pattern: PatternExpr, partial: bool = False) -> None:
        # [FUNCTION DOC] Stores the element pattern and the partial flag. Raises AssertionError
        # if pattern is not a PatternExpr.
        super().__init__()
        if not isinstance(pattern, PatternExpr):
            raise AssertionError(f"expected PatternExpr, got {type(pattern)}")
        self.pattern = pattern
        self.partial = partial

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the inner pattern.
        return f"{self.__class__.__name__}({self.pattern})"

    def _match(self, node: list[torch.fx.Node], ctx: MatchContext) -> MatchResult:  # type: ignore[override]
        # [FUNCTION DOC] Expects a non-empty list or tuple of nodes, else FailedMatch. Matches
        # the inner pattern against each element in a child context that carries over only
        # multi-user pattern bindings (so shared nodes are not revisited). Non-matching elements
        # fail the whole match unless partial (then skipped); at least one element must match.
        # Each element's match args are bundled into a tuple; the final result is bundled too.
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch("non_list")
        m = Match(ctx, self)
        # Propagating patterns with multiple users will ensure we don't revisit
        # the same nodes
        pattern_to_node = ctx.filter_multi_user_patterns()
        matched = False
        for i, child_node in enumerate(node):
            child_ctx = MatchContext(
                ctx.outputs, pattern_to_node, graph=child_node.graph
            )
            child_match = child_ctx.match(self.pattern, child_node)
            pattern_to_node = child_ctx.filter_multi_user_patterns()
            if not is_match(child_match):
                if not self.partial:
                    return FailedMatch("list[{}]: {}", i, child_match)
                continue
            matched = True
            m.extend(child_match.bundle())
        if not matched:
            return FailedMatch("list: no_match")
        return m.bundle()

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class, equal inner pattern, same partial flag.
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.pattern.pattern_eq(other.pattern)
            and self.partial == other.partial
        )


class MultiOutputPattern(PatternExpr):
    # [CLASS DOC] Pattern with several output expressions (entries may be None placeholders).
    # The first output is the anchor and must be a _TargetExpr; the others are matched by
    # searching from nodes already matched. Attributes: outputs and op (copied from the first).
    outputs: list[PatternExpr | None]

    def __init__(self, outputs: Sequence[PatternExpr | None]) -> None:
        # [FUNCTION DOC] Stores outputs and the first output's op. Raises AssertionError if the
        # first output is not a _TargetExpr or any entry is neither None nor a PatternExpr.
        super().__init__()
        if not isinstance(outputs[0], _TargetExpr):
            raise AssertionError(f"expected _TargetExpr, got {type(outputs[0])}")
        if not all(x is None or isinstance(x, PatternExpr) for x in outputs):
            raise AssertionError(outputs)
        self.outputs = list(outputs)
        self.op = outputs[0].op

    @property
    def fns(self) -> Callable[..., Any] | str | Sequence[Any]:
        # [FUNCTION DOC] Property: the target functions of the first (anchor) output.
        # This cast is checked above in __init__()
        output = typing.cast(_TargetExpr, self.outputs[0])
        return output.fns

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string listing outputs.
        return f"{self.__class__.__name__}({self.outputs})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # [FUNCTION DOC] HELPER: Source text: the outputs printed through pp, one per line, in a list.
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f",\n{'  '}"
        str_out = f"{self.__class__.__name__}([{joiner_str.join(args)}"
        str_out = f"{str_out}\n])"
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Matches the first output against node, then each remaining non-None
        # output via _match_from_anchors, merging all matches. Returns the first failure.
        output = typing.cast(_TargetExpr, self.outputs[0])
        m = ctx.match(output, node)
        if not is_match(m):
            return m

        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            child_match = self._match_from_anchors(pattern, ctx)
            if not is_match(child_match):
                return child_match
            m.extend(child_match)

        return m

    def _match_from_anchors(
        self, pattern: PatternExpr, ctx: MatchContext
    ) -> MatchResult:
        # [FUNCTION DOC] HELPER: Tries to match a secondary output pattern by iterating candidate
        # anchor nodes (users of already matched nodes). Returns the first successful Match; after
        # each failed attempt restores the context's pattern_to_node to its prior state. If no
        # candidate matches returns the last FailedMatch ("no anchor found" if none tried).
        prior = dict(ctx.pattern_to_node)
        m: MatchResult = FailedMatch("no anchor found")
        for node in pattern.find_anchor_nodes(ctx, OrderedSet()):
            m = ctx.match(pattern, node)
            if is_match(m):
                return m
            # revert any partial matches
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> MatchResult:
        # [FUNCTION DOC] Matches starting at node using a context that knows this pattern's
        # full outputs list (so the output nodes can be recovered). Returns the Match or the
        # raised FailedMatch.
        try:
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            return e

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class, same number of outputs and pairwise equal outputs.
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and len(self.outputs) == len(other.outputs)
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.outputs, other.outputs)
            )
        )


class RepeatedExpr(PatternExpr):
    # [CLASS DOC] Pattern requiring the inner target pattern to match the given node and also
    # every sibling anchor node found from it (e.g. all getitem users of a split/unbind), each
    # matched independently. Attributes: inner_pattern and op (copied from it).
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: _TargetExpr) -> None:
        # [FUNCTION DOC] Stores the inner pattern and its op.
        super().__init__()
        self.inner_pattern = inner_pattern
        self.op = inner_pattern.op

    @property
    def fns(self) -> Sequence[FnsType]:
        # [FUNCTION DOC] Property: the target functions of the inner pattern.
        return self.inner_pattern.fns

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # [FUNCTION DOC] Matches the inner pattern on node, drops its binding from the context,
        # then for each anchor node discovered from the inner pattern (excluding node itself)
        # matches the inner pattern in a fresh context and merges the result. Returns the first
        # failure, else the combined Match.
        m = ctx.match(self.inner_pattern, node)
        if not is_match(m):
            return m
        ctx.pattern_to_node.pop(
            self.inner_pattern,
        )
        # Check all anchor nodes match the pattern
        for anchor_node in self.inner_pattern.find_anchor_nodes(
            ctx, OrderedSet([node])
        ):
            anchor_m = MatchContext([self], graph=node.graph).match(
                self.inner_pattern, anchor_node
            )
            if not is_match(anchor_m):
                return anchor_m
            m.extend(anchor_m)
        return m

    def pattern_eq(self, other: Any) -> bool:
        # [FUNCTION DOC] Structural equality: same class and equal inner patterns.
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.inner_pattern.pattern_eq(
            other.inner_pattern
        )


class PatternPrettyPrinter:
    # [CLASS DOC] Serializes pattern DAGs to executable python source. Each distinct call
    # pattern is memoized as a named assignment (unique names from an FX namespace) so shared
    # sub-patterns are written once. State: namespace, memoized_objs_names, memoized_objs_pp.
    # Mainly used and tested for fused-attention patterns.
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates an empty name allocator and empty memo tables.
        self.namespace = torch.fx.graph._Namespace()
        self.memoized_objs_names: dict[PatternExpr, str] = {}
        self.memoized_objs_pp: dict[PatternExpr, str] = {}

    @staticmethod
    @functools.cache
    def run(obj: PatternExpr, output_name: str = "output") -> str:
        # [FUNCTION DOC] Static, cached by arguments. Serializes obj with a fresh printer into
        # python source lines: one assignment per memoized sub-pattern in discovery order, then
        # an assignment of the full expression to output_name. Raises AssertionError if obj has
        # no pretty_print method.
        """
        Serializes obj to python code with obj written out to `output_name`
        """

        pp = PatternPrettyPrinter()
        if not hasattr(obj, "pretty_print"):
            raise AssertionError("obj has no pretty_print method")
        out_str = obj.pretty_print(pp=pp)

        output = [
            f"{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}"
            for key in pp.memoized_objs_names
        ]

        output.append(f"{output_name} = {out_str}")

        return "\n".join(output)

    def pretty_print(self, obj: Any) -> str:
        # [FUNCTION DOC] Returns source text for obj: call patterns become (memoized) variable
        # names, tensors become tensor-construction text (may raise NotImplementedError), objects
        # with their own pretty_print delegate to it, and anything else uses repr.
        if isinstance(obj, _TargetArgsExpr):
            if memoized_name := self.memoized_objs_names.get(obj):
                return memoized_name
            else:
                return self.memoize(obj)
        if isinstance(obj, torch.Tensor):
            return _tensor_constant_repr(obj)
        if hasattr(obj, "pretty_print"):
            return obj.pretty_print(self)

        return repr(obj)

    def memoize(self, obj: _TargetArgsExpr) -> str:
        # [FUNCTION DOC] Serializes obj, allocates a unique variable name derived from its
        # target (namespace prefixes stripped), records name and source text in the memo tables
        # (side effect) and returns the name.
        obj_str = obj.pretty_print(self)
        obj_name = obj.fns_repr()
        for prefix in ("aten.", "torch.", "prims."):
            obj_name = obj_name.replace(prefix, "")

        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        return tmp_name


class _PassDictsType(Protocol):
    # [CLASS DOC] HELPER: Typing protocol for pass registries: indexed by (op, target) pair, yields a list of PatternEntry.
    def __getitem__(
        self, k: tuple[str, torch.fx.node.Target]
    ) -> list[PatternEntry]: ...


@dataclasses.dataclass
class PatternEntry:
    # [CLASS DOC] Base dataclass of a registered rewrite: a pattern plus an extra_check
    # predicate run on a Match before applying. Subclasses implement apply to rewrite the graph.
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] Abstract: rewrites the graph for a match; the base raises NotImplementedError.
        raise NotImplementedError

    def register(
        self,
        pass_dicts: _PassDictsType | Sequence[_PassDictsType],
        target: torch.fx.node.Target | None = None,
        prepend: bool = False,
    ) -> None:
        # [FUNCTION DOC] Adds this entry to the pass registry (or each registry in a sequence)
        # under the key (pattern op, target), at the front if prepend else the end. With no
        # target, registers once per target function of the pattern. Raises AssertionError if
        # the pattern lacks fns or op. Mutates the registries.
        if target is None:
            if not hasattr(self.pattern, "fns"):
                raise AssertionError("pattern has no fns attribute")
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn, prepend=prepend)
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            if not hasattr(self.pattern, "op"):
                raise AssertionError("pattern has no op attribute")
            if prepend:
                pass_dicts[(self.pattern.op, target)].insert(0, self)
            else:
                pass_dicts[(self.pattern.op, target)].append(self)
        else:
            pass_dicts = typing.cast(Sequence[_PassDictsType], pass_dicts)
            for x in pass_dicts:
                self.register(x, target, prepend=prepend)


@dataclasses.dataclass
class LoweringPatternEntry(PatternEntry):
    # [CLASS DOC] Entry whose handler is inserted as a call_function node (to be lowered later)
    # that replaces the matched output node, with the match args and kwargs as inputs.
    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] Inserts before node a call to the handler (bound to the match) with
        # the match args and kwargs, copies metadata from node, redirects all uses of node to
        # it, then erases the matched nodes. Raises AssertionError if node is not the last
        # matched node.
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            _transfer_meta(replacement.meta, node)
            node.replace_all_uses_with(replacement)
        if match.nodes[-1] is not node:
            raise AssertionError("expected match.nodes[-1] to be node")
        match.erase_nodes()


@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    # [CLASS DOC] Entry whose handler is a function that edits the FX graph directly.
    """
    A pattern that runs a function on the FX graph
    """

    handler: Callable[..., Any]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] Calls handler(match, *args, **kwargs) with the graph insertion point set before node.
        with graph.inserting_before(node):
            self.handler(match, *match.args, **match.kwargs)


@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    # [CLASS DOC] Entry that replaces the matched subgraph with a pre-traced replacement graph
    # (stored on the Match). Fields: normalize_args (maps match args/kwargs to the replacement's
    # input list) and pattern_name (optional id for telemetry). Provides the shared static
    # splice routine replace_with_graph.
    """
    The replacement pattern for the graph
    """

    normalize_args: Callable[..., list[Any]]
    pattern_name: str | None = None  # Unique identifier for per-pattern telemetry

    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: torch.fx.Graph | torch.fx.GraphModule,
        args: Sequence[torch.fx.Node],
        pass_name: str | None = None,
    ) -> list[torch.fx.Node]:
        # [FUNCTION DOC] Static. Copies replacement_graph (a GraphModule; AssertionError
        # otherwise) into graph just before the earliest matched output node, with args as its
        # inputs: call_function nodes are re-created with metadata transferred, stream/mempool
        # context merged and eager_input_vals kept; get_attr nodes must refer to graph modules
        # (else NotImplementedError) and are registered on the owning module; other ops raise
        # NotImplementedError. Then rewires uses of each matched output to the new nodes (a
        # packed multi-result is unpacked through its getitem users), propagates recompute,
        # ac_graph_id and custom tags backward from the replaced node, erases matched nodes and
        # dead pure replacement nodes. Returns the added nodes that survive.
        """
        Inserts the replacement graph into the toplevel graph at the match
        """

        added_replacement_nodes: list[torch.fx.Node] = []
        custom_context = _common_custom_context(match.nodes)

        class Replacer(torch.fx.Interpreter):
            # [CLASS DOC] HELPER: FX Interpreter that, instead of executing, re-emits each
            # call_function node of the replacement graph into the target graph and registers
            # sub-graph get_attr nodes; call_method/call_module/get_attr handlers are disabled.
            call_method = None  # type: ignore[assignment]
            call_module = None  # type: ignore[assignment]
            get_attr = None  # type: ignore[assignment]

            def run_node(self, node: torch.fx.Node) -> Any:
                # [FUNCTION DOC] HELPER: Placeholders and outputs use default behavior. For a
                # call_function node creates the equivalent node in the outer graph with the
                # already-mapped args, records it, transfers metadata and returns it (AssertionError
                # if target not callable). For get_attr nodes fetches the sub graph module,
                # reuses its name in the owning module or registers it under a unique name, and
                # adds a get_attr node (NotImplementedError if it is not a GraphModule).
                # Other ops raise NotImplementedError.
                if node.op in ("placeholder", "output"):
                    return super().run_node(node)
                target = node.target
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                if node.op == "call_function":
                    if not callable(target):
                        raise AssertionError(f"expected callable target, got {target}")
                    result = graph.call_function(target, args, kwargs)
                    added_replacement_nodes.append(result)
                    _transfer_meta(
                        new_meta=result.meta,
                        old_node=node,
                        pass_name=pass_name or "",
                    )
                    _merge_custom_context(result.meta, custom_context)
                    # This function copy-pastes the replacement graph into
                    # the graph. If the replacement graph had any eager_input_vals,
                    # we propagate those over (val/tensor_meta are handled by
                    # _transfer_meta above).
                    if "eager_input_vals" in node.meta:
                        result.meta["eager_input_vals"] = node.meta["eager_input_vals"]
                    return result
                if node.op == "get_attr":
                    # If the replacement graph contains a HOP, the subgraphs of the HOP are "get_attr" nodes.
                    # We need to fetch the subgraph of the HOP then register the subgraph to the replaced graph's root.
                    from torch._higher_order_ops.utils import (
                        unique_graph_name_with_root,
                    )

                    sub_gm = super().get_attr(target, args, kwargs)
                    if not isinstance(sub_gm, torch.fx.GraphModule):
                        raise NotImplementedError(
                            f"NYI: replacement_graph.{target} is not a graph module. Got {sub_gm}."
                        )
                    if graph.owning_module is None:
                        raise AssertionError("graph.owning_module is None")
                    graph_name = None
                    for n, mod in graph.owning_module.named_modules():
                        if sub_gm is mod:
                            graph_name = n
                            break
                    if graph_name is None:
                        if not isinstance(target, str):
                            raise AssertionError(
                                f"expected str target, got {type(target)}"
                            )
                        _, graph_name = unique_graph_name_with_root(
                            # pyrefly: ignore [unbound-name]
                            graph.owning_module,
                            target,
                        )
                        # pyrefly: ignore [unbound-name]
                        graph.owning_module.register_module(graph_name, sub_gm)
                    # pyrefly: ignore [unbound-name]
                    getattr_node = graph.get_attr(graph_name)
                    added_replacement_nodes.append(getattr_node)
                    return getattr_node

                raise NotImplementedError(f"unhandled {node}")

        output_nodes = match.output_nodes()

        if len(output_nodes) == 1:
            last_node = output_nodes[0]
        else:
            if not output_nodes[0]:
                raise AssertionError("output_nodes[0] is falsy")
            nodes = list(output_nodes[0].graph.nodes)
            indices = [
                (nodes.index(n), n)
                for n in output_nodes
                if isinstance(n, torch.fx.Node)
            ]
            last_node = min(indices, key=operator.itemgetter(0))[1]

        def percolate_tags(
            node: torch.fx.Node,
            tag_name: str,
            tag_value: str,
            input_stops: OrderedSet[torch.fx.Node],
        ) -> None:
            # [FUNCTION DOC] HELPER: Walks backward through input nodes starting at node
            # (stopping at input_stops) and sets meta[tag_name] to tag_value on each visited node.
            queue = [node]
            visited = OrderedSet[torch.fx.Node]()

            while queue:
                arg = queue.pop()
                if (
                    arg not in visited
                    and arg not in input_stops
                    and hasattr(arg, "meta")
                ):
                    visited.add(arg)
                    arg.meta[tag_name] = tag_value
                    queue.extend(arg.all_input_nodes)

        with graph.inserting_before(last_node):
            if not isinstance(replacement_graph, torch.fx.GraphModule):
                raise AssertionError(
                    f"expected GraphModule, got {type(replacement_graph)}"
                )
            replacement = Replacer(replacement_graph).run(*args)
            if isinstance(replacement, torch.fx.Node):
                replacement = [replacement]

            def maybe_getitem(node: torch.fx.Node) -> Any:
                # [FUNCTION DOC] HELPER: If node is an operator getitem call returns its index
                # argument (AssertionError if it does not have 2 args); otherwise None.
                if node.op != "call_function":
                    return None
                if node.target != operator.getitem:
                    return None
                if len(node.args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(node.args)}")
                return node.args[1]

            def replace(
                old: torch.fx.Node | None,
                new: torch.fx.Node | Sequence[torch.fx.Node] | None,
            ) -> None:
                # [FUNCTION DOC] HELPER: Redirects uses of old to new. old None requires new
                # None. new None replaces uses with None. A single node gets metadata and
                # propagated tags and takes over all uses outside the replacement nodes. A list of
                # nodes replaces getitem users of old by indexing into the list (recursively), and
                # direct uses by the list. Erases old when it has no users left (always in the list
                # case). Raises AssertionError on a bad old/new combination.
                def filter_nodes_in_newly_added_nodes(node: torch.fx.Node) -> bool:
                    # [FUNCTION DOC] HELPER: Callback that is True for nodes not created by the replacement.
                    # Do not replace the use of a node if it is being used by
                    # nodes in the replaced graph
                    return node not in added_replacement_nodes

                if old is None:
                    if new is not None:
                        raise AssertionError("expected new to be None when old is None")
                    return
                if not isinstance(old, torch.fx.Node):
                    raise AssertionError(f"expected torch.fx.Node, got {type(old)}")
                if new is None:
                    old.replace_all_uses_with(
                        None,  # type: ignore[arg-type]
                        delete_user_cb=filter_nodes_in_newly_added_nodes,
                    )
                    if len(old.users) == 0:
                        graph.erase_node(old)
                    return
                if isinstance(new, torch.fx.Node):
                    _transfer_meta(new.meta, old, pass_name=pass_name or "")

                    # Preserve the recompute tags in the replacement graph. We
                    # look at the recompute tags of the original output node to
                    # propagate the tag from the output all the way to the input
                    args_set, _ = pytree.tree_flatten(args)
                    # args (named as args in the replace_with_graph).
                    # Note that this is best effort. Since patterns are from
                    # many to many, there is no easy way to correctly map the
                    # recomputable tags. It is possible in some scenarios that we
                    # incorrectly tag some nodes as recomputables.
                    for tag_name in ["recompute", "ac_graph_id", "custom"]:
                        if tag_name in old.meta:
                            percolate_tags(
                                new, tag_name, old.meta[tag_name], OrderedSet(args_set)
                            )

                    old.replace_all_uses_with(
                        new, delete_user_cb=filter_nodes_in_newly_added_nodes
                    )
                    if len(old.users) == 0:
                        graph.erase_node(old)
                    return

                # `new` is not a node: it's a list of nodes.
                #
                # This happens when we want to replace a node that has a single
                # packed return with multiple unpacked returns. We need to do
                # some graph surgery here.
                #
                # Example:
                #   def original_graph(x):
                #      a = op(x)
                #      b = a[0]
                #      c = a[1]
                #      ...
                #
                # Assume that we want to replace op(x) with the graph
                #   def new_op(x):
                #      w = x + 1
                #      z = x + 2
                #      return (w, z)
                #
                # We need to replace `op` with the contents of `new_op`,
                # and then rewrite a[0] to be w and a[1] to be z, as so:
                #   def new_graph(x):
                #     w = x + 1
                #     z = x + 2
                #     b = w
                #     c = z
                #     ...
                old_uses = list(old.users.keys())
                for user in old_uses:
                    idx = maybe_getitem(user)
                    if idx is None:
                        # Output is used directly
                        # pyrefly: ignore [bad-argument-type]
                        old.replace_all_uses_with(new)
                    else:
                        replace(user, new[idx])
                graph.erase_node(old)

            if len(output_nodes) == len(replacement):
                for old, new in zip(output_nodes, replacement):
                    replace(old, new)
            else:
                if len(output_nodes) != 1:
                    raise AssertionError(
                        f"expected 1 output node, got {len(output_nodes)}"
                    )
                replace(output_nodes[0], replacement)

        match.erase_nodes()

        # Remove dead replacement nodes so they don't inflate user counts
        # in later lowering heuristics (e.g. should_realize_on_reuse).
        for node in reversed(added_replacement_nodes):
            if (
                not node.users
                and not node.is_impure()
                and not isinstance(node.target, torch._ops.HigherOrderOperator)
            ):
                graph.erase_node(node)

        return [node for node in added_replacement_nodes if not node._erased]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] Splices match.replacement_graph (AssertionError if unset) into graph,
        # using normalize_args on the match args/kwargs as inputs and the pattern name for
        # provenance. Mutates the graph; node is unused.
        if match.replacement_graph is None:
            raise AssertionError("match.replacement_graph is None")
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,
            self.normalize_args(*match.args, **match.kwargs),
            pass_name=self.pattern_name or "replace_with_graph",
        )


def _return_true(match: Match) -> bool:
    # [FUNCTION DOC] HELPER: Default extra_check that accepts every match.
    return True


def log_trace_failure(search_fn: Callable[..., Any], e: RuntimeError) -> None:
    # [FUNCTION DOC] Logs at info level that the replacement pattern named by search_fn failed to apply (shape mismatch while tracing) with the error.
    log.info(
        "Replacement pattern %s failed to apply due to shape mismatch: %s",
        search_fn.__name__,
        e,
    )


@dataclasses.dataclass(frozen=True)
class _TraceArgInfo:
    # [CLASS DOC] HELPER: Frozen record describing how pattern function arguments were flattened for tracing:
    # argnames (original names), flat_argnames (all unique leaf names), arg_specs (pytree spec
    # per argument) and arg_flat_names (leaf names per argument).
    argnames: list[str]
    flat_argnames: list[str]
    arg_specs: dict[str, Any]
    arg_flat_names: dict[str, list[str]]


def _unique_argname(name: str, used_names: OrderedSet[str]) -> str:
    # [FUNCTION DOC] HELPER: Returns name if unused, else name with the smallest numeric suffix
    # that is unused; adds the result to used_names (mutation).
    if name not in used_names:
        used_names.add(name)
        return name

    i = 0
    while f"{name}_{i}" in used_names:
        i += 1
    name = f"{name}_{i}"
    used_names.add(name)
    return name


def _trace_args_for_initial_trace(
    argnames: Sequence[str],
    example_inputs: Sequence[Any],
    scalar_workaround: dict[str, float | int],
) -> list[Any]:
    # [FUNCTION DOC] HELPER: Builds the argument list for the initial trace in argnames order:
    # names in scalar_workaround take that scalar, others consume example_inputs in order.
    trace_args = []
    input_idx = 0
    for argname in argnames:
        if argname in scalar_workaround:
            trace_args.append(scalar_workaround[argname])
        else:
            trace_args.append(example_inputs[input_idx])
            input_idx += 1
    return trace_args


def _trace_arg_info(argnames: Sequence[str], args: Sequence[Any]) -> _TraceArgInfo:
    # [FUNCTION DOC] HELPER: Flattens each argument with pytree and assigns unique leaf names
    # (the argument name for a single leaf, name plus index for several, none for zero leaves).
    # Returns a _TraceArgInfo with names, specs and per-argument leaf names.
    flat_argnames: list[str] = []
    arg_specs: dict[str, Any] = {}
    arg_flat_names: dict[str, list[str]] = {}
    used_names: OrderedSet[str] = OrderedSet()

    for argname, arg in zip(argnames, args):
        flat_args, spec = pytree.tree_flatten(arg)
        arg_specs[argname] = spec
        if len(flat_args) == 0:
            names: list[str] = []
        elif len(flat_args) == 1:
            names = [_unique_argname(argname, used_names)]
        else:
            names = [
                _unique_argname(f"{argname}_{i}", used_names)
                for i in range(len(flat_args))
            ]
        flat_argnames.extend(names)
        arg_flat_names[argname] = names

    return _TraceArgInfo(
        argnames=list(argnames),
        flat_argnames=flat_argnames,
        arg_specs=arg_specs,
        arg_flat_names=arg_flat_names,
    )


def _unflatten_match_kwargs(
    kwargs: Mapping[str, Any],
    arg_info: _TraceArgInfo,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Rebuilds structured arguments from flat match kwargs: for each
    # original argument gathers its leaf kwargs and unflattens them with the saved spec. If no
    # leaves were found and a default exists, uses the default; if only some leaves are
    # missing raises RuntimeError. Remaining unconsumed kwargs are passed through. Returns a new mapping.
    kwargs = dict(kwargs)
    result: dict[str, Any] = {}

    for argname in arg_info.argnames:
        flat_names = arg_info.arg_flat_names[argname]
        found_names = [flat_name for flat_name in flat_names if flat_name in kwargs]
        if len(found_names) != len(flat_names):
            if defaults is not None and argname in defaults and not found_names:
                result[argname] = defaults[argname]
                continue
            missing_names = [
                flat_name for flat_name in flat_names if flat_name not in kwargs
            ]
            raise RuntimeError(
                f"Not all inputs to pattern found in match.kwargs. Perhaps one "
                f"of the inputs is unused? argname={argname}, "
                f"flat_argnames={flat_names}, missing_argnames={missing_names}, "
                f"match.kwargs={kwargs}"
            )

        flat_values = []
        for flat_name in flat_names:
            flat_values.append(kwargs.pop(flat_name))
        result[argname] = pytree.tree_unflatten(
            flat_values, arg_info.arg_specs[argname]
        )

    result.update(kwargs)
    return result


def _get_match_node_value(node: torch.fx.Node) -> Any:
    # [FUNCTION DOC] HELPER: Returns the node's meta val if present, else its example_value (KeyError if neither).
    if "val" in node.meta:
        return node.meta["val"]
    return node.meta["example_value"]


def check_and_add_duplicate_pattern(
    pattern: PatternExpr,
    graph: torch.fx.Graph | None,
    seen_patterns: dict[str, list[str | None]],
    skip_duplicates: bool = False,
) -> bool:
    # [FUNCTION DOC] Detects duplicate registrations. The pattern's serialized source is the
    # key in seen_patterns; its value lists string forms of match graphs seen for that key
    # (None entry when no graph). A new key is recorded (relies on seen_patterns being a
    # defaultdict) and returns False. A repeat with the same graph (or no graph) returns True
    # if skip_duplicates, else raises via torch._check; a repeat with a different graph is
    # recorded and returns False.
    """
    Check if a pattern is a duplicate. Because we ignore certain types in searching, but not
    in matching, use the graph to distinguish equivalent search patterns.

    Returns True if a duplicate is found and `skip_duplicates=True` is passed in. Errors if
    `skip_duplicates` is False and a duplicate is found.
    """

    pattern_repr = PatternPrettyPrinter.run(pattern)
    equiv_pattern_reprs = seen_patterns.get(pattern_repr)
    if not equiv_pattern_reprs:
        seen_patterns[pattern_repr].append(str(graph) if graph else None)
        return False

    if graph is None:
        if skip_duplicates:
            return True
        torch._check(
            False,
            lambda: f"Duplicate pattern: {pattern_repr} with no graph",
        )

    new_graph_str = str(graph)
    for graph_str in equiv_pattern_reprs:
        if new_graph_str != graph_str:
            continue
        if skip_duplicates:
            return True
        torch._check(
            False,
            lambda: f"Duplicate pattern: {pattern_repr} with duplicated match graph {graph_str} ",
        )
    equiv_pattern_reprs.append(new_graph_str)
    return False


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: list[Any] | tuple[Any, ...],
    trace_fn: TraceFn,
    pass_dicts: _PassDictsType | Sequence[_PassDictsType],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
    search_fn_pattern: PatternExpr | None = None,
    skip_duplicates: bool = False,
    pattern_name: str | None = None,
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
) -> bool:
    # [FUNCTION DOC] Registers a search/replace rule built by tracing search_fn and replace_fn.
    # Bound methods are wrapped; raises TypeError if example_inputs is not a list or tuple.
    # Builds the pattern from search_fn (or uses search_fn_pattern), skips registration
    # (returns False) for a joint_fwd_bwd trace under inference mode or for a skipped duplicate
    # (duplicates otherwise raise), then registers a ReplacementPatternEntry in pass_dicts.
    # Its check function re-traces search_fn with the real (fake-mode) shapes of the matched
    # nodes, re-matches, runs extra_check, and on success traces replace_fn into the match's
    # replacement graph and restores structured kwargs. normalize_args flattens the structured
    # kwargs (and tangents_N extras) into the replacement input list. Side effects: mutates pass
    # registries; tracing runs with rng functionalization disabled.
    # KNOWN ISSUE: annotated to return bool, but on success it returns the registered PatternExpr.
    """
    Create a replacement rule based on example functions that get traced
    to create patterns.  This supports both training and inference when
    run on a joint forward+backward graph.

    Args:
        search_fn: traced to give original pattern
        replace_fn: traced to give replacement graph
        example_inputs: example inputs for initial trace
        trace_fn: fwd_only or joint_fwd_bwd
        pass_dict: dict of passes to register to
        extra_check: additional check to run on match(using real shapes)
    """
    argnames_static = [*inspect.signature(search_fn).parameters.keys()]

    if inspect.ismethod(search_fn):
        search_fn = _wrap_bound_method(search_fn, argnames_static)

    if inspect.ismethod(replace_fn):
        replace_argnames = [*inspect.signature(replace_fn).parameters.keys()]
        replace_fn = _wrap_bound_method(replace_fn, replace_argnames)

    if not isinstance(example_inputs, (list, tuple)):
        raise TypeError(
            f"example_inputs must be a list or tuple, got {type(example_inputs)}"
        )
    scalar_workaround = scalar_workaround or {}
    initial_trace_args = _trace_args_for_initial_trace(
        argnames_static, example_inputs, scalar_workaround
    )
    initial_arg_info = _trace_arg_info(argnames_static, initial_trace_args)
    requires_grad = pytree.tree_map(
        lambda x: isinstance(x, torch.Tensor) and x.requires_grad,
        initial_trace_args,
    )

    def check_fn(match: Match) -> bool:
        # [FUNCTION DOC] HELPER: extra_check for the entry. Rebuilds example args from the
        # matched nodes' values, refreshing tensors as fresh strided fake tensors (collecting
        # distinct symbolic sizes; integer tensors that require grad make the check fail),
        # traces search_fn (or uses the pre-traced pattern; trace RuntimeErrors are logged and
        # fail the check), re-matches the output node, then runs extra_check. On success sets
        # match.replacement_graph (traced replace_fn) and match.kwargs and returns True; else False.
        # Raises AssertionError if no fake mode is found among the args or the output node is None.
        """
        Often shapes get burned into the pattern, so our initial match ran with
        `ignore_types=(int, ...)`.

        Recheck the match with the correct shapes.
        """
        structured_match_kwargs = _unflatten_match_kwargs(
            match.kwargs, initial_arg_info
        )

        args = list(
            torch.fx.map_arg(
                [structured_match_kwargs[name] for name in argnames_static],
                _get_match_node_value,
            )
        )

        sym_args: list[torch.SymInt] = []
        fake_mode = torch._dynamo.utils.detect_fake_mode(args)
        if fake_mode is None:
            raise AssertionError("fake_mode is None")
        with fake_mode:
            invalid_args = False
            requires_grad_values = iter(pytree.tree_leaves(requires_grad))

            def refresh_arg(arg: Any) -> Any:
                # [FUNCTION DOC] HELPER: Non-tensors pass through. Tensors become new empty
                # strided tensors with the same size, stride, dtype, device and requires_grad
                # (flag consumed in order from requires_grad leaves); symbolic ints seen are added
                # to sym_args. Integer tensors flagged as requiring grad set invalid_args and are kept.
                nonlocal invalid_args
                grad = next(requires_grad_values)
                if not isinstance(arg, torch.Tensor):
                    return arg

                if grad and is_integer_dtype(arg.dtype):
                    invalid_args = True
                    return arg

                refreshed_arg = torch.empty_strided(
                    arg.size(),
                    arg.stride(),
                    dtype=arg.dtype,
                    device=arg.device,
                    requires_grad=grad,
                )
                for v in itertools.chain(refreshed_arg.shape, refreshed_arg.stride()):
                    if isinstance(v, torch.SymInt) and all(
                        statically_known_true(v != a) for a in sym_args
                    ):
                        sym_args.append(v)
                return refreshed_arg

            args = pytree.tree_map(refresh_arg, args)
            if invalid_args:
                return False

            # If we were given a pre-traced pattern then use that instead of
            # retracing. Note that this means the pattern has to be independent
            # of its args.
            specific_pattern = search_fn_pattern
            specific_arg_info = _trace_arg_info(argnames_static, args)
            specific_argnames = specific_arg_info.flat_argnames

            if not specific_pattern:
                if sym_args:
                    # AOT Autograd and make fx will dedupe symbolic shape size
                    # accesses of sym ints that appear as inputs
                    # We don't want the sym_size uses to interfere with pattern matching
                    # so we provide them as inputs.
                    # Later, when we actually do the replacement, the symbolic shape
                    # sizes will get re-traced and added to the graph.

                    def search_fn_new(*args_new: Any, **_: Any) -> Any:
                        # [FUNCTION DOC] HELPER: Calls search_fn with only the trailing real args, dropping the leading symbolic size inputs.
                        return search_fn(*args_new[len(args_new) - len(args) :])

                    try:
                        specific_graph = trace_fn(
                            search_fn_new,
                            sym_args + args,
                            get_decomp_fn=get_decomp_fn,
                        )
                    except RuntimeError as e:
                        log_trace_failure(search_fn, e)
                        return False

                    # correct argnames in the graph
                    sym_arg_names = []
                    placeholders = [
                        n for n in specific_graph.graph.nodes if n.op == "placeholder"
                    ]
                    for i, placeholder in enumerate(
                        placeholders[: len(sym_args) + len(specific_argnames)]
                    ):
                        if i < len(sym_args):
                            sym_arg_names.append(placeholder.target)
                            continue

                        with specific_graph.graph.inserting_after(placeholder):
                            new_node = specific_graph.graph.placeholder(
                                specific_argnames[i - len(sym_args)]
                            )
                            new_node.target = new_node.name
                            placeholder.replace_all_uses_with(new_node)
                            specific_graph.graph.erase_node(placeholder)

                    specific_argnames = sym_arg_names + specific_argnames
                else:
                    try:
                        specific_graph = trace_fn(
                            search_fn, args, get_decomp_fn=get_decomp_fn
                        )
                    except RuntimeError as e:
                        log_trace_failure(search_fn, e)
                        return False

                specific_pattern = fx_to_pattern(
                    specific_graph,
                    argnames=specific_argnames,
                    exclusive_arg_names=exclusive_arg_names,
                    scalar_workaround=scalar_workaround,
                )

            node = match.output_nodes()[0]
            if node is None:
                raise AssertionError("output node is None")
            specific_pattern_match = specific_pattern.match(node)

            if _should_debug_node(node.name):
                log.warning(
                    "Specific pattern match: %s%s %s %s",
                    node,
                    node.args,
                    specific_pattern_match,
                    specific_pattern,
                )

            if is_match(specific_pattern_match):
                specific_pattern_match.kwargs = _unflatten_match_kwargs(
                    specific_pattern_match.kwargs,
                    specific_arg_info,
                    defaults=structured_match_kwargs,
                )

            if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
                # trace the pattern using the shapes from the user program
                match.replacement_graph = trace_fn(
                    replace_fn, args, get_decomp_fn=get_decomp_fn
                )
                match.kwargs = structured_match_kwargs
                if len(match.nodes) == 1:
                    for n in match.replacement_graph.graph.nodes:
                        _transfer_meta(
                            new_meta=n.meta,
                            old_node=match.nodes[0],
                            pass_name="replacement",
                        )
                return True
            return False

    def normalize_args(**kwargs: Any) -> list[Any]:
        # [FUNCTION DOC] HELPER: Converts match kwargs to the positional input list for the
        # replacement graph: unflattens if needed, flattens each original argument into leaves
        # in argument order, then appends tangents_1, tangents_2, ... values. Raises
        # AssertionError if kwargs are left over.
        # Flat kwargs come from the initial match; structured kwargs are restored
        # after check_fn accepts the match.
        if not all(name in kwargs for name in argnames_static):
            kwargs = _unflatten_match_kwargs(kwargs, initial_arg_info)
        structured_args = [kwargs.pop(name) for name in argnames_static]
        args = []
        for arg in structured_args:
            args.extend(pytree.tree_leaves(arg))
        for i in range(1, len(kwargs) + 1):
            if f"tangents_{i}" not in kwargs:
                break
            args.append(kwargs.pop(f"tangents_{i}"))
        if kwargs:
            raise AssertionError(f"leftover kwargs: {kwargs!r}")
        return args

    if trace_fn is joint_fwd_bwd:
        # If inference mode is enabled during compilation, assume that we don't
        # want to match on any training graph patterns
        if torch.is_inference_mode_enabled():
            return False

    # TODO: Revisit the functionalize_rng_ops for lowmem dropout
    with functorch_config.patch(functionalize_rng_ops=False):
        if search_fn_pattern is None:
            pattern, gm = gen_pattern_and_search_gm(
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                exclusive_arg_names,
                get_decomp_fn=get_decomp_fn,
            )
        else:
            pattern = search_fn_pattern
            gm = None

        for pattern_matcher_pass in (
            pass_dicts if isinstance(pass_dicts, Sequence) else [pass_dicts]
        ):
            if isinstance(pattern_matcher_pass, PatternMatcherPass):
                if check_and_add_duplicate_pattern(
                    pattern,
                    gm.graph if gm else None,
                    pattern_matcher_pass.seen_patterns,
                    skip_duplicates=skip_duplicates,
                ):
                    return False

        pattern = ReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
            pattern_name=pattern_name,
        )
        pattern.register(pass_dicts)
        return pattern.pattern  # type: ignore[return-value]


_serialized_patterns: OrderedSet[str] = OrderedSet()


def _serialize_pattern(
    unique_name: str,
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None,
) -> PatternExpr:
    # [FUNCTION DOC] HELPER: Generates the pattern for search_fn and appends its serialized
    # python source (assigned to unique_name) to a file named after the function in the
    # serialized patterns directory. First write per function in this process truncates the
    # file and writes a header with imports; later ones append. Raises RuntimeError if the
    # directory is missing. Returns the generated PatternExpr. Side effects: file I/O and
    # updates of the module-level _serialized_patterns set.
    def get_file_template() -> str:
        # [FUNCTION DOC] HELPER: Builds the generated-file header: auto-generated notice, imports, aten/prims aliases, and an import of every PatternExpr/_TargetExpr subclass defined in this module.
        auto_generated_msg = textwrap.dedent(
            """\
            # This is an auto-generated file. Please do not modify it by hand.
            # To re-generate, run:
            # cd ~/pytorch && python torchgen/fuse/gen_patterns.py
            """
        )

        file_template = textwrap.dedent(
            """\
            # noqa: F401, E501
            {msg}
            import torch
            import torch._inductor
            import operator

            aten = torch.ops.aten
            prims = torch.ops.prims

            """
        ).format(msg=auto_generated_msg)

        pattern_matcher_imports = []
        for name in dir(torch._inductor.pattern_matcher):
            attr = getattr(torch._inductor.pattern_matcher, name)
            try:
                if isinstance(attr, type) and issubclass(
                    attr, (PatternExpr, _TargetExpr)
                ):
                    # pyrefly: ignore [bad-argument-type]
                    pattern_matcher_imports.append(name)
            except TypeError:
                pass

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    if not SERIALIZED_PATTERN_PATH.is_dir():
        raise RuntimeError(
            f"Could not find serialized patterns directory at {SERIALIZED_PATTERN_PATH}"
        )

    pattern_name = search_fn.__name__

    from torch._functorch import config as functorch_config

    with functorch_config.patch(functionalize_rng_ops=False):
        pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround)

    serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=unique_name)
    if pattern_name not in _serialized_patterns:
        write_mode = "w"
        _serialized_patterns.add(pattern_name)
    else:
        write_mode = "a"

    file_template = get_file_template()

    with open(SERIALIZED_PATTERN_PATH / f"{pattern_name}.py", write_mode) as f:
        if write_mode == "w":
            f.write(file_template)
        else:
            f.write("\n\n")
        f.write(serialized_pattern)
        f.write("\n")

    return pattern


SERIALIZED_PATTERN_PATH = Path(__file__).parent / "fx_passes" / "serialized_patterns"


# This is the set of serialized patterns that we've registered.  Used by
# test_serialized_patterns_up_to_date() to ensure the patterns are up
# to date.
@dataclasses.dataclass
class _PrecompiledPattern:
    # [CLASS DOC] HELPER: Record of a registered precompiled pattern (search function, example
    # inputs, tracer, scalar workaround and the loaded pattern), kept so tests can verify the
    # serialized patterns are up to date.
    search_fn: SearchFn
    example_inputs: Sequence[Any]
    trace_fn: TraceFn
    scalar_workaround: dict[str, float | int] | None
    search_fn_pattern: PatternExpr


_known_precompiled_patterns: list[_PrecompiledPattern] = []


def gen_register_replacement(
    unique_name: str,
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: _PassDictsType | Sequence[_PassDictsType],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
    skip_duplicates: bool = False,
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
) -> None:
    # [FUNCTION DOC] Like register_replacement but uses a precompiled pattern. If env var
    # PYTORCH_GEN_PATTERNS is set, generates and writes the serialized pattern; otherwise imports
    # the serialized module named after search_fn and fetches the attribute unique_name (logs a
    # warning if missing, then the attribute lookup raises AttributeError). Clears stored
    # constants on fake tensor example inputs to avoid leaks, records the pattern in the
    # module-level known list, and registers the replacement with the pre-traced pattern.
    # Returns None.
    # Make sure the example_inputs is materialized.
    example_inputs = tuple(example_inputs)

    if "PYTORCH_GEN_PATTERNS" in os.environ:
        pat = _serialize_pattern(
            unique_name, search_fn, example_inputs, trace_fn, scalar_workaround
        )
    else:
        pattern_name = search_fn.__name__
        m = importlib.import_module(
            f"torch._inductor.fx_passes.serialized_patterns.{pattern_name}"
        )
        if not m or not hasattr(m, unique_name):
            log.warning(
                "Precompiled pattern %r not found. Run torchgen/fuse/gen_patterns.py.",
                unique_name,
            )
        pat = getattr(m, unique_name)

    for arg in pytree.tree_iter(example_inputs):
        if isinstance(arg, FakeTensor) and maybe_get_fake_constant(arg) is not None:  # noqa: ISINSTANCE_FAKE_TENSOR
            # This can be a problem - small fake tensors (e.g. `tensor(2)`) will
            # hold onto their original constant value - and by stashing it here
            # will cause a memory leak if the constant value is on GPU.
            # Since this is just an optimization we can clear it out.
            # for c++ need to add a setter
            arg.constant = None

    _known_precompiled_patterns.append(
        _PrecompiledPattern(search_fn, example_inputs, trace_fn, scalar_workaround, pat)
    )
    register_replacement(
        search_fn,
        replace_fn,
        example_inputs,
        trace_fn,
        pass_dicts,
        extra_check,
        scalar_workaround,
        exclusive_arg_names,
        search_fn_pattern=pat,
        skip_duplicates=skip_duplicates,
        pattern_name=unique_name,
        get_decomp_fn=get_decomp_fn,
    )


@functorch_config.patch(functionalize_rng_ops=False)  # type: ignore[misc]
def gen_pattern_and_search_gm(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
) -> tuple[PatternExpr, torch.fx.GraphModule]:
    # [FUNCTION DOC] Traces search_fn with example_inputs (scalar workaround names substituted)
    # and converts the traced graph into a pattern, ignoring int, float, list, device and dtype
    # constants. Returns the pattern and the traced GraphModule. Runs with rng functionalization disabled.
    argnames = [*inspect.signature(search_fn).parameters.keys()]

    scalar_workaround = scalar_workaround or {}
    flat_inputs = _trace_args_for_initial_trace(
        argnames, example_inputs, scalar_workaround
    )
    arg_info = _trace_arg_info(argnames, flat_inputs)

    search_gm = trace_fn(search_fn, flat_inputs, get_decomp_fn=get_decomp_fn)
    return (
        fx_to_pattern(
            search_gm,
            ignore_types=(int, float, list, torch.device, torch.dtype),
            argnames=arg_info.flat_argnames,
            scalar_workaround=scalar_workaround,
            exclusive_arg_names=exclusive_arg_names,
        ),
        search_gm,
    )


def gen_pattern(
    search_fn: SearchFn,
    example_inputs: Sequence[Any],
    trace_fn: TraceFn,
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    # [FUNCTION DOC] Returns only the pattern produced by gen_pattern_and_search_gm.
    return gen_pattern_and_search_gm(
        search_fn, example_inputs, trace_fn, scalar_workaround, exclusive_arg_names
    )[0]


def register_lowering_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
    output_metadata_ignores_input_storage: bool = True,
    output_metadata_is_input: int | str | None = None,
    output_metadata_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    # [FUNCTION DOC] Returns a decorator that registers the decorated handler as a
    # LoweringPatternEntry for pattern in pass_dict (front if prepend). The handler is tagged
    # with attributes marking it as an inductor lowering function and carrying the output
    # metadata options (ignore input storage, is-input index/name, metadata function).
    """
    Register an aten to inductor IR replacement pattern.  The decorated
    function is saved and then called at lowering time allowing direct
    pattern to inductor IR conversion.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        # [FUNCTION DOC] HELPER: Registers handler and tags it with lowering attributes; raises AssertionError if handler is not callable. Returns handler.
        if not callable(handler):
            raise AssertionError(f"expected callable handler, got {handler}")
        LoweringPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        handler._inductor_lowering_function = True  # type: ignore[attr-defined]
        handler._inductor_lowering_output_metadata_ignores_input_storage = (  # type: ignore[attr-defined]
            output_metadata_ignores_input_storage
        )
        handler._inductor_lowering_output_metadata_is_input = (  # type: ignore[attr-defined]
            output_metadata_is_input
        )
        handler._inductor_lowering_output_metadata_fn = output_metadata_fn  # type: ignore[attr-defined]
        return handler

    return decorator


def register_graph_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    # [FUNCTION DOC] Returns a decorator that registers the decorated handler as a
    # GraphPatternEntry for pattern in pass_dict (front if prepend), with extra_check as filter.
    """
    Register a pattern that runs a function on the FX graph, allowing
    custom transformation code.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        # [FUNCTION DOC] HELPER: Registers handler; raises AssertionError if not callable. Returns handler unchanged.
        if not callable(handler):
            raise AssertionError(f"expected callable handler, got {handler}")
        GraphPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        return handler

    return decorator


def is_start_of_fx_graph(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] True if node is the first node of graph.
    # first node in the graph
    return node is next(iter(graph.nodes))


# match: copy_, relu_, _set_grad_enabled, manual_seed, _enter_autocast, etc
# doesn't match: __rshift__, etc
_mutation_op_re = re.compile(r"(?<!_)(_$|_[.]|(\b|_)(set|enter|exit|seed)(\b|_))(?!_)")


def fixme_incorrect_inductor_schema_op(op: torch._ops.OpOverload) -> bool:
    # [FUNCTION DOC] HELPER: True for the known inductor ops whose schema wrongly says non-mutating (currently resize_storage_bytes_ default); False for all other ops.
    if op.namespace != "inductor":
        return False

    # TODO - fix schema
    # Don't add any more!
    return op in (torch.ops.inductor.resize_storage_bytes_.default,)


def is_mutation_op(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] Heuristically decides whether node mutates its inputs. Operator overloads
    # use their schema's mutable flag (except known wrong-schema inductor ops, which fall
    # through to the name check); AutoFunctionalized targets are not mutations. Otherwise
    # call_function names and call_method names are tested against a regex for trailing
    # underscore or set/enter/exit/seed words; finally a non-None out keyword means mutation.
    # Raises AssertionError for a non-callable function target or non-string method target.
    if isinstance(
        node.target, torch._ops.OpOverload
    ) and not fixme_incorrect_inductor_schema_op(node.target):
        return node.target._schema.is_mutable
    elif isinstance(
        node.target, torch._higher_order_ops.auto_functionalize.AutoFunctionalized
    ):
        return False
    if node.op == "call_function":
        if not callable(node.target):
            raise AssertionError(f"expected callable node.target, got {node.target}")
        if _mutation_op_re.search(node.target.__name__):
            return True
    elif node.op == "call_method":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str node.target, got {type(node.target)}")
        if _mutation_op_re.search(node.target):
            return True
    return node.kwargs.get("out") is not None


def same_mutation_regions(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    # [FUNCTION DOC] True if both nodes carry the same mutation_region_id metadata. Raises
    # AssertionError if either node lacks that metadata.
    if "mutation_region_id" not in a.meta:
        raise AssertionError("mutation_region_id missing from a.meta")
    if "mutation_region_id" not in b.meta:
        raise AssertionError("mutation_region_id missing from b.meta")
    return a.meta["mutation_region_id"] == b.meta["mutation_region_id"]


def get_mutation_region_id(graph: torch.fx.Graph, node: torch.fx.Node) -> int:
    # [FUNCTION DOC] Returns node's mutation region id, computing lazily. Walks backward to the
    # nearest node that already has an id (or the graph start, id 0), then walks forward to
    # node, incrementing the id at each mutation op and storing the id in each visited node's
    # meta (side effect). Returns the id for node.
    n = node
    while "mutation_region_id" not in n.meta and not is_start_of_fx_graph(graph, n):
        n = n.prev
    mutation_region_id = n.meta.get("mutation_region_id", 0)
    while n is not node:
        n = n.next
        if is_mutation_op(n):
            mutation_region_id += 1
        n.meta["mutation_region_id"] = mutation_region_id
    return mutation_region_id


def should_compute_mutation_region_ids(graph: torch.fx.Graph) -> bool:
    # [FUNCTION DOC] True if the first node of graph has no mutation_region_id yet (ids not computed).
    return "mutation_region_id" not in next(iter(graph.nodes)).meta


def compute_mutation_region_ids(graph: torch.fx.Graph) -> None:
    # [FUNCTION DOC] Assigns every node a mutation_region_id in meta: starts at 0 and increases
    # by one at each mutation op (the mutation node itself gets the new id). Mutates metadata.
    mutation_region_id = 0
    for nd in graph.nodes:
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id


@dataclasses.dataclass(frozen=True)
class _MutationRegionRefresh:
    # [CLASS DOC] HELPER: Frozen snapshot taken before a rewrite: the node before the rewritten
    # span (anchor), the node after it (stop), the region id at the anchor and the number of
    # mutation ops inside the span; used to refresh region ids afterwards.
    anchor: torch.fx.Node
    stop: torch.fx.Node
    mutation_region_id: int
    mutation_count: int


@dataclasses.dataclass
class _GraphMutationTracker:
    # [CLASS DOC] HELPER: Records nodes created and whether a mutation node was erased while a graph handler ran.
    created_nodes: list[torch.fx.Node] = dataclasses.field(default_factory=list)
    erased_mutation_node: bool = False

    def changed_mutation_regions(self) -> bool:
        # [FUNCTION DOC] HELPER: True if a mutation node was erased or any created node is a mutation op.
        return self.erased_mutation_node or _contains_mutation_op(self.created_nodes)


@contextlib.contextmanager
def _track_graph_mutation_ops(
    graph: torch.fx.Graph,
    custom_context: dict[str, Any] | None = None,
) -> Generator[_GraphMutationTracker, None, None]:
    # [FUNCTION DOC] HELPER: Context manager yielding a tracker. While active, temporarily
    # overrides the graph instance's create_node (records new nodes and merges custom_context
    # into their meta) and erase_node (notes erased mutation ops). On exit restores the
    # original per-instance attributes (or removes the overrides), even on error.
    custom_context = custom_context or {}
    tracker = _GraphMutationTracker()
    create_node = graph.create_node
    erase_node = graph.erase_node
    sentinel = object()
    graph_dict = graph.__dict__
    create_node_attr = graph_dict.get("create_node", sentinel)
    erase_node_attr = graph_dict.get("erase_node", sentinel)

    def tracked_create_node(*args: Any, **kwargs: Any) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Creates a node via the original method, merges the custom context, records and returns it.
        created_node = create_node(*args, **kwargs)
        _merge_custom_context(created_node.meta, custom_context)
        tracker.created_nodes.append(created_node)
        return created_node

    def tracked_erase_node(to_erase: torch.fx.Node) -> Any:
        # [FUNCTION DOC] HELPER: Flags the tracker if a live mutation node is being erased, then erases it via the original method.
        if not to_erase._erased and is_mutation_op(to_erase):
            tracker.erased_mutation_node = True
        return erase_node(to_erase)

    graph_dict["create_node"] = tracked_create_node
    graph_dict["erase_node"] = tracked_erase_node
    try:
        yield tracker
    finally:
        if create_node_attr is sentinel:
            del graph_dict["create_node"]
        else:
            graph_dict["create_node"] = create_node_attr
        if erase_node_attr is sentinel:
            del graph_dict["erase_node"]
        else:
            graph_dict["erase_node"] = erase_node_attr


def _count_mutation_ops_until(start: torch.fx.Node, stop: torch.fx.Node) -> int:
    # [FUNCTION DOC] HELPER: Counts mutation ops walking forward from start up to (excluding)
    # stop. Raises AssertionError if the graph root is reached first.
    mutation_count = 0
    nd = start
    while nd is not stop:
        if nd.op == "root":
            raise AssertionError(f"unexpected root node reached: {nd}")
        if is_mutation_op(nd):
            mutation_count += 1
        nd = nd.next
    return mutation_count


def _contains_mutation_op(nodes: Iterable[torch.fx.Node]) -> bool:
    # [FUNCTION DOC] HELPER: True if any node is a mutation op.
    return any(is_mutation_op(nd) for nd in nodes)


def _replacement_changes_mutation_regions(
    entry: ReplacementPatternEntry, match: Match
) -> bool:
    # [FUNCTION DOC] HELPER: True if the matched nodes or the replacement graph's nodes contain a
    # mutation op (so region ids need refreshing). Raises AssertionError if the match has no
    # replacement graph. The entry argument is unused.
    if match.replacement_graph is None:
        raise AssertionError("match.replacement_graph must not be None")
    return _contains_mutation_op(match.nodes) or _contains_mutation_op(
        match.replacement_graph.graph.nodes
    )


def _mutation_region_refresh(
    graph: torch.fx.Graph,
    nodes: Sequence[torch.fx.Node],
) -> _MutationRegionRefresh:
    # [FUNCTION DOC] HELPER: Captures a refresh record for rewriting the given nodes: anchor is
    # the node before the earliest node and stop the node after the latest (with no nodes, the
    # whole graph is covered). Stores the anchor's region id (0 at the root) and the count of
    # mutation ops currently between anchor and stop. May lazily compute ids.
    if nodes:
        first_matched_node = min(nodes)
        last_matched_node = max(nodes)
        anchor = first_matched_node.prev
        stop = last_matched_node.next
    else:
        first_node = next(iter(graph.nodes))
        anchor = first_node.prev
        stop = anchor

    mutation_region_id = (
        0 if anchor.op == "root" else get_mutation_region_id(graph, anchor)
    )
    return _MutationRegionRefresh(
        anchor,
        stop,
        mutation_region_id,
        _count_mutation_ops_until(anchor.next, stop),
    )


def _refresh_mutation_region_ids(
    graph: torch.fx.Graph,
    refresh: _MutationRegionRefresh,
) -> None:
    # [FUNCTION DOC] HELPER: Recomputes mutation_region_id metadata after a rewrite. If the
    # anchor or stop node was erased (or the root is hit) recomputes the whole graph. Otherwise
    # renumbers nodes between anchor and stop, and if the number of mutation ops there changed
    # from the snapshot, renumbers all later nodes to the end of the graph with shifted ids.
    if refresh.anchor._erased or refresh.stop._erased:
        compute_mutation_region_ids(graph)
        return

    nd = refresh.anchor.next
    mutation_region_id = refresh.mutation_region_id
    mutation_count = 0
    while nd is not refresh.stop:
        if nd.op == "root":
            compute_mutation_region_ids(graph)
            return
        if is_mutation_op(nd):
            mutation_region_id += 1
            mutation_count += 1
        nd.meta["mutation_region_id"] = mutation_region_id
        nd = nd.next

    if mutation_count == refresh.mutation_count:
        return

    while nd.op != "root":
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id
        nd = nd.next


def _wrap_bound_method(fn: Any, argnames: list[str]) -> Any:
    # [FUNCTION DOC] HELPER: Returns a plain function forwarding to fn whose signature lists
    # only argnames as positional-or-keyword parameters (hides self for tracing).
    """
    Wrap a bound method to remove 'self' from its signature for FX tracing.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Forwards all arguments to the wrapped bound method.
        return fn(*args, **kwargs)

    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in argnames
    ]
    wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return wrapper


class PatternMatcherPass:
    # [CLASS DOC] Registry and driver for a set of pattern rewrites over FX graphs. State:
    # patterns (mapping from (op, target) to ordered list of PatternEntry), pass_name and
    # subsystem (for observers/logging) and seen_patterns (duplicate-detection table from
    # serialized pattern source to the graphs that produced it). Usable as the pass_dict
    # target of register_* helpers via item access; apply() runs the rewrites.
    """
    Registry of patterns to match and replace in FX graphs.
    """

    def __init__(
        self,
        pass_name: str | None = None,
        subsystem: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates empty pattern and seen-pattern tables and stores the optional pass name and subsystem.
        super().__init__()
        self.patterns: defaultdict[
            tuple[str, torch.fx.node.Target], list[PatternEntry]
        ] = defaultdict(list)
        self.pass_name = pass_name
        self.subsystem = subsystem

        # For a particular generated pattern repr, store all of the str representations
        # of the graph used to generate them. Because we ignore certain patterns
        # in searching, but not in matching, use the graph to distinguish if two equivalent
        # searches are actually different.
        self.seen_patterns: dict[str, list[str | None]] = defaultdict(list)

    def __getitem__(self, item: tuple[str, torch.fx.node.Target]) -> list[PatternEntry]:
        # [FUNCTION DOC] Returns (creating if absent) the entry list for the (op, target) key.
        return self.patterns[item]

    def apply(self, gm: torch.fx.GraphModule | torch.fx.Graph) -> int:
        # [FUNCTION DOC] Runs all registered patterns over gm (GraphModule or Graph with an owning
        # module; RuntimeError for other types). Returns 0 if nothing is registered. Computes
        # mutation region ids if absent, collects candidate nodes by (op, target) and visits them
        # in reverse graph order, skipping nodes with unsupported or cpu-input types. For each
        # entry (stopping if the node is erased) tries a match; discards matches spanning more
        # than one mutation region or more than one stream/mempool context; if extra_check
        # passes (symbolic result evaluated as false when undecidable) applies the entry,
        # refreshing mutation region ids when mutation ops are involved. Updates global counters
        # (match count, nodes matched, and per-pattern counts when the debug env var is set) and
        # logs matches for debug-selected nodes. Mutates the graph; returns the number of applied matches.
        """Apply all registered patterns to the graph, returning the number of matches."""
        if not self.patterns:
            return 0
        if isinstance(gm, torch.fx.GraphModule):
            graph = gm.graph
        elif isinstance(gm, torch.fx.Graph):
            graph = gm
            gm = graph.owning_module  # type: ignore[assignment]
        else:
            raise RuntimeError(
                f"The input to PatternMatcherPass must be a GraphModule or a Graph, but got {type(gm)}"
            )
        if should_compute_mutation_region_ids(graph):
            compute_mutation_region_ids(graph)
        get_mutation_region_id_partial = functools.partial(
            get_mutation_region_id, graph
        )
        count = 0
        nodes = []
        has_call_module = False
        for op, target in self.patterns:
            if op == "call_module":
                has_call_module = True
            else:
                nodes.append(graph.find_nodes(op=op, target=target, sort=False))
        if has_call_module:
            nodes.append(graph.find_nodes(op="call_module", sort=False))
        pass_name = self.pass_name if self.pass_name is not None else "pattern_matcher"
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"expected GraphModule, got {type(gm)}")
        with GraphTransformObserver(gm, pass_name, self.subsystem):
            for node in sorted(itertools.chain.from_iterable(nodes), reverse=True):
                target = extract_target(node)
                if node.op == "call_module":
                    if (node.op, target) not in self.patterns:
                        continue

                # conservatively not applying pattern for cpu input,
                # since some of the patterns induce codegen and split nodes.
                # Note: we will only skip cpu compute if disable_cpp_codegen=True
                if fallback_node_due_to_unsupported_type(node, allow_cpu_inputs=False):
                    continue

                for entry in self.patterns[(node.op, target)]:
                    if node._erased:
                        break
                    m = entry.pattern.match(node)
                    # pattern match crosses mutation barrier - discard
                    if (
                        is_match(m)
                        and len(
                            OrderedSet(map(get_mutation_region_id_partial, m.nodes))
                        )
                        != 1
                    ):
                        continue
                    # pattern match crosses stream boundary - discard
                    if (
                        is_match(m)
                        and len(
                            OrderedSet(
                                (
                                    n.meta.get("custom", {}).get("stream", 0),
                                    n.meta.get("custom", {}).get("mempool"),
                                    n.meta.get("custom", {}).get("mempool_device"),
                                )
                                for n in m.nodes
                            )
                        )
                        != 1
                    ):
                        continue
                    if _should_debug_node(node.name):
                        log.warning("%s%s %s %s", node, node.args, m, entry.pattern)

                    if is_match(m) and guard_or_false(entry.extra_check(m)):
                        count += 1
                        if isinstance(entry, ReplacementPatternEntry):
                            mutation_region_refresh = (
                                _mutation_region_refresh(graph, m.nodes)
                                if _replacement_changes_mutation_regions(entry, m)
                                else None
                            )
                        elif isinstance(entry, GraphPatternEntry):
                            mutation_region_refresh = None
                        else:
                            mutation_region_refresh = _mutation_region_refresh(
                                graph, m.nodes
                            )
                        if isinstance(entry, GraphPatternEntry):
                            with _track_graph_mutation_ops(
                                graph, _common_custom_context(m.nodes)
                            ) as mutation_tracker:
                                entry.apply(m, graph, node)
                            if mutation_tracker.changed_mutation_regions():
                                compute_mutation_region_ids(graph)
                        elif mutation_region_refresh is not None:
                            entry.apply(m, graph, node)
                            _refresh_mutation_region_ids(graph, mutation_region_refresh)
                        else:
                            entry.apply(m, graph, node)
                        counters[backend]["pattern_matcher_count"] += 1
                        counters[backend]["pattern_matcher_nodes"] += len(m.nodes)

                        # Track per-pattern counts when debug mode is active
                        if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG"):
                            if getattr(entry, "pattern_name", None):
                                pattern_name = entry.pattern_name
                            else:
                                # Fallback: use pattern class name + operation target
                                pattern_class = entry.pattern.__class__.__name__
                                target = str(node.target)
                                pattern_name = f"{pattern_class}_{target}"

                            pattern_key = f"{backend}_pattern_matcher_per_pattern"
                            counters[pattern_key][pattern_name] += 1
        return count

    def clear(self) -> None:
        # [FUNCTION DOC] Removes all registered patterns (seen_patterns is left untouched).
        self.patterns.clear()


def _not_implemented(*args: Any, **kwargs: Any) -> NoReturn:
    # [FUNCTION DOC] HELPER: Placeholder handler that always raises NotImplementedError.
    raise NotImplementedError


def fx_to_pattern(
    gm: torch.fx.GraphModule | torch.fx.Graph,
    ignore_types: Sequence[type[Any]] = (),
    argnames: Sequence[str] = (),
    scalar_workaround: dict[str, float | int] | None = None,
    exclusive_arg_names: Sequence[str] = (),
) -> PatternExpr:
    # [FUNCTION DOC] Converts a traced GraphModule (AssertionError for other types) into a
    # PatternExpr by interpreting it: placeholders become keyword-arg patterns named from
    # argnames (extra placeholders must be tangents), get_attr becomes GetAttr, call_function
    # becomes CallFunction with constants of ignore_types replaced by Ignored and
    # scalar_workaround values replaced by named keyword args. call_method and call_module are
    # unsupported. Each pattern records its user count from the node. A multi-output result
    # becomes a MultiOutputPattern. scalar_workaround values must be unique (AssertionError).
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed-length lists.
    """
    # scalar_workaround is a hack to capture dropout_p
    # see https://github.com/pytorch/pytorch/issues/97894
    scalar_workaround = scalar_workaround or {}
    inv_scalar_workaround = {v: k for k, v in scalar_workaround.items()}
    if len(inv_scalar_workaround) != len(scalar_workaround):
        raise AssertionError("scalar_workaround has duplicate values")

    def process_arg(
        x: T, ignore_types_override: Sequence[type[Any]] | None = None
    ) -> T | KeywordArg | Ignored:
        # [FUNCTION DOC] HELPER: Maps a traced constant to a pattern element: a number equal to a
        # scalar_workaround value becomes a keyword arg of that name; a value whose exact type is
        # in the ignored types (override if given) becomes Ignored; a non-empty list consisting
        # only of Ignored becomes a single Ignored; else unchanged.
        current_ignore_types = (
            ignore_types_override if ignore_types_override is not None else ignore_types
        )
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in current_ignore_types:
            return Ignored()
        if isinstance(x, list) and all(isinstance(y, Ignored) for y in x) and x:
            return Ignored()
        return x

    argnum = itertools.count()

    class Converter(torch.fx.Interpreter):
        # [CLASS DOC] HELPER: FX Interpreter that evaluates a graph into pattern objects instead
        # of values; call_method and call_module raise NotImplementedError. Uses the enclosing
        # argument counter, argnames and exclusive names.
        # pyrefly: ignore [bad-override]
        call_method = _not_implemented
        # pyrefly: ignore [bad-override]
        call_module = _not_implemented

        # pyrefly: ignore [bad-override]
        def placeholder(
            self,
            target: str,  # type: ignore[override]
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ) -> ExclusiveKeywordArg | KeywordArg:
            # [FUNCTION DOC] HELPER: Returns a keyword-arg pattern for the next placeholder,
            # named by position from argnames; placeholders beyond argnames must start with
            # tangent (AssertionError otherwise) and use their own name; with no argnames the
            # trailing numeric suffix is stripped. Exclusive names give ExclusiveKeywordArg.
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                if not target.startswith("tangent"):
                    raise AssertionError(
                        f"expected target to start with 'tangent', got {target}"
                    )
                name = target
            else:
                target = re.sub(r"_\d+$", "", target)  # de-mangle arg name
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        # pyrefly: ignore [bad-override]
        def get_attr(
            self,
            target: str,  # type: ignore[override]
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ) -> GetAttr:
            # [FUNCTION DOC] HELPER: Returns a GetAttr pattern holding the attribute value fetched from the module; AssertionError if args or kwargs are given.
            if args or kwargs:
                raise AssertionError(f"unexpected get_attr args: {args}, {kwargs}")
            return GetAttr(_get_attr(self.module, target))

        # pyrefly: ignore [bad-override]
        def call_function(
            self,
            target: str,  # type: ignore[override]
            args: Sequence[Any],
            kwargs: Mapping[str, Any],
        ) -> PatternExpr:
            # [FUNCTION DOC] HELPER: Builds a CallFunction pattern for the target. Args and
            # kwargs are mapped through process_arg (for getitem, integer constants are kept so
            # indices still match); when list is an ignored type, list-valued args get the
            # same treatment again at the top level.
            process_arg_fn = process_arg
            # Indexing is critical for matching getitem nodes, so we can't ignore int args here
            if target is operator.getitem:

                def process_arg_fn_impl(
                    x: T,
                    ignore_types_override: Sequence[type[Any]] | None = tuple(
                        t for t in ignore_types if t is not int
                    ),
                ) -> T | KeywordArg | Ignored:
                    # [FUNCTION DOC] HELPER: process_arg with an ignore list that excludes int.
                    return process_arg(x, ignore_types_override)

                process_arg_fn = process_arg_fn_impl

            args, kwargs = pytree.tree_map(process_arg_fn, (args, kwargs))
            if list in ignore_types:
                # Handle a burned in tensor size which are now [Ignored(), Ignored(), ...]
                args = [process_arg_fn(a) for a in args]
                kwargs = {k: process_arg_fn(a) for k, a in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        def run_node(self, n: torch.fx.Node) -> Any:
            # [FUNCTION DOC] HELPER: Runs the node normally, then records the original node's
            # user count on the resulting pattern (per element for tuple outputs, which must
            # match the output node's argument collection in length; AssertionError otherwise).
            rv = super().run_node(n)
            if n.op == "output" and isinstance(rv, tuple):
                args = n.args[0]
                if not isinstance(args, Collection):
                    raise AssertionError(f"expected Collection, got {type(args)}")
                if len(rv) != len(args):
                    raise AssertionError("rv and args length mismatch")
                for r, arg in zip(rv, args):
                    # pyrefly: ignore [missing-attribute]
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv

    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected GraphModule, got {type(gm)}")
    pattern = Converter(gm).run()
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern


@torch.no_grad()
def fwd_only(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    run_functional_passes: bool = True,
    get_decomp_fn: Callable[..., Any] = select_decomp_table,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Traces fn on args (real tensors, no grad) into a GraphModule with
    # make_fx under the python dispatcher, node metadata preserved and compile_on_one_rank
    # disabled, using the decomposition table from get_decomp_fn. If run_functional_passes,
    # removes no-op ops and dead code. Recompiles and returns the module.
    """Build a normalized inference graph, for use with fx_to_pattern"""
    from torch.compiler import config as compiler_config

    # Patterns are device-agnostic templates traced with fixed example tensors; keep the
    # compile-on-one-rank device handling out of pattern tracing so make_fx's single-device
    # check only validates real user graphs, not these internal fixed-device templates.
    # TODO - look into using aot autograd, asserting no mutating ops here
    with (
        # pyrefly: ignore [missing-attribute]
        compiler_config.patch(compile_on_one_rank=False),
        enable_python_dispatcher(),
        preserve_node_meta(),
    ):
        gm = make_fx(fn, get_decomp_fn(), tracing_mode="real")(*args)

    from .fx_passes.post_grad import remove_noop_ops

    if run_functional_passes:
        remove_noop_ops(gm.graph)

        # NOTE: applying early_patterns to user patterns cause
        # duplicate patterns being found in vllm. Check
        # https://github.com/pytorch/pytorch/pull/170649#issuecomment-3693427775
        # for more details.
        # from .fx_passes.joint_graph import early_patterns
        # early_patterns.apply(gm.graph)

        gm.graph.eliminate_dead_code()

    gm.recompile()
    return gm


@torch.enable_grad()
def joint_fwd_bwd(
    fn: Callable[..., Any],
    args: Sequence[Any],
    *,
    get_decomp_fn: Callable[..., Any] = select_decomp_table,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Traces fn on args with AOT autograd and captures the joint forward+backward
    # graph through a custom partition function (the compiled functions are discarded). Then
    # removes no-op ops, applies the early joint-graph patterns, resets the codegen in/out
    # specs, eliminates dead code, recompiles and returns the module. Raises AssertionError if
    # no graph was captured. Gradients enabled; compile_on_one_rank disabled during tracing.
    """Build a normalized training graph, for use with fx_to_pattern"""
    gm: torch.fx.GraphModule | None = None

    def record_joint_graph(
        joint_graph: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        # [FUNCTION DOC] HELPER: Partition callback: stores a clone of the joint graph in the
        # enclosing gm (AssertionError if already set) and then delegates to the default partitioner.
        nonlocal gm
        if gm:
            raise AssertionError("gm already set")
        gm = clone_graph(joint_graph)
        return default_partition(joint_graph, inputs, **kwargs)

    from torch.compiler import config as compiler_config

    # Keep compile-on-one-rank device handling out of pattern tracing (see fwd_only).
    # pyrefly: ignore [missing-attribute]
    with torch._guards.tracing(None), compiler_config.patch(compile_on_one_rank=False):
        aot_function(
            fn,
            # pyrefly: ignore[bad-argument-type]
            lambda gm, example_inputs: make_boxed_func(gm),
            partition_fn=record_joint_graph,
            decompositions=get_decomp_fn(),
            keep_inference_input_mutations=True,
            enable_log=False,
        )(*args)
    if not gm:
        raise AssertionError("gm was not set")

    from .fx_passes.post_grad import remove_noop_ops

    remove_noop_ops(gm.graph)

    from .fx_passes.joint_graph import early_patterns

    early_patterns.apply(gm.graph)

    # remove in/out specs
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _args(n: torch.fx.Node) -> list[torch.fx.node.Argument]:
    # [FUNCTION DOC] HELPER: Returns all Node arguments (from args and kwargs, nested) of n as a flat list.
    args: list[torch.fx.node.Argument] = []
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    return args


def stable_topological_sort(graph: torch.fx.Graph) -> None:
    # [FUNCTION DOC] Reorders graph nodes in place into a valid topological order while
    # keeping the original order wherever possible: a node whose inputs are not yet placed
    # waits on its last unplaced input, and is released after that input is placed. Raises
    # AssertionError if not all nodes could be ordered (cycle).
    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = OrderedSet[torch.fx.Node]()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    if not (not waiting and len(ready) == len(graph.nodes)):
        raise AssertionError("topological sort did not consume all nodes")


def init_once_fakemode(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Decorator for lazy pattern-initialization functions. Returns a cached
    # wrapper (cache keyed on input_device and get_decomp_fn) that calls fn once per key,
    # forwarding input_device / get_decomp_fn only if fn declares them, under fake tensor mode
    # with real mode and tracing context disabled. Restores the backend counters afterwards so
    # matches during tracing are not counted.
    """Wrapper around lazy init functions in fx_passes/"""

    _fn_params = inspect.signature(fn).parameters

    @functools.cache
    @functools.wraps(fn)
    def lazy_init(
        input_device: Any | None = None,
        get_decomp_fn: Callable[
            ..., dict[Any, Callable[..., Any]]
        ] = select_decomp_table,
    ) -> Any:
        # [FUNCTION DOC] HELPER: Cached initializer; runs fn once per distinct arguments inside fake mode and returns its result.
        counters_ref = counters[backend].copy()

        kwargs: dict[str, Any] = {}
        if "input_device" in _fn_params:
            kwargs["input_device"] = input_device
        if "get_decomp_fn" in _fn_params:
            kwargs["get_decomp_fn"] = get_decomp_fn

        with torch._guards.tracing(None), unset_fake_temporarily(), FakeTensorMode():
            result = fn(**kwargs)

        # clear view matches encountered during tracing
        counters[backend] = counters_ref

        return result

    return lazy_init


def config_flag(name: str) -> Callable[[Match], Any]:
    # [FUNCTION DOC] Returns an extra_check function that reads the inductor config attribute
    # called name (at check time) and returns its value, gating a pass behind a flag.
    """Function for extra_check to put pass behind a flag"""

    def flag_check(match: Match) -> Any:
        # [FUNCTION DOC] HELPER: Returns the current value of the config flag; match is unused.
        return getattr(config, name)

    return flag_check


def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Returns a copy of input_graph made by an FX Transformer, preserving node
    # metadata and names.
    class CopyGraph(Transformer):
        # [CLASS DOC] HELPER: Transformer that copies each node, carrying over metadata and node name.
        def run_node(self, old_node: torch.fx.Node) -> torch.fx.Node:
            # [FUNCTION DOC] HELPER: Runs the node normally; for resulting proxies copies the old node's meta and assigns a fresh unique name based on the old name.
            new_node = super().run_node(old_node)
            if isinstance(new_node, torch.fx.Proxy):
                new_node.node.meta.update(old_node.meta)
                new_node.node.name = self.new_graph._graph_namespace.create_name(
                    old_node.name, None
                )

            return new_node

    return CopyGraph(input_graph).transform()


# TODO: remove in follow up diff, used internally
_seen_patterns: OrderedSet[str] = OrderedSet()


def get_arg_value(
    node: torch.fx.Node, arg_number: int, kwarg_name: str | None = None
) -> Any:
    # [FUNCTION DOC] Returns positional argument arg_number of node if present; else the keyword
    # argument kwarg_name if given (None when absent), else None.
    if len(node.args) > arg_number:
        return node.args[arg_number]
    elif kwarg_name is None:
        return None
    else:
        return node.kwargs.get(kwarg_name)


def filter_nodes(nodes: Iterable[torch.fx.Node], fn: Any) -> list[torch.fx.Node]:
    # [FUNCTION DOC] Returns the nodes whose target equals fn (or, for an operator packet, any of its overloads), preserving order.
    fns = [fn]
    if isinstance(fn, torch._ops.OpOverloadPacket):
        fns.extend(fn.op_overloads())

    return [node for node in nodes if node.target in fns]


def extract_target(node: torch.fx.Node) -> torch.fx.node.Target:
    # [FUNCTION DOC] Returns the matching key for a node: its target, except call_module nodes
    # return the class of the referenced submodule (AssertionError if the target is not a string
    # or the graph has no owning module).
    """For call_function and call_method, we directly use the target function;
    For call_module, the target is string, and we treat the module class
     as a function.
    """
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str node.target, got {type(node.target)}")
        if node.graph.owning_module is None:
            raise AssertionError("node.graph.owning_module must not be None")
        return _get_attr(node.graph.owning_module, node.target).__class__
    return node.target
