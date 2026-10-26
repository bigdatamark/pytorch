# [MODULE DOC] Logging helpers for AOTAutograd: module-level global state tracking the name of the
# graph currently being compiled (a stack-like list, a global counter and a model name), a context
# manager that sets/clears that state, hooks that preserve forward stack traces on backward
# autograd nodes, and small message formatters. Import-time side effects: only defines the globals
# graph_being_compiled, nth_graph, model_name and callback_set; get_graph_being_compiled is an alias
# of get_aot_graph_name.
"""
Contains utils for logging in AOTAutograd, including managing the names of the graphs under
compilation, capturing user-friendly tracebacks, and debug messages.
"""

import collections
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx.traceback as fx_traceback

from .schemas import AOTConfig


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: list[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name: str) -> None:
    # [FUNCTION DOC] Sets the global model_name used as the prefix of AOT graph names.
    # Side effect: mutates module-global state. Returns nothing.
    global model_name
    model_name = name


def get_aot_compilation_context() -> tuple[list[str], str, int]:
    # [FUNCTION DOC] Returns a snapshot of the compilation naming state as a tuple of
    # (copy of the graph-being-compiled name list, model name, graph counter). No side effects.
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    # [FUNCTION DOC] Builds the current graph name: model name, double underscore, the
    # graph-being-compiled entries joined by underscores, underscore, then the global graph counter.
    # Reads globals only; no side effects. Also exported under the alias get_graph_being_compiled.
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(
    aot_config: AOTConfig, graph_name: str
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager marking that a graph is being compiled. On entry it replaces the
    # global graph_being_compiled list with one entry made of the aot_id and graph_name, and, if a
    # TracingContext is active, stores that list in its aot_graph_name (remembering the old value).
    # On exit (even on error) it increments the global nth_graph counter, empties the global list,
    # and restores the previous aot_graph_name if a tracing context is still active.
    # Args: aot_config (supplies aot_id), graph_name (str). Yields nothing.
    global graph_being_compiled
    # TODO: Don't shove the aot_id in here; set it in the context
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    old_name = None
    if tracing_context := torch._guards.TracingContext.try_get():
        old_name = tracing_context.aot_graph_name
        tracing_context.aot_graph_name = graph_being_compiled
        has_tracing_context = True
    else:
        has_tracing_context = False
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []
        if has_tracing_context:
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.aot_graph_name = old_name


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: list[torch.autograd.graph.Node]) -> None:
    # [FUNCTION DOC] Walks the autograd graph reachable from the given root nodes and registers on
    # every node a pre-hook and a post-hook so that, while backward runs, the fx stack trace and
    # grad_fn sequence number are set from the stack recorded on the node's metadata during forward
    # (empty if absent). The post-hook uses that stack plus the gradient-accumulation special marker.
    # Args: roots, list of autograd Nodes. Returns nothing. Side effects: registers hooks on nodes.
    def iter_graph(
        roots: list[torch.autograd.graph.Node],
    ) -> Iterator[torch.autograd.graph.Node]:
        # [FUNCTION DOC] HELPER: Breadth-first generator over autograd nodes reachable from roots via
        # next_functions, visiting each node once and skipping None entries. Yields nothing if roots
        # is empty.
        if not roots:
            return
        seen = set()
        q = collections.deque()
        for node in roots:
            if node is not None and node not in seen:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_: list[str]) -> Callable[[], None]:
        # [FUNCTION DOC] HELPER: Returns a callback closing over saved_stack_ that restores it as the
        # fx stack trace and clears the global callback_set flag.
        def callback() -> None:
            # [FUNCTION DOC] HELPER: Sets the fx stack trace to the saved stack and resets the global
            # callback_set flag to false. Intended to run when the autograd engine finishes backward.
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_: list[str], seq_nr: int) -> Callable[[Any], None]:
        # [FUNCTION DOC] HELPER: Returns a pre-hook closing over the forward stack and sequence number.
        def prehook(grad_output: Any) -> None:
            # [FUNCTION DOC] HELPER: Autograd node pre-hook. If no end-of-backward callback is pending,
            # queues one on the execution engine that restores the current formatted stack, and sets
            # callback_set. Then sets the fx stack trace to the node's forward stack, sets the grad_fn
            # sequence number, and marks that autograd backward is in progress. grad_output is unused.
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(  # type: ignore[attr-defined]
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)
            fx_traceback.set_grad_fn_seq_nr(seq_nr)
            fx_traceback._mark_autograd_backward()

        return prehook

    def get_posthook(
        special_stack_: list[str], seq_nr: int
    ) -> Callable[[Any, Any], None]:
        # [FUNCTION DOC] HELPER: Returns a post-hook closing over the special stack (the sequence
        # number argument is unused by the returned hook).
        def posthook(grad_input: Any, grad_output: Any) -> None:
            # [FUNCTION DOC] HELPER: Autograd node post-hook. Sets the fx stack trace to the special
            # stack (forward stack plus gradient-accumulation marker), resets the grad_fn sequence
            # number, and clears the in-backward mark. Arguments are unused.
            fx_traceback.set_stack_trace(special_stack_)
            fx_traceback.reset_grad_fn_seq_nr()
            fx_traceback._reset_autograd_backward()

        return posthook

    for node in iter_graph(roots):
        # pyrefly: ignore[missing-attribute]
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))

        special_stack = forward_node_stack.copy()
        special_stack.append(fx_traceback.GRADIENT_ACC_SPECIAL_STACK)
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))


def setup_stacktrace_preservation_hooks_from_tensors(outputs: Any) -> None:
    # [FUNCTION DOC] Accepts a tensor or a list/tuple of values, collects the grad_fn of each tensor
    # that has one, and if any exist installs the stack-trace preservation hooks on them. Non-tensors
    # and tensors without grad_fn are ignored. Returns nothing.
    roots = [
        t.grad_fn
        for t in (outputs if isinstance(outputs, (list, tuple)) else (outputs,))
        if isinstance(t, torch.Tensor) and t.grad_fn is not None
    ]
    if roots:
        setup_stacktrace_preservation_hooks(roots)


def describe_input(i: int, aot_config: AOTConfig) -> str:
    # [FUNCTION DOC] Returns a human-readable label for flat input index i: a parameter/buffer label
    # with index i when i is below aot_config.num_params_buffers, otherwise a user input label with
    # the index offset by that count.
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"


def format_guard_bug_msg(aot_config: AOTConfig, expected: str) -> str:
    # [FUNCTION DOC] Builds an error message saying graph aot_id was compiled assuming the expected
    # condition but it did not hold at runtime, indicating a guard bug to report. Returns a string.
    return (
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )
