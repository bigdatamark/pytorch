# [MODULE DOC] Defines the run_const_graph higher-order op, which executes a constant-folded
# FX GraphModule on a tuple of (weight) arguments. Registers implementations for proxy tracing,
# functionalization, fake tensor mode and the CPU key; autograd is deferred-error (not implemented).
# Import-time side effects: instantiates the run_const_graph singleton and registers all impls on it.
from typing import Any, TYPE_CHECKING

import torch
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree


class RunConstGraph(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator named "run_const_graph" that runs a constant graph
    # (a GraphModule computing folded constants) on a tuple of arguments. Stateless; a single
    # module-level instance, run_const_graph, is used.
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name "run_const_graph".
        super().__init__("run_const_graph")

    def __call__(self, graph: torch.fx.GraphModule, args: tuple[object, ...]) -> object:
        # [FUNCTION DOC] Forwards the graph module and argument tuple to the generic
        # higher-order-op dispatch and returns whatever the selected implementation returns.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(graph, args)


run_const_graph = RunConstGraph()


@run_const_graph.py_impl(ProxyTorchDispatchMode)
def run_const_graph_dispatch_mode(
    mode: ProxyTorchDispatchMode, graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    # [FUNCTION DOC] Proxy-mode (tracing) implementation. Registers the constant graph as the
    # submodule "_const_graph" on the tracer root, emits a call_function node for run_const_graph
    # with proxied args, runs the graph eagerly on the real args to get outputs, and returns the
    # outputs tracked against the new proxy.
    # Raises AssertionError if graph is not a GraphModule or the root already has a _const_graph.
    # Side effect: mutates the tracer root module.
    const_gm, weights = graph, args
    p_args = pytree.tree_map(mode.tracer.unwrap_proxy, (graph, args))  # type: ignore[union-attr]
    if not isinstance(const_gm, torch.fx.GraphModule):
        raise AssertionError(
            f"expected const_gm to be torch.fx.GraphModule, got {type(const_gm)}"
        )
    if hasattr(mode.tracer.root, "_const_graph"):  # type: ignore[union-attr]
        raise AssertionError("mode.tracer.root already has _const_graph attribute")
    mode.tracer.root.register_module("_const_graph", const_gm)  # type: ignore[union-attr]

    proxy = mode.tracer.create_proxy("call_function", run_const_graph, p_args, {})

    out = const_gm(*weights)
    return track_tensor_tree(out, proxy, constant=None, tracer=mode.tracer)


@run_const_graph.py_functionalize_impl
def run_const_graph_functional(
    ctx: "BaseFunctionalizeAPI", graph: torch.fx.GraphModule, args: tuple[Any, ...]
) -> Any:
    # [FUNCTION DOC] Functionalization implementation: unwraps functional tensors in args,
    # re-invokes run_const_graph below the functionalize key, and wraps the outputs back.
    # No mutation checking is done on the graph.
    unwrapped_args = ctx.unwrap_tensors(args)

    with ctx.redispatch_to_next():
        out = run_const_graph(graph, unwrapped_args)
        return ctx.wrap_tensors(out)  # type: ignore[arg-type]


run_const_graph.py_autograd_impl(
    autograd_not_implemented(run_const_graph, deferred_error=True)
)


@register_fake(run_const_graph, skip_cache=True)
def run_const_graph_fake_tensor_mode(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    # [FUNCTION DOC] Fake tensor implementation (cache skipped): validates that graph is a
    # GraphModule (raises AssertionError otherwise) and simply calls it on the fake args.
    if not isinstance(graph, torch.fx.GraphModule):
        raise AssertionError(
            f"expected graph to be torch.fx.GraphModule, got {type(graph)}"
        )
    return graph(*args)


@run_const_graph.py_impl(DispatchKey.CPU)
def run_const_graph_cpu(
    graph: torch.fx.GraphModule, args: tuple[object, ...]
) -> object:
    # [FUNCTION DOC] CPU-key implementation: validates graph is a GraphModule (AssertionError
    # otherwise) and calls it on the args, returning its outputs.
    if not isinstance(graph, torch.fx.GraphModule):
        raise AssertionError(
            f"expected graph to be torch.fx.GraphModule, got {type(graph)}"
        )
    return graph(*args)
