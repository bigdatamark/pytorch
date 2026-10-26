# mypy: allow-untyped-defs
# [MODULE DOC] Defines the strict_mode higher-order op and its public wrapper torch.strict_mode,
# which runs a callable on a tuple of operands as a single traced subgraph. Provides implementations
# for the dense, proxy tracing, fake tensor and functionalize paths; autograd is deferred-error.
# Import-time side effects: creates the strict_mode_op singleton and registers its impls.

import torch
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


@exposed_in("torch")
def strict_mode(callable, operands):
    # [FUNCTION DOC] Public entry point exposed as torch.strict_mode. Runs callable on operands
    # (a tuple of inputs). When Dynamo is compiling it calls strict_mode_op directly; otherwise it
    # compiles and calls the op through the generic higher-order-op compile helper.
    # Returns the callable's outputs.
    if torch.compiler.is_dynamo_compiling():
        return strict_mode_op(callable, operands)

    from torch._higher_order_ops.utils import _hop_compile_and_call

    return _hop_compile_and_call(strict_mode_op, (callable, operands))


class StrictMode(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator named "strict_mode" taking a callable and its operands.
    # Stateless; the module-level singleton strict_mode_op is the instance used.
    def __init__(self):
        # [FUNCTION DOC] Registers the operator under the name "strict_mode".
        super().__init__("strict_mode")

    def __call__(self, callable, operands):
        # [FUNCTION DOC] Forwards callable and operands to the generic higher-order-op dispatch.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(callable, operands)


strict_mode_op = StrictMode()


@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    # [FUNCTION DOC] Eager implementation: calls callable with the operands unpacked and returns
    # the result. Raises AssertionError if a dispatch mode is active (should never happen here).
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return callable(*operands)


strict_mode_op.py_autograd_impl(
    autograd_not_implemented(strict_mode_op, deferred_error=True)
)


@strict_mode_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, callable, operands):
    # [FUNCTION DOC] Proxy-mode implementation; delegates to trace_strict_mode with strict_mode_op.
    return trace_strict_mode(mode, strict_mode_op, callable, operands)


def trace_strict_mode(mode, strict_mode_op, callable, operands):
    # [FUNCTION DOC] Traces callable into an FX graph (honoring the mode's pre_dispatch flag, with
    # proxy tracing disabled during the inner trace), registers it on the tracer root under a fresh
    # "strict_graph_" name, emits a call_function node for the op with (graph, operands), then runs
    # the graph on the real operands and returns the outputs tracked to that node's proxy.
    # Side effect: adds a submodule and a node to the tracer's graph.
    pre_dispatch = getattr(mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        graph = make_fx(callable, pre_dispatch=pre_dispatch)(*operands)

    graph_name = mode.tracer.get_fresh_qualname("strict_graph_")
    mode.tracer.root.register_module(graph_name, graph)

    args = (graph, operands)

    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)

    out_proxy = mode.tracer.create_proxy(
        "call_function", strict_mode_op, proxy_args, {}, name="strict_mode"
    )

    out = graph(*operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@register_fake(strict_mode_op, skip_cache=True)
def strict_mode_fake_tensor_mode(callable, operands):
    # [FUNCTION DOC] Fake tensor implementation (cache skipped): simply calls callable on operands.
    return callable(*operands)


@strict_mode_op.py_functionalize_impl
def strict_mode_func(ctx, callable, inputs):
    # [FUNCTION DOC] Functionalization implementation: unwraps input tensors, functionalizes the
    # callable, re-invokes strict_mode_op below the functionalize key and wraps the outputs.
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    with ctx.redispatch_to_next():
        functional_callable = ctx.functionalize(callable)

        cond_return = strict_mode_op(functional_callable, unwrapped_inputs)
        return ctx.wrap_tensors(cond_return)
