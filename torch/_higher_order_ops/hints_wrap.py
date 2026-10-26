# [MODULE DOC] Defines the hints_wrapper higher-order operator, which runs a body function
# unchanged but carries a dict of context hints (for backend compilers) alongside it. Registers
# dense, autograd (deferred not-implemented error), fake tensor, functionalization and
# proxy-tracing implementations on the singleton hints_wrapper at import time.
# mypy: allow-untyped-defs
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    autograd_not_implemented,
    reenter_make_fx,
    register_fake,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


# used for wrapping a function/op with context hints
class HintsWrapper(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator named "hints_wrapper". Stateless; validates its inputs in
    # __call__ and then dispatches. One singleton instance, hints_wrapper, exists.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the HigherOrderOperator with the name "hints_wrapper".
        super().__init__("hints_wrapper")

    def __call__(self, body_fn, args, kwargs, hints):
        # [FUNCTION DOC] Validates inputs then dispatches the HOP. args is converted to a tuple and must
        # contain only tensors, ints, floats or bools; kwargs must be an empty dict (hints
        # are the only extra data); hints must be a dict with string keys and int, float,
        # bool or str values. Raises RuntimeError on any violation. Returns the dispatched result.
        r"""
        Call implementation of hints_wrapper

        Args:
            body_fn (Callable): A callable function that is within the scope
             that is being traced.

            args (Tuple of torch.Tensor/int/float/bool): A tuple of inputs to
             body_fn.

            kwargs (dict): Keyword argument to the body_fn.

            hints (dict): A dict of context hints which could be passed to
             backend compiler.
        """
        if not isinstance(args, tuple):
            args = tuple(args)

        if not all(isinstance(t, (torch.Tensor, int, float, bool)) for t in args):
            raise RuntimeError(
                f"args must be a tuple of tensors, ints, floats, or bools, got {args}"
            )

        if not isinstance(kwargs, dict):
            raise RuntimeError(f"kwargs must be a dict, got {type(kwargs)}")

        if len(kwargs) > 0:
            raise RuntimeError(
                f"kwargs except for hints are not supported, got {kwargs}"
            )

        if not isinstance(hints, dict):
            raise RuntimeError(f"hints must be a dict, got {type(hints)}")

        for k, v in hints.items():
            if not isinstance(k, str):
                raise RuntimeError(f"hints key must be a str, got {k}.")

            if not isinstance(v, (int, float, bool, str)):
                raise RuntimeError(
                    "hints must be a dict containing int, float, bool or str "
                    f"value, got value {v} for key {k}."
                )

        # pyrefly: ignore [missing-attribute]
        return super().__call__(body_fn, args, kwargs, hints)


hints_wrapper = HintsWrapper()


@hints_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def hints_wrapper_dense(body_fn, args, kwargs, hints):
    # [FUNCTION DOC] HELPER: Eager implementation: calls body_fn with args and kwargs and returns its
    # result; hints are ignored.
    return body_fn(*args, **kwargs)


hints_wrapper.py_autograd_impl(
    autograd_not_implemented(hints_wrapper, deferred_error=True)
)


@register_fake(hints_wrapper, skip_cache=True)
def hints_wrapper_fake_tensor_mode(body_func, args, kwargs, hints):
    # [FUNCTION DOC] HELPER: Fake tensor implementation: flattens args with pytree and calls body_func on
    # the flat leaves plus kwargs, returning the result; hints are ignored.
    flat_args = pytree.tree_leaves(args)
    return body_func(*flat_args, **kwargs)


@hints_wrapper.py_functionalize_impl
def hints_wrapper_functionalize(ctx, body_fn, args, kwargs, hints):
    # [FUNCTION DOC] HELPER: Functionalization rule. Unwraps args, kwargs and hints, then under a
    # redispatch to the next level functionalizes body_fn, checks that body_fn does not alias
    # or mutate its inputs (via _check_alias_and_mutation, which raises on violation), re-invokes
    # hints_wrapper with the functional body and re-wraps the outputs.
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    unwrapped_hints = ctx.unwrap_tensors(hints)
    with ctx.redispatch_to_next():
        functional_body_fn = ctx.functionalize(body_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(
            body_fn, unwrapped_args, "hints_wrapper", pre_dispatch
        )

        outputs = hints_wrapper(
            functional_body_fn,
            unwrapped_args,
            unwrapped_kwargs,
            unwrapped_hints,
        )
        return ctx.wrap_tensors(outputs)


def trace_hints_wrapper(proxy_mode, hints_wrapper, body_fn, args, kwargs, hints):
    # [FUNCTION DOC] HELPER: Traces the HOP into the FX graph. Flattens args, traces body_fn into a
    # sub-graph with reenter_make_fx, registers it on the tracer root under a unique
    # "hints_wrapper_body_graph" name, and creates a call_function node taking the sub-graph,
    # flat args and an empty kwargs, with hints passed as a keyword. Also runs body_fn eagerly
    # to compute real outputs and returns them tracked against the new proxy node.
    flat_args = tuple(pytree.tree_leaves(args))
    body_graph = reenter_make_fx(body_fn)(*flat_args, **kwargs)

    _, body_graph_name = unique_graph_id(proxy_mode, prefix="hints_wrapper_body_graph")
    proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

    new_args: tuple = (body_graph, flat_args, {})
    # merge hints into kwargs
    new_kwargs = {}
    new_kwargs["hints"] = hints

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_args)
    proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, new_kwargs)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", hints_wrapper, proxy_args, proxy_kwargs, name="hints_wrapper"
    )

    out = body_fn(*flat_args, **kwargs)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@hints_wrapper.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, body_fn, args, kwargs, hints):
    # [FUNCTION DOC] HELPER: ProxyTorchDispatchMode rule: if tracing is enabled, delegates to
    # trace_hints_wrapper; otherwise just calls hints_wrapper directly.
    if proxy_mode.enable_tracing:
        return trace_hints_wrapper(
            proxy_mode, hints_wrapper, body_fn, args, kwargs, hints
        )
    else:
        return hints_wrapper(body_fn, args, kwargs, hints)
