# [MODULE DOC]
# Defines the trace_wrapped higher-order op, which behaves like calling a
# function but appears as an opaque call_function node under proxy tensor
# tracing (used by AOTAutograd to defer backward hooks to compiled autograd).
# Also defines helpers for flex attention score_mod backward: the custom op
# zeros_and_scatter, the ModIndex autograd function (mod_index), and
# TransformGetItemToIndex. Import-time side effects: registers the custom op
# flex_lib::zeros_and_scatter (fake and vmap impls) and the dispatch
# implementations of the trace_wrapped op. Exports trace_wrapped.
"""trace_wrapped(*args, fn) is equivalent to fn(*args), but with a twist:
if you make_fx trace through this call, we will not actually trace into fn; instead,
we will directly insert it as a call_function to fn in the graph.
(Unlike make_fx, Dynamo WILL inline into fn.)
You can think of this as a one off allow_in_graph equivalent for proxy tensor tracing.

Because proxy tensor tracing does not actually run the function, there are
requirements on the behavior of fn. We are still figuring it out, but here is the current state:

1) fn SHOULD only take a single argument, which must be a tensor
2) fn MUST return a new tensor with the same metadata as the original tensor
   (e.g., zeros_like(input) is a permissible implementation of fn).
   This is verified via an extra assert that is inserted into the traced graph.
3) fn MAY have side effects, but it MAY NOT perform metadata mutation on other tensors
   participating in proxy tensor tracing (it MAY mutate other tensors, it MAY mutate Python state)
These requirements stem from the requirement that we need to continue performing proxy tensor tracing,
which assumes accurate fake tensor metadata, without actually running fn.
In the future, we may allow for a "meta" function associated with fn to allow for more interesting input-output patterns.

Note that tensors / Python state are allowed to be mutated.
This relaxed constraint is not always sound, but it is sound for backward tracing with fake
tensors as it takes place in AOTAutograd, as the backward pass is guaranteed not to depend on concrete
tensor values (via fake tensor) or Python state (because the autograd engine doesn't depend on Python).

The intended use case for this function is to allow AOTAutograd to defer complex
backward hooks to compiled autograd. AOTAutograd performs a make_fx trace which preserves
the function call as is in the graph, and only when we Dynamo through the backward graph in
compiled autograd do we inline into the function.
"""

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses import FakeTensorMode
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._pytree import tree_map_only


Tensor = torch.Tensor


__all__ = ["trace_wrapped"]


@torch.library.custom_op("flex_lib::zeros_and_scatter", mutates_args=())  # type: ignore[misc]
def zeros_and_scatter(
    shape: list[int],
    indices: list[Tensor],
    vals: Tensor,
) -> Tensor:
    # [FUNCTION DOC]
    # Custom op: builds a zero tensor of the given shape and accumulates vals
    # into it at positions given by the index tensors (scatter-add via
    # index_put). With no indices, only a scalar output (empty shape) is
    # allowed and the sum of vals is returned; otherwise raises RuntimeError.
    """Custom Op so that we can register a custom lowering for the new_output + scatter in the backwards pass"""
    if not indices:
        if shape:
            raise RuntimeError(
                "zeros_and_scatter with no indices only supports scalar outputs"
            )
        return vals.sum()
    grad = torch.zeros(shape, device=vals.device, dtype=vals.dtype)
    return torch.ops.aten.index_put(grad, indices, vals, accumulate=True)


@zeros_and_scatter.register_fake  # type: ignore[misc]
def _(
    shape: list[int],
    indices: list[Tensor],
    vals: Tensor,
) -> Tensor:
    # [FUNCTION DOC] HELPER: Fake implementation: an uninitialized tensor of
    # the requested shape with vals' dtype and device.
    return vals.new_empty(shape)


@zeros_and_scatter.register_vmap  # type: ignore[misc]
def _(info, indims, shape, indices, value):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: vmap rule. Unbatched indices are expanded to the
    # value's shape; batched indices must already match it (AssertionError
    # otherwise). Calls the op on the expanded indices and returns the result
    # paired with None, meaning the output is not batched.
    """The batching rule is special in that it returns a tensor that is not batched"""
    indices_indims: list[int | None] = indims[1] if indims[1] is not None else []
    expanded_indices = []
    for idx, idx_indim in zip(indices, indices_indims):
        # The index is not being batched, we should unsqueeze and expand to val
        if idx_indim is None:
            expanded_indices.append(idx.expand(value.shape))
        else:
            # the index is being part of the vmap batch, it should be the same size as val
            if idx.shape != value.shape:
                raise AssertionError(
                    f"idx shape {idx.shape} must match value shape {value.shape}"
                )
            expanded_indices.append(idx)

    out = torch.ops.flex_lib.zeros_and_scatter(
        shape,
        expanded_indices,
        value,
    )
    return out, None


class ModIndex(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd function performing advanced indexing of x by a list of index
    # tensors, whose backward uses the zeros_and_scatter custom op (so the
    # backward graph is a single scatter-add). Supports an empty index list for
    # scalar tensors. Automatic vmap rule is generated. Saves the index tensors
    # and the input shape on the context.
    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(x: Tensor, indices: list[Tensor]) -> Tensor:
        # [FUNCTION DOC] Index x by indices; with no indices, x must be 0-d
        # (else RuntimeError) and is returned unchanged.
        if not indices:
            if x.ndim != 0:
                raise RuntimeError(
                    "mod_index with no indices only supports scalar tensors"
                )
            return x
        return torch.ops.aten.index(x, indices)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        # [FUNCTION DOC] Save index tensors for backward and record the input
        # shape on ctx.
        x, indices = inputs
        ctx.save_for_backward(*indices)
        ctx.input_shape = x.shape

    @staticmethod
    def backward(ctx, gradOut):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Return the gradient for x, produced by scattering
        # gradOut into zeros of the input shape at the saved indices, and None
        # for the indices argument.
        indices = ctx.saved_tensors
        return (
            torch.ops.flex_lib.zeros_and_scatter(
                ctx.input_shape,
                indices,
                gradOut,
            ),
            None,
        )

    @classmethod
    @torch._export.wrappers.allow_in_pre_dispatch_graph
    def apply(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Run the normal autograd Function apply, but marked so
        # it is allowed in a pre-dispatch graph.
        return super().apply(*args, **kwargs)


mod_index = ModIndex.apply


class TransformGetItemToIndex(TorchFunctionMode):
    # [CLASS DOC]
    # Torch function mode that rewrites tensor item access whose index leaves
    # are all tensors or ints into mod_index calls (ints become 0-d int64
    # tensors on the indexed tensor's device). Stateless. All other functions
    # are passed through unchanged.
    # This is needed since we want to support calling
    # A[q_idx], where q_idx is a scalar tensor in score_mod.
    # Today, when q_idx is a scalar tensor, we implicitly convert it to a python
    # scalar and create a view. We do not want that behavior in this case, so we
    # use this torchfunctionmode to override that behavior for score_mod
    # wherever we're running it.
    #
    # We also convert integer indices to 0-D tensors so that temp[0] produces
    # the same backward graph as temp[0 * q_idx] (zeros_and_scatter with atomic_add).
    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        # [FUNCTION DOC] If func is the tensor item-access method, flatten the
        # index, and when every leaf is a tensor or int, return mod_index of
        # the tensor with integer leaves converted to tensors. Raises
        # AssertionError if the indexed object is not a tensor. Otherwise
        # calls func normally.
        if func is torch.Tensor.__getitem__:
            tensor_to_index = args[0]
            if not isinstance(tensor_to_index, torch.Tensor):
                raise AssertionError(
                    f"expected torch.Tensor, got {type(tensor_to_index)}"
                )
            index_args = pytree.tree_leaves(args[1])
            if all(isinstance(x, (torch.Tensor, int)) for x in index_args):
                converted_indices = [
                    torch.tensor(x, dtype=torch.int64, device=tensor_to_index.device)
                    if isinstance(x, int)
                    else x
                    for x in index_args
                ]
                return mod_index(tensor_to_index, converted_indices)
        return func(*args, **(kwargs or {}))


def trace_wrapped(*args: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC]
    # Public entry: invoke the trace_wrapped higher-order op with the given
    # args (keyword fn is the function to call) under no-grad mode and return
    # its result.
    with torch.no_grad():
        return _trace_wrapped_op(*args, **kwargs)


class TraceWrapped(HigherOrderOperator):
    # [CLASS DOC]
    # Higher-order operator named trace_wrapped. A single instance,
    # _trace_wrapped_op, is created at module level; its dispatch behavior is
    # registered by the py_impl decorators below.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register the operator under the name trace_wrapped.
        super().__init__("trace_wrapped")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Forward to the generic higher-order op dispatch.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(*args, **kwargs)


# TODO(jansel): need to ensure this does not get DCEed
_trace_wrapped_op = TraceWrapped()


def _assert_meta(
    grad: torch.Tensor,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Check that grad has exactly the expected size,
    # stride and dtype, raising AssertionError on any mismatch; returns grad.
    if grad.size() != size:
        raise AssertionError("size mismatch")
    if grad.stride() != stride:
        raise AssertionError("stride mismatch")
    if grad.dtype != dtype:
        raise AssertionError("dtype mismatch")
    return grad


@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(
    mode: ProxyTorchDispatchMode,
    *args: Any,
    bw_state: BackwardState | None = None,
    **kwargs: Any,
) -> Any:
    # [FUNCTION DOC]
    # Proxy-tracing implementation of trace_wrapped. Inserts one call_function
    # node (named trace_wrapped) that re-invokes the op with the extra kwargs
    # (including fn), passing the backward-state proxy if bw_state is given
    # (must be a BackwardState with a proxy, else AssertionError). Returns
    # empty_like copies of the gradient (args[1] if args[0] is None, as for
    # module backward hooks, else args[0]) tracked to the new proxy, without
    # running fn.
    def self_invoke(*args: Any, **dyn_kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Call the op under no-grad with the captured
        # kwargs; this is the function recorded in the graph.
        with torch.no_grad():
            return _trace_wrapped_op(*args, **dyn_kwargs, **kwargs)

    def unwrap_proxies(x: Any) -> Any:
        # [FUNCTION DOC] HELPER: Replace tensors by their tracer proxies,
        # recursing into lists/tuples, keeping None; any other type raises
        # AssertionError.
        if isinstance(x, torch.Tensor):
            return mode.tracer.unwrap_proxy(x)  # type: ignore[union-attr]
        if isinstance(x, (list, tuple)):
            return type(x)(map(unwrap_proxies, x))
        if x is None:
            return None
        raise AssertionError(f"unhandled type: {type(x)}")

    proxy_kwargs = {}
    if bw_state is not None:
        if not isinstance(bw_state, BackwardState):
            raise AssertionError(f"expected BackwardState, got {type(bw_state)}")
        if bw_state.proxy is None:
            raise AssertionError("bw_state.proxy must not be None")
        proxy_kwargs["bw_state"] = bw_state.proxy
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        self_invoke,
        unwrap_proxies(args),
        proxy_kwargs,
        name="trace_wrapped",
    )

    if args[0] is None:
        grad = args[1]  # module backward hooks
    else:
        grad = args[0]  # other backward hooks
    grad = tree_map_only(torch.Tensor, torch.empty_like, grad)
    track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return grad


@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC] Fake-tensor-mode implementation: always raises
    # RuntimeError since the op must not be reached under fake mode.
    raise RuntimeError("This op should never be invoked here")


@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args: Any, fn: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC] HELPER: Eager implementation: asserts no dispatch mode is
    # active (AssertionError otherwise) and returns the result of calling fn
    # with the args and remaining kwargs.
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return fn(*args, **kwargs)


_trace_wrapped_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_op, deferred_error=True)
)


@_trace_wrapped_op.py_functionalize_impl
def _trace_wrapped_functionalized(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC] HELPER: Functionalization rule: unwrap the tensor args,
    # redispatch the op below functionalization, and wrap the outputs.
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(_trace_wrapped_op(*unwrapped_args, **kwargs))


def autograd_function_backward_rewritten(original_backward: Any) -> Any:
    # [FUNCTION DOC]
    # Wrap an autograd Function backward so that all incoming gradients are made
    # contiguous before it is called. Returns the wrapper.
    def new_backward(ctx: Any, *grads: Any) -> Any:
        # [FUNCTION DOC] HELPER: Make each gradient contiguous and call the
        # original backward with ctx and them.
        # pyrefly: ignore [bad-assignment]
        grads = [g.contiguous() for g in grads]
        return original_backward(ctx, *grads)

    return new_backward
