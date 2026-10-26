# mypy: allow-untyped-defs

# [MODULE DOC]
# Defines the out_dtype higher order operator: run an allowed functional ATen op
# (linear, mm, conv, mul, div variants) and return its result cast to a requested
# dtype, without mandating the internal compute precision. Provides dense,
# proxy-tracing, fake-tensor and functionalization implementations.
# Import-time side effects: creates the singleton out_dtype and registers its
# autograd (deferred not-implemented error), proxy, fake and functionalize impls.
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    maybe_handle_decomp,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


# TODO to figure out a more generic approach
ALLOWABLE_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
]


class OutDtypeOperator(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator "out_dtype". Stateless; its call validates the
    # wrapped op and dispatches to the registered implementations.
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            argument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the operator under the name "out_dtype".
        super().__init__("out_dtype")

    def __call__(self, op, output_dtype, *args):
        # [FUNCTION DOC] Validates and dispatches an out_dtype call.
        # Args: op (an OpOverload), output_dtype (target dtype), args (op inputs).
        # Raises ValueError if op is not an OpOverload, is mutable, does not return
        # exactly one tensor, or is not in ALLOWABLE_OPS. Returns the result of
        # the dispatched implementation (tensor of output_dtype).
        # KNOWN ISSUE: the single-tensor-return error message lacks a separator
        # between its two concatenated sentences.
        if not isinstance(op, torch._ops.OpOverload):
            raise ValueError("out_dtype's first argument must be an OpOverload")
        if op._schema.is_mutable:
            raise ValueError(
                "out_dtype's first argument needs to be a functional operator"
            )
        if not (
            len(op._schema.returns) == 1
            and isinstance(op._schema.returns[0].type, torch.TensorType)
        ):
            raise ValueError(
                "out_dtype's can only apply to ops that return a single tensor"
                f"Instead got {[r.type for r in op._schema.returns]}"
            )

        if op not in ALLOWABLE_OPS:
            raise ValueError(
                f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
            )

        # pyrefly: ignore [missing-attribute]
        res = super().__call__(op, output_dtype, *args)

        return res


out_dtype = OutDtypeOperator()


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    # [FUNCTION DOC] Proxy-tracing helper: first lets maybe_handle_decomp try a
    # decomposition (returning its result if not NotImplemented). Otherwise computes
    # the output eagerly with proxy tracing disabled (op then cast to output_dtype,
    # without type promotion), records a call_function node named out_dtype in the
    # tracer's graph with unwrapped proxy args, and returns the output tracked to
    # that proxy. Side effect: adds a node to the graph being traced.
    # NB: Long-term we should put the decomposition logic into
    # ProxyTorchDispatchMode so that people do not need to call maybe_handle_decomp
    # in all HigherOrderOp proxy implementations.
    r = maybe_handle_decomp(proxy_mode, func_overload, (op, output_dtype, *args), {})
    if r is not NotImplemented:
        return r

    with disable_proxy_modes_tracing():
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        out = op(*args).to(dtype=output_dtype)

    node_args = (op, output_dtype, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(op: torch._ops.OpOverload, output_dtype: torch.dtype, *args):
    # [FUNCTION DOC] Concrete (CompositeExplicitAutograd) implementation. Uses the
    # fused int8 matmul kernel when the call is an int8xint8->int32 mm on
    # CUDA/XPU; otherwise uses the promote-compute-cast fallback. Returns a tensor.
    if is_int_mm(op, output_dtype, args):
        return torch._int_mm(*args)
    return out_dtype_fallback(op, output_dtype, *args)


def is_int_mm(op, output_dtype, args):
    # [FUNCTION DOC] HELPER: Returns True only if op is aten.mm, output_dtype is
    # int32, exactly two args are given, both are int8 tensors, and both live on
    # CUDA or XPU devices (so the int8 matmul kernel applies).
    return (
        op is torch.ops.aten.mm.default
        and output_dtype == torch.int32
        and len(args) == 2
        and args[0].dtype == torch.int8
        and args[1].dtype == torch.int8
        and (args[0].is_cuda or args[0].is_xpu)
        and (args[1].is_cuda or args[1].is_xpu)
    )


def out_dtype_fallback(op, output_dtype, *args):
    # [FUNCTION DOC] Generic implementation: computes the promoted dtype from all
    # input leaves plus a one-element tensor of output_dtype using default
    # elementwise type promotion, casts all tensor args to it, runs op, and casts
    # the result to output_dtype. Returns the resulting tensor.
    flat_inputs = pytree.arg_tree_leaves(*args) + [torch.ones(1, dtype=output_dtype)]
    promote_dtype: torch.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    casted_args = pytree.tree_map_only(
        torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    res = op(*casted_args).to(dtype=output_dtype)
    return res


out_dtype.py_autograd_impl(autograd_not_implemented(out_dtype, deferred_error=True))


@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    mode: ProxyTorchDispatchMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    # [FUNCTION DOC] HELPER: ProxyTorchDispatchMode implementation; delegates to
    # trace_out_dtype with the out_dtype operator.
    return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)


@register_fake(out_dtype, skip_cache=True)
def out_dtype_fake_tensor_mode(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    # [FUNCTION DOC] HELPER: Fake tensor implementation (not cached); reuses the
    # dense implementation, which operates on fake tensors.
    return out_dtype_dense(op, output_dtype, *args)


@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    # [FUNCTION DOC] HELPER: Functionalization implementation. Unwraps functional
    # tensors in args, redispatches out_dtype to the next key, and wraps the result.
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)

    with ctx.redispatch_to_next():
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)
