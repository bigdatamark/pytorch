# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# [MODULE DOC]
# Decompositions used only for forward-mode AD (jvp) when an op has no forward
# formula. Holds a jvp-specific registry (decomposition_table_for_jvp), a few
# temporary decompositions (trace, log_sigmoid_forward, layer/batch norm
# backward that recompute mean/variance so they track gradients), and helpers
# that torchscript decompositions and register their graphs with the JIT
# decomposition registry.
# Import-time side effects: registers decompositions into torch._decomp tables
# (silently skipped if already registered) and registers JIT graphs for ten
# aten backward/forward ops at the bottom of the file.
import inspect
from collections.abc import Callable

import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper


decomposition_table = torch._decomp.decomposition_table
decomposition_table_for_jvp: dict[torch._ops.OperatorBase, Callable] = {}
register_decomposition = torch._decomp.register_decomposition
aten = torch.ops.aten

# NOTE: [forward-mode AD decompositions mechanism]
#
# The mechanism is in VariableType,
#   IF any inputs have forward grad
#      AND there is no forward AD formula implemented
#      AND the functions are actually differentiable
#   run the decomposition
#      See run_jit_decomposition_with_args_for_jvp
#      We currently use python decompositions that we torchscript.
#
# Note that we would be building the backward graph at the decomposed level
# too, but that is OK, because we would've errored out otherwise anyway.
#
# TODO: The mechanism we are using to register decompositions doesn't
# seem to be exclusively used for jvp. So open question here is whether
# torch/csrc/jit/runtime/decomposition_registry.cpp is being used for other things.
# If that is the case, we may go down the decomposition path unexpectedly
# (and possibly produce an unintelligible error) vs erroring out earlier and
# printing that the forward AD formula is not implemented.
#
# The solution to this may be to have an explicit whitelist to control when
# to enable the decomposition.


def maybe_register_decomposition(op):
    # [FUNCTION DOC]
    # Returns a decorator that registers the decorated function as the
    # decomposition for op in the global table. Any error during registration
    # (for example a duplicate registration) is swallowed. Args: op, the
    # operator. Returns: decorator that always returns the original function.
    def decorator(f):
        # [FUNCTION DOC] HELPER: Try to register f for op; on any exception
        # return f unchanged.
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f

    return decorator


# Functions where we need a special decomposition for jvp but there's another version that
# should be used more generally (ex. for jvp we need to recompute the mean and variance for
# the backwards of a normalization function. Without jvp, it should use the saved value)
decomposition_table_for_jvp = {}


def register_decomposition_for_jvp(fn):
    # [FUNCTION DOC]
    # Returns a decorator that registers a decomposition for operator fn into
    # the jvp-only registry decomposition_table_for_jvp (not the global table).
    return register_decomposition(fn, registry=decomposition_table_for_jvp)


def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    # [FUNCTION DOC] HELPER: Look up the decomposition of operator decomp
    # (jvp table first, then the global table), strip any out-wrapper, build a
    # TorchScript graph for it, and register that graph with the JIT
    # decomposition registry. With use_python, the function is marked ignored
    # by the JIT and wrapped by a generated scripted function that calls it,
    # so the Python body is not compiled. Raises RuntimeError if no
    # decomposition is found. Side effect: mutates the JIT registry.
    if decomp in decomposition_table_for_jvp:
        decomposition_table_used = decomposition_table_for_jvp
    elif decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        raise RuntimeError(f"could not find decomposition for {decomp}")
    decomp_fn = decomposition_table_used[decomp]

    # `out_wrapper` extends a decompositions signature with
    # an `out` parameter. However jit will use the unwrapped function's
    # signature instead so we need to unwrap here to prevent an error
    decomp_fn = _maybe_remove_out_wrapper(decomp_fn)

    if use_python:
        decomp_fn = torch.jit.ignore(decomp_fn)
        sig = inspect.signature(decomp_fn)

        # Create a string wrapping the function from the signature
        # example output:
        # def wrapped_decomp(x: torch.Tensor, y: int, z: int):
        #   return decomp_fn(x, y, z)
        # Thanks copilot!
        def get_function_def(sig):
            # [FUNCTION DOC] HELPER: Build source text of a function named
            # wrapped_decomp whose parameters mirror the signature sig and
            # which returns the result of calling decomp_fn with them.
            param_def = [f"{param_str}" for param_str in sig.parameters.values()]
            param_use = [f"{param_str}" for param_str in sig.parameters]

            return f"def wrapped_decomp({', '.join(param_def)}):\n  return decomp_fn({', '.join(param_use)})\n"

        f_str = get_function_def(sig)
        graph = torch.jit.CompilationUnit(f_str).wrapped_decomp.graph
    else:
        graph = torch.jit.script(decomp_fn).graph
    torch.jit._register_decomposition(decomp, graph)


# The only decompositions here are temporary or hacks for the purposes of jvp


# TODO: do these also belong here?
@maybe_register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    # [FUNCTION DOC]
    # Decomposition of aten.trace: sum of the main diagonal of a matrix.
    return torch.sum(torch.diag(self))


@maybe_register_decomposition(aten.log_sigmoid_forward.default)
def log_sigmoid_forward(self: Tensor) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC]
    # Decomposition of aten.log_sigmoid_forward. Returns (output, buffer):
    # output is min(0, x) - log1p(exp(-|x|)); buffer is exp(-|x|) on CPU-like
    # devices, or an empty tensor on CUDA/XPU (matching those kernels).
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda or self.is_xpu:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


def recompute_mean_var(
    input: Tensor, rstd: Tensor, inner_dim_indices: list[int], keepdim: bool
):
    # [FUNCTION DOC] HELPER: Recompute mean and reciprocal std of input over
    # inner_dim_indices (population variance, keepdim as given) so they stay
    # differentiable w.r.t. input. The saved rstd is used only to back out the
    # epsilon (detached), which is then reused in the recomputed rsqrt.
    # Returns (mean, rstd).
    # for most norm decompositions, it will be the same as the core version except for here.
    # We recompute the mean and variance so that they track gradients through input

    var, mean = torch.var_mean(
        input, dim=inner_dim_indices, correction=0, keepdim=keepdim
    )
    eps = torch.pow(1 / rstd, 2) - var  # this makes me so sad inside
    eps = eps.detach()
    rstd = torch.rsqrt(var + eps)
    return mean, rstd


@register_decomposition_for_jvp(aten.native_layer_norm_backward)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: list[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    output_mask: list[bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    # [FUNCTION DOC]
    # jvp-specific decomposition of native_layer_norm_backward. Recomputes
    # mean/rstd from input (ignoring the saved mean) so results track input
    # gradients. Returns (grad_input, grad_weight, grad_bias). If the outer or
    # inner element count is zero, returns zero tensors. Outputs masked off by
    # output_mask are zero tensors instead of None (vjp cannot handle None).
    input_shape = input.shape
    input_ndim = input.dim()

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(axis))

    N = 1
    for i in inner_dims:
        N *= i
    M = 1
    for i in outer_dims:
        M *= i
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape),
            input.new_zeros(input_shape[axis:]),
            input.new_zeros(input_shape[axis:]),
        )

    mean_, rstd_ = recompute_mean_var(input, rstd, inner_dim_indices, keepdim=True)

    x_hat = (input - mean_) * rstd_
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)
    inner = a - b - c3

    if output_mask[0]:
        d_input: Tensor | None = (rstd_ / N) * inner
    else:
        d_input = torch.zeros_like(input)  # should be None but doesn't work with vjp

    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Tensor | None = torch.sum(
                grad_out * x_hat, outer_dim_indices, False
            )
        else:
            d_weight = grad_out * x_hat
    elif weight is not None:
        d_weight = torch.zeros_like(weight)  # should be None but doesn't work with vjp
    else:
        d_weight = torch.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            d_bias: Tensor | None = torch.sum(grad_out, outer_dim_indices, False)
        else:
            d_bias = grad_out.clone()
    elif bias is not None:
        d_bias = torch.zeros_like(bias)  # should be None but doesn't work with vjp
    else:
        d_bias = torch.zeros(())  # should be None but doesn't work with vjp

    return (d_input, d_weight, d_bias)


def prod(x: list[int]):
    # [FUNCTION DOC] HELPER: Product of all integers in x (1 for empty).
    r = 1
    for i in x:
        r *= i
    return r


@register_decomposition_for_jvp(aten.native_batch_norm_backward)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_invstd: Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    # [FUNCTION DOC]
    # jvp-specific decomposition of native_batch_norm_backward (channel axis
    # 1). In training mode, mean and invstd are recomputed from input so they
    # track gradients; in eval mode running statistics are used. Returns
    # (grad_input, grad_weight, grad_bias), with zero tensors in place of
    # masked-out outputs. Raises AssertionError if input rank < 2 or required
    # statistics are missing.
    input_shape = input.shape
    input_rank = input.dim()
    if input_rank < 2:
        raise AssertionError(f"rank of the input must be at least 2, got {input_rank}")

    axis = 1
    num_features = prod(input_shape) / input_shape[axis]  # type: ignore[arg-type]
    mean = save_mean
    invstd = save_invstd
    if train:
        if save_mean is None or save_invstd is None:
            raise AssertionError(
                "when train=True, save_mean and save_invstd are required"
            )

        reduciton_dims = [0] + list(range(2, input.dim()))
        if invstd is None:
            raise AssertionError("invstd must not be None for typing")
        mean, invstd = recompute_mean_var(input, invstd, reduciton_dims, keepdim=False)
    else:
        if running_mean is None or running_var is None:
            raise AssertionError(
                "running_mean and running_var must not be None when train=False"
            )
        mean = running_mean
        invstd = torch.rsqrt(running_var + eps)

    if invstd is None or mean is None:
        raise AssertionError(
            f"invstd and mean must not be None, got invstd={invstd}, mean={mean}"
        )

    broadcast_mask = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes: list[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = torch.reshape(mean, broadcast_mask)
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out, reduction_axes)
    dot_p = torch.sum(grad_out * (input - mean), reduction_axes)

    grad_mean = torch.reshape(grad_output_sum * norm, broadcast_mask)
    proj_scale = torch.reshape(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)

    if weight is None:
        grad_scale = torch.reshape(invstd, broadcast_mask) * 1.0
    else:
        grad_scale = torch.reshape(invstd * weight, broadcast_mask)

    if train:
        proj = (input - mean) * proj_scale
        grad_input = ((grad_out - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    elif weight is not None:
        grad_weight = torch.zeros_like(
            weight
        )  # should be None but doesn't work with vjp
    else:
        grad_weight = torch.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = torch.zeros_like(
            grad_output_sum
        )  # should be None but doesn't work with vjp

    return (grad_input, grad_weight, grad_bias)


@register_decomposition_for_jvp(aten.batch_norm_backward)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    save_mean: Tensor | None,
    save_var: Tensor | None,
    update: bool,
    eps: float,
    output_mask: list[bool],
    reserve: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    # [FUNCTION DOC]
    # jvp decomposition of batch_norm_backward: forwards to
    # native_batch_norm_backward, using update as the train flag and dropping
    # the reserve tensor.
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        update,
        eps,
        output_mask,
    )


_register_jit_decomposition_for_jvp(torch.ops.aten.trace.default, use_python=True)
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.nll_loss2d_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten._log_softmax_backward_data.default)
_register_jit_decomposition_for_jvp(torch.ops.aten._softmax_backward_data.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.log_sigmoid_forward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.native_layer_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.native_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.cudnn_batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.batch_norm_backward.default)
_register_jit_decomposition_for_jvp(torch.ops.aten.miopen_batch_norm_backward.default)
