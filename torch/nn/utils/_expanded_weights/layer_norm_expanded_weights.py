# mypy: allow-untyped-defs
# [MODULE DOC] Per-sample-gradient autograd function for layer_norm with ExpandedWeight
# weight/bias. Registers LayerNormPerSampleGrad for the layer norm function at import time.

import torch
import torch.nn.functional as F

from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
    sum_over_all_but_batch_and_last_n,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.layer_norm)
class LayerNormPerSampleGrad(torch.autograd.Function):
    # [CLASS DOC] Autograd function for layer norm with expanded weight/bias; backward uses the
    # native layer norm backward and fills per-sample gradients by reducing over non-parameter
    # dimensions.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        # [FUNCTION DOC] Computes layer norm with the native kernel. Raises RuntimeError if the
        # normalized shape would cover the batch dimension. Saves args, eps, mean, rstd, and
        # weight/bias when needed. Returns the output.
        # KNOWN ISSUE: the error message joins two parts without a separating space.
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        input = expanded_args[0]
        normalized_shape = expanded_args[1]
        if len(input.shape) <= len(normalized_shape):
            raise RuntimeError(
                "Expanded Weights: Layer norm should not normalize over batch dimension for per sample gradient"
                f"computations but got that normalized shape, {normalized_shape}, matched input shape."
            )
        output, mean, rstd = forward_helper(
            torch.native_layer_norm, expanded_args, expanded_kwargs
        )
        ctx.args = expanded_args

        if input.requires_grad or isinstance(expanded_kwargs["weight"], ExpandedWeight):
            ctx.weight = expanded_kwargs["weight"]
        if input.requires_grad or isinstance(expanded_kwargs["bias"], ExpandedWeight):
            ctx.bias = expanded_kwargs["bias"]
        ctx.eps = expanded_kwargs["eps"]
        ctx.mean, ctx.rstd = mean, rstd
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns the gradient tuple: the input gradient from the native layer
        # norm backward when the input requires grad, None elsewhere. Side effect: sets
        # per-sample gradients for weight and bias when they were saved.
        def weight_per_sample_grad(weight):
            # [FUNCTION DOC] HELPER: Per-sample weight gradient: the normalized input times the
            # output gradient, summed over all dimensions except batch and the weight's own
            # dimensions.
            return sum_over_all_but_batch_and_last_n(
                F.layer_norm(input, normalized_shape, eps=ctx.eps) * grad_output,
                weight.dim(),
            )

        input, normalized_shape = ctx.args
        mean, rstd = ctx.mean, ctx.rstd

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference
        if input.requires_grad:
            weight_ = unpack_expanded_weight_or_tensor(ctx.weight)
            bias_ = unpack_expanded_weight_or_tensor(ctx.bias)
            results.append(
                torch.ops.aten.native_layer_norm_backward(
                    grad_output,
                    input,
                    normalized_shape,
                    mean,
                    rstd,
                    weight_,
                    bias_,
                    (True, False, False),
                )[0]
            )
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * 4

        # set grad_sample field for weight and bias with per sample gradients
        if hasattr(ctx, "weight"):
            set_grad_sample_if_exists(ctx.weight, weight_per_sample_grad)
        if hasattr(ctx, "bias"):
            set_grad_sample_if_exists(
                ctx.bias,
                lambda bias: sum_over_all_but_batch_and_last_n(grad_output, bias.dim()),
            )
        return tuple(results)
