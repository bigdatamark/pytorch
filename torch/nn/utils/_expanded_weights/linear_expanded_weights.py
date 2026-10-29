# mypy: allow-untyped-defs
# [MODULE DOC] Per-sample-gradient autograd function for linear with ExpandedWeight weight/bias.
# Registers LinearPerSampleGrad for the linear function at import time.

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    is_batch_first,
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    # [CLASS DOC] Autograd function for linear layers with expanded weights; backward computes
    # the input gradient with a matmul and per-sample weight/bias gradients with einsum.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, _, __, *expanded_args_and_kwargs):
        # [FUNCTION DOC] Runs linear on the original weights. The first two arguments are unused
        # placeholders; the rest are input, weight and optional bias. Raises RuntimeError if the
        # input has rank below two. Saves args, bias and the batch_first flag on ctx. Returns
        # the output.
        if len(expanded_args_and_kwargs[0].shape) <= 1:
            raise RuntimeError(
                "Input does not have a batch dimension. Expanded Weights expected input "
                f"of at least rank 2, got of rank {len(expanded_args_and_kwargs[0].shape)}"
            )
        expanded_kwargs = {
            "bias": expanded_args_and_kwargs[2]
            if len(expanded_args_and_kwargs) == 3
            else None
        }
        expanded_args = expanded_args_and_kwargs[:2]
        ctx.batch_first = is_batch_first(expanded_args_and_kwargs)
        output = forward_helper(F.linear, expanded_args, expanded_kwargs)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Returns the gradient tuple (None for placeholders, matmul-based input
        # gradient when the input requires grad, None for weight and bias). Side effect: sets
        # per-sample gradients for weight and bias, transposing batch and sequence dimensions
        # first if batch_first is false.
        input, weight = ctx.args
        bias = ctx.kwargs["bias"]
        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg_names
        results.append(None)  # for op reference

        if input.requires_grad:
            results.append(grad_output.matmul(unpack_expanded_weight_or_tensor(weight)))
        else:
            results.append(None)
        results.extend([None] * 2)  # weight and bias don't compute batched gradients

        if not ctx.batch_first:
            grad_output = grad_output.transpose(0, 1)
            input = input.transpose(0, 1)

        # weight and bias get their grad_sample fields set directly if they exist
        set_grad_sample_if_exists(
            weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input)
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("n...k->nk", grad_output)
        )
        return tuple(results)
