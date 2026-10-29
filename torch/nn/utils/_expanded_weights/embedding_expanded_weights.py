# [MODULE DOC] Per-sample-gradient autograd function for embedding on ExpandedWeight weights.
# Registers EmbeddingPerSampleGrad as the handler for the embedding function at import time.
from typing import Any

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
)


@implements_per_sample_grads(F.embedding)
class EmbeddingPerSampleGrad(torch.autograd.Function):
    # [CLASS DOC] Autograd function for embedding whose weight is an ExpandedWeight. forward
    # saves input, weight and embedding options on ctx; backward fills the per-sample gradient
    # of the weight.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any, kwarg_names: list[str], _: Any, *expanded_args_and_kwargs: Any
    ) -> torch.Tensor:
        # [FUNCTION DOC] Computes the embedding lookup. Args: kwarg_names, an unused op
        # placeholder, then flattened arguments. Raises RuntimeError if the index input is
        # one-dimensional (a batch is required). Saves input, weight, padding_idx,
        # scale_grad_by_freq and sparse on ctx; returns the output.
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        if len(expanded_args[0].shape) == 1:
            raise RuntimeError(
                f"Expanded Weights needs an input with a batch size, got a 1D tensor, {expanded_args[0]}"
            )
        output = forward_helper(F.embedding, expanded_args, expanded_kwargs)
        ctx.input, ctx.weight = expanded_args
        ctx.padding_idx, ctx.scale_grad_by_freq = (
            expanded_kwargs["padding_idx"],
            expanded_kwargs["scale_grad_by_freq"],
        )
        ctx.sparse = expanded_kwargs["sparse"]
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # [FUNCTION DOC] Returns a tuple of gradients: None for the kwarg names and op
        # placeholders, the input gradient from the aten embedding backward when the input
        # requires grad (else None), and None for the remaining arguments. Side effect: sets the
        # per-sample gradient on the expanded weight.
        input, weight = ctx.input, ctx.weight
        padding_idx, scale_grad_by_freq, sparse = (
            ctx.padding_idx,
            ctx.scale_grad_by_freq,
            ctx.sparse,
        )

        def weight_per_sample_grad(weight: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds the per-sample gradient of a weight: for each batch
            # element scatter-adds the output gradient rows into a zero tensor shaped (batch,
            # weight shape) at the looked-up indices. Returns that tensor.
            batch_size = input.shape[0]
            embedding_dim = weight.shape[1]
            index = (
                input.unsqueeze(-1)
                .expand(*input.shape, embedding_dim)
                .reshape(batch_size, -1, embedding_dim)
            )
            grad_sample = torch.zeros(  # type: ignore[attr-defined]
                batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype
            )
            return grad_sample.scatter_add_(
                1, index, grad_output.reshape(batch_size, -1, embedding_dim)
            )

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference

        if input.requires_grad:
            bw_fn = torch.ops.aten.embedding_backward
            results.append(
                bw_fn(
                    grad_output,
                    input,
                    weight.shape[0],
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )
            )
        else:
            results.append(None)

        # weight doesn't compute batched gradients; no other arguments are differentiable (2 not saved from forward)
        results = results + [None] * 6

        # set grad_sample field for weight with per sample gradients
        set_grad_sample_if_exists(weight, weight_per_sample_grad)
        return tuple(results)
