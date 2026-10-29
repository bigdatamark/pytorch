# [MODULE DOC] Autograd function used by APoT fake quantization (quantize then dequantize with a clipping-range gradient mask).
import torch
from torch import Tensor
from torch.ao.quantization.experimental.quantizer import dequantize_APoT, quantize_APoT


class fake_quantize_function(torch.autograd.Function):
    # [CLASS DOC] Custom torch autograd Function: forward performs APoT quantize followed by dequantize and saves a mask of elements inside
    # [-alpha, alpha]; backward is meant to pass gradients only through those elements (straight-through estimator).
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        alpha: Tensor,
        gamma: Tensor,
        quantization_levels: Tensor,
        level_indices: Tensor,
    ) -> Tensor:
        # [FUNCTION DOC] Quantize x with quantize_APoT using alpha, gamma, quantization_levels and level_indices, build a mask of entries within
        # [-alpha, alpha] via an element-wise apply on a detached copy, dequantize the result, save the mask for backward and return the dequantized tensor.
        # KNOWN ISSUE: the element-wise apply_ is only supported for CPU tensors and is slow.
        quantized_result = quantize_APoT(
            x, alpha, gamma, quantization_levels, level_indices
        )

        # calculate mask tensor
        mask = x.detach().apply_(lambda x: (x <= alpha and x >= -alpha))

        result = dequantize_APoT(quantized_result)

        ctx.save_for_backward(mask)

        return result

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> Tensor:
        # [FUNCTION DOC] Return grad_output multiplied by the saved mask.
        # KNOWN ISSUE: saved_tensors is a tuple (not a tensor) and only one gradient is returned although forward takes five inputs; this will fail or mis-handle gradients.
        mask = ctx.saved_tensors  # type: ignore[attr-defined]
        return grad_output * mask
