# mypy: allow-untyped-defs
# [MODULE DOC] APoT quantizer: class APoTQuantizer (quantize, dequantize, quant_dequant) and module-level wrappers quantize_APoT,
# dequantize_APoT and quant_dequant_APoT that build a quantizer from qparams. Uses element-wise apply on CPU tensors via helpers in apot_utils.
import numpy as np

import torch
from torch import Tensor
from torch.ao.quantization.experimental.apot_utils import (
    apot_to_float,
    float_to_apot,
    quant_dequant_util,
)


# class to store APoT quantizer and
# implement quantize and dequantize
class APoTQuantizer:
    # [CLASS DOC] Holds APoT qparams (alpha, gamma, quantization_levels, level_indices) and converts tensors: quantize returns a TensorAPoT,
    # dequantize converts a TensorAPoT back to float, quant_dequant rounds a float tensor to the nearest level. q_apot_alpha is unimplemented.
    alpha: torch.Tensor
    gamma: torch.Tensor
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
        level_indices: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] Store the four qparam tensors on the instance.
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices

    r""" Quantizes fp Tensor to integer APoT representation.
    Conversion is based on the qparams from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: APoT Tensor representation of tensor2quantize
    """

    def quantize(self, tensor2quantize: Tensor):
        # [FUNCTION DOC] Quantize a float tensor to integer APoT representation. Applies float_to_apot element-wise (with the levels, indices and alpha)
        # on a detached view, casts to int and wraps it with this quantizer in a TensorAPoT, which is returned.
        # KNOWN ISSUE: detach shares memory and the element-wise apply is in place, so the input tensor's values are overwritten; CPU only.
        result = torch.tensor([])

        # map float_to_apot over tensor2quantize elements
        tensor2quantize = tensor2quantize.detach().apply_(
            lambda x: float_to_apot(
                x, self.quantization_levels, self.level_indices, self.alpha
            )
        )

        # convert to APoT int representation for dtype
        tensor2quantize = tensor2quantize.int()

        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT

        result = TensorAPoT(self, tensor2quantize)  # type: ignore[assignment]

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: fp reduced precision representation of input Tensor
    """

    def dequantize(self, apot_tensor) -> Tensor:
        # [FUNCTION DOC] Convert a TensorAPoT back to floating point. Flattens its data, maps each integer through apot_to_float using this
        # quantizer's levels and indices (ValueError if an index is unknown), and returns a tensor reshaped to the original size.
        # Side effect: prints the flattened data.
        # KNOWN ISSUE: leftover debug print of the data on every call.
        orig_size = apot_tensor.data.size()
        apot_tensor_data = apot_tensor.data.flatten()

        print(apot_tensor_data)

        # map apot_to_float over tensor2quantize elements
        result_temp = np.empty(shape=apot_tensor_data.size())
        for i in range(len(apot_tensor_data)):
            new_ele = apot_to_float(
                apot_tensor_data[i], self.quantization_levels, self.level_indices
            )
            result_temp[i] = new_ele

        result = torch.from_numpy(result_temp).reshape(orig_size)

        return result

    r""" Returns result of quantize -> dequantize on a fp Tensor (reduced precision)
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        apot_tensor: quantized APoT Tensor to dequantize
    Returns:
        result: fp representation of input Tensor
    """

    def quant_dequant(self, tensor2quantize: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the input tensor with every element replaced by the nearest quantization level via quant_dequant_util, applied in place.
        # KNOWN ISSUE: quant_dequant_util needs three arguments (x, levels, indices) but is called with two, so this raises TypeError; also mutates the input.
        levels_lst = list(self.quantization_levels)

        result = tensor2quantize.apply_(lambda x: quant_dequant_util(x, levels_lst))  # type: ignore[call-arg]

        return result

    def q_apot_alpha(self) -> float:
        # [FUNCTION DOC] Placeholder; always raises NotImplementedError.
        raise NotImplementedError


r""" Global method to create quantizer and call quantizer quantize_APoT
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization levels: Tensor with fp quantization levels
        level indices: Tensor with integer quantization level indices
    Returns:
        result: APoT Tensor representation of tensor2quantize
"""


def quantize_APoT(
    tensor2quantize: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    quantization_levels: Tensor,
    level_indices: Tensor,
):
    # [FUNCTION DOC] Convenience wrapper: build an APoTQuantizer from the given qparams and return its quantize result (a TensorAPoT) for tensor2quantize.
    quantizer = APoTQuantizer(
        alpha=alpha,
        gamma=gamma,
        quantization_levels=quantization_levels,
        level_indices=level_indices,
    )
    result = quantizer.quantize(tensor2quantize)
    return result


r""" Global method to create quantizer and call quantizer dequantize_APoT
    Args:
        apot_tensor: APoT Tensor to dequantize
    Returns:
        result: fp Tensor dequantized from apot_tensor
"""


def dequantize_APoT(apot_tensor) -> Tensor:
    # [FUNCTION DOC] Dequantize a TensorAPoT using the quantizer stored on it; returns the float tensor.
    quantizer = apot_tensor.quantizer
    result = quantizer.dequantize(apot_tensor)
    return result


r""" Global method to create quantizer and call quantizer quant_dequant
    Args:
        tensor2quantize: fp Tensor to quantize
        alpha: Tensor qparam alpha (clipping level)
        gamma: Tensor qparam gamma (scale factor for quantization levels)
        quantization levels: Tensor with fp quantization levels
        level indices: Tensor with integer quantization level indices
    Returns:
        result: fp reduced precision Tensor from tensor2quantize
"""


def quant_dequant_APoT(
    tensor2quantize: Tensor,
    alpha: Tensor,
    gamma: Tensor,
    quantization_levels: Tensor,
    level_indices: Tensor,
) -> Tensor:
    # [FUNCTION DOC] Convenience wrapper: build an APoTQuantizer from the given qparams and return its quant_dequant result for tensor2quantize.
    quantizer = APoTQuantizer(
        alpha=alpha,
        gamma=gamma,
        quantization_levels=quantization_levels,
        level_indices=level_indices,
    )
    result = quantizer.quant_dequant(tensor2quantize)
    return result
