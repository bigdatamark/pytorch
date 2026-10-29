# [MODULE DOC] Defines TensorAPoT, a thin container pairing an APoTQuantizer with the quantized tensor data it produced.
import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer


# class to store APoT quantized tensor
class TensorAPoT:
    # [CLASS DOC] Holds an APoT quantized tensor: the quantizer used (attribute quantizer) and the raw quantized data
    # tensor (attribute data). No validation or computation is done; int_repr exposes the stored data.
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer: APoTQuantizer, apot_data: torch.Tensor):
        # [FUNCTION DOC] Store the given quantizer and the quantized data tensor on the instance.
        self.quantizer = quantizer
        self.data = apot_data

    def int_repr(self) -> torch.Tensor:
        # [FUNCTION DOC] Return the stored quantized data tensor itself (no copy or conversion).
        return self.data
