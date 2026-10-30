# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Unit test for the integer representation of an APoT quantized tensor, using the
# observer to compute qparams and the APoT quantize function.

import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT

class TestQuantizedTensor(unittest.TestCase):
    # [CLASS DOC] Test case holding one test of the integer representation method of an APoT
    # quantized tensor; no state.
    r""" Tests int_repr on APoTQuantizer with random tensor2quantize
    and hard-coded values
    """
    def test_int_repr(self):
        # [FUNCTION DOC] Observes a fixed six-element float tensor with 4 bits and base bitwidth 2,
        # computes unsigned qparams, quantizes the tensor with APoT using those parameters, takes
        # the integer representation as int32 and asserts it equals the hard-coded level indices of
        # the nearest quantization levels.
        # generate tensor with random fp values
        tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])

        observer = APoTObserver(b=4, k=2)

        observer.forward(tensor2quantize)

        qparams = observer.calculate_qparams(signed=False)

        # get apot quantized tensor result
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                alpha=qparams[0],
                                gamma=qparams[1],
                                quantization_levels=qparams[2],
                                level_indices=qparams[3])

        qtensor_data = qtensor.int_repr().int()

        # expected qtensor values calculated based on
        # corresponding level_indices to nearest quantization level
        # for each fp value in tensor2quantize
        # e.g.
        # 0.0215 in tensor2quantize nearest 0.0208 in quantization_levels -> 3 in level_indices
        expected_qtensor_data = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.int32)

        self.assertTrue(torch.equal(qtensor_data, expected_qtensor_data))

if __name__ == '__main__':
    unittest.main()
