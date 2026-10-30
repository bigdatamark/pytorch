# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Unit tests for the APoT quantized linear layer (LinearAPoT): its forward output must
# equal that of an ordinary float Linear layer whose weight is set to the layer's quantized weight,
# for bit width 8 and base bitwidths 1 and 2.

import torch
from torch.ao.quantization.experimental.linear import LinearAPoT
from torch.nn.modules.linear import Linear
import unittest

class TestNonUniformObserver(unittest.TestCase):
    # [CLASS DOC] Test case (named like the observer tests, but exercising LinearAPoT) with two
    # tests comparing the APoT linear forward against a float linear using the dequantized APoT
    # weight. No shared state.
    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (4,4) and k=1
    """
    def test_linear_APoT_k1(self):
        # [FUNCTION DOC] Builds a 4 by 4 random weight scaled to 1000 and an integer-valued float
        # activation, applies LinearAPoT with 8 bits and k equal to 1, then builds a bias-free
        # float Linear whose weight is replaced by the layer's quantized weight cast to float.
        # Asserts the two outputs are exactly equal.
        # weight: fp tensor
        weight = 1000 * torch.rand(4, 4)

        # activation: fp32 tensor with ~ integer values
        activation = torch.randint(low=0, high=255, size=(4, 4), dtype=torch.float)

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 1)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))

    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (5,3), (3, 5) and k=2
    """
    def test_linear_APoT_k2(self):
        # [FUNCTION DOC] Same comparison as the k equal to 1 test but with a 5 by 3 weight and
        # activation and k equal to 2. The reference Linear is constructed 4 by 4 but its weight
        # parameter is then replaced, so the shape mismatch is harmless. Asserts exact equality of
        # outputs.
        # weight: fp tensor
        weight = 1000 * torch.rand(5, 3)

        # activation: fp32 tensor with ~ integer values
        # note: transpose of activation matrix will have dimension (3, 5)
        activation = torch.randint(low=0, high=255, size=(5, 3), dtype=torch.float)

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 2)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))

if __name__ == '__main__':
    unittest.main()
