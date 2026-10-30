# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Unit tests for the APoT fake quantization module and its autograd function:
# qparams agreement with the observer, forward equal to quantize then dequantize, failure
# when no qparams exist, and gradient checking of the backward pass.

import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.experimental.fake_quantize_function import fake_quantize_function
forward_helper = fake_quantize_function.forward
backward = fake_quantize_function.backward
from torch.autograd import gradcheck

class TestFakeQuantize(unittest.TestCase):
    # [CLASS DOC] Test case for the APoT fake quantize module (b=4, k=2); each test builds its
    # own observers and fake quantizers, no shared state.
    r""" Tests fake quantize calculate_qparams() method
         by comparing with result from observer calculate_qparams.
         Uses hard-coded values: alpha=1.0, b=4, k=2.
    """
    def test_fake_calc_qparams(self):
        # [FUNCTION DOC] Sets min 0 and max 1 on the fake quantizer's observer and on a standalone
        # observer, computes unsigned qparams from both, and asserts alpha, gamma, quantization
        # levels and level indices match.
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.activation_post_process.min_val = torch.tensor([0.0])
        apot_fake.activation_post_process.max_val = torch.tensor([1.0])

        alpha, gamma, quantization_levels, level_indices = apot_fake.calculate_qparams(signed=False)

        observer = APoTObserver(b=4, k=2)
        observer.min_val = torch.tensor([0.0])
        observer.max_val = torch.tensor([1.0])

        qparams_expected = observer.calculate_qparams(signed=False)

        self.assertEqual(alpha, qparams_expected[0])
        self.assertTrue(torch.equal(gamma, qparams_expected[1]))
        self.assertTrue(torch.equal(quantization_levels, qparams_expected[2]))
        self.assertTrue(torch.equal(level_indices, qparams_expected[3]))

    r""" Tests fake quantize forward() method
         by comparing result with expected
         quant_dequant_APoT mapping of input tensor.
         Uses input tensor with random values from 0 -> 1000
         and APoT observer with hard-coded values b=4, k=2
    """
    def test_forward(self):
        # [FUNCTION DOC] Observes a random tensor (values up to 1000) to get unsigned qparams, runs
        # an observing, fake-quant-enabled fake quantizer on a clone, and asserts the output is
        # exactly equal to APoT quantization followed by dequantization of the input.
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        observer = APoTObserver(b=4, k=2)
        observer.forward(X)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.enable_observer()
        apot_fake.enable_fake_quant()

        X_reduced_precision_fp = apot_fake.forward(torch.clone(X), False)

        # get X_expected by converting fp -> apot -> fp to simulate quantize -> dequantize
        X_to_apot = quantize_APoT(X, alpha, gamma, quantization_levels, level_indices)
        X_expected = dequantize_APoT(X_to_apot)

        self.assertTrue(torch.equal(X_reduced_precision_fp, X_expected))

    r""" Tests fake quantize forward() method
         throws error when qparams are None
    """
    def test_forward_exception(self):
        # [FUNCTION DOC] With the observer disabled the qparams stay unset, so calling forward on a
        # random tensor must raise; asserts some exception is raised.
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        apot_fake = APoTFakeQuantize(b=4, k=2)
        # disable observer so qparams not set, qparams are all None
        apot_fake.disable_observer()
        apot_fake.enable_fake_quant()

        with self.assertRaises(Exception):
            apot_fake.forward(torch.clone(X), False)

    r""" Tests fake quantize helper backward() method
         using torch.autograd.gradcheck function.
    """
    def test_backward(self):
        # [FUNCTION DOC] Gets unsigned qparams from an observer fed double precision random input
        # and runs gradient checking on the fake quantize function at absolute tolerance 1e-4.
        input = torch.randn(20, dtype=torch.double, requires_grad=True)

        observer = APoTObserver(b=4, k=2)
        observer(input)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False)

        gradcheck(fake_quantize_function.apply, (input, alpha, gamma, quantization_levels, level_indices), atol=1e-4)

if __name__ == '__main__':
    unittest.main()
