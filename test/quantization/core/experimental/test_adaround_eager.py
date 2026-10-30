# Owner(s): ["oncall: speech_infra"]
# [MODULE DOC] Tests for the Adaround weight-rounding optimizer on small linear-chain and
# conv-chain models: the adarounded model must have a lower output mean squared error
# versus the float model than plain round-to-nearest fake quantization of the weights.
# Running the file directly raises a RuntimeError because the test is not currently
# enabled in the test discovery list.

import copy

import torch
import torch.nn as nn
from torch.ao.quantization.experimental.adaround_optimization import (
    AdaptiveRoundingOptimizer,
)
from torch.nn import functional as F
from torch.quantization.observer import MinMaxObserver
from torch.testing._internal.common_quantization import QuantizationTestCase


def forward_wrapper(fetcher):
    # [FUNCTION DOC] HELPER: Factory for the forward-hook builder handed to the optimizer. The
    # returned hook appends the detached first input and the detached output of the hooked
    # module to the given fetcher list, so the optimizer can capture a layer's activations.
    def forward(module, input, output):
        # [FUNCTION DOC] HELPER: Forward hook that appends the detached first positional input and
        # the detached output of the module to the enclosing fetcher list; returns nothing.
        fetcher.append(input[0].detach())
        fetcher.append(output.detach())

    return forward


class TestAdaround(QuantizationTestCase):
    # [CLASS DOC] Quantization test case checking that the Adaround optimizer beats naive
    # per-tensor symmetric int8 fake quantization on MSE against float output. Holds helper
    # methods to run the optimizer, build the fake-quantized reference and build a wrapper.
    def feedforawrd_callback(
        self,
        model,
        data,
    ) -> None:
        # [FUNCTION DOC] HELPER: Default callback for the optimizer; simply runs the model on one
        # data item, discarding the result.
        model(data)

    def feedforawrd_callback_with_wrapper(self, model, data, wrapper) -> None:
        # [FUNCTION DOC] HELPER: Callback variant that runs the model on one data item through the
        # supplied wrapper module, discarding the result.
        wrapper(model, data)

    def run_adaround(self, model, img_data, wrapper=None):
        # [FUNCTION DOC] HELPER: Builds an AdaptiveRoundingOptimizer for the model and the list of
        # input batches (100 iterations, batch size 10), choosing the plain callback when no
        # wrapper is given and the wrapper callback otherwise, runs it and returns the resulting
        # adarounded model.
        adaround_optimizer = AdaptiveRoundingOptimizer(
            model,
            self.feedforawrd_callback
            if wrapper is None
            else self.feedforawrd_callback_with_wrapper,
            forward_wrapper,
            img_data,
            max_iter=100,
            batch_size=10,
            feed_forward_wrapper=wrapper,
        )
        adarounded_model = adaround_optimizer.run_adaround()
        return adarounded_model

    def get_fake_quant(self, model):
        # [FUNCTION DOC] HELPER: Deep-copies the model and, for every Linear or Conv2d module in
        # the copy, observes the weight with a min-max observer (int8 range, per-tensor symmetric),
        # fake-quantizes the weight with the computed scale and zero point and writes the result
        # back in place. Returns the copy; the input model is unchanged.
        hard_fake_quant_model = copy.deepcopy(model)
        for _, module in hard_fake_quant_model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                weight_observer = MinMaxObserver(
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                )
                weight_observer(module.weight)
                scale, zero_point = weight_observer.calculate_qparams()
                fake_quant_module = torch.fake_quantize_per_tensor_affine(
                    module.weight,
                    scale=scale,
                    zero_point=zero_point,
                    quant_min=-128,
                    quant_max=127,
                )
                module.weight.data.copy_(fake_quant_module)
        return hard_fake_quant_model

    def get_feed_forward_wrapper(self):
        # [FUNCTION DOC] HELPER: Returns a new wrapper module whose forward takes a model and a
        # sample and returns the model applied to the sample.
        class FeedForwardWrapper(nn.Module):
            # [CLASS DOC] HELPER: Trivial stateless module used as the feed forward wrapper; its
            # forward just calls the given model on the given sample.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
                super().__init__()

            def forward(self, model, sample):
                # [FUNCTION DOC] HELPER: Returns the result of calling the model on the sample.
                return model(sample)

        wrapper_module = FeedForwardWrapper()
        return wrapper_module

    def test_linear_chain(self):
        # [FUNCTION DOC] Builds a three-layer linear model, runs Adaround with calibration data of
        # fifty random batches and the wrapper callback, and compares MSE to float outputs on a
        # random input for adarounded versus fake-quantized weights. Asserts the adaround loss is
        # strictly smaller.
        class LinearChain(nn.Module):
            # [CLASS DOC] HELPER: Three stacked Linear layers (3 to 4 to 5 to 6 features) with no
            # activations, used as the model under test.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three Linear layers.
                super().__init__()
                self.linear1 = nn.Linear(3, 4)
                self.linear2 = nn.Linear(4, 5)
                self.linear3 = nn.Linear(5, 6)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the three Linear layers in order; returns the
                # final output.
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

        float_model = LinearChain()
        img_data = [torch.rand(10, 3, dtype=torch.float) for _ in range(50)]
        adarounded_model = self.run_adaround(
            float_model, img_data, self.get_feed_forward_wrapper()
        )
        fq_model = self.get_fake_quant(float_model)
        rand_input = torch.rand(10, 3)
        with torch.no_grad():
            ada_out = adarounded_model(rand_input)
            fq_out = fq_model(rand_input)
            float_out = float_model(rand_input)
            ada_loss = F.mse_loss(ada_out, float_out)
            fq_loss = F.mse_loss(fq_out, float_out)
            self.assertTrue(ada_loss.item() < fq_loss.item())

    def test_conv_chain(self):
        # [FUNCTION DOC] Same comparison as the linear test but with a three-layer Conv2d model and
        # no wrapper. Asserts adaround MSE is strictly smaller than fake-quant MSE.
        # KNOWN ISSUE: the optimizer only visits Conv1d and Linear layers, so for this Conv2d
        # model it leaves the weights unrounded and the comparison passes without exercising it.
        class ConvChain(nn.Module):
            # [CLASS DOC] HELPER: Three stacked Conv2d layers (3 to 4 to 5 to 6 channels, kernel
            # and stride 5) with no activations, used as the model under test.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three Conv2d layers.
                super().__init__()
                self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
                self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
                self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the three Conv2d layers in order; returns the
                # final output.
                x = self.conv2d1(x)
                x = self.conv2d2(x)
                x = self.conv2d3(x)
                return x

        float_model = ConvChain()
        img_data = [torch.rand(10, 3, 125, 125, dtype=torch.float) for _ in range(50)]
        adarounded_model = self.run_adaround(float_model, img_data)
        fq_model = self.get_fake_quant(float_model)
        rand_input = torch.rand(10, 3, 256, 256)
        with torch.no_grad():
            ada_out = adarounded_model(rand_input)
            fq_out = fq_model(rand_input)
            float_out = float_model(rand_input)
            ada_loss = F.mse_loss(ada_out, float_out)
            fq_loss = F.mse_loss(fq_out, float_out)
            self.assertTrue(ada_loss.item() < fq_loss.item())


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
