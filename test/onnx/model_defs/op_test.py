# Owner(s): ["module: onnx"]
# [MODULE DOC] Tiny nn.Module classes, each exercising a single operator pattern for ONNX
# tests: batch norm/pooling, concatenation, permutation, PReLU and fake quantization.

import torch
import torch.nn as nn


class DummyNet(nn.Module):
    # [CLASS DOC] Feature stack of leaky ReLU, 3-channel batch norm and average pooling.
    def __init__(self, num_classes=1000):
        # [FUNCTION DOC] Builds features; num_classes is accepted but unused.
        super().__init__()
        self.features = nn.Sequential(
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(3),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )

    def forward(self, x):
        # [FUNCTION DOC] Applies features, flattens to a column, then removes dimension one,
        # giving a 1-D result.
        output = self.features(x)
        return output.view(-1, 1).squeeze(1)


class ConcatNet(nn.Module):
    # [CLASS DOC] Stateless module that concatenates a sequence of tensors
    # along dimension one.
    def forward(self, inputs):
        # [FUNCTION DOC] Returns the concatenation of the input tensors along dimension one.
        return torch.cat(inputs, 1)


class PermuteNet(nn.Module):
    # [CLASS DOC] Stateless module that reorders dimensions of a 4-D input.
    def forward(self, input):
        # [FUNCTION DOC] Returns the input permuted so dimensions 2, 3, 0, 1 come first
        # in that order.
        return input.permute(2, 3, 0, 1)


class PReluNet(nn.Module):
    # [CLASS DOC] Module wrapping a PReLU activation with 3 learnable slopes
    # in a Sequential.
    def __init__(self) -> None:
        # [FUNCTION DOC] Builds features containing a single PReLU with 3 parameters.
        super().__init__()
        self.features = nn.Sequential(
            nn.PReLU(3),
        )

    def forward(self, x):
        # [FUNCTION DOC] Returns the PReLU stack applied to the input.
        output = self.features(x)
        return output


class FakeQuantNet(nn.Module):
    # [CLASS DOC] Module wrapping a default FakeQuantize with its observer disabled.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the fake quantizer and calls disable_observer so
        # statistics are frozen.
        super().__init__()
        self.fake_quant = torch.ao.quantization.FakeQuantize()
        self.fake_quant.disable_observer()

    def forward(self, x):
        # [FUNCTION DOC] Returns the fake-quantized version of the input.
        output = self.fake_quant(x)
        return output
