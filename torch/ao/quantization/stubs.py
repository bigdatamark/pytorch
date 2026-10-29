# [MODULE DOC] Quantization stub modules used in eager-mode quantization.
# Defines QuantStub and DeQuantStub (identity modules marking where tensors
# enter and leave the quantized region; replaced by real quantize/dequantize
# modules at convert time) and QuantWrapper (wraps a module between the two).
# Exports all three via __all__. No import-time side effects.
from typing import Any

import torch
from torch import nn
from torch.ao.quantization import QConfig


__all__ = ["QuantStub", "DeQuantStub", "QuantWrapper"]


class QuantStub(nn.Module):
    # [CLASS DOC] Identity placeholder marking where a float tensor becomes quantized.
    # Acts as an observer target during calibration and is swapped for the
    # quantized Quantize module in convert. Optionally stores a qconfig attribute.
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig: QConfig | None = None):
        # [FUNCTION DOC] Initialize the stub; sets the qconfig attribute only if a truthy qconfig is given,
        # otherwise the qconfig is later inherited from parent modules.
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Return the input tensor unchanged (identity).
        return x


class DeQuantStub(nn.Module):
    # [CLASS DOC] Identity placeholder marking where a quantized tensor becomes float.
    # Swapped for the quantized DeQuantize module in convert. Optionally stores a qconfig attribute.
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """

    def __init__(self, qconfig: Any | None = None):
        # [FUNCTION DOC] Initialize the stub; sets the qconfig attribute only if a truthy qconfig is given.
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Return the input tensor unchanged (identity).
        return x


class QuantWrapper(nn.Module):
    # [CLASS DOC] Wrapper that surrounds a module with a QuantStub on its input and a DeQuantStub on its output.
    # Holds submodules quant, dequant and module; the stubs take the wrapped module's qconfig (if any).
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """

    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

    def __init__(self, module: nn.Module):
        # [FUNCTION DOC] Build the wrapper: read the optional qconfig from the given module, register the
        # quant stub, dequant stub and the module itself as submodules, and copy the module's training mode.
        super().__init__()
        qconfig = getattr(module, "qconfig", None)
        self.add_module("quant", QuantStub(qconfig))
        self.add_module("dequant", DeQuantStub(qconfig))
        self.add_module("module", module)
        self.train(module.training)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Apply quant stub, then the wrapped module, then the dequant stub; returns the result.
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)
