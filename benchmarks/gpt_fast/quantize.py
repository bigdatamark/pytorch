# flake8: noqa: E266
# [MODULE DOC] Weight-only int8 per-channel quantization utilities for the dense gpt-fast model: a
# row-wise symmetric quantizer, a handler that builds an int8 state dict from linear layers, a
# module-swapping helper and an int8 linear layer with bfloat16 scales. No import-time side
# effects.
import torch
import torch.nn as nn
import torch.nn.functional as F


##### Quantization Primitives ######


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # [FUNCTION DOC] Symmetric per-row dynamic quantization of a 2-D tensor (axis 0 per channel).
    # Computes min/max along dim 1, takes the larger magnitude of the negated min and the max (both
    # bounded at zero), divides by half the quant range to get scales (clamped to the float32
    # epsilon, cast to the input dtype), and uses all-zero int64 zero points. Returns the rounded
    # and clamped values cast to target_dtype, the scales (one per row) and the zero points. No
    # side effects.
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


##### Weight-only int8 per-channel quantized code ######


def replace_linear_weight_only_int8_per_channel(module):
    # [FUNCTION DOC] Recursively replaces every linear child module in place with an uninitialized
    # int8 weight-only linear of the same sizes; non-linear children are recursed into. Mutates the
    # given module; returns nothing.
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(child.in_features, child.out_features),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    # [CLASS DOC] Quantization driver. Holds the model in the mod attribute; one method builds an
    # int8 state dict from its linear layers and another converts the model to int8 runtime
    # modules.
    def __init__(self, mod):
        # [FUNCTION DOC] Stores the model to be quantized in the mod attribute.
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        # [FUNCTION DOC] Builds a state dict where every linear layer's weight is replaced by its
        # int8 per-row quantization (computed from a float copy) and a matching scales entry is
        # added in the original weight dtype; both are moved to CPU. Runs without gradients. Starts
        # from the model's own state dict (a shallow dict of references), so the model is not
        # modified. Returns the dict.
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight.to("cpu")
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype).to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        # [FUNCTION DOC] Converts the held model in place to its int8 runtime form by replacing
        # linear layers, and returns the same model object.
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    # [CLASS DOC] Linear layer with int8 weights and per-output-channel bfloat16 scales, both kept
    # as buffers (not parameters). Weights are uninitialized until a state dict is loaded.
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Creates the int8 weight buffer (out_features by in_features,
        # uninitialized) and a bfloat16 scales buffer of ones.
        # KNOWN ISSUE: bias, device and dtype arguments are accepted but ignored; no bias is
        #   created.
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies a bias-free linear with the weight cast to the input dtype, then
        # multiplies the result by the per-channel scales (broadcast over the last dimension).
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
