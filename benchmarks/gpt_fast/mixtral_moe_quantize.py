# flake8: noqa: E266
# [MODULE DOC] Weight-only int8 per-channel quantization utilities for the Mixtral MoE model of the
# gpt-fast benchmark. Provides a row-wise symmetric quantizer, a handler that builds an int8 state
# dict, and int8 replacements for linear layers and the conditional (expert) feed-forward. Imports
# ConditionalFeedForward from a sibling module named mixtral_moe_model (must be on path).
from mixtral_moe_model import ConditionalFeedForward

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
    # [FUNCTION DOC] Recursively swaps modules in place for their int8 counterparts. Every linear
    # child not named gate becomes an uninitialized int8 weight-only linear of the same sizes, and
    # every ConditionalFeedForward child becomes the int8 conditional feed-forward using the shape
    # of its first weight. Other children are recursed into. Mutates the given module; returns
    # nothing.
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name != "gate":
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(
                    child.in_features, child.out_features, target_dtype=torch.int8
                ),
            )
        elif isinstance(child, ConditionalFeedForward):
            num_experts, intermediate_size, dim = child.w1.shape
            setattr(
                module,
                name,
                ConditionalFeedForwardInt8(
                    num_experts, intermediate_size, dim, target_dtype=torch.int8
                ),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    # [CLASS DOC] Quantization driver for a Mixtral model. Holds the model in the mod attribute.
    # create_quantized_state_dict builds an int8 state dict; convert_for_runtime swaps modules for
    # their int8 variants so that the state dict can be loaded.
    def __init__(self, mod):
        # [FUNCTION DOC] Stores the model to be quantized in the mod attribute.
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        # [FUNCTION DOC] Builds a state dict in which linear weights (except ones whose name ends
        # with gate) and expert weights w1, w2, w3 are replaced by int8 tensors with matching
        # scales entries. Linear layers are quantized per output row from float copies; experts are
        # quantized one expert at a time and concatenated along a new leading expert dimension.
        # Runs without gradients. Starts from the model's own state dict (the returned dict is a
        # shallow copy of references). Returns the dict with the replaced entries; the model itself
        # is not modified.
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear) and not fqn.endswith(".gate"):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)
            elif isinstance(mod, ConditionalFeedForward):
                for weight_idx in range(3):
                    weight_name = f"w{weight_idx + 1}"
                    scales_name = f"scales{weight_idx + 1}"
                    weight = getattr(mod, weight_name)
                    num_experts, intermediate_size, dim = weight.shape

                    bit8_weight_list = []
                    scales_list = []
                    for expert_idx in range(num_experts):
                        bit8_weight, scales, _ = dynamically_quantize_per_channel(
                            weight[expert_idx].float(), -128, 127, torch.int8
                        )
                        bit8_weight_list.append(
                            bit8_weight.reshape(1, intermediate_size, dim)
                        )
                        scales_list.append(scales.reshape(1, intermediate_size))

                    cur_state_dict[f"{fqn}.{weight_name}"] = torch.cat(
                        bit8_weight_list, dim=0
                    )
                    cur_state_dict[f"{fqn}.{scales_name}"] = torch.cat(
                        scales_list, dim=0
                    )

        return cur_state_dict

    def convert_for_runtime(self):
        # [FUNCTION DOC] Converts the held model in place to its int8 runtime form via the module
        # replacement helper and returns the same model object.
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
        target_dtype=None,
    ) -> None:
        # [FUNCTION DOC] Creates the weight buffer (out_features by in_features, target_dtype) and
        # a bfloat16 scales buffer of ones. Raises AssertionError when target_dtype is None.
        # KNOWN ISSUE: bias, device and dtype arguments are accepted but ignored; no bias is
        #   created.
        if target_dtype is None:
            raise AssertionError("target_dtype must not be None")
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=target_dtype)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies a bias-free linear with the weight cast to the input dtype, then
        # multiplies the result by the per-channel scales (broadcast over the last dimension).
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales


class ConditionalFeedForwardInt8(nn.Module):
    # [CLASS DOC] Int8 version of the Mixtral expert feed-forward. Stores stacked expert weights
    # w1, w2, w3 in the target dtype plus bfloat16 per-row scales buffers, all uninitialized until
    # loaded.
    def __init__(self, num_experts, intermediate_size, dim, target_dtype):
        # [FUNCTION DOC] Allocates the three stacked weight buffers (num_experts leading dimension)
        # and three scales buffers, and records target_dtype.
        super().__init__()

        self.target_dtype = target_dtype

        self.register_buffer(
            "w1", torch.empty(num_experts, intermediate_size, dim, dtype=target_dtype)
        )
        self.register_buffer(
            "w2", torch.empty(num_experts, dim, intermediate_size, dtype=target_dtype)
        )
        self.register_buffer(
            "w3", torch.empty(num_experts, intermediate_size, dim, dtype=target_dtype)
        )

        self.register_buffer(
            "scales1", torch.empty(num_experts, intermediate_size, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "scales2", torch.empty(num_experts, dim, dtype=torch.bfloat16)
        )
        self.register_buffer(
            "scales3", torch.empty(num_experts, intermediate_size, dtype=torch.bfloat16)
        )

    def forward(self, x, expert_indices):
        # [FUNCTION DOC] Computes the expert outputs for the selected experts. Gathers weights by
        # expert index for every token, casts them to the activation dtype, and runs a gated MLP
        # via einsum (silu of the first projection times the third), applying each projection's
        # scales. Takes activations x per token and expert_indices per token (active experts).
        # Returns per-token, per-active-expert outputs without combining them.
        w1_weights = self.w1.to(x.dtype)[expert_indices]  # [T, A, D, D]
        w3_weights = self.w3.to(x.dtype)[expert_indices]  # [T, A, D, D]
        w2_weights = self.w2.to(x.dtype)[expert_indices]
        x1 = F.silu(
            torch.einsum("ti,taoi -> tao", x, w1_weights)
            * self.scales1[expert_indices].to(x.dtype)
        )
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights) * self.scales3[
            expert_indices
        ].to(x.dtype)
        expert_outs = torch.einsum(
            "tao, taio -> tai", (x1 * x3), w2_weights
        ) * self.scales2[expert_indices].to(x.dtype)  # [T, A, D, D]
        return expert_outs
