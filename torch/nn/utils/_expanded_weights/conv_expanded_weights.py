# [MODULE DOC] Per-sample-gradient autograd function for conv1d/conv2d/conv3d on ExpandedWeight
# parameters. Registers ConvPerSampleGrad as the handler for the three functions at import time.
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.nn.functional as F


_P = ParamSpec("_P")
_R = TypeVar("_R")

from .conv_utils import (
    conv_args_and_kwargs,
    conv_backward,
    conv_input_for_string_padding,
    conv_picker,
)
from .expanded_weights_impl import ExpandedWeight, implements_per_sample_grads
from .expanded_weights_utils import forward_helper


@implements_per_sample_grads(F.conv1d)
@implements_per_sample_grads(F.conv2d)
@implements_per_sample_grads(F.conv3d)
class ConvPerSampleGrad(torch.autograd.Function):
    # [CLASS DOC] Autograd function computing a convolution with ExpandedWeight weight/bias and
    # storing per-sample gradients in the grad_sample field of the original weight. Registered
    # for conv1d, conv2d and conv3d. forward saves conv settings on ctx; backward delegates to
    # conv_backward.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        kwarg_names: list[str],
        conv_fn: Callable[_P, _R],
        *expanded_args_and_kwargs: Any,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Runs the convolution via forward_helper. Args: kwarg_names (names of
        # trailing keyword values), conv_fn (the functional conv), then the flattened
        # positional/keyword arguments. If padding is a string it is applied manually first and
        # padding is then set to zero. Returns the output. Raises RuntimeError for unbatched
        # input. Side effect: saves shapes, stride, padding, dilation, groups, input, weight and
        # bias on ctx.
        # KNOWN ISSUE: the two parts of the error message are joined without a separating space.
        expanded_args, expanded_kwargs = conv_args_and_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        orig_input = expanded_args[0]
        was_same_padding = expanded_kwargs["padding"] == "same"

        if isinstance(expanded_kwargs["padding"], str):
            # if padding is a string, we'll do the necessary padding (slowly) using F.pad
            kernel_size = expanded_args[1].shape[2:]
            padding, dilation = expanded_kwargs["padding"], expanded_kwargs["dilation"]
            input = conv_input_for_string_padding(
                conv_fn, padding, expanded_args[0], dilation, kernel_size
            )
            expanded_args = (input, expanded_args[1])
            # since we've already done the padding, don't need any more
            expanded_kwargs["padding"] = 0

        output = forward_helper(conv_fn, expanded_args, expanded_kwargs)
        input, weight = expanded_args
        batched_dim_size = conv_picker(conv_fn, 3, 4, 5)
        if input.dim() != batched_dim_size:
            raise RuntimeError(
                f"Expanded Weights only support convolution with batched input, got {conv_fn} with an"
                f"unbatched input of dim {input.dim()}, expected input of dim {batched_dim_size}"
            )

        # pyrefly: ignore [invalid-type-var]
        ctx.conv_fn = conv_fn

        ctx.batch_size = orig_input.shape[0]
        ctx.input_required_grad = orig_input.requires_grad
        ctx.orig_input_shape = orig_input.shape
        ctx.was_same_padding = was_same_padding
        ctx.stride, ctx.padding = expanded_kwargs["stride"], expanded_kwargs["padding"]
        ctx.dilation, ctx.groups = (
            expanded_kwargs["dilation"],
            expanded_kwargs["groups"],
        )

        if isinstance(weight, ExpandedWeight):
            ctx.input = input
        ctx.weight = weight
        ctx.bias = expanded_kwargs["bias"]

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
    # [FUNCTION DOC] Computes gradients by calling conv_backward with the saved conv function,
    # ctx and the first incoming gradient; returns the tuple of gradients (with None for
    # non-differentiable args).
        return conv_backward(ctx.conv_fn, ctx, grad_outputs[0])
