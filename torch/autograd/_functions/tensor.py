# mypy: allow-untyped-defs
# [MODULE DOC] Legacy autograd Function implementations: Type (dtype/tensor-type conversion, deprecated
# since 2.1 in favor of the tensor to method) and Resize (reshape preserving element count).
# Import-time side effects: none beyond imports.
import operator
from functools import reduce
from typing_extensions import deprecated

import torch
import torch._utils
from torch.autograd.function import Function


class Type(Function):
    # [CLASS DOC] Deprecated autograd Function converting a tensor to another tensor type.
    # Backward converts the gradient back to the input's original type (on the original device).
    @staticmethod
    @deprecated(
        "`torch.autograd._functions.Type` is deprecated as of PyTorch 2.1, "
        "please use `torch.tensor.to(dtype=dtype)` instead.",
        category=FutureWarning,
    )
    # pyrefly: ignore [bad-override]
    def forward(ctx, i, dest_type):
        # [FUNCTION DOC] Save the input's Python type and device index (-1 for non-CUDA) on ctx and
        # return the input converted to dest_type. Emits a FutureWarning (deprecated).
        ctx.input_type = type(i)
        ctx.input_device = -1 if not i.is_cuda else i.get_device()
        return i.type(dest_type)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Convert the incoming gradient back to the saved input type, switching to the saved
        # device context when the input was on a CUDA device. Returns (grad, None) for the two forward inputs.
        if ctx.input_device == -1:
            return grad_output.type(ctx.input_type), None
        else:
            with torch.accelerator.device_index(ctx.input_device):
                return grad_output.type(ctx.input_type), None


# TODO: deprecate this
class Resize(Function):
    # [CLASS DOC] Legacy autograd Function that reshapes a tensor to given sizes without changing element count.
    # Marked TODO for deprecation in the source.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, tensor, sizes):
        # [FUNCTION DOC] Reshape tensor to sizes, saving sizes, element count and the input size on ctx.
        # Raises RuntimeError if the element count differs. Quantized tensors get a self-copy then a
        # contiguous view; contiguous tensors are copied via new() before viewing; others use a contiguous view.
        ctx.sizes = sizes
        ctx.numel = reduce(operator.mul, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                (
                    "requested resize to {} ({} elements in total), "
                    "but the given tensor has a size of {} ({} elements). "
                    "autograd's resize can only change the shape of a given "
                    "tensor, while preserving the number of elements. "
                ).format(
                    "x".join(map(str, sizes)),
                    ctx.numel,
                    "x".join(map(str, tensor.size())),
                    tensor.numel(),
                )
            )
        ctx.input_sizes = tensor.size()
        if tensor.is_quantized:
            tensor.copy_(tensor)
            return tensor.contiguous().view(*sizes)
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            return result
        else:
            return tensor.contiguous().view(*sizes)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Reshape the gradient back to the saved input size; raises AssertionError if its
        # element count does not match the forward element count. Returns (grad, None).
        if grad_output.numel() != ctx.numel:
            raise AssertionError(
                f"Expected grad_output to have {ctx.numel} elements, but got {grad_output.numel()}"
            )
        return grad_output.contiguous().view(ctx.input_sizes), None
