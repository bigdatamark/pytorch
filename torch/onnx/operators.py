# [MODULE DOC] Backward-compatibility aliases for operators that help ONNX export.
# Exposes shape_as_tensor and reshape_from_tensor_shape, both bound directly to the
# internal ATen-backed torch functions. The public export list is empty. At import time
# it only imports torch and creates the two aliases.
"""This file provides a location for operators that help exporting models via onnx.

E.g. `shape_as_tensor` and `reshape_from_tensor_shape`
are to make all dynamic sizes operations traceable.

NOTE: at one point these functions were implemented differently.
Since then we have implemented these directly in ATen, so this
file is kept purely for backward-compatibility.
"""

from __future__ import annotations


__all__: list[str] = []

import torch


"""Get the shape of a tensor as a tensor.

Args:
    x (Tensor): The input tensor.

Returns:
    Tensor: A tensor of shape [len(x.shape)] containing the size of each dimension of x.

Example:
    >>> x = torch.randn(2, 3)
    >>> shape_as_tensor(x)
    tensor([2, 3])

"""
shape_as_tensor = torch._shape_as_tensor

"""Reshape a tensor to the given shape.

This function is used to make dynamic size operations traceable when exporting models via ONNX.
This function is kept for backward-compatibility. It is implemented directly in ATen.

Parameters:
    x (Tensor): the tensor to be reshaped.
    shape (Tensor): the target shape.

Returns:
    Tensor: the reshaped tensor.
"""
reshape_from_tensor_shape = torch._reshape_from_tensor
