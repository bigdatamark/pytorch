# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] HELPER: Debug-formatting utilities that turn tensors in nested arguments into short readable strings.

import torch
from torch.fx.node import Argument


def friendly_debug_info(v: object) -> Argument:
    # [FUNCTION DOC] HELPER: Returns a summary string (shape, requires_grad, dtype) for a tensor, or the string form of any other object.
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad}, dtype={v.dtype})"
    else:
        return str(v)


def map_debug_info(a: Argument) -> Argument:
    # [FUNCTION DOC] HELPER: Maps friendly_debug_info over every leaf of a nested list/tuple/dict argument and returns the same structure of strings.
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
