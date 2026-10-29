# [MODULE DOC] Public factory functions for MaskedTensor: masked_tensor (leaf constructor)
# and as_masked_tensor (differentiable constructor). Exports both via __all__.
# No import-time side effects beyond importing MaskedTensor from the core module.
# Copyright (c) Meta Platforms, Inc. and affiliates

from .core import MaskedTensor


__all__ = [
    "as_masked_tensor",
    "masked_tensor",
]


# These two factory functions are intended to mirror
#     torch.tensor - guaranteed to be a leaf node
#     torch.as_tensor - differentiable constructor that preserves the autograd history


def masked_tensor(
    data: object, mask: object, requires_grad: bool = False
) -> MaskedTensor:
    # [FUNCTION DOC] Build a MaskedTensor from data and a boolean mask, as a new leaf node
    # (like torch.tensor). requires_grad sets autograd tracking on the result. Returns the
    # MaskedTensor; validation errors (shape/dtype/layout mismatch) come from the constructor.
    return MaskedTensor(data, mask, requires_grad)


def as_masked_tensor(data: object, mask: object) -> MaskedTensor:
    # [FUNCTION DOC] Differentiable constructor (like torch.as_tensor): wraps data and mask
    # into a MaskedTensor via the internal _from_values path, preserving autograd history of
    # data. Returns the MaskedTensor.
    return MaskedTensor._from_values(data, mask)
