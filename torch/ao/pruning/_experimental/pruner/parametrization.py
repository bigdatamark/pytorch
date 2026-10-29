# mypy: allow-untyped-defs
# [MODULE DOC] Parametrization helpers for structured pruning: module_contains_param,
# the FakeStructuredSparsity row-mask parametrization, and BiasHook, a forward hook
# that re-adds a (possibly masked) bias. No import-time side effects.
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized


def module_contains_param(module, parametrization):
    # [FUNCTION DOC] Returns True if module is parametrized and any parametrization attached
    # to any of its tensors is an instance of the given parametrization class; else False.
    if is_parametrized(module):
        # see if any of the module tensors have a parametrization attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for param_list in module.parametrizations.values()
        )
    return False


# Structured Pruning Parameterizations
class FakeStructuredSparsity(nn.Module):
    # [CLASS DOC] Parametrization module holding a row-wise boolean mask as buffer mask.
    # Its forward multiplies a tensor by the mask broadcast along dimension 0, and its
    # state_dict is empty so masks are not saved twice.
    r"""
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """

    def __init__(self, mask):
        # [FUNCTION DOC] Registers mask (a tensor with one entry per row) as buffer mask.
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        # [FUNCTION DOC] Returns x multiplied by the mask reshaped to broadcast along
        # dimension 0. Raises AssertionError if the mask is not a tensor or its length
        # differs from the first dimension of x.
        if not isinstance(self.mask, torch.Tensor):
            raise AssertionError("mask must be a torch.Tensor")
        if self.mask.shape[0] != x.shape[0]:
            raise AssertionError(
                f"mask shape[0] ({self.mask.shape[0]}) must match x shape[0] ({x.shape[0]})"
            )
        shape = [1] * len(x.shape)
        shape[0] = -1
        return self.mask.reshape(shape) * x

    def state_dict(self, *args, **kwargs):
        # [FUNCTION DOC] Returns an empty mapping so the mask is not saved twice.
        # avoid double saving masks
        return {}


class BiasHook:
    # [CLASS DOC] Forward hook object that adds the module's stored _bias to the output.
    # State: param (the row-mask parametrization) and prune_bias (whether to zero the
    # bias of masked rows).
    def __init__(self, parametrization, prune_bias):
        # [FUNCTION DOC] Stores the parametrization and the prune_bias flag.
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(self, module, input, output):
        # [FUNCTION DOC] Forward-hook body. If module has a non-None _bias, optionally zeroes
        # (in place on the stored data) the entries for masked rows, reshapes the bias to
        # broadcast over dimension 1 of output, adds it to output in place, and returns
        # output. Without a _bias, returns output unchanged.
        if getattr(module, "_bias", None) is not None:
            bias = module._bias.data
            if self.prune_bias:
                bias[~self.param.mask] = 0

            # reshape bias to broadcast over output dimensions
            idx = [1] * len(output.shape)
            idx[1] = -1
            bias = bias.reshape(idx)

            output += bias
        return output
