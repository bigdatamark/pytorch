# mypy: allow-untyped-defs
# [MODULE DOC] Defines the quantized Dropout placeholder module. It exports Dropout, a pass-through
# module that lets models containing dropout run on quantized tensors. No import-time side effects.
import torch


__all__ = ["Dropout"]


class Dropout(torch.nn.Dropout):
    # [CLASS DOC] Quantized counterpart of torch.nn.Dropout, subclassing it and keeping p and inplace.
    # It does no dropping: inputs are returned unchanged in both train and eval mode. Conversion
    # helpers from_float and from_reference build a new instance from the float/reference module.
    r"""This is the quantized equivalent of :class:`~torch.nn.Dropout`.
        And this is a placeholder to enable models where fp32 tensors
        had dropout to work with quantized tensors in train and eval mode.

    Args:
        p: probability of an element to be zeroed
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    def forward(self, input):
        # [FUNCTION DOC] Returns the input tensor unchanged (no dropout applied, no copy, no mutation).
        return input

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedDropout" used in module repr.
        return "QuantizedDropout"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: builds a Dropout with the p and inplace values of the float module mod.
        # The use_precomputed_fake_quant argument is ignored.
        return cls(mod.p, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Class method: builds a Dropout with p and inplace copied from the reference module
        # mod. The scale and zero_point arguments are ignored (dropout has no quantization parameters).
        return cls(mod.p, mod.inplace)
