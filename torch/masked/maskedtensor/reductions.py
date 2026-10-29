# mypy: allow-untyped-defs
# [MODULE DOC] MaskedTensor support for reductions (sum, mean, amin, amax, argmin, argmax,
# prod, all, norm, var, std). Builds on torch.masked reduction functions, producing a
# MaskedTensor result whose mask says whether any element contributed. Main contents:
# helpers for "all" and multi-dim "any", factories for full and per-dim reductions, and the
# NATIVE/TORCH/TENSOR_REDUCE_MAP tables keyed by aten op, torch function and Tensor method,
# plus _is_reduction / _apply_reduction. Exports nothing publicly.
# Import-time side effect: builds the three maps by resolving ops and attributes by name.
# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings

import torch

from .core import is_masked_tensor
from .creation import as_masked_tensor, masked_tensor


__all__ = []  # type: ignore[var-annotated]


def _masked_all_all(data, mask=None):
    # [FUNCTION DOC] HELPER: Logical-all over every element of data, treating masked-out
    # elements (mask False) as True. With no mask, plain all. Returns a 0-dim boolean tensor.
    if mask is None:
        return data.all()
    return data.masked_fill(~mask, True).all()


def _masked_all_dim(data, dim, keepdim=False, mask=None):
    # [FUNCTION DOC] HELPER: Logical-all along dim (optionally keeping the dimension), treating
    # masked-out elements as True when a mask is given. Returns a boolean tensor.
    if mask is None:
        return torch.all(data, dim=dim, keepdim=keepdim)
    return torch.all(data.masked_fill(~mask, True), dim=dim, keepdim=keepdim)


def _masked_all(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Stand-in for a missing torch.masked all. One positional argument
    # plus one keyword (the mask) selects the full reduction; anything else is forwarded to the
    # per-dimension variant.
    if len(args) == 1 and len(kwargs) == 1:
        return _masked_all_all(args[0], mask=kwargs["mask"])
    return _masked_all_dim(*args, **kwargs)


def _multidim_any(mask, dim, keepdim):
    # [FUNCTION DOC] HELPER: Logical-any of mask over one dim or a list of dims. An int is
    # converted to a one-element list; dims are reduced from highest to lowest so indices stay
    # valid when keepdim is false. Returns the reduced boolean mask.
    if isinstance(dim, int):
        return _multidim_any(mask, [dim], keepdim)
    for d in sorted(dim, reverse=True):
        mask = torch.any(mask, dim=d, keepdim=keepdim)
    return mask


def _get_masked_fn(fn):
    # [FUNCTION DOC] HELPER: Maps a reduction name to its implementation: the local all helper
    # for the name all, otherwise the same-named function in torch.masked (AttributeError if absent).
    if fn == "all":
        return _masked_all
    return getattr(torch.masked, fn)


def _torch_reduce_all(fn):
    # [FUNCTION DOC] HELPER: Returns a function that reduces a whole MaskedTensor with the named
    # reduction.
    def reduce_all(self):
        # [FUNCTION DOC] HELPER: Full reduction of a MaskedTensor (dense or sparse COO/CSR). Uses
        # the data and mask (values of the mask when sparse). Dense: calls the masked reduction
        # directly. Sparse: reduces the values; argmin/argmax on COO computes a flat index from
        # the indices and strides. Returns a 0-dim MaskedTensor whose mask is True if any
        # element was unmasked.
        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask().values() if self.is_sparse else self.get_mask()
        # When reduction is "all", then torch.argmin/torch.argmax needs to return the index of the
        # element corresponding to the min/max, but this operation isn't supported correctly for sparse layouts.
        # Therefore, this implementation calculates it using the strides.
        if fn == "all":
            result_data = masked_fn(data, mask=mask)

        elif fn in {"argmin", "argmax"} and self.is_sparse_coo():
            sparse_idx = masked_fn(data.values(), mask=mask).to(dtype=torch.int)
            indices = (
                data.to_sparse_coo().indices()
                if not self.is_sparse_coo()
                else data.indices()
            )
            idx = indices.unbind(1)[sparse_idx]
            stride = data.size().numel() / torch.tensor(
                data.size(), device=data.device
            ).cumprod(0)
            result_data = torch.sum(idx * stride)

        # we simply pass in the values for sparse COO/CSR tensors
        elif self.is_sparse:
            result_data = masked_fn(masked_tensor(data.values(), mask))

        else:
            result_data = masked_fn(self, mask=mask)

        return as_masked_tensor(result_data, torch.any(mask))

    return reduce_all


def _torch_reduce_dim(fn):
    # [FUNCTION DOC] HELPER: Returns a function that reduces a MaskedTensor along dim(s) with the
    # named reduction.
    def reduce_dim(self, dim, keepdim=False, dtype=None):
        # [FUNCTION DOC] HELPER: Per-dimension reduction. For sparse inputs emits a warning and
        # returns NotImplemented. Raises TypeError if self is not a MaskedTensor. Otherwise applies
        # the masked reduction (the all helper gets raw data and mask) and returns a MaskedTensor
        # whose mask is the any-reduction of the input mask over the same dims.
        if self.is_sparse:
            msg = (
                f"The sparse version of {fn} is not implemented in reductions.\n"
                "If you would like this operator to be supported, please file an issue for a feature request at "
                "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
                "In the case that the semantics for the operator are not trivial, it would be appreciated "
                "to also include a proposal for the semantics."
            )
            warnings.warn(msg, stacklevel=2)
            return NotImplemented
        if not is_masked_tensor(self):
            raise TypeError("Input to reduce_dim must be a MaskedTensor")

        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask()
        if fn == "all":
            result_data = masked_fn(data, dim=dim, keepdim=keepdim, mask=mask)
        else:
            result_data = masked_fn(
                self, dim=dim, keepdim=keepdim, dtype=dtype, mask=self.get_mask()
            )
        return as_masked_tensor(result_data, _multidim_any(mask, dim, keepdim))

    return reduce_dim


def _torch_reduce(fn):
    # [FUNCTION DOC] HELPER: Builds the aten-level handler for the named reduction.
    def reduce_fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: One positional argument and no keywords means a full reduction;
        # otherwise a per-dimension reduction with the given arguments.
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        return _torch_reduce_dim(fn)(*args, **kwargs)

    return reduce_fn


def _reduce_dim_args(input, dim, keepdim=False, dtype=None):
    # [FUNCTION DOC] HELPER: Normalizes positional/keyword arguments of a per-dim reduction into
    # a fixed tuple (input, dim, keepdim, dtype).
    return input, dim, keepdim, dtype


def _torch_grad_reduce(fn):
    # [FUNCTION DOC] HELPER: Builds the handler for torch-level functions and Tensor methods
    # (the autograd-facing entry points) for the named reduction.
    def grad_reduce(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Full reduction when called with just the tensor; otherwise
        # normalizes arguments to (input, dim, keepdim, dtype) and does a per-dim reduction.
        if len(args) == 1 and len(kwargs) == 0:
            return _torch_reduce_all(fn)(args[0])
        # TODO: autograd.Function doesn't support kwarg
        input, dim, keepdim, dtype = _reduce_dim_args(*args, **kwargs)
        return _torch_reduce_dim(fn)(input, dim, keepdim, dtype)

    return grad_reduce


REDUCE_NAMES = [
    "sum",
    "mean",
    "amin",
    "amax",
    "argmin",
    "argmax",
    "prod",
    "all",
    "norm",
    "var",
    "std",
]

NATIVE_REDUCE_MAP = {
    getattr(torch.ops.aten, name): _torch_reduce(name) for name in REDUCE_NAMES
}
TORCH_REDUCE_MAP = {
    getattr(torch, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}
TENSOR_REDUCE_MAP = {
    getattr(torch.Tensor, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}

NATIVE_REDUCE_FNS = list(NATIVE_REDUCE_MAP.keys())
TORCH_REDUCE_FNS = list(TORCH_REDUCE_MAP.keys())
TENSOR_REDUCE_FNS = list(TENSOR_REDUCE_MAP.keys())


def _is_reduction(fn):
    # [FUNCTION DOC] HELPER: True if fn is a registered reduction in any of the three maps.
    return fn in NATIVE_REDUCE_MAP or fn in TORCH_REDUCE_MAP or fn in TENSOR_REDUCE_MAP


def _apply_reduction(fn, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Calls the handler registered for fn (aten, torch, then Tensor map);
    # returns NotImplemented if fn is not registered.
    if fn in NATIVE_REDUCE_MAP:
        return NATIVE_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TORCH_REDUCE_MAP:
        return TORCH_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TENSOR_REDUCE_MAP:
        return TENSOR_REDUCE_MAP[fn](*args, **kwargs)
    return NotImplemented
