# mypy: allow-untyped-defs
# [MODULE DOC] Defines WeightNormSparsifier, which zeroes blocks of a weight with the
# smallest norm and/or the smallest values inside each block. Exports WeightNormSparsifier.
# No import-time side effects.
import operator
from collections.abc import Callable
from functools import reduce

import torch
import torch.nn.functional as F

from .base_sparsifier import BaseSparsifier


__all__ = ["WeightNormSparsifier"]


def _flat_idx_to_2d(idx, shape):
    # [FUNCTION DOC] HELPER: Converts flat indices into row and column indices for a 2D shape;
    # returns the pair (rows, cols). Currently unused in this module.
    rows = idx // shape[1]
    cols = idx % shape[1]
    return rows, cols


class WeightNormSparsifier(BaseSparsifier):
    # [CLASS DOC] Block-based sparsifier. Defaults: sparsity_level (fraction of blocks removed),
    # sparse_block_shape (block rows, cols), zeros_per_block (zeros inside each block,
    # all of them by default). The norm function (norm_fn) scores blocks: absolute value
    # for 1, square for 2, or a user callable. The final mask combines a tensor-level
    # mask (whole blocks) and a block-level mask (smallest values inside each block).
    r"""Weight-Norm Sparsifier

    This sparsifier computes the norm of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.

    Args:

        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block (see note below)
        zeros_per_block: Number of zeros in a sparse block
        norm: Norm to use. Could be either `int` or a callable.
            If `int`, only L1 and L2 are implemented.

    Note::
        The `sparse_block_shape` is tuple representing (block_ROWS, block_COLS),
        irrespective of what the rows / cols mean in the data tensor. That means,
        if you were to sparsify a weight tensor in the nn.Linear, which has a
        weight shape `(Cout, Cin)`, the `block_ROWS` would refer to the output
        channels, while the `block_COLS` would refer to the input channels.

    Note::
        All arguments to the WeightNormSparsifier constructor are "default"
        arguments and could be overridden by the configuration provided in the
        `prepare` step.
    """

    def __init__(
        self,
        sparsity_level: float = 0.5,
        sparse_block_shape: tuple[int, int] = (1, 4),
        zeros_per_block: int | None = None,
        norm: Callable | int | None = None,
    ):
        # [FUNCTION DOC] Builds the default config; zeros_per_block defaults to the number of
        # elements in a block. Selects norm_fn: None means 2, a callable is used as is, 1
        # is absolute value, 2 is elementwise square; any other value raises
        # NotImplementedError. Then initializes the base class.
        if zeros_per_block is None:
            zeros_per_block = reduce(operator.mul, sparse_block_shape)
        defaults = {
            "sparsity_level": sparsity_level,
            "sparse_block_shape": sparse_block_shape,
            "zeros_per_block": zeros_per_block,
        }
        if norm is None:
            norm = 2
        if callable(norm):
            self.norm_fn = norm
        elif norm == 1:
            self.norm_fn = lambda T: T.abs()
        elif norm == 2:
            self.norm_fn = lambda T: T * T
        else:
            raise NotImplementedError(f"L-{norm} is not yet implemented.")
        super().__init__(defaults=defaults)

    def _scatter_fold_block_mask(
        self,
        output_shape,
        dim,
        indices,
        block_shape,
        mask=None,
        input_shape=None,
        device=None,
    ):
        # [FUNCTION DOC] HELPER: Zeroes, in the unfolded mask, the entries at the given indices
        # along dim, then folds the mask back to output_shape using non-overlapping
        # blocks of block_shape, replacing the mask data. If mask is None, builds an
        # all-ones mask of input_shape on device (AssertionError if input_shape is also
        # missing). Returns the mask.
        r"""Creates patches of size `block_shape` after scattering the indices."""
        if mask is None:
            if input_shape is None:
                raise AssertionError("input_shape must be provided when mask is None")
            mask = torch.ones(input_shape, device=device)
        mask.scatter_(dim=dim, index=indices, value=0)
        mask.data = F.fold(
            mask, output_size=output_shape, kernel_size=block_shape, stride=block_shape
        )
        return mask

    def _make_tensor_mask(
        self, data, input_shape, sparsity_level, sparse_block_shape, mask=None
    ):
        # [FUNCTION DOC] HELPER: Builds a mask (padded to a multiple of the block shape, then
        # cropped back) that zeroes whole blocks with the lowest scores. Sparsity level at or
        # above one gives all zeros, at or below zero gives all ones. Otherwise block
        # scores are average-pooled data, the number of blocks to zero is the rounded
        # level times block count clamped to at most count minus one, and those blocks
        # are zeroed through the scatter-fold helper. Returns the mask.
        # KNOWN ISSUE: input_shape is unused.
        r"""Creates a tensor-level mask.

        Tensor-level mask is described as a mask, where the granularity of sparsification of the
        smallest patch is the sparse_block_shape. That means, that for a given mask and a
        sparse_block_shape, the smallest "patch" of zeros/ones could be the sparse_block_shape.

        In this context, `sparsity_level` describes the fraction of sparse patches.
        """
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w

        if mask is None:
            mask = torch.ones(h + dh, w + dw, device=data.device)

        if sparsity_level >= 1.0:
            mask.data = torch.zeros_like(mask)
            return mask
        elif sparsity_level <= 0.0:
            mask.data = torch.ones_like(mask)
            return mask

        values_per_block = reduce(operator.mul, sparse_block_shape)
        if values_per_block > 1:
            # Reduce the data
            data = F.avg_pool2d(
                data[None, None, :],
                kernel_size=sparse_block_shape,
                stride=sparse_block_shape,
                ceil_mode=True,
            )
        data = data.flatten()
        num_blocks = len(data)

        data = data.repeat(1, values_per_block, 1)

        threshold_idx = round(sparsity_level * num_blocks)
        threshold_idx = max(0, min(num_blocks - 1, threshold_idx))  # Sanity check
        _, sorted_idx = torch.topk(data, k=threshold_idx, dim=2, largest=False)

        # Temp reshape for mask
        mask_reshape = mask.reshape(data.shape)  # data might be reshaped
        self._scatter_fold_block_mask(
            dim=2,
            output_shape=(h + dh, w + dw),
            indices=sorted_idx,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )
        mask.data = mask_reshape.squeeze().reshape(mask.shape)[:h, :w].contiguous()
        return mask

    def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
        # [FUNCTION DOC] HELPER: Builds a mask that, inside each block, zeroes the
        # zeros_per_block smallest values. Data is padded with NaN to a multiple of the
        # block shape and unfolded into blocks; if zeros_per_block equals the block size the
        # mask is all zeros. The mask data is replaced in place and the mask is returned
        # (with padded size; the padding is not cropped).
        # KNOWN ISSUE: padded NaN entries take part in the smallest-value selection, and
        # the returned mask keeps the padded shape rather than the data shape.
        r"""Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        """
        h, w = data.shape[-2:]
        block_h, block_w = sparse_block_shape
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w
        values_per_block = reduce(operator.mul, sparse_block_shape)

        if mask is None:
            mask = torch.ones((h + dh, w + dw), device=data.device)

        if values_per_block == zeros_per_block:
            # Everything should be sparsified
            mask.data = torch.zeros_like(mask)
            return mask

        # create a new padded tensor like data (to match the block_shape)
        padded_data = torch.ones(h + dh, w + dw, dtype=data.dtype, device=data.device)
        padded_data.fill_(torch.nan)
        padded_data[:h, :w] = data
        unfolded_data = F.unfold(
            padded_data[None, None, :],
            kernel_size=sparse_block_shape,
            stride=sparse_block_shape,
        )

        # Temp reshape for mask
        mask_reshape = mask.reshape(unfolded_data.shape)
        _, sorted_idx = torch.topk(
            unfolded_data, k=zeros_per_block, dim=1, largest=False
        )

        self._scatter_fold_block_mask(
            dim=1,
            indices=sorted_idx,
            output_shape=padded_data.shape,
            block_shape=sparse_block_shape,
            mask=mask_reshape,
        )

        mask.data = mask_reshape.squeeze().reshape(mask.shape).contiguous()
        return mask

    def update_mask(  # type: ignore[call-override, override]
        self,
        module,
        tensor_name,
        sparsity_level,
        sparse_block_shape,
        zeros_per_block,
        **kwargs,
    ):
        # [FUNCTION DOC] Recomputes the mask of module.tensor_name in place. Raises ValueError if
        # zeros_per_block is negative or larger than the block size. A level of zero or
        # less, or zero zeros per block, gives all ones; a level of one or more with a
        # fully zeroed block gives all zeros. Otherwise scores the weight with norm_fn,
        # builds the tensor-level mask, and, if blocks are only partly zeroed, combines it
        # by logical or with the block-level mask.
        values_per_block = reduce(operator.mul, sparse_block_shape)
        if zeros_per_block > values_per_block:
            raise ValueError(
                "Number of zeros per block cannot be more than the total number of elements in that block."
            )
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        mask = getattr(module.parametrizations, tensor_name)[0].mask
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = torch.zeros_like(mask)
        else:
            ww = self.norm_fn(getattr(module, tensor_name))
            tensor_mask = self._make_tensor_mask(
                data=ww,
                input_shape=ww.shape,
                sparsity_level=sparsity_level,
                sparse_block_shape=sparse_block_shape,
            )
            if values_per_block != zeros_per_block:
                block_mask = self._make_block_mask(
                    data=ww,
                    sparse_block_shape=sparse_block_shape,
                    zeros_per_block=zeros_per_block,
                )
                tensor_mask = torch.logical_or(tensor_mask, block_mask)
            mask.data = tensor_mask
