# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC]
# Defines MaskBuffer, a small reference-counted holder for a boolean mask tensor
# shared between mask-partial placements (used by embedding and gather ops).
# No import-time side effects.
from dataclasses import dataclass

import torch


@dataclass
class MaskBuffer:
    # [CLASS DOC] Dataclass holding an optional mask tensor (data) and a user
    # count (refcount). Invariant: data is set iff refcount > 0. Multiple users
    # may share one buffer as long as they materialize identical masks.
    # Methods: materialize_mask, release_mask, apply_mask.
    data: torch.Tensor | None = None
    # refcount allows shared usage of the MaskBuffer, as long as all users have the same data
    refcount: int = 0

    def materialize_mask(self, mask):
        # [FUNCTION DOC] Registers a user of the mask. On first use stores the
        # mask; later uses require an equal mask, otherwise raises RuntimeError.
        # Raises AssertionError if refcount is positive but data is missing.
        # Increments refcount.
        if self.refcount == 0:
            self.data = mask
        else:
            if self.data is None:
                raise AssertionError
            if not torch.equal(self.data, mask):
                raise RuntimeError(
                    "MaskBuffer has been materialized with conflicting data"
                )
        self.refcount += 1

    def release_mask(self):
        # [FUNCTION DOC] Drops one user; clears data when the count reaches zero.
        # Raises RuntimeError if the buffer was never materialized.
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.refcount -= 1
        if self.refcount == 0:
            self.data = None

    def apply_mask(self, tensor):
        # [FUNCTION DOC] Zeroes, in place, the entries of tensor selected by the
        # stored mask. If tensor has the same rank as the mask (gather case) the
        # mask indexes elements; otherwise (embedding case, extra trailing dim)
        # it selects whole rows along the leading dims. Mutates tensor.
        # Raises RuntimeError if the buffer has not been materialized.
        if self.refcount == 0 or self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0
