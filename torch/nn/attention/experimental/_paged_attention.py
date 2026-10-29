# mypy: allow-untyped-defs
# [MODULE DOC]
# Experimental paged attention support layered on flex attention. Exports the
# PagedAttention class, which manages a page table mapping per-batch logical
# kv blocks to physical pages of a shared kv cache, and converts block masks,
# mask_mods and score_mods from logical to physical kv indexing.
# Imports from flex_attention at import time; no other import-time side effects.
"""
This module implements Paged Attention on top of flex_attention.
This module is experimental and subject to change.
"""

import torch
from torch.nn.attention.flex_attention import (
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    BlockMask,
    noop_mask,
)


__all__ = ["PagedAttention"]


def _cdiv(x: int | float | torch.Tensor, multiple: int | float | torch.Tensor):
    # [FUNCTION DOC] HELPER: Ceiling division of x by multiple using integer floor
    # division; works elementwise for tensors. Returns the rounded-up quotient.
    return (x + multiple - 1) // multiple


class PagedAttention:
    # [CLASS DOC] Allocator and index translator for a paged kv cache used with
    # flex attention. State: n_pages and page_size; page_table (batch by page
    # slot, logical block to physical page, -1 when unallocated); capacity
    # (per-batch allocated token count); empty_pages (Python list used as a
    # stack of free physical pages); physical_to_logical (batch by physical page,
    # inverse of page_table, -1 when unmapped). Methods reserve/erase manage
    # pages, assign writes kv values into caches, and convert_logical_block_mask /
    # get_mask_mod / get_score_mod rewrite logical-index callables to physical ones.
    """
    PagedAttention supports flex attention inference with a large batch size.
    With PagedAttention, a batch of key/value tensors with varying kv length
    is split into tensor blocks of fixed length and cached in a compact way.
    Thus we can avoid redundant memory consumption due to varying kv length and
    support a larger batch size.
    """

    def __init__(
        self,
        n_pages: int,
        page_size: int,
        max_batch_size: int,
        device: str = "cuda",
    ) -> None:
        # [FUNCTION DOC] Creates the paging state for max_batch_size sequences
        # over n_pages physical pages of page_size tokens each, on the given
        # device. The page table and reverse map start filled with -1, capacities
        # at zero, and all pages free (popped from the end, lowest index first).
        # number of pages
        self.n_pages = n_pages

        # number of tokens per page
        self.page_size = page_size

        # page table: [batch, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones(
            (max_batch_size, self.n_pages), dtype=torch.int64, device=device
        )

        # capacity: batch_idx -> allocated sequence length
        self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)

        # index of empty pages that is available for allocation
        self.empty_pages = list(range(n_pages - 1, -1, -1))

        # mapping from physical page index to logical page index
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )

    def reserve(self, batch_idx: torch.Tensor, seq_len: torch.Tensor) -> None:
        # [FUNCTION DOC] Grows the allocation of one batch entry so its capacity
        # covers at least seq_len tokens. Does nothing if capacity already suffices.
        # Otherwise takes the needed number of pages from the free list, records
        # them in the page table and reverse map, and raises capacity by whole pages.
        # Raises an AssertionError if there are not enough free pages.
        # Mutates page_table, physical_to_logical, capacity and empty_pages.
        # KNOWN ISSUE: new pages are placed starting at capacity divided by page
        # size, which assumes capacity is page aligned (true as it only grows by pages).
        """
        Requests the capacity of a given batch to be at least enough to
        hold `seq_len` elements.

        Args:
            batch_idx (Tensor): batch index to be reserved; shape :math:`(1)`.
            seq_len (Tensor): minimum capacity for the given batch; shape :math:`(1)`.
        """

        if seq_len <= self.capacity[batch_idx]:
            return

        num_pages_to_allocate = _cdiv(
            seq_len - self.capacity[batch_idx], self.page_size
        )

        if len(self.empty_pages) < num_pages_to_allocate:
            raise AssertionError(
                f"requested {num_pages_to_allocate.item()} pages "
                f"but there are only {len(self.empty_pages)} empty pages"
            )

        start_page_idx = self.capacity[batch_idx] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages = torch.tensor(
            self.empty_pages[-num_pages_to_allocate:],
            device=num_pages_to_allocate.device,
        )
        self.empty_pages = self.empty_pages[:-num_pages_to_allocate]

        # update page table
        self.page_table[
            batch_idx,
            start_page_idx:end_page_idx,
        ] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx.item(),
            end_page_idx.item(),
            device=num_pages_to_allocate.device,
        )
        self.capacity[batch_idx] += num_pages_to_allocate * self.page_size

    def erase(self, batch_idx: torch.Tensor) -> None:
        # [FUNCTION DOC] Frees all pages held by one batch entry: returns them to
        # the free list, resets capacity to zero, and sets the page table row to -1.
        # Mutates page_table, capacity, empty_pages and physical_to_logical.
        # KNOWN ISSUE: the reverse map update indexes with batch_idx first and then
        # assigns on the result, which may act on a copy and leave stale entries.
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx (Tensor): batch index to be removed; shape :math:`(1)`.
        """

        # find allocated pages
        allocated_page_idx = self.page_table[batch_idx] != -1
        allocated_pages = self.page_table[batch_idx][allocated_page_idx]

        # clean metadata
        self.capacity[batch_idx] = 0
        self.empty_pages += allocated_pages.tolist()
        self.physical_to_logical[batch_idx][:, allocated_pages] = -1
        self.page_table[batch_idx] = -1

    def assign(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] Writes new key and value tensors (batch, heads, seq, dim)
        # into the shared caches at the physical slots for each batch entry and
        # input position, found through the page table. Mutates k_cache and v_cache
        # in place; returns nothing. Raises RuntimeError if k_val requires grad or
        # if batch size, head count, sequence length or hidden dims mismatch.
        # KNOWN ISSUE: the docstring names args val/cache but the parameters are
        # k_val/v_val/k_cache/v_cache; the sequence-length error message reports
        # the wrong dimension of input_pos; only k_val is checked for requires_grad.
        """
        Assigns new contents `val` to the storage `cache` at the location
        `batch_idx` and `input_pos`.

        Args:
            batch_idx (Tensor): batch index; shape :math:`(B)`.
            input_pos (Tensor): input positions to be assigned for the given batch; shape :math:`(B, S)`.
            val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
            cache (Tensor): the cache to store the values; shape:`(1, H, MAX_S, D)`
        """
        if k_val.requires_grad:
            raise RuntimeError("val must not require gradient")

        B, H, S, K_D = k_val.shape
        V_D = v_val.shape[3]
        if B != batch_idx.shape[0]:
            raise RuntimeError(
                f"Expect val and batch_idx have the same batch size "
                f"but got B={B} and B={batch_idx.shape[0]}."
            )
        if H != k_cache.shape[1]:
            raise RuntimeError(
                f"Expect val and cache has the same number of heads "
                f"but got H={H} and H={k_cache.shape[1]}."
            )
        if S != input_pos.shape[1]:
            raise RuntimeError(
                f"Expect val and input_pos has the same length "
                f"but got S={S} and S={input_pos.shape[0]}."
            )
        if K_D != k_cache.shape[3]:
            raise RuntimeError(
                f"Expect k_val and k_cache has the same hidden dim "
                f"but got D={K_D} and D={k_cache.shape[3]}."
            )
        if V_D != v_cache.shape[3]:
            raise RuntimeError(
                f"Expect v_val and v_cache has the same hidden dim "
                f"but got D={V_D} and D={v_cache.shape[3]}."
            )

        # find address
        logical_block_idx = input_pos // self.page_size  # [B, S]
        logical_block_offset = input_pos % self.page_size  # [B, S]
        physical_block_idx = torch.gather(
            self.page_table[batch_idx], 1, logical_block_idx.to(torch.int64)
        ).to(torch.int32)  # [B, S]

        addr = (physical_block_idx * self.page_size + logical_block_offset).view(
            -1
        )  # [B*S]

        k_val = k_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, K_D)
        v_val = v_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, V_D)

        k_cache[:, :, addr, :] = k_val
        v_cache[:, :, addr, :] = v_val

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        batch_idx: torch.Tensor | None = None,
        kv_len: torch.Tensor | None = None,
    ) -> BlockMask:
        # [FUNCTION DOC] Builds a physical-index BlockMask from a logical one.
        # kv_indices (and full_kv_indices if present) are mapped through the page
        # table rows of batch_idx (default: the first B entries) and padded to n_pages
        # columns; num_blocks are cloned; the mask_mod is wrapped via get_mask_mod
        # using the optional kv_len bound. Raises RuntimeError if the block mask
        # column block size differs from page_size. Returns a new BlockMask whose
        # kv sequence length is n_pages times page_size.
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
            kv_len (Optional[Tensor]): actual KV sequence length for upper bound check;
                shape :math:`(B,)` to handle multiple batches.
        """
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_size"
                f"but got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
            )

        # Increase the num columns of converted block mask from logical block mask's
        # num columns to n_pages, since a) the converted block mask
        # may have larger indices values; and b) `_ordered_to_dense` realizes
        # a dense tensor with these converted indices. There would be an IndexError
        # if using the logical block mask's num columns.

        device = block_mask.kv_num_blocks.device

        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)
        page_table = self.page_table[batch_idx]

        new_kv_num_blocks = block_mask.kv_num_blocks.clone()

        new_kv_indices = torch.zeros(
            (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
        )
        new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
            torch.gather(
                page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64)
            )
            .view(block_mask.kv_indices.shape)
            .to(torch.int32)
        )

        new_full_kv_indices, new_full_kv_num_blocks = None, None
        if block_mask.full_kv_num_blocks is not None:
            if block_mask.full_kv_indices is None:
                raise AssertionError(
                    "block_mask.full_kv_indices must not be None when full_kv_num_blocks is not None"
                )
            new_full_kv_num_blocks = block_mask.full_kv_num_blocks.clone()
            new_full_kv_indices = torch.zeros(
                (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
            )
            new_full_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(
                    page_table,
                    1,
                    block_mask.full_kv_indices.view(B, -1).to(torch.int64),
                )
                .view(block_mask.full_kv_indices.shape)
                .to(torch.int32)
            )

        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, kv_len)

        seq_lengths = (block_mask.seq_lengths[0], self.n_pages * self.page_size)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def get_mask_mod(
        self,
        mask_mod: _mask_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _mask_mod_signature:
        # [FUNCTION DOC] Returns a mask_mod taking physical kv indices. It maps
        # each physical index to its logical index via physical_to_logical, and
        # yields false for unallocated pages, negative logical positions, or
        # positions at or beyond kv_len for that batch (when given); otherwise it
        # defers to the logical mask_mod (no-op mask if None).
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
            kv_len (Optional[torch.Tensor]): actual KV sequence length for upper bound check.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Translates the physical kv index to a logical
            # one and returns the wrapped mask_mod result where the position is
            # valid (live page, within bounds), false elsewhere.
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(is_valid, mask_mod(b, h, q_idx, logical_kv_idx), False)

        return new_mask_mod

    def get_score_mod(
        self,
        score_mod: _score_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _score_mod_signature:
        # [FUNCTION DOC] Returns a score_mod taking physical kv indices. Same
        # index translation and validity rules as get_mask_mod, but invalid
        # positions get a score of negative infinity; valid ones call the logical
        # score_mod (identity if None) with the logical kv index.
        """
        Converts a score_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            score_mod (_score_mod_signature): score_mod based on the logical block index.
            `kv_len (Optional[torch.Tensor]): actual KV sequence length for upper bound check.

        """
        if score_mod is None:
            score_mod = _identity

        def new_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            # [FUNCTION DOC] HELPER: Translates the physical kv index to a logical
            # one; returns the wrapped score_mod result where valid, else negative infinity.
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(
                is_valid,
                score_mod(score, b, h, q_idx, logical_kv_idx),
                float("-inf"),
            )

        return new_score_mod
