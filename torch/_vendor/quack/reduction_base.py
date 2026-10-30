# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# [MODULE DOC] Shared base class for row-wise reduction kernels (RMSNorm, LayerNorm and
# similar) written with the CuTe DSL. It supplies the tiling arithmetic, the shared-memory
# buffer layout for cross-warp and cross-CTA reductions, and the cluster mbarrier setup that
# the concrete kernels build on. It only defines a class; importing it has no side effects.

from typing import Type, Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr

from . import copy_utils


class ReductionBase:
    # [CLASS DOC] Base class holding the configuration common to a row-reduction kernel:
    # element dtype, row length N, number of pipeline stages of the reduction buffer, and the
    # dtype used for reductions. Subclasses must override the threads-per-row hook and may
    # override the thread-count and cluster hooks; the cluster_n attribute is expected to be
    # set by the cluster hook before any tiling helper is used.
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=Float32):
        # [FUNCTION DOC] Store the element dtype, row length N, number of reduction stages
        # and the reduction accumulation dtype (default Float32) on the instance. No
        # validation is done.
        self.dtype = dtype
        self.N = N
        self.stage = stage
        self.reduction_dtype = reduction_dtype

    def _threads_per_row(self):
        # [FUNCTION DOC] HELPER: Abstract hook returning how many threads cooperate on one
        # row. The base version always raises NotImplementedError; subclasses must
        # override it.
        raise NotImplementedError()

    def _num_threads(self):
        # [FUNCTION DOC] HELPER: Default CTA thread count: 128 threads when the row length is
        # at most 16384, otherwise 256. Subclasses may override with a configured value.
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        # [FUNCTION DOC] HELPER: Default cluster-size hook; sets the cluster_n attribute to 1
        # (no clustering). Subclasses override it to pick a larger cluster on supporting GPUs.
        self.cluster_n = 1

    def _cap_cluster_n(self, vecsize: int) -> None:
        # [FUNCTION DOC] HELPER: Reduce cluster_n (in place on the instance) so that each
        # CTA of the cluster owns a distinct, non-empty slice of the row. The upper bound is
        # the number of vector blocks in the row divided by the threads per row, floored at
        # 1; the current cluster_n is only ever lowered, never raised. Takes the vector
        # width in elements.
        """Cap ``cluster_n`` so every peer CTA owns a distinct, non-empty N-tile.

        A clustered launch splits the row across ``cluster_n`` CTAs. If
        ``threads_per_row * cluster_n`` exceeds the number of vector blocks in the
        row (``N // vecsize``), one CTA tile already spans the whole row
        (``tiler_mn[1] >= N``); local_tile then collapses every peer onto tile 0,
        so the peers re-reduce the same columns and double-count in the cluster
        reduction. Capping to ``(N // vecsize) // threads_per_row`` guarantees
        ``tiler_mn[1] < N`` whenever the resulting ``cluster_n > 1``.
        """
        max_cluster_n = max(1, (self.N // vecsize) // self._threads_per_row())
        self.cluster_n = min(self.cluster_n, max_cluster_n)

    def _get_tiled_copy(self, vecsize: int = 1):
        # [FUNCTION DOC] HELPER: Build the global-memory tiled copy for a row tile. Raises an
        # AssertionError if N is not divisible by the vector size or if the thread count is
        # not a multiple of the warp size. The tile has thread-count divided by
        # threads-per-row rows and enough vector blocks per row so that cluster_n CTAs
        # together cover N. Returns a tuple of the tiled copy (built through the copy_utils 2D
        # helper), the tile shape, and threads per row.
        assert self.N % vecsize == 0, f"Input N {self.N} is not divisible by vector size {vecsize}"
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
        # [FUNCTION DOC] HELPER: Compute the shared-memory layout of the reduction scratch
        # buffer from the thread-value layout. Warps per row are derived from the layout (all
        # warps when the thread mode is flat, otherwise the first sub-mode divided by the warp
        # size, at least 1). Returns an ordered layout with modes (rows of warps, (warps per
        # row, cluster_n), stage).
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self, smem: cutlass.utils.SmemAllocator, tv_layout: cute.Layout, is_persistent: bool = False
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        # [FUNCTION DOC] HELPER: Allocate, from the given shared-memory allocator, the
        # reduction buffer (reduction dtype, 8-byte aligned) and, only when cluster_n is above
        # 1, an array of 64-bit mbarrier slots (one per stage, or two per stage when
        # persistent so that full and empty barriers are both available). Returns a pair of
        # buffer and barrier pointer; the barrier pointer is None without clustering.
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(
                Int64, num_elems=self.stage if not is_persistent else self.stage * 2
            )
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(
        self,
        tidx: Int32,
        mbar_ptr: cute.Pointer,
        num_warps: int,
        is_persistent: bool = False,
    ):
        # [FUNCTION DOC] HELPER: JIT-compiled cluster initialization, a no-op when cluster_n
        # is 1. Otherwise, threads whose index is below the stage count initialize the full
        # barrier of their stage with arrival count 1 and, in persistent mode, the matching
        # empty barrier (offset by the stage count) with an arrival count of warps times
        # cluster size. All threads then issue the mbarrier init fence and a relaxed cluster
        # arrive; the matching cluster wait is left to the caller.
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:  # Initialize full barrier
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                if const_expr(is_persistent):  # Initialize empty barrier
                    cute.arch.mbarrier_init(
                        mbar_ptr + self.stage + tidx, num_warps * self.cluster_n
                    )
            cute.arch.mbarrier_init_fence()
            # Cluster arrive after barrier init
            cute.arch.cluster_arrive_relaxed()
