# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Tile (work item) schedulers for the quack GEMM kernels. Defines enums for raster
# direction and persistence mode, argument bundles, and three scheduler classes: the base
# TileScheduler (maps CTA or cluster indices to output tile coordinates with an L2-friendly
# grouped swizzle, optionally persistent via static striding, an atomic counter, or hardware
# cluster launch control, handing out work through a shared-memory pipeline), a triangular
# variant for symmetric outputs, and a variable-M variant for ragged batches. Import-time side
# effects: none beyond imports.

from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean, const_expr

from . import utils
from .fast_math import FastDivmod
from .pipeline import PipelineStateWAdvance
from .cute_dsl_utils import mlir_namedtuple


class RasterOrderOption(IntEnum):
    # [CLASS DOC] User-facing choice of the order in which tiles are walked: along M, along N, or a
    # heuristic that picks along M when the padded N tile count exceeds the M count (else along N).
    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN


class RasterOrder(IntEnum):
    # [CLASS DOC] Resolved raster direction (no heuristic left): walk tiles with M fastest
    # or N fastest.
    AlongM = 0
    AlongN = 1


class PersistenceMode(IntEnum):
    # [CLASS DOC] How CTAs obtain work: NONE gives each CTA one tile, STATIC strides a
    # persistent CTA through a linear tile list, DYNAMIC grabs tile numbers from a global atomic
    # counter, and CLC asks the hardware cluster launch control to cancel not-yet-started
    # clusters and reuse them.
    NONE = 0
    STATIC = 1
    DYNAMIC = 2
    CLC = 3


@cute.jit
def get_raster_order_from_option(
    raster_order_option: RasterOrderOption, problem_shape_ncluster_mn: cute.Shape, group_size: Int32
) -> RasterOrder:
    # [FUNCTION DOC] Jit helper resolving a RasterOrderOption into a RasterOrder. Explicit options
    # map directly (anything not AlongM becomes along N). For the heuristic option both cluster
    # counts are rounded up to the group size and the result is along M if the N count is larger,
    # else along N.
    raster_order = (
        RasterOrder.AlongM
        if raster_order_option == RasterOrderOption.AlongM
        else RasterOrder.AlongN
    )
    if raster_order_option == RasterOrderOption.Heuristic:
        problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], group_size)
        problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], group_size)
        raster_order = (
            RasterOrder.AlongM if problem_blocks_n > problem_blocks_m else RasterOrder.AlongN
        )
    return raster_order


# Grouping arguments together that should be passed to __call__
@mlir_namedtuple
class TileSchedulerOptions(NamedTuple):
    # [CLASS DOC] Named tuple of scheduler options passed to the kernel call: the maximum
    # number of active clusters (required), raster order option (default heuristic), maximum
    # swizzle group size (default 8), optional pointer to the global tile counter, and optional
    # batch permutation tensor.
    max_active_clusters: Int32
    raster_order: cutlass.Constexpr[RasterOrderOption] = RasterOrderOption.Heuristic
    max_swizzle_size: Int32 = Int32(8)
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None


@dataclass
class TileSchedulerArguments:
    # [CLASS DOC] Dataclass of inputs for building scheduler parameters: tile counts per mode (M,
    # N, batch), raster order option, swizzle group size, cluster shape, optional tile-count
    # semaphore pointer, optional batch permutation tensor, and persistence mode (default none).
    problem_shape_ntile_mnl: cute.Shape
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    batch_idx_permute: Optional[cute.Tensor] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE


class TileScheduler:
    # [CLASS DOC] Base tile scheduler. State: a work index (linearized cluster or persistent step),
    # number of tiles executed so far, two batch-tracking counters (used by the variable-length
    # subclass), optional shared-memory scheduling buffer and pipeline, the pipeline state, and the
    # Params. Provides static construction, host grid sizing, conversion of a linear index into
    # tile coordinates, reading and writing work in shared memory, advancing to the next work item,
    # and flattening to and from MLIR values for the DSL.
    @dataclass
    class Params:
        # [CLASS DOC] Dataclass of precomputed, compile-time or runtime parameters of the base
        # scheduler: cluster counts (M, N, batch), resolved raster order, fast division helpers for
        # clusters per problem, group size, tail group size and clusters per group, number of full
        # groups, optional semaphore pointer (only kept in dynamic mode), optional batch
        # permutation, cluster shape and persistence mode.
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_fdd: FastDivmod
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        num_clusters_in_group_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        batch_idx_permute: Optional[cute.Tensor]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "TileScheduler.Params":
            # [FUNCTION DOC] Jit factory computing Params from TileSchedulerArguments. Divides tile
            # counts by the cluster shape (rounding up), resolves the raster order, picks the fast
            # and slow dimensions, clamps the group size to the fast dimension, derives the tail
            # group size (the remainder, replaced by one if zero so division is safe), full group
            # count and clusters per group. Asserts that the semaphore pointer exists in dynamic
            # mode, and keeps it only then.
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            raster_order = get_raster_order_from_option(
                args.raster_order, problem_shape_ncluster_mn, args.group_size
            )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(num_clusters_in_group),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.batch_idx_permute,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Stores the work index, tiles-executed counter, the two batch counters,
        # shared memory buffer, scheduler pipeline, pipeline state, params, and the loc and ip
        # values. No computation.
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)
# [FUNCTION DOC] Host-side conversion of scheduler arguments to Params by delegating to
# the Params factory.

    @staticmethod
    @cute.jit
    def _init_clc_mbarrier(sched_smem: Optional[cute.Tensor] = None, *, loc=None, ip=None) -> None:
        # [FUNCTION DOC] HELPER: Jit routine that initializes the cluster launch control mbarrier,
        # stored at an offset of eight integers into the first column of the scheduling buffer.
        # Asserts the buffer holds at least twelve integers per stage (four for the tile, four for
        # the response, two for the barrier). One elected thread inits the barrier with arrival
        # count one, then the memory fence and a warp sync run. Only the scheduler warp is expected
        # to call it.
        # We use 4 ints to store (pid_m, pid_n, batch_idx, is_valid),
        # another 4 ints to store clc response, and 2 ints to store the mbarrier for CLC
        # Since only the scheduler warp will touch the mbarrier (we don't use multicast when trying
        # to cancel workID), we only need the scheduler warp to initialize and sync.
        # If we use multicast when canceling workID, we would need all threads to sync.
        assert cute.size(sched_smem, mode=[0]) >= 12
        clc_mbar_ptr = sched_smem[None, 0].iterator + 8
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(clc_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_warp()

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        # [FUNCTION DOC] HELPER: Splits a hardware cluster index into the initial work index and
        # batch index. In none and CLC modes the work index is the first component and the batch is
        # the third; in static and dynamic modes the work index is the third component and the
        # batch is None.
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            current_work_idx = Int32(cluster_idx[0])
            batch_idx = Int32(cluster_idx[2])
            return current_work_idx, batch_idx
        else:
            current_work_idx = Int32(cluster_idx[2])
            batch_idx = None
            return current_work_idx, batch_idx

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TileScheduler":
        # [FUNCTION DOC] Jit factory building a scheduler for the calling CTA. The starting cluster
        # index comes from the block index (cluster size one) or cluster index, converted to the
        # initial work index. In persistent modes the buffer and pipeline must be given (asserted)
        # and the stage count is taken from the buffer's second extent. In CLC mode the scheduler
        # warp also initializes the mbarrier. Counters start at zero. Documented rule:
        # is_scheduler_warp must be true in only one warp of the whole cluster.
        """is_scheduler_warp should only be true for one warp in the whole cluster"""
        if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1):
            cluster_idx = cute.arch.block_idx()
        else:
            cluster_idx = cute.arch.cluster_idx()
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cluster_idx, loc=loc, ip=ip
        )
        stages = 0
        if const_expr(
            params.persistence_mode
            in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC, PersistenceMode.CLC]
        ):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            if is_scheduler_warp:
                TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return TileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # [FUNCTION DOC] Host-side launch grid computation. Non-persistent (none, CLC): x is
        # cluster x times the number of M,N clusters, y is cluster y, z is cluster z times the
        # batch count. Persistent (static, dynamic): the number of CTAs is the smaller of the
        # problem total and one wave (max active clusters times CTAs per cluster), divided by CTAs
        # per cluster to get persistent clusters; the grid is cluster x, cluster y, and cluster z
        # times that count.
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return (
                params.cluster_shape_mnk[0] * cute.size(params.problem_shape_ncluster_mnl[:2]),
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
            )
        else:
            num_ctas_in_problem = cute.size(
                params.problem_shape_ncluster_mnl, loc=loc, ip=ip
            ) * cute.size(params.cluster_shape_mnk)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mnk, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # [FUNCTION DOC] Maps a linear cluster index within a problem to cluster coordinates
        # (M, N) using grouped rasterization to improve L2 reuse. Splits into group and index
        # within group, uses the regular or tail group size, flips the slow coordinate for odd
        # groups (serpentine), and places the fast and slow coordinates on M or N according to
        # the raster order.
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
            # if cid_slow % 2 == 1:  # inner serpentine
            #     cid_fast_in_group = params.group_size_fdd.divisor - 1 - cid_fast_in_group
        else:  # tail part
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
            # if cid_slow % 2 == 1:  # inner serpentine
            #     cid_fast_in_group = params.group_size_tail_fdd.divisor - 1 - cid_fast_in_group
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_fdd.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _cluster_id_to_cta_id(
        self, cid_m: Int32, cid_n: Int32, *, block_zero_only: bool = False, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # [FUNCTION DOC] Converts cluster coordinates to CTA tile coordinates by scaling with the
        # cluster shape and adding the CTA's position inside the cluster (zero if the cluster has
        # one CTA or block_zero_only is set).
        if const_expr(
            block_zero_only or cute.size(self.params.cluster_shape_mnk, loc=loc, ip=ip) == 1
        ):
            bidx_in_cluster = (Int32(0), Int32(0))
        else:
            # Get the pid from cluster id
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.params.cluster_shape_mnk[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mnk[1] + bidx_in_cluster[1]
        return pid_m, pid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        # [FUNCTION DOC] Converts a work index to a WorkTileInfo. If validity is not given it is:
        # first and only tile (none mode), index below clusters per problem (CLC), or below total
        # clusters over all batches (static, dynamic). For valid work: derives the cluster id in
        # the problem and the batch (from grid position in none and CLC, else by dividing the index
        # by clusters per problem), lets an explicit bidz override the batch, swizzles, converts to
        # CTA coordinates, and applies the batch permutation if present. Invalid work has zero
        # coordinates. The tile coordinate tuple has the K slot set to None.
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            elif const_expr(params.persistence_mode == PersistenceMode.CLC):
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl[:2])
            else:
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl)
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
                cluster_id_in_problem = work_idx
                bidz_ = (
                    cute.arch.block_idx()[2]
                    if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
                    else cute.arch.cluster_idx()[2]
                )
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = (
                bidz_
                if const_expr(params.batch_idx_permute is None)
                else params.batch_idx_permute[bidz_]
            )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        # [FUNCTION DOC] Returns the current work tile as seen by the consumer. In none mode it
        # returns zero coordinates marked invalid (the first tile comes from
        # initial_work_tile_info). Otherwise waits on the pipeline, reads four values (tile M, tile
        # N, batch, valid) from the buffer at the pipeline index, fences the async proxy when the
        # cluster has several CTAs, syncs the warp, has one elected thread release the stage,
        # advances the pipeline state and returns the tile.
        params = self.params
        pid_m, pid_n, batch_idx, is_valid = Int32(0), Int32(0), Int32(0), Boolean(False)
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            pass
        # elif const_expr(params.persistence_mode == PersistenceMode.STATIC):
        #     return self._delinearize_work_idx(loc=loc, ip=ip)
        else:
            self._scheduler_pipeline.consumer_wait(self._pipeline_state)
            pid_m, pid_n, batch_idx, is_valid_i32 = [
                self._sched_smem[i, self._pipeline_state.index] for i in range(4)
            ]
            # Need this fence since the STAS from the producer is using the async proxy.
            # Without this, we get race condition / deadlock.
            if const_expr(cute.size(params.cluster_shape_mnk) > 1):
                cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                self._scheduler_pipeline.consumer_release(self._pipeline_state)
            self._pipeline_state.advance()
            is_valid = Boolean(is_valid_i32)
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, Boolean(is_valid))

    # @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        # [FUNCTION DOC] Returns the first work tile, computed from the stored initial work index.
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)
        # if is_scheduler_warp:
        # work_tile_info = self._delinearize_work_idx(block_zero_only=True, loc=loc, ip=ip)
        # self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
        # self.write_work_tile_to_smem(self._delinearize_work_idx(block_zero_only=True, loc=loc, ip=ip), loc=loc, ip=ip)

    @cute.jit
    def _fetch_next_work_idx(self, *, loc=None, ip=None) -> Int32 | Tuple[Int32, Int32, Boolean]:
        # [FUNCTION DOC] Jit routine for the scheduler warp only: obtains the next work item.
        # Static: adds the number of persistent clusters to the current index. Dynamic: lane zero
        # takes the next number from the global semaphore (atomic increment bounded by total
        # clusters minus one, or atomic add of one when M is unknown) plus the persistent cluster
        # count, then it is broadcast to the warp. CLC: an elected thread arms the mbarrier with
        # sixteen bytes and issues a cancel query, all lanes wait for it, the response (coordinates
        # and valid flag) is decoded into a work index and batch. Returns an index, or for CLC a
        # tuple of index, batch and validity; other modes return zero.
        """should only be called by the scheduler warp"""
        params = self.params
        num_persistent_clusters = (
            cute.arch.grid_dim()[2]
            if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
            else cute.arch.cluster_dim()[2]
        )
        if const_expr(params.persistence_mode == PersistenceMode.STATIC):
            return self._current_work_idx + num_persistent_clusters
            # Serpentine: alternate wave direction for a bit better load balancing
            # But currently seems a tiny bit slower, disabling for now.
            # c = Int32(cute.arch.cluster_idx()[2])
            # next_work_idx = self._current_work_idx + 2 * c + 1
            # if self.num_tiles_executed % 2 == 1:
            #     next_work_idx = self._current_work_idx + 2 * (num_persistent_clusters - 1 - c) + 1
            # return next_work_idx
        elif const_expr(params.persistence_mode == PersistenceMode.DYNAMIC):
            next_work_linear_idx = Int32(0)
            if cute.arch.lane_idx() == 0:
                # If varlen_m, problem_shape_ncluster_mnl[0] is None, so we use atomic_add
                # instead of atomic_inc, and at the end of the kernel must reset the semaphore to 0.
                #                 # cute.printf("before atomicadd, tidx = {}, bidz = {}, idx = {}", cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_idx)
                if const_expr(params.problem_shape_ncluster_mnl[0] is not None):
                    next_work_linear_idx = num_persistent_clusters + utils.atomic_inc_i32(
                        cute.size(params.problem_shape_ncluster_mnl) - 1,
                        params.tile_count_semaphore,
                    )
                else:  # varlen_m
                    next_work_linear_idx = num_persistent_clusters + utils.atomic_add_i32(
                        1, params.tile_count_semaphore
                    )
                # cute.printf("after atomicadd, tidx = {}, bidz = {}, idx = {}", cute.arch.thread_idx()[0], cute.arch.block_idx()[2], current_work_idx)
            return cute.arch.shuffle_sync(next_work_linear_idx, 0)
        elif const_expr(params.persistence_mode == PersistenceMode.CLC):
            clc_response_ptr = self._sched_smem[None, self._pipeline_state.index].iterator + 4
            mbarrier_addr = self._sched_smem[None, 0].iterator + 8
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbarrier_addr, 16, loc=loc, ip=ip)
                utils.issue_clc_query_nomulticast(mbarrier_addr, clc_response_ptr, loc=loc, ip=ip)
            cute.arch.sync_warp()
            cute.arch.mbarrier_wait(mbarrier_addr, self._pipeline_state.phase, loc=loc, ip=ip)
            bidx, bidy, bidz, valid = cute.arch.clc_response(clc_response_ptr, loc=loc, ip=ip)
            cute.arch.fence_view_async_shared()
            cluster_idx = (
                bidx // params.cluster_shape_mnk[0],
                bidy // params.cluster_shape_mnk[1],
                bidz // params.cluster_shape_mnk[2],
            )
            cluster_idx, batch_idx = type(self)._cluster_idx_to_work_idx_batch(
                params, cluster_idx, loc=loc, ip=ip
            )
            return cluster_idx, batch_idx, Boolean(valid)
        else:
            return Int32(0)

    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile_info: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        # [FUNCTION DOC] Publishes a work tile to the shared-memory buffer (nothing happens if no
        # buffer exists). Acquires a producer stage (using the flipped phase), then lanes below the
        # cluster size write. With one CTA per cluster the four values (tile M, tile N, batch,
        # valid) are stored directly and the stage committed. Otherwise each lane acts for a peer
        # CTA rank, computes that CTA's offsets inside the cluster (x fastest, then y), arrives on
        # the peer's barrier expecting sixteen bytes, and remote-stores the four values with the
        # offsets added to M and N.
        params = self.params
        if const_expr(self._sched_smem is not None):
            # producer phase is always consumer_phase ^ 1
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_acquire(pipeline_state_producer)
            sched_data = [
                work_tile_info.tile_idx[0],
                work_tile_info.tile_idx[1],
                work_tile_info.tile_idx[3],
                Int32(work_tile_info.is_valid_tile),
            ]
            lane_idx = cute.arch.lane_idx()
            if lane_idx < cute.size(params.cluster_shape_mnk):
                # cute.printf("Producer pid_m = {}, pid_n = {}, batch_idx = {}, is_valid = {}, after empty wait, idx = {}", sched_data[0], sched_data[1], sched_data[2], sched_data[3], self._current_work_idx)
                pipeline_idx = self._pipeline_state.index
                if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                    for i in cutlass.range_constexpr(4):
                        self._sched_smem[i, pipeline_idx] = sched_data[i]
                    self._scheduler_pipeline.producer_commit(self._pipeline_state)
                else:
                    peer_cta_rank_in_cluster = lane_idx
                    # Here we assume that the block idx in cluster is linearized such that
                    # x is the fastest moving direction, followed by y, then z.
                    bidx_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mnk[0]
                    bidy_in_cluster = (
                        peer_cta_rank_in_cluster // params.cluster_shape_mnk[0]
                    ) % params.cluster_shape_mnk[1]
                    mbar_ptr = self._scheduler_pipeline.producer_get_barrier(self._pipeline_state)
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, 16, peer_cta_rank_in_cluster)
                    utils.store_shared_remote_x4(
                        sched_data[0] + bidx_in_cluster,
                        sched_data[1] + bidy_in_cluster,
                        sched_data[2],
                        sched_data[3],
                        smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                        mbar_ptr=mbar_ptr,
                        peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                    )

    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Advances the scheduler by advance_count steps. Increments the executed-
        # tile counter and, when advancing by more than one, advances the pipeline state
        # accordingly. In static and dynamic modes the scheduler warp then fetches the next
        # index, converts it (block zero only) and writes it to shared memory; in CLC mode the
        # fetched index, batch and validity are used for that. Other modes do nothing more. Per
        # the documentation string: only block zero calls it, and advance_count must be one for
        # the scheduler warp.
        """is_scheduler_warp should only be true for one warp in the whole cluster.
        Moreover, we assume that only block zero in the cluster is calling this function.
        If calling with is_scheduler_warp = True, advance_count must be 1.
        """
        params = self.params
        self.num_tiles_executed += Int32(advance_count)
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if const_expr(params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
        elif const_expr(params.persistence_mode == PersistenceMode.CLC):
            # We assume here that advance_count is 1 for scheduler_warp
            if is_scheduler_warp:
                self._current_work_idx, batch, is_valid = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, batch, is_valid, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def producer_tail(self):
        # [FUNCTION DOC] Producer-side drain: if a pipeline exists, calls its producer tail with
        # a copy of the pipeline state whose phase is flipped, so the producer does not exit
        # with pending stages.
        if const_expr(self._scheduler_pipeline is not None):
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_tail(pipeline_state_producer)

    def __extract_mlir_values__(self):
        # [FUNCTION DOC] Flattens all scheduler members into one list of MLIR values
        # and returns it; records on the instance how many values each member produced
        # for later reconstruction.
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        # [FUNCTION DOC] Rebuilds a scheduler from a flat list of MLIR values using the per-member
        # counts saved by the extract method (which must have run on this instance first). Returns
        # a new instance of the same class.
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


@cute.jit
def triangular_idx_to_coord(idx: Int32) -> Tuple[Int32, Int32]:
    # [FUNCTION DOC] Jit helper mapping a linear index within a lower triangle (row-major order) to
    # a (row, col) pair: the row is derived from the square root of a transformed index (rounded
    # up, minus one) and the column is the remainder after the triangular number of rows before it.
    """
    Convert a triangular index to 2D coordinates.
    This is used to convert the linear index to 2D coordinates for triangular matrices.
    """
    row = utils.ceil((utils.sqrt(2 * idx + 2.25) - 0.5)) - 1
    col = idx - (row * (row + 1)) // 2
    return row, col


class TriangularTileScheduler(TileScheduler):
    # [CLASS DOC] Scheduler for triangular (lower-triangle) output tiles, assuming square tiles per
    # cluster. Reuses the base class state and methods, but replaces Params, construction, grid
    # size, swizzling and index conversion. Raster order is not configurable and the batch
    # permutation is not supported.
    """We assume the tile size per cluster is square (e.g., 128 x 256 per CTA, with cluster 2 x 1)"""

    @dataclass
    class Params:
        # [CLASS DOC] Dataclass of parameters for the triangular scheduler: cluster counts, a fast
        # divider for the number of triangle clusters, the inverse group size as a float, the
        # number of full groups, fast dividers for group size, tail size and their products with
        # the group size, optional semaphore pointer, cluster shape and persistence mode.
        problem_shape_ncluster_mnl: cute.Shape
        num_clusters_per_problem_fdd: FastDivmod
        group_size_inv_f32: Float32
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        group_size_mul_group_size_fdd: FastDivmod
        group_size_tail_mul_group_size_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "TriangularTileScheduler.Params":
            # [FUNCTION DOC] Jit factory building triangular Params. Asserts the cluster depth is
            # one. The clusters per problem is the triangular number of the M cluster count; groups
            # of rows are sized from the group size (clamped to the M cluster count) with a tail, a
            # tail of zero being replaced by one for division. Asserts the semaphore pointer exists
            # in dynamic mode and keeps it only then.
            assert args.cluster_shape_mnk[2] == 1
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                cute.ceil_div(problem_shape_ntile_mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            cluster_m = problem_shape_ncluster_mn[0]
            # Assume that each cluster is responsible for a square tile
            num_clusters_per_problem = cluster_m * (cluster_m + 1) // 2
            group_size = min(args.group_size, cluster_m)
            group_size_tail = cluster_m % group_size
            num_groups_regular = cluster_m // group_size
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TriangularTileScheduler.Params(
                problem_shape_ncluster_mnl,
                FastDivmod(num_clusters_per_problem),
                Float32(1.0 / group_size),
                num_groups_regular,
                FastDivmod(group_size),
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(group_size * group_size),
                FastDivmod((group_size_tail if group_size_tail > 0 else 1) * group_size),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return TriangularTileScheduler.Params.create(args, loc=loc, ip=ip)
# [FUNCTION DOC] Host-side conversion of scheduler arguments to triangular Params.

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "TriangularTileScheduler":
        # [FUNCTION DOC] Jit factory building a triangular scheduler. Same steps as the
        # base factory (initial index from the cluster position, stage count from the
        # buffer in persistent modes, mbarrier init by the scheduler warp in CLC mode) but
        # returns the triangular class. It calls the base class helpers for index
        # conversion and mbarrier setup.
        if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1):
            cluster_idx = cute.arch.block_idx()
        else:
            cluster_idx = cute.arch.cluster_idx()
        current_work_idx, _ = TileScheduler._cluster_idx_to_work_idx_batch(
            params, cluster_idx, loc=loc, ip=ip
        )
        stages = 0
        if const_expr(
            params.persistence_mode
            in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC, PersistenceMode.CLC]
        ):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            if is_scheduler_warp:
                TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return TriangularTileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # [FUNCTION DOC] Host-side launch grid computation for the triangular scheduler. The
        # cluster count along the first axis is the number of triangle clusters and one along
        # the second. Non-persistent modes return the CTA counts; persistent modes cap the
        # CTAs at one wave of active clusters and return the cluster shape with depth scaled
        # by persistent clusters.
        clusters = (params.num_clusters_per_problem_fdd.divisor, 1)
        num_ctas_mnl = (
            clusters[0] * params.cluster_shape_mnk[0],
            clusters[1] * params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
        )
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mnk, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # [FUNCTION DOC] Maps a linear cluster index within the triangle to cluster coordinates
        # (M, N). Finds the row group from the index via a square root scaled by the inverse
        # group size, subtracts the triangle before that group, then either decodes a
        # rectangular part (columns of full group blocks, using regular or tail group size) or,
        # past it, decodes the remaining triangular tail with the triangular index helper.
        # Returns M and N cluster coordinates.
        # CTA Swizzle to promote L2 data reuse
        params = self.params
        group_size = params.group_size_fdd.divisor
        group_id = (
            utils.ceil(
                (utils.sqrt(2 * cluster_id_in_problem + 2.25) - 0.5) * params.group_size_inv_f32
            )
            - 1
        )
        cid_m_start = group_id * group_size
        id_in_group = cluster_id_in_problem - (cid_m_start * (cid_m_start + 1)) // 2
        group_size_actual = (
            group_size
            if group_id < params.num_groups_regular
            else params.group_size_tail_fdd.divisor
        )
        group_col, group_remainder = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            group_col, group_remainder = divmod(id_in_group, params.group_size_mul_group_size_fdd)
        else:  # tail part
            group_col, group_remainder = divmod(
                id_in_group, params.group_size_tail_mul_group_size_fdd
            )
        cid_m_in_group, cid_n_in_group = Int32(0), Int32(0)
        if id_in_group >= group_size_actual * group_size * group_id:  # triangular tail
            cid_m_in_group, cid_n_in_group = triangular_idx_to_coord(group_remainder)
        else:
            if group_id < params.num_groups_regular:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_fdd)
            else:
                cid_n_in_group, cid_m_in_group = divmod(group_remainder, params.group_size_tail_fdd)
        cid_m = cid_m_start + cid_m_in_group
        cid_n = group_col * group_size + cid_n_in_group
        return cid_m, cid_n

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        # [FUNCTION DOC] Converts a work index to a WorkTileInfo for the triangular scheduler.
        # Validity defaults to: first tile only (none mode) or index below clusters per problem
        # times batches. Batch comes from the grid position (none, CLC) or from dividing the index
        # by clusters per problem; an explicit bidz overrides it. No batch permutation is applied.
        # Invalid work gives zero coordinates; the tile coordinate K slot is None.
        params = self.params
        if const_expr(is_valid is None):
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            else:
                is_valid = (
                    work_idx
                    < params.num_clusters_per_problem_fdd.divisor
                    * params.problem_shape_ncluster_mnl[2]
                )
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
                cluster_id_in_problem = work_idx
                bidz_ = (
                    cute.arch.block_idx()[2]
                    if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1)
                    else cute.arch.cluster_idx()[2]
                )
            else:
                bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
                cluster_id_in_problem = Int32(cluster_id_in_problem)  # divmod returns IntValue
            if const_expr(bidz is not None):
                bidz_ = bidz
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = bidz_
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        # tidx, _, _ = cute.arch.thread_idx()
        # if tidx == 0:
        #     cute.printf("bidx = {}, bidy = {}, group_id = {}, id_in_group = {}, group_size_actual = {}, group_col = {}, group_remainder = {}, cid_n_in_group = {}, cid_m_in_group = {}, cid_m = {}, cid_n = {}, is_valid = {}",
        #                 bidx, bidy, group_id, id_in_group, group_size_actual, group_col, group_remainder, cid_n_in_group, cid_m_in_group, cid_m, cid_n, is_valid)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)


@dataclass
class VarlenMTileSchedulerArguments:
    # [CLASS DOC] Dataclass of inputs for the variable-M scheduler: tile counts (M entry
    # unknown), total rows over all batches, cumulative sequence length tensor along M, raster
    # order option, swizzle group size, per-CTA tile shape, cluster shape, optional semaphore
    # pointer and persistence mode.
    problem_shape_ntile_mnl: cute.Shape
    total_m: Int32
    cu_seqlens_m: cute.Tensor
    raster_order: cutlass.Constexpr[RasterOrderOption]
    group_size: Int32
    tile_shape_mn: cutlass.Constexpr[cute.Shape]
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.NONE


class VarlenMTileScheduler(TileScheduler):
    # [CLASS DOC] Scheduler for batches whose M extent varies per batch (ragged, described by a
    # cumulative lengths tensor). Tile counts along M per batch are found at run time by a warp
    # scanning up to thirty-one batches at a time. Extra state used here: the current batch index
    # and the work index at which the current batch starts, both updated during delinearization.
    # Batch index from the cluster index is always None.
    @dataclass
    class Params:
        # [CLASS DOC] Dataclass of parameters for the variable-M scheduler: cluster counts (M entry
        # None), total rows, cumulative length tensor, resolved raster order, group size, optional
        # fast dividers (only available when the dimension involved is static), tile shape,
        # optional semaphore pointer, cluster shape and persistence mode.
        problem_shape_ncluster_mnl: cute.Shape
        total_m: Int32
        cu_seqlens_m: cute.Tensor
        raster_order: cutlass.Constexpr[RasterOrder]
        group_size: Int32
        group_size_fdd: Optional[FastDivmod]
        group_size_tail_fdd: Optional[FastDivmod]
        num_clusters_in_group_fdd: FastDivmod
        tile_shape_mn: cutlass.Constexpr[cute.Shape]
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "VarlenMTileScheduler.Params":
            # [FUNCTION DOC] Jit factory building Params for the variable-M scheduler. The M
            # cluster count is None. Raster order is along M only when explicitly requested,
            # otherwise along N (so the heuristic option gives along N). Group size, tail and
            # clusters per group are computed only when the dimensions involved are static,
            # otherwise left None; zero tail is replaced by one. Asserts the semaphore exists in
            # dynamic mode and keeps it only then.
            # KNOWN ISSUE: annotated to take the base TileSchedulerArguments, but reads total_m,
            # cu_seqlens_m and tile_shape_mn which only the variable-M arguments class defines.
            # problem_shape_ntile_mnl[0] will be None for VarlenM
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = (
                None,
                cute.ceil_div(problem_shape_ntile_mn[1], args.cluster_shape_mnk[1]),
            )
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            raster_order = const_expr(
                RasterOrder.AlongM
                if args.raster_order == RasterOrderOption.AlongM
                else RasterOrder.AlongN  # For Heuristic we also use AlongN
            )
            ncluster_fast = problem_shape_ncluster_mn[
                0 if raster_order == RasterOrder.AlongM else 1
            ]
            ncluster_slow = problem_shape_ncluster_mn[
                1 if raster_order == RasterOrder.AlongM else 0
            ]
            if const_expr(ncluster_fast is not None):
                group_size = min(args.group_size, ncluster_fast)
                group_size_tail = ncluster_fast % group_size
            else:
                group_size, group_size_tail = args.group_size, None
            num_clusters_in_group = None
            if const_expr(ncluster_slow is not None):
                num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return VarlenMTileScheduler.Params(
                problem_shape_ncluster_mnl,
                args.total_m,
                args.cu_seqlens_m,
                raster_order,
                group_size,
                FastDivmod(group_size) if ncluster_fast is not None else None,
                # Don't divide by 0
                FastDivmod(group_size_tail if group_size_tail > 0 else 1)
                if group_size_tail is not None
                else None,
                FastDivmod(num_clusters_in_group) if num_clusters_in_group is not None else None,
                args.tile_shape_mn,
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Same as the base constructor: stores work index, counters, buffer,
        # pipeline, pipeline state, params, loc and ip. Redefined here but identical in behavior.
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return VarlenMTileScheduler.Params.create(args, loc=loc, ip=ip)
# [FUNCTION DOC] Host-side conversion of arguments to variable-M Params.
# KNOWN ISSUE: annotated with the base arguments class, though the variable-M arguments class
# is what the factory actually needs.

    @staticmethod
    @cute.jit
    def _cluster_idx_to_work_idx_batch(
        params: Params, cluster_idx: Tuple[Int32, Int32, Int32], *, loc=None, ip=None
    ) -> Tuple[Int32, Optional[Int32]]:
        # [FUNCTION DOC] HELPER: Splits a cluster index into the initial work index and batch. The
        # work index is the first component in none and CLC modes, else the third. Batch is always
        # None because batches are discovered at run time.
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            current_work_idx = Int32(cluster_idx[0])
        else:
            current_work_idx = Int32(cluster_idx[2])
        batch_idx = None
        return current_work_idx, batch_idx

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: Optional[cute.Tensor] = None,
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        is_scheduler_warp: bool | Boolean = False,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenMTileScheduler":
        # [FUNCTION DOC] Jit factory building a variable-M scheduler, same steps as the
        # base factory (cluster position to initial work index, stage count from the buffer
        # in persistent modes, mbarrier init by the scheduler warp in CLC mode) returning
        # the variable-M class.
        if const_expr(cute.size(params.cluster_shape_mnk, loc=loc, ip=ip) == 1):
            cluster_idx = cute.arch.block_idx()
        else:
            cluster_idx = cute.arch.cluster_idx()
        current_work_idx, _ = VarlenMTileScheduler._cluster_idx_to_work_idx_batch(
            params, cluster_idx, loc=loc, ip=ip
        )
        stages = 0
        if const_expr(
            params.persistence_mode
            in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC, PersistenceMode.CLC]
        ):
            assert sched_smem is not None
            assert scheduler_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
        if const_expr(params.persistence_mode == PersistenceMode.CLC):
            if is_scheduler_warp:
                TileScheduler._init_clc_mbarrier(sched_smem, loc=loc, ip=ip)
        return VarlenMTileScheduler(
            current_work_idx,
            Int32(0),  # num_tiles_executed
            Int32(0),  # current_batch_idx
            Int32(0),  # num_work_idx_before_cur_batch
            sched_smem,
            scheduler_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # [FUNCTION DOC] Host-side launch grid. Upper-bounds the number of M clusters by summing,
        # over batches, a ceiling division of rows into blocks (total rows plus batch count times
        # block size minus one, divided by block size), multiplies by the N cluster count. Non-
        # persistent modes use the cluster x size times that bound along x. Persistent modes use
        # the smaller of max active clusters and the bound, scaling the depth.
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mnk[0]
        num_batch = params.problem_shape_ncluster_mnl[2]
        total_clusters_m_max = (params.total_m + num_batch * (block_size - 1)) // block_size
        total_clusters_max = total_clusters_m_max * params.problem_shape_ncluster_mnl[1]
        if const_expr(params.persistence_mode in [PersistenceMode.NONE, PersistenceMode.CLC]):
            return (
                params.cluster_shape_mnk[0] * total_clusters_max,
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2],
            )
        else:
            num_persistent_clusters = cutlass.min(max_active_clusters, total_clusters_max)
            return (
                params.cluster_shape_mnk[0],
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * num_persistent_clusters,
            )

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, num_clusters_m: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        # [FUNCTION DOC] Maps a cluster index within the current batch, with that batch's M cluster
        # count given, to cluster coordinates (M, N), with grouped serpentine rasterization. Group
        # size and clusters per group use fast dividers when known, otherwise are computed from the
        # M count (along N case). The final group may be short and is handled by dividers (tail) or
        # by an explicit actual group size (along M case). Odd groups reverse the slow coordinate.
        params = self.params
        # CTA Swizzle to promote L2 data reuse
        if const_expr(params.num_clusters_in_group_fdd is not None):
            group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
            num_clusters_in_group = params.num_clusters_in_group_fdd.divisor
        else:
            assert params.raster_order == RasterOrder.AlongN
            num_clusters_in_group = params.group_size * num_clusters_m
            group_id = cluster_id_in_problem // num_clusters_in_group
            id_in_group = cluster_id_in_problem - group_id * num_clusters_in_group
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if const_expr(params.group_size_fdd is not None and params.group_size_tail_fdd is not None):
            num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
            if (group_id + 1) * num_clusters_in_group <= num_clusters:
                cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
                # if cid_slow % 2 == 1:  # inner serpentine
                #     cid_fast_in_group = params.group_size_fdd.divisor - 1 - cid_fast_in_group
            else:  # tail part
                cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
                # if cid_slow % 2 == 1:  # inner serpentine
                #     cid_fast_in_group = params.group_size_tail_fdd.divisor - 1 - cid_fast_in_group
        else:
            assert params.raster_order == RasterOrder.AlongM
            group_size_actual = cutlass.min(
                params.group_size, num_clusters_m - group_id * params.group_size
            )
            cid_slow = id_in_group // group_size_actual
            cid_fast_in_group = id_in_group - cid_slow * group_size_actual
            # if cid_slow % 2 == 1:  # inner serpentine
            #     cid_fast_in_group = group_size_actual - 1 - cid_fast_in_group
        if group_id % 2 == 1:  # serpentine order
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else num_clusters_m
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n

    @cute.jit
    def _get_num_m_blocks(
        self, lane: Int32, bidb_start: Int32, block_size: cutlass.Constexpr[int]
    ) -> Int32:
        # [FUNCTION DOC] Warp-wide helper: lane i looks at batch index start plus lane, reads the
        # cumulative length there and the next lane's value (via shuffle) to get that batch's rows,
        # and returns the number of M blocks as a ceiling division by the block size. Lanes beyond
        # the last batch, and the last lane of the warp, return zero, since no neighbour value
        # exists for it.
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        batch_idx = lane + bidb_start
        cur_cu_seqlen = Int32(0)
        if batch_idx <= num_batch:
            cur_cu_seqlen = self.params.cu_seqlens_m[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        return (
            cute.ceil_div(seqlen, block_size)
            if batch_idx < num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,  # not used
        is_valid_: Optional[Boolean] = None,
        *,
        block_zero_only: bool = False,
        loc=None,
        ip=None,
    ) -> cutlass.utils.WorkTileInfo:
        # [FUNCTION DOC] Converts a work index to a WorkTileInfo for ragged batches using the whole
        # warp. Asserts no explicit bidz is given. Starting from the saved batch and work offset,
        # scans batches in chunks of up to thirty-one with a warp prefix sum until the chunk
        # containing the index is found (clamping to the batch count if exhausted), then picks the
        # exact batch by a ballot of lanes whose cumulative range ends at or before the index.
        # Validity defaults to true, can be passed in, requires a batch in range, and in none mode
        # also needs no tiles executed. Side effects: updates the saved current-batch and work-
        # offset members. Returns tile coordinates with the batch index.
        assert bidz is None
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_batch = self.params.problem_shape_ncluster_mnl[2]
        block_size = params.tile_shape_mn[0] * params.cluster_shape_mnk[0]
        batch_idx = self._current_batch_idx
        next_tile_idx = work_idx

        problems_end_tile = self._num_work_idx_before_cur_batch
        num_clusters_m, num_clusters_cumulative, clusters_in_problems = Int32(0), Int32(0), Int32(0)
        is_valid = True
        if const_expr(is_valid_ is not None):
            is_valid = is_valid_
        if is_valid:
            while problems_end_tile <= next_tile_idx:
                num_clusters_m = self._get_num_m_blocks(
                    lane_idx, bidb_start=batch_idx, block_size=block_size
                )
                num_clusters = num_clusters_m * params.problem_shape_ncluster_mnl[1]
                num_clusters_cumulative = utils.warp_prefix_sum(num_clusters, lane_idx)
                # Total number of blocks for the next 31 problems, same for all lanes
                clusters_in_problems = cute.arch.shuffle_sync(
                    num_clusters_cumulative, cute.arch.WARP_SIZE - 1
                )
                problems_end_tile += clusters_in_problems
                if problems_end_tile <= next_tile_idx:
                    batch_idx += cute.arch.WARP_SIZE - 1
                if batch_idx >= num_batch:
                    batch_idx = Int32(num_batch)
                    problems_end_tile = next_tile_idx + 1
        else:
            batch_idx = Int32(num_batch)

        is_valid = batch_idx < num_batch
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            is_valid &= self.num_tiles_executed == 0
        cid_m, cid_n = Int32(0), Int32(0)
        num_work_idx_before_cur_batch = self._num_work_idx_before_cur_batch
        if is_valid:
            problems_start_tile = problems_end_tile - clusters_in_problems
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, problems_end_tile = %d, num_clusters_m=%d, batch_idx = %d", self._tile_idx, problems_end_tile, num_clusters_m, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_problems = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    problems_start_tile + num_clusters_cumulative <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_problems
            num_clusters_prev_lane = (
                0
                if batch_idx_in_problems == 0
                else cute.arch.shuffle_sync(num_clusters_cumulative, batch_idx_in_problems - 1)
            )
            num_clusters_m = cute.arch.shuffle_sync(num_clusters_m, batch_idx_in_problems)
            num_work_idx_before_cur_batch = problems_start_tile + num_clusters_prev_lane
            cluster_id_in_problem = next_tile_idx - num_work_idx_before_cur_batch
            # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, cid_n=%d, cid_m=%d, is_valid = %d", self._tile_idx, batch_idx, cid_n, cid_m, is_valid)
            cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, num_clusters_m, loc=loc, ip=ip)
        pid_m, pid_n = self._cluster_id_to_cta_id(
            cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
        )
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        self._current_batch_idx = batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)
