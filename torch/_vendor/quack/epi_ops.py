# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Composable epilogue operations (EpiOps) for GEMM kernels. Each EpiOp class
# describes how one kind of epilogue tensor (scalar, broadcast row or column vector, tile load,
# tile store, vector reduction, grouped local reduction) is turned into kernel parameters, sized
# and laid out in shared memory, set up once per output tile (begin), sliced per subtile
# (begin_loop) and finalized (end_loop, end). Also holds the register-level helpers that multiply
# by or accumulate into broadcast vectors. No import-time side effects beyond imports.
"""Composable epilogue operations (EpiOps) for GEMM kernels.

Each EpiOp encapsulates a single tensor kind's behavior across the epilogue lifecycle:
smem allocation, begin (one-time per-tile setup), begin_loop (per-subtile extraction),
end (cleanup).

The ops are composed via ComposableEpiMixin. Class-level `_epi_ops` is the
static schema; `_epi_ops_to_params_dict` (called from each subclass's
`epi_to_underlying_arguments`) shadows it with an instance-level tuple of only
the active ops (those whose arg tensor is non-None). All EpiOp hook methods
below therefore assume their `param` / `arg_tensor` is non-None — the
framework guarantees inactive ops are never iterated.
"""

import math
import operator
from functools import partial
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr

from .epi_utils import assume_stride_divisibility, setup_epi_tensor
from .sm90_utils import partition_for_epilogue
from . import utils
from . import copy_utils
from . import layout_utils


class EpiContext:
    # [CLASS DOC] Plain bundle of per-tile values handed to the begin hook of an EpiOp: epilogue
    # tile shape, tiled copies for tmem-to-register and register-to-smem, tile coordinate, varlen
    # manager, epilogue barrier, thread index, optional register layout of the main output tile,
    # the tile sizes in M and N, the batch index, the number of epilogue threads and a partial of
    # the epilogue-partitioning function. Fixed slots, no mutation after construction.
    """Shared context passed to EpiOp.begin methods. Bundles common arguments.

    `tRS_rD_layout` is only populated by callers that need TileLoad — it's the
    register layout of the matmul output tile, which TileLoad uses to shape its
    own register tile so it lines up element-wise with tRS_rD in epi_visit_subtile.
    """

    __slots__ = (
        "epi_tile",
        "tiled_copy_t2r",
        "tiled_copy_r2s",
        "tile_coord_mnkl",
        "varlen_manager",
        "epilogue_barrier",
        "tidx",
        "tRS_rD_layout",
        "partition_for_epilogue_fn",
        "num_epi_threads",
        "batch_idx",
        "tile_M",
        "tile_N",
    )

    def __init__(
        self,
        gemm,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
        tRS_rD_layout=None,
    ):
        # [FUNCTION DOC] Store the given pieces and derive the tile sizes from the gemm CTA tile
        # shape, the batch index from the fourth tile coordinate entry, and the epilogue thread
        # count from the number of epilogue warps times the warp size. Builds the partition
        # function bound to the epilogue tile, thread index and the t2r copy when present
        # (otherwise the r2s copy, in which case the source side is used as reference).
        self.epi_tile = epi_tile
        self.tiled_copy_t2r = tiled_copy_t2r
        self.tiled_copy_r2s = tiled_copy_r2s
        self.tile_coord_mnkl = tile_coord_mnkl
        self.varlen_manager = varlen_manager
        self.epilogue_barrier = epilogue_barrier
        self.tidx = tidx
        self.tRS_rD_layout = tRS_rD_layout
        self.tile_M = gemm.cta_tile_shape_mnk[0]
        self.tile_N = gemm.cta_tile_shape_mnk[1]
        self.batch_idx = tile_coord_mnkl[3]
        self.num_epi_threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        self.partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )


def _get_lane_warp_layouts(tiled_copy, reference_src=True):
    # [FUNCTION DOC] HELPER: Derive lane and warp layouts along M and N from an epilogue tiled
    # copy. Inverts the thread-value layout (source layout when reference_src is true, destination
    # layout otherwise), restricts it to the tile shape, then composes with mappings from thread-
    # value index to lane index and to warp index and keeps the M part and the N part of each,
    # dropping zero strides. Returns a pair (lane layout, warp layout), each a two-mode layout
    # over M and N. Used by reductions to decide where shuffles and shared-memory reductions are
    # needed.
    """Derive lane and warp layouts along M and N from the epilogue tiled_copy.

    Follows the CUTLASS Sm90RowReduction / Sm90ColReduction pattern.
    Uses layout_src_tv_tiled (SM90, reference_src=True) or
    layout_dst_tv_tiled (SM100, reference_src=False), matching the C++ impl's
    get_layoutS_TV / get_layoutD_TV selection.

    Returns (lane_layout_MN, warp_layout_MN) where each is a 2D layout (M, N):
      lane_layout_MN[0] = lane_M: (lanes_in_M):(lane_stride_M) — e.g. 8:4
      lane_layout_MN[1] = lane_N: (lanes_in_N):(lane_stride_N) — e.g. 4:1
      warp_layout_MN[0] = warp_M: (warps_in_M):(warp_stride_M) — e.g. 4:1
      warp_layout_MN[1] = warp_N: (warps_in_N):(warp_stride_N) — e.g. 1:0

    For RowVecReduce (reduce along M): shuffle across lane_M, smem reduce across warp_M.
    For ColVecReduce (reduce along N): shuffle across lane_N, direct write (warps_in_N == 1).
    """
    # right_inverse of the TV layout gives tile_element_idx -> tv_idx.
    # SM90: use src (register) layout; SM100: use dst (smem) layout.
    layout_tv = tiled_copy.layout_src_tv_tiled if reference_src else tiled_copy.layout_dst_tv_tiled
    ref_layout = cute.right_inverse(layout_tv)
    tile_M_size, tile_N_size = cute.size(tiled_copy.tiler_mn[0]), cute.size(tiled_copy.tiler_mn[1])
    ref_layout_MN = cute.composition(
        ref_layout, cute.make_layout((tile_M_size, tile_N_size))
    )  # (tile_M, tile_N) -> tv_idx

    num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

    # tv2lane: tv_idx -> lane_idx  (lane = tv_idx % 32)
    tv2lane = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(1, 0, 0))
    ref2lane = cute.composition(tv2lane, ref_layout_MN)  # (tile_M, tile_N) -> lane_idx
    # select mode [0] = M part, [1] = N part; filter removes stride-0
    lane_M = cute.filter(cute.select(ref2lane, [0]))  # lane_m -> lane_idx
    lane_N = cute.filter(cute.select(ref2lane, [1]))  # lane_n -> lane_idx
    lane_layout_MN = layout_utils.concat_layout(lane_M, lane_N)  # (lane_M, lane_N) -> lane_idx

    # tv2warp: tv_idx -> warp_idx  (warp = tv_idx / 32)
    tv2warp = cute.make_layout((cute.arch.WARP_SIZE, num_warps, 1), stride=(0, 1, 0))
    ref2warp = cute.composition(tv2warp, ref_layout_MN)  # (tile_M, tile_N) -> warp_idx
    warp_M = cute.filter(cute.select(ref2warp, [0]))  # warp_m -> warp_idx
    warp_N = cute.filter(cute.select(ref2warp, [1]))  # warp_n -> warp_idx
    warp_layout_MN = layout_utils.concat_layout(warp_M, warp_N)  # (warp_M, warp_N) -> warp_idx

    return lane_layout_MN, warp_layout_MN


class EpiSmemBytes(NamedTuple):
    # [CLASS DOC] Named tuple of shared-memory byte counts for one epilogue op: unstaged bytes
    # allocated once per CTA tile, bytes per D (store) stage, and bytes per C (load) stage.
    # Supports addition field by field, and addition with the integer zero so the builtin sum
    # works over a collection of them.
    """Shared-memory accounting for one epilogue op.

    unstaged: allocated once per CTA tile.
    d_stage: allocated per D/store epilogue stage.
    c_stage: allocated per C/load epilogue stage.
    """

    unstaged: int = 0
    d_stage: int = 0
    c_stage: int = 0

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Return a new byte count whose three fields are the field-wise
        # sums of self and other.
        return EpiSmemBytes(
            self.unstaged + other.unstaged,
            self.d_stage + other.d_stage,
            self.c_stage + other.c_stage,
        )

    def __radd__(self, other):
        # [FUNCTION DOC] HELPER: Support the builtin sum: when the left operand is the integer
        # zero return self, otherwise defer to field-wise addition.
        if other == 0:
            return self
        return self.__add__(other)


LOCAL_REDUCE_FRAGMENT_WIDTH = 32


def grouped_local_reduce_uses_smem(axis: int, group: int) -> bool:
    # [FUNCTION DOC] Return true when a grouped reduction needs shared memory, which is only the
    # case for reduction along axis zero (M) with a group size larger than the fragment width
    # constant (32), because then values span lane fragments.
    """Return whether a grouped-M reduction crosses lane fragments."""
    return axis == 0 and group > LOCAL_REDUCE_FRAGMENT_WIDTH


class EpiOp:
    # [CLASS DOC] Base class of all epilogue ops. Holds only the field name used to look up this
    # op's argument and parameter entries. Subclasses override host-side hooks (parameter fields,
    # conversion of arguments to parameters, shared memory sizing and struct fields, TMA atoms,
    # gmem-to-smem copy) and device-side hooks (begin, begin_loop, load_s2r, end_loop, end). The
    # defaults here describe an op with no parameters, no smem and no work.
    """Base class for composable epilogue operations."""

    def __init__(self, name):
        # [FUNCTION DOC] Record the field name that identifies this op in arguments and
        # parameters.
        self.name = name

    # --- Host-side: args → params ---
    def param_fields(self):
        # [FUNCTION DOC] Return the list of triples (field name, type, default) used to generate
        # the epilogue parameters class. Must match the keys that to_params returns. Base version
        # has none.
        """Return [(field_name, type, default), ...] for auto-generating EpilogueParams.
        Must match the keys returned by to_params()."""
        return []

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Convert this op's argument fields into a mapping of parameter names to
        # values. Base version returns an empty mapping.
        """Convert this op's arg field(s) to param dict entries.
        Returns dict of {param_name: value}. Like EVT's to_underlying_arguments."""
        return {}

    def epi_m_major_score(self, arg_tensor, gemm):
        # [FUNCTION DOC] Return an integer preference for the epilogue subtile traversal order:
        # positive favors M-major, negative favors N-major. Base version is neutral (zero).
        """Preference for epilogue subtile order. Positive prefers M-major, negative N-major."""
        return 0

    # --- Host-side: smem allocation ---
    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Return an EpiSmemBytes describing the unstaged, per-D-stage and per-C-
        # stage shared memory this op needs. Base version needs none.
        """Bytes of smem needed by unstaged / D-stage / C-stage storage."""
        return EpiSmemBytes()

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Return the pair (field name, field type) to add to the shared storage
        # struct, or None if the op needs no smem. params is the full epilogue parameters object.
        # Base version returns None.
        """Return (field_name, field_type) for @cute.struct, or None if no smem needed.
        params is the full EpilogueParams object."""
        return None

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Extract this op's shared memory tensor from the epilogue storage, or None
        # if it has none. Base version returns None.
        """Extract smem tensor from storage.epi. Returns tensor or None.
        params is the full EpilogueParams object."""
        return None

    def tma_atoms(self, gemm, params):
        # [FUNCTION DOC] Return the list of TMA atoms this op uses. Base version returns an empty
        # list.
        """Return list of TMA atoms for this op."""
        return []

    def is_tile_load(self):
        # [FUNCTION DOC] Report whether the op is a tile-sized input loaded through the C load
        # pipeline. Base version returns false.
        """Whether this op is a tile-sized epilogue input loaded through the C pipeline."""
        return False

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        # [FUNCTION DOC] Return a function that copies one subtile from global to shared memory
        # for this op, or None if the op has no such load. Base version returns None.
        """Return a per-subtile gmem->smem copy function, or None."""
        return None

    # --- Device-side: kernel execution ---
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Device hook run once per output tile. Returns per-tile state that is
        # later given to begin_loop. Base version returns None.
        """One-time per-tile setup. Returns state for begin_loop."""
        return None

    def begin_loop(self, gemm, state, epi_coord):
        # [FUNCTION DOC] Device hook run per subtile. Returns the value passed to the epilogue
        # visitor; base version returns the state unchanged.
        """Per-subtile extraction. Returns value for epi_visit_subtile."""
        return state

    @cute.jit
    def load_s2r(self, gemm, param, state, stage_idx):
        # [FUNCTION DOC] Device hook that issues the smem-to-register copy for one epilogue stage.
        # Base version does nothing.
        """Issue this op's tile-load smem->register copy for one epilogue stage."""
        pass

    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] Device hook for cleanup after the epilogue visitor handled one subtile.
        # Base version does nothing.
        """Per-subtile cleanup after epi_visit_subtile."""
        pass

    def needs_async_fence(self):
        # [FUNCTION DOC] Report whether the op issues async copies that require a fence. Base
        # version returns false.
        """Whether this op issues async copies that need a fence."""
        return False

    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] Device hook run after all subtiles of a tile (for example final
        # reductions or direct writes). Base version does nothing.
        """Cleanup after all subtiles (reductions, direct writes)."""
        pass


class Scalar(EpiOp):
    # [CLASS DOC] Op that loads one scalar (or a value behind a device pointer) per tile. Uses no
    # shared memory. Optional dtype selects the loaded element type.
    """Loads a scalar value or device pointer once per tile. No smem."""

    def __init__(self, name, dtype=None):
        # [FUNCTION DOC] Store the field name via the base class and the optional element type
        # used when loading.
        super().__init__(name)
        self.dtype = dtype

    def param_fields(self):
        # [FUNCTION DOC] Declare a single parameter field named after the op, defaulting to None.
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Pass the argument attribute named after the op through unchanged as the
        # parameter.
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Load the scalar or pointed-to value using the utility helper, passing the
        # dtype only when one was given. Returns the loaded value as the per-tile state.
        if const_expr(self.dtype is not None):
            return utils.load_scalar_or_pointer(param, dtype=self.dtype)
        return utils.load_scalar_or_pointer(param)


class VecLoad(EpiOp):
    # [CLASS DOC] Base class for broadcast vector loads. Copies a per-tile vector from global
    # memory to shared memory with predicated async copies (elements past the valid extent are not
    # copied), then in begin_loop moves the needed slice to registers, converting to the
    # accumulator dtype unless convert_to_acc_dtype is false. Subclasses set dim to zero for a
    # column vector along M or one for a row vector along N; the class attribute
    # convert_to_acc_dtype defaults to true.
    """Base class for broadcast vector loads (row or col) via cp_async.

    Subclasses set `dim` to 0 (M/col) or 1 (N/row) and override `_get_gmem_vec`
    for varlen handling.
    """

    dim = None  # 0 for col (M), 1 for row (N)
    convert_to_acc_dtype = True

    def param_fields(self):
        # [FUNCTION DOC] Declare one parameter field named after the op, defaulting to None.
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Return the argument vector with stride divisibility assumptions applied,
        # keyed by the op name.
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def _tile_size(self, cta_tile_shape_mnk):
        # [FUNCTION DOC] HELPER: Return the CTA tile extent along this op's dimension (M for
        # columns, N for rows).
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        # [FUNCTION DOC] HELPER: Return the broadcast stride pair: zero stride along M with unit
        # stride along N for row vectors, the reverse for column vectors.
        # Row: stride (0,1) — broadcast along M. Col: stride (1,0) — broadcast along N.
        return (0, 1) if self.dim == 1 else (1, 0)

    def _tile_dim(self, ctx):
        # [FUNCTION DOC] HELPER: Return the tile extent in N for row vectors, in M for column
        # vectors, taken from the context.
        return ctx.tile_N if self.dim == 1 else ctx.tile_M

    def _coord_idx(self):
        # [FUNCTION DOC] HELPER: Return which tile coordinate entry indexes this vector: one for
        # row vectors, zero for column vectors.
        return 1 if self.dim == 1 else 0

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Return unstaged shared memory equal to the tile extent along this op's
        # dimension times the element size in bytes.
        return EpiSmemBytes(
            unstaged=self._tile_size(cta_tile_shape_mnk) * (arg_tensor.element_type.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Return the storage struct field named with an s_ prefix plus the op name:
        # a 16-byte aligned memory range of the vector element type sized to the tile extent.
        tensor = getattr(params, self.name)
        size = self._tile_size(gemm.cta_tile_shape_mnk)
        return (
            f"s_{self.name}",
            cute.struct.Align[cute.struct.MemRange[tensor.element_type, size], 16],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Return the vector's smem tensor from the epilogue storage as a one-
        # dimensional layout of tile extent.
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(self._tile_size(gemm.cta_tile_shape_mnk))
        )

    def needs_async_fence(self):
        # [FUNCTION DOC] Always true: the cp.async copies of the vector need a fence.
        return True

    def epi_m_major_score(self, arg_tensor, gemm):
        # [FUNCTION DOC] Prefer M-major order for row vectors (score 4, since keeping a row vector
        # in registers costs more) and mildly N-major for column vectors (score -1).
        # It costs more registers (say 4x) to keep rowvec in register vs keeping colvec in register
        return 4 if self.dim == 1 else -1

    def _get_gmem_vec(self, param, ctx):
        # [FUNCTION DOC] HELPER: Return the global vector for the current batch (the row of the
        # parameter tensor at the batch index). Subclasses override for variable-length M.
        """Get the global memory vector for this tile. Override for varlen."""
        return param[ctx.batch_idx, None]

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Copy this tile's slice of the vector from global to shared memory using a
        # one-dimensional tiled async copy over the epilogue threads. Each copy is predicated on
        # the index lying inside the valid limit (vector length minus tile start, capped at the
        # tile size) and skipped beyond the allocated smem. Then partitions the smem tensor,
        # viewed as a broadcast tile, for the epilogue (retiled for register-to-smem when a t2r
        # copy exists) and allocates a register tensor in the accumulator dtype or the original
        # dtype. Returns a list with the partitioned smem tensor and that register tensor.
        dtype = param.element_type
        num_copy_elems = const_expr(max(32, dtype.width)) // dtype.width
        thr_copy = copy_utils.tiled_copy_1d(
            dtype, ctx.num_epi_threads, num_copy_elems, is_async=True
        ).get_slice(ctx.tidx)
        mVec = self._get_gmem_vec(param, ctx)
        tile_dim = self._tile_dim(ctx)
        coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
        gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
        tVgV = thr_copy.partition_S(gVec)
        tVsV = thr_copy.partition_D(smem_tensor)
        tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_dim))
        limit = min(cute.size(mVec, mode=[0]) - coord_idx * tile_dim, tile_dim)
        for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
            if tVcV[0, m] < tile_dim:  # Guard to avoid writing beyond the smem we've allocated
                pred = cute.make_rmem_tensor(1, Boolean)
                pred[0] = tVcV[0, m] < limit
                cute.copy(thr_copy, tVgV[None, m], tVsV[None, m], pred=pred)
        tDsV = ctx.partition_for_epilogue_fn(
            cute.make_tensor(
                smem_tensor.iterator,
                cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
            )
        )
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDsV = ctx.tiled_copy_r2s.retile(tDsV)
        # Pre-allocate register tensor reused across begin_loop calls
        tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
        register_dtype = (
            gemm.acc_dtype
            if const_expr(self.convert_to_acc_dtype)
            else tDsV_sub.element_type
        )
        tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, register_dtype)
        return [tDsV, tDrV_cvt]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        # [FUNCTION DOC] Return the register tensor holding the current subtile's vector values.
        # The smem-to-register copy (with conversion to the accumulator dtype when enabled) only
        # happens when this subtile is the first in the direction orthogonal to the vector under
        # the active traversal order (first epi_m for row vectors under M-major, first epi_n for
        # column vectors otherwise); otherwise the previously loaded registers are reused.
        tDsV, tDrV_cvt = state[0], state[1]
        should_load = Boolean(True)
        if const_expr(self.dim == 1):
            if const_expr(gemm.epi_m_major):
                should_load = epi_coord[0] == 0
        else:
            if const_expr(not gemm.epi_m_major):
                should_load = epi_coord[1] == 0
        if should_load:
            tDsV_cur = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, epi_coord]
            tDrV = cute.make_rmem_tensor(tDsV_cur.layout, tDsV_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsV_cur), cute.filter_zeros(tDrV))
            if const_expr(self.convert_to_acc_dtype):
                tDrV_cvt.store(tDrV.load().to(gemm.acc_dtype))
            else:
                tDrV_cvt.store(tDrV.load())
        return tDrV_cvt


class RowVecLoad(VecLoad):
    # [CLASS DOC] Row vector of length N loaded through async copy and broadcast along M. Only
    # sets the dimension to one.
    """Loads a row vector (N,) via cp_async, broadcasts along M with stride (0,1)."""

    dim = 1


class ColVecLoad(VecLoad):
    # [CLASS DOC] Column vector of length M loaded through async copy and broadcast along N. Sets
    # the dimension to zero and supports variable-length M by offsetting into the concatenated
    # vector and using the per-batch M length for the validity limit.
    """Loads a col vector (M,) via cp_async, broadcasts along N with stride (1,0).

    Optimization: with N-major subtile loop, consecutive epi_n iterations for the same
    epi_m share the same column data. The smem→register copy only runs when epi_n == 0.
    Supports varlen_m via domain_offset.
    """

    dim = 0

    @cute.jit
    def _get_gmem_vec(self, param, ctx):
        # [FUNCTION DOC] HELPER: Return the global vector for the batch. Without varlen M it is
        # the row of the parameter at the batch index; with varlen M the parameter is offset by
        # the cumulative sequence length for the batch.
        if const_expr(not ctx.varlen_manager.varlen_m):
            mVec = param[ctx.batch_idx, None]
        else:
            mVec = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx],), param
            )
        return mVec

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Same as the base vector load setup, except the validity limit is computed
        # from the per-batch M length of the varlen manager rather than the tensor size. Copies
        # the column vector slice into smem with predication and returns a list of the partitioned
        # smem tensor and a register tensor for the converted values.
        dtype = param.element_type
        num_copy_elems = const_expr(max(32, dtype.width)) // dtype.width
        thr_copy = copy_utils.tiled_copy_1d(
            dtype, ctx.num_epi_threads, num_copy_elems, is_async=True
        ).get_slice(ctx.tidx)
        mVec = self._get_gmem_vec(param, ctx)
        tile_dim = self._tile_dim(ctx)
        coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
        gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
        tVgV = thr_copy.partition_S(gVec)
        tVsV = thr_copy.partition_D(smem_tensor)
        tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_dim))
        # ColVec uses varlen-aware limit
        limit = min(
            ctx.varlen_manager.len_m(ctx.batch_idx) - coord_idx * tile_dim,
            tile_dim,
        )
        for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
            if tVcV[0, m] < tile_dim:  # Guard to avoid writing beyond the smem we've allocated
                pred = cute.make_rmem_tensor(1, Boolean)
                pred[0] = tVcV[0, m] < limit
                cute.copy(thr_copy, tVgV[None, m], tVsV[None, m], pred=pred)
        tDsV = ctx.partition_for_epilogue_fn(
            cute.make_tensor(
                smem_tensor.iterator,
                cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
            )
        )
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDsV = ctx.tiled_copy_r2s.retile(tDsV)
        # Pre-allocate register tensor reused across begin_loop calls
        tDsV_sub = cute.group_modes(tDsV, 3, cute.rank(tDsV))[None, None, None, 0]
        register_dtype = (
            gemm.acc_dtype
            if const_expr(self.convert_to_acc_dtype)
            else tDsV_sub.element_type
        )
        tDrV_cvt = cute.make_rmem_tensor(tDsV_sub.layout, register_dtype)
        return [tDsV, tDrV_cvt]


class VecTupleLoad(EpiOp):
    # [CLASS DOC] Op that loads a tuple of broadcast vectors, delegating each element to a freshly
    # created single-vector op of the class named by vec_op_cls (the plain vector base by
    # default). Each vector gets its own smem buffer; state and values are tuples in the same
    # order.
    vec_op_cls = VecLoad

    def param_fields(self):
        # [FUNCTION DOC] Declare one parameter field named after the op, defaulting to None.
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Return the tuple of argument vectors with stride divisibility assumptions
        # applied, or None if the argument is None.
        tensors = getattr(args, self.name)
        if tensors is None:
            return {self.name: None}
        return {self.name: tuple(assume_stride_divisibility(tensor) for tensor in tensors)}

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Sum the smem requirements of the per-element vector ops over all tensors
        # in the tuple.
        return sum(
            (
                self.vec_op_cls(f"{self.name}{i}").smem_bytes(
                    tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk
                )
                for i, tensor in enumerate(arg_tensor)
            ),
            EpiSmemBytes(),
        )

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Build a storage struct with one aligned 16-byte memory range per vector,
        # sized by the element op tile extent, and return it under an s_ prefixed name.
        annotations = {}
        for i, tensor in enumerate(getattr(params, self.name)):
            size = self.vec_op_cls(f"{self.name}{i}")._tile_size(gemm.cta_tile_shape_mnk)
            annotations[f"v{i}"] = cute.struct.Align[
                cute.struct.MemRange[tensor.element_type, size], 16
            ]
        storage = type(f"{self.name}Storage", (), {"__annotations__": annotations})
        return (f"s_{self.name}", cute.struct(storage))

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Return a tuple of per-vector smem tensors taken from the nested storage
        # struct, each with one-dimensional tile-extent layout.
        storage = getattr(storage_epi, f"s_{self.name}")
        return tuple(
            getattr(storage, f"v{i}").get_tensor(
                cute.make_layout(
                    self.vec_op_cls(f"{self.name}{i}")._tile_size(gemm.cta_tile_shape_mnk)
                )
            )
            for i, _ in enumerate(getattr(params, self.name))
        )

    def needs_async_fence(self):
        # [FUNCTION DOC] Always true, since the vector copies are async.
        return True

    def epi_m_major_score(self, arg_tensor, gemm):
        # [FUNCTION DOC] Return the sum of the per-element vector scores.
        return sum(
            self.vec_op_cls(f"{self.name}{i}").epi_m_major_score(tensor, gemm)
            for i, tensor in enumerate(arg_tensor)
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Call begin of a per-element vector op for each vector with its smem
        # tensor and return the tuple of states.
        states = []
        for i, tensor in enumerate(param):
            states.append(
                self.vec_op_cls(f"{self.name}{i}").begin(gemm, tensor, smem_tensor[i], ctx)
            )
        return tuple(states)

    def begin_loop(self, gemm, state, epi_coord):
        # [FUNCTION DOC] Call begin_loop of a per-element vector op for each stored state and
        # return the tuple of values.
        values = []
        for i, tensor_state in enumerate(state):
            values.append(
                self.vec_op_cls(f"{self.name}{i}").begin_loop(gemm, tensor_state, epi_coord)
            )
        return tuple(values)


class CapturedRowVecLoad(RowVecLoad):
    # [CLASS DOC] Row vector load that keeps its original dtype in registers instead of converting
    # to the accumulator dtype.
    convert_to_acc_dtype = False


class CapturedColVecLoad(ColVecLoad):
    # [CLASS DOC] Column vector load that keeps its original dtype in registers instead of
    # converting to the accumulator dtype.
    convert_to_acc_dtype = False


class RowVecTupleLoad(VecTupleLoad):
    # [CLASS DOC] Tuple of row vector loads using the dtype-preserving element op.
    vec_op_cls = CapturedRowVecLoad


class ColVecTupleLoad(VecTupleLoad):
    # [CLASS DOC] Tuple of column vector loads using the dtype-preserving element op.
    vec_op_cls = CapturedColVecLoad


class CapturedScalarLoad(EpiOp):
    # [CLASS DOC] Op that reads the single element of a captured one-element tensor once per tile.
    # No smem.
    """Loads a captured (1,)-shaped tensor's element once per tile. No smem."""

    def param_fields(self):
        # [FUNCTION DOC] Declare one parameter field named after the op, defaulting to None.
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Pass the argument attribute through unchanged.
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Load the value of the captured tensor into registers and return it.
        return param.load()


class ScalarTupleLoad(EpiOp):
    # [CLASS DOC] Op that loads a tuple of captured one-element tensors once per tile, using one
    # scalar load op per element. No smem.
    """Loads a tuple of captured scalar tensors once per tile. No smem."""

    def param_fields(self):
        # [FUNCTION DOC] Declare one parameter field named after the op, defaulting to None.
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Pass the argument attribute through unchanged.
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Load each captured scalar tensor and return a tuple of the values.
        values = []
        for i, tensor in enumerate(param):
            values.append(
                CapturedScalarLoad(f"{self.name}{i}").begin(gemm, tensor, None, ctx)
            )
        return tuple(values)


class TileStore(EpiOp):
    # [CLASS DOC] Op for a tile-sized output tensor written through TMA, such as a post-
    # activation. An optional epi_tile_fn maps the gemm and its epilogue tile to a different tile
    # (for example half-width for gated output). The argument may be a single tensor or a tuple of
    # tensors; parameters hold the TMA atoms, TMA tensors, staged smem layouts and epilogue tiles
    # under names derived from the op name. Shared memory is accounted per D stage.
    """Tile-sized output tensor stored via TMA (e.g. postact).

    Args:
        name: field name in EpilogueArguments/Params (e.g. "mAuxOut")
        epi_tile_fn: optional (gemm, epi_tile) -> epi_tile for half-tile (GemmGated)
    """

    def __init__(self, name, epi_tile_fn=None):
        # [FUNCTION DOC] Store the name and the optional epilogue tile function.
        super().__init__(name)
        self.epi_tile_fn = epi_tile_fn

    def _tma_atom_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the TMA atom.
        return f"tma_atom_{self.name}"

    def _smem_layout_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the staged smem layout.
        return f"epi_{self.name}_smem_layout_staged"

    def _epi_tile_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for this op's epilogue tile.
        return f"epi_tile_{self.name}"

    def param_fields(self):
        # [FUNCTION DOC] Declare four parameter fields (TMA atom, tensor, staged smem layout,
        # epilogue tile), all defaulting to None so the parameter class can be built when the op
        # is inactive.
        # Defaults are None so EpilogueParams can be constructed when this op is
        # filtered out (inactive). Active calls always set all four via to_params.
        return [
            (self._tma_atom_key(), object, None),
            (self.name, object, None),
            (self._smem_layout_key(), object, None),
            (self._epi_tile_key(), object, None),
        ]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Prepare TMA store metadata for the output argument. Applies the optional
        # tile function to the gemm epilogue tile, then runs the epilogue tensor setup helper per
        # tensor. Returns a mapping with the four keys; for a tuple argument each value is a tuple
        # with one entry per tensor.
        tensors = getattr(args, self.name)
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        if isinstance(tensors, tuple):
            prepared = tuple(
                setup_epi_tensor(gemm, tensor, epi_tile=epi_tile) for tensor in tensors
            )
            return {
                self._tma_atom_key(): tuple(item[0] for item in prepared),
                self.name: tuple(item[1] for item in prepared),
                self._smem_layout_key(): tuple(item[2] for item in prepared),
                self._epi_tile_key(): tuple(item[3] for item in prepared),
            }
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensors, epi_tile=epi_tile
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
        }

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Return D-stage bytes equal to the epilogue tile element count times
        # element size in bytes (summed over tensors for a tuple argument). The tile function, if
        # any, is applied with no gemm object first.
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        # epi_tile may contain Layout entries (from SM100's compute_epilogue_tile_shape
        # fixup path), so extract the int shape first.
        if isinstance(arg_tensor, tuple):
            return sum(
                (
                    EpiSmemBytes(
                        d_stage=cute.size(cute.shape(epi_tile))
                        * (tensor.element_type.width // 8)
                    )
                    for tensor in arg_tensor
                ),
                EpiSmemBytes(),
            )
        return EpiSmemBytes(
            d_stage=cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Return the storage struct field for the output smem buffer, aligned to
        # the gemm buffer alignment, sized by the cosize of the staged layout and typed by the
        # gemm aux output dtype. For a tuple of layouts, builds a nested struct with one buffer
        # per output typed by the per-output dtypes.
        smem_layout = getattr(params, self._smem_layout_key())
        if isinstance(smem_layout, tuple):
            annotations = {}
            for i, layout in enumerate(smem_layout):
                annotations[f"v{i}"] = cute.struct.Align[
                    cute.struct.MemRange[
                        gemm.aux_out_dtypes[i],
                        cute.cosize(layout),
                    ],
                    gemm.buffer_align_bytes,
                ]
            storage = type(f"{self.name}Storage", (), {"__annotations__": annotations})
            return (f"s_{self.name}", cute.struct(storage))
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[
                    gemm.aux_out_dtype,
                    cute.cosize(smem_layout),
                ],
                gemm.buffer_align_bytes,
            ],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Return the smem tensor (or tuple of them) built from the storage using
        # the outer layout and the swizzle held in the staged layout.
        smem_layout = getattr(params, self._smem_layout_key())
        storage = getattr(storage_epi, f"s_{self.name}")
        if isinstance(smem_layout, tuple):
            return tuple(
                getattr(storage, f"v{i}").get_tensor(
                    layout.outer,
                    swizzle=layout.inner,
                )
                for i, layout in enumerate(smem_layout)
            )
        return storage.get_tensor(
            smem_layout.outer,
            swizzle=smem_layout.inner,
        )

    def tma_atoms(self, gemm, params):
        # [FUNCTION DOC] Return the TMA atom(s) as a list: the list form of a tuple, otherwise a
        # one-element list.
        tma_atom = getattr(params, self._tma_atom_key())
        return list(tma_atom) if isinstance(tma_atom, tuple) else [tma_atom]


class _TileLoadState(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple of per-tile state of a tile load: the smem-to-register tiled
    # copy, the register tile shaped like the main output register layout, the register retile
    # target for the copy and the staged smem source.
    """Per-tile register state produced by TileLoad.begin and consumed by load_s2r /
    begin_loop. tRS_rTile is the register tile partitioned to match tRS_rD's layout;
    tSR_sTile / tSR_rTile drive the per-stage smem→register copy."""

    tiled_copy_s2r: object
    tRS_rTile: object
    tSR_rTile: object
    tSR_sTile: object


class TileLoad(EpiOp):
    # [CLASS DOC] Op for an auxiliary tile-sized input loaded via TMA through the same staged
    # pipeline as the C operand and exposed to the epilogue visitor by name. Smem is accounted per
    # C stage. Several tile loads can coexist, each with its own TMA descriptor and buffer.
    # Supported only for architectures 90, 100 and 120. The layout and element type of the
    # argument are saved on the gemm object during to_params because they cannot be recovered
    # after TMA preparation.
    """Tile-sized auxiliary input loaded through the epilogue load pipeline.

    TileLoad uses the same staged gmem->smem->register pipeline as GEMM's C operand,
    but it is exposed to the epilogue as ``epi_loop_tensors[name]`` instead of as
    ``tRS_rC``. That lets custom epilogues consume extra MxN tensors without using
    the GEMM C argument.

    Its shared memory is accounted as ``EpiSmemBytes.c_stage``, so it is allocated
    per epilogue load stage. Multiple TileLoads are supported: each has its own TMA
    descriptor and smem buffer, and the pipeline transaction count includes C plus
    all enabled TileLoad buffers. Supported on SM90, SM100, and SM120.
    """

    def __init__(self, name, epi_tile_fn=None):
        # [FUNCTION DOC] Store the name and the optional epilogue tile function.
        super().__init__(name)
        self.epi_tile_fn = epi_tile_fn

    def _tma_atom_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the TMA atom.
        return f"tma_atom_{self.name}"

    def _smem_layout_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the staged smem layout.
        return f"epi_{self.name}_smem_layout_staged"

    def _epi_tile_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the epilogue tile.
        return f"epi_tile_{self.name}"

    # The original LayoutEnum and element_type can't be recovered from the
    # TMA-prepared tensor that ends up in params (`from_tensor` returns a typing
    # annotation post-TMA, not a Numeric class). We stash both on the gemm at
    # to_params time and read them back in begin(). The dtype is also exposed on
    # the params dataclass for smem_struct_field.
    def _layout_gemm_attr(self):
        # [FUNCTION DOC] HELPER: Return the attribute name on the gemm object under which the
        # original layout enum is stashed.
        return f"_tile_load_layout_{self.name}"

    def _dtype_gemm_attr(self):
        # [FUNCTION DOC] HELPER: Return the attribute name on the gemm object under which the
        # original element type is stashed.
        return f"_tile_load_dtype_{self.name}"

    def _dtype_field(self):
        # [FUNCTION DOC] HELPER: Return the parameter field name that carries the element type.
        return f"{self.name}_dtype"

    def param_fields(self):
        # [FUNCTION DOC] Declare five parameter fields (TMA atom, tensor, staged smem layout,
        # epilogue tile, dtype), all defaulting to None.
        # Defaults are None so EpilogueParams can be constructed when this op is
        # filtered out (inactive). Active calls always set all five via to_params.
        return [
            (self._tma_atom_key(), object, None),
            (self.name, object, None),
            (self._smem_layout_key(), object, None),
            (self._epi_tile_key(), object, None),
            (self._dtype_field(), object, None),
        ]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Prepare a TMA load of the argument tensor. Side effect: sets two
        # attributes on the gemm object recording the tensor layout enum and element type. Builds
        # TMA atom, tensor, staged layout (using the epilogue C stage count) and tile via the
        # setup helper, and returns a mapping of all five parameter entries.
        tensor = getattr(args, self.name)
        setattr(gemm, self._layout_gemm_attr(), cutlass.utils.LayoutEnum.from_tensor(tensor))
        setattr(gemm, self._dtype_gemm_attr(), tensor.element_type)
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile, op_type="load", stage=gemm.epi_c_stage
        )
        return {
            self._tma_atom_key(): tma_atom,
            self.name: tma_tensor,
            self._smem_layout_key(): smem_layout,
            self._epi_tile_key(): epi_tile_out,
            self._dtype_field(): tensor.element_type,
        }

    def is_tile_load(self):
        # [FUNCTION DOC] Return true: this op goes through the C load pipeline.
        return True

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Return C-stage bytes equal to the epilogue tile element count times
        # element size in bytes, applying the tile function first when present.
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        # epi_tile may contain Layout entries from SM100's compute_epilogue_tile_shape
        # fixup; extract the int shape first.
        return EpiSmemBytes(
            c_stage=cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Return the storage struct field for the staged smem buffer, aligned to
        # the gemm buffer alignment, typed by the dtype in the parameters and sized by the layout
        # cosize.
        smem_layout = getattr(params, self._smem_layout_key())
        dtype = getattr(params, self._dtype_field())
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(smem_layout)],
                gemm.buffer_align_bytes,
            ],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Return the smem tensor built from the storage with the outer layout and
        # the swizzle of the staged layout.
        smem_layout = getattr(params, self._smem_layout_key())
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            smem_layout.outer,
            swizzle=smem_layout.inner,
        )

    def tma_atoms(self, gemm, params):
        # [FUNCTION DOC] Return a one-element list with this op's TMA atom.
        return [getattr(params, self._tma_atom_key())]

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        # [FUNCTION DOC] Build the producer-side TMA copy function. Uses the gemm helper to
        # partition the per-batch tensor (offset by the varlen manager) over the CTA tile, then
        # wraps the resulting tile copy so it is issued under the epilogue pipeline. Returns that
        # wrapped function.
        tensor = getattr(params, self.name)
        batch_idx = tile_coord_mnkl[3]
        copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
            getattr(params, self._tma_atom_key()),
            varlen_manager.offset_batch_epi(tensor, batch_idx),
            gemm.cta_tile_shape_mnk[:2],
            getattr(params, self._epi_tile_key()),
            smem_tensor,
            tile_coord_mnkl,
        )
        return copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Set up the smem-to-register load. Requires architecture 90, 100 or 120
        # and a known main output register layout, else an assertion fails. Uses the t2r tiled
        # copy as the reference on architecture 100 and the tiled mma otherwise. Returns the tile
        # load state tuple.
        assert gemm.arch in (90, 100, 120), "TileLoad requires the SM90/SM100/SM120 epilogue path"
        assert ctx.tRS_rD_layout is not None
        smem_load_ref = ctx.tiled_copy_t2r if const_expr(gemm.arch == 100) else gemm.tiled_mma
        tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile = gemm.epilog_smem_load_and_partition(
            smem_load_ref,
            getattr(gemm, self._layout_gemm_attr()),
            getattr(gemm, self._dtype_gemm_attr()),
            smem_tensor,
            ctx.tRS_rD_layout,
            ctx.tidx,
        )
        # Shape: (s2r-copy-handle, register-tile-as-rD-layout, smem→r retile target,
        # smem→r staged source). begin_loop returns tRS_rTile; load_s2r uses the rest.
        return _TileLoadState(tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile)

    @cute.jit
    def load_s2r(self, gemm, param, state, stage_idx):
        # [FUNCTION DOC] Copy the given stage of the smem tile into the register retile target
        # with the s2r tiled copy.
        cute.copy(
            state.tiled_copy_s2r,
            state.tSR_sTile[None, None, None, stage_idx],
            state.tSR_rTile,
        )

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        # [FUNCTION DOC] Return the register tile partitioned like the main output; the subtile
        # coordinate is unused.
        return state.tRS_rTile


class TileTupleLoad(EpiOp):
    # [CLASS DOC] Op for a tuple of auxiliary tile loads, each with its own TMA atom, staged smem
    # buffer and epilogue tile. Layout and element type of each argument are stashed on the gemm
    # object under index-specific attribute names. Smem is accounted per C stage. Requires
    # architecture 90, 100 or 120.
    def _tma_atoms_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the tuple of TMA atoms.
        return f"tma_atoms_{self.name}"

    def _smem_layouts_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the tuple of staged smem layouts.
        return f"epi_{self.name}_smem_layouts_staged"

    def _epi_tiles_key(self):
        # [FUNCTION DOC] HELPER: Return the parameter key for the tuple of epilogue tiles.
        return f"epi_tiles_{self.name}"

    def _layout_gemm_attr(self, index):
        # [FUNCTION DOC] HELPER: Return the gemm attribute name storing the layout enum of tensor
        # number index.
        return f"_tile_tuple_load_layout_{self.name}_{index}"

    def _dtype_gemm_attr(self, index):
        # [FUNCTION DOC] HELPER: Return the gemm attribute name storing the element type of tensor
        # number index.
        return f"_tile_tuple_load_dtype_{self.name}_{index}"

    def param_fields(self):
        # [FUNCTION DOC] Declare four parameter fields (TMA atoms, tensors, staged layouts,
        # tiles), all defaulting to None.
        return [
            (self._tma_atoms_key(), object, None),
            (self.name, object, None),
            (self._smem_layouts_key(), object, None),
            (self._epi_tiles_key(), object, None),
        ]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Prepare TMA loads for every tensor in the argument tuple. Side effect:
        # records layout enum and element type for each tensor on the gemm object. Always uses the
        # default epilogue tile and the C stage count. Returns a mapping with four keys whose
        # values are tuples.
        tensors = getattr(args, self.name)
        tma_atoms, tma_tensors, smem_layouts, epi_tiles = [], [], [], []
        for i, tensor in enumerate(tensors):
            setattr(gemm, self._layout_gemm_attr(i), cutlass.utils.LayoutEnum.from_tensor(tensor))
            setattr(gemm, self._dtype_gemm_attr(i), tensor.element_type)
            tma_atom, tma_tensor, smem_layout, epi_tile = setup_epi_tensor(
                gemm, tensor, epi_tile=None, op_type="load", stage=gemm.epi_c_stage
            )
            tma_atoms.append(tma_atom)
            tma_tensors.append(tma_tensor)
            smem_layouts.append(smem_layout)
            epi_tiles.append(epi_tile)
        return {
            self._tma_atoms_key(): tuple(tma_atoms),
            self.name: tuple(tma_tensors),
            self._smem_layouts_key(): tuple(smem_layouts),
            self._epi_tiles_key(): tuple(epi_tiles),
        }

    def is_tile_load(self):
        # [FUNCTION DOC] Return true: these loads go through the C load pipeline.
        return True

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Return the sum over tensors of C-stage bytes, from the epilogue tile
        # element count times element width in bits divided by eight.
        return sum(
            (
                EpiSmemBytes(
                    c_stage=cute.size(cute.shape(epi_tile)) * tensor.element_type.width // 8
                )
                for tensor in arg_tensor
            ),
            EpiSmemBytes(),
        )

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Return a nested storage struct with one aligned memory range per tensor,
        # typed by the stashed element type and sized by the layout cosize, under an s_ prefixed
        # name.
        annotations = {}
        for i, smem_layout in enumerate(getattr(params, self._smem_layouts_key())):
            annotations[f"v{i}"] = cute.struct.Align[
                cute.struct.MemRange[
                    getattr(gemm, self._dtype_gemm_attr(i)), cute.cosize(smem_layout)
                ],
                gemm.buffer_align_bytes,
            ]
        storage = type(f"{self.name}Storage", (), {"__annotations__": annotations})
        return (f"s_{self.name}", cute.struct(storage))

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Return a tuple of smem tensors, one per staged layout, using outer layout
        # and swizzle.
        storage = getattr(storage_epi, f"s_{self.name}")
        return tuple(
            getattr(storage, f"v{i}").get_tensor(smem_layout.outer, swizzle=smem_layout.inner)
            for i, smem_layout in enumerate(getattr(params, self._smem_layouts_key()))
        )

    def tma_atoms(self, gemm, params):
        # [FUNCTION DOC] Return the TMA atoms as a list.
        return list(getattr(params, self._tma_atoms_key()))

    def load_g2s_copy_fn(self, gemm, params, smem_tensor, tile_coord_mnkl, varlen_manager, epi_pipeline):
        # [FUNCTION DOC] Build one producer TMA copy function per tensor (partitioning the per-
        # batch offset tensor over the CTA tile with its own atom, tile and smem buffer) and chain
        # them into a single function.
        copy_fns = []
        for tma_atom, tensor, epi_tile, smem in zip(
            getattr(params, self._tma_atoms_key()),
            getattr(params, self.name),
            getattr(params, self._epi_tiles_key()),
            smem_tensor,
        ):
            copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
                tma_atom,
                varlen_manager.offset_batch_epi(tensor, tile_coord_mnkl[3]),
                gemm.cta_tile_shape_mnk[:2],
                epi_tile,
                smem,
                tile_coord_mnkl,
            )
            copy_fns.append(copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline))
        return copy_utils.chain_tma_producer_copy_fns(tuple(copy_fns))

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Set up one smem-to-register load per smem buffer. Asserts architecture
        # 90, 100 or 120. Reference copy is the t2r copy on architecture 100, else the tiled mma.
        # Returns a tuple of tile load states.
        assert gemm.arch in (90, 100, 120), "TileTupleLoad requires the SM90/SM100/SM120 epilogue path"
        states = []
        smem_load_ref = ctx.tiled_copy_t2r if const_expr(gemm.arch == 100) else gemm.tiled_mma
        for i, smem in enumerate(smem_tensor):
            tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile = gemm.epilog_smem_load_and_partition(
                smem_load_ref,
                getattr(gemm, self._layout_gemm_attr(i)),
                getattr(gemm, self._dtype_gemm_attr(i)),
                smem,
                ctx.tRS_rD_layout,
                ctx.tidx,
            )
            states.append(_TileLoadState(tiled_copy_s2r, tRS_rTile, tSR_rTile, tSR_sTile))
        return tuple(states)

    @cute.jit
    def load_s2r(self, gemm, param, state, stage_idx):
        # [FUNCTION DOC] For every tile state, copy the given stage of its smem tile to its
        # registers.
        for tile_state in state:
            cute.copy(
                tile_state.tiled_copy_s2r,
                tile_state.tSR_sTile[None, None, None, stage_idx],
                tile_state.tSR_rTile,
            )

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        # [FUNCTION DOC] Return a tuple of the register tiles of all loaded tensors.
        return tuple(tile_state.tRS_rTile for tile_state in state)


@cute.jit
def vec_multiply(gemm, tRS_rD, tDrColVec, tDrRowVec):
    # [FUNCTION DOC] Multiply the accumulator register tensor in place by a column vector and then
    # by a row vector register tensor, skipping whichever is None. On architecture 100 uses packed
    # two-wide float multiplies over element pairs; otherwise elementwise multiplication.
    """Multiply tRS_rD by colvec and/or rowvec in-place. Uses packed f32x2 on SM100."""
    if const_expr(tDrColVec is not None):
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] *= tDrColVec[i]
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                tRS_rD[2 * i], tRS_rD[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    (tDrColVec[2 * i], tDrColVec[2 * i + 1]),
                )
    if const_expr(tDrRowVec is not None):
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] *= tDrRowVec[i]
        else:
            for i in cutlass.range(cute.size(tRS_rD) // 2, unroll_full=True):
                tRS_rD[2 * i], tRS_rD[2 * i + 1] = cute.arch.mul_packed_f32x2(
                    (tRS_rD[2 * i], tRS_rD[2 * i + 1]),
                    (tDrRowVec[2 * i], tDrRowVec[2 * i + 1]),
                )


@cute.jit
def colvec_reduce_accumulate(gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None):
    # [FUNCTION DOC] Accumulate into a column-vector reduction buffer (no-op if the buffer is
    # None). Each input element is optionally transformed (default identity) and optionally
    # multiplied by a scale tensor before being added. On architectures other than 100 this is
    # elementwise. On architecture 100 the buffers are viewed in a layout with the reduced mode
    # flattened and values are processed in pairs with packed multiply, fused multiply-add and
    # add, and the pair is folded into the first column entry of each row; requires an even
    # reduced extent. In that path the transform receives a pair of values.
    """Accumulate transform_fn(input) or input * rScale into a ColVecReduce buffer.

    If transform_fn is provided, accumulates transform_fn(input[i]).
    If rScale is provided, accumulates input[i] * rScale[i] (uses packed mul/fma for SM100).
    If neither, accumulates input directly (identity).
    """
    if const_expr(tDrReduce is not None):
        if const_expr(transform_fn is None):
            transform_fn = lambda x: x
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val
        else:
            tDrReduce_mn = layout_utils.convert_layout_zero_stride(tDrReduce, tDrReduce.layout)
            tRS_rInput_mn = layout_utils.convert_layout_zero_stride(tRS_rInput, tDrReduce.layout)
            if const_expr(rScale is not None):
                rScale_mn = layout_utils.convert_layout_zero_stride(rScale, tDrReduce.layout)
            for m in cutlass.range(cute.size(tDrReduce_mn, mode=[0]), unroll_full=True):
                inp = lambda n: (tRS_rInput_mn[m, 2 * n], tRS_rInput_mn[m, 2 * n + 1])
                val0 = transform_fn(inp(0))
                assert cute.size(tDrReduce_mn, mode=[1]) % 2 == 0
                if const_expr(rScale is not None):
                    row_sum = cute.arch.mul_packed_f32x2(val0, (rScale_mn[m, 0], rScale_mn[m, 1]))
                else:
                    row_sum = val0
                for n in cutlass.range(1, cute.size(tDrReduce_mn, mode=[1]) // 2, unroll_full=True):
                    val = transform_fn(inp(n))
                    if const_expr(rScale is not None):
                        row_sum = cute.arch.fma_packed_f32x2(
                            val, (rScale_mn[m, 2 * n], rScale_mn[m, 2 * n + 1]), row_sum
                        )
                    else:
                        row_sum = cute.arch.add_packed_f32x2(val, row_sum)
                tDrReduce_mn[m, 0] += row_sum[0] + row_sum[1]


@cute.jit
def rowvec_reduce_accumulate(gemm, tDrReduce, tRS_rInput, transform_fn=None, rScale=None):
    # [FUNCTION DOC] Accumulate into a row-vector reduction buffer (reduces along M, no-op if the
    # buffer is None). Each input is optionally transformed (default identity) and optionally
    # scaled before being added. Non-100 architectures work elementwise. Architecture 100
    # processes pairs with packed fused multiply-add or add and handles an odd trailing element
    # separately; the transform receives a pair there. Relies on the buffer layout being such that
    # paired entries do not alias.
    """Accumulate transform_fn(input) or input * rScale into a RowVecReduce buffer.

    Reduces along M dimension, keeping N. The zero-stride layout on M ensures
    elements at different M positions but same N column accumulate correctly.
    """
    if const_expr(tDrReduce is not None):
        if const_expr(transform_fn is None):
            transform_fn = lambda x: x
        if const_expr(gemm.arch != 100):
            for i in cutlass.range(cute.size(tDrReduce), unroll_full=True):
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val
        else:
            # Keep CUTLASS's linear fragment indexing, but use packed f32x2 arithmetic
            # for any transform that accepts and returns an f32x2 tuple.
            # We have to be careful to avoid tDrReduce[2 * i] and tDrReduce[2 * i + 1] aliasing
            # each other. For SM100, tDrReduce has layout ((32,1),1,1):((1,0),0,0) or
            # (((2,2,4),1),2,1):(((1,0,8),0),0,0), so this works. But it's error-prone.
            for i in cutlass.range(cute.size(tRS_rInput) // 2, unroll_full=True):
                acc = (tDrReduce[2 * i], tDrReduce[2 * i + 1])
                val = (tRS_rInput[2 * i], tRS_rInput[2 * i + 1])
                val = transform_fn(val)
                if const_expr(rScale is not None):
                    scale = (rScale[2 * i], rScale[2 * i + 1])
                    tDrReduce[2 * i], tDrReduce[2 * i + 1] = cute.arch.fma_packed_f32x2(
                        val, scale, acc
                    )
                else:
                    tDrReduce[2 * i], tDrReduce[2 * i + 1] = cute.arch.add_packed_f32x2(val, acc)
            if const_expr(cute.size(tRS_rInput) % 2 != 0):
                i = cute.size(tRS_rInput) - 1
                val = transform_fn(tRS_rInput[i])
                tDrReduce[i] += val * rScale[i] if const_expr(rScale is not None) else val


class VecReduce(EpiOp):
    # [CLASS DOC] Base class for row or column vector reductions to global output. Register
    # accumulator tensors are created in float32 with zero-stride layout along the reduced
    # dimension; shared memory (float32, one slot per extra reducing warp per tile element) is
    # used to combine partial results across warps. Class attributes: dim (zero for a column
    # output along M, one for a row output along N) and the subtile-order preference.
    """Base class for row/column vector reductions."""

    dim = 0  # 0 for colvec output along M, 1 for rowvec output along N
    epi_m_major_preference = 0

    def param_fields(self):
        # [FUNCTION DOC] Declare one parameter field named after the op, defaulting to None.
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Return the output vector argument with stride divisibility assumptions
        # applied, keyed by the op name.
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def epi_m_major_score(self, arg_tensor, gemm):
        # [FUNCTION DOC] Return the class-level subtile-order preference.
        return self.epi_m_major_preference

    def _tile_size(self, cta_tile_shape_mnk):
        # [FUNCTION DOC] HELPER: Return the CTA tile extent along this op's output dimension.
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        # [FUNCTION DOC] HELPER: Return the broadcast stride pair used for the register layout:
        # unit stride on M for column outputs, unit stride on N for row outputs.
        # Col: stride (1,0) broadcasts along N. Row: stride (0,1) broadcasts along M.
        return (1, 0) if self.dim == 0 else (0, 1)

    def _reduce_dim(self):
        # [FUNCTION DOC] HELPER: Return the dimension being reduced, the opposite of the output
        # dimension.
        return 1 - self.dim

    def _smem_warps(self, warp_shape_mnk):
        # [FUNCTION DOC] HELPER: Return how many warps along the reduced dimension need smem
        # slots, which is the warp count there minus one (at least zero), or zero when the warp
        # shape is unknown.
        warps = warp_shape_mnk[self._reduce_dim()] if warp_shape_mnk is not None else 1
        return max(warps - 1, 0)

    def _uses_smem(self, gemm):
        # [FUNCTION DOC] HELPER: Report whether smem is used; the base class always says yes,
        # subclasses may override.
        return True

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] Return unstaged bytes of float32 for tile extent times the number of smem
        # warps, or an empty byte count when there are no smem warps.
        smem_warps = self._smem_warps(warp_shape_mnk)
        if smem_warps == 0:
            return EpiSmemBytes()
        return EpiSmemBytes(
            unstaged=self._tile_size(cta_tile_shape_mnk) * smem_warps * (Float32.width // 8)
        )

    def smem_struct_field(self, gemm, params):
        # [FUNCTION DOC] Return the storage struct field (16-byte aligned float32 range of tile
        # extent times smem warps) or None when no smem warps are needed or smem is unused.
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if smem_warps == 0 or not self._uses_smem(gemm):
            return None
        size = self._tile_size(gemm.cta_tile_shape_mnk) * smem_warps
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[Float32, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        # [FUNCTION DOC] Return the smem tensor with layout tile extent by smem warps, or None
        # when no smem is needed.
        smem_warps = self._smem_warps(gemm.epi_smem_warp_shape_mnk())
        if smem_warps == 0 or not self._uses_smem(gemm):
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout((self._tile_size(gemm.cta_tile_shape_mnk), smem_warps))
        )

    @cute.jit
    # [FUNCTION DOC] Create a float32 register accumulator whose layout is the epilogue partition
    # of a broadcast tile with zero stride along the reduced dimension. Returns a pair of the
    # register accumulator and the smem tensor.
    def begin(self, gemm, param, smem_tensor, ctx):
        vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride())
        tDrReduce_layout = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(vec_mma_layout, Float32)
        ).layout
        tDrReduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
        return (tDrReduce, smem_tensor)

    @cute.jit
    # [FUNCTION DOC] Return the accumulator slice for the subtile coordinate. When the subtile is
    # the first along the reduced direction, the slice is zeroed first so a new reduction starts.
    # Returns the slice view.
    def begin_loop(self, gemm, state, epi_coord):
        tDrReduce = state[0]
        result = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
        if const_expr(epi_coord[self._reduce_dim()] == 0):
            cute.filter_zeros(result).fill(0.0)
        return result


class _GroupedLocalReduceParams(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple of parameters for a grouped local reduce: output tensor,
    # combine function, finalize function and a flag saying whether the result feeds the main
    # output.
    tensor: object
    combine_fn: object
    finalize_fn: object
    feeds_main: bool


class _GroupedLocalReduceFeedState(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple of state for feeding the main output: lane layout over M and
    # N and the optional local reduce state.
    lane_layout_MN: object
    local_reduce: object | None


@cute.jit
def grouped_rowvec_reduce_value(gemm, tRS_rInput, row_state, combine_fn, finalize_fn):
    # [FUNCTION DOC] Compute grouped reductions along M within one warp and broadcast the result
    # to every row lane in the group. Copies the input into a new register tensor, then for each
    # element folds across lanes by butterfly shuffles with halving row offsets using the combine
    # function, then applies the finalize function to each element. Returns None if row_state is
    # None. Asserts the group divides the lane count along M and fits in it, and that the lane
    # stride along N is one when more than one lane is in N.
    """Return same-warp grouped-M reductions broadcast to every row lane.

    Feed-main has no cross-warp replay phase, so ``group`` must fit within the
    current lane-layout M extent; cross-warp M stitching is only implemented for
    compressed aux stores.
    """
    result = None
    if const_expr(row_state is not None):
        group = const_expr(gemm.local_reduce_group)
        lane_layout_MN = row_state.lane_layout_MN
        lanes_in_M = cute.size(lane_layout_MN, mode=[0])
        lanes_in_N = cute.size(lane_layout_MN, mode=[1])
        assert group <= lanes_in_M
        assert lanes_in_M % group == 0
        if const_expr(lanes_in_N > 1):
            assert lane_layout_MN.stride[1] == 1
        result = cute.make_rmem_tensor_like(tRS_rInput, tRS_rInput.element_type)
        cute.autovec_copy(tRS_rInput, result)
        result_flt = cute.filter_zeros(result)
        if const_expr(group > 1):
            for i in cutlass.range(cute.size(result_flt), unroll_full=True):
                reduction_rows = group // 2
                while reduction_rows > 0:
                    result_flt[i] = combine_fn(
                        result_flt[i],
                        cute.arch.shuffle_sync_bfly(
                            result_flt[i],
                            offset=cute.crd2idx((reduction_rows, 0), lane_layout_MN),
                        ),
                    )
                    reduction_rows = reduction_rows // 2
        for i in cutlass.range(cute.size(result_flt), unroll_full=True):
            result_flt[i] = finalize_fn(result_flt[i])
    return result


class GroupedLocalReduce(VecReduce):
    # [CLASS DOC] Vector reduce variant for grouped local reductions, which output compressed
    # values over groups of rows or columns. Output dimension is one with an N-major preference.
    # Parameters bundle the output tensor with the combine and finalize functions and feeds-main
    # flag. Uses smem only when the reduction crosses lane fragments.
    """Store or broadcast generated grouped local-reduce values."""

    dim = 1
    epi_m_major_preference = -1

    def _uses_smem(self, gemm):
        # [FUNCTION DOC] HELPER: Return whether grouped reductions on the gemm local reduce axis
        # and group size need smem.
        return grouped_local_reduce_uses_smem(
            gemm.local_reduce_axis, gemm.local_reduce_group
        )

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        # [FUNCTION DOC] No shared memory is accounted for statically; returns an empty byte
        # count.
        return EpiSmemBytes()

    def to_params(self, gemm, args):
        # [FUNCTION DOC] Bundle the output argument with the combine function, finalize function
        # and feeds-main flag taken from the arguments into a parameter tuple, keyed by the op
        # name.
        return {
            self.name: _GroupedLocalReduceParams(
                getattr(args, self.name),
                args.local_reduce_combine_fn,
                args.local_reduce_finalize_fn,
                args.local_reduce_feeds_main,
            )
        }

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        # [FUNCTION DOC] Set up per-tile state; returns None when the param is None. Reads group
        # size and reduce axis from the gemm and asserts the group is nonzero, no larger than the
        # tile extent along that axis and divides it. For full-N groups on architecture 100 along
        # N asserts only one warp partition exists along N. When the result feeds the main output
        # (requires axis zero) the state is a feed state with the lane layout and, if an output
        # tensor exists, a float32 register accumulator; otherwise the state is a pair of a
        # float32 register accumulator and the smem tensor.
        result = None
        if const_expr(param is not None):
            group = const_expr(gemm.local_reduce_group)
            axis = const_expr(gemm.local_reduce_axis)
            tile_M, tile_N = ctx.tile_M, ctx.tile_N
            tile = tile_N if const_expr(axis == 1) else tile_M
            assert group != 0 and group <= tile
            assert tile % group == 0
            tiled_copy = (
                ctx.tiled_copy_t2r
                if ctx.tiled_copy_t2r is not None
                else ctx.tiled_copy_r2s
            )
            if const_expr(gemm.arch == 100 and axis == 1 and group == tile_N):
                _, warp_layout_MN = _get_lane_warp_layouts(
                    tiled_copy, ctx.tiled_copy_t2r is None
                )
                warps_in_N = const_expr(cute.size(warp_layout_MN, mode=[1]))
                assert warps_in_N == 1, (
                    "full-N GroupedLocalReduce requires one epilogue N-warp partition"
                )
            if const_expr(param.feeds_main):
                assert axis == 0
                lane_layout_MN, _ = _get_lane_warp_layouts(
                    tiled_copy, ctx.tiled_copy_t2r is None
                )
                local_reduce = None
                if const_expr(param.tensor is not None):
                    vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N))
                    tDrReduce_layout = ctx.partition_for_epilogue_fn(
                        cute.make_rmem_tensor(vec_mma_layout, Float32)
                    ).layout
                    local_reduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
                result = _GroupedLocalReduceFeedState(lane_layout_MN, local_reduce)
            else:
                vec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N))
                tDrReduce_layout = ctx.partition_for_epilogue_fn(
                    cute.make_rmem_tensor(vec_mma_layout, Float32)
                ).layout
                result = (cute.make_rmem_tensor(tDrReduce_layout, Float32), smem_tensor)
        return result

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        # [FUNCTION DOC] Return the per-subtile value for the visitor. Returns None when state is
        # None. In feed-main mode it returns the state itself when there is no accumulator,
        # otherwise slices the accumulator at the subtile coordinate, zeroes it and returns a feed
        # state carrying the slice. Otherwise slices the accumulator at the subtile coordinate,
        # zeroes it and returns the slice.
        result = None
        if const_expr(state is not None):
            if const_expr(gemm.local_reduce_feeds_main):
                if const_expr(state.local_reduce is None):
                    return state
                tDrReduce_cur = state.local_reduce[
                    None, None, None, epi_coord[0], epi_coord[1]
                ]
                cute.filter_zeros(tDrReduce_cur).fill(0.0)
                return _GroupedLocalReduceFeedState(
                    state.lane_layout_MN, tDrReduce_cur
                )
            tDrReduce = state[0]
            result = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
            cute.filter_zeros(result).fill(0.0)
        return result

    @cute.jit
    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] Finish grouped reduction for one subtile and write results to the global
        # output. Does nothing without params or when the op only feeds the main output and has no
        # tensor. Three paths: for N-axis groups wider than the local fragment, wait until the
        # last fragment of a group, then combine the fragments of the group in registers,
        # finalize, convert to the output dtype and have the designated thread write within the
        # valid row and group limits. For M-axis groups, first fold across lanes by butterfly
        # shuffles with the combine function; if groups span warps, stage per-warp partials in
        # shared memory, synchronize on the epilogue barrier, have the first warp of each group
        # combine, finalize and write, then synchronize again and return. Otherwise a generic path
        # finalizes each group leader value and writes it, with a vectorized four-wide float32
        # store fast path for M-axis when alignment and divisibility allow. Honors varlen M by
        # offsetting the output by cumulative sequence length. Asserts on lane layout assumptions.
        if const_expr(param is not None):
            if const_expr(param.feeds_main and param.tensor is None):
                return
            param_tensor = param.tensor
            combine_fn = const_expr(param.combine_fn)
            finalize_fn = const_expr(param.finalize_fn)
            group = const_expr(gemm.local_reduce_group)
            axis = const_expr(gemm.local_reduce_axis)
            tDrReduce = state.local_reduce if const_expr(param.feeds_main) else state[0]
            tDrReduce_cur = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            partition_for_epilogue_fn = partial(
                partition_for_epilogue,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy,
                tidx=tidx,
                reference_src=tiled_copy_t2r is None,
            )
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
            tile = tile_N if const_expr(axis == 1) else tile_M
            groups_per_cta = const_expr(tile // group)
            tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
            tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
            tDrReduce_flt = cute.filter_zeros(tDrReduce_cur)
            tDcD_flt = cute.filter_zeros(tDcD_cur)
            batch_idx = tile_coord_mnkl[3]
            limit_m = min(
                varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M,
                tile_M,
            )
            limit_n = min(cute.size(param_tensor, mode=[2]) - tile_coord_mnkl[1] * tile_N, tile_N)
            if const_expr(axis == 1):
                local_fragment_n = const_expr(cute.size(tDrReduce_cur.shape, mode=[0]))
            if const_expr(axis == 1 and group > local_fragment_n):
                assert combine_fn is not None
                assert finalize_fn is not None
                assert group % local_fragment_n == 0
                fragments_per_group = const_expr(group // local_fragment_n)
                if const_expr((epi_coord[1] + 1) % fragments_per_group == 0):
                    group_start_epi_n = const_expr(epi_coord[1] + 1 - fragments_per_group)
                    limit_groups = (
                        param_tensor.shape[2]
                        if not varlen_manager.varlen_m
                        else param_tensor.shape[1]
                    )
                    if const_expr(not varlen_manager.varlen_m):
                        mReduce = param_tensor[batch_idx, None, None]
                    else:
                        mReduce = cute.domain_offset(
                            (varlen_manager.params.cu_seqlens_m[batch_idx], None),
                            param_tensor[None, None],
                        )
                    gReduce = cute.local_tile(
                        mReduce,
                        (tile_M, groups_per_cta),
                        (tile_coord_mnkl[0], tile_coord_mnkl[1]),
                    )
                    tDrReduce_first = tDrReduce[
                        None, None, None, epi_coord[0], group_start_epi_n
                    ]
                    tDrReduce_first_flt = cute.filter_zeros(tDrReduce_first)
                    for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                        row_idx = tDcD_flt[i][0]
                        n_idx = tDcD_flt[i][1]
                        group_idx = n_idx // group
                        global_group_idx = tile_coord_mnkl[1] * groups_per_cta + group_idx
                        group_value = tDrReduce_first_flt[i]
                        for fragment_idx in cutlass.range_constexpr(1, fragments_per_group):
                            tDrReduce_fragment = tDrReduce[
                                None,
                                None,
                                None,
                                epi_coord[0],
                                group_start_epi_n + fragment_idx,
                            ]
                            tDrReduce_fragment_flt = cute.filter_zeros(tDrReduce_fragment)
                            group_value = combine_fn(group_value, tDrReduce_fragment_flt[i])
                        group_value = finalize_fn(group_value)
                        if const_expr(param_tensor.element_type != Float32):
                            group_value = group_value.to(param_tensor.element_type)
                        if (
                            n_idx % group == group - local_fragment_n
                            and row_idx < limit_m
                            and global_group_idx < limit_groups
                        ):
                            gReduce[row_idx, group_idx] = group_value
                return
            if const_expr(axis == 0):
                assert combine_fn is not None
                assert finalize_fn is not None
                lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(
                    tiled_copy, tiled_copy_t2r is None
                )
                lanes_in_M = cute.size(lane_layout_MN, mode=[0])
                lanes_in_N = cute.size(lane_layout_MN, mode=[1])
                assert lanes_in_M == LOCAL_REDUCE_FRAGMENT_WIDTH
                assert group <= tile_M
                if const_expr(group <= lanes_in_M):
                    assert lanes_in_M % group == 0
                    reduction_group = const_expr(group)
                else:
                    assert group % lanes_in_M == 0
                    reduction_group = const_expr(lanes_in_M)
                if const_expr(lanes_in_N > 1):
                    assert lane_layout_MN.stride[1] == 1
                if const_expr(reduction_group > 1):
                    for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                        reduction_rows = reduction_group // 2
                        while reduction_rows > 0:
                            tDrReduce_flt[i] = combine_fn(
                                tDrReduce_flt[i],
                                cute.arch.shuffle_sync_bfly(
                                    tDrReduce_flt[i],
                                    offset=cute.crd2idx((reduction_rows, 0), lane_layout_MN),
                                ),
                            )
                            reduction_rows = reduction_rows // 2
                if const_expr(grouped_local_reduce_uses_smem(axis, group)):
                    sReduce = state[1]
                    assert sReduce is not None
                    warp_M = warp_layout_MN[0]
                    warps_in_M = const_expr(cute.size(warp_M))
                    group_warps = const_expr(group // lanes_in_M)
                    assert group_warps <= warps_in_M
                    assert groups_per_cta * group_warps <= warps_in_M
                    warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
                    warp_m_idx = warp_layout_MN.get_hier_coord(warp_idx)[0]
                    limit_groups = (
                        param_tensor.shape[1]
                        if not varlen_manager.varlen_m
                        else param_tensor.shape[0]
                    )
                    if const_expr(not varlen_manager.varlen_m):
                        mReduce = param_tensor[batch_idx, None, None]
                    else:
                        mReduce = cute.domain_offset(
                            (varlen_manager.params.cu_seqlens_m[batch_idx], None),
                            param_tensor[None, None],
                        )
                    gReduce = cute.local_tile(
                        mReduce,
                        (groups_per_cta, tile_N),
                        (tile_coord_mnkl[0], tile_coord_mnkl[1]),
                    )
                    for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                        row_idx = tDcD_flt[i][0]
                        n_idx = tDcD_flt[i][1]
                        group_idx = row_idx // group
                        group_warp_start = group_idx * group_warps
                        group_warp_idx = warp_m_idx - group_warp_start
                        smem_warp_idx = warp_m_idx - group_idx - 1
                        if (
                            row_idx % lanes_in_M == 0
                            and group_warp_idx > 0
                            and group_warp_idx < group_warps
                        ):
                            sReduce[n_idx, smem_warp_idx] = tDrReduce_flt[i]
                    gemm.epilogue_barrier.arrive_and_wait()
                    for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                        row_idx = tDcD_flt[i][0]
                        n_idx = tDcD_flt[i][1]
                        group_idx = row_idx // group
                        group_warp_start = group_idx * group_warps
                        global_group_idx = tile_coord_mnkl[0] * groups_per_cta + group_idx
                        group_value = tDrReduce_flt[i]
                        if row_idx % group == 0 and warp_m_idx == group_warp_start:
                            for warp_offset in cutlass.range_constexpr(1, group_warps):
                                smem_warp_idx = group_warp_start + warp_offset - group_idx - 1
                                group_value = combine_fn(
                                    group_value, sReduce[n_idx, smem_warp_idx]
                                )
                            group_value = finalize_fn(group_value)
                            if const_expr(param_tensor.element_type != Float32):
                                group_value = group_value.to(param_tensor.element_type)
                            if n_idx < limit_n and global_group_idx < limit_groups:
                                gReduce[group_idx, n_idx] = group_value
                    gemm.epilogue_barrier.arrive_and_wait()
                    return
            if const_expr(axis == 1):
                limit_groups = param_tensor.shape[2] if not varlen_manager.varlen_m else param_tensor.shape[1]
                tile_shape = (tile_M, groups_per_cta)
                tile_coord = (tile_coord_mnkl[0], tile_coord_mnkl[1])
            else:
                limit_groups = param_tensor.shape[1] if not varlen_manager.varlen_m else param_tensor.shape[0]
                tile_shape = (groups_per_cta, tile_N)
                tile_coord = (tile_coord_mnkl[0], tile_coord_mnkl[1])
            if const_expr(not varlen_manager.varlen_m):
                mReduce = param_tensor[batch_idx, None, None]
            else:
                mReduce = cute.domain_offset(
                    (varlen_manager.params.cu_seqlens_m[batch_idx], None),
                    param_tensor[None, None],
                )
            gReduce = cute.local_tile(mReduce, tile_shape, tile_coord)
            if const_expr(
                axis == 0
                and param_tensor.element_type == Float32
                and param_tensor.iterator.alignment >= 16
                and cute.size(tDrReduce_flt) % 4 == 0
                and tile_N % 4 == 0
            ):
                store_vec = const_expr(4)
                for chunk in cutlass.range(
                    cute.size(tDrReduce_flt) // store_vec, unroll_full=True
                ):
                    i = chunk * store_vec
                    row_idx = tDcD_flt[i][0]
                    n_idx = tDcD_flt[i][1]
                    group_idx = row_idx // group
                    global_group_idx = tile_coord_mnkl[0] * groups_per_cta + group_idx
                    can_store_vec = (
                        row_idx % group == 0
                        and n_idx % store_vec == 0
                        and cute.size(param_tensor, mode=[2]) % store_vec == 0
                        and n_idx + store_vec <= limit_n
                        and global_group_idx < limit_groups
                    )
                    for vec_idx in cutlass.range_constexpr(1, store_vec):
                        can_store_vec = (
                            can_store_vec
                            and tDcD_flt[i + vec_idx][0] == row_idx
                            and tDcD_flt[i + vec_idx][1] == n_idx + vec_idx
                        )
                    if can_store_vec:
                        tDrStore = cute.make_rmem_tensor(store_vec, Float32)
                        for vec_idx in cutlass.range_constexpr(store_vec):
                            tDrStore[vec_idx] = finalize_fn(tDrReduce_flt[i + vec_idx])
                        gReduce_vec_ptr = cute.domain_offset(
                            (group_idx, n_idx), gReduce
                        ).iterator
                        gReduce_vec = cute.make_tensor(
                            cute.make_ptr(
                                gReduce_vec_ptr.dtype,
                                gReduce_vec_ptr.toint(),
                                gReduce_vec_ptr.memspace,
                                assumed_align=16,
                            ),
                            cute.make_layout(store_vec),
                        )
                        cute.autovec_copy(tDrStore, gReduce_vec)
                    else:
                        for vec_idx in cutlass.range_constexpr(store_vec):
                            row_idx_cur = tDcD_flt[i + vec_idx][0]
                            n_idx_cur = tDcD_flt[i + vec_idx][1]
                            group_idx_cur = row_idx_cur // group
                            global_group_idx_cur = (
                                tile_coord_mnkl[0] * groups_per_cta + group_idx_cur
                            )
                            if (
                                row_idx_cur % group == 0
                                and n_idx_cur < limit_n
                                and global_group_idx_cur < limit_groups
                            ):
                                gReduce[group_idx_cur, n_idx_cur] = finalize_fn(
                                    tDrReduce_flt[i + vec_idx]
                                )
                return
            for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                row_idx = tDcD_flt[i][0]
                n_idx = tDcD_flt[i][1]
                group_idx = n_idx // group if const_expr(axis == 1) else row_idx // group
                group_leader = n_idx % group == 0 if const_expr(axis == 1) else row_idx % group == 0
                global_group_idx = (
                    tile_coord_mnkl[1] * groups_per_cta + group_idx
                    if const_expr(axis == 1)
                    else tile_coord_mnkl[0] * groups_per_cta + group_idx
                )
                group_value = tDrReduce_flt[i]
                if const_expr(axis == 0 or finalize_fn is not None):
                    group_value = finalize_fn(group_value)
                if const_expr(param_tensor.element_type != Float32):
                    group_value = group_value.to(param_tensor.element_type)
                if const_expr(axis == 1):
                    if (
                        group_leader
                        and row_idx < limit_m
                        and global_group_idx < limit_groups
                    ):
                        gReduce[row_idx, group_idx] = group_value
                else:
                    if (
                        group_leader
                        and n_idx < limit_n
                        and global_group_idx < limit_groups
                    ):
                        gReduce[group_idx, n_idx] = group_value


class ColVecReduce(VecReduce):
    # [CLASS DOC] Column vector reduction: user code accumulates across N subtiles in registers;
    # this op allocates the registers (begin), slices them per subtile (begin_loop) and, at the
    # last N subtile of an M stripe, reduces across N lanes and warps and writes one partial value
    # per row to global memory (end_loop). Output has one partial sum per CTA tile in N. Output
    # dimension zero, N-major preference.
    """Column vector reduction: accumulates across N subtiles in registers,
    then reduces across N lanes/warps and writes to gmem per completed M stripe.

    The accumulation itself happens in epi_visit_subtile (user code).
    This op handles the register allocation (begin), per-subtile slicing (begin_loop),
    and reduction + gmem write (end_loop).
    """

    dim = 0
    epi_m_major_preference = -1

    @cute.jit
    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] At the last N subtile of the current M stripe: take the register slice,
        # derive lane and warp layouts from the tiled copy, reduce across the lanes that share a
        # row by warp reduction with addition (asserts the lane count is a power of two and lanes
        # are contiguous in N), reduce across warps in N through shared memory with a barrier, and
        # let the leader lane of the first N warp write each row's value to global memory. Writes
        # only if the CTA N tile index is within the output extent and the row is within the valid
        # M limit; handles varlen M via cumulative sequence offsets. Does nothing for earlier
        # subtiles.
        """Flush the current M stripe when the last N subtile has accumulated."""
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        if const_expr(epi_coord[1] == epi_tile_shape[1] - 1):
            tDrReduce, sDrReduce = state[0], state[1]
            tDrReduce_cur = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            reference_src = tiled_copy_t2r is None

            # ── Derive lane layout from tiled_copy ──
            lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
            # For ColVecReduce: reduce across N lanes (lanes_in_N threads share same M row)
            lanes_in_N = cute.size(lane_layout_MN, mode=[1])
            is_lane_n_leader = cute.arch.lane_idx() % lanes_in_N == 0
            # Typically lanes_in_N is 4 for Sm90
            assert lanes_in_N == 1 << int(math.log2(lanes_in_N)), (
                "lanes_in_N must be a power of 2 for butterfly reduction"
            )

            # Intra-warp shuffle reduction across N lanes
            if const_expr(lanes_in_N > 1):
                # Assumes threads for each M row are contiguous along N, so
                # warp_reduction over groups of lanes_in_N matches lane_layout_MN.
                assert lane_layout_MN.stride[1] == 1
                tDrReduce_flt = cute.filter_zeros(tDrReduce_cur)
                for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                    tDrReduce_flt[i] = cute.arch.warp_reduction(
                        tDrReduce_flt[i], operator.add, threads_in_group=lanes_in_N
                    )

            warp_N = warp_layout_MN[1]
            warps_in_N = const_expr(cute.size(warp_N))
            partition_for_epilogue_fn = partial(
                partition_for_epilogue,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy,
                tidx=tidx,
                reference_src=tiled_copy_t2r is None,
            )
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
            tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
            tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
            tDrReduce_m = layout_utils.convert_layout_zero_stride(
                tDrReduce_cur, tDrReduce_cur.layout
            )[None, 0]
            tDcD_m = layout_utils.convert_layout_zero_stride(tDcD_cur, tDrReduce_cur.layout)[
                None, 0
            ]

            # Inter-warp reduction through smem
            warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
            warp_n_idx = warp_layout_MN.get_hier_coord(warp_idx)[1]
            if const_expr(warps_in_N > 1):
                if warp_n_idx > 0 and is_lane_n_leader:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        sDrReduce[row_idx, warp_n_idx - 1] = tDrReduce_m[m]
                gemm.epilogue_barrier.arrive_and_wait()
                if warp_n_idx == 0 and is_lane_n_leader:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        for warp_n in cutlass.range_constexpr(1, warps_in_N):
                            tDrReduce_m[m] += sDrReduce[row_idx, warp_n - 1]

            # Write to gmem
            batch_idx = tile_coord_mnkl[3]
            limit_m = min(varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M, tile_M)
            limit_n_tiles = param.shape[2] if not varlen_manager.varlen_m else param.shape[1]
            if const_expr(not varlen_manager.varlen_m):
                mColVec = param[batch_idx, None, tile_coord_mnkl[1]]
            else:
                mColVec = cute.domain_offset(
                    (varlen_manager.params.cu_seqlens_m[batch_idx],),
                    param[None, tile_coord_mnkl[1]],
                )
            gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
            should_write_gmem = (
                is_lane_n_leader
                if const_expr(warps_in_N == 1)
                else warp_n_idx == 0 and is_lane_n_leader
            )
            if tile_coord_mnkl[1] < limit_n_tiles and should_write_gmem:
                for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                    row_idx = tDcD_m[m][0]
                    if row_idx < limit_m:
                        gColVec[row_idx] = tDrReduce_m[m]


class RowVecReduce(VecReduce):
    # [CLASS DOC] Row vector reduction, the M and N mirror of the column reduction: user code
    # accumulates across M subtiles in registers; at the last M subtile of an N stripe this op
    # reduces across M lanes and warps and writes one partial sum per N column to global memory.
    # Output has shape batch by number of CTA M tiles by N. Output dimension one, M-major
    # preference.
    """Row vector reduction: accumulates across M subtiles in registers,
    then reduces across M lanes/warps and writes to gmem per completed N stripe.

    Output shape is (L, ceildiv(M, tile_M), N): one partial sum per CTA-M tile per
    N column. This mirrors ColVecReduce with M/N swapped.
    """

    dim = 1
    epi_m_major_preference = 4

    @cute.jit
    def end_loop(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        # [FUNCTION DOC] At the last M subtile of the current N stripe: take the register slice,
        # derive lane and warp layouts, reduce across the lanes that share a column using
        # butterfly shuffles with addition (asserts a power of two lane count in M and contiguous
        # N lanes when several), reduce across warps in M through shared memory with a barrier,
        # and let the first M warp's leader lanes write each column value to global memory for
        # columns within the valid N limit and when the CTA M tile index is inside the output
        # extent. Output indexing differs for varlen M (no batch index). Does nothing for earlier
        # subtiles.
        """Flush the current N stripe when the last M subtile has accumulated."""
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        if const_expr(epi_coord[0] == epi_tile_shape[0] - 1):
            tDrReduce, sDrReduce = state[0], state[1]
            tDrReduce_cur = tDrReduce[None, None, None, epi_coord[0], epi_coord[1]]
            tiled_copy = tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
            reference_src = tiled_copy_t2r is None

            # ── Derive lane layout from tiled_copy ──
            lane_layout_MN, warp_layout_MN = _get_lane_warp_layouts(tiled_copy, reference_src)
            # For RowVecReduce: reduce across M lanes (lanes_in_M threads share same N col)
            lanes_in_M = cute.size(lane_layout_MN, mode=[0])
            lanes_in_N = cute.size(lane_layout_MN, mode=[1])
            is_lane_m_leader = cute.arch.lane_idx() < lanes_in_N
            assert lanes_in_M == 1 << int(math.log2(lanes_in_M)), (
                "lanes_in_M must be a power of 2 for butterfly reduction"
            )
            if const_expr(lanes_in_N > 1):
                assert lane_layout_MN.stride[1] == 1, (
                    "RowVecReduce assumes contiguous N lanes when lanes_in_N > 1"
                )

            # Intra-warp shuffle reduction across M lanes. M lanes may be either contiguous
            # (SM100 N-major output) or strided by N lanes (SM100 M-major output).
            tDrReduce_n = layout_utils.convert_layout_zero_stride(
                tDrReduce_cur, tDrReduce_cur.layout
            )[None, 0]
            if const_expr(lanes_in_M > 1):
                for n in cutlass.range(cute.size(tDrReduce_n), unroll_full=True):
                    reduction_rows = lanes_in_M // 2
                    while reduction_rows > 0:
                        tDrReduce_n[n] += cute.arch.shuffle_sync_bfly(
                            tDrReduce_n[n],
                            offset=cute.crd2idx((reduction_rows, 0), lane_layout_MN),
                        )
                        reduction_rows = reduction_rows // 2

            warp_M = warp_layout_MN[0]
            warps_in_M = const_expr(cute.size(warp_M))
            partition_for_epilogue_fn = partial(
                partition_for_epilogue,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy,
                tidx=tidx,
                reference_src=tiled_copy_t2r is None,
            )
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
            tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
            tDcD_cur = tDcD[None, None, None, epi_coord[0], epi_coord[1]]
            tDcD_n = layout_utils.convert_layout_zero_stride(tDcD_cur, tDrReduce_cur.layout)[
                None, 0
            ]

            # Inter-warp reduction through smem
            warp_idx = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
            warp_m_idx = warp_layout_MN.get_hier_coord(warp_idx)[0]
            if const_expr(warps_in_M > 1):
                if warp_m_idx > 0 and is_lane_m_leader:
                    for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                        col_idx = tDcD_n[n][1]
                        sDrReduce[col_idx, warp_m_idx - 1] = tDrReduce_n[n]
                gemm.epilogue_barrier.arrive_and_wait()
                if warp_m_idx == 0 and is_lane_m_leader:
                    for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                        col_idx = tDcD_n[n][1]
                        for warp_m in cutlass.range_constexpr(1, warps_in_M):
                            tDrReduce_n[n] += sDrReduce[col_idx, warp_m - 1]

            # Write to gmem
            batch_idx = tile_coord_mnkl[3]
            limit_m_tiles = param.shape[1] if not varlen_manager.varlen_m else param.shape[0]
            if const_expr(not varlen_manager.varlen_m):
                mRowVec = param[batch_idx, tile_coord_mnkl[0], None]
            else:
                mRowVec = param[tile_coord_mnkl[0], None]
            gRowVec = cute.local_tile(mRowVec, (tile_N,), (tile_coord_mnkl[1],))
            limit_n = min(
                cute.size(mRowVec, mode=[0]) - tile_coord_mnkl[1] * tile_N,
                tile_N,
            )
            should_write_gmem = (
                is_lane_m_leader
                if const_expr(warps_in_M == 1)
                else warp_m_idx == 0 and is_lane_m_leader
            )
            if tile_coord_mnkl[0] < limit_m_tiles and should_write_gmem:
                for n in cutlass.range(cute.size(tDcD_n, mode=[0])):
                    col_idx = tDcD_n[n][1]
                    if col_idx < limit_n:
                        gRowVec[col_idx] = tDrReduce_n[n]
