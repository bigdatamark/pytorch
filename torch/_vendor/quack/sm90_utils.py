# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Hopper (SM90) helpers for GEMM-style kernels in the CuTe DSL: shared-memory
# layout construction, tiled-MMA construction with warpgroup MMA atoms, epilogue partitioning,
# and thin wrappers that issue warpgroup MMAs with optional A/B swapping and stage indexing.
# Also exports an alias for the epilogue layout builder. Importing has no side effects.

from typing import Literal, Type, Union, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    stage: Optional[int] = None,
    major_mode_size: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    # [FUNCTION DOC] Build a swizzled shared-memory layout for the given dtype, layout enum
    # and tile. The tile shape is flattened by taking per-mode products; the major-mode size
    # defaults to the second extent for an N-major C layout and otherwise the first. The
    # swizzle atom comes from the Hopper helper for that layout, dtype and size. Atoms are
    # tiled to the tile shape, with a trailing stage mode when a stage count is given (mode
    # order is swapped for M-major layouts). Returns the unstaged or staged layout.
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    if const_expr(major_mode_size is None):
        major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged


# For compatibility with blackwell_helpers.py
make_smem_layout_epi = make_smem_layout


def make_tiled_mma(
    a_dtype: Type[Numeric],
    a_major: Literal["K", "MN"],
    b_major: Literal["K", "MN"],
    tiler_n: int,
    source: Literal["SS", "RS"] = "SS",
    atom_layout_mnk: tuple = (1, 1, 1),
    swap_AB: bool = False,
    b_dtype: Optional[Type[Numeric]] = None,
    acc_dtype: Type[Numeric] = Float32,
) -> cute.TiledMma:
    # [FUNCTION DOC] Create a Hopper warpgroup tiled MMA with tile (64, tiler_n). The B dtype
    # defaults to the A dtype and the accumulator dtype to Float32. Majors are given as the
    # strings K or MN (any other string raises KeyError); with swap_AB the two major modes are
    # exchanged (the dtypes are not). The A operand comes from registers for source RS and
    # from shared memory otherwise. Returns the tiled MMA from the CUTLASS trivial builder.
    """`b_dtype` defaults to `a_dtype`; pass it for mixed-precision MMAs (e.g. fp8).
    `acc_dtype` defaults to Float32."""
    if b_dtype is None:
        b_dtype = a_dtype
    mode = {"K": cute.nvgpu.OperandMajorMode.K, "MN": cute.nvgpu.OperandMajorMode.MN}
    a_mode, b_mode = mode[a_major], mode[b_major]
    if swap_AB:
        a_mode, b_mode = b_mode, a_mode
    a_source = warpgroup.OperandSource.RMEM if source == "RS" else warpgroup.OperandSource.SMEM
    return sm90_utils_og.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_mode,
        b_mode,
        acc_dtype,
        atom_layout_mnk=atom_layout_mnk,
        tiler_mn=(64, tiler_n),
        a_source=a_source,
    )


@dsl_user_op
def partition_for_epilogue(
    cT: cute.Tensor,
    epi_tile: cute.Tile,
    tiled_copy: cute.TiledCopy,
    tidx: Int32,
    reference_src: bool,  # do register tensors reference the src or dst layout of the tiled copy
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    # [FUNCTION DOC] Partition a coordinate tensor for the epilogue of one thread. The tensor
    # is divided into epilogue tiles, then partitioned with the thread slice of the tiled copy
    # either according to the source layout (when register tensors follow the source) or the
    # destination layout. Returns the thread partition with modes for copy element, copy M,
    # copy N, epilogue M and epilogue N.
    thr_copy = tiled_copy.get_slice(tidx)
    cT_epi = cute.flat_divide(cT, epi_tile)
    # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
    if const_expr(reference_src):
        return thr_copy.partition_S(cT_epi, loc=loc, ip=ip)
    else:
        return thr_copy.partition_D(cT_epi, loc=loc, ip=ip)


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    # A_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    # [FUNCTION DOC] JIT-compiled warpgroup GEMM. With swap_AB it calls itself with A and B
    # exchanged and swapping disabled. Otherwise it fences the warpgroup, builds a fresh MMA
    # atom (to avoid mutating the shared one), sets accumulation to off for the first K block
    # when zero_init is set, issues one MMA per K block of the fragments with accumulation on
    # afterwards, commits the group, and waits until at most wg_wait groups remain in flight
    # when wg_wait is not negative. Updates acc in place; returns None.
    if const_expr(swap_AB):
        gemm(tiled_mma, acc, tCrB, tCrA, zero_init=zero_init, wg_wait=wg_wait, swap_AB=False)
    else:
        warpgroup.fence()
        # We make a new mma_atom since we'll be modifying its attribute (accumulate).
        # Otherwise the compiler complains "operand #0 does not dominate this use"
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)


def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor:
    # [FUNCTION DOC] Allocate an FP32 register accumulator for the given output shape and
    # run a zero-initialised GEMM into it. Optional A and B indices select a stage from the
    # last mode of the fragments. With swap_AB the shape is reversed and operands and
    # indices are exchanged. The default wg_wait of -1 means no wait for the MMA group.
    # Returns the accumulator.
    if const_expr(swap_AB):
        return gemm_zero_init(
            tiled_mma, shape[::-1], tCrB, tCrA, B_idx, A_idx, wg_wait, swap_AB=False
        )
    else:
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
        return acc


def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None:
    # [FUNCTION DOC] GEMM wrapper that optionally picks a stage of A and B fragments by index
    # (last mode) and forwards to the base GEMM with the given zero_init and wg_wait; with
    # swap_AB it exchanges operands and indices first. Accumulates into acc in place and
    # returns None.
    if const_expr(swap_AB):
        gemm_w_idx(tiled_mma, acc, tCrB, tCrA, zero_init, B_idx, A_idx, wg_wait, swap_AB=False)
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool = False,
):
    # [FUNCTION DOC] Allocate the accumulator and operand fragments for a Hopper tiled MMA
    # thread slice. Whether A is in registers is read from the MMA op. Without swapping the
    # FP32 accumulator has the M by N shape, A is a fragment of the partitioned shared tensor
    # (or a register-shaped fragment of M by K when A comes from registers) and B is a
    # fragment of the partitioned shared tensor. With swap_AB the accumulator is N by M and
    # the roles of A and B are exchanged. Asserts that required shared tensors are not None.
    # Returns a tuple of accumulator, A fragment, B fragment.
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        if const_expr(not is_rs):
            assert sA is not None
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        else:
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2])))
        assert sB is not None
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    else:
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((shape_mnk[1], shape_mnk[0])), Float32
        )
        if const_expr(not is_rs):
            assert sB is not None
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_A(sB))
        else:  # B in rmem
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[1], shape_mnk[2])))
        assert sA is not None
        tCrA = thr_mma.make_fragment_B(thr_mma.partition_B(sA))
    return acc, tCrA, tCrB
