# Copyright (c) 2025-2026, Tri Dao.
# [MODULE DOC] Small helpers for Ampere-class (SM80) tensor-core MMA kernels written with the
# CuTe DSL. Currently only a function that partitions shared-memory operands and allocates the
# accumulator for a thread slice of a tiled MMA.
import cutlass.cute as cute
from cutlass import Float32, const_expr


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: cute.Tensor,
    sB: cute.Tensor,
    swap_AB: bool = False,
):
    # [FUNCTION DOC] Partition operands for one thread of a tiled MMA. Without swapping,
    # builds an FP32 register accumulator from the M and N part of the tile shape, partitions
    # the shared-memory A and B tensors, and makes register fragments from the first pipeline
    # stage of each partition. With swap_AB, the roles are exchanged: the accumulator uses the
    # N by M shape, B is partitioned as the first operand and A as the second. Returns a tuple
    # of accumulator, shared A partition, shared B partition, A fragment, B fragment. Neither
    # shared tensor may be None; this is not checked here.
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
    else:
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((shape_mnk[1], shape_mnk[0])), Float32
        )
        tCsB = thr_mma.partition_A(sB)
        tCsA = thr_mma.partition_B(sA)
        tCrB = thr_mma.make_fragment_A(tCsB[None, None, None, 0])
        tCrA = thr_mma.make_fragment_B(tCsA[None, None, None, 0])
    return acc, tCsA, tCsB, tCrA, tCrB
