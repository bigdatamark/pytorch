# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# [MODULE DOC] Small device-side helpers for the quack CUTLASS DSL kernels: pointer arithmetic,
# cluster-shared-memory remote stores with mbarrier completion, scalar math via inline PTX or NVVM
# ops (fmin, sqrt, ceil), vector packing of float pairs into 64-bit integers, a warp-level
# inclusive prefix sum, atomic increment/add on global memory, cluster launch control cancel
# queries, and tensor domain offsetting. Import-time side effects: none beyond imports; the
# functions are traced by the DSL only when called inside jitted code.

import math
from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute

from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    # [FUNCTION DOC] DSL user op returning a pointer to the element of tensor x selected by coord:
    # the tensor iterator advanced by the layout-mapped linear index of the coordinate.
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def load_scalar_or_pointer(x, dtype=Float32):
    # [FUNCTION DOC] Jit helper that accepts either a pointer or a plain value. If x is a pointer
    # (checked at compile time) it views it as a one-element tensor, reads element zero and
    # converts it to dtype (Float32 by default); otherwise returns x unchanged.
    if const_expr(isinstance(x, cute.Pointer)):
        return dtype(cute.make_tensor(x, cute.make_layout(1))[0])
    else:
        return x


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    # [FUNCTION DOC] DSL user op that translates a shared-memory pointer into the address of
    # the same location in the shared memory of another CTA in the cluster, using the PTX mapa
    # instruction. Takes the pointer and the peer CTA rank; returns the remapped 32-bit
    # address as Int32.
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    # [FUNCTION DOC] DSL user op that asynchronously stores one value to a peer CTA's shared memory
    # and signals that CTA's mbarrier with the transferred byte count (PTX st.async with
    # complete_tx). Both pointers are first mapped to the peer rank. A Python float is converted to
    # Float32; the value must then be Float32, Int32 or Int64 or an AssertionError is raised. Has
    # memory side effects and returns None.
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    # [FUNCTION DOC] DSL user op that asynchronously stores four values as one vector to a peer
    # CTA's shared memory, completing on that CTA's mbarrier (PTX st.async v4 with complete_tx).
    # Pointers are mapped to the peer rank. The element type is chosen from the first value
    # (Float32 or Int32, otherwise AssertionError); all four values are converted to that type.
    # KNOWN ISSUE: only the first value's type is checked, the others are silently converted.
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val0, (Float32, Int32)), "val must be Float32, or Int32"
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    constraint = {Float32: "f", Int32: "r"}[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0).ir_value(loc=loc, ip=ip),
            dtype(val1).ir_value(loc=loc, ip=ip),
            dtype(val2).ir_value(loc=loc, ip=ip),
            dtype(val3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, $2;\n\t"
        f"mov.{suffix} abcd.y, $3;\n\t"
        f"mov.{suffix} abcd.z, $4;\n\t"
        f"mov.{suffix} abcd.w, $5;\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} [$0], abcd, [$1];\n\t"
        "}\n",
        f"r,r,{constraint},{constraint},{constraint},{constraint}",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def fmin(a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    # [FUNCTION DOC] DSL user op returning the minimum of two numbers as Float32 via the NVVM fmin
    # op. For CUDA major version 12 the op is called with an explicit f32 result type, otherwise
    # the result type is inferred (the NVVM binding signature differs between versions).
    if cutlass.const_expr(cutlass.CUDA_VERSION.major) == 12:
        return Float32(
            nvvm.fmin(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        )
    return Float32(
        nvvm.fmin(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    # [FUNCTION DOC] DSL user op returning the square root of a number as Float32, implemented with
    # the approximate-precision PTX sqrt instruction (no side effects).
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def ceil(a: float | Float32, *, loc=None, ip=None) -> Int32:
    # [FUNCTION DOC] DSL user op that rounds a float up to the nearest integer toward positive
    # infinity using the PTX convert instruction (round to positive integer, flush denormals to
    # zero), returning the result as Int32.
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rpi.ftz.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    # [FUNCTION DOC] Jit helper that writes fill_value into out-of-bounds positions of a
    # shared-memory tensor. Builds a register tensor shaped like one vector slice, fills it, then
    # loops over the second sub-mode of the first mode and over the third mode. When a predicate
    # tensor is given, only slices whose predicate is false are overwritten; when it is None, every
    # slice is overwritten. Mutates tXsX; returns None.
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


# ---------------------------------------------------------------------------
# General-purpose DSL store / vector helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def make_vector(elem_type, *values, loc=None, ip=None):
    # [FUNCTION DOC] DSL user op that builds an MLIR vector of length equal to the number of
    # values: starts from an undefined vector of elem_type and inserts each value (converted to
    # elem_type) at its position. Returns the MLIR vector value.
    """Build an MLIR vector <N x elem_type> from N scalar DSL values.

    Example: make_vector(cutlass.Uint32, v0, v1) -> <2 x i32> MLIR vector
    """
    from cutlass._mlir import ir

    n = len(values)
    mlir_ty = elem_type.mlir_type
    vec_ty = ir.VectorType.get([n], mlir_ty)
    vec = llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
    for i, v in enumerate(values):
        vec = vector.insertelement(
            elem_type(v).ir_value(loc=loc, ip=ip),
            vec,
            position=_arith.constant(T.i32(), i, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    return vec


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    # [FUNCTION DOC] DSL user op that packs two Float32 values into one Int64 by building
    # a two-lane float vector (a first), reinterpreting its bits as a single 64-bit lane
    # and extracting it.
    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    res = cutlass.Int64(
        vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    # [FUNCTION DOC] DSL user op that reverses the packing done by the float-pair-to-int64
    # helper: reinterprets a 64-bit integer as two float lanes and returns them as a pair of
    # Float32 (lane zero first).
    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    res1 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return res0, res1


@cute.jit
def warp_prefix_sum(val: Int32, lane: Optional[Int32] = None) -> Int32:
    # [FUNCTION DOC] Jit helper computing an inclusive prefix sum of val across the lanes of a
    # warp. lane defaults to the calling thread's lane index. For each power-of-two offset below
    # the warp size it fetches the value from the lane that many positions lower (shuffle up with a
    # zero mask-and-clamp argument, noted in the code as important) and adds it if the lane index
    # is at least the offset. Returns the per-lane sum.
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def atomic_inc_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    # [FUNCTION DOC] DSL user op performing an atomic read-modify-write of the NVVM increment kind
    # on a 32-bit global-memory pointer with operand a, returning the NVVM atomic result. For CUDA
    # 12.9 the old binding signature (explicit result type) is used; for other versions the new
    # one (result type inferred).
    from cutlass import CUDA_VERSION

    # * NVVM call based on nvvm version
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )


@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    # [FUNCTION DOC] DSL user op performing an atomic add of a to the 32-bit integer at a global
    # memory pointer, returning the NVVM atomic result. Chooses between the explicit-result-type
    # and inferred-result-type NVVM binding depending on whether the CUDA version is 12.9.
    from cutlass import CUDA_VERSION

    # * NVVM call based on nvvm version
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )


@dsl_user_op
def issue_clc_query_nomulticast(
    mbar_ptr: cute.Pointer,
    clc_response_ptr: cute.Pointer,
    loc=None,
    ip=None,
) -> None:
    # [FUNCTION DOC] DSL user op that issues a cluster launch control try-cancel request, asking to
    # cancel a cluster that has not started. The response is written asynchronously to the shared
    # memory location given by clc_response_ptr and completion is signalled on the mbarrier at
    # mbar_ptr. Returns None. Note loc and ip here are ordinary (not keyword-only) parameters.
    """
    The clusterlaunchcontrol.try_cancel instruction requests atomically cancelling the launch
    of a cluster that has not started running yet. It asynchronously writes an opaque response
    to shared memory indicating whether the operation succeeded or failed. On success, the
    opaque response contains the ctaid of the first CTA of the canceled cluster.

    :param mbar_ptr: A pointer to the mbarrier address in SMEM
    :type mbar_ptr:  Pointer
    :param clc_response_ptr: A pointer to the cluster launch control response address in SMEM
    :type clc_response_ptr:  Pointer
    """
    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    clc_response_llvm_ptr = clc_response_ptr.llvm_ptr
    nvvm.clusterlaunchcontrol_try_cancel(
        clc_response_llvm_ptr,
        mbar_llvm_ptr,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def domain_offset_aligned(
    coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None
) -> cute.Tensor:
    # [FUNCTION DOC] DSL user op that offsets a tensor's origin by a coordinate while building a
    # new pointer with the original alignment (asserted unchanged by assumption, not checked).
    # Raises AssertionError if the tensor iterator is not a pointer. Returns a tensor with the
    # shifted pointer, same element type and memory space, and the original layout.
    assert isinstance(tensor.iterator, cute.Pointer)
    # We assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        elem_pointer(tensor, coord).toint(),
        tensor.memspace,
        assumed_align=tensor.iterator.alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)
