# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Helpers shared by TVM-FFI GEMM compilation: tensor permutation and major-mode
# detection for torch tensors, dtype mapping, construction of real and fake (compile-time)
# scheduler/varlen arguments and tensors, and a function that instantiates and compiles a GEMM
# class. No import-time side effects beyond imports.
# Shared utilities for TVM-FFI GEMM compilation.

from functools import partial


import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_ptr

from .compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl_utils import torch2cute_dtype_map
from .tile_scheduler import TileSchedulerOptions
from .varlen_utils import VarlenArguments


def div_for_dtype(dtype):
    # [FUNCTION DOC] Return the divisibility, in elements, that gives 16-byte alignment for a
    # dtype: 128 divided by the dtype width in bits (integer division).
    """16-byte alignment: divisibility in elements = 128 // dtype_width_bits."""
    return 128 // dtype.width


def perm3d_single(t, varlen_m=False):
    # [FUNCTION DOC] Permute one 3D tensor from layout (L, a, b) to (a, b, L). Returns the input
    # unchanged when it is None, not 3-dimensional, or varlen_m is true.
    """Permute a single 3D tensor from (L, *, *) to (*, *, L), skipping for varlen_m or 2D."""
    return t.permute(1, 2, 0) if t is not None and t.ndim == 3 and not varlen_m else t


def perm3d(A, B, D, C, varlen_m=False, varlen_k=False):
    # [FUNCTION DOC] Permute batch-first 3D tensors A, B, D, C to batch-last, honoring varlen.
    # Returns the four tensors as a tuple. With varlen_m, A, D and C are untouched and only B
    # is permuted; with varlen_k, A and B are untouched and D and C are permuted; otherwise all
    # four are permuted. None and non-3D tensors are always passed through unchanged.
    """Permute 3D tensors from (L, *, *) to (*, *, L)."""

    def _perm(t):
        # [FUNCTION DOC] HELPER: Permute a tensor from (L, a, b) to (a, b, L) if it is a non-None
        # 3D tensor, else return it unchanged.
        return t.permute(1, 2, 0) if t is not None and t.ndim == 3 else t

    if varlen_m:
        return A, _perm(B), D, C
    elif varlen_k:
        return A, B, _perm(D), _perm(C)
    else:
        return _perm(A), _perm(B), _perm(D), _perm(C)


def get_major(t, dim0, dim1):
    # [FUNCTION DOC] Return dim1 if the tensor has unit stride in its second dimension,
    # else dim0. Used to name which logical dimension is contiguous (the major mode)
    # for a 2D-style tensor.
    return dim1 if t.stride(1) == 1 else dim0


def get_majors(A_p, B_p, D_p, C_p):
    # [FUNCTION DOC] Compute major-mode labels for A, B, D and optionally C. A uses labels m/k,
    # B uses n/k, D and C use m/n. C may be None, in which case its label is None. Returns a
    # 4-tuple (a_major, b_major, d_major, c_major).
    a_major = get_major(A_p, "m", "k")
    b_major = get_major(B_p, "n", "k")
    d_major = get_major(D_p, "m", "n")
    c_major = get_major(C_p, "m", "n") if C_p is not None else None
    return a_major, b_major, d_major, c_major


def get_dtypes(A, B, D, C):
    # [FUNCTION DOC] Map torch dtypes of A, B, D and optional C to cute dtypes using the shared
    # torch-to-cute dtype table. C may be None, which yields None. Returns a 4-tuple. Raises
    # KeyError for a dtype missing from the table.
    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = torch2cute_dtype_map[B.dtype]
    d_dtype = torch2cute_dtype_map[D.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    return a_dtype, b_dtype, d_dtype, c_dtype


def make_scheduler_args(
    max_active_clusters, max_swizzle_size, tile_count_semaphore, batch_idx_permute=None
):
    # [FUNCTION DOC] Build real TileSchedulerOptions for a launch. Wraps max_active_clusters in
    # an Int32, leaves raster order unset, passes the swizzle size and optional batch index
    # permutation through, and uses the data pointer of the tile-count semaphore tensor when
    # given (else None).
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active_clusters),
        raster_order=None,
        max_swizzle_size=max_swizzle_size,
        tile_count_semaphore=(
            tile_count_semaphore.data_ptr() if tile_count_semaphore is not None else None
        ),
        batch_idx_permute=batch_idx_permute,
    )


def make_fake_scheduler_args(has_semaphore, has_batch_idx_permute, l_sym):
    # [FUNCTION DOC] Build placeholder TileSchedulerOptions for compilation. Uses fixed
    # max_active_clusters of 1 and swizzle size 8; includes a fake 4-byte-aligned global Int32
    # semaphore pointer when has_semaphore is true, and a fake Int32 batch permutation vector
    # of symbolic length l_sym when has_batch_idx_permute is true.
    return TileSchedulerOptions(
        max_active_clusters=Int32(1),
        max_swizzle_size=Int32(8),
        tile_count_semaphore=(
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=4) if has_semaphore else None
        ),
        batch_idx_permute=(
            fake_tensor(Int32, (l_sym,), leading_dim=0, divisibility=4)
            if has_batch_idx_permute
            else None
        ),
    )


def make_varlen_args(cu_seqlens_m, cu_seqlens_k, A_idx):
    if cu_seqlens_m is None and cu_seqlens_k is None:
    # [FUNCTION DOC] Package cumulative sequence-length tensors and the A index tensor into
    # VarlenArguments. Returns None when both cu_seqlens_m and cu_seqlens_k are None.
        return None
    return VarlenArguments(
        mCuSeqlensM=cu_seqlens_m,
        mCuSeqlensK=cu_seqlens_k,
        mAIdx=A_idx,
    )


def make_fake_varlen_args(varlen_m, varlen_k, gather_A, aidx_len):
    if not varlen_m and not varlen_k:
    # [FUNCTION DOC] Build placeholder VarlenArguments for compilation. Returns None unless
    # varlen_m or varlen_k is set. Otherwise creates one shared symbolic sequence count and fake
    # Int32 vectors for the M cumulative lengths (if varlen_m), K cumulative lengths (if
    # varlen_k) and the A index vector of length aidx_len (if gather_A); others are None.
        return None
    num_seqlens = cute.sym_int()
    return VarlenArguments(
        mCuSeqlensM=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_m else None
        ),
        mCuSeqlensK=(
            fake_tensor(Int32, (num_seqlens,), leading_dim=0, divisibility=4) if varlen_k else None
        ),
        mAIdx=(
            fake_tensor(Int32, (aidx_len,), leading_dim=0, divisibility=4) if gather_A else None
        ),
    )


def make_fake_gemm_tensors(
    a_dtype,
    b_dtype,
    d_dtype,
    c_dtype,
    a_major,
    b_major,
    d_major,
    c_major,
    varlen_m=False,
    varlen_k=False,
    gather_A=False,
):
    # [FUNCTION DOC] Create fake tensors mA, mB, mD, mC for compilation with shared symbolic
    # sizes. A None dtype yields None for that tensor. Returns (mA, mB, mD, mC, m, n, k, l).
    # Leading (contiguous) dimension follows the major labels. Divisibility is 16-byte alignment
    # per dtype (1 for absent D/C). Shapes: dense is (m,k,l), (n,k,l), (m,n,l); varlen_m drops
    # the batch dim of A, D, C and uses a fresh total-m symbol; varlen_k drops the batch dim of
    # and B and uses a fresh total-k symbol, with D and C keeping batch. With gather_A the
    # A row (or K) extent is an independent symbol.
    """Create fake tensors for mA, mB, mD, mC with shared sym_ints.
    Pass dtype=None to get None for that tensor (e.g. optional C).
    Returns (mA, mB, mD, mC, m, n, k, l).
    When varlen_m, m is total_m (flattened M of D/C). When varlen_k, k is total_k.
    """
    a_leading = 1 if a_major == "k" else 0
    b_leading = 1 if b_major == "k" else 0
    d_leading = 1 if d_major == "n" else 0
    c_leading = 1 if c_major == "n" else 0
    m, n, k, l = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    div_a = div_for_dtype(a_dtype)
    div_b = div_for_dtype(b_dtype)
    div_d = div_for_dtype(d_dtype) if d_dtype is not None else 1
    div_c = div_for_dtype(c_dtype) if c_dtype is not None else 1
    if varlen_m:
        # m is total_m in this case: the flattened M dimension of D/C
        m = cute.sym_int()
        a_m = cute.sym_int() if gather_A else m
        mA = fake_tensor(a_dtype, (a_m, k), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n), leading_dim=c_leading, divisibility=div_c)
    elif varlen_k:
        # k is total_k in this case: the flattened K dimension of A/B
        k = cute.sym_int()
        a_k = cute.sym_int() if gather_A else k
        mA = fake_tensor(a_dtype, (m, a_k), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    else:
        mA = fake_tensor(a_dtype, (m, k, l), leading_dim=a_leading, divisibility=div_a)
        mB = fake_tensor(b_dtype, (n, k, l), leading_dim=b_leading, divisibility=div_b)
        mD = fake_tensor(d_dtype, (m, n, l), leading_dim=d_leading, divisibility=div_d)
        mC = fake_tensor(c_dtype, (m, n, l), leading_dim=c_leading, divisibility=div_c)
    return mA, mB, mD, mC, m, n, k, l


def compile_gemm_kernel(
    GemmCls,
    a_dtype,
    tile_shape_mn,
    cluster_shape_mnk,
    pingpong,
    persistent,
    gather_A,
    is_dynamic_persistent,
    device_capacity,
    mA,
    mB,
    mD,
    mC,
    epi_args,
    scheduler_args,
    varlen_args,
    post_init=None,
    mSFA=None,
    mSFB=None,
    has_trace_ptr=False,
    use_tma_gather=False,
    concat_layout=None,
    num_warps=None,
):
    # [FUNCTION DOC] Instantiate a GEMM class for the device architecture and compile it with
    # TVM-FFI. Capability major 8 binds persistence, warp count and the arch number; 9 and 12
    # bind pingpong and persistence; 10 and 11 bind CLC persistence and TMA gather. The object is
    # built with float32 accumulation. Calls the optional post_init hook on it, uses a fake
    # environment stream, and passes scale-factor tensors only for capability majors other than
    # 8, 9, 12. A zero trace pointer is passed when has_trace_ptr is set, else None.
    # Returns the compiled kernel; the tvm-ffi option is enabled.
    """Build GemmCls instance, apply SM90 partial, and cute.compile with TVM-FFI."""
    if device_capacity[0] == 8:
        sm8x_kwargs = {"is_persistent": persistent, "num_warps": num_warps}
        sm8x_kwargs["arch"] = device_capacity[0] * 10 + device_capacity[1]
        GemmCls = partial(GemmCls, **sm8x_kwargs)
    elif device_capacity[0] in [9, 12]:
        GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
    elif device_capacity[0] in [10, 11]:
        GemmCls = partial(
            GemmCls,
            use_clc_persistence=is_dynamic_persistent,
            use_tma_gather=use_tma_gather,
        )
    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        tile_shape_mn,
        cluster_shape_mnk,
        gather_A=gather_A,
        concat_layout=concat_layout,
    )
    if post_init:
        post_init(gemm_obj)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    sf_args = () if device_capacity[0] in (8, 9, 12) else (mSFA, mSFB)
    # Trace pointer: Optional[Int64]. Compile with Int64(0) when tracing is
    # requested, None otherwise. TVM-FFI caches each variant separately.
    trace_ptr = Int64(0) if has_trace_ptr else None
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        mD,
        mC,
        epi_args,
        scheduler_args,
        varlen_args,
        stream,
        *sf_args,
        trace_ptr,
        options="--enable-tvm-ffi",
    )
