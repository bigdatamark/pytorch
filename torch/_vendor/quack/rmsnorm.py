# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# [MODULE DOC] Fused RMSNorm and LayerNorm forward/backward GPU kernels written with the CuTe
# DSL, plus Python entry points. Forward supports optional weight, bias, a fused residual add
# (with the summed residual written back) and optional per-head weights; backward is a
# persistent-grid kernel producing the input gradient and per-CTA partial weight and bias
# gradients that the host sums. Kernels are compiled lazily and cached, with launch configs
# from heuristics or an autotuner. Also contains an autograd Function, a functional wrapper, a
# drop-in nn module and reference implementations. Importing builds the autotune search spaces
# and wraps the tuned entry points but compiles nothing.

import math
from typing import Optional, Tuple, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync

import torch
from torch import Tensor

from . import utils
from . import copy_utils
from . import layout_utils
from .compile_utils import make_fake_tensor as fake_tensor
from .reduce import row_reduce
from .reduction_base import ReductionBase
from .cache import jit_cache
from .cute_dsl_utils import torch2cute_dtype_map
from .autotuner import autotune, AutotuneConfig
from .rmsnorm_config import (
    RmsNormBwdConfig,
    RmsNormFwdConfig,
    get_all_bwd_configs,
    get_all_fwd_configs,
    get_sm_count,
    prune_invalid_rmsnorm_bwd_configs,
    prune_invalid_rmsnorm_fwd_configs,
)
from cutlass.base_dsl.arch import Arch


def _bucket_T_hint(T_hint: int) -> int:
    # [FUNCTION DOC] HELPER: Bucket a row-count hint for the JIT cache key by rounding it up
    # to the next power of two (a value that is already a power of two is kept). A hint of
    # zero or below, the marker for an unknown or symbolic size, is returned as 0. Returns
    # an int.
    """Round ``T_hint`` up to the next power of 2 to bucket the JIT cache key.

    Each base-2 order of magnitude becomes a single bucket so adjacent T
    values share a compiled binary instead of triggering a recompile per row
    count. Buckets align with powers of 2 (..., 512, 1024, 2048, ...), which
    keeps the analytical heuristic thresholds (e.g. ``T_hint <= 1024``) exact
    at their power-of-2 boundaries. ``T_hint <= 0`` (the "unknown shape"
    SymInt sentinel) is preserved.
    """
    if T_hint <= 0:
        return 0
    return 1 << (T_hint - 1).bit_length()


def _ensure_contiguous(t):
    # [FUNCTION DOC] HELPER: Return a tensor whose last dimension has stride 1. While
    # compiling with torch.compile it always returns the result of a contiguous call (strides
    # cannot be inspected on fake tensors); otherwise the tensor itself is returned when the
    # last stride is already 1, avoiding a copy. Only the last dimension is checked.
    """Ensure last-dim stride is 1. Under torch.compile use unconditional .contiguous()
    (dynamo can't inspect strides on fake tensors); otherwise check first to avoid copies.
    """
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


class RMSNorm(ReductionBase):
    # [CLASS DOC] Forward kernel host object, a reduction-base subclass, for fused RMSNorm or
    # LayerNorm of rows of length N: optional residual add, weight and bias, and optional
    # saving of reciprocal std and mean. Holds the element dtype, the is_layernorm flag (two
    # reduction stages for LayerNorm, one for RMSNorm), the launch config and its unpacked
    # knobs. The cluster size is chosen at call time from the GPU architecture. Inputs may be
    # two-dimensional (rows by N) or three-dimensional (rows by heads by N, per-head mode).
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        is_layernorm: bool = False,
        config: Optional["RmsNormFwdConfig"] = None,
    ):
        # [FUNCTION DOC] Create the forward kernel object. Stage count of the reduction buffer
        # is 2 for LayerNorm and 1 for RMSNorm. When no config is given, one is taken from the
        # analytical heuristic for this N, dtype width and kind. Stores the config, the reload
        # and delayed weight-load options, and the thread, threads-per-row and cluster values.
        super().__init__(dtype, N, stage=2 if is_layernorm else 1)
        self.is_layernorm = is_layernorm
        if config is None:
            config = RmsNormFwdConfig.from_analytical_heuristic(
                N, dtype.width, is_layernorm=is_layernorm
            )
        self.config = config
        self.reload_from = config.reload_from
        self.delay_w_load = config.delay_w_load
        self._num_threads_val = config.num_threads
        self._threads_per_row_val = config.threads_per_row
        self._cluster_n_val = config.cluster_n

    def _num_threads(self):
        # [FUNCTION DOC] HELPER: Return the thread count from the config instead of the
        # base default.
        return self._num_threads_val

    def _threads_per_row(self):
        # [FUNCTION DOC] HELPER: Return the threads-per-row value from the config.
        return self._threads_per_row_val

    def _set_cluster_n(self):
        # [FUNCTION DOC] HELPER: Set cluster_n: 1 on architectures older than SM90, otherwise
        # the configured value limited to 8 on SM12x and 16 elsewhere. The architecture comes
        # from the active DSL. Side effect: updates the instance attribute.
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        # SM8x (Ampere/Ada) lacks cluster support
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        # SM12x supports cluster up to 8
        max_cluster = 8 if arch.major == 12 else 16
        self.cluster_n = min(self._cluster_n_val, max_cluster)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (b, N) or (b, H, N)
        mW: Optional[cute.Tensor],  # (N,) or (H, N)
        mB: Optional[cute.Tensor],  # (N,) or (H, N)
        mRes: Optional[cute.Tensor],  # (b, N) or (b, H, N)
        mO: cute.Tensor,  # (b, N) or (b, H, N)
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        stream: cuda.CUstream,
    ):
        # [FUNCTION DOC] JIT host entry. Takes the tensors for x, weight, bias, residual,
        # output, residual output, reciprocal std and mean (several optional) plus epsilon and
        # a stream. Asserts that x matches the configured dtype. Picks the cluster size, sets
        # the vector width to the 128-bit width of the widest participating dtype (reduced by
        # the greatest common divisor with N), caps the cluster, builds the tiled copy,
        # broadcasts weight and bias over the tile rows and the saved statistics over a
        # trailing column mode, and launches the kernel with one CTA per row tile, a grid axis
        # for the cluster and a grid axis for heads (1 unless the input is three-dimensional).
        # No return value.
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mRes, mW, mB, mO, mResO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        self._cap_cluster_n(vecsize)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW, mB = [
            layout_utils.expand(mT, dim=0, size=tiler_mn[0]) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]
        mRstd, mMean = [
            layout_utils.expand(mT, dim=cute.rank(mT), size=self.N)
            if const_expr(mT is not None)
            else None
            for mT in (mRstd, mMean)
        ]
        num_heads = mX.shape[1] if const_expr(cute.rank(mX) == 3) else 1
        self.kernel(
            mX, mW, mB, mRes, mO, mResO, mRstd, mMean, eps, tiler_mn, tiled_copy, threads_per_row
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, num_heads],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        # [FUNCTION DOC] Device kernel for one row tile (and one column slice per CTA of a
        # cluster). Allocates shared tiles for x and residual and the reduction buffer; in
        # per-head mode slices every tensor on the head given by the grid z index. Copies x
        # and residual into shared memory asynchronously (predicated for a ragged last column
        # block, skipping rows beyond the batch), loads weight and bias early unless delayed,
        # sums x and residual in FP32 and optionally writes that sum as the residual output.
        # For LayerNorm it row-reduces the sum for the mean (written out by the column-0
        # thread of the first cluster CTA when requested), optionally reloads x from shared or
        # global memory, zeroes out-of-range lanes and reduces the squared deviations; for
        # RMSNorm it reduces the sum of squares. The reciprocal std (inverse square root of
        # the mean square plus epsilon) is optionally stored. Then x may be reloaded,
        # normalised (centred for LayerNorm), scaled by weight, shifted by bias, converted to
        # the output dtype and stored for valid rows.
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        if const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # Slice per head
        if const_expr(cute.rank(mX) == 3):
            mX, mW, mB, mRes, mO, mResO, mRstd, mMean = [
                mT[None, bidz, None] if const_expr(mT is not None) else None
                for mT in (mX, mW, mB, mRes, mO, mResO, mRstd, mMean)
            ]

        shape = (cute.size(mX, mode=[0]), cute.size(mX, mode=[1]))
        idX = cute.make_identity_tensor(shape)
        # Slice for CTAs
        gX, gRes, gO, gResO, gRstd, gMean, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, mRstd, mMean, idX)
        ]
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        if const_expr(mRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
        tXgO = thr_copy_X.partition_D(gO)
        if const_expr(mResO is not None):
            tXgResO = thr_copy_X.partition_D(gResO)
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXrMean = thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_rmem_tensor_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_rmem_tensor_like(tXgRes)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        # Each copy will use the same predicate
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)

        mean, rstd = None, None
        if const_expr(self.is_layernorm):
            # LayerNorm: compute mean first, then variance
            sum_x = row_reduce(
                x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            mean = sum_x / shape[1]
            if const_expr(mMean is not None):
                # Only the thread corresponding to column 0 writes out the mean to gmem
                if (
                    tXcX[0][1] == 0
                    and row < shape[0]
                    and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
                ):
                    tXrMean[0] = mean
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            elif const_expr(self.reload_from == "gmem"):
                copy(tXgX, tXrX)
                x = tXrX.load().to(cute.Float32)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
                    x += tXrRes.load().to(cute.Float32)
            x_centered = x - mean
            if const_expr(not is_even_N):
                # OOB lanes are zero-filled for the mean pass, but they must contribute zero
                # to the variance pass (not mean^2 from (0 - mean)^2).
                tXrX_centered = cute.make_rmem_tensor_like(tXrX, Float32)
                tXrX_centered.store(x_centered)
                utils.fill_oob(tXrX_centered, tXpX, fill_value=Float32.zero)
                x_centered = tXrX_centered.load()
            sum_sq_x_sub_mean = row_reduce(
                x_centered * x_centered,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
            rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)
        else:
            # RMSNorm: compute sum of squares directly
            mean = const_expr(0.0)
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if const_expr(self.delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                copy(tXgX, tXrX)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                x += tXrRes.load().to(cute.Float32)
        x_hat = (x - mean) * rstd if const_expr(self.is_layernorm) else x * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


def _rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
    is_layernorm: bool = False,
) -> None:
    # [FUNCTION DOC] HELPER: Validate and dispatch the forward kernel in place. Asserts that
    # x, weight and residual are float16, bfloat16 or float32; returns immediately for an
    # empty x. Writes the result into the given output (and residual output, reciprocal std
    # and mean buffers when provided) by running the compiled kernel for the dtypes, N, which
    # optional outputs exist, LayerNorm flag, and per-head mode (true when weight or bias is
    # two-dimensional) found in the arguments. Returns None.
    # KNOWN ISSUE: The docstring says it returns the normalised tensor, but the function
    # returns nothing and writes into the output argument; it also lists only some of
    # the parameters.
    """RMSNorm/LayerNorm forward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,) or (H, N) for per-head weight
        eps: Small value for numerical stability
        is_layernorm: If True, compute LayerNorm instead of RMSNorm
    Returns:
        Normalized output tensor of same shape as x
    """
    # TVM FFI validates tensor devices at runtime.
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.dtype in supported_types, "Weight must be float32, float16 or bfloat16"
    if residual is not None:
        assert residual.dtype in supported_types, "Residual must be float16, bfloat16, or float32"
    if x.numel() == 0:
        return

    N = x.size(-1)
    per_head = (weight is not None and weight.dim() == 2) or (bias is not None and bias.dim() == 2)
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, out, weight, bias, residual, residual_out]
    ]
    _compile_rmsnorm_fwd(
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
        mean is not None,
        is_layernorm,
        per_head,
    )(x, weight, bias, residual, out, residual_out, rstd, mean, eps)


@jit_cache
def _compile_rmsnorm_fwd(
    dtype,
    out_dtype,
    res_dtype,
    weight_dtype,
    bias_dtype,
    res_out_dtype,
    N,
    has_rstd,
    has_mean,
    is_layernorm,
    per_head,
    config: Optional[RmsNormFwdConfig] = None,
):
    # [FUNCTION DOC] HELPER: Cached (by the jit cache decorator, keyed on the arguments)
    # compile step for the forward kernel. Creates symbolic batch (and head) sizes, fake
    # tensors for every present dtype with a divisibility hint equal to the common divisor of
    # N and the 128-bit vector widths, builds the kernel object with the optional config, and
    # compiles it with the TVM FFI option and a fake stream. Returns the compiled callable
    # taking x, weight, bias, residual, out, residual out, reciprocal std, mean and epsilon.
    batch_sym = cute.sym_int()
    head_sym = cute.sym_int() if per_head else None
    batch_shape = (batch_sym, head_sym) if per_head else (batch_sym,)
    all_dtypes = [dtype, out_dtype, res_dtype, weight_dtype, bias_dtype, res_out_dtype]
    div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
    x_cute, out_cute, res_cute, res_out_cute = [
        fake_tensor(dt, (*batch_shape, N), div)
        for dt in [dtype, out_dtype, res_dtype, res_out_dtype]
    ]
    weight_shape = (head_sym, N) if per_head else (N,)
    weight_cute, bias_cute = [
        fake_tensor(dt, weight_shape, div) for dt in [weight_dtype, bias_dtype]
    ]
    rstd_cute = fake_tensor(Float32, batch_shape) if has_rstd else None
    mean_cute = fake_tensor(Float32, batch_shape) if has_mean else None
    return cute.compile(
        RMSNorm(dtype, N, is_layernorm=is_layernorm, config=config),
        x_cute,
        weight_cute,
        bias_cute,
        res_cute,
        out_cute,
        res_out_cute,
        rstd_cute,
        mean_cute,
        Float32(0),  # eps, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    # [FUNCTION DOC] Public untuned forward RMSNorm. Allocates the output (dtype out_dtype or
    # that of x), the optional FP32 reciprocal std (shape of x without the last dimension,
    # only when store_rstd is set) and a residual output when a residual is given or
    # residual_dtype differs from the x dtype (residual_dtype defaults to the residual dtype).
    # Runs the forward in RMSNorm mode with the heuristic config. Returns a tuple of output,
    # residual output (the input x itself when no residual output was allocated) and
    # reciprocal std or None.
    # Need to wrap to handle the case where residual_out is a alias of x, which makes torch.library
    # and torch.compile unhappy. Also allocate memory for out and residual_out if they are None
    # so that _layer_norm_fwd_impl doesn't have to return them.
    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty(*x.shape[:-1], device=x.device, dtype=torch.float32) if store_rstd else None
    if residual is not None and residual_dtype is None:
        residual_dtype = residual.dtype
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty_like(
            x, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
    else:
        residual_out = None
    _rmsnorm_fwd(x, weight, out, bias, rstd, None, residual, residual_out, eps, False)
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if residual_out is None:
        residual_out = x
    return out, residual_out, rstd


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_fwd_configs()],
    key=["is_layernorm", "per_head"],
    prune_configs_by={"early_config_prune": prune_invalid_rmsnorm_fwd_configs},
)
def rmsnorm_fwd_tuned(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
    is_layernorm: bool = False,
    per_head: bool = False,
    config: Optional[RmsNormFwdConfig] = None,
) -> None:
    # [FUNCTION DOC] Autotuned forward entry, wrapped by the autotune decorator which supplies
    # a config chosen from the forward search space (cached per shape, dtype, LayerNorm flag
    # and per-head flag). Raises RuntimeError when called without a config. Writes into the
    # provided output buffers by running the kernel compiled for that config; unlike the plain
    # helper it does no dtype validation and no empty-tensor early exit. Returns None.
    """Autotuned RMSNorm/LayerNorm forward dispatch.

    The ``@autotune`` decorator injects ``config`` from the exhaustive search
    space at first call for a given (shape, dtype, ``is_layernorm``, ``per_head``)
    and caches the winner for subsequent calls. The un-tuned counterpart is
    :func:`rmsnorm_fwd`, which uses the analytical heuristic.
    """
    if config is None:
        raise RuntimeError(
            "rmsnorm_fwd_tuned requires a config (provided automatically by "
            "the @autotune decorator). Use rmsnorm_fwd for the un-tuned path."
        )
    N = x.size(-1)
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, out, weight, bias, residual, residual_out]
    ]
    _compile_rmsnorm_fwd(
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
        mean is not None,
        is_layernorm,
        per_head,
        config=config,
    )(x, weight, bias, residual, out, residual_out, rstd, mean, eps)


def rmsnorm_ref(x, w=None, bias=None, residual=None, eps=1e-6):
    # [FUNCTION DOC] Reference forward in FP32: adds the residual if present, divides by the
    # root of the mean square over the last dimension plus epsilon, multiplies by weight and
    # adds bias when given. Returns the output cast to the dtype of x, or a pair of that
    # output and the summed residual cast to the residual dtype when a residual is passed.
    x_f32 = x.float()
    if residual is not None:
        residual_f32 = residual.float()
        x_f32 = x_f32 + residual_f32
    x_norm = x_f32 / (torch.sqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps))
    out = x_norm * w if w is not None else x_norm
    if bias is not None:
        out = out + bias.float()
    if residual is None:
        return out.to(x.dtype)
    else:
        return out.to(x.dtype), x_f32.to(residual.dtype)


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6):
    # [FUNCTION DOC] Reference backward in FP32 for 2D x. Forms the normalised input from the
    # saved reciprocal std, multiplies dout by the weight when present, computes the input
    # gradient as the weighted gradient minus the normalised input times its row mean, scaled
    # by the reciprocal std, and the weight gradient as the column sum of dout times the
    # normalised input. Returns the input gradient in the dtype of x and the weight gradient
    # in the weight dtype (None without weight).
    # KNOWN ISSUE: The eps argument is accepted but never used; the reciprocal std unsqueeze
    # on dimension 1 means only 2D inputs are supported.
    """Reference implementation for RMSNorm backward pass."""
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    if w is not None:
        wdy = dout * w
    else:
        wdy = dout
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)

    # dL/dW
    if w is not None:
        dw = (dout * x_hat).sum(dim=0)
        return dx.to(x.dtype), dw.to(w.dtype)
    else:
        return dx.to(x.dtype), None


class RMSNormBackward(ReductionBase):
    # [CLASS DOC] Backward kernel host object, a reduction-base subclass. Persistent grid
    # (launched with the given SM-style CTA count): each CTA loops over row tiles, computes
    # the input gradient and accumulates FP32 partial weight and bias gradients. Holds the
    # dtype, N, per-head flag, the config and its reload options, and USE_TMA (true only if
    # the config asks for TMA, the mode is not per-head, and the per-CTA row byte sizes of x
    # and dout are multiples of 16). Uses a two-stage reduction buffer. Optional inputs are
    # the weight and the residual-output gradient; optional outputs are the weight, bias and
    # residual gradients.
    def __init__(
        self,
        dtype: cutlass.Numeric,
        N: int,
        dout_dtype: Optional[Type[cutlass.Numeric]] = None,
        T_hint: int = 0,
        per_head: bool = False,
        config: Optional["RmsNormBwdConfig"] = None,
    ):
        # [FUNCTION DOC] Create the backward kernel object (two reduction stages, FP32
        # reduction). dout dtype defaults to the x dtype. Without a config, takes one from the
        # analytical backward heuristic using N, both element widths and the row-count hint.
        # Stores the config and reload options, computes the TMA eligibility flag, and raises
        # ValueError when N is above 131072 and the dtype is 32 bits or wider (not enough
        # shared memory).
        # 2 stages for double buffering when computing mean of x_hat * wdy
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        dout_width = dout_dtype.width if dout_dtype is not None else dtype.width
        if config is None:
            config = RmsNormBwdConfig.from_analytical_heuristic(
                N, dtype.width, dout_width, T_hint=T_hint
            )
        self.config = config
        self.reload_wdy = config.reload_wdy
        self.reload_x = config.reload_x
        self.per_head = per_head
        self._num_threads_val = config.num_threads
        self._threads_per_row_val = config.threads_per_row
        self._cluster_n_val = config.cluster_n
        tile_n = N // max(1, config.cluster_n)
        row_bytes_x = tile_n * dtype.width // 8
        row_bytes_do = tile_n * dout_width // 8
        self.USE_TMA = (
            config.use_tma and not per_head and row_bytes_x % 16 == 0 and row_bytes_do % 16 == 0
        )
        self._can_use_tma = self.USE_TMA
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            # Not enough smem
            raise ValueError("RMSNormBackward does not support N > 128k with dtype >= 32 bits")

    def _num_threads(self):
        # [FUNCTION DOC] HELPER: Return the thread count from the config.
        return self._num_threads_val

    def _threads_per_row(self):
        # [FUNCTION DOC] HELPER: Return the threads-per-row value from the config.
        return self._threads_per_row_val

    def _set_cluster_n(self):
        # [FUNCTION DOC] HELPER: Set cluster_n: 1 before SM90, otherwise the configured value
        # limited to 8 on SM12x and 16 elsewhere. Side effect: updates the instance attribute.
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        # SM8x (Ampere/Ada) lacks cluster support
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        max_cluster = 8 if arch.major == 12 else 16
        self.cluster_n = min(self._cluster_n_val, max_cluster)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        # [FUNCTION DOC] JIT host entry for the backward. Takes x, weight, dout,
        # residual-output gradient, saved reciprocal std, and output tensors for dx, partial
        # dW, partial dB and residual gradient, the CTA count and a stream. Asserts the x
        # dtype, picks and caps the cluster size, builds the tiled copy with a vector width
        # from the widest participating dtype, broadcasts the weight over tile rows, creates
        # TMA atoms and views for x and dout when TMA is used, and launches the kernel with
        # grid of CTA count by cluster size by heads (heads only in per-head mode). No
        # return value.
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mW, mdO, mdResO, mdX, mdRes] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        self._cap_cluster_n(vecsize)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = (
            layout_utils.expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None
        )
        use_tma = const_expr(self.USE_TMA)
        if const_expr(use_tma):
            tma_smem_layout = cute.make_ordered_layout(tiler_mn, order=(1, 0))
            tma_op = cpasync.CopyBulkTensorTileG2SOp()
            tma_atom_X, mX_tma = cpasync.make_tiled_tma_atom(tma_op, mX, tma_smem_layout, tiler_mn)
            tma_atom_dO, mdO_tma = cpasync.make_tiled_tma_atom(
                tma_op, mdO, tma_smem_layout, tiler_mn
            )
        else:
            tma_atom_X, mX_tma, tma_atom_dO, mdO_tma = None, None, None, None
        num_blocks = sm_count
        num_heads = mX.shape[1] if const_expr(self.per_head) else 1
        self.kernel(
            mX,
            mW,
            mdO,
            mdResO,
            mRstd,
            mdX,
            mdW,
            mdB,
            mdRes,
            tma_atom_X,
            mX_tma,
            tma_atom_dO,
            mdO_tma,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[num_blocks, self.cluster_n, num_heads],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        tma_atom_X: Optional[cute.CopyAtom],
        mX_tma: Optional[cute.Tensor],
        tma_atom_dO: Optional[cute.CopyAtom],
        mdO_tma: Optional[cute.Tensor],
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        # [FUNCTION DOC] Persistent device kernel. Each CTA walks row tiles with a grid-sized
        # stride. Setup: shared multi-stage buffers for x and dout, reduction buffer and
        # cluster barriers (and TMA pipeline barriers), per-head slicing, fp32 partial
        # weight/bias accumulators zeroed, the weight tile loaded once into registers (zero
        # padded on ragged N), then a prefetch of one fewer than the stage count of tiles by
        # TMA (issued by warp 0) or cp.async (with zero fill for rows past the batch). Per
        # tile: issue the next prefetch, load reciprocal std and the optional residual-output
        # gradient, wait for the data, form the normalised x and weight times dout, row-reduce
        # their mean product (with full/empty barrier handshakes when clustered), optionally
        # reload x or the product from shared memory, compute the input gradient, add the
        # residual gradient, store it (and the residual gradient copy), and accumulate dout
        # times normalised x into the weight partial and dout into the bias partial. At the
        # end the row partials of the CTA are summed via shared memory (reusing the x buffer
        # as FP32) and written by one row to the partial output indexed by the CTA index;
        # clustered CTAs wait on the last empty barrier so peers do not exit early.
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if const_expr(self.per_head):
            bidx_start, _, bidz = cute.arch.block_idx()
        else:
            bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        # Slice per head
        if const_expr(self.per_head):
            mX, mW, mdO, mdResO, mdX, mdW, mdB, mdRes = [
                mT[None, bidz, None] if const_expr(mT is not None) else None
                for mT in (mX, mW, mdO, mdResO, mdX, mdW, mdB, mdRes)
            ]
            mRstd = mRstd[None, bidz]

        shape = mX.shape
        M = shape[0]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        USE_TMA = const_expr(self.USE_TMA)
        n_smem_stages = const_expr(self.config.smem_stages)
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], n_smem_stages), order=(1, 0, 2)
        )
        smem_align = const_expr(128 if USE_TMA else 16)
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=smem_align)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=smem_align)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout, is_persistent=True
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdResO, gdX, gdRes, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
            for mT in (mX, mdO, mdResO, mdX, mdRes, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        if const_expr(mdResO is not None):
            tXgdResO = thr_copy_X.partition_S(gdResO)
        if const_expr(mdRes is not None):
            tXgdRes = thr_copy_X.partition_D(gdRes)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_rmem_tensor_like(thr[None, None, None, 0]) for thr in (tXgX, tXgdO, tXgdX)
        ]
        tXrdResO = None
        if const_expr(mdResO is not None):
            tXrdResO = cute.make_rmem_tensor_like(tXgdResO[None, None, None, 0])
        tXrdRes = None
        if const_expr(mdRes is not None):
            tXrdRes = cute.make_rmem_tensor_like(tXgdRes[None, None, None, 0])

        # This doesn't change across iterations
        tXpX = (
            None
            if is_even_N
            else copy_utils.predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        )
        # Each copy will use the same number of elements as X
        copy = partial(copy_utils.copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            # Always compute partial weight gradients in fp32
            tXrdW = cute.make_rmem_tensor_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            # Always compute partial bias gradients in fp32
            tXrdB = cute.make_rmem_tensor_like(tXgdB, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        NUM_PIPE_STAGES = const_expr(self.config.smem_stages)

        if const_expr(USE_TMA):
            tma_mbar_ptr = smem.allocate_array(Int64, num_elems=NUM_PIPE_STAGES * 2)

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_rmem_tensor_like(tXgW)
            # Need this, otherwise rW can have arbitrary values that changes the reduction
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy(tXgW, tXrW)
            # No-op fp32 round-trip; pins tXrW into stable registers across the loop.
            tXrW.store((tXrW.load().to(Float32) + Float32(0.0)).to(tXrW.element_type))

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, NUM_PIPE_STAGES
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_PIPE_STAGES
        )
        if const_expr(USE_TMA):
            num_threads_total = cute.size(tiled_copy)
            producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads_total)
            tma_bytes_x = const_expr(cute.size(tiler_mn) * mX.element_type.width // 8)
            tma_bytes_do = const_expr(cute.size(tiler_mn) * mdO.element_type.width // 8)
            tma_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=tma_mbar_ptr,
                num_stages=NUM_PIPE_STAGES,
                producer_group=producer_group,
                consumer_group=consumer_group,
                tx_count=tma_bytes_x + tma_bytes_do,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            gX_tma = cute.local_tile(mX_tma, tiler_mn, (None, cluster_y))
            gdO_tma = cute.local_tile(mdO_tma, tiler_mn, (None, cluster_y))
            tXsX_tma, tXgX_tma = cpasync.tma_partition(
                tma_atom_X,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX_tma, 0, 2),
            )
            tXsdO_tma, tXgdO_tma = cpasync.tma_partition(
                tma_atom_dO,
                0,
                cute.make_layout(1),
                cute.group_modes(sdO, 0, 2),
                cute.group_modes(gdO_tma, 0, 2),
            )

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        next_wave_work_id = (NUM_PIPE_STAGES - 1) * gdim
        next_wave_row_id = next_wave_work_id * tiler_mn[0]

        M_ceil = cute.ceil_div(M, tiler_mn[0])
        if const_expr(USE_TMA):
            for prefetch_iter in cutlass.range_constexpr(const_expr(NUM_PIPE_STAGES - 1)):
                init_bidx = bidx_start + prefetch_iter * gdim
                if warp_id == 0:
                    if init_bidx < M_ceil:
                        tma_pipeline.producer_acquire(producer_state)
                        pipe_bar = tma_pipeline.producer_get_barrier(producer_state)
                        cute.copy(
                            tma_atom_X,
                            tXgX_tma[None, init_bidx],
                            tXsX_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        cute.copy(
                            tma_atom_dO,
                            tXgdO_tma[None, init_bidx],
                            tXsdO_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        tma_pipeline.producer_commit(producer_state)
                        producer_state.advance()
        else:
            # Pre-issue NUM_PIPE_STAGES-1 prefetches into smem stages 0..N-2.
            # The bidx loop then maintains exactly NUM_PIPE_STAGES groups in flight
            # via cp_async_wait_group(NUM_PIPE_STAGES - 1).
            for prefetch_iter in cutlass.range_constexpr(const_expr(NUM_PIPE_STAGES - 1)):
                init_bidx = bidx_start + prefetch_iter * gdim
                init_row = tXcX[None, None, None, init_bidx][0][0]
                if init_row < M:
                    copy(
                        tXgX[None, None, None, init_bidx],
                        tXsX[None, None, None, producer_state.index],
                        is_async=True,
                    )
                    copy(
                        tXgdO[None, None, None, init_bidx],
                        tXsdO[None, None, None, producer_state.index],
                        is_async=True,
                    )
                else:
                    if const_expr(tiler_mn[0] > 1):
                        utils.fill_oob(
                            tXsX[None, None, None, producer_state.index],
                            None,
                            fill_value=mX.element_type.zero,
                        )
                        utils.fill_oob(
                            tXsdO[None, None, None, producer_state.index],
                            None,
                            fill_value=mdO.element_type.zero,
                        )
                cute.arch.cp_async_commit_group()
                producer_state.advance()

        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if const_expr(USE_TMA):
                ahead_bidx = bidx + next_wave_work_id
                if warp_id == 0:
                    if ahead_bidx < M_ceil:
                        tma_pipeline.producer_acquire(producer_state)
                        pipe_bar = tma_pipeline.producer_get_barrier(producer_state)
                        cute.copy(
                            tma_atom_X,
                            tXgX_tma[None, ahead_bidx],
                            tXsX_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        cute.copy(
                            tma_atom_dO,
                            tXgdO_tma[None, ahead_bidx],
                            tXsdO_tma[None, producer_state.index],
                            tma_bar_ptr=pipe_bar,
                        )
                        tma_pipeline.producer_commit(producer_state)
                        producer_state.advance()
            else:
                # cp.async: prefetch the (NUM_PIPE_STAGES-1)-ahead batch into the
                # smem slot we're about to free up.
                ahead_bidx = bidx + next_wave_work_id
                if row + next_wave_row_id < M:
                    copy(
                        tXgX[None, None, None, ahead_bidx],
                        tXsX[None, None, None, producer_state.index],
                        is_async=True,
                    )
                    copy(
                        tXgdO[None, None, None, ahead_bidx],
                        tXsdO[None, None, None, producer_state.index],
                        is_async=True,
                    )
                else:
                    if const_expr(tiler_mn[0] > 1):
                        utils.fill_oob(
                            tXsX[None, None, None, producer_state.index],
                            None,
                            fill_value=mX.element_type.zero,
                        )
                        utils.fill_oob(
                            tXsdO[None, None, None, producer_state.index],
                            None,
                            fill_value=mdO.element_type.zero,
                        )
                cute.arch.cp_async_commit_group()
                producer_state.advance()
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            if const_expr(mdResO is not None):
                if row < M or tiler_mn[0] == 1:
                    copy(tXgdResO[None, None, None, bidx], tXrdResO)
                elif tiler_mn[0] > 1:
                    tXrdResO.fill(0.0)
            if const_expr(USE_TMA):
                tma_pipeline.consumer_wait(consumer_state)
            else:
                cute.arch.cp_async_wait_group(const_expr(NUM_PIPE_STAGES - 1))
            smem_stage = consumer_state.index
            cute.autovec_copy(tXsX[None, None, None, smem_stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, smem_stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = x * rstd
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                row_reduce(
                    x_hat * wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                # Need this fence since the STAS from the producer is using the async proxy.
                cute.arch.fence_view_async_shared()
                # It's faster to have 1 lane per warp to signal the mbar, rather than all lanes
                # Requires adjusting the thread_count when initializing the mbar
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, smem_stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    wdy *= tXrW.load().to(Float32)

            if const_expr(self.reload_x == "smem"):
                cute.autovec_copy(tXsX[None, None, None, smem_stage], tXrX)
                x = tXrX.load().to(cute.Float32)
                x_hat = x * rstd

            dx = (wdy - x_hat * mean_xhat_wdy) * rstd
            if const_expr(mdResO is not None):
                dx += tXrdResO.load().to(cute.Float32)
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy(tXrdX, tXgdX[None, None, None, bidx])
            if const_expr(mdRes is not None):
                tXrdRes.store(dx.to(tXrdRes.element_type))
                if row < M or tiler_mn[0] == 1:
                    copy(tXrdRes, tXgdRes[None, None, None, bidx])
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            if const_expr(USE_TMA):
                tma_pipeline.sync_object_empty.arrive(
                    consumer_state.index, tma_pipeline.consumer_mask
                )
            consumer_state.advance()

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                # reduction of dw_partial within the same threadblock
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_rmem_tensor_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy(tXrdW, tXgdW)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_rmem_tensor_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy(tXrdB, tXgdB)
        else:
            # dw is already in fp32, so we can directly copy to global memory
            if const_expr(mdW is not None):
                copy(tXrdW, tXgdW)
            if const_expr(mdB is not None):
                copy(tXrdB, tXgdB)

        if const_expr(self.cluster_n > 1):  # Prevent cluster from exiting early
            # Assume state contains that next useful buffer
            # So we only need to advance to num_stages - 1 times to last used buffer
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


def _rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Optional[Tensor],
    db_partial: Optional[Tensor] = None,
    dresidual_out: Optional[Tensor] = None,
    dresidual: Optional[Tensor] = None,
    sm_count: Optional[int] = None,
) -> None:
    # [FUNCTION DOC] HELPER: Validate and run the backward in place. Asserts x has 2 or 3
    # dimensions, supported dtypes, and that residual gradients match the shape of x; returns
    # at once for empty x. The per-head mode follows from three dimensions. CTA count comes
    # from the first dimension of the partial weight or bias buffer, otherwise it must be
    # passed (asserted). Fills dx, the partial weight and bias buffers and the residual
    # gradient using the cached compiled kernel keyed by dtypes, N, flags and the bucketed row
    # count (0 when the row count is symbolic). Returns None.
    # KNOWN ISSUE: The docstring claims it returns a tuple of dx and dw; the function returns
    # None and writes into the passed buffers. It also omits most parameters.
    """RMSNorm backward pass.
    Args:
        x: Input tensor of shape (M, N) or (M, H, N) for per-head
        weight: Optional weight tensor of shape (N,) or (H, N) for per-head
        dout: Upstream gradients tensor of shape (M, N) or (M, H, N)
        rstd: Reciprocal standard deviation tensor of shape (M,) or (M, H)
    Returns:
        Tuple of (dx, dw) where:
        - dx: Input gradients tensor of same shape as x
        - dw: Weight gradients tensor of same shape as weight (or None if weight is None)
    """
    assert x.dim() in (2, 3), "Input must be 2D or 3D"
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    per_head = x.dim() == 3
    if weight is not None:
        assert weight.dtype in supported_types, "Weight must be float32, float16 or bfloat16"
    if dresidual_out is not None:
        assert dresidual_out.shape == x.shape
        assert dresidual_out.dtype in supported_types, (
            "Residual must be float16, bfloat16, or float32"
        )
    if dresidual is not None:
        assert dresidual.shape == x.shape
        assert dresidual.dtype in supported_types, "Residual must be float16, bfloat16, or float32"
    if x.numel() == 0:
        return

    N = x.size(-1)
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
    dtype, dout_dtype, dx_dtype, weight_dtype, dres_dtype, dres_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, dout, dx, weight, dresidual, dresidual_out]
    ]
    T_hint = _bucket_T_hint(int(x.size(0)) if not isinstance(x.size(0), torch.SymInt) else 0)
    _compile_rmsnorm_bwd(
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        db_partial is not None,
        dres_dtype,
        dres_out_dtype,
        dw_partial is not None,
        per_head,
        T_hint=T_hint,
    )(x, weight, dout, dresidual_out, rstd, dx, dw_partial, dresidual, db_partial, sm_count)


@jit_cache
def _compile_rmsnorm_bwd(
    N,
    dtype,
    dout_dtype,
    dx_dtype,
    weight_dtype,
    has_db_partial,
    dres_dtype,
    dres_out_dtype,
    has_dw_partial,
    per_head=False,
    T_hint=0,
    config: Optional[RmsNormBwdConfig] = None,
):
    # [FUNCTION DOC] HELPER: Cached compile step for the backward kernel. Creates symbolic
    # batch sizes (a separate one for the partial buffers) and optional head size, fake
    # tensors for x, dout, dx and residual gradients with a divisibility hint, the weight,
    # FP32 reciprocal std and FP32 partial weight/bias buffers (present only when requested),
    # builds the kernel object with the row-count hint, per-head flag and optional config, and
    # compiles it with the TVM FFI option. Returns the compiled callable.
    batch_sym, batch_partial_sym = cute.sym_int(), cute.sym_int()
    head_sym = cute.sym_int() if per_head else None
    batch_shape = (batch_sym, head_sym) if per_head else (batch_sym,)
    all_dtypes = [dtype, dout_dtype, dx_dtype, dres_dtype, dres_out_dtype]
    div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
    x_cute, dout_cute, dx_cute, dres_out_cute, dres_cute = [
        fake_tensor(dt, (*batch_shape, N), div)
        for dt in [dtype, dout_dtype, dx_dtype, dres_out_dtype, dres_dtype]
    ]
    weight_shape = (head_sym, N) if per_head else (N,)
    weight_cute = fake_tensor(weight_dtype, weight_shape, div)
    rstd_cute = fake_tensor(Float32, batch_shape)
    dw_shape = (batch_partial_sym, head_sym, N) if per_head else (batch_partial_sym, N)
    dw_partial_cute = fake_tensor(Float32, dw_shape, div) if has_dw_partial else None
    db_partial_cute = fake_tensor(Float32, dw_shape, div) if has_db_partial else None
    return cute.compile(
        RMSNormBackward(
            dtype,
            N,
            dout_dtype=dout_dtype,
            T_hint=T_hint,
            per_head=per_head,
            config=config,
        ),
        x_cute,
        weight_cute,
        dout_cute,
        dres_out_cute,
        rstd_cute,
        dx_cute,
        dw_partial_cute,
        dres_cute,
        db_partial_cute,
        0,  # sm_count, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor] = None,  # grad wrt residual_out
    has_bias: bool = False,
    has_residual: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    # [FUNCTION DOC] Public untuned backward. Allocates dx, a separate residual gradient
    # only if the incoming residual-output gradient has a different dtype from dx, FP32
    # partial weight and optional bias buffers with one slice per CTA (the CTA count comes
    # from the device SM count, divided by the number of heads in per-head mode, at least
    # 1). Runs the kernel for non-empty inputs and sums the partials over the CTA axis,
    # casting to the weight dtype; for empty inputs returns zero gradients shaped like the
    # weight. When a residual existed and no separate residual gradient was made, it is the
    # same tensor as dx. Returns dx, weight gradient, bias gradient and residual gradient
    # (each possibly None).
    # KNOWN ISSUE: The bias gradient is cast using the weight dtype and, for empty input, is
    # shaped like the weight; calling with a bias gradient request but no weight fails because
    # the weight is None.
    device = x.device
    N = x.size(-1)
    per_head = x.dim() == 3
    dx = torch.empty_like(x)
    if dresidual_out is not None and dresidual_out.dtype != dx.dtype:
        dresidual = torch.empty_like(x, dtype=dresidual_out.dtype)
    else:
        dresidual = None
    sm_count = get_sm_count(N, device)
    if per_head:
        H = x.size(1)
        sm_count = max(round(sm_count / H), 1)
    else:
        H = None
    if weight is not None:
        # Always store partial gradients in fp32 for numerical accuracy
        dw_shape = (sm_count, H, N) if per_head else (sm_count, N)
        dw_partial = torch.empty(dw_shape, device=device, dtype=torch.float32)
    else:
        dw_partial = None
    db_shape = (sm_count, H, N) if per_head else (sm_count, N)
    db_partial = torch.empty(db_shape, device=device, dtype=torch.float32) if has_bias else None

    if x.numel() > 0:
        _rmsnorm_bwd(
            x, weight, dout, rstd, dx, dw_partial, db_partial, dresidual_out, dresidual, sm_count
        )
        # we have summed the partial gradients in fp32, now we convert back to the weight dtype
        dw = dw_partial.sum(dim=0).to(weight.dtype) if weight is not None else None
        db = db_partial.sum(dim=0).to(weight.dtype) if has_bias else None
    else:
        dw = torch.zeros_like(weight) if weight is not None else None
        db = torch.zeros_like(weight) if has_bias else None
    # dresidual is the same as dx in this case
    if has_residual and dresidual is None:
        dresidual = dx
    return dx, dw, db, dresidual


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_bwd_configs()],
    key=["per_head", "has_dw_partial", "has_db_partial"],
    prune_configs_by={"early_config_prune": prune_invalid_rmsnorm_bwd_configs},
)
def rmsnorm_bwd_tuned(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Optional[Tensor] = None,
    db_partial: Optional[Tensor] = None,
    dresidual_out: Optional[Tensor] = None,
    dresidual: Optional[Tensor] = None,
    sm_count: Optional[int] = None,
    per_head: bool = False,
    has_dw_partial: bool = False,
    has_db_partial: bool = False,
    config: Optional[RmsNormBwdConfig] = None,
) -> None:
    # [FUNCTION DOC] Autotuned backward entry; the autotune decorator supplies a config from
    # the backward search space, cached per shape, dtype and the per-head and partial-buffer
    # flags. Raises RuntimeError when called without config. CTA count is taken from the
    # partial weight buffer, else the partial bias buffer, else the argument; raises
    # ValueError if none of them is available. Runs the kernel compiled for that config with
    # no row-count hint, writing into the supplied buffers. Does no dtype validation or
    # empty-input exit. Returns None.
    """Autotuned RMSNorm backward dispatch.

    The ``@autotune`` decorator injects ``config`` from the exhaustive search
    space at first call for a given (shape, dtype, ``per_head``, has_*) and
    caches the winner for subsequent calls. The un-tuned counterpart is
    :func:`rmsnorm_bwd`, which uses the analytical heuristic.
    """
    if config is None:
        raise RuntimeError(
            "rmsnorm_bwd_tuned requires a config (provided automatically by "
            "the @autotune decorator). Use rmsnorm_bwd for the un-tuned path."
        )
    # The persistent grid size is encoded in the partial-accumulator shape
    # (dw_partial / db_partial have shape (sm_count, ..., N)). Derive
    # sm_count from there when available; require it explicitly only when
    # neither buffer is provided. Mirrors the _rmsnorm_bwd torch-op
    # contract.
    if dw_partial is not None:
        sm_count = dw_partial.shape[0]
    elif db_partial is not None:
        sm_count = db_partial.shape[0]
    elif sm_count is None:
        raise ValueError(
            "rmsnorm_bwd_tuned: sm_count is required when neither dw_partial "
            "nor db_partial is provided."
        )
    N = x.size(-1)
    dtype, dout_dtype, dx_dtype, weight_dtype, dres_dtype, dres_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, dout, dx, weight, dresidual, dresidual_out]
    ]
    _compile_rmsnorm_bwd(
        N,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        has_db_partial,
        dres_dtype,
        dres_out_dtype,
        has_dw_partial,
        per_head,
        T_hint=0,
        config=config,
    )(x, weight, dout, dresidual_out, rstd, dx, dw_partial, dresidual, db_partial, sm_count)


class RMSNormFunction(torch.autograd.Function):
    # [CLASS DOC] Autograd function for the fused norm. Expects the batch dimensions already
    # flattened by the wrapper so the ranks stay fixed across recompiles. Forward saves, for
    # backward, the normalised input (x, or the summed residual when a residual was used),
    # weight and reciprocal std, plus flags for bias, residual and prenorm mode.
    """Autograd wrapper for rmsnorm.

    All input reshaping (flattening batch dims, per-head layout) is done in the
    rmsnorm() wrapper BEFORE calling .apply(). This function receives already-
    flattened tensors so that tensor ranks never change between recompilations,
    which is required for torch.compile compatibility.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        eps=1e-6,
        prenorm=False,
    ):
        # [FUNCTION DOC] Forward. Makes x and the residual last-dim contiguous, decides that
        # the reciprocal std is needed when any of x, weight or bias requires grad, and runs
        # the untuned forward. Stores the saved tensors and flags on the context. Returns the
        # output alone, or the pair of output and residual output when prenorm is set.
        # Gradients are defined for the first three inputs and the residual.
        x = _ensure_contiguous(x)
        if residual is not None:
            residual = _ensure_contiguous(residual)
        need_grad = any(ctx.needs_input_grad[:3])
        out, residual_out, rstd = rmsnorm_fwd(
            x,
            weight,
            bias=bias,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            eps=eps,
            store_rstd=need_grad,
        )
        ctx.save_for_backward(x if residual is None else residual_out, weight, rstd)
        ctx.has_bias = bias is not None
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        if residual_out is None or not prenorm:
            return out
        else:
            return out, residual_out

    @staticmethod
    def backward(ctx, dout, *args):
        # [FUNCTION DOC] Backward. Reads the saved tensors, makes dout contiguous, uses the
        # extra incoming gradient as the residual-output gradient only when prenorm and a
        # residual were used, and runs the untuned backward. Returns gradients for the
        # inputs: dx, weight, bias and residual, then four None entries for the
        # non-differentiable arguments.
        # KNOWN ISSUE: When prenorm is set without a residual, the second output is the input
        # x itself but its incoming gradient is ignored, so that path's gradient is dropped.
        x, weight, rstd = ctx.saved_tensors
        dout = _ensure_contiguous(dout)
        if ctx.prenorm and ctx.has_residual:
            dresidual_out = _ensure_contiguous(args[0])
        else:
            dresidual_out = None
        dx, dw, db, dresidual = rmsnorm_bwd(
            x,
            weight,
            dout,
            rstd,
            dresidual_out,
            ctx.has_bias,
            has_residual=ctx.has_residual,
        )
        return dx, dw, db, dresidual, *([None] * 4)


def rmsnorm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
) -> Tensor:
    # [FUNCTION DOC] Public functional RMSNorm with autograd. Per-head mode is detected when
    # the weight or bias is two-dimensional, in which case the last two dimensions of x are
    # kept as the trailing shape; otherwise only the last one. Flattens the leading dimensions
    # of x and the residual into one row dimension, applies the autograd function, and
    # reshapes the result (or each element of a tuple result) back to the shape of x. Returns
    # the output tensor, or a tuple of output and residual output when prenorm is set.
    # KNOWN ISSUE: The docstring omits bias, residual, out_dtype, residual_dtype and prenorm,
    # and the tuple return in prenorm mode.
    """RMSNorm with automatic differentiation support.

    Args:
        x: Input tensor of shape (M, N) or (B, S, H, D) for per-head mode
        weight: Optional weight tensor of shape (N,) or (H, D) for per-head mode
        eps: Small value for numerical stability

    Returns:
        Normalized output tensor of same shape as x
    """
    x_shape_og = x.shape
    per_head = (weight is not None and weight.dim() == 2) or (bias is not None and bias.dim() == 2)
    last_shape = x_shape_og[-1:] if not per_head else x_shape_og[-2:]
    # Flatten batch dims before entering autograd.Function so tensor ranks
    # are determined by per_head (which dynamo guards on via the if-branch),
    # not by the original input shape. This ensures torch.compile can
    # recompile the backward subgraph correctly when switching between
    # per_head=False and per_head=True.
    x_flat = x.reshape(-1, *last_shape)
    res_flat = residual.reshape(-1, *last_shape) if residual is not None else None
    result = RMSNormFunction.apply(
        x_flat, weight, bias, res_flat, out_dtype, residual_dtype, eps, prenorm
    )
    if isinstance(result, tuple):
        return tuple(r.reshape(x_shape_og) for r in result)
    return result.reshape(x_shape_og)


class QuackRMSNorm(torch.nn.RMSNorm):
    # [CLASS DOC] Module subclass of the standard RMSNorm module that keeps its parameters and
    # constructor semantics but computes the forward pass with the fused GPU kernel through
    # the functional wrapper. The state is the base class weight and eps.
    """RMSNorm module that behaves like torch.nn.RMSNorm.

    This class provides a drop-in replacement for torch.nn.RMSNorm that uses
    the torch._vendor.quack.rmsnorm implementation under the hood.

    Args:
        dim (int): The dimension to normalize over
        eps (float, optional): A small constant for numerical stability. Default: 1e-6

    Attributes:
        weight (torch.nn.Parameter): The learnable weight parameter
        eps (float): A small constant for numerical stability
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True, device=None, dtype=None
    ):
        # [FUNCTION DOC] Forward the normalized dimension, epsilon (default 1e-6), affine
        # flag, device and dtype to the base module constructor.
        super().__init__(dim, eps, elementwise_affine, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply the fused RMSNorm kernel to the input using this module's
        # weight (which may be None when not affine) and epsilon. Returns the normalised
        # tensor of the same shape.
        """Apply RMSNorm to the input tensor.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalized tensor
        """
        return rmsnorm(x, self.weight, eps=self.eps)


def layernorm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
):
    # [FUNCTION DOC] LayerNorm forward through the shared kernel in LayerNorm mode. Asserts
    # that x is 2D with a supported dtype, weight is 1D FP32 and the optional bias is 1D FP32.
    # Allocates the output like x and optional FP32 per-row reciprocal std and mean buffers,
    # runs the kernel, and returns the output, followed by reciprocal std and/or mean in that
    # order when requested.
    """LayerNorm forward pass using the unified RMSNorm/LayerNorm kernel.

    Args:
        x: Input tensor of shape (M, N)
        weight: Weight tensor of shape (N,). Must be float32.
        bias: Optional bias tensor of shape (N,). Must be float32.
        eps: Small value for numerical stability
        return_rstd: Whether to return the reciprocal standard deviation
        return_mean: Whether to return the mean

    Returns:
        Normalized output tensor of same shape as x
        If return_rstd is True, also returns rstd tensor of shape (M,)
        If return_mean is True, also returns mean tensor of shape (M,)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight.dim() == 1, "Weight must be 1D"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert weight.dtype == torch.float32, "Weight must be float32"
    if bias is not None:
        assert bias.dim() == 1, "Bias must be 1D"
        assert bias.dtype == torch.float32, "Bias must be float32"

    M, N = x.shape
    device = x.device
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32) if return_rstd else None
    mean = torch.empty(M, device=device, dtype=torch.float32) if return_mean else None

    _rmsnorm_fwd(x, weight, out, bias, rstd, mean, None, None, eps, True)

    if return_rstd and return_mean:
        return out, rstd, mean
    elif return_rstd:
        return out, rstd
    elif return_mean:
        return out, mean
    return out


def layernorm_ref(x: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    # [FUNCTION DOC] Reference LayerNorm: converts x to FP32 and applies the standard layer
    # norm with the weight, no bias, over the shape of the weight; returns the result cast to
    # the dtype of x.
    """Reference implementation for LayerNorm."""
    x_f32 = x.float()
    return torch.nn.functional.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)


def layernorm_rstd_ref(x: torch.Tensor, eps: float = 1e-6):
    # [FUNCTION DOC] Reference reciprocal standard deviation for LayerNorm: in FP32 takes the
    # row mean, the biased variance over the last dimension, and returns the inverse square
    # root of the variance plus epsilon, with the last dimension reduced away.
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1)
    return 1.0 / torch.sqrt(var + eps)


def layernorm_mean_ref(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Reference mean for LayerNorm: the FP32 mean of x over its last dimension,
    # with that dimension reduced away.
    return x.float().mean(dim=-1)
