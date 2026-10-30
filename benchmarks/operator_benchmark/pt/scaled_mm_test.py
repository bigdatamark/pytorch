# [MODULE DOC] Operator microbenchmark for the scaled_mm functional op. Supports FP8
# tensorwise, rowwise and blockwise (DeepSeek style) scaling, MXFP8/MXFP4 and NVFP4 inputs,
# built with the helper functions from the scaled matmul CUDA test file. At import time it
# deduplicates the base shapes, builds long configs per scaling family gated by platform
# checks, and registers benchmarks only when the platform supports the op. Running as a script
# starts the runner.
from types import ModuleType

from pt.scaled_mm_common import (
    get_float8_dtype,
    get_test_scaled_matmul_cuda,
    SCALED_MM_BASE_SHAPES,
    supports_fp8_deepseek_blockwise_scaling,
)

import operator_benchmark as op_bench

import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
    SM90OrLater,
)
from torch.torch_version import TorchVersion


"""
Operator microbenchmarks for `scaled_mm`.
Uses the same dtype + scale/quantize helpers as `test/test_scaled_matmul_cuda.py`
(bf16/fp16/fp32, fp8 e4m3/e5m2, MX e8m0 scales, NVFP4 packed fp4).
"""


def _should_generate_scaled_mm_configs() -> bool:
    # [FUNCTION DOC] HELPER: Returns true when benchmarks should be registered: the torch
    # version is at least 2.9, the functional scaled_mm exists, and the device is either a
    # non-ROCm build on SM90 or newer or a ROCm build with FP8 support.
    # Benchmarks are CUDA-centric; register nothing when the op can't run.
    # Minimum requirements:
    # - PyTorch 2.9+ (scaled_mm introduced)
    # - CUDA SM90+ (compute capability >= 9.0) OR ROCm MI300+ (gfx94x)
    return (
        TorchVersion(torch.__version__) >= "2.9"
        and hasattr(torch.nn.functional, "scaled_mm")
        and (
            (torch.version.hip is None and bool(SM90OrLater))
            or (torch.version.hip is not None and bool(PLATFORM_SUPPORTS_FP8))
        )
    )


def _supports_fp8_rowwise_fp32_output() -> bool:
    # [FUNCTION DOC] HELPER: Returns whether FP32 output is available for FP8 rowwise scaling:
    # false on ROCm, without CUDA, with CUDA older than 12.9, or when device zero has compute
    # capability below 9.0. Not referenced anywhere else in this module.
    # Mirrors test_scaled_mm_vs_emulated_row_wise gating:
    # fp32 rowwise kernels are cuBLAS-only, CUDA 12.9+, and SM90-only.
    if torch.version.hip is not None:
        return False
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if TorchVersion(torch.version.cuda) < "12.9":
        return False
    return torch.cuda.get_device_capability(0) >= (9, 0)


def _supports_scaled_mm_benchmark() -> tuple[bool, str]:
    # [FUNCTION DOC] HELPER: Reports whether scaled_mm can run. Returns a pair of a flag and a
    # reason message (empty on success). Fails with a message if the functional op is missing,
    # CUDA is unavailable, a ROCm device is not gfx94x, or the CUDA capability is neither at
    # least 9.0 nor 8.9. Not referenced anywhere else in this module.
    # `scaled_mm` was introduced in PyTorch 2.9.
    if not hasattr(torch.nn.functional, "scaled_mm"):
        return False, "torch.nn.functional.scaled_mm requires PyTorch 2.9+"

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    # Mirror torch._scaled_mm support message:
    # "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+"
    if torch.version.hip is not None:
        arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx94" in arch:
            return True, ""
        return False, f"unsupported ROCm arch {arch} (requires MI300+ / gfx94x)"

    cap = torch.cuda.get_device_capability(0)
    if cap >= (9, 0) or cap == (8, 9):
        return True, ""
    return (
        False,
        f"unsupported CUDA compute capability {cap[0]}.{cap[1]} (requires >= 9.0 or 8.9)",
    )


class ScaledMMBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of scaled_mm for one scaling recipe selected by the scaling config.
    # Setup methods create low precision operands and scales and record the scale recipes and
    # swizzle settings used by forward. Class constants give block sizes for MX (32), NVFP4
    # (16) and FP8 blockwise (128) scaling. Instance state includes base_dtype, output_dtype,
    # float8_dtype and the recipe and swizzle fields.
    _MX_BLOCK_SIZE: int = 32
    _NVFP4_BLOCK_SIZE: int = 16
    _FP8_BLOCK_K: int = 128

    def _set_output_dtype(self, output_dtype: str) -> None:
        # [FUNCTION DOC] HELPER: Sets the output dtype from its name: float32 for the float32
        # name and bfloat16 for bfloat16 or any other value.
        if output_dtype == "bfloat16":
            self.output_dtype = torch.bfloat16
        elif output_dtype == "float32":
            self.output_dtype = torch.float32
        else:
            self.output_dtype = torch.bfloat16  # default

    def _set_scaled_mm_call_config(
        self,
        *,
        scale_recipe_a: ScalingType | list[ScalingType],
        scale_recipe_b: ScalingType | list[ScalingType],
        swizzle_a: SwizzleType | list[SwizzleType] | None = None,
        swizzle_b: SwizzleType | list[SwizzleType] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Records keyword-only scale recipes and optional swizzle
        # settings for both operands (single values or lists) in instance fields, so forward
        # only has to read them.
        # Store call-time config so forward() is a single straight-line call.
        self._scale_recipe_a = scale_recipe_a
        self._scale_recipe_b = scale_recipe_b
        self._swizzle_a = swizzle_a
        self._swizzle_b = swizzle_b

    def _init_fp8_tensorwise(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # [FUNCTION DOC] HELPER: Sets up FP8 tensorwise scaling. Args: M, N, K, device, helpers
        # (module of test helpers). Picks the FP8 dtype, creates random base-dtype matrices x
        # (M x K) and y (N x K, transposed to K x N) whose gradient flags come from the
        # framework auto_set, computes detached per-tensor scales, and quantizes both under
        # no-grad to saturated FP8 leaf tensors. self.inputs holds the FP8 tensors and
        # reciprocal scales; the recipe is tensorwise on both sides.
        self.float8_dtype = get_float8_dtype(self._float8_dtype_arg)

        # Base tensors carry grad in backward benches; fp8 tensors are created as leaves.
        x_base = torch.randn(
            M,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        )
        y_base = torch.randn(
            N,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        ).t()

        # Tensorwise scales; detach so the backward bench doesn't include scale computation.
        x_scale = helpers.tensor_to_scale(x_base, self.float8_dtype).float().detach()
        y_scale = helpers.tensor_to_scale(y_base, self.float8_dtype).float().detach()

        # Quantize with the same saturation logic as the reference tests.
        with torch.no_grad():
            x_lp = (
                helpers.to_fp8_saturated(x_base * x_scale, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )
            y_lp = (
                helpers.to_fp8_saturated(y_base * y_scale, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )

        self.inputs = {
            "x": x_lp,
            "y": y_lp,
            "scale_a": x_scale.reciprocal(),
            "scale_b": y_scale.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.TensorWise,
            scale_recipe_b=ScalingType.TensorWise,
        )

    def _init_fp8_rowwise(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # [FUNCTION DOC] HELPER: Sets up FP8 rowwise scaling. Same structure as the tensorwise
        # setup, but the scales are per row of x (M x 1) and per column of the transposed y (1
        # x N). Inputs hold the FP8 tensors and reciprocal scales; the recipe is rowwise on
        # both sides.
        # Row-wise scaling (per-row A scales and per-column B scales).
        # Mirrors `test_scaled_mm_vs_emulated_row_wise` in test_scaled_matmul_cuda.py.
        self.float8_dtype = get_float8_dtype(self._float8_dtype_arg)

        x_base = torch.randn(
            M,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        )
        # Start from (N, K) and transpose so mat_b is (K, N), matching scaled_mm signature.
        y_base = torch.randn(
            N,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        ).t()

        x_scales = (
            helpers.tensor_to_scale(x_base, self.float8_dtype, dim=1).float().detach()
        )  # (M, 1)
        y_scales = (
            helpers.tensor_to_scale(y_base, self.float8_dtype, dim=0).float().detach()
        )  # (1, N)

        with torch.no_grad():
            x_lp = (
                helpers.to_fp8_saturated(x_base * x_scales, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )
            y_lp = (
                helpers.to_fp8_saturated(y_base * y_scales, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )

        self.inputs = {
            "x": x_lp,
            "y": y_lp,
            "scale_a": x_scales.reciprocal(),
            "scale_b": y_scales.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.RowWise,
            scale_recipe_b=ScalingType.RowWise,
        )

    def _init_fp8_blockwise_common(
        self,
        M: int,
        N: int,
        K: int,
        device: str,
        helpers: ModuleType,
        block_m_a: int,
        block_m_b: int,
        block_k: int,
        scaling_type_a: ScalingType,
        scaling_type_b: ScalingType,
        use_padding_a: bool,
        use_padding_b: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Shared setup for FP8 blockwise scaling. Args: M, N, K, device,
        # helpers, block_m_a and block_m_b (outer block size per side: 1 or 128), block_k, the
        # scaling types per side and flags saying whether to pad the scales per side. Raises
        # RuntimeError if a CUDA device is not SM90, if K is not divisible by block_k, or if a
        # 128 sided operand has its outer dimension not divisible by block_k. Quantizes random
        # matrices with block scales; padded sides get padded then transposed scales, other
        # sides get an outer-dimension-major layout. Stores the tensors (y transposed) and
        # reciprocal scales in self.inputs and records the recipes.
        """
        Common initialization for FP8 blockwise scaling.

        Args:
            block_m_a/block_m_b: Per-side block size for M/N (1 for 1x128, 128 for 128x128).
                The kernel only supports the joint recipes (1, 1), (128, 1), (1, 128);
                the symmetric (128, 128) is not a valid kernel mode (see get_joint_scaling
                in aten/src/ATen/native/cuda/ScaledBlas.cpp).
            block_k: Block size for K dimension (always 128)
            scaling_type_a/scaling_type_b: ScalingType enum value per side
            use_padding_a/use_padding_b: If True, pad scales for 128x128; if False, use simple transpose for 1x128
        """
        self.float8_dtype = get_float8_dtype(self._float8_dtype_arg)

        # Validate SM90 support
        if device == "cuda" and torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError(
                "FP8 BlockWise (DeepSeek style) scaling is only supported on CUDA SM90 (H100)."
            )

        # Validate dimension divisibility: K is always required; the 128x128 side
        # additionally requires its outer dim (M for lhs, N for rhs) divisible by block_k.
        if K % block_k != 0:
            raise RuntimeError(
                f"FP8 BlockWise requires K divisible by {block_k}, got K={K}"
            )
        if block_m_a == block_k and M % block_k != 0:
            raise RuntimeError(
                f"FP8 BlockWise128x128 lhs requires M divisible by {block_k}, got M={M}"
            )
        if block_m_b == block_k and N % block_k != 0:
            raise RuntimeError(
                f"FP8 BlockWise128x128 rhs requires N divisible by {block_k}, got N={N}"
            )

        # Create high-precision input tensors
        x_hp = torch.randn(
            M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()
        )
        y_hp = torch.randn(
            N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()
        )

        # Quantize to FP8 with block-wise scaling
        with torch.no_grad():
            x_lp, x_scales = helpers.tensor_to_scale_block(
                x_hp, self.float8_dtype, block_m_a, block_k
            )
            y_lp, y_scales = helpers.tensor_to_scale_block(
                y_hp, self.float8_dtype, block_m_b, block_k
            )
            x_lp = x_lp.detach().requires_grad_(self.auto_set())
            y_lp = y_lp.detach().requires_grad_(self.auto_set())

        # Process scales per side based on each side's block configuration
        if use_padding_a:
            # 128x128: pad scales to multiple of 4, then transpose
            x_scales, _ = helpers._pad_128x128_scales(x_scales.detach())
            x_scales = x_scales.t()
        else:
            # 1x128: simple transpose to get "outer-dim-major" layout
            x_scales = x_scales.t().contiguous().t().detach()
        if use_padding_b:
            y_scales, _ = helpers._pad_128x128_scales(y_scales.detach())
            y_scales = y_scales.t()
        else:
            y_scales = y_scales.t().contiguous().t().detach()

        self.inputs = {
            "x": x_lp,
            "y": y_lp.t(),  # mat_b is (K, N)
            "scale_a": x_scales.reciprocal(),
            "scale_b": y_scales.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=scaling_type_a,
            scale_recipe_b=scaling_type_b,
        )

    def _init_fp8_blockwise_1x128(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # [FUNCTION DOC] HELPER: FP8 blockwise setup with 1x128 blocks on both operands, no
        # scale padding.
        # FP8 blockwise scaling with 1x128 blocks on both sides.
        self._init_fp8_blockwise_common(
            M,
            N,
            K,
            device,
            helpers,
            block_m_a=1,
            block_m_b=1,
            block_k=self._FP8_BLOCK_K,
            scaling_type_a=ScalingType.BlockWise1x128,
            scaling_type_b=ScalingType.BlockWise1x128,
            use_padding_a=False,
            use_padding_b=False,
        )

    def _init_fp8_blockwise_128x128_1x128(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # [FUNCTION DOC] HELPER: FP8 blockwise setup with 128x128 blocks on the left operand
        # and 1x128 on the right; only the left scales are padded.
        # FP8 blockwise scaling: 128x128 lhs, 1x128 rhs.
        self._init_fp8_blockwise_common(
            M,
            N,
            K,
            device,
            helpers,
            block_m_a=self._FP8_BLOCK_K,
            block_m_b=1,
            block_k=self._FP8_BLOCK_K,
            scaling_type_a=ScalingType.BlockWise128x128,
            scaling_type_b=ScalingType.BlockWise1x128,
            use_padding_a=True,
            use_padding_b=False,
        )

    def _init_fp8_blockwise_1x128_128x128(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # [FUNCTION DOC] HELPER: FP8 blockwise setup with 1x128 blocks on the left operand and
        # 128x128 on the right; only the right scales are padded.
        # FP8 blockwise scaling: 1x128 lhs, 128x128 rhs.
        self._init_fp8_blockwise_common(
            M,
            N,
            K,
            device,
            helpers,
            block_m_a=1,
            block_m_b=self._FP8_BLOCK_K,
            block_k=self._FP8_BLOCK_K,
            scaling_type_a=ScalingType.BlockWise1x128,
            scaling_type_b=ScalingType.BlockWise128x128,
            use_padding_a=False,
            use_padding_b=True,
        )

    def _init_mx_blockwise(
        self, M: int, N: int, K: int, device: str, *, mx_format: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Sets up MX (mxfp8 or mxfp4) scaling; mx_format is
        # keyword-only. Raises RuntimeError for a non-cuda device. Creates random matrices,
        # converts them with the test reference to MX data and e8m0 scales using block size 32,
        # and rearranges the scales to the blocked layout. The right operand is passed as a
        # transposed view so it is column-major. Records 1x32 blockwise recipes with no swizzle
        # on ROCm and the 32-4-4 swizzle on CUDA.
        # MX uses BlockWise1x32 with swizzled scales on CUDA, NO_SWIZZLE on HIP.
        if device != "cuda":
            raise RuntimeError(f"MX scaling requires CUDA device, got: {device}")

        # Important cuBLASLt requirement: mat_b must be column-major.
        # We satisfy this by passing a transpose view (non-contiguous) for `mat_b`.
        #
        # NOTE: we intentionally import from torch.testing._internal to reuse the exact
        # reference implementation used by test/test_scaled_matmul_cuda.py.
        from torch.testing._internal.common_quantized import to_blocked, to_mxfp

        x_hp = torch.randn(
            M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()
        )
        y_hp = torch.randn(
            N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set()
        )

        scale_a, x_lp = to_mxfp(
            x_hp.contiguous(), block_size=self._MX_BLOCK_SIZE, format=mx_format
        )
        scale_b, y_lp = to_mxfp(
            y_hp.contiguous(), block_size=self._MX_BLOCK_SIZE, format=mx_format
        )

        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)

        # HIP requires NO_SWIZZLE, CUDA uses SWIZZLE_32_4_4
        swizzle_type = (
            SwizzleType.NO_SWIZZLE
            if torch.version.hip is not None
            else SwizzleType.SWIZZLE_32_4_4
        )

        self.inputs = {
            "x": x_lp,
            "y": y_lp.t(),  # column-major mat_b
            "scale_a": scale_a,
            "scale_b": scale_b,
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=swizzle_type,
            swizzle_b=swizzle_type,
        )

    def _init_nvfp4_blockwise_and_tensorwise(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # [FUNCTION DOC] HELPER: Sets up NVFP4 scaling with a blockwise decode scale plus a
        # global tensorwise scale per operand. Raises RuntimeError for a non-cuda device, ROCm,
        # or K not divisible by 32. Creates large-magnitude random matrices, converts them to
        # packed fp4 with global scale using block size 16, and rearranges block scales.
        # Scales, recipes and swizzles are lists of two entries (blockwise with swizzle,
        # tensorwise without).
        # NVFP4 uses packed fp4 inputs and two-level scaling:
        # - blockwise (1x16) decode scales (swizzled for CUDA)
        # - tensorwise (global) decode scale (NO_SWIZZLE)
        #
        # scaled_mm expects these as LISTS for both `scale_*` and `scale_recipe_*`.
        if device != "cuda":
            raise RuntimeError(f"NVFP4 scaling requires CUDA device, got: {device}")
        if torch.version.hip is not None:
            raise RuntimeError("NVFP4 benchmarks are only wired for CUDA (non-HIP).")
        if K % 32 != 0:
            raise RuntimeError(f"NVFP4 requires K divisible by 32, got K={K}")

        # NOTE: We reuse the same reference implementation as test/test_scaled_matmul_cuda.py.
        from torch.testing._internal.common_quantized import to_blocked

        # Use nontrivial distribution so scaling isn't degenerate.
        a_ref = torch.randn((M, K), device=device, dtype=self.base_dtype) * 1000
        b_ref = torch.randn((N, K), device=device, dtype=self.base_dtype) * 1000

        a_lp, a_scale, a_global_scale = helpers.data_to_nvfp4_with_global_scale(
            a_ref, self._NVFP4_BLOCK_SIZE
        )
        b_lp, b_scale, b_global_scale = helpers.data_to_nvfp4_with_global_scale(
            b_ref, self._NVFP4_BLOCK_SIZE
        )

        a_scale = to_blocked(a_scale)
        b_scale = to_blocked(b_scale)

        self.inputs = {
            "x": a_lp,
            "y": b_lp.t(),
            "scale_a": [a_scale, a_global_scale],
            "scale_b": [b_scale, b_global_scale],
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
            swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        )

    def init(
        self,
        M,
        N,
        K,
        device,
        float8_dtype="e4m3fn",
        output_dtype="bfloat16",
        scaling="fp8_tensorwise",
    ):
        # [FUNCTION DOC] Args: M, N, K, device, float8_dtype, output_dtype, scaling. Loads the
        # shared test helper module, stores the settings, sets the output dtype and calls the
        # setup method for the requested scaling: fp8 tensorwise, rowwise or one of three
        # blockwise variants, mxfp8, mxfp4 or nvfp4. Raises ValueError for an unknown scaling
        # name. Sets the module name.
        helpers = get_test_scaled_matmul_cuda()
        self._float8_dtype_arg = float8_dtype
        self.base_dtype = torch.bfloat16
        self.scaling = scaling
        self.float8_dtype = None

        self._set_output_dtype(output_dtype)

        if scaling == "fp8_tensorwise":
            self._init_fp8_tensorwise(M, N, K, device, helpers)
        elif scaling == "fp8_rowwise":
            self._init_fp8_rowwise(M, N, K, device, helpers)
        elif scaling == "fp8_blockwise_1x128":
            self._init_fp8_blockwise_1x128(M, N, K, device, helpers)
        elif scaling == "fp8_blockwise_128x128_1x128":
            self._init_fp8_blockwise_128x128_1x128(M, N, K, device, helpers)
        elif scaling == "fp8_blockwise_1x128_128x128":
            self._init_fp8_blockwise_1x128_128x128(M, N, K, device, helpers)
        elif scaling == "mxfp8":
            self._init_mx_blockwise(M, N, K, device, mx_format="mxfp8")
        elif scaling == "mxfp4":
            self._init_mx_blockwise(M, N, K, device, mx_format="mxfp4")
        elif scaling == "nvfp4":
            self._init_nvfp4_blockwise_and_tensorwise(M, N, K, device, helpers)
        else:
            raise ValueError(f"Unsupported scaling mode: {scaling}")

        self.set_module_name("scaled_mm")

    def forward(self, x, y, scale_a, scale_b):
        # [FUNCTION DOC] Args: x, y, scale_a, scale_b. Calls scaled_mm with the scales, stored
        # recipes and output dtype, adding swizzle arguments only when they were recorded.
        # Returns the result.
        kwargs = {
            "scale_a": scale_a,
            "scale_recipe_a": self._scale_recipe_a,
            "scale_b": scale_b,
            "scale_recipe_b": self._scale_recipe_b,
            "output_dtype": self.output_dtype,
        }
        if self._swizzle_a is not None:
            kwargs["swizzle_a"] = self._swizzle_a
        if self._swizzle_b is not None:
            kwargs["swizzle_b"] = self._swizzle_b

        return torch.nn.functional.scaled_mm(x, y, **kwargs)


# Use shared base shapes from scaled_mm_common
_scaled_mm_long_shapes = []
_seen = set()
for m, n, k in SCALED_MM_BASE_SHAPES:
    shape = (m, n, k)
    if shape in _seen:
        continue
    _seen.add(shape)
    _scaled_mm_long_shapes.append([m, n, k])

# Build long configs in groups so we can gate unsupported (scaling, output_dtype)
# combinations based on the running platform.
scaled_mm_configs_long = []

if _should_generate_scaled_mm_configs():
    # FP8 tensorwise supports bf16 and fp32 output.
    scaled_mm_configs_long += op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=_scaled_mm_long_shapes,
        cross_product_configs={
            "device": ["cuda"],
            "float8_dtype": ["e4m3fn"],
            "output_dtype": ["bfloat16"],
            "scaling": ["fp8_tensorwise"],
        },
        tags=["long"],
    )

    # FP8 rowwise requires CUDA 12.9+ on CUDA builds (see `aten/src/ATen/cuda/CUDABlas.cpp:get_scale_mode`).
    if torch.version.hip is None and TorchVersion(torch.version.cuda) < "12.9":
        pass
    else:
        # Keep bf16-only for now.
        rowwise_output_dtypes = ["bfloat16"]
        scaled_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K"],
            attrs=_scaled_mm_long_shapes,
            cross_product_configs={
                "device": ["cuda"],
                "float8_dtype": ["e4m3fn"],
                "output_dtype": rowwise_output_dtypes,
                "scaling": ["fp8_rowwise"],
            },
            tags=["long"],
        )

    # MX supports both CUDA (with swizzle) and HIP (with NO_SWIZZLE).
    # NVFP4 is CUDA-only (non-HIP) due to swizzled scale requirements.
    if PLATFORM_SUPPORTS_MX_GEMM:
        scaled_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K"],
            attrs=_scaled_mm_long_shapes,
            cross_product_configs={
                "device": ["cuda"],
                "float8_dtype": ["e4m3fn"],
                "output_dtype": ["bfloat16"],
                "scaling": ["mxfp8", "mxfp4"],
            },
            tags=["long"],
        )

    # NVFP4 is CUDA-only (non-HIP)
    if torch.version.hip is None and PLATFORM_SUPPORTS_MX_GEMM:
        scaled_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K"],
            attrs=_scaled_mm_long_shapes,
            cross_product_configs={
                "device": ["cuda"],
                "float8_dtype": ["e4m3fn"],
                "output_dtype": ["bfloat16"],
                "scaling": ["nvfp4"],
            },
            tags=["long"],
        )

    # DeepSeek FP8 blockwise scaling, SM90-only. Bench all three joint recipes
    # the kernel supports (see get_joint_scaling in
    # aten/src/ATen/native/cuda/ScaledBlas.cpp): (1x128, 1x128), (128x128, 1x128),
    # and (1x128, 128x128). The symmetric (128x128, 128x128) pair is not a
    # supported kernel mode, so it is not benched.
    if supports_fp8_deepseek_blockwise_scaling():
        scaled_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K"],
            attrs=_scaled_mm_long_shapes,
            cross_product_configs={
                "device": ["cuda"],
                "float8_dtype": ["e4m3fn"],
                "output_dtype": ["bfloat16"],
                "scaling": [
                    "fp8_blockwise_1x128",
                    "fp8_blockwise_128x128_1x128",
                    "fp8_blockwise_1x128_128x128",
                ],
            },
            tags=["long"],
        )

# Generate tests for scaled_mm (register nothing on unsupported platforms).
if _should_generate_scaled_mm_configs():
    _scaled_mm_configs = scaled_mm_configs_long
else:
    _scaled_mm_configs = []

op_bench.generate_pt_test(_scaled_mm_configs, ScaledMMBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
