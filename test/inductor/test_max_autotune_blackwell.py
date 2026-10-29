# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for max-autotune matrix multiply templates on Nvidia Blackwell GPUs using
# device-side TMA (tensor memory accelerator) with persistent warp-specialized kernels. Classes
# cover: regular mm, scaled mm (per tensor and per row) and addmm persistent TMA templates; TMA
# store with fused pointwise epilogues; TMA load for epilogue inputs; and checks of the
# exhaustive and scaled config lists from the Blackwell config heuristics. Most tests need
# Blackwell hardware and are skipped otherwise. Module setup raises float32 matmul precision to
# high and teardown restores the prior value. The main block only runs with a GPU, a CPU backend
# and a big GPU.
import unittest

import torch
from torch._inductor import config
from torch._inductor.heuristics.registry import _HEURISTIC_CACHE
from torch._inductor.heuristics.template.triton import (
    BlackwellGPUGemmConfig,
    CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic,
    CUDABlackwellPersistentTMATemplateConfigHeuristic,
    CUDAScaledBlackwellTMATemplateConfigHeuristic,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._triton import has_datacenter_blackwell_tma_device


def has_tlx() -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the Triton Language Extensions module can be imported,
    # false on import error.
    """Check if TLX (Triton Language eXtensions) is available."""
    try:
        import triton.language.extra.tlx  # noqa: F401

        return True
    except ImportError:
        return False


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] Module fixture: saves the current float32 matmul precision in a module-level
    # variable and sets it to high for all tests in this module.
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")


def tearDownModule():
    # [FUNCTION DOC] Module fixture: restores the saved float32 matmul precision if one was saved,
    # then clears the saved value.
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


@instantiate_parametrized_tests
class TestMaxAutotuneBlackwell(TestCase):
    # [CLASS DOC] Parametrized end-to-end tests that compile mm, scaled mm and addmm with max
    # autotune restricted by config to the Blackwell warp-specialized persistent device-TMA
    # template, then compare against eager results and check the generated code for TMA
    # descriptor use.
    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("epilogue_subtile", (1, 2, 4))
    def test_blackwell_max_autotune_regular_mm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Compiles a matrix multiply of float16 matrices on the GPU, optionally
        # dynamic, with each input optionally transposed and with TMA store and epilogue subtile
        # count chosen by the parameters. Forces the persistent TMA template via config,
        # compares against eager within small tolerance, and checks generated code creates
        # tensor descriptors, loads through them, and stores through the descriptor or a plain
        # store depending on TMA store. Skipped without Blackwell TMA support.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Repeats both inputs eight times along both dims (for the
            # 16-byte alignment TMA needs), transposes either per the enclosing parameters, and
            # returns their matrix product.
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)
            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        M, N, K = 32, 16, 48
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )

        epilogue_subtile_regex = f"EPILOGUE_SUBTILE={epilogue_subtile}"
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                "test_configs.autotune_choice_desc_regex": epilogue_subtile_regex,
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        FileCheck().check("triton_tem_fused_mm").check(
            "triton.language.make_tensor_descriptor"
        ).check("tl.load_tensor_descriptor").check(write_api).run(code[0])

    # NOTE: the current Inductor template verifies that the scaling mode is either per-tensor or per-row
    # TODO: support additional scaling modes for Blackwell
    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_blackwell_max_autotune_scaled_mm_per_tensor_persistent_tma(
        self,
        dynamic: bool,
        tma_store: bool,
    ):
        # [FUNCTION DOC] Compiles a scaled matrix multiply with float8 operands and one scale
        # per tensor (computed from the maximum absolute value), float16 output, persistent TMA
        # template forced, dynamic or not and TMA store or not. Compares with eager (loose
        # relative tolerance) and checks generated code uses tensor descriptors. Skipped without
        # Blackwell TMA support.
        def scaled_mm(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Calls the scaled matmul on a and the transpose of b with
            # the given scales, with fast accumulation, producing float16.
            # NOTE: Inductor constrains a to be row_major and b to be col_major
            return torch._scaled_mm(
                a, b.t(), scale_a, scale_b, use_fast_accum=True, out_dtype=torch.float16
            )

        def get_scale_per_tensor(t):
            # [FUNCTION DOC] HELPER: Returns a float32 scale equal to the largest float8 e4m3
            # value divided by the largest absolute element of the tensor.
            scale = torch.finfo(torch.float8_e4m3fn).max / t.abs().max()
            return scale.to(torch.float32)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte.
        M, N, K = 32, 16, 48
        a = (torch.randn((M, K)).to(torch.float16).to(GPU_TYPE)).repeat(8, 8)
        b = (torch.randn((N, K)).to(torch.float16).to(GPU_TYPE)).repeat(8, 8)

        scale_a = get_scale_per_tensor(a)
        scale_b = get_scale_per_tensor(b)

        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(scaled_mm, dynamic=dynamic), a, b, scale_a, scale_b
            )
            c_expected = scaled_mm(a, b, scale_a, scale_b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=0.5)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        FileCheck().check("triton_tem_fused__scaled_mm").check(
            "triton.language.make_tensor_descriptor"
        ).check("tl.load_tensor_descriptor").check(write_api).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_blackwell_max_autotune_scaled_mm_per_row_persistent_tma(
        self,
        dynamic: bool,
        tma_store: bool,
    ):
        # [FUNCTION DOC] Same as the per-tensor scaled test but with a scale per row for each
        # operand (the second transposed at use), bfloat16 inputs and bfloat16 output; checks
        # results and generated TMA descriptor code. Skipped without Blackwell TMA support.
        def scaled_mm(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Calls the scaled matmul on a and the transpose of b with
            # the first scale and the transposed second scale, with fast accumulation, producing
            # bfloat16.
            # NOTE: Inductor constrains a to be row_major and b to be col_majo
            return torch._scaled_mm(
                a,
                b.t(),
                scale_a,
                scale_b.t(),
                use_fast_accum=True,
                out_dtype=torch.bfloat16,
            )

        def get_scale_per_row(t):
            # [FUNCTION DOC] HELPER: Returns a float32 per-row scale equal to the largest float8
            # e4m3 value divided by each row's largest absolute element, keeping the row
            # dimension.
            scale = (
                torch.finfo(torch.float8_e4m3fn).max
                / t.abs().max(dim=1, keepdim=True).values
            )
            return scale.to(torch.float32)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte.
        M, N, K = 32, 16, 48
        a = (torch.randn((M, K)).to(torch.bfloat16).to(GPU_TYPE)).repeat(8, 8)
        b = (torch.randn((N, K)).to(torch.bfloat16).to(GPU_TYPE)).repeat(8, 8)

        scale_a = get_scale_per_row(a)
        scale_b = get_scale_per_row(b)

        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(scaled_mm, dynamic=dynamic), a, b, scale_a, scale_b
            )
            c_expected = scaled_mm(a, b, scale_a, scale_b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=0.5)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        FileCheck().check("triton_tem_fused__scaled_mm").check(
            "triton.language.make_tensor_descriptor"
        ).check("tl.load_tensor_descriptor").check(write_api).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("epilogue_subtile", (1, 2, 4))
    def test_blackwell_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Compiles addmm with a one-dimensional bias in float16 using odd matrix
        # sizes, forced persistent TMA template, parametrized transposition, dynamic shapes, TMA
        # store and epilogue subtile. Also sets a config switch for dynamic pipelining disable
        # unless running in the internal build, as a workaround for a misaligned access issue
        # noted in the config comment. Checks generated code for TMA descriptor creation, loads
        # and the right store form, then compares with eager. Skipped without Blackwell TMA
        # support.
        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: Repeats the bias eight times and the matrices eight times
            # along both dims, applies requested transposes, and returns the addmm result.
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            x = x.repeat(8)
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        epilogue_subtile_regex = f"EPILOGUE_SUBTILE={epilogue_subtile}"
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                "test_configs.autotune_choice_desc_regex": epilogue_subtile_regex,
                # If we dynamically disable pipelining,
                # triton_blackwell_ws_persistent_device_tma template will
                # be picked and then cause mis-aligned memory access.
                # If we don't dynamically disable pipelining,
                # this template is skipped and triton_tem_fused_addmm_repeat_3
                # is used.
                #
                # Fundamentally we should fix the triton_blackwell_ws_persistent_device_tma template
                # The flag below work around the problem.
                #
                # This only happens in fbcode https://www.internalfb.com/diff/D101855575.
                # Can not repro it in OSS build.
                "triton.dynamic_disable_pipelining": not config.is_fbcode(),
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        make_desc_api = "triton.language.make_tensor_descriptor"
        read_api = "tl.load_tensor_descriptor"
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_addmm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)


@instantiate_parametrized_tests
class TestBlackwellTMAStoreFusion(TestCase):
    # [CLASS DOC] Tests that pointwise epilogues fused into a Blackwell persistent TMA matmul
    # work with TMA store; each test temporarily swaps the template heuristic config lists for
    # one test config and restores them afterwards.
    """Tests for TMA store with fused pointwise epilogues on Blackwell."""

    @staticmethod
    def _make_tma_store_test_config(epilogue_subtile: int) -> BlackwellGPUGemmConfig:
        # [FUNCTION DOC] HELPER: Static factory returning a Blackwell gemm config with 128 by
        # 128 by 64 blocks, 3 stages, 4 warps, the given epilogue subtile, warp specialization
        # off and flattening off (the warp-specialized pipeline does not support flatten with
        # descriptor store).
        """FLATTEN=False test config for TMA store pointwise epilogue fusion.

        The WS pipeline doesn't yet support FLATTEN=True + TMA descriptor store.
        """
        return BlackwellGPUGemmConfig(
            128,
            128,
            64,
            3,
            4,
            epilogue_subtile=epilogue_subtile,
            warp_specialize=False,
            flatten=False,
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_mm_sigmoid_epilogue_fusion_tma_store(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Compiles a matmul followed by a sigmoid epilogue and a multiply with
        # the input on bfloat16 data of the given shape. Replaces the matmul config lists of the
        # mm and addmm template heuristics with the single test config (restoring them in a
        # finally block), forces the template and TMA store, then checks results match eager
        # (loose tolerance) and the generated code has a fused kernel with a descriptor store.
        # Skipped without Blackwell TMA support.
        """Verify fused mm + sigmoid epilogue with TMA store (FLATTEN=False)."""

        def fn(x, W):
            # [FUNCTION DOC] HELPER: Multiplies x by the transpose of W, applies sigmoid,
            # doubles it, and multiplies by x.
            mm = torch.mm(x, W.T)
            return x * (2.0 * torch.sigmoid(mm))

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_store_test_config(epilogue_subtile)
        # Ensure the heuristic cache is populated, then patch mm_configs.
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W)
                expected = fn(x, W)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused_mm").check("tma_descriptor").check(
            ".store"
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_addmm_relu_epilogue_fusion_tma_store(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Same config-swap procedure as the sigmoid test but compiles matmul plus
        # a row bias followed by relu; checks numerical agreement with eager and that the fused
        # template kernel uses descriptors and a store. Skipped without Blackwell TMA support.
        """Verify fused addmm + relu epilogue with TMA store (FLATTEN=False)."""

        def fn(x, W, bias):
            # [FUNCTION DOC] HELPER: Computes matmul of x and the transpose of W, adds bias, and
            # applies relu.
            mm = torch.mm(x, W.T)
            return torch.relu(mm + bias)

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(N, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_store_test_config(epilogue_subtile)
        # Ensure the heuristic cache is populated, then patch mm_configs.
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W, bias)
                expected = fn(x, W, bias)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused").check("tma_descriptor").check(
            ".store"
        ).run(code[0])


@instantiate_parametrized_tests
class TestBlackwellTMALoadFusion(TestCase):
    # [CLASS DOC] Tests that pointwise epilogues using extra input tensors fuse into the
    # Blackwell persistent TMA matmul with the epilogue load going through TMA; each test swaps
    # template heuristic configs for a single test config and restores them afterwards.
    """Tests for TMA load with fused pointwise epilogues on Blackwell."""

    @staticmethod
    def _make_tma_load_test_config(epilogue_subtile: int) -> BlackwellGPUGemmConfig:
        # [FUNCTION DOC] HELPER: Static factory returning a Blackwell gemm config with 128 by
        # 128 by 64 blocks, 3 stages, 4 warps, the given epilogue subtile, warp specialization
        # off and flattening off.
        return BlackwellGPUGemmConfig(
            128,
            128,
            64,
            3,
            4,
            epilogue_subtile=epilogue_subtile,
            warp_specialize=False,
            flatten=False,
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_mm_sigmoid_epilogue_tma_load(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Compiles matmul with a sigmoid epilogue that reads the original input
        # (one epilogue load) with the TMA-load-for-epilogue option enabled and the
        # single-config swap in place; checks results against eager and that the generated code
        # has a fused kernel with descriptors and tile loads. Skipped without Blackwell TMA
        # support.
        """Verify fused mm + sigmoid epilogue with TMA load (1 epilogue load)."""

        def fn(x, W):
            # [FUNCTION DOC] HELPER: Multiplies x by the transpose of W, applies sigmoid,
            # doubles it, and multiplies by x.
            mm = torch.mm(x, W.T)
            return x * (2.0 * torch.sigmoid(mm))

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_load_test_config(epilogue_subtile=epilogue_subtile)
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_tma_load_for_template_epilogue": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W)
                expected = fn(x, W)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused_mm").check("tma_descriptor").check(
            ".load(["
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_mm_scale_bias_epilogue_tma_load(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Compiles matmul scaled by a full-size tensor and offset by another (two
        # epilogue loads) with the TMA-load option and config swap, then checks results and that
        # the fused kernel code uses descriptors and tile loads. Skipped without Blackwell TMA
        # support.
        """Verify fused mm * scale + bias epilogue with TMA load (2 epilogue loads)."""

        def fn(x, W, scale, bias):
            # [FUNCTION DOC] HELPER: Multiplies x by the transpose of W, then multiplies by
            # scale and adds bias elementwise.
            return torch.mm(x, W.T) * scale + bias

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)
        scale = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_load_test_config(epilogue_subtile=epilogue_subtile)
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_tma_load_for_template_epilogue": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W, scale, bias)
                expected = fn(x, W, scale, bias)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused").check("tma_descriptor").check(
            ".load(["
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_addmm_1d_bias_epilogue_tma_load(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        # [FUNCTION DOC] Compiles addmm with a one-dimensional bias over the transpose of W with
        # the TMA-load option and config swap; checks results against eager and that the fused
        # kernel code uses descriptors and tile loads. Skipped without Blackwell TMA support.
        """Verify fused addmm with 1D bias epilogue with TMA load."""

        def fn(bias, x, W):
            # [FUNCTION DOC] HELPER: Returns the addmm of bias, x and the transpose of W.
            return torch.addmm(bias, x, W.T)

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(N, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_load_test_config(epilogue_subtile=epilogue_subtile)
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_device_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_tma_load_for_template_epilogue": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_device_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), bias, x, W)
                expected = fn(bias, x, W)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused").check("tma_descriptor").check(
            ".load(["
        ).run(code[0])


@instantiate_parametrized_tests
class TestBlackwellExhaustiveConfigs(TestCase):
    # [CLASS DOC] Checks the config lists produced by Blackwell template heuristics. Class-level
    # sets hold the legal values per parameter (block sizes, group size, stages, warps, epilogue
    # subtile) derived from the exhaustive config generator.
    """Tests for exhaustive config generation for Blackwell templates."""

    # Expected valid values for each parameter based on _generate_exhaustive_configs
    VALID_BLOCK_SIZES = {32, 64, 128, 256}
    VALID_GROUP_M = {8}
    VALID_NUM_STAGES = {2, 3, 4, 5, 6}
    VALID_NUM_WARPS = {4, 8}
    VALID_EPILOGUE_SUBTILE = {1, 2, 4}

    @parametrize(
        "heuristic_cls",
        (
            CUDABlackwellPersistentTMATemplateConfigHeuristic,
            CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic,
        ),
    )
    def test_exhaustive_configs_parameter_variety(self, heuristic_cls):
        # [FUNCTION DOC] For each of the two persistent TMA heuristic classes, instantiates it
        # and checks its exhaustive config list has at least 128 entries, each a Blackwell gemm
        # config, and that the sets of observed values per parameter exactly equal the expected
        # valid sets.
        """Test that exhaustive configs have variety in all expected parameters."""
        heuristic = heuristic_cls()
        configs = heuristic.exhaustive_configs

        # Ensure we have at least 128 configs
        self.assertGreaterEqual(len(configs), 128)

        # Collect all unique values for each parameter
        block_m_values = set()
        block_n_values = set()
        block_k_values = set()
        group_m_values = set()
        num_stages_values = set()
        num_warps_values = set()
        epilogue_subtile_values = set()

        for cfg in configs:
            self.assertIsInstance(cfg, BlackwellGPUGemmConfig)
            block_m_values.add(cfg.block_m)
            block_n_values.add(cfg.block_n)
            block_k_values.add(cfg.block_k)
            group_m_values.add(cfg.group_m)
            num_stages_values.add(cfg.num_stages)
            num_warps_values.add(cfg.num_warps)
            epilogue_subtile_values.add(cfg.epilogue_subtile)

        # Verify multiple values exist for each parameter via set membership
        self.assertEqual(block_m_values, self.VALID_BLOCK_SIZES)
        self.assertEqual(block_n_values, self.VALID_BLOCK_SIZES)
        self.assertEqual(block_k_values, self.VALID_BLOCK_SIZES)
        self.assertEqual(group_m_values, self.VALID_GROUP_M)
        self.assertEqual(num_stages_values, self.VALID_NUM_STAGES)
        self.assertEqual(num_warps_values, self.VALID_NUM_WARPS)
        self.assertEqual(epilogue_subtile_values, self.VALID_EPILOGUE_SUBTILE)

    def test_scaled_blackwell_tma_uses_scaled_persistent_configs(self):
        # [FUNCTION DOC] Instantiates the scaled Blackwell TMA heuristic and checks every mm
        # config is a Blackwell gemm config, that the configs match the scaled persistent list
        # one-for-one on block sizes, stages and warps, and that the list is longer than the
        # small addmm persistent list.
        """Verify scaled Blackwell TMA heuristic uses blackwell_scaled_persistent_mm_configs."""
        heuristic = CUDAScaledBlackwellTMATemplateConfigHeuristic()

        for cfg in heuristic.mm_configs:
            self.assertIsInstance(cfg, BlackwellGPUGemmConfig)

        expected_configs = heuristic.blackwell_scaled_persistent_mm_configs
        self.assertEqual(len(heuristic.mm_configs), len(expected_configs))
        for actual, expected in zip(heuristic.mm_configs, expected_configs):
            self.assertEqual(actual.block_m, expected.block_m)
            self.assertEqual(actual.block_n, expected.block_n)
            self.assertEqual(actual.block_k, expected.block_k)
            self.assertEqual(actual.num_stages, expected.num_stages)
            self.assertEqual(actual.num_warps, expected.num_warps)

        addmm_configs = heuristic.blackwell_persistent_addmm_configs
        self.assertGreater(
            len(heuristic.mm_configs),
            len(addmm_configs),
            "Scaled TMA should use the larger scaled_persistent list, not the small addmm list",
        )


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
