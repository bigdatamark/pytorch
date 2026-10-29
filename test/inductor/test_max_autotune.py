# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor max-autotune GEMM/convolution autotuning on GPU (and some CPU).
# Covers Triton mm/addmm/baddbmm/bmm templates, persistent and TMA templates, decompose-K, choice
# caching, subprocess/pipelined autotuning, epilogue/prologue fusion, benchmark request classes
# and template config heuristics. Import-time side effects: when Triton on CUDA is available the
# CUDA allocator is configured to disable expandable segments; module-level setup/teardown hooks
# switch the float32 matmul precision for the whole module (highest on ROCm, high elsewhere) and
# restore it afterwards. Also defines shared helpers and test classes.
import contextlib
import functools
import inspect
import json
import logging
import math
import os
import random
import re
import tempfile
import time
import unittest
from collections.abc import Callable
from unittest import mock
from unittest.mock import patch

import sympy

import torch
import torch._inductor.async_compile
from torch import multiprocessing as mp, nn
from torch._dynamo import reset
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.testing import rand_strided, reset_rng_state
from torch._dynamo.utils import counters, same
from torch._inductor import config
from torch._inductor.autotune_process import (
    _TestBenchmarkRequest,
    _TestCodeCacheBenchmarkRequest,
    _TestEnvBenchmarkRequest,
    AsyncAutotuner,
    AutotuneProcessPool,
    ExternKernelBenchmarkRequest,
    get_visible_devices_env_var,
    run_autotune_in_subprocess,
    TritonBenchmarkRequest,
    TuningProcess,
    TuningProcessPool,
    use_pipelined_autotuning,
)
from torch._inductor.codegen.common import WorkspaceArg
from torch._inductor.graph import GraphLowering
from torch._inductor.heuristics.registry import override_template_heuristics
from torch._inductor.heuristics.template.triton import (
    BlackwellGPUGemmConfig,
    CUDAAddmmPersistentTMATemplateConfigHeuristic,
    CUDAAddMMTemplateConfigHeuristic,
    CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic,
    CUDABlackwellPersistentTMATemplateConfigHeuristic,
    CUDAMMTemplateConfigHeuristic,
    CUDAPersistentTMATemplateConfigHeuristic,
    GemmConfig,
    get_shared_memory_checker_opts,
    ROCmMMTemplateConfigHeuristic,
    XPUMMTemplateConfigHeuristic,
    XPUPersistentTMATemplateConfigHeuristic,
)
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout, FlexibleLayout
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_heuristics import CachingAutotuner, pointwise
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import (
    add_feedback_saver,
    add_preprocessing_fn,
    AlgorithmSelectorCache,
    autotune_select_algorithm,
    clear_feedback_savers,
    clear_preprocessing_fns,
    ExternKernelCaller,
    NoValidChoicesError,
    TritonTemplate,
    TritonTemplateCaller,
)
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, SM90OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    random_matrix_with_scaled_reduction_dim,
    skipIfRocm,
    skipIfTorchInductor,
    TEST_WITH_ROCM,
    TEST_XPU,
)
from torch.testing._internal.logging_utils import multiple_logs_to_string
from torch.utils._triton import (
    has_datacenter_blackwell_tma_device,
    has_triton_stable_tma_api,
    has_triton_tma_device,
)


aten = torch.ops.aten
from torch._inductor.mock_cache import global_stats, PatchCaches, Stats
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    fresh_cache,
    get_k_splits,
    run_and_get_code,
    use_decompose_k_choice,
)
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)


if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


def setUpModule():
# [FUNCTION DOC] HELPER: unittest module-level fixture. Saves the current float32 matmul precision
# in a module global, then sets it to highest on ROCm builds (TF32 accuracy issues) or high on
# other builds, so generated Triton kernels are compared against a suitable reference.
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    if torch.version.hip:
        # Temporary: TF32 accuracy issues on ROCm fail these tests; use FP32
        # as reference to ensure the generated triton kernels are adequately
        # tested. Track in: https://github.com/pytorch/pytorch/issues/169392
        torch.set_float32_matmul_precision("highest")
    else:
        torch.set_float32_matmul_precision("high")


def tearDownModule():
# [FUNCTION DOC] HELPER: unittest module-level fixture. If a precision was saved by setUpModule,
# restores that float32 matmul precision and clears the saved global; otherwise does nothing.
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


# Conditional patch for decompose_k tests - override to 10 on ROCm, no-op elsewhere
_DECOMPOSE_K_PATCH_ROCM = (
    {"triton.num_decompose_k_splits": 10} if torch.version.hip else {}
)


def benchmark_choice(choice, args, out, expected_out, timings):
# [FUNCTION DOC] HELPER: benchmarks one autotune choice into a preallocated output tensor. Runs
# the choice benchmark with the given args and out, optionally checks the output against
# expected_out with assert_close (raises AssertionError on mismatch), and copies the returned
# timing values into the timings tensor in place. Used as a subprocess/worker benchmark function.
    result = choice.benchmark(*args, out=out)
    if expected_out is not None:
        torch.testing.assert_close(out, expected_out)

    timings.copy_(torch.tensor(result))


class FailChoiceCaller(ChoiceCaller):
    # [CLASS DOC] HELPER: test double for a choice caller whose benchmark always fails. Subclasses
    # ChoiceCaller and adds no state; used to exercise error handling during autotuning.
    def benchmark(self, *args, out):
        # [FUNCTION DOC] HELPER: always raises RuntimeError, ignoring its arguments, to simulate a
        # benchmark failure for this choice.
        raise RuntimeError("This choice caller will always throw")


@unittest.mock.patch(
    "torch._inductor.select_algorithm.TritonTemplate.test_cache", new=True
)
@config.patch(enable_caching_generated_triton_templates=True)
@instantiate_parametrized_tests
class TestMaxAutotune(TestCase):
    # [CLASS DOC] Main max-autotune test suite. Patched so that the Triton template test cache is
    # enabled and generated Triton template caching is on; parametrized tests are expanded by the
    # class decorator. Tests compile small mm/addmm/baddbmm/conv/cat programs under max-autotune
    # and check numerics, generated code (via FileCheck) or autotuning behavior and counters.
    def _make_matrices(self, M, K, N, *batch_dims, dtype, device, requires_grad):
        # [FUNCTION DOC] HELPER: builds a pair of random matrices a (M by K) and b (K by N), each
        # with optional leading batch dims, using the scaled-reduction-dim generator so that sums
        # over K stay well scaled. dtype, device and requires_grad are keyword-only. Returns the
        # tuple (a, b).
        make_matrix = functools.partial(
            random_matrix_with_scaled_reduction_dim,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        a = make_matrix(M, K, *batch_dims, reduction_dim=-1)
        b = make_matrix(K, N, *batch_dims, reduction_dim=-2)
        return a, b

    @parametrize("dynamic", (False, True))
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_max_autotune_mm_plus_mm_zero_size_input(self, dynamic, search_space):
        # [FUNCTION DOC] Regression test: compiling a mm_plus_mm function with zero-row
        # inputs under max autotune must not crash. Parametrized over dynamic shapes and
        # the DEFAULT/EXHAUSTIVE GEMM search spaces; only checks that compilation and
        # execution complete.
        """
        Make sure autotuning mm_plus_mm with zero-size input works without crashes.
        """
        m, n, k = 0, 1536, 64

        def mm_plus_mm(a, b, c, d):
            # [FUNCTION DOC] HELPER: the function under test, returns the sum of two
            # matrix products.
            return a @ b + c @ d

        a = torch.randn(m, k).to(GPU_TYPE)
        b = torch.randn(k, n).to(GPU_TYPE)
        c = torch.randn(m, k).to(GPU_TYPE)
        d = torch.randn(k, n).to(GPU_TYPE)

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_search_space": search_space}
        ):
            torch.compile(mm_plus_mm, dynamic=dynamic)(a, b, c, d)

    def test_max_autotune_includes_max_autotune_pointwise_configs(self):
        # [FUNCTION DOC] Checks, using fake CUDA device properties and no real GPU, that the
        # pointwise config generator with max_autotune set returns as many configs as with
        # max_autotune_pointwise set, for 1D, 2D and 3D size hints. Fails with a message if the
        # counts differ.
        """
        Verify that `max_autotune` includes all pointwise configs from
        `max_autotune_pointwise` for 1D, 2D, and 3D pointwise kernels.
        """
        # Fake device properties for this unit test; no CUDA state is needed.
        triton_meta = {
            "device": DeviceProperties(
                type="cuda",
                index=0,
                multi_processor_count=1,
                cc=80,
                major=8,
                max_threads_per_block=1024,
                warp_size=32,
            )
        }
        inductor_meta_common = {"autotune_pointwise": False}

        for size_hints in (
            {"x": 2048},
            {"x": 128, "y": 256},
            {"x": 64, "y": 64, "z": 64},
        ):
            with self.subTest(size_hints=size_hints):
                max_autotune_configs = pointwise(
                    size_hints,
                    triton_meta=triton_meta,
                    inductor_meta={**inductor_meta_common, "max_autotune": True},
                    return_configs=True,
                )
                max_autotune_pointwise_configs = pointwise(
                    size_hints,
                    triton_meta=triton_meta,
                    inductor_meta={
                        **inductor_meta_common,
                        "max_autotune_pointwise": True,
                    },
                    return_configs=True,
                )

                self.assertEqual(
                    len(max_autotune_configs),
                    len(max_autotune_pointwise_configs),
                    "max_autotune should include all pointwise configs from max_autotune_pointwise",
                )

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(),
        "Hopper-style mm_persistent_tma template is shadowed by the Blackwell warp-specialized TMA template on data-center Blackwell. Covered by test_max_autotune_blackwell.py",
    )
    @skipIfXpu(msg="XPU TMA requires contiguous last dimension")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_max_autotune_regular_mm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
    ):
        # [FUNCTION DOC] Compiles a mm (inputs repeated 8x per dim for 16-byte TMA alignment,
        # optionally transposed) with max autotune restricted to the persistent TMA choice, with
        # and without TMA store and dynamic shapes. Checks numerics against eager and uses
        # FileCheck to confirm the generated kernel creates a TMA descriptor and uses the matching
        # load/store API (stable or experimental Triton TMA API). Skipped without device TMA
        # support or on datacenter Blackwell.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: repeats both inputs 8 times along each dim, optionally
            # transposes them per the enclosing test parameters, and returns their matrix product.
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

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        if has_triton_stable_tma_api():
            make_desc_api = "triton.language.make_tensor_descriptor"
            read_api = "tl.load_tensor_descriptor"
            if tma_store:
                # Note: The tma_descriptor0 is generated by the kernel. If the
                # code generation process changes this could change.
                write_api = "tma_descriptor0.store"
            else:
                write_api = "tl.store"
        else:
            make_desc_api = (
                "triton.language.extra.cuda.experimental_device_tensormap_create2d"
            )
            read_api = "tl._experimental_descriptor_load"
            # TMA store is not supported with the experimental API
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_mm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    def test_use_triton_tma_template_rejects_descriptor_shapes_exceeding_int32(self):
        # [FUNCTION DOC] Uses mocked matrices and output layout (no real tensors) to check that
        # the TMA template eligibility helper returns false when a descriptor dimension exceeds
        # the int32 max, and true when the dimensions are exactly at the int32 max. TMA support is
        # mocked as available.
        from torch._inductor.utils import use_triton_tma_template

        int32_max = torch.iinfo(torch.int32).max
        mat1 = mock.Mock()
        mat1.get_size.return_value = [128, int32_max + 1]
        mat2 = mock.Mock()
        mat2.get_size.return_value = [int32_max + 1, 128]
        output_layout = mock.Mock()
        output_layout.size = [128, 128]

        with (
            config.patch({"triton.enable_persistent_tma_matmul": True}),
            mock.patch("torch._inductor.utils.can_use_tma", return_value=True),
        ):
            self.assertFalse(
                use_triton_tma_template(mat1, mat2, output_layout=output_layout)
            )

        mat1.get_size.return_value = [128, int32_max]
        mat2.get_size.return_value = [int32_max, 128]

        with (
            config.patch({"triton.enable_persistent_tma_matmul": True}),
            mock.patch("torch._inductor.utils.can_use_tma", return_value=True),
        ):
            self.assertTrue(
                use_triton_tma_template(mat1, mat2, output_layout=output_layout)
            )

    def test_descriptor_shape_fits_in_int32_uses_expected_guarding(self):
        # [FUNCTION DOC] Checks guarding behavior of the int32 descriptor shape check on symbolic
        # sizes under a graph lowering context. With guards disabled it must consult the static-
        # truth query only, called once with the conjunction of bounds; with guards enabled it
        # must consult the guard-or-false query only. Both are mocked to return true; call counts
        # are asserted.
        from torch._inductor.utils import _descriptor_shape_fits_in_int32

        gm = make_fx(lambda: torch.zeros(2, 3))()
        graph = GraphLowering(gm)
        size0 = sympy.Symbol("s0", integer=True, nonnegative=True)
        size1 = sympy.Symbol("s1", integer=True, nonnegative=True)
        condition = sympy.And(
            sympy.Le(size0, torch.iinfo(torch.int32).max),
            sympy.Le(size1, torch.iinfo(torch.int32).max),
        )

        with V.set_graph_handler(graph):
            with (
                mock.patch.object(
                    V.graph.sizevars, "statically_known_true", return_value=True
                ) as statically_known_true,
                mock.patch.object(
                    V.graph.sizevars, "guard_or_false", return_value=True
                ) as guard_or_false,
            ):
                self.assertTrue(
                    _descriptor_shape_fits_in_int32(
                        [128, size0, size1], add_guards=False
                    )
                )
                statically_known_true.assert_called_once_with(condition)
                guard_or_false.assert_not_called()

            with (
                mock.patch.object(
                    V.graph.sizevars, "statically_known_true", return_value=True
                ) as statically_known_true,
                mock.patch.object(
                    V.graph.sizevars, "guard_or_false", return_value=True
                ) as guard_or_false,
            ):
                self.assertTrue(
                    _descriptor_shape_fits_in_int32(
                        [128, size0, size1], add_guards=True
                    )
                )
                guard_or_false.assert_called_once_with(condition)
                statically_known_true.assert_not_called()

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
    ):
        # [FUNCTION DOC] ROCm-only test: compiles a mm with the non-TMA persistent template
        # forced via the choice name regex, across transposition and dynamic-shape parameters.
        # Checks results against eager and that the generated kernel uses the NUM_SMS persistent-
        # grid parameter.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: repeats inputs 8 times per dim, optionally transposes them
            # per the enclosing test parameters, and returns their matrix product.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

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

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        # Verify that we are using the non-TMA persistent implementation
        FileCheck().check("triton_tem_fused_mm").check("NUM_SMS").run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_addmm_persistent(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
    ):
        # [FUNCTION DOC] ROCm-only test: like the persistent mm test but for addmm with a 1D bias
        # that is repeated 8x. Forces the non-TMA persistent choice, compares with eager, and
        # checks the generated fused addmm kernel mentions the NUM_SMS persistent-grid parameter.
        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: repeats bias and matrices 8x, optionally transposes the
            # matrices per the enclosing test parameters, and returns the addmm result.
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

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        # Verify that we are using the non-TMA persistent implementation
        FileCheck().check("triton_tem_fused_addmm").check("NUM_SMS").run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_persistent_tma_workspace_reuse(self):
        # [FUNCTION DOC] Regression test for workspace naming in the persistent TMA template.
        # Patches the TMA workspace-arg factory to hand every kernel the same WorkspaceArg object,
        # restricts configs to a single TMA config (non-TMA mm configs emptied), and compiles
        # three same-shape matmuls so template kernel rendering must give each kernel a unique
        # workspace name. The original heuristic config lists are restored in a finally block.
        # Passes if compilation succeeds.
        """
        Test that make_kernel_render creates unique workspace names.

        This test patches get_tma_workspace_arg to return the same WorkspaceArg
        instance, simulating the bug condition where templates share workspace_arg.
        The fix in make_kernel_render should create a new WorkspaceArg with a
        unique name for each kernel, preventing self-assignment bugs like
        'workspace_X = workspace_X; del workspace_X'.
        """
        from torch._inductor.codegen.common import WorkspaceZeroMode

        def three_same_shape_matmuls(a, b, c, d, e, f):
            # [FUNCTION DOC] HELPER: computes three independent matrix products and returns them
            # as a tuple.
            x = torch.mm(a, b)
            y = torch.mm(c, d)
            z = torch.mm(e, f)
            return x, y, z

        M, K, N = 4608, 2048, 7040

        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)

        mm_tma_heuristic = CUDAPersistentTMATemplateConfigHeuristic()
        mm_heuristic = CUDAMMTemplateConfigHeuristic()

        original_tma_configs = mm_tma_heuristic.mm_configs
        original_mm_configs = mm_heuristic.mm_configs

        # Create a single WorkspaceArg to be returned by all calls
        shared_workspace_arg = WorkspaceArg(
            count=1024,
            zero_mode=WorkspaceZeroMode.UNINITIALIZED,
            device=torch.device(GPU_TYPE),
            outer_name="shared_workspace",
        )

        def mock_get_tma_workspace_arg(*args, **kwargs):
            # [FUNCTION DOC] HELPER: ignores all arguments and returns the one shared
            # WorkspaceArg.
            return shared_workspace_arg

        try:
            # Force only TMA template by clearing non-TMA configs
            mm_heuristic.mm_configs = []

            # Use a single TMA config to ensure deterministic behavior
            mm_tma_heuristic.mm_configs = [GemmConfig(128, 128, 64, 4, 8, group_m=8)]

            with (
                config.patch(
                    {
                        "max_autotune_gemm": True,
                        "max_autotune_gemm_backends": "TRITON",
                        "triton.enable_persistent_tma_matmul": True,
                    }
                ),
                fresh_cache(),
                patch(
                    "torch._inductor.heuristics.template.triton.get_tma_workspace_arg",
                    mock_get_tma_workspace_arg,
                ),
            ):
                torch._dynamo.reset()
                compiled_fn = torch.compile(
                    three_same_shape_matmuls, mode="max-autotune-no-cudagraphs"
                )

                _, _ = run_and_get_code(compiled_fn, a, b, a, b, a, b)

        finally:
            mm_tma_heuristic.mm_configs = original_tma_configs
            mm_heuristic.mm_configs = original_mm_configs

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_workspace_size_bytes_accounts_for_dtype(self):
        # [FUNCTION DOC] Checks that the workspace size passed to TritonBenchmarkRequest is in
        # bytes (element count times dtype size). A fake float32 workspace arg is injected, the
        # benchmark request constructor is wrapped to record its workspace_size keyword, a single
        # TMA config is forced, and all non-None recorded sizes must equal the expected byte count
        # (and at least one must exist). Heuristic config lists are restored in a finally block.
        """workspace_size passed to benchmark request must be in bytes, not elements."""
        import sympy

        from torch._inductor.codegen.common import WorkspaceZeroMode
        from torch._inductor.utils import get_dtype_size

        count = 1024
        dtype = torch.float32
        expected_bytes = count * get_dtype_size(dtype)

        fake_ws = WorkspaceArg(
            count=sympy.Integer(count),
            zero_mode=WorkspaceZeroMode.UNINITIALIZED,
            device=torch.device(GPU_TYPE),
            outer_name="test_ws",
            dtype=dtype,
        )

        captured_sizes = []
        orig_init = TritonBenchmarkRequest.__init__

        def spy_init(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: wrapper for the benchmark request constructor that
            # records the workspace_size keyword into the enclosing list and then calls the
            # original constructor.
            captured_sizes.append(kwargs.get("workspace_size"))
            orig_init(self, *args, **kwargs)

        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)

        mm_tma_heuristic = CUDAPersistentTMATemplateConfigHeuristic()
        mm_heuristic = CUDAMMTemplateConfigHeuristic()
        original_tma_configs = mm_tma_heuristic.mm_configs
        original_mm_configs = mm_heuristic.mm_configs

        try:
            mm_heuristic.mm_configs = []
            mm_tma_heuristic.mm_configs = [GemmConfig(128, 128, 64, 4, 8, group_m=8)]

            with (
                config.patch(
                    {
                        "max_autotune_gemm": True,
                        "max_autotune_gemm_backends": "TRITON",
                        "triton.enable_persistent_tma_matmul": True,
                    }
                ),
                fresh_cache(),
                patch(
                    "torch._inductor.heuristics.template.triton.get_tma_workspace_arg",
                    return_value=fake_ws,
                ),
                patch.object(TritonBenchmarkRequest, "__init__", spy_init),
            ):
                torch._dynamo.reset()
                torch.compile(torch.mm, mode="max-autotune-no-cudagraphs")(a, b)

        finally:
            mm_tma_heuristic.mm_configs = original_tma_configs
            mm_heuristic.mm_configs = original_mm_configs

        ws_sizes = [s for s in captured_sizes if s is not None]
        self.assertTrue(len(ws_sizes) > 0, "No workspace benchmark requests created")
        for size in ws_sizes:
            self.assertEqual(size, expected_bytes)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(),
        "Hopper-style mm_persistent_tma template is shadowed by the Blackwell warp-specialized TMA template on data-center Blackwell.",
    )
    @skipIfXpu(msg="XPU TMA requires contiguous last dimension")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_strided(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
    ):
        # [FUNCTION DOC] Compiles a mm whose inputs are padded, non-contiguous (row stride rounded
        # up to a multiple of 16) tensors, with the persistent TMA choice forced; the original-
        # size inputs are repeated 8x in the compiled function. Checks numerics against eager and
        # that the kernel uses a TMA descriptor (stable or experimental API string). Note b_stride
        # is chosen from a_transposed rather than b_transposed, which looks unintended.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: repeats both inputs 8 times per dim, optionally transposes
            # them per the enclosing test parameters, and returns their product.
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

        def next_multiple_16(a: int) -> int:
            # [FUNCTION DOC] HELPER: rounds an integer up to the next multiple of 16.
            return ((a + 15) // 16) * 16

        M, N, K = 21, 31, 11
        a_shape = (K, M) if a_transposed else (M, K)
        a_stride = (
            (next_multiple_16(M), 1) if a_transposed else (next_multiple_16(K), 1)
        )
        a = torch.empty_strided(a_shape, a_stride, dtype=torch.float16).to(GPU_TYPE)
        a[:] = torch.randn(a_shape, dtype=torch.float16)
        a = a.to(GPU_TYPE)
        b_shape = (N, K) if b_transposed else (K, N)
        b_stride = (
            (next_multiple_16(K), 1) if a_transposed else (next_multiple_16(N), 1)
        )
        b = torch.empty_strided(b_shape, b_stride, dtype=torch.float16)
        b[:] = torch.randn(b_shape, dtype=torch.float16)
        b = b.to(GPU_TYPE)
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        # Verify that we are using a TMA implementation
        # depending on whether we're using the experimental API, we check for a different string
        check_str = "triton.language.extra.cuda.experimental_device_tensormap_create2d"
        if has_triton_stable_tma_api():
            check_str = "triton.language.make_tensor_descriptor"
        FileCheck().check("triton_tem_fused_mm").check(check_str).run(code[0])

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="Covered by XPU TMA")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_alignment(self, dynamic):
        # [FUNCTION DOC] Checks that when inputs have inner dims not 16-byte aligned, the
        # persistent TMA template is skipped; with only that choice allowed, compilation must
        # fail with a BackendCompilerFailed whose message mentions NoValidChoicesError.
        # Parametrized on dynamic.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: plain matrix product of its two inputs.
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_output_alignment(
        self, dynamic
    ):
        # [FUNCTION DOC] Checks that the persistent TMA template with TMA store is rejected when
        # the output tensor has strides that cannot satisfy TMA alignment; compilation with only
        # that choice must raise BackendCompilerFailed mentioning NoValidChoicesError.
        # Parametrized on dynamic.
        def mm(a, b, out):
            # [FUNCTION DOC] HELPER: writes the matrix product into the supplied out tensor and
            # returns it.
            torch.mm(a, b, out=out)
            return out

        M, N, K = 21, 31, 32
        a = torch.empty_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        a[:] = torch.randn((M, K), dtype=torch.float16)
        b = torch.empty_strided((K, N), (1, K), dtype=torch.float16, device=GPU_TYPE)
        b[:] = torch.randn((K, N), dtype=torch.float16)
        # allocate an output with a stride not divisible by 16, so it can't satisfy TMA alignment checks.
        out = torch.empty_strided((M, N), (N, 1), dtype=torch.float16, device=GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b, out)

        # Lowering to the persistent+TMA Triton template should be skipped
        # since the output doesn't have a stride of 1 in any dim
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_regular_mm_tma_dynamic_outer_dim(self):
        # [FUNCTION DOC] Compiles a mm whose first input has a dynamic outer dim under a
        # forced TMA choice (Blackwell warp-specialized name on datacenter Blackwell, else the
        # Hopper persistent TMA name) and compares the result with eager. Only checks numerics
        # and no crash.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: plain matrix product of its two inputs.
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.maybe_mark_dynamic(a, 0)

        choice_name_regex = (
            "blackwell_ws_persistent_device_tma"
            if has_datacenter_blackwell_tma_device()
            else "mm_persistent_tma"
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": choice_name_regex,
            }
        ):
            c_actual = torch.compile(mm)(a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(),
        "Hopper-style mm_persistent_tma template is shadowed by the Blackwell warp-specialized TMA template on data-center Blackwell.",
    )
    def test_persistent_tma_epilogue_fusion_store_cache(self):
        # [FUNCTION DOC] Regression test: with TMA store and epilogue fusion enabled, a mm
        # followed by relu must compile (failing autotune kernels are disallowed) and match eager.
        # Guards against a store cache not being updated, which would drop a buffer pointer from
        # the kernel signature.
        # Regression test: when epilogue fusion runs with TMA store, the
        # store_cache must be updated so that a subsequent epilogue load from
        # the same buffer hits the cache. Otherwise remove_kernel_local_buffers
        # strips the buffer pointer from the kernel signature, causing a
        # NameError at Triton compile time.
        def f(a, b):
            # [FUNCTION DOC] HELPER: repeats both inputs 8x per dim, multiplies them, returns relu
            # of result.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)
            mm = torch.mm(a, b)
            return mm.relu()

        M, N, K = 21, 31, 11
        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "epilogue_fusion": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": True,
                "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            actual = torch.compile(f)(a, b)
            expected = f(a, b)

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_zero_size_input(self, dynamic: bool):
        # [FUNCTION DOC] Regression test: compiling a sin-then-matmul function with a zero-row
        # input under max autotune must not crash; parametrized on dynamic shapes.
        """
        Make sure autotuning mm with zero-size input works without crashes.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: applies sin to the first input and returns its product with
            # the second.
            a = torch.sin(a)
            return a @ b

        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        with config.patch({"max_autotune": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    @fresh_cache()
    def test_addmm_1d_bias_no_reinterpret_tensor(self):
        # [FUNCTION DOC] Checks that the ATen addmm choice receives the 1D bias directly rather
        # than a reinterpreted view, so cuBLASLt can use its bias epilogue. Benchmarking is mocked
        # to favor ATen, the extern benchmark request constructor is wrapped to record the first
        # input shape, then the compiled result is compared with eager, the generated code must
        # not wrap the bias in a reinterpret call and the recorded bias shape must be one-
        # dimensional of size 100.
        """
        Verify that aten addmm with 1D bias does not wrap bias in reinterpret_tensor.
        This ensures cublasLt uses its optimized bias epilogue (requires dim==1).
        """

        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: addmm of the three inputs.
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        extern_bias_shape = None
        original_bmreq_init = ExternKernelBenchmarkRequest.__init__

        def tracking_bmreq_init(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: wrapper for the extern benchmark request constructor; calls
            # the original and records the sizes of the first input tensor meta into the
            # enclosing variable.
            nonlocal extern_bias_shape
            original_bmreq_init(self, *args, **kwargs)
            extern_bias_shape = tuple(self.input_tensor_meta[0].sizes)

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "ATEN,TRITON",
                    "test_configs.max_mm_configs": 1,
                }
            ),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_choice",
                mock_benchmark_choice_wrapper(aten_time=0.1, triton_time=1.0),
            ),
            mock.patch(
                "torch._inductor.autotune_process.run_autotune_in_subprocess",
                mock_benchmark_choice_wrapper(aten_time=0.1, triton_time=1.0),
            ),
            mock.patch.object(
                ExternKernelBenchmarkRequest,
                "__init__",
                tracking_bmreq_init,
            ),
        ):
            Y_compiled, code = run_and_get_code(torch.compile(addmm), x, a, b)
            Y = addmm(x, a, b)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

            # Verify addmm is called without reinterpret_tensor on bias
            FileCheck().check("addmm").run(code[0])
            self.assertNotIn("addmm(reinterpret_tensor", code[0])

            self.assertEqual((100,), extern_bias_shape)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(),
        "Hopper-style mm_persistent_tma template is shadowed by the Blackwell warp-specialized TMA template on data-center Blackwell; covered by test_max_autotune_blackwell.py",
    )
    @skipIfXpu(msg="XPU TMA requires contiguous last dimension")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
    ):
        # [FUNCTION DOC] Compiles an addmm (bias repeated 8x, matrices repeated and optionally
        # transposed) with the persistent TMA choice forced, with and without TMA store and
        # dynamic shapes. Checks numerics and uses FileCheck to confirm the fused addmm kernel
        # creates a TMA descriptor and uses the expected load/store API for the stable or
        # experimental TMA API.
        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: repeats bias and matrices 8x, optionally transposes the
            # matrices per the enclosing test parameters, and returns the addmm result.
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

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        if has_triton_stable_tma_api():
            make_desc_api = "triton.language.make_tensor_descriptor"
            read_api = "tl.load_tensor_descriptor"
            if tma_store:
                # Note: The tma_descriptor0 is generated by the kernel. If the
                # code generation process changes this could change.
                write_api = "tma_descriptor0.store"
            else:
                write_api = "tl.store"
        else:
            make_desc_api = (
                "triton.language.extra.cuda.experimental_device_tensormap_create2d"
            )
            read_api = "tl._experimental_descriptor_load"
            # TMA store is not supported with the experimental API
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_addmm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="Covered by XPU TMA")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_persistent_tma_illegal_alignment(self, dynamic):
        # [FUNCTION DOC] Checks that an addmm with non-16-byte-aligned inner dims cannot use the
        # persistent TMA template: compilation with only that choice must raise
        # BackendCompilerFailed mentioning NoValidChoicesError. Parametrized on dynamic.
        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: addmm of the three inputs.
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_addmm_tma_dynamic_outer_dim(self):
        # [FUNCTION DOC] Compiles an addmm whose matrix input has a dynamic outer dim under a
        # forced TMA choice (Blackwell warp-specialized on datacenter Blackwell, else Hopper
        # persistent TMA) and compares with eager; only checks numerics and no crash.
        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: addmm of the three inputs.
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        x = x.repeat(8)
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.maybe_mark_dynamic(a, 0)

        choice_name_regex = (
            "blackwell_ws_persistent_device_tma"
            if has_datacenter_blackwell_tma_device()
            else "mm_persistent_tma"
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": choice_name_regex,
            }
        ):
            c_actual = torch.compile(addmm)(x, a, b)
            c_expected = addmm(x, a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @fresh_cache()
    @skipIfXpu(msg="XPU doesn't support sm carveout")
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support sm carveout")
    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support persistent TMA")
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(), "B200 doesn't support sm carveout"
    )
    @parametrize("carveout", (None, 0, 27))
    @parametrize("op", ("mm", "scaled_mm"))
    def test_honor_sm_carveout_with_triton_tma(self, carveout, op: str):
        # [FUNCTION DOC] Checks that the SM carveout setting is honored by persistent TMA kernels.
        # Sets the experimental carveout (none, 0 or 27), compiles mm or scaled mm (float8 inputs)
        # with the TMA choice, profiles one run, and requires exactly one Triton template kernel
        # event whose grid size equals the device multiprocessor count minus the carveout (zero
        # when none). Side effect: leaves the global SM carveout set after the test.
        def mm_func(a, b):
            # [FUNCTION DOC] HELPER: plain matrix product of the two inputs.
            return torch.mm(a, b)

        def scaled_mm(
            a,
            b,
            scale_a,
            scale_b,
        ):
            # [FUNCTION DOC] HELPER: scaled matrix multiplication of the first two inputs using
            # the given scale tensors, producing a bfloat16 result.
            return torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)

        # Create large matrices to ensure we use all possible sms
        size = 2560
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.bfloat16)
        b = (
            torch.randn(size, size, device=GPU_TYPE, dtype=torch.bfloat16)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.tensor(1, dtype=torch.float32, device=GPU_TYPE)
        scale_b = torch.tensor(1, dtype=torch.float32, device=GPU_TYPE)

        args = (
            (a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), scale_a, scale_b)
            if op == "scaled_mm"
            else (a, b)
        )
        func = scaled_mm if op == "scaled_mm" else mm_func

        # Set the specified carveout value
        torch._C._set_sm_carveout_experimental(carveout)
        if carveout is None:
            self.assertIsNone(torch._C._get_sm_carveout_experimental())
        else:
            self.assertEqual(torch._C._get_sm_carveout_experimental(), carveout)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.native_matmul": False,
                "max_autotune_gemm_backends": "TRITON",
                "test_configs.autotune_choice_name_regex": "tma",
            }
        ):
            compiled_mm = torch.compile(func, mode="max-autotune-no-cudagraphs")
            compiled_mm(*args)  # Warm-up compilation

            with tempfile.NamedTemporaryFile() as f:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    # Run with the specified carveout
                    compiled_mm(*args)

                # Export trace and analyze results
                prof.export_chrome_trace(f.name)

                # Extract grid sizes from the trace events for TMA kernels
                kernel_name = "triton_tem_fused"
                with open(f.name) as file:
                    kernel_events = [
                        {
                            "grid": evt.get("args", {}).get("grid", []),
                            "grid_size": math.prod(evt.get("args", {}).get("grid", [])),
                        }
                        for evt in json.load(file)["traceEvents"]
                        if evt.get("cat", "") == "kernel"
                        and kernel_name in evt.get("name", "").lower()
                    ]

                # We should have exactly 1 kernel event for this run
                self.assertEqual(
                    len(kernel_events),
                    1,
                    lambda msg: f"{msg}\nExpected exactly 1 kernel event, but got {len(kernel_events)}",
                )

                # Check that grid size matches expected values based on carveout
                expected_grid_size = None
                max_grid_size = torch.cuda.get_device_properties(
                    "cuda"
                ).multi_processor_count
                careveout = 0 if carveout is None else carveout
                expected_grid_size = max_grid_size - careveout

                self.assertEqual(
                    kernel_events[0]["grid_size"],
                    expected_grid_size,
                    lambda msg: f"{msg}\nGrid size {kernel_events[0]['grid_size']} doesn't match {expected_grid_size} for carveout={carveout}",
                )

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_zero_size_input(self, dynamic):
        # [FUNCTION DOC] Regression test: compiling an addmm with a zero-row matrix input under
        # max autotune must not crash; parametrized on dynamic shapes.
        """
        Make sure autotuning addmm with zero-size input works without crashes.
        """

        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: addmm of the three inputs.
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        with config.patch({"max_autotune": True}):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_unrealized_view_bias(self, dynamic):
        # [FUNCTION DOC] Regression test: compiling an addmm whose bias is an unrealized
        # permuted view of a pointwise result under max autotune must not crash; parametrized on
        # dynamic shapes.
        """
        Make sure autotuning addmm with an unrealized view-class bias
        (here a PermuteView over a fused Pointwise) works without crashes.
        """

        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: doubles the first input, transposes it and uses it as bias
            # for addmm.
            return torch.addmm((x * 2.0).transpose(0, 1), a, b)

        x = torch.randn(8, 8).to(GPU_TYPE)
        a = torch.randn(8, 16).to(GPU_TYPE)
        b = torch.randn(16, 8).to(GPU_TYPE)
        with config.patch({"max_autotune": True}):
            torch.compile(fn, dynamic=dynamic)(x, a, b)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_autotune_conv1x1(self, search_space):
        # [FUNCTION DOC] Compiles a 1x1 convolution on a channels-last input with Triton as the
        # only GEMM and conv backend, for both search spaces, under no_grad. Checks the generated
        # code has no extern convolution call and that the output matches eager within a loose
        # absolute tolerance.
        # Assuming input has 3 channels and we want to produce 16 channels as output
        conv1x1 = (
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
            .to(memory_format=torch.channels_last)
            .to(GPU_TYPE)
        )

        # Example input tensor: batch size = 4, channels = 3, height = 32, width = 32
        # The memory format is set to `channels_last`
        input_tensor = (
            torch.randn(4, 3, 32, 32)
            .contiguous(memory_format=torch.channels_last)
            .to(GPU_TYPE)
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune_conv_backends": "TRITON",
                "max_autotune_gemm_search_space": search_space,
            }
        ):

            @torch.compile()
            def foo(mod, x):
                # [FUNCTION DOC] HELPER: applies the given module to the input.
                return mod(x)

            with torch.no_grad():
                out, code = run_and_get_code(foo, conv1x1, input_tensor)

            FileCheck().check_not("extern_kernels.convolution").run(code[0])
            self.assertEqual(conv1x1(input_tensor), out, atol=1e-2, rtol=0)

    @fresh_cache()
    @config.patch(max_autotune=True, max_fusion_size=2)
    def test_jit_fusion_matches_aot_fusion(self):
        # [FUNCTION DOC] Compiles via AOT export a function mixing pointwise ops, a scalar item
        # and a matmul, with max autotune and a fusion size limit of two, to check that the AOT
        # compile path fuses the same nodes as the JIT path. Only verifies that compilation does
        # not raise.
        # In this example, AOTInductor's JIT-compile will fuse(buf1, buf2) due
        # to proximity, we want to make sure AOT-compile pass does the same.
        # AOT could do fuse(buf2, buf4) instead if buf3 was pushed to the end
        # of the V.graph.buffers list because fuse(buf2, buf4) would have a
        # better proximity score than fuse(buf1, buf2). This scenario is possible
        # since finalizing MultiTemplateBuffers needs to replace buffers.
        def fn(x, number):
            # [FUNCTION DOC] HELPER: returns several pointwise results, an item scalar and a
            # matmul result.
            buf0 = x + x
            buf1 = number.item()
            buf2 = x * x
            buf3 = x @ x  # MultiTemplateBuffer
            buf4 = x**2
            return buf0, buf1, buf2, buf3, buf4

        inputs = (
            torch.rand([256, 256], device=GPU_TYPE),
            torch.tensor(3, device=GPU_TYPE),
        )
        torch._export.aot_compile(fn, args=inputs)

    def test_cat_addmm(self):
        # [FUNCTION DOC] Compiles a function concatenating two addmm results along dim 1 with the
        # Triton-only GEMM backend and compares against eager within loose tolerance.
        def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            # [FUNCTION DOC] HELPER: concatenates addmm of the three inputs in two
            # argument orders.
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                ],
                1,
            )

        args = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @config.patch(
        benchmark_kernel=True,
        fallback_random=True,
        max_autotune_gemm=True,
    )
    @parametrize("device", ("cpu", GPU_TYPE))
    def test_matmul_dropout(self, device):
        # [FUNCTION DOC] Compiles a forward and backward pass of matmul followed by dropout on cpu
        # and GPU with fallback random enabled, resets the RNG state before each run, and compares
        # the input gradient with eager within loose tolerance.
        def fwd(a, b):
            # [FUNCTION DOC] HELPER: matrix product followed by dropout with probability 0.1.
            x = a @ b
            x = torch.nn.functional.dropout(x, 0.1)
            return x

        def fn(a, b):
            # [FUNCTION DOC] HELPER: runs forward, sums, backpropagates and returns the gradient
            # of a.
            x = fwd(a, b).sum()
            x.backward()
            return a.grad

        N = 128
        a = torch.randn(N, N, device=device, requires_grad=True)
        b = torch.randn(N, N, device=device)

        opt_fn = torch.compile(fn)
        reset_rng_state()
        ref = fn(a, b)
        reset_rng_state()
        act = opt_fn(a, b)

        if N <= 8:
            print(f"ref\n{ref}\nact\n{act}")
        torch.testing.assert_close(ref, act, atol=1e-1, rtol=1e-1)

    @config.patch(
        max_autotune_gemm=True,
    )
    @unittest.skipIf(
        getattr(torch, GPU_TYPE).device_count() < 2,
        "Need at least 2 devices for this test",
    )
    def test_autotune_device_guard(self):
        # [FUNCTION DOC] Needs at least two devices. Compiles a matmul on tensors that live on the
        # second device while the current device is the default, in a fresh cache, and compares
        # with eager.
        x = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")
        y = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")

        def f(x, y):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return x @ y

        with fresh_cache():
            act = torch.compile(f)(x, y)
        ref = f(x, y)
        self.assertTrue(torch.allclose(act, ref, atol=4 * 1e-3, rtol=4 * 1e-3))

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/179777")
    @config.patch(max_autotune=True)
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    @parametrize("kernel_size", (1, 3))
    def test_empty_conv_input(self, search_space, kernel_size):
        # [FUNCTION DOC] Compiles a convolution with a zero-sized batch under max autotune
        # for both search spaces and kernel sizes 1 and 3, and compares against eager
        # within tolerance.
        x = torch.randn(0, 256, 14, 14, device=GPU_TYPE)
        weight = torch.randn(256, 256, kernel_size, kernel_size, device=GPU_TYPE)

        def f(x, weight):
            # [FUNCTION DOC] HELPER: generic convolution with unit stride/dilation, no padding
            # or bias.
            return torch.convolution(
                x,
                weight,
                bias=None,
                stride=[1, 1],
                padding=[0, 0],
                dilation=[1, 1],
                transposed=False,
                output_padding=[0, 0],
                groups=1,
            )

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            opt_f = torch.compile(f)
            ref = f(x, weight)
            act = opt_f(x, weight)
            self.assertTrue(torch.allclose(ref, act, atol=4 * 1e-3, rtol=4 * 1e-3))

    @config.patch(max_autotune_gemm_backends="TRITON")
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_baddmm(self, search_space):
        # [FUNCTION DOC] Compiles a module computing baddbmm (float16 bias and weights) with
        # the Triton-only GEMM backend in max-autotune mode, for both search spaces. Compares
        # with eager and, unless native matmul is enabled, expects a fused Triton baddbmm
        # template kernel.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: tiny module holding a float16 weight parameter and a bias
            # parameter and computing a batched add-matmul of the input with them.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the random float16 weight and bias parameters.
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(64, 64, 192, dtype=torch.float16)
                )
                self.bias = torch.nn.Parameter(
                    torch.randn(64, 1, 192, dtype=torch.float16)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns the baddbmm of bias, input and weight.
                return torch.ops.aten.baddbmm.default(self.bias, x, self.weight)

        x = torch.randn(
            64, 2048, 64, dtype=torch.float16, requires_grad=False, device=GPU_TYPE
        )
        mod = M().to(GPU_TYPE)

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            m_c = torch.compile(mode="max-autotune")(mod)
            out, code = run_and_get_code(m_c, x)
            self.assertEqual(out, mod(x), atol=2e-3, rtol=2e-3)

            if not config.triton.native_matmul:
                FileCheck().check("triton_tem_fused_baddbmm").run(code[0])

    @config.patch(max_autotune=True)
    def test_conv1x1_with_free_symbols(self):
        # [FUNCTION DOC] Checks that max autotune handles a 1x1 convolution whose input size
        # depends on data-dependent (nonzero) sizes without raising due to free symbols.
        # Runs twice.
        """
        Make sure there is no exception due to free symbols.
        """
        conv = nn.Conv2d(
            3, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        ).to(device=GPU_TYPE)

        @torch.compile
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: slices the input by nonzero counts of two other tensors and
            # applies the enclosing convolution.
            h = y.nonzero().size(0)
            w = z.nonzero().size(0)
            x = x[:, :, :h, :w]
            x = conv(x)
            return x

        x = torch.randn(4, 3, 224, 224).to(
            memory_format=torch.channels_last, device=GPU_TYPE
        )
        for _ in range(2):
            y = torch.randint(0, 10, (224,)).to(device=GPU_TYPE)
            z = torch.randint(0, 10, (224,)).to(device=GPU_TYPE)
            f(x, y, z)

    def _test_cat_max_autotune_impl(self, using_triton_mm):
        # [FUNCTION DOC] HELPER: shared body for cat tests, with using_triton_mm telling whether
        # the mm is a Triton kernel. Compiles three functions in max-autotune-no-cudagraphs mode
        # and checks numerics; for the first two checks exact kernel launch counts (mm plus cos
        # kernels; the second case with a multi-consumer mm gets one extra launch). The third only
        # checks numerics.
        def f(x, y):
            # [FUNCTION DOC] HELPER: cos of y, matmul of x, concatenated along dim 0.
            y = torch.cos(y)
            x = torch.mm(x, x)
            return torch.cat([x, y])

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        inps = [
            torch.randn(32, 32, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        _, code = run_and_get_code(f_c, inps[0], inps[1])
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)

        # mm kernel, and cos kernel
        count = 2 if (using_triton_mm or config.triton.native_matmul) else 1
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), count, exactly=True
        ).run(code[0])

        def f(x, y):
            # [FUNCTION DOC] HELPER: like the previous variant but also returns x plus one.
            y = torch.cos(y)
            x = torch.mm(x, x)
            out = torch.cat([x, y])
            return out, x + 1

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        _, code = run_and_get_code(f_c, inps[0], inps[1])
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)
        # x has multi-consumers (cat + add), so cat skips pointwise_cat
        # and uses ConcatKernel.  cos is no longer fused into cat, giving
        # one extra kernel launch compared to the single-consumer case.
        count = 3 if (using_triton_mm or config.triton.native_matmul) else 2
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), count, exactly=True
        ).run(code[0])

        def f(x, y):
            # [FUNCTION DOC] HELPER: returns two concatenations of the mm result and cos in
            # both orders.
            y = torch.cos(y)
            x = torch.mm(x, x)
            return torch.cat([x, y]), torch.cat([y, x])

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)

    @config.patch("trace.enabled", True)
    @config.patch({"test_configs.force_extern_kernel_in_multi_template": True})
    @config.patch("triton.native_matmul", False)
    def test_mutation_rename(self):
        # [FUNCTION DOC] Checks that dependency normalization of a mutated buffer is the same
        # before and after finalizing a multi-template buffer. Enables IR trace logging into a
        # temporary debug directory, compiles a function that mutates a diagonal view then does mm
        # and an in-place add (extern kernel forced in the multi-template), captures the pre-
        # fusion and post-fusion IR logs and uses FileCheck to require an unmet dependency on buf1
        # in each. Resets logging at the end.
        torch._logging.set_logs(ir_post_fusion=True)

        def f(x, y, z, other):
            # [FUNCTION DOC] HELPER: multiplies x and y, copies other into the diagonal of the
            # product, matmuls with z, then adds one in place to the diagonal of the result and
            # returns that view.
            mul = x * y
            diag = torch.diagonal(mul)
            diag.copy_(other)
            x = torch.mm(mul, z)
            y = torch.diagonal(x).add_(torch.tensor(1, device=GPU_TYPE))
            return y

        t = functools.partial(torch.randn, device=GPU_TYPE)
        inps = (t(3, 3), t(3, 3), t(3, 3), t(3))
        fn = torch.compile(f, mode="max-autotune-no-cudagraphs")

        (
            (
                pre_fusion_tream,
                post_fusion_stream,
            ),
            ctx,
        ) = multiple_logs_to_string(
            "torch._inductor.debug", "ir_pre_fusion", "ir_post_fusion"
        )

        with config.patch({"trace.debug_dir": tempfile.mkdtemp()}):
            with (
                self.assertLogs(
                    logging.getLogger("torch._inductor.debug"), level=logging.INFO
                ) as cm,
                ctx(),
            ):
                out = fn(*inps)

        self.assertEqual(f(*inps), out)

        pre_fusion_stream = cm.output[0]
        post_fusion_stream = cm.output[1]

        # before and after finalizing multi template buffer, deps should have the same normalization
        # wrt writes
        FileCheck().check("MultiTemplateBuffer").check("unmet").check_same("buf1").run(
            pre_fusion_stream
        )
        FileCheck().check("ExternKernelSchedulerNode").check("unmet").check_same(
            "buf1"
        ).run(post_fusion_stream)

        torch._logging.set_logs()

    @config.patch({"test_configs.force_extern_kernel_in_multi_template": True})
    def test_cat_max_autotune_extern(self):
        # [FUNCTION DOC] Runs the shared cat/max-autotune checks with an ATen extern kernel forced
        # in the multi-template, so the mm is not a Triton kernel.
        self._test_cat_max_autotune_impl(using_triton_mm=False)

    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "benchmark_epilogue_fusion": False,
        }
    )
    def test_cat_max_autotune_triton(self):
        # [FUNCTION DOC] Runs the shared cat/max-autotune checks with the Triton-only GEMM backend
        # and epilogue-fusion benchmarking disabled, so the mm is a Triton template kernel.
        self._test_cat_max_autotune_impl(using_triton_mm=True)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_conv_cat(self, search_space):
        # [FUNCTION DOC] Compiles a conv followed by concatenation of its output with output plus
        # one under no_grad in max-autotune-no-cudagraphs mode, for both search spaces. Checks the
        # output equals the module on the same input and, off ROCm, that a fused pointwise add/cat
        # kernel exists.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: tiny module with one bias-free 3x3 convolution (3 to 64
            # channels) whose forward concatenates the convolution output with that output plus
            # one along dim 0.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the convolution layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the convolution and returns the concatenation of
                # the result with the result plus one.
                x = self.conv(x)
                return torch.cat((x, x + 1))

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            with torch.no_grad():
                m = ToyModel().to(device=GPU_TYPE)
                input_tensor = torch.randn(32, 3, 64, 64).to(device=GPU_TYPE)

                # convolution is not currently plannable
                m = torch.compile(m, mode="max-autotune-no-cudagraphs")
                out, code = run_and_get_code(m, input_tensor)
                self.assertEqual(out, m(input_tensor))

                if not TEST_WITH_ROCM:
                    FileCheck().check("def triton_poi_fused_add_cat_").run(code[0])

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_conv3d(self, search_space):
        # [FUNCTION DOC] Compiles the functional 3D convolution on random CPU tensors with max
        # autotune for both search spaces and compares with eager within small tolerances.
        fn = torch.nn.functional.conv3d
        image = torch.randn([1, 3, 8, 16, 32])
        filt = torch.randn([3, 3, 7, 7, 7])

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_search_space": search_space}
        ):
            expected = fn(image, filt)
            actual = torch.compile(fn)(image, filt)
            torch.testing.assert_close(actual, expected, atol=6e-5, rtol=0.001)

    @config.patch(
        max_autotune=True, max_autotune_conv_backends="", layout_optimization=False
    )
    def test_conv_backend(self):
        # [FUNCTION DOC] With the conv backend list empty, layout optimization off and max
        # autotune on, compiling a 1x1 convolution must fail with BackendCompilerFailed
        # mentioning NoValidChoicesError.
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
        ).to(GPU_TYPE)
        inp = torch.randn([2, 3, 16, 16]).to(GPU_TYPE)

        with self.assertRaises(BackendCompilerFailed) as context:
            torch.compile(m)(inp)

        self.assertIn("NoValidChoicesError", str(context.exception))

    def test_non_contiguous_input_mm(self):
        # [FUNCTION DOC] Regression test (issue 125437): compiling a matmul under max-autotune
        # with a non-contiguous bfloat16 first input (column-major, large) must not crash and must
        # match eager.
        """
        Make sure the triton template can work with non-contiguous inputs without crash.
        Check https://github.com/pytorch/pytorch/issues/125437 for more details.
        """
        x = rand_strided(
            (50257, 2048), (1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided((2048, 768), (768, 1), dtype=torch.bfloat16, device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x, y):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return x @ y

        ref = x @ y
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    def test_non_contiguous_input_addmm(self):
        # [FUNCTION DOC] Like the non-contiguous mm test but for addmm with a closed-over bias:
        # compiles under max-autotune with a column-major first matrix and compares with eager.
        b = torch.randn((768), dtype=torch.bfloat16, device=GPU_TYPE)
        x = rand_strided(
            (50257, 2048), (1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided((2048, 768), (768, 1), dtype=torch.bfloat16, device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x, y):
            # [FUNCTION DOC] HELPER: addmm using the enclosing bias with the two inputs.
            return torch.addmm(b, x, y)

        ref = torch.addmm(b, x, y)
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    def test_non_contiguous_input_bmm(self):
        # [FUNCTION DOC] Like the non-contiguous mm test but for bmm with a broadcast batch
        # of one: compiles under max-autotune with a non-contiguous first input and compares
        # with eager.
        x = rand_strided(
            (1, 50257, 2048), (0, 1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided(
            (1, 2048, 768), (0, 768, 1), dtype=torch.bfloat16, device=GPU_TYPE
        )

        @torch.compile(mode="max-autotune")
        def f(x, y):
            # [FUNCTION DOC] HELPER: batched matrix product of the two inputs.
            return torch.bmm(x, y)

        ref = torch.bmm(x, y)
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    @unittest.skipIf(
        config.triton.native_matmul,
        "native matmul and Triton template both have accuracy fail (2.2%)",
    )
    def test_non_contiguous_input_mm_plus_mm(self):
        # [FUNCTION DOC] Compiles a sum of two matmuls with non-contiguous first operands
        # under max-autotune and compares with eager (loose tolerance). Skipped for native
        # matmul codegen.
        x1 = rand_strided((50257, 2048), (1, 50304), device=GPU_TYPE)
        y1 = rand_strided((2048, 768), (768, 1), device=GPU_TYPE)

        x2 = rand_strided((50257, 2048), (1, 50304), device=GPU_TYPE)
        y2 = rand_strided((2048, 768), (768, 1), device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x1, y1, x2, y2):
            # [FUNCTION DOC] HELPER: returns the sum of two matrix products.
            return x1 @ y1 + x2 @ y2

        ref = x1 @ y1 + x2 @ y2
        act = f(x1, y1, x2, y2)
        torch.testing.assert_close(act, ref, atol=1e-1, rtol=1e-2)

    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="",
    )
    @unittest.skipIf(
        config.triton.native_matmul, "native matmul generates when size >=2"
    )
    def test_no_valid_choices(self):
        # [FUNCTION DOC] With an empty GEMM backend list, compiling a small matmul must raise
        # BackendCompilerFailed mentioning NoValidChoicesError. Skipped for native matmul.
        a = torch.zeros([2, 2], device=GPU_TYPE)
        b = torch.zeros([2, 2], device=GPU_TYPE)
        with self.assertRaises(BackendCompilerFailed) as context:
            torch.compile(lambda a, b: a.matmul(b))(a, b)
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        config.triton.native_matmul, "Only test when template is being called"
    )
    @parametrize("multi_template", (True, False))
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_inf_timing(self, multi_template):
        # [FUNCTION DOC] Checks that when every candidate timing is infinite the compile fails
        # with BackendCompilerFailed mentioning NoValidChoicesError. The selector lookup is
        # patched to map each timing to infinity; parametrized on whether epilogue fusion
        # benchmarking (multi-template) is enabled.
        lookup = AlgorithmSelectorCache.lookup

        def mock_lookup(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: calls the original selector lookup and returns a mapping from
            # each choice to infinite time.
            timings = lookup(self, *args, **kwargs)
            return {choice: float("inf") for choice in timings}

        a = torch.zeros([16, 16], device=GPU_TYPE)
        b = torch.zeros([16, 16], device=GPU_TYPE)
        with (
            patch.object(AlgorithmSelectorCache, "lookup", mock_lookup),
            config.patch(benchmark_epilogue_fusion=multi_template),
        ):
            with self.assertRaises(BackendCompilerFailed) as context:
                torch.compile(lambda a, b: a.matmul(b))(a, b)
            self.assertIn("NoValidChoicesError", str(context.exception))

    @config.patch(force_shape_pad=True, max_autotune=True)
    def test_linear_and_cel(self):
        # [FUNCTION DOC] Simulates a GPU that is not big (is_big_gpu patched to false) with shape
        # padding forced and max autotune enabled, so the multi-template holds only extern
        # kernels. Trains a linear layer plus cross-entropy step both eager and compiled and
        # requires that loss and gradients agree within tolerance, raising AssertionError with
        # both otherwise.
        """
        Similate a GPU without enough SMs. Make sure max-autotune still
        works even when the MultiTritonTemplate encapsulates just extern
        kernels.
        """

        def mock_is_big_gpu(*args, **kwargs):
            # [FUNCTION DOC] HELPER: ignores its arguments and reports that the GPU is not big.
            return False

        B, T, C, V = 32, 1024, 768, 50257

        linear = nn.Linear(C, V).bfloat16().to(device=GPU_TYPE)
        ce = torch.nn.CrossEntropyLoss()

        def f(x, y):
            # [FUNCTION DOC] HELPER: clears gradients of input and linear parameters, computes
            # cross-entropy of the linear layer output, runs backward and returns the loss.
            x.grad = None
            linear.weight.grad = None
            linear.bias.grad = None

            loss = ce(linear(x), y)
            loss.backward()
            return loss

        x = torch.randn(B * T, C, requires_grad=True).to(GPU_TYPE).bfloat16()
        x.retain_grad()
        y = torch.randint(0, V, (B * T,)).to(GPU_TYPE)

        import torch._inductor.utils as inductor_utils

        with unittest.mock.patch.object(inductor_utils, "is_big_gpu", mock_is_big_gpu):
            opt_f = torch.compile(f)

            expect = (f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
            actual = (opt_f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
            if not same(expect, actual, tol=1e-2):
                raise AssertionError(f"ref:\n{expect}\nact:\n{actual}")

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @parametrize("dynamic", (True, False))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("sizes", ((32, 32, 32768), (64, 128, 200000), (64, 64, 177147)))
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        comprehensive_padding=False,
        shape_padding=False,
    )
    def test_max_autotune_decompose_k(self, sizes, dtype, dynamic):
        # [FUNCTION DOC] Checks decompose-K matmul with large K (float16 or bfloat16, three size
        # configs, static or dynamic). Reduced-precision reductions are disabled for the test
        # and the original settings restored at the end. Static case requires a batched matmul
        # extern kernel and a decompose_k Triton kernel whose name carries a valid split
        # divisor, plus numeric match with eager; dynamic case requires neither. Repeated with
        # relu epilogue and a transposed input. Note the restore of settings is skipped if an
        # assertion fails earlier.
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):
            fp16_red_setting = (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            )
            bf16_red_setting = (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

            M, N, K = sizes

            atol = 1e-4
            rtol = 1e-4
            # K can be huge huge, this is why the data distribution is set to iid N(0, K ** 0.5),
            # which makes the result of reductions distributed as N(0, 1).
            a, b = self._make_matrices(
                M,
                K,
                N,
                dtype=dtype,
                device=GPU_TYPE,
                requires_grad=True,
            )

            possible_splits = range(2, min(K // M, K // N) + 1)

            divisors = {split for split in possible_splits if K % split == 0}

            def check_divisors(code):
            # [FUNCTION DOC] HELPER: for each generated kernel source whose text mentions
            # decompose_k, asserts that it contains one of the valid split divisors of K (each
            # between 2 and the min of K over M and K over N) in its name.
                for kernel in code:
                    if "decompose_k" in kernel:
                        divisor_found = False
                        for divisor in divisors:
                            if f"{divisor}_split" in kernel:
                                divisor_found = True
                                break

                        self.assertTrue(
                            divisor_found,
                            lambda msg: f"{msg}\nCould not find a split in {divisors} in {kernel}",
                        )

            compiled_func = torch.compile(lambda a, b: a @ b, dynamic=dynamic)
            # We assume with the large k dim relative to m, n, decompose_k will be most performant
            out, code = run_and_get_code(compiled_func, a, b)

            if dynamic:
                FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                    "decompose_k"
                ).run(code[0])
            else:
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*.run"
                ).check("decompose_k").run(code[0])
                check_divisors(code)
                torch.testing.assert_close(out, a @ b, atol=atol, rtol=rtol)

            # Test adding epilogue also equivalent to eager
            compiled_func = torch.compile(lambda a, b: (a @ b).relu(), dynamic=dynamic)
            out, code = run_and_get_code(compiled_func, a, b)
            if dynamic:
                FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                    "decompose_k"
                ).run(code[0])
            else:
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*.run"
                ).check("decompose_k").run(code[0])
                check_divisors(code)
                torch.testing.assert_close(
                    compiled_func(a, b), (a @ b).relu(), atol=atol, rtol=rtol
                )

            # Test adding reinterpret view before subgraph
            a = a.transpose(0, 1)
            compiled_func = torch.compile(
                lambda a, b: (a.transpose(0, 1) @ b).relu(), dynamic=dynamic
            )
            out, code = run_and_get_code(compiled_func, a, b)

            if dynamic:
                FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                    "decompose_k"
                ).run(code[0])
            else:
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*_0.run"
                ).check("decompose_k").run(code[0])
                check_divisors(code)
                torch.testing.assert_close(
                    compiled_func(a, b),
                    (a.transpose(0, 1) @ b).relu(),
                    atol=atol,
                    rtol=rtol,
                )

            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                fp16_red_setting
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                bf16_red_setting
            )

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_dynamic_input(self):
        # [FUNCTION DOC] Decompose-K with a dynamic leading dim: stacks the input with itself then
        # matmul and relu. The decompose-K choice helper is mocked to accept only default
        # threshold multiples. Requires batched matmul extern kernel, decompose_k kernel and
        # symbolic size expressions in the generated code, and numeric match with eager.
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):

            def f(a, b):
            # [FUNCTION DOC] HELPER: stacks a twice along a new first dim, matmuls with b and
            # applies relu.
                a_in = torch.stack((a, a), dim=0)
                return (a_in @ b).relu()

            a, b = self._make_matrices(
                M=32,
                K=32768,
                N=64,
                dtype=torch.bfloat16,
                device=GPU_TYPE,
                requires_grad=True,
            )

            torch._dynamo.reset()
            torch._dynamo.maybe_mark_dynamic(a, 0)
            compiled_func = torch.compile(f)

            with mock.patch(
                "torch._inductor.kernel.mm.use_decompose_k_choice"
            ) as decomp_mock:
                decomp_mock.side_effect = (
                    lambda *args, **kwargs: kwargs.get("threshold_multiple", 1) == 1
                )

                out, code = run_and_get_code(compiled_func, a, b)
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*.run"
                ).check("decompose_k").check_regex(r"s[0-9]+ = s[0-9]+").check_regex(
                    r"2\*s[0-9]+"
                ).check_regex("s[0-9]+ = 32").run(code[0])
                torch.testing.assert_close(
                    out,
                    f(a, b),
                    atol=1e-4,
                    rtol=1e-4,
                )

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_dynamic_input_bwd(self):
        # [FUNCTION DOC] Decompose-K in a forward plus backward pass with a dynamic leading dim.
        # Compiles and runs once, then with the decompose-K choice mocked, resets dynamo and uses
        # a fresh cache, so the backward is recompiled inside the mock. Checks the backward graph
        # code (second entry) for batched matmul, decompose_k and symbolic size expressions.
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):

            def f(a, b):
            # [FUNCTION DOC] HELPER: concatenates 256 copies of a along dim 0, matmuls with b,
            # relu and sums.
                # 256 * s0
                a_in = torch.cat([a for _ in range(256)], dim=0)
                return (a_in @ b).relu().sum()

            a, b = self._make_matrices(
                M=8,
                K=64,
                N=32768,
                dtype=torch.bfloat16,
                device=GPU_TYPE,
                requires_grad=True,
            )

            torch._dynamo.reset()
            torch._dynamo.maybe_mark_dynamic(a, 0)
            compiled_func = torch.compile(f)
            res = compiled_func(a, b)
            res.backward()

            with mock.patch(
                "torch._inductor.kernel.mm.use_decompose_k_choice"
            ) as decomp_mock:
                decomp_mock.side_effect = (
                    lambda *args, **kwargs: kwargs.get("threshold_multiple", 1) == 1
                )

                def fwd_bwd():
                # [FUNCTION DOC] HELPER: runs the compiled function, calls backward on the output
                # and returns it.
                    out = compiled_func(a, b)
                    out.backward()
                    return out

                # The mock only affects autotuning choices, so the backward has
                # to be compiled (not served from the AOTAutograd/FX caches)
                # inside this block for the choice to show up in the code. Both
                # passes are captured together because the backward is only
                # compiled once .backward() runs.
                torch._dynamo.reset()
                with fresh_cache():
                    out, code = run_and_get_code(fwd_bwd)

                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*.run"
                ).check("decompose_k").check_regex(r"s[0-9]+ = s[0-9]+").check_regex(
                    r"256\*s[0-9]+"
                ).check_regex("s[0-9]+ = 8").run(
                    # code[1] in this case given backwards
                    code[1]
                )

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_output_stride(self):
        # [FUNCTION DOC] Checks that decompose-K honors the output stride of the original graph.
        # Forces only the decompose-K choice via template heuristic override and a mocked
        # eligibility check, compiles a matmul of a transposed input with a column-sliced input,
        # then requires the output stride to be that of the slice width, and the generated code
        # allocates the matching shape.
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):

            def f(a, b):
            # [FUNCTION DOC] HELPER: transposes the first input and returns its matrix product
            # with b.
                a = a.transpose(0, 1)
                return a @ b

            a = torch.randn((32768, 256), device=GPU_TYPE, dtype=torch.bfloat16)
            b = torch.randn((32768, 1152), device=GPU_TYPE, dtype=torch.bfloat16)

            b = b[:, :1096]

            # Force only decomposeK choice
            with (
                override_template_heuristics(
                    device_type=GPU_TYPE,
                    template_op_pairs=[
                        (torch._inductor.kernel.mm.mm_template.name, "mm")
                    ],
                ),
                mock.patch(
                    "torch._inductor.kernel.mm.use_decompose_k_choice"
                ) as decompose_mock,
            ):
                decompose_mock.return_value = True
                compiled_f = torch.compile(f)
                out, code = run_and_get_code(compiled_f, a, b)

                # Output stride equal to original gm output stride
                # If output stride is not correctly checked, this will be (1152, 1) which can cause nans
                self.assertEqual(out.stride(), (1096, 1))

                FileCheck().check_not("extern_kernels.bmm_dtype").check(
                    "decompose_k"
                ).check(
                    f" empty_strided_{GPU_TYPE}((256, 1096), (1096, 1), torch.bfloat16)"
                ).run(code[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("sizes", ((64, 128, 256), (128, 256, 512), (256, 512, 1024)))
    @config.patch(
        max_autotune=True,
    )
    def test_max_autotune_contiguous_transform_mm(self, sizes, dtype):
        # [FUNCTION DOC] ROCm-only: with the contiguous-use check mocked true, compiles a matmul
        # of a with the transpose of b and checks the result against a float64 baseline cast to
        # the test dtype, and that the generated code mentions the contiguous mm choice.
        """
        Test the contiguous subgraph transform with A * transpose(B) pattern.
        This transform makes the second matrix contiguous before the matmul.
        """
        M, N, K = sizes

        def mm_transpose(a, b):
            # [FUNCTION DOC] HELPER: matrix product of a with the transpose of b.
            return a @ b.transpose(0, 1)

        a = torch.randn(M, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)

        # Compute fp64 baseline
        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        expected_fp64 = mm_transpose(a_fp64, b_fp64)

        # Force only contiguous choice to test the transform
        with (
            mock.patch(
                "torch._inductor.heuristics.template.contiguous_mm.use_contiguous"
            ) as contiguous_mock,
        ):
            contiguous_mock.return_value = True

            compiled_func = torch.compile(mm_transpose)
            out, code = run_and_get_code(compiled_func, a, b)

            # Verify correctness against fp64 baseline
            torch.testing.assert_close(
                out, expected_fp64.to(dtype), atol=1e-2, rtol=1e-2
            )

            # Check that contiguous transform was used
            FileCheck().check("contiguous_mm").run(code[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("sizes", ((64, 128, 256), (128, 256, 512), (256, 512, 1024)))
    @config.patch(
        max_autotune=True,
    )
    def test_max_autotune_contiguous_transform_addmm(self, sizes, dtype):
        # [FUNCTION DOC] ROCm-only: the addmm variant of the contiguous transform test. With the
        # contiguous check mocked true, compiles an addmm with a transposed second matrix,
        # compares with a float64 baseline, and requires the generated code mention the contiguous
        # addmm choice.
        """
        Test the contiguous subgraph transform for addmm with non-contiguous second matrix.
        """
        M, N, K = sizes

        def addmm_transpose(inp, a, b):
            # [FUNCTION DOC] HELPER: addmm of the input with a and the transpose of b.
            return torch.addmm(inp, a, b.transpose(0, 1))

        inp = torch.randn(M, N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        a = torch.randn(M, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)

        # Compute fp64 baseline
        inp_fp64 = inp.to(torch.float64)
        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        expected_fp64 = addmm_transpose(inp_fp64, a_fp64, b_fp64)

        # Force contiguous choice to test the transform
        with (
            mock.patch(
                "torch._inductor.heuristics.template.contiguous_mm.use_contiguous"
            ) as contiguous_mock,
        ):
            contiguous_mock.return_value = True

            compiled_func = torch.compile(addmm_transpose)
            out, code = run_and_get_code(compiled_func, inp, a, b)

            # Verify correctness against fp64 baseline
            torch.testing.assert_close(
                out, expected_fp64.to(dtype), atol=1e-2, rtol=1e-2
            )

            # Check that contiguous transform was used
            FileCheck().check("contiguous_addmm").run(code[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_contiguous_transform_non_contiguous_second_matrix(
        self, dynamic
    ):
        # [FUNCTION DOC] ROCm-only: with a contiguous second matrix the generated code must NOT
        # contain the contiguous mm choice (a FileCheck failure is expected and a test failure
        # raised otherwise); with a non-contiguous second matrix and the check mocked true it
        # must. Both outputs are compared against float64 baselines.
        """
        Test that contiguous transform is only applied when the second matrix is non-contiguous.
        """
        M, N, K = 64, 128, 64

        def mm(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return a @ b

        a = torch.randn(M, K, dtype=torch.float32, device=GPU_TYPE)
        b_contiguous = torch.randn(K, N, dtype=torch.float32, device=GPU_TYPE)
        b_non_contiguous = torch.randn(
            N, K, dtype=torch.float32, device=GPU_TYPE
        ).transpose(0, 1)

        # Compute fp64 baselines without max_autotune (since fp64 doesn't work with max_autotune=True)
        a_fp64 = a.to(torch.float64)
        b_contiguous_fp64 = b_contiguous.to(torch.float64)
        b_non_contiguous_fp64 = b_non_contiguous.to(torch.float64)

        expected1_fp64 = mm(a_fp64, b_contiguous_fp64)
        expected2_fp64 = mm(a_fp64, b_non_contiguous_fp64)

        with config.patch(
            max_autotune=True,
        ):
            # Test with contiguous second matrix - should not use contiguous transform
            compiled_func_contiguous = torch.compile(mm, dynamic=dynamic)
            out1, code1 = run_and_get_code(compiled_func_contiguous, a, b_contiguous)

            # Should not contain contiguous transform
            try:
                FileCheck().check("contiguous_mm").run(code1[0])
                self.fail(
                    "Contiguous transform should not be used for contiguous matrices"
                )
            except RuntimeError:
                pass  # Expected - contiguous transform should not be used

            # Test with non-contiguous second matrix - should use contiguous transform
            with (
                mock.patch(
                    "torch._inductor.heuristics.template.contiguous_mm.use_contiguous"
                ) as contiguous_mock,
            ):
                contiguous_mock.return_value = True

                compiled_func_non_contiguous = torch.compile(mm, dynamic=dynamic)
                out2, code2 = run_and_get_code(
                    compiled_func_non_contiguous, a, b_non_contiguous
                )

                # Should contain contiguous transform
                FileCheck().check("contiguous_mm").run(code2[0])

        # Verify correctness against fp64 baselines
        torch.testing.assert_close(
            out1, expected1_fp64.to(torch.float32), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            out2, expected2_fp64.to(torch.float32), atol=1e-2, rtol=1e-2
        )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_contiguous_transform_with_epilogue(self):
        # [FUNCTION DOC] ROCm-only: contiguous transform with a relu epilogue. With the
        # contiguous check mocked true, compiles a matmul against a transposed operand followed
        # by relu, compares with a float64 baseline and requires the contiguous mm choice in the
        # generated code.
        """
        Test contiguous transform with epilogue operations like relu.
        """
        M, N, K = 128, 256, 512

        def mm_transpose_relu(a, b):
            # [FUNCTION DOC] HELPER: matmul of a with the transpose of b followed by relu.
            return (a @ b.transpose(0, 1)).relu()

        a = torch.randn(M, K, dtype=torch.float32, device=GPU_TYPE)
        b = torch.randn(N, K, dtype=torch.float32, device=GPU_TYPE)

        # Compute fp64 baseline
        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        expected_fp64 = mm_transpose_relu(a_fp64, b_fp64)

        # Force contiguous transform
        with (
            mock.patch(
                "torch._inductor.heuristics.template.contiguous_mm.use_contiguous"
            ) as contiguous_mock,
        ):
            contiguous_mock.return_value = True

            compiled_func = torch.compile(mm_transpose_relu)
            out, code = run_and_get_code(compiled_func, a, b)

            # Verify correctness against fp64 baseline
            torch.testing.assert_close(
                out, expected_fp64.to(torch.float32), atol=1e-2, rtol=1e-2
            )

            # Check that contiguous transform was used
            FileCheck().check("contiguous_mm").run(code[0])

    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_override_template_heuristics_with_custom_class(self):
        # [FUNCTION DOC] Checks the template heuristic override mechanism with a custom class.
        # Builds IR buffers of a 64 by 64 mm under a graph lowering context, subclasses the
        # registered mm heuristic so that it yields exactly the first config of the original,
        # installs it with override_template_heuristics, and verifies the registry returns an
        # instance of the custom class (also of the base class) producing a single config equal to
        # the expected one.
        """
        Test that override_template_heuristics works with a custom heuristic class.
        Verifies that get_template_heuristic returns an instance of our custom class
        and that get_template_configs yields the expected configs.
        """
        from torch._inductor.heuristics.registry import (
            get_registered_heuristic_class,
            get_template_heuristic,
        )
        from torch._inductor.kernel.mm import MMKernelInputs

        template_uid = torch._inductor.kernel.mm.mm_template.uid

        # Get the base heuristic class that would normally be used for mm_template
        base_heuristic_class = get_registered_heuristic_class(
            template_uid, GPU_TYPE, "mm"
        )
        self.assertIsNotNone(base_heuristic_class)

        # Create a dummy graph for the GraphLowering context
        gm = make_fx(lambda: torch.zeros(2, 3))()
        graph = GraphLowering(gm)

        # The graph handler is needed to create IR nodes and use V.graph.sizevars
        with V.set_graph_handler(graph), config.patch({"max_autotune": True}):
            # Create IR Buffer nodes for testing (not actual tensors)
            mat1 = Buffer(
                name="mat1",
                layout=FixedLayout(
                    torch.device(GPU_TYPE),
                    dtype=torch.float32,
                    size=(64, 64),
                ),
            )
            mat2 = Buffer(
                name="mat2",
                layout=FixedLayout(
                    torch.device(GPU_TYPE),
                    dtype=torch.float32,
                    size=(64, 64),
                ),
            )
            kernel_inputs = MMKernelInputs([mat1, mat2])

            # Get the first config from the original heuristic to use as our expected config
            base_heuristic = base_heuristic_class()
            expected_config = next(
                iter(base_heuristic.get_template_configs(kernel_inputs, "mm"))
            )

            # Create a custom heuristic class that only yields the single expected config
            class CustomMMHeuristic(base_heuristic_class):
            # [CLASS DOC] HELPER: subclass of the registered base heuristic class that restricts
            # its template configs to the single captured expected config.
                def get_template_configs(self, kernel_inputs, op_name):
                # [FUNCTION DOC] HELPER: generator ignoring its arguments that yields only the
                # captured config.
                    yield expected_config

            with override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[(template_uid, "mm")],
                override_heuristic_class=CustomMMHeuristic,
            ):
                # Get the heuristic and verify it's our custom class
                heuristic = get_template_heuristic(template_uid, GPU_TYPE, "mm")
                self.assertIsInstance(heuristic, CustomMMHeuristic)
                self.assertIsInstance(heuristic, base_heuristic_class)

                # Verify get_template_configs yields only the expected config
                configs = list(heuristic.get_template_configs(kernel_inputs, "mm"))
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0], expected_config)

    @unittest.skipIf(config.cpp_wrapper, "out_dtype override not supported for AOTI")
    def test_bmm_out_dtype(self):
        # [FUNCTION DOC] Compiles a batched matmul with float32 out_dtype of float16 inputs using
        # the ATen backend (no max autotune) and checks the code calls the dtype-aware batched
        # matmul extern and the result matches the float32 reference. Skipped for the C++ wrapper.
        def f(a, b):
            # [FUNCTION DOC] HELPER: batched matmul with float32 output dtype.
            return torch.bmm(a, b, out_dtype=torch.float32)

        a = torch.randn(2, 3, 4, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2, 4, 5, device=GPU_TYPE, dtype=torch.float16)
        expected = torch.bmm(a.float(), b.float())
        with config.patch(
            max_autotune=False,
            max_autotune_gemm_backends="ATEN",
        ):
            compiled_f = torch.compile(f)
            out, code = run_and_get_code(compiled_f, a, b)
            FileCheck().check("extern_kernels.bmm_dtype").run(code[0])
            self.assertEqual(out, expected, atol=1e-3, rtol=1e-3)

    @unittest.skipIf(config.cpp_wrapper, "out_dtype override not supported for AOTI")
    def test_triton_bmm_out_dtype(self):
        # [FUNCTION DOC] Like the ATen out_dtype bmm test but with max autotune and the Triton-
        # only backend: out_dtype is passed as a call argument and the code must contain a
        # Triton fused bmm template kernel; result compared with the float32 reference. Skipped
        # for C++ wrapper.
        def f(a, b, out_dtype=torch.float32):
            # [FUNCTION DOC] HELPER: batched matmul with a configurable output dtype defaulting
            # to float32.
            return torch.bmm(a, b, out_dtype=out_dtype)

        a = torch.randn(2, 3, 4, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2, 4, 5, device=GPU_TYPE, dtype=torch.float16)
        expected = torch.bmm(a.float(), b.float())
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
        ):
            compiled_f = torch.compile(f)
            out, code = run_and_get_code(compiled_f, a, b, out_dtype=torch.float32)
            FileCheck().check("triton_tem_fused_bmm").run(code[0])
            self.assertEqual(out, expected, atol=1e-3, rtol=1e-3)

    def test_triton_template_generated_code_cache_key(self):
        # [FUNCTION DOC] Guard test on the template code cache key: counts parameters of the
        # template generate_and_load method and of the cache make_key method, requires make_key
        # to take exactly one fewer (the caching flag) and requires generate_and_load to have
        # exactly 21 parameters. Fails when a parameter is added, as a reminder to include it in
        # the cache key.
        generate_and_load_args = len(
            inspect.signature(
                torch._inductor.select_algorithm.TritonTemplate.generate_and_load
            ).parameters
        )
        make_key_args = len(
            inspect.signature(
                torch._inductor.select_algorithm.GeneratedCodeCache.make_key
            ).parameters
        )

        # Make sure all args of generate_and_load_args are passed to make_key_args (Except generate_with_caching)
        # update this function each time new arg added to generate_and_load and make sure arg is added to make_key
        self.assertEqual(generate_and_load_args - 1, make_key_args)
        self.assertEqual(generate_and_load_args, 21)

    @fresh_cache()
    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_cache_strategy(self):
        # [FUNCTION DOC] Verifies the self-check of the generated-code cache. The template kernel
        # method that supplies input-dependent preserved state is patched to always return the
        # same string (simulating a bad cache hit), and compiling two matmuls must then raise
        # InductorError whose message says the generated code cache results in wrong output.
        def func_test1(x, y, z, m):
            # [FUNCTION DOC] HELPER: returns two independent matrix products as a tuple.
            a = torch.matmul(x, y)
            b = torch.matmul(z, m)
            return a, b

        a = torch.rand(10, 22, device=GPU_TYPE)
        b = torch.rand(22, 30, device=GPU_TYPE)
        # Test that the testing strategy works by overriding input_dependent_preserved_state and simulate a cache hit.
        with unittest.mock.patch(
            "torch._inductor.select_algorithm.TritonTemplateKernel.input_dependent_preserved_state",
            new=(lambda self: "same always"),
        ):
            with self.assertRaisesRegex(
                torch._inductor.exc.InductorError,
                r".*Generated code cache results in wrong output.*",
            ):
                torch.compile(func_test1, dynamic=False)(a, b, a, b)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_caching(self):
        # [FUNCTION DOC] Checks hit/miss counters and cache keys of the Triton template generated-
        # code cache using the mm template with four configs. Static shapes give 4 hits and 4
        # misses and (off ROCm) a specific cache key and recorded events; dynamic shapes with
        # distinct symbols give 0 hits and 8 misses with symbolic key and size events; duck-typed
        # equal sizes give 4 and 4; a ten-step matmul loop gives 36 hits and 4 misses both static
        # and dynamic; a function with symbolic expressions from concatenation gives 0 hits and 7
        # misses. Each case uses a fresh cache. Note the symbolic-shape case asserts the same
        # events string twice with different formatting.
        def reset_counters():
            # [FUNCTION DOC] HELPER: clears all dynamo counters.
            torch._dynamo.utils.counters.clear()

        def hits():
            # [FUNCTION DOC] HELPER: returns the inductor generated module cache hit counter.
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_hit"
            ]

        def misses():
            # [FUNCTION DOC] HELPER: returns the inductor generated module cache miss counter.
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_miss"
            ]

        # remove white space from x.
        def remove_white_space(x: str) -> str:
            # [FUNCTION DOC] HELPER: returns the input string with all whitespace removed.
            return re.sub(r"\s+", "", x)

        def get_cache_key_and_events() -> tuple[str, str]:
            # [FUNCTION DOC] HELPER: takes the first key of the mm template generated code cache
            # and returns it together with the string form of the recorded events for that entry.
            cache = TritonTemplate.all_templates["mm"]._generated_code_cache._cache
            cache_key = next(iter(cache))
            events = str(cache[cache_key].events)
            return cache_key, events

        def func_test1(x, y, z, m):
            # [FUNCTION DOC] HELPER: returns two independent matrix products as a tuple.
            a = torch.matmul(x, y)
            b = torch.matmul(z, m)
            return a, b

        a = torch.rand(10, 22, device=GPU_TYPE)
        b = torch.rand(22, 30, device=GPU_TYPE)

        # Valid cache hit.
        with fresh_cache():
            reset_counters()
            compile_results = torch.compile(func_test1, dynamic=False)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)
            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

            cache_key, events = get_cache_key_and_events()

            if not TEST_WITH_ROCM:
                expected = """{
                        'input_nodes':[
                            "[[10,22],[22,1],torch.float32,device(type='cuda',index=0),0]",
                            "[[22,30],[30,1],torch.float32,device(type='cuda',index=0),0]"],
                        'num_stages':1,'num_warps':2,'prefix_args':0,'suffix_args':0,'call_sizes':[10,30],
                        'layout':"[[10,30],[30,1],torch.float32,device(type='cuda',index=0),0]",
                        'num_consumer_groups':0,'num_buffers_warp_spec':0,'epilogue_fn_hash':'identity','tma_store':False,
                        'tma_load_for_template_epilogue':False,'transpose_discontiguous_tensor_descriptors_override':None,
                        'kwargs':{'EVEN_K':False,'USE_FAST_ACCUM':False,'ACC_TYPE':'tl.float32',
                        'BLOCK_M':16,'BLOCK_N':32,'BLOCK_K':16,'GROUP_M':8,'ALLOW_TF32':False},
                        'hint_override':None,'triton_meta':None}"""

                expected = expected.replace("cuda", GPU_TYPE)
                self.assertExpectedInline(
                    remove_white_space(cache_key),
                    remove_white_space(expected),
                )

                self.assertEqual(
                    remove_white_space(events),
                    remove_white_space("""[('def_kernel', ['A', 'B'], {})]"""),
                )

        # Test symbolic shapes with different symbols. Will cache miss due to different symbols in inputs.
        with fresh_cache():
            a = torch.rand(10, 22, device=GPU_TYPE)
            b = torch.rand(22, 30, device=GPU_TYPE)

            c = torch.rand(9, 21, device=GPU_TYPE)
            d = torch.rand(21, 30, device=GPU_TYPE)
            reset_counters()
            compiled_results = torch.compile(func_test1, dynamic=True)(a, b, c, d)
            eager_results = func_test1(a, b, c, d)

            self.assertEqual(compiled_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 0)
            self.assertEqual(misses(), 8)

            cache_key, events = get_cache_key_and_events()

            if not TEST_WITH_ROCM:
                expected = """{
                    'input_nodes':[
                        "[[s77,s27],[s27,1],torch.float32,device(type='cuda',index=0),0]",
                        "[[s27,s94],[s94,1],torch.float32,device(type='cuda',index=0),0]"],
                    'num_stages':1,'num_warps':2,'prefix_args':0,'suffix_args':0,'call_sizes':[s77,s94],
                    'layout':"[[s77,s94],[s94,1],torch.float32,device(type='cuda',index=0),0]",'num_consumer_groups':0,
                    'num_buffers_warp_spec':0,'epilogue_fn_hash':'identity','tma_store':False,
                    'tma_load_for_template_epilogue':False,'transpose_discontiguous_tensor_descriptors_override':None,
                    'kwargs':{'EVEN_K':False,'USE_FAST_ACCUM':False,'ACC_TYPE':'tl.float32','BLOCK_M':16,'BLOCK_N':32,
                    'BLOCK_K':16,'GROUP_M':8,'ALLOW_TF32':False},'hint_override':None,'triton_meta':None}"""
                expected = expected.replace("cuda", GPU_TYPE)
                self.assertExpectedInline(
                    remove_white_space(cache_key),
                    remove_white_space(expected),
                )

                self.assertExpectedInline(
                    remove_white_space(events),
                    remove_white_space(
                        """[('def_kernel',['A','B'],{}),('size',['A',0],{}),('size',['B',1],{}),('size',['A',1],{})]"""
                    ),
                )
                self.assertExpectedInline(
                    remove_white_space(events),
                    remove_white_space(
                        """[
                            ('def_kernel', ['A', 'B'], {}),
                            ('size', ['A', 0], {}),
                            ('size', ['B', 1], {}),
                            ('size', ['A', 1], {})]
                        """
                    ),
                )

        # Test duck typing.
        with fresh_cache():
            reset_counters()

            compile_results = torch.compile(func_test1, dynamic=True)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

        # Test loop.
        def test_func2(x):
        # [FUNCTION DOC] HELPER: squares the input matrix ten times in a loop by repeated matmul.
            for _ in range(10):
                x = torch.matmul(x, x)
            return x

        with fresh_cache():
            reset_counters()
            input = torch.rand(10, 10, device=GPU_TYPE)

            compile_results = torch.compile(test_func2, dynamic=False)(input)
            eager_results = test_func2(input)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 36)
            self.assertEqual(misses(), 4)

        with fresh_cache():
            reset_counters()
            input = torch.rand(10, 10, device=GPU_TYPE)

            compile_results = torch.compile(test_func2, dynamic=True)(input)
            eager_results = test_func2(input)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 36)
            self.assertEqual(misses(), 4)

        # No cache hit due to symbolic expressions passed i.e mm(s0 + s1, 2) vs mm(s3, 2).
        reset_counters()

        def test_func3(x, y, z, m, l):
        # [FUNCTION DOC] HELPER: returns a matmul and a second matmul on concatenated operands,
        # which produce symbolic size expressions that cannot share cache entries.
            a = torch.matmul(x, y)
            b = torch.matmul(torch.cat([x, z], 1), torch.cat([y, m, l], 0))
            return a, b

        with fresh_cache():
            a = torch.rand(10, 22, device=GPU_TYPE)
            b = torch.rand(22, 30, device=GPU_TYPE)
            c = torch.rand(10, 11, device=GPU_TYPE)
            d = torch.rand(8, 30, device=GPU_TYPE)
            e = torch.rand(3, 30, device=GPU_TYPE)

            compile_results = torch.compile(test_func3, dynamic=True)(a, b, c, d, e)
            eager_results = test_func3(a, b, c, d, e)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 0)
            self.assertEqual(misses(), 7)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_caching_bmm(self):
        # [FUNCTION DOC] Checks the generated-code cache for the bmm template: two batched
        # matmuls with identical shapes under static compilation give 4 cache hits and 4 misses
        # and match eager.
        def func_test1(x, y, z, m):
            # [FUNCTION DOC] HELPER: returns two independent batched matrix products as a tuple.
            a = torch.bmm(x, y)
            b = torch.bmm(z, m)
            return a, b

        a = torch.rand(10, 10, 22, device=GPU_TYPE)
        b = torch.rand(10, 22, 30, device=GPU_TYPE)

        def hits():
            # [FUNCTION DOC] HELPER: returns the inductor generated module cache hit counter.
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_hit"
            ]

        def misses():
            # [FUNCTION DOC] HELPER: returns the inductor generated module cache miss counter.
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_miss"
            ]

        # Valid cache hit.
        with fresh_cache():
            torch._dynamo.utils.counters.clear()
            compile_results = torch.compile(func_test1, dynamic=False)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)
            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "ATEN, TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_caching_mm_plus_mm(self):
        # [FUNCTION DOC] Checks the generated-code cache with ATen and Triton backends for two
        # identical sums of matmul pairs under static compilation: 4 hits and 4 misses, results
        # match eager.
        def func_test1(x, y, z, m):
            # [FUNCTION DOC] HELPER: returns two identical sums of two matmuls as a tuple.
            a = torch.mm(x, y)
            b = torch.mm(z, m)
            sum1 = a + b

            c = torch.mm(x, y)
            d = torch.mm(z, m)
            sum2 = c + d
            return sum1, sum2

        a = torch.rand(10, 40, device=GPU_TYPE)
        b = torch.rand(40, 30, device=GPU_TYPE)

        def hits():
            # [FUNCTION DOC] HELPER: returns the inductor generated module cache hit counter.
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_hit"
            ]

        def misses():
            # [FUNCTION DOC] HELPER: returns the inductor generated module cache miss counter.
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_miss"
            ]

        # Valid cache hit.
        with fresh_cache():
            torch._dynamo.utils.counters.clear()
            compile_results = torch.compile(func_test1, dynamic=False)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)
            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

    @fresh_cache()
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        autotune_fallback_to_aten=False,
    )
    @parametrize("num_decompose_k_splits", (0, 5, 20))
    @parametrize("decompose_k_threshold", (8, 16))
    def test_max_autotune_decompose_k_envvars(
        self, num_decompose_k_splits, decompose_k_threshold
    ):
        # [FUNCTION DOC] Checks the decompose-K configuration options for splits and threshold.
        # For a large-K and a small-K shape, clears the cached split and eligibility helpers,
        # compiles a matmul and counts generated sources that mention the decompose-K benchmark.
        # Expects none when K over M or N is below the threshold or the split count is zero;
        # otherwise expects at least one and at most the configured number of splits.
        shapes = [(32, 32, 32768), (32, 32, 256)]
        for M, N, K in shapes:
            get_k_splits.cache_clear()
            use_decompose_k_choice.cache_clear()
            a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
            b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

            with config.patch(
                {
                    "triton.num_decompose_k_splits": num_decompose_k_splits,
                    "triton.decompose_k_threshold": decompose_k_threshold,
                }
            ):
                compiled_func = torch.compile(lambda a, b: a @ b)
                _, code = run_and_get_code(compiled_func, a, b)

                decompose_count = 0
                for codegen in code:
                    if "benchmark_decompose_k_mm" in codegen:
                        decompose_count += 1

                if (
                    K // M < decompose_k_threshold
                    or K // N < decompose_k_threshold
                    or num_decompose_k_splits == 0
                ):
                    self.assertEqual(decompose_count, 0)
                else:
                    self.assertTrue(decompose_count > 0)
                    self.assertTrue(decompose_count <= num_decompose_k_splits)

    @unittest.skipIf(
        config.triton.native_matmul,
        "native matmul takes different tuning configs",
    )
    @config.patch(max_autotune=True, max_autotune_gemm_search_space="EXHAUSTIVE")
    def test_max_autotune_exhaustive(self):
        # [FUNCTION DOC] Checks the exhaustive search space benchmarks only two choices. The
        # template heuristic lookup is mocked to return the device-appropriate mm heuristic whose
        # exhaustive config list is replaced by a single config; after compiling a 1024-cubed fp16
        # matmul every benchmark_gpu counter in the inductor counters must equal 2 (ATen plus the
        # one Triton config).
        def f(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return a @ b

        M, N, K = (1024, 1024, 1024)

        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE, requires_grad=True)

        with mock.patch(
            "torch._inductor.heuristics.registry.get_template_heuristic"
        ) as config_mock:
            # Create heuristic instance and modify it before setting as mock return value
            # On ROCm, use ROCmMMTemplateConfigHeuristic; on XPU use XPUMMTemplateConfigHeuristic;
            # otherwise use CUDAMMTemplateConfigHeuristic
            if GPU_TYPE == "xpu":
                config_heuristics = XPUMMTemplateConfigHeuristic()
            elif torch.version.hip:
                config_heuristics = ROCmMMTemplateConfigHeuristic()
            else:
                config_heuristics = CUDAMMTemplateConfigHeuristic()

            # Traditionally, this would be set of all possible configs
            # We mock out the code path for the sake of the unit test
            config_heuristics.exhaustive_configs = [
                GemmConfig(32, 32, 32, 1, 8, group_m=8)
            ]
            config_mock.return_value = config_heuristics

            from torch._dynamo.utils import counters

            compiled_func = torch.compile(f)
            compiled_func(a, b)

            # Only benchmarks 2 choices, aten and the exhaustive triton config
            # Counter can be InductorBenchmarker or TritonBenchmarker
            for counter in counters["inductor"]:
                if "benchmark_gpu" in counter:
                    self.assertEqual(counters["inductor"][counter], 2)

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_mm_k_1(self):
        # [FUNCTION DOC] Compiles a matmul with inner dimension one, for sizes 90 to 99, using the
        # Triton-only backend, resetting dynamo between sizes and comparing with eager.
        def mm(x, y):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return x @ y

        for i in range(90, 100):
            torch._dynamo.reset()
            a = torch.randn((i, 1), device=GPU_TYPE, dtype=torch.float32)
            b = torch.randn((1, i), device=GPU_TYPE, dtype=torch.float32)
            compiled_f = torch.compile(mm)

            out, code = run_and_get_code(compiled_f, a, b)
            torch.testing.assert_close(out, mm(a, b), atol=1e-2, rtol=1e-2)

    @parametrize("op", ("mm", "addmm", "bmm", "baddbmm", "mm_plus_mm"))
    @parametrize("max_autotune", (False, True))
    @config.patch(
        {
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "triton.native_matmul": False,
        }
    )
    def test_autotune_gemm_choice_validation(self, op, max_autotune):
        # [FUNCTION DOC] Registers a preprocessing hook recording the types of all autotune
        # choices seen while compiling mm, addmm, bmm, baddbmm or mm_plus_mm with ATen and Triton
        # backends. With max autotune the extern and Triton choice types must both be seen;
        # without it only the extern type. The hook is removed in a finally block.
        def generate_inputs_and_func(op_name):
            # [FUNCTION DOC] HELPER: builds example inputs and the callable for the named op,
            # using extra bias for addmm and baddbmm, batch dims for bmm ops and a second pair for
            # mm_plus_mm. Raises ValueError for unsupported op names.
            # Base config with just x and w
            base_inputs = [
                torch.randn(128, 256, device=GPU_TYPE),
                torch.randn(256, 128, device=GPU_TYPE),
            ]
            func = torch.mm
            if op_name == "mm":
                # default
                pass
            elif op_name == "addmm":
                # Add bias for addmm
                base_inputs = [torch.randn(128, device=GPU_TYPE)] + base_inputs
                func = torch.addmm
            elif op_name in ["bmm", "baddbmm"]:
                # Override for batch dimensions
                base_inputs[0] = torch.randn(4, 128, 256, device=GPU_TYPE)
                base_inputs[1] = torch.randn(4, 256, 128, device=GPU_TYPE)
                func = torch.bmm
                if op_name == "baddbmm":
                    # Add batch bias
                    base_inputs = [
                        torch.torch.randn(4, 128, 128, device=GPU_TYPE)
                    ] + base_inputs
                    func = torch.baddbmm
            elif op_name == "mm_plus_mm":
                # Add second matrix pair
                base_inputs += [
                    torch.randn(128, 256, device=GPU_TYPE),
                    torch.randn(256, 128, device=GPU_TYPE),
                ]

                def mmpmm(x, w, x2, w2):
                # [FUNCTION DOC] HELPER: sum of two matrix products.
                    return torch.mm(x, w) + torch.mm(x2, w2)

                func = mmpmm
            else:
                raise ValueError(f"Unsupported op: {op_name}")
            return base_inputs, func

        choice_types_seen = set()

        def choice_validator(choices):
            # [FUNCTION DOC] HELPER: preprocessing hook which records the type of each choice into
            # the enclosing set and returns the choices unchanged.
            for choice in choices:
                choice_types_seen.add(type(choice))
            return choices

        inputs, fn = generate_inputs_and_func(op)

        add_preprocessing_fn(choice_validator)
        try:
            with config.patch({"max_autotune": max_autotune}):
                compiled_fn = torch.compile(fn, dynamic=False)
                compiled_fn(*inputs)

                if max_autotune:
                    self.assertIn(ExternKernelCaller, choice_types_seen)
                    self.assertIn(TritonTemplateCaller, choice_types_seen)
                else:
                    self.assertIn(ExternKernelCaller, choice_types_seen)
                    self.assertNotIn(TritonTemplateCaller, choice_types_seen)
        finally:
            clear_preprocessing_fns()

    @config.patch(
        {"test_configs.max_mm_configs": 4, "max_autotune_gemm_backends": "ATEN,TRITON"}
    )
    @parametrize("max_autotune_enabled", (True, False))
    def test_autotune_layout_optimization(self, max_autotune_enabled):
        # [FUNCTION DOC] Checks layout flexibility of addmm choices. A preprocessing hook asserts
        # that every choice has a fixed layout when max autotune is on and a flexible layout when
        # off (all choices extern kernels). The hook is registered for the test and the non-
        # default hooks are cleared in a finally block.
        """Test that layouts are flexible when every choice is ExternKernelChoice"""

        # we use a proxy here of bias_addmm and max-autotune because this enables us to see
        # multiple choices in both scenarios (bias_addmm, addmm, triton (max-autotune only))
        # and both bias_addmm and addmm are extern kernel choices
        def layout_checker(choices):
            # [FUNCTION DOC] HELPER: preprocessing hook; for each non-empty list of choices
            # asserts the layout class matches the enclosing max autotune setting, and returns
            # the choices.
            if choices:
                expected_layout = (
                    FixedLayout if max_autotune_enabled else FlexibleLayout
                )
                for choice in choices:
                    self.assertIsInstance(
                        choice.layout,
                        expected_layout,
                        lambda msg: f"{msg}\nExpected {expected_layout.__name__} with max_autotune={max_autotune_enabled}",
                    )
            return choices

        add_preprocessing_fn(layout_checker)

        try:
            bias = torch.randn(64, device=GPU_TYPE)
            x = torch.randn(32, 128, device=GPU_TYPE)
            w = torch.randn(128, 64, device=GPU_TYPE)

            with config.patch({"max_autotune": max_autotune_enabled}):
                compiled_fn = torch.compile(lambda b, x, w: torch.addmm(b, x, w))
                _ = compiled_fn(bias, x, w)
        finally:
            clear_preprocessing_fns(clear_defaults=False)

    @config.patch(
        {"test_configs.max_mm_configs": 4, "max_autotune_gemm_backends": "TRITON"}
    )
    def test_fixed_layout_at_lowering(self):
        # [FUNCTION DOC] Checks padding and output strides for mm, addmm, bmm and mm_plus_mm with
        # a transposed bfloat16 first operand and a float32 second operand of width 1490 that is
        # cast inside the function. In max-autotune-no-cudagraphs mode the generated code must
        # show the second matrix stride padded to 1536.
        """
        Test that max-autotune with addmm/bmm/mm_plus_mm correctly handles
        padding and maintains correct output strides. Specifically, when matrix
        b with shape (4608, 1490) is padded, its stride should become 1536.
        """

        def mm_func(a, b) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: permutes a, casts both operands to bfloat16, matmuls and
            # casts to float32.
            a_t = torch.permute(a, [1, 0]).to(torch.bfloat16)
            b_dtype = b.to(torch.bfloat16)
            # Add .to() to make sure that mm could be potentially padded
            # Strides for output are not padded
            return (a_t @ b_dtype).to(torch.float32)

        def addmm_func(a, b, bias) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: like the mm variant but using addmm with a bfloat16 bias.
            a_t = torch.permute(a, [1, 0]).to(torch.bfloat16)
            b_dtype = b.to(torch.bfloat16)
            bias_dtype = bias.to(torch.bfloat16)
            return torch.addmm(bias_dtype, a_t, b_dtype).to(torch.float32)

        def bmm_func(a, b) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: permutes a to batch-first, casts operands to bfloat16, bmm,
            # cast to float32.
            a_t = torch.permute(a, [2, 0, 1]).to(torch.bfloat16)
            b_dtype = b.to(torch.bfloat16)
            return torch.bmm(a_t, b_dtype).to(torch.float32)

        def mm_plus_mm_func(a1, b1, a2, b2) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: sum of two such transposed bfloat16 matmuls, cast to float32.
            a1_t = torch.permute(a1, [1, 0]).to(torch.bfloat16)
            b1_dtype = b1.to(torch.bfloat16)
            a2_t = torch.permute(a2, [1, 0]).to(torch.bfloat16)
            b2_dtype = b2.to(torch.bfloat16)
            return (a1_t @ b1_dtype + a2_t @ b2_dtype).to(torch.float32)

        a = torch.randn((4608, 512), device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn((4608, 1490), device=GPU_TYPE)
        bias = torch.randn(1490, device=GPU_TYPE)

        a_bmm = torch.randn((512, 4608, 8), device=GPU_TYPE, dtype=torch.bfloat16)
        b_bmm = torch.randn((8, 4608, 1490), device=GPU_TYPE)

        # Test mm_plus_mm
        a2 = torch.randn((4608, 512), device=GPU_TYPE, dtype=torch.bfloat16)
        b2 = torch.randn((4608, 1490), device=GPU_TYPE)

        # 1490 padded to 1536, check in template code
        output_code_padding_check = "stride_bk = 1536"
        funcs_and_args = [
            (mm_func, (a, b)),
            (addmm_func, (a, b, bias)),
            (bmm_func, (a_bmm, b_bmm)),
            (mm_plus_mm_func, (a, b, a2, b2)),
        ]

        for f, args in funcs_and_args:
            c_f = torch.compile(f, mode="max-autotune-no-cudagraphs")
            _, code_out = run_and_get_code(c_f, *args)
            FileCheck().check(output_code_padding_check).run(code_out[0])

    @parametrize("k", (15, 16))
    @parametrize("dynamic", (False, True))
    def test_even_k(self, k: int, dynamic: bool):
        # [FUNCTION DOC] Compiles torch.mm for a 21 by k times k by 31 float16 problem with k of
        # 15 or 16, static or dynamic, under max autotune and a fresh cache. Then inspects the
        # first key of the mm template cache and checks that it records the even-K flag as true
        # exactly when k is 16 and the shapes are static.
        M, N = 21, 31
        a = torch.randn((M, k), dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn((k, N), dtype=torch.float16, device=GPU_TYPE)

        if dynamic:
            torch._dynamo.mark_dynamic(a, 1)
            torch._dynamo.mark_dynamic(b, 0)

        with config.patch({"max_autotune": True}), fresh_cache():
            _ = torch.compile(torch.mm)(a, b)
            cache = TritonTemplate.all_templates["mm"]._generated_code_cache._cache
            cache_key = next(iter(cache))

        self.assertObjectIn(k, (15, 16))
        self.assertEqual("'EVEN_K': True" in cache_key, k == 16 and not dynamic)

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "force_pointwise_cat": True,
        }
    )
    @parametrize("epilogue", (True, False))
    def test_deferred_layout_constraint_cat_fusion(self, epilogue):
        # [FUNCTION DOC] Checks that with pointwise cat forced, a concatenation feeding a bmm,
        # whose layout constraint conflicts with padding after fusion, falls back to the ATen bmm
        # even though Triton is mocked 10x faster. The generated code must contain no Triton
        # template kernel. Parametrized on whether an extra float32 epilogue output exists.
        def mm_with_cat(a, b1, b2, d):
            # [FUNCTION DOC] HELPER: concatenates b1 and b2, adds one, casts to bfloat16, runs bmm
            # with a (cast) and returns it (with a float32 copy when the enclosing epilogue flag
            # is set) plus a fused subtraction of d.
            catted_b = torch.cat([b1, b2], dim=1)
            catted_b_add = catted_b + 1.0

            # Not padded at lowering time
            catted_bf16 = catted_b_add.to(torch.bfloat16)

            # bmm1 as output -> no padding occurs, however after fusion
            # the buffer is padded, layout constraint violated
            bmm1 = torch.bmm(a.to(torch.bfloat16), catted_bf16)
            fused = catted_bf16 - d
            if epilogue:
                bmm1_fp32 = bmm1.to(torch.float32)
                return bmm1, bmm1_fp32, fused

            return bmm1, fused

        batch = 32
        m = 512
        k1, k2 = 256, 256
        n = 1490  # Would normally trigger padding (1490 -> 1536)

        a = torch.randn(batch, m, k1 + k2, device=GPU_TYPE)
        b1 = torch.randn(batch, k1, n, device=GPU_TYPE)
        b2 = torch.randn(batch, k2, n, device=GPU_TYPE)
        d = torch.cat([b1, b2], dim=1).to(torch.bfloat16)

        with (
            fresh_cache(),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_choice",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
        ):
            compiled = torch.compile(mm_with_cat, mode="max-autotune-no-cudagraphs")
            _, code = run_and_get_code(compiled, a, b1, b2, d)

            # Despite Triton being 10x faster in benchmarks, the layout conflict
            # should force fallback to extern_kernels.bmm
            FileCheck().check_not("triton_tem").run(code[0])

    @parametrize("always_freeze", [True, False])
    def test_mm_layout_freezing_behavior(self, always_freeze):
        # [FUNCTION DOC] Checks the always-freeze-layout flag of the mm template. The kernel
        # method that reads a stride and may freeze the layout is wrapped to assert the layout is
        # fixed (flag on) or still flexible (flag off) after the call, and to record that a
        # flexible layout was seen. The flag is set on the template and restored in a finally
        # block; the test requires a flexible layout was seen. Raises AssertionError inside the
        # wrapper on violation.
        """Test that mm layout freezing behavior depends on always_freeze_layout.

        When always_freeze_layout=True, FlexibleLayout should be frozen to FixedLayout.
        When always_freeze_layout=False (default), FlexibleLayout should remain flexible
        and use layout constraints instead.
        """
        from torch._inductor import ir
        from torch._inductor.kernel.mm import mm_template
        from torch._inductor.select_algorithm import TritonTemplateKernel

        flexible_layout_called = False
        orig_stride_call = TritonTemplateKernel.get_stride_and_maybe_freeze_layout

        def tracking_get_stride(self, node):
            # [FUNCTION DOC] HELPER: replacement for the stride lookup; records whether the node
            # layout was flexible before, calls the original, and if it was flexible validates the
            # resulting layout class against the always_freeze parameter (raising AssertionError),
            # then returns the result.
            nonlocal flexible_layout_called
            flexible_layout = isinstance(node.data.layout, ir.FlexibleLayout)
            result = orig_stride_call(self, node)
            if flexible_layout:
                flexible_layout_called = True
                if always_freeze:
                    if not isinstance(node.data.layout, ir.FixedLayout):
                        raise AssertionError(
                            f"Expected FixedLayout, got {type(node.data.layout)}"
                        )
                else:
                    if not isinstance(node.data.layout, ir.FlexibleLayout):
                        raise AssertionError(
                            f"Expected FlexibleLayout, got {type(node.data.layout)}"
                        )
            return result

        def fn(x, y):
            # [FUNCTION DOC] HELPER: adds one to x to create a flexible layout buffer, then
            # matmuls with y.
            # Add intermediate computation to create FlexibleLayout buffer
            x = x + 1
            return torch.mm(x, y)

        M, K, N = 256, 128, 256
        x = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)

        # Save original value to restore later
        original_always_freeze = mm_template.always_freeze_layout

        with (
            fresh_cache(),
            patch.object(
                TritonTemplateKernel,
                "get_stride_and_maybe_freeze_layout",
                tracking_get_stride,
            ),
            config.patch({"test_configs.max_mm_configs": 1}),
        ):
            torch._dynamo.reset()
            mm_template.always_freeze_layout = always_freeze
            try:
                # Temporarily set always_freeze_layout on mm_template
                mm_template.always_freeze_layout = always_freeze
                compiled_f = torch.compile(fn, mode="max-autotune")
                compiled_f(x, y)
            finally:
                # Restore original value
                mm_template.always_freeze_layout = original_always_freeze

        self.assertTrue(flexible_layout_called)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 1,
        }
    )
    def test_deffered_layout_constraint_reintepret(self):
        # [FUNCTION DOC] Regression test (formerly a CUDA illegal memory access): two bmms consume
        # a padded-stride input, one directly after a pointwise op and one through a permute. With
        # Triton mocked faster and one config, the compile must succeed and produce a Triton
        # template kernel.
        batch, m, k, n = 4608, 40, 112, 1119

        # Shape: batch x m x k (contiguous)
        a = torch.randn(batch, m, k, dtype=torch.bfloat16, device=GPU_TYPE)

        padded_batch_stride = k * n + 48
        b = torch.empty_strided(
            size=(batch, k, n),
            stride=(padded_batch_stride, 1, k),
            dtype=torch.bfloat16,
            device=GPU_TYPE,
        )
        b.copy_(torch.randn_like(b))
        c = torch.randn(batch, k, m, dtype=torch.bfloat16, device=GPU_TYPE)

        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: bmm of a with b plus zero, b plus one, and bmm of the
            # permuted b-plus-zero with c; the bmm outputs are cast to float32.
            # Apply a pointwise op to b to make it FlexibleLayout in Inductor
            # This ensures Inductor doesn't treat it as a fixed/external layout
            # Ends up double padding
            b_flex = b + 0
            return (
                torch.bmm(a, b_flex).to(torch.float32),
                b + 1.0,
                torch.bmm(b_flex.permute(0, 2, 1), c).to(torch.float32),
            )

        with (
            mock.patch(
                "torch._inductor.autotune_process.run_autotune_in_subprocess",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_choice",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
        ):
            compiled_fn = torch.compile(fn)

            # Previously would CUDA IMA
            _, code = run_and_get_code(compiled_fn, a, b, c)

            FileCheck().check("triton_tem_fused").run(code[0])

    @config.patch(
        {
            "max_autotune": True,
        }
    )
    def test_deffered_layout_constraint_mm_cat_mm(self):
        # [FUNCTION DOC] Compiles a matmul / cat / matmul graph sharing one pointwise result,
        # with max autotune on, to check that deferred layout constraints do not crash; no
        # output assertions.
        def mm_cat_mm(t1, t2, t3, t4):
            # [FUNCTION DOC] HELPER: adds one to t2, then returns matmul with t1, a concatenation
            # with t3 and a matmul with t4.
            add = t2 + 1
            matmul1 = torch.matmul(t1, add)
            cat = torch.cat((t3, add), dim=1)
            matmul2 = torch.matmul(t4, add)
            return matmul1, cat, matmul2

        B = 3072
        M = 128
        K = 144
        N = 160
        P = 96
        t1 = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float16)
        t2 = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float16)
        t3 = torch.randn(B, P, N, device=GPU_TYPE, dtype=torch.float16)
        t4 = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float16)

        args = (t1, t2, t3, t4)

        compiled_fn = torch.compile(mm_cat_mm)

        run_and_get_code(compiled_fn, *args)

    @fresh_cache()
    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "Test requires CUDA")
    @unittest.skipIf(
        not SM90OrLater, "Requires SM90+ (H100/B200) for sufficient GPU memory"
    )
    @largeTensorTest("10 GB", device=GPU_TYPE)
    def test_max_autotune_mm_large_input_tensor_int64_indexing(self):
        # [FUNCTION DOC] Regression test (issue 171389): matmul with a second operand over 2^31
        # elements must index with 64-bit arange values. Asserts the size precondition, compiles
        # with the Triton-only backend restricted to the mm template choices and compares with
        # eager. Requires CUDA SM90+ and 10 GB; skipped on XPU and ROCm.
        """
        Test mm with input tensor exceeding 2^31 elements.
        Regression test for https://github.com/pytorch/pytorch/issues/171389
        When input tensor storage exceeds 2^31 elements, tl.arange() must be
        cast to INDEX_DTYPE (int64) to avoid integer overflow in pointer arithmetic.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return torch.mm(a, b)

        M, K, N = 1280, 65536, 65536
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)

        self.assertTrue(
            b.numel() > 2**31 - 1,
            lambda msg: f"{msg}\nTest requires tensor with >2^31 elements, got {b.numel()}",
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "test_configs.autotune_choice_name_regex": r"^triton_mm_",
            }
        ):
            result = torch.compile(mm)(a, b)

        torch.testing.assert_close(result, torch.mm(a, b), rtol=1e-2, atol=1e-2)

    @fresh_cache()
    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "Test requires CUDA")
    @largeTensorTest("6 GB", device=GPU_TYPE)
    def test_max_autotune_mm_large_storage_offset_i64_indexing(self):
        # [FUNCTION DOC] Regression test: matmul of a dynamic-shape input sliced with an offset
        # above the 32-bit range must pass the symbolic size argument as 64-bit. Asserts the
        # offset precondition, marks dim 0 dynamic, compiles with Triton-only mm templates and
        # compares with the eager result. Needs 6 GB; skipped on XPU and ROCm.
        """
        Test mm with input having dynamic storage offset exceeding i32 range.
        When a dynamic-shaped input is a slice with offset proportional to
        the dynamic dim (e.g., offset=70000*s6), the ks parameter in the
        triton template signature must use i64 to avoid overflow in pointer
        arithmetic like `A = arg_A + 70000*ks0`.
        """

        def mm(x, w):
            # [FUNCTION DOC] HELPER: slices the last eighth of rows from x using the dynamic batch
            # size and matmuls it with w.
            batch = x.shape[0] // 8
            a = x[7 * batch :]
            return torch.mm(a, w)

        K, N = 10000, 32
        batch = 32768
        x = torch.randn(8 * batch, K, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)

        expected_offset = 7 * batch * K
        self.assertTrue(
            expected_offset > 2**31 - 1,
            lambda msg: f"{msg}\nTest requires offset > i32_max, got {expected_offset}",
        )

        torch._dynamo.mark_dynamic(x, 0)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "test_configs.autotune_choice_name_regex": r"^triton_mm_",
            }
        ):
            result = torch.compile(mm)(x, w)

        a = x[7 * batch :]
        torch.testing.assert_close(result, torch.mm(a, w), rtol=1e-2, atol=1e-2)

    @fresh_cache()
    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "Test requires CUDA")
    @unittest.skipIf(
        not SM90OrLater, "Requires SM90+ (H100/B200) for sufficient GPU memory"
    )
    @largeTensorTest("10 GB", device=GPU_TYPE)
    def test_max_autotune_mm_large_output_tensor_int32_overflow(self):
        # [FUNCTION DOC] Regression test (issue 171389): matmul whose output has at least 2^32
        # elements must not take the zero-size early exit due to 32-bit overflow. Asserts the size
        # precondition, compiles with Triton-only mm templates and compares with eager. Needs
        # SM90+ and 10 GB.
        """
        Test mm with output tensor exceeding 2^32 elements.
        Regression test for https://github.com/pytorch/pytorch/issues/171389
        When M * N >= 2^32, the early exit check `if M * N == 0` can overflow
        to 0 in int32 arithmetic, causing the kernel to return immediately
        with all-zero output.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return torch.mm(a, b)

        M, K, N = 65536, 32, 65536
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)

        self.assertTrue(
            M * N >= 2**32,
            lambda msg: f"{msg}\nTest requires M*N >= 2^32 for overflow, got {M * N}",
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "test_configs.autotune_choice_name_regex": r"^triton_mm_",
            }
        ):
            result = torch.compile(mm)(a, b)

        torch.testing.assert_close(result, torch.mm(a, b), rtol=1e-2, atol=1e-2)

    @fresh_cache()
    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "Test requires CUDA")
    @unittest.skipIf(
        not SM90OrLater, "Requires SM90+ (H100/B200) for sufficient GPU memory"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(),
        "Hopper-style mm_persistent_tma template is shadowed by the Blackwell warp-specialized TMA template on data-center Blackwell.",
    )
    @largeTensorTest("10 GB", device=GPU_TYPE)
    def test_max_autotune_mm_persistent_tma_large_input_tensor_int64_indexing(self):
        # [FUNCTION DOC] Regression test (issue 171389): persistent TMA matmul with a second
        # operand over 2^31 elements. Asserts the size precondition, compiles with the Triton-only
        # backend and the persistent TMA choice forced, and compares with eager. Needs SM90+ and
        # 10 GB; skipped on XPU, ROCm and datacenter Blackwell.
        """
        Test persistent TMA mm with input tensor exceeding 2^31 elements.
        Regression test for https://github.com/pytorch/pytorch/issues/171389.
        Triton TMA descriptors require 32-bit block offsets even when the
        surrounding kernel uses int64 indexing.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return torch.mm(a, b)

        M, K, N = 1280, 65536, 65536
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)

        self.assertTrue(
            b.numel() > 2**31 - 1,
            lambda msg: f"{msg}\nTest requires tensor with >2^31 elements, got {b.numel()}",
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            result = torch.compile(mm)(a, b)

        torch.testing.assert_close(result, torch.mm(a, b), rtol=1e-2, atol=1e-2)

    @fresh_cache()
    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 1,
        }
    )
    def test_deferred_layout_constraint_reinterpret_3d(self):
        # [FUNCTION DOC] Regression test for deferred layout constraints through a rank-2 to
        # rank-3 view: the function adds zero to a padded-stride 2D input, applies an index_put to
        # force realization, views the result as 3D and uses it in two bmms and a pointwise add.
        # Benchmarking is mocked to favor Triton with one config; the compile must succeed and
        # yield a Triton template.
        batch, m, k, n = 32, 40, 1053, 40
        batch_stride = 42176

        a = torch.randn(batch, m, k, dtype=torch.bfloat16, device=GPU_TYPE)
        b = torch.empty_strided(
            size=(batch, k * n),
            stride=(batch_stride, 1),
            dtype=torch.bfloat16,
            device=GPU_TYPE,
        )
        b.copy_(torch.randn_like(b))
        c = torch.randn(batch, n, m, dtype=torch.bfloat16, device=GPU_TYPE)
        idx0 = torch.tensor([0], device=GPU_TYPE)
        idx1 = torch.tensor([0], device=GPU_TYPE)
        value = torch.zeros(1, dtype=torch.bfloat16, device=GPU_TYPE)

        def fn(a, b, c, idx0, idx1, value):
            # [FUNCTION DOC] HELPER: builds a flexible-layout copy of b, scatters value at the
            # index pair, views it as 3D and returns two float32 bmm results and the 2D buffer
            # plus one.
            # Mirror the 2D regression first: a simple pointwise op gives us a
            # rank-2 FlexibleLayout. Then force that 2D value to realize before
            # reinterpreting it as the rank-3 view from the original repro.
            b_flex_2d = b + 0
            b_flex_2d = aten.index_put.default(b_flex_2d, (idx0, idx1), value, False)
            b_view_3d = b_flex_2d.view(batch, k, n)
            return (
                torch.bmm(a, b_view_3d).to(torch.float32),
                b_flex_2d + 1.0,
                torch.bmm(b_view_3d, c).to(torch.float32),
            )

        with (
            mock.patch(
                "torch._inductor.autotune_process.run_autotune_in_subprocess",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_choice",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
        ):
            compiled_fn = torch.compile(fn)
            _, code = run_and_get_code(compiled_fn, a, b, c, idx0, idx1, value)
            FileCheck().check("triton_tem_fused").run(code[0])

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/182093")
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("use_addmm", (False, True))
    def test_triton_gemm_epilogue_fusion_truncates_accumulator(self, dtype, use_addmm):
        # [FUNCTION DOC] Verifies that Triton GEMM epilogue fusion truncates the float32
        # accumulator to the output dtype before epilogue ops for fp16/bf16 (down cast followed by
        # up cast; for addmm truncation must follow the bias add, not apply to the accumulator
        # directly) and applies no truncation for float32. Also compiles with epilogue fusion off
        # and requires the fused kernel name be absent there and the outputs equal (loose
        # tolerance on XPU fp16).
        """
        Verify that Triton GEMM epilogue fusion properly truncates the fp32
        accumulator to the output dtype before performing epilogue operations.

        When epilogue fusion is enabled and acc_dtype (fp32) differs from
        output_dtype (fp16/bf16), the generated code should:
        1. Downcast from fp32 to output dtype (truncation)
        2. Upcast back to fp32 for epilogue computation

        For float32, acc_dtype equals output_dtype so no truncation is needed.

        For addmm, verify that the bias addition happens in full precision
        BEFORE truncation (truncation should NOT be on acc directly).
        """
        if use_addmm:

            def fn(x, bias):
                # [FUNCTION DOC] HELPER: addmm of bias with x times x, then relu, minus one.
                return torch.addmm(bias, x, x).relu() - 1.0
        else:

            def fn(x):
                # [FUNCTION DOC] HELPER: x times x, then relu, minus one.
                return (x @ x).relu() - 1.0

        x = torch.randn(128, 128, dtype=dtype, device=GPU_TYPE)
        bias = torch.randn(128, dtype=dtype, device=GPU_TYPE) if use_addmm else None

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
                "epilogue_fusion": True,
                "benchmark_epilogue_fusion": False,
            }
        ):
            if use_addmm:
                out, code = run_and_get_code(torch.compile(fn), x, bias)
                expected = fn(x, bias)
            else:
                out, code = run_and_get_code(torch.compile(fn), x)
                expected = fn(x)

        tol = 5e-2 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out, expected, atol=tol, rtol=tol)

        kernel_name = (
            "triton_tem_fused_addmm_relu_sub"
            if use_addmm
            else "triton_tem_fused_mm_relu_sub"
        )
        FileCheck().check(kernel_name).run(code[0])

        # Verify the epilogue has the downcast (fp32 -> fp16/bf16) followed
        # by upcast (fp16/bf16 -> fp32) pattern for proper truncation.
        # For addmm, truncation should happen AFTER bias add, not on acc directly.
        triton_dtype = "tl.float16" if dtype == torch.float16 else "tl.bfloat16"
        if dtype in (torch.float16, torch.bfloat16):
            if use_addmm:
                # For addmm: truncation should happen on result of bias add,
                # NOT on acc directly (which would mean truncation before bias add)
                self.assertNotIn(f"acc.to({triton_dtype})", code[0])
                FileCheck().check(f".to({triton_dtype})").check(".to(tl.float32)").run(
                    code[0]
                )
            else:
                FileCheck().check(f"acc.to({triton_dtype})").check(
                    ".to(tl.float32)"
                ).run(code[0])
        else:
            # float32: no truncation casts since acc_dtype == output_dtype
            self.assertNotIn("acc.to(tl.float16)", code[0])
            self.assertNotIn("acc.to(tl.bfloat16)", code[0])

        # Verify that fused epilogue produces bitwise identical results to unfused
        torch._dynamo.reset()
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
                "epilogue_fusion": False,
                "benchmark_epilogue_fusion": False,
            }
        ):
            if use_addmm:
                out_unfused, code_unfused = run_and_get_code(torch.compile(fn), x, bias)
            else:
                out_unfused, code_unfused = run_and_get_code(torch.compile(fn), x)

        FileCheck().check_not(kernel_name).run(code_unfused[0])
        if GPU_TYPE == "xpu" and dtype == torch.float16:
            torch.testing.assert_close(out, out_unfused, atol=5e-4, rtol=5e-4)
        else:
            self.assertEqual(out, out_unfused)

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("use_addmm", (False, True))
    def test_triton_gemm_no_epilogue_no_truncation_casts(self, dtype, use_addmm):
        # [FUNCTION DOC] Verifies that a Triton GEMM with no fused epilogue ops (mm or addmm;
        # fp16, bf16, fp32) compiles to a Triton template kernel and matches eager, and that its
        # code contains no accumulator down-cast to fp16 or bf16.
        """
        Verify that Triton GEMM without epilogue fusion does not have
        truncation casts in the epilogue.

        When there are no fused epilogue operations, the accumulator should
        be stored directly without explicit truncation/upcast pattern.
        """
        if use_addmm:

            def fn(x, bias):
                # [FUNCTION DOC] HELPER: addmm of bias with x times x.
                return torch.addmm(bias, x, x)

        else:

            def fn(x):
                # [FUNCTION DOC] HELPER: matrix product of x with itself.
                return x @ x

        x = torch.randn(128, 128, dtype=dtype, device=GPU_TYPE)
        bias = torch.randn(128, dtype=dtype, device=GPU_TYPE) if use_addmm else None

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
                "epilogue_fusion": True,
                "benchmark_epilogue_fusion": False,
            }
        ):
            if use_addmm:
                out, code = run_and_get_code(torch.compile(fn), x, bias)
                expected = fn(x, bias)
            else:
                out, code = run_and_get_code(torch.compile(fn), x)
                expected = fn(x)

        # float32 needs looser tolerance due to TF32 precision differences
        tol = 5e-2 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(out, expected, atol=tol, rtol=tol)

        # Verify we got a Triton template
        kernel_name = "triton_tem_fused_addmm" if use_addmm else "triton_tem_fused_mm"
        FileCheck().check(kernel_name).run(code[0])

        # No truncation casts should exist since there are no fused epilogue ops
        self.assertNotIn("acc.to(tl.float16)", code[0])
        self.assertNotIn("acc.to(tl.bfloat16)", code[0])


@instantiate_parametrized_tests
class TestTemplateConfigPruning(TestCase):
    # [CLASS DOC] Tests for pruning GEMM configs by estimated shared memory. Class-level state
    # holds CUDA addmm and mm heuristics (plain and persistent TMA, Blackwell variants on
    # datacenter Blackwell) and a list of candidate GEMM configs built from block sizes, with sums
    # of block dims in an intermediate range. setUp saves each heuristic's config list and
    # tearDown restores them, because tests overwrite them.
    """Test class for pruning logic in GEMM autotuning."""

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: creates the shared heuristic objects and the candidate
        # config list (all combinations of block sizes 64, 128, 256 and two stage counts, keeping
        # those whose block size sum is above 192 and below 512). Uses Blackwell config class and
        # extra options (epilogue subtile, warp specialization, flatten) on datacenter Blackwell.
        super().setUpClass()
        # Initialize heuristics once for all tests
        cls.addmm_heuristic = CUDAAddMMTemplateConfigHeuristic()
        cls.mm_heuristic = CUDAMMTemplateConfigHeuristic()

        tma_addmm_heuristic_cls = CUDAAddmmPersistentTMATemplateConfigHeuristic
        tma_mm_heuristic_cls = CUDAPersistentTMATemplateConfigHeuristic
        if has_datacenter_blackwell_tma_device():
            tma_addmm_heuristic_cls = (
                CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic
            )
            tma_mm_heuristic_cls = CUDABlackwellPersistentTMATemplateConfigHeuristic
        cls.addmm_tma_heuristic = tma_addmm_heuristic_cls()
        cls.mm_tma_heuristic = tma_mm_heuristic_cls()

        block_sizes = [64, 128, 256]
        num_stages = [4, 5]
        from itertools import product

        gemm_config_cls = GemmConfig
        gemm_config_kwargs = {}
        if has_datacenter_blackwell_tma_device():
            gemm_config_cls = BlackwellGPUGemmConfig
            gemm_config_kwargs = {
                "epilogue_subtile": 2,
                "warp_specialize": True,
                "flatten": True,
            }
        cls.gemm_configs = [
            gemm_config_cls(BLOCK_M, BLOCK_N, BLOCK_K, stage, 8, **gemm_config_kwargs)
            for BLOCK_M, BLOCK_N, BLOCK_K, stage in product(
                block_sizes, block_sizes, block_sizes, num_stages
            )
            # Don't test for very large block sizes nor very small ones
            if BLOCK_M + BLOCK_N + BLOCK_K < 512 and BLOCK_M + BLOCK_N + BLOCK_K > 192
        ]

    def setUp(self):
        # [FUNCTION DOC] Saves the current config lists of the four heuristics for restoration.
        super().setUp()
        # Save original configs to restore in tearDown
        self.original_tma_mm_configs = self.mm_tma_heuristic.mm_configs
        self.original_mm_mm_configs = self.mm_heuristic.mm_configs
        self.original_addmm_tma_configs = self.addmm_tma_heuristic.mm_configs
        self.original_addmm_configs = self.addmm_heuristic.mm_configs

    def tearDown(self):
        # [FUNCTION DOC] Restores the saved config lists on the four heuristics, then calls
        # the parent.
        # Restore original configs
        self.addmm_tma_heuristic.mm_configs = self.original_addmm_tma_configs
        self.addmm_heuristic.mm_configs = self.original_addmm_configs
        self.mm_tma_heuristic.mm_configs = self.original_tma_mm_configs
        self.mm_heuristic.mm_configs = self.original_mm_mm_configs
        super().tearDown()

    @contextlib.contextmanager
    def pruning_config_context(self):
        # [FUNCTION DOC] HELPER: context manager that disables shared-memory based choice pruning,
        # enables persistent TMA matmul, sets autotune warmup to 0 and repetitions to 1 and uses a
        # fresh cache for its duration.
        """Context manager for shared memory pruning configuration."""
        with (
            config.patch(
                {
                    "max_autotune_prune_choices_based_on_shared_mem": False,
                    "triton.enable_persistent_tma_matmul": True,
                    "inductor_default_autotune_warmup": 0,
                    "inductor_default_autotune_rep": 1,
                }
            ),
            fresh_cache(),
        ):
            yield

    def create_test_tensors(
        self,
        M,
        N,
        K,
        include_bias=False,
        dtype=torch.bfloat16,
        mat1_transposed=False,
        mat2_transposed=False,
    ):
        # [FUNCTION DOC] HELPER: creates GPU tensors for an M by K and K by N matmul, optionally
        # stored transposed (created transposed then viewed back). With include_bias returns a 1D
        # bias of length N followed by the two matrices, otherwise only the two matrices.
        if mat1_transposed:
            mat1 = torch.randn(K, M, dtype=dtype, device=GPU_TYPE).t()
        else:
            mat1 = torch.randn(M, K, dtype=dtype, device=GPU_TYPE)

        if mat2_transposed:
            mat2 = torch.randn(N, K, dtype=dtype, device=GPU_TYPE).t()
        else:
            mat2 = torch.randn(K, N, dtype=dtype, device=GPU_TYPE)

        if include_bias:
            bias_1d = torch.randn(N, dtype=dtype, device=GPU_TYPE)
            return bias_1d, mat1, mat2
        return mat1, mat2

    def test_max_autotune_prune_choices(self):
        # [FUNCTION DOC] Compiles a 3 by 3 float32 matmul in max-autotune mode and requires the
        # counter of precompilation exceptions during algorithm selection to be zero (small sizes
        # must not break choice pruning).
        def mm(x, y):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return x @ y

        M, K, N = (3, 3, 3)

        x = torch.rand([M, K], device=GPU_TYPE, dtype=torch.float32)
        y = torch.rand([K, N], device=GPU_TYPE, dtype=torch.float32)

        compiled_f = torch.compile(mm, mode="max-autotune")
        compiled_f(x, y)

        self.assertEqual(
            counters["inductor"]["select_algorithm_num_precompilation_exceptions"], 0
        )

    @skipIfXpu(msg="Missing device_properties shared_memory_per_block on xpu.")
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("mat1_transposed", (False, True))
    @parametrize("mat2_transposed", (False, True))
    @parametrize("use_tma", (False, True))
    def test_shared_memory_pruning_addmm(
        self,
        dtype: torch.dtype,
        mat1_transposed: bool,
        mat2_transposed: bool,
        use_tma: bool,
    ):
        # [FUNCTION DOC] Shared memory pruning check for addmm, over dtype, operand transposition
        # and TMA vs non-TMA heuristic. Returns early for TMA with float32 or without device TMA
        # support. Empties the config list of the unused heuristic, builds a limits dict for addmm
        # and delegates to the shared checker with the selected heuristic.
        """Test shared memory pruning for addmm operation."""

        if use_tma and (dtype == torch.float32 or not has_triton_tma_device()):
            return

        def addmm_op(bias, mat1, mat2):
            # [FUNCTION DOC] HELPER: addmm of the three inputs.
            return torch.addmm(bias, mat1, mat2)

        M, K, N = 512, 512, 512
        bias_1d, mat1, mat2 = self.create_test_tensors(
            M,
            N,
            K,
            include_bias=True,
            dtype=dtype,
            mat1_transposed=mat1_transposed,
            mat2_transposed=mat2_transposed,
        )
        dtype_size = mat1.dtype.itemsize

        if use_tma:
            self.addmm_heuristic.mm_configs = []
            heuristic = self.addmm_tma_heuristic
        else:
            self.addmm_tma_heuristic.mm_configs = []
            heuristic = self.addmm_heuristic

        shared_memory_checker_opts = get_shared_memory_checker_opts("addmm", dtype_size)

        self.run_op_shared_mem_pruning_check(
            heuristic,
            addmm_op,
            (bias_1d, mat1, mat2),
            dtype_size,
            shared_memory_checker_opts,
        )

    @skipIfXpu(msg="Missing device_properties shared_memory_per_block on xpu.")
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("mat1_transposed", (False, True))
    @parametrize("mat2_transposed", (False, True))
    @parametrize("use_tma", (False, True))
    def test_shared_memory_pruning_mm(
        self,
        dtype: torch.dtype,
        mat1_transposed: bool,
        mat2_transposed: bool,
        use_tma: bool,
    ):
        # [FUNCTION DOC] Shared memory pruning check for mm, over dtype, operand transposition
        # and TMA vs non-TMA heuristic. Mirrors the addmm test without bias: returns early for
        # unsupported TMA cases, empties the other heuristic's configs and delegates to the
        # shared checker.
        if use_tma and (dtype == torch.float32 or not has_triton_tma_device()):
            return

        def mm_op(mat1, mat2):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return mat1 @ mat2

        M, K, N = 512, 512, 512
        mat1, mat2 = self.create_test_tensors(
            M,
            N,
            K,
            include_bias=False,
            dtype=dtype,
            mat1_transposed=mat1_transposed,
            mat2_transposed=mat2_transposed,
        )
        dtype_size = mat1.dtype.itemsize

        if use_tma:
            self.mm_heuristic.mm_configs = []
            heuristic = self.mm_tma_heuristic
        else:
            self.mm_tma_heuristic.mm_configs = []
            heuristic = self.mm_heuristic

        shared_memory_checker_opts = get_shared_memory_checker_opts("mm", dtype_size)

        self.run_op_shared_mem_pruning_check(
            heuristic, mm_op, (mat1, mat2), dtype_size, shared_memory_checker_opts
        )

    def run_op_shared_mem_pruning_check(
        self, heuristic, op, inputs, dtype_size, shared_memory_checker_opts
    ):
        # [FUNCTION DOC] HELPER: for every candidate config, restricts the heuristic to that
        # config, computes its estimated shared memory and the predicted over-limit flag, compiles
        # the op in max-autotune mode with precompile and autotune wrapped to capture the kernel's
        # real shared memory and whether any Triton choice got a finite time. If compilation
        # failed for all Triton choices the checker must have predicted exceeding; otherwise the
        # actual shared memory must not exceed the estimate. Skips the test when the device cannot
        # report shared memory.
        exceeds_checker = heuristic._get_exceeding_shared_memory_checker(
            **shared_memory_checker_opts
        )
        if exceeds_checker is None:
            self.skipTest("Device does not support shared memory size query")
        for c in self.gemm_configs:
            smem_estimation = heuristic.get_shared_memory_estimation(
                c, dtype_size, **shared_memory_checker_opts
            )
            # Configure heuristics to use only this specific config
            heuristic.mm_configs = [c]
            exceeds = exceeds_checker(c, dtype_size)

            original_precompile = CachingAutotuner.precompile
            original_autotune = AlgorithmSelectorCache.autotune

            captured_smem = 0
            triton_compilation_fails = True

            def mock_precompile(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: wraps the autotuner precompile; runs the original and
                # then records the shared memory size from the last compiled kernel result into
                # the enclosing variable.
                original_precompile(self, *args, **kwargs)
                # Access compile_results after precompilation
                for result in self.compile_results:
                    # Get shared memory from the compiled kernel
                    kernel = result.kernel
                    shared_mem = (
                        kernel.shared
                        if hasattr(kernel, "shared")
                        else kernel.metadata.shared
                    )
                    nonlocal captured_smem
                    captured_smem = shared_mem

            def mock_autotune(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: wraps algorithm selection autotune; runs the original,
                # marks the enclosing flag false if any Triton template caller got a finite
                # timing, and returns timings.
                timings = original_autotune(self, *args, **kwargs)
                nonlocal triton_compilation_fails
                for caller, t in timings.items():
                    if isinstance(caller, TritonTemplateCaller) and t != float("inf"):
                        triton_compilation_fails = False
                return timings

            with (
                self.pruning_config_context(),
                mock.patch.object(CachingAutotuner, "precompile", mock_precompile),
                mock.patch.object(AlgorithmSelectorCache, "autotune", mock_autotune),
            ):
                torch._dynamo.reset()
                counters.clear()
                compiled_fn = torch.compile(op, mode="max-autotune")
                run_and_get_code(compiled_fn, *inputs)

            if triton_compilation_fails:
                self.assertTrue(
                    exceeds,
                    lambda msg: f"{msg}\nConfig {c} failed to compile due to shared memory, "
                    "but the checker predicted it would NOT exceed shared memory limits.",
                )
            else:
                self.assertTrue(
                    captured_smem <= smem_estimation,
                    lambda msg: f"{msg}\nEstimated maximum smem should exceed actual smem used for config {c}",
                )


class TestMaxAutotunePrecompile(TestCase):
    # [CLASS DOC] Tests for precompilation behavior in algorithm selection, using fake choice
    # callers and compile-count counters. Has no shared state.
    def test_precompilation_threads(self):
        # [FUNCTION DOC] Checks that algorithm selection precompiles choices on worker threads.
        # Ten fake choice callers record the thread id of their precompile call; lookup is
        # replaced to just run the benchmark function and compile_threads equals the number of
        # choices. After selection all precompile calls must have happened and none on the main
        # thread (AssertionError otherwise). The debug handler is replaced during the test and
        # restored in a finally block.
        import threading
        from typing import Any
        from unittest.mock import Mock

        class FakeChoiceCaller(ChoiceCaller):
            # [CLASS DOC] HELPER: fake choice caller whose precompile records the calling thread
            # id; the other abstract methods return None or a hash of the object.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: initializes with dummy name, no inputs and a mock layout;
                # thread id unset.
                super().__init__("none", [], Mock(), description="")
                self.thread_id = None

            def precompile(self):
                # [FUNCTION DOC] HELPER: records the id of the calling thread.
                self.thread_id = threading.get_ident()

            def call_name(self) -> str:
                # [FUNCTION DOC] HELPER: returns None.
                return None

            def to_callable(self):
                # [FUNCTION DOC] HELPER: returns None.
                return None

            def hash_key(self) -> str:
                # [FUNCTION DOC] HELPER: returns a string of this object's hash.
                return str(hash(self))

            def output_node(self) -> "TensorBox":  # noqa: F821
                # [FUNCTION DOC] HELPER: returns None.
                return None

        fake_choices = [FakeChoiceCaller() for i in range(10)]
        fake_lookup_result = dict.fromkeys(fake_choices, 0.123)

        def no_lookup(
            choices: list[ChoiceCaller],
            op: str,
            inputs: str,
            benchmark: Callable[[Any], dict[ChoiceCaller, float]],
            hint_override: int | None = None,
        ) -> dict[ChoiceCaller, float] | None:
            # [FUNCTION DOC] HELPER: stand-in for the cache lookup; if a benchmark callable is
            # given runs it on the choices and returns its timings, otherwise returns None.
            if benchmark is not None:
                return benchmark(choices)

        asc = AlgorithmSelectorCache()

        def fake_benchmark_fn(*args, **kwargs):
            # [FUNCTION DOC] HELPER: ignores arguments and returns the fixed mapping of fake
            # choices to timings.
            return fake_lookup_result

        main_thread_id = threading.get_ident()
        mock_debug_handler = Mock()
        old_debug_handler = V.debug
        try:
            V.set_debug_handler(mock_debug_handler)
            with patch.object(asc, "lookup", new=no_lookup):
                with patch.object(
                    asc, "make_benchmark_fn", return_value=fake_benchmark_fn
                ):
                    with config.patch(
                        {
                            "autotune_in_subproc": False,
                            "compile_threads": len(fake_choices),
                        }
                    ):
                        asc("test_call", fake_choices, [], Mock())
            for fake_choice in fake_choices:
                if fake_choice.thread_id is None:
                    raise AssertionError(
                        "Expected all ChoiceCaller's precompile method to have been called"
                    )
                if fake_choice.thread_id == main_thread_id:
                    raise AssertionError(
                        "Expected all ChoiceCaller's precompile method to have been called on separate thread"
                    )
        finally:
            V.set_debug_handler(old_debug_handler)

    def test_filled_cache_precompile(self):
        # [FUNCTION DOC] Compiles a two-stage matmul chain (float32 then float16) in max-
        # autotune mode and checks it matches eager. Then resets dynamo and counters and
        # compiles again, requiring the precompile counter to be zero (note the second compiled
        # function is never called).
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: chained matmul in float32, then cast to float16 and chained
            # matmul again.
            a = (a @ b) @ c
            a, b, c = (t.to(torch.float16) for t in [a, b, c])
            return (a @ b) @ c

        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        inputs = [torch.rand([256, 256], device=GPU_TYPE) for _ in range(3)]
        from torch._dynamo.utils import counters

        self.assertEqual(fn(*inputs), fn_c(*inputs), atol=1e-2, rtol=1e-2)

        torch._dynamo.reset()
        counters.clear()

        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 0)

    @config.patch(autotune_local_cache=False, autotune_remote_cache=False)
    @unittest.skipIf(config.triton.native_matmul, "native matmul has counter 0")
    def test_precompilations(self):
        # [FUNCTION DOC] With local and remote autotune caches disabled, compiles a matmul chain
        # in float32 and float16 with small inputs, checks numerics and that the precompile
        # counter equals two. Skipped for native matmul.
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: chained matmul in float32, then cast to float16 and chained
            # matmul again.
            a = (a @ b) @ c
            a, b, c = (t.to(torch.float16) for t in [a, b, c])
            return (a @ b) @ c

        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        # Scale down so float16 doesn't overflow: rand [0,1) -> (a@b)@c has elements in [0, 256^2).
        # float16 max ~65504, so we keep values in a safe range (e.g. scale by 1/256).
        inputs = [torch.rand([256, 256], device=GPU_TYPE) / 256.0 for _ in range(3)]

        torch.testing.assert_close(fn_c(*inputs), fn(*inputs), atol=1e-2, rtol=1e-2)

        from torch._dynamo.utils import counters

        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 2)


@instantiate_parametrized_tests
class TestMaxAutotuneSubproc(TestCase):
    # [CLASS DOC] Tests for benchmarking and autotuning in subprocesses, including spawning a
    # child process to benchmark choices. Holds no state beyond a helper to create IR buffers.
    def _create_buffer(self, name, shape):
        # [FUNCTION DOC] HELPER: creates an IR buffer with the given name and a fixed float32
        # layout on the first GPU device with the given shape.
        return Buffer(
            name=name,
            layout=FixedLayout(
                torch.device(f"{GPU_TYPE}:0"), dtype=torch.float32, size=shape
            ),
        )

    @skipIfXpu(msg="XPU not support multiprocessing tensor reduction")
    def test_benchmark_choice_in_subproc(self):
        # [FUNCTION DOC] Builds example values for four buffers and a layout under a dummy graph,
        # binds the ATen mm plus mm choice, then starts a spawned child process that benchmarks
        # the choice and writes timings into a shared tensor. The child must exit with code zero.
        # Skipped on XPU.
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is needed to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            layout = FixedLayout(torch.device(f"{GPU_TYPE}:0"), torch.float32, (2, 2))

            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            # expected_out = (mat1 @ mat2) + (mat3 @ mat4)
            expected_out = None

            choice = aten_mm_plus_mm.bind((buf1, buf2, buf3, buf4), layout)
            # use a tensor since the mutation to a python list in a sub process
            # is not synced back to the parent process
            timings = torch.zeros(3, dtype=torch.float32)
            ctx = mp.get_context("spawn")
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            child.start()
            child.join()
            self.assertEqual(0, child.exitcode)
            print(f"timings is {timings}, out {out}, expected_out {expected_out}")

    @skipIfXpu(msg="XPU not support multiprocessing tensor reduction")
    def test_benchmark_choice_fail_in_subproc(self):
        # [FUNCTION DOC] Like the subprocess benchmark test but with a choice caller that always
        # raises and an expected output given; the spawned child process must exit with a nonzero
        # code. Skipped on XPU.
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is needed to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            layout = FixedLayout(torch.device(f"{GPU_TYPE}:0"), torch.float32, (2, 2))

            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            expected_out = (mat1 @ mat2) + (mat3 @ mat4)

            choice = FailChoiceCaller("fail_choice_caller", [], None, description="")

            # use a tensor since python list is not synced back
            timings = torch.zeros(3, dtype=torch.float32)
            ctx = mp.get_context("spawn")
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            child.start()
            child.join()
            self.assertNotEqual(0, child.exitcode)

    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("autotune_multi_device", (True, False))
    def test_max_autotune_mm_plus_mm(self, autotune_in_subproc, autotune_multi_device):
        # [FUNCTION DOC] Regression test for mm plus mm autotuning, which once crashed from a
        # Triton issue: compiles a sum of two matmuls with max autotune over all combinations of
        # autotuning in subprocess and multi-device autotuning; only checks that nothing crashes.
        """
        This crash previously due to a triton issue: https://github.com/triton-lang/triton/issues/1298 .
        With autotuning in subprocess, we don't crash anymore.
        """
        m, n, k = 2048, 1536, 64

        def mm_plus_mm(a, b, c, d):
            # [FUNCTION DOC] HELPER: sum of two matrix products.
            return a @ b + c @ d

        a = torch.randn(m, k).to(GPU_TYPE)
        b = torch.randn(k, n).to(GPU_TYPE)
        c = torch.randn(m, k).to(GPU_TYPE)
        d = torch.randn(k, n).to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": autotune_in_subproc,
                "autotune_multi_device": autotune_multi_device,
            }
        ):
            torch.compile(mm_plus_mm)(a, b, c, d)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm(self, dynamic: bool):
        # [FUNCTION DOC] Compiles sin followed by matmul with autotuning in subprocesses, static
        # and dynamic shapes; only checks it does not crash.
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: sin of a then matrix product with b.
            a = torch.sin(a)
            return a @ b

        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    def test_max_autotune_profiler_benchmarker_smoke(self):
        # [FUNCTION DOC] Smoke test: compiles a small matmul with max autotune and the torch
        # profiler benchmarker enabled, with GPU benchmarking wrapped to record each timing.
        # Requires at least one finite timing and the smallest finite timing to be above zero.
        """
        Smoke test that a simple max-autotune matmul runs with profiler
        benchmarker selection enabled through config patching, and records
        positive autotune benchmark timing.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return a @ b

        a = torch.randn(64, 32).to(GPU_TYPE)
        b = torch.randn(32, 64).to(GPU_TYPE)

        import torch._inductor.runtime.benchmarking as inductor_benchmarking

        benchmark_timings_ms = []
        original_benchmark_gpu = inductor_benchmarking.benchmarker.benchmark_gpu

        def benchmark_gpu_with_capture(*args, **kwargs):
            # [FUNCTION DOC] HELPER: calls the original GPU benchmark, appends its result to the
            # enclosing list and returns it.
            timing_ms = original_benchmark_gpu(*args, **kwargs)
            benchmark_timings_ms.append(timing_ms)
            return timing_ms

        with config.patch(
            {"max_autotune": True, "use_torch_profiler_benchmarker": True}
        ):
            with mock.patch.object(
                inductor_benchmarking.benchmarker,
                "benchmark_gpu",
                side_effect=benchmark_gpu_with_capture,
            ):
                torch.compile(mm)(a, b)

        finite_timings_ms = [t for t in benchmark_timings_ms if math.isfinite(t)]
        self.assertGreater(
            len(finite_timings_ms),
            0,
            lambda msg: f"{msg}\nExpected finite autotune benchmark timings, got {benchmark_timings_ms}",
        )
        self.assertGreater(
            min(finite_timings_ms),
            0.0,
            lambda msg: f"{msg}\nExpected autotune benchmark timing > 0, got {finite_timings_ms}",
        )

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm(self, search_space, dynamic=False):
        # [FUNCTION DOC] Compiles an addmm with autotuning in subprocesses for both search spaces
        # (dynamic flag, default false) and compares with eager. Side effect: disables fp16
        # reduced precision reduction globally and does not restore it.
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def addmm(x, a, b):
            # [FUNCTION DOC] HELPER: addmm of the three inputs.
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_search_space": search_space,
            }
        ):
            Y_compiled = torch.compile(addmm, dynamic=dynamic)(x, a, b)
            Y = addmm(x, a, b)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    def test_triton_template_with_epilogues_and_dynamic_shape(self):
        # [FUNCTION DOC] Compiles, with subprocess autotuning, the Triton-only backend and full
        # graph with dynamic shapes, a transposed matmul with bias, relu and multiply, and runs it
        # with two different second sizes, comparing each result with eager.
        def fn(
            x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, mul: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: relu of the matmul of transposed x and transposed w plus
            # bias, multiplied by mul.
            return (
                torch.nn.functional.relu(
                    torch.matmul(torch.transpose(x, 0, 1), torch.transpose(w, 0, 1))
                    + bias
                )
                * mul
            )

        M0 = 5
        M1 = 8
        K = 4
        N = 3
        w = torch.rand(N, K).to(GPU_TYPE).half()
        b = torch.rand(N).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            compiled_fn = torch.compile(
                fn, fullgraph=True, dynamic=True, mode="max-autotune-no-cudagraphs"
            )

            x0 = torch.rand(K, M0).to(GPU_TYPE).half()
            mul0 = torch.rand(M0, N).to(GPU_TYPE).half()
            y0 = compiled_fn(x0, w, b, mul0)
            y0_expected = fn(x0, w, b, mul0)
            torch.testing.assert_close(y0, y0_expected)

            x1 = torch.rand(K, M1).to(GPU_TYPE).half()
            mul1 = torch.rand(M1, N).to(GPU_TYPE).half()
            y1 = compiled_fn(x1, w, b, mul1)
            y1_expected = fn(x1, w, b, mul1)
            torch.testing.assert_close(y1, y1_expected)


@instantiate_parametrized_tests
class TestMaxAutotuneRemoteCache(TestCase):
    # [CLASS DOC] Tests for remote caching of autotune results using the mock cache classes. Test
    # setUp activates the cache patches and tearDown deactivates them.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and installs the mock cache patches.
        super().setUp()
        PatchCaches.setUp()

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and removes the mock cache patches.
        super().tearDown()
        PatchCaches.tearDown()

    @parametrize("dynamic", (False, True))
    @config.patch(
        {"compile_threads": 1, "prologue_fusion": False}
    )  # Worker processes do not register PatchCaches() properly
    def test_max_autotune_remote_caching(self, dynamic: bool):
        # [FUNCTION DOC] Checks the autotune remote cache hit/miss/put statistics with mock
        # caches. Local autotune cache is disabled and remote enabled; the Triton cache manager
        # variable is removed from the environment. Compiles the same sin-then-matmul function
        # four times (each with a fresh cache and dynamo reset) plus once with a different cache
        # key tag, and requires the autotune remote stats to equal the expected triple of hits,
        # misses and puts. Then resets stats, repeats with a function with no GEMM (an add inside
        # a module) and requires the same stats.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: sin of a then matrix product with b.
            a = torch.sin(a)
            return a @ b

        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: tiny module whose forward adds its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: returns the sum of the two inputs.
                return x + y

        def f(x, y):
            # [FUNCTION DOC] HELPER: builds a Model and returns the sum of the inputs through it.
            return Model()(x, y)

        x = torch.randn(100, 100).to(GPU_TYPE)
        y = torch.randn(100, 100).to(GPU_TYPE)

        with (
            config.patch(
                {
                    "autotune_local_cache": False,
                    "autotune_remote_cache": True,
                }
            ),
            patch.dict(os.environ),
            PatchCaches(),
        ):
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            with config.patch({"max_autotune": True}):
                for _ in range(4):
                    with fresh_cache():
                        torch.compile(mm, dynamic=dynamic)(a, b)
                    reset()
                with (
                    torch.compiler.config.patch({"cache_key_tag": "test"}),
                    fresh_cache(),
                ):
                    torch.compile(mm, dynamic=dynamic)(a, b)
                    reset()

                global_stats.report()
                self.assertEqual(global_stats.autotune_remote, Stats(2, 3, 2))

            global_stats.reset()
            for _ in range(4):
                with fresh_cache():
                    torch.compile(f, dynamic=dynamic)(x, y)
                reset()
            with torch.compiler.config.patch({"cache_key_tag": "test"}), fresh_cache():
                torch.compile(mm, dynamic=dynamic)(a, b)
                reset()
            global_stats.report()
            self.assertEqual(global_stats.autotune_remote, Stats(2, 3, 2))


class _TestTritonTemplateCaller(TritonTemplateCaller):
    # [CLASS DOC] HELPER: minimal template caller for tuning pool tests. It bypasses the parent
    # constructor, stores only a benchmark request in the bmreq attribute and prints as a fixed
    # string, which is enough for the pool to benchmark it.
    def __init__(self, bmreq: _TestBenchmarkRequest):
        # [FUNCTION DOC] HELPER: stores the benchmark request; intentionally skips
        # parent initialization.
        self.bmreq = bmreq

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: returns the fixed string used as a stand-in name.
        return "test"


class TestTuningProcess(TestCase):
    # [CLASS DOC] Tests for a single TuningProcess subprocess: timeout, exceptions, crashes, being
    # killed and device visibility. Each test creates its own process and shuts it down.
    def check_healthy(self, p: TuningProcess, device: int | None = None):
        # [FUNCTION DOC] HELPER: sends a benchmark request returning a random number to process p
        # (optionally pinned to a device) and asserts the value received back equals it.
        result = random.random()
        bmreq = _TestBenchmarkRequest(result, device=device)
        p.put(bmreq.benchmark)
        self.assertEqual(p.get(), result)

    def test_tuning_subproc_timeout(self):
        # [FUNCTION DOC] A request that sleeps 120s must make getting a result with a one
        # second timeout raise TimeoutError, and the process must still serve a healthy
        # request afterwards.
        p = TuningProcess(None)

        bmreq = _TestBenchmarkRequest(0, sleep=120)
        p.put(bmreq.benchmark)
        with self.assertRaises(TimeoutError):
            p.get(timeout=1.0)

        # Make sure the TuningProcess is still usable after a timeout.
        self.check_healthy(p)
        p.shutdown()

    def test_tuning_subproc_exception(self):
        # [FUNCTION DOC] A request that raises RuntimeError must surface the error from getting
        # its result, and the process must still serve a healthy request afterwards.
        p = TuningProcess(None)

        bmreq = _TestBenchmarkRequest(0, exc=RuntimeError("Fail"))
        p.put(bmreq.benchmark)
        with self.assertRaises(RuntimeError):
            p.get()

        # Make sure the TuningProcess is still usable after an exception.
        self.check_healthy(p)
        p.shutdown()

    def test_tuning_subproc_crash(self):
        # [FUNCTION DOC] A request that makes the subprocess exit must cause EOFError when getting
        # the result, and the process must be usable (restarted) for a healthy request afterwards.
        p = TuningProcess(None)

        bmreq = _TestBenchmarkRequest(0, crash=True)
        p.put(bmreq.benchmark)
        with self.assertRaises(EOFError):
            p.get()

        # Make sure the TuningProcess is still usable after a crash.
        self.check_healthy(p)
        p.shutdown()

    def test_tuning_subproc_killed(self):
        # [FUNCTION DOC] Kills the process and checks that a new healthy request still succeeds,
        # i.e. the process restarts lazily.
        p = TuningProcess(None)
        p.kill()
        self.check_healthy(p)
        p.shutdown()

    def test_visible_devices(self):
        # [FUNCTION DOC] For every device in the pool device list, starts a process bound to it
        # and checks a healthy request (which verifies the visible-devices variable in the child).
        device_list = TuningProcessPool.get_device_list()
        for device in device_list:
            p = TuningProcess(device)
            self.check_healthy(p, device=device)
            p.shutdown()


class TestTuningProcessPool(TestCase):
    # [CLASS DOC] Tests for the process pools that run autotune benchmarks: environment and
    # code cache resets between requests, crash and timeout recovery, device selection and
    # affinity masks, and feedback-saver registration. Uses a single device where restart
    # behavior is tested.
    # Use only one device/subprocess so we test the process restarts
    # and is usable after a crash.
    def assert_path_in_dir(self, path, expected_dir):
        # [FUNCTION DOC] HELPER: asserts that path lies inside expected_dir by comparing the
        # common path.
        self.assertEqual(
            os.path.commonpath([path, expected_dir]),
            expected_dir,
        )

    @config.patch({"autotune_multi_device": False})
    def test_tuning_pool_cache_env_resets(self):
        # [FUNCTION DOC] Checks the tuning pool reapplies the parent cache directory variable per
        # request. A request sent with a stale value in extra environment reports that value, the
        # parent environment remains unset, and a normal pool target call then reports none. Pool
        # is shut down.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRITON_CACHE_DIR", None)
            tuning_pool = TuningProcessPool()
            choice = _TestTritonTemplateCaller(
                _TestEnvBenchmarkRequest("TRITON_CACHE_DIR")
            )
            stale_triton_cache_dir = os.path.join(
                tempfile.gettempdir(), "stale_triton_cache"
            )

            try:
                process = tuning_pool.processes[0]
                process.put(
                    choice.bmreq.benchmark,
                    extra_env={"TRITON_CACHE_DIR": stale_triton_cache_dir},
                )
                self.assertEqual(process.get(), stale_triton_cache_dir)

                self.assertIsNone(os.environ.get("TRITON_CACHE_DIR"))
                self.assertIsNone(tuning_pool.target(choice))
            finally:
                tuning_pool.shutdown()

    @config.patch({"autotune_multi_device": False})
    def test_tuning_pool_cache_env_clears_codecache(self):
        # [FUNCTION DOC] Checks that the tuning pool clears in-subprocess code caches when the
        # Inductor cache directory changes: two successive requests with different directories
        # must return file paths inside the respective directories, and the paths must differ.
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            tempfile.TemporaryDirectory() as cache_dir_1,
            tempfile.TemporaryDirectory() as cache_dir_2,
        ):
            tuning_pool = TuningProcessPool()
            choice = _TestTritonTemplateCaller(_TestCodeCacheBenchmarkRequest())

            try:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir_1
                path_1 = tuning_pool.target(choice)
                self.assert_path_in_dir(path_1, cache_dir_1)

                os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir_2
                path_2 = tuning_pool.target(choice)
                self.assert_path_in_dir(path_2, cache_dir_2)
                self.assertNotEqual(path_1, path_2)
            finally:
                tuning_pool.shutdown()

    @config.patch({"pipeline_max_autotune_gemm": True})
    def test_autotune_process_pool_cache_env_resets(self):
        # [FUNCTION DOC] Same environment-reset check for the pipelined autotune process pool: a
        # task run with a stale cache dir variable reports it; after the variable is removed the
        # next task must report none. The pool is shut down in a finally block.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRITON_CACHE_DIR", None)
            autotune_pool = AutotuneProcessPool()
            bmreq = _TestEnvBenchmarkRequest("TRITON_CACHE_DIR")
            stale_triton_cache_dir = os.path.join(
                tempfile.gettempdir(), "stale_triton_cache"
            )

            try:
                os.environ["TRITON_CACHE_DIR"] = stale_triton_cache_dir
                self.assertEqual(
                    autotune_pool.submit(
                        run_autotune_in_subprocess,
                        bmreq,
                    ).result(timeout=30),
                    stale_triton_cache_dir,
                )

                os.environ.pop("TRITON_CACHE_DIR", None)
                self.assertIsNone(
                    autotune_pool.submit(
                        run_autotune_in_subprocess,
                        bmreq,
                    ).result(timeout=30)
                )
            finally:
                autotune_pool._shutdown()

    @config.patch({"pipeline_max_autotune_gemm": True})
    def test_autotune_process_pool_cache_env_clears_codecache(self):
        # [FUNCTION DOC] Same code cache reset check for the pipelined autotune process pool:
        # tasks with two different Inductor cache directories must return file paths inside each
        # directory and the two paths must differ.
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            tempfile.TemporaryDirectory() as cache_dir_1,
            tempfile.TemporaryDirectory() as cache_dir_2,
        ):
            autotune_pool = AutotuneProcessPool()
            bmreq = _TestCodeCacheBenchmarkRequest()

            try:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir_1
                path_1 = autotune_pool.submit(
                    run_autotune_in_subprocess,
                    bmreq,
                ).result(timeout=30)
                self.assert_path_in_dir(path_1, cache_dir_1)

                os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir_2
                path_2 = autotune_pool.submit(
                    run_autotune_in_subprocess,
                    bmreq,
                ).result(timeout=30)
                self.assert_path_in_dir(path_2, cache_dir_2)
                self.assertNotEqual(path_1, path_2)
            finally:
                autotune_pool._shutdown()

    @config.patch({"autotune_multi_device": False})
    def test_tuning_pool_crash(self):
        # [FUNCTION DOC] A crashing request must give infinite time for its choice, then
        # a following healthy request must yield its real result, showing the pool
        # process restarted.
        tuning_pool = TuningProcessPool()

        # First force the tuning process to crash.
        bmreq = _TestBenchmarkRequest(0, crash=True)
        choice = _TestTritonTemplateCaller(bmreq)

        timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], float("inf"))

        # Then send another request and make sure the sub-process
        # has restarted and is operational.
        bmreq = _TestBenchmarkRequest(3.14)
        choice = _TestTritonTemplateCaller(bmreq)

        timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], bmreq.result)

        tuning_pool.shutdown()

    @config.patch({"autotune_multi_device": False})
    def test_tuning_pool_timeout(self):
        # [FUNCTION DOC] A request sleeping past a one second result timeout must give infinite
        # time for its choice, then a following healthy request must yield its real result.
        tuning_pool = TuningProcessPool()

        # First force the tuning process to timeout.
        bmreq = _TestBenchmarkRequest(0, sleep=120)
        choice = _TestTritonTemplateCaller(bmreq)

        with config.patch({"max_autotune_subproc_result_timeout_seconds": 1.0}):
            timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], float("inf"))

        # Then send another request and make sure the sub-process
        # has restarted and is operational.
        bmreq = _TestBenchmarkRequest(3.14)
        choice = _TestTritonTemplateCaller(bmreq)

        timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], bmreq.result)

        tuning_pool.shutdown()

    @config.patch({"autotune_multi_device": True})
    def test_tuning_pool_multiple_devices(self):
        # [FUNCTION DOC] Restricts the visible devices variable to the last two available devices
        # while creating a multi-device pool, then benchmarks two choices and checks each returns
        # its own result.
        # Adapt the test to the available devices (and whether the backend-specific
        # visible devices env var
        # is already set in the environment); use a subset of the available devices
        # to ensure only the subset are visible to the sub-processes.
        visible_devices_env_var = get_visible_devices_env_var(GPU_TYPE)
        if visible_devices_env_var in os.environ:
            visible_devices = os.environ[visible_devices_env_var].split(",")
        else:
            device_interface = get_interface_for_device(GPU_TYPE)
            visible_devices = [str(d) for d in range(device_interface.device_count())]

        selected_visible_devices = ",".join(visible_devices[-2:])
        with unittest.mock.patch.dict(
            os.environ, {visible_devices_env_var: selected_visible_devices}
        ):
            tuning_pool = TuningProcessPool()

        choice1 = _TestTritonTemplateCaller(_TestBenchmarkRequest(3.14))
        choice2 = _TestTritonTemplateCaller(_TestBenchmarkRequest(2.718))

        timings = tuning_pool.benchmark([choice1, choice2])
        self.assertEqual(timings[choice1], choice1.bmreq.result)
        self.assertEqual(timings[choice2], choice2.bmreq.result)

        tuning_pool.shutdown()

    def test_get_visible_devices_env_var(self):
        # [FUNCTION DOC] Checks the visible-devices variable name for cuda and xpu backends.
        self.assertEqual(get_visible_devices_env_var("cuda"), "CUDA_VISIBLE_DEVICES")
        self.assertEqual(get_visible_devices_env_var("xpu"), "ZE_AFFINITY_MASK")

    @config.patch({"autotune_multi_device": True})
    def test_get_device_list_with_affinity_mask(self):
        # [FUNCTION DOC] With the XPU/CUDA affinity variable set to devices 1 and 3 and four
        # devices reported, checks the pool device list is exactly those two device indices.
        env_var = get_visible_devices_env_var(GPU_TYPE)
        env_value = "1,3"

        with (
            mock.patch(
                "torch._inductor.autotune_process.get_interface_for_device"
            ) as get_interface_mock,
            unittest.mock.patch.dict(os.environ, {env_var: env_value}, clear=False),
        ):
            get_interface_mock.return_value.device_count.return_value = 4
            self.assertEqual(TuningProcessPool.get_device_list(), [1, 3])

    def test_add_feedback_saver(self):
        # [FUNCTION DOC] Clears feedback savers, adds one saver and checks the global algorithm
        # selector cache holds exactly that function; adds a second and checks two are stored.
        # Clears at the end.
        """Test that add_feedback_saver correctly adds feedback functions."""
        from torch._inductor.select_algorithm import get_algorithm_selector_cache

        # Clear any existing feedback savers
        clear_feedback_savers()

        # Create a simple feedback saver function
        feedback_calls = []

        def simple_feedback_saver(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            # [FUNCTION DOC] HELPER: feedback saver appending a record of name, choice count,
            # timing count and whether profiled time was given to the enclosing list.
            feedback_calls.append(
                {
                    "name": name,
                    "num_choices": len(choices),
                    "num_timings": len(timings),
                    "has_profiled_time": profiled_time is not None,
                }
            )

        # Add the feedback saver
        add_feedback_saver(simple_feedback_saver)

        # Get the global cache and verify the function was added
        cache = get_algorithm_selector_cache()
        self.assertEqual(len(cache.feedback_saver_fns), 1)
        self.assertEqual(cache.feedback_saver_fns[0], simple_feedback_saver)

        # Test that we can add multiple feedback savers
        def another_feedback_saver(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            # [FUNCTION DOC] HELPER: feedback saver that does nothing.
            pass

        add_feedback_saver(another_feedback_saver)
        self.assertEqual(len(cache.feedback_saver_fns), 2)

        # Clean up
        clear_feedback_savers()

    def test_clear_feedback_savers(self):
        # [FUNCTION DOC] Adds two feedback savers, checks the cache holds two, clears them and
        # checks none remain.
        """Test that clear_feedback_savers removes all feedback functions."""
        from torch._inductor.select_algorithm import get_algorithm_selector_cache

        # Add some feedback savers first
        def feedback_saver1(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            # [FUNCTION DOC] HELPER: feedback saver that does nothing.
            pass

        def feedback_saver2(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            # [FUNCTION DOC] HELPER: feedback saver that does nothing.
            pass

        add_feedback_saver(feedback_saver1)
        add_feedback_saver(feedback_saver2)

        # Verify they were added
        cache = get_algorithm_selector_cache()
        self.assertEqual(len(cache.feedback_saver_fns), 2)

        # Clear all feedback savers
        clear_feedback_savers()

        # Verify they were cleared
        self.assertEqual(len(cache.feedback_saver_fns), 0)

    def test_feedback_saver_integration(self):
        # [FUNCTION DOC] Registers a saver, compiles a matmul with Triton-only max autotune and
        # checks the saver was called with a record having the expected keys, a positive choice
        # count and two input nodes. Saver list is cleared at start and end.
        """Test that feedback savers are actually called during autotuning."""
        # Clear any existing feedback savers
        clear_feedback_savers()

        feedback_calls = []

        def test_feedback_saver(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            # [FUNCTION DOC] HELPER: feedback saver appending name, choice count, timing count and
            # input node count to the enclosing list.
            # Store information about the call for verification
            feedback_calls.append(
                {
                    "name": name,
                    "num_choices": len(choices),
                    "num_timings": len(timings),
                    "input_node_count": len(input_nodes),
                }
            )

        # Add our test feedback saver
        add_feedback_saver(test_feedback_saver)

        # Create a simple matrix multiplication that will trigger autotuning
        def mm(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return a @ b

        a = torch.randn(32, 32, device=GPU_TYPE)
        b = torch.randn(32, 32, device=GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "triton.native_matmul": False,
            }
        ):
            torch.compile(mm)(a, b)

        # Verify that our feedback saver was called
        self.assertGreater(
            len(feedback_calls), 0, "Feedback saver should have been called"
        )

        # Verify the structure of the feedback call
        call = feedback_calls[0]
        self.assertIn("name", call)
        self.assertIn("num_choices", call)
        self.assertIn("num_timings", call)
        self.assertIn("input_node_count", call)
        self.assertGreater(call["num_choices"], 0)
        self.assertEqual(call["input_node_count"], 2)  # Two input matrices

        # Clean up
        clear_feedback_savers()


@instantiate_parametrized_tests
class TestPrologueFusion(TestCase):
    # [CLASS DOC] Tests for prologue fusion into Triton GEMM templates. Class-level config
    # enables max autotune, prologue fusion, Triton-only backend, no shape padding and 4 mm
    # configs (kept active via an exit stack closed in tearDownClass). Tests compile small graphs
    # and check kernel launch, allocation and deallocation counts and the generated masks and
    # dtype handling.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: enters a config patch into a shared exit stack for the
        # whole class.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "max_autotune": True,
                    "prologue_fusion": True,
                    "benchmark_epilogue_fusion": False,
                    "shape_padding": False,
                    "max_autotune_gemm_backends": "TRITON",
                    "test_configs.max_mm_configs": 4,  # significantly speeds up tests
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class fixture: closes the exit stack, undoing the config patch, then
        # the parent.
        cls._stack.close()
        super().tearDownClass()

    def check_code(self, code_str, num_kernels, num_allocs, num_deallocs):
        # [FUNCTION DOC] HELPER: asserts via FileCheck on code_str that the call function launches
        # exactly num_kernels kernels; if num_allocs is not None, exactly that many strided
        # allocations; and if num_deallocs is not None and the C++ wrapper is not in use, exactly
        # that many deletions.
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(),
            num_kernels,
            exactly=True,
        ).run(code_str)

        if num_allocs is not None:
            FileCheck().check(get_func_call()).check_count(
                "empty_strided", num_allocs, exactly=True
            ).run(code_str)

        # skip the deallocation check when using cpp_wrapper; most deallocations happen
        # outside of our control via RAIIAtenTensorHandle
        if num_deallocs is not None and not config.cpp_wrapper:
            FileCheck().check(get_func_call()).check_count(
                "del", num_deallocs, exactly=True
            ).run(code_str)

    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    def test_upcast(self, sizes):
        # [FUNCTION DOC] Compiles a matmul with an fp16 first operand cast to float32; checks
        # numerics and one kernel, one allocation, two deletions. With native matmul the zero mask
        # is kept so a where appears before the dot; otherwise the upcast preserves zeros and no
        # where appears.
        M, K, N = sizes

        x = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: casts x to the dtype of y and matmuls with y.
            return x.to(y.dtype) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)
        if config.triton.native_matmul:
            # native matmul preserves zero mask - need to optimize; see codegen/triton.py
            FileCheck().check("a =").check("tl.where").check("tl.dot").run(code[0])
        else:
            # upcast preserves zero mask
            FileCheck().check("a =").check_not("tl.where").check("tl.dot").run(code[0])

    @unittest.skip("Triton bug in compilation")
    def test_gather_fusion(self):
        # [FUNCTION DOC] Skipped due to a Triton compilation bug. Intended: a gathered (indexed)
        # first operand fused as prologue into one kernel with low-precision loads before the dot.
        M, K, N = (64, 128, 256)
        x = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float16, device=GPU_TYPE)

        index = torch.randperm(M, device=GPU_TYPE)

        def foo(x, y, index):
            # [FUNCTION DOC] HELPER: matmul of the index-selected rows of x with y.
            return (x[index]) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y, index)
        self.assertEqual(out, foo(x, y, index), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=3)

        # should be done in low precision
        (
            FileCheck()
            .check("for k_idx")
            .check_not("to(tl.float32)")
            .check("dot")
            .run(code[0])
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @config.patch({"triton.native_matmul": False})
    def test_low_precision(self):
        # [FUNCTION DOC] Needs FP8 support. A float8 operand cast to bfloat16 and matmul
        # should fuse into one kernel with no float32 conversion in the loop; adding one after
        # the cast must not fuse (two kernels, two allocations, three deletions). Native
        # matmul is disabled.
        M = K = N = 128

        x = torch.rand([M, K], device=GPU_TYPE).to(torch.float8_e4m3fn)
        y = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: casts x to the dtype of y and matmuls with y.
            return x.to(y.dtype) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

        # should be done in low precision, no arithmetic
        (
            FileCheck()
            .check("for k_idx")
            .check_not("to(tl.float32)")
            .check("dot")
            .run(code[0])
        )

        def foo(x, y):
            # [FUNCTION DOC] HELPER: casts x to the dtype of y, adds one, and matmuls with y.
            return (x.to(y.dtype) + 1) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)

        # should not be done in low precision, two kernels
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_downcast(self):
        # [FUNCTION DOC] Checks that a float32 to float16 downcast prologue is not fused into
        # the matmul by heuristics: two kernels, two allocations, three deletions. Skipped for
        # native matmul.
        # per heuristics, dont fuse a downcast into a mm because it would lead to more reads inside kernel
        M, K, N = (64, 128, 256)
        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float16, device=GPU_TYPE)

        def foo(x, y):
            # [FUNCTION DOC] HELPER: casts x to the dtype of y and matmuls with y.
            return x.to(y.dtype) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @parametrize("sizes", ((64, 128, 256), (64, 64, 64), (64, 120, 64)))
    @parametrize("use_async_compile", (True, False))
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_multiple_fusions(self, sizes, use_async_compile: bool):
        # [FUNCTION DOC] Compiles a matmul with prologue subtractions and an epilogue multiply,
        # with or without waiting for the async compile pool, over three sizes. Checks numerics, a
        # single kernel, and that the constant 1.1 is materialized exactly three times in the
        # kernel (no common subexpression sharing between prologues and epilogue).
        M, K, N = sizes

        def foo(x, y):
            # [FUNCTION DOC] HELPER: matmul of (x minus 1.1) and (y plus 1.1), multiplied by 1.1.
            return ((x - 1.1) @ (y + 1.1)) * 1.1

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

        # check that we do not CSE any variables between prologues, epilogues
        FileCheck().check("def triton").check_count(
            "tl.full([1], 1.1, tl.float32)", 3, exactly=True
        ).check("tl.store").run(code[0])

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/152221")
    @config.patch(
        {
            "max_autotune_gemm_backends": "Triton",
            "benchmark_epilogue_fusion": True,
            "max_epilogue_benchmarked_choices": 3,
        }
    )
    @parametrize("use_async_compile", (True, False))
    def test_pending_fusions_multiple(self, use_async_compile: bool):
        # [FUNCTION DOC] With mocked fusion benchmarks and benchmark epilogue fusion on (3
        # choices), checks the scheduler handles several pending fusions. A product of two
        # transposed-self matmuls must launch exactly two kernels; a matmul followed by relu must
        # launch one. Skipped on ROCm.
        def multi_use(x, y):
            # [FUNCTION DOC] HELPER: product of x times its transpose and y times its transpose.
            return (x @ x.T) * (y @ y.T)

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([128, 16], device=GPU_TYPE)
        y = torch.rand([128, 32], device=GPU_TYPE)

        # Mock benchmarks to keep this pending-fusion test focused on scheduler
        # behavior instead of noisy GPU microbenchmarks.
        benchmark_patches = (
            mock.patch.object(
                Scheduler,
                "benchmark_fused_nodes",
                return_value=(1.0, ""),
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_codegened_module",
                return_value=(0.5, ""),
            ),
        )

        with contextlib.ExitStack() as stack:
            for patcher in benchmark_patches:
                stack.enter_context(patcher)

            out, code = run_and_get_code(torch.compile(multi_use), x, y)

            FileCheck().check(get_func_call()).check_count(
                get_kernel_launch(), 2, exactly=True
            ).run(code[0])
            self.assertEqual(out, multi_use(x, y), atol=0.05, rtol=0.05)

            def resolve_pending(x):
                # [FUNCTION DOC] HELPER: matmul of x with itself followed by relu.
                return (x @ x).relu()

            x = torch.rand([128, 128], device=GPU_TYPE)
            out, code = run_and_get_code(torch.compile(resolve_pending), x)
            FileCheck().check(get_func_call()).check_count(
                get_kernel_launch(), 1, exactly=True
            ).run(code[0])
            self.assertEqual(out, resolve_pending(x), atol=0.05, rtol=0.05)

    @config.patch(
        {
            "max_autotune_gemm_backends": "Triton",
            "benchmark_epilogue_fusion": True,
            "max_epilogue_benchmarked_choices": 3,
        }
    )
    @parametrize("use_async_compile", (True, False))
    def test_pending_fusion_pro_and_epi(self, use_async_compile: bool):
        # [FUNCTION DOC] With epilogue fusion benchmarking enabled (3 choices), checks a float16
        # to float32 cast prologue and relu epilogue around a matmul fuse into a single kernel and
        # match eager.
        def test_multiple_fusions(x):
            # [FUNCTION DOC] HELPER: casts x to float32, matmuls it with itself and applies relu.
            y = x.to(torch.float)
            return (y @ y).relu()

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([128, 128], dtype=torch.float16, device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(test_multiple_fusions), x)
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 1, exactly=True
        ).run(code[0])
        self.assertEqual(out, test_multiple_fusions(x), atol=0.05, rtol=0.05)

    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    @parametrize("use_async_compile", (True, False))
    def test_multiple_inputs(self, sizes, use_async_compile: bool):
        # [FUNCTION DOC] Checks a prologue combining two fp16 inputs by addition, cast to float32
        # and matmul with a third input: results match eager within tolerance with one kernel, one
        # allocation and three deletions. Over three sizes and with or without async pool.
        M, K, N = sizes

        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: adds x and y, casts to float32, matmuls with z.
            return (x + y).to(torch.float) @ z

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        y = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        z = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)
        out_eager = foo(x, y, z)
        out, code = run_and_get_code(torch.compile(foo), x, y, z)
        self.assertEqual(out, out_eager, atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=3)

    def test_storage_offset_prologue(self):
        # [FUNCTION DOC] Checks prologue fusion with sliced inputs having storage offsets: two
        # row-halves of one tensor are shifted and matmul'd; result matches eager with one kernel,
        # one allocation, one deletion.
        def foo(a):
            # [FUNCTION DOC] HELPER: splits a into top and bottom halves, adds 2 to the top,
            # subtracts 2 from the bottom and matmuls them.
            q = a[:64, :]
            k = a[64:, :]
            return torch.mm(q + 2, k - 2)

        inp = torch.randn(128, 64, device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(foo), inp)
        self.assertEqual(out, foo(inp), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=1)

    @config.patch(realize_reads_threshold=1, realize_opcount_threshold=1)
    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    @parametrize("use_async_compile", (True, False))
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_prologue_multiple_nodes(self, sizes, use_async_compile: bool):
        # [FUNCTION DOC] With low realization thresholds, checks a chain of several pointwise
        # prologue ops on both operands fused into one kernel with an epilogue multiply; numerics
        # match eager and one kernel, one allocation, two deletions. Skipped for native matmul.
        M, K, N = sizes

        def foo(x, y):
            # [FUNCTION DOC] HELPER: matmul of an affine transform of x and a scaled y, times 3.
            return ((((x * 2) - 1) / 2) @ (y * 4)) * 3.0

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    @parametrize("K", (63, 64))
    def test_broadcast_x(self, K):
        # [FUNCTION DOC] Compiles dynamically a matmul whose first operand is a broadcast scalar-
        # like tensor expanded to the row count of y, plus one, for K of 63 and 64; checks
        # numerics and one kernel, one allocation and two deletions.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: expands x to a row of the first dim of y, adds one and
            # matmuls with y.
            return (x.expand([1, y.shape[0]]) + 1) @ y

        x = torch.rand([1, 1], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, 128], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo, dynamic=True), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    def test_broadcast_y(self):
        # [FUNCTION DOC] Compiles dynamically (first dim of x marked dynamic) a matmul with K and
        # N equal to one; checks numerics and one kernel, one allocation, two deletions.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return x @ y

        M = 20
        N = K = 1
        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)

        out, code = run_and_get_code(torch.compile(foo, dynamic=True), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_preserves_zero_analysis(self):
        # [FUNCTION DOC] Checks the zero-preservation analysis used for prologue masking: relu
        # preserves zero so no where mask appears in the K loop, while adding one and an unhandled
        # hypot op produce a masked where before the dot. Numerics match eager and counts are one
        # kernel, one allocation, two deletions. Skipped for native matmul.
        fns = (
            (lambda x: x.relu(), False),  # preserves zero
            (lambda x: x + 1, True),  # does not
            (
                lambda x: torch.hypot(x, x),
                True,
            ),  # not handled in analysis, conservatively assume does not preserve
        )

        def foo(x, y, fn):
            # [FUNCTION DOC] HELPER: applies the given function to x and matmuls with y.
            return fn(x) @ y

        for fn, should_mask in fns:
            x = torch.rand([64, 127], dtype=torch.float, device=GPU_TYPE)
            y = torch.rand([127, 64], dtype=torch.float, device=GPU_TYPE)

            out, code = run_and_get_code(torch.compile(foo), x, y, fn)
            self.assertEqual(out, foo(x, y, fn), atol=0.05, rtol=0.05)
            self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

            if should_mask:
                f = FileCheck().check("k_idx").check("a =").check_same("tl.where")
            else:
                f = FileCheck().check("k_idx").check("a =").check_not("tl.where")
            f.check("tl.dot").run(code[0])

    @config.patch(realize_reads_threshold=1, realize_opcount_threshold=1)
    @parametrize("benchmark_fusion", (True, False))
    @parametrize("use_async_compile", (True, False))
    def test_prologue_read_into_both_inputs(
        self, benchmark_fusion, use_async_compile: bool
    ):
        # [FUNCTION DOC] With low realization thresholds, checks a graph where one pointwise
        # result is read as both matmul inputs (unsupported for fusion). Output must match
        # eager; when epilogue benchmarking is off, it requires exactly two kernels without
        # checking allocations.
        M = K = 256

        # not supported today. it could be, but typically the pointwise nodes would get
        # inlined into separate nodes.

        def foo(x):
            # [FUNCTION DOC] HELPER: computes y from x, then matmuls y with y minus two.
            y = (x + 1) * 2
            return y @ (y - 2)

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        with config.patch(benchmark_epilogue_fusion=benchmark_fusion):
            x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)

            out, code = run_and_get_code(torch.compile(foo), x)
            self.assertEqual(out, foo(x), atol=0.05, rtol=0.05)
            # not guaranteed to fuse, but still checking correctness
            if not benchmark_fusion:
                self.check_code(
                    code[0], num_kernels=2, num_allocs=None, num_deallocs=None
                )

    @config.patch(realize_reads_threshold=1, realize_opcount_threshold=1)
    @config.patch(allow_buffer_reuse=False)
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_mismatched_prologue_group(self):
        # [FUNCTION DOC] With buffer reuse disabled and low realization thresholds, checks a
        # prologue with mismatched groups: two kernels, two allocations and four deletions (one
        # more than expected, noted in a comment as a reuse artifact). Skipped for native matmul.
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: scales and multiplies pointwise inputs, then matmuls with z.
            a = (x + 2) * 2
            b = a * y
            return b @ z

        x = torch.rand([1, 256], device=GPU_TYPE)
        y = torch.rand([256, 256], device=GPU_TYPE)
        z = torch.rand([256, 128], device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y, z)
        self.assertEqual(out, foo(x, y, z), atol=0.05, rtol=0.05)
        # there's one more dealloc than there should be because of a buffer reuse. TODO:
        # not sure why disabling buffer reuse doesn't stop
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=4)

    @skipIfXpu
    @config.patch(
        {
            "max_autotune": True,
            "benchmark_epilogue_fusion": True,
            "epilogue_fusion": True,
            "prologue_fusion": True,
        }
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    @parametrize("use_async_compile", (True, False))
    def test_lazy_template_fusion_multiple_candidates(self, use_async_compile: bool):
        # [FUNCTION DOC] Tests lazy fusion with two matmuls with cast and arithmetic prologues and
        # a shared epilogue combining both (relu and tanh). Fusion benchmarks are mocked to fixed
        # numbers; the generated code must contain the template kernel for the first matmul with
        # its prologue and the second with its prologue and epilogue chain. Skipped on XPU and
        # native matmul.
        """
        Test lazy evaluation of template fusions with multiple templates,
        multiple potential prologues, and a shared epilogue.

        This test creates a computation graph with:
        - 2 matmul templates (mm1, mm2)
        - Multiple prologue candidates for each template (type conversions + arithmetic)
        - A shared epilogue candidate that consumes both mm1 and mm2 outputs,
          creating a fusion opportunity with either template

        The lazy evaluation logic should:
        1. Defer fusion decisions until all candidates are identified
        2. Process epilogue fusions before prologue fusions
        3. Cache attempted fusions to avoid redundant compilation
        """

        def foo(a, b, c, d):
            # [FUNCTION DOC] HELPER: computes two matmuls of transformed bfloat16 inputs and
            # applies a chain of multiply, scale, relu and tanh to their product.
            # Prologues for first matmul: transform inputs a and b
            a_transformed = a.to(torch.float) + 1.0
            b_transformed = b.to(torch.float) * 2.0

            # Prologues for second matmul: transform inputs c and d
            c_transformed = c.to(torch.float) - 0.5
            d_transformed = d.to(torch.float) / 2.0

            # Two matmul templates
            mm1 = a_transformed @ b_transformed
            mm2 = c_transformed @ d_transformed

            # Shared epilogue: complex element-wise operations on both mm1 and mm2
            # This creates an epilogue node that could potentially fuse with either template
            # The chain of pointwise ops tests fusion of multiple operations
            combined = mm1 * mm2  # Multiply outputs from both matmuls
            normalized = (combined * 0.5 + 1.0).relu().tanh()

            return normalized

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        M, K, N = 64, 128, 64
        a = torch.rand([M, K], dtype=torch.bfloat16, device=GPU_TYPE)
        b = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)
        c = torch.rand([M, K], dtype=torch.bfloat16, device=GPU_TYPE)
        d = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)

        # Mock benchmarks to return deterministic results so fusion decisions
        # don't depend on noisy GPU microbenchmarks.
        with (
            mock.patch.object(
                Scheduler,
                "benchmark_fused_nodes",
                return_value=(1.0, ""),
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_codegened_module",
                return_value=(0.5, ""),
            ),
        ):
            _, code = run_and_get_code(torch.compile(foo), a, b, c, d)
        FileCheck().check("tem_fused__to_copy_add_mm_mul").check(
            "to_copy_add_div_mm_mul_relu_sub_tanh_1"
        ).run(code[0])

    @config.patch(shape_padding=True)
    @config.patch(force_shape_pad=True)
    @parametrize("sizes", ((250, 245, 128), (250, 256, 128), (256, 128, 62)))
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_prologue_masked_load(self, sizes):
        # [FUNCTION DOC] With shape padding forced, checks that no prologue fusion makes an
        # aligned load unaligned: matmul output matches eager and the code has three kernels,
        # three allocations, four deletions. KNOWN ISSUE: the sizes parameter is unpacked but
        # unused; inputs always have fixed sizes 250 by 245 and 245 by 128, so all
        # parametrizations are the same.
        M, K, N = sizes

        def foo(x, y):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return x @ y

        x = torch.rand([250, 245], device=GPU_TYPE)
        y = torch.rand([245, 128], device=GPU_TYPE)

        # we should not attempt prologue fusion if it turns an aligned load
        # into an unaligned load
        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=3, num_allocs=3, num_deallocs=4)


def autotune_select_algorithm_wrapper_return_multi():
    # [FUNCTION DOC] HELPER: returns a wrapper around the autotune selection function which always
    # sets the flag that returns a multi-template buffer.
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: forces the multi-template return flag in the keyword args and
        # forwards to the real autotune selection function, returning its result.
        kwargs["return_multi_template"] = True
        return autotune_select_algorithm(*args, **kwargs)

    return wrapper


def benchmark_choice_override_timings(benchmark_request, *args, aten_time, triton_time):
    # [FUNCTION DOC] HELPER: fake benchmark function returning a preset time by request type: the
    # aten time for extern kernel requests or callers, the triton time for Triton requests or
    # callers, and infinity for any other type. Extra positional args are ignored.
    if isinstance(
        benchmark_request, (ExternKernelBenchmarkRequest, ExternKernelCaller)
    ):
        return aten_time
    elif isinstance(benchmark_request, (TritonBenchmarkRequest, TritonTemplateCaller)):
        return triton_time
    else:
        return float("inf")


def mock_benchmark_choice_wrapper(aten_time, triton_time):
    # [FUNCTION DOC] HELPER: returns a partial application of the timing override function with
    # the given ATen and Triton times, suitable to patch a benchmark function.
    return functools.partial(
        benchmark_choice_override_timings, aten_time=aten_time, triton_time=triton_time
    )


@instantiate_parametrized_tests
class TestEpilogueFusionStaticAnalysis(TestCase):
    # [CLASS DOC] Tests of the scheduler's static analysis for fusing epilogues into Triton
    # matmul templates without benchmarking (spills, relative timings, register occupancy,
    # memory-bound epilogues). A class-level config patch (kept in an exit stack until
    # tearDownClass) enables max autotune, disables ATen fallback, epilogue benchmarking and
    # prologue fusion. Fusion decisions are steered by patching benchmark timings, register and
    # spill counts and estimated runtimes.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: enters the class-wide config patch into an exit stack.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_fallback_to_aten": False,
                    "benchmark_epilogue_fusion": False,
                    "prologue_fusion": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class fixture: closes the exit stack, undoing the config patch, then
        # the parent.
        cls._stack.close()
        super().tearDownClass()

    @contextlib.contextmanager
    def get_common_patches(
        self,
        async_compile: bool,
        persistent_tma: bool,
        *,
        aten_time: float | None = None,
        triton_time: float | None = None,
        mock_n_spills: int | None = None,
        mock_fused_n_regs: int | None = None,
        mock_unfused_n_regs: int | None = None,
        epilogue_runtime: float | None = None,
    ):
        # [FUNCTION DOC] HELPER: context manager applying the patches shared by these tests. It
        # always sets persistent TMA and the compile thread count (one thread when async compile
        # is off), makes the mm lowering return multi-template buffers, and uses a fresh cache.
        # When both times are given it also mocks choice benchmarking (in process and subprocess).
        # When spill or fused register counts are given, wraps the autotuner precompile to
        # overwrite each launcher's spills and registers. When an unfused register count is given,
        # wraps benchmark request precompile to set its register count. When an epilogue runtime
        # is given, forces the scheduler node runtime estimate. All patches are entered through an
        # exit stack for the with-block duration.
        from torch._inductor.autotune_process import TritonBenchmarkRequest
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.scheduler import BaseSchedulerNode

        common_patches = [
            config.patch(
                {
                    "triton.enable_persistent_tma_matmul": persistent_tma,
                    "compile_threads": 1
                    if not async_compile
                    else config.compile_threads,
                }
            ),
            mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                autotune_select_algorithm_wrapper_return_multi(),
            ),
            fresh_cache(),
        ]

        if aten_time is not None and triton_time is not None:
            common_patches.extend(
                [
                    mock.patch.object(
                        AlgorithmSelectorCache,
                        "benchmark_choice",
                        mock_benchmark_choice_wrapper(aten_time, triton_time),
                    ),
                    mock.patch(
                        "torch._inductor.autotune_process.run_autotune_in_subprocess",
                        mock_benchmark_choice_wrapper(aten_time, triton_time),
                    ),
                ]
            )

        if mock_n_spills is not None or mock_fused_n_regs is not None:
            original_precompile = CachingAutotuner.precompile

            def mock_precompile(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: wraps the autotuner precompile; runs the original,
                # then overwrites the spill count and fused register count (when set) on
                # every launcher.
                original_precompile(self, *args, **kwargs)
                for launcher in self.launchers:
                    if mock_n_spills is not None:
                        launcher.n_spills = mock_n_spills
                    if mock_fused_n_regs is not None:
                        launcher.n_regs = mock_fused_n_regs

            common_patches.append(
                mock.patch.object(CachingAutotuner, "precompile", mock_precompile)
            )

        if mock_unfused_n_regs is not None:
            original_bmreq_precompile = TritonBenchmarkRequest.precompile

            def mock_bmreq_precompile(self):
                # [FUNCTION DOC] HELPER: wraps the benchmark request precompile; runs the
                # original, then overwrites the register count with the mocked unfused value.
                original_bmreq_precompile(self)
                self.n_regs = mock_unfused_n_regs

            common_patches.append(
                mock.patch.object(
                    TritonBenchmarkRequest, "precompile", mock_bmreq_precompile
                )
            )

        if epilogue_runtime is not None:
            common_patches.append(
                mock.patch.object(
                    BaseSchedulerNode,
                    "_get_estimated_runtime",
                    lambda node: epilogue_runtime,
                )
            )

        with contextlib.ExitStack() as stack:
            for p in common_patches:
                stack.enter_context(p)

            yield

    def _get_mm_inputs(self):
        # [FUNCTION DOC] HELPER: returns two random bfloat16 GPU matrices of shapes 512 by 1024
        # and 1024 by 2048.
        """Common matmul inputs for epilogue fusion tests."""
        a = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1024, 2048, device=GPU_TYPE, dtype=torch.bfloat16)
        return a, b

    def _get_mm_with_epilogue_fn(self):
        # [FUNCTION DOC] HELPER: returns a function that matmuls two inputs, casts the result to
        # float32 and adds one.
        """Common function: matmul with type cast and add epilogue."""

        def f(a, b):
            # [FUNCTION DOC] HELPER: matmul of a and b, cast to float32, plus one.
            return (a @ b).to(torch.float32) + 1.0

        return f

    @contextlib.contextmanager
    def _setup_mm_heuristic(self, use_async_compile: bool):
        # [FUNCTION DOC] HELPER: context manager that replaces the CUDA mm heuristic configs with
        # a single 64 by 64 by 32 config, optionally waits for the async compile pool, and
        # restores the original configs afterwards (in a finally block).
        """Setup MM heuristic with single GemmConfig and handle cleanup."""
        mm_heuristic = CUDAMMTemplateConfigHeuristic()
        original_mm_configs = mm_heuristic.mm_configs
        gemm_config = GemmConfig(64, 64, 32, 2, 4, group_m=8)
        mm_heuristic.mm_configs = [gemm_config]

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        try:
            yield
        finally:
            mm_heuristic.mm_configs = original_mm_configs

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support in Triton")
    @skipIfXpu(msg="Bad tma config can be covered by XPU TMA")
    @unittest.skipIf(
        config.cpp_wrapper, "Skip static analysis codegen checks on cpp_wrapper"
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_bad_epilogue_fusion(self, use_async_compile: bool):
        # [FUNCTION DOC] Checks that a failure to compile a fused template-plus-epilogue kernel
        # falls back to unfused kernels safely. Uses the persistent TMA template with one good
        # config, forces the Triton choice to win, and in the failing case wraps the scheduler
        # kernel compile so a multi-node kernel yields a future or module whose compile raises.
        # With simulated failure the code must contain separate mm and pointwise cast kernels;
        # otherwise a fused kernel. Numerics are compared with eager; heuristic configs are
        # restored. Skipped without TMA, on XPU and C++ wrapper.
        def f(a, b):
            # [FUNCTION DOC] HELPER: matmul of a and b, cast to float32.
            return (a @ b).to(torch.float32)

        a = torch.randn(512, 1152, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1152, 7680, device=GPU_TYPE, dtype=torch.bfloat16)

        if GPU_TYPE == "xpu":
            tma_heuristic = XPUPersistentTMATemplateConfigHeuristic()
            mm_heuristic = XPUMMTemplateConfigHeuristic()
        else:
            tma_heuristic = CUDAPersistentTMATemplateConfigHeuristic()
            mm_heuristic = CUDAMMTemplateConfigHeuristic()

        # Save original configs to restore later
        original_tma_mm_configs = tma_heuristic.mm_configs
        original_mm_mm_configs = mm_heuristic.mm_configs

        good_tma_config = GemmConfig(128, 64, 64, 4, 8, group_m=8)
        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        original_compile_kernel = Scheduler.compile_kernel

        for simulate_fusion_failure in [True, False]:
            torch._dynamo.reset()
            tma_heuristic.mm_configs = [good_tma_config]
            # Regular mm template gets no configs
            mm_heuristic.mm_configs = []

            def mock_compile_kernel_fail_fusion(self, nodes, hint_override=None):
                # [FUNCTION DOC] HELPER: replacement for the scheduler compile-kernel method.
                # Calls the original; when failure is simulated and the node group has more than
                # one node, returns a failing result: a lambda-future wrapper whose result raises
                # RuntimeError if there was a future, else a module whose Triton attribute is an
                # object whose precompile raises. Otherwise returns the original pair.
                fut, mod = original_compile_kernel(self, nodes, hint_override)

                if simulate_fusion_failure and len(nodes) > 1:
                    if fut is not None:

                        def failing_result_fn():
                            # [FUNCTION DOC] HELPER: always raises RuntimeError.
                            raise RuntimeError

                        return torch._inductor.codecache.LambdaFuture(
                            failing_result_fn, fut.future
                        ), mod
                    else:

                        class FailingPrecompile:
                            # [CLASS DOC] HELPER: stand-in for a compiled Triton kernel whose
                            # precompile always fails.
                            def precompile(self):
                                # [FUNCTION DOC] HELPER: always raises RuntimeError.
                                raise RuntimeError

                        mod.triton_ = FailingPrecompile()
                        return None, mod

                return fut, mod

            # Different paths:
            # benchmark_epilogue_fusion: True -> always multi_template
            # causes benchmarking always
            # benchmark_epilogue_fusion: False -> TritonTemplateBuffer
            # returns speedup_from_fusion automatically as True
            # What we want: force multi template -> no benchmarking with safety
            try:
                with (
                    self.get_common_patches(use_async_compile, True),
                    mock.patch(
                        "torch._inductor.autotune_process.run_autotune_in_subprocess",
                        mock_benchmark_choice_wrapper(
                            aten_time=float("inf"), triton_time=0.1
                        ),
                    ),
                    mock.patch.object(
                        AlgorithmSelectorCache,
                        "benchmark_choice",
                        mock_benchmark_choice_wrapper(
                            aten_time=float("inf"), triton_time=0.1
                        ),
                    ),
                    mock.patch.object(
                        Scheduler,
                        "compile_kernel",
                        mock_compile_kernel_fail_fusion,
                    ),
                ):
                    compiled_f = torch.compile(f, mode="max-autotune")
                    out, code = run_and_get_code(compiled_f, a, b)

                    if not simulate_fusion_failure:
                        # Fusion should occur
                        FileCheck().check("triton_tem_fused__to_copy_mm").run(code[0])
                    else:
                        # Fusion should fail to occur, unfused kernels
                        FileCheck().check("triton_tem_fused_mm").check(
                            "triton_poi_fused__to_copy"
                        ).run(code[0])

                    torch.testing.assert_close(out, f(a, b), atol=1e-2, rtol=1e-2)
            finally:
                # Restore original configs
                tma_heuristic.mm_configs = original_tma_mm_configs
                mm_heuristic.mm_configs = original_mm_mm_configs

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/179695")
    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @unittest.skipIf(
        config.cpp_wrapper, "Skip static analysis codegen checks on cpp_wrapper"
    )
    @parametrize(
        "test_case",
        [
            "spills_reject",  # High register spillage should reject fusion
            "timing_reject",  # Triton much slower than aten should reject fusion
            "accept_with_triton_faster",  # Low spills and good timing should accept fusion
            "accept_with_aten_faster",  # Fusion even if aten is slightly faster
        ],
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_static_analysis(
        self, test_case: str, use_async_compile: bool
    ):
        # [FUNCTION DOC] Parametrized static analysis fusion cases using mocked timings and
        # spills: heavy spills reject fusion (separate mm and cast kernels), a much slower Triton
        # choice rejects fusion, a faster Triton choice or one only slightly slower than ATen
        # accepts fusion (fused kernel name). Compiles a matmul with cast-and-add epilogue with
        # one fixed mm config and checks the code accordingly. Raises RuntimeError for an unknown
        # case name. CUDA only.
        """
        Test static analysis decisions for matmul epilogue fusions.

        Tests the scheduler logic that decides whether to fuse epilogues without
        benchmarking, based on:
        1. Register spillage (n_spills <= 8 required for fusion)
        2. Runtime comparison (epilogue_runtime + ms_min_choice > choice_timings[choice])
        """
        if test_case == "spills_reject":
            mock_n_spills = 100
            triton_time = 0.1
            aten_time = float("inf")
            expect_fusion = False
        elif test_case == "timing_reject":
            mock_n_spills = 0
            triton_time = 100.0
            aten_time = 0.001
            expect_fusion = False
        elif test_case == "accept_with_triton_faster":
            mock_n_spills = 0
            triton_time = 0.1
            aten_time = float("inf")
            expect_fusion = True
        elif test_case == "accept_with_aten_faster":
            mock_n_spills = 0
            triton_time = 0.1
            aten_time = 0.09999
            expect_fusion = True
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=mock_n_spills,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm_0").run(code[0])
                elif triton_time < aten_time:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])
                else:
                    FileCheck().check_not("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @unittest.skipIf(
        config.cpp_wrapper, "Skip static analysis codegen checks on cpp_wrapper"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize("fuse_epilogue", (True, False))
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_extra_reads(
        self, fuse_epilogue: bool, use_async_compile: bool
    ):
        # [FUNCTION DOC] Checks epilogue fusion that adds extra reads (scale and bias tensors).
        # Chooses a mocked Triton time slightly below or above the break-even point derived from
        # ATen time, epilogue runtime and the extra read fraction, then requires a fused kernel
        # when fusion should be accepted and otherwise no template kernel in the code but an
        # unfused pointwise kernel. Skipped on ROCm and C++ wrapper.
        """Test epilogue fusion with extra reads (bias and scale tensors)."""

        def fn(x, w, bias, scale):
            # [FUNCTION DOC] HELPER: matmul of x and w, multiplied by scale and added with bias.
            out = torch.mm(x, w)
            return out * scale + bias

        torch._dynamo.reset()

        x = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(1024, 2048, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(512, 2048, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(512, 2048, device=GPU_TYPE, dtype=torch.bfloat16)

        epilogue_runtime = 0.5
        aten_time = 1.0
        unfused_time = aten_time + epilogue_runtime
        # 2 times extra bytes / 3
        estimated_fused = epilogue_runtime * 2 / 3

        if fuse_epilogue:
            # triton + 1 read / 2 extra memory ratio * epilogue_runtime
            # < aten_time + epilogue_runtime
            triton_time = unfused_time - estimated_fused - 0.01
        else:
            triton_time = unfused_time - estimated_fused + 0.01

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                epilogue_runtime=epilogue_runtime,
                mock_fused_n_regs=32,
                mock_n_spills=0,
            ):
                compiled_fn = torch.compile(fn)
                _, code = run_and_get_code(compiled_fn, x, w, bias, scale)

                if fuse_epilogue:
                    FileCheck().check("triton_tem_fused_add_mm_mul").run(code[0])
                else:
                    FileCheck().check_not("triton_tem").check(
                        "triton_poi_fused_add_mul"
                    ).run(code[0])

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/179694")
    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/176115")
    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @unittest.skipIf(
        config.cpp_wrapper, "Skip static analysis codegen checks on cpp_wrapper"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize(
        "test_case",
        [
            "occupancy_ratio_accept",  # ratio > 0.5, accept via Branch B
            "occupancy_ratio_reject",  # ratio <= 0.5, reject
        ],
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_occupancy_ratio(
        self, test_case: str, use_async_compile: bool
    ):
        # [FUNCTION DOC] Tests the register occupancy branch of the epilogue fusion decision. With
        # mocked unfused and fused register counts and times, fusion is accepted when the fused to
        # unfused occupancy ratio is above one half (even if ATen is slightly faster) and rejected
        # when below, checked by the fused or separate kernel names in the code. Raises
        # RuntimeError for an unknown case name. Skipped on ROCm and C++ wrapper.
        """
        Test occupancy ratio branch of _fuse_epilogue.

        Occupancy calculation (assuming regs_per_sm = 65536):
        blocks = regs_per_sm // (n_regs * threads_per_block)
        threads_per_block = num_warps * warp_size = 4 * 32 = 128
        """
        triton_time = 0.1
        epilogue_runtime = triton_time

        if test_case == "occupancy_ratio_accept":
            # blocks_unfused=5, blocks_fused=3, ratio=0.6 > 0.5 -> accept
            # aten slightly faster to verify fusion picks triton even when aten wins
            mock_unfused_n_regs, mock_fused_n_regs = 100, 160
            aten_time = 0.09
            expect_fusion = True
        elif test_case == "occupancy_ratio_reject":
            # blocks_unfused=8, blocks_fused=2, ratio=0.25 < 0.5 -> reject
            mock_unfused_n_regs, mock_fused_n_regs = 64, 200
            aten_time = 0.11
            expect_fusion = False
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=0,
                mock_fused_n_regs=mock_fused_n_regs,
                mock_unfused_n_regs=mock_unfused_n_regs,
                epilogue_runtime=epilogue_runtime,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm").run(code[0])
                else:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/176114")
    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/176113")
    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @unittest.skipIf(
        config.cpp_wrapper, "Skip static analysis codegen checks on cpp_wrapper"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize(
        "test_case",
        [
            "memory_bound_accept",
            "memory_bound_reject_low_occupancy",
        ],
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_dominating_epilogue(
        self, test_case: str, use_async_compile: bool
    ):
        # [FUNCTION DOC] Tests the memory-bound epilogue branch of the fusion decision. When
        # occupancy checks fail, fusion is still accepted if the epilogue runtime exceeds twice
        # the template time and more than one block fits per multiprocessor (mocked via register
        # counts); with only one block fits it is rejected. Verified via kernel names in the code.
        # Raises RuntimeError for an unknown case name. Skipped on ROCm and C++ wrapper.
        """
        Test memory-bound epilogue branch of _fuse_epilogue (Branch C).

        When Branches A and B fail (low occupancy), fusion can still be accepted
        if the epilogue is memory-bound (ms2 > 2*ms1) AND blocks_fused > 1.
        """
        triton_time = 0.1

        if test_case == "memory_bound_accept":
            # blocks_fused=2 > 1, ms2=0.3 > 2*ms1=0.2 -> Branch C accepts
            # aten slightly faster to verify fusion picks triton even when aten wins
            mock_unfused_n_regs, mock_fused_n_regs = 64, 256
            aten_time = 0.09
            epilogue_runtime = 0.3
            expect_fusion = True
        elif test_case == "memory_bound_reject_low_occupancy":
            # blocks_fused=1, ms2=0.3 > 2*ms1=0.2 BUT blocks_fused <= 1 -> reject
            mock_unfused_n_regs, mock_fused_n_regs = 64, 512
            aten_time = 0.11
            epilogue_runtime = 0.3
            expect_fusion = False
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=0,
                mock_fused_n_regs=mock_fused_n_regs,
                mock_unfused_n_regs=mock_unfused_n_regs,
                epilogue_runtime=epilogue_runtime,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm").run(code[0])
                else:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @parametrize("use_async_compile", (True, False))
    def test_epilogue_prologue_fusion_cache_preserved(self, use_async_compile: bool):
        # [FUNCTION DOC] Regression test: with both prologue and epilogue fusion possible and
        # prologue heuristics bypassed, compiling a matmul with pointwise ops before and after
        # must not fail with an error from taking the minimum of an empty sequence while choosing
        # a template choice. Only checks compilation succeeds; float32 inputs, ATen time infinite.
        def f(a, b):
            # [FUNCTION DOC] HELPER: adds one to a, matmuls with b and adds two.
            # Prologue: pointwise operation on input 'a' before matmul
            a_transformed = a + 1.0
            # Matmul
            mm_result = a_transformed @ b
            # Epilogue: pointwise operation on output after matmul
            return mm_result + 2.0

        torch._dynamo.reset()

        # Use float32 to avoid low precision heuristic rejection
        a = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(1024, 2048, device=GPU_TYPE, dtype=torch.float32)

        triton_time = 0.1
        aten_time = float("inf")
        epilogue_runtime = 0.05

        # Always allow prologue fusion heuristics
        def always_allow_prologue(*args):
            # [FUNCTION DOC] HELPER: stand-in prologue heuristic check that always returns true.
            return True

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=0,
                epilogue_runtime=epilogue_runtime,
            ):
                # Enable prologue fusion so both epilogue and prologue are considered
                with config.patch(prologue_fusion=True):
                    # Bypass prologue heuristics that might reject the fusion
                    with mock.patch.object(
                        Scheduler,
                        "check_prologue_fusion_heuristics_fusable",
                        always_allow_prologue,
                    ):
                        compiled_f = torch.compile(f)
                        # If the bug exists, this will fail with:
                        # "ValueError: min() arg is an empty sequence"
                        # when get_min_choice() is called during prologue fusion
                        run_and_get_code(compiled_f, a, b)


def simple_fn():
    # [FUNCTION DOC] HELPER: trivial picklable function returning 42, submitted to worker pools.
    return 42


class TestMaxAutotuneAsyncPipelined(TestMaxAutotune, TestEpilogueFusionStaticAnalysis):
    # [CLASS DOC] Re-runs all max autotune and epilogue fusion static analysis tests with
    # pipelined asynchronous autotuning turned on (single mm config, no epilogue benchmarking).
    # Inherits both test classes. SKIP_TESTS maps test name fragments to skip reasons. Adds tests
    # of the pool shutdown on inactivity and error fallback. Class fixtures enter/exit the config
    # patch; setUp and tearDown reset the process pool and the async autotuner future cache.
    """Tests for AsyncPipelinedAutotuning path."""

    SKIP_TESTS = {
        "test_inf_timing": "Logs not consistent with async pipelined autotuning",
        "test_non_contiguous_input_mm_plus_mm": "Flaky on trunk",
        "test_autotune_device_guard": "Flaky on trunk",
        "test_template_bad_epilogue_fusion": "Benchmarking path is different",
        "test_persistent_tma_epilogue_fusion_store_cache": "Epilogue fusion disabled in async pipelining",
    }

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture: calls the parent fixture, then enters a config patch
        # enabling pipelined GEMM autotuning, disabling epilogue benchmarking and limiting mm
        # configs to 1. The patch object is stored on the class for exit later.
        super().setUpClass()
        cls._async_config = config.patch(
            {
                "pipeline_max_autotune_gemm": True,
                "benchmark_epilogue_fusion": False,
                "test_configs.max_mm_configs": 1,
            }
        )
        cls._async_config.__enter__()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class fixture: exits the pipelined autotune config patch, then calls
        # the parent.
        cls._async_config.__exit__(None, None, None)
        super().tearDownClass()

    def setUp(self):
        # [FUNCTION DOC] Resets the pool inactivity-shutdown flag, then skips the test if any
        # name in SKIP_TESTS is a substring of this test's name, or if running on XPU or with the
        # C++ wrapper.
        # KNOWN ISSUE: the XPU and C++ wrapper conditions are inside the loop over skip names,
        #   so they skip every test with the reason of the first listed entry rather than a
        #   matching one.
        super().setUp()
        AutotuneProcessPool._shutdown_for_inactivity = False
        test_name = self._testMethodName
        for skip_test_name in self.SKIP_TESTS:
            if skip_test_name in test_name or TEST_XPU or config.cpp_wrapper:
                self.skipTest(self.SKIP_TESTS[skip_test_name])

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown, shuts down the autotune process pool
        # singleton, clears the inactivity flag and the future cache of the async autotuner to
        # avoid cross-test leaks.
        super().tearDown()
        AutotuneProcessPool.shutdown_instance()
        AutotuneProcessPool._shutdown_for_inactivity = False
        # Clear the AsyncAutotuner cache to prevent test pollution
        AsyncAutotuner.choice_hash_to_future.clear()

    @config.patch(max_autotune_gemm=True)
    def test_async_autotuner_cache_same_inputs(self):
        # [FUNCTION DOC] Compiles three matmuls, two with identical shapes and one different, and
        # compares with eager. Then requires the async autotuner future cache to have exactly four
        # entries. KNOWN ISSUE: the failure message says two entries but the assertion expects
        # four (one ATen and one Triton choice per unique shape).
        M, K, N = 128, 64, 256
        M2, K2, N2 = 256, 128, 64

        def three_matmuls(a1, b1, a2, b2, a3, b3):
            # [FUNCTION DOC] HELPER: returns the three matrix products of the pairs of inputs as
            # a tuple.
            return torch.mm(a1, b1), torch.mm(a2, b2), torch.mm(a3, b3)

        # Same shapes for first two matmuls
        a1 = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b1 = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)
        a2 = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b2 = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)

        # Different shapes for third matmul
        a3 = torch.randn(M2, K2, device=GPU_TYPE, dtype=torch.bfloat16)
        b3 = torch.randn(K2, N2, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled_fn = torch.compile(three_matmuls)
        result = compiled_fn(a1, b1, a2, b2, a3, b3)

        # Verify correctness
        expected = three_matmuls(a1, b1, a2, b2, a3, b3)
        for r, e in zip(result, expected):
            torch.testing.assert_close(r, e, atol=1e-2, rtol=1e-2)

        # With max_mm_configs=1, we get 2 configs total (1 per unique shape)
        # First two matmuls share the same shape, third has different shape
        # 1 aten, 1 triton config
        cache_size = len(AsyncAutotuner.choice_hash_to_future)
        self.assertEqual(
            cache_size, 4, "Cache should have 2 entries (one per unique input shape)"
        )

    @patch(
        "torch._inductor.autotune_process.AUTOTUNE_POOL_INACTIVITY_TIMEOUT",
        2,
    )
    def test_autotune_process_pool_inactivity_shutdown(self):
        # [FUNCTION DOC] With the inactivity timeout patched to 2 seconds, gets the pool
        # singleton, warms it up, submits a trivial task and checks its result; then sleeps 5
        # seconds and requires the internal worker pool and timer be released and the inactivity
        # flag be set.
        AutotuneProcessPool.shutdown_instance()
        AutotuneProcessPool._shutdown_for_inactivity = False

        pool_instance = AutotuneProcessPool.get_instance()
        pool_instance.warm_up()

        future = pool_instance.submit(simple_fn)
        result = future.result()
        self.assertEqual(result, 42)
        self.assertIsNotNone(pool_instance._pool)

        time.sleep(5)

        self.assertIsNone(pool_instance._pool)
        self.assertIsNone(pool_instance._timer)
        self.assertTrue(AutotuneProcessPool._shutdown_for_inactivity)

    @patch(
        "torch._inductor.autotune_process.AUTOTUNE_POOL_INACTIVITY_TIMEOUT",
        2,
    )
    def test_autotune_process_pool_inactivity_shutdown_warmup_only(self):
        # [FUNCTION DOC] Like the inactivity shutdown test but only warms up the pool (waiting
        # for the warmup future); after 5 seconds the pool and timer must be released and the
        # flag set.
        """Test that the pool shuts down from inactivity even when only warmup is called."""
        AutotuneProcessPool.shutdown_instance()
        AutotuneProcessPool._shutdown_for_inactivity = False

        pool_instance = AutotuneProcessPool.get_instance()
        warmup_future = pool_instance.warm_up()
        warmup_future.result()

        self.assertIsNotNone(pool_instance._pool)

        time.sleep(5)

        self.assertIsNone(pool_instance._pool)
        self.assertIsNone(pool_instance._timer)
        self.assertTrue(AutotuneProcessPool._shutdown_for_inactivity)

    @patch(
        "torch._inductor.autotune_process.AUTOTUNE_POOL_INACTIVITY_TIMEOUT",
        2,
    )
    @config.patch(max_autotune=True)
    def test_compilation_after_inactivity(self):
        # [FUNCTION DOC] With a short inactivity timeout and max autotune on, resets state,
        # compiles a matmul and checks the pipelined path filled the future cache; after waiting
        # 5 seconds the pool must have shut down so pipelined autotuning is disabled; then
        # compiling another matmul with a cleared cache must leave the future cache empty
        # (synchronous path).
        """Test that compilation after pool inactivity shutdown uses synchronous path."""

        # Reset state
        AutotuneProcessPool.shutdown_instance()
        AutotuneProcessPool._shutdown_for_inactivity = False
        AsyncAutotuner.choice_hash_to_future.clear()
        torch._dynamo.reset()

        # First compilation - should use pipelined path
        self.assertTrue(use_pipelined_autotuning())

        def matmul_fn(a, b):
            # [FUNCTION DOC] HELPER: matrix product of the two inputs.
            return torch.mm(a, b)

        a1 = torch.randn(64, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        b1 = torch.randn(32, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        a2 = torch.randn(128, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        b2 = torch.randn(64, 128, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled_fn = torch.compile(matmul_fn)
        compiled_fn(a1, b1)

        # Verify pipelined path was used (cache should have entries)
        cache_entries_after_first = len(AsyncAutotuner.choice_hash_to_future)
        self.assertGreater(cache_entries_after_first, 0)

        # Wait for inactivity shutdown
        time.sleep(5)

        self.assertTrue(AutotuneProcessPool._shutdown_for_inactivity)
        self.assertFalse(use_pipelined_autotuning())

        AsyncAutotuner.choice_hash_to_future.clear()
        torch._dynamo.reset()

        compiled_fn2 = torch.compile(matmul_fn)
        compiled_fn2(a2, b2)

        cache_entries_after_second = len(AsyncAutotuner.choice_hash_to_future)
        self.assertEqual(cache_entries_after_second, 0)

    @config.patch(max_autotune_gemm=True)
    def test_triton_error_precompilation_and_autotuning(self):
        # [FUNCTION DOC] Checks fallback to extern kernels when Triton autotuning fails. First
        # patches the autotuning step to raise NoValidChoicesError; compiling mm with and without
        # a pointwise epilogue must yield code with no Triton template kernel. Then patches the
        # async autotuner start to give every Triton choice a benchmark request that raises in the
        # subprocess, and repeats the check.
        """
        Test error handling when do_autotuning throws NoValidChoicesError
        for Triton choices. The fallback to extern kernels should still work.
        """

        def mock_do_autotuning(*args, **kwargs):
            # [FUNCTION DOC] HELPER: ignores arguments and raises NoValidChoicesError.
            raise NoValidChoicesError("Simulated: all Triton choices failed")

        a = torch.randn(64, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(32, 64, device=GPU_TYPE, dtype=torch.bfloat16)

        def mm_func(a, b, epilogue):
            # [FUNCTION DOC] HELPER: matrix product of a and b, plus one when the epilogue flag
            # is true.
            if epilogue:
                return torch.mm(a, b) + 1.0
            else:
                return torch.mm(a, b)

        def test_aten_chosen():
            # [FUNCTION DOC] HELPER: for each epilogue setting, resets dynamo, compiles the mm
            # function and checks the generated code contains no Triton template kernel.
            for epilogue in (True, False):
                torch._dynamo.reset()
                compiled_fn = torch.compile(mm_func)
                out, code = run_and_get_code(compiled_fn, a, b, epilogue)
                FileCheck().check_not("triton_tem").run(code[0])

        with mock.patch.object(
            AlgorithmSelectorCache, "do_autotuning", mock_do_autotuning
        ):
            test_aten_chosen()

        original_start = AsyncAutotuner.start
        bmreq = _TestBenchmarkRequest(
            exc=RuntimeError("Simulated benchmark failure in subprocess")
        )
        bmreq.module_cache_key = ""

        def mock_start(choices, inputs_key):
            # [FUNCTION DOC] HELPER: wrapper for the async autotuner start function; attaches
            # the failing benchmark request to every Triton template caller, then calls the
            # original start.
            for choice in choices:
                if isinstance(choice, TritonTemplateCaller):
                    choice.bmreq = bmreq
            return original_start(choices, inputs_key)

        with mock.patch.object(AsyncAutotuner, "start", mock_start):
            test_aten_chosen()


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
