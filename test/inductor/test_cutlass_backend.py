# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the CUTLASS (and Intel XPU SYCL-TLA) GEMM backend of Inductor max-
# autotune. Covers mm, addmm, bmm, int8 mm, FP8 scaled mm, sparse semi-structured mm, stream-K
# kernels, epilogue visitor tree (EVT) fusion, AOTI export, op allow/deny lists, serialization of
# ops, alignment/layout helpers and autotune caching behavior, on CUDA and XPU devices. Import-
# time side effects: may export the CUTLASS directory environment variable from the source
# checkout, may disable expandable CUDA allocator segments, and builds module-level config patch
# objects and parametrize decorators used by the test class.
import gc
import itertools
import logging
import math
import os
import re
import sysconfig
import time
import unittest
import unittest.mock as mock
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import torch


_CUTLASS_DIR_ENV_VAR = "TORCHINDUCTOR_CUTLASS_DIR"


def _maybe_set_cutlass_dir_env_from_checkout() -> None:
    # [FUNCTION DOC] HELPER: Point the CUTLASS directory environment variable at the checkout.
    # Does nothing if the variable is already set or if the default directory next to the
    # installed torch package contains a python subdirectory. Otherwise, if the source checkout
    # has a CUTLASS third_party directory with a python subdirectory, its path is stored in the
    # process environment.
    # Set this before importing torch._inductor.config below. This matters when
    # running source-checkout tests against an installed torch package.
    if _CUTLASS_DIR_ENV_VAR in os.environ:
        return

    default_cutlass_dir = (
        Path(torch.__file__).resolve().parent.parent / "third_party" / "cutlass"
    )
    if (default_cutlass_dir / "python").is_dir():
        return

    checkout_cutlass_dir = (
        Path(__file__).resolve().parents[2] / "third_party" / "cutlass"
    )
    if (checkout_cutlass_dir / "python").is_dir():
        os.environ[_CUTLASS_DIR_ENV_VAR] = str(checkout_cutlass_dir)


_maybe_set_cutlass_dir_env_from_checkout()

from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.codegen.cutlass.serialization import (
    get_cutlass_operation_serializer,
)
from torch._inductor.utils import clear_caches
from torch.export import Dim
from torch.testing._internal.common_utils import random_matrix_with_scaled_reduction_dim
from torch.testing._internal.logging_utils import log_settings
from torch.utils import _pytree as pytree


try:
    from test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from .test_aot_inductor_utils import AOTIRunnerUtil

import torch._inductor.codecache
import torch.version
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import XPUCodeCache
from torch._inductor.codegen.cutlass.kernel import CUTLASSTemplateCaller
from torch._inductor.codegen.cutlass.utils import (
    _gen_ops_cached,
    get_max_alignment,
    try_import_cutlass,
)
from torch._inductor.exc import InductorError
from torch._inductor.ir import FixedLayout
from torch._inductor.select_algorithm import NoValidChoicesError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    SM100OrLater,
    SM120OrLater,
    SM80OrLater,
    SM90OrLater,
    xfailIfSM120OrLater,
)
from torch.testing._internal.common_device_type import skipCUDAIf, skipXPUIf
from torch.testing._internal.common_utils import (
    IN_RE_WORKER,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.common_xpu import Xe2_Or_Later
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
)


# We don't need triton in this test suite.
HAS_XPU = torch.xpu.is_available()
HAS_CUDA = torch.cuda.is_available()
HAS_GPU = HAS_CUDA or HAS_XPU

if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] HELPER: Module-level fixture. Saves the current float32 matmul precision in
    # a module global and sets the precision to "high" for all tests in this module.
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")


def tearDownModule():
    # [FUNCTION DOC] HELPER: Module-level fixture. Restores the float32 matmul precision saved by
    # the module setup fixture and clears the saved global; does nothing if nothing was saved.
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


log = logging.getLogger(__name__)


def _get_path_without_sccache() -> str:
    # [FUNCTION DOC] HELPER: Return the PATH environment variable with every entry containing the
    # sccache directory removed, rejoined with colons. Used so compilations bypass sccache.
    """
    Get the PATH environment variable without sccache.
    """
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


def _get_cutlass_dir() -> str:
    # [FUNCTION DOC] HELPER: Return the configured CUTLASS directory: the XPU config value on
    # XPU, otherwise the CUDA CUTLASS config value.
    return config.xpu.cutlass_dir if GPU_TYPE == "xpu" else config.cutlass.cutlass_dir


def _check_if_instances_equal(op1, op2) -> bool:
    # [FUNCTION DOC] HELPER: Recursively compare two CUTLASS objects for equality. Lists and
    # tuples are compared as tuples (CUTLASS mixes them); different types are unequal; a custom
    # equality method, if defined on the class, must report equality; enums compare by value;
    # otherwise every instance attribute of the first object must exist in the second and compare
    # equal recursively. Returns a bool.
    """
    Utility function to check if two instances of a class are equal.
    """
    # cutlass uses list and tuple inconsistently
    if isinstance(op1, (list | tuple)):
        return tuple(op1) == tuple(op2)

    if type(op1) is not type(op2):
        return False

    # some classes have __eq__ defined but they may be insufficient
    if op1.__class__.__dict__.get("__eq__") and op1 != op2:
        return False

    if isinstance(op1, Enum):
        return op1.value == op2.value

    if hasattr(op1, "__dict__"):
        for key, value in op1.__dict__.items():
            if key not in op2.__dict__:
                return False
            if not _check_if_instances_equal(value, op2.__dict__[key]):
                return False

    return True


un_ops_under_test = [torch.relu, torch.tanh, torch.exp, torch.sigmoid]
bin_ops_under_test = [torch.add, torch.mul, torch.sub, torch.div]

evt_all_ops = parametrize(
    "op", un_ops_under_test + bin_ops_under_test, name_fn=lambda f: f.__name__
)

evt_un_ops = parametrize("op", un_ops_under_test, name_fn=lambda f: f.__name__)

evt_bin_ops = parametrize("op", bin_ops_under_test, name_fn=lambda f: f.__name__)

evt_all_shapes = parametrize("shape", itertools.product([512, 1024], repeat=2))


def gen_args(op, shape, dtype=torch.float16):
    # [FUNCTION DOC] HELPER: Build the extra argument tuple for an EVT test. For binary ops
    # returns a one-element tuple holding a random tensor of the given shape and dtype (default
    # half) on the first GPU; for unary ops returns an empty tuple.
    if op in bin_ops_under_test:
        return (torch.rand(*shape, device=f"{GPU_TYPE}:0", dtype=dtype),)
    else:
        return ()


use_evt_config = config.patch(
    {
        "max_autotune": True,
        "max_autotune_gemm_backends": "CUTLASS",
        "cutlass.cutlass_max_profiling_configs": 1,
        "benchmark_epilogue_fusion": False,  # EVT doesn't support benchmark fusion yet
        "cutlass.cutlass_tma_only": GPU_TYPE
        == "cuda",  # Only CUDA requires TMA for EVT.
        "cutlass.cutlass_epilogue_fusion_enabled": True,
    }
)

fp8_config = config.patch(
    {
        "max_autotune": True,
        "max_autotune_gemm_backends": "CUTLASS",
        "cutlass.cutlass_max_profiling_configs": 1,
        "benchmark_epilogue_fusion": False,  # EVT doesn't support benchmark fusion yet
        "cutlass.cutlass_tma_only": True,
    }
)


def select_no_algorithm(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the algorithm selection entry point used with mock
    # wraps. Ignores all arguments and always raises NoValidChoicesError, so tests can inspect
    # the choices passed in without precompiling or autotuning.
    """
    Utility function to skip precompilation and autotuning.
    """
    raise NoValidChoicesError


@instantiate_parametrized_tests
class TestCutlassBackend(TestCase):
    # [CLASS DOC] Test case for the CUTLASS backend of max-autotune GEMMs. Tests are skipped when
    # no GPU is present or on HIP. The class attribute device_type is set to the GPU type so
    # device skip decorators work. Contains tests for mm/addmm/bmm/int_mm/FP8/sparse kernels,
    # stream-K, epilogue fusion (EVT), AOTI, op allow/deny lists, caching, and more. Decorated so
    # parametrized tests are expanded.
    # device_type of each test case is necessary for skipCUDAIf decorator.
    device_type = GPU_TYPE

    def setUp(self):
        # [FUNCTION DOC] Skip when no GPU is available or on ROCm. Temporarily sets the
        # environment flag that disables automatic fresh-cache refresh to "1" while calling the
        # parent setUp (the refresh interacts badly with persistent autotune subprocesses), then
        # restores the previous value (an empty string if it was unset), and seeds the random
        # generator with 1234.
        if not HAS_GPU:
            self.skipTest(f"{GPU_TYPE} and triton are not available")
        if torch.version.hip:
            self.skipTest("CUTLASS backend is not supported on HIP")

        # The new inductor cache refresh mechanism
        # introduced with https://github.com/pytorch/pytorch/pull/122661
        # interacts badly with persistent subprocesses during
        # autotuning. So we need to disable automatic cache refresh
        # before calling setUp() on the parent class.
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )
        try:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            super().setUp()
        finally:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = (
                old_disable_fresh_cache_envvar
            )
        torch.random.manual_seed(1234)

    def tearDown(self):
        # [FUNCTION DOC] Run parent tearDown and clear inductor caches. On XPU additionally
        # closes and clears all cached XPU code-cache shared libraries, runs garbage collection
        # and empties the XPU memory cache.
        super().tearDown()
        clear_caches()
        if GPU_TYPE == "xpu":
            for dll in XPUCodeCache.dll_cache.values():
                dll.close()
            XPUCodeCache.dll_cache.clear()
            gc.collect()
            torch.xpu.empty_cache()

    def run_evt_test(self, model, op, shape, num_fusions=1):
        # [FUNCTION DOC] Shared EVT test driver. Builds all-ones half inputs of the given 2D
        # shape on the GPU (second operand transposed), compiles the model and compares the
        # result with eager. Asserts the epilogue fusion counter equals num_fusions and results
        # are close.
        M, N = shape
        a = torch.ones(M, N).to(GPU_TYPE).half()
        b = torch.ones(N, N).to(GPU_TYPE).half().t()
        extra_args = gen_args(op, (M, N))
        model = model.to(GPU_TYPE)

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            num_fusions,
        )
        torch.testing.assert_close(result, ref_result)

    def test_check_paths(self):
        # [FUNCTION DOC] Checks that the mock import directories for cuda, pydot and scipy used
        # by the CUTLASS codegen exist inside the installed torch package.
        cutlass_mock_imports_path = os.path.join(
            os.path.dirname(torch.__file__),
            "_inductor/codegen/cutlass/lib_extensions/cutlass_mock_imports",
        )
        cutlass_mock_cuda_path = os.path.join(cutlass_mock_imports_path, "cuda")
        cutlass_mock_pydot_path = os.path.join(cutlass_mock_imports_path, "pydot")
        cutlass_mock_scipy_path = os.path.join(cutlass_mock_imports_path, "scipy")
        self.assertTrue(os.path.exists(cutlass_mock_imports_path))
        self.assertTrue(os.path.exists(cutlass_mock_cuda_path))
        self.assertTrue(os.path.exists(cutlass_mock_pydot_path))
        self.assertTrue(os.path.exists(cutlass_mock_scipy_path))

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_threshold(self):
        # [FUNCTION DOC] Checks the minimum GEMM size threshold. Sets the threshold one above the
        # product of the problem dimensions so no CUTLASS choice is accepted, compiles a matmul
        # with the selection function replaced by one that raises, expects an InductorError
        # mentioning NoValidChoicesError, and verifies the choices passed to selection are empty.
        """
        Make sure Cutlass GEMM threshold works as intended.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix multiply of two tensors.
            return a @ b

        a = torch.randn(100, 10).to(GPU_TYPE).half()
        b = torch.randn(100, 10).to(GPU_TYPE).half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                # Make it slightly too large to be accepted (m*n*k)
                "cutlass.cutlass_backend_min_gemm_size": 100 * 100 * 10 + 1,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            with mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                wraps=select_no_algorithm,
            ) as sa:
                with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                    _ = torch.compile(mm, dynamic=False)(a, b)
                args, _ = sa.call_args
                _, choices, _, __ = args

                self.assertEqual(choices, [])

    @skipXPUIf(not Xe2_Or_Later, "")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_import_cutlass(self):
        # [FUNCTION DOC] Checks that CUTLASS can be imported through the helper and that the
        # cutlass_cppgen and cutlass_library modules import successfully.
        self.assertTrue(try_import_cutlass())

        import cutlass_cppgen  # type: ignore[import-not-found]  # noqa: F401
        import cutlass_library  # noqa: F401

    @skipXPUIf(not Xe2_Or_Later, "")
    def test_cutlass_key(self):
        # [FUNCTION DOC] Checks CUTLASS is importable and that computing the CUTLASS cache key
        # returns a non-None value.
        self.assertTrue(try_import_cutlass())
        from torch._inductor.codecache import cutlass_key

        self.assertIsNotNone(cutlass_key())

    @skipXPUIf(True, "CUDA-specific CUTLASS arch feature set")
    @parametrize("arch", ("103", "107"))
    def test_sm103_and_sm107_cutlass_ops_skip_int8_umma(self, arch):
        # [FUNCTION DOC] For architectures 103 and 107, generates the cached CUTLASS operation
        # list for CUDA, flattens it, and checks some operations exist but none has an int8 by
        # int8 to int32 configuration name, i.e. int8 UMMA kernels are skipped.
        from torch.utils import _pytree as pytree

        self.assertTrue(try_import_cutlass())

        ops = pytree.tree_flatten(_gen_ops_cached(arch, "13.3", "cuda"))[0]
        cutlass_ops = [op for op in ops if hasattr(op, "configuration_name")]
        int8_ops = [op for op in cutlass_ops if "s8_s8_s32" in op.configuration_name()]
        self.assertGreater(len(cutlass_ops), 0)
        self.assertEqual(int8_ops, [])

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_subproc_mm(self):
        # [FUNCTION DOC] Autotunes a large half precision mm in a subprocess with the CUTLASS
        # backend and compares with eager. Fails with an informative message if CUTLASS cannot be
        # imported. Inputs are scaled by the reduction dimension to limit accumulation error; XPU
        # uses looser tolerances.
        """
        Test autotune_in_subproc works for mm.

        NOTE: Shape like M, N, K = 100, 100, 10 would get filtered out due to
        alignment mismatch.
        """

        M, N, K = 4096, 2048, 25728

        cutlass_dir = _get_cutlass_dir()
        self.assertTrue(
            try_import_cutlass(),
            lambda msg: f"{msg}\n"
            + (
                "CUTLASS backend is required by this test but could not be imported. "
                f"Set {_CUTLASS_DIR_ENV_VAR} to a valid CUTLASS checkout; "
                f"current cutlass_dir={cutlass_dir!r}."
            ),
        )

        # Scale inputs by 1/sqrt(K) to avoid large accumulation differences
        # between CUTLASS and ATen in half precision.
        a = random_matrix_with_scaled_reduction_dim(
            M, K, dtype=torch.float16, device=GPU_TYPE, reduction_dim=-1
        )
        b = random_matrix_with_scaled_reduction_dim(
            N, K, dtype=torch.float16, device=GPU_TYPE, reduction_dim=-1
        ).t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cutlass.cutlass_max_profiling_configs": 4,
            }
        ):
            Y_compiled = torch.compile(torch.mm)(a, b)
            Y = torch.mm(a, b)
            if GPU_TYPE == "xpu":
                atol = 1e-3  # default is 1e-5
                rtol = 1e-3  # default is 1e-3
            else:
                atol = None
                rtol = None

            self.assertEqual(Y_compiled, Y, atol=atol, rtol=rtol)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_cutlass_backend_subproc_addmm(self, dtype):
        # [FUNCTION DOC] Autotunes addmm with CUTLASS in a subprocess for a given dtype, looping
        # over four bias shapes (full, column, row, vector) with alpha 2.0 and beta 0.4. Resets
        # dynamo and caches per shape and compares to eager; XPU uses looser tolerances, bf16
        # looser than fp16.
        """
        Test autotune_in_subproc works for addmm.
        """

        M, N, K = 4096, 2048, 25728

        # Scale inputs by 1/sqrt(K) so that the matmul output has O(1)
        # magnitude, avoiding large accumulation errors in half precision
        # that would require loose tolerances.
        a = random_matrix_with_scaled_reduction_dim(
            M, K, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
        )
        b = random_matrix_with_scaled_reduction_dim(
            N, K, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
        ).t()

        x_shapes = [
            (M, N),
            (M, 1),
            (1, N),
            (N,),
        ]

        alpha = 2.0
        beta = 0.4

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cutlass.cutlass_max_profiling_configs": 4,
            }
        ):
            for x_shape in x_shapes:
                torch._dynamo.reset()
                clear_caches()

                x = torch.randn(x_shape).to(GPU_TYPE).to(dtype)
                Y_compiled = torch.compile(torch.addmm)(x, a, b, alpha=alpha, beta=beta)
                Y = torch.addmm(x, a, b, alpha=alpha, beta=beta)
                if GPU_TYPE == "xpu":
                    if dtype == torch.bfloat16:
                        atol = 5e-3
                        rtol = 5e-3
                    else:
                        atol = 1e-3
                        rtol = 1e-3
                else:
                    # use default
                    atol = None
                    rtol = None
                torch.testing.assert_close(Y_compiled, Y, atol=atol, rtol=rtol)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_subproc_bmm(self):
        # [FUNCTION DOC] Autotunes a large batched matmul with CUTLASS in a subprocess and
        # compares the compiled result with eager.
        """
        Test autotune_in_subproc works for bmm.
        """

        B, M, N, K = 10, 4096, 2048, 25728

        a = torch.randn(B, M, K).to(GPU_TYPE).half()
        b = torch.randn(B, N, K).to(GPU_TYPE).half().permute(0, 2, 1)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cutlass.cutlass_max_profiling_configs": 4,
            }
        ):
            Y_compiled = torch.compile(torch.bmm)(a, b)
            Y = torch.bmm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @parametrize("dynamic", (False, True))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_diff_matmul_share_same_kernel(self, dynamic):
        # [FUNCTION DOC] Compiles a model computing two matmuls that share the left operand, with
        # subprocess autotuning, and checks the result matches eager. Then finds the generated
        # CUTLASS kernel name in the output code and asserts it appears exactly twice, i.e. both
        # matmuls use the same kernel.
        max_autotune_gemm_backends = "CUTLASS"

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the products of the first input with each of
            # the second and third inputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent initializer only.
                super().__init__()

            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Returns a tuple of the two matmuls of the first input
                # with the second and third inputs.
                ab = a @ b
                ac = a @ c
                return ab, ac

        model = MyModel()
        a = torch.randn(128, 16).to(GPU_TYPE).half()
        b = torch.randn(128, 16).to(GPU_TYPE).half().t()
        c = torch.randn(512, 16).to(GPU_TYPE).half().t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 1,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, dynamic=dynamic)
            expected = model(a, b, c)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            pattern = r"cutlass_[\w]+\.cutlass_[\w]+"
            match = re.search(pattern, codes[0])
            self.assertTrue(match is not None)
            cutlass_kernel = match.group()
            FileCheck().check_count(
                cutlass_kernel,
                2,
            ).run(codes[0])

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_number_mm_precompiles(self):
        # [FUNCTION DOC] Clears dynamo counters, compiles a single matmul with dynamic shapes and
        # several swizzle options (guaranteeing multiple choices), then checks that CUTLASS
        # appears in the code and that the select_algorithm_num_precompiles counter equals 1.
        torch._dynamo.utils.counters.clear()
        max_autotune_gemm_backends = "CUTLASS"

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing only the matmul of its first two inputs; third
            # input unused.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent initializer only.
                super().__init__()

            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Returns the matmul of the first two inputs; the third is
                # ignored.
                ab = a @ b
                return ab

        model = MyModel()
        a = torch.randn(128, 16).to(GPU_TYPE).half()
        b = torch.randn(128, 16).to(GPU_TYPE).half().t()
        c = torch.randn(512, 16).to(GPU_TYPE).half().t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 1,
                "cutlass.cutlass_max_profiling_swizzle_options": [
                    1,
                    2,
                    4,
                ],  # guarantees > 1 choices
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
                "autotune_local_cache": False,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, dynamic=True)
            expected = model(a, b, c)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(re.search(r"cutlass_.*.cutlass_.*", codes[0]))
            # Verifies expected number of precompilations
            self.assertEqual(
                torch._dynamo.utils.counters["inductor"][
                    "select_algorithm_num_precompiles"
                ],
                1,
            )

    # NOTE: right now tuned_mm doesn't support cutlass 2x, which is used by A100
    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_regular_mm(
        self,
        dynamic: bool,
        max_autotune_gemm_backends: str = "CUTLASS",
        use_aoti: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        # [FUNCTION DOC] Main mm test. Builds one or two shaped half/bf16 input pairs (two only
        # when dynamic), computes eager results, then runs either through AOTI run_multiple or
        # through torch.compile with dynamic shapes, with recompilation treated as an error when
        # dynamic. Compares actual and expected results.
        """
        Main test for mm.
        """

        # M, N, K
        shapes = [
            (128, 128, 16),
            (1024, 1024, 256),
        ]

        # M, N, K
        shapes = shapes if dynamic else shapes[0:1]

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the matmul of two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matmul of the two inputs.
                return a @ b

        model = MyModel().to(GPU_TYPE)

        inputs = [
            (
                torch.randn(M, K).to(GPU_TYPE).to(dtype),
                torch.randn(K, N).to(GPU_TYPE).to(dtype),
            )
            for (M, N, K) in shapes
        ]

        dynamic_shapes = (
            {
                "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "cutlass.cutlass_max_profiling_configs": 2,
                }
            ),
            dynamo_config.patch({"error_on_recompile": dynamic}),
        ):
            expected = [model(*input) for input in inputs]
            if use_aoti:
                actual = AOTIRunnerUtil.run_multiple(
                    model, inputs, dynamic_shapes=dynamic_shapes
                )
            else:
                compiled_model = torch.compile(model, dynamic=True)
                actual = [compiled_model(*input) for input in inputs]

            torch.testing.assert_close(actual, expected)

    @unittest.skipIf(torch.version.hip is not None, "ROCm not supported")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @skipXPUIf(not Xe2_Or_Later, "")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("num_gemms", (1, 2))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_cpp_wrapper(
        self,
        dtype: torch.dtype,
        num_gemms: int,
    ):
        # [FUNCTION DOC] Verifies the CUTLASS GEMM works under JIT cpp_wrapper mode, with one or
        # two chained GEMMs (the second of a different shape so multiple shared libraries must be
        # linked). Compiles with full graph, compares to eager, and checks the generated code
        # references the bare cutlass symbol and not the AOT-only kernels member.
        """
        Verify that CUTLASS GEMM templates work under JIT cpp_wrapper mode.
        ``num_gemms=2`` chains a second GEMM of a different shape so the
        wrapper has to link against multiple distinct .so files.
        """
        M, N, K = 128, 128, 16

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module doing one matmul, followed by a second matmul with the
            # third input when the enclosing test's num_gemms is greater than one.
            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Returns the first matmul, chained with the third input
                # when num_gemms is greater than one.
                out = a @ b
                if num_gemms > 1:
                    out = out @ c
                return out

        model = MyModel().to(GPU_TYPE)
        a = torch.randn(M, K, device=GPU_TYPE, dtype=dtype)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=dtype)
        c = torch.randn(N, N // 2, device=GPU_TYPE, dtype=dtype)
        expected = model(a, b, c)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 2,
                "cpp_wrapper": True,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, fullgraph=True)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            # JIT path must call the bare extern "C" symbol, not via the
            # AOT-only `kernels.` member.
            FileCheck().check("cutlass_").check_not("kernels.cutlass_").run(codes[0])

    @skipXPUIf(True, "XPU SYCL-TLA has not supported fp8 yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float8_e4m3fn,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_fp8_scaled_mm(
        self,
        dynamic: bool,
        max_autotune_gemm_backends: str = "CUTLASS",
        use_aoti: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        # [FUNCTION DOC] Main test for FP8 scaled mm with CUTLASS (e4m3). Quantizes weights and
        # inputs rowwise, with one or two shapes when dynamic, runs eager scaled mm as reference,
        # then runs AOTI or compiled model and compares with relaxed tolerances. TMA-only kernels
        # are forced.
        """
        Main test for mm.
        """

        # M, N, K
        shapes = [
            (128, 128, 16),
            (1024, 1024, 256),
        ]

        # M, N, K
        shapes = shapes if dynamic else shapes[0:1]

        inputs = []
        for shape in shapes:
            M, N, K = shape
            output_dtype = torch.bfloat16
            device = GPU_TYPE

            x = torch.randn(M, K, dtype=output_dtype, device=device)
            w = torch.randn(N, K, dtype=output_dtype, device=device)

            # quantize weight (prior to inference)
            w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype)
            w_t_fp8 = w_fp8.t()
            w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

            # quantize input x
            x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype)

            inputs.append((x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale))

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the scaled matmul with row-wise scales and bf16
            # output.
            def forward(self, x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale):
                # [FUNCTION DOC] HELPER: Runs the scaled matmul of the two fp8 operands with the
                # given inverse scales, no bias, bf16 output, without fast accumulation.
                y = torch._scaled_mm(
                    x_fp8,
                    w_t_fp8,
                    x_inverse_scale,
                    w_inverse_scale,
                    None,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=False,
                )
                return y

        dynamic_shapes = (
            {
                "x_fp8": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                "x_inverse_scale": {0: Dim.DYNAMIC, 1: 1},
                "w_t_fp8": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                "w_inverse_scale": {0: 1, 1: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )
        model = MyModel().to(GPU_TYPE)

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "cutlass.cutlass_max_profiling_configs": 2,
                    "benchmark_epilogue_fusion": False,  # EVT doesn't support benchmark fusion yet
                    "cutlass.cutlass_tma_only": True,
                }
            ),
            dynamo_config.patch({"error_on_recompile": dynamic}),
        ):
            expected = [model(*input) for input in inputs]
            if use_aoti:
                actual = AOTIRunnerUtil.run_multiple(
                    model, inputs, dynamic_shapes=dynamic_shapes
                )
            else:
                compiled_model = torch.compile(model, dynamic=True)
                actual = [compiled_model(*input) for input in inputs]

            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=0.05)

    @unittest.skipIf(
        torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+",
    )
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(
        os.environ,
        # note: It seems necessary to set these here, instead of `config.patch`.
        {
            "PATH": _get_path_without_sccache(),
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "1",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "CUTLASS",
        },
    )
    def test_cutlass_backend_fp8_scaled_mm_mixed_dtypes(self):
        # [FUNCTION DOC] Compiles a scaled mm taking mixed FP8 dtypes (e4m3 by e5m2) with scalar
        # scales, the second operand made column-major, using the CUTLASS backend through
        # environment variables. Compares compiled output with eager.
        """Test that CUTLASS backend works with mixed FP8 dtypes (e4m3fn x e5m2)."""
        m, k, n = 256, 256, 256

        # Create mixed FP8 dtypes: e4m3fn x e5m2
        a8 = torch.randn(m, k, device=GPU_TYPE, dtype=torch.float16).to(
            torch.float8_e4m3fn
        )
        b8 = torch.randn(k, n, device=GPU_TYPE, dtype=torch.float16).to(
            torch.float8_e5m2
        )
        # _scaled_mm requires mat2 to be column-major
        b8 = b8.t().contiguous().t()

        sa = torch.tensor(1.0, device=a8.device)
        sb = torch.tensor(2.0, device=b8.device)

        def scaled_mm_fn(a, b):
            # [FUNCTION DOC] HELPER: Scaled matmul of the two operands using the enclosing scalar
            # scales and half precision output.
            return torch._scaled_mm(
                a, b, scale_a=sa, scale_b=sb, out_dtype=torch.float16
            )

        compiled_fn = torch.compile(scaled_mm_fn, fullgraph=True)
        actual = compiled_fn(a8, b8)
        expected = scaled_mm_fn(a8, b8)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=0.05)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm(
        self,
        dynamic: bool,
        max_autotune_gemm_backends: str = "CUTLASS",
        use_aoti: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        # [FUNCTION DOC] Main addmm test. Skips bf16 on CUDA because of known accuracy problems.
        # For each of four bias shapes (full, column, row, vector) resets dynamo and caches,
        # builds inputs, computes the float32 eager reference cast to the dtype, then runs via
        # AOTI or compile and compares.
        """
        Main test for addmm.
        """
        if dtype == torch.bfloat16 and GPU_TYPE == "cuda":
            # Mismatched elements: 4539 / 16384 (27.7%)
            # Greatest absolute difference: 0.125 at index (12, 33) (up to 0.001 allowed)
            # Greatest relative difference: inf at index (15, 7) (up to 0.002 allowed)
            raise unittest.SkipTest(
                "This case with bfloat16 has known accuracy issues that need to be resolved."
            )

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing addmm of its three inputs.
            def forward(self, x, a, b):
                # [FUNCTION DOC] HELPER: Returns addmm of the bias and the two matrices.
                return torch.addmm(x, a, b)

        model = MyModel().to(GPU_TYPE)
        # M, N, K
        shapes = [
            (128, 128, 16),
            (512, 512, 128),
        ]
        shapes = shapes[0:1] if not dynamic else shapes

        x_shapes = [
            lambda M, N: (M, N),
            lambda M, N: (M, 1),
            lambda M, N: (1, N),
            lambda M, N: (N,),
        ]
        for x_shape in x_shapes:
            torch._dynamo.reset()
            clear_caches()

            inputs = [
                (
                    torch.randn(x_shape(M, N)).to(GPU_TYPE).to(dtype),
                    random_matrix_with_scaled_reduction_dim(
                        M, K, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
                    ),
                    random_matrix_with_scaled_reduction_dim(
                        N, K, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
                    ).t(),
                )
                for (M, N, K) in shapes
            ]
            dynamic_shapes = (
                {
                    "x": {
                        i: v
                        for i, v in enumerate(x_shape(Dim.DYNAMIC, Dim.DYNAMIC))
                        if v != 1
                    },
                    "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                    "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                }
                if dynamic
                else None
            )
            with (
                config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": max_autotune_gemm_backends,
                        "cutlass.cutlass_max_profiling_configs": 2,
                    }
                ),
                dynamo_config.patch({"error_on_recompile": dynamic}),
            ):
                expected = [
                    model(*[t.float() for t in input]).to(dtype) for input in inputs
                ]
                if use_aoti:
                    actual = AOTIRunnerUtil.run_multiple(
                        model, inputs, dynamic_shapes=dynamic_shapes
                    )
                else:
                    compiled_model = torch.compile(model, dynamic=dynamic)
                    actual = [compiled_model(*input) for input in inputs]

                torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm_input_reorder(self):
        # [FUNCTION DOC] Regression test for the input reorder of addmm CUTLASS choices.
        # Temporarily replaces the benchmark_choices class method of the selector cache with a
        # wrapper that records each timing per choice type (extern aten versus others, cutlass),
        # compiles addmm with ATEN and CUTLASS backends, compares to a float32 reference, and
        # checks no timing is infinite. The original method is always restored in a finally
        # block.
        """
        Without input_reorder on the CUTLASS call in tuned_addmm, the CUTLASS
        C kernel's argument order doesn't match the benchmark tensor order,
        causing illegal memory access during autotuning.

        NOTE: Uses ATEN+CUTLASS (not CUTLASS-only) because the existing addmm test
        has a pre-existing CUTLASS accuracy issue at all shapes. Also verifies that neither ATen nor CUTLASS
        choices get inf timings (which would indicate benchmark failures).
        """
        from torch._inductor.select_algorithm import (
            AlgorithmSelectorCache,
            ExternKernelCaller,
        )

        original_benchmark_choices = AlgorithmSelectorCache.benchmark_choices
        timings_by_type: dict[str, list[float]] = {"aten": [], "cutlass": []}

        @classmethod
        def tracking_benchmark_choices(cls, choices, autotune_args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement benchmark_choices class method. Delegates to the
            # original function, then appends each returned timing to the aten list if the choice
            # is an extern kernel caller and to the cutlass list otherwise (closure state).
            # Returns the original result.
            result = original_benchmark_choices.__func__(
                cls, choices, autotune_args, **kwargs
            )
            for choice, timing in result.items():
                if isinstance(choice, ExternKernelCaller):
                    timings_by_type["aten"].append(timing)
                else:
                    timings_by_type["cutlass"].append(timing)
            return result

        AlgorithmSelectorCache.benchmark_choices = tracking_benchmark_choices
        try:
            M, K, N = 256, 3520, 2048
            bias = torch.randn(N, device=GPU_TYPE, dtype=torch.bfloat16)
            x = random_matrix_with_scaled_reduction_dim(
                M, K, dtype=torch.bfloat16, device=GPU_TYPE, reduction_dim=-1
            )
            w = random_matrix_with_scaled_reduction_dim(
                K, N, dtype=torch.bfloat16, device=GPU_TYPE, reduction_dim=-2
            )

            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "ATEN,CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 2,
                }
            ):
                expected = torch.addmm(bias.float(), x.float(), w.float()).to(
                    torch.bfloat16
                )
                actual = torch.compile(torch.addmm)(bias, x, w)
                torch.testing.assert_close(actual, expected)

            self.assertTrue(
                all(t != float("inf") for t in timings_by_type["aten"]),
                lambda msg: f"{msg}\nATen benchmark failed: {timings_by_type['aten']}",
            )
            self.assertTrue(
                all(t != float("inf") for t in timings_by_type["cutlass"]),
                lambda msg: f"{msg}\nCUTLASS benchmark failed: {timings_by_type['cutlass']}",
            )
        finally:
            AlgorithmSelectorCache.benchmark_choices = original_benchmark_choices

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("use_expand", (False, True))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_bmm(
        self,
        dynamic: bool,
        use_aoti: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
        dtype: torch.dtype = torch.float16,
        use_expand: bool = False,
    ):
        # [FUNCTION DOC] Main bmm test. Builds batched inputs for one or two shapes (two only
        # when dynamic), optionally with the left operand an expanded unbatched matrix, right
        # operand a permuted tensor. Runs eager, then AOTI or compile, and compares results.
        """
        Main test for bmm.
        """

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a batched matmul.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the batched matmul of the two inputs.
                return torch.bmm(a, b)

        model = MyModel().to(GPU_TYPE)
        # B, M, N, K
        shapes = [
            (10, 4096, 2048, 25728),
            (20, 2048, 1024, 12864),
        ]
        shapes = shapes[0:1] if not dynamic else shapes

        inputs = []
        for B, M, N, K in shapes:
            if use_expand:
                A = (
                    random_matrix_with_scaled_reduction_dim(
                        M, K, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
                    )
                    .unsqueeze(0)
                    .expand(B, -1, -1)
                )
            else:
                A = random_matrix_with_scaled_reduction_dim(
                    M, K, B, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
                )

            B_tensor = random_matrix_with_scaled_reduction_dim(
                N, K, B, dtype=dtype, device=GPU_TYPE, reduction_dim=-1
            ).permute(0, 2, 1)
            inputs.append((A, B_tensor))
        dynamic_shapes = (
            {
                "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC, 2: Dim.DYNAMIC},
                "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC, 2: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            expected = [model(*input) for input in inputs]
            if use_aoti:
                actual = AOTIRunnerUtil.run_multiple(
                    model, inputs, dynamic_shapes=dynamic_shapes
                )
            else:
                compiled_model = torch.compile(model, dynamic=dynamic)
                actual = [compiled_model(*input) for input in inputs]
            torch.testing.assert_close(actual, expected)

    @skipXPUIf(True, "streamk kernels not supported on xpu cutlass backend yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_regular_mm_streamk(
        self, dynamic: bool = False, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        # [FUNCTION DOC] Autotunes mm in subprocess with only stream-K kernels allowed over four
        # shapes, including very large ones, comparing each compiled result with eager using
        # relaxed tolerances.
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        compiled_model = torch.compile(torch.mm, dynamic=dynamic)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
                "cutlass.cutlass_op_allowlist_regex": "stream_k",  # only stream-k GEMM Kernels
            }
        ):
            for M, K, N in (
                (128, 16, 128),
                (1024, 256, 1024),
                (
                    16384,
                    1024,
                    16384,
                ),
                (
                    16384,
                    1408,
                    16384,
                ),
            ):
                a = torch.randn(M, K).to(GPU_TYPE).half()
                b = torch.randn(N, K).to(GPU_TYPE).half().t()
                Y_compiled = compiled_model(a, b)
                Y = torch.mm(a, b)
                # we need relaxed numerical limits due to the sheer size of the
                # matmuls involved. Many small addition differences add up.
                torch.testing.assert_close(Y_compiled, Y, atol=0.01, rtol=0.01)

    @skipXPUIf(True, "streamk kernels not supported on xpu cutlass backend yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_streamk_with_dynamic(
        self,
    ):
        # [FUNCTION DOC] Compiling mm with dynamic shapes while only stream-K kernels are allowed
        # must fail with an InductorError naming NoValidChoicesError, since stream-K is filtered
        # out (its workspace size depends on the shape).
        """
        Test streamk with dynamic=True. Streamk should be filtered out.

        Problem is streamk can have a different workspace depending on the
        shape. Without a correct workspace, the kernel will fail at runtime.
        """

        a = torch.randn(128, 16).to(GPU_TYPE).half()
        b = torch.randn(128, 16).to(GPU_TYPE).half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_op_allowlist_regex": "stream_k",  # only stream-k GEMM Kernels
            }
        ):
            with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                _ = torch.compile(torch.mm, dynamic=True)(a, b)

    @skipXPUIf(True, "streamk kernels not supported on xpu cutlass backend yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_streamk_with_static(
        self,
    ):
        # [FUNCTION DOC] Compiling mm with static shapes for three large shapes with only
        # stream-K kernels allowed must succeed (results are not compared).
        """
        Test streamk with dynamic=False. Streamk should work.
        """

        shapes = [
            (18432, 3072, 6144),
            (9216, 3072, 6144),
            (4608, 3072, 6144),
        ]
        compiled_model = torch.compile(torch.mm, dynamic=False)

        for shape in shapes:
            M, N, K = shape
            a = torch.randn(M, K).to(GPU_TYPE).half()
            b = torch.randn(N, K).to(GPU_TYPE).half().t()

            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 1,
                    "cutlass.cutlass_op_allowlist_regex": "stream_k",  # only stream-k GEMM Kernels
                }
            ):
                _ = compiled_model(a, b)

    def _test_max_autotune_cutlass_backend_epilogue_fusion(
        self,
        dynamic: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
        fp16=True,
        expected_fuse_count=0,
        mm: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        batch_size: int | None = None,
    ):
            # [FUNCTION DOC] HELPER: Shared driver for epilogue fusion tests. Builds 2D or
            # batched inputs (optionally half), resets the epilogue fusion counter, compiles the
            # given mm function with subprocess autotuning, compares with eager, and raises
            # AssertionError if mm is missing or the counter differs from expected_fuse_count.
        # Note: The ops that are available
        # also depend on the alignment of the shapes
        # so if these shapes don't all align to at least 8 elements
        # it can happen that no Cutlass 3.x op is available
        # that allows fusions
        if batch_size is None:
            a = torch.randn(256, 32).to(GPU_TYPE)
            b = torch.randn(256, 32).to(GPU_TYPE).t()
        else:
            a = torch.randn(batch_size, 256, 32).to(GPU_TYPE)
            b = torch.randn(batch_size, 256, 32).to(GPU_TYPE).permute(0, 2, 1)
        if fp16:
            a = a.half()
            b = b.half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 4,
            }
        ):
            counters["inductor"]["cutlass_epilogue_fusion_counter"] = 0
            if mm is None:
                raise AssertionError("mm is None")
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            actual_count = counters["inductor"]["cutlass_epilogue_fusion_counter"]
            if actual_count != expected_fuse_count:
                raise AssertionError(
                    f"Expected fuse count of {expected_fuse_count} but got {actual_count}"
                )
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_fp32acc(self):
        # [FUNCTION DOC] Epilogue test for scaling the matmul by a constant in half precision;
        # expects zero fusions.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Matmul scaled by a constant.
            return (a @ b) * 3.0

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_max_autotune_cutlass_backend_chained_fusion_fp16_fp32acc(self):
        # [FUNCTION DOC] Epilogue test for a chain of scale and subtract constants after the
        # matmul; expects zero fusions.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Matmul times a constant minus a constant.
            return (a @ b) * 3.3 - 1.234

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_max_autotune_cutlass_backend_relu_fusion_fp16_fp32acc(self):
        # [FUNCTION DOC] Epilogue test with relu after scale and subtract; expects zero fusions
        # because the pointwise ops are already fused into one node.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Relu of the scaled and shifted matmul.
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_max_autotune_cutlass_backend_relu6_fusion_fp16_fp32acc(self):
        # [FUNCTION DOC] Epilogue test with relu then clamp at an upper bound after the matmul;
        # expects zero fusions because the pointwise ops are pre-fused.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Clamp of the relu of the matmul with an upper bound.
            return torch.clamp(torch.nn.functional.relu(a @ b), max=6.0)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_max_autotune_cutlass_backend_no_fusion_dtype_mismatch(self):
        # [FUNCTION DOC] Epilogue test where the output is cast to float32, differing from the
        # matmul dtype; expects zero fusions.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Matmul cast to float32 then scaled by a small constant.
            # this should not be fused, since the output dtype is different from the matmul dtype
            return (a @ b).to(torch.float32) * 0.00001

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_max_autotune_cutlass_backend_shape_dependent_normalization_fusion(self):
        # [FUNCTION DOC] Epilogue test where the matmul is divided by a size of the second
        # operand; expects zero fusions.
        def mm(a, b):
            # [FUNCTION DOC] HELPER: Matmul divided by the second operand column count.
            return (a @ b) / b.size(1)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @skipXPUIf(True, "int_mm not supported on xpu cutlass backend")
    # TODO: Enable dynamic test cases when dynamic support is added.
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @skipCUDAIf(
        torch.version.hip is None
        and HAS_CUDA
        and torch.cuda.get_device_capability() in ((10, 3), (10, 7)),
        "int_mm not supported on SM103 or SM107",
    )
    @xfailIfSM120OrLater
    @parametrize("dynamic", (False,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_int_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        # [FUNCTION DOC] Autotunes an int8 matmul (int_mm) with CUTLASS in subprocess, with the
        # second operand transposed to meet layout and alignment requirements, and compares to
        # eager.
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Int8 matmul of the two operands.
            return torch._int_mm(a, b)

        # CUTLASS only supports row-major/column-major combination of
        # layouts for this operation, thus the transpose of tensor b
        # (on the other side, Triton at the moment doesn't support
        # this combination, so it's excluded from the test).  Also,
        # for CUTLASS alignment requirements, number of columns in
        # both tensors has to be divisible by 16.
        a = torch.randint(0, 5, (100, 16), dtype=torch.int8).to(GPU_TYPE)
        b = torch.randint(0, 5, (32, 16), dtype=torch.int8).to(GPU_TYPE).T

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_force_cutlass_backend_aoti_dynamic(self):
        # [FUNCTION DOC] Exports a matmul with AOTI using forced CUTLASS (no subprocess
        # autotuning) and dynamic shapes given as fixed sizes, runs it and compares with eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matmul of its two inputs.
            def forward(self, x, w):
                # [FUNCTION DOC] HELPER: Returns the matmul of the inputs.
                return x @ w

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            model = MyModel()
            M, N, K = 16, 32, 64
            dynamic_shapes = {
                "x": {0: M, 1: K},
                "w": {0: K, 1: N},
            }

            x = torch.randn(M, K).to(GPU_TYPE).half()
            w = torch.randn(N, K).to(GPU_TYPE).half().t()

            actual = AOTIRunnerUtil.run(
                model,
                (x, w),
                dynamic_shapes=dynamic_shapes,
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_force_cutlass_backend_aoti_cexpr_codegen(self):
        # [FUNCTION DOC] Exports with AOTI a model that reshapes and strides its input along a
        # dynamic first dimension before a matmul, forcing CUTLASS, to exercise symbolic
        # expression code generation; compares with eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that takes every other element of the reshaped input,
            # makes it contiguous, rebuilds it as strided, and multiplies by the weight.
            def forward(self, x, w):
                # [FUNCTION DOC] HELPER: Reshape, slice, make contiguous and re-stride the input,
                # then matmul.
                x0, x1 = x.shape
                x = x.reshape(x0 // 2, x1, 2)[:, :, 0]
                x = x.contiguous()
                x = x.as_strided(x.size(), x.stride())

                return x @ w

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            model = MyModel()
            M, N, K = 128, 64, 64
            dynamic_shapes = {
                "x": {0: Dim.DYNAMIC},
                "w": None,
            }

            x = torch.randn(M, K).to(GPU_TYPE).half()
            w = torch.randn(N, K).to(GPU_TYPE).half().t()

            actual = AOTIRunnerUtil.run(
                model,
                (x, w),
                dynamic_shapes=dynamic_shapes,
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @skipXPUIf(True, "streamk kernels not supported on xpu cutlass backend yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    def test_aoti_workspace_ptr(self):
        # [FUNCTION DOC] AOTI run with only a stream-K 128x256x64 kernel allowed to check
        # workspace pointer handling; on SM120 or later expects NoValidChoicesError, otherwise
        # compares with eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matmul of its two inputs.
            def forward(self, x, w):
                # [FUNCTION DOC] HELPER: Returns the matmul of the inputs.
                return x @ w

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_op_allowlist_regex": "128x256x64.*stream_k",
                "cutlass.cutlass_max_profiling_configs": 1,
            }
        ):
            model = MyModel()
            M, N, K = 200, 5216, 10_432

            x = torch.randn(M, K).to(GPU_TYPE).half()
            w = torch.randn(N, K).to(GPU_TYPE).half().t()

            if SM120OrLater:
                with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                    AOTIRunnerUtil.run(
                        model,
                        (x, w),
                    )
                return

            actual = AOTIRunnerUtil.run(
                model,
                (x, w),
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual, atol=0.01, rtol=0.01)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM80OrLater or SM90OrLater, "need sm_8x exactly")
    @parametrize("dynamic", (False,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_sparse_semi_structured_mm(
        self, dynamic: bool
    ):
        # [FUNCTION DOC] Forces the CUTLASS path for sparse semi-structured tensors (a global
        # class flag is set and never reset), compiles an mm of a 2:4 sparse matrix and dense
        # matrix with subprocess autotuning, compares with dense eager, then inspects the local
        # autotune cache and asserts at least one CUTLASS kernel has a finite timing.
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        SparseSemiStructuredTensor._FORCE_CUTLASS = True

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix multiply.
            return torch.mm(a, b)

        m, n, k = 32, 8, 64
        mask = torch.tensor([0, 0, 1, 1]).tile(m, k // 4).to(GPU_TYPE).half()
        a = torch.rand(m, k).to(GPU_TYPE).half() * mask
        a_sparse = to_sparse_semi_structured(a)
        b = torch.rand(k, n).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 2,
                "autotune_local_cache": True,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a_sparse, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

        cache = torch._inductor.codecache.LocalCache().lookup(
            "sparse_semi_structured_mm"
        )
        if cache is None:
            raise AssertionError("cache is None")
        high = cache[
            f"[('cuda', 'torch.float16', {m}, {k // 2}, {k // 2}, 1, 0), "
            f"('cuda', 'torch.int16', {m}, {k // 16}, {k // 16}, 1, 0), "
            f"('cuda', 'torch.float16', {k}, {n}, {n}, 1, 0)]"
        ]["high"]
        cutlass_kernels_count = 0
        for kernel, duration in high.items():
            if kernel.startswith("cutlass_gemm") and not math.isinf(duration):
                cutlass_kernels_count += 1
        if cutlass_kernels_count <= 0:
            raise AssertionError(
                f"Expected cutlass_kernels_count > 0, got {cutlass_kernels_count}"
            )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_op_denylist(
        self,
    ):
        # [FUNCTION DOC] Sets the deny list regex to pingpong, compiles addmm with the selection
        # function mocked to raise, expects NoValidChoicesError, then verifies the selected op is
        # addmm and that no CUTLASS template choice has a name containing pingpong while at least
        # one remains.
        def my_addmm(x, a, b, alpha, beta):
            # [FUNCTION DOC] HELPER: addmm with alpha and beta arguments swapped into keyword
            # positions.
            return torch.addmm(x, a, b, alpha=beta, beta=alpha)

        x = torch.randn((128, 128)).to(GPU_TYPE).half()
        a = torch.randn(128, 128).to(GPU_TYPE).half()
        b = torch.randn(128, 128).to(GPU_TYPE).half().t()

        with fresh_cache():
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 2,
                    "cutlass.cutlass_op_allowlist_regex": "",
                    "cutlass.cutlass_op_denylist_regex": "pingpong",
                }
            ):
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
                    with self.assertRaisesRegex(
                        InductorError, r".*NoValidChoicesError.*"
                    ):
                        torch.compile(my_addmm, dynamic=False)(x, a, b, 1.0, 2.0)
                    args, _ = sa.call_args
                    op_name, choices, _, __ = args
                    if op_name != "addmm":
                        raise AssertionError(
                            f"Expected op_name 'addmm', got {op_name!r}"
                        )
                    cuda_template_count = 0
                    for choice in choices:
                        if isinstance(choice, CUTLASSTemplateCaller):
                            choice_info = choice.info_dict()
                            op_conf_name = choice_info.get("op_conf_name", "")
                            if not isinstance(op_conf_name, str):
                                raise AssertionError(
                                    f"Expected op_conf_name to be str, got {type(op_conf_name)}"
                                )
                            if "pingpong" in op_conf_name:
                                raise AssertionError(
                                    "All pingpong Kernels should have been filtered"
                                )
                            cuda_template_count += 1
                    if cuda_template_count <= 0:
                        raise AssertionError("No CUTLASSTemplateCaller choices")

    @skipXPUIf(True, "Intel cutlass doesn't have pingpong kernels yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_op_allowlist(
        self,
    ):
        # [FUNCTION DOC] Sets the allow list regex (stream_k on SM100 or later, otherwise
        # pingpong) with no deny list, mocks the selection function to raise, and inspects the
        # choices passed in.
        def addmm(x, a, b, alpha, beta):
            # [FUNCTION DOC] HELPER: addmm with explicit alpha and beta.
            return torch.addmm(x, a, b, alpha=alpha, beta=beta)

        x = torch.randn((128, 128)).to(GPU_TYPE).half()
        a = torch.randn(128, 128).to(GPU_TYPE).half()
        b = torch.randn(128, 128).to(GPU_TYPE).half().t()
        allowlist_regex = "stream_k" if SM100OrLater else "pingpong"

        with fresh_cache():
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 2,
                    "cutlass.cutlass_op_allowlist_regex": allowlist_regex,
                    "cutlass.cutlass_op_denylist_regex": None,
                }
            ):
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
                    with self.assertRaisesRegex(
                        InductorError, r".*NoValidChoicesError.*"
                    ):
                        torch.compile(addmm, dynamic=False)(x, a, b, 1.0, 1.0)
                    args, _ = sa.call_args
                    op_name, choices, _, __ = args
                    if op_name != "addmm":
                        raise AssertionError(
                            f"Expected op_name 'addmm', got {op_name!r}"
                        )
                    cuda_template_count = 0
                    for choice in choices:
                        if isinstance(choice, CUTLASSTemplateCaller):
                            choice_info = choice.info_dict()
                            op_conf_name = choice_info.get("op_conf_name", "")
                            if not isinstance(op_conf_name, str):
                                raise AssertionError(
                                    f"Expected op_conf_name to be str, got {type(op_conf_name)}"
                                )
                            if allowlist_regex not in op_conf_name:
                                raise AssertionError(
                                    f"Only {allowlist_regex} kernels should have been allowed"
                                )
                            cuda_template_count += 1
                    if cuda_template_count <= 0:
                        raise AssertionError("No CUTLASSTemplateCaller choices")

    @skipXPUIf(True, "fp8 not supported on xpu cutlass backend yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_fp8_scaled_mm_fast_accum_filtering(
        self,
    ):
        # [FUNCTION DOC] Quantizes bf16 inputs to e4m3 rowwise and compiles a scaled mm twice
        # (fast accumulation true unless on SM100 or later, then false) with the selection
        # function mocked to raise. Checks every CUTLASS choice name contains fastaccum only when
        # fast accumulation was requested and never otherwise, and that at least one CUTLASS
        # choice remains.
        float8_dtype = torch.float8_e4m3fn
        # Only bf16 output type is supported for row-wise scaling, not fp32
        output_dtype: torch.dtype = torch.bfloat16
        device = GPU_TYPE
        M, K, N = 128, 128, 128  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=output_dtype, device=device)
        w = torch.randn(N, K, dtype=output_dtype, device=device)
        bias = None
        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, float8_dtype)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, float8_dtype)

        def linear(
            x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias, use_fast_accum
        ):
            # [FUNCTION DOC] HELPER: Scaled mm of the quantized operands with the given scales,
            # bias and the fast accumulation flag, bf16 output (closure variable).
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=output_dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        linear_compiled = torch.compile(linear, backend="inductor")

        def run_test(use_fast_accum):
            # [FUNCTION DOC] HELPER: Under a fresh cache and CUTLASS-only max-autotune config,
            # runs the compiled scaled mm with the given fast accumulation flag with selection
            # mocked to raise, expects NoValidChoicesError, and checks the fastaccum naming of
            # the surviving CUTLASS choices.
            with fresh_cache():
                with config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "CUTLASS",
                        "cutlass.cutlass_max_profiling_configs": 2,
                    }
                ):
                    with mock.patch(
                        "torch._inductor.kernel.mm.autotune_select_algorithm",
                        wraps=select_no_algorithm,
                    ) as sa:
                        with self.assertRaisesRegex(
                            InductorError, r".*NoValidChoicesError.*"
                        ):
                            linear_compiled(
                                x_fp8,
                                x_inverse_scale,
                                w_t_fp8,
                                w_inverse_scale,
                                bias,
                                use_fast_accum,
                            )

                        args, _ = sa.call_args
                        _, choices, _, _ = args
                        cuda_template_count = 0
                        for choice in choices:
                            if isinstance(choice, CUTLASSTemplateCaller):
                                choice_info = choice.info_dict()
                                op_conf_name = choice_info.get("op_conf_name", "")
                                if not isinstance(op_conf_name, str):
                                    raise AssertionError(
                                        f"Expected op_conf_name to be str, got {type(op_conf_name)}"
                                    )
                                if use_fast_accum:
                                    if "fastaccum" not in op_conf_name:
                                        raise AssertionError(
                                            "Only fastaccum Kernels should have been allowed"
                                        )
                                else:
                                    if "fastaccum" in op_conf_name:
                                        raise AssertionError(
                                            "fastaccum Kernels should have been filtered"
                                        )
                                cuda_template_count += 1
                        if cuda_template_count <= 0:
                            raise AssertionError("No CUTLASSTemplateCaller choices")

        if not SM100OrLater:
            run_test(True)
        run_test(False)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_shape_coverage_mm(
        self,
    ):
        # [FUNCTION DOC] Compiles mm for five input shape pairs with static shapes, with
        # selection mocked to raise. For each, asserts selection was called for mm and that at
        # least one CUTLASS template choice exists; correctness is not checked and nothing is
        # compiled to kernels.
        """
        Checks if cutlass backend produces some ops for a variety of shapes.

        This test doesn't compile and check the correctness of the ops.

        NOTE: K has to be even.
        """

        inputs = [
            (
                torch.randn(128, 500).to(GPU_TYPE).half(),
                torch.randn(500, 576).to(GPU_TYPE).half(),
            ),
            (
                torch.randn(500, 128).to(GPU_TYPE).half(),
                torch.randn(128, 576).to(GPU_TYPE).half(),
            ),
            (
                torch.randn(128, 250).to(GPU_TYPE).half(),
                torch.randn(250, 576).to(GPU_TYPE).half(),
            ),
            (
                torch.randn(250, 128).to(GPU_TYPE).half(),
                torch.randn(128, 576).to(GPU_TYPE).half(),
            ),
            (
                torch.randn(125, 128).to(GPU_TYPE).half(),
                torch.randn(128, 576).to(GPU_TYPE).half(),
            ),
        ]

        with (
            fresh_cache(),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 2,
                }
            ),
            mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                wraps=select_no_algorithm,
            ) as sa,
        ):
            for input in inputs:
                A, B = input
                M, K = A.shape
                _, N = B.shape

                with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                    torch.compile(torch.mm, dynamic=False)(*input)

                self.assertTrue(
                    sa.called,
                    lambda msg: f"{msg}\nautotune_select_algorithm was not called  with shape M={M}, N={N}, K={K}",
                )
                args, _ = sa.call_args
                op_name, choices, _, __ = args
                if op_name != "mm":
                    raise AssertionError(f"Expected op_name 'mm', got {op_name!r}")
                cuda_template_count = 0
                for choice in choices:
                    if isinstance(choice, CUTLASSTemplateCaller):
                        choice_info = choice.info_dict()
                        op_conf_name = choice_info.get("op_conf_name", "")
                        if not isinstance(op_conf_name, str):
                            raise AssertionError(
                                f"Expected op_conf_name to be str, got {type(op_conf_name)}"
                            )
                        cuda_template_count += 1

                self.assertGreater(
                    cuda_template_count,
                    0,
                    lambda msg: f"{msg}\n"
                    + (
                        "No CUTLASSTemplateCaller choices found for matmul with shape "
                        f"M={M}, N={N}, K={K}"
                    ),
                )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM80OrLater, "need sm_80")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_get_max_alignment(self):
        # [FUNCTION DOC] Unit test of the alignment helper on CPU fixed layouts: contiguous case
        # gives 4, strided case 4, odd stride gives 1, offset 6 for half gives 2, aligned offset
        # gives 8, and float32 dtype gives 4.
        l4 = FixedLayout(
            torch.device("cpu"), torch.half, size=[1, 2, 4], stride=[0, 4, 1]
        )
        m4 = get_max_alignment(l4)
        self.assertEqual(
            m4, 4, "Wrong max alignment. Should have been 4. (simple, contiguous case)"
        )

        l4_2 = FixedLayout(
            torch.device("cpu"), torch.half, size=[1, 4, 2], stride=[0, 1, 4]
        )
        m4_2 = get_max_alignment(l4_2)
        self.assertEqual(
            m4_2,
            4,
            "Wrong max alignment. Should have been 4. Did not deal with strides correctly",
        )

        l1 = FixedLayout(
            torch.device("cpu"), torch.half, size=[2, 4, 2], stride=[23, 1, 4]
        )
        m1 = get_max_alignment(l1)
        self.assertEqual(
            m1,
            1,
            "Wrong max alignment. Should have been 1. Did not take stride into account correctly",
        )

        l2 = FixedLayout(
            torch.device("cpu"), torch.half, size=[1, 2, 4], stride=[0, 4, 1], offset=6
        )
        m2 = get_max_alignment(l2)
        self.assertEqual(
            m2, 2, "Wrong max alignment. Should have been 2. (due to choice of offset)"
        )

        l8 = FixedLayout(
            torch.device("cpu"),
            torch.half,
            size=[2, 2, 8],
            stride=[32, 8, 1],
            offset=24,
        )
        m8 = get_max_alignment(l8)
        self.assertEqual(m8, 8, "Wrong max alignment. Should have been 8.")

        l4 = FixedLayout(
            torch.device("cpu"),
            torch.float32,
            size=[2, 2, 8],
            stride=[32, 8, 1],
            offset=24,
        )
        m4 = get_max_alignment(l4)
        self.assertEqual(
            m4, 4, "Wrong max alignment. Should have been 4 (due to float32 dtype )."
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_standalone_runner(self):
        # [FUNCTION DOC] Enables generation of a standalone runner, compiles mm while capturing
        # the CUTLASS source, writes the first source to a temporary file, builds it with the
        # standalone compile command (on FBCODE rewrites a link flag for libcuda), runs the build
        # command and, unless on a remote execution worker in FBCODE, the executable; both must
        # return status zero. Removes the temp files.
        max_autotune_gemm_backends = "CUTLASS"

        a = torch.randn(128, 16).to(GPU_TYPE).half()
        b = torch.randn(128, 16).to(GPU_TYPE).half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
                "cutlass.generate_test_runner": True,  # put standalone runner in the generated code
            }
        ):
            from tempfile import NamedTemporaryFile

            from torch._inductor.codegen.cutlass.utils import (
                cutlass_standalone_runner_compile_command,
                CUTLASSCompileSourceCapturingContext,
            )

            # Run compilation, check results just in case, and save
            # CUTLASS-based generated code.
            with CUTLASSCompileSourceCapturingContext(GPU_TYPE) as ctx:
                compiled = torch.compile(torch.mm, dynamic=False)

                expected = torch.mm(a, b)
                actual = compiled(a, b)

                torch.testing.assert_close(actual, expected)

                sources = ctx.sources

            if len(sources) < 1:
                raise AssertionError(f"Expected len(sources) >= 1, got {len(sources)}")

            # Get names for temporary source and executable files.
            suffix = ".cpp" if GPU_TYPE == "xpu" else ".cu"
            cu_file = NamedTemporaryFile("w", suffix=suffix, delete=False)  # noqa: SIM115
            cu_file.close()
            exe_file = NamedTemporaryFile("w", suffix="", delete=False)  # noqa: SIM115
            exe_file.close()

            # Save the generated code into the .cu file.
            with open(cu_file.name, "w") as file:
                file.write(sources[0])

            # Get command to compile .cu file, and run the
            # compilation.
            command = cutlass_standalone_runner_compile_command(
                GPU_TYPE, Path(cu_file.name), Path(exe_file.name)
            )

            if IS_FBCODE:
                # hack to bypass the following error:
                # error while loading shared libraries: IX}: invalid mode for dlopen(): Invalid argument
                platform_path = sysconfig.get_config_var("LIBDIR")
                cuda_path = os.path.realpath(os.path.join(platform_path, "libcuda.so"))
                command = command.replace("-lcuda ", f"-L{cuda_path} ")

            repro_message = (
                f"Reproduce with: {command}\n"
                f"exe_file.name: {exe_file.name}\n"
                f"cu_file.name: {cu_file.name}\n"
            )

            retcode = os.system(command)
            self.assertEqual(retcode, 0, repro_message)

            # Run the executable generated.
            if not IS_FBCODE or not IN_RE_WORKER:
                retcode = os.system(exe_file.name)
                self.assertEqual(retcode, 0, repro_message)

            # Remove temporary files.
            os.remove(cu_file.name)
            os.remove(exe_file.name)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_integration(self):
        # [FUNCTION DOC] Autotunes mm with ATEN, TRITON and CUTLASS together while capturing
        # debug logs of the CUTLASS codegen logger; checks the result and that the log reports a
        # positive number of ops.
        """
        Test if cutlass backend can be autotune with other backends
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Matmul of the two operands.
            return a @ b

        a = torch.randn(128, 16).to(GPU_TYPE).half()
        b = torch.randn(128, 16).to(GPU_TYPE).half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 2,
                # needed for log searching
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
            }
        ):
            with (
                log_settings("+inductor"),
                self.assertLogs(
                    logger="torch._inductor.codegen.cutlass", level=logging.DEBUG
                ) as test_log,
            ):
                Y_compiled = torch.compile(mm, dynamic=False)(a, b)
                Y = mm(a, b)
                torch.testing.assert_close(Y_compiled, Y)

            output = "\n".join(record.getMessage() for record in test_log.records)

            match = re.search(
                r"Got cutlass configs: total number of ops: (\d+)", output
            )
            if not match:
                raise AssertionError("Expect to find the cutlass configs log")
            num_ops = int(match.group(1))
            self.assertTrue(num_ops > 0, "The number of ops should be greater than 0")

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_maybe_append_choice_caching(self):
        # [FUNCTION DOC] Wraps the CUTLASS template render method with a counter, compiles a
        # model that repeats a matmul ten times with caching codegen enabled, checks results, and
        # asserts render is called ten plus two times (cache reuse across identical matmuls).
        """
        Test if maybe_append_choice's caching leads to correct results and
        shorter maybe_append_choice time.
        """

        NUM_ITERATIONS = 10

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the matmul and division by 32 repeatedly to the
            # first input, using the enclosing iteration count.
            def forward(self, A, B):
                # [FUNCTION DOC] HELPER: Loops the iteration count times updating the left
                # operand with its matmul by the second operand divided by 32; returns the final
                # left operand.
                for _ in range(NUM_ITERATIONS):
                    A = A @ B / 32
                return A

        model = TestModule().to(GPU_TYPE)
        A = torch.randn(1024, 1024, dtype=torch.bfloat16, device=GPU_TYPE)
        B = torch.randn(1024, 1024, dtype=torch.bfloat16, device=GPU_TYPE).t()

        expected = model(A, B)

        # Track render calls
        from torch._inductor.codegen.cutlass.gemm_template import CUTLASSGemmTemplate

        original_render = CUTLASSGemmTemplate.render
        render_call_count = 0

        def counting_render(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement render method that increments the enclosing call
            # counter and delegates to the original render.
            nonlocal render_call_count
            render_call_count += 1
            return original_render(self, *args, **kwargs)

        with mock.patch.object(CUTLASSGemmTemplate, "render", counting_render):
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                    "cutlass.enable_caching_codegen": True,
                    "cutlass.cutlass_max_profiling_configs": 2,
                }
            ):
                compiled_model = torch.compile(model, fullgraph=True)
                actual = compiled_model(A, B)

        torch.testing.assert_close(actual, expected)

        self.assertEqual(render_call_count, NUM_ITERATIONS + 2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_multiple_mm(self):
        # [FUNCTION DOC] Compiles a module with two matmuls of different shapes with caching
        # codegen enabled, counting render calls; asserts the call count equals the number of
        # matmuls plus twice that number and results match eager.
        """
        Test multiple matrix multiplications with different shapes in a single nn.Module.
        """

        class MultipleMMModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning two independent matmuls.
            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: Returns the matmuls of the first pair and second pair of
                # inputs.
                # First mm with shape (128, 64) @ (64, 32) -> (128, 32)
                mm1 = a @ b
                # Second mm with shape (256, 128) @ (128, 64) -> (256, 64)
                mm2 = c @ d
                return mm1, mm2

        model = MultipleMMModel().to(GPU_TYPE)

        # Create tensors with different shapes
        a = torch.randn(128, 64).to(GPU_TYPE).half()
        b = torch.randn(32, 64).to(GPU_TYPE).half().t()
        c = torch.randn(256, 128).to(GPU_TYPE).half()
        d = torch.randn(64, 128).to(GPU_TYPE).half().t()

        # Track render calls
        from torch._inductor.codegen.cutlass.gemm_template import CUTLASSGemmTemplate

        original_render = CUTLASSGemmTemplate.render
        render_call_count = 0

        def counting_render(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement render method that increments the enclosing call
            # counter and delegates to the original render.
            nonlocal render_call_count
            render_call_count += 1
            return original_render(self, *args, **kwargs)

        with mock.patch.object(CUTLASSGemmTemplate, "render", counting_render):
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 2,
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                    "cutlass.enable_caching_codegen": True,
                }
            ):
                # Get expected results
                expected = model(a, b, c, d)

                # Compile and run
                compiled_model = torch.compile(model)
                actual = compiled_model(a, b, c, d)

                # Verify results
                torch.testing.assert_close(actual, expected)

        num_matmuls = 2
        expected_count = num_matmuls + num_matmuls * 2
        self.assertEqual(render_call_count, expected_count)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_multiple_mm_with_dynamic_shape(self):
        # [FUNCTION DOC] Like the multiple matmul test but one matmul has dynamic inputs and one
        # uses fixed module attributes; compiled with dynamic shapes; expects same render count
        # formula and results.
        """
        Test multiple matrix multiplications where one has dynamic shapes.
        """

        class MultipleMMDynamicModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with fixed half tensors c and d on the GPU, and a
            # forward that returns a dynamic input matmul along with the fixed matmul.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates and stores random half tensors c and d (d
                # transposed) on the GPU.
                super().__init__()
                self.c = torch.randn(64, 256).to(GPU_TYPE).half()
                self.d = torch.randn(128, 256).to(GPU_TYPE).half().t()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matmul of the two inputs and the matmul of
                # the stored tensors.
                # dynamic shape matmul
                mm1 = a @ b
                # static shape matmul
                mm2 = self.c @ self.d
                return mm1, mm2

        model = MultipleMMDynamicModel().to(GPU_TYPE)

        # Create tensors with different shapes
        a = torch.randn(128, 64).to(GPU_TYPE).half()
        b = torch.randn(32, 64).to(GPU_TYPE).half().t()

        # Track render calls
        from torch._inductor.codegen.cutlass.gemm_template import CUTLASSGemmTemplate

        original_render = CUTLASSGemmTemplate.render
        render_call_count = 0

        def counting_render(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement render method that increments the enclosing call
            # counter and delegates to the original render.
            nonlocal render_call_count
            render_call_count += 1
            return original_render(self, *args, **kwargs)

        with mock.patch.object(CUTLASSGemmTemplate, "render", counting_render):
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 2,
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                    "cutlass.enable_caching_codegen": True,
                }
            ):
                # Get expected results
                expected = model(a, b)

                # Compile and run
                compiled_model = torch.compile(model, dynamic=True)
                actual = compiled_model(a, b)

                # Verify results
                torch.testing.assert_close(actual, expected)

        num_matmuls = 2
        expected_count = num_matmuls + num_matmuls * 2
        self.assertEqual(render_call_count, expected_count)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_matmul_same_tensor(self):
        # [FUNCTION DOC] Compiles mm where the second operand is the transpose of the same tensor
        # as the first and compares to eager.
        max_autotune_gemm_backends = "CUTLASS"

        M = 128
        A = torch.randn(M, M).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            compiled = torch.compile(torch.mm)

            torch.testing.assert_close(A @ A.t(), compiled(A, A.t()))

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_matmul_nonzero_offset(self):
        # [FUNCTION DOC] Compiles mm on slices of a tensor that have a non-zero storage offset
        # and compares to eager.
        max_autotune_gemm_backends = "CUTLASS"

        M = 129
        A = torch.randn(M, M - 1).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cutlass.cutlass_max_profiling_configs": 2,
            }
        ):
            compiled = torch.compile(torch.mm)
            torch.testing.assert_close(
                A[1:, :] @ A[1:, :].t(), compiled(A[1:, :], A[1:, :].t())
            )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_flexible_layout(self):
        # [FUNCTION DOC] Compiles a model whose matmul input has a flexible (not yet fixed)
        # layout, with CUTLASS autotuning; passes if compile and run succeed (no result check).
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module matmuling a zero tensor with the transposed input.
            def forward(self, B):
                # [FUNCTION DOC] HELPER: Returns the matmul of a zeros tensor of the input shape
                # with the transposed input.
                A = torch.zeros_like(B)
                return A @ B.t()

        M = 1024
        B = torch.randn(M, M).to(GPU_TYPE).half()
        model = TestModel().to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 1,
            }
        ):
            _ = torch.compile(model)(B)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @use_evt_config
    def test_evt_flexible_layout(self):
        # [FUNCTION DOC] Same as the flexible layout test but with a relu epilogue under the EVT
        # config; asserts the epilogue fusion counter equals one.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu to a matmul with a zero tensor.
            def forward(self, B):
                # [FUNCTION DOC] HELPER: Returns relu of the matmul of zeros with the transposed
                # input.
                A = torch.zeros_like(B)
                return (A @ B.t()).relu()

        M = 1024
        B = torch.randn(M, M).to(GPU_TYPE).half()
        model = TestModel().to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 1,
            }
        ):
            _ = torch.compile(model)(B)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_filtered_ops_cache(self):
        # [FUNCTION DOC] Compiles a model that chains a hundred identical-shape matmuls and
        # asserts the filtered-ops cache miss counter is exactly one, so op filtering runs once
        # and other lowerings hit the cache.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module performing a hundred chained matmuls with the transposed
            # input.
            def forward(self, B):
                # [FUNCTION DOC] HELPER: Repeats a hundred times the matmul of an accumulator
                # initially zero with the transposed input and returns the accumulator.
                A = torch.zeros_like(B)
                for _ in range(100):
                    A = A @ B.t()
                return A

        M = 1024
        B = torch.randn(M, M).to(GPU_TYPE).half()
        model = TestModel().to(GPU_TYPE)

        counters.clear()
        with (
            fresh_cache(),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cutlass.cutlass_max_profiling_configs": 1,
                }
            ),
        ):
            _ = torch.compile(model)(B)

        # All 100 matmuls share one input shape/dtype/layout, so they map to a
        # single filtered-ops cache key: op filtering runs exactly once (one
        # miss) and the remaining lowerings hit the cache.
        self.assertEqual(counters["inductor"]["cutlass_filtered_ops_cache_miss"], 1)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("use_aoti", (False, True))
    def test_compilation_time(self, use_aoti):
        # [FUNCTION DOC] Measures wall time of compiling and running a matmul with CUTLASS, via
        # AOTI or compile; checks results (looser tolerances on XPU) and that elapsed time is
        # under 100 seconds on XPU and 120 seconds otherwise.
        M = 1024
        A = torch.randn(M, M).to(GPU_TYPE).half()
        B = torch.randn(M, M).to(GPU_TYPE).half().t()

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a matmul of two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matmul of the inputs.
                return a @ b

        model = MyModel().to(GPU_TYPE)
        expected = model(A, B)

        start_time = time.time()
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cutlass.cutlass_max_profiling_configs": 1,
            }
        ):
            if use_aoti:
                actual = AOTIRunnerUtil.run(
                    model,
                    (A, B),
                )
            else:
                actual = torch.compile(model, fullgraph=True)(A, B)

            if GPU_TYPE == "xpu":
                atol = 1e-4  # default is 1e-5
                rtol = 1e-3  # default is 1e-3
                expected_time = 100
            else:
                atol = None
                rtol = None
                expected_time = 120
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        self.assertTrue(time.time() - start_time < expected_time)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    @evt_all_ops
    @evt_all_shapes
    def test_evt_fusions_basic(self, op, shape):
        # [FUNCTION DOC] Parametrized over op and shape under the EVT config. Builds a model that
        # applies relu to a matmul and then the op (with an extra tensor for binary ops) and
        # delegates to the shared EVT driver expecting one fusion.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu after matmul then the enclosing op with
            # extra args.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Matmul, relu, then the op applied with the extra
                # arguments.
                res = (a @ b).relu()  # add extra activation to not hit addmm path
                return op(res, *extra_args)

        self.run_evt_test(TestModel(), op, shape)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    def test_evt_reshaped_external_read_fusion(self):
        # [FUNCTION DOC] Regression test for EVT shape propagation when an epilogue reads an
        # external buffer that is a compatible reshape of the GEMM output (3D versus the
        # flattened 2D template output). Compiles and checks results are close and the fusion
        # counter equals one.
        """
        Regression test: when an epilogue node reads an external buffer whose
        shape is a compatible reshape of the GEMM output (e.g. [128, 128, 128]
        vs the 2D template output [16384, 128] where 128*128 == 16384), the
        CUTLASS EVT shape propagation used to raise a dimension mismatch:
            RuntimeError: Dimension mismatch between accum(1, 16384, 128),
            arg(128, 128, 128).
        The external read is now normalized to the 2D template shape (valid
        because the read is contiguous, so the row-major flatten is
        memory-equivalent), allowing the epilogue to be fused and producing
        correct results.
        """
        torch._dynamo.utils.counters.clear()
        M, N, K = 16384, 128, 256
        reshaped = (128, 128, N)  # 128 * 128 == M

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes the matmul to 3D, adds a 3D bias and
            # applies relu.
            def forward(self, a, b, bias3d):
                # [FUNCTION DOC] HELPER: Matmul, reshape to the enclosing 3D shape, add the bias,
                # relu.
                out = a @ b  # (M, N)
                out = out.reshape(*reshaped)  # (128, 128, N)
                out = out + bias3d  # external read with the reshaped shape
                return out.relu()

        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)
        bias3d = torch.randn(*reshaped, device=GPU_TYPE, dtype=torch.float16)

        model = TestModel().to(GPU_TYPE)
        ref = model(a, b, bias3d)
        result = torch.compile(model, fullgraph=True)(a, b, bias3d)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    @evt_bin_ops
    def test_evt_broadcasting(self, op):
        # [FUNCTION DOC] Parametrized over binary and unary ops under the EVT config. Compiles a
        # model that returns both the raw accumulator and the op applied to the relu of the
        # accumulator with all-ones half inputs; asserts one epilogue fusion and results match
        # eager.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matmul result and the op applied on its
            # relu.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Returns the accumulator and the op of the relu of the
                # accumulator with the extra arguments.
                acc = a @ b
                return acc, op(acc.relu(), *extra_args)

        M = 1024
        N = 512
        a = torch.ones(M, N).to(GPU_TYPE).half()
        b = torch.ones(N, N).to(GPU_TYPE).half().t()
        extra_args = gen_args(op, (M, N))
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    @evt_un_ops
    def test_evt_activations(self, op):
        # [FUNCTION DOC] Parametrized over unary activations under the EVT config. Compiles a
        # model returning the accumulator and the activation of it for random half inputs;
        # asserts one epilogue fusion and results are close with relaxed tolerances.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matmul accumulator and the op applied to
            # it.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Returns the accumulator and the op applied to it with
                # the extra arguments.
                acc = a @ b
                return acc, op(acc, *extra_args)

        M = 1024
        N = 512
        a = torch.randn(M, N).to(GPU_TYPE).half()
        b = torch.randn(N, N).to(GPU_TYPE).half().t()
        extra_args = gen_args(op, (M, N))
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    def test_evt_scalar_constant(self):
        # [FUNCTION DOC] Checks that numeric literals in the epilogue (scale by 0.5 then subtract
        # 1.0, the latter exercising signed immediates) are lowered to scalar immediate nodes in
        # the EVT, so that rendering succeeds; asserts one fusion and close results.
        # Python scalar literals in the epilogue (e.g. * 0.5, - 1.0) are lowered
        # by the CUTLASS EVT frontend into immediate-constant ("imm_*") nodes.
        # These are not input buffers, so EVT arg rendering must recover the
        # constant value and emit it as a scalar (see get_constant_value /
        # constant handling in evt_extensions.py); without it the renderer
        # raises KeyError on the imm_ name. The negative literal additionally
        # exercises the name-parsing fallback for signed immediates.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying the matmul by a constant then subtracting a
            # constant.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the matmul times 0.5 minus 1.0.
                return (a @ b) * 0.5 - 1.0

        M = 1024
        N = 512
        a = torch.randn(M, N).to(GPU_TYPE).half()
        b = torch.randn(N, N).to(GPU_TYPE).half().t()
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b)
        ref_result = model(a, b)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    def test_evt_gelu(self):
        # [FUNCTION DOC] Checks that gelu, decomposed by Inductor, is re-folded into the native
        # CUTLASS GELU functor for the EVT; asserts one fusion and close results.
        # gelu is decomposed by Inductor into x * 0.5 * (1 + erf(x / sqrt(2)));
        # the EVT codegen re-folds this back into the native CUTLASS GELU
        # functor (see _fuse_activations in python_evt.py).
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying gelu to a matmul.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns gelu of the matmul.
                return torch.nn.functional.gelu(a @ b)

        M = 1024
        N = 512
        a = torch.randn(M, N).to(GPU_TYPE).half()
        b = torch.randn(N, N).to(GPU_TYPE).half().t()
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b)
        ref_result = model(a, b)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    def test_evt_silu_fusion(self):
        # [FUNCTION DOC] Checks that SiLU applied to a half precision matmul is fused into the
        # GEMM epilogue via EVT; asserts one fusion and close results.
        """Test that SiLU activation can be fused into GEMM epilogue via EVT."""

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying silu to a matmul.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns silu of the matmul.
                return torch.nn.functional.silu(a @ b)

        M, N = 1024, 512
        a = torch.randn(M, N, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(N, N, device=GPU_TYPE, dtype=torch.float16).t()
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b)
        ref_result = model(a, b)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    def test_evt_llama_mlp_pattern(self):
        # [FUNCTION DOC] Checks the Llama MLP pattern (silu of a gate projection multiplied by an
        # up projection) compiles with at least one epilogue fusion and matches eager.
        """Test the Llama MLP pattern: silu(x @ gate.T) * (x @ up.T)."""

        class LlamaMLP(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the Llama style gated MLP of an input and two
            # weights.
            def forward(self, x, gate_w, up_w):
                # [FUNCTION DOC] HELPER: Returns silu of the input times the transposed gate
                # weight, multiplied elementwise by the input times the transposed up weight.
                return torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())

        M, K, N = 256, 256, 256
        x = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        gate_w = torch.randn(N, K, device=GPU_TYPE, dtype=torch.float16)
        up_w = torch.randn(N, K, device=GPU_TYPE, dtype=torch.float16)
        model = LlamaMLP().to(GPU_TYPE)

        ref_result = model(x, gate_w, up_w)
        result = torch.compile(model)(x, gate_w, up_w)

        self.assertGreaterEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    def test_evt_aux_load_mul(self):
        # [FUNCTION DOC] Checks multiplying a matmul output by a second external tensor of the
        # same shape (an auxiliary load) is fused via EVT; asserts one fusion and close results.
        """Test that multiplying GEMM output with an external buffer uses AuxLoad."""

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying a matmul by an auxiliary tensor.
            def forward(self, a, b, aux):
                # [FUNCTION DOC] HELPER: Returns the matmul times the auxiliary tensor.
                return (a @ b) * aux

        M, N, K = 256, 256, 256
        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)
        aux = torch.randn(M, N, device=GPU_TYPE, dtype=torch.float16)
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b, aux)
        ref_result = model(a, b, aux)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result, atol=1e-2, rtol=1e-2)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    @evt_all_ops
    def test_evt_mixed_dtypes(self, op):
        # [FUNCTION DOC] Parametrized over all ops. Model applies the op to the relu of a matmul
        # then adds a float32 tensor. First compiles with epilogue fusion disabled as baseline
        # (fusion counter must be zero), resets dynamo, compiles again with fusion enabled
        # (counter must be one) and compares the two results.
        M = 1024
        N = 256

        fp32_tensor = torch.ones(M, N).to(GPU_TYPE).float()

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the op to relu of a matmul and adding a float32
            # tensor from the enclosing scope.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Matmul, relu, op with extra args, then add of the
                # float32 tensor.
                acc = a @ b
                out0 = op(acc.relu(), *extra_args)
                out1 = torch.add(out0, fp32_tensor)
                return out1

        model = TestModel().to(GPU_TYPE)
        a = torch.ones(M, N).to(GPU_TYPE).half()
        b = torch.ones(N, N).to(GPU_TYPE).half().t()
        extra_args = gen_args(op, (M, N), dtype=torch.float16)

        # baseline is cutlass kernel + triton
        # matches expected casting behavior
        with config.patch({"cutlass.cutlass_epilogue_fusion_enabled": False}):
            ref_result = torch.compile(model)(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            0,
        )

        torch._dynamo.reset()
        result = torch.compile(model)(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )

        torch.testing.assert_close(result, ref_result)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    @evt_all_ops
    def test_evt_multi_op(self, op):
        # [FUNCTION DOC] Parametrized over all ops. Uses a model adding the accumulator to the
        # relu of the op applied to the relu of the accumulator, and runs the shared EVT driver
        # on a fixed shape.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a multi-op epilogue that also reuses the
            # accumulator.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Adds the accumulator to the relu of the op of the relu
                # of the accumulator.
                acc = a @ b
                return torch.add(op(acc.relu(), *extra_args).relu(), acc)

        self.run_evt_test(TestModel(), op, (1024, 512))

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    @evt_all_ops
    def test_evt_reuse_matmul_input(self, op):
        # [FUNCTION DOC] Parametrized over all ops. Epilogue adds the first matmul input to the
        # result, so the shape must be square; runs the shared EVT driver for a square shape.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose epilogue re-reads the matmul left input.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Adds the first input to the relu of the op of the relu
                # of the accumulator.
                acc = a @ b
                return torch.add(op(acc.relu(), *extra_args).relu(), a)

        self.run_evt_test(TestModel(), op, (1024, 1024))  # shape needs to be square

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_all_ops
    @parametrize(
        "dynamic", (False, True)
    )  # To not drastically increase test time we only test dynamic on this test
    @xfailIfSM120OrLater
    def test_evt_multi_output(self, op, dynamic):
        # [FUNCTION DOC] Parametrized over ops and dynamic flag (dynamic only here, to limit test
        # time). Compiles a model with two outputs (op result and op result plus relu) for one
        # static or two shapes, and after each shape checks the cumulative fusion counter is two
        # times the number of shapes run so far and results match eager.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the op result and its sum with the relu of the
            # accumulator.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Returns the op of the relu and that value plus the relu.
                acc = a @ b
                z0 = acc.relu()
                z = op(z0, *extra_args)
                y = z + z0
                return z, y

        M = 1024
        N = 512
        shapes = [(512, 512)] if not dynamic else [(1024, 64), (128, 256)]
        for i, shape in enumerate(shapes):
            M, N = shape
            a = torch.ones(M, N).to(GPU_TYPE).half()
            b = torch.ones(N, N).to(GPU_TYPE).half().t()
            extra_args = gen_args(op, (M, N))
            model = TestModel().to(GPU_TYPE)

            result = torch.compile(model)(a, b, extra_args)
            ref_result = model(a, b, extra_args)

            self.assertEqual(
                torch._dynamo.utils.counters["inductor"][
                    "cutlass_epilogue_fusion_counter"
                ],
                2 * (i + 1),
            )
            torch.testing.assert_close(result, ref_result)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    @use_evt_config
    def test_evt_return_accumulator(self):
        # [FUNCTION DOC] Checks an epilogue that returns both the raw accumulator and a derived
        # add output; asserts one fusion and that results match eager.
        op = torch.add

        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the accumulator and the add of its relu with
            # an extra arg.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Returns the accumulator and the op of the relu of the
                # accumulator.
                acc = a @ b
                return acc, op(acc.relu(), *extra_args)

        M = 1024
        N = 512
        a = torch.ones(M, N).to(GPU_TYPE).half()
        b = torch.ones(N, N).to(GPU_TYPE).half().t()
        extra_args = gen_args(op, (M, N))
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_un_ops
    def test_evt_reshape(self, op):
        # [FUNCTION DOC] Parametrized over unary ops. Matmul on flattened 3D input followed by a
        # view back to 3D and the op; checks one fusion and results with all-ones inputs.
        # mm [M, N] -> view [batch, seq, N] -> pointwise op
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module flattening a 3D input, matmul, reshaping back and
            # applying the op.
            def forward(self, a, b, extra_args):
                # [FUNCTION DOC] HELPER: Flatten leading dims, matmul, view back to batch and
                # sequence, apply op.
                # a is 3D [batch, seq, K], b is 2D [K, N]
                batch, seq = a.shape[0], a.shape[1]
                acc = a.view(-1, a.shape[-1]) @ b
                acc = acc.view(batch, seq, -1)
                return op(acc, *extra_args)

        N = 512
        # Use batch=2, seq=512 so that M=batch*seq=1024.
        # Use ones (like run_evt_test) to avoid fp16 numerical noise.
        a = torch.ones(2, 512, N).to(GPU_TYPE).half()
        b = torch.ones(N, N).to(GPU_TYPE).half().t()
        extra_args = gen_args(op, (2, 512, N))
        model = TestModel().to(GPU_TYPE)

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            counters["inductor"]["cutlass_epilogue_fusion_counter"],
            1,
        )
        torch.testing.assert_close(result, ref_result)

    def _test_gemm_operation_serialization(
        self, arch: str, cuda_version: str, min_ops=1000
    ):
        # [FUNCTION DOC] HELPER: Shared check of operation serialization. Generates the cached
        # CUTLASS operations for an arch and version, flattens them, requires more than min_ops,
        # requires all configuration names to be unique, then serializes and deserializes each
        # with the CUTLASS serializer and checks round-trip equality via the equality helper.
        """
        Testing serialization for GEMM operations generated by CUTLASS.
        This should cover GroupedGemmOperation as well.
        """
        full_ops = _gen_ops_cached(arch, cuda_version, GPU_TYPE)
        ops = pytree.tree_flatten(full_ops)[0]

        # sanity check
        self.assertGreater(len(ops), min_ops, "Too few ops generated")

        # test if configuration name is unique
        op_config_names = [op.configuration_name() for op in ops]
        self.assertEqual(len(op_config_names), len(set(op_config_names)))

        serializer = get_cutlass_operation_serializer()
        self.assertIsNotNone(serializer)

        serialized_ops = [serializer.serialize(op) for op in ops]
        deserialized_ops = [
            serializer.deserialize(serialized_op) for serialized_op in serialized_ops
        ]
        for op, deserialized_op in zip(ops, deserialized_ops, strict=False):
            self.assertTrue(_check_if_instances_equal(op, deserialized_op))

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("arch", ("90", "100"))
    @parametrize("cuda_version", ("12.4", "12.8"))
    def test_gemm_operation_serialization_cuda(self, arch: str, cuda_version: str):
        # [FUNCTION DOC] Runs the serialization check for the parametrized CUDA arch and CUDA
        # version.
        self._test_gemm_operation_serialization(arch, cuda_version)

    @skipXPUIf(not Xe2_Or_Later, "")
    @unittest.skipIf(not HAS_XPU, "XPU not available")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("arch", ("Xe12", "Xe20"))
    @parametrize("xpu_version", ("20250201", "20250301"))
    def test_gemm_operation_serialization_xpu(self, arch: str, xpu_version: str):
        # [FUNCTION DOC] Normalizes the XPU arch name, then runs the serialization check with a
        # lower minimum operation count of 40.
        from torch._inductor.codegen.cutlass.utils import _normalize_xpu_arch

        arch = _normalize_xpu_arch(arch)
        self._test_gemm_operation_serialization(arch, xpu_version, min_ops=40)

    @unittest.skipIf(
        torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+",
    )
    @skipXPUIf(True, "XPU SYCL-TLA has not supported fp8 yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @fp8_config
    @parametrize("float8_dtype", (torch.float8_e4m3fn,))
    @parametrize(
        "shape",
        (
            (
                512,
                128,
                64,
            ),
        ),
    )
    @xfailIfSM120OrLater
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize("input_dtype", (torch.bfloat16, torch.float16))
    def test_fp8_rowwise_scaling(
        self,
        float8_dtype: torch.dtype,
        shape: tuple[int, int, int],
        has_bias: bool,
        use_fast_accum: bool,
        input_dtype: torch.dtype,
    ):
        # [FUNCTION DOC] Compiles a row-wise scaled FP8 mm, optionally with bias and fast
        # accumulation, versus eager; asserts output dtype bf16 for both and closeness with
        # relaxed tolerances.
        # Only bf16 output type is supported for row-wise scaling, not fp32
        output_dtype: torch.dtype = torch.bfloat16
        device = GPU_TYPE
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=input_dtype, device=device)
        w = torch.randn(N, K, dtype=input_dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=input_dtype).to(torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, float8_dtype)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, float8_dtype)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: Scaled mm of the quantized operands with the given scales
            # and bias using the enclosing output dtype and fast accumulation flag.
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=output_dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        linear_compiled = torch.compile(linear, backend="inductor")
        y_compiled = linear_compiled(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        self.assertEqual(y_eager.dtype, output_dtype)
        self.assertEqual(y_compiled.dtype, output_dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(
        torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+",
    )
    @skipXPUIf(True, "XPU SYCL-TLA has not supported fp8 yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @fp8_config
    @parametrize("float8_dtype", (torch.float8_e4m3fn,))
    @parametrize(
        "shape",
        (
            (
                512,
                1024,
            ),
        ),
    )
    @xfailIfSM120OrLater
    @parametrize("use_fast_accum", (True,))
    @parametrize("use_aoti", (False, True))
    @parametrize("dynamic", (False, True))
    def test_fp8_rowwise_scaling_multiple_linear(
        self,
        float8_dtype: torch.dtype,
        shape: tuple[int, int],
        use_fast_accum: bool,
        use_aoti: bool = False,
        dynamic: bool = False,
    ):
        # [FUNCTION DOC] Simulates a realistic scenario: two chained row-wise scaled FP8 linear
        # layers with requantization between them, run via AOTI or compile, optionally dynamic.
        # Skipped when both AOTI and dynamic are set. Compares with eager using relaxed
        # tolerances.
        """
        This test is meant to simulate a more realistic scenario.
        """
        if dynamic and use_aoti:
            self.skipTest("Accuracy issues when both AOTI and dynamic are enabled")
        # Only bf16 output type is supported for row-wise scaling, not fp32
        output_dtype: torch.dtype = torch.bfloat16
        device = GPU_TYPE
        M, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, N, dtype=output_dtype, device=device)
        w1 = torch.randn(N, N, dtype=output_dtype, device=device)
        w2 = torch.randn(N, N, dtype=output_dtype, device=device)

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two pre-quantized transposed fp8 weights and their
            # scales, and a forward that quantizes activations, runs the first scaled mm,
            # requantizes and runs the second.
            def __init__(self, w1, w2, float8_dtype):
                # [FUNCTION DOC] HELPER: Quantizes both weights rowwise, stores scales,
                # transposed fp8 weights and the fp8 dtype as attributes.
                super().__init__()
                w1_fp8, self.w1_inverse_scale = _quantize_rowwise(w1, float8_dtype)
                w2_fp8, self.w2_inverse_scale = _quantize_rowwise(w2, float8_dtype)

                self.w1_t_fp8 = w1_fp8.t()
                self.w2_t_fp8 = w2_fp8.t()

                self.float8_dtype = float8_dtype

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the input, runs two scaled mms with a
                # requantization of the first result between them, with bf16 output, and returns
                # the second.
                x_fp8, x_inverse_scale = _quantize_rowwise(x, self.float8_dtype)
                y1 = torch._scaled_mm(
                    x_fp8,
                    self.w1_t_fp8,
                    x_inverse_scale.view(-1, 1),
                    self.w1_inverse_scale.view(1, -1),
                    out_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                )

                y1_fp8, y1_inverse_scale = _quantize_rowwise(y1, self.float8_dtype)
                y2 = torch._scaled_mm(
                    y1_fp8,
                    self.w2_t_fp8,
                    y1_inverse_scale.view(-1, 1),
                    self.w2_inverse_scale.view(1, -1),
                    out_dtype=output_dtype,
                    use_fast_accum=use_fast_accum,
                )
                return y2

        model = TestModule(w1, w2, float8_dtype).to(GPU_TYPE)

        dynamic_shapes = (
            {
                "x": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )

        expected = model(x)

        if use_aoti:
            actual = AOTIRunnerUtil.run(
                model,
                (x,),
                dynamic_shapes=dynamic_shapes,
            )
        else:
            compiled_model = torch.compile(model, fullgraph=True, dynamic=dynamic)
            actual = compiled_model(x)

        torch.testing.assert_close(expected, actual, rtol=1e-2, atol=0.05)

    @unittest.skipIf(
        torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+",
    )
    @skipXPUIf(True, "XPU SYCL-TLA has not supported fp8 yet")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @fp8_config
    @parametrize("float8_dtype", (torch.float8_e4m3fn,))
    @parametrize(
        "shape",
        (
            (
                512,
                128,
                64,
            ),
        ),
    )
    @xfailIfSM120OrLater
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False,))
    @parametrize("input_dtype", (torch.bfloat16, torch.float16))
    def test_fp8_tensorwise_scaling(
        self,
        float8_dtype: torch.dtype,
        shape: tuple[int, int, int],
        has_bias: bool,
        use_fast_accum: bool,
        input_dtype: torch.dtype,
    ):
        # [FUNCTION DOC] Compiles a tensor-wise scaled FP8 mm in max-autotune mode, optionally
        # with bias, versus eager; output dtype equals input dtype; checks dtypes and closeness
        # with a small absolute tolerance since accumulation order depends on the selected
        # kernel.
        device = GPU_TYPE
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        output_dtype = input_dtype
        # input and output dtypes of _scaled_mm do not need to be the same, but
        # typically in a model they are
        x = torch.randn(M, K, dtype=input_dtype, device=device)
        w = torch.randn(N, K, dtype=input_dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=input_dtype)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, float8_dtype)
        w_t_fp8 = w_fp8.t()

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, float8_dtype)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: Scaled mm of the quantized operands with tensor-wise scales
            # and bias, with enclosing output dtype and fast accumulation flag.
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=output_dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        y_compiled = linear_compiled(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        self.assertEqual(y_eager.dtype, output_dtype)
        self.assertEqual(y_compiled.dtype, output_dtype)
        # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
        # autotuning for the compiled case, the results can be different because of
        # the way blocks of results are accumulated (float addition not associative), so
        # setting a small absolute tolerance in these tests
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @skipXPUIf(not Xe2_Or_Later, "")
    @skipCUDAIf(not SM90OrLater, "need sm_90")
    @xfailIfSM120OrLater
    def test_config_number_post_filtering(self) -> None:
        # [FUNCTION DOC] For four layout combos and two dtypes, compiles mm with caches disabled
        # and selection mocked to raise BackendCompilerFailed with NoValidChoicesError, records
        # the number of choices per combination, and asserts the number of distinct counts is one
        # (two on XPU).
        """
        Test if cutlass backend produces the same number of configs after filtering
        regardless of layout and dtype.
        """
        layouts = ["rr", "rc", "cr", "cc"]
        dtypes = [torch.float16, torch.bfloat16]

        config_counts = {}

        for layout in layouts:
            for dtype in dtypes:
                a = torch.randn(128, 128, dtype=dtype).to(GPU_TYPE)
                b = torch.randn(128, 128, dtype=dtype).to(GPU_TYPE)
                if layout[0] == "c":
                    a = a.t()
                if layout[1] == "c":
                    b = b.t()

                with config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "CUTLASS",
                        # needed for log searching
                        "force_disable_caches": True,
                        "cutlass.cutlass_max_profiling_swizzle_options": [2],
                    }
                ):
                    with mock.patch(
                        "torch._inductor.kernel.mm.autotune_select_algorithm",
                        wraps=select_no_algorithm,
                    ) as sa:
                        with self.assertRaisesRegex(
                            BackendCompilerFailed, r".*NoValidChoicesError.*"
                        ):
                            _ = torch.compile(torch.mm, dynamic=False)(a, b)
                        args, _ = sa.call_args
                        _, choices, _, __ = args

                        config_counts[(layout, dtype)] = len(choices)

        # Check that all config counts are equal
        all_counts = list(config_counts.values())
        # XPU has more configs for bf16 than f16.
        expected_count = 2 if GPU_TYPE == "xpu" else 1
        self.assertTrue(
            len(set(all_counts)) == expected_count,
            lambda msg: f"{msg}\nConfig counts should be equal across all layout/dtype combinations. "
            f"Got counts: {config_counts}",
        )


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
