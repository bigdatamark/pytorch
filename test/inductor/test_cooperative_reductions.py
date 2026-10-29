# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for Triton cooperative reductions (reductions split across a grid
# with a grid barrier), the variance two-step vs Welford heuristic, and fixed Triton configs
# injected through a custom choices handler. All GPU-dependent; tests run only when a GPU is
# available. No import-time side effects besides imports.
from typing import Any

import sympy

import torch
import torch._inductor
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import FixedTritonConfig, TritonKernel
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import assert_close
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestVarianceReductionHeuristic(TestCase):
    # [CLASS DOC] Tests that the compiler picks the two-step variance path or the Welford
    # reduction for var_mean depending on dtype, reduction size and whether the reduction is
    # split. CUDA only (each test skips otherwise).
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, then resets dynamo state so each test
        # compiles fresh.
        super().setUp()
        torch._dynamo.reset()

    def _skip_if_not_cuda(self):
        # [FUNCTION DOC] HELPER: Skips the current test unless the GPU type is cuda.
        if GPU_TYPE != "cuda":
            self.skipTest("CUDA-specific variance heuristic")

    def _dtypes(self):
        # [FUNCTION DOC] HELPER: Returns the dtypes to test: float16, plus bfloat16 when CUDA
        # reports bf16 support.
        dtypes = [torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        return dtypes

    def test_var_mean_uses_two_step_for_non_split_reductions(self):
        # [FUNCTION DOC] For a 4 by 4096 half-precision input, compiles var_mean over the last
        # dim with zero correction and checks it matches eager and that the generated code uses a
        # plain sum and no Welford reduce helper. Skips unless CUDA.
        self._skip_if_not_cuda()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns var_mean over the last dim with zero correction.
            return torch.var_mean(x, dim=-1, correction=0)

        for dtype in self._dtypes():
            x = torch.randn([4, 4096], device=GPU_TYPE, dtype=dtype)
            expected = fn(x)
            result, (source_code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x
            )

            self.assertEqual(result, expected)
            self.assertIn("tl.sum", source_code)
            self.assertNotIn("welford_reduce", source_code)

    def test_var_mean_keeps_welford_for_float32_reductions(self):
        # [FUNCTION DOC] For a float32 4 by 4096 input, checks the compiled var_mean matches
        # eager and that the generated code still uses Welford helpers (no two-step variance for
        # float32). Skips unless CUDA.
        self._skip_if_not_cuda()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns var_mean over the last dim with zero correction.
            return torch.var_mean(x, dim=-1, correction=0)

        x = torch.randn([4, 4096], device=GPU_TYPE, dtype=torch.float32)
        expected = fn(x)
        result, (source_code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)

        self.assertEqual(result, expected)
        self.assertIn("welford_", source_code)

    @config.patch("triton.force_cooperative_reductions", True)
    def test_var_mean_keeps_welford_for_small_reductions(self):
        # [FUNCTION DOC] With cooperative reductions forced on, checks that a small (512 wide)
        # half-precision var_mean still compiles to Welford helpers and matches eager. Skips
        # unless CUDA.
        self._skip_if_not_cuda()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns var_mean over the last dim with zero correction.
            return torch.var_mean(x, dim=-1, correction=0)

        for dtype in self._dtypes():
            x = torch.randn([4, 512], device=GPU_TYPE, dtype=dtype)
            expected = fn(x)
            result, (source_code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x
            )

            self.assertEqual(result, expected)
            self.assertIn("welford_", source_code)

    @config.patch(
        {
            "triton.cooperative_reductions": False,
            "triton.force_cooperative_reductions": False,
            "triton.use_two_step_variance_threshold": 2_000_000,
        }
    )
    def test_var_mean_keeps_welford_for_split_reductions(self):
        # [FUNCTION DOC] With cooperative reductions disabled and a high two-step threshold,
        # checks that a 2 by 1M-element half-precision var_mean (a split reduction) keeps Welford
        # helpers and matches eager. Skips unless CUDA.
        self._skip_if_not_cuda()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns var_mean over the last dim with zero correction.
            return torch.var_mean(x, dim=-1, correction=0)

        for dtype in self._dtypes():
            x = torch.randn([2, 1024 * 1024], device=GPU_TYPE, dtype=dtype)
            expected = fn(x)
            result, (source_code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x
            )

            self.assertEqual(result, expected)
            self.assertIn("welford_", source_code)


class _TestingHeuristics(InductorChoices):
    # [CLASS DOC] HELPER: InductorChoices subclass for tests that forces every Triton kernel to
    # use given cooperative and persistent reduction overrides and a fixed config. State:
    # cooperative, persistent, cfg, and call_count (number of kernel-kwargs requests).
    def __init__(self, *, cooperative: bool, persistent: bool, cfg: dict[str, int]):
        # [FUNCTION DOC] HELPER: Stores the keyword-only cooperative flag, persistent flag and
        # config mapping, and initializes the call counter to zero.
        super().__init__()
        self.cooperative = cooperative
        self.persistent = persistent
        self.cfg = cfg
        self.call_count = 0

    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Increments the call counter and returns the incoming kernel
        # kwargs extended with the override_cooperative_reduction and
        # override_persistent_reduction settings and a FixedTritonConfig built from the
        # stored config. Arguments other than kernel_kwargs are unused.
        self.call_count += 1
        return {
            **kernel_kwargs,
            "override_cooperative_reduction": self.cooperative,
            "override_persistent_reduction": self.persistent,
            "fixed_config": FixedTritonConfig(self.cfg),
        }


@config.patch(
    {
        "triton.cooperative_reductions": True,
        "triton.force_cooperative_reductions": True,
    }
)
@instantiate_parametrized_tests
class CooperativeReductionTests(TestCase):
    # [CLASS DOC] Tests cooperative reductions with cooperative reductions enabled and forced by
    # config. Parametrized tests are instantiated by decorator. Subclasses below rerun all tests
    # with persistent reductions or multi-kernel toggled. setUp zeroes the generated kernel
    # count.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, zeroes the inductor generated-kernel
        # counter and resets dynamo.
        super().setUp()
        torch._inductor.metrics.generated_kernel_count = 0
        torch._dynamo.reset()

    def run_and_check(self, fn, args, dtype=None, *, expect_kernel_count=1):
        # [FUNCTION DOC] HELPER: Casts args to float64 for float16 and float32 inputs (otherwise
        # keeps the dtype) and runs fn eagerly as reference, converting it to dtype (float64 if
        # None), compiles fn with fullgraph and runs it on args collecting generated code, then
        # compares with dtype-dependent tolerances (looser for half precision). Tuple or list
        # results are compared element by element, non-tensor items by equality. Also asserts the
        # code uses a cooperative grid (fixed-config or cooperative_reduction decorator), a
        # cooperative launch flag on CUDA, and, unless multi-kernel is used, that the generated
        # kernel count equals expect_kernel_count. Returns the generated source code string.
        # Cooperative reductions accumulate in a different order than eager, so a
        # single fixed tolerance is too tight for large-magnitude reductions
        # (e.g. sum over ~1M elements). Scale rtol with the input dtype's
        # precision, keeping atol tight and the float64/default path tight. sum is
        # a large-magnitude relative-error problem, so rtol alone covers it; a loose
        # atol would slash sensitivity for small-magnitude outputs (softmax ~1e-6,
        # mean over 1M ~1e-3) sharing this parametrization.
        RTOL, ATOL = {
            torch.float16: (1e-2, 1e-6),
            torch.bfloat16: (1e-2, 1e-6),
            torch.float32: (1e-3, 1e-6),
        }.get(dtype, (1e-5, 1e-6))

        # Compute the reference in float64 for reduced-precision inputs so the
        # eager baseline does not itself carry the same accumulation error that
        # the compiled reduction does.
        ref_dtype = dtype
        if dtype in (torch.float16, torch.float32):
            ref_dtype = torch.float64

        # Cast to the determined reference dtype
        args_ref = [tensor.to(ref_dtype) for tensor in args]

        # Calculate expected output
        raw_expected = fn(*args_ref)

        if isinstance(raw_expected, (tuple, list)):
            # If it's a tuple or list, apply .to(dtype) to each tensor within it
            # Also, handle cases where dtype might not be provided (e.g., for bool reductions)
            if dtype is not None:
                expected = type(raw_expected)(
                    [
                        t.to(dtype) if isinstance(t, torch.Tensor) else t
                        for t in raw_expected
                    ]
                )
            else:
                expected = type(raw_expected)(
                    [
                        t.to(torch.float64) if isinstance(t, torch.Tensor) else t
                        for t in raw_expected
                    ]
                )
        else:
            # If it's a single tensor
            if dtype is not None:
                expected = raw_expected.to(dtype)
            else:
                expected = raw_expected.to(torch.float64)

        fn_compiled = torch.compile(fn, fullgraph=True)
        result, (source_code,) = run_and_get_code(fn_compiled, *args)

        # For comparison, ensure result is also a tuple/list if expected is
        if isinstance(expected, (tuple, list)):
            if isinstance(result, torch.Tensor):
                result = (result,)
            elif not isinstance(result, type(expected)):
                result = type(expected)(result)

            if dtype is not None:
                result = type(result)(
                    [t.to(dtype) if isinstance(t, torch.Tensor) else t for t in result]
                )
            else:
                result = type(result)(
                    [
                        t.to(torch.float64) if isinstance(t, torch.Tensor) else t
                        for t in result
                    ]
                )
        else:
            if dtype is not None and isinstance(result, torch.Tensor):
                result = result.to(dtype)
            elif isinstance(result, torch.Tensor):
                result = result.to(torch.float64)

        # Apply assert_close with fixed tolerances for tensor comparisons
        if isinstance(result, torch.Tensor) and isinstance(expected, torch.Tensor):
            assert_close(result, expected, rtol=RTOL, atol=ATOL)
        elif isinstance(result, (tuple, list)) and isinstance(expected, (tuple, list)):
            # Iterate through elements for comparison
            for r_item, e_item in zip(result, expected):
                if isinstance(r_item, torch.Tensor) and isinstance(
                    e_item, torch.Tensor
                ):
                    assert_close(r_item, e_item, rtol=RTOL, atol=ATOL)
                else:
                    # Fallback to assertEqual for non-tensor elements (e.g., bool, int)
                    self.assertEqual(r_item, e_item)
        else:
            # Fallback to assertEqual for other types not handled by assert_close
            self.assertEqual(result, expected)

        if "@triton_heuristics.fixed_config" in source_code:
            self.assertIn("cooperative_reduction_grid", source_code)
        else:
            self.assertIn("@triton_heuristics.cooperative_reduction", source_code)
        if GPU_TYPE == "cuda":
            self.assertIn("'launch_cooperative_grid': True", source_code)
        if "async_compile.multi_kernel" not in source_code:
            self.assertEqual(
                torch._inductor.metrics.generated_kernel_count, expect_kernel_count
            )
        return source_code

    @parametrize(
        "name",
        [
            "sum",
            "mean",
            "prod",
            "amin",
            "amax",
            "min",
            "max",
            "var_mean",
            "std",
            "softmax",
        ],
    )
    @parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_reduction_fns(self, name, dtype):
        # [FUNCTION DOC] Parametrized over reduction names (sum, mean, prod, amin, amax, min,
        # max, var_mean, std, softmax) and float dtypes. Reduces the sum of two 1M-element rows
        # along the last dim and checks the result through the shared checker.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies the selected reduction to the sum of the two inputs
            # over the last dim.
            return reduction_fn(x + y, dim=-1)

        reduction_fn = getattr(torch, name)
        args = [torch.randn(1, 1024**2, device=GPU_TYPE, dtype=dtype) for _ in range(2)]
        self.run_and_check(fn, args, dtype)

    def test_bool_reduction_fns(self):
        # [FUNCTION DOC] Compiles six full boolean reductions (any and all over equality,
        # inequality and ordering comparisons) of two 1024-element tensors and checks
        # correctness. Unless multi-kernel is used, also verifies that none of the per-split-id
        # conditional blocks appear before the grid barrier and all six appear after it.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a list of six any/all reductions over comparisons of
            # the two inputs.
            return [
                torch.any(x == y),
                torch.all(x == y),
                torch.any(x != y),
                torch.all(x != y),
                torch.any(x < y),
                torch.all(x > y),
            ]

        args = [torch.randn(1024, device=GPU_TYPE) for _ in range(2)]
        source_code = self.run_and_check(fn, args)
        if "async_compile.multi_kernel" in source_code:
            return
        before, after = source_code.split("triton_helpers.x_grid_barrier")
        self.assertEqual(before.count("if rsplit_id == ("), 0)
        self.assertEqual(after.count("if rsplit_id == ("), 6)

    @parametrize("bs", [1, 2, 5, 15])
    @parametrize("count", [1024**2 + 1, 1024**2 - 1, 1024])
    def test_non_power_of_2(self, bs, count):
        # [FUNCTION DOC] Parametrized over batch sizes and reduction lengths around a power of
        # two. Checks mean and (std plus min) of a random matrix compile correctly with
        # cooperative reductions.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the mean, and the sum of std and min, of the input.
            return x.mean(), x.std() + x.min()

        args = [torch.randn([bs, count], device=GPU_TYPE)]
        self.run_and_check(fn, args)

    def test_chained_reductions(self):
        # [FUNCTION DOC] Compiles eight chained residual softmax steps on a 4 by 100000 input and
        # checks correctness. Unless multi-kernel is used, asserts the number of grid barriers (8
        # with online softmax, else 16) and that exactly five GPU buffers are allocated.
        def fn(x):
            # [FUNCTION DOC] HELPER: Repeats eight times adding the row softmax of the tensor to
            # itself; returns the result.
            for _ in range(8):
                x = x + torch.softmax(x, 1)
            return x

        args = [torch.randn(4, 100000, device=GPU_TYPE)]
        source_code = self.run_and_check(fn, args)
        if "async_compile.multi_kernel" in source_code:
            return

        # With online softmax, the computation of max and sum are done
        # jointly and they share a single barrier call.
        expected_num_barrier = 8 if config.online_softmax else 16
        self.assertEqual(
            source_code.count("triton_helpers.x_grid_barrier"), expected_num_barrier
        )
        self.assertEqual(source_code.count(f"empty_strided_{GPU_TYPE}"), 5)

    def test_reduce_split(self):
        # [FUNCTION DOC] Checks that a vector norm of a large matrix and a column sum of a small
        # one compile into two kernels (expect_kernel_count of 2) with correct results.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the vector norm of the first input and the dim-0 sum
            # of the second.
            a1 = torch.linalg.vector_norm(a)
            b1 = torch.sum(b, dim=0)
            return a1, b1

        inps = [
            torch.rand(2048, 512, device=GPU_TYPE),
            torch.rand(20, 20, device=GPU_TYPE),
        ]
        self.run_and_check(fn, inps, expect_kernel_count=2)


@config.patch("triton.persistent_reductions", not config.triton.persistent_reductions)
class NoPersistCooperativeReductionTests(CooperativeReductionTests):
    # [CLASS DOC] Reruns every cooperative reduction test with the persistent reductions config
    # flipped from its default.
    pass


@config.patch("triton.multi_kernel", int(not config.triton.multi_kernel))
class MultiKernelCooperativeReductionTests(CooperativeReductionTests):
    # [CLASS DOC] Reruns every cooperative reduction test with the multi-kernel config flipped
    # from its default.
    pass


@config.patch(
    {
        "triton.cooperative_reductions": True,
    }
)
@instantiate_parametrized_tests
class TestFixedConfigs(TestCase):
    # [CLASS DOC] Tests cooperative-capable reductions compiled with fixed Triton configs
    # supplied by a testing choices handler, covering persistent and non-persistent, cooperative
    # and non-cooperative modes and non-power-of-two split sizes.
    def _check(self, fn, args, *, persistent=False, cooperative=True, cfg):
        # [FUNCTION DOC] HELPER: Computes the eager result, installs a testing heuristics handler
        # with the given flags and config as the choices handler while compiling and running fn,
        # then asserts the result equals eager, the handler was consulted exactly once, and the
        # code uses a fixed-config decorator.
        expected = fn(*args)
        heuristic = _TestingHeuristics(
            persistent=persistent, cooperative=cooperative, cfg=cfg
        )
        with torch._inductor.virtualized.V.set_choices_handler(heuristic):
            result, (source_code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), *args
            )
        self.assertEqual(result, expected)
        self.assertEqual(heuristic.call_count, 1)
        self.assertIn("@triton_heuristics.fixed_config(", source_code)

    @parametrize(
        "persistent,cooperative,cfg",
        [
            (False, False, {"XBLOCK": 1, "R0_BLOCK": 128}),
            (False, False, {"XBLOCK": 2, "R0_BLOCK": 128}),
            (True, False, {"XBLOCK": 1}),
            (True, False, {"XBLOCK": 2}),
            (False, True, {"XBLOCK": 1, "R0_BLOCK": 128, "RSPLIT": 16}),
            (False, True, {"XBLOCK": 2, "R0_BLOCK": 128, "RSPLIT": 16}),
            (True, True, {"XBLOCK": 1, "RSPLIT": 16}),
            (True, True, {"XBLOCK": 2, "RSPLIT": 16}),
            (False, True, {"XBLOCK": 1, "R0_BLOCK": 128, "RSPLIT": 17}),
            (False, True, {"XBLOCK": 2, "R0_BLOCK": 128, "RSPLIT": 17}),
            (True, True, {"XBLOCK": 1, "RSPLIT": 17}),
            (True, True, {"XBLOCK": 2, "RSPLIT": 17}),
        ],
    )
    def test_fixed_configs(self, persistent, cooperative, cfg):
        # [FUNCTION DOC] Parametrized over persistent, cooperative and config combinations
        # (XBLOCK, R0_BLOCK, RSPLIT values). Checks a softmax-plus-input function on an 8 by 8000
        # tensor under each fixed config.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns softmax of the input plus one over the last dim,
            # added to the input.
            return torch.softmax(x + 1, dim=-1) + x

        args = [torch.randn(8, 8000, device=GPU_TYPE)]
        self._check(fn, args, persistent=persistent, cooperative=cooperative, cfg=cfg)

    @parametrize(
        "persistent,x,r,rsplit",
        [
            (False, 1, 8000, 17),
            (False, 4, 8123, 33),
            (False, 9, 8000, 17),
            (False, 1, 8192, 33),
            (False, 3, 8192, 17),
            (True, 1, 7567, 17),
            (True, 4, 8000, 17),
            (True, 9, 8000, 37),
            (True, 1, 8192, 17),
            (True, 3, 8192, 40),
        ],
    )
    def test_welford_non_power_of_2_rsplit(self, persistent, x, r, rsplit):
        # [FUNCTION DOC] Parametrized over shapes and non-power-of-two split counts. Checks
        # var_mean over the last dim with a fixed config (XBLOCK 64, 8 warps, plus an R0_BLOCK of
        # 64 when not persistent).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns var_mean over the last dim.
            return torch.var_mean(x, dim=-1)

        cfg = {"XBLOCK": 64, "RSPLIT": rsplit, "num_warps": 8}
        if not persistent:
            cfg["R0_BLOCK"] = 64
        args = [torch.randn(x, r, device=GPU_TYPE)]
        self._check(fn, args, persistent=persistent, cfg=cfg)

    @parametrize("persistent", [True, False])
    def test_min_max_non_power_of_2_rsplit(self, persistent):
        # [FUNCTION DOC] Checks amin, amax, argmin and argmax with a split count of 33 for
        # persistent and non-persistent configs, first on an integer matrix of increasing and
        # decreasing ranges and then on float rows containing zeros followed by positive or
        # negative infinity.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns amin, amax, argmin and argmax over the last dim as a
            # tuple.
            return (
                torch.amin(x, dim=-1),
                torch.amax(x, dim=-1),
                torch.argmin(x, dim=-1),
                torch.argmax(x, dim=-1),
            )

        cfg = {"XBLOCK": 2, "RSPLIT": 33, "num_warps": 8}
        if not persistent:
            cfg["R0_BLOCK"] = 32

        args = [
            torch.stack(
                [
                    torch.arange(10, 4096, device=GPU_TYPE),
                    -torch.arange(10, 4096, device=GPU_TYPE),
                ]
            )
        ]
        self._check(fn, args, persistent=persistent, cfg=cfg)
        args = [
            torch.stack(
                [
                    torch.tensor(
                        [0.0] * 150 + [float("inf")] * 150,
                        device=GPU_TYPE,
                        dtype=torch.float32,
                    ),
                    torch.tensor(
                        [0.0] * 150 + [-float("inf")] * 150,
                        device=GPU_TYPE,
                        dtype=torch.float32,
                    ),
                ]
            )
        ]
        self._check(fn, args, persistent=persistent, cfg=cfg)

    @parametrize("persistent", [False, True])
    @parametrize("rsplit", [32, 33])
    def test_fixed_config_with_larger_xblock_than_xnumel(self, persistent, rsplit):
        # [FUNCTION DOC] Parametrized over persistence and split counts 32 and 33. Uses an XBLOCK
        # of 128, larger than the number of outputs, on boolean any/all reductions and a mean of
        # two 1024-element tensors and checks correctness.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a list of any, all, any-not-equal, all-not-equal
            # reductions and the mean of the sum of the inputs.
            return [
                torch.any(x == y),
                torch.all(x == y),
                torch.any(x != y),
                torch.all(x != y),
                torch.mean(x + y),
            ]

        cfg = {"XBLOCK": 128, "RSPLIT": rsplit, "num_warps": 16, "num_stages": 1}
        if not persistent:
            cfg["R0_BLOCK"] = 64
        args = [torch.randn(1024, device=GPU_TYPE) for _ in range(2)]
        self._check(fn, args, persistent=persistent, cfg=cfg)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
