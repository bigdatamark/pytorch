# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that Inductor's strict numerics mode compiles sums over the innermost
# dimension into the Triton inner-tree reduction ordering, and that the results are bitwise equal
# to the eager inner-tree sum. Covers persistent, looped and split reductions, dtypes, dynamic
# shapes, layouts, signed zero, special values, fusion with other reductions, combo kernels and
# TMA. Defines the case tables used by the parametrized tests; the test class is instantiated for
# CUDA only.
"""Tests for strict inner-contiguous sum ordering."""

import os
import unittest
from unittest import mock

import torch
from torch._inductor import config, metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._native.ops.sum.inner_tree_plan import compute_inner_tree_params, vec_size
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TEST_CUTEDSL,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._triton import has_triton_reduction_ordering


def _singleton_input(device):
    # [FUNCTION DOC] HELPER: Builds a 1 by 1 float tensor with strides (1, 0) viewing a single
    # negative-zero element on the given device; used as a singleton-dim layout case.
    return torch.as_strided(torch.full((1,), -0.0, device=device), (1, 1), (1, 0))


SUM_CASES = (
    ("persistent_fp16", (64, 256), 1, torch.float16),
    ("looped_bf16", (8, 12000), 1, torch.bfloat16),
    ("split_fp32", (8, 65536), 1, torch.float32),
    ("persistent_fp64", (64, 256), 1, torch.float64),
    ("looped_fp64", (8, 12000), 1, torch.float64),
    ("split_fp64", (8, 65536), 1, torch.float64),
)
INNER_TREE_CALL = "reduction_ordering=tl.constexpr(tl.ReductionOrdering.INNER_TREE)"

SUM_VARIANTS = (
    ("autotune", (64, 256), 1, torch.float32, False, {"max_autotune": True}),
)

DYNAMIC_CASES = (("plan_change", (512, 65537), {}),)

OUT_OF_SCOPE_CASES = (
    ("multidim", lambda z: torch.sum(z, (0, 1))),
    ("dtype", lambda z: torch.sum(z, 1, dtype=torch.float64)),
)

LAYOUT_CASES = (
    (
        "outer_strided",
        lambda d: (torch.randn(128, 300, device=d)[::2], -1),
        True,
    ),
    ("singleton_dim1", lambda d: (_singleton_input(d), 1), True),
    (
        "noncollapsible",
        lambda d: (torch.randn(16, 8, 300, device=d)[::2], -1),
        False,
    ),
)

SIGNED_ZERO_CASES = (
    ("multirow_fp32", 4, 1, True, torch.float32),
    ("persistent_fp32", 64, 128, False, torch.float32),
    ("persistent_fp64", 64, 128, False, torch.float64),
)

FUSION_CASES = (
    "nested",
    "mix",
    "multi_kernel",
    "multi_output",
)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON
    and torch.version.hip is None
    and has_triton_reduction_ordering(),
    "requires CUDA, tl.ReductionOrdering, and the eager inner-tree implementation",
)
class StrictNumericsTest(TestCase):
    # [CLASS DOC] CUDA, non-ROCm test class that needs a Triton with reduction ordering support.
    # It sets the environment flag enabling the eager inner-tree sum, so eager and compiled sums
    # can be compared bitwise. Helpers compile in strict numerics mode with caches disabled and
    # return the generated code.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setup, patches the process environment to turn on the
        # eager inner-tree sum (restored by a cleanup) and seeds the random generator with 0.
        super().setUp()
        env_patch = mock.patch.dict(os.environ, {"PYTORCH_SUM_INNER_TREE": "1"})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        torch.manual_seed(0)

    def _run(self, fn, *args, **cfg):
        # [FUNCTION DOC] HELPER: Compiles the given function with full graph under a config that
        # sets numerics to strict, disables caches and applies extra config overrides, after
        # resetting dynamo. Returns the result and all generated code joined into one string.
        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        return result, "\n".join(codes)

    def _assert_bitwise_equal(self, eager, result):
        # [FUNCTION DOC] HELPER: Asserts bitwise equality of eager and compiled results by
        # comparing their contiguous bytes; does nothing unless the CuTe DSL is available.
        if not TEST_CUTEDSL:
            return
        self.assertEqual(
            eager.contiguous().reshape(-1).view(torch.uint8),
            result.contiguous().reshape(-1).view(torch.uint8),
        )

    def _check_sum(self, device, shape, dim, dtype, keepdim=False, **cfg):
        # [FUNCTION DOC] HELPER: Creates a random tensor, compares the eager sum over the dim
        # against the strict compiled sum (bitwise), and asserts the generated code contains the
        # inner-tree reduction ordering marker. Returns the generated code.
        x = torch.randn(*shape, device=device, dtype=dtype)

        def fn(z):
            # [FUNCTION DOC] HELPER: Sums its argument along the enclosing dim with the enclosing
            # keepdim flag.
            return torch.sum(z, dim, keepdim=keepdim)

        eager = fn(x)
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(eager, result)
        self.assertIn(INNER_TREE_CALL, code)
        return code

    @parametrize("case", SUM_CASES, name_fn=lambda c: c[0])
    def test_sum_bitwise(self, device, case):
        # [FUNCTION DOC] Parametrized over the sum case table: checks bitwise equal sums, then
        # checks code shape by case name: persistent cases use the persistent reduction heuristic,
        # looped cases contain an offset loop and exactly one inner-tree marker, and split cases
        # contain two markers.
        name, shape, dim, dtype = case
        code = self._check_sum(device, shape, dim, dtype)
        if name.startswith("persistent"):
            self.assertIn("@triton_heuristics.persistent_reduction(", code)
        elif name.startswith("looped"):
            self.assertIn("for r0_offset in", code)
            self.assertEqual(code.count(INNER_TREE_CALL), 1)
        else:
            self.assertEqual(code.count(INNER_TREE_CALL), 2)

    @parametrize("case", SUM_VARIANTS, name_fn=lambda c: c[0])
    def test_sum_variants(self, device, case):
        # [FUNCTION DOC] Parametrized over sum variants (currently the max-autotune config): runs
        # the sum check with the variant's shape, dim, dtype, keepdim and config overrides.
        _, shape, dim, dtype, keepdim, cfg = case
        self._check_sum(device, shape, dim, dtype, keepdim, **cfg)

    @parametrize(
        "input_dtype",
        (torch.float32, torch.float64),
        name_fn=lambda dtype: str(dtype).removeprefix("torch."),
    )
    def test_special_values_match_eager(self, device, input_dtype):
        # [FUNCTION DOC] Builds a matrix of rows with a NaN, positive infinity, negative infinity,
        # an infinity pair, negative zeros and large cancelling values, sums rows in strict mode,
        # checks bitwise equality with eager and that the inner-tree marker is present.
        # Parametrized on float32 and float64.
        x = torch.zeros(6, 300, device=device, dtype=input_dtype)
        x[0, 0] = torch.nan
        x[1, 0] = torch.inf
        x[2, 0] = -torch.inf
        x[3, :2] = torch.tensor(
            [torch.inf, -torch.inf], device=device, dtype=input_dtype
        )
        x[4] = -0.0
        x[5] = torch.tensor(
            [1e20, 1.0, -1e20, 1.0], device=device, dtype=input_dtype
        ).repeat(75)

        def fn(z):
            # [FUNCTION DOC] HELPER: Sums its argument along dim 1.
            return z.sum(1)

        result, code = self._run(fn, x)
        self._assert_bitwise_equal(fn(x), result)
        self.assertIn(INNER_TREE_CALL, code)

    @skipIfNoCuteDSL
    @parametrize("case", DYNAMIC_CASES, name_fn=lambda c: c[0])
    def test_dynamic_sum(self, device, case):
        # [FUNCTION DOC] Needs the CuTe DSL. Compiles a sum over dim 1 with dynamic shapes under
        # strict numerics and compares bitwise to eager for each size in the case, which changes
        # the reduction plan.
        _, sizes, cfg = case

        def fn(z):
            # [FUNCTION DOC] HELPER: Sums its argument along dim 1.
            return torch.sum(z, 1)

        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for n in sizes:
                x = torch.randn(8, n, device=device)
                self._assert_bitwise_equal(fn(x), compiled(x))

    @parametrize("case", OUT_OF_SCOPE_CASES, name_fn=lambda c: c[0])
    def test_out_of_scope_uses_default_order(self, device, case):
        # [FUNCTION DOC] Parametrized over cases not covered by strict ordering (multi-dim sum,
        # sum with an output dtype change): the generated code must not contain the inner-tree
        # marker.
        _, fn = case
        _, code = self._run(fn, torch.randn(64, 300, device=device))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", LAYOUT_CASES, name_fn=lambda c: c[0])
    def test_layout_eligibility(self, device, case):
        # [FUNCTION DOC] Parametrized over layouts: builds an input with the case's builder,
        # compiles a sum, and checks the inner-tree marker presence equals the case's eligibility
        # flag; if eligible also checks bitwise equality with eager.
        _, make_input, eligible = case
        x, dim = make_input(device)

        def fn(z):
            # [FUNCTION DOC] HELPER: Sums its argument along the enclosing dim.
            return torch.sum(z, dim)

        result, code = self._run(fn, x)
        self.assertEqual(INNER_TREE_CALL in code, eligible)
        if eligible:
            self._assert_bitwise_equal(fn(x), result)

    def test_unbacked_reduction_size_uses_default_order(self, device):
        # [FUNCTION DOC] Checks that a sum whose reduction size is data dependent (an unbacked
        # size after boolean masking) is not given the inner-tree ordering, while the result
        # equals eager.
        def fn(z):
            # [FUNCTION DOC] HELPER: Selects positive elements of the input and sums them along
            # dim 0.
            return z[z > 0].sum(0)

        x = torch.randn(1024, device=device)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            result, code = self._run(fn, x)
        self.assertEqual(result, fn(x))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", SIGNED_ZERO_CASES, name_fn=lambda c: c[0])
    def test_signed_zero(self, device, case):
        # [FUNCTION DOC] Parametrized over signed zero cases: sums a tensor of negative zeros
        # along dim 1 and checks the result matches eager bitwise. It also checks the generated
        # reduction block size constant against the value computed from the vector size and inner-
        # tree parameters (or from the load count for the multi-row case, which disables
        # persistent reductions).
        _, rows, n, multirow, dtype = case
        x = torch.full((rows, n), -0.0, device=device, dtype=dtype)

        def fn(z):
            # [FUNCTION DOC] HELPER: Sums its argument along dim 1.
            return torch.sum(z, 1)

        cfg = {"triton.persistent_reductions": False} if multirow else {}
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(fn(x), result)
        vector_size = vec_size(x.element_size())
        if multirow:
            num_loads = (n + vector_size - 1) // vector_size
            rblock = (1 << (num_loads - 1).bit_length()) * vector_size
        else:
            rblock = compute_inner_tree_params(n, 1, vector_size).batch_total_elements
        self.assertIn(f"R0_BLOCK: tl.constexpr = {rblock}", code)

    def _make_fusion_case(self, kind, device):
        # [FUNCTION DOC] HELPER: Builds a fusion scenario for the given kind (nested, mix,
        # multi_kernel, otherwise multi_output). Returns the function, its input tuple, the index
        # of the output to compare (or none), the config overrides, a mapping of metric names to
        # expected values and an expected generated kernel count (or none).
        cfg = {}
        expected_metrics = {}
        kernel_count = None
        result_index = None

        if kind == "nested":
            batch, width, group = 8, 4096, 16
            args = (torch.randn(batch, width, device=device),)
            cfg = {"triton.nested_reduction": True}
            expected_metrics = {"codegen_nested_reduction": 0}

            def fn(x):
                # [FUNCTION DOC] HELPER: Nested reduction case: takes the max along the last dim,
                # adds it to the input and realizes the result, then reshapes into groups and sums
                # the last dim.
                outer = x.amax(-1, keepdim=True)
                y = torch.ops._inductor_test.realize(x + outer)
                return y.reshape(batch, width // group, group).sum(-1)

        elif kind == "mix":
            args = (torch.randn(8, 12000, device=device),)
            cfg = {
                "triton.mix_order_reduction": True,
                "triton.mix_order_reduction_non_strict_mode": True,
            }
            result_index = 0
            expected_metrics = {"codegen_mix_order_reduction": 0}

            def fn(x):
                # [FUNCTION DOC] HELPER: Mixed order reduction case: returns the sum along the
                # last dim and the product along dim 0.
                return x.sum(-1), x.prod(0)

        elif kind == "multi_kernel":
            args = (torch.randn(8, 12000, device=device),)
            cfg = {"triton.multi_kernel": True}

            def fn(x):
                # [FUNCTION DOC] HELPER: Multi-kernel case: sums the input along dim 1.
                return x.sum(1)

        else:
            args = (torch.randn(8, 300, device=device),)
            cfg = {"online_softmax": True}
            result_index = 1
            kernel_count = 2

            def fn(x):
                # [FUNCTION DOC] HELPER: Online softmax case: returns the softmax along the last
                # dim and the sum along the last dim.
                return torch.softmax(x, -1), x.sum(-1)

        return fn, args, result_index, cfg, expected_metrics, kernel_count

    @parametrize("kind", FUSION_CASES)
    def test_fusion_preserves_strict_sum(self, device, kind):
        # [FUNCTION DOC] Parametrized over fusion kinds: compiles the scenario in strict mode and
        # checks the selected output bitwise against eager, that the inner-tree marker appears
        # once, that the given metrics have the expected values and the generated kernel count if
        # specified. Multi-kernel must not use multi-kernel compilation and must be looped;
        # multi_output also compares the first output to eager.
        fn, args, index, cfg, expected_metrics, kernel_count = self._make_fusion_case(
            kind, device
        )
        eager = fn(*args)
        metrics.reset()
        result, code = self._run(fn, *args, **cfg)
        expected = eager if index is None else eager[index]
        actual = result if index is None else result[index]
        self._assert_bitwise_equal(expected, actual)
        self.assertEqual(code.count(INNER_TREE_CALL), 1)
        for metric, expected_value in expected_metrics.items():
            self.assertEqual(getattr(metrics, metric), expected_value)
        if kernel_count is not None:
            self.assertEqual(metrics.generated_kernel_count, kernel_count)
        if kind == "multi_kernel":
            self.assertNotIn("async_compile.multi_kernel(", code)
            self.assertIn("for r0_offset in", code)
        elif kind == "multi_output":
            self.assertEqual(eager[0], result[0])

    def test_combo_kernel_preserves_strict_sum_blocks(self, device):
        # [FUNCTION DOC] Compiles two sums of separate inputs with combo kernels enabled (no
        # autotune, no peak memory threshold), checks both bitwise against eager, that the marker
        # is present and that no combo grid metadata appears in the code.
        args = (
            torch.randn(8, 12000, device=device),
            torch.randn(8, 12000, device=device),
        )

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sums of both arguments along dim 1.
            return a.sum(1), b.sum(1)

        eager = fn(*args)
        result, code = self._run(
            fn,
            *args,
            combo_kernels=True,
            combo_kernels_autotune=0,
            combo_kernel_peak_memory_pct_threshold=None,
        )
        for expected, actual in zip(eager, result, strict=True):
            self._assert_bitwise_equal(expected, actual)
        self.assertIn(INNER_TREE_CALL, code)
        self.assertNotIn("combo_grid_meta", code)

    @unittest.skipIf(not SM90OrLater, "requires TMA support")
    @parametrize("kind", ("multirow", "split"))
    def test_tma_preserves_strict_sum(self, device, kind):
        # [FUNCTION DOC] Requires SM90 or later. Parametrized on kind: a multi-row small tensor or
        # a one-row tensor with large cancelling values at the batch positions of the inner-tree
        # plan. Compiles with tensor descriptors and aligned inputs, checks bitwise equality with
        # eager, and for the split kind checks two markers and a tensor descriptor in the code
        # (otherwise a store call).
        if kind == "multirow":
            x = torch.randn(64, 5, device=device)
        else:
            x = torch.zeros(1, 65536, device=device)
            params = compute_inner_tree_params(
                x.shape[1], x.shape[0], vec_size(x.element_size())
            )
            for batch, value in enumerate((1e20, 1, -1e20, 1)):
                x[0, batch * params.batch_total_elements] = value

        def fn(z):
            # [FUNCTION DOC] HELPER: Sums its argument along dim 1.
            return torch.sum(z, 1)

        eager = fn(x)
        if kind == "split":
            self._assert_bitwise_equal(eager, torch.ones_like(eager))
        result, code = self._run(
            fn,
            x,
            assume_aligned_inputs=True,
            **{"triton.use_tensor_descriptor": True},
        )
        self._assert_bitwise_equal(eager, result)
        if kind == "split":
            self.assertEqual(code.count(INNER_TREE_CALL), 2)
        self.assertIn("tensor_descriptor" if kind == "split" else "tl.store", code)


instantiate_device_type_tests(StrictNumericsTest, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
