# Owner(s): ["module: inductor"]

# [MODULE DOC] Inductor tests for the Triton "mix order reduction" optimization, which fuses an
# inner (row) reduction and an outer (column) reduction over the same input into one kernel.
# Contains a shared base class, tests of skip cases, numerics/metrics tests (including RMSNorm and
# LayerNorm backward, TMA, dynamic shapes, mocked scheduler fusion checks), an over-fusion
# regression test, CPU-runnable autotune heuristic tests, and a variant class that reruns all
# tests with the config flag flipped. Most tests need a GPU. Import time has no side effects
# besides the config-patching decorators; the main block runs tests only when a GPU exists.
import unittest
from unittest import mock
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch import nn
from torch._dynamo.utils import same
from torch._inductor import metrics, utils
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.runtime.triton_heuristics import persistent_reduction
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.test_case import run_tests, TestCase
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    recover_orig_fp32_precision,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._triton import has_triton_tma_device


class TestBase(TestCase):
    # [CLASS DOC] HELPER: Base test case for this file. Resets inductor metrics and dynamo
    # compilation metrics before every test, and offers a numeric comparison helper.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Per-test setup. Calls the parent setup, then clears the inductor
        # metrics counters and the dynamo compilation metrics so tests can assert on fresh counts.
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

    def check_numeric(self, f, args, tol=1e-3):
        # [FUNCTION DOC] HELPER: Runs function f eagerly and compiled on the given arguments
        # and asserts the results match under the dynamo comparison utility with tolerance
        # tol (default 1e-3).
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertTrue(same(ref, act, tol=tol))


class SkipPatternTest(TestBase):
    # [CLASS DOC] Documents cases where mix-order reduction must not be applied; each test checks
    # the number of generated kernels or the mix-order metric stays at the unfused value.
    """
    Illustate the cases that we skip mix-order reduction. We skip in cases
    like when the outer reduction is followed by a pointwise that load
    the un-reduced tensor.
    """

    @inductor_config.patch(split_reductions=False)
    def test_dimension_too_close(self):
        # [FUNCTION DOC] Compiles a function doing a row sum and a column sum of a square 768 by
        # 768 GPU tensor with split reductions disabled, and asserts two kernels were generated,
        # i.e. the reductions were not fused because the two reduction sizes are too close.
        """
        Skip if the two reduction size are too close.
        We require one reduction dimension to be much larger so we can split
        that dimension and make it efficient.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum over dim 1 and the sum over dim 0
            # of the input.
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0)
            return out1, out2

        x = torch.randn(768, 768, device=GPU_TYPE)
        torch.compile(f)(x)
        self.assertEqual(2, metrics.generated_kernel_count)


# Cooperative reductions disable split reductions, which are necessary for mix order
# reductions.
@inductor_config.patch(
    {
        "triton.cooperative_reductions": False,
        "triton.force_cooperative_reductions": False,
    }
)
@instantiate_parametrized_tests
class MixOrderReductionTest(TestBase):
    # [CLASS DOC] Main test suite for mix-order reduction. Class decorators disable cooperative
    # reductions (they turn off split reductions needed here) and expand parametrized tests. The
    # subclass NoMixOrderReductionTest reruns every test with the feature flag inverted, so most
    # tests assert the metric equals the flag value, or skip when the feature is off.
    @parametrize(
        "name",
        [
            "sum",
            "prod",
            "mean",
        ],
    )
    @parametrize("dtype", (torch.bfloat16, torch.float))
    @parametrize("swap", (False, True))
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768), (32, 1024, 768)))
    def test_mix_order_reduction(self, name, dtype, swap, split_reductions, shape):
        # [FUNCTION DOC] Parametrized over reduction name (sum, prod, mean), dtype, operand order,
        # split_reductions and shape. Compiles a function returning an inner-dim reduction and an
        # outer-dim reduction (over dims 0, or 0 and 1 for 3D input), compares with eager (looser
        # tolerance for bfloat16), and asserts the mix-order metric equals the config flag. Skips
        # prod with 3D shapes because prod accepts only a single dim.
        # torch.prod does not accept tuple for dim argument
        if name == "prod" and len(shape) == 3:
            self.skipTest("Invalid combination")

        def f(x):
            # [FUNCTION DOC] HELPER: Closure computing the row-wise reduction and the outer
            # reduction of x using reduction_fn from the enclosing scope, in an order
            # controlled by swap.
            def outer_red():
                # [FUNCTION DOC] HELPER: Reduces x over all leading dims (0 and 1 for 3D, 0 for
                # 2D); raises AssertionError if x is neither 2D nor 3D.
                if len(shape) == 3:
                    return reduction_fn(x, dim=(0, 1))
                else:
                    assert len(shape) == 2  # noqa: S101
                    return reduction_fn(x, dim=0)

            if swap:
                return outer_red(), reduction_fn(x, dim=-1)
            else:
                return reduction_fn(x, dim=-1), outer_red()

        reduction_fn = getattr(torch, name)
        x = torch.randn(shape, dtype=dtype, device=GPU_TYPE)

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = f(x)
        act = opt_f(x)
        tol = 1e-3 if dtype == torch.float else 1e-2
        self.assertTrue(
            same(ref, act, tol=tol), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_xmask(self):
        # [FUNCTION DOC] Checks the x mask is correct in mix-order kernels: the input is a view of
        # the first rows of a larger buffer whose extra row is filled with a huge value, and
        # xblock initial size is 2, so reading beyond the rows would corrupt results. Compares
        # compiled vs eager and asserts the mix-order metric equals the config flag. Skipped if
        # the feature is disabled.
        """
        Make sure xmask is setup properly
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum over dim 0 and the sum over dim 1 of x.
            return x.sum(dim=0), x.sum(dim=1)

        M, N = 32768 + 1023, 768
        EXTRA_ROW = 1
        buf = torch.randn(M + EXTRA_ROW, N, device=GPU_TYPE)
        x = buf[:M, :]
        # make sure wrong xmask error loud if read excess elements
        buf[M:, :] = 1000000

        opt_f = torch.compile(
            f,
            options={
                "triton.mix_order_reduction_initial_xblock": 2,
            },
        )

        ref = f(x)
        act = opt_f(x)

        self.assertTrue(
            same(ref, act, tol=1e-3), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_avoid_non_coalesced_access(self):
        # [FUNCTION DOC] Verifies numerics of a case where one operand is a transposed view,
        # then asserts that no mix-order reduction was generated, since forcing the x block
        # to 1 would make accesses to the transposed tensor non-coalesced. Skipped if the
        # feature is disabled.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the last-dim sum of the sum of x and y, and the sum
            # of x over dims 0 and 1.
            return (x + y).sum(dim=-1), x.sum(dim=(0, 1))

        x = torch.randn(128, 256, 768, device=GPU_TYPE)
        y = torch.randn(128, 768, 256, device=GPU_TYPE).transpose(1, 2)
        self.check_numeric(f, (x, y))

        # we skip mix order reduction for such kernel since
        # we force XBLOCK to be 1, the access to tensor y would be
        # very inefficient.
        # TODO: support XBLOCK larger than 1. But in that case, we
        # would have bigger restriction on rnumel to avoid exploding
        # shared memory.
        self.assertEqual(metrics.codegen_mix_order_reduction, 0)

    @inductor_config.patch(split_reductions=False)
    def test_fuse_non_contiguous_pointwise(self):
        # [FUNCTION DOC] Regression test: a pointwise op using both a row reduction result and
        # unreduced data of x is fused next to the reduction, which could break the contiguity
        # invariant. Only checks numerics of the compiled function with split reductions off.
        # Skipped if disabled.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        # Regression: mix-order reduction can appear valid pre-fusion, but a pointwise
        # fused into one side can change access patterns and break the contiguity
        # invariant. This test builds a reduction + pointwise path plus a second
        # reduction, matching the shape/ordering pattern seen in the E2E failure.

        def f(x):
            # [FUNCTION DOC] HELPER: Returns a pointwise product of the row sum with the first
            # column of x, and the column sum of x.
            # First reduction (contiguous on its own).
            r1 = x.sum(dim=1)
            # Pointwise depends on both reduced and unreduced data, so fusing it
            # with the reduction can change access strides.
            y = r1 * x[:, 0]
            # Second reduction across a different dimension to trigger mix-order logic.
            r2 = x.sum(dim=0)
            return y, r2

        # Large, asymmetric shape encourages mix-order reduction heuristics.
        x = torch.randn(32768, 768, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    @inductor_config.patch(coordinate_descent_tuning=True)
    def test_XBLOCK_coordest_tuning(self):
        # [FUNCTION DOC] With coordinate descent tuning enabled, checks numerics and that exactly
        # one mix-order reduction kernel is generated, i.e. x block tuning is skipped for such
        # kernels and compilation still works. Skipped if the feature is disabled.
        """
        We should skip XBLOCK coordinate descent tuning for
        mix order reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the last-dim sum and the dim 0 sum of x.
            return x.sum(dim=-1), x.sum(dim=0)

        x = torch.randn(32768, 256, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(metrics.codegen_mix_order_reduction, 1)

    @inductor_config.patch(unroll_reductions_threshold=1)
    def test_3layer_split_reduction(self):
        # [FUNCTION DOC] Uses a very tall, 2-column input and unroll threshold 1 to force a
        # three-layer split reduction. Checks numerics and that the mix-order count is 1 only
        # when cooperative reductions are enabled, otherwise 0 (mix order is not applied for more
        # than two layers).
        """
        Use a larger M and smaller N to trigger a 3 layer split reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the last-dim sum and the dim 0 sum of x.
            return x.sum(dim=-1), x.sum(dim=0)

        M = 32768 * 1024 if torch.version.hip is not None else 32768 * 256
        x = torch.randn(M, 2, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        # We don't do mix order reduction for split redutions
        # with more than 2 layers
        self.assertEqual(
            metrics.codegen_mix_order_reduction,
            1
            if inductor_config.triton.cooperative_reductions
            or inductor_config.triton.force_cooperative_reductions
            else 0,
        )

    def test_independent_split_size(self):
        # [FUNCTION DOC] Verifies the mix-order split size config is honored: for split sizes 8
        # and 16 it resets dynamo, checks numerics and metric, compiles and fetches the
        # generated code, and asserts the code contains the matching RSPLIT_SIZE kernel meta
        # entry. Skipped if disabled.
        """
        Make sure mix order reduction can pick the split size it wants
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the last-dim sum and the dim 0 sum of x.
            return x.sum(dim=-1), x.sum(dim=0)

        def check_one_split_size(split_size):
            # [FUNCTION DOC] HELPER: For one split size, resets dynamo, patches the split size
            # config, checks numerics and the mix-order metric, and asserts the generated code
            # mentions that RSPLIT_SIZE. Uses the input x from the enclosing test.
            torch._dynamo.reset()

            with inductor_config.patch(
                "triton.mix_order_reduction_split_size", split_size
            ):
                self.check_numeric(f, (x,))
                self.assertEqual(
                    inductor_config.triton.mix_order_reduction,
                    metrics.codegen_mix_order_reduction,
                )

                _, (code,) = utils.run_and_get_code(torch.compile(f), x)
                self.assertTrue(f"'RSPLIT_SIZE': {split_size}" in code)

        x = torch.randn(32768, 768, dtype=torch.float, device=GPU_TYPE)

        check_one_split_size(8)
        check_one_split_size(16)

    @inductor_config.patch(split_reductions=False)
    def test_non_contiguous_input(self):
        # [FUNCTION DOC] Uses a permuted (non-contiguous) 3D input with split reductions off,
        # checks compiled numerics, and asserts the mix-order metric equals the config flag.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the last-dim sum and the sum over dims 0 and 1 of x.
            return x.sum(dim=-1), x.sum(dim=[0, 1])

        x = torch.randn(1024, 32, 768, dtype=torch.float, device=GPU_TYPE).permute(
            1, 0, 2
        )
        self.check_numeric(f, (x,))
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @inductor_config.patch(split_reductions=False)
    def test_multi_workspace_allocation(self):
        # [FUNCTION DOC] Compiles a function with two inputs of different shapes, each producing
        # a row and a column sum, with split reductions off. Checks numerics and that the metric
        # is 2 when the feature is on (one fused kernel per input, each needing its own
        # workspace), else 0.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the dim 0 sum and dim 1 sum of x followed by
            # those of y.
            return x.sum(dim=0), x.sum(dim=1), y.sum(dim=0), y.sum(dim=1)

        x = torch.randn(4096 * 64, 32, device=GPU_TYPE)
        y = torch.randn(4098 * 64, 34, device=GPU_TYPE)

        self.check_numeric(f, (x, y))
        expected_mix_order_reduction = (
            0 if not inductor_config.triton.mix_order_reduction else 2
        )
        self.assertEqual(
            expected_mix_order_reduction, metrics.codegen_mix_order_reduction
        )

    @parametrize(
        "wdtype",
        [
            torch.bfloat16,  # extra down cast for dw is needed
            torch.float,
        ],
    )
    @parametrize("split_reductions", (False, True))
    @parametrize(
        "shape", ((1000000, 256), (32768, 2048), (32768, 768), (32768 + 1023, 768))
    )
    @parametrize("max_autotune", (False, True))
    @parametrize("initial_xblock", (1, 2))
    @parametrize("add_1dim", (False, True))
    # The test OOM in CI sometimes. Ask for more memory to make it stable.
    @largeTensorTest("16GB", device=GPU_TYPE, inductor=True)
    def test_rms_norm_bwd(
        self,
        wdtype,
        split_reductions,
        shape,
        max_autotune,
        initial_xblock,
        add_1dim,
    ):
        # [FUNCTION DOC] Parametrized RMSNorm forward/backward test (weight dtype, split
        # reductions, shape, max autotune, initial xblock, extra singleton dim). Skips most max-
        # autotune and extra-dim combinations to save resources. Compares input and weight grads
        # of the compiled version against eager (tolerance 1e-2) and asserts the mix-order metric
        # equals the flag. Needs about 16GB of device memory.
        # max_autotune can be slow and cost resource, trim down the tests
        # for max autotune
        if max_autotune and not (
            wdtype == torch.bfloat16
            and not split_reductions
            and shape in ((32768, 768), (32769, 768))
            and initial_xblock == 1
            and inductor_config.triton.mix_order_reduction
        ):
            self.skipTest("Skip non-critical tests to save resources.")

        if shape != (1000000, 256) and add_1dim:
            self.skipTest("Skip non-critical tests to save resources.")

        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: RMSNorm computed manually: casts x to float, scales by
            # reciprocal root of the mean square plus eps, multiplies by w, and casts back to
            # the original dtype.
            orig_dtype = x.dtype

            x = x.float()
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)
            return y

        def fwd_bwd(f):
            # [FUNCTION DOC] HELPER: Clears grads of x and w, runs f on x, w, eps, backpropagates
            # dy and returns the grads of x and w. Uses x, w, eps and dy from the enclosing test.
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        torch.manual_seed(1337)

        # M, N = 1152 * 500, 384
        M, N = shape
        x = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        if add_1dim:
            x = x[:, None, :]

        w = torch.randn(N, dtype=wdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
                "triton.mix_order_reduction_initial_xblock": initial_xblock,
                **(
                    {
                        "max_autotune": True,
                        "coordinate_descent_tuning": True,
                    }
                    if max_autotune
                    else {}
                ),
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(
            same(ref, act, tol=1e-2), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @unittest.skipUnless(
        has_triton_tma_device(), "Test requires Triton TMA device support"
    )
    @parametrize("allow_multi_stages", (False, True))
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
        }
    )
    # The 1M-row input plus its grads needs a fair amount of device memory.
    @largeTensorTest("16GB", device=GPU_TYPE, inductor=True)
    def test_rms_norm_bwd_tma(self, allow_multi_stages):
        # [FUNCTION DOC] Regression test for stale xoffset with device-side TMA in the mix-
        # order loop, using a 1M row input and bfloat16 tensors. Compares grads with eager,
        # asserts the metric, and when the feature is on checks the backward code has the
        # uses_tma and uses_device_tma meta entries and advances xoffset by XBLOCK.
        # Parametrized by whether multi stages are allowed. Requires TMA support, 16GB of
        # memory, tensor descriptors and aligned inputs.
        """
        Regression test for https://github.com/pytorch/pytorch/issues/186241.

        With TMA enabled, the mix-order reduction loop used to leave the scalar
        `xoffset` (which TMA descriptors index off) frozen while only advancing
        the `xindex` tensor, so every loop iteration read/wrote the first tile
        and produced wrong results. Triton cannot pipeline device-side tensor
        map creation, so these kernels use one stage.

        ``allow_multi_stages=True`` verifies that device-side TMA still forces
        NUM_STAGES=1; ``False`` disables multi-staging globally as a sanity check.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: RMSNorm computed manually in float and cast back to
            # the input dtype.
            orig_dtype = x.dtype
            x = x.float()
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            return (x * rsqrt[:, None] * w).to(dtype=orig_dtype)

        def fwd_bwd(compiled_f):
            # [FUNCTION DOC] HELPER: Clears grads, runs compiled_f on x, w, eps from the enclosing
            # test, backpropagates dy and returns the grads of x and w.
            x.grad = None
            w.grad = None
            out = compiled_f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        torch.manual_seed(1337)
        # Large M so the RSPLIT loop runs many iterations per program: this is
        # what surfaces the stale-xoffset bug (a single iteration would hide it).
        M, N = 1000000, 256
        x = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": False,
                "triton.mix_order_reduction_allow_multi_stages": allow_multi_stages,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )
        if inductor_config.triton.mix_order_reduction:
            # The fix advances the scalar xoffset inside the mix-order loop so
            # TMA descriptors point at the right tile each iteration.
            (
                FileCheck()
                .check("'uses_tma': True")
                .check("'uses_device_tma': True")
                .check("xoffset += XBLOCK")
                .run(bwd_wrapper)
            )

    @unittest.skipUnless(
        has_triton_tma_device(), "Test requires Triton TMA device support"
    )
    @parametrize("allow_multi_stages", (False, True))
    @parametrize("shape", ((32768, 256), (32768 + 1023, 768)))
    @inductor_config.patch(
        {
            "triton.use_tensor_descriptor": True,
            "assume_aligned_inputs": True,
            # Pin the split size so each program runs several loop iterations
            # (iterations = RSPLIT_SIZE // XBLOCK) regardless of the input size
            # and SM count. This is what lets a small input surface the
            # stale-xoffset bug that test_rms_norm_bwd_tma reproduces at 1M rows.
            "triton.mix_order_reduction_split_size": 64,
        }
    )
    def test_rms_norm_bwd_tma_small(self, allow_multi_stages, shape):
        # [FUNCTION DOC] Smaller TMA variant of the stale xoffset regression test. The split size
        # is pinned to 64 so each program loops several times. Runs for two shapes (one with row
        # count not divisible by the block) and both multi-stage settings; compares grads, asserts
        # the metric, and checks generated code for TMA markers and the xoffset advance when the
        # feature is on.
        """
        Small, CI-friendly variant of test_rms_norm_bwd_tma (regression test for
        https://github.com/pytorch/pytorch/issues/186241).

        With the split size pinned, every program advances its TMA descriptor
        offset across multiple loop iterations, so a stale xoffset still
        produces wrong results at these sizes. The non-divisible-M shape also
        exercises the masked tail with TMA descriptors.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: RMSNorm computed manually in float and cast back to
            # the input dtype.
            orig_dtype = x.dtype
            x = x.float()
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            return (x * rsqrt[:, None] * w).to(dtype=orig_dtype)

        def fwd_bwd(compiled_f):
            # [FUNCTION DOC] HELPER: Clears grads, runs compiled_f on x, w, eps from the enclosing
            # test, backpropagates dy and returns the grads of x and w.
            x.grad = None
            w.grad = None
            out = compiled_f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        torch.manual_seed(1337)
        M, N = shape
        x = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": False,
                "triton.mix_order_reduction_allow_multi_stages": allow_multi_stages,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )
        if inductor_config.triton.mix_order_reduction:
            (
                FileCheck()
                .check("'uses_tma': True")
                .check("'uses_device_tma': True")
                .check("xoffset += XBLOCK")
                .run(bwd_wrapper)
            )

    @parametrize(
        "wbdtype",
        [
            torch.bfloat16,  # extra down cast for dw/db is needed
            torch.float,
        ],
    )
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768)))
    def test_layer_norm_bwd_with_bias(self, wbdtype, split_reductions, shape):
        # [FUNCTION DOC] Parametrized layer norm backward test with weight and bias of dtype
        # wbdtype over float input. Compares x, weight and bias grads of compiled vs eager
        # (tolerance 1e-2) and asserts the mix-order metric equals the config flag.
        def f(x, w, b, eps):
            # [FUNCTION DOC] HELPER: Layer norm of x over the last dim with weight and bias
            # cast to float.
            return F.layer_norm(x, x.shape[-1:], w.float(), b.float(), eps)

        def fwd_bwd(f):
            # [FUNCTION DOC] HELPER: Clears grads of x, w, b, runs f, backpropagates dy and
            # returns the three grads. Uses names from the enclosing test.
            x.grad = None
            w.grad = None
            b.grad = None
            out = f(x, w, b, eps)
            out.backward(dy)
            return x.grad, w.grad, b.grad

        # M, N = 1152 * 500, 384
        M, N = shape
        xdtype = torch.float
        x = torch.randn(M, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(
            same(ref, act, tol=1e-2), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("dynamic_dims", ([0], [1], [0, 1]))
    def test_rms_norm_bwd_with_dynamic_shape(self, dynamic_dims):
        # [FUNCTION DOC] RMSNorm backward with a 3D input marked dynamic on dims 0 and 1; note
        # that the parametrized dynamic_dims argument is not used for marking here. Compares grads
        # with eager and asserts the metric equals the flag. Skipped if the feature is disabled.
        # KNOWN ISSUE: the dynamic_dims parameter is ignored; the marking always uses dims
        # 0 and 1.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: Functional RMS norm of x over the last dim with weight
            # w and eps.
            return F.rms_norm(x, x.shape[-1:], weight=w, eps=eps)

        def fwd_bwd(f):
            # [FUNCTION DOC] HELPER: Clears grads, runs f, backpropagates dy and returns grads of
            # x and w.
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        M0, M1, N = 251, 223, 128
        wbdtype = torch.float
        xdtype = torch.float
        x = torch.randn(M0, M1, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        torch._dynamo.mark_dynamic(x, (0, 1))
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": False,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(
            same(ref, act, tol=1e-2), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("dynamic_dims", ([0], [1], [0, 1]))
    def test_layer_norm_bwd_with_dynamic_shape(self, dynamic_dims):
        # [FUNCTION DOC] Layer norm backward (no bias) on a 3D input marked dynamic on the
        # parametrized dynamic_dims. Compares grads with eager and asserts the metric equals the
        # flag. Skipped if the feature is disabled.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: Functional layer norm of x over the last dim with weight
            # w, no bias.
            return F.layer_norm(x, x.shape[-1:], weight=w, bias=None, eps=eps)

        def fwd_bwd(f):
            # [FUNCTION DOC] HELPER: Clears grads, runs f, backpropagates dy and returns grads of
            # x and w.
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        M0, M1, N = 251, 223, 128
        wbdtype = torch.float
        xdtype = torch.float
        x = torch.randn(M0, M1, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        torch._dynamo.mark_dynamic(x, dynamic_dims)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(f)

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(
            same(ref, act, tol=1e-2), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768)))
    def test_layer_norm_bwd_no_bias(self, split_reductions, shape):
        # [FUNCTION DOC] Layer norm backward without bias for two shapes and split
        # reduction settings. Compares grads with eager (tolerance 1e-2) and asserts the
        # metric equals the flag.
        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: Functional layer norm of x with weight w and no bias.
            return F.layer_norm(x, x.shape[-1:], w, bias=None, eps=eps)

        def fwd_bwd(f):
            # [FUNCTION DOC] HELPER: Clears grads, runs f, backpropagates dy and returns grads of
            # x and w.
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        # M, N = 1152 * 500, 384
        M, N = shape
        xdtype = torch.float
        wbdtype = torch.float
        x = torch.randn(M, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(
            same(ref, act, tol=1e-2), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("split_reductions", (False, True))
    @parametrize("dtype", [torch.bfloat16, torch.float])
    def test_rms_norm_sharing_weights(self, split_reductions, dtype):
        # [FUNCTION DOC] Applies RMS norm with one shared weight to three inputs, computes
        # reference grads in float64 cast back to the dtype, and compares with grads from the
        # compiled function (loose tolerance for bfloat16). Asserts three mix-order kernels were
        # generated and that the backward wrapper code contains exactly one MixOrderReductionGrid,
        # i.e. the weight kernel is shared. Skipped if the feature is disabled.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(xs, w, eps):
            # [FUNCTION DOC] HELPER: Applies RMS norm with shared weight w to each tensor in xs
            # and returns the list of outputs.
            ys = []
            for x in xs:
                ys.append(F.rms_norm(x, x.shape[-1:], w, eps=eps))
            return ys

        num_norm = 3
        M, N = 32768, 768
        xs = [
            torch.randn(M, N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
            for _ in range(num_norm)
        ]
        w = torch.randn(N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        dys = [torch.randn_like(xs[0]) for _ in range(num_norm)]
        eps = 1e-5

        ref = f(xs, w, eps)

        # use float64 to compute ref_grads for precision
        # and cast back to original dtype
        xs_f64 = [x.to(torch.float64) for x in xs]
        w_f64 = w.to(torch.float64)
        dys_f64 = [dy.to(torch.float64) for dy in dys]
        ref_f64 = f(xs_f64, w_f64, eps)
        ref_grads_f64 = torch.autograd.grad(ref_f64, [*xs_f64, w_f64], dys_f64)
        ref_grads = [g.to(dtype) for g in ref_grads_f64]

        act = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )(xs, w, eps)
        act_grads, (wrapper,) = utils.run_and_get_code(
            lambda: torch.autograd.grad(act, [*xs, w], dys)
        )
        # bfloat16 cause big numerical instability for grad_weight
        tol = 1e-3 if dtype == torch.float32 else 0.5
        self.assertTrue(same((ref, ref_grads), (act, act_grads), tol=tol))
        self.assertEqual(
            metrics.codegen_mix_order_reduction,
            num_norm,
        )

        # a single mix order reduction kernel get shared
        FileCheck().check_count("MixOrderReductionGrid", 1, exactly=True).run(wrapper)

    @parametrize("split_reductions", (False, True))
    @parametrize("dtype", [torch.bfloat16, torch.float])
    @parametrize("has_bias", [False, True])
    def test_layer_norm_sharing_weights(self, split_reductions, dtype, has_bias):
        # [FUNCTION DOC] Same as the RMS norm sharing test but with layer norm and optional bias.
        # Compares outputs and x grads with tolerance by dtype (XPU gets a looser bfloat16 value),
        # compares weight/bias grads only for float32, asserts three mix-order kernels and exactly
        # one MixOrderReductionGrid in the backward wrapper. Skipped if the feature is disabled.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(xs, w, bias, eps):
            # [FUNCTION DOC] HELPER: Applies layer norm with shared weight w and bias to each
            # tensor in xs and returns the list of outputs.
            ys = []
            for x in xs:
                ys.append(F.layer_norm(x, x.shape[-1:], w, bias=bias, eps=eps))
            return ys

        num_norm = 3
        M, N = 32768, 768
        xs = [
            torch.randn(M, N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
            for _ in range(num_norm)
        ]
        w = torch.randn(N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        b = (
            torch.randn(N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
            if has_bias
            else None
        )
        dys = [torch.randn_like(xs[0]) for _ in range(num_norm)]
        eps = 1e-5

        ref = f(xs, w, b, eps)
        act = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )(xs, w, b, eps)

        inputs_for_grad = [*xs, w]
        if has_bias:
            inputs_for_grad.append(b)
        ref_grads = torch.autograd.grad(ref, inputs_for_grad, dys)
        act_grads, (wrapper,) = utils.run_and_get_code(
            lambda: torch.autograd.grad(act, inputs_for_grad, dys)
        )
        tol = 1e-3 if dtype == torch.float32 else 1e-2
        if GPU_TYPE == "xpu":
            tol = 1e-3 if dtype == torch.float32 else 2e-2
        self.assertTrue(same((ref, ref_grads[:-2]), (act, act_grads[:-2]), tol=tol))
        if dtype == torch.float32:
            # bfloat16 cause big numerical instability for grad_weight
            # and grad_bias
            torch.testing.assert_close(
                ref_grads[-2:], act_grads[-2:], atol=tol, rtol=tol
            )
        self.assertEqual(
            metrics.codegen_mix_order_reduction,
            num_norm,
        )

        # a single mix order reduction kernel get shared
        FileCheck().check_count("MixOrderReductionGrid", 1, exactly=True).run(wrapper)

    @inductor_config.patch(split_reductions=False)
    def test_dont_fuse_nodes_that_introduce_producer_consumer_rel(self):
        # [FUNCTION DOC] Builds a row sum, a column sum and a pointwise op over x plus the column
        # sum. Checks numerics, that exactly one mix-order kernel exists, and that the pointwise
        # op is not fused into it (which would make the inner reduction a producer of the outer
        # one). Skipped if the feature is disabled.
        # KNOWN ISSUE: the final assertion passes the kernel count as the message argument of the
        # true check, so it always passes; it was probably meant to be an equality check.
        """
        The test constructs an inner reduction, an outer reduction and
        a pointwise kernel.

        The inner reduction and outer reduction will be fused first.
        We don't further fuse the pointwise kernel (with the inner reduction part)
        since that introduces producer/consumer relationship between
        the inner and outer reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the dim 1 sum, and the dim 0 sum (keeping the
            # dim) plus x.
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0, keepdim=True) + x
            return out1, out2

        x = torch.randn(32768, 768, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(1, metrics.codegen_mix_order_reduction)
        # two kernels
        # one is the mix-order reduction kernel
        # the other is the piontwise kernel
        self.assertTrue(2, metrics.generated_kernel_count)

    @patch("torch._inductor.scheduler.MixOrderReduction.is_split_reduction")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_numel_rnumel")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_common_read")
    @patch("torch._inductor.scheduler.MixOrderReduction.has_mix_reduction_orders")
    def test_mix_order_reduction_non_strict_mode(
        self,
        mock_has_mix_reduction_orders: mock.Mock,
        mock_get_common_read: mock.Mock,
        mock_get_numel_rnumel: mock.Mock,
        mock_is_split_reduction: mock.Mock,
    ):
        # [FUNCTION DOC] Uses mocked scheduler nodes and patched MixOrderReduction helpers (mix
        # orders present, constant sizes, not split) inside a dummy graph handler to check
        # can_fuse: it returns False when non-strict mode is off and True when it is on. The mock
        # arguments are the patched class methods injected by the decorators. Skipped if the
        # feature is disabled.
        """
        This tests whether we can skip some non-critical checks
        when non_strict mode is on
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        from torch._inductor.scheduler import BaseSchedulerNode

        mock_node_1 = mock.create_autospec(BaseSchedulerNode)
        mock_node_2 = mock.create_autospec(BaseSchedulerNode)

        mock_node_1.is_gpu.return_value = True
        mock_node_2.is_gpu.return_value = True

        mock_node_1.get_device.return_value.type = "cuda"
        mock_node_1.is_reduction.return_value = True
        mock_node_2.is_reduction.return_value = True
        mock_node_1.has_strict_sum.return_value = False
        mock_node_2.has_strict_sum.return_value = False

        from torch._inductor.utils import OrderedSet

        mock_node_1.ancestors = OrderedSet()
        mock_node_2.ancestors = OrderedSet()
        mock_node_1.get_operation_names.return_value = OrderedSet()
        mock_node_2.get_operation_names.return_value = OrderedSet()

        mock_has_mix_reduction_orders.return_value = True
        mock_get_common_read.return_value = "common_read"
        from sympy import Integer

        mock_get_numel_rnumel.return_value = (Integer(1), Integer(1))
        mock_is_split_reduction.return_value = False

        mock_node_1.read_writes = mock.Mock()
        mock_node_1.read_writes.reads = []

        # Create a dummy graph
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(lambda: torch.zeros(2, 3))()
        graph = GraphLowering(gm)

        with (
            V.set_graph_handler(graph),
            inductor_config.patch(
                {"triton.mix_order_reduction_non_strict_mode": False}
            ),
        ):
            self.assertFalse(MixOrderReduction.can_fuse(mock_node_1, mock_node_2))
        with (
            V.set_graph_handler(graph),
            inductor_config.patch(
                {
                    "triton.mix_order_reduction_non_strict_mode": True,
                }
            ),
        ):
            self.assertTrue(MixOrderReduction.can_fuse(mock_node_1, mock_node_2))

    @patch("torch._inductor.scheduler.MixOrderReduction.is_split_reduction")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_numel_rnumel")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_common_read")
    @patch("torch._inductor.scheduler.MixOrderReduction.has_mix_reduction_orders")
    def test_mix_order_reduction_data_dependent_checks_skip_fusion(
        self,
        mock_has_mix_reduction_orders: mock.Mock,
        mock_get_common_read: mock.Mock,
        mock_get_numel_rnumel: mock.Mock,
        mock_is_split_reduction: mock.Mock,
    ):
        # [FUNCTION DOC] Same mock setup as the non-strict mode test, but the numel and rnumel
        # are unbacked symbolic ints from the graph shape environment. Asserts can_fuse returns
        # False with non-strict mode off, because data-dependent sizes cannot be checked. Mock
        # arguments are the patched class methods injected by decorators. Skipped if the
        # feature is disabled.
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        from torch._inductor.scheduler import BaseSchedulerNode
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        mock_node_1 = mock.create_autospec(BaseSchedulerNode)
        mock_node_2 = mock.create_autospec(BaseSchedulerNode)

        mock_node_1.is_gpu.return_value = True
        mock_node_2.is_gpu.return_value = True
        mock_node_1.get_device.return_value.type = "cuda"
        mock_node_1.is_reduction.return_value = True
        mock_node_2.is_reduction.return_value = True
        mock_node_1.has_strict_sum.return_value = False
        mock_node_2.has_strict_sum.return_value = False

        from torch._inductor.utils import OrderedSet

        mock_node_1.ancestors = OrderedSet()
        mock_node_2.ancestors = OrderedSet()
        mock_node_1.get_operation_names.return_value = OrderedSet()
        mock_node_2.get_operation_names.return_value = OrderedSet()

        mock_has_mix_reduction_orders.return_value = True
        mock_get_common_read.return_value = ["common_read"]
        mock_is_split_reduction.return_value = False

        mock_node_1.read_writes = mock.Mock()
        mock_node_1.read_writes.reads = []

        from torch._inductor.graph import GraphLowering

        gm = make_fx(lambda: torch.zeros(2, 3))()
        graph = GraphLowering(gm)
        nrow = graph.sizevars.shape_env.create_unbacked_symint().node.expr
        ncol = graph.sizevars.shape_env.create_unbacked_symint().node.expr
        mock_get_numel_rnumel.return_value = (nrow, ncol)

        with (
            V.set_graph_handler(graph),
            inductor_config.patch(
                {"triton.mix_order_reduction_non_strict_mode": False}
            ),
        ):
            self.assertFalse(MixOrderReduction.can_fuse(mock_node_1, mock_node_2))

    @inductor_config.patch({"triton.mix_order_reduction_non_strict_mode": True})
    def test_no_recompile(self):
        # [FUNCTION DOC] With non-strict mode on, compiles a row and column sum with a
        # dynamic first dimension, checks results and one mix-order kernel, then calls with
        # two other batch sizes and asserts the dynamo compilation metrics contain only one
        # entry (no recompilation).
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the dim 1 sum and the dim 0 sum of x.
            return x.sum(dim=1), x.sum(dim=0)

        x0 = torch.randn(2048, 1024, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x0, (0,))
        opt_f = torch.compile(f)

        ref = f(x0)
        act = opt_f(x0)

        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)
        self.assertEqual(metrics.codegen_mix_order_reduction, 1)

        opt_f(torch.randn(4096, 1024, device=GPU_TYPE))
        opt_f(torch.randn(512, 1024, device=GPU_TYPE))

        compile_metrics = torch._dynamo.utils._compilation_metrics
        self.assertEqual(len(compile_metrics), 1, "Don't recompile")

    @skipIfXpu(msg="https://github.com/intel/intel-xpu-backend-for-triton/issues/6398")
    def test_additive_rnumel(self):
        # [FUNCTION DOC] Regression test for issue 176375: concatenates two inputs with dynamic
        # second dims, multiplies with a sliced weight, adds the total sum, and returns two
        # reductions, so the reduction size is an additive symbolic expression. Compares against
        # eager and asserts the mix-order metric equals the flag. Skipped on XPU.
        """
        Fix https://github.com/pytorch/pytorch/issues/176375
        """
        x = torch.randn(32768, 300, device=GPU_TYPE)
        y = torch.randn(32768, 200, device=GPU_TYPE)
        w = torch.randn(550, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(y, 1)

        def f(x, y, w):
            # [FUNCTION DOC] HELPER: Concatenates x and y along dim 1, scales by a weight slice
            # matching the width, adds the global sum, and returns the dim 0 and dim 1 sums.
            z = torch.cat((x, y), dim=1)

            # Slice w_pool to match dynamic dim. This avoids a guard that would
            # resolve s0+s1 to a concrete value (640), which is essential for
            # keeping rnumel symbolic in the generated code.
            w = w[: z.shape[-1]]  # [s0+s1], no concrete-equality guard
            z = z * w
            scale = z.sum()
            z = z + scale
            return z.sum(dim=0), z.sum(dim=1)

        ref = f(x, y, w)
        act = torch.compile(f)(x, y, w)

        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)

        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @recover_orig_fp32_precision
    def test_additive_num_splits(self):
        # [FUNCTION DOC] Checks that an additive split count is parenthesized in generated code:
        # runs a linear plus layer norm model on a dynamic-shape input, compares weight and bias
        # grads of the compiled and eager models (tolerance 1e-3). Sets high matmul precision; the
        # fp32 precision is restored by the decorator.
        """
        When the `num_splits` is an additive expression, a pair of
        parenthesis is required.
        """
        torch.set_float32_matmul_precision("high")
        linear1 = nn.Linear(1000, 1000).to(GPU_TYPE)
        norm = nn.LayerNorm(1000).to(GPU_TYPE)

        def model(x):
            # [FUNCTION DOC] HELPER: Drops the last position along dim 1, flattens to rows of 1000
            # and applies the linear layer and layer norm from the enclosing test.
            return norm(linear1(x[:, :-1].reshape(-1, 1000)))

        compiled_model = torch.compile(model)
        x = torch.randn(32, 200, 1000, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 1)
        compiled_model(x).sum().backward()

        act = linear1.weight.grad, linear1.bias.grad

        linear1.zero_grad()
        norm.zero_grad()
        model(x).sum().backward()
        ref = linear1.weight.grad, linear1.bias.grad

        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)

    @largeTensorTest("36GB", device=GPU_TYPE, inductor=True)
    def test_out_of_shared_memory(self):
        # [FUNCTION DOC] Regression test for issue 175250 (out of shared memory). Builds a
        # compiled attention-like forward with dual RMS norms, QKV projections, rotary embeddings
        # and repeated KV heads at a large sequence length, runs forward and backward, and asserts
        # more than one mix-order kernel was generated. Skipped if disabled; needs 36GB of memory.
        """
        Fix https://github.com/pytorch/pytorch/issues/175250
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        NUM_HEADS = 32
        NUM_KV_HEADS = 8
        HEAD_DIM = 128
        HIDDEN_SIZE = NUM_HEADS * HEAD_DIM * 2
        SEQ_LEN = 8192 * 2

        def rotate_half(x):
            # [FUNCTION DOC] HELPER: Splits the last dim in halves and returns the concatenation
            # of the negated second half with the first half.
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin):
            # [FUNCTION DOC] HELPER: Applies rotary embeddings to q and k using cos and sin
            # broadcast over the heads dim; returns the rotated q and k.
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
            return (q * cos) + (rotate_half(q) * sin), (k * cos) + (
                rotate_half(k) * sin
            )

        @torch.compile
        def forward(
            x,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            embed_norm,
            hidden_norm,
            cos,
            sin,
        ):
            # [FUNCTION DOC] HELPER: Compiled forward of the test model: splits the input into
            # embeds and hidden halves, normalizes each, recombines, projects to q, k, v, applies
            # rotary embeddings, repeats kv heads, does unfused matrix products and returns the
            # output projection plus the residual.
            batch, seq_len, _ = x.shape

            # Eagle3 first layer: split concatenated [embeds, hidden] input
            mid = x.shape[2] // 2
            embeds, hidden = x.split(mid, dim=-1)

            # Dual RMSNorm (pow, sum, div, mul in backward)
            embeds = embed_norm(embeds)
            hidden = hidden_norm(hidden)
            residual = hidden

            # Recombine for attention input (2 * HIDDEN_SIZE)
            x = torch.cat([embeds, hidden], dim=-1)

            # Adding a graph break here "fixes" the issue
            # by breaking up the fused op
            # torch._dynamo.graph_break()

            # Q/K/V projections from 2*hidden_size input
            q = q_proj(x).view(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k_proj(x).view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v_proj(x).view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = torch.repeat_interleave(k, NUM_HEADS // NUM_KV_HEADS, dim=1)
            v = torch.repeat_interleave(v, NUM_HEADS // NUM_KV_HEADS, dim=1)
            out = q.contiguous() @ k.contiguous().transpose(-2, -1) @ v.contiguous()

            out = out.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
            return o_proj(out) + residual

        # Layers
        embed_norm = nn.RMSNorm(HIDDEN_SIZE).to(GPU_TYPE)
        hidden_norm = nn.RMSNorm(HIDDEN_SIZE).to(GPU_TYPE)
        # Q/K/V project from 2*HIDDEN_SIZE (concatenated embeds + hidden)
        q_proj = nn.Linear(2 * HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False).to(
            GPU_TYPE
        )
        k_proj = nn.Linear(2 * HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False).to(
            GPU_TYPE
        )
        v_proj = nn.Linear(2 * HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False).to(
            GPU_TYPE
        )
        o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False).to(GPU_TYPE)

        # Block mask - simple causal only
        def causal_mask(_b, _h, q, kv):
            # [FUNCTION DOC] HELPER: Causal mask predicate returning whether q index is at least
            # kv index. It is not used elsewhere in the test.
            return q >= kv

        # Rotary embeddings (precomputed, no grad needed)
        inv_freq = 1.0 / (
            500000.0
            ** (
                torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=GPU_TYPE)
                / HEAD_DIM
            )
        )
        pos = torch.arange(1, SEQ_LEN + 1, dtype=torch.float32, device=GPU_TYPE)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        cos, sin = emb.cos(), emb.sin()

        # Input: 2*HIDDEN_SIZE to match split [embeds, hidden]
        x = torch.randn(
            1, SEQ_LEN, 2 * HIDDEN_SIZE, device=GPU_TYPE, requires_grad=True
        )

        out = forward(
            x,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            embed_norm,
            hidden_norm,
            cos,
            sin,
        )
        loss = out.sum()
        loss.backward()
        self.assertTrue(metrics.codegen_mix_order_reduction > 1)

    @inductor_config.patch("triton.mix_order_reduction", True)
    @inductor_config.patch("triton.mix_order_reduction_non_strict_mode", True)
    def test_dimension_refactoring_mismatch(self):
        # [FUNCTION DOC] Reproduces a mismatch between the dimension factorization seen during
        # fusion and the one used by code generation after a reduction split (13 by 8472 RMSNorm
        # backward). Runs forward/backward eagerly and compiled and compares grads (tolerance
        # 1e-3). Skipped if the feature is disabled.
        # KNOWN ISSUE: the final metric assertion is greater-or-equal zero, which always holds.
        """
        This reproduces an issue where `simplify_and_reorder()` produces a different
        dimension factorization than `_original_ranges` used during fusion decision.
        For example, fusion might see (13, 8472) but codegen sees (26, 4236) after
        the reduction split optimization adds a factor of 2 to the pointwise dimensions.

        We skip fusing split reductions for node1 in this case.
        """

        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        # Reproduce the RMSNorm backward pattern that triggered the bug.
        # The key is:
        # - Shape (M, N) = (13, 8472) where N=8472 is large enough to trigger split
        # - RMSNorm backward creates reductions along both dimensions
        # - The feature dimension reduction (8472) gets split with factor 2
        # - Mix order reduction tries to fuse these, but groups don't match after split
        def f(x, w, eps):
            # [FUNCTION DOC] HELPER: RMSNorm computed manually in float and cast back to
            # the input dtype.
            orig_dtype = x.dtype
            x = x.float()
            # RMSNorm forward: y = x * rsqrt(mean(x^2) + eps) * w
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)
            return y

        def fwd_bwd(compiled_f):
            # [FUNCTION DOC] HELPER: Clears grads, runs compiled_f on x, w, eps from the enclosing
            # test, backpropagates dy and returns the grads of x and w.
            x.grad = None
            w.grad = None
            out = compiled_f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        # Use the exact shape from the bug report: (13, 8472)
        # 8472 = 2 * 4236, so split with factor 2 gives sub-reductions of 4236
        M, N = 13, 8472
        x = torch.randn(M, N, dtype=torch.float32, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=torch.float32, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(f)

        ref = fwd_bwd(f)
        act = fwd_bwd(opt_f)
        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)
        self.assertGreaterEqual(metrics.codegen_mix_order_reduction, 0)

    def test_keepdim_shape_mismatch(self):
        # [FUNCTION DOC] Regression test for keepdim handling: a custom normalization module with
        # parameters of shape one by hidden is run forward and backward eagerly and compiled,
        # grads are compared (tolerance 1e-3), and at least one mix-order kernel must be
        # generated. Skipped if the feature is disabled.
        """
        Test that MixOrderReduction correctly handles keepdim=True reductions.

        This test reproduces a bug where the final reduction in MixOrderReduction
        generates `view(nsplit, rnumel).sum(dim=0)` which produces shape [rnumel],
        but the expected output should be [1, rnumel] when keepdim=True.

        The error manifests as:
        RuntimeError: Function CompiledFunctionBackward returned an invalid gradient
        at index N - got [2048] but expected shape compatible with [1, 2048]
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        # Create a model that produces reductions with keepdim=True in the backward pass
        # This pattern is common in normalization layers like RMSNorm/LayerNorm
        class KeepDimReductionModel(nn.Module):
            # [CLASS DOC] HELPER: Small normalization module (mean/variance normalize, then scale
            # and shift) with weight and bias parameters shaped one by hidden_size, used to get
            # keepdim reductions in the backward pass.
            def __init__(self, hidden_size):
                # [FUNCTION DOC] HELPER: Initializes weight as ones and bias as zeros of shape
                # one by hidden_size.
                super().__init__()
                # Using shape [1, hidden_size] to ensure keepdim=True in backward
                self.weight = nn.Parameter(torch.ones(1, hidden_size))
                self.bias = nn.Parameter(torch.zeros(1, hidden_size))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Normalizes x over the last dim using mean and biased
                # variance (eps 1e-5), then applies the weight and bias; returns the result.
                # x: [batch, hidden_size]
                # Normalization-like operation that produces keepdim reductions in backward
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                x_norm = (x - mean) / (var + 1e-5).sqrt()
                return x_norm * self.weight + self.bias

        M, N = 32768, 2048  # Large batch to trigger mix order reduction
        model = KeepDimReductionModel(N).to(GPU_TYPE)

        x = torch.randn(M, N, dtype=torch.float32, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)

        def fwd_bwd(model, x, dy):
            # [FUNCTION DOC] HELPER: Clears grads of x and model, runs model on x, backpropagates
            # dy, and returns the grads of x, weight and bias.
            x.grad = None
            model.zero_grad()
            out = model(x)
            out.backward(dy)
            return x.grad, model.weight.grad, model.bias.grad

        # Reference (eager)
        ref = fwd_bwd(model, x, dy)

        # Compiled with mix order reduction
        compiled_model = torch.compile(model)
        act = fwd_bwd(compiled_model, x, dy)

        # Verify numerical correctness
        self.assertTrue(
            same(ref, act, tol=1e-3), lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}"
        )

        # Verify mix order reduction was used
        self.assertGreater(
            metrics.codegen_mix_order_reduction,
            0,
            "Mix order reduction should be triggered",
        )

    def test_multi_dim_reduction_output_shape(self):
        # [FUNCTION DOC] Regression test for issue 178080: compiles a function with a sum over
        # dims 1 and 2 and a sum over dim 0 of a 3D tensor, and checks both output shapes and
        # values match eager.
        """
        Regression test for https://github.com/pytorch/pytorch/issues/178080:
        two reductions over different dims in the same compiled function must
        produce correctly-shaped outputs.

        x.sum(dim=(1, 2)) over [32768, 512, 2] -> [32768]
        x.sum(dim=0)      over [32768, 512, 2] -> [512, 2]
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum over dims 1 and 2 and the sum over dim
            # 0 of x.
            return x.sum(dim=(1, 2)), x.sum(dim=0)

        x = torch.randn(32768, 512, 2, device=GPU_TYPE)
        ref = f(x)
        act = torch.compile(f)(x)

        self.assertEqual(list(act[0].shape), list(ref[0].shape))
        self.assertEqual(list(act[1].shape), list(ref[1].shape))
        self.assertTrue(same(ref, act, tol=1e-3))


class OverFusionTest(TestBase):
    # [CLASS DOC] Regression tests for over-fusion in mix-order reduction (issue 179423), where
    # absorbing too many pointwise nodes creates kernels with too many loads.
    """
    Regression test for mix-order reduction over-fusion in transformer backward
    passes. When can_fuse_with absorbs too many pointwise nodes into a
    mixed-order kernel, the resulting kernel has excessive buffer reads (loads)
    per RSPLIT loop iteration, causing register spills and performance
    regression. See #179423.
    """

    @skipIfXpu(
        msg="XPU selects Flash Attention for SDPA backward; the current SYCL TLA"
        "implementation does not guarantee precision on PVC. Re-enable once oneDNN"
        "adds SDPA backward support. See https://github.com/intel/torch-xpu-ops/issues/4094"
    )
    @inductor_config.patch(
        {
            "triton.mix_order_reduction": True,
            "triton.mix_order_reduction_max_reads": 10,
            # These assertions inspect scheduler/codegen metrics populated only
            # during fresh compilation; a warm FX graph cache bypasses that path.
            "force_disable_caches": True,
        }
    )
    def test_max_reads_limits_fusion(self):
        # [FUNCTION DOC] Builds a three-block transformer (GQA attention with QK RMS norm, squared
        # leaky relu MLP) in bfloat16, runs backward eagerly and compiled with the max reads limit
        # set to 10 and caches disabled, compares input grads (tolerance 5e-2), and asserts mix-
        # order kernels were still generated while some fusions were rejected. Skipped without a
        # GPU or on XPU.
        """
        Verify that max_reads limits over-fusion in a transformer backward
        pass without disabling mix-order reduction entirely.

        Uses the exact model pattern from #179423: GQA attention with QK-norm
        and squared leaky-relu MLP. The QK-norm creates extra intermediate
        buffers in the backward pass that push read counts above the threshold.
        """
        if not HAS_GPU:
            self.skipTest("requires GPU")

        num_heads = 8
        num_kv_heads = 4
        dim = 512
        head_dim = dim // num_heads

        class Attention(nn.Module):
            # [CLASS DOC] HELPER: Grouped-query attention module with q, k, v and output
            # projections, QK RMS norm, repeated kv heads and causal scaled dot product attention.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates four bias-free linear layers (q, k, v, output
                # projection) using dim, num_kv_heads and head_dim from the enclosing test.
                super().__init__()
                self.c_q = nn.Linear(dim, dim, bias=False)
                self.c_k = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
                self.c_v = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
                self.proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Projects x to q, k, v, RMS-normalizes q and k,
                # expands kv heads to match the query heads, applies causal attention and
                # the output projection.
                B, T, D = x.shape
                q = self.c_q(x).reshape(B, T, num_heads, head_dim)
                k = self.c_k(x).reshape(B, T, num_kv_heads, head_dim)
                v = self.c_v(x).reshape(B, T, num_kv_heads, head_dim)
                q = F.rms_norm(q, (q.size(-1),))
                k = F.rms_norm(k, (k.size(-1),))
                q = q.transpose(1, 2)
                k = k.transpose(1, 2).repeat_interleave(
                    num_heads // num_kv_heads, dim=1
                )
                v = v.transpose(1, 2).repeat_interleave(
                    num_heads // num_kv_heads, dim=1
                )
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                return self.proj(y.transpose(1, 2).reshape(B, T, D))

        class Block(nn.Module):
            # [CLASS DOC] HELPER: Transformer block with pre-norm attention and a pre-norm MLP
            # using a squared leaky relu activation, both with residual connections.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates two RMS norms, an attention module and the two
                # MLP linear layers.
                super().__init__()
                self.attn_norm = nn.RMSNorm(dim)
                self.mlp_norm = nn.RMSNorm(dim)
                self.attn = Attention()
                self.fc1 = nn.Linear(dim, dim * 4, bias=False)
                self.fc2 = nn.Linear(dim * 4, dim, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the attention and MLP sub-layers with residual
                # additions and returns the result.
                x = x + self.attn(self.attn_norm(x))
                h = self.mlp_norm(x)
                x = x + self.fc2(F.leaky_relu(self.fc1(h), negative_slope=0.5).square())
                return x

        model = nn.Sequential(*[Block() for _ in range(3)]).to(GPU_TYPE).bfloat16()

        x = torch.randn(
            8, 2048, dim, device=GPU_TYPE, dtype=torch.bfloat16, requires_grad=True
        )
        dy = torch.randn_like(x)

        out_ref = model(x)
        out_ref.backward(dy)
        grad_ref = x.grad.clone()
        x.grad = None

        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        out_act = compiled(x)
        out_act.backward(dy)
        grad_act = x.grad.clone()

        self.assertTrue(same(grad_ref, grad_act, tol=5e-2))

        # max_reads should limit over-fusion, not disable mix_order entirely
        self.assertGreater(metrics.codegen_mix_order_reduction, 0)
        self.assertGreater(metrics.rejected_mix_order_reduction_fusion, 0)


class MixOrderReductionHeuristicTest(TestBase):
    # [CLASS DOC] CPU-runnable tests of the NUM_STAGES choices in the mix-order persistent
    # reduction autotune heuristic, run via the return-configs mode with no GPU.
    """
    CPU-runnable unit tests for the mix-order persistent_reduction autotuning
    heuristic. These exercise the config generation logic directly (via
    ``return_configs=True``) without needing a GPU.
    """

    def test_uses_tma_tracks_emitted_descriptors(self):
        # [FUNCTION DOC] For no, host and device TMA states, builds a TritonKernel without
        # calling its constructor, sets the host descriptor mapping and the device TMA flag, and
        # asserts the uses_tma property is true unless there is no TMA and uses_device_tma is
        # true only for device.
        for source in (None, "host", "device"):
            with self.subTest(source=source):
                kernel = object.__new__(TritonKernel)
                host_descriptors = {}
                if source == "host":
                    host_descriptors["arg"] = mock.sentinel.descriptor
                kernel.host_tma_descriptor_args = host_descriptors
                kernel._emitted_device_tma = source == "device"
                self.assertEqual(kernel.uses_tma, source is not None)
                self.assertEqual(kernel.uses_device_tma, source == "device")

    def _gen_num_stages(
        self, *, tma, allow_multi_stages, device_tma=False, rsplit_size=256, rnumel=256
    ):
        # [FUNCTION DOC] HELPER: Calls the persistent reduction heuristic in return-configs mode
        # for a fake CPU device with the mix-order meta (split size, multi-stage permission,
        # optional TMA flags) and the given rnumel. Asserts at least one config exists and returns
        # the set of NUM_STAGES values in the configs.
        """
        Drive the mix-order branch of persistent_reduction and return the set of
        NUM_STAGES values across the generated configs.
        """
        inductor_meta = {
            "RSPLIT_SIZE": rsplit_size,
            "mix_order_reduction_allow_multi_stages": allow_multi_stages,
        }
        if tma:
            inductor_meta["uses_tma"] = True
        if device_tma:
            inductor_meta["uses_device_tma"] = True
        configs = persistent_reduction(
            {"x": 4096, "r0_": rnumel},
            triton_meta={
                "device": DeviceProperties(
                    type="cpu", index=0, multi_processor_count=1, cc=0
                )
            },
            inductor_meta=inductor_meta,
            return_configs=True,
        )
        self.assertTrue(configs, "expected at least one config")
        return {c.kwargs["NUM_STAGES"] for c in configs}

    def test_num_stages_support_matrix(self):
        # [FUNCTION DOC] Checks that with multi stages allowed, no TMA gives the stage set of just
        # 3 and device TMA gives just 1.
        cases = (
            ("no TMA", False, False, {3}),
            ("device TMA", True, True, {1}),
        )
        for name, tma, device_tma, expected in cases:
            with self.subTest(name):
                stages = self._gen_num_stages(
                    tma=tma,
                    device_tma=device_tma,
                    allow_multi_stages=True,
                )
                self.assertEqual(stages, expected)

    def test_num_stages_single_when_multi_stage_disabled(self):
        # [FUNCTION DOC] Checks that with multi stages disallowed, the stage set is exactly 1 both
        # with and without TMA.
        """
        When multi-staging is disabled, NUM_STAGES collapses to 1 regardless of
        whether TMA is active.
        """
        for tma in (False, True):
            stages = self._gen_num_stages(tma=tma, allow_multi_stages=False)
            self.assertEqual(stages, {1}, f"tma={tma}: expected {{1}}, got {stages}")

    def test_device_tma_disables_multi_stages_across_shapes(self):
        # [FUNCTION DOC] Checks that with device TMA the stage set is exactly 1 for a grid of
        # split sizes and reduction sizes.
        for rsplit_size in (16, 64, 128, 256):
            for rnumel in (256, 1024, 8192, 16384):
                stages = self._gen_num_stages(
                    tma=True,
                    device_tma=True,
                    allow_multi_stages=True,
                    rsplit_size=rsplit_size,
                    rnumel=rnumel,
                )
                self.assertEqual(stages, {1})


@inductor_config.patch(
    "triton.mix_order_reduction", not inductor_config.triton.mix_order_reduction
)
class NoMixOrderReductionTest(MixOrderReductionTest):
    # [CLASS DOC] Reruns all MixOrderReductionTest tests with the mix_order_reduction config
    # set to the opposite of its default, so the feature-off (or on) path is covered.
    # Defines nothing new.
    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
