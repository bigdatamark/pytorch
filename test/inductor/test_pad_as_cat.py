# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for two graph optimizations: sharing one concatenation between
# several consumers (cat multi-consumer) and rewriting padding as a concatenation with zeros
# (pad-as-cat). Tests are GPU-only. Import-time side effect: enables the inductor metrics
# logging artifact so the bytes-accessed counter is populated. Runs via the inductor
# run_tests.
"""Tests for cat multi-consumer and pad-as-cat optimizations."""

import torch
from torch._dynamo.utils import counters
from torch._inductor import metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


# required so that metrics.num_bytes_accessed is populated
torch._logging.set_logs(inductor_metrics=True)


class TestCatMultiConsumer(TestCase):
    # [CLASS DOC] Tests that a cat whose input is also consumed elsewhere is not recomputed,
    # and that a single-consumer cat still fuses into one pointwise kernel. Both tests use the
    # inductor metrics byte counter and disable the FX graph cache.
    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_cat_to_fp16(self):
        # [FUNCTION DOC] Compiles a function returning a row-wise concatenation of the input
        # with a zero block plus a float16 cast of the input. Checks both outputs equal eager
        # results and that the bytes accessed counter is below the unoptimized estimate where
        # the input would be read twice. Requires a GPU; resets the metrics counter before
        # running.
        """Multi-consumer cat avoids duplicate computation."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input concatenated along
            # dimension zero with a zero tensor of 6 rows and 768 columns, and the input cast
            # to float16.
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])

        # Without the optimization x would be read twice (once by cat, once
        # by to_fp16). With the optimization ConcatKernel shares x so it is
        # read only once.
        z = ref[0]
        y = ref[1]
        x_bytes = x.nelement() * x.element_size()
        z_bytes = z.nelement() * z.element_size()
        y_bytes = y.nelement() * y.element_size()
        unoptimized_bytes = 2 * x_bytes + z_bytes + y_bytes
        self.assertLess(
            metrics.num_bytes_accessed,
            unoptimized_bytes,
            "Optimization should avoid reading x twice.",
        )

    @torch._inductor.config.patch(fx_graph_cache=False)
    @requires_gpu()
    def test_single_consumer_cat_unchanged(self):
        # [FUNCTION DOC] Compiles a function returning only the zero-padded concatenation of
        # the input and checks the result equals eager. Then asserts the bytes accessed
        # counter is exactly the input size plus the output size, i.e. the zero fill is fused
        # into the cat kernel. Requires a GPU; resets the metrics counter first.
        """Single-consumer cat unchanged."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input concatenated along
            # dimension zero with a zero tensor of 6 rows and 768 columns.
            return torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        metrics.reset()
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result, ref)

        # Single-consumer cat should use pointwise_cat which fuses the
        # zeros fill into the cat kernel.  Total bytes: read x + write z.
        x_bytes = x.nelement() * x.element_size()
        z_bytes = ref.nelement() * ref.element_size()
        expected_bytes = x_bytes + z_bytes
        self.assertEqual(
            metrics.num_bytes_accessed,
            expected_bytes,
            lambda msg: f"{msg}\nExpected {expected_bytes} bytes, got {metrics.num_bytes_accessed}.",
        )


class TestPadAsCat(TestCase):
    # [CLASS DOC] Tests that functional padding is counted as rewritten into a cat via the
    # pad_rewritten_as_cat inductor counter, for multi-consumer and single-consumer inputs.
    @requires_gpu()
    def test_mul_pad_addmm(self):
        # [FUNCTION DOC] Compiles a function that pads a scaled bfloat16 tensor by 192
        # trailing columns and also feeds the scaled tensor to a matrix multiply with bias.
        # Checks both results against eager (loose tolerance for the matmul), that the
        # generated code contains a tensor reinterpretation (zero-copy buffer sharing), and
        # that the pad rewrite counter is positive.
        """Multi-consumer F.pad uses ConcatKernel zero-copy."""
        counters.clear()

        def fn(x, scale, bias, weight):
            # [FUNCTION DOC] HELPER: Function under test. Scales the input, pads the product
            # on the last dimension with 192 zeros, and also computes bias plus product times
            # weight. Returns both.
            mul_result = x * scale
            padded = torch.nn.functional.pad(mul_result, [0, 192])
            mm_result = torch.addmm(bias, mul_result, weight)
            return padded, mm_result

        x = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(2880, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x, scale, bias, weight)
        ref = fn(x, scale, bias, weight)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1], atol=1e-2, rtol=1e-2)
        self.assertIn("reinterpret_tensor", code)
        self.assertGreater(counters["inductor"]["pad_rewritten_as_cat"], 0)

    @requires_gpu()
    def test_single_consumer_pad(self):
        # [FUNCTION DOC] Compiles a function that pads a scaled float tensor by 192 trailing
        # columns, checks the result equals eager, and asserts the pad rewrite counter is
        # positive.
        """Single-consumer F.pad is decomposed into cat, which fuses via pointwise_cat."""
        counters.clear()

        def fn(x, scale):
            # [FUNCTION DOC] HELPER: Function under test. Multiplies the input by scale and
            # pads the last dimension with 192 zeros.
            return torch.nn.functional.pad(x * scale, [0, 192])

        x = torch.randn(128, 2880, device=GPU_TYPE)
        scale = torch.randn(128, 2880, device=GPU_TYPE)

        compiled = torch.compile(fn)
        result = compiled(x, scale)
        ref = fn(x, scale)

        self.assertEqual(result, ref)
        self.assertGreater(counters["inductor"]["pad_rewritten_as_cat"], 0)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
