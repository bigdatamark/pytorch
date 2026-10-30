# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor scatter-related optimizations: rewriting a constant-filled
# tensor followed by a scatter into a pointwise computation (checked through byte-count and
# match metrics), and the partitioned scatter FX pass that reduces atomic contention for
# accumulating index_put (checked via counters, accuracy against eager, config gates and GPU
# memory budget). Import-time side effects: turns on inductor metrics logging, reads the
# DO_PERF_TEST environment variable, and sets the default device to the GPU when one is
# available. The runner only starts when a GPU is present.

import copy
import os
import unittest

import torch
from torch import nn
from torch._dynamo.utils import counters, same
from torch._inductor import config, metrics
from torch._inductor.fx_passes.reduced_atomic_contention import _compute_num_partitions
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestScatterOpt(TestCase):
    # [CLASS DOC] Tests the full-plus-scatter to pointwise rewrite on CPU: each test compiles a
    # function, compares with eager, and inspects inductor metrics (bytes accessed, pattern
    # match counts). State is reset before every test.
    def setUp(self):
        # [FUNCTION DOC] Resets inductor metrics and clears the dynamo counters before each
        # test.
        super().setUp()
        metrics.reset()
        counters.clear()

    def check_metric(self, val=1):
        # [FUNCTION DOC] HELPER: Asserts the number of scatter-upon-constant-tensor matches
        # recorded in the metrics equals the expected value (default one).
        self.assertEqual(val, metrics.num_matches_for_scatter_upon_const_tensor)

    def do_acc_test(self, f, *args):
        # [FUNCTION DOC] HELPER: Runs the function eagerly and compiled on the same arguments
        # and asserts the results agree within a tolerance of 1e-3, with a message showing both
        # values.
        expect = f(*args)
        actual = torch.compile(f)(*args)
        self.assertTrue(
            same(expect, actual, tol=1e-3), lambda msg: f"{msg}\n{expect=}\n{actual=}\n"
        )

    def test_3d_tensor(self):
        # [FUNCTION DOC] Compiles a fill then scatter along the last dim of a 3D tensor and
        # checks accuracy plus that the bytes accessed equal the output size plus the index
        # size, meaning the fill and scatter were fused into a single pass.
        L, M, N = 2, 1024, 2048

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a 3D float tensor with a constant, scatters a second
            # constant along dim 2 at the given indices, and returns it.
            y = torch.full([L, M, N], 3.14, dtype=torch.float)
            y.scatter_(2, x.unsqueeze(2), 2.718)
            return y

        x = torch.randint(0, N, (L, M), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = (
            L * M * N * torch.float.itemsize + L * M * torch.int64.itemsize
        )
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_non_last_dim(self):
        # [FUNCTION DOC] Checks the optimization applies when scattering along dim 0 of a 2D
        # tensor; accuracy and bytes accessed (output plus index) are verified.
        """
        Test the case that the scatter dimension is not the last one.
        """
        M, N = 1024, 2048

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a 2D float tensor with a constant, scatters a
            # constant along dim 0 using the unsqueezed index, and returns it.
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(0, x.unsqueeze(0), 2.718)
            return y

        x = torch.randint(0, M, (N,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + N * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_neg_scatter_dim(self):
        # [FUNCTION DOC] Checks that a negative scatter dim (last dim) is handled; accuracy and
        # the bytes accessed equal output plus index sizes.
        M, N = 1024, 2048

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a 2D tensor with a constant, scatters a constant
            # along dim minus one at the unsqueezed index, and returns it.
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(-1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_shorter_index_tensor(self):
        # [FUNCTION DOC] Uses an index tensor with half as many rows as the output; accuracy
        # must hold and no pattern is matched (pattern matcher count is zero) since shorter
        # indexes are unsupported.
        M, N = 1024, 2048

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a 2D tensor with a constant and scatters a constant
            # along dim 1 at the unsqueezed (shorter) index; returns the tensor.
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M // 2,), dtype=torch.int64)
        self.do_acc_test(f, x)

        # no match since the index tensor is shorter. May support it in future.
        self.assertEqual(0, counters["inductor"]["pattern_matcher_count"])

    def test_nonzero_const_tensor(self):
        # [FUNCTION DOC] Checks the rewrite works for a non-zero fill value; accuracy and bytes
        # accessed (output plus index) are verified.
        M, N = 1024, 2048

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a 2D tensor with a non-zero constant, scatters a
            # constant along dim 1 at the unsqueezed index, and returns it.
            y = torch.full([M, N], 3.14, dtype=torch.float)
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        x = torch.randint(0, N, (M,), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_dense(self):
        # [FUNCTION DOC] Uses a dense index (one index per half of the columns, not a single
        # column per row), which cannot be optimized; accuracy is checked and bytes accessed
        # must be at least the expected sum (lower bound only, because of a known dependency-
        # accounting imprecision).
        M, N = 1024, 2048

        def f(x):
            # [FUNCTION DOC] HELPER: Fills a 2D tensor with zeros and scatters a constant along
            # dim 1 using the full 2D index tensor; returns it.
            y = torch.full([M, N], 0, dtype=torch.float)
            y.scatter_(1, x, 0.618)
            return y

        x = torch.randint(0, N, (M, N // 2), dtype=torch.int64)
        self.do_acc_test(f, x)
        expected_num_bytes = M * N * torch.float.itemsize + M * (N // 2) * (
            torch.int64.itemsize + torch.float.itemsize
        )
        # Use assertGreaterEqual rather than assertEqual due to the issue related
        # to StarDep mentioned here: https://github.com/pytorch/pytorch/pull/129043#discussion_r1651699706
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

    def test_can_not_optimize_due_to_non_const(self):
        # [FUNCTION DOC] Scatters into a tensor passed as an argument rather than a constant
        # tensor, so no rewrite applies; checks accuracy, a lower bound on bytes accessed, and
        # then the exact byte count including a known overestimate for the mutation kernels
        # (three kernels are generated: copy in, scatter, copy back).
        M, N = 1024, 2048

        def f(x, y):
            # [FUNCTION DOC] HELPER: Scatters a constant in place into the argument tensor along
            # dim 1 at the given index and returns that tensor.
            y.scatter_(1, x, 0.618)
            return y

        x = torch.randint(0, N, (M, 1), dtype=torch.int64)
        y = torch.randn([M, N])
        self.do_acc_test(f, x, y)

        # The generated code is quite in-efficient.
        # There are 3 kernels
        # 1. copy from arg to buf
        # 2. scatter upon buf
        # 3. copy buf back to arg
        # Link to the wrapper: https://gist.github.com/shunting314/d43b74e680b3e5b514f7c28160c39f40
        expected_num_bytes = 4 * M * N * torch.float.itemsize + M * (
            torch.int64.itemsize + torch.float.itemsize
        )
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

        # the second kernel and third kernel are both mutation kernel. So we
        # overestimated the memory accessed
        # Update the test once the overestimiation is fixed.
        over_estimate = M * torch.float.itemsize + M * N * torch.float.itemsize
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes + over_estimate)

    def test_dtype_preserved(self):
        # [FUNCTION DOC] For float16 and bfloat16 constant tensors, checks the rewritten result
        # keeps the low-precision dtype (not promoted to float32), values agree with eager, and
        # the rewrite matched once. Metrics are reset per dtype.
        """
        The full+scatter -> pointwise rewrite must preserve the const tensor's
        dtype. torch.where with Python scalar branches promotes to the default
        float dtype (float32), so for a low-precision const tensor (e.g. f16)
        the rewrite previously produced a float32 result. This is exactly the
        pattern emitted by nll_loss_backward / cross_entropy backward on f16
        inputs.
        """
        M, N = 1024, 2048

        for dtype in (torch.float16, torch.bfloat16):
            metrics.reset()

            def f(x):
                # [FUNCTION DOC] HELPER: Fills a 2D tensor of the loop's dtype with a constant,
                # scatters a constant along dim 1 at the unsqueezed index, and returns it.
                y = torch.full([M, N], 3.14, dtype=dtype)
                y.scatter_(1, x.unsqueeze(1), 2.718)
                return y

            x = torch.randint(0, N, (M,), dtype=torch.int64)
            expect = f(x)
            actual = torch.compile(f)(x)
            self.assertEqual(expect.dtype, dtype)
            self.assertEqual(
                actual.dtype,
                dtype,
                lambda msg: f"{msg}\ndtype not preserved for {dtype}",
            )
            self.assertTrue(
                same(expect, actual, tol=1e-2),
                lambda msg: f"{msg}\n{expect=}\n{actual=}\n",
            )
            self.check_metric()

    def test_cross_entropy_loss(self):
        # [FUNCTION DOC] Runs cross entropy loss backward on a bf16 linear model eagerly and
        # compiled, compares weight gradients, and checks the pattern matched once. The
        # vocabulary size is reduced unless DO_PERF_TEST is set; in that mode it skips on xpu,
        # resets peak memory stats, benchmarks the compiled step, and prints time and peak
        # memory.
        """
        Match full+scatter in CEL and replaces it with a pointwise.

        Perf data on an A100 GPU:
        Without the scatter optimization:
          ms=47.340, peak_mem=10.524 GB
        With the scatter optimization:
          ms=42.768, peak_mem=7.227 GB
        """
        B, T, D, V = 32, 1024, 768, 50257
        if not DO_PERF_TEST:
            # use a smaller V if not doing perf test to avoid OOM
            # in CI
            V = V // 100
        ref_model = nn.Linear(D, V).to(torch.bfloat16)
        opt_model = copy.deepcopy(ref_model)
        ce = nn.CrossEntropyLoss()

        def f(m, x, label):
        # [FUNCTION DOC] HELPER: Computes cross entropy of the model output against labels
        # (flattened) and runs backward; returns nothing.
            ce(m(x).view(-1, V), label.view(-1)).backward()

        opt_f = torch.compile(f)

        x = torch.randn(B, T, D).to(torch.bfloat16)
        label = torch.randint(0, V, (B, T)).to(torch.int64)

        f(ref_model, x, label)
        ref_grad = ref_model.weight.grad
        opt_f(opt_model, x, label)
        act_grad = opt_model.weight.grad
        if not torch.allclose(ref_grad, act_grad, atol=1e-3, rtol=1e-3):
            raise AssertionError(f"{ref_grad=}\n{act_grad=}")

        self.check_metric()

        if DO_PERF_TEST:
            if GPU_TYPE == "xpu":
                raise unittest.SkipTest(
                    "torch.xpu.reset_peak_memory_stats not implemented."
                )
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                opt_f(opt_model, x, label)
            ms = benchmarker.benchmark_gpu(lambda: opt_f(opt_model, x, label))
            peak_mem = torch.cuda.max_memory_allocated() / 10**9
            print(f"{ms=:.3f}, {peak_mem=:.3f} GB")


class TestPartitionedScatterOpt(TestCase):
    # [CLASS DOC] Tests the partitioned scatter FX pass for accumulating index_put. setUp
    # enables the config flag and saves the enable and force flags, tearDown restores them.
    # Tests inspect counters for applied or skipped reasons and compare compiled output with
    # eager.
    """Tests for the partitioned scatter FX pass (reduced_atomic_contention.py)."""

    def setUp(self):
        # [FUNCTION DOC] Resets metrics, counters and dynamo, saves the enabled and force config
        # values for restoration, and turns the pass on.
        super().setUp()
        metrics.reset()
        counters.clear()
        torch._dynamo.reset()
        self._saved_enabled = config.partitioned_scatter_enabled
        self._saved_force = config.partitioned_scatter_force
        config.partitioned_scatter_enabled = True

    def tearDown(self):
        # [FUNCTION DOC] Restores the saved enabled and force config values, then calls the
        # parent tearDown.
        config.partitioned_scatter_enabled = self._saved_enabled
        config.partitioned_scatter_force = self._saved_force
        super().tearDown()

    def _check_accuracy(self, f, args, *, atol=1e-1, rtol=1e-2, exact=False):
        # [FUNCTION DOC] HELPER: Runs a function eagerly and through inductor (no grad, full
        # graph) and asserts every output (single or sequence) matches: bitwise equal when exact
        # is set, else close in float32 within the given tolerances.
        """Run f eagerly and through Inductor, assert the results agree."""
        with torch.no_grad():
            expected = f(*args)
            compiled_f = torch.compile(f, backend="inductor", fullgraph=True)
            actual = compiled_f(*args)

        if isinstance(expected, (list, tuple)):
            outputs = list(zip(expected, actual))
        else:
            outputs = [(expected, actual)]
        for e, a in outputs:
            if exact:
                self.assertTrue(torch.equal(e, a), f"expected={e}\nactual={a}")
            else:
                self.assertTrue(
                    torch.allclose(e.float(), a.float(), atol=atol, rtol=rtol),
                    f"expected={e}\nactual={a}",
                )

    def _make_scatter_inputs(
        self, N, output_shape, dtype=torch.float32, index_high=None, dim=0
    ):
        # [FUNCTION DOC] HELPER: Builds CPU inputs for a one-index scatter-add along a
        # dimension: an all-zero output of the given shape and dtype, a random int64 index of
        # length N, and a value tensor shaped like the output with that dimension replaced by N
        # (small ints for int32, normal random otherwise). The index upper bound defaults to
        # half the dim size (at least one).
        """
        Create (out, idx, vals) for a 1-index scatter-add along `dim`.
        index_high controls contention: smaller = more writes per slot.
        """
        if index_high is None:
            index_high = output_shape[dim] // 2 or 1
        idx = torch.randint(0, index_high, (N,), dtype=torch.int64)
        val_shape = list(output_shape)
        val_shape[dim] = N
        if dtype in (torch.int32,):
            vals = torch.randint(0, 4, val_shape, dtype=dtype)
        else:
            vals = torch.randn(val_shape, dtype=dtype)
        out = torch.zeros(output_shape, dtype=dtype)
        return out, idx, vals

    def test_pr_reference_accuracy(self):
        # [FUNCTION DOC] Seeds the RNG and runs three accumulating index_puts with only four
        # distinct slots on a 10k-row input; accuracy is checked with loose tolerance and the
        # pass must apply at least three times.
        """Scaled-down PR benchmark: three scatter-adds with high contention (4 slots)."""
        torch.manual_seed(42)
        N, D, n_small = 10_000, 10, 51

        def f(out0, out1, out2, idx0, idx1, idx2, vals):
            # [FUNCTION DOC] HELPER: Applies three accumulating index_puts with separate outputs
            # and indexes but shared values; returns the three results.
            out0 = out0.index_put([idx0], vals, accumulate=True)
            out1 = out1.index_put([idx1], vals, accumulate=True)
            out2 = out2.index_put([idx2], vals, accumulate=True)
            return out0, out1, out2

        vals = torch.randn(N, D, dtype=torch.float32)
        out0 = torch.zeros(n_small, D, dtype=torch.float32)
        out1 = torch.zeros(n_small, D, dtype=torch.float32)
        out2 = torch.zeros(n_small, D, dtype=torch.float32)
        idx0 = torch.randint(0, 4, (N,), dtype=torch.int64)
        idx1 = torch.randint(0, 4, (N,), dtype=torch.int64)
        idx2 = torch.randint(0, 4, (N,), dtype=torch.int64)

        self._check_accuracy(
            f, (out0, out1, out2, idx0, idx1, idx2, vals), atol=1.0, rtol=1e-2
        )
        self.assertGreaterEqual(counters["inductor"]["partitioned_scatter_applied"], 3)

    def test_accuracy_int32_exact(self):
        # [FUNCTION DOC] Integer 32-bit accumulating index_put must match eager exactly.
        """Integer scatter-add must be bit-for-bit identical to eager (addition is associative)."""
        torch.manual_seed(4)
        N = 8192

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Returns the output with values accumulated at the index
            # positions.
            return out.index_put([idx], vals, accumulate=True)

        out, idx, vals = self._make_scatter_inputs(N, (8,), torch.int32, index_high=8)
        self._check_accuracy(f, (out, idx, vals), exact=True)

    def test_accuracy_inplace_index_put(self):
        # [FUNCTION DOC] Checks the in-place accumulating index_put variant is also matched and
        # correct.
        """index_put_ (in-place) is also matched and produces correct output."""
        torch.manual_seed(7)
        N, n, D = 8192, 8, 4

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Accumulates values in place into out at the index positions
            # and returns out.
            out.index_put_([idx], vals, accumulate=True)
            return out

        out = torch.zeros(n, D, dtype=torch.float32)
        idx = torch.randint(0, 4, (N,), dtype=torch.int64)
        vals = torch.randn(N, D, dtype=torch.float32)

        self._check_accuracy(f, (out, idx, vals), atol=1.0, rtol=1e-2)

    def test_skip_accumulate_false(self):
        # [FUNCTION DOC] An index_put without accumulation is not matched: results are exact and
        # the applied counter stays zero.
        """index_put with accumulate=False doesn't match the registered patterns."""
        n = 256

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Returns the output with values written (not accumulated) at
            # the index.
            return out.index_put([idx], vals, accumulate=False)

        torch.manual_seed(10)
        out = torch.zeros(n, dtype=torch.float32)
        idx = torch.randperm(n, dtype=torch.int64)
        vals = torch.randn(n, dtype=torch.float32)

        self._check_accuracy(f, (out, idx, vals), exact=True)
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)

    def test_pass_disabled(self):
        # [FUNCTION DOC] With the enable flag turned off, compiles an accumulating index_put and
        # checks the result is correct and the applied counter is zero.
        """When partitioned_scatter_enabled=False the pass is a no-op."""
        config.partitioned_scatter_enabled = False

        N, n = 8192, 8

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Returns the output with values accumulated at the index
            # positions.
            return out.index_put([idx], vals, accumulate=True)

        out = torch.zeros(n, dtype=torch.float32)
        idx = torch.randint(0, 4, (N,), dtype=torch.int64)
        vals = torch.randn(N, dtype=torch.float32)

        with torch.no_grad():
            expected = f(out, idx, vals)
            actual = torch.compile(f, backend="inductor", fullgraph=True)(
                out, idx, vals
            )
        self.assertTrue(same(expected, actual, tol=1e-3))
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)

    def test_force_mode_bypasses_heuristic_gates(self):
        # [FUNCTION DOC] Compiles two accumulating scatters, one with too few indices and one
        # with low contention. Without force both are skipped (applied zero, a skip reason
        # counter positive); with the force flag set temporarily both are applied (count two)
        # and outputs match eager.
        """
        partitioned_scatter_force=True bypasses the min_index_size and
        min_contention_ratio gates while still enforcing correctness constraints.
        """

        def f(out_a, idx_a, vals_a, out_b, idx_b, vals_b):
            # [FUNCTION DOC] HELPER: Applies two independent accumulating index_puts and returns
            # both results.
            out_a = out_a.index_put([idx_a], vals_a, accumulate=True)
            out_b = out_b.index_put([idx_b], vals_b, accumulate=True)
            return out_a, out_b

        torch.manual_seed(15)
        # Op A: index_size=100 < min_index_size → normally skipped by gate 6
        out_a = torch.zeros(10, dtype=torch.float32)
        idx_a = torch.randint(0, 5, (100,), dtype=torch.int64)
        vals_a = torch.randn(100, dtype=torch.float32)

        # Op B: contention_ratio=0.25 < min_contention_ratio → normally skipped by gate 7
        out_b = torch.zeros(16384, dtype=torch.float32)
        idx_b = torch.randint(0, 16384, (4096,), dtype=torch.int64)
        vals_b = torch.randn(4096, dtype=torch.float32)

        args = (out_a, idx_a, vals_a, out_b, idx_b, vals_b)

        with torch.no_grad():
            expected = f(*args)
            torch.compile(f, backend="inductor", fullgraph=True)(*args)
        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_index_too_small"]
            + counters["inductor"]["partitioned_scatter_skipped_low_contention"],
            0,
        )

        counters.clear()
        torch._dynamo.reset()
        saved_force = config.partitioned_scatter_force
        try:
            config.partitioned_scatter_force = True
            with torch.no_grad():
                forced = torch.compile(f, backend="inductor", fullgraph=True)(*args)
        finally:
            config.partitioned_scatter_force = saved_force

        self.assertEqual(
            counters["inductor"]["partitioned_scatter_applied"],
            2,
            "force mode must apply to both ops regardless of heuristic gates",
        )
        for e, a in zip(expected, forced):
            self.assertTrue(
                torch.allclose(e, a, atol=1e-1, rtol=1e-2),
                f"force mode output mismatch: expected={e[:5]} actual={a[:5]}",
            )

    def test_skip_cpu_device(self):
        # [FUNCTION DOC] With force enabled, a CPU scatter is still not rewritten: applied
        # counter is zero, the CPU device skip counter is positive, and results match eager.
        """CPU scatters are never rewritten, even with force enabled."""
        torch.manual_seed(21)
        N, n = 8192, 8

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Returns the output with values accumulated at the index
            # positions.
            return out.index_put([idx], vals, accumulate=True)

        out = torch.zeros(n, dtype=torch.float32, device="cpu")
        idx = torch.randint(0, 4, (N,), dtype=torch.int64, device="cpu")
        vals = torch.randn(N, dtype=torch.float32, device="cpu")

        config.partitioned_scatter_force = True
        with torch.no_grad():
            expected = f(out, idx, vals)
            actual = torch.compile(f, backend="inductor", fullgraph=True)(
                out, idx, vals
            )

        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 0)
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_cpu_device"],
            0,
            "CPU scatter should be skipped by the device gate",
        )
        self.assertTrue(torch.allclose(expected, actual, atol=1e-1, rtol=1e-2))

    def test_compute_num_partitions_tight_budget(self):
        # [FUNCTION DOC] Calls the partition count helper with a 20 MB budget for a one-million
        # element float32 output: eight partitions would need more extra memory than available,
        # so the result is four.
        """Memory constraint picks P=4: P=8 overhead (28 MB) exceeds 20 MB budget."""
        # P=4: overhead = 1M * 4 * 3 = 12 MB ≤ 20 MB
        # P=8: overhead = 1M * 4 * 7 = 28 MB > 20 MB
        result = _compute_num_partitions(20_000_000, 1_000_000, 4, min_p=2, max_p=128)
        self.assertEqual(result, 4)

    def test_compute_num_partitions_diminishing_returns_cap(self):
        # [FUNCTION DOC] With effectively unlimited memory, checks the partition count helper's
        # cap that depends on writes per slot: 128 (limited by the maximum), 16, and the minimum
        # of two for sparse writes.
        """Diminishing-returns cap limits P when memory is not the bottleneck."""
        available = 10**12  # effectively unlimited

        # writes_per_slot=64 → cap=256, min(256, max_p=128) = 128
        result = _compute_num_partitions(
            available,
            1024,
            4,
            min_p=2,
            max_p=128,
            index_size=1024,
            scatter_dim_size=16,
        )
        self.assertEqual(result, 128)

        # writes_per_slot=4 → cap=16
        result = _compute_num_partitions(
            available,
            1024,
            4,
            min_p=2,
            max_p=128,
            index_size=64,
            scatter_dim_size=16,
        )
        self.assertEqual(result, 16)

        # writes_per_slot=0.5 → cap=max(2, 2)=2
        result = _compute_num_partitions(
            available,
            1024,
            4,
            min_p=2,
            max_p=128,
            index_size=8,
            scatter_dim_size=16,
        )
        self.assertEqual(result, 2)

    def test_skip_low_contention_ratio_multidim(self):
        # [FUNCTION DOC] Regression test that the contention gate divides index size by the
        # scatter dim size rather than by total output elements, so a vocab by dim output with
        # many indices still gets the pass applied; also checks output accuracy.
        """
        Contention gate uses index_size / scatter_dim_size, not index_size / output_numel.

        For [vocab=4096, dim=64] with N=50000 indices:
          wrong: 50000 / (4096*64) = 0.19 → skip
          right: 50000 / 4096 = 12.2 → apply
        """
        vocab, dim, N = 4096, 64, 50_000

        def f(out, idx, vals):
            # [FUNCTION DOC] HELPER: Returns the output with rows accumulated at the index
            # positions.
            return out.index_put([idx], vals, accumulate=True)

        out = torch.zeros(vocab, dim, dtype=torch.float32)
        idx = torch.randint(0, vocab, (N,), dtype=torch.int64)
        vals = torch.randn(N, dim, dtype=torch.float32)

        with torch.no_grad():
            expected = f(out, idx, vals)
            actual = torch.compile(f, backend="inductor", fullgraph=True)(
                out.clone(), idx, vals
            )

        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-1, rtol=1e-2),
            f"multidim scatter mismatch: {expected[:3]} vs {actual[:3]}",
        )
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_applied"],
            0,
            "pass skipped for [vocab, dim] output — contention gate likely still uses "
            "output_numel instead of scatter_dim_size",
        )

    @unittest.skipUnless(HAS_GPU, "requires GPU for CUDA-aware memory tracking")
    @config.patch(partitioned_scatter_min_contention_ratio=1.0)
    def test_memory_aware_partition_count(self):
        # [FUNCTION DOC] GPU test (contention ratio pinned to one) of the memory aware partition
        # count. First sets the non-model memory floor so headroom allows four partitions and
        # expects the pass to apply with correct output; then sets the floor to total GPU memory
        # and expects it to be skipped for the memory budget with a still-correct result. The
        # floor option is restored.
        """
        Verify that live tensor memory constrains num_partitions.

        The contention gate is pinned low so this exercises the memory budget
        in isolation (this graph's ratio is 11M/10M = 1.1).

        Graph: out(40 MB) + idx(88 MB) + vals(44 MB) + persistent(400 MB) inputs.
        The budget is sized off the peak at the index_put *allocation* point, where
        all four inputs plus the 40 MB output are live (612 MB) — idx/vals are only
        freed once the node completes. Headroom is derived from those sizes rather
        than hardcoded so it tracks any change to them.

        Sub-test 1: headroom fits P-1 extra output buffers → P=4 applied.
        Sub-test 2: floor = total_gpu → no headroom → pass skips.
        """
        torch.manual_seed(12)
        N = 11_000_000
        output_size = 10_000_000
        persist_n = 100_000_000

        def f(out, idx, vals, persistent):
            # [FUNCTION DOC] HELPER: Accumulating index_put into the output followed by adding
            # the sum of a large persistent tensor; returns the result.
            scattered = out.index_put([idx], vals, accumulate=True)
            return scattered + persistent.sum()

        out = torch.zeros(output_size, dtype=torch.float32, device=GPU_TYPE)
        idx = torch.randint(0, output_size, (N,), dtype=torch.int64, device=GPU_TYPE)
        vals = torch.randn(N, dtype=torch.float32, device=GPU_TYPE)
        persistent = torch.randn(persist_n, dtype=torch.float32, device=GPU_TYPE)

        with torch.no_grad():
            expected = f(out, idx, vals, persistent)

        _, total_gpu = torch.cuda.mem_get_info()

        out_bytes = output_size * 4
        baseline_peak = out_bytes + N * 8 + N * 4 + persist_n * 4 + out_bytes
        target_p = 4
        headroom = baseline_peak + (target_p - 1) * out_bytes

        saved_floor = config.partitioned_scatter_non_model_floor_bytes
        try:
            config.partitioned_scatter_non_model_floor_bytes = total_gpu - headroom
            with torch.no_grad():
                actual = torch.compile(f, backend="inductor", fullgraph=True)(
                    out, idx, vals, persistent
                )
        finally:
            config.partitioned_scatter_non_model_floor_bytes = saved_floor

        self.assertGreater(
            counters["inductor"]["partitioned_scatter_applied"],
            0,
            f"real memory pressure: pass should apply (headroom {headroom} bytes "
            f"leaves {(target_p - 1) * out_bytes} bytes for P={target_p})",
        )
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
            f"real memory pressure: output mismatch "
            f"expected[:5]={expected[:5]}\nactual[:5]={actual[:5]}",
        )

        counters.clear()
        torch._dynamo.reset()

        saved_floor = config.partitioned_scatter_non_model_floor_bytes
        try:
            config.partitioned_scatter_non_model_floor_bytes = total_gpu
            with torch.no_grad():
                actual_skipped = torch.compile(f, backend="inductor", fullgraph=True)(
                    out, idx, vals, persistent
                )
        finally:
            config.partitioned_scatter_non_model_floor_bytes = saved_floor

        self.assertEqual(
            counters["inductor"]["partitioned_scatter_applied"],
            0,
            "floor=total_gpu: no headroom, pass must skip",
        )
        self.assertGreater(
            counters["inductor"]["partitioned_scatter_skipped_memory_budget"],
            0,
        )
        self.assertTrue(
            torch.allclose(expected, actual_skipped, atol=1e-2, rtol=1e-2),
            "floor=total_gpu: output should still be correct (pass gracefully skips)",
        )

    @unittest.skipUnless(HAS_GPU, "requires GPU for CUDA-aware memory tracking")
    @config.patch(partitioned_scatter_min_contention_ratio=1.0)
    def test_memory_budget_shared_across_scatters(self):
        # [FUNCTION DOC] GPU test that several scatters in one graph share one memory budget:
        # with headroom for only one extra buffer, exactly one of three scatters may be
        # rewritten and outputs must match eager. The floor option is restored afterwards.
        """
        Several scatters in one graph must share the budget, not each claim it.

        The memory profile describes the untransformed graph, so a candidate
        evaluated after an earlier rewrite has to be charged for that scatter's
        expanded buffer — otherwise N scatters each commit the full budget.

        Three 20 MB scatters here share room for one extra buffer, so exactly
        one may be rewritten.
        """
        torch.manual_seed(13)
        S = 5_000_000
        N = 20_000_000

        def f(out0, out1, out2, idx, vals):
            # [FUNCTION DOC] HELPER: Applies three accumulating index_puts to separate outputs
            # with shared index and values; returns the three results.
            a = out0.index_put([idx], vals, accumulate=True)
            b = out1.index_put([idx], vals, accumulate=True)
            c = out2.index_put([idx], vals, accumulate=True)
            return a, b, c

        outs = [torch.zeros(S, dtype=torch.float32, device=GPU_TYPE) for _ in range(3)]
        idx = torch.randint(0, S, (N,), dtype=torch.int64, device=GPU_TYPE)
        vals = torch.randn(N, dtype=torch.float32, device=GPU_TYPE)
        args = (*outs, idx, vals)

        with torch.no_grad():
            expected = f(*args)

        _, total_gpu = torch.cuda.mem_get_info()

        # Each out dies right after its own scatter, so the peak at every
        # index_put is the three inputs plus that node's 20 MB output.
        out_bytes = S * 4
        baseline_peak = 3 * out_bytes + N * 8 + N * 4 + out_bytes
        # Room for a single P=2 buffer: one scatter fits, the other two must not.
        headroom = baseline_peak + out_bytes

        saved_floor = config.partitioned_scatter_non_model_floor_bytes
        try:
            config.partitioned_scatter_non_model_floor_bytes = total_gpu - headroom
            with torch.no_grad():
                actual = torch.compile(f, backend="inductor", fullgraph=True)(*args)
        finally:
            config.partitioned_scatter_non_model_floor_bytes = saved_floor

        applied = counters["inductor"]["partitioned_scatter_applied"]
        self.assertEqual(
            applied,
            1,
            f"budget fits one expanded buffer but {applied} scatters were "
            f"rewritten: the per-invocation budget is not shared",
        )
        for i, (e, a) in enumerate(zip(expected, actual)):
            self.assertTrue(
                torch.allclose(e, a, atol=1e-2, rtol=1e-2),
                f"output {i} mismatch: expected[:5]={e[:5]}\nactual[:5]={a[:5]}",
            )

    @unittest.skipUnless(HAS_GPU, "requires GPU")
    def test_perf_atomic_contention(self):
        # [FUNCTION DOC] GPU performance test: times three accumulating index_puts with the pass
        # disabled and enabled, prints the speedup and applied count, requires the pass to apply
        # and the speedup to exceed a 1.5 floor. Changes the enable flag and resets dynamo;
        # leaves it enabled.
        """
        Reproduce the PR benchmark: three index_put(accumulate=True) ops with
        moderate contention (8 slots). CI on ROCm MI300X delivers ≈1.8×, so the
        1.5× floor catches regressions with enough slack for hardware variance.
        """
        torch.manual_seed(42)
        N, D, n = 1_000_000, 100, 501
        MIN_SPEEDUP = 1.5

        def scatter_fn(out0, out1, out2, idx, vals):
            # [FUNCTION DOC] HELPER: Applies three accumulating index_puts and returns the three
            # results.
            out0 = out0.index_put([idx], vals, accumulate=True)
            out1 = out1.index_put([idx], vals, accumulate=True)
            out2 = out2.index_put([idx], vals, accumulate=True)
            return out0, out1, out2

        vals = torch.randn(N, D, dtype=torch.float32, device=GPU_TYPE)
        out0 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
        out1 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
        out2 = torch.zeros(n, D, dtype=torch.float32, device=GPU_TYPE)
        idx = torch.randint(0, 8, (N,), dtype=torch.int64, device=GPU_TYPE)
        inputs = (out0, out1, out2, idx, vals)

        config.partitioned_scatter_enabled = False
        torch._dynamo.reset()
        baseline_fn = torch.compile(scatter_fn, backend="inductor", fullgraph=True)
        with torch.no_grad():
            for _ in range(3):
                baseline_fn(*inputs)
        baseline_ms = benchmarker.benchmark_gpu(lambda: baseline_fn(*inputs))

        config.partitioned_scatter_enabled = True
        torch._dynamo.reset()
        counters.clear()
        partitioned_fn = torch.compile(scatter_fn, backend="inductor", fullgraph=True)
        with torch.no_grad():
            for _ in range(3):
                partitioned_fn(*inputs)
        partitioned_ms = benchmarker.benchmark_gpu(lambda: partitioned_fn(*inputs))

        speedup = baseline_ms / partitioned_ms
        n_applied = counters["inductor"]["partitioned_scatter_applied"]

        print(
            f"\nbaseline={baseline_ms:.3f} ms  "
            f"partitioned={partitioned_ms:.3f} ms  "
            f"speedup={speedup:.2f}×  applied={n_applied}"
        )

        self.assertGreater(
            n_applied,
            0,
            "pass did not apply to any op — check gates/config",
        )
        self.assertGreater(
            speedup,
            MIN_SPEEDUP,
            f"expected ≥{MIN_SPEEDUP:.1f}× speedup (moderate contention, 8 slots), "
            f"got {speedup:.2f}×  "
            f"(baseline={baseline_ms:.3f} ms, partitioned={partitioned_ms:.3f} ms)",
        )

        config.partitioned_scatter_enabled = True


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
