# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor deterministic mode. Checks that toggling the global
# deterministic algorithms switch updates the inductor deterministic config, that deterministic
# mode disables matmul padding benchmarks, max-autotune template autotuning and coordinate-
# descent tuning, that batch-invariant persistent reductions and deterministic cumulative sums
# give bitwise-stable results, that reorder-for-locality keeps random draw order, and that a few
# HuggingFace models give bitwise-identical results across two runs. Defines the repository root
# path at import.
import contextlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    IS_BIG_GPU,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


@instantiate_parametrized_tests
class DeterministicTest(TestCase):
    # [CLASS DOC] Deterministic mode test case with parametrized tests. setUp enters a fresh
    # Inductor cache context that is exited in tearDown so tests do not share compile caches.
    def setUp(self) -> None:
        # [FUNCTION DOC] Runs the base setUp, then creates an exit stack holding a fresh-
        # cache context.
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        # [FUNCTION DOC] Closes the exit stack (leaving the fresh-cache context) and runs
        # base teardown.
        self._exit_stack.close()
        super().tearDown()

    def test_use_deterministic_algorithsm(self):
        # [FUNCTION DOC] Saves the current deterministic algorithms state, then toggles it on,
        # off and on (warn-only mode) and asserts the inductor deterministic config follows each
        # value. Always restores the saved state in a final block.
        old_val = torch.are_deterministic_algorithms_enabled()
        try:
            for new_val in [True, False, True]:
                torch.use_deterministic_algorithms(new_val, warn_only=True)
                self.assertEqual(inductor_config.deterministic, new_val)
        finally:
            torch.use_deterministic_algorithms(old_val, warn_only=True)

    @parametrize("deterministic", [False, True])
    def test_mm_padding(self, deterministic):
        # [FUNCTION DOC] Parametrized by the deterministic flag. Compiles a matrix multiply of
        # two odd sized (2049 square) tensors on the GPU under that config and checks the result
        # against eager. Asserts the matmul-padding benchmark counter is zero in deterministic
        # mode and positive otherwise.
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Compiled function returning the matrix product of its
                # two inputs.
                return x @ y

            inps = [torch.rand([2049, 2049], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @inductor_config.patch(max_autotune=True)
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_max_autotune(self, deterministic):
        # [FUNCTION DOC] Parametrized by the deterministic flag; runs under max-autotune and
        # requires a big GPU. Compiles a matrix multiply of two 2048 square tensors, compares to
        # eager, and asserts the algorithm-selection autotune counter is zero when deterministic
        # and positive otherwise.
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Compiled function returning the matrix product of its
                # two inputs.
                return x @ y

            inps = [torch.rand([2048, 2048], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] == 0)
            else:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] > 0)

    def test_pointwise_coordesc_tuning(self):
        # [FUNCTION DOC] Compiles a pointwise add of one in max-autotune mode on the GPU, checks
        # the result against eager and asserts the coordinate-descent tuning benchmark counter
        # is positive.
        @torch.compile(mode="max-autotune")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one to its input.
            return x + 1

        x = torch.randn(2048, device=GPU_TYPE)
        self.assertEqual(f(x), x + 1)

        self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @parametrize("deterministic", [False, True])
    def test_reduction_coordesc_tuning(self, deterministic):
        # [FUNCTION DOC] Parametrized by the deterministic flag; enables coordinate descent
        # tuning in the config. Compiles a last-dimension sum of a 2048 square GPU tensor,
        # checks it against eager, and asserts the coordinate-descent benchmark counter is zero
        # when deterministic, positive otherwise.
        with inductor_config.patch(
            deterministic=deterministic, coordinate_descent_tuning=True
        ):

            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Compiled function summing its input over the
                # last dimension.
                return x.sum(dim=-1)

            inp = torch.rand([2048, 2048], device=GPU_TYPE)

            out = foo(inp)
            self.assertEqual(out, inp.sum(dim=-1))

            if deterministic:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU + Triton")
    @inductor_config.patch(batch_invariant=True)
    def test_persistent_reduction_batch_invariance(self):
        # [FUNCTION DOC] With the batch-invariant config on and a GPU with triton, compiles a
        # layer norm over bfloat16 data (768 features, 1024 rows). After computing the full
        # result, repeatedly halves the number of rows (resetting dynamo each time to force
        # recompile) and asserts each prefix output is bitwise equal to the matching prefix of
        # the full result.
        H = 768
        FULL = 1024

        def fn(x, w, b):
            # [FUNCTION DOC] HELPER: Applies layer norm over the last dimension with the given
            # weight and bias.
            return torch.nn.functional.layer_norm(x, (H,), weight=w, bias=b)

        torch.manual_seed(0)
        w = torch.randn(H, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(H, device=GPU_TYPE, dtype=torch.bfloat16)
        x_full = torch.randn(FULL, H, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        torch._dynamo.reset()
        out_full = compiled(x_full, w, b)
        self.assertEqual(out_full, fn(x_full, w, b))

        # Halving sweep, matching what the benchmark harness does.
        size = FULL // 2
        while size >= 1:
            torch._dynamo.reset()
            out = compiled(x_full[:size].contiguous(), w, b)
            ref = out_full[:size].contiguous()
            self.assertTrue(
                torch.equal(ref, out),
                lambda msg: f"{msg}\npersistent reduction diverged at size={size} (FULL={FULL})",
            )
            size //= 2

    @unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU + Triton")
    def test_cumsum_deterministic(self):
        # [FUNCTION DOC] Requires GPU and triton. Turns on deterministic algorithms (error mode)
        # and compares a compiled cumulative sum of a large, odd sized float tensor bitwise (via
        # integer views) against eager and across five repeated compiled calls.
        # KNOWN ISSUE: sets the global deterministic algorithms flag and never restores it in
        # this test.
        from torch._inductor.utils import run_and_get_code

        x = torch.randn(1_000_003, device=GPU_TYPE, dtype=torch.float32)
        compiled = torch.compile(lambda v: torch.cumsum(v, dim=0))

        torch.use_deterministic_algorithms(True, warn_only=False)
        eager = torch.cumsum(x, dim=0)
        result, (_,) = run_and_get_code(compiled, x)
        for _ in range(5):
            self.assertEqual(result.view(torch.int32), compiled(x).view(torch.int32))
        self.assertEqual(result.view(torch.int32), eager.view(torch.int32))

    def test_reorder_for_locality_preserves_randint_order(self):
        # [FUNCTION DOC] With random fallback enabled, checks that compiling a function that
        # seeds the generator, draws a four-row random integer tensor and then discards a two-
        # row draw returns the same values as eager execution seeded the same way, i.e.
        # scheduling reorder keeps the order of random draws.
        with inductor_config.patch(fallback_random=True):

            def fn():
                # [FUNCTION DOC] HELPER: Seeds the generator, draws a four-row and a two-row
                # random integer tensor, discarding the second, and returns the first.
                torch.manual_seed(0)
                out = torch.randint(0, 100, (4, 1), dtype=torch.int64)
                _ = torch.randint(0, 100, (2, 1), dtype=torch.int64)
                return out

            compiled = torch.compile(fn, backend="inductor")

            torch.manual_seed(0)
            eager = fn()

            torch.manual_seed(0)
            compiled_out = compiled()

            torch.testing.assert_close(eager, compiled_out)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180681")
    @unittest.skipIf(IS_FBCODE, "Skipping run2run determinism test in fbcode")
    @parametrize("model_name", ["GoogleFnet", "BertForMaskedLM", "DistillGPT2"])
    @parametrize("training_or_inference", ["training", "inference"])
    @parametrize("precision", ["float32", "bfloat16", "float16", "amp"])
    def test_run2run_determinism(self, model_name, training_or_inference, precision):
        # [FUNCTION DOC] Parametrized by model name, training or inference, and precision.
        # Skipped on ROCm and in fbcode. Runs the HuggingFace benchmark script in a subprocess
        # with caches disabled and the deterministic env flag on, saving model outputs to a
        # temporary pickle; then runs it again with benchmarking results distorted (inverse) and
        # a flag comparing outputs with the saved file. Asserts the second run stdout reports
        # the result as bitwise equivalent. The first run result is not checked.
        """
        Test run2run determinism for a few huggingface models.

        The test assumes benchmarks/dynamo/huggingface.py can be found from
        the current working directory.
        """

        def _setup_env(env):
            # [FUNCTION DOC] HELPER: Fills the subprocess environment mapping: disables the
            # autotune, fx-graph remote and fx-graph caches and, if the enclosing determinism
            # toggle is true, sets the deterministic env flag.
            env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"  # disable autotune cache
            env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
            env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
            if enable_determinism:
                env["TORCHINDUCTOR_DETERMINISTIC"] = "1"

        # set to false if you want to check how the test fails without
        # the deterministic mode
        enable_determinism = True
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_pkl = os.path.join(tmpdir, "saved.pkl")
            cmd = (
                f"{sys.executable} {REPO_ROOT}/benchmarks/dynamo/huggingface.py --backend inductor"
                + f" --{precision} --accuracy --only {model_name} --{training_or_inference}"
                + f" --disable-cudagraphs --save-model-outputs-to={saved_pkl}"
            )
            print("Command", cmd)
            env = os.environ.copy()
            _setup_env(env)
            out = subprocess.run(cmd.split(), capture_output=True, env=env)

            # We don't check the accuracy against eager here because some
            # of the combination between model and precision can not
            # pass that accuracy test. But it's still valuable to make
            # sure we generate bitwise equivalent result from run to run.
            # self.assertTrue("pass" in out.stdout.decode())

            cmd = (
                f"{sys.executable} {REPO_ROOT}/benchmarks/dynamo/huggingface.py --backend inductor"
                + f" --{precision} --accuracy --only {model_name} --{training_or_inference}"
                + f" --disable-cudagraphs --compare-model-outputs-with={saved_pkl}"
            )
            print("Command", cmd)

            # distort benchmarking results
            env["TORCHINDUCTOR_DISTORT_BENCHMARKING_RESULT"] = "inverse"
            out = subprocess.run(cmd.split(), capture_output=True, env=env)
            self.assertTrue(
                "The result is bitwise equivalent to the previously saved result"
                in out.stdout.decode(),
                lambda msg: f"{msg}\nstdout: {out.stdout.decode()}, stderr: {out.stderr.decode()}",
            )


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON:
        run_tests()
