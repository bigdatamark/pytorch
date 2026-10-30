# Owner(s): ["module: autograd"]

# [MODULE DOC] Smoke test that runs the functional autograd benchmark script in a subprocess for a
# few models and checks that it exits successfully and writes output. Import time reads an
# environment variable to detect coverage collection.
import os

import subprocess
import unittest

from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    slowTest,
    TemporaryFileName,
    TestCase,
)

PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))


# This is a very simple smoke test for the functional autograd benchmarking script.
class TestFunctionalAutogradBenchmark(TestCase):
    # [CLASS DOC] Test case that launches the benchmark script as a child process through a shared
    # helper; it keeps no state of its own. One test runs the quick models and a slow-marked test
    # runs the large ones.
    def _test_runner(self, model, disable_gpu=False):
        # [FUNCTION DOC] HELPER: Runs the benchmark script with a temporary output file, zero
        # timed iterations (warmup only), only the vjp task and only the given model name;
        # optionally passes a negative gpu index to disable the GPU, and uses the python
        # executable name on Windows. Asserts the exit code is zero and the output file is not
        # empty.
        # Note about windows:
        # The temporary file is exclusively open by this process and the child process
        # is not allowed to open it again. As this is a simple smoke test, we choose for now
        # not to run this on windows and keep the code here simple.
        with TemporaryFileName() as out_file:
            cmd = [
                "python3",
                "../benchmarks/functional_autograd_benchmark/functional_autograd_benchmark.py",
            ]
            if IS_WINDOWS:
                cmd[0] = "python"
            # Only run the warmup
            cmd += ["--num-iters", "0"]
            # Only run the vjp task (fastest one)
            cmd += ["--task-filter", "vjp"]
            # Only run the specified model
            cmd += ["--model-filter", model]
            # Output file
            cmd += ["--output", out_file]
            if disable_gpu:
                cmd += ["--gpu", "-1"]

            res = subprocess.run(cmd, check=False)

            self.assertTrue(res.returncode == 0)
            # Check that something was written to the file
            self.assertTrue(os.stat(out_file).st_size > 0)

    @unittest.skipIf(
        PYTORCH_COLLECT_COVERAGE,
        "Can deadlocks with gcov, see https://github.com/pytorch/pytorch/issues/49656",
    )
    def test_fast_tasks(self):
        # [FUNCTION DOC] Runs the helper for each of the fast models in turn (resnet18, the two
        # ppl regression models, wav2letter, transformer, multiheadattn). Skipped when coverage
        # collection is enabled because of a known deadlock with gcov.
        fast_tasks = [
            "resnet18",
            "ppl_simple_reg",
            "ppl_robust_reg",
            "wav2letter",
            "transformer",
            "multiheadattn",
        ]

        for task in fast_tasks:
            self._test_runner(task)

    @slowTest
    @unittest.skipIf(
        IS_WINDOWS,
        "NamedTemporaryFile on windows does not have all the features we need.",
    )
    def test_slow_tasks(self):
        # [FUNCTION DOC] Slow-test only (and skipped on Windows). Runs the helper for fcn_resnet
        # and detr with the GPU disabled because CI GPUs lack memory; the deepspeech model is
        # deliberately left out as too slow.
        slow_tasks = ["fcn_resnet", "detr"]
        # deepspeech is voluntarily excluded as it takes too long to run without
        # proper tuning of the number of threads it should use.

        for task in slow_tasks:
            # Disable GPU for slow test as the CI GPU don't have enough memory
            self._test_runner(task, disable_gpu=True)


if __name__ == "__main__":
    run_tests()
