# Owner(s): ["module: unknown"]
# [MODULE DOC] Memory-leak regression tests for the OpenMP-backed parallel-for path on CPU.
# Runs a one-by-one max pool network repeatedly and checks that process resident memory
# (read through psutil) does not grow strictly across the last measurements.
# Import time: tries to import psutil (sets a flag when missing), creates a CPU device
# object. Test class attributes also build a random input tensor and a model at class
# definition. Executes the test runner when run as a script.

import collections
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


device = torch.device("cpu")


class Network(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny module whose only layer is a max pool with window and stride 1,
    # held as a class attribute. Used to exercise the parallel-for pooling kernel.
    maxp1 = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the class-level max pool layer to the input tensor
        # and returns the pooled tensor (same shape, since window and stride are 1).
        return self.maxp1(x)


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
class TestOpenMP_ParallelFor(TestCase):
    # [CLASS DOC] Leak checks for OpenMP parallel-for. Skipped entirely when psutil is
    # missing. Class attributes hold a random CPU input batch (20x1x80x80) and a shared
    # Network model. Helpers func and func_rss measure resident memory; the two tests
    # differ only in thread count.
    batch = 20
    channels = 1
    side_dim = 80
    x = torch.randn([batch, channels, side_dim, side_dim], device=device)
    model = Network()

    def func(self, runs):
        # [FUNCTION DOC] HELPER: Runs the model on the shared input for ten rounds, each of
        # runs forward passes, recording the process resident set size after every round.
        # Returns a deque holding only the last five readings (the first five are warm-up).
        p = psutil.Process()
        # warm up for 5 runs, then things should be stable for the last 5
        last_rss = collections.deque(maxlen=5)
        for _ in range(10):
            for _ in range(runs):
                self.model(self.x)
            last_rss.append(p.memory_info().rss)
        return last_rss

    def func_rss(self, runs):
        # [FUNCTION DOC] HELPER: Collects the last five memory readings from func and
        # fails the test (with the readings in the message) if every reading is strictly
        # larger than the one before it, which indicates a leak.
        last_rss = list(self.func(runs))
        # Check that the sequence is not strictly increasing
        is_increasing = True
        for idx in range(len(last_rss)):
            if idx == 0:
                continue
            is_increasing = is_increasing and (last_rss[idx] > last_rss[idx - 1])
        self.assertTrue(
            not is_increasing,
            msg=lambda msg: f"{msg}\nmemory usage is increasing, {str(last_rss)}",
        )

    def test_one_thread(self):
        # [FUNCTION DOC] Sets the global torch thread count to 1 (side effect, not
        # restored), then runs the memory-growth check with 300 passes per round.
        # Regression test for issue 32284.
        """Make sure there is no memory leak with one thread: issue gh-32284"""
        torch.set_num_threads(1)
        self.func_rss(300)

    def test_n_threads(self):
        # [FUNCTION DOC] Sets the global torch thread count to the smaller of five and the
        # physical core count (side effect, not restored), then runs the memory-growth
        # check with 300 passes per round.
        """Make sure there is no memory leak with many threads"""
        ncores = min(5, psutil.cpu_count(logical=False))
        torch.set_num_threads(ncores)
        self.func_rss(300)


if __name__ == "__main__":
    run_tests()
