# Owner(s): ["module: dynamo"]
# [MODULE DOC] Test for the torch.compile benchmarking utility bench_all. The tabulate
# dependency is probed in a guarded import and sets HAS_TABULATE, which skips the test class
# when absent. Device tests are instantiated for every device except cpu.

import unittest

import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    recover_orig_fp32_precision,
    run_tests,
    TestCase,
)


try:
    import tabulate  # noqa: F401  # type: ignore[import]

    from torch.utils.benchmark.utils.compile import bench_all

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


@unittest.skipIf(not HAS_TABULATE, "tabulate not available")
class TestCompileBenchmarkUtil(TestCase):
    # [CLASS DOC] Device-generic test that runs bench_all on a toy model for inference and
    # training and checks the returned table text. Skipped entirely when tabulate is
    # unavailable.
    # bench_all's _enable/_disable_tensor_cores restore via the legacy
    # set_float32_matmul_precision, which can't reproduce the "none" default
    # of the per-backend matmul flags.
    @recover_orig_fp32_precision
    def test_training_and_inference(self, device):
        # [FUNCTION DOC] Resets dynamo, builds a toy model on the device, and benchmarks it with
        # 5 iterations: the inference table must contain the Inference and Eager labels and a
        # dash; with an SGD optimizer the training table must contain Train, Eager and a dash.
        # The decorator restores the original float32 matmul precision settings afterwards.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module with one uninitialized 2x2 parameter, multiplied
            # elementwise with the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the weight parameter from an uninitialized 2x2
                # tensor.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.Tensor(2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied elementwise by the weight.
                return x * self.weight

        torchdynamo.reset()
        model = ToyModel().to(device)

        inference_table = bench_all(model, torch.ones(1024, 2, 2, device=device), 5)
        self.assertTrue(
            "Inference" in inference_table
            and "Eager" in inference_table
            and "-" in inference_table
        )

        training_table = bench_all(
            model,
            torch.ones(1024, 2, 2, device=device),
            5,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        )
        self.assertTrue(
            "Train" in training_table
            and "Eager" in training_table
            and "-" in training_table
        )


instantiate_device_type_tests(TestCompileBenchmarkUtil, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()
