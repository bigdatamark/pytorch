# Owner(s): ["oncall: distributed checkpointing"]

# [MODULE DOC] Tests for DefaultStager and CheckpointStagerConfig in the experimental
# distributed checkpoint package: synchronous versus asynchronous (future-returning)
# staging, option combinations, accelerator tensors, resource cleanup and repeated use.
# Import-time side effect: instantiates device-type specific test classes (including XPU)
# into this module's globals; GPU-only tests are skipped without an accelerator.

from concurrent.futures import Future
from unittest import skipIf

import torch
from torch.distributed.checkpoint._experimental.staging import (
    CheckpointStagerConfig,
    DefaultStager,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TEST_ACCELERATOR, TestCase


class TestDefaultStager(TestCase):
    # [CLASS DOC] Device-generic test case for DefaultStager. Each test builds a stager
    # from a CheckpointStagerConfig, stages a state dict (returned directly when sync,
    # as a Future when async) and closes the stager.
    def setUp(self) -> None:
        # [FUNCTION DOC] Builds a sample state dict with a linear layer state dict,
        # optimizer params, epoch, step, a random tensor and a nested mapping.
        super().setUp()
        # Create a test state dictionary with various data types
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
            "tensor": torch.randn(3, 4),
            "nested": {"inner_tensor": torch.ones(2, 2), "inner_value": 42},
        }

    @skipIf(not TEST_ACCELERATOR, reason="requires GPU")
    def test_sync_staging(self) -> None:
        # [FUNCTION DOC] With async staging off, stage returns a dict directly; checks it
        # contains the expected keys and scalar values, then closes the stager.
        """Test synchronous staging."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Stage the state dict
        staged_dict = stager.stage(self.state_dict)

        # Verify that a state dict is returned (not a Future)
        self.assertIsInstance(staged_dict, dict)

        # Verify the staged state dictionary
        self.assertIn("model", staged_dict)
        self.assertIn("optimizer", staged_dict)
        self.assertEqual(staged_dict["epoch"], 5)
        self.assertEqual(staged_dict["step"], 1000)
        self.assertIn("tensor", staged_dict)
        self.assertIn("nested", staged_dict)

        # Clean up
        stager.close()

    @skipIf(not TEST_ACCELERATOR, reason="requires GPU")
    def test_async_staging(self) -> None:
        # [FUNCTION DOC] With async staging on, stage returns a Future; waits for it and
        # checks the resulting dict has the expected keys and scalar values. Closes the
        # stager.
        """Test asynchronous staging."""
        options = CheckpointStagerConfig(use_async_staging=True)
        stager = DefaultStager(options)

        # Stage the state dict
        result = stager.stage(self.state_dict)

        # Verify that a Future is returned
        self.assertIsInstance(result, Future)

        # Wait for the Future to complete
        staged_dict = result.result()

        # Verify the staged state dictionary
        self.assertIn("model", staged_dict)
        self.assertIn("optimizer", staged_dict)
        self.assertEqual(staged_dict["epoch"], 5)
        self.assertEqual(staged_dict["step"], 1000)

        # Clean up
        stager.close()

    def test_cuda_non_blocking_without_cuda(self) -> None:
        # [FUNCTION DOC] Skipped when an accelerator exists. Otherwise requesting
        # non-blocking copy must make DefaultStager construction raise AssertionError.
        """Test that non-blocking copy fails when CUDA is not available."""
        if torch.accelerator.is_available():
            self.skipTest("CUDA is available, cannot test CUDA unavailable scenario")

        options = CheckpointStagerConfig(use_non_blocking_copy=True)
        with self.assertRaises(AssertionError):
            DefaultStager(options)

    def test_different_option_combinations(self) -> None:
        # [FUNCTION DOC] Runs staging under several configs (all off, pinned only, shared
        # memory only, plus async-only and non-blocking-only when an accelerator exists),
        # each in a subtest. Expects a Future for async configs and a dict otherwise, checks
        # the model key is present, and closes each stager.
        """Test various combinations of staging options."""
        test_cases = [
            # All disabled
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=False,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
            # Only pinned memory
            CheckpointStagerConfig(
                use_pinned_memory=True,
                use_shared_memory=False,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
            # Only shared memory
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
        ]

        if torch.accelerator.is_available():
            # Only async staging
            test_cases.append(
                CheckpointStagerConfig(
                    use_pinned_memory=torch.accelerator.is_available(),
                    use_shared_memory=False,
                    use_async_staging=True,
                    use_non_blocking_copy=False,
                )
            )
            # Only CUDA non-blocking copy
            test_cases.append(
                CheckpointStagerConfig(
                    use_pinned_memory=torch.accelerator.is_available(),
                    use_shared_memory=False,
                    use_async_staging=False,
                    use_non_blocking_copy=torch.accelerator.is_available(),
                )
            )

        for options in test_cases:
            with self.subTest(options=options):
                stager = DefaultStager(options)

                # Test staging works with these options
                if options.use_async_staging and torch.accelerator.is_available():
                    result = stager.stage(self.state_dict)
                    self.assertIsInstance(result, Future)
                    staged_dict = result.result()
                else:
                    staged_dict = stager.stage(self.state_dict)

                self.assertIsInstance(staged_dict, dict)
                self.assertIn("model", staged_dict)

                stager.close()

    @skipIf(not TEST_ACCELERATOR, reason="requires GPU")
    def test_cuda_tensors_staging(self) -> None:
        # [FUNCTION DOC] Stages a state dict mixing accelerator and CPU tensors with sync
        # staging; raises AssertionError if the result is not a dict and checks the three
        # top-level keys are present. Does not verify tensors actually ended up on CPU.
        """Test staging with CUDA tensors."""
        # Create state dict with CUDA tensors
        device = torch.accelerator.current_accelerator()
        cuda_state_dict = {
            "cuda_tensor": torch.randn(3, 4).to(device),
            "cpu_tensor": torch.randn(2, 3),
            "mixed_model": {
                "weight": torch.randn(5, 5).to(device),
                "bias": torch.randn(5).to(device),
            },
        }

        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        staged_dict = stager.stage(cuda_state_dict)
        if not isinstance(staged_dict, dict):
            raise AssertionError(f"Expected dict, got {type(staged_dict)}")

        # Verify tensors are staged (should be moved to CPU)
        self.assertIn("cuda_tensor", staged_dict)
        self.assertIn("cpu_tensor", staged_dict)
        self.assertIn("mixed_model", staged_dict)

        stager.close()

    @skipIf(not TEST_ACCELERATOR, reason="requires GPU")
    def test_resource_cleanup(self) -> None:
        # [FUNCTION DOC] Checks the stager's internal state dict stager exists, then closes
        # the stager. Nothing is asserted after close.
        """Test that resources are properly cleaned up."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Verify initial state
        self.assertIsNotNone(stager._state_dict_stager)

        # Close and verify cleanup
        stager.close()

    def test_multiple_staging_operations(self) -> None:
        # [FUNCTION DOC] Reuses one sync stager for three different state dicts (linear,
        # conv, optimizer settings), using pinned memory and non-blocking copy only if an
        # accelerator exists, and checks each result is a dict containing the input keys.
        """Test multiple staging operations with the same stager."""
        options = CheckpointStagerConfig(
            use_async_staging=False,
            use_pinned_memory=torch.accelerator.is_available(),
            use_shared_memory=False,
            use_non_blocking_copy=torch.accelerator.is_available(),
        )
        stager = DefaultStager(options)

        # Stage multiple different state dicts
        state_dicts = [
            {"model1": torch.nn.Linear(5, 3).state_dict()},
            {"model2": torch.nn.Conv2d(3, 16, 3).state_dict()},
            {"optimizer": {"lr": 0.001, "momentum": 0.9}},
        ]

        staged_results = []
        for state_dict in state_dicts:
            staged_dict = stager.stage(state_dict)
            staged_results.append(staged_dict)

        # Verify all staging operations succeeded
        self.assertEqual(len(staged_results), 3)
        for i, result in enumerate(staged_results):
            self.assertIsInstance(result, dict)
            # Verify the result contains the expected keys
            for key in state_dicts[i]:
                self.assertIn(key, result)

        stager.close()


instantiate_device_type_tests(TestDefaultStager, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
