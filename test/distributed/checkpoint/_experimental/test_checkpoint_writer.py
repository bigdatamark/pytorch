# Owner(s): ["oncall: distributed checkpointing"]

# [MODULE DOC] Tests for CheckpointWriter and CheckpointWriterConfig in the experimental
# distributed checkpoint package. Uses a recording WriterHook and a mock barrier to check
# the file layout written for a rank, that the barrier and the pre/post commit hooks are
# invoked with the right arguments, and that both are optional.

import os
import shutil
import tempfile
from typing import Any
from unittest.mock import MagicMock

import torch
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
    WriterHook,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class MockWriterHook(WriterHook):
    # [CLASS DOC] HELPER: WriterHook that records whether and with what path and keyword
    # arguments pre_commit and post_commit were called, for assertions.
    """Mock implementation of WriterHook for testing."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the called flags to false and the recorded
        # paths and keyword arguments to None.
        self.pre_commit_called = False
        self.commit_called = False
        self.pre_commit_path: str | None = None
        self.commit_path: str | None = None
        self.pre_commit_kwargs: dict[str, Any] | None = None
        self.commit_kwargs: dict[str, Any] | None = None

    def pre_commit(self, path: str, **kwargs: Any):
        # [FUNCTION DOC] HELPER: Records that the pre-commit hook ran, with its path and
        # keyword arguments.
        self.pre_commit_called = True
        self.pre_commit_path = path
        self.pre_commit_kwargs = kwargs

    def post_commit(self, path: str, **kwargs: Any):
        # [FUNCTION DOC] HELPER: Records that the post-commit hook ran (stored in the
        # commit_called, commit_path and commit_kwargs fields) with its path and keyword
        # arguments.
        self.commit_called = True
        self.commit_path = path
        self.commit_kwargs = kwargs


class TestCheckpointWriterConfig(TestCase):
    # [CLASS DOC] Tests defaults and overrides of CheckpointWriterConfig.
    def test_default_values(self):
        # [FUNCTION DOC] Checks the default write barrier timeout is 600 seconds.
        """Test that CheckpointWriterConfig has the correct default values."""
        options = CheckpointWriterConfig()
        self.assertEqual(options.write_barrier_timeout_secs, 600)

    def test_custom_values(self):
        # [FUNCTION DOC] Checks an explicit write barrier timeout (300) is stored.
        """Test that CheckpointWriterConfig can be initialized with custom values."""
        options = CheckpointWriterConfig(write_barrier_timeout_secs=300)
        self.assertEqual(options.write_barrier_timeout_secs, 300)


class TestCheckpointWriter(TestCase):
    # [CLASS DOC] Test case for CheckpointWriter. Fixtures: temp directory, rank 0 of 1,
    # default config, a mock barrier, a recording hook, a writer wired to them and a small
    # state dict.
    def setUp(self):
        # [FUNCTION DOC] Creates the temp directory, rank info, config, mock barrier,
        # recording hook, the writer using them, and the sample state dict.
        super().setUp()
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test objects
        self.rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
        )
        self.options = CheckpointWriterConfig()
        self.mock_barrier = MagicMock()
        self.mock_hook = MockWriterHook()

        # Create the checkpoint writer
        self.writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=self.mock_barrier,
            commit_hook=self.mock_hook,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self):
        # [FUNCTION DOC] Recursively deletes the temporary directory. Does not call the base
        # class teardown.
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_write_creates_checkpoint_file(self):
        # [FUNCTION DOC] Writes the state dict to a new path and checks the per-rank file is
        # created inside it and loads back with the saved model, optimizer, epoch and step.
        """Test that write creates a checkpoint file with the correct content."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write
        self.writer.write(checkpoint_path, self.state_dict)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

        # Load the checkpoint and verify its contents
        loaded_state_dict = torch.load(expected_file_path)
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertEqual(loaded_state_dict["epoch"], 5)
        self.assertEqual(loaded_state_dict["step"], 1000)

    def test_write_calls_barrier(self):
        # [FUNCTION DOC] Checks one write triggers exactly one barrier execution.
        """Test that write calls the barrier with the correct parameters."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write
        self.writer.write(checkpoint_path, self.state_dict)

        # Verify that the barrier was called
        self.mock_barrier.execute_barrier.assert_called_once()

    def test_write_calls_commit_hooks(self):
        # [FUNCTION DOC] Writes with an extra keyword argument and checks both the pre
        # and post commit hooks ran with the checkpoint path and received that keyword.
        """Test that write calls the commit hooks with the correct parameters."""
        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Call write with additional kwargs
        kwargs = {"extra": "value"}
        self.writer.write(checkpoint_path, self.state_dict, **kwargs)

        # Verify that the pre_commit hook was called with the correct parameters
        self.assertTrue(self.mock_hook.pre_commit_called)
        self.assertEqual(self.mock_hook.pre_commit_path, checkpoint_path)
        self.assertEqual(
            self.mock_hook.pre_commit_kwargs is not None
            and self.mock_hook.pre_commit_kwargs["extra"],
            "value",
        )

        # Verify that the commit hook was called with the correct parameters
        self.assertTrue(self.mock_hook.commit_called)
        self.assertEqual(self.mock_hook.commit_path, checkpoint_path)
        self.assertEqual(
            self.mock_hook.commit_kwargs is not None
            and self.mock_hook.commit_kwargs["extra"],
            "value",
        )

    def test_write_without_barrier(self):
        # [FUNCTION DOC] Builds a writer with no barrier, writes, and checks the per-rank
        # file exists.
        """Test that write works correctly without a barrier."""
        # Create a writer without a barrier
        writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=None,
            commit_hook=self.mock_hook,
        )

        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_barrier")

        # Call write
        writer.write(checkpoint_path, self.state_dict)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

    def test_write_without_commit_hook(self):
        # [FUNCTION DOC] Builds a writer with no commit hook, writes, and checks the file
        # exists and the (shared mock) barrier was still run exactly once.
        """Test that write works correctly without a commit hook."""
        # Create a writer without a commit hook
        writer = CheckpointWriter(
            config=self.options,
            rank_info=self.rank_info,
            barrier=self.mock_barrier,
            commit_hook=None,
        )

        # Set up the checkpoint path
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_hook")

        # Call write
        writer.write(checkpoint_path, self.state_dict)

        # Verify that the checkpoint file exists
        expected_file_path = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(expected_file_path))

        # Verify that the barrier was still called
        self.mock_barrier.execute_barrier.assert_called_once()

    def test_close(self):
        # [FUNCTION DOC] Calls close on the writer and only checks it does not raise.
        """Test that close doesn't raise any exceptions."""
        # This is a no-op in the base class, so just verify it doesn't raise
        self.writer.close()


if __name__ == "__main__":
    run_tests()
