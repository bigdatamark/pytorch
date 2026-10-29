# Owner(s): ["oncall: distributed checkpointing"]

# [MODULE DOC] Tests for the experimental checkpoint builder factory functions
# make_sync_checkpointer and make_async_checkpointer: they check the returned checkpointer
# type and a real save/load round trip on a temporary directory (single rank, no barrier).

import os
import shutil
import tempfile

import torch
from torch.distributed.checkpoint._experimental.barriers import BarrierConfig
from torch.distributed.checkpoint._experimental.builder import (
    make_async_checkpointer,
    make_sync_checkpointer,
)
from torch.distributed.checkpoint._experimental.checkpointer import (
    AsyncCheckpointer,
    SyncCheckpointer,
)
from torch.distributed.checkpoint._experimental.config import CheckpointerConfig
from torch.distributed.checkpoint._experimental.staging import CheckpointStagerConfig
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMakeCheckpointer(TestCase):
    # [CLASS DOC] Test case for the checkpointer factories. State per test: a temporary
    # directory, a single-rank RankInfo and a small state dict (a linear layer state dict,
    # optimizer param groups, epoch 5, step 1000).
    def setUp(self) -> None:
        # [FUNCTION DOC] Creates the temp directory, the rank info (world size 1, rank 0)
        # and the sample state dict used by every test.
        super().setUp()
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create real objects for testing
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def tearDown(self) -> None:
        # [FUNCTION DOC] Recursively deletes the temporary directory. Does not call the base
        # class teardown.
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_make_sync_checkpointer(self) -> None:
        # [FUNCTION DOC] Builds a sync checkpointer with barriers disabled, checks the type,
        # saves (returns None in sync mode), checks the per-rank checkpoint file exists, and
        # loads back verifying the epoch value.
        """Test creating a synchronous checkpointer using make_sync_checkpointer."""

        # Create sync checkpointer using factory function with no barrier
        config = CheckpointerConfig(barrier_config=BarrierConfig(barrier_type=None))
        checkpointer = make_sync_checkpointer(config=config, rank_info=self.rank_info)

        # Verify it's a SyncCheckpointer instance
        self.assertIsInstance(checkpointer, SyncCheckpointer)

        # Test that it works for sync operations
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_factory_sync")
        result = checkpointer.save(checkpoint_path, self.state_dict)
        self.assertIsNone(result)  # Sync mode returns None

        # Verify checkpoint was created
        checkpoint_file = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

        # Test loading
        loaded_state_dict = checkpointer.load(checkpoint_path)
        self.assertEqual(loaded_state_dict["epoch"], 5)

    def test_make_sync_checkpointer_with_config_first(self) -> None:
        # [FUNCTION DOC] Same flow as the basic sync test (type, save returns None, file
        # exists) using a different directory; no load check.
        # KNOWN ISSUE: despite the name, nothing here tests argument order, since both
        # arguments are passed by keyword; it duplicates the basic test.
        """Test creating a synchronous checkpointer with config as first parameter."""
        # Create sync checkpointer with config as first parameter
        config = CheckpointerConfig(barrier_config=BarrierConfig(barrier_type=None))
        checkpointer = make_sync_checkpointer(config=config, rank_info=self.rank_info)

        # Verify it's a SyncCheckpointer instance
        self.assertIsInstance(checkpointer, SyncCheckpointer)

        # Test that it works for sync operations
        checkpoint_path = os.path.join(
            self.temp_dir, "checkpoint_factory_sync_config_first"
        )
        result = checkpointer.save(checkpoint_path, self.state_dict)
        self.assertIsNone(result)  # Sync mode returns None

        # Verify checkpoint was created
        checkpoint_file = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

    def test_make_sync_checkpointer_with_custom_config(self) -> None:
        # [FUNCTION DOC] Builds a sync checkpointer passing arguments in the opposite
        # keyword order with a no-barrier config; checks type, save result, file existence
        # and a load round trip. Largely duplicates the basic sync test.
        """Test creating a synchronous checkpointer with a custom config."""
        # Create a custom config with no barrier
        config = CheckpointerConfig(barrier_config=BarrierConfig(barrier_type=None))

        # Create sync checkpointer with the custom config
        checkpointer = make_sync_checkpointer(rank_info=self.rank_info, config=config)

        # Verify it's a SyncCheckpointer instance
        self.assertIsInstance(checkpointer, SyncCheckpointer)

        # Test that it works for sync operations
        checkpoint_path = os.path.join(
            self.temp_dir, "checkpoint_factory_sync_custom_config"
        )
        result = checkpointer.save(checkpoint_path, self.state_dict)
        self.assertIsNone(result)  # Sync mode returns None

        # Verify checkpoint was created
        checkpoint_file = os.path.join(
            checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(checkpoint_file))

        # Test loading
        loaded_state_dict = checkpointer.load(checkpoint_path)
        self.assertEqual(loaded_state_dict["epoch"], 5)

    def test_make_async_checkpointer(self) -> None:
        # [FUNCTION DOC] Builds an async checkpointer whose staging uses non-blocking copy
        # and pinned memory only when an accelerator is available. Saves, expecting stage
        # and write futures, waits on both, checks the checkpoint file and load round trip,
        # and always closes the checkpointer in a finally block.
        """Test creating an asynchronous checkpointer using make_async_checkpointer."""
        # Create async checkpointer using factory function with default parameters
        config: CheckpointerConfig = CheckpointerConfig()
        config.staging_config = CheckpointStagerConfig(
            use_non_blocking_copy=torch.accelerator.is_available(),
            use_pinned_memory=torch.accelerator.is_available(),
        )
        checkpointer = make_async_checkpointer(config=config, rank_info=self.rank_info)

        try:
            # Verify it's an AsyncCheckpointer instance
            self.assertIsInstance(checkpointer, AsyncCheckpointer)

            # Test that it works for async operations
            checkpoint_path = os.path.join(self.temp_dir, "checkpoint_factory_async")
            stage_future, write_future = checkpointer.save(
                checkpoint_path, self.state_dict
            )

            # Verify futures are returned
            self.assertIsNotNone(stage_future)
            self.assertIsNotNone(write_future)

            # Wait for completion
            stage_future.result()
            write_future.result()

            # Verify checkpoint was created
            checkpoint_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(checkpoint_file))

            # Test loading
            loaded_state_dict = checkpointer.load(checkpoint_path)
            self.assertEqual(loaded_state_dict["epoch"], 5)

        finally:
            # Clean up
            checkpointer.close()


if __name__ == "__main__":
    run_tests()
