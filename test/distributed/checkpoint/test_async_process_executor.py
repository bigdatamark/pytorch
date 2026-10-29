# Owner(s): ["oncall: distributed checkpointing"]

# [MODULE DOC] Tests for the process-based async checkpoint executor
# (_ProcessBasedAsyncCheckpointExecutor) and its _ProcessGroupInitInfo settings: failure in
# background process group creation, continued serving after a failed save, use of a prefix
# store via DCP_USE_PREFIX_STORE, and GC-related environment flags. Defines a configurable
# stub StorageWriter. Import-time side effect: exits the process (status 0) under dev-asan.

import os
import sys
from unittest.mock import patch

import torch
import torch.testing._internal.common_utils as common
from torch import distributed as dist
from torch.distributed.checkpoint._async_process_executor import (
    _ProcessBasedAsyncCheckpointExecutor,
    _ProcessGroupInitInfo,
)
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.testing._internal.common_distributed import skip_if_win32
from torch.testing._internal.common_utils import (
    retry_on_connect_failures,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestStorageWriter(StorageWriter):
    # [CLASS DOC] HELPER: Minimal in-memory StorageWriter stub. Tracks a behavior string
    # ("success" or "fail_once") and a call counter; write_data raises on the first call
    # when behavior is fail_once, otherwise returns a completed future. All other hooks are
    # no-ops. Not collected as a test despite the Test prefix.
    """Unified test storage writer with configurable behaviors."""

    def __init__(
        self,
        behavior="success",
    ):
        # [FUNCTION DOC] HELPER: Stores the behavior string and resets the call counter to 0.
        """
        Create a test storage writer with specified behavior.

        Args:
            behavior: "success", "fail_once"
        """
        self.behavior = behavior
        self.call_count = 0

    def _should_fail(self):
        # [FUNCTION DOC] HELPER: Increments the call counter and returns True only when the
        # behavior is fail_once and this is the first call; False otherwise.
        """Determine if this call should fail based on the configured behavior."""
        self.call_count += 1

        if self.behavior == "success":
            return False
        elif self.behavior == "fail_once":
            return self.call_count == 1

        return False

    # Implement all required StorageWriter methods directly
    def reset(self, checkpoint_id=None):
        # [FUNCTION DOC] HELPER: No-op reset hook; ignores checkpoint_id.
        """Reset for new checkpoint."""

    def set_up_storage_writer(self, is_coordinator):
        # [FUNCTION DOC] HELPER: No-op setup hook.
        """Set up storage writer."""

    def prepare_local_plan(self, plan):
        # [FUNCTION DOC] HELPER: Returns the given local plan unchanged.
        """Prepare local plan."""
        return plan

    def prepare_global_plan(self, plans):
        # [FUNCTION DOC] HELPER: Returns the given list of plans unchanged.
        """Prepare global plan."""
        return plans

    def write_data(self, plan, planner):
        # [FUNCTION DOC] HELPER: Counts the call and raises RuntimeError naming the behavior
        # and call count when the policy says fail; otherwise returns an already-completed
        # future holding one fake result mapping (success flag and 100 bytes written).
        """Write data with policy-based failure behavior."""
        from torch.futures import Future

        # Check if we should fail based on policy
        if self._should_fail():
            raise RuntimeError(
                f"TestStorageWriter: {self.behavior} policy triggered failure on call {self.call_count}"
            )

        # Return a Future that completes to simple WriteResult-like objects
        future = Future()
        result = [{"success": True, "bytes_written": 100}]
        future.set_result(result)
        return future

    def finish(self, metadata, results):
        # [FUNCTION DOC] HELPER: No-op; returns None.
        """Finish checkpoint."""
        return None

    def storage_meta(self):
        # [FUNCTION DOC] HELPER: Returns None (no storage metadata).
        """Return storage metadata."""
        return None

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        # [FUNCTION DOC] HELPER: Accepts any checkpoint id; always returns True.
        """Validate checkpoint ID."""
        return True


class TestAsyncProcessExecutor(DTensorTestBase):
    # [CLASS DOC] Multi-process test case checking error handling of the background-process
    # async save executor.
    """Test suite for async checkpoint process executor error handling using public APIs."""

    @with_comms
    def test_checkpoint_save_failure_continues_serving(self) -> None:
        # [FUNCTION DOC] Three phases with DCP_USE_PREFIX_STORE removed from the environment:
        # (1) get_free_port patched to return -1 so background process group creation fails
        # and a ValueError surfaces from the save future; (2) a save with a fail_once writer
        # yields a future whose exception mentions the policy failure, and on rank 0 the
        # port lookup ran exactly once (a new process was started); (3) a new executor
        # with a succeeding writer completes without calling get_free_port, showing the
        # background process was reused. Environment and patches are restored on exit.
        """Test that checkpoint save failure doesn't exit process, continues serving."""

        test_state_dict = {
            "model": {"weight": torch.randn(4, 4), "bias": torch.randn(4)},
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
        }

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DCP_USE_PREFIX_STORE", None)

            # 1. Simulate a failure in creating PG in background process.
            with patch(
                "torch.distributed.checkpoint._async_process_executor.get_free_port",
                return_value=-1,
            ):
                with self.assertRaises(ValueError) as _:
                    proc_executor = _ProcessBasedAsyncCheckpointExecutor()
                    fut = proc_executor.execute_save(
                        staging_future_or_state_dict=test_state_dict,
                    )
                    fut.result()

            # 2. Attempt save with failing storage writer
            with patch(
                "torch.distributed.checkpoint._async_process_executor.get_free_port",
                return_value=get_free_port(),
            ) as mock_get_free_port:
                proc_executor = _ProcessBasedAsyncCheckpointExecutor()
                fut = proc_executor.execute_save(
                    staging_future_or_state_dict=test_state_dict,
                    storage_writer=TestStorageWriter(behavior="fail_once"),
                )
                self.assertIn(
                    "fail_once policy triggered failure", str(fut.exception())
                )
                # Verify new process was created for this attempt
                if dist.get_rank() == 0:
                    mock_get_free_port.assert_called_once()

            # 3. Second save attempt with successful storage writer - process should still be alive
            with patch(
                "torch.distributed.checkpoint._async_process_executor.get_free_port",
            ) as mock_get_free_port:
                proc_executor = _ProcessBasedAsyncCheckpointExecutor()
                fut = proc_executor.execute_save(
                    staging_future_or_state_dict=test_state_dict,
                    storage_writer=TestStorageWriter(behavior="success"),
                )
                result = fut.result()
                # Verify process is still alive
                mock_get_free_port.assert_not_called()
                # Verify successful save
                self.assertIsNotNone(result)


class TestAsyncProcessExecutorPrefixStore(TestCase):
    # [CLASS DOC] Single-process test case for the executor when DCP_USE_PREFIX_STORE is on.
    @skip_if_win32()
    @retry_on_connect_failures
    def test_checkpoint_save_with_prefix_store_enabled(self) -> None:
        # [FUNCTION DOC] Sets DCP_USE_PREFIX_STORE=1 with a free local MASTER_PORT, initializes
        # a one-rank gloo default group, and runs an async save with a succeeding stub
        # writer; expects a non-None result and that no new free port was requested.
        # Side effect: the default process group is initialized and never destroyed here.
        # Skipped on Windows; retried on connect failures.
        """Test that checkpoint save works when DCP_USE_PREFIX_STORE is enabled."""

        test_state_dict = {
            "model": {"weight": torch.randn(4, 4), "bias": torch.randn(4)},
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
        }

        master_addr = "localhost"
        master_port = str(common.find_free_port())

        with patch.dict(
            os.environ,
            {
                "DCP_USE_PREFIX_STORE": "1",
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            },
        ):
            with patch(
                "torch.distributed.checkpoint._async_process_executor.get_free_port"
            ) as mock_get_free_port:
                dist.init_process_group(
                    backend=dist.Backend.GLOO,
                    rank=0,
                    world_size=1,
                )

                proc_executor = _ProcessBasedAsyncCheckpointExecutor()
                fut = proc_executor.execute_save(
                    staging_future_or_state_dict=test_state_dict,
                    storage_writer=TestStorageWriter(behavior="success"),
                )
                result = fut.result()
                self.assertIsNotNone(result)
                mock_get_free_port.assert_not_called()


class TestProcessGroupInitInfo(DTensorTestBase):
    # [CLASS DOC] Multi-process test case for _ProcessGroupInitInfo, which captures rank,
    # world size, store address and port, prefix-store and GC flags from the environment.
    """Test suite for _ProcessGroupInitInfo."""

    @with_comms
    def test_process_group_init_info_with_default_pg(self) -> None:
        # [FUNCTION DOC] With the prefix store variable unset, checks the info object has the
        # current global rank and world size, a store address, a positive port, and
        # use_prefix_store false.
        """Test that ProcessGroupInitInfo correctly initializes."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DCP_USE_PREFIX_STORE", None)

            pg_init_info = _ProcessGroupInitInfo()

            self.assertEqual(pg_init_info.global_rank, dist.get_rank())
            self.assertEqual(pg_init_info.world_size, dist.get_world_size())
            self.assertIsNotNone(pg_init_info.tcp_store_master_addr)
            self.assertGreater(pg_init_info.tcp_store_master_port, 0)
            self.assertEqual(pg_init_info.use_prefix_store, False)

    @with_comms
    def test_process_group_init_info_with_prefix_store_env_var(self) -> None:
        # [FUNCTION DOC] With the prefix store variable set to 1 and valid MASTER_ADDR and
        # MASTER_PORT, use_prefix_store is true; if the port is missing, the address is
        # missing, or the port is not a number, construction raises CheckpointException.
        """Test that ProcessGroupInitInfo handles DCP_USE_PREFIX_STORE environment variable."""

        # Flag enabled, addr/port correctly defined
        with patch.dict(
            os.environ,
            {
                "DCP_USE_PREFIX_STORE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            },
        ):
            pg_init_info = _ProcessGroupInitInfo()
        self.assertTrue(pg_init_info.use_prefix_store)

        # Missing port
        with patch.dict(
            os.environ, {"DCP_USE_PREFIX_STORE": "1", "MASTER_ADDR": "localhost"}
        ):
            with self.assertRaises(CheckpointException):
                pg_init_info = _ProcessGroupInitInfo()
        # Missing addr
        with patch.dict(
            os.environ, {"DCP_USE_PREFIX_STORE": "1", "MASTER_PORT": "12345"}
        ):
            with self.assertRaises(CheckpointException):
                pg_init_info = _ProcessGroupInitInfo()
        # Invalid port
        with patch.dict(
            os.environ,
            {
                "DCP_USE_PREFIX_STORE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "a",
            },
        ):
            with self.assertRaises(CheckpointException):
                pg_init_info = _ProcessGroupInitInfo()

    @with_comms
    def test_process_group_init_info_without_prefix_store_env_var(self) -> None:
        # [FUNCTION DOC] use_prefix_store is false when the variable is 0, unset, 2, true,
        # false or empty, i.e. only the exact value 1 enables it.
        """Test that ProcessGroupInitInfo defaults to not using prefix store."""

        # Env var set to 0
        with patch.dict(os.environ, {"DCP_USE_PREFIX_STORE": "0"}):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.use_prefix_store)

        # Missing env var
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DCP_USE_PREFIX_STORE", None)
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.use_prefix_store)

        # Invalid env var
        with patch.dict(os.environ, {"DCP_USE_PREFIX_STORE": "2"}):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.use_prefix_store)

        with patch.dict(os.environ, {"DCP_USE_PREFIX_STORE": "true"}):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.use_prefix_store)

        with patch.dict(os.environ, {"DCP_USE_PREFIX_STORE": "false"}):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.use_prefix_store)

        with patch.dict(os.environ, {"DCP_USE_PREFIX_STORE": ""}):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.use_prefix_store)

    @with_comms
    def test_process_group_init_info_gc_env_vars(self) -> None:
        # [FUNCTION DOC] Checks disable_automatic_gc and disable_manual_gc follow the
        # DCP_DISABLE_AUTOMATIC_GC and DCP_DISABLE_MANUAL_GC variables independently: true
        # for 1, false when unset or 0.
        """Test that ProcessGroupInitInfo correctly reads GC-related environment variables."""

        # Test with both GC env vars enabled
        with patch.dict(
            os.environ,
            {
                "DCP_DISABLE_AUTOMATIC_GC": "1",
                "DCP_DISABLE_MANUAL_GC": "1",
            },
        ):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertTrue(pg_init_info.disable_automatic_gc)
            self.assertTrue(pg_init_info.disable_manual_gc)

        # Test with automatic GC disabled, manual GC enabled
        with patch.dict(
            os.environ,
            {
                "DCP_DISABLE_AUTOMATIC_GC": "1",
            },
        ):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertTrue(pg_init_info.disable_automatic_gc)
            self.assertFalse(pg_init_info.disable_manual_gc)

        # Test with automatic GC enabled, manual GC disabled
        with patch.dict(
            os.environ,
            {
                "DCP_DISABLE_MANUAL_GC": "1",
            },
        ):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.disable_automatic_gc)
            self.assertTrue(pg_init_info.disable_manual_gc)

        # Test with both GC env vars disabled
        with patch.dict(
            os.environ,
            {
                "DCP_DISABLE_AUTOMATIC_GC": "0",
                "DCP_DISABLE_MANUAL_GC": "0",
            },
        ):
            pg_init_info = _ProcessGroupInitInfo()
            self.assertFalse(pg_init_info.disable_automatic_gc)
            self.assertFalse(pg_init_info.disable_manual_gc)


if __name__ == "__main__":
    run_tests()
