# Owner(s): ["oncall: distributed checkpointing"]

# [MODULE DOC] Tests for CheckpointProcess in the experimental distributed checkpoint
# package: the request/response data types, the process config defaults, and a real child
# process that runs a CheckpointWriter (init, write with plain/future state dicts, init
# timeout and failure, graceful/forced close, broken pipe, shared-memory tensors across
# processes). Also defines module-level init functions that must be picklable for the
# child process. Spawns subprocesses and writes to temporary directories during tests.

import os
import tempfile
import time
from concurrent.futures import Future
from typing import Any

import torch
from torch.distributed.checkpoint._experimental.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
    RequestType,
    WorkerRequest,
    WorkerResponse,
)
from torch.distributed.checkpoint._experimental.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterConfig,
)
from torch.distributed.checkpoint._experimental.types import RankInfo
from torch.testing._internal.common_utils import run_tests, TestCase


def subprocess_init_fn(name: str, parent_pid: int) -> None:
    # [FUNCTION DOC] HELPER: Child-process init hook for the happy path. Raises
    # AssertionError if the name is not "test-checkpointer", if it runs in the parent's
    # own process, or if the current process's parent is not the given parent pid.
    """Initialize the subprocess with some basic checks.

    This is similar to the subprocess_init_routine in checkpointing_test.py.
    """
    if name != "test-checkpointer":
        raise AssertionError(f"Unexpected subprocess name: {name}")
    if os.getpid() == parent_pid:
        raise AssertionError("This was supposed to run in a different process")
    if os.getppid() != parent_pid:
        raise AssertionError("This was supposed to run as a child to main process")


def failing_subprocess_init_fn(name: str, parent_pid: int) -> None:
    # [FUNCTION DOC] HELPER: Child-process init hook that ignores its arguments and always
    # raises RuntimeError, used to test init failure propagation.
    """Initialize function that raises an exception."""
    # Acknowledge parameters to avoid unused variable warnings
    _ = name
    _ = parent_pid
    raise RuntimeError("Subprocess initialization failed")


def timedout_subprocess_init_fn(**kwargs: Any) -> None:
    # [FUNCTION DOC] HELPER: Child-process init hook that ignores its keyword arguments
    # and sleeps 3 seconds, to exceed a short init timeout.
    # Acknowledge parameters to avoid unused variable warnings
    _ = kwargs
    time.sleep(3)  # Simulate a long initialization


def ckpt_writer_init_fn(**kwargs: Any) -> CheckpointWriter:
    # [FUNCTION DOC] HELPER: Module-level (picklable) factory run in the child process.
    # Returns a CheckpointWriter built from the config and rank_info keyword arguments.
    """Initialize a CheckpointWriter in the subprocess.

    This function is called in the subprocess to create a CheckpointWriter instance.
    It's important that this function is defined at the module level so it can be pickled.
    """
    return CheckpointWriter(
        config=kwargs.get("config"),
        rank_info=kwargs.get("rank_info"),
    )


def failing_ckpt_writer_init_fn(**kwargs: Any) -> CheckpointWriter:
    # [FUNCTION DOC] HELPER: Writer factory that ignores arguments and always raises
    # RuntimeError. Not used by any test in this file.
    """Initialize function that raises an exception."""
    # Acknowledge parameters to avoid unused variable warnings
    _ = kwargs
    raise RuntimeError("CheckpointWriter initialization failed")


def shared_tensor_verifier_init_fn(**kwargs: Any) -> CheckpointWriter:
    # [FUNCTION DOC] HELPER: Writer factory run in the child process. Defines a
    # CheckpointWriter subclass whose write checks tensors in the received state dict are
    # in shared memory and returns the instance built from config and rank_info arguments.
    """Initialize a CheckpointWriter that verifies shared memory tensors."""

    class SharedTensorVerifier(CheckpointWriter):
        # [CLASS DOC] HELPER: CheckpointWriter that does not write files. Its write method
        # asserts tensors arriving through inter-process transfer are shared-memory
        # tensors and stamps the shared tensor so the parent can detect same-memory access.
        def __init__(self, config=None, rank_info=None, **init_kwargs):
            # [FUNCTION DOC] HELPER: Calls the base writer constructor with the given or a
            # default writer config, the rank info, and no barrier or commit hook; extra
            # keyword arguments are ignored.
            # Acknowledge unused kwargs to avoid linting warnings
            _ = init_kwargs
            super().__init__(
                config=config or CheckpointWriterConfig(),
                rank_info=rank_info,
                barrier=None,
                commit_hook=None,
            )

        def write(self, state_dict, path, **__):
            # [FUNCTION DOC] HELPER: Ignores the path. If a shared tensor entry exists,
            # raises AssertionError unless it is in shared memory, then sets its first
            # element to 42.0 (visible to the parent). If a regular tensor entry exists,
            # raises AssertionError unless it has also been moved to shared memory. Returns
            # None.
            # Acknowledge parameters to avoid unused variable warnings
            _ = path

            # Verify shared memory tensor behavior directly with assertions
            if "shared_tensor" in state_dict:
                shared_tensor = state_dict["shared_tensor"]
                # Critical assertion: shared tensor should remain in shared memory in subprocess
                if not shared_tensor.is_shared():
                    raise AssertionError(
                        "Shared tensor should be in shared memory in subprocess"
                    )

                shared_tensor[0] = 42.0

            if "regular_tensor" in state_dict:
                # Note: ForkingPickler moves regular tensors to shared memory during IPC - this is acceptable
                if not state_dict["regular_tensor"].is_shared():
                    raise AssertionError(
                        "Regular tensor should also be in shared memory in subprocess"
                    )

            return None

    verifier = SharedTensorVerifier(
        config=kwargs.get("config"),
        rank_info=kwargs.get("rank_info"),
    )
    return verifier


class TestRequestTypes(TestCase):
    # [CLASS DOC] Tests the RequestType enum and the WorkerRequest and WorkerResponse
    # records used for parent-child communication.
    """Test the request/response data structures."""

    def test_request_type_enum(self) -> None:
        # [FUNCTION DOC] Checks the string values of the PING, WRITE_CHECKPOINT and
        # TERMINATE_PROCESS request types.
        """Test RequestType enum values."""
        self.assertEqual(RequestType.PING.value, "ping")
        self.assertEqual(RequestType.WRITE_CHECKPOINT.value, "write_checkpoint")
        self.assertEqual(RequestType.TERMINATE_PROCESS.value, "exit")

    def test_worker_request(self) -> None:
        # [FUNCTION DOC] Builds a WorkerRequest and checks the request type and payload
        # fields are stored.
        """Test WorkerRequest dataclass."""
        request = WorkerRequest(request_type=RequestType.PING, payload={"test": "data"})
        self.assertEqual(request.request_type, RequestType.PING)
        self.assertEqual(request.payload["test"], "data")

    def test_worker_response(self) -> None:
        # [FUNCTION DOC] Builds a WorkerResponse and checks request type, success flag,
        # empty error message and payload are stored.
        """Test WorkerResponse dataclass."""
        response = WorkerResponse(
            request_type=RequestType.PING,
            success=True,
            error_msg=None,
            payload={"result": "success"},
        )
        self.assertEqual(response.request_type, RequestType.PING)
        self.assertTrue(response.success)
        self.assertIsNone(response.error_msg)
        self.assertEqual(response.payload["result"], "success")


class TestCheckpointProcessConfig(TestCase):
    # [CLASS DOC] Tests the defaults and overrides of CheckpointProcessConfig.
    """Test CheckpointProcessConfig configuration."""

    def test_default_options(self) -> None:
        # [FUNCTION DOC] Checks defaults: 30 second init timeout and 60 second shutdown
        # timeout.
        """Test default CheckpointProcessConfig."""
        options = CheckpointProcessConfig()
        # Test default values
        self.assertEqual(options.subprocess_init_timeout_secs, 30)
        self.assertEqual(options.subprocess_shutdown_timeout_secs, 60)

    def test_custom_options(self) -> None:
        # [FUNCTION DOC] Checks explicit init (10) and shutdown (30) timeouts are kept.
        """Test custom CheckpointProcessConfig."""
        options = CheckpointProcessConfig(
            subprocess_init_timeout_secs=10, subprocess_shutdown_timeout_secs=30
        )
        self.assertEqual(options.subprocess_init_timeout_secs, 10)
        self.assertEqual(options.subprocess_shutdown_timeout_secs, 30)


class TestCheckpointProcess(TestCase):
    # [CLASS DOC] Integration tests that start a real CheckpointProcess (child process
    # hosting a CheckpointWriter). Fixtures: a single-rank RankInfo, a default writer
    # config and a small state dict. Most tests wait for process_creation_future, use the
    # process, and close it to terminate the child.
    def setUp(self) -> None:
        # [FUNCTION DOC] Creates the rank info (world size 1, rank 0), default writer
        # config and sample state dict shared by the tests.
        super().setUp()
        """Set up common test fixtures."""
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
        )
        self.writer_config = CheckpointWriterConfig()
        self.test_state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

    def _create_checkpoint_process(
        self,
        subprocess_init_fn_override=None,
        subprocess_init_args_override=None,
        writer_init_fn_override=None,
        subprocess_init_timeout_secs=30,
    ):
        # [FUNCTION DOC] HELPER: Starts and returns a CheckpointProcess using the fixtures.
        # Optional overrides replace the child init function, its arguments (default name
        # "test-checkpointer" and current pid) and the writer factory; the init timeout is
        # set in the process config. Side effect: spawns a child process.
        """Helper to create CheckpointProcess."""
        config = CheckpointProcessConfig(
            subprocess_init_timeout_secs=subprocess_init_timeout_secs,
        )

        return CheckpointProcess(
            rank_info=self.rank_info,
            config=config,
            subprocess_init_fn=subprocess_init_fn_override or subprocess_init_fn,
            subprocess_init_args=subprocess_init_args_override
            or (
                "test-checkpointer",
                os.getpid(),
            ),
            checkpoint_writer_init_fn=writer_init_fn_override or ckpt_writer_init_fn,
            checkpoint_writer_init_args={
                "config": self.writer_config,
                "rank_info": self.rank_info,
            },
        )

    def test_checkpoint_process_initialization(self) -> None:
        # [FUNCTION DOC] Starts the process, waits for creation, checks the child is
        # alive, closes it, and checks the child is no longer alive.
        """Test that CheckpointProcess initializes and closes correctly."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for the process creation future to complete
        checkpoint_process.process_creation_future.result()

        # Verify process is alive
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())

        checkpoint_process.close()

        # Verify process is terminated
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_checkpoint_write_sync_state_dict(self) -> None:
        # [FUNCTION DOC] Writes a plain state dict through the child process into a temp
        # directory, checks a future is returned, waits for it, then loads the per-rank file
        # and verifies model, optimizer, epoch and step. Closes the process.
        """Test writing a checkpoint with synchronous state dict."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint
            future = checkpoint_process.write(self.test_state_dict, checkpoint_path)

            # Verify future is returned
            self.assertIsInstance(future, Future)

            # Wait for completion
            future.result()

            # Verify checkpoint file was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

            # Verify checkpoint content
            loaded_state_dict = torch.load(expected_file)
            self.assertIn("model", loaded_state_dict)
            self.assertIn("optimizer", loaded_state_dict)
            self.assertEqual(loaded_state_dict["epoch"], 5)
            self.assertEqual(loaded_state_dict["step"], 1000)

        checkpoint_process.close()

    def test_checkpoint_write_future_state_dict(self) -> None:
        # [FUNCTION DOC] Passes a Future (resolved after a short delay by a thread pool)
        # as the state dict, waits for the write to finish and checks the checkpoint file
        # exists. Shuts down the executor and closes the process.
        """Test writing a checkpoint with Future state dict."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Create a Future that resolves to the state dict
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)

        def get_state_dict():
            time.sleep(0.1)  # Simulate some processing time
            return self.test_state_dict

        future_state_dict = executor.submit(get_state_dict)

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint with Future state dict
            write_future = checkpoint_process.write(future_state_dict, checkpoint_path)

            # Wait for completion
            write_future.result()

            # Verify checkpoint file was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

        executor.shutdown(wait=True)
        checkpoint_process.close()

    def test_checkpoint_write_with_kwargs(self) -> None:
        # [FUNCTION DOC] Writes with extra keyword arguments, expecting them to be accepted
        # and forwarded without error, and checks the checkpoint file exists.
        """Test checkpoint writing with additional kwargs."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint with kwargs
            future = checkpoint_process.write(
                self.test_state_dict,
                checkpoint_path,
                custom_arg="test_value",
                another_arg=42,
            )

            # Wait for completion
            future.result()

            # Verify checkpoint was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

        checkpoint_process.close()

    def test_subprocess_initialization_timeout(self) -> None:
        # [FUNCTION DOC] Uses a child init that sleeps 3 seconds with a 1 second timeout;
        # waiting on the creation future must raise TimeoutError whose message contains
        # "Timed out". The process is not closed afterwards.
        """Test subprocess initialization timeout."""

        # Create checkpoint process with a very short timeout by mocking the initialization
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=timedout_subprocess_init_fn,
            subprocess_init_timeout_secs=1,
        )

        # This should timeout
        with self.assertRaises(TimeoutError) as cm:
            checkpoint_process.process_creation_future.result()

        self.assertIn("Timed out", str(cm.exception))

    def test_subprocess_initialization_failure(self) -> None:
        # [FUNCTION DOC] Uses a child init that raises; waiting on the creation future must
        # raise RuntimeError. The process is not closed afterwards.
        """Test subprocess initialization failure."""
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=failing_subprocess_init_fn
        )

        # The subprocess should fail to initialize
        # We expect this to raise an exception when we try to use it
        with self.assertRaises(RuntimeError):
            checkpoint_process.process_creation_future.result()

    def test_graceful_termination(self) -> None:
        # [FUNCTION DOC] Starts the process, checks it is alive, closes it and checks it
        # has stopped (same checks as the initialization test).
        """Test graceful termination of subprocess."""
        checkpoint_process = self._create_checkpoint_process()

        checkpoint_process.process_creation_future.result()
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())
        checkpoint_process.close()
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_forced_termination(self) -> None:
        # [FUNCTION DOC] Replaces the process join method with one that reports a timeout,
        # then closes, intending to trigger the forced kill path.
        # KNOWN ISSUE: makes no assertion about the process state afterwards (a comment
        # admits it may be flaky), so it only checks that close does not raise.
        """Test forced termination when graceful termination fails."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Mock the join method to simulate timeout
        def mock_join(timeout=None):
            # Acknowledge timeout parameter to avoid unused variable warning
            _ = timeout
            return False  # Simulate timeout

        checkpoint_process.process.join = mock_join

        # This should trigger forced termination
        checkpoint_process.close()

        # Process should still be terminated (killed)
        # Note: This test might be flaky depending on timing

    def test_communication_error_handling(self):
        # [FUNCTION DOC] Closes the parent end of the pipe to the child, then a write must
        # raise RuntimeError whose message contains "Child process terminated unexpectedly".
        # The process is not closed afterwards.
        """Test handling of communication errors."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process.process_creation_future.result()

        # Close the pipe to simulate communication failure
        checkpoint_process._parent_end.close()

        # Attempting to write should raise an error
        with self.assertRaises(RuntimeError) as cm:
            future = checkpoint_process.write(self.test_state_dict, "/tmp/test")
            future.result()

        self.assertIn("Child process terminated unexpectedly", str(cm.exception))

    def test_shared_memory_tensor_ipc(self):
        # [FUNCTION DOC] Uses the verifier writer in the child. Sends a state dict with a
        # shared-memory tensor and a regular tensor, waits for the write (any child
        # assertion failure fails the test), checks the shared tensor kept its data pointer
        # and sharing status, and that the child's marker value 42.0 is visible in the
        # parent, proving both processes use the same memory. Closes the process.
        """Test that shared memory tensors are backed by the same memory across processes."""

        checkpoint_process = self._create_checkpoint_process(
            writer_init_fn_override=shared_tensor_verifier_init_fn,
        )

        checkpoint_process.process_creation_future.result()

        # Create tensors and put them in shared memory
        shared_tensor = torch.randn(100, 100)
        shared_tensor.share_memory_()

        shared_tensor_data_ptr = shared_tensor.data_ptr()

        regular_tensor = torch.randn(50, 50)
        # Don't put regular tensor in shared memory for comparison

        # Verify initial shared memory status
        self.assertTrue(
            shared_tensor.is_shared(), "Shared tensor should be in shared memory"
        )
        self.assertFalse(
            regular_tensor.is_shared(), "Regular tensor should not be in shared memory"
        )

        # Create state dict with mixed tensor types
        test_state_dict = {
            "shared_tensor": shared_tensor,
            "regular_tensor": regular_tensor,
        }

        # Write to subprocess - the SharedTensorVerifier will:
        # 1. Verify the tensor is still in shared memory
        # 2. Check the marker value (42.0) to confirm same memory
        # 3. Modify specific positions to prove same memory access
        future = checkpoint_process.write(test_state_dict, "")

        try:
            result = (
                future.result()
            )  # This will raise an exception if the subprocess assertions fail
            self.assertIsNone(result)  # SharedTensorVerifier returns None on success
        except Exception as e:
            self.fail(f"Subprocess assertions failed: {e}")

        # assert shared tensor is still in same shared memory
        self.assertEqual(
            shared_tensor_data_ptr,
            shared_tensor.data_ptr(),
            "Shared tensor should still be in same shared memory",
        )
        self.assertTrue(
            shared_tensor.is_shared(), "Shared tensor should still be in shared memory"
        )

        # CRITICAL TEST: Verify that modifications made by subprocess are visible in main process
        # This definitively proves that both processes access the same memory

        self.assertAlmostEqual(
            shared_tensor[0][0],
            42.0,
            places=6,
            msg=lambda msg: f"{msg}\nExpected subprocess signature 42.0, got {shared_tensor[0]}. "
            f"Shared memory not working - subprocess modifications not visible!",
        )

        checkpoint_process.close()


if __name__ == "__main__":
    run_tests()
