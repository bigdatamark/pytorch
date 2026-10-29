# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the fsspec-based checkpoint storage (FsspecWriter/FsspecReader and
# the FileSystem abstraction): an FSDP model and optimizer round trip, overwrite semantics,
# cleanup of partially written streams on failure, and graceful behavior when the stream has
# no file descriptor (in-memory filesystem). Defines a local with_temp_dir decorator.
# Import-time side effect: picks the accelerator device type and its default backend.

import shutil
import tempfile
from collections.abc import Callable
from functools import wraps
from typing import Any
from unittest.mock import patch

import fsspec
import fsspec.implementations.memory

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint._fsspec_filesystem import (
    FileSystem,
    FsspecReader,
    FsspecWriter,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.checkpoint.utils import CheckpointException
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
BACKEND = torch.distributed.get_default_backend_for_device(device_type)


def with_temp_dir(
    func: Callable | None = None,
) -> Callable | None:
    # [FUNCTION DOC] HELPER: Decorator factory used bare on a test method. Raises
    # AssertionError if no function is given; otherwise returns a wrapper that creates a
    # temp directory on rank 0 (or when no process group exists), broadcasts its path to
    # all ranks, stores it as the test's temp_dir attribute, runs the test, and finally
    # removes the directory on rank 0, ignoring errors. Prints the directory it creates.
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    if func is None:
        raise AssertionError("Expected func to not be None")

    @wraps(func)
    def wrapper(self, *args: tuple[object], **kwargs: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Sets up the shared temp_dir (rank 0 creates and broadcasts
        # it), calls the wrapped test with the given arguments, and always cleans up the
        # directory afterward on rank 0.
        # Only create temp_dir when rank is 0 (or no pg)
        if not dist.is_initialized() or dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            print(f"Using temp directory: {temp_dir}")
        else:
            temp_dir = ""
        object_list = [temp_dir]

        # Broadcast temp_dir to all the other ranks
        if dist.is_initialized():
            dist.broadcast_object_list(object_list)
        self.temp_dir = object_list[0]

        try:
            func(self, *args, **kwargs)
        finally:
            if not dist.is_initialized() or dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper


class MyTestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Four-stage MLP (8->16->32->64->8) used as the FSDP test model.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates net1 to net4 (Linear and ReLU stages).
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs x through net1, net2, net3, net4 in order.
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestFSSpec(ShardedTensorTestBase):
    # [CLASS DOC] Two-rank tests of DCP save/load through FsspecWriter and FsspecReader.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks for these tests.
        return 2

    @with_comms(backend=BACKEND, init_rpc=False)
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsspec(self):
        # [FUNCTION DOC] Trains one Adam step on an FSDP model, saves the sharded model and
        # optimizer state dicts with FsspecWriter, checks a second FSDP model differs, then
        # loads the model with FsspecReader and the optimizer via
        # load_sharded_optimizer_state_dict, and asserts parameters and the Adam first and
        # second moment buffers of the first state entry match. Needs 2+ GPUs.
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(MyTestModule().to(device_type))
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optim),
            }

            dcp.save(
                state_dict=state_dict,
                storage_writer=FsspecWriter(CHECKPOINT_DIR),
                planner=dcp.DefaultSavePlanner(),
            )

        model_2 = FSDP(MyTestModule().to(device_type))
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    self.assertNotEqual(n_p1[1], n_p2[1])

        # now load the model and ensure the values are the same
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model_2.state_dict(),
            }

            dcp.load(
                state_dict=state_dict,
                storage_reader=FsspecReader(CHECKPOINT_DIR),
                planner=dcp.DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=FsspecReader(CHECKPOINT_DIR),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                model_2, optim_2, optim_state["optim"]
            )
            optim_2.load_state_dict(flattened_osd)

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    self.assertEqual(n_p1[1], n_p2[1])

        def opt_at(opt, idx):
            # [FUNCTION DOC] HELPER: Returns the idx-th per-parameter state mapping of the
            # optimizer, in state insertion order.
            return list(iter(opt.state.values()))[idx]

        # Adam lazily creates its state
        self.assertEqual(opt_at(optim, 0)["exp_avg"], opt_at(optim_2, 0)["exp_avg"])
        self.assertEqual(
            opt_at(optim, 0)["exp_avg_sq"], opt_at(optim_2, 0)["exp_avg_sq"]
        )

    @with_comms(backend=BACKEND, init_rpc=False)
    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_overwrite(self):
        # [FUNCTION DOC] Saves one tensor with overwrite disabled, saves another to the same
        # location with overwrite enabled, loads and asserts the second value is returned,
        # then asserts a further save with overwrite disabled raises CheckpointException
        # mentioning that the checkpoint already exists. Needs 2+ GPUs.
        t1, t2 = torch.randn(10), torch.randn(10)

        dcp.save(
            {"random": t1}, storage_writer=FsspecWriter(self.temp_dir, overwrite=False)
        )
        dcp.save(
            {"random": t2}, storage_writer=FsspecWriter(self.temp_dir, overwrite=True)
        )

        sd = {"random": torch.zeros(10)}
        dcp.load(sd, checkpoint_id=self.temp_dir)
        self.assertTrue(torch.allclose(sd["random"], t2))

        with self.assertRaisesRegex(
            CheckpointException, ".*Checkpoint already exists.*"
        ):
            dcp.save(
                {"random": t2},
                storage_writer=FsspecWriter(self.temp_dir, overwrite=False),
            )


class TestFileSystem(TestCase):
    # [CLASS DOC] Single-process tests of the fsspec-backed FileSystem helper and writer.
    @with_temp_dir
    def test_remove_on_fail(self):
        # [FUNCTION DOC] Verifies create_stream cleanup: a file opened for writing is removed
        # when the body raises OSError, while a file opened for reading that raises is kept.
        fs = FileSystem()
        path = fs.init_path(self.temp_dir)

        write_file = fs.concat_path(path, "writeable")
        with self.assertRaises(OSError):
            with fs.create_stream(write_file, "w") as s:
                s.write("aaa")
                raise OSError("fail")
        self.assertFalse(fs.exists(write_file))

        read_file = fs.concat_path(path, "readable")
        with fs.create_stream(read_file, "w") as s:
            s.write("bbb")
        self.assertTrue(fs.exists(read_file))

        with self.assertRaises(OSError):
            with fs.create_stream(read_file, "r") as s:
                raise OSError("fail")
        self.assertTrue(fs.exists(read_file))

    @patch("os.sync")
    def test_fsspec_without_fileno_support(self, mock_os_sync):
        # [FUNCTION DOC] With os.sync mocked, saves a tensor with FsspecWriter to the fsspec
        # memory filesystem (no file descriptor support) without a process group, while
        # spying on the in-memory file flush. Asserts flush was called, the checkpoint loads
        # back with equal values, and os.sync was called at most twice.
        """
        fsspec's "memory://" protocol simulates cloud storage
        by not supporting .fileno() and raising io.UnsupportedOperation.
        This tests that the stream is flushed and fsync degrades gracefully.
        """
        checkpoint_dir = "memory://test_checkpoint_with_no_file_no"

        # Create a dummy state dict
        state_dict = {"tensor": torch.randn(10)}

        # Spy on the MemoryFile flush method to ensure it gets called
        with patch.object(
            fsspec.implementations.memory.MemoryFile, "flush", autospec=True
        ) as mock_flush:
            # Save using FsspecWriter
            dcp.save(
                state_dict=state_dict,
                storage_writer=FsspecWriter(checkpoint_dir),
                planner=dcp.DefaultSavePlanner(),
                no_dist=True,
            )

            # Assert that flush() was explicitly called on the stream
            self.assertTrue(
                mock_flush.called, "Expected stream.flush() to be called explicitly."
            )

        # Verify it saved properly and can be loaded
        load_dict = {"tensor": torch.zeros(10)}
        dcp.load(
            state_dict=load_dict,
            storage_reader=FsspecReader(checkpoint_dir),
            planner=dcp.DefaultLoadPlanner(),
            no_dist=True,
        )

        self.assertTrue(torch.allclose(state_dict["tensor"], load_dict["tensor"]))

        # os.sync() may be called on backends that don't support per-file fsync
        self.assertLessEqual(mock_os_sync.call_count, 2)


if __name__ == "__main__":
    run_tests()
