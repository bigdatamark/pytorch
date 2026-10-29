# Owner(s): ["oncall: distributed"]
# [MODULE DOC]
# Multi-process tests for the top-level distributed checkpoint save/load API (dcp.save and
# dcp.load): automatic storage reader/writer detection from checkpoint_id, and the
# cross-rank key consistency check. Defines a small MLP test module. Requires GPUs.
import os
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class MyTestModule(nn.Module):
    # [CLASS DOC] HELPER: small four-stage MLP (8 -> 16 -> 32 -> 64 -> 8) with ReLUs, used
    # only to produce a state dict for checkpoint tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: builds the four sub-networks net1..net4.
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: runs x through net1, net2, net3, net4 in order; returns output.
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestSaveAndLoadAPI(DTensorTestBase):
    # [CLASS DOC] Multi-process test case (DTensorTestBase) running with a world size of 2
    # that exercises dcp.save, dcp.load and dcp.utils helpers.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of spawned test processes, fixed at 2.
        return 2

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_auto_detect(self):
        # [FUNCTION DOC] Wraps the model in FSDP, saves and loads its state dict under a temp
        # directory with only checkpoint_id given (reader/writer auto-detected). Then patches
        # validate_checkpoint_id on the file system reader and writer to return false and
        # checks that save and load still work (with the patch active). Finally asserts that
        # an unrecognized id ("abc://abc.abc") makes save and load raise RuntimeError
        # matching "Cannot detect". Skipped with fewer than 4 GPUs.
        # KNOWN ISSUE: the model is wrapped in FSDP twice (the second wraps the first).
        model = FSDP(MyTestModule().to(self.device_type))
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = FSDP(model, device_mesh=device_mesh)
        dcp.save(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))
        dcp.load(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))

        with patch.object(
            dcp.FileSystemReader, "validate_checkpoint_id", return_value=False
        ):
            with patch.object(
                dcp.FileSystemWriter, "validate_checkpoint_id", return_value=False
            ):
                dcp.save(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )
                dcp.load(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )

        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.save(model.state_dict(), checkpoint_id="abc://abc.abc")
        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.load(model.state_dict(), checkpoint_id="abc://abc.abc")

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_assert_same_keys(self):
        # [FUNCTION DOC] Checks dcp.utils._assert_same_keys: passes when all ranks hold the
        # same state dict keys, and raises AssertionError after rank 0 adds key "abc" while
        # other ranks add "def". Skipped with fewer than 2 GPUs.
        """Test the `_assert_same_keys` function."""
        model = MyTestModule()
        state_dict = model.state_dict()
        # Check across ranks; expect true
        dcp.utils._assert_same_keys(state_dict)

        # Introduces difference; expect false
        if self.rank == 0:
            state_dict["abc"] = torch.rand(1)
        else:
            state_dict["def"] = torch.rand(1)

        with self.assertRaises(AssertionError):
            dcp.utils._assert_same_keys(state_dict)


if __name__ == "__main__":
    run_tests()
