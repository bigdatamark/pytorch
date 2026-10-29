# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for checkpoint format conversion utilities: DCP to torch.save file,
# torch.save file to DCP, and loading a torch.save file directly into an FSDP-sharded model
# with a broadcasting reader. Defines a small uneven-width MLP used as the test model.

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    dcp_to_torch_save,
    DynamicMetaLoadPlanner,
    torch_save_to_dcp,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class SimpleModelUneven(nn.Module):
    # [CLASS DOC] HELPER: Four-layer MLP (5->10->15->30->5) with ReLU between layers, built
    # deterministically (fixed seed) so separate instances start with identical weights.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 (a global side effect) and
        # creates the four Linear layers and a ReLU module.
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies net1..net3 each followed by ReLU, then net4
        # without activation; returns the output tensor.
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        # [FUNCTION DOC] HELPER: Returns a random 4x5 tensor on the accelerator device.
        return torch.rand(4, 5, device=device_type)


class TestFormatUtils(DTensorTestBase):
    # [CLASS DOC] Tests for dcp_to_torch_save, torch_save_to_dcp and the broadcasting
    # torch.save reader with a dynamic-metadata load planner.
    @with_temp_dir
    def test_dcp_to_torch_save(self) -> None:
        # [FUNCTION DOC] Saves a model as a DCP checkpoint, converts it to a single
        # torch.save file in the same temp directory, loads that file and asserts it equals
        # the model's state dict under the model key.
        model = SimpleModelUneven()
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        torch_path = self.temp_dir + "/model.pt"
        dcp_to_torch_save(self.temp_dir, torch_path)

        loaded_sd = torch.load(torch_path)
        self.assertEqual(loaded_sd, {"model": model.state_dict()})

    @with_temp_dir
    def test_torch_save_to_dcp(self) -> None:
        # [FUNCTION DOC] Writes a model state dict with torch.save, converts it to a DCP
        # checkpoint, loads that into a new model instance and asserts the state dicts match.
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}
        torch_path = self.temp_dir + "/model.pt"
        torch.save(sd, torch_path)

        torch_save_to_dcp(torch_path, self.temp_dir)

        model = SimpleModelUneven()
        dcp.load({"model": model}, checkpoint_id=self.temp_dir)

        self.assertEqual({"model": model.state_dict()}, sd)

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_online_torch_save_to_dcp(self) -> None:
        # [FUNCTION DOC] Rank 0 writes a torch.save file; after a barrier, all ranks load it
        # straight into an FSDP-wrapped model via DCP load with a dynamic-metadata planner
        # and the broadcasting torch.save reader, then assert the state dicts match.
        # Needs 2+ GPUs.
        """Tests loading a model saved by torch.save directly into a sharded model
        using dcp.load
        """
        # Save a model with torch.save
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}

        torch_fn = self.temp_dir + "/model.pt"
        if dist.get_rank() == 0:
            torch.save(sd, torch_fn)
        dist.barrier()

        # Load into a sharded model
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = SimpleModelUneven().to(self.device_type)
        model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,
        )
        dcp.load(
            {"model": model},
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingTorchSaveReader(),
            checkpoint_id=torch_fn,
        )

        self.assertEqual(sd["model"], model.state_dict())


if __name__ == "__main__":
    run_tests()
