# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Multi-GPU tests that save and load tensor-parallel (DTensor) model state dicts
# with torch.distributed.checkpoint, including loading into a model built on the meta device.
# Defines a small uneven-shape model used by the meta device test. No import-time side effects.

from copy import deepcopy

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class UnevenShardedModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Three layer MLP (linear 5->10, ReLU, linear 10->15, linear 15->1) whose
    # dimensions do not divide evenly across typical world sizes, so tensor parallel sharding
    # produces uneven shards. Parameters are created directly on the given device.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 5 (side effect) so weights are
        # deterministic, then builds the three linear layers and the ReLU on the given device.
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(5, 10, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 15, device=device)
        self.net3 = torch.nn.Linear(15, 1, device=device)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies net1, ReLU, net2, net3 in sequence to the input and
        # returns the result.
        return self.net3(self.net2(self.relu(self.net1(x))))


class TestTpCheckpoint(DTensorTestBase):
    # [CLASS DOC] Multi-process DTensor test case; each test runs on every rank with a process
    # group set up by with_comms and a shared temporary checkpoint directory (self.temp_dir).
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint(self):
        # [FUNCTION DOC] Parallelizes an MLP with column/row tensor parallelism, saves its state
        # dict with the default planners, takes an SGD step so parameters differ, loads the
        # checkpoint back in place and checks local shards equal the originally saved values.
        # Requires at least 2 GPUs.
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # create model and move it to GPU with id rank
        model = MLPModule(self.device_type).to(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        original_state_dict = deepcopy(model.state_dict())

        dcp.save(
            state_dict=original_state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # Update the parameters so model.state_dict() will be different from original_state_dict.
        torch.manual_seed(0)
        inp = torch.rand(20, 10).to(self.rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        state_dict = model.state_dict()

        # ensure the current model parameters are different from original_state_dict before loading from checkpoint
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertNotEqual(param1.to_local(), param2.to_local())

        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )

        # now load from checkpoint to check current model parameters are the same as original_state_dict
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertEqual(param1.to_local(), param2.to_local())

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint_load_on_meta_device(self):
        # [FUNCTION DOC] Saves a tensor-parallel uneven-shape model, then builds a second model
        # on the meta device, loads the checkpoint into its state dict in place (the same dict
        # object must be kept), assigns it into the model and checks no parameter is still meta
        # and all local shards match the saved ones. Requires at least 2 GPUs.
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # create model and move it to GPU with id rank
        model = UnevenShardedModel(self.device_type).to(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
            "net3": ColwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan=parallelize_plan)
        original_state_dict = {
            "model": model.state_dict(),
        }

        dcp.save(
            state_dict=original_state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
        )

        model2 = parallelize_module(
            UnevenShardedModel("meta"), tp_mesh, parallelize_plan=parallelize_plan
        )
        model2_sd_before_load = model2.state_dict()
        state_dict_to_load = {"model": model2_sd_before_load}

        dcp.load(
            state_dict=state_dict_to_load,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
        )
        # We need to make sure state_dict_to_load["model"] is the same as state_dict_after_load["model"],
        # since we are doing in-place loading.
        self.assertTrue(state_dict_to_load["model"] is model2_sd_before_load)

        model2.load_state_dict(state_dict_to_load["model"], assign=True)
        state_dict_after_load = {"model": model2.state_dict()}

        self.assertEqual(
            len(original_state_dict["model"]), len(state_dict_to_load["model"])
        )
        self.assertEqual(
            len(original_state_dict["model"]), len(state_dict_after_load["model"])
        )

        for name, param in original_state_dict["model"].items():
            param_to_load = state_dict_to_load["model"][name]
            param_after_load = state_dict_after_load["model"][name]

            # we need to explicitly check the device is not meta as the assertEqual check
            # currently doesn't handle DTensor with meta device.
            self.assertTrue(not param_to_load.is_meta)
            self.assertTrue(not param_after_load.is_meta)
            self.assertEqual(param.to_local(), param_to_load.to_local())
            self.assertEqual(param.to_local(), param_after_load.to_local())


if __name__ == "__main__":
    run_tests()
