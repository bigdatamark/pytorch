# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests that save an FSDP model's sharded state dict with DCP and
# load it into a second FSDP model, both with the same process group and with a resharded
# (even/odd rank split) process group.

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class FsdpModelStateCheckpoint(DTensorTestBase):
    # [CLASS DOC] Test case for FSDP sharded model state save/load round trips with and
    # without resharding.
    @property
    def backend(self):
        # [FUNCTION DOC] HELPER: Returns a combined backend string using gloo for CPU and
        # the default backend of the test device type for the accelerator.
        curr_backend = dist.get_default_backend_for_device(self.device_type)
        return f"cpu:gloo,{self.device_type}:{curr_backend}"

    def _test_fsdp_model_state(self, process_group) -> None:
        # [FUNCTION DOC] HELPER: Wraps a meta-device Linear in FSDP, runs a backward pass,
        # and saves its sharded state dict. Builds a second FSDP model (optionally on the
        # given process group), checks its full params differ, loads the checkpoint into it
        # and asserts weight and bias equal the first model's.
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(torch.nn.Linear(8, 8, device="meta"))
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
            }

            dist_cp.save(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
                planner=DefaultSavePlanner(),
            )

        model_2 = FSDP(
            torch.nn.Linear(8, 8, device="meta"), process_group=process_group
        )

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertNotEqual(model.weight, model_2.weight)
                self.assertNotEqual(model.bias, model_2.bias)

        # now load the model and ensure the values are the same
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model_2.state_dict(),
            }

            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                planner=DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertEqual(model.weight, model_2.weight)
                self.assertEqual(model.bias, model_2.bias)

    @skip_if_lt_x_gpu(2)
    @with_comms
    @with_temp_dir
    def test_fsdp_model_state_no_resharding(self):
        # [FUNCTION DOC] Runs the round trip with the default process group. Needs 2+ GPUs.
        self._test_fsdp_model_state(process_group=None)

    def _create_new_dist_group(self):
        # [FUNCTION DOC] HELPER: Creates two new groups, one of even ranks and one of odd
        # ranks (every rank participates in creating both), and returns the one containing
        # the calling rank.
        world_size = dist.get_world_size()
        group1 = [i for i in range(world_size) if i % 2 == 0]
        group2 = [i for i in range(world_size) if i % 2 != 0]

        # create new fsdp group for resharding
        fsdp_0 = dist.new_group(ranks=group1)
        fsdp_1 = dist.new_group(ranks=group2)
        if dist.get_rank() % 2 == 0:
            my_fsdp = fsdp_0
        else:
            my_fsdp = fsdp_1

        return my_fsdp

    @skip_if_lt_x_gpu(4)
    @with_comms
    @with_temp_dir
    def test_fsdp_model_state_with_resharding(self):
        # [FUNCTION DOC] Runs the round trip where the loading model shards over an even/odd
        # rank subgroup, so the checkpoint is resharded on load. Needs 4+ GPUs.
        self._test_fsdp_model_state(process_group=self._create_new_dist_group())


if __name__ == "__main__":
    run_tests()
