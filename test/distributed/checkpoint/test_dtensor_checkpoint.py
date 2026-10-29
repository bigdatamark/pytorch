# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Multi-GPU test that saves and loads a module whose state dict holds DTensors
# (sharded and replicated, on the full mesh and on a submesh of even ranks) plus module
# extra state, using DCP's default save/load planners and a FileSystemWriter/Reader.
# Defines a small test module and a DTensorTestBase subclass; SUBMESH_TENSOR_SIZE is the
# global length of the submesh tensors. No import-time side effects.

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
    zeros,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


SUBMESH_TENSOR_SIZE = 6


class MyTestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module owning four DTensor parameters (sdt, rdt, submesh_sdt,
    # submesh_rdt) and non-parameter extra state (an int and a tensor) that is exported
    # and restored through get_extra_state/set_extra_state so it appears in the state dict.
    def __init__(
        self,
        sdt: DTensor,
        rdt: DTensor,
        submesh_sdt: DTensor,
        submesh_rdt: DTensor,
        extra_state: int = 1,
        extra_state_tensor: torch.Tensor = torch.zeros(1),
    ) -> None:
        # [FUNCTION DOC] HELPER: Wraps each DTensor argument in a Parameter and stores the
        # extra state int and tensor in private attributes.
        # KNOWN ISSUE: the default extra_state_tensor is one shared tensor object created
        # at definition time, so default-constructed modules alias it.
        super().__init__()
        self.sdt = torch.nn.Parameter(sdt)
        self.rdt = torch.nn.Parameter(rdt)
        self.submesh_sdt = torch.nn.Parameter(submesh_sdt)
        self.submesh_rdt = torch.nn.Parameter(submesh_rdt)
        self._extra_state = extra_state
        self._extra_state_tensor = extra_state_tensor

    @property
    def extra_state(self) -> int:
        # [FUNCTION DOC] HELPER: Property returning the integer extra state.
        return self._extra_state

    @extra_state.setter
    def extra_state(self, new_extra_state: int) -> None:
        # [FUNCTION DOC] HELPER: Property setter replacing the integer extra state.
        self._extra_state = new_extra_state

    @property
    def extra_state_tensor(self) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Property returning the extra state tensor.
        return self._extra_state_tensor

    @extra_state_tensor.setter
    def extra_state_tensor(self, new_extra_state_tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Property setter replacing the extra state tensor.
        self._extra_state_tensor = new_extra_state_tensor

    def get_extra_state(self) -> dict[str, int | torch._tensor.Tensor]:
        # [FUNCTION DOC] HELPER: Module hook returning a mapping with the keys extra_state
        # and extra_state_tensor, which becomes the module's _extra_state state dict entry.
        return {
            "extra_state": self._extra_state,
            "extra_state_tensor": self._extra_state_tensor,
        }

    def set_extra_state(self, state: dict[str, int | torch._tensor.Tensor]) -> None:
        # [FUNCTION DOC] HELPER: Module hook restoring the int and tensor extra state from
        # the mapping produced by get_extra_state; raises KeyError if a key is missing.
        self._extra_state = state["extra_state"]  # pyre-ignore[8]
        self._extra_state_tensor = state["extra_state_tensor"]  # pyre-ignore[8]


class DTensorPlanner(DTensorTestBase):
    # [CLASS DOC] Multi-process test case (one process per GPU) exercising DCP save/load of
    # DTensor state dicts through the default planners.
    def create_dtensor_model(
        self,
        tensor_to_shard: torch.tensor,
        tensor_to_replicate: torch.tensor,
    ) -> torch.nn.Module:
        # [FUNCTION DOC] HELPER: Builds a MyTestModule on the current device type. The two
        # input tensors are distributed over a mesh of all ranks (one sharded on dim 0, one
        # replicated); two zero tensors of length SUBMESH_TENSOR_SIZE are created on a
        # submesh of even ranks (sharded and replicated). Returns a tuple of the model,
        # the sharded DTensor and the replicated DTensor.
        # KNOWN ISSUE: the return annotation says Module but a 3-tuple is returned.
        mesh = DeviceMesh(
            device_type=self.device_type,
            mesh=range(dist.get_world_size()),
        )
        sharded_dt = distribute_tensor(tensor_to_shard, mesh, placements=[Shard(0)])
        replicated_dt = distribute_tensor(
            tensor_to_replicate, mesh, placements=[Replicate()]
        )

        # Only even rank will be part of the mesh.
        submesh = DeviceMesh(
            device_type=self.device_type,
            mesh=[i for i in range(dist.get_world_size()) if i % 2 == 0],
        )
        submesh_tensor_size = [SUBMESH_TENSOR_SIZE]
        submesh_sharded_dt = zeros(
            submesh_tensor_size,
            device_mesh=submesh,
            placements=[Shard(0)],
        )
        submesh_replicated_dt = zeros(
            submesh_tensor_size, device_mesh=submesh, placements=[Replicate()]
        )

        model = MyTestModule(
            sharded_dt,
            replicated_dt,
            submesh_sharded_dt,
            submesh_replicated_dt,
        ).to(self.device_type)

        return (
            model,
            sharded_dt,
            replicated_dt,
        )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_distributed_tensor_planner(self) -> None:
        # [FUNCTION DOC] Saves the model state dict with DefaultSavePlanner, rebuilds the
        # model with values scaled by ten, loads the checkpoint into the new state dict
        # with DefaultLoadPlanner, and checks per key: full-mesh sharded and replicated
        # local tensors equal the originals, submesh tensors are zeros on even ranks and
        # empty on odd ranks, and extra state is back to its saved values. Needs at least
        # 2 GPUs; uses a temp directory as the checkpoint path.
        CHECKPOINT_DIR = self.temp_dir

        local_tensor = torch.arange(0, 4, dtype=torch.float32)
        local_tensor_2 = torch.arange(4, 8, dtype=torch.float32)
        (model, sharded_dt, replicated_dt) = self.create_dtensor_model(
            local_tensor, local_tensor_2
        )
        state_dict = model.state_dict()

        """
        When the model is initialized, the state_dict on rank 0 are as follows when there are 4 GPUs.
        rank 0:
            OrderedDict(
                [
                    (
                        'rdt',
                        DTensor(
                            local_tensor=tensor([4., 5., 6., 7.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 1, 2, 3]),
                            placements=[Replicate()]
                        )
                    ),
                    (
                        'sdt',
                        DTensor(
                            local_tensor=tensor([0.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 1, 2, 3]),
                            placements=[Shard(dim=0)])
                        ),
                    ),
                    (
                        'submesh_sdt',
                        DTensor(
                            local_tensor=tensor([8., 9.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 2]),
                            placements=[Shard(dim=0)]
                        ),
                    ),
                    (
                        'submesh_rdt',
                        DTensor(
                            local_tensor=tensor([12., 13., 14., 15.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 2]),
                            placements=[Replicate()]
                        )
                    ),
                    (
                        '_extra_state',
                        {'extra_state': 1, 'extra_state_tensor': tensor([0.])}
                    )
                ]
            )
        """

        dist_cp.save(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=dist_cp.DefaultSavePlanner(),
        )
        model, _, _ = self.create_dtensor_model(local_tensor * 10, local_tensor_2 * 10)
        state_dict = model.state_dict()

        """
        When the model is re-initialized, we have changed the params in state_dict.
        The updated values are as follows, when there are 4 GPUs:
        rank 0:
            OrderedDict(
                [
                    (
                        'rdt',
                        DTensor(
                            local_tensor=tensor([40., 50., 60., 70.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 1, 2, 3]),
                            placements=[Replicate()],
                        )
                    ),
                    (
                        'sdt',
                        DTensor(
                            local_tensor=tensor([0.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 1, 2, 3]),
                            placements=[Shard(dim=0)],
                        )
                    ),
                    (
                        'submesh_sdt',
                        DTensor(
                            local_tensor=tensor([80., 90.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 2]),
                            placements=[Shard(dim=0)]
                        )
                    ),
                    ('submesh_rdt',
                        DTensor(
                            local_tensor=tensor([120., 130., 140., 150.], device=f'{self.device_type}:0'),
                            device_mesh=DeviceMesh:([0, 2]),
                            placements=[Replicate()]
                        )
                    ),
                    (
                        '_extra_state', {'extra_state': 10, 'extra_state_tensor': tensor([10.])}
                    )
                ]
            )
        """

        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=dist_cp.DefaultLoadPlanner(),
        )

        """
        After loading the model from the checkpoint, we want to make sure that the values in state_dict
        match the values that are originally saved to the checkpoint.
        """
        for k, v in state_dict.items():
            if k == "sdt":
                self.assertEqual(sharded_dt.to_local(), v.to_local())
            if k == "rdt":
                self.assertEqual(replicated_dt.to_local(), v.to_local())

            if k == "submesh_sdt":
                if self.rank % 2 == 0:
                    shard_size = int(SUBMESH_TENSOR_SIZE / v.device_mesh.size())
                    self.assertEqual(v.to_local().size(), torch.Size([shard_size]))
                    self.assertEqual(v.to_local(), torch.zeros([shard_size]))
                else:
                    self.assertEqual(v.to_local().size(), torch.Size([0]))
                    self.assertEqual(v.to_local(), torch.tensor([]))

            if k == "submesh_rdt":
                if self.rank % 2 == 0:
                    shard_size = SUBMESH_TENSOR_SIZE
                    self.assertEqual(v.to_local().size(), torch.Size([shard_size]))
                    self.assertEqual(v.to_local(), torch.zeros([shard_size]))
                else:
                    self.assertEqual(v.to_local().size(), torch.Size([0]))
                    self.assertEqual(v.to_local(), torch.tensor([]))

            if k == "_extra_state":
                self.assertEqual(1, v["extra_state"])
                self.assertEqual(torch.tensor([0.0]), v["extra_state_tensor"])


if __name__ == "__main__":
    run_tests()
