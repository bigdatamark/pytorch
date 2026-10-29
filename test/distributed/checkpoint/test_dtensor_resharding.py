# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-GPU tests for distributed checkpoint (DCP) resharding of DTensors.
# Covers placement changes on 1D and 2D device meshes, 1D<->2D mesh changes, empty and
# uneven shards, and a custom "checkpointable" tensor subclass and LocalShardsWrapper
# for uneven shards. Defines module-level placement lists (incl. all distinct 2D pairs).
# Import-time side effect: sets the module logger level to INFO.
import logging
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint._extension import ZStandard
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard, zeros
from torch.distributed.tensor._shards_wrapper import LocalShardsWrapper
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    get_test_extension_registry,
    Rot13Example,
    with_temp_dir,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHECKPOINT_DIR = "checkpoint"

ONE_D_PLACEMENTS = [
    [Shard(0)],
    [Replicate()],
]
ONE_D_TO_ONE_D_PLACEMENTS = [
    ([Replicate()], [Shard(0)]),
    ([Shard(0)], [Replicate()]),
]

TWO_D_PLACEMENTS = [
    [Replicate(), Replicate()],
    [Replicate(), Shard(0)],
    [Shard(0), Replicate()],
    [Shard(0), Shard(0)],
]
TWO_D_TO_TWO_D_PLACEMENTS = []
for p1 in TWO_D_PLACEMENTS:
    for p2 in TWO_D_PLACEMENTS:
        if p1 != p2:
            TWO_D_TO_TWO_D_PLACEMENTS.append((p1, p2))


@instantiate_parametrized_tests
class TestDTensorReshardPlacementChange(DTensorTestBase):
    # [CLASS DOC] Tests saving a DTensor with one placement and loading it into a DTensor
    # with a different placement on the same mesh shape and world size.
    """
    Test DCP reshard for DTensor with placements changes and without world_size change and mesh_tensor change.
    """

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    @parametrize("extensions", [None, [Rot13Example()], [ZStandard()]])
    def test_1d_to_1d_reshard_placement_change(self, extensions) -> None:
        # [FUNCTION DOC] For each (Replicate<->Shard(0)) pair on a 1D mesh, saves a 4x4
        # DTensor with FileSystemWriter (optionally using no extension, Rot13 or ZStandard),
        # loads into a zero DTensor of the new placement, then checks the redistributed
        # full tensor and the round-tripped original placement match. Needs 2+ GPUs.
        CHECKPOINT_DIR = self.temp_dir

        for one_d_to_one_d_placements in ONE_D_TO_ONE_D_PLACEMENTS:
            original_placement, new_placement = one_d_to_one_d_placements

            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (self.world_size,)
            device_mesh = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, device_mesh, placements=original_placement
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(
                    path=CHECKPOINT_DIR, _extensions=extensions
                ),
                planner=dist_cp.DefaultSavePlanner(),
            )

            zero_dtensor = zeros(
                [4, 4], device_mesh=device_mesh, placements=new_placement
            )
            state_dict_to_load = {"dtensor": zero_dtensor}

            dist_cp.load(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.FileSystemReader(
                    CHECKPOINT_DIR, _extension_registry=get_test_extension_registry()
                ),
                planner=dist_cp.DefaultLoadPlanner(),
            )

            # materialzie the whole tensor to compare with the original global_tensor
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=[Replicate()],
            )
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            # redistribute the tensor back to its original placement for comparison.
            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                device_mesh,
                placements=original_placement,
            )
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_2d_to_2d_reshard_placement_change(self) -> None:
        # [FUNCTION DOC] Same as the 1D test but on a (2, world_size/2) mesh, iterating over
        # every ordered pair of distinct 2D placements; saves, loads into a zero DTensor,
        # and verifies full-tensor equality and round trip back to the original placement.
        # Needs 4+ GPUs.
        CHECKPOINT_DIR = self.temp_dir
        for two_d_to_two_d_placements in TWO_D_TO_TWO_D_PLACEMENTS:
            original_placement, new_placement = two_d_to_two_d_placements

            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor,
                mesh_2d,
                placements=original_placement,
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            zero_dtensor = zeros([4, 4], device_mesh=mesh_2d, placements=new_placement)
            state_dict_to_load = {"dtensor": zero_dtensor}

            dist_cp.load(
                state_dict=state_dict_to_load,
                storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                planner=dist_cp.DefaultLoadPlanner(),
            )

            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=[Replicate(), Replicate()],
            )
            self.assertEqual(global_tensor, state_dict_to_load["dtensor"].to_local())

            state_dict_to_load["dtensor"] = state_dict_to_load["dtensor"].redistribute(
                mesh_2d,
                placements=original_placement,
            )
            self.assertEqual(
                state_dict_to_save["dtensor"].to_local(),
                state_dict_to_load["dtensor"].to_local(),
            )


class TestDTensorReshardMeshChange(DTensorTestBase):
    # [CLASS DOC] Tests DCP loading into DTensors whose device mesh dimensionality differs
    # from the one used at save time (1D to 2D and 2D to 1D), plus uneven/empty shard cases.
    """
    Test DCP reshard for DTensor with placements changes and mesh_tensor change.
    """

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_1d_to_2d_reshard_mesh_change(self) -> None:
        # [FUNCTION DOC] Saves a 4x4 DTensor on a 1D mesh (Shard(0) or Replicate), then for
        # every 2D placement loads into a zero DTensor on a 2D mesh and checks the fully
        # replicated result equals the original global tensor. Needs 2+ GPUs.
        CHECKPOINT_DIR = self.temp_dir
        for placements_1d in ONE_D_PLACEMENTS:
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (self.world_size,)
            mesh_1d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, mesh_1d, placements=placements_1d
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            for placements_2d in TWO_D_PLACEMENTS:
                mesh_shape = (2, self.world_size // 2)
                mesh_2d = init_device_mesh(self.device_type, mesh_shape)

                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_2d, placements=placements_2d
                )
                state_dict_to_load = {"dtensor": zero_dtensor}

                dist_cp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # materialzie the whole tensor to compare with the original global_tensor
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_2d,
                    placements=[Replicate(), Replicate()],
                )
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_2d_to_1d_reshard_mesh_change(self) -> None:
        # [FUNCTION DOC] Inverse of the 1D to 2D test: saves on a 2D mesh with each 2D
        # placement, loads into 1D-mesh DTensors (Shard(0) or Replicate), and compares the
        # fully replicated tensor to the global tensor. Needs 4+ GPUs.
        CHECKPOINT_DIR = self.temp_dir
        for placements_2d in TWO_D_PLACEMENTS:
            global_tensor = torch.arange(16, dtype=torch.float).view(4, 4)
            mesh_shape = (2, self.world_size // 2)
            mesh_2d = init_device_mesh(self.device_type, mesh_shape)
            dtensor = distribute_tensor(
                global_tensor, mesh_2d, placements=placements_2d
            )
            state_dict_to_save = {"dtensor": dtensor}

            dist_cp.save(
                state_dict=state_dict_to_save,
                storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=dist_cp.DefaultSavePlanner(),
            )

            for placements_1d in ONE_D_PLACEMENTS:
                mesh_shape = (self.world_size,)
                mesh_1d = init_device_mesh(self.device_type, mesh_shape)

                zero_dtensor = zeros(
                    [4, 4], device_mesh=mesh_1d, placements=placements_1d
                )
                state_dict_to_load = {"dtensor": zero_dtensor}

                dist_cp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
                    planner=dist_cp.DefaultLoadPlanner(),
                )

                # materialzie the whole tensor to compare with the original global_tensor
                state_dict_to_load["dtensor"] = state_dict_to_load[
                    "dtensor"
                ].redistribute(
                    mesh_1d,
                    placements=[Replicate()],
                )
                self.assertEqual(
                    global_tensor, state_dict_to_load["dtensor"].to_local()
                )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(2)
    def test_dtensor_checkpoint_resharding_with_empty_shard(self):
        # [FUNCTION DOC] Saves a one-element tensor sharded over all ranks (so most ranks
        # hold empty shards), then loads it into a 2D mesh with Shard(0) on both dims.
        # Passes if save and load complete without error; no value check is made.
        """
        Test dtensor checkpoint resharding with dtensor containing empty shards.
        """
        tensor = torch.rand(1).to(self.device_type)
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])
        ref_state_dict = {"dtensor": dtensor}

        dist_cp.save(
            state_dict=ref_state_dict,
            storage_writer=dist_cp.FileSystemWriter(path=self.temp_dir),
        )

        tensor = torch.rand(1).to(self.device_type)
        mesh_2 = init_device_mesh(self.device_type, (2, self.world_size // 2))
        dtensor = distribute_tensor(tensor, mesh_2, [Shard(0), Shard(0)])
        state_dict = {"dtensor": dtensor}
        dist_cp.load(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(self.temp_dir),
        )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_dtensor_checkpoint_with_uneven_shards(self) -> None:
        # [FUNCTION DOC] Builds a 14x1 DTensor from per-rank local tensors (4 rows each, the
        # last rank only 2), saves it, then loads into a plain CPU full tensor (resharding
        # load). Prints the loaded tensor per rank; asserts nothing beyond no exceptions.
        # Needs 4 GPUs: the row counts assume world size 4.
        """
        Saving a dtensor with uneven shards.
        rank 0  -> [[0], [1], [2], [3]]
        rank 1  -> [[4], [5], [6], [7]]
        rank 2  -> [[8], [9], [10], [11]]
        rank 3  -> [[12], [13]]
        """
        CHECKPOINT_DIR = self.temp_dir
        mesh_shape = (self.world_size,)
        mesh_1 = init_device_mesh(self.device_type, mesh_shape)
        my_rank = dist.get_rank()
        # Make the last shard uneven
        if my_rank == self.world_size - 1:
            local_tensor = torch.arange(
                start=my_rank * 4, end=(my_rank * 4) + 2, dtype=torch.float
            ).view(2, 1)
        else:
            local_tensor = torch.arange(
                start=my_rank * 4, end=(my_rank + 1) * 4, dtype=torch.float
            ).view(4, 1)
        dtensor = DTensor.from_local(
            local_tensor,
            mesh_1,
            [Shard(0)],
            run_check=True,
            shape=torch.Size([14, 1]),
            stride=torch.Size([1, 1]),
        )

        state_dict_to_save = {"uneven_sharded_dtensor": dtensor}

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=dist_cp.DefaultSavePlanner(),
        )

        loading_full_tensor = torch.rand([14, 1], dtype=torch.float, device="cpu")
        print(f"rank {my_rank} loading_dtensor for load :\n {loading_full_tensor}")
        state_dict_to_load = {
            "uneven_sharded_dtensor": loading_full_tensor
        }  # re-sharding load.
        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.FileSystemReader(self.temp_dir),
        )


class CheckpointableDistTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Wrapper tensor subclass holding one local shard plus its fqn,
    # offsets and overall size. It implements the DCP checkpointable-tensor protocol
    # (chunk list, write items, shard lookup) so uneven shards can be saved/loaded. Any
    # dispatched torch op on it raises NotImplementedError, so it is not usable in training.
    """
    A distributed checkpointable tensor representation. Unlike Dtensor, this representation
    cannot be used for distributed training.

    Supports distributed tensor save/loads that has uneven shards. (DTensor cannot support the same)
    """

    _local_tensor: torch.Tensor
    _shard_offsets: torch.Size
    _overall_size: torch.Size

    @staticmethod
    def __new__(
        cls,
        fqn: str,
        local_tensor: torch.Tensor,
        shard_offsets: list[int],
        overall_size: list[int],
    ) -> "CheckpointableDistTensor":
        # [FUNCTION DOC] HELPER: Creates a wrapper subclass instance sized to overall_size
        # with the local tensor's dtype, device and layout; stores fqn, the local tensor,
        # shard offsets and overall size as attributes. Returns the new instance.
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            overall_size,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
        )

        r._fqn = fqn
        r._local_tensor = local_tensor
        r._shard_offsets = torch.Size(shard_offsets)
        r._overall_size = torch.Size(overall_size)

        return r

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Ignores all arguments and calls the base initializer.
        super().__init__()

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Always raises NotImplementedError naming the op; no
        # operation is supported on this tensor type.
        raise NotImplementedError(
            f"{func} is not supported for CheckpointableDistTensor!"
        )

    def __create_chunk_list__(self):
        # [FUNCTION DOC] HELPER: Returns a one-element list with a ChunkStorageMetadata
        # describing this shard (its offsets and the local tensor's size).
        return [
            ChunkStorageMetadata(
                offsets=self._shard_offsets, sizes=self._local_tensor.size()
            )
        ]

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        # [FUNCTION DOC] HELPER: Returns one SHARD-type WriteItem for this shard, indexed by
        # the stored fqn and offsets, with chunk, tensor properties and overall size.
        # The fqn and object arguments are ignored; the stored fqn is used instead.
        return [
            WriteItem(
                index=MetadataIndex(fqn=self._fqn, offset=self._shard_offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=self._shard_offsets, sizes=self._local_tensor.size()
                    ),
                    properties=TensorProperties.create_from_tensor(self._local_tensor),
                    size=self._overall_size,
                ),
            )
        ]

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns the local tensor if the index fqn and offset match
        # this shard; otherwise raises AssertionError describing the mismatch.
        if not (self._fqn == index.fqn and self._shard_offsets == index.offset):
            raise AssertionError(
                f"fqn/offset mismatch: {self._fqn} vs {index.fqn}, "
                f"{self._shard_offsets} vs {index.offset}"
            )
        return self._local_tensor

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Builds a descriptive string with fqn, local tensor, shard
        # offset and overall size.
        # KNOWN ISSUE: reads a nonexistent attribute named _shard_offset (the stored one is
        # _shard_offsets), so calling repr raises AttributeError.
        return (
            f"CheckpointableDistributedTensor("
            f"fqn={self._fqn}, "
            f"local_tensor={self._local_tensor}, "
            f"shard_offset={self._shard_offset}, "
            f"overall_size={self._overall_size})"
        )


class TestCheckpointableReshard(DTensorTestBase):
    # [CLASS DOC] Tests DCP resharding loads where per-rank shard sizes are uneven and the
    # save-time and load-time shard plans differ, via the checkpointable tensor API and
    # via DTensor wrapping a LocalShardsWrapper.
    """
    Test DCP reshard loads when shard sizes are uneven across the ranks.
    """

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_uneven_reshard_with_checkpointable_api(self) -> None:
        # [FUNCTION DOC] Each rank saves its slice (per a fixed offset/length plan) of a
        # 16x1 tensor as a CheckpointableDistTensor, then loads a different plan's slice
        # into a random tensor and asserts it equals the expected arange values.
        # Requires exactly 4 ranks (plans have four entries).
        """
        Saves a 1d distributed tensor that has shards with uneven sizes using Checkpointable API.
        Loads them back with a different shard plan (resharding). By default this UT runs with
        NUM_DEVICES = 4.
        """
        saving_1d_shard_plan = [
            (0, 4),
            (4, 3),
            (7, 4),
            (11, 5),
        ]  # offset, length tuples.
        loading_1d_shard_plan = [(0, 2), (2, 4), (6, 6), (12, 4)]
        CHECKPOINT_DIR = self.temp_dir
        my_rank = dist.get_rank()
        saving_shard_offset, saving_shard_length = saving_1d_shard_plan[my_rank]
        saving_local_tensor = torch.arange(
            start=saving_shard_offset,
            end=saving_shard_offset + saving_shard_length,
            dtype=torch.float,
        ).view(saving_shard_length, 1)
        logger.info(f"[{my_rank}] saving_local_tensor : {saving_local_tensor}")  # noqa: G004
        saving_cp_dist_tensor = CheckpointableDistTensor(
            fqn="checkpointable_tensor",
            local_tensor=saving_local_tensor,
            shard_offsets=[saving_shard_offset, 0],
            overall_size=[16, 1],
        )
        state_dict_to_save = {"checkpointable_tensor": saving_cp_dist_tensor}

        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=dist_cp.DefaultSavePlanner(),
        )

        loading_shard_offset, loading_shard_length = loading_1d_shard_plan[my_rank]
        loading_local_tensor = torch.rand([loading_shard_length, 1], dtype=torch.float)
        logger.info(
            f"[{my_rank}] loading_local_tensor (initialized with random vals) : {loading_local_tensor}"  # noqa: G004
        )
        expected_loaded_local_val_tensor = torch.arange(
            start=loading_shard_offset,
            end=loading_shard_offset + loading_shard_length,
            dtype=torch.float,
        ).view(loading_shard_length, 1)

        loading_cp_dist_tensor = CheckpointableDistTensor(
            fqn="checkpointable_tensor",
            local_tensor=loading_local_tensor,
            shard_offsets=[loading_shard_offset, 0],
            overall_size=[16, 1],
        )
        state_dict_to_load = {"checkpointable_tensor": loading_cp_dist_tensor}
        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.FileSystemReader(self.temp_dir),
        )
        if not torch.equal(loading_local_tensor, expected_loaded_local_val_tensor):
            raise AssertionError(
                "Expected loading_local_tensor to equal expected_loaded_local_val_tensor"
            )

    @with_comms
    @with_temp_dir
    @skip_if_lt_x_gpu(4)
    def test_uneven_reshard_with_dtensor_shards_wrapper_api(self) -> None:
        # [FUNCTION DOC] Same scenario as the checkpointable API test, but each shard is a
        # LocalShardsWrapper inside a DTensor on a CPU 1D mesh (Shard(0), global 16x1).
        # Saves with one plan, loads with another, asserts the loaded values, then
        # barriers. Requires exactly 4 ranks. (Its docstring wrongly says Checkpointable API.)
        """
        Saves a 1d distributed tensor that has shards with uneven sizes using Checkpointable API.
        Loads them back with a different shard plan (resharding). By default this UT runs with
        NUM_DEVICES = 4.
        """
        # NB: saving shardin plan and loading sharding plans are different and their
        #     shard lengths are uneven.
        saving_1d_shard_plan = [
            (0, 4),
            (4, 3),
            (7, 4),
            (11, 5),
        ]  # offset, length tuples.
        loading_1d_shard_plan = [(0, 6), (6, 2), (8, 1), (9, 7)]
        cp_path = self.temp_dir
        my_rank = dist.get_rank()

        # 1d device mesh on CPU device
        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh("cpu", mesh_shape)

        saving_shard_offset, saving_shard_length = saving_1d_shard_plan[my_rank]
        saving_local_tensor = torch.arange(
            start=saving_shard_offset,
            end=saving_shard_offset + saving_shard_length,
            dtype=torch.float,
        ).view(saving_shard_length, 1)

        # In order to support uneven shards we have to wrap the original shards in LocalShardsWrapper.
        saving_local_shard_wrapper = LocalShardsWrapper(
            local_shards=[saving_local_tensor], local_offsets=[(saving_shard_offset, 0)]
        )

        logger.info(
            f"[{my_rank}] saving_local_shard_warpper : {saving_local_shard_wrapper}"  # noqa: G004
        )

        saving_cp_dist_tensor = DTensor.from_local(
            local_tensor=saving_local_shard_wrapper,
            device_mesh=device_mesh,
            placements=[Shard(0)],
            shape=torch.Size([16, 1]),
            stride=torch.Size([1, 1]),
        )

        # put the DTensor in a state dict and call DCP save.
        state_dict_to_save = {"checkpointable_tensor": saving_cp_dist_tensor}
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(path=cp_path),
            planner=dist_cp.DefaultSavePlanner(),
        )

        loading_shard_offset, loading_shard_length = loading_1d_shard_plan[my_rank]
        loading_local_tensor = torch.rand(
            [loading_shard_length, 1], dtype=torch.float, device="cpu"
        )
        loading_local_shard_wrapper = LocalShardsWrapper(
            local_shards=[loading_local_tensor],
            local_offsets=[(loading_shard_offset, 0)],
        )

        expected_loaded_local_val_tensor = torch.arange(
            start=loading_shard_offset,
            end=loading_shard_offset + loading_shard_length,
            dtype=torch.float,
        ).view(loading_shard_length, 1)

        loading_cp_dist_tensor = DTensor.from_local(
            local_tensor=loading_local_shard_wrapper,
            device_mesh=device_mesh,
            placements=[Shard(0)],
            shape=torch.Size([16, 1]),
            stride=torch.Size([1, 1]),
        )
        state_dict_to_load = {"checkpointable_tensor": loading_cp_dist_tensor}

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp.FileSystemReader(path=cp_path),
        )
        logger.info(
            f"[{my_rank}] loaded_shards_wrapper : {loading_local_shard_wrapper}"  # noqa: G004
        )
        if not torch.equal(loading_local_tensor, expected_loaded_local_val_tensor):
            raise AssertionError(
                "Expected loading_local_tensor to equal expected_loaded_local_val_tensor"
            )
        dist.barrier()


# TODO: Add dtensor resharding test when world size changes.
if __name__ == "__main__":
    run_tests()
