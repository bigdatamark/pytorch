# Owner(s): ["oncall: distributed"]
# [MODULE DOC]
# Tests for the distributed checkpoint planners and plan helpers: default local/global save
# plans, plan deduplication, save plan caching in SavePlanner class-level caches, load plans
# including resharding across different world sizes, read item creation from chunk lists,
# merging delta plans, global plan validation, strict/partial loading in DefaultLoadPlanner,
# and a multi-process save/load of a CheckpointableTensor. Exits the whole process at import
# if running under dev-asan (multiprocessing spawn is unsupported there). Provides the helper
# create_sharded_tensor used to build one-dimensional ShardedTensors without real comms.

import sys
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import (
    TensorProperties as TensorProperties_Shard,
)
from torch.distributed.checkpoint import CheckpointableTensor
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import (
    _create_default_local_metadata,
    _validate_global_plan,
    create_default_global_save_plan,
    create_default_local_load_plan,
    create_default_local_save_plan,
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.filesystem import CURRENT_DCP_VERSION
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    SavePlan,
    SavePlanner,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _compare_save_plans,
    _merge_delta_local_plans,
    create_read_items_for_chunk_list,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed.distributed_utils import (
    with_dist,
    with_fake_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def create_sharded_tensor(rank, world_size, shards_per_rank, shard_size=8):
    # [FUNCTION DOC] HELPER: builds a 1D ShardedTensor of shards_per_rank times world_size
    # shards, each shard_size elements, placed consecutively (shard i owned by rank i divided
    # by shards_per_rank, on cpu). Only the shards owned by the given rank get real random
    # local data; metadata describes all shards. Returns the ShardedTensor assembled from the
    # local shards and global metadata (no communication performed).
    shards_metadata = []
    local_shards = []
    for idx in range(world_size * shards_per_rank):
        shard_rank = idx // shards_per_rank
        shard_md = ShardMetadata(
            shard_offsets=[idx * shard_size],
            shard_sizes=[shard_size],
            placement=f"rank:{shard_rank}/cpu",
        )
        shards_metadata.append(shard_md)
        if shard_rank == rank:
            shard = Shard.from_tensor_and_offsets(
                torch.rand(*shard_md.shard_sizes),
                shard_offsets=shard_md.shard_offsets,
                rank=rank,
            )
            local_shards.append(shard)

    sharded_tensor_md = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=torch.Size([shard_size * len(shards_metadata)]),
        tensor_properties=TensorProperties_Shard.create_from_tensor(torch.zeros(1)),
    )

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md
    )


class TestCheckpointableTensorDistributed(DTensorTestBase):
    # [CLASS DOC] Multi-process test case (4 ranks, cpu) for saving and loading plain tensors
    # that carry checkpointable-tensor attributes (global shape and offsets, local offsets and
    # sizes) through the default planners.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of test processes, fixed at 4.
        return 4

    @property
    def device_type(self) -> str:
        # [FUNCTION DOC] Returns the device type for the test, always cpu.
        return "cpu"

    @with_comms
    @with_temp_dir
    def test_checkpointable_tensor_shard_save_load(self):
        # [FUNCTION DOC] Each rank annotates a small float tensor as a shard of a global 1D
        # tensor, saves it with dcp.save, and the checkpoint metadata is checked: one tensor
        # entry of full size with one chunk per rank. Then a meta-device target with the same
        # annotations is loaded in place by dcp.load; asserts the result is a real cpu tensor
        # with the expected shape attributes and values equal to what was saved.
        shard_size = 4
        rank = dist.get_rank()
        start = rank * shard_size
        expected = torch.arange(start, start + shard_size, dtype=torch.float32)

        tensor = expected.clone()
        tensor.global_shape = (self.world_size * shard_size,)
        tensor.global_offsets = ((start,),)
        tensor.local_offsets = ((0,),)
        tensor.local_sizes = ((shard_size,),)
        self.assertIsInstance(tensor, CheckpointableTensor)

        dcp.save({"proto": tensor}, checkpoint_id=self.temp_dir)
        dist.barrier()

        metadata = dcp.FileSystemReader(self.temp_dir).read_metadata()
        tensor_metadata = metadata.state_dict_metadata["proto"]
        self.assertIsInstance(tensor_metadata, TensorStorageMetadata)
        self.assertEqual(
            torch.Size([self.world_size * shard_size]), tensor_metadata.size
        )
        self.assertEqual(
            [
                ChunkStorageMetadata(
                    offsets=torch.Size([rank * shard_size]),
                    sizes=torch.Size([shard_size]),
                )
                for rank in range(self.world_size)
            ],
            sorted(tensor_metadata.chunks, key=lambda chunk: tuple(chunk.offsets)),
        )

        target = torch.empty(shard_size, device="meta")
        target.global_shape = (self.world_size * shard_size,)
        target.global_offsets = ((start,),)
        target.local_offsets = ((0,),)
        target.local_sizes = ((shard_size,),)
        state_dict = {"proto": target}

        dcp.load(state_dict, checkpoint_id=self.temp_dir)

        loaded = state_dict["proto"]
        self.assertFalse(loaded.is_meta)
        self.assertEqual(torch.device("cpu"), loaded.device)
        self.assertIsInstance(loaded, CheckpointableTensor)
        self.assertEqual(torch.Size([shard_size]), loaded.size())
        self.assertEqual((self.world_size * shard_size,), loaded.global_shape)
        self.assertEqual(((start,),), loaded.global_offsets)
        self.assertEqual(expected, loaded)


class TestSavePlan(TestCase):
    # [CLASS DOC] Single-process tests of save and load planning for a state dict containing
    # a tensor, a Python list and a ShardedTensor; process group state is faked per rank.
    # Caching tests mutate SavePlanner class-level caches and clean them up at the end.
    @with_fake_comms(rank=1, world_size=4)
    def test_local_plan(self):
        # [FUNCTION DOC] With rank 1 of 4, builds the default local save plan for the
        # state dict. Non-coordinator plan has three items; checks the tensor item (tensor
        # write type, size, properties, full chunk) and the sharded tensor item (shard type,
        # offset 8, chunk size 8). The coordinator plan also has three items and includes the
        # tensor item and a byte stream item for the list with no tensor data.
        tensor = torch.rand(10)
        val = [1, 2, 3]
        st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)
        state_dict = {"tensor": tensor, "value": val, "st": st}
        plan = create_default_local_save_plan(state_dict, False)
        self.assertEqual(3, len(plan.items))
        wi = plan.items[0]
        self.assertEqual(wi.index, MetadataIndex("tensor", [0]))
        self.assertEqual(wi.type, WriteItemType.TENSOR)
        self.assertEqual(wi.tensor_data.size, tensor.size())
        self.assertEqual(
            wi.tensor_data.properties,
            TensorProperties.create_from_tensor(torch.zeros(1)),
        )
        self.assertEqual(wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(wi.tensor_data.chunk.sizes, torch.Size([10]))

        st_wi = plan.items[2]
        self.assertEqual(st_wi.index, MetadataIndex("st", [8]))
        self.assertEqual(st_wi.type, WriteItemType.SHARD)
        self.assertEqual(st_wi.tensor_data.size, st.size())
        self.assertEqual(
            st_wi.tensor_data.properties,
            TensorProperties.create_from_tensor(torch.zeros(1)),
        )
        self.assertEqual(st_wi.tensor_data.chunk.offsets, torch.Size([8]))
        self.assertEqual(st_wi.tensor_data.chunk.sizes, torch.Size([8]))

        # Coordinator rank, should include replicated items as well
        plan = create_default_local_save_plan(state_dict, True)
        self.assertEqual(3, len(plan.items))

        tensor_wi = next(wi for wi in plan.items if wi.type == WriteItemType.TENSOR)
        self.assertEqual(tensor_wi.index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_wi.tensor_data.size, tensor.size())
        self.assertEqual(
            tensor_wi.tensor_data.properties,
            TensorProperties.create_from_tensor(tensor),
        )
        self.assertEqual(tensor_wi.tensor_data.chunk.offsets, torch.Size([0]))
        self.assertEqual(tensor_wi.tensor_data.chunk.sizes, torch.Size([10]))

        bytes_wi = next(wi for wi in plan.items if wi.type == WriteItemType.BYTE_IO)
        self.assertEqual(bytes_wi.index, MetadataIndex("value"))
        self.assertIsNone(bytes_wi.tensor_data)

    @with_fake_comms(rank=1, world_size=4)
    def test_local_plan_with_caching(self):
        # [FUNCTION DOC] Uses a save planner with plan caching. The first local plan stays
        # pending (not in the class-level cache) until finish_plan promotes it; after that the
        # cache holds an equal plan. A second create_local_plan returns an empty plan marked
        # unusable with no planner or storage data. Removes the cache entries it created.
        tensor = torch.rand(10)
        val = [1, 2, 3]
        st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)
        state_dict = {"tensor": tensor, "value": val, "st": st}
        planner = DefaultSavePlanner(enable_plan_caching=True)
        planner.set_up_planner(state_dict, is_coordinator=False)
        # First iteration, should create a new plan
        first_plan = planner.create_local_plan()

        # Plan should be pending, not yet in the class-level cache
        self.assertNotIn(planner._cached_plans_key, SavePlanner._cached_save_plan)

        # Promote the pending plan by calling finish_plan
        planner.finish_plan(first_plan)

        # Validate that the plan has been cached after finish_plan
        cached_plan = SavePlanner._cached_save_plan[planner._cached_plans_key]
        self.assertEqual(first_plan, cached_plan)

        # second iteration, should create an empty unusable plan
        second_plan = planner.create_local_plan()
        self.assertFalse(second_plan.usable)
        self.assertEqual(0, len(second_plan.items))
        self.assertIsNone(second_plan.planner_data)
        self.assertIsNone(second_plan.storage_data)

        # Clean up class-level caches
        key = planner._cached_plans_key
        SavePlanner._cached_save_plan.pop(key, None)
        SavePlanner._cached_final_save_plan.pop(key, None)

    def test_global_plan(self):
        # [FUNCTION DOC] Builds local save plans for four simulated ranks, dedups them and
        # creates the default global plan. Asserts each item keeps its type and tensor data,
        # the global metadata has an entry (bytes or tensor kind) for each item, tensor
        # metadata matches size and properties, and the index hint set by the global plan
        # points to the matching chunk.
        def create_data(rank):
            # [FUNCTION DOC] HELPER: under a faked process group for the rank (world size 4),
            # builds a state dict (tensor, list, sharded tensor) and returns its default local
            # save plan; rank 0 acts as coordinator.
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        all_plans = dedup_save_plans(all_plans)
        final_plans, metadata = create_default_global_save_plan(all_plans=all_plans)

        # The default global plan updates all indexes to include hints
        for new_plan, old_plan in zip(final_plans, all_plans):
            for new_item, old_item in zip(new_plan.items, old_plan.items):
                self.assertEqual(new_item.index, old_item.index)
                self.assertEqual(new_item.type, old_item.type)
                self.assertEqual(new_item.tensor_data, old_item.tensor_data)
                self.assertIn(new_item.index.fqn, metadata.state_dict_metadata)

                item_md = metadata.state_dict_metadata[new_item.index.fqn]
                if new_item.type == WriteItemType.BYTE_IO:
                    self.assertTrue(isinstance(item_md, BytesStorageMetadata))
                else:
                    self.assertTrue(isinstance(item_md, TensorStorageMetadata))
                    self.assertEqual(item_md.size, old_item.tensor_data.size)
                    self.assertEqual(
                        item_md.properties, old_item.tensor_data.properties
                    )

                    self.assertIsNotNone(new_item.index.index)
                    # Make sure the hint is correct
                    self.assertEqual(
                        item_md.chunks[new_item.index.index], old_item.tensor_data.chunk
                    )

    def test_dedup_plans(self):
        # [FUNCTION DOC] Dedups the four ranks' local plans; asserts the plan count is
        # unchanged, each deduped plan differs from its original, and has fewer items
        # (replicated items are kept on one rank only).
        def create_data(rank):
            # [FUNCTION DOC] HELPER: returns the default local save plan for a simulated rank
            # of 4 (tensor, list, sharded tensor); rank 0 is coordinator.
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        deduped_plans = dedup_save_plans(all_plans)

        # Number of plans should remain unchanged
        self.assertEqual(len(all_plans), len(deduped_plans))

        # Number of items in the deduped plans should be less than the original plans
        for new_plan, old_plan in zip(deduped_plans, all_plans):
            self.assertFalse(_compare_save_plans(new_plan, old_plan))
            self.assertTrue(len(new_plan.items) < len(old_plan.items))

    def test_global_plan_with_caching(self):
        # [FUNCTION DOC] Exercises global plan caching across three iterations with a caching
        # planner. Iteration one caches the global plan, the input plans and the metadata.
        # Iteration two with identical plans returns plans that are empty and unusable, with
        # the same metadata. Iteration three with a changed (larger) tensor returns a usable
        # nonempty plan for rank 0 only, new metadata, and refreshed caches. Cleans up the
        # class-level caches at the end.
        # KNOWN ISSUE: the comment before the second iteration checks says the plans are
        # "empty and usable" but the assertions require them to be unusable.
        def create_data(rank):
            # [FUNCTION DOC] HELPER: with a faked group for the rank, sets up a caching save
            # planner with the standard state dict (tensor of 10 elements) and returns its
            # local plan; rank 0 is coordinator.
            with with_dist(rank=rank, world_size=4):
                planner = DefaultSavePlanner(enable_plan_caching=True)
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
                return planner.create_local_plan()

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        planner = DefaultSavePlanner(enable_plan_caching=True)
        # First iteration, should create a new plan
        first_global_plan, first_metadata = planner.create_global_plan(all_plans)

        # Validate that the plan has been cached
        cached_global_plan = SavePlanner._cached_global_plan[planner._cached_plans_key]
        self.assertEqual(cached_global_plan, first_global_plan)

        # Validate that all_plans are cached
        cached_all_plans = SavePlanner._cached_all_plans[planner._cached_plans_key]
        self.assertEqual(cached_all_plans, all_plans)

        # Second iteration, should return empty plans
        # Recreate the plans as the previous ones are deduped.
        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        second_global_plan, second_metadata = planner.create_global_plan(all_plans)
        # All the plans should be empty and usable
        for plan in second_global_plan:
            self.assertFalse(plan.usable)
            self.assertEqual(0, len(plan.items))
            self.assertIsNone(plan.planner_data)
            self.assertIsNone(plan.storage_data)

        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(
            second_metadata, planner._cached_metadata[planner._cached_plans_key]
        )

        # Validate that all_plans are cached and remain unchanged.
        cached_all_plans = SavePlanner._cached_all_plans[planner._cached_plans_key]
        self.assertEqual(cached_all_plans, all_plans)

        # Third iteration with changed plans
        def create_data_v2(rank):
            # [FUNCTION DOC] HELPER: same as create_data but the tensor has 20 elements, so
            # the plan of the rank that holds the tensor differs from the earlier iterations.
            with with_dist(rank=rank, world_size=4):
                planner = DefaultSavePlanner(enable_plan_caching=True)
                tensor = torch.rand(20)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
                return planner.create_local_plan()

        all_plans = [
            create_data_v2(0),
            create_data_v2(1),
            create_data_v2(2),
            create_data_v2(3),
        ]
        third_global_plan, third_metadata = planner.create_global_plan(all_plans)
        # Only the rank 0 plan should be non-empty. The rest should be empty
        tensor_plan = third_global_plan[0]
        self.assertNotEqual(0, len(tensor_plan.items))
        self.assertTrue(tensor_plan.usable)

        # Validate that all_plans are updated and cached
        cached_all_plans = SavePlanner._cached_all_plans[planner._cached_plans_key]
        self.assertEqual(cached_all_plans, all_plans)

        for plan in third_global_plan[1:]:
            self.assertFalse(plan.usable)
            self.assertEqual(0, len(plan.items))
            self.assertIsNone(plan.planner_data)
            self.assertIsNone(plan.storage_data)

        # Global metadata should be different as one plan has changed
        self.assertNotEqual(second_metadata, third_metadata)
        # Validate that the metadata is cached
        self.assertEqual(
            third_metadata, planner._cached_metadata[planner._cached_plans_key]
        )

        # Validate that the new plan has been cached
        cached_global_plan = SavePlanner._cached_global_plan[planner._cached_plans_key][
            0
        ]
        self.assertEqual(cached_global_plan, tensor_plan)

        # Clean up class-level caches
        key = planner._cached_plans_key
        SavePlanner._cached_save_plan.pop(key, None)
        SavePlanner._cached_final_save_plan.pop(key, None)
        SavePlanner._cached_all_plans.pop(key, None)
        SavePlanner._cached_global_plan.pop(key, None)
        SavePlanner._cached_metadata.pop(key, None)

    def test_finish_plan_with_caching(self):
        # [FUNCTION DOC] With a caching planner as coordinator, finish_plan on a real plan
        # stores the result in the final-plan cache; a second finish_plan call given an
        # empty unusable plan returns the cached finished plan. Removes cache entries after.
        planner = DefaultSavePlanner(enable_plan_caching=True)
        tensor = torch.rand(10)
        val = [1, 2, 3]
        state_dict = {"tensor": tensor, "value": val}
        planner.set_up_planner(state_dict, is_coordinator=True)
        plan = planner.create_local_plan()

        # First iteration, should create a new plan
        first_finished_plan = planner.finish_plan(plan)

        # Validate that the plan has been cached
        cached_finished_plan = SavePlanner._cached_final_save_plan[
            planner._cached_plans_key
        ]
        self.assertEqual(first_finished_plan, cached_finished_plan)

        # second iteration, should return the cached plan
        second_finished_plan = planner.finish_plan(SavePlan([], usable=False))
        self.assertEqual(second_finished_plan, first_finished_plan)

        # Clean up class-level caches
        key = planner._cached_plans_key
        SavePlanner._cached_save_plan.pop(key, None)
        SavePlanner._cached_final_save_plan.pop(key, None)

    def test_caching_after_validation_failure(self):
        # [FUNCTION DOC] Regression test: when global plan validation reports errors (patched
        # to return one message), create_global_plan raises ValueError and must leave the
        # all-plans, global-plan, metadata and save-plan caches without the planner's key. A
        # second save with the same planner then succeeds (no KeyError) and fills the caches.
        # Cleans up the class-level caches at the end.
        def create_data(rank):
            # [FUNCTION DOC] HELPER: with a faked group for the rank, sets up a caching save
            # planner and returns its local plan; rank 0 is coordinator.
            with with_dist(rank=rank, world_size=4):
                planner = DefaultSavePlanner(enable_plan_caching=True)
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                planner.set_up_planner(state_dict, is_coordinator=(rank == 0))
                return planner.create_local_plan()

        planner = DefaultSavePlanner(enable_plan_caching=True)
        key = planner._cached_plans_key

        tensor = torch.rand(10)
        val = [1, 2, 3]
        state_dict = {"tensor": tensor, "value": val}
        planner.set_up_planner(state_dict, is_coordinator=True)

        local_plan = planner.create_local_plan()
        self.assertTrue(local_plan.usable)
        self.assertNotIn(key, SavePlanner._cached_save_plan)

        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]

        with patch(
            "torch.distributed.checkpoint.default_planner._validate_global_plan",
            return_value=["mock validation error"],
        ):
            with self.assertRaises(ValueError):
                planner.create_global_plan(all_plans)

        self.assertNotIn(key, SavePlanner._cached_all_plans)
        self.assertNotIn(key, SavePlanner._cached_global_plan)
        self.assertNotIn(key, SavePlanner._cached_metadata)
        self.assertNotIn(key, SavePlanner._cached_save_plan)

        # Second save should succeed, not KeyError
        planner.set_up_planner(state_dict, is_coordinator=True)
        local_plan_2 = planner.create_local_plan()
        self.assertTrue(local_plan_2.usable)

        global_plan, metadata = planner.create_global_plan(all_plans)
        self.assertIsNotNone(global_plan)
        self.assertIsNotNone(metadata)
        self.assertIn(key, SavePlanner._cached_all_plans)
        self.assertIn(key, SavePlanner._cached_global_plan)
        self.assertIn(key, SavePlanner._cached_metadata)

        # Clean up class-level caches
        SavePlanner._cached_save_plan.pop(key, None)
        SavePlanner._cached_final_save_plan.pop(key, None)
        SavePlanner._cached_all_plans.pop(key, None)
        SavePlanner._cached_global_plan.pop(key, None)
        SavePlanner._cached_metadata.pop(key, None)

    def test_local_load_plan(self):
        # [FUNCTION DOC] Creates default local metadata from rank 1's state dict and a local
        # load plan against it. Asserts three read items: the sharded tensor and the tensor
        # each copy from the identical storage index into the same destination with offset 0
        # and full length, and the list becomes a byte stream item.
        def create_state_dict(rank):
            # [FUNCTION DOC] HELPER: returns the standard state dict (tensor, list, sharded
            # tensor) built under a faked group for the rank with world size 4.
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                return {"tensor": tensor, "value": val, "st": st}

        state_dict = create_state_dict(1)
        metadata = _create_default_local_metadata(state_dict)

        load_plan = create_default_local_load_plan(state_dict, metadata)
        # This will create 3 entries
        self.assertEqual(3, len(load_plan.items))
        st_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "st")
        tensor_item = next(
            ri for ri in load_plan.items if ri.dest_index.fqn == "tensor"
        )
        bytes_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "value")

        self.assertEqual(st_item.type, LoadItemType.TENSOR)
        # This is an exact copy
        self.assertEqual(st_item.dest_index, MetadataIndex("st", [8]))
        self.assertEqual(st_item.dest_offsets, torch.Size([0]))
        self.assertEqual(st_item.storage_index, MetadataIndex("st", [8]))
        self.assertEqual(st_item.storage_offsets, torch.Size([0]))
        self.assertEqual(st_item.lengths, torch.Size([8]))

        self.assertEqual(tensor_item.type, LoadItemType.TENSOR)
        self.assertEqual(tensor_item.dest_index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_item.dest_offsets, torch.Size([0]))
        self.assertEqual(tensor_item.storage_index, MetadataIndex("tensor", [0]))
        self.assertEqual(tensor_item.storage_offsets, torch.Size([0]))
        self.assertEqual(tensor_item.lengths, torch.Size([10]))

        self.assertEqual(bytes_item.type, LoadItemType.BYTE_IO)
        self.assertEqual(bytes_item.dest_index, MetadataIndex("value"))

    def test_load_with_resharding(self):
        # [FUNCTION DOC] Resharding of a 128 element sharded tensor. Going from 8 ranks to 4,
        # rank 1's destination shard needs two read items, one from each of two source shards
        # with lengths 16. Going from 4 ranks to 8, rank 1 needs one read item that takes the
        # upper half (offset 16, length 16) of a single source shard.
        def create_state_dict(rank, world_size):
            # [FUNCTION DOC] HELPER: returns a state dict with a 1D sharded tensor of 128
            # elements split evenly across world_size ranks, built under a faked group.
            with with_dist(rank=rank, world_size=world_size):
                return {
                    "st": create_sharded_tensor(
                        rank=rank,
                        world_size=world_size,
                        shards_per_rank=1,
                        shard_size=128 // world_size,
                    )
                }

        # Rank 1 has a 16 bytes shard from [16, 32[
        world8_state_dict = create_state_dict(rank=1, world_size=8)
        world8_metadata = _create_default_local_metadata(world8_state_dict)

        # Rank 1 has a 32 bytes shard from [32, 64[
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        world4_metadata = _create_default_local_metadata(world4_state_dict)

        # First scenario, going from world=8 to world=4, need to load 2 shards
        # Each 4-world shard has 32 elements, so it needs to load 2 shards
        load_plan = create_default_local_load_plan(world4_state_dict, world8_metadata)
        self.assertEqual(2, len(load_plan.items))
        low_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])
        )
        high_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([16])
        )

        self.assertEqual(low_ri.storage_index, MetadataIndex("st", [32]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(low_ri.dest_index, MetadataIndex("st", [32]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([16]))

        self.assertEqual(high_ri.storage_index, MetadataIndex("st", [48]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex("st", [32]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([16]))
        self.assertEqual(high_ri.lengths, torch.Size([16]))

        # Second scenario, going from world=4 to world=8, need to load half of 1 shard
        # rank1 on 8-world needs to load the upper half of the rank0 4-world shard
        load_plan = create_default_local_load_plan(world8_state_dict, world4_metadata)
        self.assertEqual(1, len(load_plan.items))
        ri = load_plan.items[0]
        self.assertEqual(ri.storage_index, MetadataIndex("st", [0]))
        self.assertEqual(ri.storage_offsets, torch.Size([16]))
        self.assertEqual(ri.dest_index, MetadataIndex("st", [16]))
        self.assertEqual(ri.dest_offsets, torch.Size([0]))
        self.assertEqual(ri.lengths, torch.Size([16]))

    def test_load_with_world_size_diff_by_one(self):
        # [FUNCTION DOC] Loads a 120 element sharded tensor saved with 4 ranks into the
        # layout for 3 ranks (rank 1). Asserts two read items: one reading from the shard
        # at 30 with source offset 10 and length 20 into destination offset 0, and one from
        # the shard at 60 with source offset 0 and length 20 into destination offset 20.
        def create_state_dict(rank, world_size):
            # [FUNCTION DOC] HELPER: returns a state dict with a 1D sharded tensor of 120
            # elements split evenly across world_size ranks, built under a faked group.
            with with_dist(rank=rank, world_size=world_size):
                return {
                    "st": create_sharded_tensor(
                        rank=rank,
                        world_size=world_size,
                        shards_per_rank=1,
                        shard_size=120 // world_size,
                    )
                }

        # rank 1 has a 30 bytes shard from [30, 60[
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        world4_metadata = _create_default_local_metadata(world4_state_dict)

        # rank 1 has a 40 bytes shard from [40, 80[
        world3_state_dict = create_state_dict(rank=1, world_size=3)

        load_plan = create_default_local_load_plan(world3_state_dict, world4_metadata)
        self.assertEqual(2, len(load_plan.items))
        # this is [30, 60] to load [40, 60]
        low_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])
        )
        # this is [60, 90] to load [60, 80]
        high_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([20])
        )

        self.assertEqual(low_ri.storage_index, MetadataIndex("st", [30]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([10]))
        self.assertEqual(low_ri.dest_index, MetadataIndex("st", [40]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([20]))

        self.assertEqual(high_ri.storage_index, MetadataIndex("st", [60]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex("st", [40]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([20]))
        self.assertEqual(high_ri.lengths, torch.Size([20]))


class TestPlannerHelpers(TestCase):
    # [CLASS DOC] Single-process tests for functions in the planner helper module: read item
    # creation from chunk lists, merging of delta plans, and save plan comparison.
    def test_create_read_item_from_chunks(self):
        # [FUNCTION DOC] Saved tensor of 16 elements in two chunks of 8; the requested chunk
        # is offset 4, size 7. Asserts two read items: elements 4..7 from the first chunk
        # (source offset 4, length 4, destination offset 0) and elements 8..10 from the second
        # (source offset 0, length 3, destination offset 4), both into the chunk at offset 4.
        tensor_md = TensorStorageMetadata(
            properties=TensorProperties.create_from_tensor(torch.empty([16])),
            size=torch.Size([16]),
            chunks=[
                ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([8])),
                ChunkStorageMetadata(offsets=torch.Size([8]), sizes=torch.Size([8])),
            ],
        )

        chunk = ChunkStorageMetadata(offsets=torch.Size([4]), sizes=torch.Size([7]))
        read_items = create_read_items_for_chunk_list("foo", tensor_md, [chunk])

        self.assertEqual(2, len(read_items))
        self.assertEqual(MetadataIndex("foo", [4]), read_items[0].dest_index)
        self.assertEqual(torch.Size([0]), read_items[0].dest_offsets)

        self.assertEqual(MetadataIndex("foo", [0]), read_items[0].storage_index)
        self.assertEqual(torch.Size([4]), read_items[0].storage_offsets)

        self.assertEqual(torch.Size([4]), read_items[0].lengths)

        self.assertEqual(MetadataIndex("foo", [4]), read_items[1].dest_index)
        self.assertEqual(torch.Size([4]), read_items[1].dest_offsets)

        self.assertEqual(MetadataIndex("foo", [8]), read_items[1].storage_index)
        self.assertEqual(torch.Size([0]), read_items[1].storage_offsets)

        self.assertEqual(torch.Size([3]), read_items[1].lengths)

    def test_merge_delta_local_plans(self):
        # [FUNCTION DOC] Merging cached per-rank plans with delta plans: where a delta plan is
        # usable it replaces the cached plan for that rank; where it is an empty unusable plan
        # the cached plan is kept. Covers both changed, only the first, only the second, and
        # none changed. Plans are compared by item index, type and tensor data.
        def create_data(rank):
            # [FUNCTION DOC] HELPER: returns the default local save plan of a simulated rank
            # of 4 (tensor, list, sharded tensor); rank 0 is coordinator.
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        def _validate_plans(plan1: SavePlan, plan2: SavePlan):
            # [FUNCTION DOC] HELPER: asserts two plans have the same item count and, item by
            # item, equal index, type and tensor data.
            self.assertEqual(len(plan1.items), len(plan2.items))
            for item1, item2 in zip(plan1.items, plan2.items):
                self.assertEqual(item1.index, item2.index)
                self.assertEqual(item1.type, item2.type)
                self.assertEqual(item1.tensor_data, item2.tensor_data)

        cached_plans = [create_data(0), create_data(1)]
        delta_plans = [create_data(2), create_data(3)]

        # Both the plans changed.
        # Merge plan should have both the plans from the delta plans
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        self.assertEqual(2, len(merged_plans))
        _validate_plans(delta_plans[0], merged_plans[0])
        _validate_plans(delta_plans[1], merged_plans[1])

        # Only the first plan changed.
        # Merge plan should have the first plan from the delta plans and the second plan from the cached plans
        delta_plans = [create_data(2), SavePlan([], usable=False)]
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        _validate_plans(delta_plans[0], merged_plans[0])
        _validate_plans(cached_plans[1], merged_plans[1])

        # Only the second plan changed.
        # Merge plan should have the first plan from the cached plans and the second plan from the delta plans
        delta_plans = [SavePlan([], usable=False), create_data(3)]
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        _validate_plans(cached_plans[0], merged_plans[0])
        _validate_plans(delta_plans[1], merged_plans[1])

        # None of the plans changed. Cached plans should be returned
        delta_plans = [SavePlan([], usable=False), SavePlan([], usable=False)]
        merged_plans = _merge_delta_local_plans(cached_plans, delta_plans)
        _validate_plans(cached_plans[0], merged_plans[0])
        _validate_plans(cached_plans[1], merged_plans[1])

    def test_compare_save_plans(self):
        # [FUNCTION DOC] Asserts the plan comparison helper reports plans from different
        # ranks as not equal and each plan as equal to itself.
        def create_data(rank):
            # [FUNCTION DOC] HELPER: returns the default local save plan of a simulated rank
            # of 4 (tensor, list, sharded tensor); rank 0 is coordinator.
            with with_dist(rank=rank, world_size=4):
                tensor = torch.rand(10)
                val = [1, 2, 3]
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                state_dict = {"tensor": tensor, "value": val, "st": st}
                return create_default_local_save_plan(state_dict, rank == 0)

        plan1 = create_data(0)
        plan2 = create_data(1)
        self.assertFalse(_compare_save_plans(plan1, plan2))
        self.assertTrue(_compare_save_plans(plan1, plan1))
        self.assertTrue(_compare_save_plans(plan2, plan2))


class TestValidateGlobalPlan(TestCase):
    # [CLASS DOC] Single-process tests for _validate_global_plan, which returns a list of
    # error messages (empty when the plan is valid) for the given plans and metadata.
    def _make_metadata(self, chunks, size):
        # [FUNCTION DOC] HELPER: returns Metadata with a single float32 tensor entry named
        # param, of the given size and chunk list.
        storage = TensorStorageMetadata(
            properties=TensorProperties(dtype=torch.float32),
            size=torch.Size(size),
            chunks=chunks,
        )
        return Metadata(state_dict_metadata={"param": storage})

    def test_non_overlapping_chunks(self):
        # [FUNCTION DOC] Four adjacent chunks of one element covering a size 4 tensor yield
        # no validation errors.
        chunks = [
            ChunkStorageMetadata(offsets=torch.Size([i]), sizes=torch.Size([1]))
            for i in range(4)
        ]
        metadata = self._make_metadata(chunks, [4])
        self.assertEqual(_validate_global_plan([SavePlan([])], metadata), [])

    def test_detect_overlapping_chunks(self):
        # [FUNCTION DOC] Two chunks overlapping at index 1 produce at least one error.
        chunks = [
            ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([2])),
            ChunkStorageMetadata(offsets=torch.Size([1]), sizes=torch.Size([2])),
        ]
        metadata = self._make_metadata(chunks, [4])
        self.assertGreater(len(_validate_global_plan([SavePlan([])], metadata)), 0)


class TestLoadPlanner(TestCase):
    # [CLASS DOC] Single-process end-to-end tests of DefaultLoadPlanner behavior using
    # dcp.save and dcp.load of small linear modules to a temporary directory.
    @with_temp_dir
    def test_strict(self):
        # [FUNCTION DOC] Saves a linear module, then loads into a module with an extra
        # parameter: succeeds when partial loading is allowed, and raises CheckpointException
        # mentioning a missing key in the checkpoint when partial loading is disallowed.
        original_module = nn.Linear(2, 2)
        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        new_module = nn.Linear(2, 2)
        new_module.extra_param = nn.Parameter(torch.randn(2, 2))
        dcp.load(
            state_dict={"module": new_module},
            checkpoint_id=self.temp_dir,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        with self.assertRaisesRegex(CheckpointException, "Missing key in checkpoint"):
            dcp.load(
                state_dict={"module": new_module},
                checkpoint_id=self.temp_dir,
                planner=DefaultLoadPlanner(allow_partial_load=False),
            )

    @with_temp_dir
    def test_load_different_sizes_throws(self):
        # [FUNCTION DOC] Saves a 2x2 linear module and loads into a 3 input linear module;
        # asserts CheckpointException with a size mismatch message.
        original_module = nn.Linear(2, 2)
        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        new_module = nn.Linear(3, 2)
        with self.assertRaisesRegex(CheckpointException, "Size mismatch"):
            dcp.load(
                state_dict={"module": new_module},
                checkpoint_id=self.temp_dir,
                planner=DefaultLoadPlanner(),
            )

    @with_temp_dir
    def test_version_key_in_planner_data(self):
        # [FUNCTION DOC] Saves and loads a linear module with a default load planner and
        # asserts the planner's checkpoint metadata version equals the current DCP version.
        original_module = nn.Linear(2, 2)

        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        new_module = nn.Linear(2, 2)
        planner = DefaultLoadPlanner()
        dcp.load(
            state_dict={"module": new_module},
            checkpoint_id=self.temp_dir,
            planner=planner,
        )

        self.assertEqual(planner.metadata.version, CURRENT_DCP_VERSION)


if __name__ == "__main__":
    run_tests()
