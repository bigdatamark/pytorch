# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for the legacy save_state_dict/load_state_dict checkpoint entry points:
# default local metadata creation for sharded, replicated and bytes entries, and propagation
# of per-rank failures injected at every storage writer/reader stage into a
# CheckpointException (including the no-distributed mode). Defines fault-injecting storage
# stubs. Import-time side effects: computes the module globals device_type and backend from
# the current accelerator, and exits the process (status 0) under dev-asan.

import os
import sys
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.futures
import torch.nn
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint import (
    CheckpointException,
    load_state_dict,
    save_state_dict,
    StorageReader,
    StorageWriter,
)
from torch.distributed.checkpoint.default_planner import _create_default_local_metadata
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = torch.distributed.get_default_backend_for_device(device_type)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a 4x4 sharded zeros tensor, a regular 4x4 parameter, two
    # optional extra attributes (initially None), and a registered sharded-tensor state dict
    # hook so ShardedTensor entries appear in the state dict. Not referenced by tests here.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the sharded tensor from spec, the regular parameter,
        # empty extras, and registers the state dict hook.
        super().__init__()
        self.sharded: ShardedTensor = sharded_tensor.zeros(self.spec(), 4, 4)
        self.regular = torch.nn.Parameter(torch.ones(4, 4))
        self.extra_sharded: ShardedTensor | None = None
        self.extra_param: torch.nn.Parameter | None = None
        self._register_state_dict_hook(state_dict_hook)

    def spec(self) -> ChunkShardingSpec:
        # [FUNCTION DOC] HELPER: Returns a chunk sharding spec splitting dim 0 over rank 0
        # on device 0 and rank 1 on device 1 of the current accelerator type.
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        return ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )


class TestDistributedCheckpointing(ShardedTensorTestBase):
    # [CLASS DOC] Two-process test case for default local metadata generation.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Always 2 processes.
        return 2

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_tensor_metadata_with_missing_rank_spec(self) -> None:
        # [FUNCTION DOC] Builds a sharded tensor whose spec lists only rank 1, creates default
        # local metadata, and checks that entry has exactly one chunk.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:1/{device_type}:1",
            ],
        )

        st = sharded_tensor.zeros(spec, 4, 4, dtype=torch.float64)
        md = _create_default_local_metadata({"st": st})
        st_md = md.state_dict_metadata["st"]

        self.assertEqual(1, len(st_md.chunks))

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_default_metadata(self) -> None:
        # [FUNCTION DOC] Creates default local metadata for a state dict with a sharded
        # tensor (two ranks), a replicated tensor and a plain list. Checks the list maps to
        # bytes metadata, and the tensors map to tensor metadata with float32 dtype, the
        # right global size, and 1 chunk (replicated) or 2 chunks (sharded).
        device = f"{device_type}:{dist.get_rank()}"
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )

        state_dict = {
            "sharded": sharded_tensor.rand(
                spec,
                (
                    10,
                    10,
                ),
            ),
            "replicated": torch.rand(4, device=device),
            "bytes": [1, 2, 3, 4],
        }

        metadata = _create_default_local_metadata(state_dict)
        self.assertTrue("bytes" in metadata.state_dict_metadata)
        self.assertIsInstance(
            metadata.state_dict_metadata["bytes"], BytesStorageMetadata
        )

        self.assertTrue("replicated" in metadata.state_dict_metadata)
        self.assertIsInstance(
            metadata.state_dict_metadata["replicated"], TensorStorageMetadata
        )
        md = metadata.state_dict_metadata["replicated"]
        self.assertEqual(md.size, state_dict["replicated"].size())
        self.assertEqual(md.properties.dtype, torch.float32)
        self.assertEqual(1, len(md.chunks))

        self.assertTrue("sharded" in metadata.state_dict_metadata)
        self.assertIsInstance(
            metadata.state_dict_metadata["sharded"], TensorStorageMetadata
        )
        md = metadata.state_dict_metadata["sharded"]
        self.assertEqual(md.properties.dtype, torch.float32)
        self.assertEqual(md.size, state_dict["sharded"].size())
        self.assertEqual(2, len(md.chunks))


class TestStorageBase:
    # [CLASS DOC] HELPER: Mixin for fault-injecting storage stubs. fail_conf maps a hook name
    # to the collection of ranks that should fail at that hook; rank is the current process
    # rank (0 when distributed is not initialized).
    def __init__(self, fail_conf):
        # [FUNCTION DOC] HELPER: Stores fail_conf and the current rank.
        self.fail_conf = fail_conf
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def _get_ranks(self, name):
        # [FUNCTION DOC] HELPER: Returns the ranks configured to fail for the hook name, or
        # None when the hook is not configured.
        return self.fail_conf.get(name, None)

    def _fail_rank(self, name):
        # [FUNCTION DOC] HELPER: Raises ValueError if this rank is configured to fail at hook
        # name; otherwise does nothing.
        ranks = self._get_ranks(name)
        if ranks is not None and self.rank in ranks:
            raise ValueError(f"rank fail {self.rank} for {name}")

    def _fail_rank_async(self, name, result=None):
        # [FUNCTION DOC] HELPER: Returns a future that holds a ValueError if this rank is
        # configured to fail at hook name, else completed with the given result.
        ranks = self._get_ranks(name)
        fut = Future()
        if ranks is not None and self.rank in ranks:
            fut.set_exception(ValueError(f"async rank fail {self.rank} for {name}"))
        else:
            fut.set_result(result)
        return fut


class FaultyStorageWriter(TestStorageBase, StorageWriter):
    # [CLASS DOC] HELPER: StorageWriter stub that writes nothing and raises per fail_conf at
    # each stage: set up, local plan, global plan, write_data (sync and async), and finish.
    def __init__(self, fail_conf):
        # [FUNCTION DOC] HELPER: Passes fail_conf to the base mixin.
        super().__init__(fail_conf)

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        return

    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_set_up_storage_writer.
        self._fail_rank("fail_set_up_storage_writer")

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_prepare_local_plan;
        # otherwise returns the plan unchanged.
        self._fail_rank("fail_prepare_local_plan")
        return plan

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_prepare_global_plan;
        # otherwise returns the plans unchanged.
        self._fail_rank("fail_prepare_global_plan")
        return plans

    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_write_data; otherwise
        # returns a future that fails on ranks configured for fail_write_data_async and
        # completes with an empty result list elsewhere.
        self._fail_rank("fail_write_data")
        return self._fail_rank_async("fail_write_data_async", [])

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_finish; else no-op.
        self._fail_rank("fail_finish")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return True


class FaultyStorageReader(TestStorageBase, StorageReader):
    # [CLASS DOC] HELPER: StorageReader stub that serves a fixed Metadata object, reads no
    # data, and raises per fail_conf at each stage: set up, local plan, global plan,
    # read_data (sync and async) and read_metadata.
    def __init__(self, metadata, fail_conf):
        # [FUNCTION DOC] HELPER: Passes fail_conf to the base mixin and keeps the metadata
        # object to be returned by read_metadata.
        super().__init__(fail_conf)
        self.metadata = metadata

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        return

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_set_up_storage_reader.
        self._fail_rank("fail_set_up_storage_reader")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_prepare_local_plan;
        # otherwise returns the plan unchanged.
        self._fail_rank("fail_prepare_local_plan")
        return plan

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_prepare_global_plan;
        # otherwise returns the plans unchanged.
        self._fail_rank("fail_prepare_global_plan")
        return plans

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_read_data; otherwise
        # returns a future that fails on ranks configured for fail_read_data_async and
        # completes with no value elsewhere. Loads no tensor data.
        self._fail_rank("fail_read_data")
        return self._fail_rank_async("fail_read_data_async")

    def read_metadata(self) -> Metadata:
        # [FUNCTION DOC] HELPER: Raises on ranks configured for fail_read_metadata; otherwise
        # returns the stored metadata.
        self._fail_rank("fail_read_metadata")
        return self.metadata

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return True


class TestDistributedFailure(ShardedTensorTestBase):
    # [CLASS DOC] Multi-process test case checking that failures injected on chosen ranks at
    # each save/load stage surface as CheckpointException on the failing ranks, with and
    # without an initialized process group.
    def get_spec(self):
        # [FUNCTION DOC] HELPER: Returns a chunk sharding spec on dim 0 with one placement per
        # rank of the world, each on the same-numbered device of the accelerator type.
        return ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{r}/{device_type}:{r}" for r in range(dist.get_world_size())
            ],
        )

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_dummy_writer_works(self) -> None:
        # [FUNCTION DOC] Saves a state dict (sharded, replicated, bytes) with a writer that
        # has no failures configured; passes if no exception is raised.
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        save_state_dict(state_dict, FaultyStorageWriter({}))

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_dummy_reader_works(self) -> None:
        # [FUNCTION DOC] Loads a state dict using default local metadata and a reader with
        # no failures configured; passes if no exception is raised.
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }
        metadata = _create_default_local_metadata(state_dict)

        load_state_dict(state_dict, FaultyStorageReader(metadata, {}))

    def _test_dist_failure(self, callback, kwargs):
        # [FUNCTION DOC] HELPER: Runs callback. The bad ranks are the first value in kwargs
        # (empty if kwargs is empty). With no bad ranks the callback must succeed. Otherwise
        # it must raise CheckpointException whose failures only name bad ranks, each with a
        # ValueError as first wrapped error unless ignore_exception_type is set, and every
        # bad rank must appear in the failures.
        # KNOWN ISSUE: bad ranks come from the first kwargs value, so this relies on the
        # failure hook being the first keyword argument, ahead of ignore_exception_type.
        bad_ranks = next(iter(kwargs.values())) if len(kwargs) > 0 else []

        # Empty bad_ranks means it must work
        if len(bad_ranks) == 0:
            callback()
        else:
            with self.assertRaises(CheckpointException) as cm:
                callback()
            e = cast(CheckpointException, cm.exception)
            for rank, wrapped_ex in e.failures.items():
                ex = wrapped_ex[0]
                self.assertTrue(
                    rank in bad_ranks, msg=lambda msg: f"{msg}\n{rank} did not fail"
                )
                if not kwargs.get("ignore_exception_type", False):
                    self.assertEqual(ValueError, type(ex), str(ex))

            failed_ranks = e.failures.keys()
            for rank in bad_ranks:
                self.assertTrue(
                    rank in failed_ranks,
                    msg=lambda msg: f"{msg}\n{rank} was supposed to fail was fine",
                )

    def _test_save(self, state_dict, coordinator=0, **kwargs):
        # [FUNCTION DOC] HELPER: Saves state_dict through save_state_dict with a
        # FaultyStorageWriter built from kwargs, the given coordinator rank, and no_dist
        # set when no process group exists; then checks the failure outcome.
        no_dist = not dist.is_initialized()

        def _save():
            save_state_dict(
                state_dict,
                storage_writer=FaultyStorageWriter(kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        self._test_dist_failure(_save, kwargs)

    def _test_load(self, state_dict, coordinator=0, **kwargs):
        # [FUNCTION DOC] HELPER: Loads state_dict via load_state_dict using default local
        # metadata and a FaultyStorageReader built from kwargs, the given coordinator rank,
        # and no_dist set when no process group exists; then checks the failure outcome.
        no_dist = not dist.is_initialized()

        def _load():
            metadata = _create_default_local_metadata(state_dict)
            load_state_dict(
                state_dict,
                storage_reader=FaultyStorageReader(metadata, kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        self._test_dist_failure(_load, kwargs)

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_save_error_handling(self) -> None:
        # [FUNCTION DOC] With 4 ranks, injects failures at each save stage on chosen ranks
        # (setup, finish, global plan, local plan, write_data, async write) and with
        # coordinator rank 1, checking each is reported as expected.
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        self._test_save(state_dict, fail_set_up_storage_writer=[0])
        self._test_save(state_dict, fail_finish=[0])
        self._test_save(state_dict, fail_prepare_global_plan=[0])

        self._test_save(state_dict, fail_prepare_local_plan=[0])
        self._test_save(state_dict, fail_write_data=[2])
        self._test_save(state_dict, fail_write_data_async=[3])

        self._test_save(state_dict, coordinator=1, fail_set_up_storage_writer=[1])
        self._test_save(state_dict, coordinator=1, fail_finish=[1])

    def test_save_error_handling_no_dist(self) -> None:
        # [FUNCTION DOC] Asserts no process group exists, then repeats the save failure
        # injections at every stage with rank 0 as the only rank.
        state_dict = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}

        self.assertFalse(dist.is_initialized())

        self._test_save(state_dict, fail_set_up_storage_writer=[0])
        self._test_save(state_dict, fail_finish=[0])
        self._test_save(state_dict, fail_prepare_global_plan=[0])

        self._test_save(state_dict, fail_prepare_local_plan=[0])
        self._test_save(state_dict, fail_write_data=[0])
        self._test_save(state_dict, fail_write_data_async=[0])

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend()
    def test_load_error_handling(self) -> None:
        # [FUNCTION DOC] With 4 ranks, checks a clean load succeeds and that failures
        # injected at each load stage (setup, global plan, read_metadata, local plan, read
        # data sync and async) on chosen ranks and with several coordinator ranks are
        # reported; read_metadata failures skip the exception type check.
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        self._test_load(state_dict)
        self._test_load(state_dict, fail_set_up_storage_reader=[0])
        self._test_load(state_dict, fail_prepare_global_plan=[0])
        self._test_load(state_dict, fail_read_metadata=[0], ignore_exception_type=True)
        self._test_load(state_dict, fail_prepare_local_plan=[1])
        self._test_load(state_dict, fail_read_data=[3])
        self._test_load(state_dict, fail_read_data_async=[1])

        self._test_load(state_dict, coordinator=3, fail_set_up_storage_reader=[0])
        self._test_load(
            state_dict,
            coordinator=1,
            fail_read_metadata=[3],
            ignore_exception_type=True,
        )
        self._test_load(state_dict, coordinator=2, fail_read_data=[0])
        self._test_load(state_dict, coordinator=3, fail_read_data_async=[2])
        self._test_load(state_dict, coordinator=1, fail_prepare_global_plan=[1])

    def test_load_error_handling_no_dist(self) -> None:
        # [FUNCTION DOC] Repeats the load checks with no process group and rank 0 only.
        state_dict = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}
        self._test_load(state_dict)
        self._test_load(state_dict, fail_set_up_storage_reader=[0])
        self._test_load(state_dict, fail_read_metadata=[0], ignore_exception_type=True)
        self._test_load(state_dict, fail_prepare_local_plan=[0])
        self._test_load(state_dict, fail_prepare_global_plan=[0])
        self._test_load(state_dict, fail_read_data=[0])
        self._test_load(state_dict, fail_read_data_async=[0])


if __name__ == "__main__":
    run_tests()
