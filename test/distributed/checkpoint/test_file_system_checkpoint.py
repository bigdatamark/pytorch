# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the legacy save_state_dict/load_state_dict API with the
# FileSystemWriter/FileSystemReader storage: plain tensors (no_dist), ShardedTensor
# save/load, resharding on load across sharding specs, byte objects, sharded<->plain tensor
# switching, and plan caching. Defines a state-dict comparison helper and small models.
# Import-time side effects: picks the accelerator device type and default backend; exits
# the process (status 0) when running under dev-asan.

import os
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint._extension import ZStandard
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    MyShardedModel1,
)
from torch.testing._internal.distributed.checkpoint_utils import (
    get_test_extension_registry,
    Rot13Example,
    with_temp_dir,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = torch.distributed.get_default_backend_for_device(device_type)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
) -> bool:
    # [FUNCTION DOC] HELPER: Asserts through the test case that both state dicts have the
    # same size and key set, then compares values per key: ShardedTensor local shards are
    # compared pairwise and plain tensors with an exact-equality check. Returns True when
    # all checks pass; failures surface as assertion errors from the test case.
    # KNOWN ISSUE: the lambda passed as the assertTrue message is not called as a string
    # builder by unittest, so the key-specific message is probably never shown.
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, ShardedTensor):
            for local_shard_1, local_shard_2 in zip(
                value_1.local_shards(), value_2.local_shards()
            ):
                self.assertTrue(
                    torch.equal(local_shard_1.tensor, local_shard_2.tensor),
                    lambda msg: f"{msg}\nKey {key}'s shard does not match",
                )
        elif isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2),
                lambda msg: f"{msg}\nKey {key}'s tensor does not match",
            )

    return True


class MyTestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with two Linear layers and an EmbeddingBag, used to
    # produce a state dict of plain tensors.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear_1 (5->5), linear_2 (5->1) and emb (5x10).
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# The ShardedModels are borrowed from test/distributed/_sharded_tensor/test_sharded_tensor.py
class MyShardedModel3(torch.nn.Module):
    # [CLASS DOC] HELPER: Module holding a single random 10x20 ShardedTensor laid out by the
    # given sharding spec (no RRefs).
    def __init__(
        self,
        spec: ShardingSpec,
    ) -> None:
        # [FUNCTION DOC] HELPER: Creates the sharded_tensor attribute via sharded_tensor.rand
        # with the spec and size 10x20.
        super().__init__()
        self.sharded_tensor: ShardedTensor = sharded_tensor.rand(
            spec, 10, 20, init_rrefs=False
        )


class TestDistributedStateDictSaveLoad(TestCase):
    # [CLASS DOC] Single-process tests of save/load with the no_dist option.
    def test_read_write_only_tensor(self) -> None:
        # [FUNCTION DOC] Saves a plain-tensor state dict to a temp directory without a
        # process group (default layout, then single_file_per_rank), loads into a freshly
        # initialized module state dict, and checks it differs before and matches after.
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path)
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            state_dict_to_load_to = MyTestModule().state_dict()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path, single_file_per_rank=True)
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            state_dict_to_load_to = MyTestModule().state_dict()

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    # [CLASS DOC] Two-rank test of save/load round trip for a sharded model, parametrized
    # over storage extensions.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks for these tests.
        return 2

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    @parametrize("extensions", [None, [Rot13Example()], [ZStandard()]])
    def test_read_write_shard_tensor(self, extensions) -> None:
        # [FUNCTION DOC] Rank 0's temp directory is broadcast to all ranks. Saves a
        # ShardedTensor-containing model state dict via FileSystemWriter (with the given
        # extension list), loads it into a second model using a reader with the test
        # extension registry, and verifies values differ before and match after loading.
        paths = [tempfile.mkdtemp()]
        dist.broadcast_object_list(paths)

        path = paths[0]

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )

        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path, _extensions=extensions)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec, init_rrefs=False)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(
            path=path, _extension_registry=get_test_extension_registry()
        )
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()


class TestDistributedReshardOnLoad(ShardedTensorTestBase):
    # [CLASS DOC] Two-rank tests that save with one sharding layout and load with another
    # (different chunk/enumerable specs, row vs column sharding, sharded vs plain tensors).
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks for these tests.
        return 2

    def get_file_path(self) -> str:
        # [FUNCTION DOC] HELPER: Rank 0 creates a temporary directory and broadcasts its
        # path to every rank; returns that shared path.
        paths = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(paths)
        return paths[0]

    def load_tensor(self, tensor: ShardedTensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Gathers a ShardedTensor onto rank 0 (a zero tensor on the
        # first accelerator) and returns it there; other ranks pass no output and get None.
        res = (
            torch.zeros(tensor.shape, device=f"{device_type}:0")
            if dist.get_rank() == 0
            else None
        )
        tensor.gather(out=res)
        return res

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_load_with_different_shard_plan(self) -> None:
        # [FUNCTION DOC] For every ordered pair of distinct sharding specs (two chunk specs,
        # two enumerable specs over a 10x20 tensor), recreates the directory, saves a model
        # with the first spec, loads into a model with the second, gathers both tensors on
        # rank 0 and asserts they are close.
        path = self.get_file_path()

        # We hardcode the assumption of how many shards are around
        self.assertEqual(self.world_size, dist.get_world_size())

        specs = [
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    f"rank:0/{device_type}:0",
                    f"rank:1/{device_type}:1",
                ],
            ),
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    f"rank:0/{device_type}:0",
                    f"rank:1/{device_type}:1",
                    f"rank:1/{device_type}:1",
                    f"rank:0/{device_type}:0",
                ],
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[2, 20],
                        placement=f"rank:0/{device_type}:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[2, 0],
                        shard_sizes=[1, 20],
                        placement=f"rank:1/{device_type}:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[3, 0],
                        shard_sizes=[3, 20],
                        placement=f"rank:0/{device_type}:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[6, 0],
                        shard_sizes=[3, 20],
                        placement=f"rank:1/{device_type}:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[9, 0],
                        shard_sizes=[1, 20],
                        placement=f"rank:0/{device_type}:0",
                    ),
                ]
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[8, 20],
                        placement=f"rank:1/{device_type}:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8, 0],
                        shard_sizes=[2, 20],
                        placement=f"rank:0/{device_type}:0",
                    ),
                ]
            ),
        ]

        for s0 in specs:
            for s1 in specs:
                if s0 == s1:
                    continue

                if dist.get_rank() == 0:
                    shutil.rmtree(path, ignore_errors=True)
                    os.makedirs(path)
                dist.barrier()

                model_to_save = MyShardedModel3(s0)
                model_to_save._register_state_dict_hook(state_dict_hook)
                state_dict_to_save = model_to_save.state_dict()

                fs_writer = FileSystemWriter(path=path)
                save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

                dist.barrier()

                model_to_load = MyShardedModel3(s1)
                model_to_load._register_state_dict_hook(state_dict_hook)
                state_dict_to_load_to = model_to_load.state_dict()
                dist.barrier()

                fs_reader = FileSystemReader(path=path)
                load_state_dict(
                    state_dict=state_dict_to_load_to, storage_reader=fs_reader
                )

                dist.barrier()
                store_tensor = self.load_tensor(model_to_save.sharded_tensor)
                dist.barrier()
                load_tensor = self.load_tensor(model_to_load.sharded_tensor)

                if dist.get_rank() == 0:
                    self.assertTrue(
                        torch.allclose(store_tensor, load_tensor),
                        msg=lambda msg: f"{msg}\n{s0} vs {s1}",
                    )

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_load_rowwise_to_colwise(self) -> None:
        # [FUNCTION DOC] Saves a model sharded along dim 0 and loads into one sharded along
        # dim 1, then gathers both on rank 0 and asserts the full tensors are close.
        path = self.get_file_path()
        self.assertEqual(self.world_size, dist.get_world_size())

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        src_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        dst_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )

        if dist.get_rank() == 0:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)

        model_to_save = MyShardedModel3(src_spec).to(dist.get_rank())
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        model_to_load = MyShardedModel3(dst_spec).to(dist.get_rank())
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        fs_reader = FileSystemReader(path=path)

        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        # We can't use torch.allclose since each ST has a different sharding spec
        store_tensor = self.load_tensor(model_to_save.sharded_tensor)
        load_tensor = self.load_tensor(model_to_load.sharded_tensor)

        if dist.get_rank() == 0:
            self.assertTrue(torch.allclose(store_tensor, load_tensor))

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_save_load_bytes(self) -> None:
        # [FUNCTION DOC] Saves a state dict of non-tensor objects (a list and a string),
        # loads into a dict with different values, and asserts the originals are restored.
        path = self.get_file_path()

        state_dict_to_save = {"bytes0": [1], "bytes1": "string"}

        fs_writer = FileSystemWriter(path=path)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        state_dict_to_load = {"bytes0": [2], "bytes1": "other"}

        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

        self.assertEqual([1], state_dict_to_load["bytes0"])
        self.assertEqual("string", state_dict_to_load["bytes1"])

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    def test_switch_between_sharded_tensor_to_tensor(self) -> None:
        # [FUNCTION DOC] For each pair of 1D sharding specs, saves a sharded tensor plus a
        # replicated tensor, then loads the sharded entry into a plain tensor and the
        # replicated entry into a sharded tensor. On rank 0, asserts that both gathered
        # results match what was saved.
        path = self.get_file_path()
        tensor_size = 32

        specs = [
            ChunkShardingSpec(
                dim=0,
                placements=[
                    f"rank:0/{device_type}:0",
                    f"rank:1/{device_type}:1",
                ],
            ),
            ChunkShardingSpec(
                dim=0,
                placements=[
                    f"rank:0/{device_type}:0",
                    f"rank:1/{device_type}:1",
                    f"rank:1/{device_type}:1",
                    f"rank:0/{device_type}:0",
                ],
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[8],
                        placement=f"rank:1/{device_type}:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8],
                        shard_sizes=[tensor_size - 8],
                        placement=f"rank:0/{device_type}:0",
                    ),
                ]
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[10],
                        placement=f"rank:0/{device_type}:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[10],
                        shard_sizes=[tensor_size - 10],
                        placement=f"rank:1/{device_type}:1",
                    ),
                ]
            ),
        ]

        for save_spec in specs:
            for load_spec in specs:
                save_dict = {
                    "sharded": sharded_tensor.rand(save_spec, tensor_size),
                    "replicated": torch.rand(tensor_size, device=self.rank),
                }

                fs_writer = FileSystemWriter(path=path)
                save_state_dict(state_dict=save_dict, storage_writer=fs_writer)

                # Freaky Friday the tensors
                load_dict = {
                    "sharded": torch.zeros(tensor_size, device=self.rank),
                    "replicated": sharded_tensor.zeros(load_spec, tensor_size),
                }

                fs_reader = FileSystemReader(path=path)
                load_state_dict(state_dict=load_dict, storage_reader=fs_reader)

                save_dict_sharded = self.load_tensor(save_dict["sharded"])
                load_dict_replicated = self.load_tensor(load_dict["replicated"])

                if dist.get_rank() == 0:
                    self.assertTrue(
                        torch.allclose(save_dict_sharded, load_dict["sharded"]),
                        lambda msg: f"{msg}\nsave-spec {save_spec} load-spec {load_spec}",
                    )
                    self.assertTrue(
                        torch.allclose(save_dict["replicated"], load_dict_replicated),
                        lambda msg: f"{msg}\nsave-spec {save_spec} load-spec {load_spec}",
                    )


class TestDistributedStateDictSaveLoadWithCaching(ShardedTensorTestBase):
    # [CLASS DOC] Two-rank test of repeated saves using a planner with plan caching enabled.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks for these tests.
        return 2

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(2)
    @requires_accelerator_dist_backend()
    @with_temp_dir
    def test_read_write_shard_tensor(self) -> None:
        # [FUNCTION DOC] Saves and reloads a sharded model state dict twice using a default
        # save planner with plan caching enabled (the second save reuses the writer and a
        # new planner), verifying each time that a fresh model matches after loading.
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
            ],
        )

        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=self.temp_dir)
        save_state_dict(
            state_dict=state_dict_to_save,
            storage_writer=fs_writer,
            planner=DefaultSavePlanner(enable_plan_caching=True),
        )

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec, init_rrefs=False)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(
            path=self.temp_dir, _extension_registry=get_test_extension_registry()
        )
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()

        # Save Attempt 2
        save_state_dict(
            state_dict=state_dict_to_save,
            storage_writer=fs_writer,
            planner=DefaultSavePlanner(enable_plan_caching=True),
        )

        dist.barrier()

        # Create a new model
        model_to_load = MyShardedModel1(spec, init_rrefs=False)
        # This is not the correct hook for loading the state dict
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        dist.barrier()

        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # Test load.
        fs_reader = FileSystemReader(
            path=self.temp_dir, _extension_registry=get_test_extension_registry()
        )
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()


instantiate_parametrized_tests(TestDistributedStateDictSaveLoadWithSharedTensor)

if __name__ == "__main__":
    run_tests()
