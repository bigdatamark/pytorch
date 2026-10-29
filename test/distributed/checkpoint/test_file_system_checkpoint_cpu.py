# Owner(s): ["oncall: distributed"]
# [MODULE DOC] CPU/gloo variant of the file-system checkpoint tests: save/load with
# FileSystemWriter/Reader parametrized over writer thread counts, with Rot13 and ZStandard
# storage extensions, a Stateful blob object, sharded tensors, resharding on load, byte
# objects and sharded<->plain tensor switching. Import-time side effect: exits the process
# (status 0) under dev-asan; defines the thread count set used for parametrization.

import sys
import tempfile
from typing import Any, IO

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
    load,
    load_state_dict,
    save,
    save_state_dict,
)
from torch.distributed.checkpoint._extension import ZStandard
from torch.distributed.checkpoint.stateful import Stateful
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
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


_THREAD_COUNTS = {1, 2}


def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
) -> bool:
    # [FUNCTION DOC] HELPER: Asserts both state dicts have equal length and key sets, then
    # per key compares ShardedTensor local shards pairwise, plain tensors by exact
    # equality, and Stateful objects by equality. Returns True if no assertion fails.
    # KNOWN ISSUE: the lambda passed as the assertTrue message is never evaluated into a
    # string by unittest, so the per-key message is probably not shown.
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
        elif isinstance(value_1, Stateful):
            self.assertEqual(value_1, value_2)

    return True


class MyTestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module (two Linear layers, one EmbeddingBag) whose state
    # dict supplies plain tensors for the tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear_1 (5->5), linear_2 (5->1) and emb (5x10).
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# The ShardedModels are borrowed from test/distributed/_sharded_tensor/test_sharded_tensor.py
class MyShardedModel3(torch.nn.Module):
    # [CLASS DOC] HELPER: Module holding one random 10x20 ShardedTensor laid out by a spec.
    def __init__(
        self,
        spec: ShardingSpec,
    ) -> None:
        # [FUNCTION DOC] HELPER: Builds the sharded_tensor attribute with the spec, 10x20.
        super().__init__()
        self.sharded_tensor: ShardedTensor = sharded_tensor.rand(
            spec, 10, 20, init_rrefs=False
        )


class BlobState:
    # [CLASS DOC] HELPER: Minimal Stateful object whose state is a mapping with one key,
    # "blob", holding a bytes value; compares equal to another BlobState with equal state.
    def __init__(self, value: IO[bytes]) -> Any:
        # [FUNCTION DOC] HELPER: Stores a mapping with the single entry "blob" -> value.
        # KNOWN ISSUE: the return annotation is Any rather than None.
        self.state = {"blob": value}

    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns the internal state mapping itself (not a copy).
        return self.state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: Replaces the internal state with the given mapping.
        self.state = state_dict

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: True only if other is a BlobState with an equal state.
        return isinstance(other, BlobState) and self.state == other.state

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a string showing the blob value.
        return f"BlobState({self.state['blob']})"


class TestDistributedStateDictSaveLoad(TestCase):
    # [CLASS DOC] Single-process (no_dist) save/load test, parametrized by thread count.
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_only_tensor(self, thread_count) -> None:
        # [FUNCTION DOC] Saves a plain-tensor state dict with a FileSystemWriter using the
        # given thread count, loads it into a new module state dict without a process
        # group, and checks the dicts differ before loading and match afterward.
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
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


class TestDistributedStateDictSaveLoadRot13(TestCase):
    # [CLASS DOC] Tests save/load through the Rot13 example storage extension.
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_tensor_and_blob(self, thread_count) -> None:
        # [FUNCTION DOC] Saves module tensors plus a BlobState with the Rot13 extension via
        # the save API, then loads with a reader holding the test extension registry (the
        # extension is discovered from checkpoint metadata) and verifies equality.
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()
            state_dict_to_save["test_blob"] = BlobState(b"SomeBlobForTesting")

            fs_writer = FileSystemWriter(
                path=path,
                thread_count=thread_count,
                _extensions=[Rot13Example()],
            )
            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            state_dict_to_load_to = MyTestModule().state_dict()
            state_dict_to_load_to["test_blob"] = BlobState(b"")

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding.  Note there is no extension
            # specification here; it is determined dynamically from the metadata.
            fs_reader = FileSystemReader(
                path=path, _extension_registry=get_test_extension_registry()
            )
            load(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadZStandard(TestCase):
    # [CLASS DOC] Tests save/load through the ZStandard compression extension.
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_only_tensor(self, thread_count) -> None:
        # [FUNCTION DOC] Saves module tensors plus a BlobState with the ZStandard extension,
        # loads with a default reader and verifies equality after loading. Despite the name,
        # it also covers a non-tensor blob object.
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()
            state_dict_to_save["test_blob"] = BlobState(b"SomeBlobForTesting")

            fs_writer = FileSystemWriter(
                path=path,
                thread_count=thread_count,
                _extensions=[ZStandard()],
            )
            save(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
            )

            state_dict_to_load_to = MyTestModule().state_dict()
            state_dict_to_load_to["test_blob"] = BlobState(b"")

            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # Load from file without any resharding
            fs_reader = FileSystemReader(path=path)
            load(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
            )

            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)


class TestDistributedStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    # [CLASS DOC] Two-rank gloo test of sharded-tensor model save/load round trip.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks for these tests.
        return 2

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_shard_tensor(self, thread_count) -> None:
        # [FUNCTION DOC] Broadcasts a temp directory from rank 0, saves a sharded model
        # state dict with the given writer thread count, loads into a second model, and
        # asserts values differ before and match after loading.
        paths = [tempfile.mkdtemp()]
        dist.broadcast_object_list(paths)

        path = paths[0]

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # Test save
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
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
        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
        dist.barrier()


class TestDistributedReshardOnLoad(ShardedTensorTestBase):
    # [CLASS DOC] Two-rank gloo tests of loading checkpoints saved with a different
    # sharding layout, or switching between sharded and plain tensors.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 2, the number of ranks for these tests.
        return 2

    def get_file_path(self) -> str:
        # [FUNCTION DOC] HELPER: Rank 0 makes a temp directory and broadcasts its path;
        # every rank returns that path.
        paths = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(paths)
        return paths[0]

    def load_tensor(self, tensor: ShardedTensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Gathers a ShardedTensor to rank 0 into a CPU zero tensor and
        # returns it there; other ranks return None.
        res = torch.zeros(tensor.shape, device="cpu") if dist.get_rank() == 0 else None
        tensor.gather(out=res)
        return res

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_load_with_different_shard_plan(self, thread_count) -> None:
        # [FUNCTION DOC] For every ordered pair of distinct sharding specs (chunk and
        # enumerable, over a 10x20 tensor), saves a model with the first and loads into one
        # with the second, then compares gathered full tensors on rank 0. Unlike the GPU
        # version it does not clear the directory between iterations.
        path = self.get_file_path()

        # We hardcode the assumption of how many shards are around
        self.assertEqual(self.world_size, dist.get_world_size())

        specs = [
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                ],
            ),
            # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                    "rank:1",
                    "rank:0",
                ],
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[2, 0],
                        shard_sizes=[1, 20],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[3, 0],
                        shard_sizes=[3, 20],
                        placement="rank:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[6, 0],
                        shard_sizes=[3, 20],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[9, 0],
                        shard_sizes=[1, 20],
                        placement="rank:0",
                    ),
                ]
            ),
            # This requires the tensors to be [10, 20]
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[8, 20],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8, 0],
                        shard_sizes=[2, 20],
                        placement="rank:0",
                    ),
                ]
            ),
        ]

        for s0 in specs:
            for s1 in specs:
                if s0 == s1:
                    continue

                dist.barrier()

                model_to_save = MyShardedModel3(s0)
                model_to_save._register_state_dict_hook(state_dict_hook)
                state_dict_to_save = model_to_save.state_dict()

                fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
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

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_load_rowwise_to_colwise(self, thread_count) -> None:
        # [FUNCTION DOC] Saves a model sharded on dim 0, loads into one sharded on dim 1,
        # and asserts the gathered tensors are close on rank 0.
        # KNOWN ISSUE: moves the models with a cuda call even though this is a gloo/CPU
        # test, so it needs CUDA despite the CPU-oriented setup.
        path = self.get_file_path()
        self.assertEqual(self.world_size, dist.get_world_size())

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        src_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        dst_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        model_to_save = MyShardedModel3(src_spec).cuda(dist.get_rank())
        model_to_save._register_state_dict_hook(state_dict_hook)
        state_dict_to_save = model_to_save.state_dict()

        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        model_to_load = MyShardedModel3(dst_spec).cuda(dist.get_rank())
        model_to_load._register_state_dict_hook(state_dict_hook)
        state_dict_to_load_to = model_to_load.state_dict()

        fs_reader = FileSystemReader(path=path)

        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        # We can't use torch.allclose since each ST has a different sharding spec
        store_tensor = self.load_tensor(model_to_save.sharded_tensor)
        load_tensor = self.load_tensor(model_to_load.sharded_tensor)

        if dist.get_rank() == 0:
            self.assertTrue(torch.allclose(store_tensor, load_tensor))

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_save_load_bytes(self, thread_count) -> None:
        # [FUNCTION DOC] Saves a state dict of a list and a string, loads into a dict with
        # different values, and asserts the saved values are restored.
        path = self.get_file_path()

        state_dict_to_save = {"bytes0": [1], "bytes1": "string"}

        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        state_dict_to_load = {"bytes0": [2], "bytes1": "other"}

        fs_reader = FileSystemReader(path=path)
        load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

        self.assertEqual([1], state_dict_to_load["bytes0"])
        self.assertEqual("string", state_dict_to_load["bytes1"])

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_switch_between_sharded_tensor_to_tensor(self, thread_count) -> None:
        # [FUNCTION DOC] For each pair of 1D sharding specs, saves a sharded tensor and a
        # rank-0-broadcast replicated tensor, then loads the sharded entry into a plain
        # tensor and the replicated entry into a sharded tensor; rank 0 asserts both
        # gathered results match.
        path = self.get_file_path()
        tensor_size = 32

        specs = [
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                ],
            ),
            ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0",
                    "rank:1",
                    "rank:1",
                    "rank:0",
                ],
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[8],
                        placement="rank:1",
                    ),
                    ShardMetadata(
                        shard_offsets=[8],
                        shard_sizes=[tensor_size - 8],
                        placement="rank:0",
                    ),
                ]
            ),
            EnumerableShardingSpec(
                shards=[
                    ShardMetadata(
                        shard_offsets=[0],
                        shard_sizes=[10],
                        placement="rank:0",
                    ),
                    ShardMetadata(
                        shard_offsets=[10],
                        shard_sizes=[tensor_size - 10],
                        placement="rank:1",
                    ),
                ]
            ),
        ]

        for save_spec in specs:
            for load_spec in specs:
                save_dict = {
                    "sharded": sharded_tensor.rand(save_spec, tensor_size),
                    "replicated": torch.rand(tensor_size, device="cpu"),
                }
                dist.broadcast(save_dict["replicated"], src=0)

                fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
                save_state_dict(state_dict=save_dict, storage_writer=fs_writer)

                # Freaky Friday the tensors
                load_dict = {
                    "sharded": torch.zeros(tensor_size, device="cpu"),
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


instantiate_parametrized_tests(TestDistributedStateDictSaveLoad)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadRot13)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadWithSharedTensor)
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadZStandard)
instantiate_parametrized_tests(TestDistributedReshardOnLoad)

if __name__ == "__main__":
    run_tests()
