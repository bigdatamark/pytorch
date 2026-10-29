# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for torch.distributed.checkpoint utilities: MetadataIndex lookup in state
# dicts, the DCP logger, exception wrapping for cross-rank serialization, bounded file views, and
# the _DistWrapper collectives. Exits at import time (status 0) under dev-asan builds. Defines the
# helper create_sharded_tensor used by the lookup tests.

import io
import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.c10d_logger import _c10d_logger
from torch.distributed.checkpoint.api import _wrap_exception
from torch.distributed.checkpoint.logger import _dcp_logger
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.utils import (
    _create_file_view,
    _DistWrapper,
    find_state_dict_object,
)
from torch.distributed.distributed_c10d import _object_to_tensor, _tensor_to_object
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.distributed_utils import with_fake_comms


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def create_sharded_tensor(rank, world_size, shards_per_rank):
    # [FUNCTION DOC] HELPER: Builds a 1-D ShardedTensor of world_size * shards_per_rank shards,
    # each of size 8 with consecutive offsets; shard idx is placed on rank idx // shards_per_rank.
    # Only shards owned by the given rank get a local random tensor. Returns the ShardedTensor
    # assembled from local shards and the global metadata (float32 properties).
    shards_metadata = []
    local_shards = []
    for idx in range(world_size * shards_per_rank):
        shard_rank = idx // shards_per_rank
        shard_md = ShardMetadata(
            shard_offsets=[idx * 8], shard_sizes=[8], placement=f"rank:{shard_rank}/cpu"
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
        size=torch.Size([8 * len(shards_metadata)]),
        tensor_properties=TensorProperties.create_from_tensor(torch.zeros(1)),
    )

    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md
    )


class TestMedatadaIndex(TestCase):
    # [CLASS DOC] Tests MetadataIndex equality/hash semantics and find_state_dict_object lookups
    # for plain and sharded entries, plus a check that the DCP logger is separate from c10d's.
    def test_init_convert_offset(self):
        # [FUNCTION DOC] An offset given as a list equals the same offset given as torch.Size.
        a = MetadataIndex("foo", [1, 2])
        b = MetadataIndex("foo", torch.Size([1, 2]))
        self.assertEqual(a, b)

    def test_index_hint_ignored_on_equals(self):
        # [FUNCTION DOC] The optional index hint does not affect MetadataIndex equality.
        a = MetadataIndex("foo")
        b = MetadataIndex("foo", index=99)
        self.assertEqual(a, b)

    def test_index_hint_ignored_on_hash(self):
        # [FUNCTION DOC] The optional index hint does not affect the MetadataIndex hash.
        a = MetadataIndex("foo")
        b = MetadataIndex("foo", index=99)
        self.assertEqual(hash(a), hash(b))

    def test_flat_data(self):
        # [FUNCTION DOC] For non-sharded entries (tensor, list) the lookup returns the stored
        # object regardless of offset or index hint. An unknown FQN raises ValueError mentioning
        # FQN, and an offset on a non-ShardedTensor entry raises ValueError mentioning
        # ShardedTensor.
        state_dict = {
            "a": torch.rand(10),
            "b": [1, 2, 3],
        }

        a = find_state_dict_object(state_dict, MetadataIndex("a"))
        self.assertEqual(a, state_dict["a"])
        a = find_state_dict_object(state_dict, MetadataIndex("a", [0]))
        self.assertEqual(a, state_dict["a"])
        a = find_state_dict_object(state_dict, MetadataIndex("a", index=99))
        self.assertEqual(a, state_dict["a"])

        b = find_state_dict_object(state_dict, MetadataIndex("b"))
        self.assertEqual(b, state_dict["b"])
        b = find_state_dict_object(state_dict, MetadataIndex("b", index=1))
        self.assertEqual(b, state_dict["b"])

        with self.assertRaisesRegex(ValueError, "FQN"):
            find_state_dict_object(state_dict, MetadataIndex("c"))
        with self.assertRaisesRegex(ValueError, "ShardedTensor"):
            find_state_dict_object(state_dict, MetadataIndex("b", [1]))

    @with_fake_comms(rank=0, world_size=2)
    def test_sharded_tensor_lookup(self):
        # [FUNCTION DOC] With fake comms (rank 0 of 2, 3 shards per rank), looking up a shard by
        # offset returns the matching local shard tensor whether the index hint is correct, wrong
        # or out of range. A missing offset raises ValueError (no offset was provided); an offset
        # matching no shard raises ValueError (Could not find shard).
        st = create_sharded_tensor(rank=0, world_size=2, shards_per_rank=3)
        state_dict = {"st": st}

        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8]))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # good hint
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=1))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # bad hint
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=2))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # broken hint
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=99))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        with self.assertRaisesRegex(ValueError, "no offset was provided"):
            find_state_dict_object(state_dict, MetadataIndex("st"))

        with self.assertRaisesRegex(ValueError, "Could not find shard"):
            find_state_dict_object(state_dict, MetadataIndex("st", [1]))

    def test_dcp_logger(self):
        # [FUNCTION DOC] Checks the DCP logger is a distinct object from the c10d logger and that
        # the c10d logger has exactly one handler.
        self.assertTrue(_c10d_logger is not _dcp_logger)
        self.assertEqual(1, len(_c10d_logger.handlers))


class TestWrapException(TestCase):
    # [CLASS DOC] Tests that _wrap_exception output survives pickling and keeps usable tracebacks.
    def test_wrap_exception_serializable_via_object_to_tensor(self):
        # [FUNCTION DOC] Wraps a raised ValueError, round-trips it through _object_to_tensor and
        # _tensor_to_object, and checks the exception message and a non-empty StackSummary survive.
        """Verify _wrap_exception produces a result that _object_to_tensor can serialize.

        Python 3.13+ adds a _code attribute to FrameSummary containing
        bytecode objects that cannot be pickled. _wrap_exception must
        clear these so that _object_to_tensor (used by gather_object and
        scatter_object_list) succeeds instead of raising
        "TypeError: cannot pickle code objects".
        """
        try:
            raise ValueError("test error")
        except ValueError as e:
            wrapped = _wrap_exception(e)

        # _object_to_tensor / _tensor_to_object are what gather_object
        # and scatter_object_list use to serialize objects across ranks.
        # This would raise "TypeError: cannot pickle code objects"
        # on Python 3.13+ without the fix.
        byte_tensor, size = _object_to_tensor(wrapped, torch.device("cpu"), None)
        restored = _tensor_to_object(byte_tensor, size.item(), None)

        self.assertIsInstance(restored[0], ValueError)
        self.assertEqual(str(restored[0]), "test error")
        self.assertIsInstance(restored[1], traceback.StackSummary)
        self.assertGreater(len(restored[1]), 0)

    def test_wrap_exception_preserves_traceback_formatting(self):
        # [FUNCTION DOC] Formats the wrapped traceback of a RuntimeError and checks the raising
        # source line text still appears.
        """Verify that clearing _code does not break traceback formatting."""
        try:
            raise RuntimeError("format test")
        except RuntimeError as e:
            wrapped = _wrap_exception(e)

        formatted = "".join(traceback.format_list(wrapped[1]))
        self.assertIn("raise RuntimeError", formatted)


class TestReaderView(TestCase):
    # [CLASS DOC] Tests _create_file_view, which exposes a (offset, length) window of a seekable
    # stream; setUp builds front, middle and back 5-byte views over a buffer of letters A-Z.
    def setUp(self):
        # [FUNCTION DOC] Creates three independent in-memory buffers holding bytes A-Z and wraps
        # them as 5-byte file views starting at offsets 0, 10 and 21 (end of buffer minus 5).
        super().setUp()
        buffer = io.BytesIO(bytearray(range(ord("A"), ord("Z") + 1)))
        self.front_view = _create_file_view(buffer, 0, 5)

        buffer = io.BytesIO(bytearray(range(ord("A"), ord("Z") + 1)))
        self.middle_view = _create_file_view(buffer, 10, 5)

        buffer = io.BytesIO(bytearray(range(ord("A"), ord("Z") + 1)))
        self.back_view = _create_file_view(buffer, len(buffer.getbuffer()) - 5, 5)

    def testShortRead(self):
        # [FUNCTION DOC] Reading 3 bytes returns the first 3 bytes of each view.
        self.assertEqual(self.front_view.read(3), b"ABC")
        self.assertEqual(self.middle_view.read(3), b"KLM")
        self.assertEqual(self.back_view.read(3), b"VWX")

    def testLongRead(self):
        # [FUNCTION DOC] Reading more than the window size returns only the 5 bytes in the window.
        self.assertEqual(self.front_view.read(10), b"ABCDE")
        self.assertEqual(self.middle_view.read(10), b"KLMNO")
        self.assertEqual(self.back_view.read(10), b"VWXYZ")

    def testAllRead(self):
        # [FUNCTION DOC] Reading with size -1 returns the whole window of each view.
        self.assertEqual(self.front_view.read(-1), b"ABCDE")
        self.assertEqual(self.middle_view.read(-1), b"KLMNO")
        self.assertEqual(self.back_view.read(-1), b"VWXYZ")

    def testShortReadinto(self):
        # [FUNCTION DOC] readinto a 3-byte buffer fills it and returns 3 for each view.
        ba = bytearray(3)

        self.assertEqual(self.front_view.readinto(ba), 3)
        self.assertEqual(ba, b"ABC")

        self.assertEqual(self.middle_view.readinto(ba), 3)
        self.assertEqual(ba, b"KLM")

        self.assertEqual(self.back_view.readinto(ba), 3)
        self.assertEqual(ba, b"VWX")

    def testLongReadinto(self):
        # [FUNCTION DOC] readinto an 8-byte buffer returns 5 and fills only the first 5 bytes; a
        # second call returns 0 and leaves the buffer unchanged, for each view.
        ba = bytearray(8)
        self.assertEqual(self.front_view.readinto(ba), 5)
        self.assertEqual(ba, b"ABCDE\0\0\0")
        self.assertEqual(self.front_view.readinto(ba), 0)
        self.assertEqual(ba, b"ABCDE\0\0\0")

        self.assertEqual(self.middle_view.readinto(ba), 5)
        self.assertEqual(ba, b"KLMNO\0\0\0")
        self.assertEqual(self.middle_view.readinto(ba), 0)
        self.assertEqual(ba, b"KLMNO\0\0\0")

        self.assertEqual(self.back_view.readinto(ba), 5)
        self.assertEqual(ba, b"VWXYZ\0\0\0")
        self.assertEqual(self.back_view.readinto(ba), 0)
        self.assertEqual(ba, b"VWXYZ\0\0\0")


class TestDistWrapper(DTensorTestBase):
    # [CLASS DOC] Multi-GPU tests of _DistWrapper (gather, scatter, broadcast, barrier) over
    # subgroups of a 2-D device mesh; world size is min(4, accelerator count).
    @property
    def world_size(self):
        # [FUNCTION DOC] Property: number of ranks, capped at 4 and at the accelerator count.
        return min(4, torch.accelerator.device_count())

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_object(self):
        # [FUNCTION DOC] Gathers each rank's id over the mesh dimension-1 group to coordinator rank
        # 0; only group coordinators get the list of ranks, others get None. Raises AssertionError
        # on mismatch.
        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        torch.random.manual_seed(dist.get_rank())

        dist_wrapper = _DistWrapper(
            mesh_2d.get_group(1), use_dist=True, coordinator_rank=0
        )

        rank = mesh_2d.get_rank()
        half_world_size = self.world_size // 2
        gathered_objects = dist_wrapper.gather_object(rank)
        expected_objects = (
            list(range(rank, rank + half_world_size))
            if rank % half_world_size == 0
            else None
        )
        if gathered_objects != expected_objects:
            raise AssertionError(f"Expected {expected_objects}, got {gathered_objects}")

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_scatter_object(self):
        # [FUNCTION DOC] Group coordinators scatter a list of consecutive ranks over the
        # dimension-1 group; each rank must receive its own rank id. Raises AssertionError on
        # mismatch.
        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        torch.random.manual_seed(dist.get_rank())

        dist_wrapper = _DistWrapper(
            mesh_2d.get_group(1), use_dist=True, coordinator_rank=0
        )

        rank = mesh_2d.get_rank()
        half_world_size = self.world_size // 2

        objects = (
            list(range(rank, rank + half_world_size))
            if rank % half_world_size == 0
            else None
        )
        scattered_objects = dist_wrapper.scatter_object(objects)
        expected_objects = rank
        if scattered_objects != expected_objects:
            raise AssertionError(
                f"Expected {expected_objects}, got {scattered_objects}"
            )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_broadcast_object_with_nonzero_coordinator(self):
        # [FUNCTION DOC] On the WORLD group with coordinator_rank 1, only rank 1 supplies a payload
        # and every rank must receive 1 from broadcast_object.
        # Everybody uses WORLD, but src is coordinator_rank=1
        dist_wrapper = _DistWrapper(
            group=dist.group.WORLD,
            use_dist=True,
            coordinator_rank=1,
        )

        rank = dist.get_rank()
        # only local rank 1 supplies the payload
        payload: int | None = rank if rank == 1 else None

        result = dist_wrapper.broadcast_object(payload)
        # every rank should receive the value from global rank 1
        if result != 1:
            raise AssertionError(f"Expected 1, got {result}")

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_broadcast_object_global_local_mismatch(self):
        # [FUNCTION DOC] Regression test (issue 152310): with a subgroup whose coordinator_rank is
        # a local index, broadcast_object must source from the global coordinator rank, so every
        # rank receives dist_wrapper.global_coordinator_rank.
        # reproduces issue 152310

        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        dist_wrapper = _DistWrapper(
            group=mesh_2d.get_group(1),
            use_dist=True,
            coordinator_rank=1,  # local coordinator index within the subgroup
        )

        rank = mesh_2d.get_rank()

        # only the local coordinator in each subgroup provides payload
        payload: int | None = rank if dist_wrapper.is_coordinator else None
        got = dist_wrapper.broadcast_object(payload)

        # ensure we broadcast from the *global* coordinator rank,
        # not the local index.  For rows [0,1] this is global rank 1;
        # for rows [2,3] this is global rank 3.
        expected = dist_wrapper.global_coordinator_rank
        if got != expected:
            raise AssertionError(f"Expected {expected}, got {got}")

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_barrier(self):
        # [FUNCTION DOC] Calls barrier on a mesh subgroup wrapper; passes if no exception is
        # raised.
        mesh_2d = dist.init_device_mesh(self.device_type, (2, self.world_size // 2))
        torch.random.manual_seed(dist.get_rank())

        dist_wrapper = _DistWrapper(
            mesh_2d.get_group(1), use_dist=True, coordinator_rank=0
        )

        # No exception should be raised.
        dist_wrapper.barrier()


if __name__ == "__main__":
    run_tests()
