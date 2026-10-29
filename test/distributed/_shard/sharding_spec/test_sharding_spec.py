# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the sharding spec API (DevicePlacementSpec, ChunkShardingSpec,
# EnumerableShardingSpec), the internal helpers for split sizes, chunk sharding params, tensor
# and shard consistency checks, overlap validation, spec inference from shard metadata, and a
# user-defined GridShardingSpec exercised through the sharded tensor constructors. Defines the
# GridShardingSpec dataclass. Most spec-only tests need two CUDA GPUs and otherwise pass as
# skipped; the custom-spec tensor tests need 4 GPUs and NCCL.
import copy
from dataclasses import dataclass

import torch
from torch.distributed._shard import _shard_tensor, sharded_tensor
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import (
    _infer_sharding_spec_from_shards_metadata,
    ChunkShardingSpec,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    get_chunk_sharding_params,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)


class TestShardingSpec(TestCase):
    # [CLASS DOC] Single-process test case for sharding spec construction, validation and helper
    # functions; no process group is used.
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "2 CUDA GPUs are needed")
    def test_device_placement(self):
        # [FUNCTION DOC] Checks DevicePlacementSpec accepts device strings, torch devices and
        # rank-qualified strings, and that malformed strings raise ValueError (cannot parse
        # remote device) or RuntimeError (invalid device string).
        # valid devices
        DevicePlacementSpec("cuda:0")
        DevicePlacementSpec(torch.device(0))
        DevicePlacementSpec(torch.device("cuda:0"))
        DevicePlacementSpec("rank:0/cuda:0")
        DevicePlacementSpec("rank:0/cpu")
        DevicePlacementSpec("rank:0")

        # invalid devices
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            DevicePlacementSpec("cuda:foo")
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            DevicePlacementSpec("foo:0")
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            DevicePlacementSpec("rank:0/cuda:foo")
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            DevicePlacementSpec("rank:0/cpu2")

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "2 CUDA GPUs are needed")
    def test_chunked_sharding_spec(self):
        # [FUNCTION DOC] Checks ChunkShardingSpec construction: valid dims (including negative)
        # and placement forms are accepted, a named dimension raises NotImplementedError, a
        # non-integer dim raises ValueError, and bad placement strings raise ValueError or
        # RuntimeError.
        # Test valid specs.
        ChunkShardingSpec(0, [torch.device(0), torch.device(1)])
        ChunkShardingSpec(0, [torch.device("cuda:0"), torch.device("cuda:1")])
        ChunkShardingSpec(-1, ["cuda:0", "cuda:1"])
        ChunkShardingSpec(0, ["rank:0/cuda:0", "rank:0/cuda:1"])
        ChunkShardingSpec(0, ["rank:0", "rank:1"])
        ChunkShardingSpec(0, ["rank:0/cpu", "rank:1/cpu"])

        # Test unimplemented error
        with self.assertRaisesRegex(NotImplementedError, "not support named dimension"):
            # Named dimension.
            ChunkShardingSpec("N", ["cuda:0", "cuda:1"])

        # Test invalid specs
        with self.assertRaisesRegex(ValueError, "needs to be an integer"):
            ChunkShardingSpec(None, ["cuda:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "needs to be an integer"):
            ChunkShardingSpec({}, ["cuda:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ChunkShardingSpec(0, ["random:0", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ChunkShardingSpec(0, ["cuda:foo", "cuda:1"])
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ChunkShardingSpec(0, ["rank:foo", "cuda:1"])
        with self.assertRaisesRegex(RuntimeError, "Expected one of"):
            ChunkShardingSpec(0, ["rank:0/foo", "cuda:1"])
        with self.assertRaisesRegex(RuntimeError, "Expected one of"):
            ChunkShardingSpec(0, ["rank:0/random:0", "cuda:1"])
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            ChunkShardingSpec(0, ["rank:0/cuda:foo", "cuda:1"])

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "2 CUDA GPUs are needed")
    def test_enumerable_sharding_spec(self):
        # [FUNCTION DOC] Checks EnumerableShardingSpec and check_tensor. Valid layouts (row-wise,
        # row and column, uneven sizes) pass check_tensor against the full tensor size. Invalid
        # inputs must raise ValueError: unparsable placement, mismatched offset and size lengths,
        # negative offsets or sizes, empty shard list, inconsistent shard ranks, overlapping
        # shards, tensor rank mismatch, shards exceeding the tensor, and shard volume not
        # matching tensor volume.
        # test valid specs

        # test row-wise sharding
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:1",
                ),
            ]
        )
        check_tensor(spec.shards, torch.Size([10, 5]))

        # test row and column sharding
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[3, 3],
                    placement="cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 3],
                    shard_sizes=[3, 3],
                    placement="cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[3, 0],
                    shard_sizes=[3, 3],
                    placement="cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[3, 3],
                    shard_sizes=[3, 3],
                    placement="cuda:3",
                ),
            ]
        )
        check_tensor(spec.shards, torch.Size([6, 6]))

        # test uneven shard sizes.
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[2, 4],
                    placement="cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 4],
                    shard_sizes=[4, 2],
                    placement="cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[2, 0],
                    shard_sizes=[4, 4],
                    placement="cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[4, 4],
                    shard_sizes=[2, 2],
                    placement="cuda:3",
                ),
            ]
        )
        check_tensor(spec.shards, torch.Size([6, 6]))

        # test invalid sharding
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            ShardMetadata(shard_offsets=[0], shard_sizes=[1], placement="cuda:foo")

        with self.assertRaisesRegex(ValueError, "same number of elements"):
            ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, "shard_offsets should be >=0"):
            ShardMetadata(shard_offsets=[-1, 0], shard_sizes=[1, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, "shard_sizes should be >= 0"):
            ShardMetadata(shard_offsets=[0, 0], shard_sizes=[-1, 1], placement="cuda:0")

        with self.assertRaisesRegex(ValueError, "Empty shard list provided"):
            EnumerableShardingSpec([])

        with self.assertRaisesRegex(ValueError, "Found inconsistent ranks for shards"):
            EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_offsets=[0, 0], shard_sizes=[1, 1], placement="cpu"
                    ),
                    ShardMetadata(
                        shard_offsets=[0, 0, 0], shard_sizes=[1, 1, 1], placement="cpu"
                    ),
                ]
            )

        with self.assertRaisesRegex(ValueError, "Shards.*overlap"):
            EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_offsets=[0, 0], shard_sizes=[3, 3], placement="cpu"
                    ),
                    ShardMetadata(
                        shard_offsets=[2, 0], shard_sizes=[3, 3], placement="cpu"
                    ),
                ]
            )

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:1",
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "Rank of tensor is.*but shards rank"):
            check_tensor(spec.shards, torch.Size([10, 10, 10]))

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:1",
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "exceeds tensor dim"):
            check_tensor(spec.shards, torch.Size([10, 3]))

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="cuda:1",
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "does not match tensor volume"):
            check_tensor(spec.shards, torch.Size([10, 10]))

    def test_get_split_size(self):
        # [FUNCTION DOC] Checks get_split_size, the ceiling of dim size over number of chunks,
        # for uneven, even, single-chunk and one-per-element cases.
        self.assertEqual(3, get_split_size(11, 4))
        self.assertEqual(3, get_split_size(12, 4))
        self.assertEqual(4, get_split_size(13, 4))
        self.assertEqual(2, get_split_size(5, 4))

        self.assertEqual(11, get_split_size(11, 1))
        self.assertEqual(1, get_split_size(11, 11))

    def test_get_chunked_dim_size(self):
        # [FUNCTION DOC] Checks get_chunked_dim_size, the size of one chunk given dim size, split
        # size and chunk index; the last chunk may be smaller and chunks past the end are size 0.
        self.assertEqual(3, get_chunked_dim_size(11, 3, 0))
        self.assertEqual(2, get_chunked_dim_size(11, 3, 3))
        self.assertEqual(4, get_chunked_dim_size(13, 4, 0))
        self.assertEqual(1, get_chunked_dim_size(13, 4, 3))
        self.assertEqual(0, get_chunked_dim_size(5, 2, 3))

    def test_get_chunk_sharding_params(self):
        # [FUNCTION DOC] Checks get_chunk_sharding_params for a 4-way chunk spec on a size 21
        # dim: the offset and length for given ranks, first with natural placement order and then
        # after the placements list is permuted, where the offset follows the position of the
        # rank in the placements.
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]
        spec = ChunkShardingSpec(
            dim=0,
            placements=ranks,
        )
        result = get_chunk_sharding_params(21, 4, spec, 1)
        self.assertEqual(6, result[0])
        self.assertEqual(6, result[1])
        result = get_chunk_sharding_params(21, 4, spec, 3)
        self.assertEqual(18, result[0])
        self.assertEqual(3, result[1])
        ranks[1], ranks[2] = ranks[2], ranks[1]
        ranks[0], ranks[3] = ranks[3], ranks[0]
        spec.placements = ranks
        result = get_chunk_sharding_params(21, 4, spec, 1)
        self.assertEqual(12, result[0])
        self.assertEqual(6, result[1])
        result = get_chunk_sharding_params(21, 4, spec, 3)
        self.assertEqual(0, result[0])
        self.assertEqual(6, result[1])

    def _infer_enum_sharding_spec_case(self):
        # [FUNCTION DOC] HELPER: Builds three hand-written lists of shard metadata that are not a
        # regular chunking and checks that inferring a spec from each yields an
        # EnumerableShardingSpec holding the same shards. Not named as a test, so it only runs
        # when called by the inference test.
        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[10, 5],
                placement="cuda:1",
            ),
        ]
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0],
                shard_sizes=[16],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[16],
                shard_sizes=[9],
                placement="cuda:1",
            ),
        ]
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="rank:0/cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="rank:1/cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="rank:2/cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="rank:3/cuda:3",
            ),
        ]
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

    def _infer_chunk_sharding_spec_case(self, placements, sharding_dim, st_size):
        # [FUNCTION DOC] HELPER: Builds the shard metadata that a chunk sharding of a tensor of
        # size st_size along sharding_dim would produce for the given placements (offset by split
        # size times index, size from the chunked dim size, stored at the position of the
        # placement rank), then checks that inference returns a ChunkShardingSpec with the same
        # dim and placements.
        world_size = len(placements)
        split_size = get_split_size(st_size[sharding_dim], world_size)
        shards_metadata = [None] * world_size
        for idx, placement in enumerate(placements):
            shard_size = copy.deepcopy(st_size)
            offsets = [0] * len(st_size)
            offsets[sharding_dim] = split_size * idx
            shard_size[sharding_dim] = get_chunked_dim_size(
                st_size[sharding_dim], split_size, idx
            )
            shards_metadata[placement.rank()] = ShardMetadata(
                shard_offsets=offsets,
                shard_sizes=shard_size,
                placement=placement,
            )

        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, ChunkShardingSpec))
        self.assertEqual(spec.dim, sharding_dim)
        self.assertEqual(spec.placements, placements)

    def test_infer_sharding_spec_from_shards_metadata(self):
        # [FUNCTION DOC] Runs the enumerable inference case, then for each random chunk spec
        # generated with a fixed seed runs the chunk inference case over tensor shapes of rank 2
        # to 5 and sharding dims 0 to 4.
        self._infer_enum_sharding_spec_case()
        chunk_specs = _chunk_sharding_specs_list_for_test([0, 0, 1, 1], seed=31)
        for spec in chunk_specs:
            self._infer_chunk_sharding_spec_case(spec.placements, 0, [4, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 0, [5, 15, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 1, [12, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 2, [4, 18, 15])
            self._infer_chunk_sharding_spec_case(spec.placements, 3, [7, 12, 16, 37])
            self._infer_chunk_sharding_spec_case(
                spec.placements, 4, [50, 4, 18, 15, 77]
            )

    def test_check_overlapping(self):
        # [FUNCTION DOC] Checks validate_non_overlapping_shards_metadata on 2D and 3D shard
        # layouts: adjacent, diagonal and grid layouts pass, while shards intersecting in every
        # dimension raise ValueError mentioning overlap.
        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="cuda:1",
            ),
        ]
        validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[4, 0],
                shard_sizes=[5, 5],
                placement="cuda:1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 4],
                shard_sizes=[5, 5],
                placement="cuda:1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[5, 0, 5],
                shard_sizes=[5, 5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 5, 5],
                shard_sizes=[5, 5, 5],
                placement="cuda:1",
            ),
        ]
        validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[5, 0, 5],
                shard_sizes=[5, 5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 4, 5],
                shard_sizes=[5, 5, 5],
                placement="cuda:1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[5, 0, 5],
                shard_sizes=[5, 5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 4, 9],
                shard_sizes=[5, 5, 5],
                placement="cuda:1",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="cuda:3",
            ),
        ]
        validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="cuda:1",
            ),
        ]
        validate_non_overlapping_shards_metadata(shards)

        shards = [
            ShardMetadata(
                shard_offsets=[0, 0, 0],
                shard_sizes=[5, 5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0, 0],
                shard_sizes=[5, 5, 5],
                placement="cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[10, 0, 0],
                shard_sizes=[5, 5, 5],
                placement="cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[10, 3, 0],
                shard_sizes=[5, 5, 5],
                placement="cuda:3",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)


# Custom ShardingSpec, an simple example to do grid sharding
@dataclass
class GridShardingSpec(ShardingSpec):
    # [CLASS DOC] Example user-defined ShardingSpec (dataclass) that tiles a 2D tensor into
    # square blocks of side grid_size placed on the given placements. Fields: grid_size and
    # placements. Only build_metadata is implemented; shard raises NotImplementedError.
    grid_size: int
    placements: list[torch.distributed._remote_device | str]

    def __post_init__(self):
        # [FUNCTION DOC] Dataclass post-init hook: converts, in place in the placements list,
        # every entry that is not already a remote device into a remote device parsed from it.
        for i, remote_device in enumerate(self.placements):
            if not isinstance(remote_device, torch.distributed._remote_device):
                self.placements[i] = torch.distributed._remote_device(remote_device)

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: TensorProperties,
    ) -> ShardedTensorMetadata:
        # [FUNCTION DOC] Builds ShardedTensorMetadata for a 2D tensor: raises AssertionError if
        # the tensor is not 2D, if either size is not divisible by grid_size, or if the number of
        # chunk tiles differs from the number of placements. Returns one ShardMetadata per tile
        # with the stated sizes and placements, plus the given size and properties.
        # KNOWN ISSUE: the tile count per axis is computed by dividing the size by grid_size, but
        # offsets use the row or column index times grid_size and the placement index uses the
        # row index times the row chunk count, which is only correct when the row and column
        # chunk counts are equal.
        tensor_num_dim = len(tensor_sizes)
        if tensor_num_dim != 2:
            raise AssertionError("only support 2-dim tensor for grid sharding")
        shards_metadata = []

        def chunk_num(dim_size, grid_size):
            # [FUNCTION DOC] HELPER: Returns dim_size divided by grid_size using integer
            # division; raises AssertionError if the division is not exact.
            if dim_size % grid_size != 0:
                raise AssertionError("only support dim_size mod grid_size == 0")
            return dim_size // grid_size

        row_chunks = chunk_num(tensor_sizes[0], self.grid_size)
        col_chunks = chunk_num(tensor_sizes[1], self.grid_size)

        if row_chunks * col_chunks != len(self.placements):
            raise AssertionError(
                f"Expected row_chunks * col_chunks == len(self.placements), "
                f"got {row_chunks * col_chunks} vs {len(self.placements)}"
            )
        for row_idx in range(row_chunks):
            for col_idx in range(col_chunks):
                shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=[
                            row_idx * self.grid_size,
                            col_idx * self.grid_size,
                        ],
                        shard_sizes=[self.grid_size, self.grid_size],
                        placement=self.placements[row_idx * row_chunks + col_idx],
                    )
                )
        return ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=tensor_sizes,
            tensor_properties=tensor_properties,
        )

    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> ShardedTensor:
        # [FUNCTION DOC] Not supported: always raises NotImplementedError, regardless of
        # arguments.
        raise NotImplementedError("GridShardingSpec.shard not implemented yet!")


class TestCustomShardingSpec(ShardedTensorTestBase):
    # [CLASS DOC] Test case on ShardedTensorTestBase for a user-defined ShardingSpec subclass.
    def test_custom_sharding_spec(self):
        # [FUNCTION DOC] Builds a GridShardingSpec of four placements and block size 4, produces
        # metadata for an 8 by 8 float tensor, and validates it with check_tensor.
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]

        grid_spec = GridShardingSpec(grid_size=4, placements=ranks)

        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        meta = grid_spec.build_metadata(torch.Size((8, 8)), tensor_properties)
        check_tensor(meta.shards_metadata, torch.Size((8, 8)))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_sharding_spec_tensor_ctor(self):
        # [FUNCTION DOC] Creates a ones sharded tensor of 4 by 4 with a GridShardingSpec over
        # four ranks and asserts each rank holds a single 2 by 2 local shard of ones on its own
        # cuda device.
        """Test sharded_tensor.ones(...) with the custom
        grid sharding spec.
        """

        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]

        grid_spec = GridShardingSpec(grid_size=2, placements=ranks)

        st = sharded_tensor.ones(grid_spec, 4, 4)

        # Validate local shard is initialized with torch.ones
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        self.assertEqual((2, 2), local_shard.size())
        self.assertEqual(local_shard, torch.ones(2, 2))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_sharding_spec_shard_tensor(self):
        # [FUNCTION DOC] Asserts that sharding a plain tensor through the private shard tensor
        # entry point with GridShardingSpec raises NotImplementedError, since the spec's shard
        # method is unimplemented.
        """Test custom spec can be invoked from the
        _shard_tensor callsite.
        """

        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]

        grid_spec = GridShardingSpec(grid_size=2, placements=ranks)

        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            _shard_tensor(torch.randn(8, 8), grid_spec)


if __name__ == "__main__":
    run_tests()
