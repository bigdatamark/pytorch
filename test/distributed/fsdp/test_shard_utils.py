# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Two-rank tests for the FSDP shard helpers that build a ChunkShardingSpec-style
# ShardedTensor and a row-sharded DTensor from a full tensor. Defines one FSDPTest class and one
# DTensorTestBase class plus module-level device_type (current accelerator, else cpu). Runs
# tests via run_tests when executed as a script; no import-time side effects beyond that.

import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestShardUtilsDistributed(FSDPTest):
    # [CLASS DOC] Distributed test case (2 ranks) checking _create_chunk_sharded_tensor
    # against the unsharded source tensor. Stateless apart from the FSDPTest base fixtures.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the number of ranks to spawn for this test class: always 2.
        return 2

    def _create_tensor(self, *size):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 and returns a uniform random
        # tensor of the given shape on device_type, so every rank builds identical data.
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).to(device=device_type)

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        # [FUNCTION DOC] For several 1-D and 2-D shapes (including sizes not divisible by the
        # world size), builds a chunk-sharded ShardedTensor on each rank, gathers it to rank 0
        # and asserts it equals the original tensor. Skipped when fewer than 2 GPUs exist.
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)

            sharded_tensor = _create_chunk_sharded_tensor(
                tensor,
                self.rank,
                self.world_size,
                torch.accelerator.device_count(),
                _get_default_group(),
            )
            output = (
                torch.empty(*size).to(device=device_type) if self.rank == 0 else None
            )
            sharded_tensor.gather(0, output)
            if self.rank == 0:
                self.assertEqual(tensor, output)


class TestShardUtilsDistributedDTensor(DTensorTestBase):
    # [CLASS DOC] Distributed test case (2 ranks) checking _create_chunk_dtensor, which wraps
    # a rank's chunk of a full tensor as a DTensor over a device mesh. No own state.
    @property
    def world_size(self):
        # [FUNCTION DOC] Returns the number of ranks to spawn for this test class: always 2.
        return 2

    def _create_tensor(self, *size):
        # [FUNCTION DOC] HELPER: Seeds the global RNG with 0 and returns a uniform random
        # tensor of the given shape on device_type, so every rank builds identical data.
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).to(device=device_type)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_create_chunk_dtensor(self):
        # [FUNCTION DOC] For several shapes, builds a chunked DTensor from the full tensor and
        # checks the local shard equals this rank's piece of a dim-0 chunk split; when the local
        # shard is empty, asserts the rank index is beyond the number of chunks produced.
        device_mesh = self.build_device_mesh()

        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)
            tensor_chunks = torch.chunk(tensor, self.world_size, dim=0)

            dtensor = _create_chunk_dtensor(tensor, self.rank, device_mesh)
            local_tensor = dtensor.to_local()

            if local_tensor.numel() != 0:
                self.assertEqual(local_tensor, tensor_chunks[self.rank])
            else:
                self.assertEqual(self.rank >= len(tensor_chunks), True)


if __name__ == "__main__":
    run_tests()
