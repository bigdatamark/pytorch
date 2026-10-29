# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Four-rank tests of torch.equal and torch.allclose on ShardedTensor operands.
# Import-time side effects: computes device_type and the default backend, and if the dev-asan
# flag is set it prints a message to stderr and exits the process with status 0.

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)
backend = torch.distributed.get_default_backend_for_device(device_type)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorBinaryOps(ShardedTensorTestBase):
    # [CLASS DOC] Multi-process test case for binary comparison ops on ShardedTensor. The
    # class attribute seed (starts at 42) is shared state, bumped on every call to
    # get_random_tensors so each test pair gets fresh random data.
    """Test base for binary comparison functions such as torch.equal, torch.allclose etc. for ShardedTensor"""

    seed = 42

    def get_random_tensors(
        self, spec1, spec2, *sizes, pg1=None, pg2=None, seed_offset=0
    ):
        # [FUNCTION DOC] HELPER: Creates two random ShardedTensors of the given sizes using
        # spec1/pg1 and spec2/pg2 (default group if None). The RNG is seeded with the class
        # seed for the first and class seed plus seed_offset for the second, so with offset
        # 0 the values match. Increments the class seed afterwards; returns (st1, st2).
        pg1 = _get_default_group() if pg1 is None else pg1
        pg2 = _get_default_group() if pg2 is None else pg2
        torch.manual_seed(TestShardedTensorBinaryOps.seed)
        st1 = sharded_tensor.rand(spec1, sizes, process_group=pg1)
        torch.manual_seed(TestShardedTensorBinaryOps.seed + seed_offset)
        st2 = sharded_tensor.rand(spec2, sizes, process_group=pg2)

        TestShardedTensorBinaryOps.seed += 1
        return st1, st2

    def get_gpu_specs(self):
        # [FUNCTION DOC] HELPER: Returns two dim-0 chunk sharding specs over four ranks on
        # the accelerator: the identity placement and one with ranks swapped pairwise
        # (1,0,3,2), which yields a different placement order.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )

        alt_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:1/{device_type}:1",
                f"rank:0/{device_type}:0",
                f"rank:3/{device_type}:3",
                f"rank:2/{device_type}:2",
            ],
        )
        return spec, alt_spec

    def _test_common_failures(self, cmp_op):
        # [FUNCTION DOC] HELPER: Asserts that comparison function cmp_op returns False for
        # pairs differing in one local shard's values, size, placement order, values
        # (ones vs zeros), dtype, requires_grad, or pin_memory (cpu specs). Also asserts it
        # raises RuntimeError about needing the same ProcessGroup when the two tensors use
        # different (even equal-rank) groups. Creates new process groups (collective).
        spec, alt_spec = self.get_gpu_specs()

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        if self.rank == 0:
            torch.nn.init.uniform_(st1.local_shards()[0].tensor)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 5)
        self.assertFalse(cmp_op(st1, st2))

        st1, st2 = self.get_random_tensors(spec, alt_spec, 10, 10)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.zeros(spec, 10, 10)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, dtype=torch.double)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, requires_grad=True)
        self.assertFalse(cmp_op(st1, st2))

        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        st1 = sharded_tensor.ones(cpu_spec, 10, 10)
        st2 = sharded_tensor.ones(cpu_spec, 10, 10, pin_memory=True)
        self.assertFalse(cmp_op(st1, st2))

        pg = dist.new_group([1, 0, 3, 2])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

        pg = dist.new_group([0, 1, 2, 3])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_torch_equal_tensor_specs(self):
        # [FUNCTION DOC] Runs the shared failure cases against torch.equal.
        self._test_common_failures(torch.equal)

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_torch_equal(self):
        # [FUNCTION DOC] Checks torch.equal is True for two identically seeded sharded
        # tensors with the same spec.
        """Test torch.equal(ShardedTensor, ShardedTensor)"""

        spec, _ = self.get_gpu_specs()
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.equal(st1, st2))

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_torch_allclose_tensor_specs(self):
        # [FUNCTION DOC] Runs the shared failure cases against torch.allclose.
        self._test_common_failures(torch.allclose)

    @with_comms(backend=backend)
    @skip_if_lt_x_gpu(4)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_torch_allclose(self):
        # [FUNCTION DOC] Checks torch.allclose is True for identical sharded tensors (also
        # with zero atol), False for differently seeded ones with default tolerances, and
        # True for those with atol 1 since random values lie in the unit interval.
        """Test torch.allclose(ShardedTensor, ShardedTensor)"""

        spec, _ = self.get_gpu_specs()

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.allclose(st1, st2))
        self.assertTrue(torch.allclose(st1, st2, atol=0))

        # compare different arrays
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, seed_offset=1)
        self.assertFalse(torch.allclose(st1, st2))
        # sharded_tensor.rand produces uniform values in the [0,1] range.
        self.assertTrue(torch.allclose(st1, st2, atol=1))


if __name__ == "__main__":
    run_tests()
