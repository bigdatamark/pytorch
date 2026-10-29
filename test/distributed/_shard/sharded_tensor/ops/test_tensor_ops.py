# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Multi-process GPU tests of basic tensor operations on ShardedTensor: deepcopy,
# in-place copy_, clone, detach and requires_grad_. All use a 4-way row-chunk sharding of a
# 12 by 5 tensor. Import-time side effect: computes module-level device_type and backend from
# the current accelerator (cpu fallback).

import copy

import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import (
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)
backend = torch.distributed.get_default_backend_for_device(device_type)


class TestTensorOps(ShardedTensorTestBase):
    # [CLASS DOC] Distributed test case with one test per ShardedTensor basic operation; each
    # needs enough GPUs and an nccl or xccl backend, with RPC not initialized.
    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_deep_copy(self):
        # [FUNCTION DOC] Deep-copies a random sharded tensor; asserts the copy has the same
        # type, equal local tensor contents, and is a different object.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        copied_st = copy.deepcopy(st)
        self.assertTrue(type(copied_st) is type(st))
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_inplace_copy(self):
        # [FUNCTION DOC] Copies an all-ones sharded tensor into a random one and checks they
        # become equal. Then, under no-grad, copies into a sharded tensor that requires grad
        # (source does not) and checks the local tensors match.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        ones_st = sharded_tensor.ones(spec, (12, 5))
        self.assertFalse(torch.equal(ones_st, st))
        st.copy_(ones_st)
        self.assertTrue(torch.equal(st, ones_st))

        # no grad inplace_copy should work between two with different requires_grad
        st_with_grad = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        self.assertTrue(st_with_grad.requires_grad)
        self.assertFalse(ones_st.requires_grad)
        with torch.no_grad():
            st_with_grad.copy_(ones_st)
            self.assertEqual(st_with_grad.local_tensor(), ones_st.local_tensor())

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_clone(self):
        # [FUNCTION DOC] Clones a random sharded tensor; asserts same type, equal local
        # tensor contents, and a distinct object.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        copied_st = st.clone()
        self.assertTrue(type(copied_st) is type(st))
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_detach(self):
        # [FUNCTION DOC] Creates a sharded tensor requiring grad, checks its local shards
        # require grad, then detaches and asserts the result and all its local shards do not.
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        local_shards = st.local_shards()
        # before set requires_grad, all local shards should not require grads
        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)

        detached_st = st.detach()
        self.assertFalse(detached_st.requires_grad)

        for local_shard in detached_st.local_shards():
            self.assertFalse(local_shard.tensor.requires_grad)

    @with_comms(init_rpc=False, backend=backend)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    def test_set_requires_grad(self):
        # [FUNCTION DOC] Checks local shards start without grad, calls requires_grad_ on the
        # sharded tensor, then asserts the tensor and the previously fetched local shards
        # all require grad (the change propagates to shards in place).
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:0/{device_type}:0",
                f"rank:1/{device_type}:1",
                f"rank:2/{device_type}:2",
                f"rank:3/{device_type}:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        local_shards = st.local_shards()
        # before set requires_grad, all local shards should not require grads
        for local_shard in local_shards:
            self.assertFalse(local_shard.tensor.requires_grad)

        st.requires_grad_()
        self.assertTrue(st.requires_grad)

        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)


if __name__ == "__main__":
    run_tests()
