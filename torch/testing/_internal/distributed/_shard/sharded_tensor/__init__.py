# mypy: allow-untyped-defs
# [PACKAGE DOC] HELPER: Test infrastructure for sharded tensor tests. Exports TEST_GPU_NUM (the
# fixed world size, 4), ShardedTensorTestBase (a multi-process test case that can initialize and
# tear down a process group plus RPC), and the with_comms decorator. Submodules _test_ops_common
# and _test_st_common hold sharding spec generators and sample models.

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    tp_transports,
)


TEST_GPU_NUM = 4


class ShardedTensorTestBase(MultiProcessTestCase):
    # [CLASS DOC] HELPER: Base class for sharded tensor tests that run one process per rank
    # (world size fixed at TEST_GPU_NUM). setUp spawns the worker processes; helper methods set up
    # and tear down the process group and RPC, and compare two sharded tensors for equality.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Returns the fixed number of ranks, TEST_GPU_NUM.
        return TEST_GPU_NUM

    def init_pg(self, backend="nccl"):
        # [FUNCTION DOC] HELPER: Initializes the default process group for this rank using a
        # file-based rendezvous at the test's file name. Raises RuntimeError for backends other
        # than nccl, gloo, mpi, hccl or xccl. For nccl or xccl, also sets the current accelerator
        # device index to this rank. Side effect: global process group state.
        if backend not in ["nccl", "gloo", "mpi", "hccl", "xccl"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # set device for nccl pg for collectives
        if backend == "nccl" or backend == "xccl":
            torch.accelerator.set_device_index(self.rank)

    def init_rpc(self):
        # [FUNCTION DOC] HELPER: Initializes RPC for this rank with the TensorPipe backend and a
        # file-based init method, naming workers by rank. Each rank gets a device map to every
        # worker that maps the own rank to the peer rank and the peer rank to the own rank.
        # Side effect: global RPC agent state.
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            _transports=tp_transports()
        )
        rpc_backend_options.init_method = f"file://{self.file_name}"
        for rank in range(self.world_size):
            rpc_backend_options.set_device_map(
                f"worker{rank}", {rank: self.rank, self.rank: rank}
            )

        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

    def init_comms(self, init_rpc=True, backend="nccl"):
        # [FUNCTION DOC] HELPER: Sets up communication for the test: RPC first (unless init_rpc is
        # false), then the process group for the given backend.
        if init_rpc:
            self.init_rpc()
        self.init_pg(backend=backend)

    def destroy_comms(self, destroy_rpc=True):
        # [FUNCTION DOC] HELPER: Tears down communication: waits at a barrier so all ranks finish,
        # shuts down RPC if destroy_rpc is true, then destroys the default process group.
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()

        if destroy_rpc:
            rpc.shutdown()
        dist.destroy_process_group()

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the parent setup, then spawns the per-rank worker processes.
        super().setUp()
        self._spawn_processes()

    def assert_sharded_tensor_equal(self, st1, st2):
        # [FUNCTION DOC] HELPER: Asserts two sharded tensors match: same number of local shards
        # with equal shard tensors and shard metadata pairwise, equal overall metadata, equal
        # sharding specs, and the same number of remote shards. Fails the test on any mismatch.
        st1_local_shards = st1.local_shards()
        st2_local_shards = st2.local_shards()
        self.assertEqual(len(st1_local_shards), len(st2_local_shards))
        for i, st1_local_shard in enumerate(st1_local_shards):
            self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
            self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)

        self.assertEqual(st1.metadata(), st2.metadata())
        self.assertEqual(st1.sharding_spec(), st2.sharding_spec())
        self.assertEqual(len(st1.remote_shards()), len(st2.remote_shards()))


# wrapper to initialize comms (processgroup + rpc)
def with_comms(func=None, init_rpc=True, backend="nccl"):
    # [FUNCTION DOC] HELPER: Decorator for test methods of ShardedTensorTestBase. Usable bare or
    # with arguments (when func is None it returns a partial for later application). The wrapped
    # test first exits the process with the multi-GPU skip exit code if an accelerator backend
    # (nccl, xccl, hccl) is requested but there is no accelerator, its default backend differs, or
    # fewer devices than the world size exist. Otherwise it sets up comms, runs the test, then
    # tears comms down. Returns the wrapper.
    # KNOWN ISSUE: if the test body raises, destroy_comms is skipped (no try/finally).
    if func is None:
        return partial(
            with_comms,
            init_rpc=init_rpc,
            backend=backend,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Skip-or-run wrapper described in with_comms; passes all arguments
        # through to the test function and returns None (the test's return value is dropped).
        # Skip test if backend requires accelerator but not enough devices available
        acc = torch.accelerator.current_accelerator()
        if backend in ["nccl", "xccl", "hccl"]:
            if (
                acc is None
                or backend != dist.get_default_backend_for_device(acc)
                or torch.accelerator.device_count() < self.world_size
            ):
                sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        self.init_comms(init_rpc=init_rpc, backend=backend)
        func(self, *args, **kwargs)
        self.destroy_comms(destroy_rpc=init_rpc)

    return wrapper
