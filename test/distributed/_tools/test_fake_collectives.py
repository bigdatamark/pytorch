# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests that the fake collectives tooling (CollectiveOp, collective_ops,
# non_functional_collectives) correctly recognises the process group and computes communication
# tensor sizes for every c10d and functional collective, using a fake process group of world size
# 4 under FakeTensorMode on CUDA. Defines a TorchDispatchMode that checks each intercepted
# collective. No import-time side effects beyond aliasing torch.ops namespaces.
import unittest

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import FakeWork, ProcessGroup
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import (
    all_gather_single,
    all_gather_single_autograd,
    all_gather_single_coalesced,
    all_reduce,
    all_reduce_coalesced,
    all_to_all_single,
    all_to_all_single_autograd,
    broadcast,
    reduce_scatter_single,
    reduce_scatter_single_autograd,
    reduce_scatter_single_coalesced,
    wait_tensor,
)
from torch.distributed._tools.fake_collectives import (
    collective_ops,
    CollectiveOp,
    non_functional_collectives,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten
c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd


class TestFakeCollectives(TestCase):
    # [CLASS DOC] Test case that drives one of each supported collective under fake tensors and
    # the CollectiveTest dispatch mode.
    def _setup_distributed(self):
        # [FUNCTION DOC] HELPER: Initialises the global default process group with the fake
        # backend (rank 0, world size 4, a FakeStore) and sets the current CUDA device. Side
        # effect: creates global distributed state that the caller must destroy.
        world_size = 4
        store = FakeStore()
        dist.init_process_group("fake", rank=0, world_size=world_size, store=store)
        torch.cuda.set_device(torch.cuda.current_device())

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_collectives(self):
        # [FUNCTION DOC] Under FakeTensorMode and the CollectiveTest checker, issues
        # non-functional collectives (broadcast, all reduce, reduce, send, recv, all gather,
        # reduce scatter, scatter, gather, all to all, barrier) and functional ones (including
        # autograd and coalesced variants) on fake CUDA tensors. Assertions happen in the
        # dispatch mode. Always destroys the process group in a finally block. Skipped without
        # CUDA.
        try:
            self._setup_distributed()
            with FakeTensorMode(), CollectiveTest(test=self):
                test_tensor_list = [torch.randn(100, device="cuda") for _ in range(4)]
                test_tensor_list_2 = [torch.randn(400, device="cuda") for _ in range(4)]
                test_tensor = torch.randn(100, device="cuda")
                # Used as gather output or scatter input
                test_tensor2 = torch.randn(400, device="cuda")

                # Testing non-functional collective operations
                dist.broadcast(test_tensor, src=0)
                dist.all_reduce(test_tensor)
                dist.reduce(test_tensor, dst=0)
                dist.send(test_tensor, dst=1)
                dist.recv(test_tensor, src=1)
                dist.all_gather(test_tensor_list, test_tensor)
                dist.reduce_scatter(test_tensor, test_tensor_list)
                dist.reduce_scatter_single(test_tensor, test_tensor2)
                dist.scatter(test_tensor, scatter_list=test_tensor_list, src=0)
                dist.gather(test_tensor, gather_list=test_tensor_list, dst=0)
                dist.all_gather_single(test_tensor2, test_tensor)
                dist.all_to_all(test_tensor_list, test_tensor_list)
                dist.all_to_all_single(test_tensor2, test_tensor2)
                dist.barrier()

                # Testing functional collectives
                wait_tensor(test_tensor)
                broadcast(test_tensor, src=0, group=dist.group.WORLD)
                all_reduce(test_tensor, reduceOp="avg", group=dist.group.WORLD)
                all_gather_single(test_tensor, gather_dim=0, group=dist.group.WORLD)
                all_gather_single_autograd(
                    test_tensor, gather_dim=0, group=dist.group.WORLD
                )
                reduce_scatter_single(
                    test_tensor2, scatter_dim=0, reduceOp="sum", group=dist.group.WORLD
                )
                reduce_scatter_single_autograd(
                    test_tensor2, scatter_dim=0, reduceOp="sum", group=dist.group.WORLD
                )
                all_to_all_single(
                    test_tensor,
                    output_split_sizes=[0],
                    input_split_sizes=[1],
                    group=dist.group.WORLD,
                )
                all_reduce_coalesced(
                    test_tensor_list, reduceOp="avg", group=dist.group.WORLD
                )
                all_gather_single_coalesced(test_tensor_list, group=dist.group.WORLD)
                reduce_scatter_single_coalesced(
                    test_tensor_list_2,
                    scatter_dim=[0] * 4,
                    reduceOp="sum",
                    group=dist.group.WORLD,
                )
                all_to_all_single_autograd(
                    test_tensor,
                    output_split_sizes=[0],
                    input_split_sizes=[1],
                    group=dist.group.WORLD,
                )
        finally:
            if dist.group.WORLD is not None:
                dist.destroy_process_group()


class CollectiveTest(TorchDispatchMode):
    # [CLASS DOC] TorchDispatchMode that executes each op, and for recognised collectives
    # asserts: the process group is a ProcessGroup equal to the world group with size 4 and a
    # non-empty name; the computed communication size equals the expected size (barriers and
    # wait_tensor are exempt); and non-functional collectives return a fake work object. Holds
    # the owning test case. The class attribute collective_size_exclude lists ops exempt from
    # size checks.
    collective_size_exclude = {
        c10d.barrier.default,
        c10d.monitored_barrier_.default,
        _c10d_functional.wait_tensor.default,
    }

    def __init__(self, test: TestFakeCollectives, _dispatch_key=None):
        # [FUNCTION DOC] Initialises the dispatch mode with the optional dispatch key and stores
        # the test case used for assertions.
        super().__init__(_dispatch_key)
        self.test = test

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Runs the intercepted op and returns its result unchanged. For ops in
        # collective_ops (except wait_tensor) checks the process group extracted from the args;
        # for those not excluded, compares the computed communication size with the expected size
        # from get_expected_size; for non-functional collectives other than the monitored
        # barrier, checks the last result element (or the result itself) unboxes to a FakeWork.
        # Failures surface as test assertion errors.
        res = func(*args, **(kwargs or {}))

        if func in collective_ops:
            if func != _c10d_functional.wait_tensor.default:
                pg = CollectiveOp.get_process_group(func, args)
                self.test.assertIsInstance(
                    pg, ProcessGroup, "Error: pg is not an instance of ProcessGroup"
                )
                self.test.assertEqual(
                    pg, dist.group.WORLD, "Error: pg is not equal to dist.group.WORLD"
                )
                self.test.assertEqual(
                    pg.size(),
                    4,
                    lambda msg: f"{msg}\nError: Expected pg.size() to be 4, but got {pg.size()}",
                )
                self.test.assertNotEqual(
                    pg.name(), "", "Error: pg.name() should not be an empty string"
                )

            if func not in CollectiveTest.collective_size_exclude:
                # Compute expected communication tensor size
                computed_size = CollectiveOp.get_comm_tensor_size(
                    func, res, args, kwargs
                )
                expected_size = self.get_expected_size(func, res, args, kwargs)

                self.test.assertEqual(
                    computed_size,
                    expected_size,
                    msg=lambda msg: f"{msg}\nSize mismatch for {func.__name__}: expected {expected_size}, got {computed_size}",
                )

        if (
            func in non_functional_collectives
            and func != c10d.monitored_barrier_.default
        ):
            work = res[-1] if isinstance(res, (tuple, list)) else res
            self.test.assertIsInstance(FakeWork.unbox(work), FakeWork)

        return res

    @staticmethod
    def get_expected_size(func, res, args, kwargs):
        # [FUNCTION DOC] Looks up the expected byte count of communicated data for an op in a
        # fixed mapping (assuming world size 4 and the specific 100 and 400 element float tensors
        # used by the test) and returns it. The result, args and kwargs arguments are unused.
        # Raises ValueError for an op not in the mapping.
        """Return expected tensor size for collectives explicitly used in run_test()."""
        WORLD_SIZE, TENSOR_100, TENSOR_400 = 4, 100 * 4, 400 * 4
        TENSOR_LIST_100, TENSOR_LIST_400 = (
            WORLD_SIZE * TENSOR_100,
            WORLD_SIZE * TENSOR_400,
        )

        size_map = {
            # Non-functional collectives
            c10d.broadcast_.default: TENSOR_100,
            c10d.allreduce_.default: TENSOR_100,
            c10d.reduce_.default: TENSOR_100,
            c10d.send.default: TENSOR_100,
            c10d.recv_.default: TENSOR_100,
            c10d.allgather_.default: TENSOR_LIST_100,
            c10d.reduce_scatter_.default: TENSOR_LIST_100,
            c10d._reduce_scatter_base_.default: TENSOR_400,
            c10d.scatter_.default: TENSOR_LIST_100,
            c10d.gather_.default: TENSOR_LIST_100,
            c10d._allgather_base_.default: TENSOR_400,
            c10d.alltoall_.default: TENSOR_LIST_100,
            c10d.alltoall_base_.default: TENSOR_400,
            # Functional collectives
            _c10d_functional.broadcast.default: TENSOR_100,
            _c10d_functional.all_reduce.default: TENSOR_100,
            _c10d_functional.all_gather_into_tensor.default: TENSOR_LIST_100,
            _c10d_functional_autograd.all_gather_into_tensor.default: TENSOR_LIST_100,
            _c10d_functional.reduce_scatter_tensor.default: TENSOR_400,
            _c10d_functional_autograd.reduce_scatter_tensor.default: TENSOR_400,
            _c10d_functional.all_to_all_single.default: TENSOR_100,
            _c10d_functional_autograd.all_to_all_single.default: TENSOR_100,
            _c10d_functional.all_reduce_coalesced.default: TENSOR_LIST_100,
            _c10d_functional.all_gather_into_tensor_coalesced.default: TENSOR_LIST_400,
            _c10d_functional.reduce_scatter_tensor_coalesced.default: TENSOR_LIST_100,
        }

        if func in size_map:
            return size_map[func]

        raise ValueError(f"Unhandled function: {func}")


if __name__ == "__main__":
    run_tests()
