# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the threaded process group backend: collectives
# (broadcast_object_list, all_gather, broadcast, scatter, reduce_scatter, all_reduce,
# all_to_all, gather, coalesced all_reduce, sub-groups) run with 4 ranks as threads in one
# process. Covers both the spawn_threads_and_init_comms decorator and the
# MultiThreadedTestCase base class, plus error propagation, skips and autograd backward thread
# affinity. Exits at import if distributed is unavailable. Defines device_type (current
# accelerator type or cpu) and DEFAULT_WORLD_SIZE.

import operator
import os
import sys
import threading
from functools import reduce
from unittest import skip, SkipTest

import torch
import torch.autograd
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
    spawn_threads_and_init_comms,
)
from torch.testing._internal.common_utils import IS_SANDCASTLE, run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

DEFAULT_WORLD_SIZE = 4


class TestCollectivesWithWrapper(TestCase):
    # [CLASS DOC] Tests using the spawn_threads_and_init_comms decorator to run test bodies on
    # 4 rank threads, including how errors and skips raised in a rank propagate to the caller.
    @spawn_threads_and_init_comms(world_size=4)
    def test_broadcast_object_list(self):
        # [FUNCTION DOC] Rank 0 provides value 99, other ranks None; after
        # broadcast_object_list (default source 0) element 0 of the list equals 99
        # on every rank.
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

    def test_collective_error_on_rank_zero(self):
        # [FUNCTION DOC] Defines a 4-rank threaded test where each rank does an
        # all_gather, then rank 0 raises AssertionError before the second all_gather;
        # asserts the invocation raises RuntimeError (the other ranks' failure is surfaced
        # rather than hanging).
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            # [FUNCTION DOC] HELPER: Per-rank body: all_gather of a rank-filled 3x3 tensor,
            # rank 0 raises AssertionError, then a second all_gather for the remaining ranks.
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            output_tensors = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(output_tensors, input_tensor)

            if dist.get_rank() == 0:
                raise AssertionError("Mimic real test failure.")  # fail on rank 0

            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        with self.assertRaises(RuntimeError):
            _test_method(self)

    def test_collective_error_on_rank_non_zero(self):
        # [FUNCTION DOC] Same as the rank-zero failure test, but rank 1 raises; asserts the
        # call raises RuntimeError.
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            # [FUNCTION DOC] HELPER: Per-rank body: all_gather, then rank 1 raises
            # AssertionError, others do a second all_gather.
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            output_tensors = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(output_tensors, input_tensor)

            if dist.get_rank() == 1:
                raise AssertionError("Mimic real test failure.")  # fail on rank 1

            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        with self.assertRaises(RuntimeError):
            _test_method(self)

    def test_collective_error_on_rank_non_zero_all(self):
        # [FUNCTION DOC] Same pattern, but every rank with rank greater than zero raises;
        # asserts the call raises RuntimeError.
        @spawn_threads_and_init_comms(world_size=4)
        def _test_method(self):
            # [FUNCTION DOC] HELPER: Per-rank body: all_gather, then ranks above zero raise
            # AssertionError, rank 0 attempts a second all_gather.
            input_tensor = torch.ones(3, 3) * dist.get_rank()  # perform 1st all gather
            output_tensors = [
                torch.empty_like(input_tensor) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(output_tensors, input_tensor)

            if dist.get_rank() > 0:
                raise AssertionError(
                    "Mimic real test failure."
                )  # fail on all non-zero rank

            dist.all_gather(output_tensors, input_tensor)  # perform 2nd all gather

        with self.assertRaises(RuntimeError):
            _test_method(self)

    def test_skip(self):
        # [FUNCTION DOC] Checks that a unittest skip raised inside the threaded test
        # body surfaces as SkipTest to the caller; the assertion is skipped when
        # running on Sandcastle.
        @spawn_threads_and_init_comms(world_size=4)
        @skip("check if skip exception can be captured correctly.")
        def _test_method(self):
            # [FUNCTION DOC] HELPER: Empty test body decorated with skip.
            pass

        if not IS_SANDCASTLE:
            with self.assertRaises(SkipTest):
                _test_method(self)

    @spawn_threads_and_init_comms(world_size=4)
    def test_all_to_all_single_tensor(self):
        # [FUNCTION DOC] all_to_all_single with tensor split sizes (all ones): each rank sends
        # rank-filled rows and receives one row from each rank; output rows equal the rank
        # indices in order.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        send = torch.full((world_size, 2), rank)
        sizes = torch.ones(world_size, dtype=torch.int64)

        out = torch.zeros(world_size, 2, dtype=send.dtype)
        dist.all_to_all_single(out, send, sizes, sizes)
        self.assertEqual(out.tolist(), list(zip(range(world_size), range(world_size))))

    @spawn_threads_and_init_comms(world_size=4)
    def test_all_to_all_single_list(self):
        # [FUNCTION DOC] Same as the tensor-sizes variant, but split sizes are passed
        # as Python lists.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        send = torch.full((world_size, 2), rank)
        sizes = [1] * world_size

        out = torch.zeros(world_size, 2, dtype=send.dtype)
        dist.all_to_all_single(out, send, sizes, sizes)
        self.assertEqual(out.tolist(), list(zip(range(world_size), range(world_size))))

    @spawn_threads_and_init_comms(world_size=4)
    def test_all_to_all_single_none(self):
        # [FUNCTION DOC] all_to_all_single without split sizes (even split); the output has
        # one row from each rank in rank order.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        send = torch.full((world_size, 2), rank)

        out = torch.zeros(world_size, 2, dtype=send.dtype)
        dist.all_to_all_single(out, send)
        self.assertEqual(out.tolist(), list(zip(range(world_size), range(world_size))))


class TestCollectivesWithBaseClass(MultiThreadedTestCase):
    # [CLASS DOC] Tests collectives with the MultiThreadedTestCase base class: 4 rank threads,
    # with the init barrier environment variable enabled during each test. Each test body
    # executes once per rank thread.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property: returns the world size, 4.
        return 4

    def setUp(self):
        # [FUNCTION DOC] Sets TORCH_DIST_INIT_BARRIER to 1 in the environment, runs the parent
        # setUp and spawns the rank threads.
        os.environ["TORCH_DIST_INIT_BARRIER"] = "1"
        super().setUp()
        self._spawn_threads()

    def tearDown(self):
        # [FUNCTION DOC] Runs the parent tearDown, then resets TORCH_DIST_INIT_BARRIER to 0
        # in the environment.
        super().tearDown()
        os.environ["TORCH_DIST_INIT_BARRIER"] = "0"

    def test_allgather(self):
        # [FUNCTION DOC] all_gather of a rank-filled 3x3 tensor; each output entry i equals a
        # tensor filled with i.
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [
            torch.empty_like(input_tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(output_tensors, input_tensor)
        for rank, out_tensor in enumerate(output_tensors):
            self.assertEqual(out_tensor, torch.ones(3, 3) * rank)

    def test_broadcast(self):
        # [FUNCTION DOC] For each source rank, broadcasts a clone of the rank-filled tensor
        # and checks every rank receives the source's values.
        input_tensor = torch.ones(3, 3) * dist.get_rank()
        for rank in range(self.world_size):
            cloned_input = input_tensor.clone()
            dist.broadcast(cloned_input, src=rank)
            self.assertEqual(cloned_input, torch.ones(3, 3) * rank)

    def test_scatter(self):
        # [FUNCTION DOC] Rank 0 scatters a list of rank-filled tensors; every rank receives a
        # tensor filled with its own rank.
        if dist.get_rank() == 0:
            scatter_list = [torch.ones(3, 3) * rank for rank in range(self.world_size)]
        else:
            scatter_list = None
        output_tensor = torch.empty(3, 3)

        dist.scatter(output_tensor, scatter_list)
        self.assertEqual(output_tensor, torch.ones(3, 3) * dist.get_rank())

    def test_reduce_scatter(self):
        # [FUNCTION DOC] reduce_scatter with default sum: rank r receives r times world
        # size per element. Then repeats with the AVG op and expects each rank to receive
        # its own rank value.
        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(self.world_size)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)
        expected_tensor = torch.ones(3, 3) * dist.get_rank() * self.world_size
        self.assertEqual(output_tensor, expected_tensor)

        output_tensor = torch.empty(3, 3)
        dist.reduce_scatter(output_tensor, to_reduce_scatter, op=dist.ReduceOp.AVG)
        expected_tensor = torch.ones(3, 3) * dist.get_rank()
        self.assertEqual(output_tensor, expected_tensor)

    def test_broadcast_object_list(self):
        # [FUNCTION DOC] Same as the decorator-based version: rank 0 supplies 99,
        # broadcast_object_list distributes it, and element 0 is checked on every rank. Also
        # prints rank and world size.
        val = 99 if dist.get_rank() == 0 else None
        object_list = [val] * dist.get_world_size()
        print(f"{dist.get_rank()} -> {dist.get_world_size()}")

        dist.broadcast_object_list(object_list=object_list)
        self.assertEqual(99, object_list[0])

    def test_all_reduce(self):
        # [FUNCTION DOC] all_reduce (default sum) of rank-filled tensors equals the sum of
        # ranks 0..3, i.e. 6 per element.
        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        self.assertEqual(output, torch.ones(3, 3) * res_num)

    def test_all_to_all(self):
        # [FUNCTION DOC] all_to_all where each rank sends world-size tensors filled with
        # consecutive values starting at rank times world size; checks each rank gets the
        # tensor at position rank from every sender.
        rank = self.rank
        world_size = self.world_size
        input_tensor_list = [
            torch.ones(3, 3) * x
            for x in range(rank * world_size, (rank + 1) * world_size)
        ]
        output_tensor_list = [torch.empty_like(tensor) for tensor in input_tensor_list]
        dist.all_to_all(output_tensor_list, input_tensor_list)
        expected_tensor_list = [
            torch.ones(3, 3) * x
            for x in range(rank, world_size * world_size, world_size)
        ]
        self.assertEqual(expected_tensor_list, output_tensor_list)

    def test_all_reduce_ops(self):
        # [FUNCTION DOC] all_reduce with PRODUCT, MIN, MAX, BAND, BOR and BXOR on scalar
        # tensors holding rank plus one; compares to the result of folding the matching
        # operator over 1..world size (MIN is 1, MAX is world size).
        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.PRODUCT)
        expected = reduce(operator.mul, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.MIN)
        self.assertEqual(1, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.MAX)
        self.assertEqual(self.world_size, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.BAND)
        expected = reduce(operator.and_, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.BOR)
        expected = reduce(operator.or_, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

        tensor = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(tensor, op=ReduceOp.BXOR)
        expected = reduce(operator.xor, range(1, self.world_size + 1))
        self.assertEqual(expected, tensor.item())

    def test_assert_equal_on_rank(self):
        # [FUNCTION DOC] Broadcasts rank 0's random tensor and uses assertEqualOnRank
        # (equal on rank 0) and assertNotEqualOnRank (differs on rank 1) because the RNG is
        # shared among threads.
        # RNG is shared across threads. So instead of asserting on all threads
        # we only assert on rank 0
        self_tensor = torch.rand(3, 3)
        rank_0_tensor = self_tensor.clone()
        dist.broadcast(rank_0_tensor, src=0)
        self.assertEqualOnRank(rank_0_tensor, self_tensor, rank=0)
        self.assertNotEqualOnRank(rank_0_tensor, self_tensor, rank=1)

    def test_subpg(self):
        # [FUNCTION DOC] Creates two subgroups ([0,1] and [2,3]) and all_reduces concurrently
        # within each; ranks 0 and 1 expect 1, ranks 2 and 3 expect 5.
        subpg0 = dist.new_group([0, 1])
        subpg1 = dist.new_group([2, 3])
        current_rank = dist.get_rank()
        output = torch.ones(3, 3) * current_rank

        # call all_reduce on subpg0 and subpg1 concurrently
        if current_rank in [0, 1]:
            dist.all_reduce(output, group=subpg0)
        else:
            dist.all_reduce(output, group=subpg1)

        if current_rank in [0, 1]:
            self.assertEqual(output, torch.ones(3, 3) * 1)
        else:
            self.assertEqual(output, torch.ones(3, 3) * 5)

    def test_using_pg_from_another_thread(self):
        # [FUNCTION DOC] Starts a separate Python thread that all_reduces a requires-grad
        # tensor over the WORLD group and joins it, checking that a process group can be used
        # from a different thread.
        def stuff_in_other_thread(pg):
            # [FUNCTION DOC] HELPER: Thread body: all_reduce of a random 4-element requires-
            # grad tensor on the given group.
            x = torch.rand(4, requires_grad=True)
            dist.all_reduce(x, group=pg)

        t = threading.Thread(target=stuff_in_other_thread, args=(dist.group.WORLD,))
        t.start()
        t.join()

    def test_gather(self):
        # [FUNCTION DOC] Rank 0 gathers rank-filled tensors from all ranks; on rank 0
        # gather_list entry i equals a tensor filled with i. Other ranks pass no list.
        if dist.get_rank() == 0:
            gather_list = [torch.empty(3, 3) for _ in range(self.world_size)]
        else:
            gather_list = None
        input_tensor = torch.ones(3, 3) * dist.get_rank()

        dist.gather(input_tensor, gather_list)
        if dist.get_rank() == 0:
            for i in range(self.world_size):
                self.assertEqual(gather_list[i], torch.ones(3, 3) * i)

    def test_all_reduce_coalesced(self):
        # [FUNCTION DOC] all_reduce_coalesced on two tensors (rank-filled and twice that);
        # results equal the rank sum (6) and twice it.
        t0 = torch.ones(3, 3) * dist.get_rank()
        t1 = torch.ones(3, 3) * dist.get_rank() * 2
        dist.all_reduce_coalesced([t0, t1])
        res_num = ((0 + self.world_size - 1) * self.world_size) / 2
        self.assertEqual(t0, torch.ones(3, 3) * res_num)
        self.assertEqual(t1, torch.ones(3, 3) * (res_num * 2))

    @skip_if_lt_x_gpu(1)
    def test_bwd_sees_fwd_pg(self):
        # [FUNCTION DOC] Requires at least one GPU. Applies a custom autograd Function on a
        # rank-valued tensor on device_type and runs backward, verifying backward executes in
        # the same thread as forward, with the process group still initialized and usable
        # (all_reduce giving 12).
        fwd_tid = threading.current_thread().ident

        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function that doubles the rank tensor and checks,
            # in backward, that thread identity and distributed state match the forward.
            @staticmethod
            def forward(ctx, rank):
                # [FUNCTION DOC] HELPER: Returns twice the input rank tensor, saving result
                # and rank for backward; raises AssertionError if the rank value does not
                # equal the current dist rank.
                result = rank * 2

                ctx.save_for_backward(result, rank)
                if int(rank.item()) != dist.get_rank():
                    raise AssertionError(
                        f"Expected rank.item() == dist.get_rank(), got {int(rank.item())} vs {dist.get_rank()}"
                    )
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Asserts backward runs on the forward thread, dist is
                # initialized and the rank matches; all_reduces the saved result (expected 12)
                # and returns the gradient times the result.
                result, rank = ctx.saved_tensors
                bwd_tid = threading.current_thread().ident

                self.assertEqual(
                    fwd_tid,
                    bwd_tid,
                    lambda msg: f"{msg}\nbwd not running in the same thread a fwd for rank {rank.item()}",
                )
                self.assertTrue(dist.is_initialized())
                self.assertEqual(int(rank.item()), dist.get_rank())
                dist.all_reduce(result)
                self.assertEqual(int(result.item()), 12)  # (0 + 1 + 2 + 3) * 2

                return grad_output * result

        x = torch.tensor(
            [dist.get_rank()], dtype=torch.float, device=device_type, requires_grad=True
        )
        x = MyFunc.apply(x)
        x.sum().backward()


if __name__ == "__main__":
    run_tests()
