# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the UCC process group backend: environment-based init does not
# disturb root logging, store timeout handling, basic collectives (broadcast, allreduce,
# allgather, reduce, reduce_scatter_base, send/recv, barrier) on the CPU, and
# DistributedDataParallel usage over UCC including unused-parameter handling. Many GPU or
# multi-device variants are skipped (marked as timing out or unsupported by UCC). Import side
# effect: if c10d or UCC is unavailable, prints a message to stderr and exits with status 0.

import copy
import logging
import math
import operator
import os
import random
import sys
import tempfile
from functools import reduce

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_ucc_available():
    print("c10d UCC not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common
from test_c10d_common import (
    gpus_for_rank,
    ModuleForDdpCommHook,
    SparseGradientModule,
    Task,
)

import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_ucc,
    skip_if_lt_x_gpu,
    verify_ddp_error_logged,
)
from torch.testing._internal.common_utils import (
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    TestCase,
)


def simple_reduce_tests(rank, world_size):
    # [FUNCTION DOC] HELPER: Builds (reduce op, this rank's input, expected result) triples
    # for a group of the given size: sum, product, min, max on a single float per rank; bitwise
    # AND for four iterations (each rank sets a different extra bit); bitwise OR and XOR for
    # factors one to four, where each rank pre-combines several consecutive values to emulate a
    # larger world. Returns the list of triples; no side effects.
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([world_size]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = 1 << i
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    return tests


class RendezvousEnvTest(TestCase):
    # [CLASS DOC] Single-process test case for initializing a UCC process group from
    # environment variables.
    @requires_ucc()
    @retry_on_connect_failures
    def test_logging_init(self):
        # [FUNCTION DOC] Sets world size 1, rank 0, localhost master address and a free port
        # in the process environment (side effect, never restored), initializes a ucc group via
        # the env init method, and asserts the root logger's handlers are unchanged
        # compared with before, then destroys the group.
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"

        previous_handlers = logging.root.handlers

        c10d.init_process_group(backend="ucc", init_method="env://")

        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        c10d.destroy_process_group()


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    # [CLASS DOC] Timeout tests inherited from the shared abstract test case, specialized
    # here with a UCC-backed default store timeout test.
    @requires_ucc()
    @retry_on_connect_failures
    def test_default_store_timeout_ucc(self):
        # [FUNCTION DOC] Runs the shared default-store-timeout check for the ucc backend.
        self._test_default_store_timeout("ucc")


class ProcessGroupUCCTest(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test case that builds a UCC process group over a file store
    # and exercises basic collectives and point-to-point operations. UCC limits noted in
    # the tests: single input tensor only; GPU variants are not enabled here.
    def _create_process_group_ucc(self):
        # [FUNCTION DOC] HELPER: Creates and returns a UCC process group for this rank and
        # world size using a file store on the test's temporary file.
        store = c10d.FileStore(self.file_name, self.world_size)
        return c10d.ProcessGroupUCC(store, self.rank, self.world_size)

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setup, then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs parent teardown, then removes the file-store file,
        # ignoring OS errors.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_ucc()
    def test_empty_tensors(self):
        # [FUNCTION DOC] Broadcasts a zero-element float tensor and asserts the result still
        # has zero elements and equals the input (dtype not compared).
        pg = self._create_process_group_ucc()

        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqual(xs[0], output[0], exact_dtype=False)

    # TODO: add error check testing

    def _test_broadcast_basics(self, fn):
        # [FUNCTION DOC] HELPER: With each rank in turn as root, broadcasts a one-element
        # tensor (made by applying fn to a tensor holding the rank) and asserts every rank gets
        # the root's value. Then uses the convenience overload with root zero and asserts
        # the result is one. Single-tensor input only.
        pg = self._create_process_group_ucc()

        def broadcast(xs, rootRank, rootTensor):
            # [FUNCTION DOC] HELPER: Runs a broadcast with the given root rank and root
            # tensor index, waits on its future, and returns the resulting tensors.
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0], exact_dtype=False)

            # TODO: UCC currently does not support multi tensor input

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    @requires_ucc()
    def test_broadcast_basics(self):
        # [FUNCTION DOC] Runs the broadcast checks with plain clones of CPU tensors.
        self._test_broadcast_basics(lambda t: t.clone())

    # TODO: test_broadcast_basics_cuda times out locally

    def _test_allreduce_basics(self, fn):
        # [FUNCTION DOC] HELPER: For every case from the reduce test table, allreduces the
        # input (after applying fn) with that op and compares to the expected value (dtype
        # ignored). Then checks the default overload (sum) of a one-element tensor.
        pg = self._create_process_group_ucc()

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0], exact_dtype=False)

        # TODO: UCC currently does not support multi tensor input

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

    @requires_ucc()
    def test_allreduce_basics(self):
        # [FUNCTION DOC] Runs the allreduce checks with plain clones of CPU tensors.
        self._test_allreduce_basics(lambda t: t.clone())

    # TODO: test_allreduce_basics_cuda times out locally

    def _test_allgather_basics(self, fn):
        # [FUNCTION DOC] HELPER: Each rank contributes one one-element tensor holding its
        # rank; allgather into a prefilled list of world-size tensors and asserts the output
        # lists the ranks in order. Only one input tensor per rank (the loop over counts is a
        # single iteration because UCC supports only that).
        pg = self._create_process_group_ucc()

        # TODO: Run with N input tensor per rank; for now, UCC only supports single tensor input so N=1
        for n in [1]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            expected_output = [
                [fn(torch.tensor([i])) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            self.assertEqual(expected_output, result)

    def test_allgather_basics(self):
        # [FUNCTION DOC] Runs the allgather checks with plain clones of CPU tensors.
        # KNOWN ISSUE: unlike its siblings this test has no requires_ucc decorator.
        self._test_allgather_basics(lambda t: t.clone())

    def _test_reduce_basics(self, fn):
        # [FUNCTION DOC] HELPER: For every case in the reduce test table and every root rank,
        # reduces the input (after applying fn) to that root with that op; on the root rank
        # only, asserts the result equals the expected value (dtype ignored).
        pg = self._create_process_group_ucc()
        for op, input, output in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                fut = pg.reduce([tmp], opts).get_future()
                fut.wait()
                result = fut.value()
                if root == self.rank:
                    self.assertEqual(output, result[0], exact_dtype=False)

    @requires_ucc()
    def test_reduce_basics(self):
        # [FUNCTION DOC] Runs the reduce checks with plain clones of CPU tensors.
        self._test_reduce_basics(lambda t: t.clone())

    # TODO: test_reduce_basics_cuda times out locally

    @requires_ucc()
    def test_send_recv_all_to_all(self):
        # [FUNCTION DOC] Every rank sends a one-element tensor holding its rank to every
        # other rank and receives one from each, using tag zero. Waits for all send and recv
        # works asserting completion, then asserts each received slot holds the sender's rank.
        pg = self._create_process_group_ucc()

        # Preallocate tensors for input/output
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])

    # TODO: test_barrier_implies_wait fails with numerical mismatch, will investigate later
    @skip_but_pass_in_sandcastle("fails with numerical mismatch, skip for now")
    @requires_ucc()
    def test_barrier_implies_wait(self):
        # [FUNCTION DOC] Starts sixteen allreduces on 100 by 100 tensors without keeping the
        # work handles, runs a barrier, and asserts each tensor holds its fill value times the
        # world size (barrier must complete earlier work). Skipped (passes) in sandcastle
        # because of a known numerical mismatch.
        pg = self._create_process_group_ucc()

        # Kick off allreduce operations
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # Note: leak the returned work handle
            pg.allreduce(tensor)

        # Barrier should ensure all previous work has completed
        pg.barrier().get_future().wait()

        for i, tensor in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

    @requires_ucc()
    def _test_reduce_scatter_base_basics(self, fn):
        # [FUNCTION DOC] HELPER: Each rank passes a (world, world, 10) tensor filled with its
        # rank plus one into the base reduce-scatter with a 10-element output, and asserts the
        # output equals the sum over ranks of those values.
        # KNOWN ISSUE: the requires_ucc decorator sits on this helper (not on the test that
        # calls it), and the input holds world times more data than the 10-element output
        # per rank needs.
        pg = self._create_process_group_ucc()
        n = self.world_size
        input = fn(torch.ones(n, n, 10) * (self.rank + 1.0))
        output = fn(torch.zeros(10))
        expected_output = fn(torch.ones(10) * (n + 1) * n / 2)
        fut = pg._reduce_scatter_base(output, input).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(result[0], expected_output)

    def test_reduce_scatter_base_basics(self):
        # [FUNCTION DOC] Runs the base reduce-scatter checks with plain clones of CPU tensors.
        self._test_reduce_scatter_base_basics(lambda t: t.clone())


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    # [CLASS DOC] Multi-process DDP tests over the UCC backend, reusing shared DDP test logic
    # from the common test case. Many multi-GPU variants are skipped as unsupported or
    # timing out.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setup, then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def _get_process_group(self):
        # [FUNCTION DOC] HELPER: Initializes the default ucc process group from the test
        # store for this rank and world size, and returns the default group object.
        store = self._get_store()
        c10d.init_process_group(
            "ucc", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_ucc_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        # [FUNCTION DOC] HELPER: Creates the ucc process group and delegates to the shared DDP
        # check with the given device list, device ids, multi-device flag and whether
        # gradients are views of the bucket.
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_ucc()
    def test_ucc_backend_cpu_module(self):
        # [FUNCTION DOC] DDP check with a module on CPU.
        self._test_ucc_backend([torch.device("cpu")], None)

    @requires_ucc()
    def test_ucc_backend_cpu_module_grad_is_view(self):
        # [FUNCTION DOC] DDP check with a CPU module and gradients as views of buckets.
        self._test_ucc_backend(
            [torch.device("cpu")], None, gradient_as_bucket_view=True
        )

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ucc_backend_1gpu_module_device_ids_integer_list(self):
        # [FUNCTION DOC] DDP check with a one-GPU module whose device ids are given as
        # integers; uses this rank's first assigned GPU. Needs at least two GPUs.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, int_devices)

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ucc_backend_1gpu_module_device_ids_torch_device_list(self):
        # [FUNCTION DOC] Same as the integer variant but device ids are given as torch
        # device objects. Needs at least two GPUs.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, devices)

    # TODO: test_ucc_backend_2gpu_module and test_ucc_backend_4gpu_module
    # require broadcast_coalesced which is not supported by ucc currently
    @skip_but_pass_in_sandcastle(
        "requires broadcast coalesced, which is not supported by ucc currently"
    )
    @requires_ucc()
    @skip_if_lt_x_gpu(4)
    def test_ucc_backend_2gpu_module(self):
        # [FUNCTION DOC] Multi-device DDP check with two GPUs per rank and no device ids.
        # Skipped (passes) in sandcastle since broadcast coalescing is unsupported by UCC.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, None, multi_device=True)

    @skip_but_pass_in_sandcastle(
        "requires broadcast coalesced, which is not supported by ucc currently"
    )
    @requires_ucc()
    @skip_if_lt_x_gpu(8)
    def test_ucc_backend_4gpu_module(self):
        # [FUNCTION DOC] Multi-device DDP check with four GPUs per rank and no device ids;
        # needs eight GPUs. Skipped (passes) in sandcastle since UCC lacks broadcast
        # coalescing.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_ucc_backend(devices, None, multi_device=True)

    def _test_global_local_unused_params_grad(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        # [FUNCTION DOC] HELPER: Wraps a three-task module (rank 0 uses task 0, other ranks
        # use task 1, the third is never used) in DDP with unused-parameter detection first on
        # CPU, then on this rank's first GPU, using the optional bucket-view and static-graph
        # flags. After forward all grads are unset; after backward the two used tasks have
        # grads on every rank while the globally unused task stays unset.
        """
        By simulating a multi-task training, this test is to make sure:
        1) DDP does not touch the grad of globally unused parameters.
        2) DDP does update the grad of locally unused parameters.
        """

        class GlobalLocalUnusedParamModule(nn.Module):
            # [CLASS DOC] HELPER: Module with three Task submodules t0, t1 and task_unused;
            # forward uses t0 on rank 0 and t1 otherwise, never task_unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three Task submodules.
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                # [FUNCTION DOC] HELPER: Returns the tuple of the three tasks' parameters.
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                # [FUNCTION DOC] HELPER: Applies task 0 to x if rank is 0, else task 1.
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # [FUNCTION DOC] HELPER: Runs forward on the DDP model with input 8 and this
            # rank, asserts all three parameter grads are unset, runs backward on the mean,
            # and asserts the first two have grads while the third is still unset.
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            t0_p, t1_p, task_unused_p = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            # However the globally unused parameter should still have None grad.
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

        process_group = self._get_process_group()

        # Test on CPU
        cpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(gpu_model)

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        # [FUNCTION DOC] Runs the unused-parameter check with default options. Skipped
        # (passes) in sandcastle because it times out.
        self._test_global_local_unused_params_grad()

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        # [FUNCTION DOC] Unused-parameter check with gradients as bucket views; skipped in
        # sandcastle (times out).
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_static_graph(self):
        # [FUNCTION DOC] Unused-parameter check with static graph enabled; skipped in
        # sandcastle (times out).
        self._test_global_local_unused_params_grad(static_graph=True)

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        # [FUNCTION DOC] Builds a two-task module where rank 0 uses both tasks and other
        # ranks use only the second, wraps it in DDP with unused-parameter detection on CPU and
        # then on this rank's first GPU, and verifies grads are unset after forward and set for
        # all parameters on every rank after backward. Skipped (passes) in sandcastle (times
        # out).
        """
        An empty unused_parameters array does not imply find_unused_parameters =
        false. This test makes sure that DDP allreduces unused parameters
        accordingly where the forward pass in some process uses all parameters.
        This unit test creates a module that uses all parameters in rank = 0, and
        has unused parameters in other ranks.
        """

        class FindUnusedParamModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two Task submodules; rank 0 chains t0 then t1,
            # other ranks use only t1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two Task submodules.
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()

            def task_parameters(self):
                # [FUNCTION DOC] HELPER: Returns the tuple of both tasks' parameters.
                return (self.t0.p, self.t1.p)

            def forward(self, x, rank):
                # [FUNCTION DOC] HELPER: On rank 0 applies task 0 then task 1; otherwise only
                # task 1.
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # [FUNCTION DOC] HELPER: Runs forward with input 8 and this rank, asserts all
            # parameter grads are unset, runs backward on the mean, and asserts all now have
            # grads.
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]

        process_group = self._get_process_group()

        # Test on CPU
        cpu_model = DistributedDataParallel(
            FindUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            FindUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(gpu_model)

    @requires_ucc()
    def test_ignored_output(self):
        # [FUNCTION DOC] Wraps a small two-layer model in DDP (CPU, default options), runs
        # four forward passes whose output is discarded without backward, then four iterations
        # with cross-entropy loss and backward. Passes if no error is raised.
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called.
        """
        process_group = self._get_process_group()

        class IgnoredOutput(nn.Module):
            # [CLASS DOC] HELPER: Bias-free two-layer MLP (2 to 10 to 4) with relu after each
            # layer and a softmax over classes at the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1 (2 to 10), fc2 (10 to 4), both without
                # bias, and a relu.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, relu, fc2, relu, then softmax along
                # dimension one; returns class probabilities.
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutput().float(),
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_ucc()
    def test_ignored_output_with_unused_parameters(self):
        # [FUNCTION DOC] Like the ignored-output test but the model has an extra layer never
        # used in forward and DDP is created with unused-parameter detection: four discarded
        # forward passes with no backward, then four iterations with loss and backward.
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called, if not all model
        parameters participated in computing the model output.
        """
        process_group = self._get_process_group()

        class IgnoredOutputWithUnusedParameters(nn.Module):
            # [CLASS DOC] HELPER: Same MLP as the ignored-output model plus a third linear
            # layer that forward never uses, so its parameters receive no gradient.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates bias-free fc1 (2 to 10), fc2 (10 to 4), fc3
                # (4 to 4) and a relu.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, relu, fc2, relu, then softmax along
                # dimension one, skipping fc3.
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutputWithUnusedParameters().float(),
            process_group=process_group,
            find_unused_parameters=True,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        # [FUNCTION DOC] HELPER: Runs cross-entropy backward of the plain model on the full
        # random batch and of the DDP model on this rank's slice of that batch, then asserts
        # the first parameter's coalesced sparse gradients are equal.
        mult = 2
        batch_size = mult * self.world_size
        criterion = nn.CrossEntropyLoss()
        input = torch.randint(0, 10, [batch_size, 2])
        target = torch.randint(0, 10, [batch_size])

        # Run with entire batch against single process version
        criterion(vanilla_model(input), target).backward()

        # Run with partial batch against multi process version
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()

        # Check that the gradients are sparse and identical
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(
            vanilla_parameter.grad.coalesce(), ddp_parameter.grad.coalesce()
        )

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_save_load_checkpoint(self):
        # [FUNCTION DOC] Initializes a ucc group from a file URL and trains three models on
        # GPU from zeroed weights: a DDP model and a plain model checkpointed after three SGD
        # steps (rank 0 saves the DDP state dict to a file in the temp directory, all ranks
        # load it with device remapping, the plain model after stripping the module prefix),
        # and a DDP model trained for six steps straight. Asserts all parameters then match.
        # Needs two GPUs.
        # KNOWN ISSUE: the checkpoint file path is fixed in the temp directory and never
        # removed.
        dist.init_process_group(
            "ucc",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        class TestModel(nn.Module):
            # [CLASS DOC] HELPER: Bias-free two-layer MLP (2 to 10 to 4) with relu after each
            # layer and a softmax over classes at the output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1 (2 to 10), fc2 (10 to 4), both without
                # bias, and a relu.
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, relu, fc2, relu, then softmax along
                # dimension one; returns class probabilities.
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        def train_loop(model, optimizer, iterations):
            # [FUNCTION DOC] HELPER: Runs the given number of steps: zero grads, forward on the
            # enclosing test's input, cross-entropy loss against its target, backward, optimizer
            # step. Relies on the enclosing scope's input, criterion and target variables.
            for _ in range(iterations):
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        model_withload = TestModel().float().to(device_id)
        model_withoutload = TestModel().float().to(device_id)

        ddp_withload = DistributedDataParallel(
            model_withload,
            device_ids=[device_id],
        )
        ddp_withoutload = DistributedDataParallel(
            model_withoutload,
            device_ids=[device_id],
        )

        # ensure that all the three models start with the same set of parameters. By default they are randomized on construction
        for p in ddp_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in model_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in ddp_withoutload.parameters():
            with torch.no_grad():
                p.zero_()

        batch_size = 4
        criterion = nn.CrossEntropyLoss()

        optimizer_withload = torch.optim.SGD(ddp_withload.parameters(), lr=0.001)
        optimizer_non_ddp_withload = torch.optim.SGD(
            model_withload.parameters(), lr=0.001
        )
        optimizer_withoutload = torch.optim.SGD(ddp_withoutload.parameters(), lr=0.001)

        input = torch.rand([batch_size, 2], dtype=torch.float).to(device_id)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # run the model for 6 iterations, with a checkpoint in the middle
        train_loop(ddp_withload, optimizer_withload, 3)

        # zero out parameters of both DDP and non-DDP models and reload them from the DDP state dict
        checkpoint_path = tempfile.gettempdir() + "/model.checkpoint"
        if self.rank == 0:
            torch.save(ddp_withload.state_dict(), checkpoint_path)

        dist.barrier()
        map_location = {"cuda:0": f"cuda:{self.rank:d}"}
        ddp_state_dict = torch.load(checkpoint_path, map_location=map_location)

        for model in [ddp_withload, model_withload]:
            for p in model.parameters():
                with torch.no_grad():
                    p.zero_()
        ddp_withload.load_state_dict(ddp_state_dict)
        # the non-DDP model needs to first remove the prefix of "module." from the DDP state dict
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            ddp_state_dict, "module."
        )
        model_withload.load_state_dict(ddp_state_dict)

        train_loop(ddp_withload, optimizer_withload, 3)
        train_loop(model_withload, optimizer_non_ddp_withload, 3)

        # re-run the model with the same inputs for 6 iterations with no checkpoint
        train_loop(ddp_withoutload, optimizer_withoutload, 6)

        for p_withload, p_withoutload, p_non_ddp_withload in zip(
            ddp_withload.parameters(),
            ddp_withoutload.parameters(),
            model_withload.parameters(),
        ):
            self.assertEqual(p_withload, p_withoutload)
            self.assertEqual(p_non_ddp_withload, p_withoutload)

    def _test_sparse_gradients(self, gradient_as_bucket_view=False):
        # [FUNCTION DOC] HELPER: Seeds the RNG identically on all ranks, wraps a copy of the
        # sparse-gradient test model in DDP (optional bucket-view gradients), and verifies
        # sparse gradients against the single-process model.
        process_group = self._get_process_group()

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    # TODO: backward pass: input tensor has to be dense
    @skip_but_pass_in_sandcastle("backward pass: input tensor has to be dense")
    @requires_ucc()
    def test_sparse_gradients(self):
        # [FUNCTION DOC] Sparse gradient check with defaults; skipped (passes) in sandcastle
        # because the backward pass needs dense input.
        self._test_sparse_gradients()

    # TODO: backward pass: input tensor has to be dense
    @skip_but_pass_in_sandcastle("backward pass: input tensor has to be dense")
    @requires_ucc()
    def test_sparse_gradients_grad_is_view(self):
        # [FUNCTION DOC] Sparse gradient check with gradients as bucket views; skipped in
        # sandcastle (backward needs dense input).
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_ucc()
    def test_ddp_comm_hook_future_passing_cpu(self):
        # [FUNCTION DOC] Wraps the hook test module on CPU in DDP, registers the shared simple
        # hook (from the common base class), and uses the shared helper to check that the
        # resulting gradients equal the hook's constant value of twice ones rather than the
        # default reduction.
        """
        This unit test verifies whether the Future object is passed properly.
        The callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Test on CPU
        cpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().cpu(), process_group=process_group
        )

        # Register DDP Communication Hook
        cpu_model.register_comm_hook(None, self._simple_hook)

        # check whether the grads are equal to what then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        # [FUNCTION DOC] HELPER: Puts the hook test module on this rank's first GPU, wraps it
        # in DDP with the given group and bucket-view flag, registers the hook with its state
        # if a hook is given, and returns the DDP model.
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_ucc(self):
        # [FUNCTION DOC] GPU version of the future-passing hook test: builds a GPU DDP model
        # with the shared simple hook and checks gradients equal twice ones. Needs two GPUs.
        """
        This unit test verifies whether the Future object is passed properly using ucc backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_ucc()
    def test_ddp_invalid_comm_hook_init(self):
        # [FUNCTION DOC] Checks register_comm_hook validation: a non-callable hook raises a
        # type error saying the hook must be callable, and a hook whose bucket parameter is
        # annotated with a wrong type raises a value error about the bucket annotation.
        """
        This unit test makes sure that register_comm_hook properly checks the format
        of hook defined by user. The Python hook must be callable. This test also
        checks whether bucket annotation checked properly if defined.
        """
        process_group = self._get_process_group()

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        with self.assertRaisesRegex(TypeError, "Communication hook must be callable."):
            model.register_comm_hook(state=None, hook=1)

        with self.assertRaisesRegex(
            ValueError, "bucket annotation should be dist.GradBucket."
        ):

            def comm_hook(
                state: object, bucket: int
            ) -> torch.futures.Future[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Invalid hook (bucket annotated as an integer) that
                # returns an empty future; only used to trigger registration validation.
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_ucc()
    def test_ddp_invalid_comm_hook_return_type(self):
        # [FUNCTION DOC] Checks that registering a hook annotated to return an integer raises
        # a value error about the return annotation (and that the error is logged on the
        # model), and that an unannotated hook returning a non-future is accepted at
        # registration. The final block also runs forward (arguments 8 and the rank) and
        # backward inside the assertRaises context after registering the hook.
        # KNOWN ISSUE: forward and backward sit inside the expected-error block; the expected
        # runtime error about the callback result can only come from them, but the model used
        # takes different inputs in other tests, so it is unclear the intended path runs.
        """
        This test checks whether return annotation checked properly if defined. It also
        checks whether an internal error is thrown if return type is incorrect and user
        hasn't specified any return type annotation.
        """
        process_group = self._get_process_group()

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        expected_err = (
            "Communication hook: return annotation should be torch.futures.Future"
        )
        with self.assertRaisesRegex(
            ValueError,
            expected_err,
        ):

            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                # [FUNCTION DOC] HELPER: Invalid hook whose return annotation is an integer
                # instead of a future; only used to trigger registration validation.
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

        verify_ddp_error_logged(model, expected_err)

        with self.assertRaisesRegex(
            RuntimeError,
            "callback must return a torch.futures.Future object, but got",
        ):

            def comm_hook(state: object, bucket: dist.GradBucket):
                # [FUNCTION DOC] HELPER: Unannotated hook that returns the integer one
                # instead of a future (an invalid result).
                return 1

            model.register_comm_hook(state=None, hook=comm_hook)

            # Run forward
            output = model(8, self.rank)

            # Run backward
            output.mean().backward()

    @requires_ucc()
    def test_ddp_comm_hook_register_just_once(self):
        # [FUNCTION DOC] Registers a trivial hook once successfully, then registers again
        # and asserts a runtime error saying a hook can only be registered once.
        """
        DDP communication hook can only be registered once. This test validates whether
        the error is thrown properly when register_comm_hook is called more than once.
        """
        process_group = self._get_process_group()

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        def dummy_hook(state, bucket):
            # [FUNCTION DOC] HELPER: Hook returning a future already resolved with the
            # bucket's buffer wrapped in a list (no communication).
            fut = torch.futures.Future()
            fut.set_result([bucket.buffer()])
            return fut

        model.register_comm_hook(None, dummy_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "register_comm_hook or register_builtin_comm_hook can only be called once.",
        ):
            model.register_comm_hook(None, dummy_hook)

    # TODO: backward pass: input tensor must be dense
    @skip_but_pass_in_sandcastle("backward pass: input tensor has to be dense")
    @requires_ucc()
    def test_ddp_comm_hook_sparse_gradients(self):
        # [FUNCTION DOC] Repeats the sparse gradient check with a registered hook that
        # allreduces the bucket buffer through the process group and divides the result by the
        # world size. Skipped (passes) in sandcastle (backward needs dense input).
        """
        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a
        simple hook that does allreduce and works with ucc backend for this test.
        """
        process_group = self._get_process_group()

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
        )

        def allreduce_hook_ucc(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            # [FUNCTION DOC] HELPER: DDP hook: starts an asynchronous allreduce of the bucket
            # buffer on the process group and returns a future chained with a division by the
            # world size.
            def div_by_world_size(fut):
                # [FUNCTION DOC] HELPER: Waits on the allreduce future and returns the first
                # result tensor divided by the world size.
                # Divide the result by 2 * world_size.
                return fut.wait()[0] / self.world_size

            # Prepare allreduced grad bucket tensors by running an async work.
            fut = process_group.allreduce([bucket.buffer()]).get_future()
            return fut.then(div_by_world_size)

        ddp_model.register_comm_hook(None, allreduce_hook_ucc)

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    # [CLASS DOC] Multi-process communication tests from the shared abstract case, run with
    # the ucc backend (sequence numbers, group membership, dtype checks, barrier arguments).
    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property naming the cpu device.
        return "cpu"

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setup, then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs parent teardown, then removes the file-store file,
        # ignoring OS errors.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_ucc(self):
        # [FUNCTION DOC] Runs the shared default-group sequence number check for ucc.
        self._test_sequence_num_set_default_pg(backend="ucc")

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_ucc_new_group(self):
        # [FUNCTION DOC] Runs the shared new-group sequence number check for ucc.
        self._test_sequence_num_set_new_group(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_sequence_num_incremented_ucc_default(self):
        # [FUNCTION DOC] Runs the shared check that the default group's sequence number
        # increments with collectives, for ucc.
        self._test_sequence_num_incremented_default_group("ucc")

    @skip_if_lt_x_gpu(4)
    @requires_ucc()
    def test_sequence_num_incremented_ucc_subgroup(self):
        # [FUNCTION DOC] Runs the shared subgroup sequence number increment check for ucc
        # (needs four GPUs). If the world size is below four, returns the result of the
        # sandcastle skip helper without running the check.
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("ucc")

    @skip_but_pass_in_sandcastle("Fails on M60")
    @requires_ucc()
    def test_ucc_barrier_device_ids(self):
        # [FUNCTION DOC] Initializes a ucc group over a file store and asserts that calling
        # barrier with device ids raises a runtime error saying device ids are not supported.
        # Skipped (passes) in sandcastle (fails on M60 GPUs).
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="ucc", rank=self.rank, world_size=self.world_size, store=store
        )

        with self.assertRaisesRegex(RuntimeError, "device_ids not supported"):
            c10d.barrier(device_ids=[self.rank])

    @skip_but_pass_in_sandcastle("Fails on M60")
    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_ucc_warn_not_in_group(self):
        # [FUNCTION DOC] Runs the shared warn-when-not-in-group check for ucc; skipped in
        # sandcastle (fails on M60).
        self._test_warn_not_in_group(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_ucc_rank_membership(self):
        # [FUNCTION DOC] Runs the shared rank membership check for ucc.
        self._test_rank_membership(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_tensor_dtype_mismatch(self):
        # [FUNCTION DOC] Runs the shared tensor dtype mismatch check for ucc.
        self._test_tensor_dtype_mismatch(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_tensor_dtype_complex(self):
        # [FUNCTION DOC] Runs the shared complex tensor dtype check for ucc.
        self._test_tensor_dtype_complex(backend="ucc")


class UccProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    # [CLASS DOC] Dispatched-collective tests from the shared base class, run with the ucc
    # backend; both tests are skipped (pass) in sandcastle and need at least one GPU.
    @skip_but_pass_in_sandcastle("Fails on M60")
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        # [FUNCTION DOC] Runs the shared dispatched-collectives check (reduce, broadcast,
        # all_reduce, all_gather, reduce_scatter, barrier, all_to_all, scatter) with ucc.
        # includes reduce, broadcast, all_reduce, all_gather, reduce_scatter, barrier, all_to_all, scatter
        self._test_collectives(backend="ucc")

    @skip_but_pass_in_sandcastle("Fails on M60")
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        # [FUNCTION DOC] Initializes a ucc group over a file store, runs all_gather_single of
        # a 10 by 10 ones tensor on cuda into a zero output tensor, and asserts the output
        # equals the input.
        # KNOWN ISSUE: the output tensor has the same shape as the input, so it only looks
        # sized for a world of one; a gather over several ranks would normally need a larger
        # output.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "ucc",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_single(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)


if __name__ == "__main__":
    if torch.cuda._initialized:
        raise AssertionError(
            "test_distributed must not have initialized CUDA context on main process"
        )

    run_tests()
