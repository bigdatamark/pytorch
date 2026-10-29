# Owner(s): ["oncall: distributed"]
# This test file contains positive tests for c10d with NCCL backend.
# During the test, it is expected that ProcessGroup will not be aborted, destroyed or incur fatal error.
# Please be mindful of this when adding tests here.
# If you need to add tests for group creation, abort or destroy, please add tests in test_c10d_nccl.py.

# There are two ways to launch tests in this file:
# 1. Run this file directly with `python test_c10d_ops_nccl.py`
# 2. Use multi-process launcher, e.g. `torchrun --standalone --nproc-per-node 2 test_c10d_ops_nccl.py`

# [MODULE DOC] Positive (no abort or destroy) tests of c10d collectives and point-to-point
# operations on the legacy NCCL process group, all in one class run by the continuous multi-
# process harness, which builds the default group once and runs each test on every rank.
# Covers empty tensors, broadcast, reduce, all_reduce (including sparse, float8, average,
# premul sum and unsupported bitwise ops), all_gather variants, gather, scatter,
# reduce_scatter variants, barrier, send/recv, CUDA graph capture and watchdog behavior, and
# premul sum reduce ops. Import-time side effects: exits the process with status 0 if c10d
# or NCCL is unavailable, or under dev-asan.
import math
import os
import sys
import unittest

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


import torch.distributed as dist
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcContinuousTest,
    requires_nccl,
    requires_nccl_version,
)
from torch.testing._internal.common_utils import (
    IS_LINUX,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)


class ProcessGroupNCCLOpTest(MultiProcContinuousTest):
    # [CLASS DOC] All NCCL op tests, using the default group of the harness (self.pg, backend
    # nccl, world size from the harness). Tests find their GPUs through rank_to_GPU and are
    # skipped (but reported as passing in sandcastle) when fewer than two GPUs exist. Most tests
    # exercise the raw group API (work objects with options structures) rather than the functional
    # dist API.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by the
        # harness; returns nccl.
        return "nccl"

    @classmethod
    def opts(cls, high_priority_stream=False):
        # [FUNCTION DOC] HELPER: Class method building stock NCCL options whose high
        # priority stream flag is set from the argument (default off); returned to the
        # harness for group creation.
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    @property
    def rank_to_GPU(self):
        # [FUNCTION DOC] HELPER: Property returning the mapping from each rank to its list of GPU
        # indices, computed by the common multi-GPU helper for this world size.
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "nccl")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_empty_tensors(self):
        # [FUNCTION DOC] Runs broadcast, allreduce and reduce on a one-element list holding an
        # empty float tensor on this rank's first GPU, then allgather and reduce_scatter with
        # lists of empty tensors, waiting each work. Asserts every tensor remains empty and no
        # call fails.
        pg = self.pg
        local_device_idx = self.rank_to_GPU[self.rank][0]

        xs = [torch.FloatTensor([]).cuda(local_device_idx)]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        ys = [
            [
                torch.FloatTensor([]).cuda(local_device_idx)
                for _ in range(self.world_size)
            ]
        ]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())

        ys = [torch.FloatTensor([]).cuda(local_device_idx)]
        xs = [
            [
                torch.FloatTensor([]).cuda(local_device_idx)
                for _ in range(self.world_size)
            ]
        ]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_broadcast_ops(self):
        # [FUNCTION DOC] For each rank as root, broadcasts a one-element tensor (checks every rank
        # ends with the root's rank value), then tests multiple tensors per rank (one per local
        # GPU): the root's tensors are replaced with a square tensor of size root plus one filled
        # with root plus one, each local tensor index is tried as root tensor, and all tensors on
        # all ranks must equal the expected one.
        pg = self.pg

        def broadcast(xs, rootRank, rootTensor):
            # [FUNCTION DOC] HELPER: Broadcasts the tensor list through the group with the given
            # root rank and root tensor index in the options, waits for completion, and returns
            # the same list.
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()
            return xs

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = torch.tensor([self.rank]).cuda(self.rank_to_GPU[self.rank][0])
            output = broadcast([x], i, 0)
            self.assertEqual(torch.tensor([i]), output[0])

            expected_tensor = torch.empty([i + 1, i + 1]).fill_(i + 1)
            xs = [
                torch.empty([i + 1, i + 1]).fill_(-1).cuda(device=device_idx)
                for device_idx in self.rank_to_GPU[self.rank]
            ]

            # test with multiple input tensors (multiple gpu in one rank)
            for j in range(len(xs)):
                if self.rank == i:
                    xs[j] = expected_tensor.cuda(device=self.rank_to_GPU[self.rank][j])

                broadcast(xs, i, j)

                for tensor in xs:
                    self.assertEqual(tensor, expected_tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_sparse_allreduce_ops(self):
        # [FUNCTION DOC] Builds a 2 by 3 sparse COO tensor on the rank's device and all-reduces
        # it; expects the result as a dense tensor equal to twice the input (the expected values
        # assume two ranks). A RuntimeError saying NCCL does not support sparse all_reduce is
        # swallowed so the test passes on builds without it; any other error is re-raised.
        pg = self.pg

        indices = torch.tensor([[0, 1]])
        values = torch.tensor([[1, 2, 0], [4, 0, 6]])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(2, 3)).to(
            self.rank
        )

        # sparse allreduce call is wrapped in a try catch since the c10d API is only available in the nccl experimental branch
        try:
            tensor_list = [sparse_tensor]
            work = pg.allreduce(tensor_list)
            work.wait()

            # tensor_list is a list of size 1, with the allreduce output as a dense tensor
            a = torch.tensor([[2, 4, 0], [8, 0, 12]]).to(self.rank)
            self.assertEqual(tensor_list[0], a)
        except RuntimeError as e:
            if "NCCL does not support all_reduce with sparse tensors" in str(e):
                pass
            else:
                # Rethrow the exception if it's a different error
                raise

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allreduce_ops(self):
        # [FUNCTION DOC] Checks all_reduce for SUM (n(n+1)/2 over the world size n), AVG when NCCL
        # is 2.10 or newer, premul sum with scalar and tensor factors for half, float, double and
        # bfloat16 when NCCL is 2.11.1 or newer, PRODUCT (factorial of the world size), MIN (1)
        # and MAX (world size), and asserts the bitwise ops BAND, BOR and BXOR raise ValueError
        # saying they cannot be used with NCCL.
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allreduce(tensors, op):
            # [FUNCTION DOC] HELPER: Runs the group allreduce on the tensor list with the given
            # reduce op in the options and waits for completion; results are in place.
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.SUM)

        ndev = self.world_size
        self.assertEqual(
            torch.tensor([ndev * (ndev + 1) // 2]),
            tensors[0],
        )

        # Avg (only available for NCCL 2.10+)
        if torch.cuda.nccl.version() >= (2, 10, 0):
            tensors = [torch.tensor([self.rank + 1.0]).cuda(local_device_id)]

            allreduce(tensors, c10d.ReduceOp.AVG)
            ndev = self.world_size
            self.assertEqual(
                torch.tensor([ndev * (ndev + 1.0) / (2.0 * ndev)]),
                tensors[0],
            )

        # Premul Sum
        if torch.cuda.nccl.version() >= (2, 11, 1):
            for dtype in torch.half, torch.float, torch.double, torch.bfloat16:
                for factor in (
                    3.0,
                    torch.tensor([5.0], device=local_device_id, dtype=dtype),
                ):
                    tensors = [
                        torch.tensor([self.rank + 1])
                        .cuda(local_device_id)
                        .to(dtype=dtype)
                    ]

                    allreduce(tensors, c10d._make_nccl_premul_sum(factor))

                    self.assertEqual(
                        factor
                        * torch.tensor(
                            [self.world_size * (self.world_size + 1) / 2],
                            dtype=dtype,
                            device=local_device_id,
                        ),
                        tensors[0],
                    )

        # Product
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.PRODUCT)
        self.assertEqual(torch.tensor([math.factorial(self.world_size)]), tensors[0])

        # Min
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.MIN)
        self.assertEqual(torch.tensor([1]), tensors[0])

        # Max
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]

        allreduce(tensors, c10d.ReduceOp.MAX)
        self.assertEqual(torch.tensor([self.world_size]), tensors[0])

        for op, err in zip(
            (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR),
            ("ReduceOp.BAND", "ReduceOp.BOR", "ReduceOp.BXOR"),
        ):
            with self.assertRaisesRegex(ValueError, "Cannot use " + err + " with NCCL"):
                allreduce(tensors, op)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/157879")
    @requires_nccl_version((2, 24), "Need NCCL 2.24+ for Float8")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skip_but_pass_in_sandcastle_if(
        not PLATFORM_SUPPORTS_FP8, "Float8 requires sm >= 90"
    )
    def test_allreduce_float8(self):
        # [FUNCTION DOC] Skipped when running on Linux (the decorator references an issue link),
        # when NCCL is older than 2.24, with fewer than two GPUs, or when the platform has no
        # float8 support. All-reduces 1024 ones of type float8 e4m3fn through the functional API
        # and asserts the result equals the world size.
        device = torch.device("cuda", self.rank_to_GPU[self.rank][0])

        numel = 1024
        tensor = torch.ones(numel, dtype=torch.float32, device=device).to(
            torch.float8_e4m3fn
        )
        dist.all_reduce(tensor)

        expected = (
            torch.empty_like(tensor).fill_(self.world_size).to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(tensor, expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_alltoall_ops_with_cudafree_race(self):
        # [FUNCTION DOC] Sets the current device, creates many temporary tensors, then ten times
        # starts an all_to_all_single of 1000 by 1000 float tensors, frees cached CUDA memory
        # (which triggers cudaFree) before waiting, and finally synchronizes. Only checks that
        # nothing crashes or hangs; output values are not verified.
        pg = self.pg
        opts = c10d.AllToAllOptions()
        local_device = f"cuda:{self.rank_to_GPU[self.rank][0]}"
        torch.cuda.set_device(local_device)
        input = torch.rand(1000, 1000, device=local_device)
        output = torch.rand(1000, 1000, device=local_device)
        race_tensors = []
        # create some tensors to race with alltoall collective
        for _ in range(10):
            tmp = []
            for i in range(5):
                tmp.append(torch.rand(10 ** (3 + i), device=local_device))
            race_tensors.append(tmp)

        for _ in range(10):
            race_tensors.pop()
            work = pg.all_to_all_single(output, input, [], [], opts)
            # this triggers cudaFree
            torch.cuda.empty_cache()
            work.wait()
        torch.cuda.synchronize(device=local_device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allreduce_in_cudagraph(self):
        # [FUNCTION DOC] Warms up with one all_reduce of a one-element tensor (value must
        # equal the world size). Twice captures a graph containing an all_reduce and a
        # broadcast from rank 0; capture must not change the value; replays the graph twice
        # and expects the value to be multiplied by the world size on each replay, which is
        # verified after the second replay.
        local_device_idx = self.rank_to_GPU[self.rank][0]
        # This device setting is needed by the CUDAGraph API to understand on
        # which device to find the current stream
        torch.cuda.set_device(local_device_idx)
        xs = torch.FloatTensor([1]).cuda(local_device_idx)

        # single warmup
        c10d.all_reduce(xs, group=self.pg)
        # 1 + 1 + ...  = world_size
        expected_val = self.world_size
        self.assertEqual(xs.item(), expected_val)

        # Use a loop to test re-capture
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                c10d.all_reduce(xs, group=self.pg)
                c10d.broadcast(xs, src=0, group=self.pg)
            # Graph capture should not change the tensor value
            self.assertEqual(xs.item(), expected_val)

            graph.replay()
            expected_val *= self.world_size
            graph.replay()
            expected_val *= self.world_size
            self.assertEqual(xs.item(), expected_val)

    @unittest.skipIf(TEST_WITH_ROCM, "https://github.com/pytorch/pytorch/issues/157896")
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_nccl_watchdog_cudagraph(self):
        # [FUNCTION DOC] Skipped on ROCm. Ten times: runs 30 eager all_reduce calls on a one-
        # element tensor, captures a graph with in-place adds and three all_reduce calls, and
        # replays it 100 times. Passes if the watchdog does not crash when event queries are
        # disallowed during capture. No values are checked.
        # test that the watchdog does not crash graphs with disallowed event query
        pg = self.pg
        rank = self.rank_to_GPU[self.rank][0]
        with torch.cuda.device(rank):
            for _ in range(10):
                xs = [torch.FloatTensor([1]).cuda(rank)]
                for _ in range(30):
                    pg.allreduce(xs[0]).wait()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    xs[0] += 0.0
                    pg.allreduce(xs[0]).wait()
                    pg.allreduce(xs[0]).wait()
                    pg.allreduce(xs[0]).wait()
                    xs[0] += 0.0

                for _ in range(100):
                    graph.replay()

    @requires_nccl()
    @requires_nccl_version(
        (2, 29), "Need NCCL 2.29+ for multisegment memory in CUDA graph"
    )
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_nccl_cudagraph_multisegment(self):
        # [FUNCTION DOC] Needs NCCL 2.29 or newer. Turns on expandable segments, runs three eager
        # all_gather_single calls of a large tensor filled with rank plus one and checks the
        # output sum, then captures one all_gather_single into a CUDA graph, replays it and checks
        # the sum again. Verifies multisegment memory works in graphs.
        # KNOWN ISSUE: the allocator setting is turned back to false at the end of the test only
        # on success (no cleanup block), and the original value is not restored.
        # Prior to NCCL 2.29, this would cause an Invalid Memory Access (IMA)
        # because NCCL didn't properly handle multisegment memory in graphs.
        local_device_idx = self.rank_to_GPU[self.rank][0]
        torch.cuda.set_device(local_device_idx)
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

        b, t, d = 64, 1, 101024
        inp = torch.ones((b, t, d), device="cuda") * (self.rank + 1)

        for _ in range(3):
            output = inp.new_empty((self.world_size, b, t, d))
            c10d.all_gather_single(output, inp, group=self.pg)

        expected_sum = inp.numel() * sum(range(1, self.world_size + 1))
        self.assertEqual(output.sum().item(), expected_sum)

        static_inp = inp.clone()
        static_output = inp.new_empty((self.world_size, b, t, d))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            c10d.all_gather_single(static_output, static_inp, group=self.pg)

        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(static_output.sum().item(), expected_sum)
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_ops(self):
        # [FUNCTION DOC] For each root rank: reduces tensors holding rank plus one to that root
        # and checks the root has n(n+1)/2 while others keep their input; asserts BAND, BOR and
        # BXOR raise ValueError saying they cannot be used with NCCL; and, when NCCL is 2.11.1 or
        # newer, compares a premul sum reduce (scalar and tensor factors) against a plain sum of
        # inputs pre-multiplied by the factor, checking at the root.
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce(xs, rootRank, rootTensor, op=None):
            # [FUNCTION DOC] HELPER: Runs the group reduce on the tensor list with the given root
            # rank and root tensor index, setting the reduce op in the options only when one is
            # provided, and waits for completion.
            opts = c10d.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            if op:
                opts.reduceOp = op
            work = pg.reduce(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.world_size):
            tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]

            reduce(tensors, rt, 0)

            if self.rank == rt:
                self.assertEqual(
                    torch.tensor([self.world_size * (self.world_size + 1) // 2]),
                    tensors[0],
                )
            else:
                self.assertEqual(
                    torch.tensor([self.rank + 1]),
                    tensors[0],
                )

            for op, err in zip(
                (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR),
                ("ReduceOp.BAND", "ReduceOp.BOR", "ReduceOp.BXOR"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "Cannot use " + err + " with NCCL"
                ):
                    reduce(tensors, self.rank, rt, op)

            # Premul sum
            if torch.cuda.nccl.version() >= (2, 11, 1):
                for factor in (3.0, torch.tensor([5.0], device=local_device_id)):
                    if isinstance(factor, torch.Tensor):
                        factor_ref = factor.cpu().item()
                    else:
                        factor_ref = factor
                    float_tensors = [
                        torch.tensor(
                            [self.rank + 1.0], device=f"cuda:{local_device_id}"
                        )
                    ]
                    float_tensors_ref = [
                        torch.tensor(
                            [(self.rank + 1.0) * factor_ref],
                            device=f"cuda:{local_device_id}",
                        )
                    ]

                    reduce(float_tensors_ref, rt, 0)
                    reduce(float_tensors, rt, 0, c10d._make_nccl_premul_sum(factor))
                    if self.rank == rt:
                        self.assertEqual(float_tensors_ref[0], float_tensors[0])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allgather_ops(self):
        # [FUNCTION DOC] Builds one 2 by 2 tensor filled with 2 per local GPU and output lists
        # filled with -1 (one entry per GPU per rank), runs the list-style allgather, and asserts
        # every output entry equals the 2-filled expected tensor.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]

        def allgather(output_ts, input_ts):
            # [FUNCTION DOC] HELPER: Runs the group allgather with the given output lists and
            # input tensors, and returns the result of waiting for the work.
            work = pg.allgather(output_ts, input_ts)
            return work.wait()

        tensors = [torch.empty(2, 2).fill_(2).cuda(device=i) for i in local_device_ids]
        output_tensors = []
        expected_output = []

        output_per_gpu = (
            [torch.empty(2, 2).fill_(-1)] * len(local_device_ids) * self.world_size
        )
        expected_per_gpu = (
            [torch.empty(2, 2).fill_(2)] * len(local_device_ids) * self.world_size
        )

        for gpu in local_device_ids:
            output_tensors.append([t.cuda(device=gpu) for t in output_per_gpu])
            expected_output.append([t.cuda(device=gpu) for t in expected_per_gpu])

        allgather(output_tensors, tensors)

        # Verification
        self.assertEqual(output_tensors, expected_output)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allgather_base_ops(self):
        # [FUNCTION DOC] Each rank contributes a one-element tensor holding its rank;
        # all_gather_single into a world-size tensor must equal the sequence of ranks in order.
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            # [FUNCTION DOC] HELPER: Runs the group all_gather_single on the output and input
            # tensors and waits for completion.
            work = pg.all_gather_single(output_t, input_t)
            work.wait()

        # allgather_base is GPU number agnostic.
        # Each rank contribute one tensor regardless of GPU counts
        tensor = torch.tensor([self.rank]).cuda(local_device_id)
        output_t = torch.empty((self.world_size), dtype=tensor.dtype).cuda(
            local_device_id
        )

        allgather_base(output_t, tensor)

        # Verification
        self.assertEqual(torch.arange(self.world_size), output_t)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allgather_base_basics(self):
        # [FUNCTION DOC] Asserts all_gather_single raises ValueError when the output size is not
        # world size times the input size, and TypeError when the output dtype differs from the
        # input dtype (that case also has a mis-sized output, so the type check must be reported).
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            # [FUNCTION DOC] HELPER: Runs the group all_gather_single on the output and input
            # tensors and waits for completion.
            work = pg.all_gather_single(output_t, input_t)
            work.wait()

        # anticipate an error
        with self.assertRaisesRegex(
            ValueError,
            "output tensor size must be equal to world_size times input tensor size",
        ):
            tensor = torch.tensor([self.rank]).cuda(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=tensor.dtype).cuda(
                local_device_id
            )
            # fails the check because output_t is not correctly sized
            allgather_base(output_t, tensor)

        # anticipate an error
        with self.assertRaisesRegex(
            TypeError, "output tensor must have the same type as input tensor"
        ):
            tensor = torch.tensor([self.rank], dtype=torch.float).cuda(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=torch.long).cuda(
                local_device_id
            )
            # fails the check because the dtype is different
            allgather_base(output_t, tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_gather_ops(self):
        # [FUNCTION DOC] For each root rank, gathers one-element tensors holding the sender's rank
        # into per-GPU output lists (filled with -1) at the root, and the root asserts the lists
        # equal the ranks in order.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            # [FUNCTION DOC] HELPER: Sets the root rank in the options; the root calls the group
            # gather with its output lists and the input tensors, other ranks pass an empty output
            # list; then waits for the work.
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()

        # init input
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([self.rank]).cuda(device_id))

        # init output
        output_ts = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            output_ts.append([])
            for rank in range(self.world_size):
                output_ts[idx].append(torch.tensor([-1]).cuda(gpu_idx))

        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for rank in range(self.world_size):
            gather(output_ts, tensors, rank)
            if rank == self.rank:
                self.assertEqual(expected, output_ts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_gather_stress(self):
        # [FUNCTION DOC] Repeats the gather test across 1000 prepared sets of input and output
        # tensors for each root rank, with the root verifying the output after each call.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def gather(output_t, input_t, rootRank):
            # [FUNCTION DOC] HELPER: Sets the root rank in the options; the root calls the group
            # gather with its output lists and the input tensors, other ranks pass an empty output
            # list; then waits for the work.
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                work = pg.gather([], input_t, opts)
            work.wait()

        stress_length = 1000

        # init input
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([self.rank]).cuda(device_id))

        # init output
        output_ts = []
        for i in range(stress_length):
            output_ts.append([[] for _ in range(num_gpus)])
            for idx, ls in enumerate(output_ts[i]):
                gpu_idx = local_device_ids[idx]
                for _ in range(self.world_size):
                    ls.append(torch.tensor([-1]).cuda(gpu_idx))

        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for i in range(stress_length):
            for rank in range(self.world_size):
                gather(output_ts[i], tensors[i], rank)
                # Verification
                if rank == self.rank:
                    self.assertEqual(output_ts[i], expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_gather_checks(self):
        # [FUNCTION DOC] Asserts argument validation of the group gather: a root rank of -1 or
        # equal to the world size raises ValueError (invalid root rank), passing an integer
        # instead of an options object raises TypeError (incompatible function arguments), and an
        # empty input tensor list raises RuntimeError from the dispatcher (no tensor arguments).
        pg = self.pg
        device_id = self.rank_to_GPU[self.rank][0]

        # init input
        tensor = torch.tensor([self.rank]).cuda(device_id)

        # init output
        output_ts = []
        for _ in range(self.world_size):
            output_ts.append(torch.tensor([-1]).cuda(device_id))

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([output_ts], [tensor], opts)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            pg.gather([output_ts], [tensor], 0)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([output_ts], [tensor], opts)

        with self.assertRaisesRegex(
            # throws error message from dispatcher
            RuntimeError,
            "There were no tensor arguments to this function",
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([output_ts], [], opts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_scatter_ops(self):
        # [FUNCTION DOC] For each root rank, scatters per-GPU lists of tensors (element r
        # holds the value r) from the root and asserts the output on every rank equals that
        # rank's own value.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            # [FUNCTION DOC] HELPER: Sets the root rank in the options; the root calls the group
            # scatter with its output and input lists, other ranks pass an empty input list; then
            # waits for the work.
            opts = c10d.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()

        # init output
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).cuda(device_id))

        # init input
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).cuda(gpu_idx))

        # test each rank to scatter
        expected = [torch.tensor([self.rank])]
        for rank in range(self.world_size):
            scatter(tensors, scatter_list, rank)
            self.assertEqual(expected, tensors)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_scatter_stress(self):
        # [FUNCTION DOC] Repeats the scatter test across 1000 prepared sets of tensors for each
        # root rank, verifying the output after each call.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            # [FUNCTION DOC] HELPER: Sets the root rank in the options; the root calls the group
            # scatter with its output and input lists, other ranks pass an empty input list; then
            # waits for the work.
            opts = c10d.ScatterOptions()
            opts.rootRank = rootRank
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            else:
                work = pg.scatter(output_t, [], opts)
            work.wait()

        stress_length = 1000

        # init output
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([-1]).cuda(device_id))

        # init input
        scatter_list = []
        for i in range(stress_length):
            scatter_list.append([[] for _ in range(num_gpus)])
            for idx, ls in enumerate(scatter_list[i]):
                gpu_idx = local_device_ids[idx]
                for rank in range(self.world_size):
                    ls.append(torch.tensor([rank]).cuda(gpu_idx))

        # test each rank to scatter
        expected = [torch.tensor([self.rank])]
        for i in range(stress_length):
            for rank in range(self.world_size):
                scatter(tensors[i], scatter_list[i], rank)
                # Verification
                self.assertEqual(tensors[i], expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_scatter_checks(self):
        # [FUNCTION DOC] Asserts argument validation of the group scatter: a root rank of -1 or
        # equal to the world size raises ValueError (invalid root rank), an integer instead of an
        # options object raises TypeError, and an empty output list raises RuntimeError from the
        # dispatcher (no tensor arguments).
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        # init output
        tensors = []
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).cuda(device_id))

        # init input
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).cuda(gpu_idx))

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter(tensors, scatter_list, opts)

        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            pg.scatter(tensors, scatter_list, 0)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter(tensors, scatter_list, opts)

        with self.assertRaisesRegex(
            # throws error message from dispatcher
            RuntimeError,
            "There were no tensor arguments to this function",
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], scatter_list, opts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_base_basics(self):
        # [FUNCTION DOC] Asserts reduce_scatter_single raises ValueError when the input size is
        # not output size times world size, and TypeError when input and output dtypes differ (the
        # dtype case also has a mis-sized output).
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            # [FUNCTION DOC] HELPER: Runs the group reduce_scatter_single on the output and input
            # tensors and waits for completion.
            work = pg.reduce_scatter_single(output_t, input_t)
            work.wait()

        # anticipate an error
        with self.assertRaisesRegex(
            ValueError,
            "input tensor must be the same size as output size times world size",
        ):
            input_t = torch.tensor([self.rank]).cuda(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=input_t.dtype).cuda(
                local_device_id
            )
            # fails the check because output_t is not correctly sized
            reduce_scatter_base(output_t, input_t)

        # anticipate an error
        with self.assertRaisesRegex(
            TypeError, "input tensor must be the same type as the output tensor."
        ):
            tensor = torch.tensor([self.rank], dtype=torch.float).cuda(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=torch.long).cuda(
                local_device_id
            )
            # fails the check because the dtype is different
            reduce_scatter_base(output_t, tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_v(self):
        # [FUNCTION DOC] Uneven reduce_scatter through the functional API: every rank supplies a
        # list of ones tensors of sizes 0..world size minus 1, and rank r receives a tensor of
        # size r that must equal the world size in every element.
        device = torch.device("cuda", self.rank_to_GPU[self.rank][0])
        # A list of tensors with different sizes
        input_list = [torch.ones(i, device=device) for i in range(self.world_size)]
        # The i-th output should have size i
        output = torch.zeros(self.rank, device=device)
        work = c10d.reduce_scatter(output, input_list, group=self.pg, async_op=True)
        expected = torch.ones(self.rank, device=device) * self.world_size
        work.wait()
        self.assertEqual(expected, output)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_all_gather_v(self):
        # [FUNCTION DOC] Uneven all_gather through the functional API: rank r contributes a tensor
        # of size r filled with r, and every rank must receive the list of tensors of sizes
        # 0..world size minus 1 with those values.
        device = torch.device("cuda", self.rank_to_GPU[self.rank][0])
        # A list of tensors with different sizes
        output_list = [torch.zeros(i, device=device) for i in range(self.world_size)]
        # The i-th input has size i, filled with value i
        input = torch.ones(self.rank, device=device) * self.rank
        work = c10d.all_gather(output_list, input, group=self.pg, async_op=True)
        expected = [torch.ones(i, device=device) * i for i in range(self.world_size)]
        work.wait()
        self.assertEqual(expected, output_list)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_ops(self):
        # [FUNCTION DOC] Using lists-of-lists inputs where rank r holds the values r+1, r+2, ...
        # (one per destination), checks SUM, MIN, MAX and PRODUCT results for each local GPU
        # against closed-form expected values. Then repeats the four ops with a single output
        # tensor and a flat list of inputs, and when NCCL is 2.11.1 or newer compares premul sum
        # (scalar and tensor factor) against a plain sum of pre-scaled inputs.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
        num_gpus = len(local_device_ids)

        def reduce_scatter(outputs, input_lists, op):
            # [FUNCTION DOC] HELPER: Runs the group reduce_scatter with the given outputs, input
            # lists and reduce op set in the options, and waits for completion.
            opts = c10d.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(outputs, input_lists, opts)
            work.wait()

        output = [torch.tensor([0]).cuda(i) for i in local_device_ids]

        #  GPU/rank
        #   0         [1], [2], [3], [4]
        #   1         [2], [3], [4], [5]
        #   2         [3], [4], [5], [6]
        #   3         [4], [5], [6], [7]

        # Sum
        tensor_lists = []
        input_per_gpu = []

        for i in range(self.world_size):
            input_per_gpu.append(torch.tensor([self.rank + i + 1]))

        for gpu in local_device_ids:
            tensor_lists.append([t.cuda(device=gpu) for t in input_per_gpu])

        reduce_scatter(output, tensor_lists, c10d.ReduceOp.SUM)

        for i in range(num_gpus):
            expected = torch.tensor(
                [
                    (1 + self.world_size) * self.world_size // 2
                    + self.world_size * self.rank
                ]
            )

            self.assertEqual(expected, output[i])

        # Min
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MIN)

        for i in range(num_gpus):
            expected = torch.tensor([self.rank + 1 + i])
            self.assertEqual(expected, output[i])

        # Max
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MAX)

        for i in range(num_gpus):
            expected = torch.tensor([self.rank + self.world_size + i])
            self.assertEqual(expected, output[i])

        # Product
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.PRODUCT)

        for i in range(num_gpus):
            prod_val = math.perm(self.rank + self.world_size, self.world_size)

            expected = torch.tensor([prod_val])
            self.assertEqual(expected, output[i])

        # Test the input params overridden scenarios, aka, when the input is
        # a list and output is just one tensor.
        # Sum
        output_tensor = torch.empty_like(input_per_gpu[0][0]).cuda(self.rank)
        input_list = [tensor[0].cuda(self.rank) for tensor in input_per_gpu]
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.SUM).wait()
        expected = torch.tensor(
            (1 + self.world_size) * self.world_size // 2 + self.world_size * self.rank
        )
        self.assertEqual(expected, output_tensor)

        # Min
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.MIN).wait()
        expected = torch.tensor(self.rank + 1)
        self.assertEqual(expected, output_tensor)

        # Max
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.MAX).wait()
        expected = torch.tensor(self.rank + self.world_size)
        self.assertEqual(expected, output_tensor)

        # Product
        pg.reduce_scatter(output_tensor, input_list, c10d.ReduceOp.PRODUCT).wait()
        prod_val = self.rank + 1
        for k in range(1, self.world_size):
            prod_val = prod_val * (self.rank + 1 + k)
        expected = torch.tensor(prod_val)
        self.assertEqual(expected, output_tensor)

        if torch.cuda.nccl.version() >= (2, 11, 1):
            for factor in (3.0, torch.tensor([5.0], device=self.rank)):
                if isinstance(factor, torch.Tensor):
                    factor_ref = factor.cpu().item()
                else:
                    factor_ref = factor
                output = [t.float() for t in output]
                tensor_lists = [[t.float() for t in tl] for tl in tensor_lists]
                output_ref = [t.float() for t in output]
                tensor_lists_ref = [
                    [t.float() * factor_ref for t in tl] for tl in tensor_lists
                ]
                reduce_scatter(output, tensor_lists, c10d._make_nccl_premul_sum(factor))
                reduce_scatter(output_ref, tensor_lists_ref, c10d.ReduceOp.SUM)
                self.assertEqual(output_ref, output)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_base_ops(self):
        # [FUNCTION DOC] Each rank supplies the sequence 0..world size minus 1 as input and a one-
        # element output; after reduce_scatter_single the output must equal the rank times the
        # world size.
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            # [FUNCTION DOC] HELPER: Runs the group reduce_scatter_single on the output and input
            # tensors and waits for completion.
            work = pg.reduce_scatter_single(output_t, input_t)
            work.wait()

        # reduce_scatter_base is GPU number agnostic.
        # Each rank contribute one tensor regardless of GPU counts
        output_t = torch.empty([1]).cuda(local_device_id)
        tensor = torch.arange(self.world_size, dtype=output_t.dtype).cuda(
            local_device_id
        )

        reduce_scatter_base(output_t, tensor)

        # Verification
        self.assertEqual(output_t[0], self.rank * self.world_size)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/157897")
    @requires_nccl_version((2, 24), "Need NCCL 2.24+ for Float8")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skip_but_pass_in_sandcastle_if(
        not PLATFORM_SUPPORTS_FP8, "Float8 requires sm >= 90"
    )
    def test_reduce_scatter_float8(self):
        # [FUNCTION DOC] Skipped on Linux, with NCCL older than 2.24, with fewer than two GPUs, or
        # without float8 platform support. Reduce-scatters ones of type float8 e5m2 (world size
        # times 1024 elements into 1024) through the functional API and asserts each output
        # element equals the world size.
        device = torch.device("cuda", self.rank_to_GPU[self.rank][0])

        numel = 1024
        output_tensor = torch.zeros(numel, dtype=torch.float32, device=device).to(
            torch.float8_e5m2
        )
        input_tensor = torch.ones(
            self.world_size * numel, dtype=torch.float32, device=device
        ).to(torch.float8_e5m2)
        dist.reduce_scatter_single(output_tensor, input_tensor)

        expected = (
            torch.empty_like(output_tensor).fill_(self.world_size).to(torch.float8_e5m2)
        )
        torch.testing.assert_close(output_tensor, expected)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_bfloat16(self):
        # [FUNCTION DOC] Reduce-scatters ones of type bfloat16 (world size times 1024 elements
        # into 1024) through the functional API and asserts each output element equals the world
        # size. Needs at least two GPUs but has no NCCL availability decorator.
        device = torch.device("cuda", self.rank_to_GPU[self.rank][0])

        numel = 1024
        output_tensor = torch.zeros(numel, dtype=torch.float32, device=device).to(
            torch.bfloat16
        )
        input_tensor = torch.ones(
            self.world_size * numel, dtype=torch.float32, device=device
        ).to(torch.bfloat16)
        # currently only reduce_scatter_tensor supports bfloat16
        dist.reduce_scatter_single(output_tensor, input_tensor)

        expected = (
            torch.empty_like(output_tensor).fill_(self.world_size).to(torch.bfloat16)
        )
        torch.testing.assert_close(output_tensor, expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_barrier(self):
        # [FUNCTION DOC] Starts all_reduce works on tensor lists spanning 1 up to the number
        # of local GPUs, calls the group barrier and waits for it, and then asserts every
        # tensor equals its value times the world size, showing the barrier ordered after
        # the prior works.
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]

        def allreduce(tensors):
            # [FUNCTION DOC] HELPER: Starts the group allreduce on the tensors with default
            # options and returns the work without waiting.
            opts = c10d.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work

        # Making the collective to operate on
        # 1, 2, 3, 4, .... len(local_device_ids) GPUs
        tensors_list = [[] for _ in range(len(local_device_ids))]

        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                tensors_list[i - 1].append(
                    torch.tensor([j + 1]).cuda(local_device_ids[j])
                )

        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)

        # Barrier will ensure that all previous work is completed
        pg.barrier().wait()

        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                self.assertEqual(
                    torch.tensor([(j + 1) * self.world_size]), tensors_list[i - 1][j]
                )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_send_recv(self):
        # [FUNCTION DOC] Seeds the random generator so both ranks create the same 10 by 10 tensor;
        # rank 0 sends it to rank 1, which receives into a fresh random tensor and asserts
        # equality. Other ranks do nothing.
        device = self.rank_to_GPU[self.rank][0]

        # Generate the same random tensor
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_send_recv_complex(self):
        # [FUNCTION DOC] Same as the send/recv test but with a complex float tensor: rank 0 sends,
        # rank 1 receives and compares.
        device = self.rank_to_GPU[self.rank][0]

        # Generate the same random tensor
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_send_recv_object_list(self):
        # [FUNCTION DOC] Rank 0 sends a list of objects (value 99) to rank 1 with the device
        # argument, and rank 1 receives into its own list and asserts the first element is 99.
        device = self.rank_to_GPU[self.rank][0]

        val = 99 if self.rank == 0 else None
        object_list = [val] * self.world_size
        if self.rank == 0:
            dist.send_object_list(object_list, 1, device=device)
        if self.rank == 1:
            dist.recv_object_list(object_list, 0, device=device)
            self.assertEqual(object_list[0], 99)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_tensor_register_hook(self):
        # [FUNCTION DOC] Sets the environment variable that enables the tensor register allocator
        # hook, runs an all_gather_single of each rank id into a world-size tensor and asserts it
        # equals the ranks in order, then removes the variable. Side effect: the variable is
        # removed only if the test reaches the end.
        os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "1"

        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            # [FUNCTION DOC] HELPER: Runs the group all_gather_single on the output and input
            # tensors and waits for completion.
            work = pg.all_gather_single(output_t, input_t)
            work.wait()

        # allgather_base is GPU number agnostic.
        # Each rank contribute one tensor regardless of GPU counts
        tensor = torch.tensor([self.rank]).cuda(local_device_id)
        output_t = torch.empty((self.world_size), dtype=tensor.dtype).cuda(
            local_device_id
        )

        allgather_base(output_t, tensor)

        # Verification
        self.assertEqual(torch.arange(self.world_size), output_t)

        # Unset env
        del os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"]

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_op_premul_sum(self):
        # [FUNCTION DOC] Skipped if NCCL is older than 2.11.1. Checks the PREMUL_SUM reduce op
        # with scalar and tensor factors for half, float and double in all_reduce, reduce (every
        # root) and reduce_scatter against closed-form expected values, then asserts
        # PREMUL_SUM(2.0) yields a ReduceOp, two equal ones compare equal, and the PREMUL_SUM enum
        # value is 8.
        if torch.cuda.nccl.version() < (2, 11, 1):
            self.skipTest("NCCL 2.11.1+ is required for PREMUL_SUM")

        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allreduce(tensors, op):
            # [FUNCTION DOC] HELPER: Runs the group allreduce on the tensor list with the given
            # reduce op and waits for completion.
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        def reduce(tensors, op, root=0):
            # [FUNCTION DOC] HELPER: Runs the group reduce on the tensor list with the given
            # reduce op and root rank (default 0) and waits for completion.
            opts = c10d.ReduceOptions()
            opts.reduceOp = op
            opts.rootRank = root
            work = pg.reduce(tensors, opts)
            work.wait()

        def reduce_scatter(output, input_lists, op):
            # [FUNCTION DOC] HELPER: Runs the group reduce_scatter with the given output, input
            # lists and reduce op, and waits for completion.
            opts = c10d.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(output, input_lists, opts)
            work.wait()

        # allreduce
        for dtype in (torch.half, torch.float, torch.double):
            scalar_factor = 3.0
            tensors = [
                torch.tensor([self.rank + 1]).cuda(local_device_id).to(dtype=dtype)
            ]

            allreduce(tensors, c10d.ReduceOp.PREMUL_SUM(scalar_factor))

            expected = scalar_factor * torch.tensor(
                [self.world_size * (self.world_size + 1) / 2],
                dtype=dtype,
                device=f"cuda:{local_device_id}",
            )
            self.assertEqual(expected, tensors[0])

        for dtype in (torch.half, torch.float, torch.double):
            tensor_factor = torch.tensor(
                [5.0], device=f"cuda:{local_device_id}", dtype=dtype
            )
            tensors = [
                torch.tensor([self.rank + 1]).cuda(local_device_id).to(dtype=dtype)
            ]

            allreduce(tensors, c10d.ReduceOp.PREMUL_SUM(tensor_factor))

            expected = tensor_factor * torch.tensor(
                [self.world_size * (self.world_size + 1) / 2],
                dtype=dtype,
                device=f"cuda:{local_device_id}",
            )
            self.assertEqual(expected, tensors[0])

        #  reduce
        for root in range(self.world_size):
            for dtype in (torch.half, torch.float, torch.double):
                # Test with scalar factor
                scalar_factor = 2.0
                tensors = [
                    torch.tensor([self.rank + 1]).cuda(local_device_id).to(dtype=dtype)
                ]

                reduce(tensors, c10d.ReduceOp.PREMUL_SUM(scalar_factor), root)

                if self.rank == root:
                    expected = scalar_factor * torch.tensor(
                        [self.world_size * (self.world_size + 1) / 2],
                        dtype=dtype,
                        device=f"cuda:{local_device_id}",
                    )
                    self.assertEqual(expected, tensors[0])

                # Test with tensor factor
                tensor_factor = torch.tensor(
                    [4.0], device=f"cuda:{local_device_id}", dtype=dtype
                )
                tensors = [
                    torch.tensor([self.rank + 1]).cuda(local_device_id).to(dtype=dtype)
                ]

                reduce(tensors, c10d.ReduceOp.PREMUL_SUM(tensor_factor), root)

                if self.rank == root:
                    expected = tensor_factor * torch.tensor(
                        [self.world_size * (self.world_size + 1) / 2],
                        dtype=dtype,
                        device=f"cuda:{local_device_id}",
                    )
                    self.assertEqual(expected, tensors[0])

        #  reduce_scatter
        for dtype in (torch.half, torch.float, torch.double):
            # Test with scalar factor
            scalar_factor = 3.0
            output = [torch.zeros(1, dtype=dtype, device=f"cuda:{local_device_id}")]
            input_lists = [
                [
                    torch.tensor([i + 1], dtype=dtype, device=f"cuda:{local_device_id}")
                    for i in range(self.world_size)
                ]
            ]

            reduce_scatter(output, input_lists, c10d.ReduceOp.PREMUL_SUM(scalar_factor))

            # Each rank receives sum of (rank+1) from all processes, multiplied by factor
            expected = scalar_factor * torch.tensor(
                [self.world_size * (self.rank + 1)],
                dtype=dtype,
                device=f"cuda:{local_device_id}",
            )
            self.assertEqual(expected, output[0])

            # Test with tensor factor
            tensor_factor = torch.tensor(
                [5.0], device=f"cuda:{local_device_id}", dtype=dtype
            )
            output = [torch.zeros(1, dtype=dtype, device=f"cuda:{local_device_id}")]
            input_lists = [
                [
                    torch.tensor([i + 1], dtype=dtype, device=f"cuda:{local_device_id}")
                    for i in range(self.world_size)
                ]
            ]

            reduce_scatter(output, input_lists, c10d.ReduceOp.PREMUL_SUM(tensor_factor))

            expected = tensor_factor * torch.tensor(
                [self.world_size * (self.rank + 1)],
                dtype=dtype,
                device=f"cuda:{local_device_id}",
            )
            self.assertEqual(expected, output[0])

        # Test that PREMUL_SUM is callable and returns a ReduceOp
        premul_op = c10d.ReduceOp.PREMUL_SUM(2.0)
        self.assertIsInstance(premul_op, c10d.ReduceOp)

        # Test equality comparison
        premul_op1 = c10d.ReduceOp.PREMUL_SUM(2.0)
        premul_op2 = c10d.ReduceOp.PREMUL_SUM(2.0)
        self.assertEqual(premul_op1, premul_op2)

        # Like other ReduceOps, PREMUL_SUM should have a unique integer value.
        self.assertEqual(c10d.ReduceOp.PREMUL_SUM, 8)


if __name__ == "__main__":
    run_tests()
