# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process tests that collectives of the c10d backends run under CUDA graph
# capture and replay. Exits silently at import (sys.exit 0) if torch.distributed is not
# available. Defines one abstract test class and, at import, generates one concrete test class
# per CUDA backend via instantiate_backend_tests. Runnable directly through run_tests.

import sys
import unittest

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10dBackendTest,
    CUDA_BACKENDS,
    instantiate_backend_tests,
)

from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM


ASYNC_OPS = (False, True)


@unittest.skipIf(
    TEST_WITH_ROCM, "RCCL does not support all collectives under HIP graph capture"
)
class AbstractCUDAGraphsTest(C10dBackendTest):
    # [CLASS DOC] Abstract (skipped on ROCm) test suite checking that collectives can be captured
    # into a torch.cuda.CUDAGraph and replayed. Relies on attributes supplied by the generated
    # subclass: rank, world_size, device, dtypes, complex_dtypes, backend_name and
    # supports_cuda_graph_barrier. Per-collective helpers take a dtype and async flag; test
    # methods drive them under subTest for every dtype and sync/async mode.
    def _tensor(self, dtype, rank=None):
        # [FUNCTION DOC] HELPER: Builds a 4-element tensor on the test device filled with rank +
        # 1 (this rank by default), created as float32 and then cast to dtype. Returns the
        # tensor.
        rank = self.rank if rank is None else rank
        return torch.full((4,), rank + 1, dtype=torch.float32, device=self.device).to(
            dtype
        )

    def _capture_and_replay(self, op, async_op):
        # [FUNCTION DOC] HELPER: Captures the callable op into a new CUDA graph, then replays it
        # and synchronizes the device. During capture, asserts that op returned a work handle and
        # waits on it when async_op is true, otherwise asserts it returned None. Raises on
        # assertion failure.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            work = op()
            if async_op:
                self.assertIsNotNone(work)
                work.wait()
            else:
                self.assertIsNone(work)
        graph.replay()
        torch.cuda.synchronize()

    def _test_all_reduce(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays all_reduce on a per-rank tensor, then
        # asserts every element equals the sum of 1..world_size.
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.all_reduce(tensor, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)

    def _test_broadcast(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays broadcast from rank 0, then asserts the
        # tensor equals rank 0's tensor on every rank.
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.broadcast(tensor, src=0, async_op=async_op),
            async_op,
        )
        self.assertEqual(tensor, self._tensor(dtype, rank=0))

    def _test_all_gather(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays all_gather into a list of empty tensors,
        # then asserts entry i equals the tensor made by rank i.
        tensor = self._tensor(dtype)
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        self._capture_and_replay(
            lambda: dist.all_gather(output, tensor, async_op=async_op),
            async_op,
        )
        for rank, result in enumerate(output):
            self.assertEqual(result, self._tensor(dtype, rank))

    def _test_all_gather_single(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays all_gather_single into one flat output of
        # world_size times the input size, then asserts it equals the concatenation of every
        # rank's tensor in rank order.
        tensor = self._tensor(dtype)
        output = torch.empty(
            self.world_size * tensor.numel(),
            dtype=dtype,
            device=self.device,
        )
        self._capture_and_replay(
            lambda: dist.all_gather_single(output, tensor, async_op=async_op),
            async_op,
        )
        expected = torch.cat(
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    def _test_reduce(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays reduce to rank 0; only on rank 0 asserts
        # the tensor equals the sum of 1..world_size.
        tensor = self._tensor(dtype)
        self._capture_and_replay(
            lambda: dist.reduce(tensor, dst=0, async_op=async_op),
            async_op,
        )
        if self.rank == 0:
            expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
            self.assertEqual(tensor, expected)

    def _test_gather(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays gather to rank 0 (output list is only
        # allocated on rank 0, None elsewhere); on rank 0 asserts entry i equals rank i's tensor.
        tensor = self._tensor(dtype)
        output = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        self._capture_and_replay(
            lambda: dist.gather(tensor, output, dst=0, async_op=async_op),
            async_op,
        )
        if self.rank == 0:
            for rank, result in enumerate(output):
                self.assertEqual(result, self._tensor(dtype, rank))

    def _test_scatter(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays scatter from rank 0 (input list only on
        # rank 0); asserts each rank received the tensor made for its own rank.
        output = torch.empty(4, dtype=dtype, device=self.device)
        inputs = (
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
            if self.rank == 0
            else None
        )
        self._capture_and_replay(
            lambda: dist.scatter(output, inputs, src=0, async_op=async_op),
            async_op,
        )
        self.assertEqual(output, self._tensor(dtype, self.rank))

    def _test_reduce_scatter(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays reduce_scatter with world_size identical
        # input tensors per rank; asserts the output equals the sum of 1..world_size.
        inputs = [self._tensor(dtype) for _ in range(self.world_size)]
        output = torch.empty_like(inputs[0])
        self._capture_and_replay(
            lambda: dist.reduce_scatter(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(output, sum(range(1, self.world_size + 1)))
        self.assertEqual(output, expected)

    def _test_reduce_scatter_single(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays reduce_scatter_single over a concatenated
        # input; asserts the 4-element output equals the sum of 1..world_size.
        inputs = torch.cat([self._tensor(dtype) for _ in range(self.world_size)])
        output = torch.empty(4, dtype=dtype, device=self.device)
        self._capture_and_replay(
            lambda: dist.reduce_scatter_single(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.full_like(output, sum(range(1, self.world_size + 1)))
        self.assertEqual(output, expected)

    def _test_all_to_all(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays all_to_all where each rank sends its own
        # tensor to every peer; asserts output i equals the tensor made by rank i.
        inputs = [self._tensor(dtype, self.rank) for _ in range(self.world_size)]
        outputs = [torch.empty_like(inputs[0]) for _ in range(self.world_size)]
        self._capture_and_replay(
            lambda: dist.all_to_all(outputs, inputs, async_op=async_op),
            async_op,
        )
        for rank, result in enumerate(outputs):
            self.assertEqual(result, self._tensor(dtype, rank))

    def _test_all_to_all_single(self, dtype, async_op):
        # [FUNCTION DOC] HELPER: Captures and replays all_to_all_single over a concatenated input
        # of this rank's tensor; asserts the output equals the rank-ordered concatenation of
        # every rank's tensor.
        inputs = torch.cat(
            [self._tensor(dtype, self.rank) for _ in range(self.world_size)]
        )
        output = torch.empty_like(inputs)
        self._capture_and_replay(
            lambda: dist.all_to_all_single(output, inputs, async_op=async_op),
            async_op,
        )
        expected = torch.cat(
            [self._tensor(dtype, rank) for rank in range(self.world_size)]
        )
        self.assertEqual(output, expected)

    def test_collectives(self):
        # [FUNCTION DOC] Initializes the process group, runs a warmup all_reduce and
        # synchronizes, then runs each collective helper (all_reduce, broadcast, all_gather,
        # all_gather_single, reduce, gather, scatter, reduce_scatter, reduce_scatter_single,
        # all_to_all, all_to_all_single) for every dtype in self.dtypes and both sync/async
        # modes, each in a subTest.
        self._init_pg()
        warmup = torch.ones(1, device=self.device)
        dist.all_reduce(warmup)
        torch.cuda.synchronize()

        tests = (
            self._test_all_reduce,
            self._test_broadcast,
            self._test_all_gather,
            self._test_all_gather_single,
            self._test_reduce,
            self._test_gather,
            self._test_scatter,
            self._test_reduce_scatter,
            self._test_reduce_scatter_single,
            self._test_all_to_all,
            self._test_all_to_all_single,
        )
        for test in tests:
            for dtype in self.dtypes:
                for async_op in ASYNC_OPS:
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(dtype, async_op)

    def test_barrier(self):
        # [FUNCTION DOC] Skips the test if the backend does not support barrier under CUDA
        # graphs. Otherwise initializes the process group, runs a warmup all_reduce, and
        # captures/replays barrier (device_ids set to this rank) in both sync and async modes.
        if not self.supports_cuda_graph_barrier:
            self.skipTest(f"{self.backend_name} barrier does not support CUDA graphs")
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        for async_op in ASYNC_OPS:
            self._capture_and_replay(
                lambda: dist.barrier(
                    device_ids=[self.rank],
                    async_op=async_op,
                ),
                async_op,
            )

    def test_complex_collectives(self):
        # [FUNCTION DOC] Initializes the process group and warms up, then runs all_reduce,
        # broadcast and reduce helpers under graph capture for every dtype in self.complex_dtypes
        # and both sync/async modes, each in a subTest.
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        for dtype in self.complex_dtypes:
            for async_op in ASYNC_OPS:
                for test in (
                    self._test_all_reduce,
                    self._test_broadcast,
                    self._test_reduce,
                ):
                    with self.subTest(
                        collective=test.__name__,
                        dtype=dtype,
                        async_op=async_op,
                    ):
                        test(dtype, async_op)


instantiate_backend_tests(
    globals(), "CUDAGraphs", AbstractCUDAGraphsTest, CUDA_BACKENDS
)


if __name__ == "__main__":
    run_tests()
