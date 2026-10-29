# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Backend-parameterized ProcessGroup API tests: rank normalization in
# new_group, group and work sequence numbers, collective timing, group init order, work
# registry cleanup, async work lifetime, stream-ordered wait, and work result/future APIs.
# AbstractProcessGroupTest is instantiated once per entry of C10D_BACKENDS. Import-time
# side effects: exits with status 0 if torch.distributed is unavailable.
import math
import sys

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

from torch._C._distributed_c10d import WorkResult
from torch.testing._internal.common_utils import get_cycles_per_ms, run_tests


class AbstractProcessGroupTest(C10dBackendTest):
    # [CLASS DOC] Backend-agnostic tests of ProcessGroup and Work behavior on a two-rank
    # group. Capability flags (sequence numbers, timing, work results, etc.) injected by
    # instantiate_backend_tests decide which tests are skipped for a backend.
    def test_new_group_rank_normalization(self):
        # [FUNCTION DOC] Creates a group from a tensor of all ranks and checks the reported
        # group ranks are the plain list of ints; a broadcast from rank 0 zeroes every
        # rank's tensor. Also asserts that a float tensor of ranks raises TypeError (ranks
        # must be a sequence of integers) and that a duplicate rank given as a mix of int
        # and tensor raises ValueError.
        self._init_pg()

        ranks = torch.arange(self.world_size)
        group = dist.new_group(ranks=ranks)
        self.assertEqual(
            dist.get_process_group_ranks(group), list(range(self.world_size))
        )

        tensor = torch.tensor([self.rank], device=self.device)
        dist.broadcast(tensor, src=0, group=group)
        self.assertEqual(tensor, torch.zeros_like(tensor))

        with self.assertRaisesRegex(TypeError, "ranks must be a sequence of integers"):
            dist.new_group(ranks=torch.arange(self.world_size, dtype=torch.float))

        with self.assertRaisesRegex(ValueError, "must not contain duplicate entries"):
            dist.new_group(ranks=[0, torch.tensor(0)])

    def test_sequence_numbers(self):
        # [FUNCTION DOC] Skipped if the backend lacks sequence numbers. Checks the default
        # group sequence number starts at 0 and increments by one per all_reduce (3 calls),
        # and that a new subgroup has its own counter (0 then 1 after one all_reduce) while
        # the default group's counter stays at 3.
        if not self.supports_sequence_numbers:
            self.skipTest(f"{self.backend_name} does not support sequence numbers")
        self._init_pg()
        default_pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(default_pg._get_sequence_number_for_group(), 0)
        for sequence_number in range(1, 4):
            dist.all_reduce(torch.ones(1, device=self.device))
            self.assertEqual(
                default_pg._get_sequence_number_for_group(), sequence_number
            )

        subgroup = dist.new_group(list(range(self.world_size)))
        self.assertEqual(subgroup._get_sequence_number_for_group(), 0)
        dist.all_reduce(torch.ones(1, device=self.device), group=subgroup)
        self.assertEqual(subgroup._get_sequence_number_for_group(), 1)
        self.assertEqual(default_pg._get_sequence_number_for_group(), 3)

    def test_work_sequence_number(self):
        # [FUNCTION DOC] Skipped if the backend does not report work sequence numbers. Three
        # async all_reduce calls on the default group must return work whose sequence
        # number is 1, 2, 3 (matching the group counter); then an async all_reduce on a new
        # subgroup must report sequence number 1.
        if not self.supports_work_sequence_number:
            self.skipTest(f"{self.backend_name} does not report work seq numbers")
        self._init_pg()
        default_pg = dist.distributed_c10d._get_default_group()
        for expected in range(1, 4):
            work = dist.all_reduce(torch.ones(1, device=self.device), async_op=True)
            self.assertEqual(work._get_sequence_number(), expected)
            self.assertEqual(default_pg._get_sequence_number_for_group(), expected)
            work.wait()

        subgroup = dist.new_group(list(range(self.world_size)))
        work = dist.all_reduce(
            torch.ones(1, device=self.device), group=subgroup, async_op=True
        )
        self.assertEqual(work._get_sequence_number(), 1)
        work.wait()

    def test_work_duration(self):
        # [FUNCTION DOC] Skipped if collective timing is unsupported. A work completed
        # before timing is enabled must raise RuntimeError when its duration is queried.
        # After enabling collectives timing on the default group, a new all_reduce work
        # (after wait and a CUDA synchronize on cuda) must report a positive finite duration.
        if not self.supports_collectives_timing:
            self.skipTest(f"{self.backend_name} does not support collectives timing")
        self._init_pg()
        default_pg = dist.distributed_c10d._get_default_group()
        tensor = torch.ones(1024, device=self.device)

        untimed_work = dist.all_reduce(tensor, async_op=True)
        untimed_work.wait()
        default_pg._enable_collectives_timing()
        # Timing is only enabled for collectives issued afterwards.
        with self.assertRaises(RuntimeError):
            untimed_work._get_duration()

        work = dist.all_reduce(tensor, async_op=True)
        work.wait()
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        duration = work._get_duration()
        self.assertGreater(duration, 0)
        self.assertTrue(math.isfinite(duration))

    def test_different_group_initialization_order(self):
        # [FUNCTION DOC] Creates a full-world group and then a singleton group containing
        # rank 0 (on cuda sets the device to 0 first). Rank 0 runs an all_reduce on the
        # singleton and checks the tensor is unchanged; all ranks then meet at a barrier on
        # the full group. Verifies group creation order does not hang or break.
        self._init_pg()
        if self.device_type == "cuda":
            torch.cuda.set_device(0)
        full_group = dist.new_group(list(range(self.world_size)))
        singleton = dist.new_group([0])
        if self.rank == 0:
            tensor = torch.ones(1, device=self.device)
            dist.all_reduce(tensor, group=singleton)
            self.assertEqual(tensor, torch.ones_like(tensor))
        dist.barrier(group=full_group)

    def test_wait_unregisters_work(self):
        # [FUNCTION DOC] Within the context allowing in-flight collectives as graph input,
        # asserts the global work registry is empty, holds one entry after an async
        # all_reduce, and is empty again after wait on that work.
        self._init_pg()
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            tensor = torch.ones(1, device=self.device)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(tensor, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

    def test_async_work_lifetime(self):
        # [FUNCTION DOC] Issues an async all_reduce, drops the work handle, and after a
        # barrier asserts the tensor holds the sum over ranks of (rank + 1). Then repeats
        # with the input tensor and the work both deleted, expecting only a clean barrier
        # (no crash), proving the collective keeps its buffers alive.
        self._init_pg()
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        del work
        dist.barrier()
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)

        tensor = torch.ones(4, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        del tensor
        del work
        dist.barrier()

    def test_work_wait_is_stream_ordered(self):
        # [FUNCTION DOC] Skipped on non-cuda backends. Warms up, then sleeps the current
        # stream about 250 ms and issues an async all_reduce; after wait returns, asserts
        # the current stream is still busy (query false), showing wait only orders the
        # current stream rather than blocking the host. Then synchronizes.
        if self.device_type != "cuda":
            self.skipTest(f"{self.backend_name} does not use CUDA")
        self._init_pg()
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        torch.cuda._sleep(int(250 * get_cycles_per_ms()))
        work = dist.all_reduce(torch.ones(4, device=self.device), async_op=True)

        work.wait()

        self.assertFalse(torch.cuda.current_stream().query())
        torch.cuda.synchronize()

    def test_work_result_future(self):
        # [FUNCTION DOC] Skipped if the backend lacks work results. Waits on the future
        # result of an async all_reduce, expecting it converts to WorkResult.SUCCESS, and
        # asserts the work reports completed.
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        self._init_pg()
        work = dist.all_reduce(torch.ones(4, device=self.device), async_op=True)

        result = work.get_future_result().wait()

        self.assertEqual(WorkResult(result), WorkResult.SUCCESS)
        self.assertTrue(work.is_completed())

    def test_work_future(self):
        # [FUNCTION DOC] Waits on the future from an async all_reduce and asserts the tensor
        # holds the sum over ranks of (rank + 1).
        self._init_pg()
        tensor = torch.full((4,), float(self.rank + 1), device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        future = work.get_future()
        future.wait()
        expected = torch.full_like(tensor, sum(range(1, self.world_size + 1)))
        self.assertEqual(tensor, expected)


instantiate_backend_tests(
    globals(), "ProcessGroup", AbstractProcessGroupTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()
