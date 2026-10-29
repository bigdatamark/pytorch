# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Multi-process tests for c10d reconfigurable (fault tolerant) process
# groups. Defines an abstract test mixin that initializes a process group with
# reconfigure enabled, exchanges reconfigure handles through a file store, and
# exercises reconfigure, shrink, grow, abort and timeout recovery. Concrete test
# classes are generated at import time, one per backend in the backend list (gloo on
# cpu, nccl2 on cuda), and registered in module globals with skip decorators when
# the backend or three GPUs are unavailable. Also has two single-process contract
# tests. Exits the process with status zero at import if torch.distributed is unavailable.
import os
import sys
import time
import unittest
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import WorkResult
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TEST_CUDA,
    TestCase,
)


@dataclass(frozen=True)
class FaultToleranceBackend:
    # [CLASS DOC] Frozen record describing one backend under test: backend name, device type
    # string, and a flag saying whether the backend reports work results (default false).
    # Consumed by the test class factory below.
    name: str
    device_type: str
    supports_work_result: bool = False


FAULT_TOLERANCE_BACKENDS = [
    FaultToleranceBackend("gloo", "cpu"),
    FaultToleranceBackend("nccl2", "cuda", supports_work_result=True),
]


class AbstractFaultToleranceTest:
    # [CLASS DOC] Test mixin (not a test case itself) combined with MultiProcessTestCase by
    # the factory. Expects the subclass to define backend_name, device_type and
    # supports_work_result as class attributes. Runs 3 ranks, each in its own spawned process.
    # Per-test state set up here: the file store, the default process group and the backend
    # implementation for the rank device. Helpers build a reconfigurable group, collect
    # handles of all ranks, reconfigure to a chosen handle list and check an all_reduce sum.
    # The test methods cover reconfigure, shrink, scale and recovery scenarios.
    @property
    def world_size(self):
        # [FUNCTION DOC] Property: number of ranks used by every test, always 3.
        return 3

    @property
    def device(self):
        # [FUNCTION DOC] Property: device for this rank. For cuda device_type returns the
        # string for the GPU whose index equals the rank, otherwise returns the device type
        # string itself.
        if self.device_type == "cuda":
            return f"cuda:{self.rank}"
        return self.device_type

    def setUp(self):
        # [FUNCTION DOC] Test fixture: calls the parent setUp and then spawns the worker
        # processes that run the test body.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Test fixture: destroys the default process group if one is
        # initialized, calls the parent tearDown, then tries to delete the store file,
        # silently ignoring OS errors.
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _create_store(self):
        # [FUNCTION DOC] HELPER: returns a file based store over the test file with
        # world size ranks.
        return dist.FileStore(self.file_name, self.world_size)

    def _init_reconfigurable_pg(self):
        # [FUNCTION DOC] HELPER: creates the store, selects the GPU for this rank on cuda, and
        # initializes the default process group with reconfigure enabled (30 second timeout).
        # Saves the store, the default group and the backend implementation on self, and
        # asserts that both the distributed module and the backend report reconfigure support.
        # Side effect: initializes global process group state.
        self.store = self._create_store()
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            timeout=timedelta(seconds=30),
            enable_reconfigure=True,
        )
        self.pg = c10d._get_default_group()
        self.backend = dist.get_backend_impl(self.pg, torch.device(self.device))
        self.assertTrue(dist._supports_reconfigure())
        self.assertTrue(self.backend.supports_reconfigure)

    def _collect_handles(self, key_prefix):
        # [FUNCTION DOC] HELPER: publishes this rank reconfigure handle in the store under a
        # key made of the prefix and the rank, then reads every rank handle back (blocking
        # until all are present) and returns them as decoded strings ordered by rank.
        handle = dist._get_reconfigure_handle()
        self.store.set(f"{key_prefix}_{self.rank}", handle)
        return [
            self.store.get(f"{key_prefix}_{rank}").decode("utf-8")
            for rank in range(self.world_size)
        ]

    def _store_barrier(self, key_prefix):
        # [FUNCTION DOC] HELPER: store based barrier. Sets a marker key for this rank, then
        # blocks reading the marker keys of all ranks.
        self.store.set(f"{key_prefix}_{self.rank}", "1")
        for rank in range(self.world_size):
            self.store.get(f"{key_prefix}_{rank}")

    def _reconfigure(self, uuid, handles):
        # [FUNCTION DOC] HELPER: calls the distributed reconfigure function with the given
        # uuid and handle list, using a 30 second timeout, and waits for the returned work to
        # finish. Errors raised by the wait propagate.
        work = dist._reconfigure(
            uuid,
            handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

    def _create_reconfigured_pg(self, name, uuid):
        # [FUNCTION DOC] HELPER: initializes a reconfigurable group, gathers handles
        # under the name plus init suffix, reconfigures with the given uuid, asserts
        # the world size and rank are unchanged, and returns the handles collected
        # again under the name plus post suffix.
        self._init_reconfigurable_pg()
        handles = self._collect_handles(f"{name}_init")
        self._reconfigure(uuid, handles)
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)
        return self._collect_handles(f"{name}_post")

    def _assert_all_reduce_sum(self, expected_value):
        # [FUNCTION DOC] HELPER: each rank all_reduces a 4 element tensor filled
        # with its rank plus one, and asserts the result on cpu equals a tensor
        # filled with the expected value.
        tensor = torch.full((4,), dist.get_rank() + 1.0, device=self.device)
        dist.all_reduce(tensor)
        expected = torch.full((4,), expected_value, dtype=tensor.dtype)
        self.assertEqual(tensor.cpu(), expected)

    def test_reconfigure_basic(self):
        # [FUNCTION DOC] Test: reconfiguring a freshly initialized group with all
        # ranks succeeds.
        self._create_reconfigured_pg("ft_basic", 100)

    def test_reconfigure_then_all_reduce(self):
        # [FUNCTION DOC] Test: after a full reconfigure, all_reduce over all ranks gives the
        # sum of one through world size.
        self._create_reconfigured_pg("ft_all_reduce", 200)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_then_send_recv(self):
        # [FUNCTION DOC] Test: after a full reconfigure, each rank sends to the next rank and
        # receives from the previous rank through the backend directly (even ranks send first,
        # odd ranks receive first to avoid deadlock) and checks the received tensor equals the
        # sender rank plus one.
        self._create_reconfigured_pg("ft_send_recv", 300)

        rank = dist.get_rank()
        send_rank = (rank + 1) % self.world_size
        recv_rank = (rank - 1 + self.world_size) % self.world_size
        send_tensor = torch.full((4,), rank + 1.0, device=self.device)
        recv_tensor = torch.zeros(4, device=self.device)

        if rank % 2 == 0:
            send_work = self.backend.send([send_tensor], send_rank, 0)
            recv_work = self.backend.recv([recv_tensor], recv_rank, 0)
        else:
            recv_work = self.backend.recv([recv_tensor], recv_rank, 0)
            send_work = self.backend.send([send_tensor], send_rank, 0)

        send_work.wait()
        recv_work.wait()
        self.assertEqual(recv_tensor.cpu(), torch.full((4,), recv_rank + 1.0))

    def test_work_explicit_timeout_includes_prelaunch_stall(self):
        # [FUNCTION DOC] Test: skipped unless the backend reports work results. After a full
        # reconfigure it stalls the GPU stream with a sleep, launches an async all_reduce and
        # waits with a 50 ms timeout. Expects a backend error mentioning timeout even though
        # the stream has not finished, the stream query to be false, the work to be completed,
        # and the future result to be the timeout result code. Synchronizes at the end.
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        self._create_reconfigured_pg("ft_work_timeout", 1300)
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        torch.cuda._sleep(int(500 * get_cycles_per_ms()))
        work = dist.all_reduce(torch.ones(4, device=self.device), async_op=True)

        with self.assertRaisesRegex(dist.DistBackendError, "timed out"):
            work.wait(timeout=timedelta(milliseconds=50))

        self.assertFalse(torch.cuda.current_stream().query())
        self.assertTrue(work.is_completed())
        self.assertEqual(
            WorkResult(work.get_future_result().wait()), WorkResult.TIMEOUT
        )
        torch.cuda.synchronize()

    def test_work_reports_communicator_error(self):
        # [FUNCTION DOC] Test: skipped unless the backend reports work results. Rank 0
        # launches an async all_reduce while other ranks idle, aborts the backend, then
        # expects the work to be completed but unsuccessful, with a backend error exception, a
        # communicator error work result, and wait raising an error containing the NCCL
        # operation failed text. Other ranks only sleep one second.
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        self._create_reconfigured_pg("ft_work_error", 1301)
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()

        if self.rank == 0:
            work = dist.all_reduce(torch.ones(1, device=self.device), async_op=True)
            time.sleep(0.5)
            self.backend.abort()
            self.assertTrue(work.is_completed())
            self.assertFalse(work.is_success())
            self.assertIsInstance(work.exception(), dist.DistBackendError)
            self.assertEqual(
                WorkResult(work.get_future_result().wait()),
                WorkResult.COMM_ERROR,
            )
            with self.assertRaisesRegex(dist.DistBackendError, "NCCL operation failed"):
                work.wait()
        else:
            time.sleep(1)

    def test_shrink_exclude_last_rank(self):
        # [FUNCTION DOC] Test: shrinks the group by excluding the last rank. The excluded rank
        # only waits at a store barrier. Survivors reconfigure with a new uuid and the first
        # handles, then check world size is one smaller, rank is unchanged, all_reduce sums
        # correctly and a broadcast from group rank 0 delivers 42.
        handles = self._create_reconfigured_pg("ft_shrink_last", 400)
        excluded_rank = self.world_size - 1
        if self.rank == excluded_rank:
            self._store_barrier("ft_shrink_last_done")
            return

        self._reconfigure(401, handles[:excluded_rank])
        self.assertEqual(dist.get_world_size(), self.world_size - 1)
        self.assertEqual(dist.get_rank(), self.rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size)))

        tensor = torch.zeros(4, device=self.device)
        if dist.get_rank() == 0:
            tensor.fill_(42.0)
        dist.broadcast(tensor, group_src=0)
        self.assertEqual(tensor.cpu(), torch.full((4,), 42.0))
        self._store_barrier("ft_shrink_last_done")

    def test_shrink_exclude_middle_rank(self):
        # [FUNCTION DOC] Test: shrinks the group by excluding the middle rank.
        # Survivors reconfigure with the remaining handles and check that ranks
        # above the excluded one shift down by one, the world size shrinks, and
        # all_reduce gives the sum over survivors.
        handles = self._create_reconfigured_pg("ft_shrink_middle", 500)
        excluded_rank = self.world_size // 2
        if self.rank == excluded_rank:
            self._store_barrier("ft_shrink_middle_done")
            return

        surviving_handles = [
            handle for rank, handle in enumerate(handles) if rank != excluded_rank
        ]
        self._reconfigure(501, surviving_handles)

        expected_rank = self.rank if self.rank < excluded_rank else self.rank - 1
        self.assertEqual(dist.get_world_size(), self.world_size - 1)
        self.assertEqual(dist.get_rank(), expected_rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size)))
        self._store_barrier("ft_shrink_middle_done")

    def test_reconfigure_scale_down_up(self):
        # [FUNCTION DOC] Test: each rank first reconfigures to its own single rank group
        # (unique uuid per rank), checks world size one and rank zero, then all ranks
        # reconfigure back to the full group, and finally each shrinks to a single rank group
        # again. Ends with a store barrier.
        self._init_reconfigurable_pg()
        # Each rank shrinks to its own disjoint group, so each needs a unique uuid.
        self._reconfigure(600 + self.rank, [dist._get_reconfigure_handle()])
        self.assertEqual(dist.get_world_size(), 1)
        self.assertEqual(dist.get_rank(), 0)

        handles = self._collect_handles("ft_scale_down_up")
        self._reconfigure(603, handles)
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)

        self._reconfigure(604 + self.rank, [dist._get_reconfigure_handle()])
        self.assertEqual(dist.get_world_size(), 1)
        self.assertEqual(dist.get_rank(), 0)
        self._store_barrier("ft_scale_down_up_done")

    def test_reconfigure_single_to_all(self):
        # [FUNCTION DOC] Test: each rank reconfigures to a single rank group using a
        # unique uuid, then all ranks reconfigure to the full group and an
        # all_reduce sum is verified.
        self._init_reconfigurable_pg()
        # Each rank shrinks to its own disjoint group, so each needs a unique uuid.
        self._reconfigure(700 + self.rank, [dist._get_reconfigure_handle()])

        handles = self._collect_handles("ft_single_to_all")
        self._reconfigure(703, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_identity(self):
        # [FUNCTION DOC] Test: reconfiguring to the same full membership a second time (new
        # uuid) leaves a working group; verified with an all_reduce sum.
        self._create_reconfigured_pg("ft_identity", 800)
        handles = self._collect_handles("ft_identity_again")
        self._reconfigure(801, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_late_join(self):
        # [FUNCTION DOC] Test: only the lower half of the ranks form a group first, while the
        # remaining ranks do not reconfigure yet; then all ranks reconfigure together (late
        # join) and an all_reduce sum over all ranks is verified.
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_late_join_initial")
        initial_world_size = self.world_size // 2
        if self.rank < initial_world_size:
            self._reconfigure(900, handles[:initial_world_size])

        handles = self._collect_handles("ft_late_join_all")
        self._reconfigure(901, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_merge_split(self):
        # [FUNCTION DOC] Test: ranks are split into a lower and an upper group, each
        # reconfiguring with its own uuid, then all ranks merge through one more reconfigure
        # and an all_reduce sum over all ranks is verified.
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_merge_split_initial")
        split = self.world_size // 2
        if self.rank < split:
            self._reconfigure(1000, handles[:split])
        else:
            self._reconfigure(1001, handles[split:])

        handles = self._collect_handles("ft_merge_split_all")
        self._reconfigure(1002, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_after_abort(self):
        # [FUNCTION DOC] Test: after a full reconfigure the backend is aborted. The reported
        # error is a communicator error for nccl2 and success for other backends. A
        # reconfigure with a fresh uuid must then recover the group, verified with an
        # all_reduce sum.
        # Port of torchcomms' ReconfigureTest.test_reconfigure_after_abort:
        # abort() (a revoke in reconfigurable mode) must be recoverable by a
        # reconfigure() with a fresh uuid.
        self._create_reconfigured_pg("ft_abort", 1200)
        self.backend.abort()

        from torch._C._distributed_c10d import ErrorType

        is_nccl = self.backend_name == "nccl2"
        expected = ErrorType.COMM_ERROR if is_nccl else ErrorType.SUCCESS
        self.assertEqual(self.backend.get_error(), expected)

        handles = self._collect_handles("ft_abort_recover")
        self._reconfigure(1201, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_after_timeout(self):
        # [FUNCTION DOC] Test: after a working all_reduce, the backend timeout is set to 50 ms
        # and rank 1 alone launches an all_reduce that cannot complete. Either wait raises an
        # error saying timed out, or the backend error becomes the timeout code within 10
        # seconds and a further all_reduce raises timed out. The timeout is restored to 30
        # seconds, ranks synchronize at a store barrier, then a reconfigure with a new uuid
        # recovers the group, verified with an all_reduce sum.
        from torch._C._distributed_c10d import ErrorType

        self._create_reconfigured_pg("ft_timeout", 1300)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))
        self.backend.set_timeout(timedelta(milliseconds=50))

        if self.rank == 1:
            tensor = torch.ones(4, device=self.device)
            work = dist.all_reduce(tensor, async_op=True)
            try:
                work.wait()
            except RuntimeError as error:
                self.assertRegex(str(error), "[Tt]imed out")
            else:
                deadline = time.monotonic() + 10
                while (
                    self.backend.get_error() == ErrorType.SUCCESS
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
                self.assertEqual(self.backend.get_error(), ErrorType.TIMEOUT)
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    dist.all_reduce(tensor, async_op=True)
            del work

        self.backend.set_timeout(timedelta(seconds=30))
        self._store_barrier("ft_timeout_observed")

        handles = self._collect_handles("ft_timeout_recover")
        self._reconfigure(1301, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_rejects_reused_uuid(self):
        # [FUNCTION DOC] Test: reusing a reconfigure uuid must be rejected. Non
        # nccl2 backends: each rank reconfigures to a single rank group then repeats
        # with the same uuid and expects an already used error. nccl2: all ranks
        # reconfigure with one uuid, then repeat it with a 500 ms timeout; rank 0
        # must see already used and other ranks a wait timeout error. After a store
        # barrier the original group must still all_reduce correctly.
        self._init_reconfigurable_pg()
        if self.backend_name != "nccl2":
            uuid = 1100 + self.rank
            self._reconfigure(uuid, [dist._get_reconfigure_handle()])
            with self.assertRaisesRegex(RuntimeError, "already used"):
                self._reconfigure(uuid, [dist._get_reconfigure_handle()])
            return

        uuid = 1100
        handles = self._collect_handles("ft_reused_uuid_initial")
        self._reconfigure(uuid, handles)
        handles = self._collect_handles("ft_reused_uuid_current")
        error = "already used" if self.rank == 0 else "Wait timeout"
        with self.assertRaisesRegex(RuntimeError, error):
            dist._reconfigure(
                uuid,
                handles,
                timeout=timedelta(milliseconds=500),
            ).wait()
        self._store_barrier("ft_reused_uuid_rejected")
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_timeout_is_retryable(self):
        # [FUNCTION DOC] Test: only for nccl2. Rank 0 starts a reconfigure naming only two of
        # the three handles with a 500 ms timeout and expects a timed out error. After a store
        # barrier all ranks reconfigure with a new uuid and all handles, showing a timed out
        # reconfigure is retryable; verified with an all_reduce sum.
        if self.backend_name != "nccl2":
            self.skipTest("nonblocking NCCL initialization behavior")
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_timeout_retry_initial")

        if self.rank == 0:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                dist._reconfigure(
                    1400,
                    handles[:2],
                    timeout=timedelta(milliseconds=500),
                ).wait()
        self._store_barrier("ft_timeout_retry_observed")

        handles = self._collect_handles("ft_timeout_retry_current")
        self._reconfigure(1401, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))


def _make_fault_tolerance_test_class(backend):
# [FUNCTION DOC] HELPER: factory building the concrete test class for one backend record.
# Creates a class combining the mixin and MultiProcessTestCase, sets backend_name, device_type
# and supports_work_result on it, names it backend name capitalized plus FaultToleranceTest,
# and wraps it with skip decorators: skipped if the backend is unavailable and, for cuda, if
# the test flag or fewer than 3 GPUs are present. Returns the decorated class.
    class FaultToleranceTest(AbstractFaultToleranceTest, MultiProcessTestCase):
        # [CLASS DOC] HELPER: empty concrete test case joining the mixin and the multi-process
        # test case; attributes are filled in by the enclosing factory.
        pass

    FaultToleranceTest.backend_name = backend.name
    FaultToleranceTest.device_type = backend.device_type
    FaultToleranceTest.supports_work_result = backend.supports_work_result
    FaultToleranceTest.__name__ = f"{backend.name.capitalize()}FaultToleranceTest"
    FaultToleranceTest.__qualname__ = FaultToleranceTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend.name),
        f"{backend.name} backend is not available",
    )(FaultToleranceTest)
    if backend.device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 3,
            "fault tolerance CUDA tests require at least 3 GPUs",
        )(cls)
    return cls


for backend in FAULT_TOLERANCE_BACKENDS:
    class_name = f"{backend.name.capitalize()}FaultToleranceTest"
    globals()[class_name] = _make_fault_tolerance_test_class(backend)


class ReconfigureContractTest(TestCase):
    # [CLASS DOC] Single process test case for the process group reconfigure API contract.
    def test_reconfigure_rejects_multiple_backends(self) -> None:
        # [FUNCTION DOC] Test: a process group with both a gloo cpu backend and an
        # nccl cuda backend registered must raise an error mentioning multiple
        # backends when asking for reconfigure support, for a reconfigure handle, or
        # when reconfiguring with default options.
        pg = dist.ProcessGroup(0, 1)
        pg._register_backend(torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO)
        pg._register_backend(torch.device("cuda"), dist.ProcessGroup.BackendType.NCCL)

        msg = "multiple backends"
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.supports_reconfigure
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.reconfigure(c10d.ReconfigureOptions())


class BackendCapabilityContractTest(TestCase):
    # [CLASS DOC] Single process test case checking base backend bindings are safe to call on
    # any backend.
    """Ensures base-Backend bindings are safe to call on any backend (e.g. Gloo)
    without throwing, so hasattr-based capability detection stays valid."""

    def test_get_error_returns_success_on_base_backend(self) -> None:
        # [FUNCTION DOC] Test: a plain base backend object has a get_error attribute and it
        # returns the success error type.
        from torch._C._distributed_c10d import Backend as C10DBackend, ErrorType

        backend = C10DBackend(0, 1)
        self.assertTrue(hasattr(backend, "get_error"))
        self.assertEqual(backend.get_error(), ErrorType.SUCCESS)


if __name__ == "__main__":
    run_tests()
