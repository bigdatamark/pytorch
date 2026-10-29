# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for Python-implemented ProcessGroup subclasses (the PyProcessGroup
# trampoline). Defines small fake process groups (single-rank group returning real or wrapped Work
# objects, attribute-overriding group, store-constructed group, groups advertising coalescing,
# reconfigure and window support) and test cases that check DDP over them, that C++-side virtual
# calls dispatch into the Python overrides, delegation of the reconfigure and window helpers,
# stream blocking with Work objects, and batch_isend_irecv over registered Python backends.
# Importing has no side effects beyond imports; running as a script calls run_tests.

import os
import time
import unittest
import weakref
from datetime import timedelta

import test_c10d_common

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._C._distributed_c10d import _create_work_from_future
from torch.distributed.distributed_c10d import (
    _coalescing_manager,
    _get_default_group,
    _register_process_group,
    _unregister_process_group,
    ReconfigureOptions,
)
from torch.futures import Future
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def create_work(result):
        # [FUNCTION DOC] HELPER: Builds a Future already completed with result and wraps it into a
        # c10d Work object through the C++ helper; returns that Work.
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class MyWork(dist._Work):
    # [CLASS DOC] HELPER: Python Work subclass holding a result and an already-completed future.
    # Keeps only a weak reference to the owning process group, and bumps that group's wait_count
    # and get_future_count counters when waited on or asked for its future.
    def __init__(self, result, pg):
        # [FUNCTION DOC] HELPER: Stores the result, creates a future completed with it, and keeps
        # a weak reference to the owning process group pg.
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.pg_ = weakref.ref(pg)

    def wait(self, timeout):
        # [FUNCTION DOC] HELPER: Increments the owning group's wait_count and returns true; the
        # timeout is ignored.
        self.pg_().wait_count += 1
        return True

    def get_future(self):
        # [FUNCTION DOC] HELPER: Increments the owning group's get_future_count and returns the
        # completed future.
        self.pg_().get_future_count += 1
        return self.future_


class LonelyRankProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] HELPER: Pure-Python ProcessGroup for rank 0 and world size 1. broadcast,
    # allgather and allreduce do not communicate; they return either a Work created from a
    # completed future (use_wrapper true) or a MyWork that is also kept in an internal list.
    # Exposes wait_count and get_future_count, incremented by MyWork, so tests can verify DDP
    # waited on its work.
    """
    This PG only supports world_size of 1
    """

    def __init__(self, rank, world, use_wrapper):
        # [FUNCTION DOC] HELPER: Initializes the base group and records rank, world, a use_wrapper
        # flag, zeroed wait and get-future counters and an empty list of created works. Raises
        # AssertionError unless rank is 0 and world is 1.
        super().__init__(rank, world)
        if rank != 0:
            raise AssertionError(f"Expected rank == 0, got {rank}")
        if world != 1:
            raise AssertionError(f"Expected world == 1, got {world}")

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self.use_wrapper = use_wrapper
        self._work = []

    def broadcast(self, tensor_list, opts):
        # [FUNCTION DOC] HELPER: Returns a Work holding the unchanged tensor list (no data
        # movement), made by the wrapper helper or as a tracked MyWork depending on use_wrapper.
        if self.use_wrapper:
            return create_work(tensor_list)
        res = MyWork(tensor_list, self)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        # [FUNCTION DOC] HELPER: Copies each input tensor into the matching tensor of the first
        # output list in place (single rank), then returns a Work holding the output tensors,
        # either wrapped or as a tracked MyWork.
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)
        if self.use_wrapper:
            return create_work(output_tensors)

        res = MyWork(output_tensors, self)
        self._work.append(res)

        return res

    def allreduce(self, tensors, opts):
        # [FUNCTION DOC] HELPER: Leaves the tensors unchanged (world size 1) and returns a Work
        # holding them, either wrapped or as a tracked MyWork.
        if self.use_wrapper:
            return create_work(tensors)
        res = MyWork(tensors, self)
        self._work.append(res)
        return res

    def getSize(self):
        # [FUNCTION DOC] HELPER: Returns the world size, which is always 1.
        return self._world

    def getBackendName(self):
        # [FUNCTION DOC] HELPER: Returns the backend name "lonely-pg".
        return "lonely-pg"

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a short string showing world size and rank.
        return f"PLG w:{self._world} r:{self._rank}"


class DummyAttrProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] HELPER: ProcessGroup subclass that overrides the rank (123), size (456), backend
    # name ("dummy-attr") and the group name and description setters and getters; names and
    # descriptions are stored with a "py:" prefix. Used to check that C++ accessors dispatch to
    # the Python overrides.
    def getRank(self):
        # [FUNCTION DOC] HELPER: Returns the fixed rank 123.
        return 123

    def getSize(self):
        # [FUNCTION DOC] HELPER: Returns the fixed size 456.
        return 456

    def getBackendName(self):
        # [FUNCTION DOC] HELPER: Returns the backend name "dummy-attr".
        return "dummy-attr"

    def setGroupName(self, name) -> None:
        # [FUNCTION DOC] HELPER: Stores the group name with the "py:" prefix.
        self._group_name = "py:" + name

    def getGroupName(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the stored prefixed group name.
        return self._group_name

    def setGroupDesc(self, group_desc) -> None:
        # [FUNCTION DOC] HELPER: Stores the group description with the "py:" prefix.
        self._group_desc = "py:" + group_desc

    def getGroupDesc(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the stored prefixed group description.
        return self._group_desc


class StoreProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] HELPER: ProcessGroup built with the store, rank and size constructor; reports
    # backend name "store-pg" so tests can show a Python subclass built this way is routed through
    # the Python trampoline.
    """
    A ProcessGroup constructed with the 3-arg (store, rank, size) constructor.
    Used to verify the store is accessible and that a Python subclass built
    this way is routed through the PyProcessGroup trampoline.
    """

    def __init__(self, store, rank, world):
        # [FUNCTION DOC] HELPER: Calls the base constructor with store, rank and world and
        # remembers rank and world.
        super().__init__(store, rank, world)
        self._rank = rank
        self._world = world

    def getBackendName(self):
        # [FUNCTION DOC] HELPER: Returns the backend name "store-pg".
        return "store-pg"


class CoalescingProcessGroup(test_c10d_common.DummyProcessGroup):
    # [CLASS DOC] HELPER: Dummy test group that reports coalescing support and counts
    # start_coalescing and end_coalescing calls; end_coalescing returns an empty-list Work.
    # Inherits send and recv behavior from the dummy group.
    """
    A ProcessGroup that advertises coalescing support and records coalescing
    calls, so both the coalescing manager and the batch_isend_irecv coalescing
    path can be exercised against a pure-Python backend. send/recv (which add 1
    and 2 respectively) are inherited from DummyProcessGroup.
    """

    def __init__(self, rank, world):
        # [FUNCTION DOC] HELPER: Calls the dummy group constructor and zeroes the start and end
        # coalescing counters.
        super().__init__(rank, world)
        self.start_coalescing_count = 0
        self.end_coalescing_count = 0

    @property
    def supports_coalescing(self):
        # [FUNCTION DOC] HELPER: Property returning true.
        return True

    def start_coalescing(self, device):
        # [FUNCTION DOC] HELPER: Counts a start-coalescing call; the device is ignored.
        self.start_coalescing_count += 1

    def end_coalescing(self, device):
        # [FUNCTION DOC] HELPER: Counts an end-coalescing call and returns a Work holding an empty
        # list; the device is ignored.
        self.end_coalescing_count += 1
        return create_work([])


class ReconfigurableProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] HELPER: Python ProcessGroup that advertises reconfigure support, returns a fixed
    # handle string, and stores the options passed to reconfigure (initially none) so tests can
    # inspect what the distributed helpers forwarded.
    """
    A Python ProcessGroup that records reconfigure calls. Used to verify the
    torch.distributed reconfigure helpers delegate to ProcessGroup.
    """

    def __init__(self, rank, world):
        # [FUNCTION DOC] HELPER: Calls the base constructor and initializes the recorded
        # reconfigure options to none.
        super().__init__(rank, world)
        self.reconfigure_opts = None

    @property
    def supports_reconfigure(self):
        # [FUNCTION DOC] HELPER: Property returning true.
        return True

    def get_reconfigure_handle(self):
        # [FUNCTION DOC] HELPER: Returns the fixed handle string for rank 0.
        return "handle-for-rank-0"

    def reconfigure(self, opts):
        # [FUNCTION DOC] HELPER: Records the options and returns a Work completed with none.
        self.reconfigure_opts = opts
        return create_work(None)


class WindowProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] HELPER: Python ProcessGroup that advertises window support and records the
    # tensor passed to new_window (sentinel string until called) and returns a fake window string.
    """
    A Python ProcessGroup that records new_window calls. Used to verify the
    torch.distributed window helpers delegate to ProcessGroup.
    """

    def __init__(self, rank, world):
        # [FUNCTION DOC] HELPER: Calls the base constructor and sets the recorded window tensor to
        # the sentinel "unset".
        super().__init__(rank, world)
        self.new_window_tensor = "unset"

    @property
    def supports_window(self):
        # [FUNCTION DOC] HELPER: Property returning true.
        return True

    def new_window(self, tensor=None):
        # [FUNCTION DOC] HELPER: Records the tensor argument (default none) and returns the string
        # "fake-window".
        self.new_window_tensor = tensor
        return "fake-window"


# We cannot use parametrize as some tests are defined on the base class and use _get_process_group
class AbstractDDPSingleRank(test_c10d_common.CommonDistributedDataParallelTest):
    # [CLASS DOC] HELPER: Base for single-rank DDP tests over LonelyRankProcessGroup, run in
    # thread mode. Subclasses define use_wrapper to choose whether the group returns wrapped Work
    # or the MyWork subclass. Reuses the shared DDP process-group test from the common test class.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the base setUp and starts the test threads.
        super().setUp()
        self._spawn_threads()

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property: the world size is 1.
        return 1

    def _get_process_group(self):
        # [FUNCTION DOC] HELPER: Builds a LonelyRankProcessGroup for this test's rank, world size
        # and use_wrapper setting.
        return LonelyRankProcessGroup(self.rank, self.world_size, self.use_wrapper)

    def test_ddp_invoke_work_object(self):
        # [FUNCTION DOC] Trains a small module under DDP over the lonely group for one backward
        # pass, then runs the plain module on the same input and checks the bias gradients are
        # equal. When not using the wrapper, also asserts the group's wait and get-future counters
        # were incremented.
        pg = self._get_process_group()

        torch.manual_seed(123)
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        wrapped_model = model
        input_tensor = torch.rand(2)
        model = DDP(model, process_group=pg)
        model(input_tensor).sum().backward()

        ddp_grad = wrapped_model[0].bias.grad.clone()

        wrapped_model.zero_grad()
        wrapped_model(input_tensor).sum().backward()
        self.assertEqual(wrapped_model[0].bias.grad, ddp_grad)
        if not self.use_wrapper:
            self.assertTrue(pg.wait_count > 0)
            self.assertTrue(pg.get_future_count > 0)

    def test_ddp_with_pypg(self):
        # [FUNCTION DOC] Runs the shared DDP process-group check on cpu with the lonely group.
        pg = self._get_process_group()

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None)

    def test_ddp_with_pypg_with_grad_views(self):
        # [FUNCTION DOC] Same as the shared DDP check on cpu with the lonely group, but with
        # gradients used as bucket views.
        pg = self._get_process_group()

        self._test_ddp_with_process_group(
            pg, [torch.device("cpu")], device_ids=None, gradient_as_bucket_view=True
        )

    def test_ddp_no_init_sync(self):
        # [FUNCTION DOC] Wraps a module in DDP with init_sync disabled and checks the group saw no
        # wait and no get-future calls during construction.
        pg = self._get_process_group()

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        model = DDP(model, process_group=pg, init_sync=False)

        self.assertEqual(pg.wait_count, 0)
        self.assertEqual(pg.get_future_count, 0)


class TestDDPWithWorkSubclass(AbstractDDPSingleRank, MultiThreadedTestCase):
    # [CLASS DOC] HELPER: Runs the single-rank DDP tests with the lonely group returning the
    # MyWork subclass.
    @property
    def use_wrapper(self):
        # [FUNCTION DOC] HELPER: Property returning false.
        return False


class TestDDPWithWorkWrapper(AbstractDDPSingleRank, MultiThreadedTestCase):
    # [CLASS DOC] HELPER: Runs the single-rank DDP tests with the lonely group returning Work
    # created from a future.
    @property
    def use_wrapper(self):
        # [FUNCTION DOC] HELPER: Property returning true.
        return True


class BlockWork(dist._Work):
    # [CLASS DOC] HELPER: Work whose future is never completed by itself; tests complete it
    # manually to release a stream blocked with block_current_stream.
    """
    Dummy work that is used to test blocking the current stream.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Calls the base constructor and creates a pending future.
        super().__init__()
        self.future_ = torch.futures.Future()

    def get_future(self):
        # [FUNCTION DOC] HELPER: Returns the pending future.
        return self.future_


class TestPyProcessGroup(TestCase):
    # [CLASS DOC] Single-process tests that Python ProcessGroup subclasses are called by the C++
    # layer: attribute overrides, store constructor, functional all-gather, direct reduce-scatter,
    # reduce, gather, scatter, all-to-all, recv-any-source, the coalescing manager, abort and
    # shutdown, reconfigure and window delegation, and Work stream blocking (cuda).
    def test_attr_overrides(self):
        # [FUNCTION DOC] Checks name, rank and size report the overridden values, and that setting
        # group name and description yield the prefixed values.
        pg = DummyAttrProcessGroup(0, 1)
        self.assertEqual(pg.name(), "dummy-attr")
        self.assertEqual(pg.rank(), 123)
        self.assertEqual(pg.size(), 456)

        pg._set_group_name("name")
        self.assertEqual(pg.group_name, "py:name")

        pg._set_group_desc("desc")
        self.assertEqual(pg.group_desc, "py:desc")

    def test_store_constructor(self):
        # [FUNCTION DOC] Builds a group with a hash store holding a key, and checks the group
        # store is the same object and returns the key. Also checks the Python backend name, rank
        # and size overrides are used.
        store = dist.HashStore()
        store.set("test_key", "test_value")

        pg = StoreProcessGroup(store, 0, 1)

        # The store passed to the 3-arg constructor is accessible via the PG
        # and is the same store object we passed in.
        group_store = pg.get_group_store()
        self.assertIs(group_store, store)
        self.assertEqual(group_store.get("test_key"), b"test_value")

        # A Python subclass built via the 3-arg constructor must be routed
        # through the PyProcessGroup trampoline so the getBackendName override
        # dispatches back into Python; a raw C++ ProcessGroup would return the
        # base backend name ("undefined") instead.
        self.assertEqual(pg.name(), "store-pg")
        self.assertEqual(pg.rank(), 0)
        self.assertEqual(pg.size(), 1)

    def test_all_gather_single(self):
        # [FUNCTION DOC] Registers a dummy group under a name, calls the functional
        # all-gather-into-tensor out op with size 1 and waits, then checks the dummy
        # all-gather-single override ran and the output equals the input. Unregisters the group in
        # a finally block.
        # all_gather_into_tensor_out calls group->all_gather_single(...) in C++,
        # which dispatches through the PyProcessGroup trampoline into the Python
        # DummyProcessGroup.all_gather_single override (it copies the input into
        # each chunk). Without the override the base impl needs a backend and
        # raises, so a correct output proves the override the PR adds was hit.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        _register_process_group("test_all_gather_single", pg)
        try:
            input = torch.ones(2)
            output = torch.empty(2)
            torch.ops._c10d_functional.all_gather_into_tensor_out(
                input, 1, "test_all_gather_single", out=output
            )
            torch.ops._c10d_functional.wait_tensor(output)
            self.assertIn("all_gather_single", pg.collectives_called)
            self.assertEqual(output, torch.ones(2))
        finally:
            _unregister_process_group("test_all_gather_single")

    def test_reduce_scatter_single(self):
        # [FUNCTION DOC] Calls reduce_scatter_single directly on a dummy group and checks it was
        # recorded and the output holds the input values.
        # No functional collective dispatches to reduce_scatter_single (they use
        # the coalesced variant), so this is a direct sanity check that the
        # override runs and forwards its buffers.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.ones(2)
        output = torch.zeros(2)
        pg.reduce_scatter_single(output, input).wait()
        self.assertIn("reduce_scatter_single", pg.collectives_called)
        self.assertEqual(output, torch.ones(2))

    # reduce / gather / scatter / alltoall share their Python name with the
    # bound method, so a plain instance call hits the Python override directly
    # (a sanity check that it runs and forwards its buffers; no functional
    # collective dispatches to these as ProcessGroup virtuals). recvAnysource is
    # bound under the non-shadowed name recv_anysource, so pg.recv_anysource()
    # routes through the C++ virtual into the trampoline. DummyProcessGroup
    # records each dispatched collective and applies a recognizable mutation
    # (reduce adds 3, recvAnysource adds 4, gather/scatter/alltoall copy).
    def test_reduce(self):
        # [FUNCTION DOC] Calls reduce on a dummy group, waits, checks the call was recorded and
        # the tensor was incremented by three.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        tensor = torch.zeros(4)
        pg.reduce([tensor]).wait()
        self.assertIn("reduce", pg.collectives_called)
        self.assertEqual(tensor, torch.zeros(4) + 3)

    def test_gather(self):
        # [FUNCTION DOC] Calls gather on a dummy group, waits, checks the call was recorded and
        # the output equals the input.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.arange(4, dtype=torch.float32)
        output = torch.zeros(4)
        pg.gather([[output]], [input]).wait()
        self.assertIn("gather", pg.collectives_called)
        self.assertEqual(output, input)

    def test_scatter(self):
        # [FUNCTION DOC] Calls scatter on a dummy group, waits, checks the call was recorded and
        # the output equals the input.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.arange(4, dtype=torch.float32)
        output = torch.zeros(4)
        pg.scatter([output], [[input]]).wait()
        self.assertIn("scatter", pg.collectives_called)
        self.assertEqual(output, input)

    def test_alltoall(self):
        # [FUNCTION DOC] Calls alltoall on a dummy group, waits, checks the call was recorded and
        # the output equals the input.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.arange(4, dtype=torch.float32)
        output = torch.zeros(4)
        pg.alltoall([output], [input]).wait()
        self.assertIn("alltoall", pg.collectives_called)
        self.assertEqual(output, input)

    def test_recv_anysource(self):
        # [FUNCTION DOC] Calls recv_anysource on a dummy group with source tag 7, waits, checks
        # the recvAnysource call was recorded and the tensor was incremented by four.
        # recv_anysource is bound to the recvAnysource virtual under a different
        # name, so this routes through the C++ trampoline into the override.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        tensor = torch.zeros(4)
        pg.recv_anysource([tensor], 7).wait()
        self.assertIn("recvAnysource", pg.collectives_called)
        self.assertEqual(tensor, torch.zeros(4) + 4)

    def test_coalescing_manager(self):
        # [FUNCTION DOC] Enters the coalescing manager (async) on the coalescing group with an
        # empty body, waits, and checks one start call, one end call and one collected work.
        # The coalescing manager calls _start_coalescing / _end_coalescing, which
        # route through the C++ virtual into the PyProcessGroup trampoline and
        # dispatch to the start_coalescing / end_coalescing overrides; the work
        # returned by end_coalescing is collected.
        pg = CoalescingProcessGroup(0, 1)
        device = torch.device("cpu")
        with _coalescing_manager(pg, device, async_ops=True) as cm:
            pass
        cm.wait()

        self.assertEqual(pg.start_coalescing_count, 1)
        self.assertEqual(pg.end_coalescing_count, 1)
        self.assertEqual(len(cm.works), 1)

    def test_abort_shutdown(self) -> None:
        # [FUNCTION DOC] Calls abort and shutdown on a group that does not override them; only
        # checks they do not raise.
        # verify this are noops
        pg = DummyAttrProcessGroup(0, 1)
        pg.abort()
        pg.shutdown()

    def test_reconfigure_delegation(self) -> None:
        # [FUNCTION DOC] Checks the distributed helpers report reconfigure support and the handle
        # for a Python group, and that the reconfigure helper returns a work and forwards an
        # options object carrying the uuid, handles, timeout and hints unchanged.
        pg = ReconfigurableProcessGroup(0, 1)

        self.assertTrue(dist._supports_reconfigure(group=pg))
        self.assertEqual(dist._get_reconfigure_handle(group=pg), "handle-for-rank-0")

        timeout = timedelta(seconds=30)
        work = dist._reconfigure(
            uuid=7,
            handles=["a", "b"],
            group=pg,
            timeout=timeout,
            hints={"k": "v"},
        )
        self.assertIsNotNone(work)

        # The helper builds a ReconfigureOptions and forwards it unchanged.
        opts = pg.reconfigure_opts
        self.assertIsNotNone(opts)
        self.assertEqual(opts.uuid, 7)
        self.assertEqual(opts.handles, ["a", "b"])
        self.assertEqual(opts.timeout, timeout)
        self.assertEqual(opts.hints, {"k": "v"})

    def test_reconfigure_rejects_multiple_backends(self) -> None:
        # [FUNCTION DOC] Registers cpu and cuda backends on one group, then checks that reading
        # supports_reconfigure, getting the reconfigure handle and calling reconfigure each raise
        # RuntimeError mentioning multiple backends.
        pg = dist.ProcessGroup(0, 1)
        pg._register_backend(torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO)
        pg._register_backend(torch.device("cuda"), dist.ProcessGroup.BackendType.NCCL)

        msg = "multiple backends"
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.supports_reconfigure
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.reconfigure(ReconfigureOptions())

    def test_window_delegation(self) -> None:
        # [FUNCTION DOC] Checks the distributed helpers report window support, that creating a
        # window without a tensor forwards none, and that a given tensor is forwarded as the same
        # object; both return the fake window string.
        pg = WindowProcessGroup(0, 1)

        self.assertTrue(dist._supports_window(group=pg))

        # With no tensor, new_window is called with tensor=None.
        self.assertEqual(dist._new_window(group=pg), "fake-window")
        self.assertIsNone(pg.new_window_tensor)

        # A tensor is forwarded through to ProcessGroup.new_window.
        t = torch.zeros(4)
        self.assertEqual(dist._new_window(t, group=pg), "fake-window")
        self.assertIs(pg.new_window_tensor, t)

    @unittest.skipIf(not TEST_CUDA, "no cuda/xpu")
    def test_block_current_stream(self) -> None:
        # [FUNCTION DOC] Cuda only. On a side stream, records an event that resolves, then blocks
        # the current stream on a Work with block_current_stream and checks a new event stays
        # unresolved until the Work future is completed, after which synchronizing resolves it.
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        with stream:
            # nothing in queue so instantly resolves
            event1 = torch.cuda.Event()
            event1.record()
            time.sleep(0.1)
            self.assertTrue(event1.query())

            work = BlockWork()
            work.block_current_stream()

            # stream is blocked so doesn't resolve
            event = torch.cuda.Event()
            event.record()
            time.sleep(0.1)
            self.assertFalse(event.query())

            # resolve the work
            work.get_future().set_result(None)

            stream.synchronize()
            self.assertTrue(event.query())

    @unittest.skipIf(not TEST_CUDA, "no cuda/xpu")
    def test_block_current_stream_use_after_free(self) -> None:
        # [FUNCTION DOC] Cuda only. Blocks a side stream on two Works, completes and deletes the
        # second while the first still blocks, checks a later event is still unresolved, then
        # completes the first and checks it resolves (guards against the CPU control tensor being
        # freed too early).
        """
        This tests that the CPU control tensor is not freed before the CUDA kernel executes.
        """
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with stream:
            a = BlockWork()
            a.block_current_stream()

            b = BlockWork()
            b.block_current_stream()

            # unblock b first though a is still blocking
            b.get_future().set_result(None)
            # delete b
            del b

            # a is still blocking so this doesn't resolve
            event = torch.cuda.Event()
            event.record()
            time.sleep(0.1)
            self.assertFalse(event.query())

            # unblock a
            a.get_future().set_result(None)

            stream.synchronize()
            self.assertTrue(event.query())


class TestBatchSendRecv(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test case checking batch_isend_irecv over registered pure-Python
    # backends, with and without coalescing support. Uses the default world size of the
    # multi-process base class.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Runs the base tearDown and removes the file-store file, ignoring OSError.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @staticmethod
    def create_dummy(store, group_rank, group_size, timeout):
        # [FUNCTION DOC] HELPER: Backend creator function with the store, rank, size, timeout
        # signature; returns a dummy Python group and ignores the store and timeout.
        return test_c10d_common.DummyProcessGroup(group_rank, group_size)

    @staticmethod
    def create_coalescing(store, group_rank, group_size, timeout):
        # [FUNCTION DOC] HELPER: Backend creator function; returns a coalescing-capable Python
        # group and ignores the store and timeout.
        return CoalescingProcessGroup(group_rank, group_size)

    def test_batch_isend_irecv(self):
        # [FUNCTION DOC] Registers a dummy backend, sets the MASTER_ADDR and MASTER_PORT
        # environment variables (side effect), inits the default group, and runs batch_isend_irecv
        # with one send and one receive to the next rank. Waits on each request, then checks the
        # send tensor was incremented by one and the receive tensor by two (the per-op fallback
        # was used). Ends with a barrier and process group destruction.
        # batch_isend_irecv over a Python ProcessGroup that does not advertise
        # coalescing falls back to per-op send/recv. DummyProcessGroup.send adds
        # 1 and recv adds 2 to verify the ops are dispatched into Python.
        dist.Backend.register_backend("dummy", TestBatchSendRecv.create_dummy)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        peer = (self.rank + 1) % self.world_size
        send_tensor = torch.zeros(2, 2)
        recv_tensor = torch.zeros(2, 2)
        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, peer),
                dist.P2POp(dist.irecv, recv_tensor, peer),
            ]
        )
        for req in reqs:
            req.wait()

        self.assertEqual(send_tensor, torch.zeros(2, 2) + 1)
        self.assertEqual(recv_tensor, torch.zeros(2, 2) + 2)

        dist.barrier()
        dist.destroy_process_group()

    def test_batch_isend_irecv_coalescing(self):
        # [FUNCTION DOC] Registers a coalescing backend, sets MASTER_ADDR and MASTER_PORT (side
        # effect), inits the default group, and runs batch_isend_irecv with a send and a receive.
        # Checks one start and one end coalescing call, a single returned work, and the same
        # tensor increments as the dummy test. Ends with a barrier and process group destruction.
        # A Python ProcessGroup that advertises supports_coalescing routes
        # batch_isend_irecv through the coalescing manager: start/endCoalescing
        # wrap the sends/recvs and the single end-coalescing work is returned.
        dist.Backend.register_backend("coalescing", TestBatchSendRecv.create_coalescing)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "coalescing", rank=self.rank, world_size=self.world_size
        )

        pg = _get_default_group()
        peer = (self.rank + 1) % self.world_size
        send_tensor = torch.zeros(2, 2)
        recv_tensor = torch.zeros(2, 2)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, peer),
                dist.P2POp(dist.irecv, recv_tensor, peer),
            ]
        )
        for work in works:
            work.wait()

        self.assertEqual(pg.start_coalescing_count, 1)
        self.assertEqual(pg.end_coalescing_count, 1)
        self.assertEqual(len(works), 1)
        self.assertEqual(send_tensor, torch.zeros(2, 2) + 1)
        self.assertEqual(recv_tensor, torch.zeros(2, 2) + 2)

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
