# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests that a pure-Python subclass of the c10d Backend (and a Work subclass)
# can plug into ProcessGroup and into init_process_group via a registered custom backend.
# Contains RecordingWork and RecordingBackend (a mock that logs calls and mutates tensors
# with distinct constants), the create_process_group helper, and two test classes: a
# single-process TestPyBackend and a two-process TestPyBackendProcessGroup. Import-time
# side effects: none beyond imports.
import os
import weakref
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    Backend as C10DBackend,
    ErrorType,
    ReconfigureOptions,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TestCase


class RecordingWork(dist._Work):
    # [CLASS DOC] HELPER: Python Work implementation for the mock backend. Holds a result
    # value, an already-completed Future carrying that result, and a weak reference to the
    # creating backend so that wait and get_future calls can be counted on it.
    def __init__(self, result, backend):
        # [FUNCTION DOC] HELPER: Stores the result, creates a future resolved immediately
        # with the result, and keeps only a weak reference to the backend (avoids a cycle).
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.backend_ = weakref.ref(backend)

    def wait(self, timeout):
        # [FUNCTION DOC] HELPER: Increments the backend's wait counter (the timeout is
        # ignored) and returns True. Assumes the backend is still alive.
        self.backend_().wait_count += 1
        return True

    def get_future(self):
        # [FUNCTION DOC] HELPER: Increments the backend's get_future counter and returns
        # the pre-completed future holding the work result.
        self.backend_().get_future_count += 1
        return self.future_


class RecordingBackend(C10DBackend):
    # [CLASS DOC] HELPER: Mock c10d Backend written in Python. Every collective appends a
    # tuple (operation name, arguments) to the calls list, performs a simple tensor
    # mutation (add a per-op constant in place, or copy inputs to outputs), and returns a
    # RecordingWork kept in an internal list. Also records coalescing, sequence number,
    # hook, timing, eager-connect, tensor-alloc, suspend/resume, abort and shutdown calls
    # in plain attributes, and advertises all optional capabilities as supported.
    def __init__(self, rank, world, name="python-backend"):
        # [FUNCTION DOC] HELPER: Initializes the base Backend with rank and world size,
        # stores the name and an Options object (backend name, 5 second timeout), and
        # zeroes or clears all recording attributes (counters, calls list, flags).
        super().__init__(rank, world)
        self._name = name
        self._options = C10DBackend.Options(name, timeout=timedelta(seconds=5))
        self.wait_count = 0
        self.get_future_count = 0
        self.calls = []
        self._work = []
        self.reconfigure_opts = None
        self.start_coalescing_count = 0
        self.end_coalescing_count = 0
        self.sequence_number = 0
        self.registered_hook = None
        self.wait_for_pending_works_count = 0
        self.collectives_timing_enabled = False
        self.eager_device = None
        self.tensor_alloc_device_idx = None
        self.allocate_args = None
        self.aborted = False
        self.shut_down = False
        self.suspended = False
        self.resumed = False

    def _new_work(self, result=None):
        # [FUNCTION DOC] HELPER: Creates a RecordingWork carrying result, appends it to the
        # internal work list (keeping it alive) and returns it.
        work = RecordingWork(result, self)
        self._work.append(work)
        return work

    @property
    def supports_splitting(self):
        # [FUNCTION DOC] HELPER: Capability property; always True.
        return True

    @property
    def supports_coalescing(self):
        # [FUNCTION DOC] HELPER: Capability property; always True.
        return True

    @property
    def supports_time_estimate(self):
        # [FUNCTION DOC] HELPER: Capability property; always True.
        return True

    @property
    def supports_shrinking(self):
        # [FUNCTION DOC] HELPER: Capability property; always True.
        return True

    @property
    def supports_reconfigure(self):
        # [FUNCTION DOC] HELPER: Capability property; always True.
        return True

    @property
    def supports_window(self):
        # [FUNCTION DOC] HELPER: Capability property; always True.
        return True

    @property
    def options(self):
        # [FUNCTION DOC] HELPER: Property returning the Options object created at init.
        return self._options

    def getBackendName(self):
        # [FUNCTION DOC] HELPER: Returns the backend name given at construction.
        return self._name

    def set_timeout(self, timeout):
        # [FUNCTION DOC] HELPER: Records a set_timeout call with the timeout; does not
        # change the stored options.
        self.calls.append(("set_timeout", timeout))

    def shrink(self, ranks_to_exclude, shrink_flags=0, opts_override=None):
        # [FUNCTION DOC] HELPER: Records the excluded ranks, flags and override options, and
        # returns a new RecordingBackend with the same rank and size named
        # shrunk-python-backend (no real shrinking is done).
        self.calls.append(("shrink", ranks_to_exclude, shrink_flags, opts_override))
        return RecordingBackend(self.rank(), self.size(), "shrunk-python-backend")

    def get_reconfigure_handle(self):
        # [FUNCTION DOC] HELPER: Returns a fixed string handle for this backend.
        return "handle-for-python-backend"

    def reconfigure(self, opts):
        # [FUNCTION DOC] HELPER: Stores the options object and returns a work with no result.
        self.reconfigure_opts = opts
        return self._new_work(None)

    def start_coalescing(self):
        # [FUNCTION DOC] HELPER: Increments the start-coalescing counter.
        self.start_coalescing_count += 1

    def end_coalescing(self):
        # [FUNCTION DOC] HELPER: Increments the end-coalescing counter and returns a work
        # whose result is an empty list.
        self.end_coalescing_count += 1
        return self._new_work([])

    def broadcast(self, tensor_list, opts):
        # [FUNCTION DOC] HELPER: Records the call, adds 1 in place to each tensor, and
        # returns a work holding the tensor list.
        self.calls.append(("broadcast", opts))
        for tensor in tensor_list:
            tensor.add_(1)
        return self._new_work(tensor_list)

    def allreduce(self, tensor_list, opts):
        # [FUNCTION DOC] HELPER: Records the call, adds 2 in place to each tensor, and
        # returns a work holding the tensor list.
        self.calls.append(("allreduce", opts))
        for tensor in tensor_list:
            tensor.add_(2)
        return self._new_work(tensor_list)

    def allreduce_sparse(self, tensor_list, opts):
        # [FUNCTION DOC] HELPER: Records the call only (tensors untouched) and returns a work
        # holding the tensor list.
        self.calls.append(("allreduce_sparse", opts))
        return self._new_work(tensor_list)

    def allreduce_coalesced(self, tensor_list, opts):
        # [FUNCTION DOC] HELPER: Records the call, adds 3 in place to each tensor, and
        # returns a work holding the tensor list.
        self.calls.append(("allreduce_coalesced", opts))
        for tensor in tensor_list:
            tensor.add_(3)
        return self._new_work(tensor_list)

    def reduce(self, tensor_list, opts):
        # [FUNCTION DOC] HELPER: Records the call, adds 4 in place to each tensor, and
        # returns a work holding the tensor list.
        self.calls.append(("reduce", opts))
        for tensor in tensor_list:
            tensor.add_(4)
        return self._new_work(tensor_list)

    def allgather(self, output_tensors, input_tensors, opts):
        # [FUNCTION DOC] HELPER: Records the call; pairs each list of output tensors with an
        # input tensor and copies that input into every tensor of the list. Returns a work
        # holding the output lists.
        self.calls.append(("allgather", opts))
        for output_tensor_list, input_tensor in zip(output_tensors, input_tensors):
            for output_tensor in output_tensor_list:
                output_tensor.copy_(input_tensor)
        return self._new_work(output_tensors)

    def all_gather_single(self, output_tensor, input_tensor, opts):
        # [FUNCTION DOC] HELPER: Records the call, copies the input into the output tensor
        # (shapes must be compatible) and returns a work holding the output tensor.
        self.calls.append(("all_gather_single", opts))
        output_tensor.copy_(input_tensor)
        return self._new_work(output_tensor)

    def all_gather_single_coalesced(self, outputs, inputs, opts):
        # [FUNCTION DOC] HELPER: Records the call and copies each input into the matching
        # output; returns a work holding the outputs.
        self.calls.append(("all_gather_single_coalesced", opts))
        for output, input in zip(outputs, inputs):
            output.copy_(input)
        return self._new_work(outputs)

    def allgather_coalesced(self, output_lists, input_list, opts):
        # [FUNCTION DOC] HELPER: Records the call; for each pair of output list and input
        # tensor copies the input into every tensor of the list. Returns a work holding the
        # output lists.
        self.calls.append(("allgather_coalesced", opts))
        for output_list, input in zip(output_lists, input_list):
            for output in output_list:
                output.copy_(input)
        return self._new_work(output_lists)

    def gather(self, output_tensors, input_tensors, opts):
        # [FUNCTION DOC] HELPER: Records the call; if output tensors were supplied, copies
        # each input tensor into the matching tensor of the first output list. Returns a
        # work holding the output tensors.
        self.calls.append(("gather", opts))
        if output_tensors:
            for output, input in zip(output_tensors[0], input_tensors):
                output.copy_(input)
        return self._new_work(output_tensors)

    def scatter(self, output_tensors, input_tensors, opts):
        # [FUNCTION DOC] HELPER: Records the call; if input tensors were supplied, copies
        # each tensor of the first input list into the matching output tensor. Returns a
        # work holding the output tensors.
        self.calls.append(("scatter", opts))
        if input_tensors:
            for output, input in zip(output_tensors, input_tensors[0]):
                output.copy_(input)
        return self._new_work(output_tensors)

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        # [FUNCTION DOC] HELPER: Records the call; for each output and its input list copies
        # the list element indexed by this backend's rank into the output (no reduction is
        # performed). Returns a work holding the output tensors.
        self.calls.append(("reduce_scatter", opts))
        for output, input_list in zip(output_tensors, input_tensors):
            output.copy_(input_list[self.rank()])
        return self._new_work(output_tensors)

    def reduce_scatter_single(self, output_tensor, input_tensor, opts):
        # [FUNCTION DOC] HELPER: Records the call and copies the leading slice of the input
        # (first dimension, output element count long) into the output; no reduction.
        # Returns a work holding the output tensor.
        self.calls.append(("reduce_scatter_single", opts))
        output_tensor.copy_(input_tensor.narrow(0, 0, output_tensor.numel()))
        return self._new_work(output_tensor)

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts):
        # [FUNCTION DOC] HELPER: Records the call and, for each output/input pair, copies the
        # leading slice of the input (output element count long) into the output; no
        # reduction. Returns a work holding the outputs.
        self.calls.append(("reduce_scatter_single_coalesced", opts))
        for output, input in zip(outputs, inputs):
            output.copy_(input.narrow(0, 0, output.numel()))
        return self._new_work(outputs)

    def all_to_all_single(
        self, output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
    ):
        # [FUNCTION DOC] HELPER: Records the call with both split size lists and options,
        # copies the whole input into the output (split sizes are ignored) and returns a
        # work holding the output tensor.
        self.calls.append(
            ("all_to_all_single", output_split_sizes, input_split_sizes, opts)
        )
        output_tensor.copy_(input_tensor)
        return self._new_work(output_tensor)

    def alltoall(self, output_tensors, input_tensors, opts):
        # [FUNCTION DOC] HELPER: Records the call and copies each input tensor into the
        # output tensor at the same position; returns a work holding the outputs.
        self.calls.append(("alltoall", opts))
        for output, input in zip(output_tensors, input_tensors):
            output.copy_(input)
        return self._new_work(output_tensors)

    def send(self, tensor_list, dst, tag=0):
        # [FUNCTION DOC] HELPER: Records destination and tag, adds 1 in place to each tensor
        # (nothing is transmitted) and returns a work holding the tensor list.
        self.calls.append(("send", dst, tag))
        for tensor in tensor_list:
            tensor.add_(1)
        return self._new_work(tensor_list)

    def recv(self, tensor_list, src, tag=0):
        # [FUNCTION DOC] HELPER: Records source and tag, adds 2 in place to each tensor and
        # returns a work holding the tensor list.
        self.calls.append(("recv", src, tag))
        for tensor in tensor_list:
            tensor.add_(2)
        return self._new_work(tensor_list)

    def recv_anysource(self, tensor_list, tag=0):
        # [FUNCTION DOC] HELPER: Records the tag, adds 5 in place to each tensor and returns
        # a work holding the tensor list.
        self.calls.append(("recv_anysource", tag))
        for tensor in tensor_list:
            tensor.add_(5)
        return self._new_work(tensor_list)

    def barrier(self, opts):
        # [FUNCTION DOC] HELPER: Records the call and returns a work with no result.
        self.calls.append(("barrier", opts))
        return self._new_work(None)

    def monitored_barrier(self, opts, wait_all_ranks=False):
        # [FUNCTION DOC] HELPER: Records the options and wait_all_ranks flag; returns nothing.
        self.calls.append(("monitored_barrier", opts, wait_all_ranks))

    def _set_sequence_number_for_group(self):
        # [FUNCTION DOC] HELPER: Sets the stored sequence number to the fixed value 123.
        self.sequence_number = 123

    def _get_sequence_number_for_group(self):
        # [FUNCTION DOC] HELPER: Returns the stored sequence number (initially 0).
        return self.sequence_number

    def _register_on_completion_hook(self, hook):
        # [FUNCTION DOC] HELPER: Stores the hook; it is never invoked by this backend.
        self.registered_hook = hook

    def _wait_for_pending_works(self):
        # [FUNCTION DOC] HELPER: Increments the pending-works wait counter.
        self.wait_for_pending_works_count += 1

    def _enable_collectives_timing(self):
        # [FUNCTION DOC] HELPER: Sets the collectives-timing flag to True.
        self.collectives_timing_enabled = True

    def split(self, store, ranks, opts):
        # [FUNCTION DOC] HELPER: Records store, ranks and options; returns a new backend
        # named split-backend whose rank is this rank's position in the ranks list (raises
        # an error if absent) and whose size is the number of ranks.
        self.calls.append(("split", store, ranks, opts))
        return RecordingBackend(ranks.index(self.rank()), len(ranks), "split-backend")

    def merge(self, store, opts, rank, size):
        # [FUNCTION DOC] HELPER: Records the arguments and returns a new backend named
        # merged-backend with the given rank and size.
        self.calls.append(("merge", store, opts, rank, size))
        return RecordingBackend(rank, size, "merged-backend")

    def eager_connect_single_device(self, device):
        # [FUNCTION DOC] HELPER: Stores the device it was asked to connect to.
        self.eager_device = device

    def get_error(self):
        # [FUNCTION DOC] HELPER: Always reports the SUCCESS error type.
        return ErrorType.SUCCESS

    def supports_tensor_alloc(self, device_idx):
        # [FUNCTION DOC] HELPER: Stores the queried device index and returns True.
        self.tensor_alloc_device_idx = device_idx
        return True

    def allocate_tensor(self, size, dtype, device):
        # [FUNCTION DOC] HELPER: Stores the (size, dtype, device) arguments and returns a
        # new uninitialized tensor of that size, dtype and device.
        self.allocate_args = (size, dtype, device)
        return torch.empty(size, dtype=dtype, device=device)

    def suspend(self):
        # [FUNCTION DOC] HELPER: Sets the suspended flag.
        self.suspended = True

    def resume(self):
        # [FUNCTION DOC] HELPER: Sets the resumed flag.
        self.resumed = True

    def memory_stats(self):
        # [FUNCTION DOC] HELPER: Returns a fixed mapping with allocated set to 7.
        return {"allocated": 7}

    def abort(self):
        # [FUNCTION DOC] HELPER: Sets the aborted flag.
        self.aborted = True

    def shutdown(self):
        # [FUNCTION DOC] HELPER: Sets the shut_down flag.
        self.shut_down = True


def create_process_group(backend):
    # [FUNCTION DOC] HELPER: Builds a ProcessGroup over an in-memory HashStore using the
    # backend's rank and size, registers the backend as the custom backend for cpu, makes
    # it the default backend, sets group name pybackend-test and a description, and
    # returns the group.
    group = dist.ProcessGroup(dist.HashStore(), backend.rank(), backend.size())
    group._register_backend(
        torch.device("cpu"), dist.ProcessGroup.BackendType.CUSTOM, backend
    )
    group._set_default_backend(dist.ProcessGroup.BackendType.CUSTOM)
    group._set_group_name("pybackend-test")
    group._set_group_desc("python backend test")
    return group


class TestPyBackend(TestCase):
    # [CLASS DOC] Single-process tests that wrap a RecordingBackend in a ProcessGroup and
    # verify Python overrides of properties, collectives and backend-only methods are
    # dispatched and their results/side effects are visible.
    def test_attr_overrides(self) -> None:
        # [FUNCTION DOC] With a one-rank group, checks rank and size, that all capability
        # properties are true on the backend and group, that the backend options expose the
        # backend name, and that group name, description, bound device id and the symmetric
        # memory rendezvous flag can be set and read back.
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        self.assertEqual(group.rank(), 0)
        self.assertEqual(group.size(), 1)

        for attr in (
            "supports_splitting",
            "supports_coalescing",
            "supports_time_estimate",
            "supports_shrinking",
            "supports_reconfigure",
            "supports_window",
        ):
            self.assertTrue(getattr(backend, attr))

        self.assertTrue(group.supports_reconfigure)
        self.assertTrue(group.supports_window)
        self.assertEqual(
            group._get_backend(torch.device("cpu")).options.backend, "python-backend"
        )

        group._set_group_name("name")
        group._set_group_desc("desc")
        self.assertEqual(group.group_name, "name")
        self.assertEqual(group.group_desc, "desc")

        group.bound_device_id = torch.device("cpu:0")
        self.assertEqual(group.bound_device_id, torch.device("cpu:0"))

        group.use_pg_for_symm_mem_rendezvous = True
        self.assertTrue(group.use_pg_for_symm_mem_rendezvous)

    def test_collective_overrides(self) -> None:
        # [FUNCTION DOC] Calls allreduce, broadcast, all_gather_single (plain and coalesced),
        # reduce_scatter_single (plain and coalesced), all_to_all_single and recv_anysource
        # through the group, waiting each work, and asserts the mock's tensor effects
        # (+2, +1, copies, +5) and the recorded call names; finally the wait counter is
        # positive.
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        allreduce_tensor = torch.zeros(2)
        work = group.allreduce([allreduce_tensor])
        self.assertTrue(work.wait())
        self.assertEqual(allreduce_tensor, torch.full((2,), 2.0))

        broadcast_tensor = torch.zeros(2)
        group.broadcast([broadcast_tensor]).wait()
        self.assertEqual(broadcast_tensor, torch.ones(2))

        output = torch.zeros(2)
        input = torch.ones(2)
        group.all_gather_single(output, input).wait()
        self.assertEqual(output, input)
        self.assertEqual(backend.calls[-1][0], "all_gather_single")

        outputs = [torch.zeros(2), torch.zeros(2)]
        inputs = [torch.ones(2), torch.full((2,), 3.0)]
        group.all_gather_single_coalesced(outputs, inputs).wait()
        self.assertEqual(outputs, inputs)
        self.assertEqual(backend.calls[-1][0], "all_gather_single_coalesced")

        rs_output = torch.zeros(2)
        rs_input = torch.arange(2.0)
        group.reduce_scatter_single(rs_output, rs_input).wait()
        self.assertEqual(rs_output, rs_input)
        self.assertEqual(backend.calls[-1][0], "reduce_scatter_single")

        rs_outputs = [torch.zeros(2), torch.zeros(2)]
        rs_inputs = [torch.arange(4.0), torch.arange(2.0, 6.0)]
        group.reduce_scatter_single_coalesced(rs_outputs, rs_inputs).wait()
        self.assertEqual(rs_outputs, [torch.arange(2.0), torch.arange(2.0, 4.0)])
        self.assertEqual(backend.calls[-1][0], "reduce_scatter_single_coalesced")

        a2a_output = torch.zeros(2)
        a2a_input = torch.ones(2)
        group.all_to_all_single(a2a_output, a2a_input, [], []).wait()
        self.assertEqual(a2a_output, a2a_input)
        self.assertEqual(backend.calls[-1][0], "all_to_all_single")

        recv_any = torch.zeros(2)
        group.recv_anysource([recv_any], 0).wait()
        self.assertEqual(recv_any, torch.full((2,), 5.0))

        self.assertGreater(backend.wait_count, 0)

    def test_collective_returns_none(self) -> None:
        # [FUNCTION DOC] Replaces allreduce, broadcast, barrier, recv_anysource and
        # end_coalescing on the backend with lambdas that mutate tensors but return None,
        # then asserts the group-level calls return None instead of failing, while the
        # tensor side effects still occur.
        backend = RecordingBackend(0, 1)
        backend.allreduce = lambda tensors, opts: [t.add_(2) for t in tensors] and None
        backend.broadcast = lambda tensors, opts: [t.add_(1) for t in tensors] and None
        backend.barrier = lambda opts: None
        group = create_process_group(backend)

        t = torch.zeros(2)
        self.assertIsNone(group.allreduce([t]))
        self.assertEqual(t, torch.full((2,), 2.0))

        t = torch.zeros(2)
        self.assertIsNone(group.broadcast([t]))
        self.assertEqual(t, torch.ones(2))

        self.assertIsNone(group.barrier())

        backend.recv_anysource = (
            lambda tensors, tag: [t.add_(5) for t in tensors] and None
        )
        t = torch.zeros(2)
        self.assertIsNone(group.recv_anysource([t], 0))
        self.assertEqual(t, torch.full((2,), 5.0))

        backend.end_coalescing = lambda: None
        group._start_coalescing(torch.device("cpu"))
        self.assertIsNone(group._end_coalescing(torch.device("cpu")))

    def test_backend_only_overrides(self) -> None:
        # [FUNCTION DOC] Exercises group-level methods that delegate to Python overrides:
        # set_timeout, shrink, reconfigure handle and reconfigure, start/end coalescing,
        # sequence numbers, monitored_barrier, completion hook registration, pending-works
        # wait, collectives timing, split_group, merge_remote_group, then direct backend
        # calls (eager connect, error, tensor alloc support and allocation, suspend,
        # resume, memory stats) and finally group abort and shutdown, asserting the
        # recorded state for each.
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        timeout = timedelta(seconds=3)
        group.set_timeout(timeout)
        self.assertEqual(backend.calls[-1], ("set_timeout", timeout))

        shrunk = group._get_backend(torch.device("cpu")).shrink([1], 7, None)
        self.assertEqual(shrunk.name(), "shrunk-python-backend")
        self.assertEqual(backend.calls[-1], ("shrink", [1], 7, None))

        self.assertEqual(group.get_reconfigure_handle(), "handle-for-python-backend")
        opts = ReconfigureOptions()
        group.reconfigure(opts).wait()
        self.assertIs(backend.reconfigure_opts, opts)

        group._start_coalescing(torch.device("cpu"))
        group._end_coalescing(torch.device("cpu")).wait()
        self.assertEqual(backend.start_coalescing_count, 1)
        self.assertEqual(backend.end_coalescing_count, 1)

        backend._set_sequence_number_for_group()
        self.assertEqual(backend._get_sequence_number_for_group(), 123)

        group.monitored_barrier(timeout, True)
        self.assertEqual(backend.calls[-1][0], "monitored_barrier")

        group._register_on_completion_hook(lambda work_info: None)
        self.assertTrue(group._has_hooks())
        self.assertIsNotNone(backend.registered_hook)

        group._wait_for_pending_works()
        group._enable_collectives_timing()
        self.assertEqual(backend.wait_for_pending_works_count, 1)
        self.assertTrue(backend.collectives_timing_enabled)

        store = dist.HashStore()
        split_opts = C10DBackend.Options("split")
        split = group.split_group([0], opts=split_opts)
        self.assertEqual(
            split._get_backend(torch.device("cpu")).name(), "split-backend"
        )

        merge = group.merge_remote_group(store, 1)
        self.assertEqual(
            merge._get_backend(torch.device("cpu")).name(), "merged-backend"
        )

        device = torch.device("cpu:0")
        backend.eager_connect_single_device(device)
        self.assertEqual(backend.eager_device, device)

        self.assertEqual(backend.get_error(), ErrorType.SUCCESS)
        self.assertTrue(backend.supports_tensor_alloc(device))
        self.assertEqual(backend.tensor_alloc_device_idx, device)

        allocated = backend.allocate_tensor(
            4,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(allocated.shape, (4,))
        self.assertEqual(allocated.dtype, torch.float64)
        self.assertEqual(backend.allocate_args, (4, torch.float64, torch.device("cpu")))

        backend.suspend()
        backend.resume()
        self.assertTrue(backend.suspended)
        self.assertTrue(backend.resumed)
        self.assertEqual(backend.memory_stats(), {"allocated": 7})

        group.abort()
        group.shutdown()
        self.assertTrue(backend.aborted)
        self.assertTrue(backend.shut_down)


class TestPyBackendProcessGroup(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test that registers RecordingBackend under the name
    # pybackend and drives it through init_process_group and the public dist API.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown, then removes the store file,
        # ignoring OS errors.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @staticmethod
    def create_backend(store, group_rank, group_size, timeout):
        # [FUNCTION DOC] HELPER: Backend factory registered with the dist Backend registry;
        # ignores store and timeout and returns a RecordingBackend with the given rank and
        # size named registered-python-backend.
        return RecordingBackend(group_rank, group_size, "registered-python-backend")

    def test_init_process_group_with_pybackend(self):
        # [FUNCTION DOC] Registers the pybackend name for cpu, sets MASTER_ADDR and
        # MASTER_PORT (6789) in the environment, and initializes the default group. Checks
        # the group's cpu backend is the Python backend (not a ProcessGroup), then runs
        # blocking, async, future-based and multiple concurrent all_reduce calls (each
        # result is the input plus 2, or 3 from ones) and a batch_isend_irecv to a peer,
        # asserting one start/end coalescing and send/recv tensor effects. Destroys the
        # group at the end.
        dist.Backend.register_backend(
            "pybackend", TestPyBackendProcessGroup.create_backend, devices=["cpu"]
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("pybackend", rank=self.rank, world_size=self.world_size)

        pg = _get_default_group()
        backend = pg._get_backend(torch.device("cpu"))
        self.assertIsInstance(backend, C10DBackend)
        self.assertNotIsInstance(backend, dist.ProcessGroup)
        self.assertEqual(backend.name(), "registered-python-backend")

        allreduce_tensor = torch.zeros(2)
        dist.all_reduce(allreduce_tensor)
        self.assertEqual(allreduce_tensor, torch.full((2,), 2.0))

        sync_tensor = torch.zeros(2)
        dist.all_reduce(sync_tensor, async_op=False)
        self.assertEqual(sync_tensor, torch.full((2,), 2.0))

        async_tensor = torch.zeros(2)
        work = dist.all_reduce(async_tensor, async_op=True)
        self.assertIsNotNone(work)
        work.wait()
        self.assertEqual(async_tensor, torch.full((2,), 2.0))

        future_tensor = torch.ones(2)
        work = dist.all_reduce(future_tensor, async_op=True)
        fut = work.get_future()
        fut.wait()
        self.assertEqual(future_tensor, torch.full((2,), 3.0))

        async_tensors = [torch.zeros(2) for _ in range(4)]
        works = [dist.all_reduce(t, async_op=True) for t in async_tensors]
        for w in works:
            w.wait()
        for t in async_tensors:
            self.assertEqual(t, torch.full((2,), 2.0))

        peer = (self.rank + 1) % self.world_size
        send_tensor = torch.zeros(2)
        recv_tensor = torch.zeros(2)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, peer),
                dist.P2POp(dist.irecv, recv_tensor, peer),
            ]
        )
        for work in works:
            work.wait()

        self.assertEqual(backend.start_coalescing_count, 1)
        self.assertEqual(backend.end_coalescing_count, 1)
        self.assertEqual(send_tensor, torch.ones(2))
        self.assertEqual(recv_tensor, torch.full((2,), 2.0))

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
