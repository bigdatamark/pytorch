# Owner(s): ["module: multiprocessing"]
# [MODULE DOC] Tests for torch.multiprocessing tensor sharing between processes: shared memory
# via file descriptor and file system strategies, storage identity preservation, process pools,
# autograd variable sharing, CUDA IPC (tensors, events, allocator cleanup) and meta tensors.
# Module level worker functions run in child processes. Module constants gate CUDA IPC and
# multi-GPU tests and set repeat counts and a waiting limit; importing runs device-type test
# instantiation for the device-generic class.
import contextlib
import copy
import gc
import os
import sys
import time
import unittest
from sys import platform

import torch
import torch.cuda
import torch.multiprocessing as mp
import torch.utils.hooks
from torch.nn import Parameter
from torch.testing._internal.common_cuda import IS_JETSON
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    load_tests,
    run_tests,
    skipIfRocm,
    skipIfRocmVersionAtLeast,
    slowTest,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
    TEST_WITH_TSAN,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

TEST_REPEATS = 30
HAS_SHM_FILES = os.path.isdir("/dev/shm")
MAX_WAITING_TIME_IN_SECONDS = 30

TEST_CUDA_IPC = (
    torch.cuda.is_available()
    and sys.platform != "darwin"
    and sys.platform != "win32"
    and not IS_JETSON
    and not TEST_WITH_ROCM
)  # https://github.com/pytorch/pytorch/issues/90940

TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1


class SubProcess(mp.Process):
    # [CLASS DOC] HELPER: Process subclass that adds three to a given (shared) tensor in place
    # when run; it is marked as a daemon at construction. Used by the inherit-tensor test.
    def __init__(self, tensor):
        # [FUNCTION DOC] HELPER: Initializes the base process, stores the tensor and marks the
        # process as a daemon.
        super().__init__()
        self.tensor = tensor
        self.daemon = True

    def run(self):
        # [FUNCTION DOC] HELPER: Child entry point; adds three to the stored tensor in place.
        self.tensor.add_(3)


def _test_cuda_ipc_deadlock_actor(queue, iterations):
    # [FUNCTION DOC] HELPER: Actor loop for the CUDA IPC deadlock test: for the given number of
    # iterations, removes one item from the queue if it is not empty, then sleeps ten ms.
    for _ in range(iterations):
        if not queue.empty():
            queue.get()
        time.sleep(0.01)


def _test_cuda_ipc_deadlock_learner(queue, iterations):
    # [FUNCTION DOC] HELPER: Learner loop for the deadlock test: creates a small LSTM on CUDA
    # and, per iteration, puts a deep copy of its state dict on the queue if the queue is not
    # full, then sleeps ten ms.
    net = torch.nn.LSTM(1, 1).cuda()
    for _ in range(iterations):
        if not queue.full():
            queue.put(copy.deepcopy(net.state_dict()))
        time.sleep(0.01)


def simple_fill(queue, event):
    # [FUNCTION DOC] HELPER: Child worker that takes a list of tensors from the queue, sets the
    # whole first tensor to four through the shared memory, then sets the event.
    data = queue.get()
    data[0][:] = 4
    event.set()


def simple_pool_fill(tensor):
    # [FUNCTION DOC] HELPER: Pool worker that fills the given (shared) tensor with four in place
    # and returns a new tensor equal to it plus one.
    tensor.fill_(4)
    return tensor.add(1)


def send_tensor(queue, event, device, dtype):
    # [FUNCTION DOC] HELPER: Child worker that creates a 5x5 tensor of ones on the device with
    # the dtype, puts the same tensor on the queue twice, then waits for the event before exiting.
    t = torch.ones(5, 5, device=device, dtype=dtype)
    queue.put(t)
    queue.put(t)
    event.wait()


def send_and_delete_tensors(queue, event, device, dtype, count, size=5):
    # [FUNCTION DOC] HELPER: Child worker that sends the given count of 1-D tensors, each filled
    # with its loop index, on the queue, deleting its own reference right after each send, then
    # waits for the event.
    for i in range(count):
        t = torch.full([size], i, device=device, dtype=dtype)
        queue.put(t)
        del t
    event.wait()


def limbo_cleanup_worker(q, e):
    # [FUNCTION DOC] HELPER: Child worker that creates a CUDA tensor of zeros, sends it on the
    # queue, waits for the event, then deletes its reference so cleanup happens at exit.
    t = torch.zeros(5, 5, device="cuda")
    q.put(t)
    e.wait()
    del t


def send_tensor_with_untyped_storage(queue, event):
    # [FUNCTION DOC] HELPER: Child worker that splits a two-element CUDA tensor of ones into two
    # chunks and, for each, obtains the raw sharing fields of its untyped storage (device,
    # handles, sizes, offsets, reference counter, event data) and gathers them with tensor
    # metadata into a spec mapping. Sends the list of specs on the queue, then waits for the event.
    tensors = torch.ones(2, device="cuda").chunk(2, dim=0)
    specs = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        (
            storage_device,
            storage_handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        specs.append(
            {
                "tensor_cls": type(tensor),
                "tensor_size": tensor.shape,
                "tensor_stride": tensor.stride(),
                "tensor_offset": tensor.storage_offset(),
                "dtype": tensor.dtype,
                "requires_grad": tensor.requires_grad,
                "storage_cls": type(storage),
                "storage_device": storage_device,
                "storage_handle": storage_handle,
                "storage_size_bytes": storage_size_bytes,
                "storage_offset_bytes": storage_offset_bytes,
                "ref_counter_handle": ref_counter_handle,
                "ref_counter_offset": ref_counter_offset,
                "event_handle": event_handle,
                "event_sync_required": event_sync_required,
            }
        )
    queue.put(specs)
    event.wait()


def receive_and_send_sum(queue, out_queue, event, device, dtype, count, size=5):
    # [FUNCTION DOC] HELPER: Child worker that receives the given count of tensors from the
    # queue, accumulates their sum into a zero tensor on the device, puts the sum on the output
    # queue, then waits for the event.
    s = torch.full([size], 0, device=device, dtype=dtype)
    for _ in range(count):
        t = queue.get()
        s += t
    out_queue.put(s)
    event.wait()


def receive_and_send(queue, out_queue, event, count):
    # [FUNCTION DOC] HELPER: Child worker that receives the given count of tensors from the
    # queue and puts a clone of each on the output queue, then waits for the event.
    for _ in range(count):
        t = queue.get()
        out_queue.put(t.clone())
    event.wait()


def sum_tensors(inq, outq):
    # [FUNCTION DOC] HELPER: Child worker working on CUDA device one: takes a list of tensors
    # from the input queue and, for each, puts a tuple of its sum, device index, element count
    # and storage size on the output queue.
    with torch.cuda.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put(
                (
                    tensor.sum().item(),
                    tensor.get_device(),
                    tensor.numel(),
                    tensor.storage().size(),
                )
            )


def queue_get_exception(inqueue, outqueue):
    # [FUNCTION DOC] HELPER: Child worker that closes the standard error descriptor to hide the
    # expected message, then tries to create a CUDA tensor and puts the raised exception on the
    # output queue, or a no-exception marker string if nothing was raised.
    os.close(2)  # hide expected error message
    try:
        torch.zeros(5, 5).cuda()
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put("no exception")


# Multiply by two in a separate stream
def cuda_multiply_two(queue, ready, done):
    # [FUNCTION DOC] HELPER: Child worker that signals ready, then on a fresh CUDA stream takes a
    # CUDA event and tensor from the queue, waits on the event, doubles the tensor in place,
    # records the event again and signals done.
    ready.set()
    with torch.cuda.stream(torch.cuda.Stream()):
        cuda_event, tensor = queue.get()
        cuda_event.wait()
        tensor.mul_(2)
        cuda_event.record()
        done.set()
        del cuda_event


def requires_grad_variable_sharing(queue, ready):
    # [FUNCTION DOC] HELPER: Child worker that takes a variable from the queue, sets the ready
    # event and puts the variable's requires-grad flag back on the queue.
    var = queue.get()
    ready.set()
    queue.put(var.requires_grad)


def integer_parameter_serialization(queue, done, finish):
    # [FUNCTION DOC] HELPER: Child worker that takes an integer parameter from the queue, does a
    # trivial arithmetic operation on it, deletes it, sets the done event and waits for finish.
    iparam = queue.get()
    iparam + 1
    del iparam
    done.set()
    finish.wait()


def autograd_sharing(queue, ready, master_modified, device, is_parameter):
    # [FUNCTION DOC] HELPER: Child worker for the autograd sharing test. Takes a variable from the
    # queue, sets ready, waits for the master's modification, then checks that data equals the
    # expected values with the master's change, no gradient is present, no backward hooks, and the
    # type is Parameter or plain Tensor as requested. Overwrites the data with ones, sets a
    # gradient of ones, and puts the combined success flag on the queue.
    var = queue.get()
    ready.set()
    master_modified.wait()

    expected_var = torch.arange(1.0, 26, device=device).view(5, 5)
    expected_var[0, 0] = 1000
    is_ok = var.data.equal(expected_var)
    var.data[:] = torch.ones(5, 5, device=device)

    is_ok &= var.grad is None
    is_ok &= not var._backward_hooks
    if is_parameter:
        is_ok &= type(var) is Parameter
    else:
        is_ok &= type(var) is torch.Tensor
    var._grad = torch.ones(5, 5, device=device)

    queue.put(is_ok)


def mixed_type_producer(queue, event):
    # [FUNCTION DOC] HELPER: Child worker that ten times sends a float CUDA tensor and a byte
    # CUDA tensor on the queue, then waits for and clears the event.
    for _ in range(10):
        float_tensor = torch.ones(2, 2).float().cuda()
        byte_tensor = torch.zeros(2, 2).byte().cuda()

        queue.put(float_tensor)
        queue.put(byte_tensor)
        event.wait()
        event.clear()


def simple_autograd_function(a=1):
    # [FUNCTION DOC] HELPER: Runs backward on a random tensor made to require gradients, then
    # returns the square of the argument (default one). Used to test autograd in pool workers.
    torch.rand(3).requires_grad_(True).mean().backward()
    return a**2


def _event_multiprocess_child(event, p2c, c2p):
    # [FUNCTION DOC] HELPER: Child for the event test. Reports ready on the child-to-parent
    # queue, waits for the parent's record notification, synchronizes the shared event, then
    # reports completion.
    c2p.put(0)  # notify parent child is ready
    p2c.get()  # wait for record in parent
    event.synchronize()
    c2p.put(1)  # notify parent synchronization is done


def _event_handle_importer_consumer(handle, p2c, c2p):
    # [FUNCTION DOC] HELPER: Child that imports an event from an IPC handle, reports ready, waits
    # for the parent's record notice, synchronizes the event, reports done, then waits for the
    # parent to finish before the event is destroyed.
    e1 = torch.cuda.Event.from_ipc_handle(0, handle)
    c2p.put(0)  # notify parent child is ready
    p2c.get()  # wait for record in parent
    e1.synchronize()
    c2p.put(1)  # notify synchronization is done in child
    p2c.get()  # wait for parent to finish before destructing child event


def _event_handle_exporter_consumer(handle, p2c, c2p):
    # [FUNCTION DOC] HELPER: Child that imports an event from an IPC handle on the current
    # device, spins the GPU for a while on a new stream, records the event, notifies the parent
    # and waits for the parent to finish synchronizing before destroying the event.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        e1 = torch.cuda.Event.from_ipc_handle(torch.cuda.current_device(), handle)
        torch.cuda._sleep(50000000)  # spin for about 50 ms
        e1.record()
        c2p.put(0)
        # wait for parent process finished synchronization before
        # destructing e1
        p2c.get()


@contextlib.contextmanager
def fs_sharing():
    # [FUNCTION DOC] HELPER: Context manager that switches the tensor sharing strategy to file
    # system on entry and always restores the previous strategy on exit.
    prev_strategy = mp.get_sharing_strategy()
    mp.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        mp.set_sharing_strategy(prev_strategy)


class leak_checker:
    # [CLASS DOC] HELPER: Context manager used by tests to detect leaked shared memory files for
    # the pids it checks (the current pid initially; tests add child pids). State: the checked
    # pid list, the owning test case and the list of next free file descriptors recorded on entry.
    def __init__(self, test_case):
        # [FUNCTION DOC] HELPER: Records the current pid as checked and stores the test case.
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Records the next free file descriptor numbers at start and returns
        # the checker itself.
        self.next_fds = self._get_next_fds(10)
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] HELPER: On exit, collects CUDA IPC memory when CUDA is available. If the
        # body raised no exception, asserts there are no leftover shared memory files for the
        # checked pids. Never suppresses exceptions (returns false).
        # KNOWN ISSUE: the descriptor leak comparison is commented out as too flaky, so only the
        # shared memory file check is enforced.
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        if args[0] is None:
            # Check that the 10th available file-descriptor at the end of the
            # test is no more than 4 higher than the 10th available at the
            # start. This attempts to catch file descriptor leaks, but allows
            # one-off initialization that may use up a file descriptor
            # TODO: Disabled because this check is too flaky
            # available_fds = self._get_next_fds(10)
            # self.test_case.assertLessEqual(
            #     available_fds[-1] - self.next_fds[-1], 5)
            self.test_case.assertFalse(self.has_shm_files())
        return False

    def check_pid(self, pid):
        # [FUNCTION DOC] HELPER: Adds a pid to the list of checked pids (those whose shared
        # memory files the leak check looks for).
        self.checked_pids.append(pid)

    def _get_next_fds(self, n=1):
        # [FUNCTION DOC] HELPER: Duplicates a descriptor n times to learn the lowest free
        # descriptor numbers, closes the duplicates and returns the list of numbers.
        # dup uses the lowest-numbered unused descriptor for the new descriptor
        fds = [os.dup(0) for i in range(n)]
        for fd in fds:
            os.close(fd)
        return fds

    def has_shm_files(self, wait=True):
        # [FUNCTION DOC] HELPER: Reports whether shared memory files remain for the checked pids.
        # Returns false when the shared memory directory does not exist. With the file system
        # strategy and waiting enabled, it polls every half second up to the maximum waiting time
        # for the files to disappear. Returns the final boolean result.
        if not HAS_SHM_FILES:
            return False

        result = self._has_shm_files()
        if not result or mp.get_sharing_strategy() != "file_system" or not wait:
            return result

        total_waiting_time = 0
        waiting_time = 0.5

        while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and result:
            time.sleep(waiting_time)
            total_waiting_time += waiting_time
            result = self._has_shm_files()

        return result

    def _has_shm_files(self):
        # [FUNCTION DOC] HELPER: Runs garbage collection, then lists the shared memory directory
        # and returns true if any entry name starts with the torch prefix plus any checked pid.
        gc.collect()
        names = ["torch_" + str(pid) for pid in self.checked_pids]
        for filename in os.listdir("/dev/shm"):
            for name in names:
                if filename.startswith(name):
                    return True
        return False


class _MultiprocessingTestMixin:
    # [CLASS DOC] HELPER: Mixin providing shared test implementations (sharing, autograd sharing,
    # empty tensor, integer parameter) for the device-generic and plain test classes. Expects to
    # be combined with a test case class.
    """Shared helpers used by both device-type and non-device-type test classes."""

    def _test_sharing(self, ctx=mp, device="cpu", dtype=torch.float, repeat=1):
        # [FUNCTION DOC] HELPER: Sharing check using the given multiprocessing context, device and
        # dtype, repeated the given number of times inside a leak checker. Each repetition runs a
        # fill test and a receive test (nested helpers below). Meta devices skip value checks.
        def test_fill():
            # [FUNCTION DOC] HELPER: Puts a zero tensor and a column view of it on a queue,
            # starts a daemon child that fills the first tensor with four, polls until the child
            # sets its event (up to the maximum waiting time), asserts the change is visible to the
            # parent in both tensors (except for meta) and the child exits.
            x = torch.zeros(5, 5).to(device, dtype)
            q = ctx.Queue()
            e = ctx.Event()

            data = [x, x[:, 1]]
            q.put(data)

            p = ctx.Process(target=simple_fill, args=(q, e))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            total_waiting_time = 0
            waiting_time = 0.5
            is_set = False
            while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and not is_set:
                time.sleep(waiting_time)
                total_waiting_time += waiting_time
                is_set = e.is_set()

            self.assertTrue(is_set)
            if device != "meta":
                self.assertTrue(data[0].eq(4).all())
                self.assertTrue(data[1].eq(4).all())

            p.join(100)
            self.assertFalse(p.is_alive())

        def test_receive():
            # [FUNCTION DOC] HELPER: Starts a daemon child that sends the same tensor twice,
            # receives both, checks values (sizes only for meta) and that their storages match in
            # type and contents, releases the tensors, sets the event and joins the child.
            # KNOWN ISSUE: the data pointer comparison compares the first storage with itself,
            # not with the second, so it can never fail.
            q = ctx.Queue()
            e = ctx.Event()

            p = ctx.Process(target=send_tensor, args=(q, e, device, dtype))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            t1 = q.get()
            t2 = q.get()
            if device == "meta":
                self.assertEqual(t1.size(), t2.size())
            else:
                self.assertTrue(t1.eq(1).all())
            s1 = t1.storage()
            s2 = t2.storage()
            self.assertEqual(type(s1), type(s2))
            self.assertEqual(s1.data_ptr(), s1.data_ptr())
            if device == "meta":
                self.assertEqual(s1.size(), s2.size())
            else:
                self.assertEqual(s1, s2)

            del t1, t2

            e.set()
            p.join(100)
            self.assertFalse(p.is_alive())

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()

    def _test_autograd_sharing(self, var, ctx=mp, is_parameter=False):
        # [FUNCTION DOC] HELPER: Shares a variable with a child, runs the child's checks, and
        # verifies results. Starts a daemon child, optionally registers an unserializable
        # backward hook if the variable requires gradients, sets a zero gradient and sends the
        # variable. After the child is ready, modifies the data and gradient, signals the child,
        # asserts the child reported success, and that the parent sees the child's writes to the
        # data (ones) but its own gradient. Joins the child.
        device = str(var.device).split(":")[0]

        ready = ctx.Event()
        master_modified = ctx.Event()
        queue = ctx.Queue()
        p = ctx.Process(
            target=autograd_sharing,
            args=(queue, ready, master_modified, device, is_parameter),
        )
        p.daemon = True
        p.start()

        @torch.utils.hooks.unserializable_hook
        def hook(*unused):
            # [FUNCTION DOC] HELPER: No-op hook function marked as unserializable so that
            # sending the variable skips it.
            pass

        if var.requires_grad:
            var.register_hook(hook)
        var._grad = torch.zeros(5, 5, device=device)
        queue.put(var)

        ready.wait()
        var.data[0, 0] = 1000
        var.grad.data[:] = torch.ones(5, 5, device=device) * 4
        master_modified.set()

        worker_ok = queue.get()
        self.assertTrue(worker_ok)

        self.assertEqual(var.data, torch.ones(5, 5, device=device))
        self.assertEqual(var.grad.data, torch.ones(5, 5, device=device) * 4)
        p.join(100)
        self.assertFalse(p.is_alive())

    def _test_empty_tensor_sharing(self, dtype, device):
        # [FUNCTION DOC] HELPER: Sends an empty tensor of the given dtype and device through a
        # queue and asserts the received value equals the original.
        q = mp.Queue()
        empty = torch.tensor([], dtype=dtype, device=device)
        q.put(empty)
        out = q.get(timeout=1)
        self.assertEqual(out, empty)

    def _test_integer_parameter_serialization(self, device):
        # [FUNCTION DOC] HELPER: Sends a non-gradient integer parameter to a spawned child that
        # reads it and signals done. The parent waits up to the maximum waiting time, deletes its
        # parameter (collecting CUDA IPC memory for CUDA), always releases the child and joins it,
        # then asserts the child exited with code zero.
        param = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int64, device=device), requires_grad=False
        )

        ctx = mp.get_context("spawn")
        done = ctx.Event()
        finish = ctx.Event()
        queue = ctx.Queue()
        p = ctx.Process(
            target=integer_parameter_serialization, args=(queue, done, finish)
        )
        p.start()
        try:
            queue.put(param)
            self.assertTrue(done.wait(MAX_WAITING_TIME_IN_SECONDS))
            del param
            if torch.device(device).type == "cuda":
                torch.cuda.ipc_collect()
        finally:
            finish.set()
            p.join(100)

        self.assertEqual(
            0,
            p.exitcode,
            msg=lambda msg: f'{msg}\nFailed to serialize successfully for "{device}" device!',
        )


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class TestMultiprocessingDeviceType(_MultiprocessingTestMixin, TestCase):
    # [CLASS DOC] Device-generic sharing tests instantiated for each available device type; skipped
    # under thread sanitizer. Combines the shared mixin with the test case base class.
    """Device-generic multiprocessing tests, instantiated per available backend."""

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Collects CUDA IPC memory after each test when CUDA is available.
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    def _get_ctx(self, device):
        # [FUNCTION DOC] HELPER: Returns the default multiprocessing module for CPU devices and a
        # spawn context for all other devices.
        if torch.device(device).type == "cpu":
            return mp
        return mp.get_context("spawn")

    def test_simple_sharing(self, device):
        # [FUNCTION DOC] Runs the basic sharing check with float dtype on the test device, using
        # a spawn context for non-CPU devices.
        ctx = self._get_ctx(device)
        self._test_sharing(ctx, device, torch.float)

    @dtypes(torch.float32, torch.int64)
    def test_empty_tensor_sharing(self, device, dtype):
        # [FUNCTION DOC] Runs the empty tensor sharing check for the float and int64 dtypes
        # given by the decorator, on the test device.
        self._test_empty_tensor_sharing(dtype, torch.device(device))

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/92131")
    @unittest.skipIf(
        TEST_WITH_ASAN,
        "non-deterministically hangs with ASAN "
        "https://github.com/pytorch/pytorch/issues/94024",
    )
    def test_variable_sharing(self, device):
        # [FUNCTION DOC] Runs the autograd sharing check for a leaf variable with requires-grad
        # both true and false. Skipped on ROCm and under address sanitizer.
        ctx = self._get_ctx(device)
        for requires_grad in [True, False]:
            var = (
                torch.arange(1.0, 26, device=device)
                .view(5, 5)
                .requires_grad_(requires_grad)
            )
            self._test_autograd_sharing(var, ctx)

    def test_parameter_sharing(self, device):
        # [FUNCTION DOC] Runs the autograd sharing check for a Parameter, expecting the child to
        # see a Parameter type.
        ctx = self._get_ctx(device)
        param = Parameter(torch.arange(1.0, 26, device=device).view(5, 5))
        self._test_autograd_sharing(param, ctx, is_parameter=True)

    def test_integer_parameter_serialization(self, device):
        # [FUNCTION DOC] Runs the integer parameter serialization check on the test device.
        self._test_integer_parameter_serialization(device=device)

    # See https://github.com/pytorch/pytorch/issues/14997
    @unittest.skipIf(TEST_WITH_ASAN, "non-deterministically hangs with ASAN")
    def test_leaf_variable_sharing(self, device):
        # [FUNCTION DOC] For requires-grad true and false, sends a leaf variable to a child and
        # asserts the child sees the same requires-grad flag. Skipped under address sanitizer.
        ctx = self._get_ctx(device)
        for requires_grad in [True, False]:
            var = (
                torch.arange(1.0, 26, device=device)
                .view(5, 5)
                .requires_grad_(requires_grad)
            )
            self.assertTrue(var.is_leaf)
            ready = ctx.Event()
            queue = ctx.Queue()
            p = ctx.Process(target=requires_grad_variable_sharing, args=(queue, ready))
            p.daemon = True
            p.start()
            queue.put(var)
            ready.wait()
            worker_requires_grad = queue.get()
            self.assertTrue(worker_requires_grad == requires_grad)

    def test_non_leaf_variable_sharing(self, device):
        # [FUNCTION DOC] Asserts that putting a non-leaf variable (requires grad, produced by an
        # operation) on a queue raises a runtime error mentioning requires_grad.
        var0 = torch.arange(1.0, 26, device=device).view(5, 5).requires_grad_(True)
        var = var0 * 2
        queue = mp.SimpleQueue()
        self.assertRaisesRegex(RuntimeError, r"requires_grad", lambda: queue.put(var))

    def test_is_shared_accelerator(self, device):
        # [FUNCTION DOC] Skips for CPU; for other devices asserts a freshly created tensor is
        # already marked as shared.
        if torch.device(device).type == "cpu":
            self.skipTest("CPU tensors are not shared by default")
        t = torch.randn(5, 5, device=device)
        self.assertTrue(t.is_shared())


instantiate_device_type_tests(TestMultiprocessingDeviceType, globals())


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class TestMultiprocessing(_MultiprocessingTestMixin, TestCase):
    # [CLASS DOC] Non-device-generic multiprocessing tests: file descriptor and file system
    # strategies, pools, inheritance, autograd with fork and spawn, CUDA IPC and events, meta
    # tensors and thread names. Skipped under thread sanitizer.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Collects CUDA IPC memory after each test when CUDA is available.
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    def _test_preserve_sharing(self, ctx=mp, repeat=1):
        # [FUNCTION DOC] HELPER: Repeats a check that storage sharing is preserved through a
        # queue inside a leak checker.
        def do_test():
            # [FUNCTION DOC] HELPER: Sends a storage, a tensor and two views through a queue and
            # asserts the received values are equal and all share the same underlying storage
            # handle as the received storage.
            x = torch.randn(5, 5)
            data = [x.storage(), x, x[2], x[:, 1]]
            q = ctx.Queue()
            q.put(data)
            new_data = q.get(timeout=1)
            self.assertEqual(new_data, data, atol=0, rtol=0)
            storage_cdata = data[0]._cdata
            self.assertEqual(new_data[0]._cdata, storage_cdata)
            for t in new_data[1:]:
                self.assertEqual(t.storage()._cdata, storage_cdata)

        with leak_checker(self):
            for _ in range(repeat):
                do_test()

    def _test_pool(self, ctx=mp, repeat=1):
        # [FUNCTION DOC] HELPER: Repeats a pool check inside a leak checker.
        def do_test():
            # [FUNCTION DOC] HELPER: Creates a pool of two workers (registering their pids with the
            # leak checker), maps the fill function over four zero tensors, asserts results are
            # fives and the original buffers are fours, then closes and joins the pool.
            p = ctx.Pool(2)
            for proc in p._pool:
                lc.check_pid(proc.pid)

            buffers = [torch.zeros(2, 2) for i in range(4)]
            results = p.map(simple_pool_fill, buffers, 1)
            self.assertEqual(len(results), len(buffers))
            for r in results:
                self.assertEqual(r, torch.ones(2, 2) * 5, atol=0, rtol=0)
            for b in buffers:
                self.assertEqual(b, torch.ones(2, 2) * 4, atol=0, rtol=0)

            p.close()
            p.join()

        with leak_checker(self) as lc:
            for _ in range(repeat):
                do_test()

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    @unittest.skipIf(
        TEST_WITH_ASAN,
        "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326",
    )
    def test_fd_sharing(self):
        # [FUNCTION DOC] Runs the sharing check with the default (file descriptor) strategy for
        # the repeat count. Skipped on macOS and under address sanitizer.
        self._test_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    def test_fd_preserve_sharing(self):
        # [FUNCTION DOC] Runs the preserve sharing check with the default strategy for the repeat
        # count. Skipped on macOS.
        self._test_preserve_sharing(repeat=TEST_REPEATS)

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    def test_fd_pool(self):
        # [FUNCTION DOC] Runs the pool check with the default strategy for the repeat count.
        # Skipped on macOS.
        self._test_pool(repeat=TEST_REPEATS)

    # torch_shm_manager cannot resolve librocprofiler-sdk.so.1 in CI child processes.
    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(
        TEST_WITH_ASAN,
        "seems to hang with ASAN, see https://github.com/pytorch/pytorch/issues/5326",
    )
    def test_fs_sharing(self):
        # [FUNCTION DOC] Runs the sharing check under the file system strategy; repeats once on
        # macOS (slow) and otherwise the full repeat count. Skipped on new ROCm and address
        # sanitizer.
        with fs_sharing():
            # The test works but is very slow on MacOS, see https://github.com/pytorch/pytorch/pull/93183,
            # so run it only once there. The delay is in waiting for the child process to terminate (join)
            repeat = 1 if IS_MACOS else TEST_REPEATS
            self._test_sharing(repeat=repeat)

    @skipIfRocmVersionAtLeast([7, 14])
    def test_fs_preserve_sharing(self):
        # [FUNCTION DOC] Runs the preserve sharing check under the file system strategy.
        with fs_sharing():
            self._test_preserve_sharing(repeat=TEST_REPEATS)

    @skipIfRocmVersionAtLeast([7, 14])
    def test_fs_pool(self):
        # [FUNCTION DOC] Runs the pool check under the file system strategy.
        with fs_sharing():
            self._test_pool(repeat=TEST_REPEATS)

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.skipIf(not HAS_SHM_FILES, "don't not how to check if shm files exist")
    def test_fs(self):
        # [FUNCTION DOC] Under the file system strategy, repeatedly checks shared memory files
        # appear when a storage is put on a queue. Skipped if no shared memory directory exists.
        def queue_put():
            # [FUNCTION DOC] HELPER: Asserts no shared memory file exists, puts a double storage on
            # a queue, sleeps briefly for the asynchronous serialization, asserts a file now
            # exists, then reads the item back.
            x = torch.DoubleStorage(4)
            q = mp.Queue()
            self.assertFalse(lc.has_shm_files())
            q.put(x)
            time.sleep(0.05)  # queue serializes asynchronously
            self.assertTrue(lc.has_shm_files(wait=False))
            q.get()

        with fs_sharing(), leak_checker(self) as lc:
            for _ in range(TEST_REPEATS):
                queue_put()

    def test_inherit_tensor(self):
        # [FUNCTION DOC] Shares a zero tensor, starts the process subclass that adds three to it,
        # joins with a timeout of two seconds; prints a message if it is too slow, otherwise
        # asserts the tensor equals three.
        t = torch.zeros(5, 5)
        p = SubProcess(t.share_memory_())
        p.start()
        p.join(2)
        if p.exitcode is None:
            print("test_inherit_tensor: SubProcess too slow")
        else:
            self.assertEqual(t, torch.ones(5, 5) * 3, atol=0, rtol=0)

    @unittest.skipIf(IS_WINDOWS, "Test needs to use fork multiprocessing")
    def test_autograd_errors(self):
        # [FUNCTION DOC] Using a forked pool of three, maps the autograd helper. When CUDA, MPS or
        # XPU is available, asserts the pool raises an error about being unable to handle autograd;
        # otherwise it must succeed. Skipped on Windows.
        ctx = mp.get_context("fork")
        simple_autograd_function()
        # Autograd only uses thread when GPUs are involved
        if (
            torch.cuda.is_available()
            or torch.backends.mps.is_available()
            or torch.xpu.is_available()
        ):
            with self.assertRaisesRegex(RuntimeError, r"Unable to handle autograd"):
                with ctx.Pool(3) as pool:
                    pool.map(simple_autograd_function, [1, 2, 3])
        else:
            with ctx.Pool(3) as pool:
                pool.map(simple_autograd_function, [1, 2, 3])

    def test_autograd_fine_with_spawn(self):
        # [FUNCTION DOC] Asserts that the autograd helper works in a pool of three created from a
        # spawn context.
        ctx = mp.get_context("spawn")
        simple_autograd_function()
        with ctx.Pool(3) as pool:
            pool.map(simple_autograd_function, [1, 2, 3])

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_memory_allocation(self):
        # [FUNCTION DOC] Sends five int tensors from a spawned child that deletes each after
        # sending, and checks the first one received. Skipped without CUDA IPC.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        e = ctx.Event()
        p = ctx.Process(
            target=send_and_delete_tensors, args=(q, e, "cuda", torch.int, 5)
        )
        p.start()
        t = []
        for _ in range(5):
            t.append(q.get())
        self.assertEqual(t[0], torch.full([5], 0, dtype=torch.int32))
        del t
        e.set()
        p.join(1)

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_ipc_limbo_cleanup_at_exit(self):
        # [FUNCTION DOC] Checks that a child holding a sent CUDA tensor exits cleanly (code zero)
        # after deleting it at exit. Skipped without CUDA IPC.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        e = ctx.Event()

        p = ctx.Process(target=limbo_cleanup_worker, args=(q, e))
        p.start()

        t_received = q.get()
        e.set()
        p.join(5)
        self.assertEqual(p.exitcode, 0)
        del t_received

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_ipc_deadlock(self):
        # [FUNCTION DOC] Starts an actor and a learner child sharing a size-one queue for a hundred
        # iterations; asserts neither is alive ten seconds after start (no deadlock). Skipped
        # without CUDA IPC.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(1)
        processes = dict(
            a=ctx.Process(target=_test_cuda_ipc_deadlock_actor, args=(queue, 100)),
            l=ctx.Process(target=_test_cuda_ipc_deadlock_learner, args=(queue, 100)),
        )

        for p in processes.values():
            p.start()

        for p in processes.values():
            p.join(10)

        for p in processes.values():
            self.assertFalse(p.is_alive())

    @slowTest
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_cuda_send_many(self, name=None, size=5, count=100000):
        # [FUNCTION DOC] Slow test with three chained children: sender, forwarder, summer. Checks
        # the first element of the summed result equals the arithmetic series total. Skipped
        # without CUDA IPC.
        ctx = mp.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        q3 = ctx.Queue()
        e1 = ctx.Event()
        e2 = ctx.Event()
        e3 = ctx.Event()
        p1 = ctx.Process(
            target=send_and_delete_tensors,
            args=(q1, e1, "cuda", torch.long, count, size),
        )
        p2 = ctx.Process(target=receive_and_send, args=(q1, q2, e2, count))
        p3 = ctx.Process(
            target=receive_and_send_sum,
            args=(q2, q3, e3, "cuda", torch.long, count, size),
        )
        p1.start()
        p2.start()
        p3.start()
        result = q3.get()
        self.assertEqual(result[0], int(count * (count - 1) / 2))
        del result
        e1.set()
        e2.set()
        e3.set()
        p1.join(1)
        p2.join(1)
        p3.join(1)

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @unittest.skipIf(not TEST_MULTIGPU, "found only 1 GPU")
    def test_cuda_small_tensors(self):
        # [FUNCTION DOC] Sends five small CUDA tensors alternating over two devices to a child
        # that reports their sums, devices and sizes; asserts sums, devices and element counts
        # match (the storage size check is intentionally disabled), then drops references and
        # collects IPC memory. Needs CUDA IPC and more than one GPU.
        # Check multiple small tensors which will likely use the same
        # underlying cached allocation
        ctx = mp.get_context("spawn")
        tensors = []
        for i in range(5):
            device = i % 2
            tensors += [torch.arange(i * 5.0, (i + 1) * 5).cuda(device)]

        inq = ctx.Queue()
        outq = ctx.Queue()
        inq.put(tensors)
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        p.start()

        results = []
        for _ in range(5):
            results.append(outq.get())
        p.join()

        for i, _tensor in enumerate(tensors):
            v, device, tensor_size, storage_size = results[i]
            self.assertEqual(v, torch.arange(i * 5.0, (i + 1) * 5).sum())
            self.assertEqual(device, i % 2)
            self.assertEqual(tensor_size, 5)

            # You might think this should be the case, but it's not!  After
            # data from the CUDA caching allocator goes through IPC, the
            # size of the storage is the size of the *cached cudaMalloc for
            # the entire memory block* of the storage, not just the storage.
            # See Note [CUDA IPC and the caching allocator] for more info
            #
            # self.assertEqual(storage_size, 5)

        # Collect current process (producer) files, make sure nothing holds
        # ref to the sent tensors
        del _tensor
        del tensors

        # We need to collect, as CUDA MP implementation holds one shared
        # memory 'file' for performance reason
        torch.cuda.ipc_collect()

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_bad_call(self):
        # [FUNCTION DOC] After initializing CUDA in the parent, starts a forked child that tries to
        # use CUDA, and asserts it reports a runtime error. Skipped on Windows or without CUDA.
        # Initialize CUDA
        t = torch.zeros(5, 5).cuda().cpu()
        inq = mp.Queue()
        outq = mp.Queue()
        p = mp.Process(target=queue_get_exception, args=(inq, outq))
        p.start()
        inq.put(t)
        p.join()
        self.assertIsInstance(outq.get(), RuntimeError)

    @unittest.skipIf(IS_WINDOWS, "not applicable to Windows (only fails with fork)")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_wrong_cuda_fork(self):
        # [FUNCTION DOC] Runs a script in a subprocess that uses CUDA before forking workers, and
        # asserts the stderr contains the message that CUDA cannot be re-initialized in a forked
        # subprocess. Skipped on Windows or without CUDA.
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch
from torch.multiprocessing import Process
def run(rank):
    torch.cuda.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        x = torch.rand(20, 2).cuda()
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
        )
        self.assertRegex(stderr, "Cannot re-initialize CUDA in forked subprocess.")

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_rebuild_cuda_tensor(self):
        # [FUNCTION DOC] Receives storage specs from a child and rebuilds a CUDA tensor from each
        # with the reductions rebuild function, then compares to a list of ones. Skipped without
        # CUDA IPC.
        # KNOWN ISSUE: the expectation compares a list of tensors to a list of scalar ones, which
        # may be unreliable; the loop variable is also deleted after the loop.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        event = ctx.Event()

        proc = ctx.Process(
            target=send_tensor_with_untyped_storage,
            args=(queue, event),
        )
        proc.start()

        specs = queue.get()
        tensors = []
        for spec in specs:
            tensors.append(mp.reductions.rebuild_cuda_tensor(**spec))
        self.assertEqual(tensors, [1, 1])

        del tensors, spec
        event.set()

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event(self):
        # [FUNCTION DOC] Sends an interprocess event and tensor to a child that doubles the tensor
        # on another stream after waiting for the event. The parent adds one after a GPU sleep,
        # records the event, sends them, waits for the child and checks the tensor values are
        # all four. Skipped without CUDA IPC.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        ready = ctx.Event()
        done = ctx.Event()
        p = ctx.Process(target=cuda_multiply_two, args=(queue, ready, done))
        p.start()

        ready.wait()
        with torch.cuda.stream(torch.cuda.Stream()):
            tensor = torch.cuda.FloatTensor([1, 1, 1, 1])
            # Use a sleep kernel to test events. Without the event, the
            # multiply happens before the add.
            event = torch.cuda.Event(interprocess=True)
            torch.cuda._sleep(20000000)  # about 30 ms
            tensor.add_(1)
            event.record()
            queue.put((event, tensor))
            done.wait()  # must wait until subprocess records event
            event.synchronize()
            self.assertEqual(list(tensor), [4, 4, 4, 4])
        p.join()

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_multiprocess(self):
        # [FUNCTION DOC] Shares an interprocess event with a child; records it after a GPU sleep
        # and checks query is false until the child has synchronized, then true. Skipped without
        # CUDA IPC.
        event = torch.cuda.Event(enable_timing=False, interprocess=True)
        self.assertTrue(event.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=_event_multiprocess_child,
            args=(event, p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for until child process is ready
        torch.cuda._sleep(50000000)  # spin for about 50 ms
        event.record()
        p2c.put(0)  # notify child event is recorded

        self.assertFalse(event.query())
        c2p.get()  # wait for synchronization in child
        self.assertTrue(event.query())
        p.join()

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    @unittest.skipIf(not TEST_MULTIGPU, "found only 1 GPU")
    def test_event_handle_multi_gpu(self):
        # [FUNCTION DOC] Creates IPC handles on a different device than the event's device, for an
        # unrecorded and a recorded event. Needs CUDA IPC and at least two GPUs.
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        with torch.cuda.device(d0):
            e0 = torch.cuda.Event(enable_timing=False, interprocess=True)

        with torch.cuda.device(d1):
            # create handle on different device from un-recorded event
            e0.ipc_handle()

        with torch.cuda.device(d0):
            e1 = torch.cuda.Event(enable_timing=False, interprocess=True)
            stream = torch.cuda.Stream()
            torch.cuda._sleep(50000000)  # spin for about 50 ms
            e1.record(stream)

        with torch.cuda.device(d1):
            # create handle on different device from recorded event
            e1.ipc_handle()

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_handle_importer(self):
        # [FUNCTION DOC] Sends an IPC handle of an event to a child that imports it; the parent
        # records the event after a GPU sleep and checks its query state before and after the
        # child synchronizes. Skipped without CUDA IPC.
        e0 = torch.cuda.Event(enable_timing=False, interprocess=True)
        self.assertTrue(e0.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=_event_handle_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to become ready
        torch.cuda._sleep(50000000)  # spin for about 50 ms
        e0.record()
        p2c.put(0)  # notify child event is recorded

        self.assertFalse(e0.query())
        c2p.get()  # wait for synchronization in child
        self.assertTrue(e0.query())
        p2c.put(1)  # notify child that parent is done
        p.join()

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_event_handle_exporter(self):
        # [FUNCTION DOC] Sends an IPC handle to a child that records the imported event; the
        # parent checks the event is incomplete, synchronizes it, then checks it is complete.
        # Skipped without CUDA IPC.
        e0 = torch.cuda.Event(enable_timing=False, interprocess=True)

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=_event_handle_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()
        # wait for event in child process is recorded
        c2p.get()

        self.assertFalse(e0.query())
        e0.synchronize()
        self.assertTrue(e0.query())
        p2c.put(0)
        p.join()

    def test_empty_tensor_sharing_meta(self):
        # [FUNCTION DOC] Runs the empty tensor sharing check for float32 and int64 on the meta
        # device.
        self._test_empty_tensor_sharing(torch.float32, torch.device("meta"))
        self._test_empty_tensor_sharing(torch.int64, torch.device("meta"))

    def test_tensor_sharing_meta(self):
        # [FUNCTION DOC] Sends a one-element float32 meta tensor through a queue and asserts the
        # received value equals the original.
        dtype = torch.float32
        device = torch.device("meta")
        q = mp.Queue()
        empty = torch.tensor([1], dtype=dtype, device=device)
        q.put(empty)
        out = q.get(timeout=1)
        self.assertEqual(out, empty)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/167522"
    )
    def test_meta_simple(self):
        # [FUNCTION DOC] Runs the sharing check with a spawn context on the meta device. Skipped
        # on Linux and in slow mode (linked issue), so rarely runs.
        self._test_sharing(mp.get_context("spawn"), "meta", torch.float)

    # Check sharing a cudaMalloc allocation with different types of storage.
    # (Issue #11422)
    def _test_mixed_types_cuda_sharing(self, ctx=mp):
        # [FUNCTION DOC] HELPER: Child producer sends ten pairs of float and byte CUDA tensors that
        # share one allocation; checks received values equal ones and zeros, releases them and
        # signals the child, then sleeps five seconds and joins.
        all_ones = torch.ones(2, 2).float()
        all_zeros = torch.zeros(2, 2).byte()
        queue = ctx.Queue()
        event = ctx.Event()

        p = ctx.Process(target=mixed_type_producer, args=(queue, event))

        p.start()

        for _ in range(10):
            float_tensor = queue.get()
            byte_tensor = queue.get()
            self.assertEqual(float_tensor, all_ones)
            self.assertEqual(byte_tensor, all_zeros)
            del float_tensor, byte_tensor
            event.set()

        time.sleep(5)
        p.join()

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_mixed_types_cuda_sharing(self):
        # [FUNCTION DOC] Runs the mixed types CUDA sharing check with a spawn context. Skipped
        # without CUDA IPC.
        self._test_mixed_types_cuda_sharing(mp.get_context("spawn"))

    def test_empty_shared(self):
        # [FUNCTION DOC] Makes an empty tensor shared in memory; passes if no error is raised.
        t = torch.tensor([])
        t.share_memory_()

    def _test_is_shared(self):
        # [FUNCTION DOC] HELPER: Asserts a new tensor is not shared, moves it to shared memory, and
        # asserts it is shared.
        t = torch.randn(5, 5)
        self.assertFalse(t.is_shared())
        t.share_memory_()
        self.assertTrue(t.is_shared())

    @unittest.skipIf(
        platform == "darwin", "file descriptor strategy is not supported on macOS"
    )
    def test_is_shared(self):
        # [FUNCTION DOC] Runs the shared flag check with the default strategy. Skipped on macOS.
        self._test_is_shared()

    # torch_shm_manager cannot resolve librocprofiler-sdk.so.1 in CI child processes.
    @skipIfRocmVersionAtLeast([7, 14])
    def test_fs_is_shared(self):
        # [FUNCTION DOC] Runs the shared flag check under the file system strategy. Skipped on
        # new ROCm.
        with fs_sharing():
            self._test_is_shared()

    @unittest.skipIf(sys.platform != "linux", "Only runs on Linux; requires prctl(2)")
    def test_set_thread_name(self):
        # [FUNCTION DOC] Sets the current thread name through the multiprocessing helper and
        # asserts reading it back gives the same name. Linux only.
        name = "test name"
        mp._set_thread_name(name)
        self.assertEqual(mp._get_thread_name(), name)


if __name__ == "__main__":
    run_tests()
