# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the c10d one-sided window API
# (Backend.new_window / Window.put / signal / wait_signal /
# map_remote_tensor). Modeled on torchcomms' WindowRmaTest; parameterized over
# backends like test_c10d_fault_tolerance.py so additional backends can be
# enabled by extending WINDOW_BACKENDS.
# [MODULE DOC] Two-process tests of the one-sided window API (register, put, signal,
# wait_signal, map_remote_tensor, get_attr) over memory allocated from the backend mempool.
# An abstract mixin holds all tests; a factory builds one concrete multi-process test class
# per (backend, device) entry in WINDOW_BACKENDS (currently nccl2 on cuda) and publishes it
# as a module global named after the backend. Import side effects: exits the process with
# status 0 if the distributed package is unavailable, and creates those module globals.

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


WINDOW_BACKENDS = [
    ("nccl2", "cuda"),
]


class AbstractWindowTest:
    # [CLASS DOC] Mixin with all window tests, meant to be combined with a multi-process test
    # case. The concrete subclass must provide the class attributes backend_name and
    # device_type. Each test spawns two processes, initializes a process group over a
    # file store, skips if the backend lacks window support, and uses tensors allocated from
    # the backend mempool so they can be registered in a window.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property fixing the number of ranks at two.
        return 2

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property giving the device of the configured type indexed
        # by this process's rank.
        return torch.device(f"{self.device_type}:{self.rank}")

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setup, then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the process group if one is initialized, runs the
        # parent teardown, and removes the file-store file, ignoring OS errors.
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        # [FUNCTION DOC] HELPER: Sets the cuda device for this rank (cuda only), creates the
        # process group for the test backend using a file store and a 60 second timeout,
        # and stores the backend implementation for this rank's device on self.backend. Skips
        # the test if windows are unsupported by the distributed build or the backend.
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        self.backend = dist.get_backend_impl(device=self.device)
        if not dist._supports_window() or not self.backend.supports_window:
            self.skipTest(f"{self.backend_name} does not support windows")

    def _make_pool(self):
        # [FUNCTION DOC] HELPER: Returns a CUDA memory pool backed by the backend's memory
        # allocator, used to allocate window-registrable tensors.
        return torch.cuda.MemPool(self.backend.mem_allocator)

    def _probe_window_support(self, pool):
        # [FUNCTION DOC] HELPER: Allocates a one-element tensor from the pool, then tries to
        # register and deregister it in a throwaway window. A runtime error from registration
        # makes the test skip with the error text. Collective: all ranks must call it.
        # Symmetric-window registration needs a transport that can expose
        # VMM-backed memory across ranks (NVLink intra-node or IB inter-node).
        # The probe is collective, so every rank succeeds or skips together.
        with torch.cuda.use_mem_pool(pool):
            probe_buf = torch.empty(1, dtype=torch.float, device=self.device)
        try:
            probe_win = dist._new_window()
            probe_win.tensor_register(probe_buf)
            probe_win.tensor_deregister()
        except RuntimeError as e:
            self.skipTest(f"symmetric window registration unsupported: {e}")

    def _window_ring_put(self, count, dtype, async_op, use_tensor_in_new_window):
        # [FUNCTION DOC] HELPER: Shared ring test body. Allocates a payload of count elements
        # (filled with the rank) and a window buffer of count times world size elements from
        # the pool, creates the window (passing the buffer at creation or registering it
        # afterwards, per use_tensor_in_new_window). For four iterations each rank puts a
        # payload into the next rank's slot, waits for the previous rank's signal (both
        # awaited explicitly if async_op), maps its own window tensor and asserts its slot
        # holds the previous rank's value for that iteration. Deregisters at the end.
        # Ring put: each rank puts its payload into the next rank's window and
        # signals it; each rank waits for the previous rank's signal and
        # verifies the received slice. Port of torchcomms' _window_put_test.
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            input_tensor = torch.full(
                [count], self.rank, dtype=dtype, device=self.device
            )
            win_buf = torch.ones(
                [count * self.world_size], dtype=dtype, device=self.device
            )
        dist.barrier()

        if use_tensor_in_new_window:
            win = dist._new_window(win_buf)
        else:
            win = dist._new_window()
            win.tensor_register(win_buf)
        dist.barrier()

        dst_rank = (self.rank + 1) % self.world_size
        src_rank = (self.rank - 1 + self.world_size) % self.world_size

        for iteration in range(4):
            input_tensor.fill_(self.rank + iteration * self.world_size)
            work = win.put(input_tensor, dst_rank, dst_rank * count, async_op)
            if async_op:
                work.wait()

            wait_work = win.wait_signal(src_rank, async_op)
            if async_op:
                wait_work.wait()

            local_tensor = win.map_remote_tensor(self.rank)
            torch.cuda.synchronize()
            received = local_tensor[self.rank * count : (self.rank + 1) * count]
            expected = torch.full(
                [count],
                src_rank + iteration * self.world_size,
                dtype=dtype,
                device=self.device,
            )
            self.assertEqual(received, expected, msg=f"iteration {iteration}")
            dist.barrier()

        torch.cuda.synchronize()
        win.tensor_deregister()
        del win
        del pool
        torch.cuda.synchronize()

    def test_put_signal_wait_sync(self):
        # [FUNCTION DOC] Ring put of 1024 floats with synchronous put and wait_signal.
        self._init_pg()
        self._window_ring_put(1024, torch.float, False, False)

    def test_put_signal_wait_async(self):
        # [FUNCTION DOC] Ring put of 1024 floats with asynchronous put and wait_signal,
        # waiting on the returned work objects.
        self._init_pg()
        self._window_ring_put(1024, torch.float, True, False)

    def test_signal_wait(self):
        # [FUNCTION DOC] Creates a window over a one-element pooled tensor, signals the next
        # rank and waits for a signal from the previous rank (both synchronous), then
        # synchronizes and deregisters. Passes if nothing hangs or raises.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.zeros(1, device=self.device)
        win = dist._new_window(win_buf)
        win.signal((self.rank + 1) % self.world_size, False)
        win.wait_signal((self.rank - 1) % self.world_size, False)
        torch.cuda.synchronize()
        win.tensor_deregister()

    def test_put_dtypes_and_sizes(self):
        # [FUNCTION DOC] Runs the synchronous ring put for three combinations of element
        # count and dtype: 4 floats, 1024 int32 and 65536 int8.
        self._init_pg()
        for count, dtype in [
            (4, torch.float),
            (1024, torch.int),
            (1024 * 64, torch.int8),
        ]:
            self._window_ring_put(count, dtype, False, False)

    def test_new_window_with_tensor(self):
        # [FUNCTION DOC] Ring put where the buffer is handed to the window constructor
        # rather than registered afterwards.
        self._init_pg()
        self._window_ring_put(1024, torch.float, False, True)

    def test_map_remote_tensor_local(self):
        # [FUNCTION DOC] Registers a pooled arange tensor, maps the window tensor for the
        # caller's own rank, and asserts it has the same dtype, shape and values as the
        # registered buffer. Deregisters afterwards.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        count = 1024
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.arange(count, dtype=torch.float, device=self.device)
        win = dist._new_window()
        win.tensor_register(win_buf)
        dist.barrier()

        local_tensor = win.map_remote_tensor(self.rank)
        self.assertEqual(local_tensor.dtype, win_buf.dtype)
        self.assertEqual(local_tensor.shape, win_buf.shape)
        self.assertEqual(local_tensor, win_buf)

        win.tensor_deregister()

    def test_window_attr(self):
        # [FUNCTION DOC] Creates a window over a pooled tensor and asserts that the
        # attributes queried for the next rank report an access type that is an instance of
        # the WindowAccessType enum.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
        win = dist._new_window(win_buf)
        dist.barrier()
        from torch._C._distributed_c10d import WindowAccessType

        attr = win.get_attr((self.rank + 1) % self.world_size)
        self.assertIsInstance(attr.access_type, WindowAccessType)
        win.tensor_deregister()

    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_register_errors(self):
        # [FUNCTION DOC] Checks runtime errors: registering a tensor not from the backend
        # mempool (message mentions the mempool); put or signal before registration
        # ("not registered"); registering the same pooled tensor twice (double registration).
        # Skipped on ROCm 7.14 and later.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        win = dist._new_window()
        # Registering a tensor outside the backend mempool must fail.
        plain = torch.ones(16, dtype=torch.float, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "mempool|MemPool"):
            win.tensor_register(plain)
        # Ops before registration must fail.
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            win.put(plain, 0, 0, False)
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            win.signal(0, False)
        # Double registration must fail.
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
        win.tensor_register(win_buf)
        dist.barrier()
        with self.assertRaisesRegex(
            RuntimeError, "double registration|Double registration"
        ):
            win.tensor_register(win_buf)
        win.tensor_deregister()

    def test_put_out_of_bounds(self):
        # [FUNCTION DOC] Puts a 16-element source at offset 8 into a 16-element window and
        # expects a runtime error saying the write exceeds the window size.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
            src = torch.ones(16, dtype=torch.float, device=self.device)
        win = dist._new_window(win_buf)
        dist.barrier()
        with self.assertRaisesRegex(RuntimeError, "exceeds the window size"):
            win.put(src, (self.rank + 1) % self.world_size, 8, False)
        win.tensor_deregister()

    def test_put_argument_validation_is_rank_local(self):
        # [FUNCTION DOC] Checks that put validates its arguments locally, raising runtime
        # errors for a negative offset ("nonnegative"), a huge offset (exceeds or overflows),
        # a source dtype differing from the window ("does not match"), and, on rank 0 only,
        # a source tensor outside any registered region ("not in a registered"), without
        # needing the other rank to participate. A barrier precedes deregistration.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(16, dtype=torch.float, device=self.device)
        win = dist._new_window(win_buf)
        dst_rank = (self.rank + 1) % self.world_size

        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            win.put(win_buf[:1], dst_rank, -1, False)
        with self.assertRaisesRegex(RuntimeError, "exceeds|overflows"):
            win.put(win_buf[:1], dst_rank, sys.maxsize, False)
        with torch.cuda.use_mem_pool(pool):
            wrong_dtype = torch.ones(1, dtype=torch.int, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            win.put(wrong_dtype, dst_rank, 0, False)

        # Only one rank calls put. A missing source must fail locally rather than
        # entering collective window registration and mismatching the barrier.
        if self.rank == 0:
            unregistered = torch.ones(1, device=self.device)
            with self.assertRaisesRegex(RuntimeError, "not in a registered"):
                win.put(unregistered, dst_rank, 0, False)
        dist.barrier()
        win.tensor_deregister()

    def test_put_uses_destination_window_metadata(self):
        # [FUNCTION DOC] Each rank allocates a window buffer of 16 plus rank elements after
        # rank-dependent padding, so offsets differ per rank. Each rank puts its first
        # element (rank plus 10) into the last slot of the next rank's window, using the
        # destination's size, then waits for the previous rank's signal and asserts its own
        # last element equals the previous rank's value. Verifies the destination window's
        # metadata, not the local one, determines placement.
        self._init_pg()
        pool = self._make_pool()
        self._probe_window_support(pool)
        with torch.cuda.use_mem_pool(pool):
            # Keep rank-dependent padding alive so the logical tensor starts at
            # a different offset in each rank's registered allocator segment.
            padding = torch.empty(1 + self.rank * 4096, device=self.device)
            win_buf = torch.zeros(16 + self.rank, device=self.device)
        win_buf[0] = self.rank + 10
        win = dist._new_window(win_buf)
        dst_rank = (self.rank + 1) % self.world_size
        src_rank = (self.rank - 1 + self.world_size) % self.world_size
        dst_size = 16 + dst_rank

        win.put(win_buf[:1], dst_rank, dst_size - 1, False)
        win.wait_signal(src_rank, False)
        torch.cuda.synchronize()
        self.assertEqual(win_buf[-1], src_rank + 10)

        win.tensor_deregister()
        del padding


def _make_window_test_class(backend_name, device_type):
    # [FUNCTION DOC] HELPER: Builds a concrete test class combining the abstract mixin with
    # the multi-process test case, sets its backend_name and device_type attributes and a
    # class name derived from the capitalized backend name. Wraps it to skip when the backend
    # is unavailable and, for cuda, when the machine has no CUDA or fewer than two GPUs.
    class WindowTest(AbstractWindowTest, MultiProcessTestCase):
        # [CLASS DOC] HELPER: Concrete window test class; behavior comes entirely from the
        # mixin and base class, attributes are assigned by the enclosing factory.
        pass

    WindowTest.backend_name = backend_name
    WindowTest.device_type = device_type
    WindowTest.__name__ = f"{backend_name.capitalize()}WindowTest"
    WindowTest.__qualname__ = WindowTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(WindowTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "window CUDA tests require at least 2 GPUs",
        )(cls)
    return cls


for backend_name, device_type in WINDOW_BACKENDS:
    globals()[f"{backend_name.capitalize()}WindowTest"] = _make_window_test_class(
        backend_name, device_type
    )


if __name__ == "__main__":
    run_tests()
