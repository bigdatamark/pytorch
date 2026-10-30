# Owner(s): ["module: tests"]
# [MODULE DOC] Tests for the generic torch accelerator API (device index, stream and event
# handling, context managers, memory statistics, device capability) that run on whichever
# accelerator backend is present. Import-time side effects: if no accelerator is detected it
# prints a message to standard error, replaces the test case base with a no-op one and exits the
# process. Device-specific variants (all but cpu, with MPS and XPU allowed) are generated into
# this module by the device type test instantiation.

import gc
import operator
import sys
import unittest
from contextlib import nullcontext

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    TEST_ACCELERATOR,
    TEST_MPS,
    TEST_MULTIACCELERATOR,
    TEST_XPU,
    TestCase,
)


if not TEST_ACCELERATOR:
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest
    # Skip because failing when run on cuda build with no GPU, see #150059 for example
    sys.exit()


class TestAccelerator(TestCase):
    # [CLASS DOC] Test case for the accelerator API; each test acts on the current accelerator,
    # with the device type attribute supplied by the instantiation helper. Some tests require
    # several accelerators or exclude MPS or XPU.
    def test_current_accelerator(self):
        # [FUNCTION DOC] Checks the accelerator is available, that the current accelerator type
        # equals the test device type with no index set, and that setting the device index with a
        # cpu device raises ValueError mentioning a mismatch with the current accelerator.
        self.assertTrue(torch.accelerator.is_available())
        self.assertEqual(torch.accelerator.current_accelerator().type, self.device_type)
        self.assertIsNone(torch.accelerator.current_accelerator().index)
        with self.assertRaisesRegex(
            ValueError, "doesn't match the current accelerator"
        ):
            torch.accelerator.set_device_index("cpu")

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_generic_multi_device_behavior(self):
        # [FUNCTION DOC] Needs at least two accelerators. Switches the current device index to the
        # next device, then back, verifying the index each time; then sets a stream created on the
        # other device as current and checks the current device follows it and stays there after
        # synchronizing the original device.
        orig_device = torch.accelerator.current_device_index()
        target_device = (orig_device + 1) % torch.accelerator.device_count()

        torch.accelerator.set_device_index(target_device)
        self.assertEqual(target_device, torch.accelerator.current_device_index())
        torch.accelerator.set_device_index(orig_device)
        self.assertEqual(orig_device, torch.accelerator.current_device_index())

        s1 = torch.Stream(target_device)
        torch.accelerator.set_stream(s1)
        self.assertEqual(target_device, torch.accelerator.current_device_index())
        torch.accelerator.synchronize(orig_device)
        self.assertEqual(target_device, torch.accelerator.current_device_index())

    def test_generic_stream_behavior(self):
        # [FUNCTION DOC] Makes one stream current, computes a sum on the host, then on a second
        # stream copies both operands to the accelerator without blocking, switches back to the
        # first stream, records and waits on an event on the second stream, adds on the device,
        # records again, synchronizes, and checks the event completed and the device result equals
        # the host result.
        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event = torch.Event()
        a = torch.randn(1000)
        b = torch.randn(1000)
        c = a + b
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream(), s2)
        a_acc = a.to(torch.accelerator.current_accelerator(), non_blocking=True)
        b_acc = b.to(torch.accelerator.current_accelerator(), non_blocking=True)
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        torch.accelerator.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(c_acc.cpu(), c)

    def test_current_stream_query(self):
        # [FUNCTION DOC] Checks the current stream can be queried by device object, device index
        # or device string and always gives the same stream, and that querying with a cpu device
        # raises ValueError about a mismatch with the current accelerator.
        s = torch.accelerator.current_stream()
        self.assertEqual(torch.accelerator.current_stream(s.device), s)
        self.assertEqual(torch.accelerator.current_stream(s.device.index), s)
        self.assertEqual(torch.accelerator.current_stream(str(s.device)), s)
        other_device = torch.device("cpu")
        with self.assertRaisesRegex(
            ValueError, "doesn't match the current accelerator"
        ):
            torch.accelerator.current_stream(other_device)

    def test_stream_compare_with_non_stream(self):
        # [FUNCTION DOC] Regression check for comparing a stream with other values: equality with
        # ints, strings, floats, None and plain objects is False and inequality True (membership
        # in a mixed list works); ordering comparisons between two streams raise TypeError; a
        # second stream constructed from the same id, device index and device type compares equal.
        # Comparing with a non-Stream must not blindly cast it to THPStream*
        # and read garbage fields (#188033)
        s1 = torch.Stream()
        for other in (42, "hello", 3.14, None, object()):
            self.assertFalse(s1 == other)
            self.assertTrue(s1 != other)
        self.assertTrue(s1 in [1, "a", s1])
        s2 = torch.Stream()
        for op in (operator.lt, operator.le, operator.gt, operator.ge):
            with self.assertRaises(TypeError):
                op(s1, s2)
        sid, di, dt = s1.stream_id, s1.device_index, s1.device_type
        same = torch.Stream(stream_id=sid, device_index=di, device_type=dt)
        self.assertTrue(s1 == same)
        self.assertFalse(s1 != same)

    def test_device_context_manager(self):
        # [FUNCTION DOC] Checks the device index context manager with no index leaves the current
        # device unchanged, and with index zero switches to device zero inside the block and
        # restores the previous index afterwards.
        prev_device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), prev_device)
        self.assertEqual(torch.accelerator.current_device_index(), prev_device)
        with torch.accelerator.device_index(0):
            self.assertEqual(torch.accelerator.current_device_index(), 0)
        self.assertEqual(torch.accelerator.current_device_index(), prev_device)

    def test_device_index_fullgraph(self):
        # [FUNCTION DOC] Compiles a function that uses the device index context manager with the
        # device of its input, using the eager backend and requiring a single graph, and checks
        # the compiled result equals the uncompiled result on device zero.
        def fn(x):
            # [FUNCTION DOC] HELPER: Enters the device index context for the input tensor device,
            # computes the sine of the input plus one inside it, and returns the result.
            with torch.accelerator.device_index(x.device.index):
                x = torch.sin(x + 1)
            return x

        x = torch.randn((2, 2), device=0)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_multi_device_context_manager(self):
        # [FUNCTION DOC] Needs at least two accelerators. Sets device zero current, enters the
        # context manager for device one, checks the current index is one inside and zero after
        # leaving.
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        with torch.accelerator.device_index(dst_device):
            self.assertEqual(torch.accelerator.current_device_index(), dst_device)
        self.assertEqual(torch.accelerator.current_device_index(), src_device)

    def test_stream_context_manager(self):
        # [FUNCTION DOC] Checks that using a new stream as a context manager makes it the current
        # stream inside the block and restores the previous current stream afterwards.
        prev_stream = torch.accelerator.current_stream()
        with torch.Stream() as s:
            self.assertEqual(torch.accelerator.current_stream(), s)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    def test_stream_context_manager_reentrance(self):
        # [FUNCTION DOC] Checks the stream context manager can be entered twice on the same stream
        # and nested across different streams (including re-entering an outer stream), with the
        # current stream correct at each level and the original stream restored at the end.
        prev_stream = torch.accelerator.current_stream()
        s0 = torch.Stream()
        with s0, s0:
            self.assertEqual(torch.accelerator.current_stream(), s0)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)
        s1 = torch.Stream()
        with s0:
            self.assertEqual(torch.accelerator.current_stream(), s0)
            with s1:
                self.assertEqual(torch.accelerator.current_stream(), s1)
                with s0:
                    self.assertEqual(torch.accelerator.current_stream(), s0)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_multi_device_stream_context_manager(self):
        # [FUNCTION DOC] Needs at least two accelerators. With device zero current, enters a
        # stream created on device one and checks the current device and stream switch to it while
        # the device zero current stream is unchanged; after leaving, checks the device index and
        # both current streams are restored.
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        src_prev_stream = torch.accelerator.current_stream()
        dst_prev_stream = torch.accelerator.current_stream(dst_device)
        with torch.Stream(dst_device) as dst_stream:
            self.assertEqual(torch.accelerator.current_device_index(), dst_device)
            self.assertEqual(torch.accelerator.current_stream(), dst_stream)
            self.assertEqual(
                torch.accelerator.current_stream(src_device), src_prev_stream
            )
        self.assertEqual(torch.accelerator.current_device_index(), src_device)
        self.assertEqual(torch.accelerator.current_stream(), src_prev_stream)
        self.assertEqual(torch.accelerator.current_stream(dst_device), dst_prev_stream)

    @unittest.skipIf(TEST_MPS, "MPS doesn't support pin memory!")
    def test_pin_memory_on_non_blocking_copy(self):
        # [FUNCTION DOC] Skipped on MPS. Copies a host tensor to the accelerator, then back to the
        # host without blocking, synchronizes, and checks the resulting host tensor is pinned and
        # equal to the device data.
        t_acc = torch.randn(100).to(torch.accelerator.current_accelerator())
        t_host = t_acc.to("cpu", non_blocking=True)
        torch.accelerator.synchronize()
        self.assertTrue(t_host.is_pinned())
        self.assertEqual(t_acc.cpu(), t_host)

    def test_generic_event_behavior(self):
        # [FUNCTION DOC] Checks that asking for elapsed time between two events raises ValueError
        # when timing was not enabled (explicitly or by default), and when timing is enabled but
        # the events were not recorded; each message is matched by regular expression.
        event1 = torch.Event(enable_timing=False)
        event2 = torch.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.Event(enable_timing=True)
        event2 = torch.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be recorded before calculating elapsed time",
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.Event()
        event2 = torch.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

    @unittest.skipIf(TEST_MPS, "MPS doesn't support torch.accelerator memory API!")
    def test_memory_stats(self):
        # [FUNCTION DOC] Skipped on MPS. Allocates and frees a tensor to initialize the device
        # allocator, collects garbage, checks the allocator is initialized and empties the cache.
        # Verifies the memory statistics mapping has every combination of the four byte
        # categories, three pools and four metrics, then exercises allocated, reserved and peak
        # counters (grow after an allocation, return to baseline after freeing and resetting
        # peaks). Finally allocates exactly 1024 bytes and checks the active bytes current counter
        # rises by 1024 with freed still zero, and that after freeing the freed counter equals
        # 1024 and peak values are restored.
        # Ensure that device allocator is initialized
        acc = torch.accelerator.current_accelerator()
        tmp = torch.randn(100, device=acc)
        del tmp
        gc.collect()
        self.assertTrue(torch._C._accelerator_isAllocatorInitialized())
        torch.accelerator.empty_cache()

        pool_type = ["all", "small_pool", "large_pool"]
        metric_type = ["peak", "current", "allocated", "freed"]
        stats_type = [
            "allocated_bytes",
            "reserved_bytes",
            "active_bytes",
            "requested_bytes",
        ]
        mem_stats = torch.accelerator.memory_stats()
        expected_stats = [
            f"{st}.{pt}.{mt}"
            for st in stats_type
            for pt in pool_type
            for mt in metric_type
        ]
        missing_stats = [stat for stat in expected_stats if stat not in mem_stats]
        self.assertEqual(
            len(missing_stats),
            0,
            lambda msg: f"{msg}\nMissing expected memory statistics: {missing_stats}",
        )

        prev_allocated = torch.accelerator.memory_allocated()
        prev_reserved = torch.accelerator.memory_reserved()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        self.assertGreaterEqual(prev_allocated, 0)
        self.assertGreaterEqual(prev_reserved, 0)
        self.assertGreater(prev_max_allocated, 0)
        self.assertGreater(prev_max_reserved, 0)
        tmp = torch.ones(256, device=acc)
        self.assertGreater(torch.accelerator.memory_allocated(), prev_allocated)
        self.assertGreaterEqual(torch.accelerator.memory_reserved(), prev_reserved)
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.memory_allocated(), prev_allocated)
        self.assertEqual(torch.accelerator.memory_reserved(), prev_reserved)
        torch.accelerator.reset_accumulated_memory_stats()
        prev_max_allocated = torch.accelerator.max_memory_allocated()
        prev_max_reserved = torch.accelerator.max_memory_reserved()
        # Activate 1kB memory
        prev_active_current = torch.accelerator.memory_stats()[
            "active_bytes.all.current"
        ]
        tmp = torch.randn(256, device=acc)
        # Detect if the current active memory is 1kB
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            1024 + prev_active_current,
        )
        self.assertEqual(torch.accelerator.memory_stats()["active_bytes.all.freed"], 0)
        del tmp
        gc.collect()
        torch.accelerator.empty_cache()
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.current"],
            prev_active_current,
        )
        self.assertEqual(
            torch.accelerator.memory_stats()["active_bytes.all.freed"], 1024
        )
        torch.accelerator.reset_peak_memory_stats()
        self.assertEqual(torch.accelerator.max_memory_allocated(), prev_max_allocated)
        self.assertEqual(torch.accelerator.max_memory_reserved(), prev_max_reserved)

    @unittest.skipIf(TEST_MPS, "MPS doesn't support torch.accelerator memory API!")
    def test_get_memory_info(self):
        # [FUNCTION DOC] Skipped on MPS. Calls the memory info query and checks both free and
        # total byte counts are non-negative.
        free_bytes, total_bytes = torch.accelerator.get_memory_info()
        self.assertGreaterEqual(free_bytes, 0)
        self.assertGreaterEqual(total_bytes, 0)

    @unittest.skipIf(
        TEST_XPU,
        "bare-bones/opaque bit-field dtypes cause undefined behavior on XPU, see https://github.com/pytorch/pytorch/issues/179888",
    )
    def test_device_capability_supported_dtypes(self):
        # [FUNCTION DOC] Skipped on XPU. Queries the device capability (skips the test on
        # RuntimeError) and checks the supported dtype entry is a non-empty set. For every dtype
        # attribute of the torch module, in a subtest, creates an empty tensor of that dtype on
        # the accelerator and converts it to a reference supported dtype and back; this must
        # succeed for supported dtypes and raise RuntimeError or TypeError for unsupported ones.
        try:
            caps = torch.accelerator.get_device_capability()
        except RuntimeError:
            self.skipTest("Backend doesn't support get_device_capability")

        supported_dtypes = caps["supported_dtypes"]
        self.assertIsInstance(supported_dtypes, set)
        self.assertGreater(len(supported_dtypes), 0)

        acc = torch.accelerator.current_accelerator()
        reference_dtype = next(iter(supported_dtypes))

        all_dtypes = [
            getattr(torch, name)
            for name in dir(torch)
            if isinstance(getattr(torch, name), torch.dtype)
        ]
        for dtype in all_dtypes:
            with self.subTest(dtype=dtype):
                ctx = (
                    nullcontext()
                    if dtype in supported_dtypes
                    else self.assertRaises((RuntimeError, TypeError))
                )
                with ctx:
                    t = torch.empty(16, dtype=dtype, device=acc)
                    t = t.to(reference_dtype)
                    t = t.to(dtype)


instantiate_device_type_tests(
    TestAccelerator,
    globals(),
    except_for=("cpu",),
    allow_mps=True,
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
