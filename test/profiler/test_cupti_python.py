# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests for the pylibcupti wrapper over the real libcupti library, gated on
# libcupti 13.3 or newer. The wrapper symbols are imported at load only when cupti-python is
# available.

import ctypes
import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUPTI, TEST_CUPTI_V13_3
from torch.testing._internal.common_utils import run_tests, TestCase


# These tests drive the real libcupti through _PyLibCupti, so they import its symbols;
# cupti-python imports its enums at load, so guard the import on TEST_CUPTI. TEST_CUPTI_V13_3
# additionally requires a loaded libcupti >= 13.3 (the v2 user-defined-record API the wrapper
# targets) -- e.g. the 13.3 wheel LD_PRELOADed; torch's bundled libcupti is typically 13.1.
if TEST_CUPTI:
    from torch.profiler._cupti.cupti_python import CUPTI_SUCCESS, CuptiError, pylibcupti


@unittest.skipIf(not TEST_CUPTI_V13_3, "requires a loaded libcupti >= 13.3")
class TestPyLibCupti(TestCase):
    # [CLASS DOC] Exercises the version, record accessor, error check, subscribe, activity
    # lifecycle, hardware trace and finalize entry points of the CUPTI wrapper against the live
    # library. Class skipped unless libcupti 13.3 or newer is loaded.
    def test_get_version(self):
        # [FUNCTION DOC] Asserts the reported libcupti version number is at least 130300.
        self.assertGreaterEqual(pylibcupti().get_version(), 130300)

    def test_get_next_record_fn_address(self):
        # [FUNCTION DOC] The next-record function address must be positive. The test then calls
        # it through ctypes with null subscriber and buffer and a zero size and asserts the
        # status is not success rather than crashing.
        # cuptiActivityGetNextRecord_v2 is present on >= 13.2, so its address (used
        # by the native decode worker) is nonzero here.
        addr = pylibcupti().get_next_record_fn_address()
        self.assertGreater(addr, 0)
        # ...and it is the live function, not garbage: invoke it through the address
        # the same way the native worker does -- int(subscriber, buffer, validSize,
        # &record) -> CUptiResult. With no subscriber/buffer CUPTI returns a defined
        # non-success status (CUPTI_ERROR_INVALID_PARAMETER) rather than crashing.
        get_next = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )(addr)
        record = ctypes.c_void_p()
        self.assertNotEqual(
            get_next(None, None, 0, ctypes.byref(record)), CUPTI_SUCCESS
        )

    def test_result_string_and_check(self):
        # [FUNCTION DOC] The error check helper must not raise for the success code, and the
        # result string helper must give CUPTI_SUCCESS for it. A failing code must raise
        # CuptiError whose message holds the operation name and the invalid parameter error
        # string.
        lib = pylibcupti()
        # A successful rc never raises and decodes to CUPTI's real success string --
        # the actual cuptiGetResultString lookup, not the "rc=<n>" fallback.
        lib._check(CUPTI_SUCCESS, "noop")
        self.assertEqual(lib._result_string(CUPTI_SUCCESS), "CUPTI_SUCCESS")
        # A non-success rc raises CuptiError carrying the op name and the decoded
        # CUPTI error string (rc=1 -> CUPTI_ERROR_INVALID_PARAMETER).
        with self.assertRaises(CuptiError) as cm:
            lib._check(1, "deliberately_bad")
        self.assertIn("deliberately_bad", str(cm.exception))
        self.assertIn("CUPTI_ERROR_INVALID_PARAMETER", str(cm.exception))

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_v2_subscribe_timestamp_roundtrip(self):
        # [FUNCTION DOC] Needs CUDA. Subscribes (skips if unsupported), asserts the timestamp
        # for the subscriber is positive and that toggling kernel latency timestamps returns a
        # boolean, then always unsubscribes.
        torch.cuda.init()
        lib = pylibcupti()
        try:
            sub_handle = lib.subscribe()
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            # get_timestamp's _v2 form requires an active subscriber and shares the
            # activity-record timebase, so it returns a positive monotonic value.
            self.assertGreater(lib.get_timestamp(sub_handle), 0)
            # best-effort per-subscriber toggle: returns a bool, never raises.
            self.assertIsInstance(
                lib.enable_kernel_latency_timestamps(sub_handle, True), bool
            )
        finally:
            lib.unsubscribe(sub_handle)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_v2_activity_lifecycle(self):
        # [FUNCTION DOC] Needs CUDA. Walks the subscription-scoped sequence the monitor uses:
        # push and pop an external correlation id and check it round-trips, arm user defined
        # records with the native buffer callback addresses, enable concurrent kernel and
        # runtime kinds with field registry selections, run a kernel, flush, and check the
        # dropped record count is non-negative. Also calls the versioned dropped-count function
        # directly and asserts success, then disables the kinds, disarms, and always
        # unsubscribes.
        # Drive the subscription-scoped wrapper surface against real libcupti, the
        # same sequence the monitor runs: push/pop external correlation, arm UDR
        # with the native buffer callbacks, enable/disable kinds with field
        # selections, flush, and read the dropped-record count.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.records import FIELD_REGISTRY

        monitor_ext = torch._C._profiler._cupti_monitor
        torch.cuda.init()
        lib = pylibcupti()
        try:
            sub = lib.subscribe()
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            # external-correlation push/pop round-trips the id (v2 form, under sub).
            self.assertTrue(
                lib.activity_push_external_correlation_id(4242, sub_handle=sub)
            )
            self.assertEqual(
                lib.activity_pop_external_correlation_id(sub_handle=sub), 4242
            )
            lib.arm_user_defined_records(
                sub,
                monitor_ext.buffer_request_callback_address(),
                monitor_ext.buffer_complete_callback_address(),
            )
            # CONCURRENT_KERNEL + RUNTIME; enabling RUNTIME also exercises
            # disable_noisy_runtime_apis (best-effort, inside activity_enable).
            kinds = [ActivityKind.CONCURRENT_KERNEL, ActivityKind.RUNTIME]
            for kind in kinds:
                lib.activity_enable(sub, kind, FIELD_REGISTRY[kind])
            torch.randn(4096, device="cuda").sum()
            torch.cuda.synchronize()
            lib.activity_flush_all()  # must not raise
            self.assertGreaterEqual(lib.activity_get_num_dropped_records(sub, 0, 0), 0)
            # The count is best-effort -- the wrapper maps any bad status to 0 -- so
            # the count alone can't tell a real read from a rejected one. Assert the
            # status directly: with a subscriber active CUPTI rejects the global-scope
            # accessor, so only the _v2 form may succeed here.
            dropped = ctypes.c_size_t()
            rc = lib._lib.cuptiActivityGetNumDroppedRecords_v2(
                ctypes.c_void_p(sub), None, 0, ctypes.byref(dropped)
            )
            self.assertEqual(rc, CUPTI_SUCCESS)
            for kind in kinds:
                lib.activity_disable(sub, kind)
            lib.disarm_user_defined_records(sub)
        finally:
            lib.unsubscribe(sub)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_activity_enable_hw_trace_toggle(self):
        # [FUNCTION DOC] Needs CUDA. Hardware trace is a global toggle, so a child Python
        # process enables then disables it. A child exit code of two means the platform rejects
        # it and the test skips; any other nonzero code fails with the captured stderr.
        # HW-trace is a GLOBAL toggle -- it perturbs process-wide CUPTI state (and
        # breaks a sibling UDR session), so exercise it in an isolated child that
        # inherits this process's libcupti via LD_PRELOAD: enable then restore to
        # the default disabled state. exit(2) means the platform rejects HW trace.
        script = (
            "import sys, torch\n"
            "torch.cuda.init()\n"
            "from torch.profiler._cupti.cupti_python import pylibcupti, CuptiError\n"
            "lib = pylibcupti()\n"
            "try:\n"
            "    lib.activity_enable_hw_trace(True)\n"
            "except CuptiError:\n"
            "    sys.exit(2)\n"
            "lib.activity_enable_hw_trace(False)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        if proc.returncode == 2:
            self.skipTest("HW trace unsupported on this platform")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_finalize_in_subprocess(self):
        # [FUNCTION DOC] Needs CUDA. Runs the global finalize call in a child Python process so
        # sibling tests are unaffected, and asserts a zero exit code, reporting stderr on
        # failure.
        # finalize() is cuptiFinalize -- a global, process-wide teardown the monitor
        # never calls. Run it in a child (which inherits this process's libcupti via
        # LD_PRELOAD) so it can't tear CUPTI down for sibling tests; a clean exit
        # means the wrapper's call + rc-check succeeded.
        script = (
            "import torch; torch.cuda.init(); "
            "from torch.profiler._cupti.cupti_python import pylibcupti; "
            "pylibcupti().finalize()"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    run_tests()
