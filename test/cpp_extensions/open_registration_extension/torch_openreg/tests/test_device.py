# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Device management tests for the openreg backend: device count (two),
# switching and context managers via torch.accelerator, device capability, device
# objects, tensor placement and fork poisoning. No import-time side effects.

import multiprocessing

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase


class TestDevice(TestCase):
    # [CLASS DOC] Test cases for openreg device queries, selection and initialization.
    # Several tests change the current device index; some restore it, some do not.
    def test_device_count(self):
        # [FUNCTION DOC] The accelerator device count must be exactly two.
        """Test device count query"""
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

    def test_device_switch(self):
        # [FUNCTION DOC] Sets current device index to one then zero and checks it is read back.
        """Test switching between devices"""
        torch.accelerator.set_device_index(1)
        self.assertEqual(torch.accelerator.current_device_index(), 1)

        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_device_context(self):
        # [FUNCTION DOC] device_index context with None leaves the current device unchanged;
        # with index one it switches inside and restores the original on exit.
        """Test device context manager"""
        device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), device)
        self.assertEqual(torch.accelerator.current_device_index(), device)

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        self.assertEqual(torch.accelerator.current_device_index(), device)

    def test_invalid_device_index(self):
        # [FUNCTION DOC] Setting device index two raises RuntimeError "The device index is out
        # of range".
        """Test error handling for invalid device index"""
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.accelerator.set_device_index(2)

    def test_device_capability(self):
        # [FUNCTION DOC] The capability of openreg:0 must list every dtype from the full dtype
        # set (including complex32 and quantized types) as supported.
        capability = torch.accelerator.get_device_capability("openreg:0")
        supported_dtypes = capability["supported_dtypes"]
        expected_dtypes = get_all_dtypes(include_complex32=True, include_qint=True)

        self.assertTrue(all(dtype in supported_dtypes for dtype in expected_dtypes))

    def test_device_properties(self):
        # [FUNCTION DOC] Device objects built from "openreg:0" and "openreg" have type openreg
        # and index zero or None respectively.
        """Test device properties"""
        device = torch.device("openreg:0")
        self.assertEqual(device.type, "openreg")
        self.assertEqual(device.index, 0)

        device = torch.device("openreg")
        self.assertEqual(device.type, "openreg")
        self.assertIsNone(device.index)

    def test_tensor_device(self):
        # [FUNCTION DOC] Tensors created on "openreg" and "openreg:1" report type openreg and
        # the requested index.
        """Test tensor device assignment"""
        x = torch.randn(2, 3, device="openreg")
        self.assertEqual(x.device.type, "openreg")

        x = torch.randn(2, 3, device="openreg:1")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.device.index, 1)

    def test_device_guard(self):
        # [FUNCTION DOC] Entering device_index one switches the current device and exit
        # restores it.
        """Test device guard context manager"""
        original_device = torch.accelerator.current_device_index()

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)

        self.assertEqual(torch.accelerator.current_device_index(), original_device)

    def test_device_switch_persistence(self):
        # [FUNCTION DOC] After setting device one, new tensors on plain "openreg" land on
        # index one; the finally block restores the old index.
        """Test that device switch persists across operations"""
        old_index = torch.accelerator.current_device_index()
        try:
            torch.accelerator.set_device_index(1)

            x = torch.randn(2, 3, device="openreg")
            self.assertEqual(x.device.index, 1)

            y = torch.randn(3, 3, device="openreg")
            self.assertEqual(y.device.index, 1)
        finally:
            # reset default device index
            torch.accelerator.set_device_index(old_index)

    def test_device_count_consistency(self):
        # [FUNCTION DOC] Device count is two, and every index below it can be set and read back.
        # Side effect: leaves the last index (one) as the current device.
        """Test device count consistency"""
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

        # Test that we can access all devices
        for i in range(count):
            torch.accelerator.set_device_index(i)
            self.assertEqual(torch.accelerator.current_device_index(), i)

    @skipIfWindows(msg="Fork not available on Windows")
    def test_device_poison_fork(self):
        # [FUNCTION DOC] Initializes openreg in the parent, then forks a child that tries to
        # init again and sends the exception through a queue; the parent requires a RuntimeError
        # saying OpenReg cannot be re-initialized in a forked subprocess and to use spawn.
        # Skipped on Windows.
        # First, initialize in the parent process
        torch.openreg.init()

        def child(q):
            # [FUNCTION DOC] HELPER: Child-process body: re-initializes openreg and puts any
            # raised exception on the queue.
            try:
                # Second, try to initialize in the child process
                torch.openreg.init()
            except Exception as e:
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()

        self.assertTrue(not q.empty())

        exc = q.get()
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Cannot re-initialize OpenReg in forked subprocess. "
                "To use OpenReg with multiprocessing, you must use the 'spawn' start method"
            ),
        ):
            raise exc


if __name__ == "__main__":
    run_tests()
