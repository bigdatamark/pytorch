# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the DeviceGuard context manager from torch._dynamo.device_interface.
# One suite uses a Mock device interface; the other is instantiated per available device
# type (including MPS and XPU) and uses the real interface. Nothing runs at import time
# except test class generation.
from unittest.mock import Mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.device_interface import DeviceGuard, get_interface_for_device
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class TestDeviceGuard(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests DeviceGuard against a Mock interface whose exchange_device returns
    # 0 and maybe_exchange_device returns 1, so calls and index bookkeeping can be checked.
    """
    Unit tests for the DeviceGuard class using a mock DeviceInterface.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Create a Mock device interface where exchange_device
        # reports previous index 0 and maybe_exchange_device reports index 1.
        super().setUp()
        self.device_interface = Mock()

        self.device_interface.exchange_device = Mock(return_value=0)
        self.device_interface.maybe_exchange_device = Mock(return_value=1)

    def test_device_guard(self):
        # [FUNCTION DOC] With index 1, entering must call exchange_device once with 1 and
        # record prev_idx 0; exiting must call maybe_exchange_device once with 0. The idx
        # stays 1 afterwards only because the mock returns 1 from that call.
        device_guard = DeviceGuard(self.device_interface, 1)

        with device_guard as _:
            self.device_interface.exchange_device.assert_called_once_with(1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.device_interface.maybe_exchange_device.assert_called_once_with(0)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    def test_device_guard_no_index(self):
        # [FUNCTION DOC] With no index, neither exchange_device nor maybe_exchange_device
        # may be called, and prev_idx stays -1 and idx stays None inside and after the guard.
        device_guard = DeviceGuard(self.device_interface, None)

        with device_guard as _:
            self.device_interface.exchange_device.assert_not_called()
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.device_interface.maybe_exchange_device.assert_not_called()
        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


class TestDeviceGuardWithInterface(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic tests (instantiated per device type below) of DeviceGuard
    # with the real interface for that device type; test methods receive a device string.
    """
    Unit tests for the DeviceGuard class using a real DeviceInterface.
    """

    def test_device_guard_no_index(self, device):
        # [FUNCTION DOC] Looks up the interface for the device type, builds a guard with
        # no index, and checks the current device is unchanged inside the guard and that
        # prev_idx is -1 and idx is None both inside and after it.
        device_interface = get_interface_for_device(torch.device(device).type)
        current_device = device_interface.current_device()

        device_guard = DeviceGuard(device_interface, None)

        with device_guard as _:
            self.assertEqual(device_interface.current_device(), current_device)
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


instantiate_device_type_tests(
    TestDeviceGuardWithInterface, globals(), allow_mps=True, allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
