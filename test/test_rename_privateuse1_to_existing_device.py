# Owner(s): ["module: PrivateUse1"]

# [MODULE DOC] Test for renaming the PrivateUse1 backend to a name that is already
# an existing device name. Defines a dummy device module and one test case that
# renames the backend to "maia", checks that a second rename fails, and checks that
# autocast and device-index parsing only work once a device module is registered.
# Importing has no side effects; the rename itself happens inside the test.
import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class DummyPrivateUse1Module:
    # [CLASS DOC] Stand-in for a user supplied PrivateUse1 device module. It has no
    # state and only static methods that report the device as available, report autocast
    # as enabled with float16 as the autocast dtype, ignore attempts to change autocast
    # settings, and list float16 as the only AMP-supported dtype.
    @staticmethod
    def is_available():
        # [FUNCTION DOC] Always reports that the dummy device is available (returns True).
        return True

    @staticmethod
    def is_autocast_enabled():
        # [FUNCTION DOC] Always reports autocast as enabled on the dummy device (returns True).
        return True

    @staticmethod
    def get_autocast_dtype():
        # [FUNCTION DOC] Returns float16 as the autocast dtype of the dummy device.
        return torch.float16

    @staticmethod
    def set_autocast_enabled(enable):
        # [FUNCTION DOC] No-op setter for the autocast enabled flag; the argument is ignored.
        pass

    @staticmethod
    def set_autocast_dtype(dtype):
        # [FUNCTION DOC] No-op setter for the autocast dtype; the argument is ignored.
        pass

    @staticmethod
    def get_amp_supported_dtype():
        # [FUNCTION DOC] Returns a list holding only float16 as the dtypes supported by AMP.
        return [torch.float16]


class TestRenamePrivateuseoneToExistingBackend(TestCase):
    # [CLASS DOC] Test case holding the single test of renaming PrivateUse1 to an
    # existing backend name. The test mutates process-global backend-name state, so it
    # is skipped under TorchDynamo.
    @skipIfTorchDynamo(
        "TorchDynamo exposes https://github.com/pytorch/pytorch/issues/166696"
    )
    def test_external_module_register_with_existing_backend(self):
        # [FUNCTION DOC] Renames the PrivateUse1 backend to "maia" and verifies behavior.
        # Steps: a second rename to another name must raise a RuntimeError saying the name
        # was already set; the C++ backend name must read back as "maia"; the torch module
        # attribute for maia must not exist yet (AttributeError); autocast on that device
        # type must fail with an AssertionError before registration. After registering the
        # dummy module as the maia device module, the attribute works, autocast succeeds,
        # and device index parsing accepts both a string and a device object with an index.
        # Side effect: sets the process-wide PrivateUse1 backend name, which cannot be reset.
        torch.utils.rename_privateuse1_backend("maia")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "maia")

        with self.assertRaises(AttributeError):
            torch.maia.is_available()

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("maia", DummyPrivateUse1Module)

        torch.maia.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("maia:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("maia:2")), 2)


if __name__ == "__main__":
    run_tests()
