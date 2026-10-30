# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Tests for torch private-use-one device module registration
# (register_device_module, rename_privateuse1_backend) with a dummy backend module providing
# autocast hooks. Defines the dummy module and a TestCase; no import-time side effects beyond
# imports.
import sys

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class DummyPrivateUse1Module:
    # [CLASS DOC] HELPER: Stub backend module used as a stand-in runtime module for the private-
    # use-one device; holds only static methods reporting availability and fixed autocast
    # settings.
    @staticmethod
    def is_available():
        # [FUNCTION DOC] HELPER: Always reports the device as available.
        return True

    @staticmethod
    def is_autocast_enabled():
        # [FUNCTION DOC] HELPER: Always reports autocast as enabled.
        return True

    @staticmethod
    def get_autocast_dtype():
        # [FUNCTION DOC] HELPER: Returns the autocast dtype, always float16.
        return torch.float16

    @staticmethod
    def set_autocast_enabled(enable):
        # [FUNCTION DOC] HELPER: No-op setter for autocast enabled state; the argument is
        # ignored.
        pass

    @staticmethod
    def set_autocast_dtype(dtype):
        # [FUNCTION DOC] HELPER: No-op setter for autocast dtype; the argument is ignored.
        pass

    @staticmethod
    def get_amp_supported_dtype():
        # [FUNCTION DOC] HELPER: Returns a list with float16 as the only AMP-supported dtype.
        return [torch.float16]


class TestExtensionUtils(TestCase):
    # [CLASS DOC] Test case covering registering external device modules for the private-use-one
    # backend and renaming that backend. Each test may leave global state, which tearDown cleans
    # up.
    def tearDown(self):
        # [FUNCTION DOC] Cleanup: reads the current private-use-one backend name, deletes the
        # torch attribute of that name if present and removes the matching torch submodule entry
        # from sys.modules if present. Does not reset the backend name itself.
        # Clean up
        backend_name = torch._C._get_privateuse1_backend_name()
        if hasattr(torch, backend_name):
            delattr(torch, backend_name)
        if f"torch.{backend_name}" in sys.modules:
            del sys.modules[f"torch.{backend_name}"]

    def test_external_module_register(self):
        # [FUNCTION DOC] Checks registration errors and success: registering for a built-in
        # device raises RuntimeError, an invalid device type name raises RuntimeError listing
        # expected names, the module attribute is initially missing (AttributeError),
        # registering the dummy for privateuseone makes it callable, and registering again
        # raises RuntimeError (override unsupported).
        # Built-in module
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("cuda", torch.cuda)

        # Wrong device type
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dummmy", DummyPrivateUse1Module)

        with self.assertRaises(AttributeError):
            torch.privateuseone.is_available()  # type: ignore[attr-defined]

        torch._register_device_module("privateuseone", DummyPrivateUse1Module)

        torch.privateuseone.is_available()  # type: ignore[attr-defined]

        # No supporting for override
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("privateuseone", DummyPrivateUse1Module)

    @skipIfTorchDynamo(
        "accelerator doesn't compose with privateuse1 : https://github.com/pytorch/pytorch/issues/166696"
    )
    def test_external_module_register_with_renamed_backend(self):
        # [FUNCTION DOC] Renames the private-use-one backend to foo (skipped under Dynamo),
        # checks a second rename raises RuntimeError, the name reads back as foo, the module is
        # missing before registration, and autocast with foo raises AssertionError until the
        # dummy module is registered. After registering, autocast works and device index parsing
        # of foo strings and torch.device values gives 1 and 2. Mutates global backend naming.
        torch.utils.rename_privateuse1_backend("foo")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "foo")

        with self.assertRaises(AttributeError):
            torch.foo.is_available()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("foo", DummyPrivateUse1Module)

        torch.foo.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("foo:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("foo:2")), 2)


if __name__ == "__main__":
    run_tests()
