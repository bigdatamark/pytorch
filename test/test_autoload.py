# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Test that checks the device backend autoload switch: the environment variable for
# autoload and the variable set by the custom backend extension on import must agree.

import os

from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceBackendAutoload(TestCase):
    # [CLASS DOC] Test case for automatic loading of out-of-tree device backends.
    def test_autoload(self):
        # [FUNCTION DOC] Reads the autoload switch variable and the imported-marker variable from
        # the environment (both default to 0) and asserts the two string values are equal. The
        # marker is set by the custom backend extension package when it is imported.
        switch = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

        # After importing the extension, the value of this environment variable should be true
        # See: test/cpp_extensions/torch_test_cpp_extension/__init__.py
        is_imported = os.getenv("IS_CUSTOM_DEVICE_BACKEND_IMPORTED", "0")

        # Both values should be equal
        self.assertEqual(is_imported, switch)


if __name__ == "__main__":
    run_tests()
