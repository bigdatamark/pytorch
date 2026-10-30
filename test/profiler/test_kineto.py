# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Test that Kineto initialises correctly in daemon mode, instantiated for the cuda
# device type only. Runs subprocesses; no import-time side effects beyond test instantiation.
import os
import subprocess
import sys
from unittest.mock import patch

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleKinetoInitializationTest(TestCase):
    # [CLASS DOC] Device-generic test class checking Kineto initialisation behaviour when the
    # daemon environment variable is set.
    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    def test_kineto_profiler_with_environment_variable(self, device):
        # [FUNCTION DOC] Runs under a patched environment enabling the Kineto daemon. Spawns a
        # Python subprocess that imports torch and initialises the device module when available;
        # a non-zero exit fails the test. Then runs a second process that only loads the torch C
        # shared library via ctypes and asserts stderr does not contain the daemon config loader
        # registration message, so Kineto must not initialise on direct library load.
        """
        This test checks whether kineto works with torch in daemon mode, please refer to issue #112389 and #131020.
        Besides that, this test will also check that kineto will not be initialized when user loads the shared library
        directly.
        """
        device_type = device.split(":")[0]
        script = f"""
import torch
if torch.{device_type}.is_available():
    device_module = torch.{device_type}
    if hasattr(device_module, 'init'):
        device_module.init()
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )
        # import the shared library directly - it triggers static init but doesn't call kineto_init
        env = os.environ.copy()
        env["KINETO_USE_DAEMON"] = "1"
        if "KINETO_DAEMON_INIT_DELAY_S" in env:
            env.pop("KINETO_DAEMON_INIT_DELAY_S")
        _, stderr = TestCase.run_process_no_exception(
            f"from ctypes import CDLL; CDLL('{torch._C.__file__}')"
        )
        self.assertNotRegex(
            stderr.decode("ascii"),
            "Registering daemon config loader",
            "kineto should not be initialized when the shared library is imported directly",
        )


instantiate_device_type_tests(
    SimpleKinetoInitializationTest,
    globals(),
    only_for=("cuda",),
)

if __name__ == "__main__":
    run_tests()
