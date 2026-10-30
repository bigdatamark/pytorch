# Owner(s): ["module: onnx"]
# [MODULE DOC] Test that importing torch.onnx does not eagerly import the heavy optional packages
# onnxruntime and onnxscript. Defines one test class; running the file as a script invokes the
# common test runner.

import subprocess
import sys
import tempfile

import pytorch_test_common

from torch.testing._internal import common_utils


class TestLazyONNXPackages(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Export test case verifying lazy import behavior of ONNX-related packages; each
    # test spawns a fresh interpreter, so no state is shared between tests.
    def _test_package_is_lazily_imported(self, pkg, torch_pkg="torch.onnx"):
        # [FUNCTION DOC] HELPER: Runs a subprocess Python interpreter with import timing enabled
        # that imports torch.onnx, in a temporary working directory, and fails the test if the
        # given package name (preceded by a space) appears in the captured stderr import-time log.
        # The torch_pkg argument is unused. The subprocess is checked, so a nonzero exit raises
        # CalledProcessError.
        with tempfile.TemporaryDirectory() as wd:
            r = subprocess.run(
                [sys.executable, "-Ximporttime", "-c", "import torch.onnx"],
                capture_output=True,
                text=True,
                cwd=wd,
                check=True,
            )

        # The extra space makes sure we're checking the package, not any package containing its name.
        self.assertTrue(
            f" {pkg}" not in r.stderr,
            f"`{pkg}` should not be imported, full importtime: {r.stderr}",
        )

    def test_onnxruntime_is_lazily_imported(self):
        # [FUNCTION DOC] Asserts that onnxruntime is not imported as a side effect of importing
        # torch.onnx.
        self._test_package_is_lazily_imported("onnxruntime")

    def test_onnxscript_is_lazily_imported(self):
        # [FUNCTION DOC] Asserts that onnxscript is not imported as a side effect of importing
        # torch.onnx.
        self._test_package_is_lazily_imported("onnxscript")


if __name__ == "__main__":
    common_utils.run_tests()
