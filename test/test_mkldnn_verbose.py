# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests that the oneDNN (MKLDNN) verbose mode switch works. Each test launches the
# helper script mkldnn_verbose.py in a subprocess with a verbose level and counts the
# output lines that begin with the oneDNN verbose marker. No import-time side effects.

from torch.testing._internal.common_utils import TestCase, run_tests
import os
import subprocess
import sys

class TestMKLDNNVerbose(TestCase):
    # [CLASS DOC] Test case holding the two subprocess based checks for oneDNN verbose output
    # (verbose level one must print messages, level zero must print none).
    def test_verbose_on(self):
        # [FUNCTION DOC] Runs the helper script mkldnn_verbose.py with verbose level 1 in a shell
        # subprocess, merging stderr into stdout, and counts output lines starting with the
        # oneDNN verbose marker. If the script prints the message saying MKLDNN could not be
        # put in verbose mode, the test returns early (treated as a skip-like pass). Otherwise
        # it asserts at least one verbose line was seen.
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkldnn_verbose.py --verbose-level=1', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("onednn_verbose"):
                    num = num + 1
                elif line == 'Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope.':
                    return
        self.assertTrue(num > 0, 'oneDNN verbose messages not found.')

    def test_verbose_off(self):
        # [FUNCTION DOC] Runs the helper script mkldnn_verbose.py with verbose level 0 in a shell
        # subprocess (stderr merged into stdout) and asserts that no output line starts with
        # the oneDNN verbose marker.
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkldnn_verbose.py --verbose-level=0', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("onednn_verbose"):
                    num = num + 1
        self.assertEqual(num, 0, 'unexpected oneDNN verbose messages found.')

if __name__ == '__main__':
    run_tests()
