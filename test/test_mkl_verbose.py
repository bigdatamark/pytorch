# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests that the MKL verbose switch controls oneMKL logging. Each test launches the
# helper script mkl_verbose.py in a subprocess at a given verbose level and counts output lines
# that start with the MKL verbose marker. No import-time side effects.

from torch.testing._internal.common_utils import TestCase, run_tests
import os
import subprocess
import sys

class TestMKLVerbose(TestCase):
    # [CLASS DOC] Test case with two subprocess-based checks, one for verbose level one (messages
    # expected) and one for level zero (no messages expected).
    def test_verbose_on(self):
        # [FUNCTION DOC] Runs the helper script in a shell subprocess at verbose level one with
        # stdout and stderr merged, and counts lines starting with the MKL verbose marker. If the
        # line saying MKL could not be put into verbose mode appears, the test returns early
        # (treated as pass). Otherwise it asserts at least one verbose line was seen.
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkl_verbose.py --verbose-level=1', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("MKL_VERBOSE"):
                    num = num + 1
                elif line == 'Failed to set MKL into verbose mode. Please consider to disable this verbose scope.':
                    return
        self.assertTrue(num > 0, 'oneMKL verbose messages not found.')

    def test_verbose_off(self):
        # [FUNCTION DOC] Runs the helper script in a shell subprocess at verbose level zero
        # with output merged, counts lines starting with the MKL verbose marker, and
        # asserts that none appeared.
        num = 0
        loc = os.path.dirname(os.path.abspath(__file__))
        with subprocess.Popen(f'{sys.executable} -u {loc}/mkl_verbose.py --verbose-level=0', shell=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                line = str(line, 'utf-8').strip()
                if line.startswith("MKL_VERBOSE"):
                    num = num + 1
        self.assertEqual(num, 0, 'unexpected oneMKL verbose messages found.')

if __name__ == '__main__':
    run_tests()
