# Owner(s): ["oncall: jit"]
# [MODULE DOC] Runs the whole TorchScript test suite from test_jit under the simple JIT executor.
# It appends a command line flag selecting that executor to the argument list before the star
# import of test_jit, so every imported test case runs with that executor mode. When executed
# as a script it launches the test runner, but only on Python versions older than 3.14.

import sys
sys.argv.append("--jit-executor=simple")
from test_jit import *  # noqa: F403

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
