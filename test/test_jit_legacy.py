# Owner(s): ["oncall: jit"]
# [MODULE DOC] Runs the whole test_jit suite under the legacy JIT graph executor. At import it
# appends the legacy executor flag to the command line arguments, then (when run as a script)
# parses the command line so the executor setting is known before test classes are created.
# It star-imports every test from test_jit, so those tests are collected here too. When run as
# a script it calls the test runner, except on Python 3.14 or newer where nothing is run.

import sys
sys.argv.append("--jit-executor=legacy")
from torch.testing._internal.common_utils import parse_cmd_line_args, run_tests

if __name__ == '__main__':
    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from test_jit import *  # noqa: F403

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
