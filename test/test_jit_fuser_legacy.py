# Owner(s): ["oncall: jit"]

# [MODULE DOC] Runs the whole fuser test suite from test_jit_fuser under the legacy graph
# executor. At import time it appends the legacy executor flag to the process arguments, and
# when run as a script it parses command line args first so the executor setting is in effect
# before the star-imported tests are instantiated. Defines no tests of its own.
import sys
sys.argv.append("--jit-executor=legacy")

if __name__ == "__main__":
    from torch.testing._internal.common_utils import parse_cmd_line_args

    # The value of GRAPH_EXECUTOR depends on command line arguments so make sure they're parsed
    # before instantiating tests.
    parse_cmd_line_args()

from test_jit_fuser import *  # noqa: F403

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
