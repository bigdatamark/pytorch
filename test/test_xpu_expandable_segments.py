# Owner(s): ["module: intel"]
# [MODULE DOC] XPU variant of the test_xpu suite run with the expandable-segments allocator. Re-
# exports the TestXpu and TestXpuOpsXPU test classes from test_xpu and, when run as a script on
# a machine with an available XPU and not on Windows, loads the disabled-tests list, turns on
# the allocator setting expandable_segments and sets the TestXpu class flag expandable_segments
# before running all tests. Import has the side effect of temporarily adding the repo root to
# the module search path to import tools.stats helpers.
import pathlib
import sys

from test_xpu import TestXpu, TestXpuOpsXPU  # noqa: F401

import torch
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import get_disabled_tests


sys.path.remove(str(REPO_ROOT))

if __name__ == "__main__":
    if torch.xpu.is_available() and not IS_WINDOWS:
        get_disabled_tests(".")

        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        TestXpu.expandable_segments = True

        run_tests()
