# Owner(s): ["oncall: jit"]
# [MODULE DOC] Smoke test for the lazy tensor metrics bindings. Defines one module-level test
# function checking that the lazy metrics counter name list is empty in a fresh process, and runs
# the shared test runner when executed directly.

import torch._lazy.metrics
from torch.testing._internal.common_utils import run_tests


def test_metrics():
    # [FUNCTION DOC] Reads the list of lazy metrics counter names and raises an assertion
    # error (including the names) if any counter exists; passes when no lazy computation has
    # recorded one.
    names = torch._lazy.metrics.counter_names()
    if len(names) != 0:
        raise AssertionError(f"Expected no counter names, but got {names}")


if __name__ == "__main__":
    run_tests()
