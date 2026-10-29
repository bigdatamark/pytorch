# [MODULE DOC] Script that exports test timing data from test-infra by calling get_test_times
# and get_test_class_times from tools.stats.import_test_stats. Import-time side effect: appends
# the repository root to the module search path so the tools package can be imported.
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.stats.import_test_stats import get_test_class_times, get_test_times


def main() -> None:
    # [FUNCTION DOC] Entry point: print a status message, then fetch and write per-test and
    # per-test-class timing data via the imported helpers. Side effects: network access and
    # file writes performed by those helpers.
    print("Exporting test times from test-infra")
    get_test_times()
    get_test_class_times()


if __name__ == "__main__":
    main()
