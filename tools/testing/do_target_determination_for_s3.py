# /// script
# dependencies = ["boto3==1.35.42"]
# ///
# [MODULE DOC] Script that runs test target determination (heuristic test prioritization) for
# all discovered tests, downloads the needed historical stats first, prints the aggregated
# result, emits it as metrics when running in CI, and writes td_results.json at the repo root.
# Import time: temporarily puts the repo root on sys.path while importing tools modules.
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import (
    copy_additional_previous_failures,
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
)
from tools.stats.upload_metrics import emit_metric
from tools.testing.discover_tests import TESTS
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    TestPrioritizations,
)


sys.path.remove(str(REPO_ROOT))


def import_results() -> TestPrioritizations:
    # [FUNCTION DOC] Loads previously saved target determination results from the additional
    # CI files directory in the repo and returns them as prioritizations; if the file is
    # missing, prints a notice and returns an empty prioritization.
    if not (REPO_ROOT / ".additional_ci_files/td_results.json").exists():
        print("No TD results found")
        return TestPrioritizations([], {})
    with open(REPO_ROOT / ".additional_ci_files/td_results.json") as f:
        td_results = json.load(f)
        tp = TestPrioritizations.from_json(td_results)

    return tp


def main() -> None:
    # [FUNCTION DOC] Downloads and copies historical test stats and heuristic inputs (network
    # and file writes), computes aggregated heuristics for all discovered tests, prints a
    # summary, emits two metrics when the CI variable equals true, and writes the final
    # prioritizations as td_results.json in the repo root.
    # KNOWN ISSUE: a comment says the metrics are split into 3 but only 2 are emitted; and the
    # initial aggregated heuristics object is immediately overwritten (redundant).
    selected_tests = TESTS

    aggregated_heuristics: AggregatedHeuristics = AggregatedHeuristics(selected_tests)

    get_test_times()
    get_test_class_times()
    get_test_file_ratings()
    get_test_class_ratings()
    get_td_heuristic_historial_edited_files_json()
    get_td_heuristic_profiling_json()
    copy_pytest_cache()
    copy_additional_previous_failures()

    aggregated_heuristics = get_test_prioritizations(selected_tests)

    test_prioritizations = aggregated_heuristics.get_aggregated_priorities()

    print("Aggregated Heuristics")
    print(test_prioritizations.get_info_str(verbose=False))

    if os.getenv("CI") == "true":
        print("Emitting metrics")
        # Split into 3 due to size constraints
        emit_metric(
            "td_results_final_test_prioritizations",
            {"test_prioritizations": test_prioritizations.to_json()},
        )
        emit_metric(
            "td_results_aggregated_heuristics",
            {"aggregated_heuristics": aggregated_heuristics.to_json()},
        )

    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps(test_prioritizations.to_json()))


if __name__ == "__main__":
    main()
