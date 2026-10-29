# [MODULE DOC] Entry point for target determination: runs every registered heuristic over a
# list of tests and aggregates their prioritization results.
from __future__ import annotations

import sys
from typing import Any

from tools.testing.target_determination.heuristics import (
    AggregatedHeuristics,
    HEURISTICS,
    TestPrioritizations,
)


def get_test_prioritizations(
    tests: list[str], file: Any = sys.stdout
) -> AggregatedHeuristics:
    # [FUNCTION DOC] Prints the received test names to the given stream (default standard
    # output), then asks each registered heuristic for its prediction confidence over the
    # tests and adds it to an aggregate, printing each heuristic's summary. An exception from
    # one heuristic is caught and printed so the rest still run. Returns the aggregate.
    aggregated_results = AggregatedHeuristics(tests)
    print(f"Received {len(tests)} tests to prioritize", file=file)
    for test in tests:
        print(f"  {test}", file=file)

    for heuristic in HEURISTICS:
        try:
            new_rankings: TestPrioritizations = heuristic.get_prediction_confidence(
                tests
            )
            aggregated_results.add_heuristic_results(heuristic, new_rankings)

            print(f"Results from {heuristic.__class__.__name__}")
            print(new_rankings.get_info_str(verbose=False), file=file)
        except Exception as e:
            print(f"Error in {heuristic.__class__.__name__}: {e}", file=file)

    return aggregated_results
