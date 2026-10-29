# [MODULE DOC] Parser that converts gcov JSON output into CoverageRecord objects.

from __future__ import annotations

from typing import Any

from .coverage_record import CoverageRecord


class GcovCoverageParser:
    # [CLASS DOC] Wraps the parsed gcov JSON of a single test and converts it into a list of
    # CoverageRecord objects via parse. Holds the data it was given and does not mutate it.
    """
    Accepts a parsed json produced by gcov --json-format -- typically,
    representing a single C++ test and produces a list
    of CoverageRecord(s).
    """

    def __init__(self, llvm_coverage: dict[str, Any]) -> None:
        # [FUNCTION DOC] Stores the parsed gcov JSON mapping (the argument is
        # named llvm_coverage).
        self._llvm_coverage = llvm_coverage

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if the path contains third-party and
        # should be skipped.
        """
        Returns True if file path should not be processed.
        This is repo-specific and only makes sense for the current state of
        ovrsource.
        """
        return "third-party" in path

    def parse(self) -> list[CoverageRecord]:
        # [FUNCTION DOC] Builds one CoverageRecord per file in the gcov JSON, skipping third-
        # party files. A line with a zero count is uncovered, otherwise covered. Lines are
        # returned sorted and deduplicated. Returns the list of records.
        # The JSON format is described in the gcov source code
        # https://gcc.gnu.org/onlinedocs/gcc/Invoking-Gcov.html
        records: list[CoverageRecord] = []
        for file_info in self._llvm_coverage["files"]:
            filepath = file_info["file"]
            if self._skip_coverage(filepath):
                continue
            # parse json file
            covered_lines: set[int] = set()
            uncovered_lines: set[int] = set()
            for line in file_info["lines"]:
                line_number = line["line_number"]
                count = line["count"]
                if count == 0:
                    uncovered_lines.update([line_number])
                else:
                    covered_lines.update([line_number])

            records.append(
                CoverageRecord(filepath, sorted(covered_lines), sorted(uncovered_lines))
            )

        return records
