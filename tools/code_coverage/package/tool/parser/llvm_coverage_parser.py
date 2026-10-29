# [MODULE DOC] Parser that converts llvm-cov export JSON into CoverageRecord objects.

from __future__ import annotations

from typing import Any

from .coverage_record import CoverageRecord
from .llvm_coverage_segment import LlvmCoverageSegment, parse_segments


class LlvmCoverageParser:
    # [CLASS DOC] Wraps the parsed llvm-cov JSON of a single test and converts it into a list
    # of CoverageRecord objects via parse. Holds the data it was given and does not mutate it.
    """
    Accepts a parsed json produced by llvm-cov export -- typically,
    representing a single C++ test and produces a list
    of CoverageRecord(s).

    """

    def __init__(self, llvm_coverage: dict[str, Any]) -> None:
        # [FUNCTION DOC] Stores the parsed llvm-cov export JSON mapping.
        self._llvm_coverage = llvm_coverage

    @staticmethod
    def _skip_coverage(path: str) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if the path is inside a third-party folder and
        # should be skipped.
        """
        Returns True if file path should not be processed.
        This is repo-specific and only makes sense for the current state of
        ovrsource.
        """
        return "/third-party/" in path

    @staticmethod
    def _collect_coverage(
        segments: list[LlvmCoverageSegment],
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] HELPER: Walks the ordered segments, comparing each with the previous
        # one to collect covered and uncovered line numbers. Lines that are covered anywhere
        # are removed from the uncovered set. Returns both lists sorted.
        """
        Stateful parsing of coverage segments.
        """
        covered_lines: set[int] = set()
        uncovered_lines: set[int] = set()
        prev_segment = LlvmCoverageSegment(1, 0, 0, 0, 0, None)
        for segment in segments:
            covered_range, uncovered_range = segment.get_coverage(prev_segment)
            covered_lines.update(covered_range)
            uncovered_lines.update(uncovered_range)
            prev_segment = segment

        uncovered_lines.difference_update(covered_lines)
        return sorted(covered_lines), sorted(uncovered_lines)

    def parse(self, repo_name: str) -> list[CoverageRecord]:
        # [FUNCTION DOC] Builds one CoverageRecord per file in each export unit of
        # the llvm-cov JSON, skipping third-party files, using the file's
        # segments. Returns the list of records.
        # KNOWN ISSUE: repo_name is unused, and the missing-filename check runs after the
        # third-party check, which would fail on a None filename.
        # The JSON format is described in the LLVM source code
        # https://github.com/llvm-mirror/llvm/blob/master/tools/llvm-cov/CoverageExporterJson.cpp
        records: list[CoverageRecord] = []
        for export_unit in self._llvm_coverage["data"]:
            for file_info in export_unit["files"]:
                filepath = file_info["filename"]
                if self._skip_coverage(filepath):
                    continue

                if filepath is None:
                    continue

                segments = file_info["segments"]

                covered_lines, uncovered_lines = self._collect_coverage(
                    parse_segments(segments)
                )

                records.append(CoverageRecord(filepath, covered_lines, uncovered_lines))

        return records
