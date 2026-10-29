# [MODULE DOC] Defines LlvmCoverageSegment (one entry of the llvm-cov segment list) and the
# function that builds segments from raw JSON arrays.

from __future__ import annotations

from typing import NamedTuple


class LlvmCoverageSegment(NamedTuple):
    # [CLASS DOC] Immutable record of one llvm-cov segment: line, column,
    # execution count, whether the count is meaningful, whether it starts a
    # region, and an optional gap-entry flag. Provides helpers to compute covered
    # and uncovered line ranges between consecutive segments.
    line: int
    col: int
    segment_count: int
    has_count: int
    is_region_entry: int
    is_gap_entry: int | None

    @property
    def has_coverage(self) -> bool:
        # [FUNCTION DOC] Property: True when the segment's execution count is above zero.
        return self.segment_count > 0

    @property
    def is_executable(self) -> bool:
        # [FUNCTION DOC] Property: True when the segment has a count at all, meaning that code
        # is executable there.
        return self.has_count > 0

    def get_coverage(
        self, prev_segment: LlvmCoverageSegment
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Computes the lines between the previous segment and this one. Returns
        # a pair of lists (covered lines, uncovered lines): empty lists when the previous
        # segment is not executable; otherwise the line range from the previous segment's line
        # up to this segment's line (inclusive when this segment starts after column one) goes
        # to covered or uncovered depending on the previous segment's count.
        # Code adapted from testpilot.testinfra.runners.gtestcoveragerunner.py
        if not prev_segment.is_executable:
            return [], []

        # this segment ends at the line if col == 1
        # (so segment effectively ends on the line) and
        # line+1 if col is > 1 (so it touches at least some part of last line).
        end_of_segment = self.line if self.col == 1 else self.line + 1
        lines_range = list(range(prev_segment.line, end_of_segment))
        return (lines_range, []) if prev_segment.has_coverage else ([], lines_range)


def parse_segments(raw_segments: list[list[int]]) -> list[LlvmCoverageSegment]:
    # [FUNCTION DOC] Converts a list of raw segment arrays (5 or 6 elements each) from llvm
    # export JSON into LlvmCoverageSegment objects, using no gap flag for 5-element arrays.
    # Raises AssertionError for any other length.
    """
    Creates LlvmCoverageSegment from a list of lists in llvm export json.
    each segment is represented by 5-element array.
    """
    ret: list[LlvmCoverageSegment] = []
    for raw_segment in raw_segments:
        if not (len(raw_segment) == 5 or len(raw_segment) == 6):
            raise AssertionError(
                f"list is not compatible with llvmcom export: expected 5 or 6 elements, got {len(raw_segment)}"
            )
        if len(raw_segment) == 5:
            ret.append(
                LlvmCoverageSegment(
                    raw_segment[0],
                    raw_segment[1],
                    raw_segment[2],
                    raw_segment[3],
                    raw_segment[4],
                    None,
                )
            )
        else:
            ret.append(LlvmCoverageSegment(*raw_segment))

    return ret
