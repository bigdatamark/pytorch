# [MODULE DOC] Defines CoverageRecord, the per-file coverage result shared by the parsers.

from __future__ import annotations

from typing import Any, NamedTuple


class CoverageRecord(NamedTuple):
    # [CLASS DOC] Immutable record of coverage for one file: the file path, the covered line
    # numbers and an optional list of uncovered line numbers.
    filepath: str
    covered_lines: list[int]
    uncovered_lines: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the record as a mapping with the keys filepath, covered_lines
        # and uncovered_lines.
        return {
            "filepath": self.filepath,
            "covered_lines": self.covered_lines,
            "uncovered_lines": self.uncovered_lines,
        }
