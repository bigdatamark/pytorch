#!/usr/bin/env python3
# [MODULE DOC] Validates the grouped review-surface taxonomy embedded in the CODEOWNERS file:
# parses the declared hierarchy and the grouped path rules, checks every git-tracked path is
# covered, and reports uncovered, duplicate, stale, ineffective and order-inconsistent rules.
# Run as a script; with --lintrunner it emits a JSON lint message instead of a report.
# No import-time side effects.
"""Check that the proposed CODEOWNERS review surfaces cover every tracked path."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


LINTER_CODE = "CODEOWNERS_TAXONOMY"
HIERARCHY_MARKER = "# Review-surface hierarchy:"
SECTION_MARKER = "# Draft grouped review-surface definitions:"
GROUP_PREFIX = "# ["


@dataclass(frozen=True)
class TaxonomyPattern:
    # [CLASS DOC] Immutable record of one taxonomy rule: the review-surface group name, the
    # path (without leading slash; a trailing slash means a directory), and the 1-based line
    # number in CODEOWNERS. Hashable so it can be placed in sets.
    """A grouped taxonomy rule at its actual CODEOWNERS source position."""

    group: str
    path: str
    line_number: int


@dataclass
class Report:
    # [CLASS DOC] Result of the coverage analysis: tracked path count, number of patterns, set
    # of groups that are the effective owner of some path, lists of uncovered paths, mappings
    # of overridden and duplicated paths to their patterns, stale (never matching) and
    # ineffective (always overridden) patterns, and source-order mismatch messages.
    tracked: int
    pattern_count: int
    groups: set[str]
    uncovered: list[str]
    overridden: dict[str, list[TaxonomyPattern]]
    duplicates: dict[str, list[TaxonomyPattern]]
    stale: list[TaxonomyPattern]
    ineffective: list[TaxonomyPattern]
    source_order_mismatches: list[str]


def parse_declared_groups(codeowners: Path) -> list[str]:
    # [FUNCTION DOC] Reads the CODEOWNERS file and returns the group names listed in the
    # hierarchy block (comment lines of the form dash, label, colon, comma separated names)
    # between the hierarchy and rules markers. Raises ValueError if a marker is missing, a
    # group is declared twice, or no groups are found.
    """Parse the review surfaces declared by the hierarchy."""
    lines = codeowners.read_text().splitlines()
    try:
        start = lines.index(HIERARCHY_MARKER) + 1
        end = lines.index(SECTION_MARKER)
    except ValueError as error:
        raise ValueError("missing review-surface hierarchy or rules marker") from error

    groups = []
    for line in lines[start:end]:
        if line.startswith("# - ") and ": " in line:
            groups.extend(line.split(": ", 1)[1].split(", "))
    duplicates = sorted(group for group in set(groups) if groups.count(group) > 1)
    if duplicates:
        raise ValueError(f"duplicate hierarchy surfaces: {', '.join(duplicates)}")
    if not groups:
        raise ValueError("review-surface hierarchy contains no surfaces")
    return groups


def is_valid_owner(owner: str) -> bool:
    # [FUNCTION DOC] HELPER: True if owner is a plausible user or org/team handle (at sign then
    # one or two slash separated parts of letters, digits, dash, underscore, dot) or an email
    # with exactly one at sign and non-empty local and domain parts.
    """Check the syntax of a GitHub user, team, or email CODEOWNER."""
    if owner.startswith("@"):
        parts = owner[1:].split("/")
        return len(parts) <= 2 and all(
            part
            and all(character.isalnum() or character in "-_." for character in part)
            for part in parts
        )
    if owner.count("@") != 1:
        return False
    local, domain = owner.split("@")
    return bool(local and domain)


def parse_patterns(codeowners: Path) -> tuple[list[TaxonomyPattern], list[str]]:
    # [FUNCTION DOC] Parses the grouped rules after the rules marker. Group headers are
    # bracketed comment lines; rule lines are active (start with a slash, must have valid
    # owners) or commented out (comment then slash). Returns (patterns in file order with
    # leading slash removed, group names in order). Raises ValueError for a missing marker,
    # repeated group, rule before any group, missing or invalid owners, wildcard or root
    # paths, unrecognized lines, or no patterns at all.
    """Parse grouped rules while preserving their actual CODEOWNERS order."""
    lines = codeowners.read_text().splitlines()
    try:
        start = lines.index(SECTION_MARKER) + 1
    except ValueError as error:
        raise ValueError(f"missing taxonomy marker: {SECTION_MARKER}") from error

    patterns = []
    section_groups = []
    group = None
    for line_number, raw_line in enumerate(lines[start:], start=start + 1):
        line = raw_line.strip()
        if line.startswith(GROUP_PREFIX) and line.endswith("]"):
            group = line[len(GROUP_PREFIX) : -1]
            if group in section_groups:
                raise ValueError(f"{codeowners}:{line_number}: repeated group: {group}")
            section_groups.append(group)
        elif line.startswith(("# /", "/")):
            location = f"{codeowners}:{line_number}"
            active = line.startswith("/")
            fields = (line if active else line[2:]).split()
            pattern = fields[0]
            if group is None:
                raise ValueError(f"{location}: pattern has no group")
            if active:
                owners = []
                for field in fields[1:]:
                    if field.startswith("#"):
                        break
                    owners.append(field)
                if not owners:
                    raise ValueError(f"{location}: active pattern has no owners")
                invalid_owners = [
                    owner for owner in owners if not is_valid_owner(owner)
                ]
                if invalid_owners:
                    raise ValueError(
                        f"{location}: invalid active owners: {', '.join(invalid_owners)}"
                    )
            if pattern == "/" or any(character in pattern for character in "*?[\\"):
                raise ValueError(f"{location}: invalid taxonomy path: {pattern!r}")
            patterns.append(TaxonomyPattern(group, pattern[1:], line_number))
        elif line and not line.startswith("#"):
            raise ValueError(f"{codeowners}:{line_number}: invalid taxonomy entry")

    if not patterns:
        raise ValueError("taxonomy section contains no patterns")
    return patterns, section_groups


def tracked_paths(repo: Path) -> list[str]:
    # [FUNCTION DOC] Runs git ls-tree on HEAD in the given repo and returns the sorted list of
    # all committed file paths. Raises CalledProcessError if git fails.
    """List paths committed in HEAD, including submodule entries."""
    output = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "-z", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout.decode()
    return sorted(path for path in output.split("\0") if path)


def analyze(paths: list[str], patterns: list[TaxonomyPattern]) -> Report:
    # [FUNCTION DOC] For each path, finds matching patterns (the exact path or any ancestor
    # directory with trailing slash). No match means uncovered. The effective pattern is the
    # longest match; it is compared to the match defined latest in the file, and a difference
    # is a source-order mismatch (later rule would override a more specific one). Also
    # collects duplicate patterns, stale patterns that match nothing, and ineffective ones
    # that match but are never the effective one. Returns a Report; pure computation.
    """Measure coverage and verify that source order preserves specificity."""
    patterns_by_path: dict[str, list[TaxonomyPattern]] = defaultdict(list)
    for pattern in patterns:
        patterns_by_path[pattern.path].append(pattern)

    matched_patterns: set[TaxonomyPattern] = set()
    effective_patterns: set[TaxonomyPattern] = set()
    effective_groups = set()
    source_order_mismatches = []
    uncovered = []
    overridden = {}
    for path in paths:
        parts = path.split("/")
        candidates = [path]
        candidates.extend(
            "/".join(parts[:index]) + "/" for index in range(1, len(parts))
        )
        matches = []
        for candidate in candidates:
            matches.extend(patterns_by_path.get(candidate, []))
        if not matches:
            uncovered.append(path)
            continue
        if len(matches) > 1:
            overridden[path] = matches
        matched_patterns.update(matches)
        effective_pattern = max(matches, key=lambda pattern: len(pattern.path))
        effective_patterns.add(effective_pattern)
        effective_groups.add(effective_pattern.group)
        source_pattern = max(matches, key=lambda pattern: pattern.line_number)
        if effective_pattern != source_pattern:
            source_order_mismatches.append(
                f"{path}: {source_pattern.group}:{source_pattern.path} overrides "
                f"{effective_pattern.group}:{effective_pattern.path}"
            )

    duplicates = {
        path: entries for path, entries in patterns_by_path.items() if len(entries) > 1
    }
    return Report(
        tracked=len(paths),
        pattern_count=len(patterns),
        groups=effective_groups,
        uncovered=uncovered,
        overridden=overridden,
        duplicates=duplicates,
        stale=[pattern for pattern in patterns if pattern not in matched_patterns],
        ineffective=[
            pattern
            for pattern in patterns
            if pattern in matched_patterns and pattern not in effective_patterns
        ],
        source_order_mismatches=source_order_mismatches,
    )


def print_report(report: Report) -> None:
    # [FUNCTION DOC] Prints counts and a coverage percentage to stdout, followed by a section
    # listing entries for each non-empty failure category (uncovered, duplicate, stale,
    # ineffective, source-order mismatches).
    """Print the coverage summary and each integrity failure."""
    covered = report.tracked - len(report.uncovered)
    percentage = 100 * covered / report.tracked if report.tracked else 100.0
    print(f"Tracked paths: {report.tracked}")
    print(f"Covered paths: {covered}")
    print(f"Uncovered paths: {len(report.uncovered)}")
    print(f"Coverage: {percentage:.4f}%")
    print(f"Review surfaces: {len(report.groups)}")
    print(f"Patterns: {report.pattern_count}")
    print(f"Paths with overrides: {len(report.overridden)}")
    print(f"Duplicate patterns: {len(report.duplicates)}")
    print(f"Stale patterns: {len(report.stale)}")
    print(f"Ineffective patterns: {len(report.ineffective)}")
    print(f"Source-order mismatches: {len(report.source_order_mismatches)}")

    duplicate_entries = {
        path: ", ".join(pattern.group for pattern in patterns)
        for path, patterns in report.duplicates.items()
    }
    sections = [
        ("Uncovered paths", report.uncovered),
        (
            "Duplicate patterns",
            [f"{path}: {groups}" for path, groups in duplicate_entries.items()],
        ),
        (
            "Stale patterns",
            [f"{pattern.group}:{pattern.path}" for pattern in report.stale],
        ),
        (
            "Ineffective patterns",
            [f"{pattern.group}:{pattern.path}" for pattern in report.ineffective],
        ),
        ("Source-order mismatches", report.source_order_mismatches),
    ]
    for title, entries in sections:
        if entries:
            print(f"\n{title}:")
            print("\n".join(f"  {entry}" for entry in entries))


def emit_lint_error(description: str) -> None:
    # [FUNCTION DOC] Prints one JSON object to stdout (flushed) describing an error-severity
    # lint message for the CODEOWNERS file with the given description.
    """Emit a lintrunner-compatible taxonomy error."""
    print(
        json.dumps(
            {
                "path": "CODEOWNERS",
                "line": None,
                "char": None,
                "code": LINTER_CODE,
                "severity": "error",
                "name": "invalid-taxonomy",
                "original": None,
                "replacement": None,
                "description": description,
            }
        ),
        flush=True,
    )


def main() -> int:
    # [FUNCTION DOC] Entry point. Finds the git repo root, parses CODEOWNERS and analyzes
    # coverage. On parse errors returns 2 after printing to stderr (or, with --lintrunner,
    # emits a lint message and returns 0). Otherwise prints the report (not in lintrunner mode)
    # plus missing or undeclared group notes. In lintrunner mode emits one summary lint error
    # if any problem exists and returns 0; otherwise returns 1 if there were problems, else 0.
    """Run the taxonomy coverage check for the current Git repository."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lintrunner", action="store_true")
    args = parser.parse_args()
    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    try:
        codeowners = repo / "CODEOWNERS"
        declared_groups = parse_declared_groups(codeowners)
        patterns, section_groups = parse_patterns(codeowners)
        report = analyze(tracked_paths(repo), patterns)
    except ValueError as error:
        if args.lintrunner:
            emit_lint_error(
                f"CODEOWNERS taxonomy validation failed: {error}. "
                "Run `python scripts/codeowners/check_taxonomy.py` for details."
            )
            return 0
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not args.lintrunner:
        print_report(report)
    declared_group_set = set(declared_groups)
    section_group_set = set(section_groups)
    missing_groups = sorted(declared_group_set - report.groups)
    missing_sections = sorted(declared_group_set - section_group_set)
    undeclared_sections = sorted(section_group_set - declared_group_set)
    if not args.lintrunner:
        if missing_groups:
            print(f"Missing effective surfaces: {', '.join(missing_groups)}")
        if missing_sections:
            print(f"Missing grouped sections: {', '.join(missing_sections)}")
        if undeclared_sections:
            print(f"Undeclared grouped sections: {', '.join(undeclared_sections)}")
    has_errors = bool(missing_groups or missing_sections or undeclared_sections) or any(
        (
            report.uncovered,
            report.duplicates,
            report.stale,
            report.ineffective,
            report.source_order_mismatches,
        )
    )
    if args.lintrunner and has_errors:
        failure_counts = [
            (len(report.uncovered), "uncovered paths"),
            (len(report.duplicates), "duplicate patterns"),
            (len(report.stale), "stale patterns"),
            (len(report.ineffective), "ineffective patterns"),
            (len(report.source_order_mismatches), "source-order mismatches"),
            (len(missing_groups), "missing effective surfaces"),
            (len(missing_sections), "missing grouped sections"),
            (len(undeclared_sections), "undeclared grouped sections"),
        ]
        failures = [f"{count} {name}" for count, name in failure_counts if count]
        emit_lint_error(
            "CODEOWNERS taxonomy validation failed: "
            + ", ".join(failures)
            + ". Run `python scripts/codeowners/check_taxonomy.py` for details."
        )
        return 0
    return int(has_errors)


if __name__ == "__main__":
    sys.exit(main())
