#!/usr/bin/env python3
"""
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)
has valid ownership information in a comment header. Valid means:
  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]
  - Each owner label actually exists in PyTorch
  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS

The set of valid labels is a snapshot checked in at LABELS_FILE rather than fetched at
runtime, so that linting neither needs network access nor breaks when a label is deleted
upstream. Refresh it with:

  python3 .github/scripts/export_pytorch_labels.py pytorch pytorch \
      --output-file tools/linter/adapters/pytorch_labels.json
"""

# [MODULE DOC]
# Lintrunner adapter that checks that each test file has an ownership comment header listing
# owner labels, that every label exists in the checked-in label snapshot, and that it is a
# module or oncall label or an accepted exception.
# Main contents: LintSeverity/LintMessage types, check_labels, check_file and main.
# Import-time side effect: reads and parses pytorch_labels.json next to this file into the
# PYTORCH_LABELS set (raises if the file is missing or invalid).
from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "TESTOWNERS"


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enumeration of lint severities (error, warning,
    # advice, disabled) serialized into the JSON output consumed by lintrunner.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] HELPER: Immutable record of one lint finding in the lintrunner JSON format:
    # path, line and char (each optional), linter code, severity, short name, optional
    # original and replacement text (for autofix), and an optional description.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


LABELS_FILE = Path(__file__).parent / "pytorch_labels.json"
PYTORCH_LABELS = set(json.loads(LABELS_FILE.read_text()))
# Team/owner labels usually start with "module: " or "oncall: ", but the following are acceptable exceptions
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]
OWNERS_PREFIX = "# Owner(s): "
GLOB_EXCEPTIONS = ["**/test/run_test.py"]


def check_labels(
    labels: list[str], filename: str, line_number: int
) -> list[LintMessage]:
    # [FUNCTION DOC] Validates the owner labels found on one header line. Emits an
    # invalid-label error for each label missing from the snapshot, and an invalid-owner error
    # for each label that neither starts with a module or oncall prefix nor is in the
    # acceptable list. Returns the messages tagged with the file name and line number.
    # KNOWN ISSUE: the invalid-owner text names ACCEPTABLE_OWNERS_LABELS and a tools/linters
    # path, but the constant is ACCEPTABLE_OWNER_LABELS and the directory is tools/linter.
    lint_messages = []
    for label in labels:
        if label not in PYTORCH_LABELS:
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=line_number,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[invalid-label]",
                    original=None,
                    replacement=None,
                    description=(
                        f"{label} is not a PyTorch label "
                        "(please choose from https://github.com/pytorch/pytorch/labels). "
                        "If it was added recently, refresh the checked-in label snapshot "
                        "as described in tools/linter/adapters/testowners_linter.py"
                    ),
                )
            )

        if label.startswith(("module:", "oncall:")) or label in ACCEPTABLE_OWNER_LABELS:
            continue

        lint_messages.append(
            LintMessage(
                path=filename,
                line=line_number,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[invalid-owner]",
                original=None,
                replacement=None,
                description=(
                    f"{label} is not an acceptable owner "
                    "(please update to another label or edit ACCEPTABLE_OWNERS_LABELS "
                    "in tools/linters/adapters/testowners_linter.py)"
                ),
            )
        )

    return lint_messages


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Scans the file line by line for lines starting with the owners prefix,
    # decodes each remainder as a JSON list of labels and validates it. If no such line exists,
    # adds a no-owner-info error. Returns all messages. Raises OSError if unreadable and a JSON
    # error if an owners line is not valid JSON.
    lint_messages = []
    has_ownership_info = False

    with open(filename) as f:
        for idx, line in enumerate(f):
            if not line.startswith(OWNERS_PREFIX):
                continue

            has_ownership_info = True
            labels = json.loads(line[len(OWNERS_PREFIX) :])
            lint_messages.extend(check_labels(labels, filename, idx + 1))

    if has_ownership_info is False:
        lint_messages.append(
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[no-owner-info]",
                original=None,
                replacement=None,
                description="Missing a comment header with ownership information.",
            )
        )

    return lint_messages


def main() -> None:
    # [FUNCTION DOC] Command line entry point. Parses the list of file paths, checks them one
    # by one in this process and prints each message as a JSON line on standard output.
    # KNOWN ISSUE: GLOB_EXCEPTIONS is defined but never used in this file.
    parser = argparse.ArgumentParser(
        description="test ownership linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()
    lint_messages = []

    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
