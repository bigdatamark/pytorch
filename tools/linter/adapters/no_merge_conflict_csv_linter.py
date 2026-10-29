# [MODULE DOC] Lint adapter that enforces a merge-conflict-resistant layout for CSV files:
# each non-blank line is followed by three blank lines. Checks files in parallel worker
# processes and prints warning-level JSON lint messages carrying the reformatted text.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from enum import Enum
from typing import NamedTuple


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enum of the severities lintrunner understands: error,
    # warning, advice and disabled.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] HELPER: Record for one lint finding in lintrunner's JSON schema (path, line,
    # char, code, severity, name, original, replacement, description; None means unset). Instances
    # are converted to a mapping and printed as one JSON object per line.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Compute the expected layout of a CSV file by dropping blank lines and
    # following each remaining line with three newline characters (the last three are removed).
    # Returns an empty list if it equals the file content, otherwise a single warning message with
    # the original and replacement text. Reads the file twice, once as raw bytes and once as text.
    # KNOWN ISSUE: the original is decoded from bytes while the replacement is built from
    # text mode reads, so files with DOS line endings always differ from the computed replacement.
    with open(filename, "rb") as f:
        original = f.read().decode("utf-8")
    replacement = ""
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) > 0:
                replacement += line
                replacement += "\n" * 3
        replacement = replacement[:-3]

        if replacement == original:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="MERGE_CONFLICTLESS_CSV",
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses options, configures logging to stderr, then checks every
    # file given on the command line in a process pool sized to the CPU count, printing each lint
    # message as a JSON line. If a check raises, logs which file failed and re-raises.
    parser = argparse.ArgumentParser(
        description="Format csv files to have 3 lines of space between each line to prevent merge conflicts.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
