# [MODULE DOC] Generic lint adapter that runs grep with an extended regex pattern over the
# given files and reports each match as a lint error, optionally proposing a whole-file
# replacement computed with sed. Supports an allowlist pattern (skip files containing it), a
# per-line allowlist, a match-first-only mode, and caps on reported matches and file size.
# Module globals LINTER_NAME and ERROR_DESCRIPTION are set from the command line in main and
# used when printing messages. Prints JSON lint messages to stdout.
"""
Generic linter that greps for a pattern and optionally suggests replacements.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple


IS_WINDOWS: bool = os.name == "nt"
MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB in bytes
MAX_MATCHES_PER_FILE: int = 100  # Maximum number of matches to report per file
MAX_ORIGINAL_SIZE: int = (
    512 * 1024
)  # 512KB - don't compute replacement if original is larger


class LintSeverity(str, Enum):
    # [CLASS DOC] HELPER: String-valued enum of the severities lintrunner understands: error,
    # warning, advice and disabled.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


LINTER_NAME: str = ""
ERROR_DESCRIPTION: str | None = None


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


def as_posix(name: str) -> str:
    # [FUNCTION DOC] HELPER: Convert a path to forward-slash form when running on Windows; returns
    # it unchanged elsewhere.
    return name.replace("\\", "/") if IS_WINDOWS else name


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] HELPER: Run an external command with captured output, logging the command and
    # its elapsed time at debug level. Returns the finished process; does not raise on a nonzero
    # exit status.
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def print_lint_message(
    name: str,
    severity: LintSeverity = LintSeverity.ERROR,
    path: str | None = None,
    line: int | None = None,
    original: str | None = None,
    replacement: str | None = None,
    description: str | None = None,
) -> None:
    # [FUNCTION DOC] Build a lint message from the arguments and print it as a JSON line on
    # stdout. The code field comes from the global linter name, and the description falls back to
    # the global error description when none is given. Side effects: writes to stdout, reads
    # module globals.
    """
    Create a LintMessage and print it as JSON.

    Accepts the same arguments as LintMessage constructor.
    """
    char = None
    code = LINTER_NAME
    description = description or ERROR_DESCRIPTION
    lint_message = LintMessage(
        path, line, char, code, severity, name, original, replacement, description
    )
    print(json.dumps(lint_message._asdict()), flush=True)


def group_lines_by_file(lines: list[str]) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Group grep output lines by file name. Input lines look like name,
    # line number and content separated by colons; the result maps each file name to the list of
    # text after the first colon. Empty lines are skipped.
    """
    Group matching lines by filename.

    Args:
        lines: List of grep output lines in format "filename:line:content"

    Returns:
        Dictionary mapping filename to list of line remainders (without filename prefix)
    """
    grouped: dict[str, list[str]] = {}
    for line in lines:
        if not line:
            continue
        # Extract filename and remainder from "filename:line:content" format
        parts = line.split(":", 1)
        filename = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""
        if filename not in grouped:
            grouped[filename] = []
        grouped[filename].append(remainder)
    return grouped


def check_allowlist(
    filename: str,
    allowlist_pattern: str,
) -> bool:
    # [FUNCTION DOC] Report whether a file should be skipped because it contains the allowlist
    # pattern, found by running grep on that file. Returns false immediately when the pattern is
    # empty, and also false when grep finds nothing. On an exception it prints a command-failed
    # lint message and returns false.
    # KNOWN ISSUE: run_command never uses the check option, so the branch formatting a
    # CalledProcessError is effectively dead code; a grep exit status of 2 (a real grep error) is
    # treated like no match.
    """
    Check if a file matches the allowlist pattern.

    Args:
        filename: Path to the file to check
        allowlist_pattern: Pattern to grep for in the file

    Returns:
        True if the file should be skipped (allowlist pattern matched), False otherwise.
        Prints error message and returns False if there was an error running grep.
    """
    if not allowlist_pattern:
        return False

    try:
        proc = run_command(["grep", "-nEHI", allowlist_pattern, filename])
    except Exception as err:
        print_lint_message(
            name="command-failed",
            description=(
                f"Failed due to {err.__class__.__name__}:\n{err}"
                if not isinstance(err, subprocess.CalledProcessError)
                else (
                    "COMMAND (exit code {returncode})\n"
                    "{command}\n\n"
                    "STDERR\n{stderr}\n\n"
                    "STDOUT\n{stdout}"
                ).format(
                    returncode=err.returncode,
                    command=" ".join(as_posix(x) for x in err.cmd),
                    stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                    stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                )
            ),
        )
        return False

    # allowlist pattern was found, abort lint
    if proc.returncode == 0:
        return True

    return False


def lint_file(
    filename: str,
    line_remainders: list[str],
    allowlist_pattern: str,
    replace_pattern: str,
    error_name: str,
    line_allowlist_pattern: re.Pattern[str] | None = None,
) -> None:
    # [FUNCTION DOC] Emit lint messages for one file given its grep match remainders. Skips the
    # file if the allowlist pattern is present, drops lines matching the line allowlist, and then
    # either (a) when a replace pattern is given and the file is small enough, runs sed over the
    # file and prints a single message carrying the original and replaced text, or (b) prints one
    # message per matching line up to a maximum, followed by a too-many-matches message if the
    # limit was exceeded. Errors reading or running sed print a command-failed message. Side
    # effects: reads the file, runs sed, prints to stdout.
    """
    Lint a file with one or more pattern matches, printing LintMessages as they're created.

    Args:
        filename: Path to the file being linted
        line_remainders: List of line remainders (format: "line:content" without filename prefix)
        allowlist_pattern: Pattern to check for allowlisting
        replace_pattern: Pattern for sed replacement
        error_name: Human-readable error name
        line_allowlist_pattern: Compiled regex; lines matching this are skipped
    """
    if not line_remainders:
        return

    should_skip = check_allowlist(filename, allowlist_pattern)
    if should_skip:
        return

    if line_allowlist_pattern:
        line_remainders = [
            lr
            for lr in line_remainders
            if not line_allowlist_pattern.search(
                lr.split(":", 1)[1] if ":" in lr else lr
            )
        ]
        if not line_remainders:
            return

    # Check if file is too large to compute replacement
    file_size = os.path.getsize(filename)
    compute_replacement = replace_pattern and file_size <= MAX_ORIGINAL_SIZE

    # Apply replacement to entire file if pattern is specified and file is not too large
    original = None
    replacement = None
    if compute_replacement:
        # When we have a replacement, report a single message with line=None
        try:
            with open(filename) as f:
                original = f.read()

            proc = run_command(["sed", "-r", replace_pattern, filename])
            replacement = proc.stdout.decode("utf-8")
        except Exception as err:
            print_lint_message(
                name="command-failed",
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )
            return

        print_lint_message(
            path=filename,
            name=error_name,
            original=original,
            replacement=replacement,
        )
    else:
        # When no replacement, report each matching line (up to MAX_MATCHES_PER_FILE)
        total_matches = len(line_remainders)
        matches_to_report = min(total_matches, MAX_MATCHES_PER_FILE)

        for line_remainder in line_remainders[:matches_to_report]:
            # line_remainder format: "line_number:content"
            split = line_remainder.split(":", 1)
            line_number = int(split[0]) if split[0] else None
            print_lint_message(
                path=filename,
                line=line_number,
                name=error_name,
            )

        # If there are more matches than the limit, print an error
        if total_matches > MAX_MATCHES_PER_FILE:
            print_lint_message(
                path=filename,
                name="too-many-matches",
                description=f"File has {total_matches} matches, only showing first {MAX_MATCHES_PER_FILE}",
            )


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses command-line options (pattern, allowlist patterns, linter
    # name, error name and description, replace pattern, match-first-only, verbose), rejecting
    # duplicated options, sets the module globals, configures logging to stderr, and filters out
    # oversized or inaccessible files with warnings. Runs grep over the remaining files in batches
    # sized to stay below the command-line length limit, groups the output per file and calls the
    # per-file linter. If grep itself raises, prints a command-failed message and exits with
    # status zero.
    # KNOWN ISSUE: the batch size is computed by integer division and could be zero if the
    # argument length is huge relative to the file count, which would make the range step invalid;
    # in match-first-only mode the output has no line numbers, so messages carry no line.
    parser = argparse.ArgumentParser(
        description="grep wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="pattern to grep for",
    )
    parser.add_argument(
        "--allowlist-pattern",
        help="if this pattern is true in the file, we don't grep for pattern",
    )
    parser.add_argument(
        "--linter-name",
        required=True,
        help="name of the linter",
    )
    parser.add_argument(
        "--match-first-only",
        action="store_true",
        help="only match the first hit in the file",
    )
    parser.add_argument(
        "--error-name",
        required=True,
        help="human-readable description of what the error is",
    )
    parser.add_argument(
        "--error-description",
        required=True,
        help="message to display when the pattern is found",
    )
    parser.add_argument(
        "--replace-pattern",
        help=(
            "the form of a pattern passed to `sed -r`. "
            "If specified, this will become proposed replacement text."
        ),
    )
    parser.add_argument(
        "--line-allowlist-pattern",
        help="if a matching line also matches this pattern, that line is skipped",
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

    # Check for duplicate arguments before parsing
    seen_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            arg_name = arg.split("=")[0]
            if arg_name in seen_args:
                parser.error(
                    f"argument {arg_name}: not allowed to be specified multiple times"
                )
            seen_args.add(arg_name)

    args = parser.parse_args()

    global LINTER_NAME, ERROR_DESCRIPTION
    LINTER_NAME = args.linter_name
    ERROR_DESCRIPTION = args.error_description

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # Filter out files that are too large before running grep
    filtered_filenames = []
    for filename in args.filenames:
        try:
            file_size = os.path.getsize(filename)
            if file_size > MAX_FILE_SIZE:
                print_lint_message(
                    path=filename,
                    severity=LintSeverity.WARNING,
                    name="file-too-large",
                    description=f"File size ({file_size} bytes) exceeds {MAX_FILE_SIZE} bytes limit, skipping",
                )
            else:
                filtered_filenames.append(filename)
        except OSError as err:
            print_lint_message(
                path=filename,
                name="file-access-error",
                description=f"Failed to get file size: {err}",
            )

    # If all files were filtered out, nothing to do
    if not filtered_filenames:
        return

    files_with_matches = []
    if args.match_first_only:
        files_with_matches = ["--files-with-matches"]

    lines = []
    try:
        # Split the grep command into multiple batches to avoid hitting the
        # command line length limit of ~1M on my machine
        arg_length = sum(len(x) for x in filtered_filenames)
        batches = arg_length // 750000 + 1
        batch_size = len(filtered_filenames) // batches
        for i in range(0, len(filtered_filenames), batch_size):
            proc = run_command(
                [
                    "grep",
                    "-nEHI",
                    *files_with_matches,
                    args.pattern,
                    *filtered_filenames[i : i + batch_size],
                ]
            )
            lines.extend(proc.stdout.decode().splitlines())
    except Exception as err:
        print_lint_message(
            name="command-failed",
            description=(
                f"Failed due to {err.__class__.__name__}:\n{err}"
                if not isinstance(err, subprocess.CalledProcessError)
                else (
                    "COMMAND (exit code {returncode})\n"
                    "{command}\n\n"
                    "STDERR\n{stderr}\n\n"
                    "STDOUT\n{stdout}"
                ).format(
                    returncode=err.returncode,
                    command=" ".join(as_posix(x) for x in err.cmd),
                    stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                    stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                )
            ),
        )
        sys.exit(0)

    line_allowlist_re = (
        re.compile(args.line_allowlist_pattern) if args.line_allowlist_pattern else None
    )

    # Group lines by file to call lint_file once per file
    grouped_lines = group_lines_by_file(lines)

    for filename, line_remainders in grouped_lines.items():
        lint_file(
            filename,
            line_remainders,
            args.allowlist_pattern,
            args.replace_pattern,
            args.error_name,
            line_allowlist_re,
        )


if __name__ == "__main__":
    main()
