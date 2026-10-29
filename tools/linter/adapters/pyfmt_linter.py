# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "usort==1.1.3",
#   "isort==6.0.1",
#   "ruff==0.14.4",
# ]
# ///
# [MODULE DOC] Lint adapter that formats Python files by applying isort, then usort, then ruff
# format, and reports a warning with the original and formatted text when they differ. Files
# are processed in parallel worker processes, capped by the MAX_JOBS environment variable or
# the CPU count. Prints JSON lint messages. Import-time side effects: imports isort and usort
# and computes the repository root.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import isort
import usort


IS_WINDOWS: bool = os.name == "nt"
REPO_ROOT = Path(__file__).absolute().parents[3]


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


def as_posix(name: str) -> str:
    # [FUNCTION DOC] HELPER: Convert a path to forward-slash form on Windows; unchanged elsewhere.
    # Not used in this module.
    return name.replace("\\", "/") if IS_WINDOWS else name


def format_error_message(filename: str, err: Exception) -> LintMessage:
    # [FUNCTION DOC] HELPER: Wrap an exception into an advice-level command-failed LintMessage for
    # the given file.
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="PYFMT",
        severity=LintSeverity.ADVICE,
        name="command-failed",
        original=None,
        replacement=None,
        description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
    )


def run_isort(content: str, path: Path) -> str:
    # [FUNCTION DOC] HELPER: Sort imports with isort using the repository configuration. For every
    # file other than this script, usort skip comments are temporarily turned into isort split
    # markers before sorting and turned back afterwards, so both tools respect the same
    # boundaries. Returns the new content.
    isort_config = isort.Config(settings_path=str(REPO_ROOT))

    is_this_file = path.samefile(__file__)
    if not is_this_file:
        content = re.sub(r"(#.*\b)usort:\s*skip\b", r"\g<1>isort: split", content)

    content = isort.code(content, config=isort_config, file_path=path)

    if not is_this_file:
        content = re.sub(r"(#.*\b)isort: split\b", r"\g<1>usort: skip", content)

    return content


def run_usort(content: str, path: Path) -> str:
    # [FUNCTION DOC] HELPER: Sort imports with usort, using the configuration found for the path,
    # and return the new content.
    usort_config = usort.Config.find(path)

    return usort.usort_string(content, path=path, config=usort_config)


def run_ruff_format(content: str, path: Path) -> str:
    # [FUNCTION DOC] HELPER: Format the content with ruff format, run as a subprocess reading
    # standard input, using the repository pyproject configuration and the file name for context.
    # Returns the formatted text; raises a ValueError carrying ruff's output if ruff exits
    # with failure.
    try:
        return subprocess.check_output(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--config",
                str(REPO_ROOT / "pyproject.toml"),
                "--stdin-filename",
                str(path),
                "-",
            ],
            input=content,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.output) from exc


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Read a file, run isort, usort and ruff format over it in that order (isort
    # first to fix blank-line style) and return an empty list if nothing changed, otherwise a
    # warning LintMessage with original and replacement. Any exception becomes an advice-level
    # command-failed message instead of propagating.
    path = Path(filename).absolute()
    original = replacement = path.read_text(encoding="utf-8")

    try:
        # NB: run isort first to enforce style for blank lines
        replacement = run_isort(replacement, path=path)
        replacement = run_usort(replacement, path=path)
        replacement = run_ruff_format(replacement, path=path)

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="PYFMT",
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except Exception as err:
        return [format_error_message(filename, err)]


def _default_num_workers() -> int | None:
    # [FUNCTION DOC] HELPER: Return the worker count: the positive integer value of MAX_JOBS when
    # set, otherwise the CPU count.
    # Forks one process per file batch, so respect MAX_JOBS to cap parallelism.
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses options (verbose, number of workers, file names),
    # configures logging to stderr, then checks all files in a process pool and prints each lint
    # message as a JSON line. If a check raises, logs the failing file and re-raises.
    parser = argparse.ArgumentParser(
        description="Format files with usort + ruff-format.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=None,
        help="number of parallel workers (defaults to MAX_JOBS or the CPU count)",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()
    num_workers = args.num_workers or _default_num_workers()

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
        max_workers=num_workers,
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
