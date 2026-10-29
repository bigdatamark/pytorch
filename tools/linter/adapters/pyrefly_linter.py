# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==1.26.4 ; python_version >= '3.10' and python_version <= '3.11'",
#   "numpy==2.1.0 ; python_version >= '3.12' and python_version <= '3.13'",
#   "numpy==2.3.4 ; python_version >= '3.14'",
#   "expecttest==0.3.0",
#   "pyrefly==0.58.0",
#   "sympy==1.13.3",
#   "spmd_types==0.2.1",
#   "types-requests==2.27.25",
#   "types-pyyaml==6.0.2",
#   "types-tabulate==0.8.8",
#   "types-protobuf==5.29.1.20250403",
#   "types-setuptools==79.0.0.20250422",
#   "types-jinja2==2.11.9",
#   "types-colorama==0.4.6",
#   "filelock==3.18.0",
#   "junitparser==2.1.1",
#   "rich==14.1.0",
#   "optree==0.17.0",
#   "types-openpyxl==3.1.5.20250919",
#   "types-python-dateutil==2.9.0.20251008",
#   "packaging",
#   "libcst",
#   "isort",
#   "usort",
# ]
#
# [tool.uv]
# dependency-metadata = [
#   { name = "spmd-types", version = "0.2.1", requires-dist = [] },
# ]
# ///
# [MODULE DOC] Lint adapter that runs the pyrefly type checker on the project (as selected by
# the config file, not by individual file arguments) with JSON output and converts each
# reported error into a lint message, ignoring errors named deprecated. Supports options to
# remove unused ignores and to add suppressions. Also scans stderr for internal-error lines.
# Prints JSON lint messages. Import-time effect: compiles two regular expressions.
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


# Note: This regex pattern is kept for reference but not used for pyrefly JSON parsing
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)

# torch/_dynamo/variables/tensor.py:363: error: INTERNAL ERROR
INTERNAL_ERROR_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<severity>\S+?):?
    \s(?P<message>INTERNAL\sERROR.*)
    $
    """
)


def run_command(
    args: list[str],
    *,
    extra_env: dict[str, str] | None,
    retries: int,
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] HELPER: Run a command with captured output, logging the command line and
    # elapsed time at debug level, and return the finished process without raising on a
    # nonzero status.
    # KNOWN ISSUE: the extra_env and retries parameters are accepted but ignored.
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


# Severity mapping (currently only used for stderr internal errors)
# Pyrefly JSON output doesn't include severity, so all errors default to ERROR
severities = {
    "error": LintSeverity.ERROR,
    "note": LintSeverity.ADVICE,
}


def check_pyrefly_installed(code: str) -> list[LintMessage]:
    # [FUNCTION DOC] Verify that the pyrefly executable runs by asking for its version. Returns an
    # empty list on success or a single error LintMessage with the stderr text if it exits
    # with failure.
    # KNOWN ISSUE: only a failing exit status is handled; if the executable is missing the
    # resulting OS error propagates uncaught.
    cmd = ["pyrefly", "--version"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return []
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"Could not run '{' '.join(cmd)}': {msg}",
            )
        ]


def in_github_actions() -> bool:
    # [FUNCTION DOC] HELPER: Return whether the GITHUB_ACTIONS environment variable is set to a
    # non-empty value. Not used in this module.
    return bool(os.getenv("GITHUB_ACTIONS"))


def check_files(
    code: str, config: str, remove_unused_ignores: bool, suppress: bool
) -> list[LintMessage]:
    # [FUNCTION DOC] Run pyrefly check with JSON output using the given config, adding flags to
    # remove unused ignores or suppress errors when requested, and return lint messages. Returns
    # command-failed messages if the process cannot start or exits with a status other than 0 or
    # 1. Decodes the first JSON object from stdout (trailing GitHub Actions lines are ignored),
    # drops errors named deprecated, and builds one error message per remaining entry; malformed
    # output yields a json-parse-error message. Also adds messages for internal errors matched
    # in stderr.
    # KNOWN ISSUE: the severity expression for the deprecated name can never choose advice because
    # those entries are filtered out first.
    try:
        pyrefly_commands = [
            "pyrefly",
            "check",
            "--config",
            config,
            "--output-format=json",
        ]
        if remove_unused_ignores:
            pyrefly_commands.append("--remove-unused-ignores")
        if suppress:
            pyrefly_commands.append("--suppress-errors")
        proc = run_command(
            [*pyrefly_commands],
            extra_env={},
            retries=0,
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    stderr = str(proc.stderr, "utf-8").strip()
    if proc.returncode not in (0, 1):
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=stderr,
            )
        ]

    # Parse JSON output from pyrefly. In GitHub Actions, pyrefly appends
    # ::error commands to stdout after the JSON, so use raw_decode to parse
    # only the first JSON object and ignore trailing output.
    try:
        if stdout:
            result, _ = json.JSONDecoder().raw_decode(stdout)
            errors = result.get("errors", [])
        else:
            errors = []
        errors = [error for error in errors if error["name"] != "deprecated"]
        rc = [
            LintMessage(
                path=error["path"],
                name=error["name"],
                description=error.get(
                    "description", error.get("concise_description", "")
                ),
                line=error["line"],
                char=error["column"],
                code=code,
                severity=LintSeverity.ADVICE
                if error["name"] == "deprecated"
                else LintSeverity.ERROR,
                original=None,
                replacement=None,
            )
            for error in errors
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="json-parse-error",
                original=None,
                replacement=None,
                description=f"Failed to parse pyrefly JSON output: {e}",
            )
        ]

    # Still check stderr for internal errors
    rc += [
        LintMessage(
            path=match["file"],
            name="INTERNAL ERROR",
            description=match["message"],
            line=int(match["line"]),
            char=None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in INTERNAL_ERROR_RE.finditer(stderr)
    ]
    return rc


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses options (code, verbose, config, remove-unused-ignores,
    # suppress), configures logging at info level to stderr, runs the install check and the
    # pyrefly check, and prints each lint message as a JSON line.
    # KNOWN ISSUE: the help text for the config option says mypy .ini file although it is a
    # pyrefly config; the verbose option has no effect.
    parser = argparse.ArgumentParser(
        description="pyrefly wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--code",
        default="PYREFLY",
        help="the code this lint should report as",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an mypy .ini config file",
    )
    parser.add_argument(
        "--remove-unused-ignores",
        action="store_true",
        help="clean up unused ignores",
    )
    parser.add_argument(
        "--suppress",
        action="store_true",
        help="add suppressions",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    lint_messages = check_pyrefly_installed(args.code) + check_files(
        args.code, args.config, args.remove_unused_ignores, args.suppress
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
