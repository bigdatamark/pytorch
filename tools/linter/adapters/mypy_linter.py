# [MODULE DOC] Lint adapter that runs mypy (through the dmypy daemon locally, or plain mypy
# under GitHub Actions) on the given files and converts its output into JSON lint messages.
# Prefers an adjacent stub file over a source file when one exists. Parses diagnostics from
# stdout with a regular expression and internal-error lines from stderr with another. Maps
# mypy severities (error, note) to lint severities. Prints messages as JSON lines.
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
from pathlib import Path
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


# tools/linter/flake8_linter.py:15:13: error: Incompatibl...int")  [assignment]
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


# Severity is either "error" or "note":
# https://github.com/python/mypy/blob/8b47a032e1317fb8e3f9a818005a6b63e9bf0311/mypy/errors.py#L46-L47
severities = {
    "error": LintSeverity.ERROR,
    "note": LintSeverity.ADVICE,
}


def check_mypy_installed(code: str) -> list[LintMessage]:
    # [FUNCTION DOC] Verify that mypy can be run with the current Python interpreter by asking for
    # its version. Returns an empty list on success or a single error LintMessage carrying mypy's
    # stderr text if the command exits with failure.
    cmd = [sys.executable, "-mmypy", "-V"]
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
    # non-empty value.
    return bool(os.getenv("GITHUB_ACTIONS"))


def check_files(
    filenames: list[str],
    config: str,
    retries: int,
    code: str,
) -> list[LintMessage]:
    # [FUNCTION DOC] Run mypy on the given files with the given config and return lint messages.
    # Converts file names to relative paths (a workaround for a dmypy bug with absolute paths),
    # uses plain mypy under GitHub Actions and dmypy otherwise. Returns a command-failed message
    # if the process cannot start or exits with a status other than 0 or 1; otherwise returns one
    # message per parsed diagnostic plus one per internal error found in stderr. Side effects:
    # runs a subprocess.
    # dmypy has a bug where it won't pick up changes if you pass it absolute
    # file names, see https://github.com/python/mypy/issues/16768
    filenames = [os.path.relpath(f) for f in filenames]
    try:
        mypy_commands = ["dmypy", "run", "--"]
        if in_github_actions():
            mypy_commands = ["mypy"]
        proc = run_command(
            [*mypy_commands, f"--config={config}"] + filenames,
            extra_env={},
            retries=retries,
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

    rc = [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ] + [
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
    # [FUNCTION DOC] Entry point. Parses options (retries, config, code, verbose, file names),
    # configures logging to stderr, builds an ordered mapping of files to check substituting a
    # stub file for a source file when the stub exists, then prints the installation check and
    # mypy results as JSON lines.
    # KNOWN ISSUE: the stub name is derived by replacing every occurrence of the text .py in the
    # path, which would also alter directory names containing it.
    parser = argparse.ArgumentParser(
        description="mypy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out mypy",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an mypy .ini config file",
    )
    parser.add_argument(
        "--code",
        default="MYPY",
        help="the code this lint should report as",
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
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    # Use a dictionary here to preserve order. mypy cares about order,
    # tragically, e.g. https://github.com/python/mypy/issues/2015
    filenames: dict[str, bool] = {}

    # If a stub file exists, have mypy check it instead of the original file, in
    # accordance with PEP-484 (see https://www.python.org/dev/peps/pep-0484/#stub-files)
    for filename in args.filenames:
        if filename.endswith(".pyi"):
            filenames[filename] = True
            continue

        stub_filename = filename.replace(".py", ".pyi")
        if Path(stub_filename).exists():
            filenames[stub_filename] = True
        else:
            filenames[filename] = True

    lint_messages = check_mypy_installed(args.code) + check_files(
        list(filenames), args.config, args.retries, args.code
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
