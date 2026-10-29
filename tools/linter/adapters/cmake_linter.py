# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cmakelint==1.4.1",
# ]
# ///
# [MODULE DOC] Lintrunner adapter that runs cmakelint with a given config file on CMake files
# in parallel threads and prints one JSON error message per reported line to stdout.
# Run as a script; the commented block above declares the cmakelint dependency. No
# import-time side effects beyond imports and constants.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import time
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "CMAKE"


class LintSeverity(str, Enum):
    # [CLASS DOC] String enum of lintrunner severities: error, warning, advice, disabled.
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    # [CLASS DOC] Immutable record of one lintrunner finding: path, line, char, code, severity,
    # name, original text, replacement text and description (several may be None). Printed
    # as a JSON object via its dict conversion.
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


# CMakeLists.txt:901: Lines should be <= 80 characters long [linelength]
RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] Run the command list with captured output and no exit status check; return
    # the CompletedProcess. Logs command and elapsed milliseconds at debug level; OSError from
    # process launch propagates.
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


def check_file(
    filename: str,
    config: str,
) -> list[LintMessage]:
    # [FUNCTION DOC] Run cmakelint with the given config on one file. Returns one error message
    # (line number, no column) per output line matching the result pattern; the bracketed
    # check name is used as the message name. If the tool cannot be launched (OSError) returns
    # a single command-failed error without location. Exit status is not checked.
    try:
        proc = run_command(
            ["cmakelint", f"--config={config}", filename],
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]


# Script entry: parses the config path and file names, lints all files in a thread pool and
# prints each message as JSON; unexpected exceptions are logged and re-raised.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cmakelint runner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="location of cmakelint config",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                filename,
                args.config,
            ): filename
            for filename in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise
