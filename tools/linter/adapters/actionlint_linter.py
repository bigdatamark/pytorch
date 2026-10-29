# [MODULE DOC] Lintrunner adapter that runs the actionlint binary on GitHub workflow files in
# parallel threads and prints one JSON lint message per finding to stdout. Run as a script
# with a required binary path and file list; nothing runs at import time except imports.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "ACTIONLINT"


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


RESULTS_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?P<char>\d+):
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] Run the command list, capturing output (no check, no timeout), and
    # return the CompletedProcess. Logs the command and elapsed milliseconds at debug level;
    # OSError from process launch propagates.
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
    binary: str,
    file: str,
) -> list[LintMessage]:
    # [FUNCTION DOC] Run actionlint on one file, suppressing three known-benign messages
    # (runs-on mapping, freethreaded input of setup-python, workflow_dispatch input limit).
    # Returns one error message per line of output matching the result regex (file, line,
    # char, message, bracketed code). If the binary cannot be launched (OSError) returns a
    # single command-failed error without location. Exit status is not checked.
    try:
        proc = run_command(
            [
                binary,
                "-ignore",
                '"runs-on" section must be sequence node but got mapping node with "!!map" tag',
                "-ignore",
                'input "freethreaded" is not defined in action "actions/setup-python@v',
                # GitHub increased workflow_dispatch limit to 25 inputs (Dec 2025).
                # actionlint fixed this in v1.7.10; remove after upgrading from v1.7.7.
                "-ignore",
                'maximum number of inputs for "workflow_dispatch" event is 10 but',
                file,
            ]
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
            char=int(match["char"]),
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]


# [FUNCTION DOC] Script entry (module level, not a function): parses the binary path and file
# names, prints a JSON error and exits 0 if the binary is missing, otherwise lints all files
# in a thread pool and prints each message as JSON; unexpected errors are logged and re-raised.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="actionlint runner",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="actionlint binary path",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    if not os.path.exists(args.binary):
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Could not find actionlint binary at {args.binary},"
                " you may need to run `lintrunner init`."
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                args.binary,
                filename,
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
