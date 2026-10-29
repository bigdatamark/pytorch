# [MODULE DOC] Script that checks the installed lintrunner version against a minimum (0.10.7)
# by running the lintrunner command with the version flag. If the installed version is older,
# prints an advice-level JSON lint message asking the user to upgrade. Everything runs under
# the main guard, so importing has no effects.
# KNOWN ISSUE: when the version output cannot be parsed, an error message is built but never
# printed and the script exits with status zero, so a missing lintrunner is silently ignored.
from __future__ import annotations

import json
import subprocess
import sys
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "LINTRUNNER_VERSION"


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


def toVersionString(version_tuple: tuple[int, int, int]) -> str:
    # [FUNCTION DOC] HELPER: Format a three-integer version tuple as a dotted string.
    return ".".join(str(x) for x in version_tuple)


if __name__ == "__main__":
    version_str = (
        subprocess.run(["lintrunner", "-V"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    import re

    version_match = re.compile(r"lintrunner (\d+)\.(\d+)\.(\d+)").match(version_str)

    if not version_match:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description="Lintrunner is not installed, did you forget to run `make setup-lint && make lint`?",
        )
        sys.exit(0)

    curr_version = int(version_match[1]), int(version_match[2]), int(version_match[3])
    min_version = (0, 10, 7)

    if curr_version < min_version:
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ADVICE,
            name="command-failed",
            original=None,
            replacement=None,
            description="".join(
                (
                    f"Lintrunner is out of date (you have v{toVersionString(curr_version)} ",
                    f"instead of v{toVersionString(min_version)}). ",
                    "Please run `pip install lintrunner -U` to update it",
                )
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
