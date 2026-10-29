# [MODULE DOC] Filter script that reads lintrunner JSON messages (one per line) from standard
# input and prints a JSON list of GitHub check annotations to standard output. Messages lacking
# a path or line are skipped with a note on standard error. Paths are made relative to the git
# root (found by running git at import time). All work runs at import time; there are no
# functions. Every annotation gets the failure level.
from __future__ import annotations

import json
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


# From: https://docs.github.com/en/rest/reference/checks
class GitHubAnnotationLevel(str, Enum):
    # [CLASS DOC] String enum of the annotation severity levels accepted by the GitHub checks
    # API: notice, warning and failure. Only failure is used by this script.
    NOTICE = "notice"
    WARNING = "warning"
    FAILURE = "failure"


class GitHubAnnotation(NamedTuple):
    # [CLASS DOC] Immutable record of one GitHub check annotation: file path, start and end line,
    # optional start and end column, severity level, message, optional title and optional raw
    # details. Converted to a dict for JSON output.
    path: str
    start_line: int
    end_line: int
    start_column: int | None
    end_column: int | None
    annotation_level: GitHubAnnotationLevel
    message: str
    title: str | None
    raw_details: str | None


PYTORCH_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    .decode("ascii")
    .strip()
)

annotations = []
for line in sys.stdin:
    lint_message = json.loads(line)

    path = lint_message.get("path")
    line = lint_message.get("line")

    code = lint_message["code"]
    severity = lint_message["severity"]
    name = lint_message["name"]
    description = lint_message.get("description")

    # These fields are required by the GitHub API, but optional in lintrunner.
    # If they don't exist, just skip.
    if path is None or line is None:
        print(f"No path/line for lint: ({code}) {name}", file=sys.stderr)
        continue

    # normalize path relative to git root
    path = Path(path).relative_to(PYTORCH_ROOT)

    annotations.append(
        GitHubAnnotation(
            path=str(path),
            start_line=int(line),
            end_line=int(line),
            start_column=None,
            end_column=None,
            annotation_level=GitHubAnnotationLevel.FAILURE,
            message=description,
            title=f"({code}) {name}",
            raw_details=None,
        )._asdict()
    )

print(json.dumps(annotations), flush=True)
