# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyyaml==6.0.2",
# ]
# ///
# [MODULE DOC] Lint adapter for GitHub workflow files that ensures jobs triggered by push,
# schedule, pull_request or pull_request_target are guarded so they do not run on forks, i.e.
# the job condition must check that the repository (or its owner) is pytorch. It can propose
# an automatic rewrite adding or extending the condition. Prefers the C-based YAML loader when
# available. Files are checked in parallel processes when run as a script.
# KNOWN ISSUE: the docstring's last sentence is incomplete (it ends with the words so this is).
"""
This a linter that ensures that jobs that can be triggered by push,
pull_request, or schedule will check if the repository owner is 'pytorch'.  This
ensures that forks will not run jobs.

There are some edge cases that might be caught, and this prevents workflows from
being reused in other organizations, but as of right now, there are no workflows
with both push/pull_request/etc and workflow_call triggers simultaneously, so
this is.

There is also a setting in Github repos that can disable all workflows for that
repo.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from yaml import load


if TYPE_CHECKING:
    from collections.abc import Callable


# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]


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


def load_yaml(path: Path) -> Any:
    # [FUNCTION DOC] HELPER: Load a YAML file with the chosen safe loader; returns parsed data.
    with open(path) as f:
        return load(f, Loader)


def gen_lint_message(
    filename: str | None = None,
    original: str | None = None,
    replacement: str | None = None,
    description: str | None = None,
) -> LintMessage:
    # [FUNCTION DOC] HELPER: Build an error LintMessage with the fixed code NO_WORKFLOWS_ON_FORK
    # and rule name format, with optional path, original text, replacement text and description.
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="NO_WORKFLOWS_ON_FORK",
        severity=LintSeverity.ERROR,
        name="format",
        original=original,
        replacement=replacement,
        description=description,
    )


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Check one workflow file and return lint messages. Only mappings whose
    # triggers (stored under the boolean key that YAML uses for the word on) include push,
    # schedule, pull_request or pull_request_target are considered. Jobs that have a needs entry
    # or an if condition of false are exempt; other jobs must have a condition that mentions the
    # pytorch repository or owner, otherwise they are marked bad. Then rewrites the file text line
    # by line, inserting or extending the condition for each bad job, and returns a message with
    # original and replacement. Gives only a description-only message when an existing condition
    # spans multiple lines, and an error message when a condition cannot be parsed. Returns an
    # empty list if nothing changes.
    logging.debug("Checking file %s", filename)

    workflow = load_yaml(Path(filename))
    bad_jobs: dict[str, str | None] = {}
    if type(workflow) is not dict:
        return []

    # yaml parses "on" as True
    triggers = workflow.get(True, {})
    triggers_to_check = ["push", "schedule", "pull_request", "pull_request_target"]
    if not any(trigger in triggers_to_check for trigger in triggers):
        return []

    jobs = workflow.get("jobs", {})
    for job, definition in jobs.items():
        if definition.get("needs"):
            # The parent job will have the if statement
            continue

        if_statement = definition.get("if")

        if if_statement is None:
            bad_jobs[job] = None
        elif type(if_statement) is bool and not if_statement:
            # if: false
            pass
        else:
            if_statement = str(if_statement)
            valid_checks: list[Callable[[str], bool]] = [
                lambda x: "github.repository == 'pytorch/pytorch'" in x
                and "github.event_name != 'schedule' || github.repository == 'pytorch/pytorch'"
                not in x,
                lambda x: "github.repository_owner == 'pytorch'" in x,
            ]
            if not any(f(if_statement) for f in valid_checks):
                bad_jobs[job] = if_statement

    with open(filename) as f:
        lines = f.readlines()

    smart_enough = True
    original = "".join(lines)
    iterator = iter(range(len(lines)))
    replacement = ""
    for i in iterator:
        line = lines[i]
        # Search for job name
        re_match = re.match(r"( +)([-_\w]*):", line)
        if not re_match or re_match.group(2) not in bad_jobs:
            replacement += line
            continue
        job_name = re_match.group(2)

        failure_type = bad_jobs[job_name]
        if failure_type is None:
            # Just need to add an if statement
            replacement += (
                f"{line}{re_match.group(1)}  if: github.repository_owner == 'pytorch'\n"
            )
            continue

        # Search for if statement
        while re.match(r"^ +if:", line) is None:
            replacement += line
            i = next(iterator)
            line = lines[i]
        if i + 1 < len(lines) and not re.match(r"^ +(.*):", lines[i + 1]):
            # This is a multi line if statement
            smart_enough = False
            break

        if_statement_match = re.match(r"^ +if: ([^#]*)(#.*)?$", line)
        # Get ... in if: ... # comments
        if not if_statement_match:
            return [
                gen_lint_message(
                    description=f"Something went wrong when looking at {job_name}.",
                )
            ]

        if_statement = if_statement_match.group(1).strip()

        # Handle comment in if: ... # comments
        comments = if_statement_match.group(2) or ""
        if comments:
            comments = " " + comments

        # Too broad of a check, but should catch everything
        needs_parens = "||" in if_statement

        # Handle ${{ ... }}
        has_brackets = re.match(r"\$\{\{(.*)\}\}", if_statement)
        internal_statement = (
            has_brackets.group(1).strip() if has_brackets else if_statement
        )

        if needs_parens:
            internal_statement = f"({internal_statement})"
        new_line = f"{internal_statement} && github.repository_owner == 'pytorch'"

        # I don't actually know if we need the ${{ }} but do it just in case
        new_line = "${{ " + new_line + " }}" + comments

        replacement += f"{re_match.group(1)}  if: {new_line}\n"

    description = (
        "Please add checks for if: github.repository_owner == 'pytorch' in the following jobs in this file: "
        + ", ".join(job for job in bad_jobs)
    )

    if not smart_enough:
        return [
            gen_lint_message(
                filename=filename,
                description=description,
            )
        ]

    if replacement == original:
        return []

    return [
        gen_lint_message(
            filename=filename,
            original=original,
            replacement=replacement,
            description=description,
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="workflow consistency linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

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
