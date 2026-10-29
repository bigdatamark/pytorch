# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ruff==0.14.4",
# ]
# ///
# [MODULE DOC] Lint adapter for the ruff linter. Runs ruff check with JSON output over the
# given files, maps each finding to a lint message with a severity derived from the rule code
# (or an explicit code-to-severity option), optionally appends rule explanations and disable
# hints, and then, in parallel threads and unless disabled, runs ruff in fix-only mode on
# each file with findings to offer a whole-file replacement. Messages are printed as JSON
# lines. Includes helpers for running commands with a timeout and retry.
"""Adapter for https://github.com/charliermarsh/ruff."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import enum
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, BinaryIO


LINTER_CODE = "RUFF"
SYNTAX_ERROR = "E999"
IS_WINDOWS: bool = os.name == "nt"


class LintSeverity(str, enum.Enum):
    # [CLASS DOC] HELPER: String-valued enum of severities (error, warning, advice, disabled) in
    # the form lintrunner expects.
    """Severity of a lint message."""

    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


@dataclasses.dataclass(frozen=True)
class LintMessage:
    # [CLASS DOC] HELPER: Immutable record for one lint finding following the lintrunner schema
    # (path, line, char, code, severity, name, original, replacement, description). Can be
    # converted to a mapping and printed as a JSON line.
    """A lint message defined by https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html."""

    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None

    def asdict(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Return the message fields as a mapping.
        return dataclasses.asdict(self)

    def display(self) -> None:
        # [FUNCTION DOC] HELPER: Print the message as a JSON line on stdout for lintrunner.
        """Print to stdout for lintrunner to consume."""
        print(json.dumps(self.asdict()), flush=True)


def as_posix(name: str) -> str:
    # [FUNCTION DOC] HELPER: Convert a path to forward-slash form on Windows; unchanged elsewhere.
    return name.replace("\\", "/") if IS_WINDOWS else name


def _run_command(
    args: list[str],
    *,
    timeout: int | None,
    stdin: BinaryIO | None,
    input: bytes | None,
    check: bool,
    cwd: os.PathLike[Any] | None,
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] HELPER: Run a command once with captured output, never through a shell,
    # logging the command and elapsed time at debug level. Passes either the given input bytes or
    # stdin, a timeout, a check flag and a working directory. Returns the finished process; may
    # raise TimeoutExpired or CalledProcessError (when check is set).
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        if input is not None:
            return subprocess.run(
                args,
                capture_output=True,
                shell=False,
                input=input,
                timeout=timeout,
                check=check,
                cwd=cwd,
            )

        return subprocess.run(
            args,
            stdin=stdin,
            capture_output=True,
            shell=False,
            timeout=timeout,
            check=check,
            cwd=cwd,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def run_command(
    args: list[str],
    *,
    retries: int = 0,
    timeout: int | None = None,
    stdin: BinaryIO | None = None,
    input: bytes | None = None,
    check: bool = False,
    cwd: os.PathLike[Any] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] HELPER: Run a command, retrying up to the given number of times only when it
    # times out, sleeping one second between attempts and logging a warning. Returns the finished
    # process; re-raises the timeout error once retries run out.
    remaining_retries = retries
    while True:
        try:
            return _run_command(
                args, timeout=timeout, stdin=stdin, input=input, check=check, cwd=cwd
            )
        except subprocess.TimeoutExpired as err:
            if remaining_retries == 0:
                raise err
            remaining_retries -= 1
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",
                retries - remaining_retries,
                retries,
                err,
            )
            time.sleep(1)


def add_default_options(parser: argparse.ArgumentParser) -> None:
    # [FUNCTION DOC] HELPER: Add the shared options (retries, verbose, file names) to the parser.
    """Add default options to a parser.

    This should be called the last in the chain of add_argument calls.
    """
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of times to retry if the linter times out.",
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


def explain_rule(code: str) -> str:
    # [FUNCTION DOC] HELPER: Ask ruff to explain a rule by its code (JSON output) and return a
    # newline-prefixed text containing the rule's linter and summary. Raises CalledProcessError if
    # ruff fails.
    proc = run_command(
        ["ruff", "rule", "--output-format=json", code],
        check=True,
    )
    rule = json.loads(str(proc.stdout, "utf-8").strip())
    return f"\n{rule['linter']}: {rule['summary']}"


def get_issue_severity(code: str) -> LintSeverity:
    # [FUNCTION DOC] HELPER: Map a ruff rule code to a default severity: advice for style,
    # complexity and type-note prefixes, error for undefined names, syntax errors and pylint error
    # rules, and warning for everything else.
    # "B901": `return x` inside a generator
    # "B902": Invalid first argument to a method
    # "B903": __slots__ efficiency
    # "B950": Line too long
    # "C4": Flake8 Comprehensions
    # "C9": Cyclomatic complexity
    # "E2": PEP8 horizontal whitespace "errors"
    # "E3": PEP8 blank line "errors"
    # "E5": PEP8 line length "errors"
    # "T400": type checking Notes
    # "T49": internal type checker errors or unmatched messages
    if any(
        code.startswith(x)
        for x in (
            "B9",
            "C4",
            "C9",
            "E2",
            "E3",
            "E5",
            "T400",
            "T49",
            "PLC",
            "PLR",
        )
    ):
        return LintSeverity.ADVICE

    # "F821": Undefined name
    # "E999": syntax error
    if any(code.startswith(x) for x in ("F821", SYNTAX_ERROR, "PLE")):
        return LintSeverity.ERROR

    # "F": PyFlakes Error
    # "B": flake8-bugbear Error
    # "E": PEP8 "Error"
    # "W": PEP8 Warning
    # possibly other plugins...
    return LintSeverity.WARNING


def format_lint_message(
    message: str, code: str, rules: dict[str, str], show_disable: bool
) -> str:
    # [FUNCTION DOC] HELPER: Build the message text: optionally append the explanation for the
    # code, always append a pointer to the ruff rules documentation, and optionally append a hint
    # on disabling with a noqa comment.
    # KNOWN ISSUE: the documentation link does not include the rule code, and the explanation and
    # pointer are appended after a period, so messages ending in punctuation get doubled periods.
    if rules:
        message += f".\n{rules.get(code) or ''}"
    message += ".\nSee https://beta.ruff.rs/docs/rules/"
    if show_disable:
        message += f".\n\nTo disable, use `  # noqa: {code}`"
    return message


def check_files(
    filenames: list[str],
    severities: dict[str, LintSeverity],
    *,
    config: str | None,
    retries: int,
    timeout: int,
    explain: bool,
    show_disable: bool,
) -> list[LintMessage]:
    # [FUNCTION DOC] Run ruff check as a subprocess with JSON output over all the given files
    # (using the config if given, with retries and a timeout) and return one LintMessage per
    # finding. Optionally gathers explanations for each distinct rule code. Severity comes from
    # the explicit mapping, falling back to the default mapping. Returns a single command-failed
    # message if ruff cannot be run or exits with failure.
    # KNOWN ISSUE: a timeout that persists after all retries raises TimeoutExpired, which is not
    # caught here and propagates.
    try:
        proc = run_command(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--exit-zero",
                "--quiet",
                "--output-format=json",
                *([f"--config={config}"] if config else []),
                *filenames,
            ],
            retries=retries,
            timeout=timeout,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as err:
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
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        f"COMMAND (exit code {err.returncode})\n"
                        f"{' '.join(as_posix(x) for x in err.cmd)}\n\n"
                        f"STDERR\n{err.stderr.decode('utf-8').strip() or '(empty)'}\n\n"
                        f"STDOUT\n{err.stdout.decode('utf-8').strip() or '(empty)'}"
                    )
                ),
            )
        ]

    stdout = str(proc.stdout, "utf-8").strip()
    vulnerabilities = json.loads(stdout)

    if explain:
        all_codes = {v["code"] for v in vulnerabilities}
        rules = {code: explain_rule(code) for code in all_codes}
    else:
        rules = {}

    def lint_message(vuln: dict[str, Any]) -> LintMessage:
        # [FUNCTION DOC] HELPER: Convert one ruff finding into a LintMessage, treating an empty
        # code as the syntax error code and showing the disable hint only when the finding has a
        # real code.
        code = vuln["code"] or SYNTAX_ERROR
        return LintMessage(
            path=vuln["filename"],
            name=code,
            description=(
                format_lint_message(
                    vuln["message"],
                    code,
                    rules,
                    show_disable and bool(vuln["code"]),
                )
            ),
            line=int(vuln["location"]["row"]),
            char=int(vuln["location"]["column"]),
            code=LINTER_CODE,
            severity=severities.get(code, get_issue_severity(code)),
            original=None,
            replacement=None,
        )

    return [lint_message(v) for v in vulnerabilities]


def check_file_for_fixes(
    filename: str,
    *,
    config: str | None,
    retries: int,
    timeout: int,
) -> list[LintMessage]:
    # [FUNCTION DOC] Run ruff in fix-only mode on one file, feeding its content on standard input,
    # and return an empty list if the output is identical to the original, else a warning
    # LintMessage named format with the original and fixed text. Returns a command-failed error
    # message if ruff cannot be run. Reads the file; does not write it.
    try:
        with open(filename, "rb") as f:
            original = f.read()
        with open(filename, "rb") as f:
            proc_fix = run_command(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--fix-only",
                    "--exit-zero",
                    *([f"--config={config}"] if config else []),
                    "--stdin-filename",
                    filename,
                    "-",
                ],
                stdin=f,
                retries=retries,
                timeout=timeout,
                check=True,
            )
    except (OSError, subprocess.CalledProcessError) as err:
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
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        f"COMMAND (exit code {err.returncode})\n"
                        f"{' '.join(as_posix(x) for x in err.cmd)}\n\n"
                        f"STDERR\n{err.stderr.decode('utf-8').strip() or '(empty)'}\n\n"
                        f"STDOUT\n{err.stdout.decode('utf-8').strip() or '(empty)'}"
                    )
                ),
            )
        ]

    replacement = proc_fix.stdout
    if original == replacement:
        return []

    return [
        LintMessage(
            path=filename,
            name="format",
            description="Run `lintrunner -a` to apply this patch.",
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
        )
    ]


def _default_num_workers() -> int | None:
    # [FUNCTION DOC] HELPER: Return the worker count: the positive integer value of MAX_JOBS when
    # set, otherwise the CPU count.
    # Forks one process per file batch, so respect MAX_JOBS to cap parallelism.
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses options (workers, config, explain, show-disable, timeout,
    # severity overrides, no-fix and the shared options), configures logging to stderr and builds
    # the severity override mapping, raising an AssertionError for a malformed override. Runs the
    # check over all files and prints the messages, then unless disabled or there were no
    # findings, runs the fix check for each file with findings in a thread pool and prints its
    # messages. Logs and re-raises any failure.
    parser = argparse.ArgumentParser(
        description=f"Ruff linter. Linter code: {LINTER_CODE}. Use with RUFF-FIX to auto-fix issues.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=None,
        help="number of parallel workers (defaults to MAX_JOBS or the CPU count)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the `pyproject.toml` or `ruff.toml` file to use for configuration",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Explain a rule",
    )
    parser.add_argument(
        "--show-disable",
        action="store_true",
        help="Show how to disable a lint message",
    )
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="Seconds to wait for ruff",
    )
    parser.add_argument(
        "--severity",
        action="append",
        help="map code to severity (e.g. `F401:advice`). This option can be used multiple times.",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Do not suggest fixes",
    )
    add_default_options(parser)
    args = parser.parse_args()
    num_workers = args.num_workers or _default_num_workers()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    severities: dict[str, LintSeverity] = {}
    if args.severity:
        for severity in args.severity:
            parts = severity.split(":", 1)
            if len(parts) != 2:
                raise AssertionError(f"invalid severity `{severity}`")
            severities[parts[0]] = LintSeverity(parts[1])

    lint_messages = check_files(
        args.filenames,
        severities=severities,
        config=args.config,
        retries=args.retries,
        timeout=args.timeout,
        explain=args.explain,
        show_disable=args.show_disable,
    )
    for lint_message in lint_messages:
        lint_message.display()

    if args.no_fix or not lint_messages:
        # If we're not fixing, we can exit early
        return

    files_with_lints = {lint.path for lint in lint_messages if lint.path is not None}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file_for_fixes,
                path,
                config=args.config,
                retries=args.retries,
                timeout=args.timeout,
            ): path
            for path in files_with_lints
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    lint_message.display()
            except Exception:  # Catch all exceptions for lintrunner
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
