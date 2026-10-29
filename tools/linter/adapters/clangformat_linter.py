# [MODULE DOC] Lintrunner adapter that runs clang-format on files in parallel threads and
# reports a format message (original and replacement text) for every file whose formatted
# output differs. Prints JSON lines to stdout. Defines IS_WINDOWS (platform flag); no other
# import-time side effects.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple


IS_WINDOWS: bool = os.name == "nt"


def _default_num_workers() -> int | None:
    # [FUNCTION DOC] HELPER: worker count default. Returns the MAX_JOBS environment value when
    # it is a positive integer, otherwise the CPU count (which may be None).
    # clang-format must respect MAX_JOBS to cap parallelism.
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


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


def as_posix(name: str) -> str:
    # [FUNCTION DOC] HELPER: on Windows replace backslashes with forward slashes in name;
    # on other platforms return name unchanged.
    return name.replace("\\", "/") if IS_WINDOWS else name


def _run_command(
    args: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] HELPER: run the command once with captured output, a timeout, and
    # check enabled (shell mode on Windows so batch scripts are found). Logs command and
    # elapsed time at debug level. Raises TimeoutExpired or CalledProcessError on failure.
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
            shell=IS_WINDOWS,  # So batch scripts are found.
            timeout=timeout,
            check=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def run_command(
    args: list[str],
    *,
    retries: int,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    # [FUNCTION DOC] Run the command, retrying only on timeout up to the given number of
    # retries with a one second sleep and a warning log between attempts. Returns the
    # CompletedProcess; re-raises TimeoutExpired once retries are exhausted. Other errors
    # (CalledProcessError, OSError) propagate immediately.
    remaining_retries = retries
    while True:
        try:
            return _run_command(args, timeout=timeout)
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


def check_file(
    filename: str,
    binary: str,
    retries: int,
    timeout: int,
) -> list[LintMessage]:
    # [FUNCTION DOC] Format one file with clang-format and compare against its current bytes.
    # Returns an empty list if identical; a single warning with original and replacement text
    # if different (the tool writes the formatted result to stdout; the file is not modified).
    # A timeout yields a timeout error message; OSError or a failing exit yields an advice
    # level command-failed message including the command, stderr and stdout.
    try:
        with open(filename, "rb") as f:
            original = f.read()
        proc = run_command(
            [binary, filename],
            retries=retries,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGFORMAT",
                severity=LintSeverity.ERROR,
                name="timeout",
                original=None,
                replacement=None,
                description=(
                    "clang-format timed out while trying to process a file. "
                    "Please report an issue in pytorch/pytorch with the "
                    "label 'module: lint'"
                ),
            )
        ]
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGFORMAT",
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
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
        ]

    replacement = proc.stdout
    if original == replacement:
        return []

    return [
        LintMessage(
            path=filename,
            line=None,
            char=None,
            code="CLANGFORMAT",
            severity=LintSeverity.WARNING,
            name="format",
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
            description="See https://clang.llvm.org/docs/ClangFormat.html.\nRun `lintrunner -a` to apply this patch.",
        )
    ]


def main() -> None:
    # [FUNCTION DOC] CLI entry point. Parses binary path, retries (default 3), timeout
    # (default 90 s), verbose flag, worker count and file names; configures logging to stderr.
    # If the binary path does not exist prints a JSON init-error and exits with status 0.
    # Otherwise checks all files in a thread pool and prints each lint message as JSON on
    # stdout; unexpected exceptions are logged and re-raised.
    parser = argparse.ArgumentParser(
        description="Format files with clang-format.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-format binary path",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out clang-format",
    )
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="seconds to wait for clang-format",
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
        help=(
            "number of clang-format processes to run in parallel. Defaults to "
            "the MAX_JOBS environment variable if set, otherwise the CPU count."
        ),
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
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

    binary = os.path.normpath(args.binary) if IS_WINDOWS else args.binary
    if not Path(binary).exists():
        lint_message = LintMessage(
            path=None,
            line=None,
            char=None,
            code="CLANGFORMAT",
            severity=LintSeverity.ERROR,
            name="init-error",
            original=None,
            replacement=None,
            description=(
                f"Could not find clang-format binary at {binary}, "
                "did you forget to run `lintrunner init`?"
            ),
        )
        print(json.dumps(lint_message._asdict()), flush=True)
        sys.exit(0)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(check_file, x, binary, args.retries, args.timeout): x
            for x in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
