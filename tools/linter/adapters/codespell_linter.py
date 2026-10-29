# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "codespell[toml]==2.4.1",
# ]
# ///
# [MODULE DOC] Lintrunner adapter that runs codespell on files in a process pool with the
# repository dictionary of accepted words, and also validates the dictionary itself (no
# duplicates, case-insensitively sorted, no forbidden words). Prints JSON error messages.
# Import-time side effects: none beyond defining constants (REPO_ROOT, DICTIONARY path,
# forbidden words, 1 GB size cap).
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).absolute().parents[3]
DICTIONARY = REPO_ROOT / "tools" / "linter" / "dictionary.txt"

FORBIDDEN_WORDS = {
    "multipy",  # project pytorch/multipy is dead  # codespell:ignore multipy
}

MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB in bytes


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


def format_error_message(
    filename: str,
    error: Exception | None = None,
    *,
    message: str | None = None,
) -> LintMessage:
    # [FUNCTION DOC] Build an error-severity "spelling error" LintMessage for the file without
    # line or column. If no message is given but an exception is, the description names the
    # exception class and text and advises fixing the error or adding words to the dictionary.
    if message is None and error is not None:
        message = (
            f"Failed due to {error.__class__.__name__}:\n{error}\n"
            "Please either fix the error or add the word(s) to the dictionary file.\n"
            "HINT: all-lowercase words in the dictionary can cover all case variations."
        )
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="CODESPELL",
        severity=LintSeverity.ERROR,
        name="spelling error",
        original=None,
        replacement=None,
        description=message,
    )


def run_codespell(path: Path) -> str:
    # [FUNCTION DOC] Run codespell as a module under the current interpreter on the file with
    # the repository ignore-words dictionary, from the temp directory (to avoid codespell
    # reading a repo pyproject). Returns combined output on success; raises ValueError carrying
    # the output when codespell exits non-zero (i.e. misspellings found).
    try:
        return subprocess.check_output(
            [
                sys.executable,
                "-m",
                "codespell_lib",
                "--ignore-words",
                str(DICTIONARY),
                str(path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            # codespell (<= 2.4.2 and main) crashes reading a pyproject.toml that
            # has an array-of-tables directly under [tool] (e.g. the
            # [[tool.dynamic-metadata]] scikit-build-core requires): it feeds the
            # whole [tool] table to configparser.read_dict, which fails on the list
            # value. codespell also auto-reads ./pyproject.toml, so run it from a
            # directory that has none and pass the dictionary directly (the sole
            # [tool.codespell] setting we use). Upstream bug: codespell-project/codespell#3969.
            cwd=tempfile.gettempdir(),
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.output) from exc


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Spell-check one file. Files above the size cap give a warning and are
    # skipped; a size lookup failure gives an error message. Otherwise codespell is run and
    # any failure (including misspellings) becomes a single spelling error message; success
    # returns an empty list.
    path = Path(filename).absolute()

    # Check if file is too large
    try:
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            return [
                LintMessage(
                    path=filename,
                    line=None,
                    char=None,
                    code="CODESPELL",
                    severity=LintSeverity.WARNING,
                    name="file-too-large",
                    original=None,
                    replacement=None,
                    description=f"File size ({file_size} bytes) exceeds {MAX_FILE_SIZE} bytes limit, skipping",
                )
            ]
    except OSError as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CODESPELL",
                severity=LintSeverity.ERROR,
                name="file-access-error",
                original=None,
                replacement=None,
                description=f"Failed to get file size: {err}",
            )
        ]

    try:
        run_codespell(path)
    except Exception as err:
        return [format_error_message(filename, err)]
    return []


def check_dictionary(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Validate the dictionary file: lines must be unique, sorted
    # case-insensitively, and contain no forbidden word (in original or lower case). The first
    # violated rule, or any read error, is returned as a single error message; otherwise an
    # empty list.
    """Check the dictionary file for duplicates."""
    path = Path(filename).absolute()
    try:
        words = path.read_text(encoding="utf-8").splitlines()
        words_set = set(words)
        if len(words) != len(words_set):
            raise ValueError("The dictionary file contains duplicate entries.")
        # pyrefly: ignore [bad-argument-type]
        uncased_words = list(map(str.lower, words))
        if uncased_words != sorted(uncased_words):
            raise ValueError(
                "The dictionary file is not sorted alphabetically (case-insensitive)."
            )
        for forbidden_word in sorted(
            FORBIDDEN_WORDS & (words_set | set(uncased_words))
        ):
            raise ValueError(
                f"The dictionary file contains a forbidden word: {forbidden_word!r}. "
                "Please remove it from the dictionary file and use 'codespell:ignore' "
                "inline comment instead."
            )
    except Exception as err:
        return [format_error_message(filename, err)]
    return []


def _default_num_workers() -> int | None:
    # [FUNCTION DOC] HELPER: worker count default. Returns the MAX_JOBS environment value when
    # it is a positive integer, otherwise the CPU count (which may be None).
    # Forks one process per file batch, so respect MAX_JOBS to cap parallelism.
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


def main() -> None:
    # [FUNCTION DOC] CLI entry point. Parses verbose flag, worker count and file names,
    # configures logging to stderr, then in a process pool spell-checks every file and also
    # validates the dictionary file. Prints each lint message as JSON to stdout; unexpected
    # exceptions are logged with the target name and re-raised.
    parser = argparse.ArgumentParser(
        description="Check files for spelling mistakes using codespell.",
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
        futures[executor.submit(check_dictionary, str(DICTIONARY))] = str(DICTIONARY)
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
