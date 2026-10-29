# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "packaging==25.0",
#   "tomli==2.2.1 ; python_version < '3.11'",
# ]
# ///
# [MODULE DOC] Lintrunner adapter that enforces that no file requires a newer minimum CMake
# than the root CMakeLists.txt declares. Checks CMake files (cmake_minimum_required lines),
# pyproject build-system requirements and requirements text files for CMake pins, using a
# process pool, and prints JSON error messages. Import-time side effects: reads the root
# CMakeLists.txt to compute CMAKE_MINIMUM_VERSION (raises RuntimeError if the declaration is
# missing, or OSError if the file is unreadable).
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import logging
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from packaging.requirements import Requirement
from packaging.version import Version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).absolute().parents[3]

LINTER_CODE = "CMAKE_MINIMUM_REQUIRED"

CMAKE_MINIMUM_REQUIRED_PATTERN = re.compile(
    r"cmake_minimum_required\(VERSION\s+(?P<version>\d+\.\d+(\.\d+)?)\b.*\)",
    flags=re.IGNORECASE,
)


def _root_cmake_minimum_version() -> Version:
    # [FUNCTION DOC] HELPER: scan the root CMakeLists.txt line by line and return the version
    # of the first minimum-required declaration as a Version. Raises RuntimeError if there
    # is none; file read errors propagate.
    # The root CMakeLists.txt is the single source of truth for the minimum
    # CMake version -- scikit-build-core reads the same cmake_minimum_required
    # to decide which CMake to provision. Every other file is validated against
    # it, so bumping the root raises the enforced floor repo-wide.
    cmakelists = REPO_ROOT / "CMakeLists.txt"
    with cmakelists.open(encoding="utf-8") as f:
        for line in f:
            if match := CMAKE_MINIMUM_REQUIRED_PATTERN.search(line):
                return Version(match.group("version"))
    raise RuntimeError(f"No cmake_minimum_required found in {cmakelists}")


CMAKE_MINIMUM_VERSION = _root_cmake_minimum_version()


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
    line: int | None = None,
    message: str | None = None,
) -> LintMessage:
    # [FUNCTION DOC] Build an error-severity LintMessage for the file (optional line, no
    # column) named "CMake minimum version". If no message text is given but an exception is
    # given, the description becomes a failure text with the exception class name and text.
    if message is None and error is not None:
        message = f"Failed due to {error.__class__.__name__}:\n{error}"
    return LintMessage(
        path=filename,
        line=line,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name="CMake minimum version",
        original=None,
        replacement=None,
        description=message,
    )


def check_cmake(path: Path) -> list[LintMessage]:
    # [FUNCTION DOC] Scan a CMake file for minimum-required declarations; at the first one whose
    # version is greater than the root minimum, return a single error with its line number.
    # Returns an empty list otherwise. File read errors propagate (not caught here).
    # The root CMakeLists.txt defines CMAKE_MINIMUM_VERSION, so it trivially
    # satisfies this check; every other cmake file must not require a newer
    # CMake than the environment provides.
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if match := CMAKE_MINIMUM_REQUIRED_PATTERN.search(line):
                version = match.group("version")
                if Version(version) > CMAKE_MINIMUM_VERSION:
                    return [
                        format_error_message(
                            str(path),
                            line=i,
                            message=(
                                f"The environment can only provide CMake {CMAKE_MINIMUM_VERSION}, "
                                f"but found requiring {version}."
                            ),
                        )
                    ]
    return []


def check_requirement(
    requirement: Requirement,
    path: Path,
    *,
    line: int | None = None,
) -> LintMessage | None:
    # [FUNCTION DOC] Check one parsed requirement. Ignores anything not named cmake. For the
    # first exact-match or at-least specifier whose version (trailing wildcard removed) is
    # below the root minimum, returns an error message (with optional line); else None.
    if requirement.name.lower() != "cmake":
        return None

    for spec in requirement.specifier:
        if (
            spec.operator in ("==", ">=")
            and Version(spec.version.removesuffix(".*")) < CMAKE_MINIMUM_VERSION
        ):
            return format_error_message(
                str(path),
                line=line,
                message=(
                    f"CMake minimum version must be at least {CMAKE_MINIMUM_VERSION}, "
                    f"but found {spec}."
                ),
            )

    return None


def check_pyproject(path: Path) -> list[LintMessage]:
    # [FUNCTION DOC] Parse a pyproject TOML file and check each build-system requires entry with
    # check_requirement; returns the messages found. A parse or read error yields one error
    # message; a file of unexpected shape (no build-system table or requires list) yields an
    # empty list. Messages carry no line number.
    try:
        pyproject = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as err:
        return [format_error_message(str(path), err)]

    if not isinstance(pyproject, dict):
        return []
    if not isinstance(pyproject.get("build-system"), dict):
        return []

    build_system = pyproject["build-system"]
    requires = build_system.get("requires")
    if not isinstance(requires, list):
        return []
    return list(
        filter(
            None,
            (check_requirement(Requirement(req), path=path) for req in requires),
        )
    )


def check_requirements(path: Path) -> list[LintMessage]:
    # [FUNCTION DOC] Check a requirements-style text file line by line, skipping blank, comment
    # and option lines and lines that fail to parse as requirements. Returns the messages
    # from check_requirement with line numbers; a read error yields a single error message.
    try:
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as err:
        return [format_error_message(str(path), err)]

    lint_messages = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(line)
        except Exception:
            continue
        lint_message = check_requirement(requirement, path=path, line=i)
        if lint_message is not None:
            lint_messages.append(lint_message)

    return lint_messages


def check_file(filename: str) -> list[LintMessage]:
    # [FUNCTION DOC] Dispatch by lower-cased base name: CMake files (CMakeLists and cmake
    # modules or templates) go to check_cmake, pyproject.toml to check_pyproject, names
    # matching requirements text patterns to check_requirements; others give an empty list.
    path = Path(filename).absolute()
    basename = path.name.lower()
    if basename in ("cmakelists.txt", "cmakelists.txt.in") or basename.endswith(
        (".cmake", ".cmake.in")
    ):
        return check_cmake(path)
    if basename == "pyproject.toml":
        return check_pyproject(path)
    if fnmatch.fnmatch(basename, "*requirements*.txt") or fnmatch.fnmatch(
        basename, "*requirements*.in"
    ):
        return check_requirements(path)
    return []


def main() -> None:
    # [FUNCTION DOC] CLI entry point. Parses verbose flag and file names, configures logging to
    # stderr, checks all files in a process pool and prints each lint message as JSON to
    # stdout; unexpected exceptions are logged with the file name and re-raised.
    parser = argparse.ArgumentParser(
        description="Check consistency of cmake minimum version in requirement files.",
        fromfile_prefix_chars="@",
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
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

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


if __name__ == "__main__":
    main()
