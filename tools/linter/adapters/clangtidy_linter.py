# [MODULE DOC] Lintrunner adapter that runs clang-tidy on C++ files in parallel threads using
# a compile_commands.json build directory, parses its text output into JSON lint messages,
# and keeps only findings under the PyTorch root. Import-time side effects: locates the SCM
# root (raises RuntimeError if none), queries the system compiler for its include search
# directories (subprocess call, may raise), and builds the module-level include_args list.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from sysconfig import get_paths as gp
from typing import NamedTuple


# PyTorch directory root
def scm_root() -> str:
    # [FUNCTION DOC] Walk up from the current directory until a directory containing a git
    # entry or a mercurial directory is found; return its absolute path. Raises RuntimeError
    # when the filesystem root is reached without finding one.
    path = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        if os.path.isdir(os.path.join(path, ".hg")):
            return path
        n = len(path)
        path = os.path.dirname(path)
        if len(path) == n:
            raise RuntimeError("Unable to find SCM root")


PYTORCH_ROOT = scm_root()


def _default_num_workers() -> int | None:
    # [FUNCTION DOC] HELPER: worker count default. Returns the MAX_JOBS environment value when
    # it is a positive integer, otherwise the CPU count (which may be None).
    # clang-tidy is memory hungry, so respect MAX_JOBS to cap parallelism.
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


# Returns '/usr/local/include/python<version number>'
def get_python_include_dir() -> str:
    # [FUNCTION DOC] HELPER: return the include directory of the running Python interpreter
    # as reported by the sysconfig paths table.
    return gp()["include"]


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


# c10/core/DispatchKey.cpp:281:26: error: 'k' used after it was moved [bugprone-use-after-move]
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
            check=False,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


# Severity is either "error" or "note":
# https://github.com/python/mypy/blob/8b47a032e1317fb8e3f9a818005a6b63e9bf0311/mypy/errors.py#L46-L47
severities = {
    "error": LintSeverity.ERROR,
    "warning": LintSeverity.WARNING,
}


def clang_search_dirs() -> list[str]:
    # [FUNCTION DOC] Pick the first available compiler among clang, gcc, cpp, cc and run it in
    # preprocess-only verbose mode on empty input. Parse its stderr and return the list of
    # directories listed between the include search start line and the end-of-list line.
    # Raises RuntimeError if no compiler is found, CalledProcessError if the compiler fails.
    # KNOWN ISSUE: the error message prints the already-filtered (empty) compiler list, so it
    # never names the compilers that were looked for.
    # Compilers are ordered based on fallback preference
    # We pick the first one that is available on the system
    compilers = ["clang", "gcc", "cpp", "cc"]
    compilers = [c for c in compilers if shutil.which(c) is not None]
    if len(compilers) == 0:
        raise RuntimeError(f"None of {compilers} were found")
    compiler = compilers[0]

    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-", "-v"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    stderr = result.stderr.decode().strip().split("\n")
    search_start = r"#include.*search starts here:"
    search_end = r"End of search list."

    append_path = False
    search_paths = []
    for line in stderr:
        if re.match(search_start, line):
            if append_path:
                continue
            else:
                append_path = True
        elif re.match(search_end, line):
            break
        elif append_path:
            search_paths.append(line.strip())

    return search_paths


include_args = []
include_dir = [
    "/usr/lib/llvm-11/include/openmp",
    get_python_include_dir(),
    os.path.join(PYTORCH_ROOT, "third_party/pybind11/include"),
    # For header-only lints (no compile_commands.json entry) to resolve <ATen/...>.
    os.path.join(PYTORCH_ROOT, "aten/src"),
    PYTORCH_ROOT,
] + clang_search_dirs()
for dir in include_dir:
    include_args += ["--extra-arg", f"-I{dir}"]


def check_file(
    filename: str,
    binary: str,
    build_dir: Path,
    std: str | None,
) -> list[LintMessage]:
    # [FUNCTION DOC] Run clang-tidy on one file using the build directory's compile database,
    # the module-level include arguments plus the build directory include paths, and an
    # optional C++ standard flag. Temporarily changes the process working directory to the
    # build directory (restored in a finally block; process-wide, so racy across threads) to
    # resolve reported relative paths. Returns one message per parsed output line whose
    # absolute path is under the PyTorch root; severity from the table, defaulting to error;
    # negative columns become None. OSError yields a single command-failed error message.
    # Explicitly pass include path for linters that only check headers.
    # build/aten/src covers generated <ATen/...> headers (Functions.h etc.).
    build_include_args = include_args + [
        "--extra-arg",
        f"-I{build_dir}",
        "--extra-arg",
        f"-I{build_dir}/aten/src",
    ]
    cmd = [
        binary,
        f"-p={build_dir}",
        *build_include_args,
        filename,
    ]
    # Only add -- and -std flag if std is explicitly specified
    if std is not None:
        cmd.extend(["--", f"-std={std}"])

    try:
        proc = run_command(cmd)
    except OSError as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    lint_messages = []
    try:
        # Change the current working directory to the build directory, since
        # clang-tidy will report files relative to the build directory.
        saved_cwd = os.getcwd()
        os.chdir(build_dir)

        for match in RESULTS_RE.finditer(proc.stdout.decode()):
            # Convert the reported path to an absolute path.
            abs_path = str(Path(match["file"]).resolve())
            if not abs_path.startswith(PYTORCH_ROOT):
                continue
            message = LintMessage(
                path=abs_path,
                name=match["code"],
                description=match["message"],
                line=int(match["line"]),
                char=int(match["column"])
                if match["column"] is not None and not match["column"].startswith("-")
                else None,
                code="CLANGTIDY",
                severity=severities.get(match["severity"], LintSeverity.ERROR),
                original=None,
                replacement=None,
            )
            lint_messages.append(message)
    finally:
        os.chdir(saved_cwd)

    return lint_messages


def main() -> None:
    # [FUNCTION DOC] CLI entry point. Parses binary, build dir, optional C++ standard, verbose
    # flag, worker count and file names; configures logging to stderr. If the binary is
    # missing prints a JSON error and exits with status 0. Otherwise runs check_file for all
    # files in a thread pool using absolute binary and build paths and prints each lint
    # message as JSON; unexpected exceptions are logged and re-raised.
    parser = argparse.ArgumentParser(
        description="clang-tidy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="clang-tidy binary path",
    )
    parser.add_argument(
        "--build-dir",
        "--build_dir",
        required=True,
        help=(
            "Where the compile_commands.json file is located. "
            "Gets passed to clang-tidy -p"
        ),
    )
    parser.add_argument(
        "--std",
        default=None,
        help=(
            "C++ standard to use for compilation (e.g., c++17, c++20). "
            "If not specified, uses the standard from compile_commands.json."
        ),
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
            "number of clang-tidy processes to run in parallel. Defaults to the "
            "MAX_JOBS environment variable if set, otherwise the CPU count. "
            "Lower this to reduce peak memory usage (clang-tidy is memory hungry)."
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

    if not os.path.exists(args.binary):
        err_msg = LintMessage(
            path="<none>",
            line=None,
            char=None,
            code="CLANGTIDY",
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Could not find clang-tidy binary at {args.binary},"
                " you may need to run `lintrunner init`."
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    abs_build_dir = Path(args.build_dir).resolve()

    # Get the absolute path to clang-tidy and use this instead of the relative
    # path such as .lintbin/clang-tidy. The problem here is that os.chdir is
    # per process, and the linter uses it to move between the current directory
    # and the build folder. And there is no .lintbin directory in the latter.
    # When it happens in a race condition, the linter command will fails with
    # the following no such file or directory error: '.lintbin/clang-tidy'
    binary_path = os.path.abspath(args.binary)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                filename,
                binary_path,
                abs_build_dir,
                args.std,
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


if __name__ == "__main__":
    main()
