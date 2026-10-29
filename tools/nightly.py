#!/usr/bin/env python3
# Much of the logging code here was forked from https://github.com/ezyang/ghstack
# Copyright (c) Edward Z. Yang <ezyang@mit.edu>
# [MODULE DOC] Command line tool that sets up a PyTorch development environment from the
# pre-built nightly binaries. Subcommands: checkout (check out the commit matching the
# nightly wheel, optionally on a new branch) and pull (merge that commit into the current
# branch). It creates or reuses a virtual environment, downloads the nightly torch wheel,
# installs its dependencies plus dev tools, and moves/links the compiled artifacts from
# the wheel into the repo's torch directory, then writes a .pth file pointing to the repo.
# Import-time side effects: runs the binary build matrix generator script as a subprocess
# (which writes nightly_source_matrix.json), then loads that file into PIP_SOURCES.
# Also defines platform flags and global LOGGER/VERBOSE state set by main.
r"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.

You can use this script to check out a new nightly branch with the following::

    $ ./tools/nightly.py checkout -b my-nightly-branch
    $ source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows

Or if you would like to check out the nightly commit in detached HEAD mode::

    $ ./tools/nightly.py checkout
    $ source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows

Or if you would like to reuse an existing virtual environment, you can pass in
the prefix argument (--prefix)::

    $ ./tools/nightly.py checkout -b my-nightly-branch -p my-env
    $ source my-env/bin/activate  # or `& .\my-env\Scripts\Activate.ps1` on Windows

To install the nightly binaries built with CUDA, you can pass in the flag --cuda::

    $ ./tools/nightly.py checkout -b my-nightly-branch --cuda
    $ source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows

To install the nightly binaries built with ROCm, you can pass in the flag --rocm::

    $ ./tools/nightly.py checkout -b my-nightly-branch --rocm
    $ source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows

You can also use this tool to pull the nightly commits into the current branch as
well. This can be done with::

    $ ./tools/nightly.py pull
    $ source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows

To create the virtual environment with a specific Python interpreter, you can
pass in the --python argument::

    $ ./tools/nightly.py --python /path/to/python3.12
    $ source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows

Pulling will recreate a fresh virtual environment and reinstall the development
dependencies as well as the nightly binaries into the repo directory.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import itertools
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from ast import literal_eval
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from platform import system as platform_system
from typing import Any, cast, NamedTuple, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator


try:
    from packaging.version import Version
except ImportError:
    Version = None  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).absolute().parent.parent
GITHUB_REMOTE_URL = "https://github.com/pytorch/pytorch.git"
PACKAGES_TO_INSTALL = (
    "torch",
    "numpy",
    "cmake",
    "ninja",
    "packaging",
    "ruff",
    "mypy",
    "pytest",
    "hypothesis",
    "ipython",
    "rich",
    "clang-format",
    "clang-tidy",
    "sphinx",
)
DEFAULT_VENV_DIR = REPO_ROOT / "venv"


LOGGER: logging.Logger | None = None
VERBOSE: bool = False
DATETIME_FORMAT = "%Y-%m-%d_%Hh%Mm%Ss"
SHA1_RE = re.compile(r"(?P<sha1>[0-9a-fA-F]{40})")
USERNAME_PASSWORD_RE = re.compile(r":\/\/(.*?)\@")
LOG_DIRNAME_RE = re.compile(
    r"(?P<datetime>\d{4}-\d\d-\d\d_\d\dh\d\dm\d\ds)_"
    r"(?P<uuid>[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})",
)


PLATFORM = platform_system().replace("Darwin", "macOS")
LINUX = PLATFORM == "Linux"
MACOS = PLATFORM == "macOS"
WINDOWS = PLATFORM == "Windows"
POSIX = LINUX or MACOS


class PipSource(NamedTuple):
    # [CLASS DOC] Immutable description of one pip index to install nightly wheels from:
    # name (for example cpu or cuda-<version>), index_url, the set of platform names that
    # support it, and the accelerator kind. Built from entries of the nightly source matrix.
    name: str
    index_url: str
    supported_platforms: set[str]
    accelerator: str


# Generate: .github/scripts/nightly_source_matrix.json
GENERATE_MATRIX_SCRIPT = (
    REPO_ROOT / ".github" / "scripts" / "generate_binary_build_matrix.py"
)
subprocess.check_call(
    [sys.executable, str(GENERATE_MATRIX_SCRIPT)],
    cwd=GENERATE_MATRIX_SCRIPT.parent,
)

# See: .github/scripts/nightly_source_matrix.json
NIGHTLY_SOURCE_FILE = GENERATE_MATRIX_SCRIPT.with_name("nightly_source_matrix.json")
NIGHTLY_SOURCE_MATRIX = json.loads(NIGHTLY_SOURCE_FILE.read_text(encoding="utf-8"))
PIP_SOURCES = {name: PipSource(**data) for name, data in NIGHTLY_SOURCE_MATRIX.items()}


class Formatter(logging.Formatter):
    # [CLASS DOC] Logging formatter that scrubs secrets from every formatted record.
    # State: redactions, a mapping from sensitive substring to its replacement text.
    # INFO and DEBUG records are emitted as the bare message; other levels use the format.
    redactions: dict[str, str]

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        # [FUNCTION DOC] Initializes the base formatter with the given format strings and
        # starts with an empty redaction mapping.
        super().__init__(fmt, datefmt)
        self.redactions = {}

    # Remove sensitive information from URLs
    def _filter(self, s: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the text with URL credentials replaced by
        # placeholders and every registered redaction needle replaced by its replacement.
        s = USERNAME_PASSWORD_RE.sub(r"://<USERNAME>:<PASSWORD>@", s)
        for needle, replace in self.redactions.items():
            s = s.replace(needle, replace)
        return s

    def formatMessage(self, record: logging.LogRecord) -> str:
        # [FUNCTION DOC] Returns the raw message for INFO and DEBUG records without any
        # level prefix; for other levels defers to the base class formatting.
        if record.levelno == logging.INFO or record.levelno == logging.DEBUG:
            # Log INFO/DEBUG without any adornment
            return record.getMessage()
        else:
            # I'm not sure why, but formatMessage doesn't show up
            # even though it's in the typeshed for Python >3
            return super().formatMessage(record)

    def format(self, record: logging.LogRecord) -> str:
        # [FUNCTION DOC] Formats the record with the base class, then applies redaction.
        return self._filter(super().format(record))

    def redact(self, needle: str, replace: str = "<REDACTED>") -> None:
        # [FUNCTION DOC] Registers a substring to be replaced in all later output. Empty
        # needles are ignored. Mutates the redactions mapping; returns nothing.
        """Redact specific strings; e.g., authorization tokens.  This won't
        retroactively redact stuff you've already leaked, so make sure
        you redact things as soon as possible.
        """
        # Don't redact empty strings; this will lead to something
        # that looks like s<REDACTED>t<REDACTED>r<REDACTED>...
        if needle == "":
            return
        self.redactions[needle] = replace


@contextlib.contextmanager
def timer(logger: logging.Logger, prefix: str) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that measures the wall time of its body and logs an
    # info line with the prefix and elapsed seconds.
    # KNOWN ISSUE: the log line is not in a finally block, so nothing is logged if the
    # body raises.
    """Timed context manager"""
    start_time = time.perf_counter()
    yield
    logger.info("%s took %.3f [s]", prefix, time.perf_counter() - start_time)


F = TypeVar("F", bound=Callable[..., Any])


def timed(prefix: str) -> Callable[[F], F]:
    # [FUNCTION DOC] Decorator factory. The decorated function logs the prefix on entry
    # and its elapsed time on exit using the global LOGGER, which must already be set.
    """Decorator for timing functions"""

    def decorator(f: F) -> F:
        # [FUNCTION DOC] HELPER: Wraps the given function with the timing behavior,
        # preserving its metadata.
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Logs the prefix, runs the wrapped function inside
            # the timer and returns its result unchanged.
            logger = cast(logging.Logger, LOGGER)
            logger.info(prefix)
            with timer(logger, prefix):
                return f(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


class Venv:
    # [CLASS DOC] Manages a Python virtual environment at a prefix path: creation,
    # validation, running python/pip/uv/wheel commands inside it, and installing or
    # downloading packages from a given PipSource. State: prefix, pip_source,
    # base_executable (interpreter used to create it), cached executable and bindir,
    # and an environment mapping merged into every subprocess (pip/uv index, colors).
    """Virtual environment manager"""

    AGGRESSIVE_UPDATE_PACKAGES = (
        "uv",
        "pip",
        "setuptools<82",
        "packaging",
        "wheel",
        "build[uv]",
    )

    def __init__(
        self,
        prefix: Path | str,
        pip_source: PipSource,
        *,
        base_executable: Path | str | None = None,
    ) -> None:
        # [FUNCTION DOC] Resolves the base interpreter (default: the running one; a
        # non-absolute name is looked up on PATH) and records the absolute prefix, source
        # and subprocess environment. Raises RuntimeError if the interpreter is not found.
        base_executable = Path(base_executable or sys.executable)
        if not base_executable.is_absolute():
            base_exec = shutil.which(str(base_executable))
            if base_exec is None:
                raise RuntimeError(
                    f"Could not find Python executable {base_executable}",
                )
            base_executable = Path(base_exec)

        self.prefix = Path(prefix).absolute()
        self.pip_source = pip_source
        self.base_executable = base_executable.absolute()
        self._executable: Path | None = None
        self._bindir: Path | None = None
        self._env = {
            "PIP_EXTRA_INDEX_URL": self.pip_source.index_url,
            "UV_INDEX": self.pip_source.index_url,
            "UV_PYTHON_DOWNLOADS": "never",
            "FORCE_COLOR": "1",
            "CLICOLOR_FORCE": "1",
        }

    def is_venv(self) -> bool:
        # [FUNCTION DOC] Returns true if the prefix is a directory containing pyvenv.cfg.
        """Check if the prefix is a virtual environment."""
        return self.prefix.is_dir() and (self.prefix / "pyvenv.cfg").is_file()

    @property
    def bindir(self) -> Path:
        # [FUNCTION DOC] Property: the Scripts directory on Windows, otherwise bin, under
        # the prefix; cached. Raises AssertionError if the prefix is not a virtual env.
        """Get the bin directory for the virtual environment."""
        if not self.is_venv():
            raise AssertionError("Not a virtual environment")
        if self._bindir is None:
            if WINDOWS:
                self._bindir = self.prefix / "Scripts"
            else:
                self._bindir = self.prefix / "bin"
        return self._bindir

    @property
    def executable(self) -> Path:
        # [FUNCTION DOC] Property: the venv's python executable path (python.exe on
        # Windows), cached. Raises AssertionError if the prefix is not a venv or the file
        # is missing or not executable.
        """Get the Python executable for the virtual environment."""
        if not self.is_venv():
            raise AssertionError("Not a virtual environment")
        if self._executable is None:
            executable = self.bindir / ("python.exe" if WINDOWS else "python")
            if not (executable.is_file() or executable.is_symlink()):
                raise AssertionError(f"{executable} is not a file or symlink")
            if not os.access(executable, os.X_OK):
                raise AssertionError(f"{executable} is not executable")
            self._executable = executable
        return self._executable

    def site_packages(self, python: Path | str | None = None) -> Path:
        # [FUNCTION DOC] Asks the interpreter (default: the venv python) for its site
        # directories and returns the first existing one named site-packages. Raises
        # RuntimeError if none exists. Runs a subprocess.
        """Get the site-packages directory for the virtual environment."""
        output = self.python(
            "-c",
            "import site; [print(p) for p in site.getsitepackages()]",
            python=python,
            capture_output=True,
        ).stdout
        # pyrefly: ignore [bad-argument-type]
        candidates = list(map(Path, filter(None, map(str.strip, output.splitlines()))))
        candidates = [p for p in candidates if p.is_dir() and p.name == "site-packages"]
        if not candidates:
            raise RuntimeError(
                f"No site-packages directory found for executable {python}"
            )
        return candidates[0]

    @property
    def activate_script(self) -> Path:
        # [FUNCTION DOC] Property: path of the shell activation script (Scripts/activate
        # on Windows, bin/activate elsewhere). Does not check that it exists.
        """Get the activation script for the virtual environment."""
        if WINDOWS:
            # Assume PowerShell
            return self.prefix / "Scripts" / "activate"
        # Assume POSIX-compliant shell: Bash, Zsh, etc.
        return self.prefix / "bin" / "activate"

    @property
    def activate_command(self) -> str:
        # [FUNCTION DOC] Property: shell command text that activates the venv (dot-source
        # on Windows, source with a quoted path elsewhere).
        """Get the command to activate the virtual environment."""
        if WINDOWS:
            # Assume PowerShell
            return f'. "{self.activate_script}"'
        # Assume Bash, Zsh, etc.
        # POSIX standard should use dot `. venv/bin/activate` rather than `source`
        return f"source {shlex.quote(str(self.activate_script))}"

    @timed("Creating virtual environment")
    def create(
        self,
        *,
        remove_if_exists: bool = False,
        assume_yes: bool = False,
    ) -> Path:
        # [FUNCTION DOC] Creates the venv at the prefix with the base interpreter. If the
        # prefix exists: raises RuntimeError unless remove_if_exists; when set, refuses
        # (RuntimeError) if it is not a venv or is one of the interpreters' prefixes,
        # asks for confirmation unless assume_yes (exits with status 1 on refusal), then
        # deletes it. Adds a gitignore, and on Linux appends a snippet to the activate
        # script that extends the library path with NVIDIA package libraries. Returns the
        # result of ensure (the prefix path). Side effects: filesystem, subprocesses, stdout.
        """Create a virtual environment."""
        if self.prefix.exists():
            if remove_if_exists:
                # If the venv directory already exists, remove it first
                if not self.is_venv():
                    raise RuntimeError(
                        f"The path {self.prefix} already exists and is not a virtual environment. "
                        "Please remove it manually or choose a different prefix."
                    )
                if any(
                    Path(p).absolute().samefile(self.prefix)
                    for p in [
                        sys.prefix,
                        sys.exec_prefix,
                        sys.base_prefix,
                        sys.base_exec_prefix,
                    ]
                ):
                    raise RuntimeError(
                        f"The path {self.prefix} trying to remove is the same as the interpreter "
                        "to run this script. Please choose a different prefix or deactivate the "
                        "current virtual environment."
                    )
                if any(
                    Path(
                        self.base_python(
                            "-c",
                            f"import os, sys; print(os.path.abspath({p}))",
                            capture_output=True,
                        ).stdout.strip()
                    )
                    .absolute()
                    .samefile(self.prefix)
                    for p in [
                        "sys.prefix",
                        "sys.exec_prefix",
                        "sys.base_prefix",
                        "sys.base_exec_prefix",
                    ]
                ):
                    raise RuntimeError(
                        f"The Python executable {self.base_executable} trying to remove is the "
                        "same as the interpreter to create the virtual environment. Please choose "
                        "a different prefix or a different Python interpreter."
                    )
                if not assume_yes:
                    answer = input(
                        f"The virtual environment {self.prefix} already exists. "
                        "Do you want to remove it and recreate it? [y/N] "
                    )
                    if answer.lower() not in ("y", "yes"):
                        if answer.lower() not in ("n", "no", ""):
                            print(f"Invalid answer: {answer!r}")
                        else:
                            print(f"Aborting due to existing prefix: {self.prefix}")
                        sys.exit(1)

                print(f"Removing existing venv: {self.prefix}")
                _remove_existing(self.prefix)
            else:
                raise RuntimeError(f"Path {self.prefix} already exists.")

        print(f"Creating venv (Python {self.base_python_version()}): {self.prefix}")
        self.base_python("-m", "venv", str(self.prefix))
        if not self.is_venv():
            raise AssertionError("Failed to create virtual environment.")
        (self.prefix / ".gitignore").write_text("*\n", encoding="utf-8")

        if LINUX:
            activate_script = self.activate_script
            st_mode = activate_script.stat().st_mode
            # The activate script may be read-only and we need to add write permissions
            activate_script.chmod(st_mode | 0o200)
            with activate_script.open(mode="a", encoding="utf-8") as f:
                f.write(
                    "\n"
                    + textwrap.dedent(
                        f"""
                        # Add NVIDIA PyPI packages to LD_LIBRARY_PATH
                        export LD_LIBRARY_PATH="$(
                            {self.executable.name} - <<EOS
                        import glob
                        import itertools
                        import os
                        import site

                        nvidia_libs = [
                            p.rstrip("/")
                            for p in itertools.chain.from_iterable(
                                glob.iglob(f"{{site_dir}}/{{pattern}}/", recursive=True)
                                for site_dir in site.getsitepackages()
                                for pattern in ("nvidia/**/lib", "cu*/**/lib")
                            )
                        ]
                        ld_library_path = os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
                        print(os.pathsep.join(dict.fromkeys(nvidia_libs + ld_library_path)))
                        EOS
                        )"
                        """
                    ).strip()
                    + "\n"
                )
            # Change the file mode back
            activate_script.chmod(st_mode)

        return self.ensure()

    def ensure(self) -> Path:
        # [FUNCTION DOC] Creates the venv (replacing any existing path) if it is not one;
        # otherwise checks that its major.minor Python version matches the base
        # interpreter (RuntimeError on mismatch) and upgrades the core packaging tools.
        # Returns the prefix.
        """Ensure the virtual environment exists."""
        if not self.is_venv():
            return self.create(remove_if_exists=True)
        if (
            self.python_version().split(".")[:2]
            != self.base_python_version().split(".")[:2]
        ):
            raise RuntimeError(
                f"Python version mismatch: venv has Python {self.python_version()} "
                f"but base Python is {self.base_python_version()}. "
                "Please recreate the virtual environment with the correct Python version."
            )

        self.pip_install(*self.AGGRESSIVE_UPDATE_PACKAGES, upgrade=True)
        return self.prefix

    def python(
        self,
        *args: str,
        python: Path | str | None = None,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Runs the given interpreter (default venv python) with the args
        # as a text-mode subprocess. Environment is the process environment plus the
        # venv mapping plus an optional env override; check defaults to true (raises
        # CalledProcessError on failure). Other keyword args go to subprocess run.
        """Run a Python command in the virtual environment."""
        if python is None:
            python = self.executable
        cmd = [str(python), *args]
        env = popen_kwargs.pop("env", None) or {}
        check = popen_kwargs.pop("check", True)
        # pyrefly: ignore [no-matching-overload]
        return subprocess.run(
            cmd,
            check=check,
            text=True,
            encoding="utf-8",
            env={**os.environ, **self._env, **env},
            **popen_kwargs,
        )

    def base_python(
        self,
        *args: str,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Same as python but always uses the base interpreter.
        """Run a Python command in the base environment."""
        return self.python(*args, python=self.base_executable, **popen_kwargs)

    def python_version(self, *, python: Path | str | None = None) -> str:
        # [FUNCTION DOC] Runs the interpreter to print its version as major.minor.micro
        # followed by ABI flags; returns the stripped string.
        """Get the Python version for the virtual environment.

        Return a string like "3.13.7", "3.13.7t", "3.13.7d", "3.13.7td", etc.
        """
        return self.python(
            "-c",
            (
                "import sys; print('{0.major}.{0.minor}.{0.micro}{1}'."
                "format(sys.version_info, getattr(sys, 'abiflags', '')))"
            ),
            python=python,
            capture_output=True,
        ).stdout.strip()

    def base_python_version(self) -> str:
        # [FUNCTION DOC] Version string of the base interpreter (see python_version).
        """Get the Python version for the base environment.

        Return a string like "3.13.7", "3.13.7t", "3.13.7d", "3.13.7td", etc.
        """
        return self.python_version(python=self.base_executable)

    def uv(
        self,
        *args: str,
        python: Path | str | None = None,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Runs the uv program from the venv bin directory with the args,
        # targeting the given interpreter (default venv python) through the UV_PYTHON
        # variable. Environment and check handling are the same as in python.
        """Run a uv command in the virtual environment."""
        if python is None:
            python = self.executable
        cmd = [str(self.bindir / "uv"), *args]
        env = popen_kwargs.pop("env", None) or {}
        check = popen_kwargs.pop("check", True)
        # pyrefly: ignore [no-matching-overload]
        return subprocess.run(
            cmd,
            check=check,
            text=True,
            encoding="utf-8",
            env={**os.environ, **self._env, **env, "UV_PYTHON": str(python)},
            **popen_kwargs,
        )

    @timed("Installing packages")
    def uv_pip_install(
        self,
        *packages: str,
        prerelease: bool = False,
        upgrade: bool = False,
        no_deps: bool = False,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Prints the packages and runs uv pip install with optional
        # verbose, prerelease, upgrade and no-deps flags. Returns the completed process.
        # KNOWN ISSUE: the prerelease flag is passed to uv without a value, but uv's
        # prerelease option expects a strategy argument, so the next argument would be
        # consumed as its value.
        """Run a pip install command in the virtual environment."""
        uv_pip_args = []
        if VERBOSE:
            uv_pip_args.append("-v")
        if prerelease:
            uv_pip_args.append("--prerelease")
        if upgrade:
            uv_pip_args.append("--upgrade")
            verb = "Upgrading"
        else:
            verb = "Installing"
        if no_deps:
            uv_pip_args.append("--no-deps")
        print(f"{verb} package(s) ({self.pip_source.index_url}):")
        for package in packages:
            print(f"  - {os.path.basename(package)}")
        return self.uv("pip", "install", *uv_pip_args, *packages, **popen_kwargs)

    def pip(self, *args: str, **popen_kwargs: Any) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Runs the pip module of the venv python with the given args.
        """Run a pip command in the virtual environment."""
        return self.python("-m", "pip", *args, **popen_kwargs)

    @timed("Installing packages")
    def pip_install(
        self,
        *packages: str,
        prerelease: bool = False,
        upgrade: bool = False,
        no_deps: bool = False,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Prints the packages and runs pip install with optional verbose,
        # pre-release, upgrade and no-deps flags. Returns the completed process.
        """Run a pip install command in the virtual environment."""
        pip_args = []
        if VERBOSE:
            pip_args.append("-v")
        if prerelease:
            pip_args.append("--pre")
        if upgrade:
            pip_args.append("--upgrade")
            verb = "Upgrading"
        else:
            verb = "Installing"
        if no_deps:
            pip_args.append("--no-deps")
        print(f"{verb} package(s) ({self.pip_source.index_url}):")
        for package in packages:
            print(f"  - {os.path.basename(package)}")
        return self.pip("install", *pip_args, *packages, **popen_kwargs)

    @timed("Downloading packages")
    def pip_download(
        self,
        *packages: str,
        prerelease: bool = False,
        no_deps: bool = False,
        **popen_kwargs: Any,
    ) -> list[Path]:
        # [FUNCTION DOC] Downloads the packages with pip into a new temporary directory
        # that is deleted at interpreter exit, prints what was downloaded, and returns
        # the list of downloaded file paths. Raises CalledProcessError if pip fails.
        """Download a package in the virtual environment."""
        tmpdir = tempfile.TemporaryDirectory(prefix="pip-download-")
        atexit.register(tmpdir.cleanup)
        tempdir = Path(tmpdir.name).absolute()
        print(
            f"Downloading package(s) ({self.pip_source.index_url}): "
            f"{', '.join(packages)}"
        )
        pip_args = []
        if VERBOSE:
            pip_args.append("-v")
        if prerelease:
            pip_args.append("--pre")
        if no_deps:
            pip_args.append("--no-deps")
        self.pip("download", f"--dest={tempdir}", *pip_args, *packages, **popen_kwargs)
        files = list(tempdir.iterdir())
        print(f"Downloaded {len(files)} file(s) to {tempdir}:")
        for file in files:
            print(f"  - {file.name}")
        return files

    def wheel(
        self,
        *args: str,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Runs the wheel module of the venv python with the given args.
        """Run a wheel command in the virtual environment."""
        return self.python("-m", "wheel", *args, **popen_kwargs)

    @timed("Unpacking wheel file")
    def wheel_unpack(
        self,
        wheel: Path | str,
        dest: Path | str,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Unpacks a .whl file into the destination directory using the
        # wheel tool. Raises AssertionError if the path is not a wheel file.
        """Unpack a wheel into a directory."""
        wheel = Path(wheel).absolute()
        dest = Path(dest).absolute()
        if not (wheel.is_file() and wheel.suffix.lower() == ".whl"):
            raise AssertionError(f"{wheel} is not a valid wheel file")
        return self.wheel("unpack", f"--dest={dest}", str(wheel), **popen_kwargs)

    @contextlib.contextmanager
    def extracted_wheel(self, wheel: Path | str) -> Generator[Path]:
        # [FUNCTION DOC] Context manager: unpacks the given local wheel into a temporary
        # directory and yields the single unpacked subdirectory (the site directory
        # contents). Raises RuntimeError if there is not exactly one subdirectory. The
        # temporary directory is removed on exit.
        """Download and extract a wheel into a temporary directory."""
        with tempfile.TemporaryDirectory(prefix="wheel-") as tempdir:
            self.wheel_unpack(wheel, tempdir)
            subdirs = [p for p in Path(tempdir).absolute().iterdir() if p.is_dir()]
            if len(subdirs) != 1:
                raise RuntimeError(
                    f"Expected exactly one directory in {tempdir}, "
                    f"got {[str(d) for d in subdirs]}."
                )
            yield subdirs[0]


def git(*args: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Builds a git command line (as a list) that runs in the repo
    # root with the given arguments. Does not execute anything.
    return ["git", "-C", str(REPO_ROOT), *args]


@functools.lru_cache
def logging_base_dir() -> Path:
    # [FUNCTION DOC] HELPER: Returns (creating if needed, cached) the nightly/log
    # directory under the repo root.
    base_dir = REPO_ROOT / "nightly" / "log"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


@functools.lru_cache
def logging_run_dir() -> Path:
    # [FUNCTION DOC] HELPER: Creates (once, cached) a per-run log directory named by
    # timestamp and a uuid inside the base log directory, and returns it.
    base_dir = logging_base_dir()
    cur_dir = base_dir / f"{datetime.now().strftime(DATETIME_FORMAT)}_{uuid.uuid1()}"
    cur_dir.mkdir(parents=True, exist_ok=True)
    return cur_dir


@functools.lru_cache
def logging_record_argv() -> None:
    # [FUNCTION DOC] HELPER: Writes the command line of this run into an argv file in
    # the run log directory (only once because of caching).
    s = subprocess.list2cmdline(sys.argv)
    (logging_run_dir() / "argv").write_text(s, encoding="utf-8")


def logging_record_exception(e: BaseException) -> None:
    # [FUNCTION DOC] HELPER: Writes the exception type and message to an exception file
    # in the run log directory, adding captured stdout and stderr for a failed subprocess.
    text = f"{type(e).__name__}: {e}"
    if isinstance(e, subprocess.CalledProcessError):
        text += f"\n\nstdout: {e.stdout}\n\nstderr: {e.stderr}"
    (logging_run_dir() / "exception").write_text(text, encoding="utf-8")


def logging_rotate() -> None:
    # [FUNCTION DOC] HELPER: Deletes all but the newest 1000 entries (sorted by name,
    # newest first) of the log base directory, skipping names that do not look like run logs.
    log_base = logging_base_dir()
    old_logs = sorted(log_base.iterdir(), reverse=True)
    for stale_log in old_logs[1000:]:
        # Sanity check that it looks like a log
        if LOG_DIRNAME_RE.fullmatch(stale_log.name) is not None:
            shutil.rmtree(stale_log)


@contextlib.contextmanager
def logging_manager(*, debug: bool = False) -> Generator[logging.Logger, None, None]:
    # [FUNCTION DOC] Context manager that configures the pytorch-nightly logger with a
    # console handler (info, or debug when requested) and a file handler in the run log
    # directory, records argv, rotates old logs, and yields the logger. On any exception
    # it logs it, saves it to the run directory, prints the log path and exits with
    # status 1.
    # KNOWN ISSUE: the handlers are never removed, so repeated use would duplicate output.
    """Setup logging. If a failure starts here we won't
    be able to save the user in a reasonable way.

    Logging structure: there is one logger (the root logger)
    and in processes all events.  There are two handlers:
    stderr (INFO) and file handler (DEBUG).
    """
    formatter = Formatter(fmt="%(levelname)s: %(message)s", datefmt="")
    root_logger = logging.getLogger("pytorch-nightly")
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = logging_run_dir() / "nightly.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging_record_argv()

    try:
        logging_rotate()
        print(f"log file: {log_file}")
        yield root_logger
    except Exception as e:
        logging.exception("Fatal exception")  # noqa: LOG015
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)
    except BaseException as e:
        # You could logging.debug here to suppress the backtrace
        # entirely, but there is no reason to hide it from technically
        # savvy users.
        logging.info("", exc_info=True)  # noqa: LOG015
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)


def check_branch(subcommand: str, branch: str | None) -> str | None:
    # [FUNCTION DOC] For the checkout subcommand, returns an error message if the working
    # tree (ignoring untracked files) is dirty or if the named branch already exists;
    # returns None when all is fine or for other subcommands. Runs git subprocesses.
    """Checks that the branch name can be checked out."""
    if subcommand != "checkout":
        return None
    # next check that the local repo is clean
    cmd = git("status", "--untracked-files=no", "--porcelain")
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    if stdout.strip():
        return "Need to have clean working tree to checkout!\n\n" + stdout
    # next check that the branch name doesn't already exist (if a branch name is provided)
    if branch is not None:
        cmd = git("show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
        p = subprocess.run(cmd, capture_output=True, check=False)  # type: ignore[assignment]
        if not p.returncode:
            return f"Branch {branch!r} already exists"
    return None


@timed("Installing dependencies")
def install_packages(venv: Venv, packages: Iterable[str]) -> None:
    # [FUNCTION DOC] Removes duplicate package specs (keeping order) and installs the
    # rest into the venv with uv; does nothing for an empty list.
    """Install dependencies to deps environment"""
    # install packages
    packages = list(dict.fromkeys(packages))
    if packages:
        venv.uv_pip_install(*packages)


def _ensure_commit(git_sha1: str) -> None:
    # [FUNCTION DOC] HELPER: Checks whether the commit exists in the local repo and, if
    # not, fetches it from the GitHub remote. Raises CalledProcessError if fetch fails.
    """Make sure that we actually have the commit locally"""
    cmd = git("cat-file", "-e", git_sha1 + r"^{commit}")
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode == 0:
        # we have the commit locally
        return
    # we don't have the commit, must fetch
    cmd = git("fetch", GITHUB_REMOTE_URL, git_sha1)
    subprocess.check_call(cmd)


def _nightly_version(site_dir: Path) -> str:
    # [FUNCTION DOC] HELPER: Reads the git version recorded in the unpacked wheel's
    # torch version file, fetches that commit if needed, extracts the nightly commit
    # hash from its subject line, ensures that commit is available, and returns it.
    # Raises RuntimeError if the git version line or the hash cannot be found.
    # first get the git version from the installed module
    version_file = site_dir / "torch" / "version.py"
    with version_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith("git_version"):
                continue
            git_version = literal_eval(line.partition("=")[2].strip())
            break
        else:
            raise RuntimeError(f"Could not find git_version in {version_file}")

    print(f"Found released git version {git_version}")
    # now cross reference with nightly version
    _ensure_commit(git_version)
    cmd = git("show", "--no-patch", "--format=%s", git_version)
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    m = SHA1_RE.search(stdout)
    if m is None:
        raise RuntimeError(
            f"Could not find nightly release in git history:\n  {stdout}"
        )
    nightly_version = m.group("sha1")
    print(f"Found nightly release version {nightly_version}")
    # now checkout nightly version
    _ensure_commit(nightly_version)
    return nightly_version


@timed("Checking out nightly PyTorch")
def checkout_nightly_version(branch: str | None, site_dir: Path) -> None:
    # [FUNCTION DOC] Resolves the nightly commit from the wheel and checks it out,
    # detached when branch is None, else on a new branch with that name.
    """Gets the nightly version and then checks it out."""
    nightly_version = _nightly_version(site_dir)
    if branch is None:
        # Detached mode - explicitly use --detach flag
        cmd = git("checkout", "--detach", nightly_version)
    else:
        # Branch mode
        cmd = git("checkout", "-b", branch, nightly_version)
    subprocess.check_call(cmd)


@timed("Pulling nightly PyTorch")
def pull_nightly_version(site_dir: Path) -> None:
    # [FUNCTION DOC] Resolves the nightly commit from the wheel and merges it into the
    # current branch.
    """Fetches the nightly version and then merges it ."""
    nightly_version = _nightly_version(site_dir)
    cmd = git("merge", nightly_version)
    subprocess.check_call(cmd)


def _get_listing_linux(source_dir: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Lists shared objects in the source dir and in its lib
    # subdirectory (including versioned names).
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.so"),
            (source_dir / "lib").glob("*.so.*"),
        )
    )


def _get_listing_macos(source_dir: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Lists shared objects in the source dir plus dylibs in its
    # lib subdirectory.
    # oddly, these are .so files even on Mac
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.dylib"),
        )
    )


def _get_listing_windows(source_dir: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Lists pyd files in the source dir plus lib and dll files in
    # its lib subdirectory.
    # KNOWN ISSUE: the dll pattern lacks a wildcard (it matches only a file literally
    # named .dll), so real dll files are not listed.
    return list(
        itertools.chain(
            source_dir.glob("*.pyd"),
            (source_dir / "lib").glob("*.lib"),
            (source_dir / "lib").glob(".dll"),
        )
    )


def _glob_pyis(d: Path) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of posix-style relative paths of all stub
    # files found recursively under the directory.
    return {p.relative_to(d).as_posix() for p in d.rglob("*.pyi")}


def _find_missing_pyi(source_dir: Path, target_dir: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns, sorted, the stub files present in the source dir
    # but absent from the target dir (as paths under the source dir).
    source_pyis = _glob_pyis(source_dir)
    target_pyis = _glob_pyis(target_dir)
    missing_pyis = sorted(source_dir / p for p in (source_pyis - target_pyis))
    return missing_pyis


def _get_listing(source_dir: Path, target_dir: Path) -> list[Path]:
    # [FUNCTION DOC] HELPER: Builds the list of items to move from the wheel's torch
    # directory: platform binaries, missing stubs, the version file, the generated test
    # directory, bin and include. Raises RuntimeError on an unknown platform.
    if LINUX:
        listing = _get_listing_linux(source_dir)
    elif MACOS:
        listing = _get_listing_macos(source_dir)
    elif WINDOWS:
        listing = _get_listing_windows(source_dir)
    else:
        raise RuntimeError(f"Platform {platform_system()!r} not recognized")
    listing.extend(_find_missing_pyi(source_dir, target_dir))
    listing.append(source_dir / "version.py")
    listing.append(source_dir / "testing" / "_internal" / "generated")
    listing.append(source_dir / "bin")
    listing.append(source_dir / "include")
    return listing


def _remove_existing(path: Path) -> None:
    # [FUNCTION DOC] HELPER: Deletes the path if it exists, recursively for directories.
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _move_single(
    src: Path,
    source_dir: Path,
    target_dir: Path,
    mover: Callable[[Path, Path], None],
    verb: str,
) -> None:
    # [FUNCTION DOC] HELPER: Moves one file or directory tree from the source dir to the
    # same relative place in the target dir, first deleting any existing target. The
    # mover callable (copy or link) is applied per file; the verb is used for verbose output.
    relpath = src.relative_to(source_dir)
    trg = target_dir / relpath
    _remove_existing(trg)
    # move over new files
    if src.is_dir():
        trg.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(src):
            relroot = Path(root).relative_to(src)
            for name in files:
                relname = relroot / name
                s = src / relname
                t = trg / relname
                if VERBOSE:
                    print(f"{verb} {s} -> {t}")
                mover(s, t)
            for name in dirs:
                (trg / relroot / name).mkdir(parents=True, exist_ok=True)
    else:
        if VERBOSE:
            print(f"{verb} {src} -> {trg}")
        mover(src, trg)


def _copy_files(listing: list[Path], source_dir: Path, target_dir: Path) -> None:
    # [FUNCTION DOC] HELPER: Copies each listed item into the target dir, with metadata.
    for src in listing:
        # pyrefly: ignore [bad-argument-type]
        _move_single(src, source_dir, target_dir, shutil.copy2, "Copying")


def _link_files(listing: list[Path], source_dir: Path, target_dir: Path) -> None:
    # [FUNCTION DOC] HELPER: Hard-links each listed item into the target dir.
    for src in listing:
        _move_single(src, source_dir, target_dir, os.link, "Linking")


@timed("Moving nightly files into repo")
def move_nightly_files(site_dir: Path) -> None:
    # [FUNCTION DOC] Moves the nightly binaries and related files from the unpacked wheel
    # into the repo's torch directory: copies on Windows, hard links elsewhere with a
    # fall back to copying if linking fails. Mutates the repo tree.
    """Moves PyTorch files from temporary installed location to repo."""
    # get file listing
    source_dir = site_dir / "torch"
    target_dir = REPO_ROOT / "torch"
    listing = _get_listing(source_dir, target_dir)
    # copy / link files
    if WINDOWS:
        _copy_files(listing, source_dir, target_dir)
    else:
        try:
            _link_files(listing, source_dir, target_dir)
        except Exception:
            _copy_files(listing, source_dir, target_dir)


@timed("Writing pytorch-nightly.pth")
def write_pth(venv: Venv) -> None:
    # [FUNCTION DOC] Writes a pytorch-nightly path file into the venv's site-packages so
    # that the repo root is importable, with a header explaining how to remove it.
    """Writes Python path file for this dir."""
    (venv.site_packages() / "pytorch-nightly.pth").write_text(
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{REPO_ROOT}\n",
        encoding="utf-8",
    )


def parse_dependencies(
    venv: Venv,
    wheel_site_dir: Path,
) -> list[str]:
    # [FUNCTION DOC] Reads the wheel's metadata file (from its single dist-info
    # directory) with the venv interpreter and returns requirement strings whose
    # environment markers evaluate true. Raises RuntimeError if the dist-info directory
    # or metadata file is missing or ambiguous.
    """Parse dependencies from the torch wheel's metadata."""
    dist_info_dirs = list(wheel_site_dir.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one .dist-info directory in {wheel_site_dir}, "
            f"got {dist_info_dirs}"
        )
    dist_info_dir = dist_info_dirs[0]
    if not (dist_info_dir / "METADATA").is_file():
        raise RuntimeError(
            f"Expected METADATA file in {dist_info_dir}, but it does not exist."
        )

    # Use the Python interpreter in the virtual environment instead of the interpreter
    # running this script, so that we can evaluate markers correctly.
    dependencies = (
        venv.python(
            "-c",
            textwrap.dedent(
                """
                from packaging.metadata import Metadata

                with open("METADATA", encoding="utf-8") as f:
                    metadata = Metadata.from_email(f.read())
                for req in metadata.requires_dist:
                    if req.marker is None or req.marker.evaluate():
                        print(req)
                """
            ).strip(),
            cwd=dist_info_dir,
            capture_output=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return [dep.strip() for dep in dependencies]


def install(
    *,
    venv: Venv,
    packages: Iterable[str],
    subcommand: str = "checkout",
    branch: str | None = None,
    fresh_venv: bool = False,
    assume_yes: bool = False,
) -> None:
    # [FUNCTION DOC] Full workflow: ensure or recreate the venv, download the nightly
    # torch wheel without dependencies (RuntimeError unless exactly one wheel), install
    # its dependencies and the extra packages, then check out or merge the matching
    # commit (ValueError for an unknown subcommand), move the binaries into the repo,
    # write the path file and log activation instructions.
    """Development install of PyTorch"""
    if not fresh_venv:
        print(f"Using existing venv: {venv.prefix}")
        venv.ensure()
    else:
        venv.create(remove_if_exists=True, assume_yes=assume_yes)

    packages = [p for p in packages if p != "torch"]

    downloaded_files = venv.pip_download("torch", prerelease=True, no_deps=True)
    if len(downloaded_files) != 1:
        raise RuntimeError(f"Expected exactly one torch wheel, got {downloaded_files}")
    torch_wheel = downloaded_files[0]
    if not (
        torch_wheel.name.startswith("torch-") and torch_wheel.name.endswith(".whl")
    ):
        raise RuntimeError(f"Expected exactly one torch wheel, got {torch_wheel}")

    with venv.extracted_wheel(torch_wheel) as wheel_site_dir:
        dependencies = parse_dependencies(venv, wheel_site_dir)
        install_packages(venv, [*dependencies, *packages])

        if subcommand == "checkout":
            checkout_nightly_version(branch, wheel_site_dir)
        elif subcommand == "pull":
            pull_nightly_version(wheel_site_dir)
        else:
            raise ValueError(f"Subcommand {subcommand} must be one of: checkout, pull.")
        move_nightly_files(wheel_site_dir)

    write_pth(venv)
    cast(logging.Logger, LOGGER).info(
        "-------\n"
        "PyTorch Development Environment set up!\n"
        "Please activate to enable this environment:\n\n"
        "  $ %s\n",
        venv.activate_command,
    )


def make_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] Builds the argument parser with checkout and pull subcommands; each
    # accepts python interpreter, prefix, fresh, yes, verbose and optional cuda/rocm
    # version options. Returns the parser.
    def find_executable(name: str) -> Path:
        # [FUNCTION DOC] HELPER: Argument type converter that resolves an executable name
        # on PATH to an absolute path or raises an argument type error if not found.
        executable = shutil.which(name)
        if executable is None:
            raise argparse.ArgumentTypeError(
                f"Could not find executable {name} in PATH."
            )
        return Path(executable).absolute()

    parser = argparse.ArgumentParser()
    # subcommands
    subcmd = parser.add_subparsers(dest="subcmd", help="subcommand to execute")
    checkout = subcmd.add_parser("checkout", help="checkout a new branch")
    checkout.add_argument(
        "-b",
        "--branch",
        help="Branch name to checkout (if omitted, checks out in detached HEAD mode)",
        dest="branch",
        default=None,
        metavar="NAME",
    )
    pull = subcmd.add_parser(
        "pull", help="pulls the nightly commits into the current branch"
    )
    # general arguments
    subparsers = [checkout, pull]
    for subparser in subparsers:
        subparser.add_argument(
            "--python",
            "--base-executable",
            type=find_executable,
            help=(
                "Path to Python interpreter to use for creating the virtual environment. "
                "Defaults to the interpreter running this script."
            ),
            dest="base_executable",
            default=None,
            metavar="PYTHON",
        )
        subparser.add_argument(
            "--prefix",
            "-p",
            type=lambda p: Path(p).absolute(),
            help='Path to virtual environment directory (e.g. "./venv")',
            dest="prefix",
            default=str(DEFAULT_VENV_DIR),
            metavar="PATH",
        )
        subparser.add_argument(
            "--fresh",
            help="Remove existing virtual environment if it exists",
            dest="fresh",
            default=False,
            action="store_true",
        )
        subparser.add_argument(
            "--yes",
            "-y",
            help=(
                "Automatic yes to prompts; assume 'yes' as answer to all prompts "
                "(e.g., removing existing venv)"
            ),
            dest="yes",
            default=False,
            action="store_true",
        )
        subparser.add_argument(
            "--verbose",
            "-v",
            help="Provide debugging info",
            dest="verbose",
            default=False,
            action="store_true",
        )
        subparser.add_argument(
            "--cuda",
            help=(
                "CUDA version to install "
                "(defaults to the latest version available on the platform)"
            ),
            dest="cuda",
            nargs="?",
            default=argparse.SUPPRESS,
            metavar="VERSION",
        )
        subparser.add_argument(
            "--rocm",
            help=(
                "ROCm version to install "
                "(defaults to the latest version available on the platform)"
            ),
            dest="rocm",
            nargs="?",
            default=argparse.SUPPRESS,
            metavar="VERSION",
        )
    return parser


def parse_arguments() -> argparse.Namespace:
    # [FUNCTION DOC] Parses the command line, defaulting branch to none, and reports a
    # parser error if both cuda and rocm are requested. Returns the namespace.
    parser = make_parser()
    args = parser.parse_args()
    args.branch = getattr(args, "branch", None)
    if hasattr(args, "cuda") and hasattr(args, "rocm"):
        parser.error("Cannot specify both CUDA and ROCm versions.")
    return args


def main() -> None:
    # [FUNCTION DOC] Entry point: parses arguments, sets global verbosity, validates the
    # branch (exits with the message on problems), selects the pip source (cpu by default,
    # or the requested/latest CUDA or ROCm one for this platform, exiting if unavailable),
    # then runs the install workflow under the logging manager. Sets global LOGGER.
    # KNOWN ISSUE: sorting versions uses the packaging Version class, which is None if the
    # optional import failed, so the cuda/rocm paths would then fail.
    """Main entry point"""
    global LOGGER, VERBOSE
    args = parse_arguments()
    VERBOSE = args.verbose

    status = check_branch(args.subcmd, args.branch)
    if status:
        sys.exit(status)

    pip_source = None
    for toolkit in ("CUDA", "ROCm"):
        accel = toolkit.lower()
        if hasattr(args, accel):
            requested = getattr(args, accel)
            available_sources = {
                src.name[len(f"{accel}-") :]: src
                for src in PIP_SOURCES.values()
                if src.name.startswith(f"{accel}-")
                and PLATFORM in src.supported_platforms
            }
            if not available_sources:
                print(f"No {toolkit} versions available on platform {PLATFORM}.")
                sys.exit(1)
            if requested is not None:
                pip_source = available_sources.get(requested)
                if pip_source is None:
                    print(
                        f"{toolkit} {requested} is not available on platform {PLATFORM}. "
                        f"Available version(s): {', '.join(sorted(available_sources, key=Version))}"
                    )
                    sys.exit(1)
            else:
                pip_source = available_sources[max(available_sources, key=Version)]

    if pip_source is None:
        pip_source = PIP_SOURCES["cpu"]  # always available

    with logging_manager(debug=args.verbose) as logger:
        LOGGER = logger
        venv = Venv(
            prefix=args.prefix,
            pip_source=pip_source,
            base_executable=args.base_executable,
        )
        install(
            venv=venv,
            packages=PACKAGES_TO_INSTALL,
            subcommand=args.subcmd,
            branch=args.branch,
            fresh_venv=args.fresh,
            assume_yes=args.yes,
        )


if __name__ == "__main__":
    main()
