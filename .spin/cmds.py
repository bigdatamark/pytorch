# [MODULE DOC] Custom spin commands for the PyTorch repo (loaded by the spin CLI).
# Defines click commands for regenerating version.py, type stubs, clang-tidy files and
# GitHub workflows, setting up and running lintrunner (lint, fixlint, quicklint, quickfix),
# cleaning, building docs, editable and regular installs, and a pyrefly command group.
# Import-time side effects: adds the repo root to the front of the module search path
# and imports the repo clean tool from it. Also defines the linter category sets.
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mktemp

import click
import spin


# tomllib is built in on Python 3.11+, and spin depends on tomli for older versions.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


CWD = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(CWD))  # this only affects the current process
from tools.clean import clean as _clean


def file_digest(file, algorithm: str):
    # [FUNCTION DOC] HELPER: Return a hash object for the open binary file using the named
    # algorithm. Uses the standard library file digest helper when available, otherwise
    # reads the file in 8192-byte chunks into a new hash object. Consumes the file position.
    try:
        return hashlib.file_digest(file, algorithm)
    except AttributeError:
        pass  # Fallback to manual implementation below
    hash = hashlib.new(algorithm)
    while chunk := file.read(8192):
        hash.update(chunk)
    return hash


def _hash_file(file):
    # [FUNCTION DOC] HELPER: Return the hex SHA-256 digest of the file at the given path.
    with open(file, "rb") as f:
        hash = file_digest(f, "sha256")
    return hash.hexdigest()


def _hash_files(files):
    # [FUNCTION DOC] HELPER: Return a mapping from each given file path to its SHA-256 hex digest.
    hashes = {file: _hash_file(file) for file in files}
    return hashes


def _read_hashes(hash_file: Path):
    # [FUNCTION DOC] HELPER: Parse a stored hash file into a mapping of file name to hash.
    # Each line holds a 64-char hash, two separator chars, then the file name (stripped).
    # Returns an empty mapping when the hash file does not exist.
    if not hash_file.exists():
        return {}
    with hash_file.open("r") as f:
        lines = f.readlines()
    hashes = {}
    for line in lines:
        hash = line[:64]
        file = line[66:].strip()
        hashes[file] = hash
    return hashes


def _updated_hashes(hash_file, files_to_hash):
    # [FUNCTION DOC] HELPER: Hash the given files and compare with the hashes stored in the
    # hash file. Returns the new mapping of file to hash when it differs from the stored one
    # (including when nothing is stored), otherwise None. Does not write anything.
    old_hashes = _read_hashes(hash_file)
    new_hashes = _hash_files(files_to_hash)
    if new_hashes != old_hashes:
        return new_hashes
    return None


@click.command()
def regenerate_version():
    # [FUNCTION DOC] Spin command: run the tools.generate_torch_version module in a
    # subprocess with debug mode off, which regenerates version.py.
    """Regenerate version.py."""
    cmd = [
        sys.executable,
        "-m",
        "tools.generate_torch_version",
        "--is-debug=false",
    ]
    spin.util.run(cmd)


TYPE_STUBS = [
    (
        "Pytorch type stubs",
        Path(".lintbin/.pytorch-type-stubs.sha256"),
        [
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/tags.yaml",
            "tools/autograd/deprecated.yaml",
        ],
        [
            sys.executable,
            "-m",
            "tools.pyi.gen_pyi",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--deprecated-functions-path",
            "tools/autograd/deprecated.yaml",
        ],
    ),
    (
        "Datapipes type stubs",
        None,
        [],
        [
            sys.executable,
            "torch/utils/data/datapipes/gen_pyi.py",
        ],
    ),
]


@click.command()
def regenerate_type_stubs():
    # [FUNCTION DOC] Spin command: regenerate type stubs for each entry of the stub table.
    # Entries with a hash file are regenerated only when the hashed input files changed,
    # and the new hashes are then written to that hash file (creating its directory).
    # Entries without a hash file are always regenerated. Prints progress messages.
    """Regenerate type stubs."""
    for name, hash_file, files_to_hash, cmd in TYPE_STUBS:
        if hash_file:
            if hashes := _updated_hashes(hash_file, files_to_hash):
                click.echo(
                    f"Changes detected in type stub files for {name}. Regenerating..."
                )
                spin.util.run(cmd)
                hash_file.parent.mkdir(parents=True, exist_ok=True)
                with hash_file.open("w") as f:
                    for file, hash in hashes.items():
                        f.write(f"{hash}  {file}\n")
                click.echo("Type stubs and hashes updated.")
            else:
                click.echo(f"No changes detected in type stub files for {name}.")
        else:
            click.echo(f"No hash file for {name}. Regenerating...")
            spin.util.run(cmd)
            click.echo("Type stubs regenerated.")


@click.command()
def regenerate_clangtidy_files():
    # [FUNCTION DOC] Spin command: run the clang-tidy build-file generator module in a
    # subprocess to regenerate the clang-tidy files.
    """Regenerate clang-tidy files."""
    cmd = [
        sys.executable,
        "-m",
        "tools.linter.clang_tidy.generate_build_files",
    ]
    spin.util.run(cmd)


#: These linters are expected to need less than 3s cpu time total
VERY_FAST_LINTERS = {
    "ATEN_CPU_GPU_AGNOSTIC",
    "C10_NODISCARD",
    "C10_UNUSED",
    "CALL_ONCE",
    "CMAKE_MINIMUM_REQUIRED",
    "CODEOWNERS_TAXONOMY",
    "CONTEXT_DECORATOR",
    "COPYRIGHT",
    "CUBINCLUDE",
    "DEPLOY_DETECTION",
    "ERROR_PRONE_ISINSTANCE",
    "EXEC",
    "HEADER_ONLY_LINTER",
    "IMPORT_LINTER",
    "INCLUDE",
    "ISINSTANCE_FAKE_TENSOR",
    "LINTRUNNER_VERSION",
    "MERGE_CONFLICTLESS_CSV",
    "META_NO_CREATE_UNBACKED",
    "NEWLINE",
    "NOQA",
    "NO_WORKFLOWS_ON_FORK",
    "ONCE_FLAG",
    "PYBIND11_INCLUDE",
    "PYBIND11_SPECIALIZATION",
    "PYPIDEP",
    "PYPROJECT",
    "RAWCUDA",
    "RAWCUDADEVICE",
    "RAWTHROW",
    "ROOT_LOGGING",
    "SYMPY_MINMAX",
    "TABS",
    "TESTOWNERS",
    "TYPEIGNORE",
    "TYPENOSKIP",
    "UNSPECIFIED_BACKEND",
    "WORKFLOWSYNC",
}


#: These linters are expected to take a few seconds, but less than 10s cpu time total
FAST_LINTERS = {
    "CLANGTIDY_EXECUTORCH_COMPATIBILITY",
    "CMAKE",
    "DOCSTRING_LINTER",
    "GHA",
    "NATIVEFUNCTIONS",
    "PYREFLY",
    "RUFF",
    "SET_LINTER",
    "SHELLCHECK",
    "SPACES",
}


#: These linters are expected to take more than 10s cpu time total;
#: some need more than 1 hour.
SLOW_LINTERS = {
    "ACTIONLINT",
    "CLANGFORMAT",
    "CLANGTIDY",
    "CODESPELL",
    "FLAKE8",
    "GB_REGISTRY",
    "GENERATED_SHIMS_VERSION",
    "PYFMT",
    "STABLE_SHIM_USAGE",
    "STABLE_SHIM_VERSION",
    "TEST_DEVICE_BIAS",
    "TEST_HAS_MAIN",
    "SCOPED_LIBRARY",
}


ALL_LINTERS = VERY_FAST_LINTERS | FAST_LINTERS | SLOW_LINTERS


LINTRUNNER_CACHE_INFO = (
    Path(".lintbin/.lintrunner.sha256"),
    [
        "requirements.txt",
        "pyproject.toml",
        ".lintrunner.toml",
    ],
)


LINTRUNNER_BASE_CMD = [
    "uvx",
    "--python",
    "3.10",
    "lintrunner@0.12.7",
]


def _check_linter_python_versions():
    # [FUNCTION DOC] HELPER: Validate the lintrunner config file in the current directory.
    # Every linter whose command runs a uv script must pin Python 3.10 via the python flag.
    # Raises a click exception listing the codes of all linters that do not.
    invalid_linters = []

    with Path(".lintrunner.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
    for linter in config["linter"]:
        command = linter.get("command", [])
        if command[:2] != ["uv", "run"] or "--script" not in command:
            continue
        try:
            python_index = command.index("--python")
        except ValueError:
            python_index = -1
        if python_index == -1 or command[python_index + 1 : python_index + 2] != [
            "3.10"
        ]:
            invalid_linters.append(linter["code"])

    if invalid_linters:
        raise click.ClickException(
            "Linters using `uv run --script` must specify `--python 3.10`: "
            + ", ".join(invalid_linters)
        )


@click.command()
def setup_lint():
    # [FUNCTION DOC] Spin command: initialize lintrunner (pinned version, run through uvx).
    # Raises a called-process error on failure; output is captured, not shown.
    """Set up lintrunner with current CI version."""
    cmd = LINTRUNNER_BASE_CMD + ["init"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _check_linters():
    # [FUNCTION DOC] HELPER: Check that the python-version rule holds and that the set of
    # linters reported by lintrunner matches the union of the three category sets.
    # Raises a click exception naming unknown or missing linters. Returns the two (empty)
    # sets of unknown and missing linters when everything matches.
    _check_linter_python_versions()
    cmd = LINTRUNNER_BASE_CMD + ["list"]
    ret = spin.util.run(cmd, output=False, stderr=subprocess.PIPE)
    linters = {l.strip() for l in ret.stdout.decode().strip().split("\n")[1:]}
    unknown_linters = linters - ALL_LINTERS
    missing_linters = ALL_LINTERS - linters
    if unknown_linters:
        raise click.ClickException(
            f"Unknown linters found; please add them to the correct category "
            f"in .spin/cmds.py: {', '.join(sorted(unknown_linters))}"
        )
    if missing_linters:
        raise click.ClickException(
            f"Missing linters found; please update the corresponding category "
            f"in .spin/cmds.py: {', '.join(sorted(missing_linters))}"
        )
    return unknown_linters, missing_linters


@spin.util.extend_command(
    setup_lint,
    doc=f"""
        If configuration has changed, update lintrunner.

        Compares the stored old hashes of configuration files with new ones and
        performs setup via setup-lint if the hashes have changed.
        Hashes are stored in {LINTRUNNER_CACHE_INFO[0]}; the following files are
        considered: {", ".join(LINTRUNNER_CACHE_INFO[1])}.
        """,
)
@click.pass_context
def lazy_setup_lint(ctx, parent_callback, **kwargs):
    # [FUNCTION DOC] Spin command extending setup-lint: runs the parent setup only when the
    # hashes of the lint configuration files changed, then stores the new hashes. Always
    # regenerates version.py and the type stubs afterwards and validates the linter
    # categories (which can raise a click exception). The parent callback is passed in.
    if hashes := _updated_hashes(*LINTRUNNER_CACHE_INFO):
        click.echo(
            "Changes detected in lint configuration files. Setting up linting tools..."
        )
        parent_callback(**kwargs)
        hash_file = LINTRUNNER_CACHE_INFO[0]
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with hash_file.open("w") as f:
            for file, hash in hashes.items():
                f.write(f"{hash}  {file}\n")
        click.echo("Linting tools set up and hashes updated.")
    else:
        click.echo("No changes detected in lint configuration files. Skipping setup.")
    click.echo("Regenerating version...")
    ctx.invoke(regenerate_version)
    click.echo("Regenerating type stubs...")
    ctx.invoke(regenerate_type_stubs)
    click.echo("Done.")
    _check_linters()


def _check_arg(check_arg, arg, args_iter):
    # [FUNCTION DOC] HELPER: Match one command-line argument against an option name.
    # Handles the joined form with an equals sign (returns the stripped value) and the
    # separate form (consumes and returns the next item of the shared iterator, raising
    # StopIteration if none is left). Returns None if the argument is not that option or
    # has the joined form with a longer name. Note an empty value is returned as an empty string.
    if arg.startswith(check_arg):
        found_arg, sep, value = arg.partition("=")
        if sep == "=":
            if found_arg == check_arg:
                return value.strip()
        else:
            if arg == check_arg:
                return next(args_iter).strip()
    return None


def _process_lintrunner_args(lintrunner_args):
    # [FUNCTION DOC] HELPER: Split lintrunner arguments into those spin handles and the rest.
    # Extracts the take and skip option values as sets of linter codes, the tee-json file
    # name, and the all-files flag. Records whether any non-option argument (a path) exists.
    # Returns the remaining arguments, take, skip, tee file, has-paths and has-all-files flags.
    take = None
    skip = None
    args_iter = iter(arg.strip() for arg in lintrunner_args)
    remaining_args = []
    tee_file = None
    has_paths = False
    has_all_files = False
    for arg in args_iter:
        if _take := _check_arg("--take", arg, args_iter):
            take = set(_take.split(","))
        elif _skip := _check_arg("--skip", arg, args_iter):
            skip = set(_skip.split(","))
        elif _tee_file := _check_arg("--tee-json", arg, args_iter):
            tee_file = _tee_file.strip()
        elif arg == "--all-files":
            has_all_files = True
        else:
            if not arg.startswith("-"):
                has_paths = True
            remaining_args.append(arg)
    return remaining_args, take, skip, tee_file, has_paths, has_all_files


def _run_lintrunner(
    default_linters,
    take,
    skip,
    apply_patches=False,
    all_files=False,
    lintrunner_args=None,
    return_json_output=False,
):
    # [FUNCTION DOC] HELPER: Run lintrunner for a set of linters and report the outcome.
    # The linters are the default set narrowed by the take set and reduced by the skip set;
    # if none remain it prints a message and skips the run. Otherwise it builds the command
    # with optional patch application, all-files and extra args, runs it without exiting,
    # and (when JSON output is requested) reads then deletes a temporary tee file.
    # Returns a pair: whether lint problems were found, and the JSON text (None if not requested).
    cmd = LINTRUNNER_BASE_CMD
    if return_json_output:
        tee_file = mktemp(prefix="spinlint_", suffix=".json")
        tee_cmd = ["--tee-json", tee_file]
    else:
        tee_file = None
        tee_cmd = []
    linters = default_linters.copy()
    if take is not None:
        linters &= take
    if skip is not None:
        linters -= skip
    if not linters:
        click.echo("No linters to run after applying --take/--skip filters.")
        click.echo("Skipping lintrunner execution.")
        lint_found = False
        if return_json_output:
            json_output = ""
        else:
            json_output = None
    else:
        full_cmd = (
            cmd
            + tee_cmd
            + [
                "--take",
                ",".join(linters),
            ]
            + (["--apply-patches"] if apply_patches else [])
            + (["--all-files"] if all_files else [])
            + (list(lintrunner_args) if lintrunner_args else [])
        )
        p = spin.util.run(full_cmd, sys_exit=False)
        lint_found = bool(p.returncode)
        if tee_file:
            tee_path = Path(tee_file)
            json_output = tee_path.read_text()
            tee_path.unlink()
        else:
            json_output = None
    return lint_found, json_output


@click.command()
@click.option("-a", "--apply-patches", is_flag=True)
@click.argument("lintrunner_args", metavar="", nargs=-1)
@click.pass_context
def lint(ctx, *, lintrunner_args, apply_patches, **kwargs):
    # [FUNCTION DOC] Spin command: lint the whole tree. Runs the lazy setup, then runs the
    # very fast and fast linters on all files unless explicit paths were given, and the slow
    # linters on changed files (all files only if all-files was requested). Optionally
    # concatenates the JSON outputs into the tee-json file. Prints a failure message and
    # exits with status 1 if any linter found problems.
    """Lint all files."""
    ctx.invoke(lazy_setup_lint)
    lintrunner_args, take, skip, tee_file, has_paths, has_all_files = (
        _process_lintrunner_args(lintrunner_args)
    )
    all_files = has_all_files or not has_paths
    all_files_linters = VERY_FAST_LINTERS | FAST_LINTERS
    changed_files_linters = SLOW_LINTERS
    write_json_output = bool(tee_file)
    lint_found_all, json_output_all = _run_lintrunner(
        all_files_linters,
        take=take,
        skip=skip,
        apply_patches=apply_patches,
        all_files=all_files,
        lintrunner_args=lintrunner_args,
        return_json_output=write_json_output,
    )
    # Slow linters default to changed files only so a bare `spin lint` stays
    # fast locally, but must honor an explicit --all-files (e.g. trunk CI) so
    # they continuously check the whole tree rather than just the merge-base diff.
    lint_found_changed, json_output_changed = _run_lintrunner(
        changed_files_linters,
        take=take,
        skip=skip,
        apply_patches=apply_patches,
        all_files=has_all_files,
        lintrunner_args=lintrunner_args,
        return_json_output=write_json_output,
    )
    lint_found = lint_found_all or lint_found_changed
    if write_json_output:
        Path(tee_file).write_text(json_output_all + json_output_changed)
    if lint_found:
        click.secho("Lint failed!", fg="red")
        raise SystemExit(1)


@click.command()
@click.argument("lintrunner_args", metavar="", nargs=-1)
@click.pass_context
def fixlint(ctx, *, lintrunner_args, **kwargs):
    # [FUNCTION DOC] Spin command: invoke the lint command with patch application enabled.
    """Autofix all files."""
    ctx.invoke(lint, lintrunner_args=lintrunner_args, apply_patches=True)


@click.command()
@click.option("-a", "--apply-patches", is_flag=True)
@click.argument("lintrunner_args", metavar="", nargs=-1)
@click.pass_context
def quicklint(ctx, *, lintrunner_args, apply_patches, **kwargs):
    # [FUNCTION DOC] Spin command: run the lazy setup, then run lintrunner directly on its
    # default (changed) files with the extra arguments, adding patch application on request.
    """Lint changed files."""
    ctx.invoke(lazy_setup_lint)
    cmd = LINTRUNNER_BASE_CMD + list(lintrunner_args)
    if apply_patches:
        cmd += ["--apply-patches"]
    spin.util.run(cmd)


@click.command()
@click.argument("lintrunner_args", metavar="", nargs=-1)
@click.pass_context
def quickfix(ctx, *, lintrunner_args, **kwargs):
    # [FUNCTION DOC] Spin command: invoke quicklint with patch application enabled.
    # KNOWN ISSUE: the extra lintrunner arguments are accepted but not forwarded.
    """Autofix changed files."""
    ctx.invoke(quicklint, apply_patches=True)


@click.command()
def clean():
    # [FUNCTION DOC] Spin command: remove git-ignored files using the repo clean tool.
    """Clean, that is remove all files in .gitignore except in the NOT-CLEAN-FILES section."""
    _clean()


@click.command()
def regenerate_github_workflows():
    # [FUNCTION DOC] Spin command: run the CI workflow generator script from the .github
    # directory to regenerate the GitHub workflows.
    """Regenerate GitHub workflows from templates."""
    cmd = [sys.executable, "scripts/generate_ci_workflows.py"]
    spin.util.run(cmd, cwd="./.github")


#: Canary imports for the docs build environment. If any of these are missing
#: in the active env, the `make` invocation in spin docs would fail deep inside
#: a Sphinx extension or a figure-generation script with an unhelpful traceback.
#: Listing them up-front lets us surface a single actionable hint instead.
_DOCS_DEP_CANARIES = ("sphinx", "matplotlib")


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("make_args", nargs=-1, type=click.UNPROCESSED)
def docs(make_args):
    # [FUNCTION DOC] Spin command: build the documentation by running make in the docs
    # directory, passing extra arguments through (default target is html). First checks
    # that the canary dependencies are importable, raising a click exception if not. Prints
    # a warning to standard error when node is missing and KaTeX prerender is not disabled.
    """Build documentation.

    Wraps `make` in `docs/`. With no arguments runs `make html`. Pass any
    Make target or argument through, e.g.

    \b
        spin docs                     # html (default)
        spin docs doctest             # run doctests
        spin docs coverage            # API coverage report
        spin docs html-stable         # release-style build
        spin docs serve               # local http server on build/html
        spin docs html O="-D katex_prerender=0"  # no Node.js needed

    Requires the docs build dependencies (`docs/requirements.txt`) to be
    installed in the active environment. The default build also prerenders
    math with KaTeX, which needs Node.js (`node`) on PATH; pass
    `O="-D katex_prerender=0"` to render math in the browser instead. (Use
    `O=` rather than `SPHINXOPTS=` so the Makefile's default options are kept.)
    """
    missing = [
        name for name in _DOCS_DEP_CANARIES if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise click.ClickException(
            f"Docs build dependencies missing: {', '.join(missing)}.\n"
            f"Install them with:\n\n"
            f"    pip install -r docs/requirements.txt"
        )

    # docs/source/conf.py sets katex_prerender=True, so sphinxcontrib-katex
    # spawns a Node.js server to prerender math. Missing node otherwise fails
    # deep in the Sphinx run with a bare "No such file or directory: 'node'".
    # Only a warning: node is unnecessary when the user disables prerender.
    prerender_disabled = any("katex_prerender=0" in arg for arg in make_args)
    if not prerender_disabled and shutil.which("node") is None:
        click.echo(
            "Warning: `node` not found on PATH. The docs build prerenders math "
            "with KaTeX (katex_prerender=True), which needs Node.js. Either "
            "install nodejs, or skip prerender with:\n\n"
            '    spin docs html O="-D katex_prerender=0"\n',
            err=True,
        )
    cmd = ["make", *(make_args or ("html",))]
    spin.util.run(cmd, cwd="docs")


def _pip_install_cmd(editable):
    # [FUNCTION DOC] HELPER: Build the install command list, using uv pip when uv is found
    # on the path and the current interpreter's pip otherwise. Adds the editable flag on
    # request and always installs the current directory verbosely without build isolation.
    """Build the pip install command, preferring uv when available."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install"]
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd += ["-e"]
    return cmd + [".", "-v", "--no-build-isolation"]


@click.command()
def develop():
    # [FUNCTION DOC] Spin command: run an editable install of the repo via the helper command.
    """Build PyTorch (editable install).

    Runs an editable pip install using uv when available, falling back to
    regular pip.  Build configuration comes from the environment, e.g.
    `BUILD_CONFIG spin develop`.
    """
    spin.util.run(_pip_install_cmd(editable=True))


# Alias so `spin editable` also works.
editable = click.command(name="editable")(develop.callback)
editable.help = develop.help


@click.command()
def install():
    # [FUNCTION DOC] Spin command: run a non-editable install of the repo via the helper command.
    """Install PyTorch (non-editable).

    Runs a regular pip install using uv when available, falling back to
    regular pip.  Build configuration comes from the environment, e.g.
    `BUILD_CONFIG spin install`.
    """
    spin.util.run(_pip_install_cmd(editable=False))


PYREFLY_LINTER_SCRIPT = CWD / "tools" / "linter" / "adapters" / "pyrefly_linter.py"
PYREFLY_CONFIG = CWD / "pyrefly.toml"


def _pyrefly_version() -> str:
    # [FUNCTION DOC] HELPER: Read the pyrefly linter adapter and extract the pinned pyrefly
    # version. Raises a runtime error if no pin is found.
    """Read the pyrefly version pinned in the linter adapter so spin stays in sync."""
    text = PYREFLY_LINTER_SCRIPT.read_text()
    match = re.search(r'"pyrefly==([^"]+)"', text)
    if not match:
        raise RuntimeError(
            f"Could not find pinned pyrefly version in {PYREFLY_LINTER_SCRIPT}"
        )
    return match.group(1)


def _pyrefly_base_cmd() -> list[str]:
    # [FUNCTION DOC] HELPER: Return the command list that runs the pinned pyrefly via uvx
    # with Python 3.12.
    return ["uvx", "--python", "3.12", f"pyrefly@{_pyrefly_version()}"]


def _pyrefly_init():
    # [FUNCTION DOC] HELPER: Run the pyrefly init subcommand through spin.
    cmd = _pyrefly_base_cmd() + ["init"]
    spin.util.run(cmd)


@click.group()
def pyrefly():
    # [FUNCTION DOC] Click group for pyrefly subcommands. Its body prints a startup message
    # whenever any subcommand is invoked.
    """Commands for managing PyRefly stubs and checks."""
    click.echo("Starting Pyrefly...")


@pyrefly.command()
@click.argument("files", metavar="", nargs=-1)
def infer(files):
    # [FUNCTION DOC] Spin subcommand: print a warning that infer is experimental, then run
    # pyrefly infer with the repo pyrefly config file on the given files.
    """Infer type annotations using `pyrefly infer`.

    Note: pyrefly's `infer` is still under development and has a known bug
    around imports. Review generated annotations before committing.
    """
    click.echo(
        "Warning: `pyrefly infer` is experimental and has a known bug with "
        "imports. Review the generated annotations carefully."
    )
    cmd = _pyrefly_base_cmd() + ["infer", "--config", str(PYREFLY_CONFIG)] + list(files)
    spin.util.run(cmd)
