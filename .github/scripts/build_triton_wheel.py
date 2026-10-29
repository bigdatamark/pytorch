#!/usr/bin/env python3
# [MODULE DOC] Script that builds a Triton wheel for CI: clones the Triton (or Intel XPU
# backend) repository, checks out a release branch or pinned commit, rewrites the package
# version, builds a wheel with setup.py and copies it into the current directory. Reads pin
# and version files under .ci/docker. Import-time: only computes script and repo paths.

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory


SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent


def read_triton_pin(device: str = "cuda") -> str:
    # [FUNCTION DOC] Returns the stripped contents of the pinned Triton commit file in the
    # docker commit-pins directory: the XPU file for device xpu, otherwise the default one.
    # Raises OSError if the file is missing.
    triton_file = "triton.txt"
    if device == "xpu":
        triton_file = "triton-xpu.txt"
    with open(REPO_DIR / ".ci" / "docker" / "ci_commit_pins" / triton_file) as f:
        return f.read().strip()


def read_triton_version(device: str = "cuda") -> str:
    # [FUNCTION DOC] Returns the stripped Triton version string from the docker directory
    # (XPU-specific file for device xpu, otherwise the default one). Raises OSError if the
    # file is missing.
    triton_version_file = "triton_version.txt"
    if device == "xpu":
        triton_version_file = "triton_xpu_version.txt"
    with open(REPO_DIR / ".ci" / "docker" / triton_version_file) as f:
        return f.read().strip()


def check_and_replace(inp: str, src: str, dst: str) -> str:
    # [FUNCTION DOC] HELPER: Returns inp with every occurrence of src replaced by dst.
    # Raises RuntimeError if src does not occur in inp.
    # KNOWN ISSUE: the error message has a stray dollar sign before the searched text.
    """Checks that `src` can be found in `input` and replaces it with `dst`"""
    if src not in inp:
        raise RuntimeError(f"Can't find ${src} in the input")
    return inp.replace(src, dst)


def patch_init_py(
    path: Path, *, version: str, expected_version: str | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Rewrites the version assignment in the given Triton package
    # init file in place, replacing the expected old version with the new version. Raises
    # RuntimeError if the old assignment is not found.
    # KNOWN ISSUE: when expected_version is not given it is read for the default cuda
    # device, not for the device actually being built.
    if not expected_version:
        expected_version = read_triton_version()
    with open(path) as f:
        orig = f.read()
    # Replace version
    orig = check_and_replace(
        orig, f"__version__ = '{expected_version}'", f'__version__ = "{version}"'
    )
    with open(path, "w") as f:
        f.write(orig)


def build_triton(
    *,
    version: str,
    commit_hash: str,
    device: str = "cuda",
    py_version: str | None = None,
    release: bool = False,
    with_clang_ldd: bool = False,
) -> Path:
    # [FUNCTION DOC] Builds a Triton wheel and returns its path in the current directory.
    # Sets build job count from MAX_JOBS or the usable CPU count (and XPU link jobs), clones
    # the device-specific repo into a temporary directory, checks out the release branch
    # (release mode) or the given commit, sets wheel-name environment variables, patches the
    # version, runs ROCm packaging when device is rocm, builds with setup.py and copies the
    # wheel to the current directory (ROCm wheels are then patched). Subprocess failures raise
    # CalledProcessError. KNOWN ISSUE: the py_version argument is never used.
    env = os.environ.copy()
    # Default to the CPUs available to this process: os.sched_getaffinity respects
    # cgroup/cpuset limits (e.g. inside an OSDC container), unlike os.cpu_count()
    # which reports the whole host. A pre-set MAX_JOBS still wins.
    if hasattr(os, "sched_getaffinity"):
        default_jobs = len(os.sched_getaffinity(0))
    else:
        default_jobs = os.cpu_count() or 1
    max_jobs = int(env.get("MAX_JOBS", default_jobs))
    env["MAX_JOBS"] = str(max_jobs)

    if device == "xpu" and "TRITON_PARALLEL_LINK_JOBS" not in env:
        env["TRITON_PARALLEL_LINK_JOBS"] = str(max_jobs // 2 or 1)

    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"

        triton_repo = "https://github.com/openai/triton"
        if device == "rocm":
            triton_pkg_name = "triton-rocm"
        elif device == "xpu":
            triton_pkg_name = "triton-xpu"
            triton_repo = "https://github.com/intel/intel-xpu-backend-for-triton"
        else:
            triton_pkg_name = "triton"
        check_call(["git", "clone", triton_repo, "triton"], cwd=tmpdir)
        if release:
            ver, rev, patch = version.split(".")
            if device == "xpu":
                # XPU uses the patch version in the release branch name
                check_call(
                    ["git", "checkout", f"release/{ver}.{rev}.{patch}"],
                    cwd=triton_basedir,
                )
            else:
                check_call(
                    ["git", "checkout", f"release/{ver}.{rev}.x"], cwd=triton_basedir
                )
        else:
            check_call(["git", "fetch", "origin", commit_hash], cwd=triton_basedir)
            check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

        # change built wheel name and version
        env["TRITON_WHEEL_NAME"] = triton_pkg_name
        if sys.platform != "win32":
            env["TRITON_EXT_ENABLED"] = "ON"
        if with_clang_ldd:
            env["TRITON_BUILD_WITH_CLANG_LLD"] = "1"

        patch_init_py(
            triton_pythondir / "triton" / "__init__.py",
            version=f"{version}",
            expected_version=read_triton_version(device),
        )

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/package_triton_wheel.sh"],
                cwd=triton_basedir,
                shell=True,
            )
            print("ROCm libraries setup for triton installation...")

        # old triton versions have setup.py in the python/ dir,
        # new versions have it in the root dir.
        triton_setupdir = (
            triton_basedir
            if (triton_basedir / "setup.py").exists()
            else triton_pythondir
        )

        check_call(
            [sys.executable, "setup.py", "bdist_wheel"], cwd=triton_setupdir, env=env
        )

        whl_path = next(iter((triton_setupdir / "dist").glob("*.whl")))
        shutil.copy(whl_path, Path.cwd())

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/patch_triton_wheel.sh", Path.cwd()],
                cwd=triton_basedir,
            )

        return Path.cwd() / whl_path.name


def main() -> None:
    # [FUNCTION DOC] Command-line entry point. Parses release, device, commit hash, version
    # and clang-lld options, defaults the version and commit from the pin files, and calls
    # the wheel build. The wheel path result is discarded.
    # KNOWN ISSUE: the device choice aarch64 is accepted but read_triton_pin/version and the
    # build treat it like cuda; the clang flag name is spelled ldd rather than lld.
    from argparse import ArgumentParser

    parser = ArgumentParser("Build Triton binaries")
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "rocm", "xpu", "aarch64"]
    )
    parser.add_argument("--py-version", type=str)
    parser.add_argument("--commit-hash", type=str)
    parser.add_argument("--with-clang-ldd", action="store_true")
    parser.add_argument("--triton-version", type=str, default=None)
    args = parser.parse_args()

    triton_version = read_triton_version(args.device)
    if args.triton_version:
        triton_version = args.triton_version

    build_triton(
        device=args.device,
        commit_hash=(
            args.commit_hash if args.commit_hash else read_triton_pin(args.device)
        ),
        version=triton_version,
        py_version=args.py_version,
        release=args.release,
        with_clang_ldd=args.with_clang_ldd,
    )


if __name__ == "__main__":
    main()
