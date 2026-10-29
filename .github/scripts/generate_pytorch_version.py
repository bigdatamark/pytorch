#!/usr/bin/env python3

# [MODULE DOC] Script that prints the PyTorch version string for a binary build. If HEAD is
# exactly on a (non-ciflow) git tag it prints the tag without a leading v or trailing rc suffix;
# otherwise it prints a nightly version made of the base version from version.txt, a dev date
# stamp and an optional local build suffix (cu<version> or arch type plus version). Options can
# also come from the NO_BUILD_SUFFIX, GPU_ARCH_TYPE and GPU_ARCH_VERSION environment variables.
# Import-time side effects: briefly modifies sys.path to import tools.strtobool.
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Because this file is run as a script, we need to add the repo root to the
# path to be able to import tools.strtobool.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.strtobool import strtobool


sys.path.pop(0)


LEADING_V_PATTERN = re.compile("^v")
TRAILING_RC_PATTERN = re.compile("-rc[0-9]*$")
LEGACY_BASE_VERSION_SUFFIX_PATTERN = re.compile("a0$")


class NoGitTagException(Exception):
    # [CLASS DOC] Error raised when a release version is requested but HEAD is not on a usable
    # git tag. Carries no extra state.
    pass


def get_pytorch_root() -> Path:
    # [FUNCTION DOC] Returns the git top-level directory of the current working directory by
    # running git; raises if git fails or the directory is not in a repository.
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )


def get_tag() -> str:
    # [FUNCTION DOC] Returns the git tag that exactly matches HEAD with any leading v and
    # trailing rc suffix removed, or an empty string when HEAD has no exact tag or the tag is a
    # ciflow tag.
    root = get_pytorch_root()
    try:
        dirty_tag = (
            subprocess.check_output(["git", "describe", "--tags", "--exact"], cwd=root)
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return ""
    # Strip leading v that we typically do when we tag branches
    # ie: v1.7.1 -> 1.7.1
    tag = re.sub(LEADING_V_PATTERN, "", dirty_tag)
    # Strip trailing rc pattern
    # ie: 1.7.1-rc1 -> 1.7.1
    tag = re.sub(TRAILING_RC_PATTERN, "", tag)
    # Ignore ciflow tags
    if tag.startswith("ciflow/"):
        return ""
    return tag


def get_base_version() -> str:
    # [FUNCTION DOC] Reads version.txt at the repo root and returns its stripped text with a
    # trailing legacy marker removed. Raises if the file is missing.
    root = get_pytorch_root()
    dirty_version = Path(root / "version.txt").read_text().strip()
    # Strips trailing a0 from version.txt, not too sure why it's there in the
    # first place
    return re.sub(LEGACY_BASE_VERSION_SUFFIX_PATTERN, "", dirty_version)


class PytorchVersion:
    # [CLASS DOC] Computes release and nightly version strings for a build. State: GPU arch type,
    # GPU arch version and a flag disabling the local build suffix. Methods return the suffix,
    # the release version (needs a git tag) and the nightly version (dev date plus suffix).
    def __init__(
        self,
        gpu_arch_type: str,
        gpu_arch_version: str,
        no_build_suffix: bool,
    ) -> None:
        # [FUNCTION DOC] Stores the GPU arch type, GPU arch version and the no-build-suffix
        # flag; no validation or side effects.
        self.gpu_arch_type = gpu_arch_type
        self.gpu_arch_version = gpu_arch_version
        self.no_build_suffix = no_build_suffix

    def get_post_build_suffix(self) -> str:
        # [FUNCTION DOC] Returns the local version suffix: empty if suffixes are disabled; a plus
        # sign and cu plus the version without dots for cuda; otherwise a plus sign followed by
        # the arch type and version (for example cpu with an empty version gives plus cpu).
        if self.no_build_suffix:
            return ""
        if self.gpu_arch_type == "cuda":
            return f"+cu{self.gpu_arch_version.replace('.', '')}"
        return f"+{self.gpu_arch_type}{self.gpu_arch_version}"

    def get_release_version(self) -> str:
        # [FUNCTION DOC] Returns the cleaned git tag followed by the build suffix. Raises
        # NoGitTagException if there is no usable tag. Runs git (twice) as a side effect.
        if not get_tag():
            raise NoGitTagException(
                "Not on a git tag, are you sure you want a release version?"
            )
        return f"{get_tag()}{self.get_post_build_suffix()}"

    def get_nightly_version(self) -> str:
        # [FUNCTION DOC] Returns the base version, a dev marker with today's local date as eight
        # digits, and the build suffix.
        date_str = datetime.today().strftime("%Y%m%d")
        build_suffix = self.get_post_build_suffix()
        return f"{get_base_version()}.dev{date_str}{build_suffix}"


def main() -> None:
    # [FUNCTION DOC] Parses options (no-build-suffix flag, GPU arch type, GPU arch version, each
    # defaulting from an environment variable), then prints the release version, falling back to
    # the nightly version when there is no git tag.
    parser = argparse.ArgumentParser(
        description="Generate pytorch version for binary builds"
    )
    parser.add_argument(
        "--no-build-suffix",
        action="store_true",
        help="Whether or not to add a build suffix typically (+cpu)",
        default=strtobool(os.environ.get("NO_BUILD_SUFFIX", "False")),
    )
    parser.add_argument(
        "--gpu-arch-type",
        type=str,
        help="GPU arch you are building for, typically (cpu, cuda, rocm)",
        default=os.environ.get("GPU_ARCH_TYPE", "cpu"),
    )
    parser.add_argument(
        "--gpu-arch-version",
        type=str,
        help="GPU arch version, typically (10.2, 4.0), leave blank for CPU",
        default=os.environ.get("GPU_ARCH_VERSION", ""),
    )
    args = parser.parse_args()
    version_obj = PytorchVersion(
        args.gpu_arch_type, args.gpu_arch_version, args.no_build_suffix
    )
    try:
        print(version_obj.get_release_version())
    except NoGitTagException:
        print(version_obj.get_nightly_version())


if __name__ == "__main__":
    main()
