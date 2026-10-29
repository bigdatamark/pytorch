# [MODULE DOC] Script that fetches optional third-party dependencies (NCCL and Eigen) by
# shallow-cloning pinned release tags into the third_party directory. Pins come from
# files in the CI docker pins directory and the third_party directory. Run with no
# argument to fetch all, or with a function name to call just that one. Import-time
# side effects: none beyond computing paths.
import os
from pathlib import Path
from subprocess import check_call


repo_root = Path(__file__).absolute().parent.parent
third_party_path = repo_root / "third_party"


def _read_file(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Returns the whole text of the file with surrounding
    # whitespace stripped.
    with path.open(encoding="utf-8") as f:
        return f.read().strip()


def _checkout_by_tag(repo: str, tag: str) -> None:
    # [FUNCTION DOC] HELPER: Shallow-clones (depth one) the given tag of the repository
    # URL inside the third_party directory. Raises CalledProcessError if git fails.
    check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            repo,
        ],
        cwd=third_party_path,
    )


def read_nccl_pin(cuda_version: str = "") -> str:
    # [FUNCTION DOC] Returns the pinned NCCL release tag. The CUDA version defaults to
    # the desired-CUDA or CUDA-version environment variable; for 12.6 (or the cu126
    # label) a separate pin file is used, otherwise the default pin file. Reads the pin
    # from the CI docker commit pins directory; raises an OS error if the file is missing.
    # Default NCCL version
    nccl_file = "nccl.txt"

    # If NCCL version diverges for different CUDA versions, uncomment the
    # following block and add the appropriate file (using CUDA 11 as an example)

    # 12.6 builds for sm50, needs a lower version
    if not cuda_version:
        cuda_version = os.getenv("DESIRED_CUDA", os.getenv("CUDA_VERSION", ""))
    if cuda_version.startswith("12.6") or cuda_version == "cu126":
        nccl_file = "nccl-cu126.txt"

    nccl_pin_path = repo_root / ".ci" / "docker" / "ci_commit_pins" / nccl_file
    return _read_file(nccl_pin_path)


def checkout_nccl() -> None:
    # [FUNCTION DOC] Prints and clones the pinned NCCL tag from NVIDIA's GitHub into
    # third_party unless an nccl directory already exists there.
    release_tag = read_nccl_pin()
    print(f"-- Checkout nccl release tag: {release_tag}")
    nccl_basedir = third_party_path / "nccl"
    if not nccl_basedir.exists():
        _checkout_by_tag("https://github.com/NVIDIA/nccl", release_tag)


def checkout_eigen() -> None:
    # [FUNCTION DOC] Prints and clones the Eigen tag named in the third_party pin file
    # from GitLab into third_party unless an eigen directory already exists there.
    eigen_tag = _read_file(third_party_path / "eigen_pin.txt")
    print(f"-- Checkout Eigen release tag: {eigen_tag}")
    eigen_basedir = third_party_path / "eigen"
    if not eigen_basedir.exists():
        _checkout_by_tag("https://gitlab.com/libeigen/eigen", eigen_tag)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # If no arguments are given checkout all optional dependency
        checkout_nccl()
        checkout_eigen()
    else:
        # Otherwise just call top-level function of choice
        globals()[sys.argv[1]]()
