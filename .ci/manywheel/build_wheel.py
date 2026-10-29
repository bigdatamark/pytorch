#!/usr/bin/env python3
# [MODULE DOC] Command-line script (argument: output directory) that builds a PyTorch wheel
# inside a manylinux container. Build flags are expected from the caller's environment; this
# script adds only architecture-dependent BLAS settings (MKL paths on x86_64, ARM Compute
# Library plus OpenBLAS or NVPL on aarch64) and then runs the build module to produce the
# wheel. Main entries: configure_blas_env, main. No import-time side effects.
"""Build a PyTorch wheel inside a manylinux container.

Usage: build_wheel.py <output_dir>

Expects all build env vars (USE_CUDA, TORCH_CUDA_ARCH_LIST, etc.) to be set
by the caller (GitHub Actions workflow env). This script only adds the BLAS
plumbing that depends on the host architecture.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def configure_blas_env() -> None:
    # [FUNCTION DOC] Print the detected architecture and settings, then set process
    # environment variables: on x86_64 point CMake at MKL under /opt/intel if present; on
    # aarch64 require /acl, enable MKLDNN with ACL, and choose NVPL (cuda-aarch64) or
    # OpenBLAS (cpu-aarch64 or cpu) after checking the library exists. Exits the process with
    # an error message if a required library is missing. Other architectures are untouched.
    """Tell CMake which BLAS to use, based on architecture and GPU type.

    On x86, MKL from /opt/intel is wired in via CMAKE_{INCLUDE,LIBRARY}_PATH.
    On aarch64, CMake otherwise hunts for MKL (which doesn't exist there);
    we explicitly pick OpenBLAS or NVPL and enable ACL for oneDNN.
    """
    arch = platform.machine()
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "")
    print(
        f"build_wheel.py: ARCH={arch} GPU_ARCH_TYPE={gpu_arch_type or 'unset'} "
        f"DESIRED_CUDA={os.environ.get('DESIRED_CUDA', 'unset')}"
    )

    if arch == "x86_64":
        if Path("/opt/intel/include").is_dir():
            os.environ["CMAKE_INCLUDE_PATH"] = "/opt/intel/include"
            os.environ["CMAKE_LIBRARY_PATH"] = "/opt/intel/lib:/lib"
        return

    if arch != "aarch64":
        return

    if not Path("/acl").is_dir():
        sys.exit("ERROR: ARM Compute Library not found at /acl")
    os.environ["USE_MKLDNN"] = "1"
    os.environ["USE_MKLDNN_ACL"] = "1"
    os.environ["ACL_ROOT_DIR"] = "/acl"

    if gpu_arch_type == "cuda-aarch64":
        nvpl = Path("/usr/local/lib/libnvpl_blas_lp64_gomp.so.0")
        if not nvpl.is_file():
            sys.exit(f"ERROR: NVPL BLAS not found at {nvpl}")
        print("Using NVPL BLAS/LAPACK and ACL for MKLDNN on CUDA aarch64")
        os.environ["BLAS"] = "NVPL"
    elif gpu_arch_type in ("cpu-aarch64", "cpu"):
        openblas = Path("/opt/OpenBLAS/lib/libopenblas.so.0")
        if not openblas.is_file():
            sys.exit(f"ERROR: OpenBLAS not found at {openblas}")
        print("Using OpenBLAS and ACL for MKLDNN on CPU aarch64")
        os.environ["BLAS"] = "OpenBLAS"
        os.environ["OpenBLAS_HOME"] = "/opt/OpenBLAS"


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses the output directory, configures BLAS env variables,
    # installs the build package with pip and builds the wheel with no build isolation into
    # the output directory. Raises a called-process error on failure.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    configure_blas_env()

    subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(args.output_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
