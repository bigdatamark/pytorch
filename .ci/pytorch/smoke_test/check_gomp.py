# [MODULE DOC] Smoke-test script verifying that OpenMP is not loaded twice. It loads the
# system libgomp and the installed libtorch_cpu shared library with CPU affinity limited to
# four cores, asks for the maximum OpenMP thread count, and fails if that is one (the sign of
# a duplicated libgomp). Main entries: get_gomp_thread, main. No import-time side effects.
import ctypes
import os
import sys
from pathlib import Path


def get_gomp_thread():
    # [FUNCTION DOC] Compute the path of libtorch_cpu under the interpreter's installation
    # prefix and Python-version site-packages directory, pick the AlmaLinux libgomp path (or
    # the Ubuntu multiarch path if absent), set GOMP_CPU_AFFINITY to 0-3 in this process, load
    # both libraries and return the OpenMP maximum thread count as an integer. Raises an OS
    # error if a library cannot be loaded.
    # KNOWN ISSUE: assumes the standard lib/pythonX.Y/site-packages layout and ignores the
    # real purelib location.
    """
    Retrieves the maximum number of OpenMP threads after loading the `libgomp.so.1` library
    and the `libtorch_cpu.so` library. It then queries the
    maximum number of threads available for OpenMP parallel regions using the
    `omp_get_max_threads` function.

    Returns:
        int: The maximum number of OpenMP threads available.

    Notes:
        - The function assumes the default path for `libgomp.so.1` on AlmaLinux OS.
        - The path to `libtorch_cpu.so` is constructed based on the Python executable's
          installation directory.
        - This function is specific to environments where PyTorch and OpenMP are used
          together and may require adjustments for other setups.
    """
    python_path = Path(sys.executable).resolve()
    python_prefix = (
        python_path.parent.parent
    )  # Typically goes to the Python installation root

    # Get the additional ABI flags (if any); it may be an empty string.
    abiflags = getattr(sys, "abiflags", "")

    # Construct the Python directory name correctly (e.g., "python3.14t").
    python_version = (
        f"python{sys.version_info.major}.{sys.version_info.minor}{abiflags}"
    )

    libtorch_cpu_path = (
        python_prefix
        / "lib"
        / python_version
        / "site-packages"
        / "torch"
        / "lib"
        / "libtorch_cpu.so"
    )

    # use the default gomp path of AlmaLinux OS
    libgomp_path = "/usr/lib64/libgomp.so.1"
    # if it does not exist, try Ubuntu path
    if not os.path.exists(libgomp_path):
        libgomp_path = f"/usr/lib/{os.uname().machine}-linux-gnu/libgomp.so.1"

    os.environ["GOMP_CPU_AFFINITY"] = "0-3"

    libgomp = ctypes.CDLL(libgomp_path)
    libgomp = ctypes.CDLL(libtorch_cpu_path)

    libgomp.omp_get_max_threads.restype = ctypes.c_int
    libgomp.omp_get_max_threads.argtypes = []

    omp_max_threads = libgomp.omp_get_max_threads()
    return omp_max_threads


def main():
    # [FUNCTION DOC] Entry point. Gets the OpenMP thread count, prints it, and raises a
    # runtime error if it equals one.
    omp_max_threads = get_gomp_thread()
    print(
        f"omp_max_threads after loading libgomp.so and libtorch_cpu.so: {omp_max_threads}"
    )
    if omp_max_threads == 1:
        raise RuntimeError(
            "omp_max_threads is 1. Check whether libgomp.so is loaded twice."
        )


if __name__ == "__main__":
    main()
