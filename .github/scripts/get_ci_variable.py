#!/usr/bin/env python3
# [MODULE DOC] Command line helper that prints one CI constant from generate_binary_build_matrix:
# the stable CUDA version or the minimum supported python version. No import-time side effects.
"""Helper script - Return CI variables such as stable cuda, min python version, etc."""

import argparse
import sys


def main(args: list[str]) -> None:
    # [FUNCTION DOC] Parses the argument list for the stable-CUDA or min-python flag and prints
    # the stable CUDA version or the first (oldest) python version of the full list. If both
    # flags are given only the CUDA one is printed; with neither, prints nothing. Imports the
    # matrix module lazily.
    import generate_binary_build_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-stable-version",
        action="store_true",
        help="get cuda stable version",
    )
    parser.add_argument(
        "--min-python-version",
        action="store_true",
        help="get min supported python version",
    )
    options = parser.parse_args(args)
    if options.cuda_stable_version:
        return print(generate_binary_build_matrix.CUDA_STABLE)
    if options.min_python_version:
        return print(generate_binary_build_matrix.FULL_PYTHON_VERSIONS[0])


if __name__ == "__main__":
    main(sys.argv[1:])
