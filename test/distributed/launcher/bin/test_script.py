#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Worker script launched by the elastic launcher tests (test_run.py, launch_test.py).
# Prints the torchelastic environment variables set by the agent, then either raises a
# RuntimeError (when --fail is passed) or touches a file named after the global RANK inside
# the directory given by --touch-file-dir so the test can verify which workers ran.
# No import-time side effects; main() only runs when executed as a script.

import argparse
import os
from pathlib import Path


def parse_args():
    # [FUNCTION DOC] HELPER: builds the command line parser and returns the parsed namespace.
    # Options: --fail (boolean flag, default off) and --touch-file-dir / --touch_file_dir (string
    # directory path). Reads sys.argv; argparse exits the process on invalid arguments.
    parser = argparse.ArgumentParser(description="test script")

    parser.add_argument(
        "--fail",
        default=False,
        action="store_true",
        help="forces the script to throw a RuntimeError",
    )

    # file is used for assertions
    parser.add_argument(
        "--touch-file-dir",
        "--touch_file_dir",
        type=str,
        help="dir to touch a file with global rank as the filename",
    )
    return parser.parse_args()


def main():
    # [FUNCTION DOC] Entry point of the worker. Parses arguments, then prints every variable in a
    # fixed list of torchelastic variables (LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, ...,
    # TEST_SENTINEL_PARENT, TORCHELASTIC_ERROR_FILE). Raises KeyError if any is not set in the
    # environment. With --fail it then raises RuntimeError; otherwise it creates an empty file
    # named by the RANK variable in the touch-file directory (side effect on disk) and prints it.
    args = parse_args()
    env_vars = [
        "LOCAL_RANK",
        "RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "LOCAL_WORLD_SIZE",
        "WORLD_SIZE",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "OMP_NUM_THREADS",
        "TEST_SENTINEL_PARENT",
        "TORCHELASTIC_ERROR_FILE",
    ]

    print("Distributed env vars set by agent:")
    for env_var in env_vars:
        value = os.environ[env_var]
        print(f"{env_var} = {value}")

    if args.fail:
        raise RuntimeError("raising exception since --fail flag was set")
    else:
        file = os.path.join(args.touch_file_dir, os.environ["RANK"])
        Path(file).touch()
        print(f"Success, created {file}")


if __name__ == "__main__":
    main()
