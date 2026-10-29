#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Worker script for the launcher tests that checks the legacy --local-rank
# command line argument injected by the launcher matches the LOCAL_RANK environment variable.
# No import-time side effects; main() only runs when executed as a script.

import argparse
import os


def parse_args():
    # [FUNCTION DOC] HELPER: parses the required integer option --local-rank / --local_rank and
    # returns the namespace. Exits the process if it is missing or not an integer.
    parser = argparse.ArgumentParser(description="test script")

    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        required=True,
        help="The rank of the node for multi-node distributed training",
    )

    return parser.parse_args()


def main():
    # [FUNCTION DOC] Prints start and end markers, reads the expected rank from the LOCAL_RANK
    # environment variable (KeyError if unset) and compares it with the --local-rank argument.
    # Raises RuntimeError describing both values when they differ.
    print("Start execution")
    args = parse_args()
    expected_rank = int(os.environ["LOCAL_RANK"])
    actual_rank = args.local_rank
    if expected_rank != actual_rank:
        raise RuntimeError(
            "Parameters passed: --local-rank that has different value "
            f"from env var: expected: {expected_rank}, got: {actual_rank}"
        )
    print("End execution")


if __name__ == "__main__":
    main()
