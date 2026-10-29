#!/usr/bin/env python3
# [MODULE DOC] HELPER: Test worker script run as a subprocess by multiprocessing tests.
# Reads the RANK environment variable (KeyError if unset) and parses an optional
# --exitcode (default 0) plus a positional comma-separated msg. If exitcode is nonzero it
# writes "exit <code> from <rank>" to stderr and exits with that code. Otherwise, for each
# comma-separated message it prints one line to stdout and one to stderr, each naming the
# message and rank. Only runs when executed as a script.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test binary, exits with exitcode")
    parser.add_argument("--exitcode", type=int, default=0)
    parser.add_argument("msg", type=str)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    exitcode = args.exitcode
    if exitcode != 0:
        print(f"exit {exitcode} from {rank}", file=sys.stderr)
        sys.exit(exitcode)
    else:
        for msg in args.msg.split(","):
            print(f"{msg} stdout from {rank}")
            print(f"{msg} stderr from {rank}", file=sys.stderr)
