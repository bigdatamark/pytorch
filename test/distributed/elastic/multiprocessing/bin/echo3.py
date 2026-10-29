#!/usr/bin/env python3
# [MODULE DOC] HELPER: Test worker script run as a subprocess by multiprocessing tests.
# Reads RANK from the environment (KeyError if unset) and parses --segfault plus a
# positional msg. If --segfault is given with any non-empty value it reads memory at
# address zero via ctypes, killing the process with SIGSEGV; otherwise it prints the
# message and rank to stdout.
# KNOWN ISSUE: --segfault uses a bool type converter, so any non-empty string (even
# "False") is truthy and triggers the crash.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ctypes
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="test binary, triggers a segfault (SIGSEGV)"
    )
    parser.add_argument("--segfault", type=bool, default=False)
    parser.add_argument("msg", type=str)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])

    if args.segfault:
        ctypes.string_at(0)
    else:
        print(f"{args.msg} from {rank}")
