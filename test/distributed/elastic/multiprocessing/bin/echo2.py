#!/usr/bin/env python3
# [MODULE DOC] HELPER: Test worker script run as a subprocess by multiprocessing tests.
# Reads RANK from the environment (KeyError if unset) and parses --raises plus a positional
# msg. If --raises is given with any non-empty value it raises a RuntimeError naming the
# rank; otherwise it prints the message and rank to stdout.
# KNOWN ISSUE: --raises uses a bool type converter, so any non-empty string (even "False")
# is truthy and enables raising; only the default (flag absent) avoids the error.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test binary, raises a RuntimeError")
    parser.add_argument("--raises", type=bool, default=False)
    parser.add_argument("msg", type=str)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])

    if args.raises:
        raise RuntimeError(f"raised from {rank}")
    else:
        print(f"{args.msg} from {rank}")
