#!/usr/bin/env python3
# [MODULE DOC] HELPER: Test worker script run as a subprocess by multiprocessing tests.
# Reads RANK from the environment (KeyError if unset) and parses --exitcode (default 0)
# plus a positional msg. A nonzero exitcode prints "exit <code> from <rank>" to stderr and
# exits immediately with that code. With exitcode zero it sleeps 1000 seconds (a
# long-running worker meant to be killed or timed out by the test) before printing the
# message to stdout and stderr.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import time


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
        time.sleep(1000)
        print(f"{args.msg} stdout from {rank}")
        print(f"{args.msg} stderr from {rank}", file=sys.stderr)
