#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Worker script that records whether torch.distributed.is_torchelastic_launched()
# returns true, by writing the resulting boolean text to the file given by --out-file. The
# launcher tests run it both under torchrun and directly via runpy and compare the file.
# No import-time side effects beyond importing torch.distributed.

"""
This is a test script that launches as part of the test cases in
run_test.py, to validate the correctness of
the method ``torch.distributed.is_torchelastic_launched()``. To do so,
we run this script with and without torchelastic and validate that the
boolean value written to the out_file is indeed what we expect (e.g.
should be False when not launched with torchelastic, True when launched with)
The script itself is not a test case hence no assertions are made in this script.

see: - test/distributed/launcher/run_test.py#test_is_torchelastic_launched()
     - test/distributed/launcher/run_test.py#test_is_not_torchelastic_launched()
"""

import argparse

import torch.distributed as dist


def parse_args():
    # [FUNCTION DOC] HELPER: parses the single option --out-file / --out_file (path of the file to
    # write) and returns the namespace. Exits the process on invalid arguments.
    parser = argparse.ArgumentParser(description="test script")
    parser.add_argument(
        "--out-file",
        "--out_file",
        help="file to write indicating whether this script was launched with torchelastic",
    )
    return parser.parse_args()


def main():
    # [FUNCTION DOC] Opens the out file for writing (truncating it) and writes the string form of
    # the is_torchelastic_launched result, either True or False, with no trailing newline.
    # Raises if --out-file is missing (open on None) or cannot be written.
    args = parse_args()
    with open(args.out_file, "w") as out:
        out.write(f"{dist.is_torchelastic_launched()}")


if __name__ == "__main__":
    main()
