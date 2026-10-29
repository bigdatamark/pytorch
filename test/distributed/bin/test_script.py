#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] HELPER: Trivial smoke-test script used as a launch target by distributed
# launcher tests. Running it as main prints a success line; no import-time side effects.


def main():
    # [FUNCTION DOC] HELPER: Prints "Success, smoke test" to stdout and returns nothing.
    print("Success, smoke test")


if __name__ == "__main__":
    main()
