#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]
# [MODULE DOC] HELPER: Test worker script used to simulate a hung process. When run as a
# script it sleeps 600 seconds and then prints "finished work"; tests are expected to kill
# it earlier. No import-time side effects.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time


if __name__ == "__main__":
    time.sleep(600)
    print("finished work")
