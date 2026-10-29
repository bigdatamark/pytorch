#!/usr/bin/env python3

# [MODULE DOC] Provides the default log level for elastic loggers. No import-time side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def get_log_level() -> str:
    # [FUNCTION DOC] Return the default log level name, the string WARNING.
    """
    Return default log level for pytorch.
    """
    return "WARNING"
