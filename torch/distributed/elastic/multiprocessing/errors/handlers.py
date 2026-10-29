#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Multiprocessing error-reporting module

# [MODULE DOC] Exposes a factory returning the default ErrorHandler used to record
# worker errors. Only imports ErrorHandler; no import-time side effects.

from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler


__all__ = ["get_error_handler"]


def get_error_handler() -> ErrorHandler:
    # [FUNCTION DOC] Returns a new default ErrorHandler instance (no arguments, no side effects).
    return ErrorHandler()
