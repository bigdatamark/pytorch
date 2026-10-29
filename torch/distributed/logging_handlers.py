#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Holds the registry of logging handlers used by torch.distributed.
# Defines an empty public export list and a private mapping from handler
# name to logging handler, containing only a default no-op NullHandler.
# Import-time side effect: constructs that NullHandler.
import logging


__all__: list[str] = []

_log_handlers: dict[str, logging.Handler] = {
    "default": logging.NullHandler(),
}
