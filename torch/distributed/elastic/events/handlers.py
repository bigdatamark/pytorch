#!/usr/bin/env python3
# [MODULE DOC] Registry of named logging handlers for elastic events. Import creates a console
# stream handler and two null handlers keyed by destination name in a module-level mapping.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging


_log_handlers: dict[str, logging.Handler] = {
    "console": logging.StreamHandler(),
    "dynamic_rendezvous": logging.NullHandler(),
    "null": logging.NullHandler(),
}


def get_logging_handler(destination: str = "null") -> logging.Handler:
    # [FUNCTION DOC] Returns the registered logging handler for the destination name (console,
    # dynamic_rendezvous or null; default null). Raises a key error for unknown names.
    global _log_handlers
    return _log_handlers[destination]
