#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] HELPER: Builds the logger used by sharded tensor code. Looks up a handler by destination name
# in the log handler registry, attaches a timestamped formatter, and returns a non-propagating
# DEBUG-level logger named after the handler type. No import-time side effects beyond imports.
import logging

from torch.distributed._shard.sharded_tensor.logging_handlers import _log_handlers


__all__: list[str] = []


def _get_or_create_logger() -> logging.Logger:
    # [FUNCTION DOC] HELPER: Creates (or reconfigures) the logger named "sharding-spec-<handler class name>" using the
    # default handler: sets DEBUG level, applies a formatter with time/file/line/level/process/thread, disables
    # propagation and adds the handler. Side effect: each call attaches the handler again to the shared logger.
    # Returns the logging.Logger.
    logging_handler, log_handler_name = _get_logging_handler()
    logger = logging.getLogger(f"sharding-spec-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = "default",
) -> tuple[logging.Handler, str]:
    # [FUNCTION DOC] HELPER: Looks up the handler registered for the destination name (default "default") in the
    # handler registry and returns it with its class name. Raises KeyError for an unregistered destination.
    log_handler = _log_handlers[destination]
    log_handler_name = type(log_handler).__name__
    return (log_handler, log_handler_name)
