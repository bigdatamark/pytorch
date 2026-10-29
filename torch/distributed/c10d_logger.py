#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC]
# Logging utilities for c10d (torch.distributed) APIs: a logger factory, a message builder
# describing the current distributed state, and two decorators (exception logging and
# wait-counter timing). Import-time side effect: creates the module-level c10d logger
# (_c10d_logger) with the default destination handler, level DEBUG, no propagation.
# Exports nothing publicly (empty __all__).

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
from torch.monitor import _WaitCounter


__all__: list[str] = []

_DEFAULT_DESTINATION = "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    # [FUNCTION DOC] HELPER: returns the logger named for the destination's handler, setting
    # level DEBUG, no propagation, a formatter with time, file, line, level, process and
    # thread, and attaching the handler. Mutates logging global state.
    # KNOWN ISSUE: each call adds the handler again, so repeated calls duplicate handlers.
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"c10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> tuple[logging.Handler, str]:
    # [FUNCTION DOC] HELPER: looks up the handler registered for the destination and returns it
    # with a display name made of its class name and the destination. Raises KeyError for an
    # unknown destination.
    log_handler = _log_handlers[destination]
    log_handler_name = f"{type(log_handler).__name__}-{destination}"
    return (log_handler, log_handler_name)


# pyrefly: ignore [unknown-name]
global _c10d_logger
_c10d_logger = _get_or_create_logger()


def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: builds a mapping of string fields describing a call. Always has
    # the function name. If the distributed package is initialized, also adds process group
    # name, backend, world size, group size, global and group rank (group taken from a group
    # or process_group keyword), and the NCCL version for the nccl backend.
    # KNOWN ISSUE: the group name is looked up from a keyword named pg, not the group found above.
    if dist.is_initialized():
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict = {
            "func_name": f"{func_name}",
            "pg_name": f"{dist._get_process_group_name(kwargs.get('pg'))}",  # type: ignore[arg-type]
            "backend": f"{dist.get_backend(group)}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(group)}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(group)}",
        }
        if msg_dict["backend"] == "nccl":
            nccl_version = torch.cuda.nccl.version()
            msg_dict["nccl_version"] = ".".join(str(v) for v in nccl_version)
    else:
        msg_dict = {
            "func_name": f"{func_name}",
        }
    return msg_dict


_T = TypeVar("_T")
_P = ParamSpec("_P")


def _exception_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] HELPER: decorator. The wrapped function behaves as the original, but any
    # exception is first logged at debug level (message mapping plus error text) and re-raised.
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        # [FUNCTION DOC] HELPER: calls the wrapped function; on exception logs and re-raises it.
        try:
            return func(*args, **kwargs)
        except Exception as error:
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
            msg_dict["error"] = f"{error}"
            _c10d_logger.debug(msg_dict)
            raise

    return wrapper


def _time_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] HELPER: decorator. The wrapped function behaves as the original but runs
    # inside a wait counter guard named after the function, for timing metrics.
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        # [FUNCTION DOC] HELPER: runs the wrapped function under the wait counter and returns its result.
        with _WaitCounter(f"pytorch.wait_counter.c10d.{func.__name__}").guard():
            func_return = func(*args, **kwargs)
        return func_return

    return wrapper
