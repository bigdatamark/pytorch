# [MODULE DOC]
# Logging helpers for Dynamo/Inductor: a list of the loggers they own and a
# step-numbered logger factory with an optional progress bar.
# Import-time side effects: defines a global step counter; the progress bar
# (pbar, num_steps) is only created if disable_progress is false, which it is
# not by default, so normally nothing is created.
"""Logging utilities for Dynamo and Inductor.

This module provides specialized logging functionality including:
- Step-based logging that prepends step numbers to log messages
- Progress bar management for compilation phases
- Centralized logger management for Dynamo and Inductor components

The logging system helps track the progress of compilation phases and provides structured
logging output for debugging and monitoring.
"""

import itertools
import logging
from collections.abc import Callable
from typing import Any

from torch.hub import _Faketqdm, tqdm


# Disable progress bar by default, not in dynamo config because otherwise get a circular import
disable_progress = True


# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers() -> list[logging.Logger]:
    # [FUNCTION DOC]
    # Returns a new list of the standard logging.Logger objects for the symbolic
    # shapes module, torch._dynamo and torch._inductor. No side effects.
    return [
        logging.getLogger("torch.fx.experimental.symbolic_shapes"),
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
    ]


# Creates a logging function that logs a message with a step # prepended.
# get_step_logger should be lazily called (i.e. at runtime, not at module-load time)
# so that step numbers are initialized properly. e.g.:

# @functools.cache
# def _step_logger():
#     return get_step_logger(logging.getLogger(...))

# def fn():
#     _step_logger()(logging.INFO, "msg")

_step_counter = itertools.count(1)

# Update num_steps if more phases are added: Dynamo, AOT, Backend
# This is very inductor centric
# _inductor.utils.has_triton() gives a circular import error here

if not disable_progress:
    try:
        import triton  # noqa: F401

        num_steps = 3
    except ImportError:
        num_steps = 2
    pbar = tqdm(total=num_steps, desc="torch.compile()", delay=0)


def get_step_logger(logger: logging.Logger) -> Callable[..., None]:
    # [FUNCTION DOC]
    # Takes a logger and returns a logging function (level, msg, **kwargs) that
    # prefixes messages with a unique step number. Side effects: advances the
    # global step counter; if progress is enabled, advances the progress bar and
    # sets its postfix to the logger name.
    # Callers should invoke it lazily (at runtime) so step numbers are meaningful.
    if not disable_progress:
        pbar.update(1)
        if not isinstance(pbar, _Faketqdm):
            pbar.set_postfix_str(f"{logger.name}")

    step = next(_step_counter)

    def log(level: int, msg: str, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Logs msg at the given level on the captured
        # logger, prefixed with the captured step number. Defaults stacklevel to
        # 2 so the reported caller is the code calling this function. Extra
        # keyword arguments are forwarded to the logger.
        if "stacklevel" not in kwargs:
            kwargs["stacklevel"] = 2
        logger.log(level, "Step %s: %s", step, msg, **kwargs)

    return log
