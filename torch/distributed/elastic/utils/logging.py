#!/usr/bin/env python3

# [MODULE DOC] Logger setup helpers for elastic: get_logger returns a stdlib logger whose level comes from the LOGLEVEL environment variable (default from get_log_level) and whose name defaults to the caller's module name. No import-time side effects.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import warnings

from torch.distributed.elastic.utils.log_level import get_log_level


def get_logger(name: str | None = None) -> logging.Logger:
    # [FUNCTION DOC] Return a logger for name, or for the calling module's name when name is not given (derived from the call stack, two frames up). Sets the logger level from LOGLEVEL or the default WARNING. Side effect: mutates the shared logger's level.
    """
    Util function to set up a simple logger that writes
    into stderr. The loglevel is fetched from the LOGLEVEL
    env. variable or WARNING as default. The function will use the
    module name of the caller if no name is provided.

    Args:
        name: Name of the logger. If no name provided, the name will
              be derived from the call stack.
    """

    # Derive the name of the caller, if none provided
    # Use depth=2 since this function takes up one level in the call stack
    return _setup_logger(name or _derive_module_name(depth=2))


def _setup_logger(name: str | None = None) -> logging.Logger:
    # [FUNCTION DOC] HELPER: Get the stdlib logger with the given name and set its level from the LOGLEVEL environment variable, falling back to the default log level. Returns the logger.
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOGLEVEL", get_log_level()))
    return logger


def _derive_module_name(depth: int = 1) -> str | None:
    # [FUNCTION DOC] HELPER: Return the name of the module that owns the stack frame at the given depth, or the file base name without extension if the module cannot be found. On any failure emits a RuntimeWarning and returns None.
    """
    Derives the name of the caller module from the stack frames.

    Args:
        depth: The position of the frame in the stack.
    """
    try:
        stack = inspect.stack()
        if depth >= len(stack):
            raise AssertionError
        # FrameInfo is just a named tuple: (frame, filename, lineno, function, code_context, index)
        frame_info = stack[depth]

        module = inspect.getmodule(frame_info[0])
        if module:
            module_name = module.__name__
        else:
            # inspect.getmodule(frame_info[0]) does NOT work (returns None) in
            # binaries built with @mode/opt
            # return the filename (minus the .py extension) as modulename
            filename = frame_info[1]
            module_name = os.path.splitext(os.path.basename(filename))[0]
        return module_name
    except Exception as e:
        warnings.warn(
            f"Error deriving logger module name, using <None>. Exception: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
