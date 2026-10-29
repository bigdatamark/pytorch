# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Defines FlightRecorderLogger, a process-wide singleton wrapper around a stdlib logger named Flight Recorder. No import-time side effects beyond class definition.
import logging
from collections.abc import Callable
from typing import Any


__all__ = ["FlightRecorderLogger"]


class FlightRecorderLogger:
    # [CLASS DOC] Singleton logger facade. The first construction creates the shared instance with an info-level logger writing bare messages to a stream handler; later constructions return the same instance. Exposes debug, info, warning, error and critical as properties returning the matching logger methods, plus set_log_level.
    _instance: Any | None = None
    logger: logging.Logger

    def __init__(self) -> None:
        # [FUNCTION DOC] Re-bind the logger attribute to the stdlib logger named Flight Recorder (runs on every construction; harmless since it is the same logger).
        self.logger: logging.Logger = logging.getLogger("Flight Recorder")

    def __new__(cls) -> Any:
        # [FUNCTION DOC] Singleton constructor. On first call creates the instance, configures the shared logger (info level, message-only formatter, one stream handler) and stores it on the class; always returns the single instance.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger("Flight Recorder")
            cls._instance.logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            cls._instance.logger.addHandler(ch)
        return cls._instance

    def set_log_level(self, level: int) -> None:
        # [FUNCTION DOC] Set the logging level of the underlying logger.
        self.logger.setLevel(level)

    @property
    def debug(self) -> Callable[..., None]:
        # [FUNCTION DOC] Property returning the logger's debug method.
        return self.logger.debug

    @property
    def info(self) -> Callable[..., None]:
        # [FUNCTION DOC] Property returning the logger's info method.
        return self.logger.info

    @property
    def warning(self) -> Callable[..., None]:
        # [FUNCTION DOC] Property returning the logger's warning method.
        return self.logger.warning

    @property
    def error(self) -> Callable[..., None]:
        # [FUNCTION DOC] Property returning the logger's error method.
        return self.logger.error

    @property
    def critical(self) -> Callable[..., None]:
        # [FUNCTION DOC] Property returning the logger's critical method.
        return self.logger.critical
