# [MODULE DOC] Logging setup helper for lumen CLI tasks. No import-time side effects.
"""
Logger Utility helpers for CLI tasks.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO):
    # [FUNCTION DOC] Configure the root logger to write to standard output at the given
    # level with a timestamp, level and logger name format. Global side effect; has no
    # effect if the root logger already has handlers.
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
