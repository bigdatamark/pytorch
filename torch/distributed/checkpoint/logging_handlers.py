# [MODULE DOC] Registers the DCP logger name with the shared distributed log-handler table.
# Defines DCP_LOGGER_NAME. Import-time side effect: adds a no-op handler under that name
# to the global handler mapping in torch.distributed.logging_handlers.
import logging

from torch.distributed.logging_handlers import _log_handlers


__all__: list[str] = []

DCP_LOGGER_NAME = "dcp_logger"

_log_handlers.update(
    {
        DCP_LOGGER_NAME: logging.NullHandler(),
    }
)
