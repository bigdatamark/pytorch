# [PACKAGE DOC] torch.monitor: re-exports the C++ monitoring bindings (events, stats, event
# handlers, wait counters) from torch._C._monitor via star import, plus the Python
# TensorboardEventHandler that forwards Stat events to a TensorBoard SummaryWriter and the
# STAT_EVENT name constant. Import-time side effects: none beyond importing the native module.
from typing import TYPE_CHECKING

from torch._C._monitor import *  # noqa: F403
from torch._C._monitor import _WaitCounter, _WaitCounterTracker


if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

STAT_EVENT = "torch.monitor.Stat"


class TensorboardEventHandler:
    # [CLASS DOC] Callable event handler that writes monitor events to a TensorBoard writer.
    # State: the writer given at construction. Only events named STAT_EVENT are handled; their
    # data entries become scalars. Intended to be registered with register_event_handler.
    """
    TensorboardEventHandler is an event handler that will write known events to
    the provided SummaryWriter.

    This currently only supports ``torch.monitor.Stat`` events which are logged
    as scalars.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MONITOR)
        >>> # xdoctest: +REQUIRES(module:tensorboard)
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> from torch.monitor import TensorboardEventHandler, register_event_handler
        >>> writer = SummaryWriter("log_dir")
        >>> register_event_handler(TensorboardEventHandler(writer))
    """

    def __init__(self, writer: "SummaryWriter") -> None:
        # [FUNCTION DOC] Stores the SummaryWriter used for all later events. No validation or I/O.
        """
        Constructs the ``TensorboardEventHandler``.
        """
        self._writer = writer

    def __call__(self, event: Event) -> None:
        # [FUNCTION DOC] Handles one event: if its name is the Stat event name, writes each
        # key/value pair of the event data as a scalar to the writer, using the event timestamp as
        # wall time. Other events are ignored. Side effect: writes to the TensorBoard writer.
        if event.name == STAT_EVENT:
            for k, v in event.data.items():
                self._writer.add_scalar(k, v, walltime=event.timestamp.timestamp())
