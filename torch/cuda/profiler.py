# mypy: allow-untyped-defs
# [MODULE DOC] CUDA profiler control: start/stop collection through the CUDA runtime
# and a profile context manager. Also defines DEFAULT_FLAGS, a list of legacy
# profiler option strings (not used within this module).
import contextlib

from . import check_error, cudart


__all__ = ["start", "stop", "profile"]

DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


def start():
    # [FUNCTION DOC] Starts CUDA profiler collection via the runtime; check_error raises on failure.
    r"""Starts cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
    check_error(cudart().cudaProfilerStart())


def stop():
    # [FUNCTION DOC] Stops CUDA profiler collection via the runtime; check_error raises on failure.
    r"""Stops cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
    check_error(cudart().cudaProfilerStop())


@contextlib.contextmanager
def profile():
    # [FUNCTION DOC] Context manager that starts profiling on entry and always stops it on exit.
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> model = torch.nn.Linear(20, 30).cuda()
        >>> inputs = torch.randn(128, 20).cuda()
        >>> with torch.cuda.profiler.profile() as prof:
        ...     model(inputs)
    """
    try:
        start()
        yield
    finally:
        stop()
