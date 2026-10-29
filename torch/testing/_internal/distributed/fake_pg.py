# mypy: allow-untyped-defs

# [MODULE DOC] Registers the fake process group backend for tests. Re-exports the C++
# FakeProcessGroup and FakeStore (see __all__) and, at import time, registers a factory for the
# Backend.FAKE backend on cpu, cuda, hpu and xpu devices using the extended backend API. No other
# logic lives here.
import torch.distributed as dist
from torch._C._distributed_c10d import FakeProcessGroup, FakeStore


# FakeStore is a no-op Key-Value store (implemented in C++) for initialization
# of the fake process group; one can either use FakeStore or HashStore. It used
# to be a Python class defined here, so it is re-exported to keep
# `from ...fake_pg import FakeStore` working.
__all__ = ["FakeProcessGroup", "FakeStore"]


def _create_fake_pg(common_opts, backend_opts):
    # [FUNCTION DOC] HELPER: Backend factory for the fake backend. Takes common options (group rank
    # and group size) and backend options and returns a new FakeProcessGroup built through its
    # internal creator. No communication happens; collectives produce meaningless results.
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convenient tool when playing
    with distributed but don't care about the actual data.
    """
    return FakeProcessGroup._create_internal(
        common_opts.group_rank, common_opts.group_size, backend_opts
    )


dist.Backend.register_backend(
    dist.Backend.FAKE,
    _create_fake_pg,
    extended_api=True,
    devices=["cpu", "cuda", "hpu", "xpu"],
)
