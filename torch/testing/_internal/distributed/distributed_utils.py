# mypy: allow-untyped-defs

# [MODULE DOC] Test helpers that initialize torch.distributed with a mock process group. Defines
# MockProcessGroup, a no-communication ProcessGroup subclass, and registers it as the
# "mock_process_group" backend at import time (global side effect on dist.Backend). Exports
# mock_init_dist, the with_dist context manager and the with_fake_comms decorator, which let tests
# query rank and world size without real communication.
from contextlib import contextmanager
from datetime import timedelta
from functools import partial, wraps

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


class MockProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] Minimal ProcessGroup that performs no communication; it only carries a rank and
    # world size. Its only override reports the backend name "mock_process_group".
    def getBackendName(self):
        # [FUNCTION DOC] Returns the fixed backend name string "mock_process_group".
        return "mock_process_group"


def create_mock_pg(prefix_store, rank, world_size, timeout):
    # [FUNCTION DOC] Backend factory registered for "mock_process_group". Ignores the store and
    # timeout and returns a new MockProcessGroup built from the given rank and world size.
    return MockProcessGroup(rank, world_size)


dist.Backend.register_backend("mock_process_group", create_mock_pg)


def mock_init_dist(rank, world_size):
    # [FUNCTION DOC] Initializes the default process group as a mock one for the given rank and
    # world size. Raises an AssertionError if a default group already exists. Side effects: creates
    # an in-memory HashStore, pre-increments the store based barrier counter so the init barrier
    # believes all other ranks arrived, then calls init_process_group with the mock backend, group
    # name "fake" and a one second timeout. Relies on c10d internals.
    # !!! WARNING !!!
    # Kids don't try this at home, this is a cute pile of hacks that
    # depends on a small mountain of c10d internals
    if dist.is_initialized():
        raise AssertionError("Expected dist to not be initialized")
    store = dist.HashStore()
    # Trick _store_based_barrier into believing everyone else already checked-in
    # Zero is the group index
    store.add(f"{c10d.STORE_BASED_BARRIER_PREFIX}:0", world_size - 1)
    dist.init_process_group(
        backend="mock_process_group",
        rank=rank,
        world_size=world_size,
        store=store,
        group_name="fake",
        timeout=timedelta(seconds=1),
    )


@contextmanager
def with_dist(rank=0, world_size=2):
    # [FUNCTION DOC] Context manager that initializes the mock process group (via mock_init_dist)
    # with the given rank and world size, yields, and always destroys the process group on exit.
    # Raises whatever mock_init_dist raises.
    """
    Context manager that initializer c10d with a fake process group.
    """
    mock_init_dist(rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()


def with_fake_comms(func=None, rank=0, world_size=2):
    # [FUNCTION DOC] Decorator factory for test methods. Used bare it wraps the function;
    # used with rank and world_size arguments it returns a partial that wraps later. The
    # wrapped method runs inside with_dist, so a mock process group exists for the call. The
    # wrapper's return value is discarded.
    # KNOWN ISSUE: the wrapper does not return the result of the wrapped function.
    """
    Function wrapper that inits a fake process group designed for testing.
    Right now only querying for world size is available
    """
    if func is None:
        return partial(with_fake_comms, rank=rank, world_size=world_size)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the wrapped test method inside a mock process group context.
        # Forwards all arguments; propagates exceptions; returns nothing.
        with with_dist(rank, world_size):
            func(self, *args, **kwargs)

    return wrapper
