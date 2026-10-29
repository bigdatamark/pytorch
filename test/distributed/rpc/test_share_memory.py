#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests that RPC can pickle modules whose tensors live in shared memory, using a
# custom pickler that routes storages and tensors through torch.multiprocessing reductions.
# Exits with status 0 at import when torch.distributed is unavailable. Contains a context
# manager switching the sharing strategy, a pickler subclass, two worker functions, and one
# two-process test case.
import contextlib
import copyreg
import os
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.rpc as rpc
import torch.multiprocessing.reductions as TorchMpReductions
from torch import multiprocessing
from torch.distributed.rpc.api import _use_rpc_pickler
from torch.distributed.rpc.internal import _InternalRPCPickler
from torch.testing._internal.common_utils import run_tests, TestCase


@contextlib.contextmanager
def fs_sharing():
    # [FUNCTION DOC] HELPER: Context manager that temporarily sets the torch.multiprocessing
    # sharing strategy to file_system and restores the previous strategy on exit, even if the
    # body raises.
    prev_strategy = multiprocessing.get_sharing_strategy()
    multiprocessing.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        multiprocessing.set_sharing_strategy(prev_strategy)


class ShareMemoryRPCPickler(_InternalRPCPickler):
    # [CLASS DOC] HELPER: RPC pickler that serializes storages, tensors and Parameters with
    # the torch.multiprocessing reductions (reduce_storage / reduce_tensor) so shared-memory
    # tensors are sent by handle. Its _dispatch_table is a copy of the copyreg table plus
    # those entries.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the dispatch table: copies the global copyreg table
        # and overrides entries for all torch storage classes, tensor classes, Tensor and
        # Parameter.
        # KNOWN ISSUE: the bare attribute read of _dispatch_table before the assignment is a
        # no-op statement.
        super().__init__()
        self._dispatch_table
        # pyre-fixme[4]: Attribute must be annotated.
        self._dispatch_table = copyreg.dispatch_table.copy()

        for t in torch._storage_classes:
            self._dispatch_table[t] = TorchMpReductions.reduce_storage

        for t in torch._tensor_classes:
            self._dispatch_table[t] = TorchMpReductions.reduce_tensor
        self._dispatch_table[torch.Tensor] = TorchMpReductions.reduce_tensor
        self._dispatch_table[torch.nn.parameter.Parameter] = (
            TorchMpReductions.reduce_tensor
        )


def worker_loop(a):
    # [FUNCTION DOC] HELPER: Entry point of the spawned second process; joins the RPC group
    # as worker1 (rank 1 of 2) and immediately shuts down. The argument is unused.
    rpc.init_rpc("worker1", rank=1, world_size=2)
    rpc.shutdown()


def worker_fn(m):
    # [FUNCTION DOC] HELPER: Remote target that does nothing; receives the shared module.
    pass


class TestRPCPickler(TestCase):
    # [CLASS DOC] Test case holding the shared-memory RPC pickler test.
    def test_case(self):
        # [FUNCTION DOC] Sets MASTER_ADDR/MASTER_PORT (localhost:29500) in the environment and
        # spawns a peer process running worker_loop under the file_system sharing strategy.
        # Here it uses ShareMemoryRPCPickler to init RPC as worker0, shares a Linear module's
        # memory, sends it with rpc.remote to worker1 and waits on the RRef. RPC is always
        # shut down and the peer joined at the end.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        with fs_sharing():
            r = multiprocessing.spawn(worker_loop, join=False)

            try:
                with _use_rpc_pickler(ShareMemoryRPCPickler()):
                    rpc.init_rpc("worker0", rank=0, world_size=2)
                    m = torch.nn.Linear(1, 2)
                    m.share_memory()
                    rref = rpc.remote("worker1", worker_fn, args=(m,))

                    rref.to_here()
            finally:
                rpc.shutdown()
                r.join()


if __name__ == "__main__":
    run_tests()
