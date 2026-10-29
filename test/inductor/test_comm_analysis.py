# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the device resolution used when inductor estimates NCCL collective
# runtime from an FX node (torch._inductor.comm_analysis). Import-time side effect: if the
# distributed package or NCCL is unavailable, or common_distributed cannot be imported, a message
# goes to stderr and the whole process exits with status 0 (tests skipped). Contains a graph-
# tracing helper and one TestCase using fake and real (hash store) process groups.

import sys
import unittest
import warnings

import torch
import torch.distributed as dist


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

try:
    from torch.testing._internal.common_distributed import requires_nccl
except ImportError:
    print("common_distributed not importable, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


def _get_all_gather_node(group_size, group_name):
    # [FUNCTION DOC] HELPER: Traces a small function with make_fx that calls the functional
    # all_gather_into_tensor collective followed by wait_tensor on a 4x4 ones tensor, using the
    # given group size and group name. Returns the first call_function node whose target string
    # mentions all_gather_into_tensor; raises RuntimeError if the traced graph has no such node.
    """Trace a simple all_gather function and return the collective FX node."""

    def func(inp, group_size, group_name):
        # [FUNCTION DOC] HELPER: Traced body: all-gathers the input over the named group, then
        # waits on the result and returns the waited tensor.
        out = torch.ops._c10d_functional.all_gather_into_tensor(
            inp, group_size, group_name
        )
        wait = torch.ops._c10d_functional.wait_tensor(out)
        return wait

    gm = make_fx(func)(torch.ones(4, 4), group_size, group_name)
    for n in gm.graph.nodes:
        if n.op == "call_function" and "all_gather_into_tensor" in str(n.target):
            return n
    raise RuntimeError("No all_gather_into_tensor node found in traced graph")


class TestNcclEstimateDeviceResolution(TestCase):
    # [CLASS DOC] Tests that the NCCL runtime estimator picks the right backend for a process
    # group (fake, multi-backend, single NCCL). Each test initializes the default process group
    # and registers it under a fixed group name, and must destroy it afterwards.
    """
    Tests for the device resolution fix in _nccl_estimate() inside
    estimate_nccl_collective_runtime_from_fx_node.
    """

    def _init_pg(self, backend, world_size=2):
        # [FUNCTION DOC] HELPER: Initializes the default process group with the given backend
        # (rank 0, world size default 2) over a FakeStore, registers it as a named group with
        # c10d, and returns (group, group name, group size). Side effect: global default process
        # group state.
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(
            backend=backend, rank=0, world_size=world_size, store=store
        )
        pg = dist.group.WORLD
        group_name = "test_comm_analysis"
        torch._C._distributed_c10d._register_process_group(group_name, pg)
        return pg, group_name, pg.size()

    def _init_pg_real_store(self, backend, world_size=1):
        # [FUNCTION DOC] HELPER: Same as the fake-store initializer but uses a real HashStore and
        # a default world size of 1, so real backends (gloo/nccl) can be constructed. Registers
        # the group under the fixed test name and returns (group, group name, group size). Side
        # effect: global default process group state.
        store = dist.HashStore()
        dist.init_process_group(
            backend=backend, rank=0, world_size=world_size, store=store
        )
        pg = dist.group.WORLD
        group_name = "test_comm_analysis"
        torch._C._distributed_c10d._register_process_group(group_name, pg)
        return pg, group_name, pg.size()

    def _destroy_pg(self):
        # [FUNCTION DOC] HELPER: Destroys the default process group created by a test.
        dist.destroy_process_group()

    def test_fake_backend_falls_back_to_analytical(self):
        # [FUNCTION DOC] With the fake backend, estimating the collective runtime with the NCCL
        # estimator enabled must still yield a positive value and must equal the purely analytical
        # estimate (the estimator cannot be used, so it falls back). The process group is
        # destroyed in a finally clause.
        """FAKE backend: _nccl_estimate returns None, falls back to analytical formula."""
        pg, group_name, group_size = self._init_pg("fake")
        try:
            node = _get_all_gather_node(group_size, group_name)
            from torch._inductor.comm_analysis import (
                estimate_nccl_collective_runtime_from_fx_node,
            )

            est_ms = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=True
            )
            self.assertGreater(est_ms, 0)

            est_ms_analytical = estimate_nccl_collective_runtime_from_fx_node(
                node, use_nccl_estimator=False
            )
            self.assertEqual(est_ms, est_ms_analytical)
        finally:
            self._destroy_pg()

    @requires_nccl()
    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_multi_backend_pg_resolves_to_nccl(self):
        # [FUNCTION DOC] Requires NCCL and CUDA. Builds a combined cpu gloo plus cuda nccl group
        # on a real store. Checks that the group's default device is the CPU, that the cuda device
        # resolves to a backend that supports time estimation, and that the cpu device resolves to
        # a backend that does not. FutureWarning from the default-device lookup is suppressed.
        """
        Multi-backend PG ("cpu:gloo,cuda:nccl"): We should resolve to the cuda device's backend.
        """
        torch.cuda.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store("cpu:gloo,cuda:nccl")
        try:
            from torch.distributed.distributed_c10d import _get_pg_default_device

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                default_device = _get_pg_default_device(pg)
            self.assertEqual(default_device, torch.device("cpu"))

            nccl_backend = pg._get_backend(torch.device("cuda"))
            self.assertTrue(nccl_backend.supports_time_estimate)

            gloo_backend = pg._get_backend(torch.device("cpu"))
            self.assertFalse(gloo_backend.supports_time_estimate)
        finally:
            self._destroy_pg()

    @requires_nccl()
    @unittest.skipUnless(TEST_CUDA, "requires CUDA")
    def test_single_nccl_backend_resolves_correctly(self):
        # [FUNCTION DOC] Requires NCCL and CUDA. Builds a single-backend NCCL group on a real
        # store and checks that the backend resolved for the cuda device supports time estimation.
        """Single NCCL backend PG: cuda device resolves to NCCL with time estimation."""
        torch.cuda.set_device(0)
        pg, group_name, group_size = self._init_pg_real_store("nccl")
        try:
            backend = pg._get_backend(torch.device("cuda"))
            self.assertTrue(backend.supports_time_estimate)
        finally:
            self._destroy_pg()


if __name__ == "__main__":
    run_tests()
