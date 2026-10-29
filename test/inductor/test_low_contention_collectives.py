# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Test module for the low-contention collectives FX graph pass in
# Inductor. It checks that replace_collectives_with_low_contention rewrites an
# all-gather node into the symmetric-memory low-contention variant only when the
# overlap check passes or is explicitly skipped through configuration.
# Importing it has no side effects beyond the imports; running it as a script
# invokes the standard test runner.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch._inductor import config
from torch._inductor.fx_passes.low_contention_collectives import (
    replace_collectives_with_low_contention,
)
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not dist.is_available(), "requires distributed")
class TestLowContentionCollectives(TestCase):
    # [CLASS DOC] Test case for the low-contention collective replacement pass.
    # Skipped entirely when torch.distributed is unavailable. Holds a graph builder
    # helper and one test; it keeps no per-instance state.
    def _build_ag_graph(self):
        # [FUNCTION DOC] HELPER: Builds and returns a small FX graph with one placeholder
        # input (fake value of shape 1024 by 1024), a functional all_gather_into_tensor
        # node over a group named test_group with world size 2 (fake value 2048 by 1024),
        # and a wait_tensor node on it that feeds the graph output. There is no compute
        # in the graph, so no collective can overlap with computation.
        """Build an FX graph: input -> all_gather -> wait_tensor -> output (no compute)."""
        c10d_fn = torch.ops._c10d_functional
        graph = torch.fx.Graph()
        inp = graph.placeholder("input")
        inp.meta["val"] = torch.empty(1024, 1024)

        ag = graph.call_function(
            c10d_fn.all_gather_into_tensor.default,
            args=(inp, 2, "test_group"),
        )
        ag.meta["val"] = torch.empty(2048, 1024)

        wait = graph.call_function(c10d_fn.wait_tensor.default, args=(ag,))
        wait.meta["val"] = torch.empty(2048, 1024)

        graph.output(wait)
        return graph

    @patch(
        "torch._inductor.fx_passes.low_contention_collectives._enable_symm_mem",
        return_value=True,
    )
    def test_skip_overlap_check(self, _mock):
        # [FUNCTION DOC] Verifies the skip_overlap_check configuration of the pass. The
        # symmetric-memory enablement check is mocked to report true (via the patch
        # decorator argument, unused here) and the minimum bytes per rank is set to zero.
        # With skip_overlap_check false, the pass leaves the all-gather in place because
        # there is no overlapping compute. With it true, the all-gather is replaced by
        # the symm_mem low-contention all-gather and the original op disappears. Each
        # case uses a fresh graph and a scoped config patch.
        c10d_fn = torch.ops._c10d_functional
        symm_mem = torch.ops.symm_mem
        lc_config = {
            "aten_distributed_optimizations.low_contention_min_bytes_per_rank": 0,
        }

        def get_targets(graph):
            # [FUNCTION DOC] HELPER: Returns the list of call targets of all call_function
            # nodes in the given FX graph, in node order.
            return [n.target for n in graph.nodes if n.op == "call_function"]

        # Default: no compute overlap -> collective NOT replaced
        graph = self._build_ag_graph()
        with config.patch(
            {
                **lc_config,
                "aten_distributed_optimizations.low_contention_skip_overlap_check": False,
            }
        ):
            replace_collectives_with_low_contention(graph)
        self.assertIn(c10d_fn.all_gather_into_tensor.default, get_targets(graph))

        # skip_overlap_check=True: collective replaced despite no compute
        graph = self._build_ag_graph()
        with config.patch(
            {
                **lc_config,
                "aten_distributed_optimizations.low_contention_skip_overlap_check": True,
            }
        ):
            replace_collectives_with_low_contention(graph)
        self.assertIn(symm_mem._low_contention_all_gather.default, get_targets(graph))
        self.assertNotIn(c10d_fn.all_gather_into_tensor.default, get_targets(graph))


if __name__ == "__main__":
    run_tests()
