# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Test for the DTensor op-coverage debug utility. It checks that tracing a small
# MLP with the inductor decomposition table yields a forward and a backward FX graph.
# No import-time side effects beyond imports.

import torch
import torch.nn as nn
from torch.distributed.tensor.debug._op_coverage import get_inductor_decomp_graphs
from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleMLP(nn.Module):
    # [CLASS DOC] HELPER: Tiny model used as the tracing target: Linear(50, 32), ReLU,
    # Linear(32, 8), followed by a sigmoid on the output.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the two linear layers (net1, net2) and the ReLU.
        super().__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies net1, ReLU, net2 and sigmoid to the input batch and
        # returns the resulting tensor.
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))


class TestOpCoverage(TestCase):
    # [CLASS DOC] Single-test case class for the op-coverage tracing helper; no state.
    def test_trace_with_inductor_decomp(self):
        # [FUNCTION DOC] Calls get_inductor_decomp_graphs on SimpleMLP with one random 8x50
        # input and no kwargs. Fails if the result does not hold exactly two graphs, and
        # asserts both are torch.fx.GraphModule (forward then backward).
        model = SimpleMLP()
        args = (torch.randn(8, 50),)
        kwargs = {}
        graphs = get_inductor_decomp_graphs(model, args, kwargs)
        if len(graphs) != 2:
            raise AssertionError(f"Expect fwd + bwd graphs, got {len(graphs)}")
        self.assertIsInstance(graphs[0], torch.fx.GraphModule)
        self.assertIsInstance(graphs[1], torch.fx.GraphModule)


if __name__ == "__main__":
    run_tests()
