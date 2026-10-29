# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for the block traversal mode of the FX net minimizer (_MinimizerBase from
# torch.fx.passes.net_min_base). A small traced three-layer module is minimized against mocked
# comparison behavior and the culprit node sets are checked.
# Running the file directly raises a RuntimeError since it is not used by test discovery.

from unittest import mock

import torch
from torch.fx.passes.net_min_base import (
    _MinimizerBase,
    _MinimizerSettingBase,
    FxNetMinimizerResultMismatchError,
)
from torch.fx.passes.tools_common import Names
from torch.testing._internal.common_utils import TestCase


class TestNetMinBaseBlock(TestCase):
    # [CLASS DOC] Test case with a fixture building a traced Linear, Linear, ReLU module and a
    # minimizer in block traverse mode with a mocked compare function. Tests simulate which
    # nodes cause a mismatch and assert the minimizer reports exactly those nodes.
    def setUp(self) -> None:
        # [FUNCTION DOC] Fixture run before each test. Defines and traces a three-layer module
        # (linear, linear2, relu), stores a mock compare function, a sample input of shape 2 by
        # 10, block mode settings, the _MinimizerBase instance and an empty report list on self.
        super().setUp()
        # Setup test fixtures for each test method

        class SimpleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying linear (10 to 5), linear2 (5 to 5) and a
            # relu in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two Linear layers and the ReLU layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Runs the input through linear, linear2 and relu and
                # returns the result.
                x = self.linear(x)
                x = self.linear2(x)
                x = self.relu(x)
                return x

        self.compare_fn = mock.MagicMock()

        self.module = torch.fx.symbolic_trace(SimpleModule())
        self.sample_input = (torch.randn(2, 10),)
        self.settings = _MinimizerSettingBase(traverse_method="block")
        self.minimizer = _MinimizerBase(
            module=self.module,
            sample_input=self.sample_input,
            settings=self.settings,
            compare_fn=self.compare_fn,
        )
        self.report = []

    def assert_problematic_nodes(self, culprit_names: Names) -> None:
        # [FUNCTION DOC] HELPER: Asserts the minimizer finds exactly the given node names.
        # Patches the minimizer's run-and-compare step with a side effect that raises
        # FxNetMinimizerResultMismatchError whenever all culprit names are present among the
        # interior nodes of the submodule being checked. Then runs minimize and compares the set
        # of returned node names to the culprit names.
        """
        Quick helper function to assert that a set of nodes (when present together in a subgraph) cause a discrepancy
        """
        with mock.patch("torch.fx.passes.net_min_base._MinimizerBase._run_and_compare"):

            def run_and_compare_side_effect(
                split_module: torch.fx.GraphModule,
                submod_name: str,
                output_names: Names,
                report_idx: int = -1,
            ) -> None:
                # [FUNCTION DOC] HELPER: Mock side effect for the run-and-compare step. Looks up
                # the submodule by name, collects the names of its graph nodes without the first
                # (input) and last (output) node, and raises FxNetMinimizerResultMismatchError
                # if all culprit names are included. Returns nothing otherwise.
                submodule = getattr(split_module, submod_name)

                # Remove input/output layer
                names = set([node.name for node in submodule.graph.nodes][1:-1])
                if set(culprit_names) <= names:
                    raise FxNetMinimizerResultMismatchError

            self.minimizer._run_and_compare.side_effect = run_and_compare_side_effect

            # Every single node should be a discrepancy
            culprits = self.minimizer.minimize()
            self.assertEqual({node.name for node in culprits}, set(culprit_names))

    def test_no_discrepancy(self) -> None:
        # [FUNCTION DOC] With run_a and run_b patched to return the same zero tensor and the
        # compare function reporting a match, minimize must return an empty set of culprits.
        # No discrepancies should handle gracefully with an empty set
        with (
            mock.patch("torch.fx.passes.net_min_base._MinimizerBase.run_a"),
            mock.patch("torch.fx.passes.net_min_base._MinimizerBase.run_b"),
        ):
            # Have both run_a and run_b return the same result
            return_value = torch.zeros((2, 5))
            self.minimizer.run_a.return_value = return_value
            self.minimizer.run_b.return_value = return_value
            self.compare_fn.return_value = (0, True)

            # There should be no discrepancy between the two, and thus we should receive an empty set
            culprits = self.minimizer.minimize()
            self.assertEqual(culprits, set())

    def test_all_nodes_discrepancy(self) -> None:
        # [FUNCTION DOC] All three nodes (linear, linear2, relu) are culprits; the minimizer
        # must report all of them.
        self.assert_problematic_nodes(["linear", "linear2", "relu"])

    def test_first_node_discrepancy(self) -> None:
        # [FUNCTION DOC] Only the first node (linear) causes the mismatch; the minimizer must
        # report just it.
        self.assert_problematic_nodes(["linear"])

    def test_last_node_discrepancy(self) -> None:
        # [FUNCTION DOC] Only the last node (relu) causes the mismatch; the minimizer must
        # report just it.
        self.assert_problematic_nodes(["relu"])

    def test_middle_node_discrepancy(self) -> None:
        # [FUNCTION DOC] Only the middle node (linear2) causes the mismatch; the minimizer must
        # report just it.
        self.assert_problematic_nodes(["linear2"])

    def test_contiguous_partial_discrepancy_end(self) -> None:
        # [FUNCTION DOC] The last two nodes (linear2 and relu) together cause the mismatch; the
        # minimizer must report both.
        self.assert_problematic_nodes(["linear2", "relu"])

    def test_continugous_partial_discrepancy_beginning(self) -> None:
        # [FUNCTION DOC] The first two nodes (linear and linear2) together cause the mismatch;
        # the minimizer must report both. The test name has a typo (continugous).
        self.assert_problematic_nodes(["linear", "linear2"])


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
