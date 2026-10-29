# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for the activation checkpointing helpers: GraphInfoProvider (graph
# bookkeeping for what can be recomputed), KnapsackEvaluator (peak memory and recompute runtime
# of a knapsack decision), and the dp_knapsack and dp_knapsack_sliding_hirschberg solvers.
# Defines three TestCase classes on small hand-made graphs and tables, plus a run_tests entry
# point. No import-time side effects.
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack import (
    dp_knapsack,
    dp_knapsack_sliding_hirschberg,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)
from torch.fx.graph import Graph
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGraphInfoProvider(TestCase):
    # [CLASS DOC] Tests graph construction and derived graphs of GraphInfoProvider. setUp builds
    # a six-node chain (node1 to output, plus an edge from node1 to output) where node1, node2
    # and node5 are the recomputable banned nodes, each with unit memory and runtime.
    """
    Test class for GraphInfoProvider.
    The test class sets up a small graph example and tests the methods validating the graph building logic.
    """

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Fixture. Stores the node order, the edge list, the three
        # recomputable banned nodes, unit memory and runtime lists, and a GraphInfoProvider
        # built from them.
        super().setUp()
        self.graph_nodes_in_order = [
            "node1",
            "node2",
            "node3",
            "node4",
            "node5",
            "output",
        ]
        self.graph_edges = [
            ("node1", "node2"),
            ("node2", "node3"),
            ("node3", "node4"),
            ("node4", "node5"),
            ("node5", "output"),
            ("node1", "output"),
        ]
        self.all_recomputable_banned_nodes = ["node1", "node2", "node5"]
        self.recorded_knapsack_input_memories = [1.0, 1.0, 1.0]
        self.recorded_knapsack_input_runtimes = [1.0, 1.0, 1.0]
        self.graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )

    def test_inialize_from_graph(self):
        # [FUNCTION DOC] Builds an fx graph by hand with the same shape as the fixture, then
        # calls the provider factory (spelled with a typo in the source) and checks the node
        # order, the edges (compared sorted) and the recomputable banned node names come out as
        # the string names.
        joint_graph = Graph()
        node1 = joint_graph.placeholder("node1")
        node2 = joint_graph.call_function(lambda x: x, (node1,))
        node2.name = "node2"
        node3 = joint_graph.call_function(lambda x: x, (node2,))
        node3.name = "node3"
        node4 = joint_graph.call_function(lambda x: x, (node3,))
        node4.name = "node4"
        node5 = joint_graph.call_function(lambda x: x, (node4,))
        node5.name = "node5"
        output = joint_graph.call_function(lambda x, y: (x, y), (node5, node1))
        output.name = "output"
        all_recomputable_banned_nodes = [node1, node2, node5]
        recorded_knapsack_input_memories = [1.0, 1.0, 1.0]
        recorded_knapsack_input_runtimes = [1.0, 1.0, 1.0]
        graph_info_provider = GraphInfoProvider.inialize_from_graph(
            joint_graph=joint_graph,
            all_recomputable_banned_nodes=all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=recorded_knapsack_input_runtimes,
        )
        self.assertEqual(
            graph_info_provider.graph_nodes_in_order,
            ["node1", "node2", "node3", "node4", "node5", "output"],
        )
        self.assertEqual(
            sorted(graph_info_provider.graph_edges),
            sorted(
                [
                    ("node1", "node2"),
                    ("node2", "node3"),
                    ("node3", "node4"),
                    ("node4", "node5"),
                    ("node5", "output"),
                    ("node1", "output"),
                ]
            ),
        )
        self.assertEqual(
            graph_info_provider.all_recomputable_banned_nodes,
            ["node1", "node2", "node5"],
        )

    def test_get_non_ac_peak_memory(self):
        # [FUNCTION DOC] Checks the non-AC peak memory equals the sum of the recorded memories.
        self.assertEqual(
            self.graph_info_provider.get_non_ac_peak_memory(),
            sum(self.recorded_knapsack_input_memories),
        )

    def test_get_theoretical_max_runtime(self):
        # [FUNCTION DOC] Checks the theoretical max runtime equals the sum of the
        # recorded runtimes.
        self.assertEqual(
            self.graph_info_provider.get_theoretical_max_runtime(),
            sum(self.recorded_knapsack_input_runtimes),
        )

    def test_get_knapsack_memory_input(self):
        # [FUNCTION DOC] Checks the knapsack memory input equals the recorded memory list.
        self.assertEqual(
            self.graph_info_provider.get_knapsack_memory_input(),
            self.recorded_knapsack_input_memories,
        )

    def test_get_knapsack_runtime_input(self):
        # [FUNCTION DOC] Checks the knapsack runtime input equals the recorded runtime list.
        self.assertEqual(
            self.graph_info_provider.get_knapsack_runtime_input(),
            self.recorded_knapsack_input_runtimes,
        )

    def test_recomputable_node_only_graph(self):
        # [FUNCTION DOC] Checks the graph restricted to recomputable nodes has exactly those
        # nodes in order and only the direct edge from node1 to node2.
        recomputable_node_only_graph = (
            self.graph_info_provider.recomputable_node_only_graph
        )
        expected_nodes = self.all_recomputable_banned_nodes
        expected_edges = [("node1", "node2")]
        self.assertEqual(list(recomputable_node_only_graph.nodes), expected_nodes)
        self.assertEqual(
            sorted(recomputable_node_only_graph.edges), sorted(expected_edges)
        )

    def test_recomputable_node_only_graph_with_larger_graph_context(self):
        # [FUNCTION DOC] Checks the recomputable-node graph that keeps the larger graph context:
        # same nodes, and edges node1 to node2 and node2 to node5 (indirect paths count, except
        # those passing through another recomputable node, so no edge from node1 to node5).
        recomputable_node_only_graph_with_larger_graph_context = self.graph_info_provider.recomputable_node_only_graph_with_larger_graph_context
        expected_nodes = self.all_recomputable_banned_nodes
        # node1 does not have an indirect path to node5 because of node2
        # node2 has an indirect path to node5
        expected_edges = [("node1", "node2"), ("node2", "node5")]
        self.assertEqual(
            sorted(recomputable_node_only_graph_with_larger_graph_context.nodes),
            sorted(expected_nodes),
        )
        self.assertEqual(
            sorted(recomputable_node_only_graph_with_larger_graph_context.edges),
            sorted(expected_edges),
        )

    def test_full_joint_nx_graph(self):
        # [FUNCTION DOC] Builds a fresh provider and checks the full joint graph drops the
        # output node and every edge touching it.
        graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )
        full_joint_nx_graph = graph_info_provider.full_joint_nx_graph
        expected_nodes = [
            node for node in self.graph_nodes_in_order if node != "output"
        ]
        expected_edges = [
            (u, v) for u, v in self.graph_edges if u != "output" and v != "output"
        ]
        self.assertEqual(list(full_joint_nx_graph.nodes), expected_nodes)
        self.assertEqual(sorted(full_joint_nx_graph.edges), sorted(expected_edges))

    def test_simplified_fx_joint_graph(self):
        # [FUNCTION DOC] Builds a fresh provider and checks the simplified fx joint graph has
        # the nodes in the original order and, via user links, exactly the original edge set.
        graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )
        simplified_fx_joint_graph = graph_info_provider.simplified_fx_joint_graph
        expected_nodes = self.graph_nodes_in_order
        expected_edges = self.graph_edges
        self.assertEqual(
            [node.name for node in simplified_fx_joint_graph.nodes], expected_nodes
        )
        self.assertEqual(
            sorted(
                [
                    (node.name, user.name)
                    for node in simplified_fx_joint_graph.nodes
                    for user in node.users
                ]
            ),
            sorted(expected_edges),
        )


class TestKnapsackEvaluator(TestCase):
    # [CLASS DOC] Tests KnapsackEvaluator on a six-node graph with recomputable memories 0.1,
    # 0.2, 0.2 and runtimes 100, 50, 51. setUp also installs a fake knapsack algorithm: a lookup
    # table from memory budget to runtime, saved indices and recomputable indices, defaulting to
    # saving everything.
    """
    Test class for KnapsackEvaluator.
    The test class sets up a small graph example and tests the methods validating the knapsack evaluation logic.
    """

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Fixture. Builds the graph data, a provider, an evaluator and a
        # fake knapsack function mapping budgets 0.1 to 0.5 to canned results (unknown budgets
        # save all).
        super().setUp()
        self.graph_nodes_in_order = [
            "node1",
            "node2",
            "node3",
            "node4",
            "node5",
            "output",
        ]
        self.graph_edges = [
            ("node1", "node2"),
            ("node2", "node3"),
            ("node3", "node4"),
            ("node4", "node5"),
            ("node5", "output"),
            ("node1", "output"),
        ]
        self.all_recomputable_banned_nodes = ["node1", "node2", "node5"]
        self.recorded_knapsack_input_memories = [0.1, 0.2, 0.2]
        self.recorded_knapsack_input_runtimes = [100.0, 50.0, 51.0]
        self.graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )
        self.knapsack_evaluator = KnapsackEvaluator(
            graph_info_provider=self.graph_info_provider
        )
        self.knapsack_algo = lambda memory_values, runtime_values, memory_budget: {
            0.1: (101.0, [0], [1, 2]),
            0.2: (101.0, [0], [1, 2]),
            0.3: (50.0, [0, 2], [1]),
            0.4: (50.0, [0, 2], [1]),
            0.5: (0.0, [0, 1, 2], []),
        }.get(memory_budget, (0.0, [0, 1, 2], []))

    def test_evaluate_knapsack_output_not_accounting_for_backward_pass(self):
        # [FUNCTION DOC] Evaluates saving node 0 and recomputing nodes 1 and 2 without backward
        # accounting; expects peak memory 0.1 and recomputation runtime 101.
        saved_nodes_idxs = [0]
        recomputable_node_idxs = [1, 2]
        result = self.knapsack_evaluator.evaluate_knapsack_output(
            saved_nodes_idxs=saved_nodes_idxs,
            recomputable_node_idxs=recomputable_node_idxs,
        )
        self.assertEqual(result["peak_memory"], 0.1)
        self.assertEqual(result["recomputation_runtime"], 101.0)

    def test_evaluate_knapsack_output_accounting_for_backward_pass(self):
        # [FUNCTION DOC] Same choice with backward accounting on; expects peak memory 0.5 and
        # the same recomputation runtime 101.
        saved_nodes_idxs = [0]
        recomputable_node_idxs = [1, 2]
        result = self.knapsack_evaluator.evaluate_knapsack_output(
            saved_nodes_idxs=saved_nodes_idxs,
            recomputable_node_idxs=recomputable_node_idxs,
            account_for_backward_pass=True,
        )
        self.assertEqual(result["peak_memory"], 0.5)
        self.assertEqual(result["recomputation_runtime"], 101.0)

    def test_evaluate_knapsack_output_with_wrong_sized_values(self):
        # [FUNCTION DOC] Passes index lists that do not cover all three recomputable nodes and
        # expects an AssertionError.
        saved_nodes_idxs = [0]
        recomputable_node_idxs = [1]
        with self.assertRaises(AssertionError):
            self.knapsack_evaluator.evaluate_knapsack_output(
                saved_nodes_idxs=saved_nodes_idxs,
                recomputable_node_idxs=recomputable_node_idxs,
            )

    def test_evaluate_distribution_of_results_for_knapsack_algo(self):
        # [FUNCTION DOC] Runs the distribution evaluation over budgets 0.1, 0.2, 0.3 using the
        # fake algorithm; checks one result per budget and spot-checks the budget, peak memory,
        # runtime, non-AC peak memory, theoretical max runtime and the two percentage fields.
        memory_budget_values = [0.1, 0.2, 0.3]
        results = (
            self.knapsack_evaluator.evaluate_distribution_of_results_for_knapsack_algo(
                knapsack_algo=self.knapsack_algo,
                memory_budget_values=memory_budget_values,
            )
        )
        self.assertEqual(len(results), len(memory_budget_values))
        self.assertEqual(results[0]["memory_budget"], 0.1)
        self.assertEqual(results[0]["peak_memory"], 0.1)
        self.assertEqual(results[0]["recomputation_runtime"], 101)
        self.assertEqual(results[1]["non_ac_peak_memory"], 0.5)
        self.assertEqual(results[1]["theoretical_max_runtime"], 201)
        self.assertEqual(results[2]["percentage_of_theoretical_peak_memory"], 0.3 / 0.5)
        self.assertEqual(
            results[2]["percentage_of_theoretical_peak_runtime"], 50.0 / 201
        )

    def test_get_knee_point_memory_budget(self):
        # [FUNCTION DOC] Searches for the knee point of the memory/runtime trade-off for several
        # upper budgets (lower bound 0.1, five iterations) and compares to expected values. Ties
        # between budgets with the same result are resolved arbitrarily, per the inline note.
        """
        Checks if the method correctly estimates the knee point in the memory budget
        where the trade-off between memory usage and recomputation runtime is optimal.

        If memory budget and runtime are considered as equal cost, then the knee point
        is where the distance from 0 is smallest.
        """
        max_mem_budget_to_expected_knee_point = {
            0.1: 0.1,
            0.2: 0.1,
            0.3: 0.3,
            0.4: 0.4,  # 0.3 and 0.4 provide the same algo output so this is arbitrary
            0.5: 0.4,
        }
        for (
            max_mem_budget,
            expected_knee_point,
        ) in max_mem_budget_to_expected_knee_point.items():
            knee_point_memory_budget = (
                self.knapsack_evaluator.get_knee_point_memory_budget(
                    knapsack_algo=self.knapsack_algo,
                    max_mem_budget=max_mem_budget,
                    min_mem_budget=0.1,
                    iterations=5,
                )
            )
            self.assertEqual(knee_point_memory_budget, expected_knee_point)

    def test_get_backward_memory_from_topologically_sorted_graph(self):
        # [FUNCTION DOC] Calls a private evaluator method that walks the larger-context graph
        # recomputing and dropping nodes with only node1 saved, starting from 0.1. Prints the
        # result and the expected list, then compares memory values (approximately) and event
        # labels pairwise. Pairs are compared with a zip, so a shorter result would not fail
        # the test.
        result = self.knapsack_evaluator._get_backward_memory_from_topologically_sorted_graph(
            node_graph=self.graph_info_provider.recomputable_node_only_graph_with_larger_graph_context,
            node_memories=self.graph_info_provider.all_node_memories,
            saved_nodes_set={"node1"},
            peak_memory_after_forward_pass=0.1,
        )
        expected_result = [
            (0.1, "Initial Peak/Current Memory"),
            (0.3, "Recomputing Node: node5"),
            (0.5, "Recomputing Predecessor of node5: node2"),
            (0.3, "Dropping Node: node5"),
            (0.1, "Dropping Node(already saved): node2"),
            (0.0, "Dropping Node(already saved): node1"),
        ]
        print(result, expected_result)
        for result_item, expected_result_item in zip(result, expected_result):
            self.assertAlmostEqual(result_item[0], expected_result_item[0])
            self.assertEqual(result_item[1], expected_result_item[1])


class TestActivationCheckpointingKnapsack(TestCase):
    # [CLASS DOC] Tests the two dynamic programming knapsack solvers on a shared table of cases
    # (memory, runtime, budget, expected runtime, saved indices, recomputable indices),
    # including empty input, zero budget and zero costs.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture. Builds the list of thirteen test case tuples shared by
        # both solver tests.
        super().setUp()
        # (memory, runtime, max_memory, expected_runtime, expected_saved, expected_recomputable)
        self.test_cases = [
            ([2, 3, 2, 4, 1], [1, 2, 1, 3, 2], 5, 5.0, [3, 4], [2, 1, 0]),
            ([1, 1, 1], [1, 2, 3], 3, 6.0, [0, 1, 2], []),
            ([10, 20, 30], [1, 2, 3], 5, 0.0, [], [2, 1, 0]),
            ([1, 2, 3], [10, 20, 30], 1, 10.0, [0], [2, 1]),
            ([1, 1, 1], [2, 2, 2], 2, 4.0, [0, 1], [2]),
            ([0, 2, 3], [5, 2, 3], 5, 10.0, [0, 1, 2], []),
            ([1, 2, 3], [0, 2, 3], 3, 3.0, [2], [0, 1]),
            ([100, 200, 300], [1000, 2000, 3000], 500, 5000.0, [1, 2], [0]),
            ([0.5, 1.5, 2.0], [1.0, 2.0, 3.0], 2.0, 3.0, [1, 0], [2]),
            ([], [], 10, 0.0, [], []),
            ([1, 2, 3], [1, 2, 3], 0, 0.0, [], [2, 1, 0]),
            ([0, 0, 0], [1, 2, 3], 0, 6.0, [0, 1, 2], []),
            ([1, 2, 3], [0, 0, 0], 6, 0.0, [], [2, 1, 0]),
        ]

    def _run_knapsack_and_check(
        self,
        func,
        memory,
        runtime,
        max_memory,
        expected_runtime,
        expected_saved,
        expected_recomputable,
    ):
        # [FUNCTION DOC] HELPER: Calls the given solver with memory, runtime and budget, and
        # checks the returned runtime exactly, and the saved and recomputable index lists
        # after sorting.
        result_runtime, result_saved, result_recomputable = func(
            memory, runtime, max_memory
        )
        self.assertEqual(result_runtime, expected_runtime)
        self.assertEqual(sorted(result_saved), sorted(expected_saved))
        self.assertEqual(sorted(result_recomputable), sorted(expected_recomputable))

    def test_dp_knapsack(self):
        # [FUNCTION DOC] Runs the plain knapsack solver on every shared case, one subtest
        # per case.
        for i, (
            memory,
            runtime,
            max_memory,
            expected_runtime,
            expected_saved,
            expected_recomputable,
        ) in enumerate(self.test_cases):
            with self.subTest(f"dp_knapsack_case_{i}"):
                self._run_knapsack_and_check(
                    dp_knapsack,
                    memory,
                    runtime,
                    max_memory,
                    expected_runtime,
                    expected_saved,
                    expected_recomputable,
                )

    def test_dp_knapsack_sliding_hirschberg(self):
        # [FUNCTION DOC] Runs the sliding Hirschberg knapsack solver on every shared case, one
        # subtest per case.
        for i, (
            memory,
            runtime,
            max_memory,
            expected_runtime,
            expected_saved,
            expected_recomputable,
        ) in enumerate(self.test_cases):
            with self.subTest(f"dp_knapsack_sliding_hirschberg_case_{i}"):
                self._run_knapsack_and_check(
                    dp_knapsack_sliding_hirschberg,
                    memory,
                    runtime,
                    max_memory,
                    expected_runtime,
                    expected_saved,
                    expected_recomputable,
                )


if __name__ == "__main__":
    run_tests()
