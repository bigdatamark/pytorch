# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for the internal directed graph class used by PackageExporter to track
# dependencies: successor and predecessor queries, node attributes, edges, iteration,
# membership, transitive closure and path extraction. Falls back to a non-relative import of the
# shared base test case. Run as a script, it starts the test runner.

from torch.package._digraph import DiGraph
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDiGraph(PackageTestCase):
    # [CLASS DOC] Test case building small graphs and checking the query methods of the
    # graph class.
    """Test the DiGraph structure we use to represent dependencies in PackageExporter"""

    def test_successors(self):
        # [FUNCTION DOC] Checks successors lists both targets of a node with two outgoing edges
        # and is empty for a node added without edges.
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("bar", list(g.successors("foo")))
        self.assertIn("baz", list(g.successors("foo")))
        self.assertEqual(len(list(g.successors("qux"))), 0)

    def test_predecessors(self):
        # [FUNCTION DOC] Checks predecessors lists the source of an edge for each target and is
        # empty for a node added without edges.
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("foo", list(g.predecessors("bar")))
        self.assertIn("foo", list(g.predecessors("baz")))
        self.assertEqual(len(list(g.predecessors("qux"))), 0)

    def test_successor_not_in_graph(self):
        # [FUNCTION DOC] Asking for the successors of an unknown node must raise ValueError.
        g = DiGraph()
        with self.assertRaises(ValueError):
            g.successors("not in graph")

    def test_predecessor_not_in_graph(self):
        # [FUNCTION DOC] Asking for the predecessors of an unknown node must raise ValueError.
        g = DiGraph()
        with self.assertRaises(ValueError):
            g.predecessors("not in graph")

    def test_node_attrs(self):
        # [FUNCTION DOC] Adds a node with two keyword attributes and checks both are readable
        # through the node mapping.
        g = DiGraph()
        g.add_node("foo", my_attr=1, other_attr=2)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)
        self.assertEqual(g.nodes["foo"]["other_attr"], 2)

    def test_node_attr_update(self):
        # [FUNCTION DOC] Adds a node twice with a different value for the same attribute and
        # checks the second value replaces the first.
        g = DiGraph()
        g.add_node("foo", my_attr=1)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)

        g.add_node("foo", my_attr="different")
        self.assertEqual(g.nodes["foo"]["my_attr"], "different")

    def test_edges(self):
        # [FUNCTION DOC] Adds four edges (one in a separate component) and checks the edge
        # collection has exactly four entries, each a source and target pair.
        g = DiGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(1, 3)
        g.add_edge(4, 5)

        edge_list = list(g.edges)
        self.assertEqual(len(edge_list), 4)

        self.assertIn((1, 2), edge_list)
        self.assertIn((2, 3), edge_list)
        self.assertIn((1, 3), edge_list)
        self.assertIn((4, 5), edge_list)

    def test_iter(self):
        # [FUNCTION DOC] Adds three nodes and checks that iterating the graph yields exactly
        # those nodes.
        g = DiGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        nodes = set()
        nodes.update(g)

        self.assertEqual(nodes, {1, 2, 3})

    def test_contains(self):
        # [FUNCTION DOC] Checks the membership test is true for an added node and false for an
        # absent one.
        g = DiGraph()
        g.add_node("yup")

        self.assertTrue("yup" in g)
        self.assertFalse("nup" in g)

    def test_contains_non_hashable(self):
        # [FUNCTION DOC] Checks the membership test with an unhashable list on an empty graph
        # returns False instead of raising.
        g = DiGraph()
        self.assertFalse([1, 2, 3] in g)

    def test_forward_closure(self):
        # [FUNCTION DOC] On a graph with two branches joining at one node, checks the forward
        # transitive closure of two start nodes includes the start node and everything reachable
        # from it.
        g = DiGraph()
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("5", "4")
        g.add_edge("4", "3")
        self.assertTrue(g.forward_transitive_closure("1") == {"1", "2", "3"})
        self.assertTrue(g.forward_transitive_closure("4") == {"4", "3"})

    def test_all_paths(self):
        # [FUNCTION DOC] Builds a graph with two routes from node one to node three plus an
        # unrelated branch, and requests the paths subgraph as dot text. Splits the text on
        # semicolons, drops the header and trailer pieces and strips newlines to remove ordering
        # effects, then expects exactly the five edges that lie on a route from the source to
        # the target.
        g = DiGraph()
        g.add_edge("1", "2")
        g.add_edge("1", "7")
        g.add_edge("7", "8")
        g.add_edge("8", "3")
        g.add_edge("2", "3")
        g.add_edge("5", "4")
        g.add_edge("4", "3")

        result = g.all_paths("1", "3")
        # to get rid of indeterminism
        actual = {i.strip("\n") for i in result.split(";")[2:-1]}
        expected = {
            '"2" -> "3"',
            '"1" -> "7"',
            '"7" -> "8"',
            '"1" -> "2"',
            '"8" -> "3"',
        }
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    run_tests()
