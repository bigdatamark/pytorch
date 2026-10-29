# Owner(s): ["module: fx"]

# [MODULE DOC] Tests for GraphTransformObserver (torch.fx.passes.graph_transform_observer): it
# records created and erased nodes during graph rewrites, writes dot files of input and output
# graphs to a log directory, attaches NodeSource provenance to node metadata, and removes its
# hooks on exit. The main guard raises RuntimeError since the file is not run directly.
import copy
import os
import tempfile

import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.traceback import NodeSourceAction
from torch.testing._internal.common_utils import TestCase


class TestGraphTransformObserver(TestCase):
    # [CLASS DOC] Tests the GraphTransformObserver context manager around pattern replacement and
    # manual graph edits, covering node tracking, dot file output, hooks cleanup and deepcopy.
    def test_graph_transform_observer(self):
        # [FUNCTION DOC] Replaces a negation pattern with relu inside the observer using a
        # temporary log directory. Asserts the observer lists relu as created and neg as erased,
        # and that dot files for the input and output graphs named after the current pass count
        # exist in the log directory. Side effect: creates a temporary directory that is not
        # removed.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that negates its input and adds the result to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the negation of x added to itself.
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern for the rewrite: negation of x.
            return torch.neg(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement for the rewrite: relu of x.
            return torch.relu(x)

        traced = symbolic_trace(M())

        log_url = tempfile.mkdtemp()

        with GraphTransformObserver(
            traced, "replace_neg_with_relu", log_url=log_url
        ) as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        current_pass_count = GraphTransformObserver.get_current_pass_count()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_input_graph.dot",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_output_graph.dot",
                )
            )
        )

    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_graph_transform_observer_node_tracking(self):
        # [FUNCTION DOC] Runs with provenance tracking level 1. Applies two successive rewrites
        # (neg to relu, then relu to cos) under the observer and checks that created and erased
        # nodes are reported, that all hook lists on the traced module are empty after exit, and
        # that from_node provenance chains hold the replaced node with the replace and create
        # actions, the pass name and graph id. Then repeats on an exported module where a manually
        # added add node replaces all uses of neg, checking its provenance names neg and the pass
        # test.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that negates its input and adds the result to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the negation of x added to itself.
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern for the rewrite: negation of x.
            return torch.neg(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: First replacement: relu of x.
            return torch.relu(x)

        def replacement2(x):
            # [FUNCTION DOC] HELPER: Second replacement: cosine of x.
            return torch.cos(x)

        traced = symbolic_trace(M())

        def check_node_source(node_source, node_name, target, id, pass_name, action):
            # [FUNCTION DOC] HELPER: Asserts that a NodeSource has the expected name, target, pass
            # name, graph id and action.
            self.assertEqual(node_source.name, node_name)
            self.assertEqual(node_source.target, target)
            self.assertEqual(node_source.pass_name, pass_name)
            self.assertEqual(node_source.graph_id, id)
            self.assertEqual(node_source.action, action)

        with GraphTransformObserver(traced, "replace_neg_with_relu") as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        self.assertEqual(len(traced._replace_hooks), 0)
        self.assertEqual(len(traced._create_node_hooks), 0)
        self.assertEqual(len(traced._erase_node_hooks), 0)
        self.assertEqual(len(traced._deepcopy_hooks), 0)

        for node in traced.graph.nodes:
            if node.name == "relu":
                from_node = node.meta["from_node"]
                self.assertTrue(len(from_node) == 1)
                check_node_source(
                    from_node[0],
                    "neg",
                    str(torch.neg),
                    id(traced.graph),
                    "replace_neg_with_relu",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )

        with GraphTransformObserver(traced, "replace_relu_with_cos") as ob:
            subgraph_rewriter.replace_pattern(traced, replacement, replacement2)

            self.assertTrue("cos" in ob.created_nodes)
            self.assertTrue("relu" in ob.erased_nodes)

        for node in traced.graph.nodes:
            if node.name == "cos":
                from_node = node.meta["from_node"]
                self.assertTrue(len(from_node) == 1)
                check_node_source(
                    from_node[0],
                    "relu",
                    str(torch.relu),
                    id(traced.graph),
                    "replace_relu_with_cos",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )
                check_node_source(
                    from_node[0].from_node[0],
                    "neg",
                    str(torch.neg),
                    id(traced.graph),
                    "replace_neg_with_relu",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )

        class SimpleLinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns the negation of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the negation of x.
                return torch.neg(x)

        model = SimpleLinearModel()
        gm = torch.export.export(model, (torch.rand(10),), strict=True).module()

        with GraphTransformObserver(gm, "test"):
            add_node = gm.graph.call_function(torch.ops.aten.add.default, (1, 1))
            neg_node = next(
                iter([node for node in gm.graph.nodes if node.name == "neg"])
            )
            neg_node.replace_all_uses_with(replace_with=add_node)

        from_node = add_node.meta["from_node"]
        self.assertTrue(len(from_node) == 1)
        check_node_source(
            from_node[0],
            "neg",
            str(torch.ops.aten.neg.default),
            id(gm.graph),
            "test",
            [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
        )

    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_graph_transform_observer_deepcopy(self):
        # [FUNCTION DOC] Runs with provenance tracking level 1. Deep copies an exported graph
        # module inside the observer and checks the copy has the same node names and that none of
        # the replace, create-node, erase-node or deepcopy hook lists remain on the copy after the
        # context exits.
        class SimpleLinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns the negation of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the negation of x.
                return torch.neg(x)

        model = SimpleLinearModel()
        gm = torch.export.export(model, (torch.rand(10),), strict=True).module()

        with GraphTransformObserver(gm, "test"):
            gm2 = copy.deepcopy(gm)

        nodes = [node.name for node in gm.graph.nodes]
        nodes2 = [node.name for node in gm2.graph.nodes]
        self.assertEqual(nodes, nodes2)

        # deepcopied graph modules should not have hooks after exiting
        # the context
        self.assertEqual(len(gm2._replace_hooks), 0)
        self.assertEqual(len(gm2._create_node_hooks), 0)
        self.assertEqual(len(gm2._erase_node_hooks), 0)
        self.assertEqual(len(gm2._deepcopy_hooks), 0)

    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_graph_transform_observer_replace(self):
        # [FUNCTION DOC] Runs with provenance tracking level 1. Inside the observer, replaces all
        # uses of the add node of a traced module with a newly created aten add node renamed
        # new_add, then checks the new node has exactly one from_node entry naming the original
        # add node with pass name test, i.e. the replaced node is not duplicated in the
        # provenance.
        # the node should should not be duplicated
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing a sum with one, then returning two different
            # products of it (times two and times three).
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to x and returns a tuple of the sum times two and
                # times three.
                y = x + 1
                z = y * 2
                w = y * 3
                return z, w

        model = Model()
        gm = symbolic_trace(model)

        with GraphTransformObserver(gm, "test"):
            for node in gm.graph.nodes:
                if node.name == "add":
                    new_node = gm.graph.call_function(
                        torch.ops.aten.add.Tensor, (node.args[0], node.args[1])
                    )
                    node.replace_all_uses_with(new_node)
                    new_node.name = "new_add"

        self.assertEqual(len(new_node.meta["from_node"]), 1)
        self.assertEqual(new_node.meta["from_node"][0].name, "add")
        self.assertEqual(new_node.meta["from_node"][0].pass_name, "test")


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
