# Owner(s): ["oncall: quantization"]
# Copied from pytorch/test/fx/test_subgraph_rewriter.py
# [MODULE DOC] Quantization-suite copy of the FX subgraph rewriter tests. Defines one
# JitTestCase subclass whose tests trace small modules or functions with symbolic_trace, call
# replace_pattern from torch.fx.subgraph_rewriter, lint the graph and compare outputs against a
# hand-written reference. Importing appends the parent test directory to the system path.
# Running the file directly raises a RuntimeError pointing to the test_fx entry.

import os
import sys

import torch
from torch.fx import symbolic_trace, subgraph_rewriter
from torch.fx.annotate import annotate
# Make the helper files in test/ importable
from torch.fx.experimental.rewriter import RewritingTracer

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_fx.py TESTNAME\n\n"
                       "instead.")

class TestSubgraphRewriter(JitTestCase):
    # [CLASS DOC] Collection of tests for replace_pattern. Each test builds a model, traces it
    # into a GraphModule, rewrites it with a pattern and replacement, then checks graph validity
    # and numerical equality with a traced reference. No state is kept between tests.

    def test_subgraph_rewriter_preserves_logic(self):
        # [FUNCTION DOC] Rewrites a graph using a pattern identical to the replacement and
        # verifies the graph still lints and gives the same outputs as the reference on a random
        # 1x3 input; the rewrite must be a no-op in effect.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing negation plus relu of the input and
            # returning the sum added to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns val added to itself, where val is negation of x
                # plus relu of x.
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: negation of x plus relu of x. Used as both
            # pattern and replacement.
            return torch.neg(x) + torch.relu(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference computation equal to the model, traced to produce
            # expected outputs.
            val = torch.neg(x) + torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        # Replace `pattern` with the same pattern (shouldn't change
        # the underlying logic)
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        # [FUNCTION DOC] Replaces a one-node pattern (negation) with relu in a graph where the
        # negation result is used twice; checks the rewritten module equals a reference that
        # uses relu in its place.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model negating the input and returning the result added to
            # itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes negation of x and returns it added to itself.
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: One-operator pattern: negation of x.
            return torch.neg(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: relu of x.
            return torch.relu(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference with relu substituted for negation, added to
            # itself.
            val = torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_single_pattern_match(self):
        # [FUNCTION DOC] Replaces a two-operator pattern (negation plus relu) matching exactly
        # once with a single relu and compares against the reference graph outputs after
        # linting.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing negation plus relu and returning the sum
            # added to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes negation of x plus relu of x and returns it
                # added to itself.
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: negation of x plus relu of x.
            return torch.neg(x) + torch.relu(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: relu of x.
            return torch.relu(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference: relu of x added to itself.
            val = torch.relu(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_output = comparison_fn(x)
        test_output = traced.forward(x)
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        # [FUNCTION DOC] Checks that all non-overlapping matches are replaced: two concatenate-
        # then-sum computations on w1 and w2 are each rewritten to a stack, and outputs on
        # random 1x3 inputs match the reference.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two identical concatenate-and-sum subgraphs on
            # w1 and w2, combined with x using the max of each.
            def forward(self, x, w1, w2):
                # [FUNCTION DOC] HELPER: Returns x plus max of m1 plus max of m2, where m1 and
                # m2 are concatenate-then-sum results.
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            # [FUNCTION DOC] HELPER: Pattern graph: concatenate w1 and w2 then sum.
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            # [FUNCTION DOC] HELPER: Replacement graph: stack of w1 and w2.
            return torch.stack([w1, w2])

        def comparison(x, w1, w2):
            # [FUNCTION DOC] HELPER: Reference with each concatenate-and-sum replaced by a
            # stack.
            m1 = torch.stack([w1, w2])
            m2 = torch.stack([w1, w2])
            return x + torch.max(m1) + torch.max(m2)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, w1, w2)
        test_outs = traced.forward(x, w1, w2)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_graph_argument_order(self):
        # [FUNCTION DOC] Uses a matrix multiply pattern as both pattern and replacement and
        # checks that argument order is preserved, so the rewritten module equals the reference
        # on a 3x4 and a 4x5 input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning the matrix product of x and y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and y.
                return torch.mm(x, y)

        def pattern(x, y):
            # [FUNCTION DOC] HELPER: Pattern graph: matrix product of x and y, also used as
            # replacement.
            return torch.mm(x, y)

        def comparison(x, y):
            # [FUNCTION DOC] HELPER: Reference matrix product of x and y.
            return torch.mm(x, y)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)
        y = torch.randn(4, 5)

        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_correct_output_replacement(self):
        # [FUNCTION DOC] Replaces relu applied to x with negation where the matched node's
        # output feeds a later addition with another term; checks the rewritten output is wired
        # to the right consumer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding negation of y to relu of x and returning the
            # sum added to itself.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes val from y and x as described and returns val
                # added to itself.
                val = torch.neg(y) + torch.relu(x)
                return torch.add(val, val)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: relu of x.
            return torch.relu(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: negation of x.
            return torch.neg(x)

        def comparison(x, y):
            # [FUNCTION DOC] HELPER: Reference with negation in place of relu on x.
            val = torch.neg(y) + torch.neg(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x, y)
        test_outs = traced.forward(x, y)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_traced_as_callable(self):
        # [FUNCTION DOC] Passes already symbolically traced Module instances as pattern and
        # replacement (rather than plain functions) and checks that sigmoid replaces the
        # negation plus relu subgraph.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing negation plus relu and returning the sum
            # added to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes negation of x plus relu of x and returns it
                # added to itself.
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        class Pattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Pattern module with no state, computing negation plus relu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns negation of x plus relu of x.
                return torch.neg(x) + torch.relu(x)

        class Replacement(torch.nn.Module):
            # [CLASS DOC] HELPER: Replacement module with no state, computing sigmoid.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sigmoid of x.
                return torch.sigmoid(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference: sigmoid of x added to itself.
            val = torch.sigmoid(x)
            return torch.add(val, val)

        traced = symbolic_trace(M())
        traced_pattern = symbolic_trace(Pattern())
        traced_replacement = symbolic_trace(Replacement())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, traced_pattern, traced_replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_is_entire_graph(self):
        # [FUNCTION DOC] Uses a pattern covering the whole graph (negation then adding the
        # result to itself) and replaces it with sigmoid followed by concatenation; the traced
        # replacement serves as reference.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model negating x and returning the result added to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes negation of x and returns it added to itself.
                a = torch.neg(x)
                return torch.add(a, a)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: negation of x added to itself.
            a = torch.neg(x)
            return torch.add(a, a)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: concatenation of the sigmoid of x with
            # itself.
            a = torch.sigmoid(x)
            return torch.cat([a, a])

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(replacement)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(self):
        # [FUNCTION DOC] Verifies that the output node of a matched pattern may have users
        # outside the match: relu is replaced by sigmoid although the relu result is consumed
        # twice later.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing relu of x then negation of it minus itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns negation of y minus y, where y is relu of x.
                y = torch.relu(x)
                return torch.neg(y) - y

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: relu of x.
            return torch.relu(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: sigmoid of x.
            return torch.sigmoid(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference with sigmoid in place of relu.
            y = torch.sigmoid(x)
            return torch.neg(y) - y

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(self):
        # [FUNCTION DOC] Verifies that a pattern whose internal (non-output) nodes have users
        # outside the match is not applied: replace_pattern must return an empty list of
        # matches.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with several concatenate nodes and addmm calls whose
            # intermediate results are reused outside the matching region.
            def forward(self, x, w1, w2, b1, b2):
                # [FUNCTION DOC] HELPER: Builds concatenations, addmm results and sums,
                # returning two scalar sums.
                m0 = torch.cat([w1, w2])  # noqa: F841
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t0 = torch.addmm(b1, m1, m2.t())  # noqa: F841
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.sum(t1), torch.sum(t2)

        def pattern(x, w1, w2, b1, b2):
            # [FUNCTION DOC] HELPER: Pattern graph: two concatenations, then addmm with a
            # transposed second operand.
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        def replacement(x, w1, w2, b1, b2):
            # [FUNCTION DOC] HELPER: Replacement graph: concatenation of x, w1 and w2.
            return torch.cat([x, w1, w2])

        traced = symbolic_trace(M())

        # Result should be [] since no matches can be found
        res = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        self.assertEqual(res, [])

    def test_subgraph_rewriter_placeholder_matching(self):
        # [FUNCTION DOC] Checks that a pattern placeholder may match a graph node that itself
        # has inputs (an addition), so a dequantize, sigmoid, cast-to-float16 chain is removed
        # and only the add of 3 remains; compares with the reference.
        """
        This tests that a placeholder Node can be matched to a Node with
        a different number of input Nodes. In the example below, the
        original traced Module looks like this:
            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_function  <built-in function add>                                     (x, 3)                    {}
            call_method    dequantize                                                  (add,)                    {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, torch.float16)  {}
            output         output                                                      (to,)                     {}
        while the pattern we want to match looks like this:
            opcode         target                                                      args                      kwargs
            -------------  ----------------------------------------------------------  ------------------------  --------
            placeholder    x                                                           ()                        {}
            call_method    dequantize                                                  (x,)                      {}
            call_function  <built-in method sigmoid of type object at 0x7f7c1f440fe0>  (dequantize,)             {}
            call_method    to                                                          (sigmoid, torch.float16)  {}
            output         output                                                      (to,)                     {}
        Here, we want to be able to match the original graph's
        `call_function.add` Node with the pattern graph's
        `plaeholder.x` Node.
        Credit to Jerry Zhang (GitHub: jerryzh168) for this test case
        """
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that adds 3 in place, dequantizes, applies sigmoid
            # and casts to a dtype read from an attribute set to float16.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores float16 in the dtype attribute.
                super().__init__()
                self.dtype = torch.float16

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the addition, dequantize, sigmoid and cast
                # chain to x and returns it.
                x += 3
                x = x.dequantize()
                x = torch.sigmoid(x)
                dtype = self.dtype
                x = x.to(dtype)
                return x

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: dequantize, sigmoid, cast to float16.
            x = x.dequantize()
            x = torch.sigmoid(x)
            x = x.to(torch.float16)
            return x

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: returns x unchanged.
            return x

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference: x plus 3.
            return x + 3

        traced = symbolic_trace(M())
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_replaces_referenced_submodules(self):
        # [FUNCTION DOC] Verifies that submodules referenced by the replacement are added to the
        # rewritten GraphModule and unreferenced ones (sigmoid) are removed, while the shared
        # relu submodule is kept as a ReLU; also compares outputs against a reference module.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with sigmoid and relu submodules, adding 1 to the
            # input and applying sigmoid then relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sigmoid and relu submodules as attributes.
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 1 to x, then applies sigmoid and relu submodules.
                x = x + 1
                return self.submod(self.sigmoid(x))

        class Pattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Pattern module holding sigmoid and relu submodules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates sigmoid and relu submodules.
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies sigmoid then relu submodule to x.
                return self.submod(self.sigmoid(x))

        class Replacement(torch.nn.Module):
            # [CLASS DOC] HELPER: Replacement module holding an identity submodule and a relu
            # submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates identity and relu submodules.
                super().__init__()
                self.id = torch.nn.Identity()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies identity then relu submodule to x.
                return self.submod(self.id(x))

        class Comparison(torch.nn.Module):
            # [CLASS DOC] HELPER: Reference module equal to the expected rewritten model.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates identity and relu submodules.
                super().__init__()
                self.id = torch.nn.Identity()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 1 to x then applies identity and relu submodules.
                x = x + 1
                return self.submod(self.id(x))

        traced = symbolic_trace(M())
        comparison = Comparison()

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        traced.graph.lint()

        ref_outs = comparison(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)

        traced.get_submodule("id")
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            traced.get_submodule("sigmoid")

        submod = traced.get_submodule("submod")
        self.assertEqual(type(submod), torch.nn.ReLU)

    def test_subgraph_rewriter_annotations_int(self):
        # [FUNCTION DOC] Checks that an integer-annotated local variable traced by
        # RewritingTracer and the annotate helper traced with symbolic_trace both give the
        # placeholder node type int; raises AssertionError if either differs.

        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Model using a native annotation on a local variable, traced
            # via the AST rewriting tracer.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Assigns x to an int-annotated local and returns the sum
                # of x and it.
                y: int = x
                return torch.add(x, y)

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Model using the annotate helper to mark a value as int.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Annotates x as int and returns the sum of x and the
                # annotated value.
                y = annotate(x, int)
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M1())

        module = M2()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        for n, m in zip(symbolic_traced.graph.nodes, graph.nodes):
            if n.op == 'placeholder':
                if n.type is not int:
                    raise AssertionError(f"Expected n.type to be int, got {n.type}")
                if m.type is not int:
                    raise AssertionError(f"Expected m.type to be int, got {m.type}")

    def test_subgraph_writer_replace_consecutive_submodules(self):
        # [FUNCTION DOC] Rewrites three consecutive sigmoid calls to exp each, verifying that
        # each of the chained matches is replaced and results equal the reference.

        def f(x):
            # [FUNCTION DOC] HELPER: Applies sigmoid three times in a row.
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern graph: sigmoid of x.
            return torch.sigmoid(x)

        def replacement(x):
            # [FUNCTION DOC] HELPER: Replacement graph: exp of x.
            return torch.exp(x)

        def comparison(x):
            # [FUNCTION DOC] HELPER: Reference applying exp three times.
            x = torch.exp(x)
            x = torch.exp(x)
            return torch.exp(x)

        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        x = torch.randn(3, 4)

        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        traced.graph.lint()

        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)
        self.assertEqual(ref_outs, test_outs)
