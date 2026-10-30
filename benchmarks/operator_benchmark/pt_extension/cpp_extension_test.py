# [MODULE DOC] Unit tests checking that the custom consume operator from the compiled benchmark
# cpp extension survives tracing: traced graphs must keep every consumed computation. Importing
# the extension module registers the operator namespace used by the tests.
import unittest

import benchmark_cpp_extension  # noqa: F401

import torch


class TestConsumeOp(unittest.TestCase):
    # [CLASS DOC] Tests for the operator_benchmark consume custom op under torch.jit.trace.
    def test_jit_consume_op(self):
        # [FUNCTION DOC] Traces a function that consumes the sum of a tensor six times in a
        # loop, then checks the graph contains six sum nodes and that the traced function
        # returns the plain sum.
        iters = 6

        def foo(x):
            # [FUNCTION DOC] HELPER: Loops six times feeding the sum of x into the consume op;
            # returns the last consume result.
            for i in range(iters):
                result = torch.ops.operator_benchmark._consume(torch.sum(x))
            return result

        r = torch.jit.trace(foo, (torch.rand(2, 2)))

        graph = str(r.graph)
        occurrence = graph.count("aten::sum")

        x = torch.rand(2, 2)
        value = r(x)
        self.assertEqual(value, torch.sum(x))
        self.assertEqual(occurrence, iters)

    def test_jit_consume_op_for_list_input(self):
        # [FUNCTION DOC] Same as the previous test but consumes a list of tensors (chunk of the
        # input): checks six chunk nodes in the traced graph and that the output matches chunk
        # elementwise.
        iters = 6

        def foo(x):
            # [FUNCTION DOC] HELPER: Loops six times feeding the chunks of x into the consume
            # op; returns the last consume result.
            for i in range(iters):
                result = torch.ops.operator_benchmark._consume(torch.chunk(x, 2))
            return result

        r = torch.jit.trace(foo, torch.rand(2, 2))

        graph = str(r.graph)
        occurrence = graph.count("aten::chunk")

        x = torch.rand(2, 2)
        value = r(x)

        self.assertTrue(
            all(torch.allclose(t1, t2) for t1, t2 in zip(value, torch.chunk(x, 2)))
        )
        self.assertEqual(occurrence, iters)
