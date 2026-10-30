# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT tests for the graph pass that decomposes operators into simpler ops. Covers a
# built-in decomposition (variance) and a user-registered decomposition (square).

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestOpDecompositions(JitTestCase):
    # [CLASS DOC] JitTestCase with tests of the run-decompositions graph pass.
    def test_op_decomposition(self):
        # [FUNCTION DOC] Scripts a variance function, checks the graph contains the variance
        # op, runs the decomposition pass on the graph in place, then checks the scripted
        # output equals eager output on a random 10x10 input and that the variance op is gone
        # from the graph.
        def foo(x):
            # [FUNCTION DOC] HELPER: Eager function returning the unbiased variance of its
            # input tensor.
            return torch.var(x, unbiased=True)

        # TODO: more robust testing
        foo_s = torch.jit.script(foo)
        FileCheck().check("aten::var").run(foo_s.graph)
        torch._C._jit_pass_run_decompositions(foo_s.graph)
        inp = torch.rand([10, 10])
        self.assertEqual(foo(inp), foo_s(inp))
        FileCheck().check_not("aten::var").run(foo_s.graph)

    def test_registered_decomposition(self):
        # [FUNCTION DOC] Registers a custom decomposition (square expressed as a power of two)
        # for the square op, runs the pass on a scripted square function's graph, and checks the
        # square op was replaced by a pow op and that results still equal torch.square on a
        # random vector. Side effect: the registration is global and is not undone by the test.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the elementwise square of
            # its input.
            return torch.square(x)

        @torch.jit.script
        def square_decomp(x):
            # [FUNCTION DOC] HELPER: Scripted function computing the square as a power of two;
            # its graph is the decomposition registered for the square op.
            return torch.pow(x, 2)

        torch.jit._register_decomposition(
            torch.ops.aten.square.default, square_decomp.graph
        )
        torch._C._jit_pass_run_decompositions(foo.graph)
        FileCheck().check_not("aten::square").check("aten::pow").run(foo.graph)
        x = torch.rand([4])
        self.assertEqual(foo(x), torch.square(x))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
