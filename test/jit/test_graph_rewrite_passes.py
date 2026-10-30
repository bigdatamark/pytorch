# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for JIT graph rewrite passes. Currently covers the fuse_linear pass that
# turns matmul with a transposed weight (plus optional in-place bias add) into a single linear
# operation. Not runnable directly: running the file as a script raises an error that points to
# the umbrella test runner.

import torch
import torch._C
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestGraphRewritePasses(JitTestCase):
    # [CLASS DOC] JitTestCase with tests for graph rewrite passes.
    def test_fuse_linear(self):
        # [FUNCTION DOC] Traces a functional linear module with and without bias, applies the
        # fuse_linear pass and checks with FileCheck that a linear operation appears while
        # matmul, addmm, in-place add and transpose operations are gone, that the fused node
        # keeps the source range of the original matmul, and that the traced module still runs.
        # Then traces a module doing a three-dimensional matmul and verifies the pass leaves it
        # unfused (matmul stays, no linear) and it still runs.
        class FunctionalLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing matmul of the input with the transpose of a
            # stored weight and adding the stored bias in place when the bias is not None.
            # Weight and bias are plain tensors kept as attributes. Only traced.
            def __init__(self, weight, bias):
                # [FUNCTION DOC] HELPER: Stores the given weight and bias tensors as attributes.
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns matmul of the input with the transposed weight;
                # when a bias is present, adds it in place to that result first. Returns the
                # result tensor.
                res = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    res.add_(self.bias)
                return res

        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        for has_bias in [True, False]:
            bias = b1 if has_bias else None
            model = torch.jit.trace(FunctionalLinear(w1, bias), [x1])
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    source_range_1 = node.sourceRange()
            torch._C._jit_pass_fuse_linear(model.graph)
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    source_range_2 = node.sourceRange()
            FileCheck().check("aten::linear").run(model.graph)
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            self.assertTrue(source_range_1 == source_range_2)
            # make sure it runs
            model(x1)

        # check matmuls are not fused
        class Matmul(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that multiplies the input by a stored weight
            # with matmul, used to check that the fuse pass does not fuse a non-linear
            # matmul. Only traced.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Stores the given weight tensor as an attribute.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns matmul of the input with the stored weight
                # without transposing it.
                return torch.matmul(x, self.weight)

        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        torch._C._jit_pass_fuse_linear(model.graph)
        # check 3d matmul is not fused
        FileCheck().check("aten::matmul").run(model.graph)
        FileCheck().check_not("aten::linear").run(model.graph)
        # make sure it runs
        model(x)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
