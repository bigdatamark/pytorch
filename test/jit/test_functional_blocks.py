# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the create_functional_graphs JIT pass. It checks that side-effect-free
# tensor operations are grouped into functional subgraph nodes while in-place mutation stays
# outside them. Not runnable directly: running the file as a script raises an error that points
# to the umbrella test runner.

import os
import sys

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestFunctionalBlocks(JitTestCase):
    # [CLASS DOC] JitTestCase with a single test of the functional graph creation pass.
    def test_subgraph_creation(self):
        # [FUNCTION DOC] Scripts a function that mixes pure operations, an in-place add on one
        # tensor and a branch, runs the create_functional_graphs pass on its graph, then uses
        # FileCheck patterns on the printed graph. Verifies that uses of the first two inputs
        # are sunk into a functional graph node, that the functional node is immediately
        # followed by the final addition left outside it, and that the in-place add stays
        # outside the functional graph while the later multiplication is inside it.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Function under test. Adds one to each of three tensors,
            # mutates the third in place, multiplies, branches on a comparison of the second
            # with a constant, and returns the sum of all three. Only scripted, never called.
            x = x + 1
            y = y + 1
            z = z + 1
            z.add_(2)
            z = z * z
            y = y * z
            if y < 2:
                y = y + 5
            return x + y + z

        graph = torch.jit.script(fn).graph
        self.run_pass("create_functional_graphs", graph)

        # all uses of x and y should be sunk
        FileCheck().check(r"%x").check_not(r"%x").check("FunctionalGraph").check(
            r"%x"
        ).run(graph)
        FileCheck().check(r"%y").check_not(r"%y").check("FunctionalGraph").check(
            r"%y"
        ).run(graph)

        # Don't allow any outputs which escape scope, so there is one final addition in the graph
        FileCheck().check("Tensor = prim::Functional").check_next("aten::add").run(
            graph
        )

        # z + 1, z.add_(2) considered non functional, z = z * z should be considered functional
        FileCheck().check("add").check("add_").check_not("mul").check(
            "FunctionalGraph"
        ).run(graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
