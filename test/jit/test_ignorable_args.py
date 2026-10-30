# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests that TorchScript code generation omits trailing arguments equal to schema
# defaults (for slice) and keeps explicit keyword arguments (for the out argument of add). Not
# runnable directly: running the file as a script raises an error that points to the umbrella
# test runner.

import os
import sys

import torch
from torch._C import parse_ir
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests that Python slice class is supported in TorchScript
class TestIgnorableArgs(JitTestCase):
    # [CLASS DOC] JitTestCase for printing and round-tripping of calls whose trailing arguments
    # equal defaults.
    def test_slice_ignorable_args_for_slice(self):
        # [FUNCTION DOC] Parses a textual IR graph that builds a 2 by 6 integer tensor and
        # applies two slice operations (dimension 0 starting at 2, dimension 1 ending at 1, with
        # the other bounds unset or default). Builds a function from it, makes an export/import
        # copy, and uses FileCheck to assert the printed source drops trailing default arguments
        # in the nested slice calls. Also asserts the original and copy return equal results.
        graph_str = """graph():
            %13 : int = prim::Constant[value=0]()
            %10 : bool = prim::Constant[value=0]()
            %8 : NoneType = prim::Constant()
            %0 : int = prim::Constant[value=1]()
            %1 : int = prim::Constant[value=2]()
            %2 : int = prim::Constant[value=3]()
            %3 : int = prim::Constant[value=4]()
            %4 : int = prim::Constant[value=9]()
            %5 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %6 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %7 : int[][] = prim::ListConstruct(%5, %6)
            %val.1 : Tensor = aten::tensor(%7, %8, %8, %10)
            %16 : Tensor = aten::slice(%val.1, %13, %1, %8, %0)
            %20 : Tensor = aten::slice(%16, %0, %8, %0, %0)
            return (%20)"""
        graph = parse_ir(graph_str)
        function = self.createFunctionFromGraph(graph)
        function_copy = self.getExportImportCopy(function)
        src = str(function.code)
        # For a signature:
        # aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
        # We ignore trailing arguments after start=2 for dim 0
        # and after end=1 for dim 1
        # because in %16, %15 and %0 are default values for the schema.
        FileCheck().check(
            "torch.slice(torch.slice(torch.tensor(_0), 0, 2), 1, None, 1)"
        ).run(src)
        self.assertEqual(function(), function_copy())

    def test_add_out_ignorable_args(self):
        # [FUNCTION DOC] Scripts a function calling add with an explicit out argument and checks
        # with FileCheck that the generated source still shows the out keyword argument.
        @torch.jit.script
        def fn(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function adding two tensors, writing the result
            # into the second tensor via the out argument; returns nothing.
            torch.add(x, y, out=y)

        FileCheck().check("torch.add(x, y, out=y)").run(fn.code)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
