# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for calling tensor methods by name, here the getitem special
# method. Importing the module appends the test directory to the system path.

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestTensorMethods(JitTestCase):
    # [CLASS DOC] JitTestCase suite for tensor method calls in TorchScript, covering valid and
    # invalid getitem calls.
    def test_getitem(self):
        # [FUNCTION DOC] Checks that calling getitem on a tensor with an index tensor matches
        # eager results and that the scripted graph contains an aten index node.
        def tensor_getitem(inp: torch.Tensor):
            # [FUNCTION DOC] HELPER: Builds a long tensor of indices zero and two and returns
            # the input indexed by it.
            indices = torch.tensor([0, 2], dtype=torch.long)
            return inp.__getitem__(indices)

        inp = torch.rand(3, 4)
        self.checkScript(tensor_getitem, (inp,))

        scripted = torch.jit.script(tensor_getitem)
        FileCheck().check("aten::index").run(scripted.graph)

    def test_getitem_invalid(self):
        # [FUNCTION DOC] Checks that scripting a call of getitem with no arguments raises a
        # RuntimeError saying exactly one argument is expected, with the method access
        # highlighted.
        def tensor_getitem_invalid(inp: torch.Tensor):
            # [FUNCTION DOC] HELPER: Calls getitem on the input without arguments; never
            # successfully compiled.
            return inp.__getitem__()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "expected exactly 1 argument", "inp.__getitem__"
        ):
            torch.jit.script(tensor_getitem_invalid)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
