# Owner(s): ["module: inductor"]
# [MODULE DOC] Small test of pad_mm helper utilities. No import-time side effects.
import operator

import torch
from torch._inductor.fx_passes.pad_mm import get_non_view_def
from torch._inductor.test_case import run_tests, TestCase


class PadMMUtilsTest(TestCase):
    # [CLASS DOC] Tests for helper utilities of the pad_mm pass.
    def test_get_non_view_def_traverses_getitem(self):
        # [FUNCTION DOC] Builds a graph with a placeholder, a sort call and a getitem of its
        # first result, and asserts the non-view definition lookup passes through the getitem
        # and returns the sort node.
        graph = torch.fx.Graph()
        arg = graph.placeholder("arg")
        sort = graph.call_function(torch.ops.aten.sort.default, (arg,))
        getitem = graph.call_function(operator.getitem, (sort, 0))

        self.assertIs(get_non_view_def(getitem), sort)


if __name__ == "__main__":
    run_tests()
