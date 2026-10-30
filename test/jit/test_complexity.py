# Owner(s): ["oncall: jit"]

# [MODULE DOC] Informational test of how much control flow and non-tensor work remains in
# optimized TorchScript graphs for generated nn.functional and nn.Module tests.
# Defines two graph-statistic helpers and one TestCase; it only prints statistics and
# asserts nothing. Run directly, the file raises RuntimeError because it is not wired
# into discover_tests. No import-time side effects besides extending sys.path with the
# test directory.
import contextlib
import os
import sys
import unittest

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    set_default_dtype,
    suppress_warnings,
)
from torch.testing._internal.jit_metaprogramming_utils import (
    get_all_nn_module_tests,
    get_nn_functional_compiled_fn_and_inputs,
    get_nn_functional_tests,
    get_nn_mod_test_name,
    try_get_nn_module_compiled_mod_and_inputs,
)
from torch.testing._internal.jit_utils import enable_profiling_mode, JitTestCase


def num_ifs_loops(graph):
    # [FUNCTION DOC] HELPER: Count loop and if nodes in the body of a graph. Converts the graph
    # to text, keeps only the text before the first occurrence of the word return, and returns
    # the number of Loop plus If node mentions there. If no return is found the slice silently
    # drops the last character instead, which is harmless for counting.
    graph_str = str(graph)
    # only look at body of graph
    graph_body = graph_str[0 : graph_str.find("return")]
    return graph_body.count("prim::Loop") + graph_body.count("prim::If")


def num_non_tensor_nodes(block):
    # [FUNCTION DOC] HELPER: Recursively count nodes in a block that produce no Tensor output.
    # Skips constants, bailout nodes and attribute getters (kind substring match). For every
    # other node it first adds the counts from its nested blocks, then adds one if none of its
    # outputs has a type whose text mentions Tensor (a node with no outputs counts as one).
    # Returns the integer total.
    num_non_tensor = 0
    for node in block.nodes():
        kind = node.kind()
        # GetAttr don't provide useful signal here, since they are non-optimizable except with freezing
        # Constant is not executed, bailouts should be a separate tests, don't provide useful signal here
        if kind == "prim::Constant" or "prim::Bailout" in kind or "GetAttr" in kind:
            continue
        for b in node.blocks():
            num_non_tensor += num_non_tensor_nodes(b)
        tensor_out = False
        for out in node.outputs():
            if "Tensor" in str(out.type()):
                tensor_out = True
                break
        num_non_tensor += int(not tensor_out)
    return num_non_tensor


class TestComplexity(JitTestCase):
    # [CLASS DOC] JitTestCase that prints, per generated nn.functional test and per nn.Module
    # test, the number of if/loop nodes and non-tensor nodes in the last optimized graph.
    # Gradient mode is disabled and the default dtype is double for every test.
    def setUp(self):
        # [FUNCTION DOC] Remember whether grad mode was enabled, then disable grad mode and enter
        # the set_default_dtype context (double) through an ExitStack kept on the instance.
        super().setUp()
        self.grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(set_default_dtype(torch.double))

    def tearDown(self):
        # [FUNCTION DOC] Undo setUp: close the ExitStack (restoring the default dtype), restore the
        # saved grad mode, then run the base class tearDown.
        self._stack.close()
        torch.set_grad_enabled(self.grad_enabled)
        super().tearDown()

    @suppress_warnings
    def test_generated_functional_tests(self):
        # [FUNCTION DOC] Under profiling mode, for every generated nn.functional test spec
        # build the scripted function and inputs, run it six times so the profiling executor
        # optimizes it, then record the test name with if/loop and non-tensor counts of the
        # last executed optimized graph. Prints a header row and one row per test; there are
        # no assertions. Warnings are suppressed by the decorator.
        with enable_profiling_mode():
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            for test in get_nn_functional_tests():
                test_name = test[0]

                fn, inputs = get_nn_functional_compiled_fn_and_inputs(*test)
                for _ in range(6):
                    fn(*inputs)

                g = torch.jit.last_executed_optimized_graph()
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))
        for line in stats:
            print(line)

    @suppress_warnings
    @unittest.skipIf(IS_FBCODE, "Causes a RecursionError in fbcode")
    def test_nn_module_tests(self):
        # [FUNCTION DOC] Same statistics as the functional variant, but over all generated
        # nn.Module tests. Specs for which no compiled module can be built are skipped. Each
        # module is run six times in profiling mode before reading the last optimized graph;
        # rows are printed and nothing is asserted. Skipped under fbcode because it hits a
        # RecursionError there.
        with enable_profiling_mode():
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            for test in get_all_nn_module_tests():
                out = try_get_nn_module_compiled_mod_and_inputs(**test)
                if not out:
                    continue

                mod, inputs = out
                test_name = get_nn_mod_test_name(**test)
                for _ in range(6):
                    mod(*inputs)

                g = torch.jit.last_executed_optimized_graph()
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))

            for line in stats:
                print(line)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
