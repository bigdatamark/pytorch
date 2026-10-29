# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for the FX subgraph matcher utilities: SubgraphMatcher and
# SubgraphMatcherWithNameNodeMap from torch.fx.passes.utils. Covers matching with get_attr
# nodes, list arguments, literal ignoring, variadic default arguments and name-to-node maps from
# exported patterns.
# The module is not run by the normal test discovery; running it as a script just raises a
# RuntimeError. Import time adds the parent test directory to sys.path.

import os
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.export import export
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import unittest

from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.jit_utils import JitTestCase


class WrapperModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Thin torch.nn.Module that stores a callable in the fn attribute and
    # forwards every call to it, so plain functions can be passed to torch.export.export.
    def __init__(self, fn: Callable):
        # [FUNCTION DOC] HELPER: Stores the callable fn on the module after the
        # base Module initialization.
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped function with the given positional and
        # keyword arguments and returns its result.
        return self.fn(*args, **kwargs)


class TestMatcher(JitTestCase):
    # [CLASS DOC] JitTestCase based test class exercising SubgraphMatcher and
    # SubgraphMatcherWithNameNodeMap on traced, make_fx and exported graphs. Every test builds a
    # target graph and a pattern graph, matches, and asserts on the number of matches or on the
    # name node map contents.
    def test_subgraph_matcher_with_attributes(self):
        # [FUNCTION DOC] Checks that a pattern whose get_attr parameters have different shapes
        # and names than the target (5x5 versus 3x3) still matches a traced addmm graph exactly
        # once. Asserts one match.
        class LargeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a bias and a weight parameter (3x3 ones) used
            # as the matching target.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two 3x3 parameters _weight and _bias.
                super().__init__()
                self._weight = torch.nn.Parameter(torch.ones(3, 3))
                self._bias = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the aten addmm operator with the bias, the
                # input and the weight.
                return torch.ops.aten.addmm.default(self._bias, x, self._weight)

        # Large Model graph:
        # opcode         name           target              args                 kwargs
        # -------------  -------------  ------------------  -------------------  --------
        # placeholder    x              x                   ()                   {}
        # get_attr       _bias          _bias               ()                   {}
        # get_attr       _weight        _weight             ()                   {}
        # call_function  addmm_default  aten.addmm.default  (_bias, x, _weight)  {}
        # output         output         output              (addmm_default,)     {}
        large_model_graph = symbolic_trace(LargeModel()).graph

        class PatternModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Same structure as LargeModel but with 5x5 parameters named
            # _weight_1 and _bias_1; its traced graph is the pattern.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two 5x5 parameters _weight_1 and _bias_1.
                super().__init__()
                self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
                self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the aten addmm operator with the pattern bias,
                # the input and the pattern weight.
                return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

        pattern_graph = torch.fx.symbolic_trace(PatternModel()).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(large_model_graph)
        self.assertEqual(len(match_result), 1)

    def test_subgraph_matcher_with_list(self):
        # [FUNCTION DOC] Checks that a pattern with a list argument containing a placeholder (z)
        # matches a target with a literal number in that list position, while the shape lookup
        # parts agree. Asserts exactly one match.
        def original(x, y):
            # [FUNCTION DOC] HELPER: Target function reshaping the input with a list of a
            # literal five and the second input's first dimension.
            return torch.ops.aten.view(x, [5, y.shape[0]])

        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, z):
            # [FUNCTION DOC] HELPER: Pattern function using the same view operator but with a
            # placeholder in place of the literal five.
            return torch.ops.aten.view(x, [z, y.shape[0]])

        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 1)

    def test_subgraph_matcher_with_list_bad(self):
        # [FUNCTION DOC] Negative test: the pattern's list arguments have a different number of
        # elements (three and one) than the target's (two and two), so no match is found.
        # Asserts zero matches.
        def original(x, y):
            # [FUNCTION DOC] HELPER: Target function calling the aten reshape alias copy
            # operator with two-element list arguments.
            return torch.ops.aten._reshape_alias_copy.default(
                x, [1, y.shape[0]], [y.shape[1], y.shape[1]]
            )

        original_graph = torch.fx.symbolic_trace(original).graph

        def pattern(x, y, b):
            # [FUNCTION DOC] HELPER: Pattern function calling the same operator with a
            # three-element size list and a one-element stride list.
            return torch.ops.aten._reshape_alias_copy.default(
                x, [b, y.shape[0], y.shape[1]], [y.shape[1]]
            )

        pattern_graph = torch.fx.symbolic_trace(pattern).graph

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 0)

    def test_subgraph_matcher_ignore_literals(self):
        # [FUNCTION DOC] Uses make_fx graphs of adding one (target) and adding two (pattern),
        # after dead code elimination. By default the differing literal prevents a match (zero
        # matches); with ignore_literals set to true the matcher finds one match.
        def original(x):
            # [FUNCTION DOC] HELPER: Target function adding the literal one to its input.
            return x + 1

        original_graph = make_fx(original)(torch.ones(3, 3)).graph
        original_graph.eliminate_dead_code()

        def pattern(x):
            # [FUNCTION DOC] HELPER: Pattern function adding the literal two to its input.
            return x + 2

        pattern_graph = make_fx(pattern)(torch.ones(4, 4)).graph
        pattern_graph.eliminate_dead_code()

        subgraph_matcher = SubgraphMatcher(pattern_graph)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 0)

        subgraph_matcher = SubgraphMatcher(pattern_graph, ignore_literals=True)
        match_result = subgraph_matcher.match(original_graph)
        self.assertEqual(len(match_result), 1)

    def test_variatic_arg_matching(self):
        # [FUNCTION DOC] Verifies matching of the max_pool2d_with_indices operator when the
        # target graph, traced from MaxPool2d modules, omits some arguments (only stride, only
        # padding, or both). The fully explicit pattern graph must match itself and every one of
        # these variants once.
        inputs = (torch.randn(20, 16, 50, 32),)

        def maxpool(x, kernel_size, stride, padding, dilation):
            # [FUNCTION DOC] HELPER: Function applying the aten max_pool2d_with_indices operator
            # with five explicit arguments; its traced graph is the pattern.
            return torch.ops.aten.max_pool2d_with_indices.default(
                x, kernel_size, stride, padding, dilation
            )

        maxpool_graph = torch.fx.symbolic_trace(maxpool).graph

        maxpool_matcher = SubgraphMatcher(maxpool_graph)
        match_result = maxpool_matcher.match(maxpool_graph)
        self.assertEqual(len(match_result), 1)

        # Graph only contains "stride" argument
        maxpool_s = torch.nn.MaxPool2d(kernel_size=2, stride=1).eval()
        maxpool_s_graph = make_fx(maxpool_s)(*inputs).graph
        match_s_result = maxpool_matcher.match(maxpool_s_graph)
        self.assertEqual(len(match_s_result), 1)

        # Graph only contains "padding" argument
        maxpool_p = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        maxpool_p_graph = make_fx(maxpool_p)(*inputs).graph
        match_p_result = maxpool_matcher.match(maxpool_p_graph)
        self.assertEqual(len(match_p_result), 1)

        # Graph only contains "stride, padding" argument
        maxpool_sp = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        maxpool_sp_graph = make_fx(maxpool_sp)(*inputs).graph
        match_sp_result = maxpool_matcher.match(maxpool_sp_graph)
        self.assertEqual(len(match_sp_result), 1)

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_split_to_graph_and_name_node_map(self):
        # [FUNCTION DOC] Checks the internal helper that splits an exported pattern graph module
        # into the graph and a name-to-node map: the output of the module after the split must
        # equal the output before for the first two return values. Skipped on Windows.
        """Testing the internal helper function for splitting the pattern graph"""
        from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
            _split_to_graph_and_name_node_map,
        )

        def pattern(x, weight):
            # [FUNCTION DOC] HELPER: Pattern function doing a convolution, a relu and a
            # multiplication by two, returning the relu, the doubled relu and a dictionary
            # naming the conv and relu results.
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        pattern_gm = export(
            WrapperModule(pattern), example_inputs, strict=True
        ).module()
        before_split_res = pattern_gm(*example_inputs)
        pattern_gm, _ = _split_to_graph_and_name_node_map(pattern_gm)
        after_split_res = pattern_gm(*example_inputs)
        self.assertEqual(before_split_res[0], after_split_res[0])
        self.assertEqual(before_split_res[1], after_split_res[1])

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_matcher_with_name_node_map_function(self):
        # [FUNCTION DOC] Runs SubgraphMatcherWithNameNodeMap with a function-based pattern
        # against an exported target graph, checks the name node map has conv and relu entries
        # for each match, annotates the matched conv node's meta, and verifies the annotation is
        # visible on the node in the target graph. Raises AssertionError on mismatch. Skipped
        # on Windows.
        """Testing SubgraphMatcherWithNameNodeMap with function pattern"""

        def target_graph(x, weight):
            # [FUNCTION DOC] HELPER: Target function scaling the input by two and the weight by
            # three, then convolution, relu, doubling and adding the two relu results.
            x = x * 2
            weight = weight * 3
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu2 = relu * 2
            return relu + relu2

        def pattern(x, weight):
            # [FUNCTION DOC] HELPER: Pattern function doing convolution, relu and doubling,
            # returning both values plus a dictionary mapping conv and relu names to
            # their results.
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu_mul_by_two = relu * 2
            return relu, relu_mul_by_two, {"conv": conv, "relu": relu}

        example_inputs = (
            torch.randn(1, 3, 3, 3) * 10,
            torch.randn(3, 3, 3, 3),
        )
        pattern_gm = export(
            WrapperModule(pattern), example_inputs, strict=True
        ).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        target_gm = export(
            WrapperModule(target_graph), example_inputs, strict=True
        ).module()
        internal_matches = matcher.match(target_gm.graph)
        for internal_match in internal_matches:
            name_node_map = internal_match.name_node_map
            if "conv" not in name_node_map:
                raise AssertionError("Expected 'conv' in name_node_map")
            if "relu" not in name_node_map:
                raise AssertionError("Expected 'relu' in name_node_map")
            name_node_map["conv"].meta["custom_annotation"] = "annotation"
            # check if we correctly annotated the target graph module
            for n in target_gm.graph.nodes:
                if n == name_node_map["conv"]:
                    if not (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    ):
                        raise AssertionError(
                            "Expected custom_annotation to be 'annotation'"
                        )

    @unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
    def test_matcher_with_name_node_map_module(self):
        # [FUNCTION DOC] Same flow as the function pattern test but with module patterns: a
        # Linear based Pattern module returns the linear output and a dictionary of linear and
        # x, and the target is a module with one Linear layer. Checks the map keys and the
        # annotation on the matched linear node. Raises AssertionError on mismatch.
        # Skipped on Windows.
        """Testing SubgraphMatcherWithNameNodeMap with module pattern"""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Target module wrapping one Linear layer of size 5 by 5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Linear layer stored as linear.
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input.
                return self.linear(x)

        class Pattern(torch.nn.Module):
            # [CLASS DOC] HELPER: Pattern module with one Linear layer whose forward also
            # returns a dictionary naming the linear output and the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Linear layer stored as linear.
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear output and a dictionary with the
                # linear result and the input under the names linear and x.
                linear = self.linear(x)
                # Note: we can't put "weight": self.linear.weight in dictionary since
                # nn.Parameter is not an allowed output type in dynamo
                return linear, {"linear": linear, "x": x}

        example_inputs = (torch.randn(3, 5),)
        pattern_gm = export(Pattern(), example_inputs, strict=True).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        target_gm = export(M(), example_inputs, strict=True).module()
        internal_matches = matcher.match(target_gm.graph)
        for internal_match in internal_matches:
            name_node_map = internal_match.name_node_map
            if "linear" not in name_node_map:
                raise AssertionError("Expected 'linear' in name_node_map")
            if "x" not in name_node_map:
                raise AssertionError("Expected 'x' in name_node_map")
            name_node_map["linear"].meta["custom_annotation"] = "annotation"
            # check if we correctly annotated the target graph module
            for n in target_gm.graph.nodes:
                if n == name_node_map["linear"]:
                    if not (
                        "custom_annotation" in n.meta
                        and n.meta["custom_annotation"] == "annotation"
                    ):
                        raise AssertionError(
                            "Expected custom_annotation to be 'annotation'"
                        )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
