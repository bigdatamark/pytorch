# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for FX graph splitting utilities: splitting by tags, a splitter base
# class with a custom operator-support check, and output type preservation. Defines a
# scripted dataclass and a wrapped (opaque to tracing) function used by one test. Importing
# the file compiles the dataclass with the TorchScript compiler. Running the file directly
# raises an error stating the tests are not currently used and must be enabled in test
# discovery.

import dataclasses
from collections import defaultdict

import torch
import torch.fx.passes.operator_support as op_support
import torch.fx.passes.splitter_base as splitter_base
from torch.fx.passes.split_utils import split_by_tags
from torch.testing._internal.common_utils import TestCase


@torch.jit.script
@dataclasses.dataclass
class DummyDataClass:
    # [CLASS DOC] HELPER: Dataclass with three integer fields a, b and c, compiled with
    # TorchScript.
    a: int
    b: int
    c: int


@torch.fx.wrap
def wrapped_add(_dataclass, y):
    # [FUNCTION DOC] HELPER: Wrapped function (kept as a single call node by tracing).
    # Returns the c field of the dataclass plus y.
    return _dataclass.c + y


class TestFXSplit(TestCase):
    # [CLASS DOC] Tests for tag-based splitting preserving metadata and for splitting
    # through the splitter base class with a dataclass-producing call node. No state.
    def test_split_preserve_node_meta(self):
        # [FUNCTION DOC] Splits a traced three-op module by tags, one op per tag, after
        # storing each node's name in its meta. Asserts every non-output node in each child
        # graph still has the meta name equal to its node name, and that placeholder nodes
        # of the split module also keep the meta (copied correctly).
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing x plus x, y times y and returning their
            # difference.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the doubled x minus the squared y.
                x = x + x
                y = y * y
                return x - y

        gm = torch.fx.symbolic_trace(TestModule())
        for node in gm.graph.nodes:
            node.meta["name"] = node.name
            if node.name == "add":
                node.tag = "a"
            elif node.name == "mul":
                node.tag = "b"
            elif node.name == "sub":
                node.tag = "c"

        split_gm = split_by_tags(gm, ["a", "b", "c"])
        for m in split_gm.children():
            for n in m.graph.nodes:
                if n.op != "output":
                    self.assertIn("name", n.meta)
                    self.assertEqual(n.meta["name"], n.name)

        # Validate that metadata is copied correctly for graph placeholder nodes
        for node in split_gm.graph.nodes:
            if node.op == "placeholder":
                self.assertIn("name", node.meta)
                self.assertEqual(node.meta["name"], node.name)

    def test_dataclass_as_graph_entry(self):
        # [FUNCTION DOC] Edge case: the graph contains a call_function node with no input
        # dependencies (a dataclass constructed inside forward and passed to a wrapped
        # function with the input). Builds a splitter with an operator-support object that
        # accepts only the wrapped function, calls it, asserts a result is returned (no
        # exception) and that the split module's output equals the original module's on a
        # random input.
        """
        Test that splitting works when the graph entry is a dataclass instance
        and a wrapped function is called with it, resulting in a call_function
        node with no input dependencies. This tests the edge case fixed in D81232435
        where call_function nodes with no dependencies should be handled properly
        in the starter_nodes() method.

        Graph visualization:
        y (input)    DummyDataClass(2,3,4) (no input deps, result as a call_function_node)
            \              /
             \            /
              wrapped_add
                  |
              z (output)
        """  # noqa: W605

        class TestModuleWithFunctionEntry(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that builds a dataclass value from constants and
            # passes it with the input to the wrapped function.
            def forward(self, y):
                # [FUNCTION DOC] HELPER: Creates the dataclass value from three constants
                # and returns the wrapped function applied to it and y.
                # This creates a call_function node with no input dependencies
                dummy_data_class = DummyDataClass(2, 3, 4)
                z = wrapped_add(dummy_data_class, y)
                return z

        mod = TestModuleWithFunctionEntry()
        gm = torch.fx.symbolic_trace(mod)

        # Create custom operator support to mark wrapped_add as supported
        class CustomOpSupport(op_support.OperatorSupportBase):
            # [CLASS DOC] HELPER: Operator support that accepts only the wrapped function.
            def is_node_supported(self, submodules, node) -> bool:
                # [FUNCTION DOC] HELPER: Returns True if the node target is the wrapped
                # function. Submodules argument is unused.
                return node.target is wrapped_add

        # Create a simple splitter to test the edge case
        class TestSplitter(splitter_base._SplitterBase):
            # [CLASS DOC] HELPER: Minimal splitter built on the private splitter base class
            # with default settings.
            def __init__(self, module, sample_input, operator_support):
                # [FUNCTION DOC] HELPER: Creates default base settings and forwards the
                # module, sample input, operator support and settings to the base
                # constructor.
                settings = splitter_base._SplitterSettingBase()
                super().__init__(module, sample_input, operator_support, settings)

        # Create splitter instance - this tests the fix where call_function nodes
        # with no input dependencies are properly handled in starter_nodes()
        splitter = TestSplitter(
            module=gm,
            sample_input=[torch.randn(2, 3)],
            operator_support=CustomOpSupport(),
        )

        # This should not raise an exception (tests the fix from D81232435)
        # The fix allows call_function nodes with no dependencies as valid starter nodes
        split_result = splitter()

        # Verify the splitting worked correctly
        self.assertIsNotNone(split_result)

        # Test that the split module produces the same result as the original
        test_input = torch.randn(2, 3)
        original_result = mod(test_input)
        split_module_result = split_result(test_input)
        self.assertTrue(torch.equal(original_result, split_module_result))


class TestSplitByTags(TestCase):
    # [CLASS DOC] Tests splitting a symbolically traced module by tags and checking the
    # resulting submodule names, order and contents plus the original-to-split name mapping.
    class TestModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with four linear layers; three inputs go through the
        # first three, the results are concatenated and fed to the fourth.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates four linear layers of sizes 2 to 3, 4 to 5, 6
            # to 7 and 8 to 6.
            super().__init__()
            self.linear1 = torch.nn.Linear(2, 3)
            self.linear2 = torch.nn.Linear(4, 5)
            self.linear3 = torch.nn.Linear(6, 7)
            self.linear4 = torch.nn.Linear(8, 6)

        def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor,
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the first three layers to the three inputs,
            # concatenates the outputs and returns the fourth layer applied to the result.
            v1 = self.linear1(x1)
            v2 = self.linear2(x2)
            v3 = self.linear3(x3)
            v4 = torch.cat([v1, v2, v3])
            return self.linear4(v4)

    @staticmethod
    def trace_and_tag(
        module: torch.nn.Module, tags: list[str]
    ) -> tuple[torch.fx.GraphModule, dict[str, list[str]]]:
        # [FUNCTION DOC] HELPER: Static helper. Symbolically traces the module and assigns a
        # tag to every node: the first tag for nodes whose name mentions the first layer,
        # the second tag for those mentioning the second layer, the third tag for everything
        # else. Records the names of module-call nodes per tag (all nodes for the first
        # two). Returns the graph module and a mapping from tag to node names.
        """
        Test simple gm consists of nodes with tag (only show call_module nodes here):
            linear1 - tag: "red"
            linear2 - tag: "blue"
            linear3, linear4 - tag: "green"

        At the beginning we have:
            gm:
                linear1
                linear2
                linear3
                linear4

        split_gm = split_by_tags(gm, tags)

        Then we have:
            split_gm:
                red:
                    linear1
                blue:
                    linear2
                green:
                    linear3
                    linear4
        """
        tag_node = defaultdict(list)
        gm: torch.fx.GraphModule = torch.fx.symbolic_trace(module)

        # Add tag to all nodes and build dictionary record tag to call_module nodes
        for node in gm.graph.nodes:
            if "linear1" in node.name:
                node.tag = tags[0]
                tag_node[tags[0]].append(node.name)
            elif "linear2" in node.name:
                node.tag = tags[1]
                tag_node[tags[1]].append(node.name)
            else:
                node.tag = tags[2]
                if node.op == "call_module":
                    tag_node[tags[2]].append(node.name)
        return gm, tag_node

    def test_split_by_tags(self) -> None:
        # [FUNCTION DOC] Splits the tagged module by tags requesting the name mapping.
        # Asserts the split module's children are named after the tags in order, that each
        # child contains the expected module-call nodes in order, and that the original-to-
        # split qualified name mapping equals the expected mapping of the four layers to
        # their tag-prefixed names.
        tags = ["red", "blue", "green"]
        module = TestSplitByTags.TestModule()
        gm, tag_node = TestSplitByTags.trace_and_tag(module, tags)
        split_gm, orig_to_split_fqn_mapping = split_by_tags(
            gm, tags, return_fqn_mapping=True
        )
        # Ensure split_gm has (and only has) ordered submodules named
        # red_0, blue_1, green_2
        for idx, (name, _) in enumerate(split_gm.named_children()):
            if idx < len(tags):
                self.assertTrue(
                    name == tags[idx],
                    lambda msg: f"{msg}\nsplit_gm has an incorrect submodule named {name}",
                )

        # Ensure each submodule has expected (ordered) call_module node(s).
        # For example, a submodule named split_gm.red_0 has (and only has) linear1;
        # split_gm.green_2 has (and only has) linear3 and linear4 with order
        sub_graph_idx = 0
        for sub_name, sub_graph_module in split_gm.named_children():
            node_idx = 0
            for node in sub_graph_module.graph.nodes:
                if node.op != "call_module":
                    continue
                self.assertTrue(
                    node.name == tag_node[f"{sub_name}"][node_idx],
                    # pyre-fixme[61]: `name` is undefined, or not always defined.
                    lambda msg: f"{msg}\n{sub_name} has incorrectly include {node.name}",
                )
                node_idx += 1
            sub_graph_idx += 1

        self.assertEqual(
            orig_to_split_fqn_mapping,
            {
                "linear1": "red.linear1",
                "linear2": "blue.linear2",
                "linear3": "green.linear3",
                "linear4": "green.linear4",
            },
            lambda msg: f"{msg}\n{orig_to_split_fqn_mapping=}",
        )


class TestSplitOutputType(TestCase):
    # [CLASS DOC] Tests that splitting by tags preserves the output type and value for an
    # exported module.
    class TestModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with a convolution, a multiplication by 0.5 and a relu.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates a 3 to 16 channel 3 by 3 convolution with bias
            # and a relu.
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Convolves, halves the result and applies relu.
            conv = self.conv(x)
            conv = conv * 0.5
            relu = self.relu(conv)
            return relu

    @staticmethod
    def trace_and_tag(
        module: torch.nn.Module, inputs: torch.Tensor, tags: list[str]
    ) -> tuple[torch.fx.GraphModule, dict[str, list[str]]]:
        # [FUNCTION DOC] HELPER: Static helper. Exports the module in strict mode with the
        # given inputs and takes the resulting graph module. Tags nodes by name: conv nodes
        # get the first tag, mul nodes the second, everything else the third. Returns the
        # graph module and a mapping from tag to node names.
        """
        Test simple gm consists of nodes with tag (only show call_module nodes here):
            conv - tag: "red"
            mul - tag: "blue"
            relu - tag: "green"

        At the beginning we have:
            gm:
                conv
                mul
                relu

        split_gm = split_by_tags(gm, tags)

        Then we have:
            split_gm:
                red:
                    conv
                blue:
                    mul
                green:
                    relu
        """
        tag_node = defaultdict(list)
        gm: torch.fx.GraphModule = torch.export.export(
            module, (inputs,), strict=True
        ).module()
        # Add tag to all nodes and build dictionary record tag to call_module nodes
        for node in gm.graph.nodes:
            if "conv" in node.name:
                node.tag = tags[0]
                tag_node[tags[0]].append(node.name)
            elif "mul" in node.name:
                node.tag = tags[1]
                tag_node[tags[1]].append(node.name)
            else:
                node.tag = tags[2]
                if node.op == "call_module":
                    tag_node[tags[2]].append(node.name)
        return gm, tag_node

    def test_split_by_tags(self) -> None:
        # [FUNCTION DOC] Exports and tags the module, splits it by tags, runs the original
        # module and the split module on a random 1 by 3 by 224 by 224 input and asserts the
        # outputs have the same type and equal values.
        tags = ["red", "blue", "green"]
        module = TestSplitOutputType.TestModule()

        inputs = torch.randn((1, 3, 224, 224))

        gm, _ = TestSplitOutputType.trace_and_tag(module, inputs, tags)
        split_gm, _ = split_by_tags(gm, tags, return_fqn_mapping=True)

        gm_output = module(inputs)
        split_gm_output = split_gm(inputs)

        self.assertTrue(type(gm_output) is type(split_gm_output))
        self.assertTrue(torch.equal(gm_output, split_gm_output))


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
