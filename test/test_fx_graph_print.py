# Owner(s): ["module: fx"]

# [MODULE DOC] Tests for GraphModule.print_readable and its additional_meta option, which appends
# chosen node.meta entries as trailing comments in the printed code. Graphs are built with make_fx
# or symbolic_trace, meta is attached by hand, and the returned string is checked. No import-time
# side effects besides running run_tests when executed as main.
import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFxGraphPrint(TestCase):
    # [CLASS DOC] TestCase grouping the print_readable additional_meta tests. Stateless; each test
    # traces a tiny function or module, sets meta on call_function nodes, and inspects output.
    """Tests for GraphModule.print_readable() with additional_meta."""

    def test_print_readable_with_list_of_meta_keys(self):
        # [FUNCTION DOC] Traces sin plus cos, stores two custom meta keys (a string and a multiple
        # of ten) on every call_function node, prints with both keys requested, and asserts each
        # key/value pair for the first three nodes appears in the returned text.
        """Test print_readable with a list of meta keys to include."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function returning sin plus cos of its input.
            return torch.sin(x) + torch.cos(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Manually add some meta keys to nodes for testing
        call_fn_idx = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["my_custom_key"] = f"value_{call_fn_idx}"
                node.meta["another_key"] = call_fn_idx * 10
                call_fn_idx += 1

        output = gm.print_readable(
            print_output=False, additional_meta=["my_custom_key", "another_key"]
        )

        # The actual meta values should appear in the generated code
        self.assertIn("my_custom_key: value_0", output)
        self.assertIn("another_key: 0", output)
        self.assertIn("my_custom_key: value_1", output)
        self.assertIn("another_key: 10", output)
        self.assertIn("my_custom_key: value_2", output)
        self.assertIn("another_key: 20", output)

    def test_print_readable_with_list_missing_keys(self):
        # [FUNCTION DOC] Sets only one meta key on call_function nodes, requests it together with
        # a key that no node has, and asserts the existing key is printed while the missing key
        # name never appears in the output.
        """Test print_readable with a list containing keys that don't exist in meta."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function applying sin to its input.
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Add only one key
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["existing_key"] = "exists"

        output = gm.print_readable(
            print_output=False,
            additional_meta=["existing_key", "nonexistent_key"],
        )

        # Only existing_key should appear
        self.assertIn("existing_key: exists", output)
        self.assertNotIn("nonexistent_key", output)

    def test_print_readable_with_seq_nr(self):
        # [FUNCTION DOC] Assigns increasing sequence numbers starting at zero to
        # call_function nodes, prints with seq_nr requested, and asserts values 0, 1 and 2
        # show up in the output.
        """Test print_readable with seq_nr in node.meta."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function computing sin and cos of the input
            # and adding them.
            y = torch.sin(x)
            z = torch.cos(x)
            return y + z

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Manually set seq_nr for testing (normally set during tracing)
        seq_nr = 0
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["seq_nr"] = seq_nr
                seq_nr += 1

        output = gm.print_readable(print_output=False, additional_meta=["seq_nr"])

        # seq_nr values should appear in the generated code
        self.assertIn("seq_nr: 0", output)
        self.assertIn("seq_nr: 1", output)
        self.assertIn("seq_nr: 2", output)

    def test_print_readable_none_annotation(self):
        # [FUNCTION DOC] Prints with additional_meta set to None (the default) and asserts it
        # works without error and the output still mentions sin.
        """Test print_readable with additional_meta=None (default behavior)."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function applying sin to its input.
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Default behavior - no custom annotation
        output = gm.print_readable(print_output=False, additional_meta=None)

        # Should still work without errors
        self.assertIn("sin", output)

    def test_print_readable_with_nested_modules(self):
        # [FUNCTION DOC] Symbolically traces a module that calls a nested submodule, tags the
        # call_function nodes of the top graph with a layer meta value, prints with that key, and
        # asserts the tag appears in the output.
        """Test print_readable with nested GraphModules."""

        class Inner(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward applies relu; used as a submodule.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns relu of the input.
                return torch.relu(x)

        class Outer(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding an Inner submodule and applying it to sin of x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: initializes the parent module and registers Inner.
                super().__init__()
                self.inner = Inner()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the inner submodule to sin of the input and
                # returns it.
                return self.inner(torch.sin(x))

        model = Outer()
        traced = symbolic_trace(model)

        # Add meta to nodes
        for node in traced.graph.nodes:
            if node.op == "call_function":
                node.meta["layer"] = "outer"

        output = traced.print_readable(print_output=False, additional_meta=["layer"])

        self.assertIn("layer: outer", output)

    def test_print_readable_additional_meta_ordering(self):
        # [FUNCTION DOC] Adds a dict-valued meta entry plus a plain one, prints with the plain key
        # requested, and for any output line holding both the key text and the annotation marker
        # asserts the key text comes first. If no line holds both, nothing is asserted.
        """Test that additional_meta appears before other meta info in the comment."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function applying sin to its input.
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Add custom meta that would also appear via annotation
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["custom"] = {"test": "value"}
                node.meta["my_key"] = "my_value"

        output = gm.print_readable(print_output=False, additional_meta=["my_key"])

        # The additional_meta should appear before the custom annotation
        for line in output.split("\n"):
            if "my_key: my_value" in line and "Annotation:" in line:
                my_key_pos = line.find("my_key: my_value")
                annotation_pos = line.find("Annotation:")
                self.assertLess(
                    my_key_pos,
                    annotation_pos,
                    "additional_meta should appear before Annotation",
                )
                break

    def test_print_readable_with_empty_list(self):
        # [FUNCTION DOC] Prints with an empty list of meta keys and asserts no error occurs and
        # sin is in the output.
        """Test print_readable with an empty list of meta keys."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function applying sin to its input.
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Empty list should not add any annotation
        output = gm.print_readable(print_output=False, additional_meta=[])

        # Should work without errors
        self.assertIn("sin", output)

    def test_print_readable_multiple_meta_keys_formatting(self):
        # [FUNCTION DOC] Sets three meta keys on call_function nodes, requests all three, and
        # asserts each key/value pair appears and that the output contains a comma plus space
        # separator (a weak check, since the output may contain that elsewhere).
        """Test that multiple meta keys are formatted correctly with commas."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function applying sin to its input.
            return torch.sin(x)

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["key1"] = "val1"
                node.meta["key2"] = "val2"
                node.meta["key3"] = "val3"

        output = gm.print_readable(
            print_output=False, additional_meta=["key1", "key2", "key3"]
        )

        # All three keys should appear comma-separated
        self.assertIn("key1: val1", output)
        self.assertIn("key2: val2", output)
        self.assertIn("key3: val3", output)
        # Check comma separation
        self.assertIn(", ", output)

    def test_print_readable_seq_nr_with_multiple_ops(self):
        # [FUNCTION DOC] Assigns sequence numbers starting at 100 to successive call_function
        # nodes of a graph with sin, cos, add and relu, prints with seq_nr requested, and asserts
        # values 100 through 103 appear.
        """Test print_readable with seq_nr across multiple different operations."""

        def forward(x):
            # [FUNCTION DOC] HELPER: traced function computing relu of sin plus cos of the input.
            a = torch.sin(x)
            b = torch.cos(x)
            c = torch.relu(a + b)
            return c

        x = torch.randn(3)
        gm = make_fx(forward)(x)

        # Manually set seq_nr
        seq_nr = 100
        for node in gm.graph.nodes:
            if node.op == "call_function":
                node.meta["seq_nr"] = seq_nr
                seq_nr += 1

        output = gm.print_readable(print_output=False, additional_meta=["seq_nr"])

        # Check that seq_nr values appear
        self.assertIn("seq_nr: 100", output)
        self.assertIn("seq_nr: 101", output)
        self.assertIn("seq_nr: 102", output)
        self.assertIn("seq_nr: 103", output)


if __name__ == "__main__":
    run_tests()
