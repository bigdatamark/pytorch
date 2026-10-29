# Owner(s): ["module: fx"]
# [MODULE DOC] Tests that FX tracing with make_fx supports opaque custom class objects: an
# opaque counter is registered as a symbolic opaque type and passed through a traced function.
# Import time registers OpaqueCounter via register_custom_class.

import torch
from torch._library.opaque_object import register_custom_class
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


# Define a simple opaque type for testing
class OpaqueCounter(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Opaque test type deriving from CustomClassBase that stores an integer
    # in the attribute value. Registered below with symbolic (reference) semantics.
    """A simple opaque object that holds a counter."""

    def __init__(self, value: int):
        # [FUNCTION DOC] HELPER: Stores the initial integer as the attribute value.
        self.value = value

    def get_value(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the stored counter value.
        """Get the current counter value."""
        return self.value


# Register it as an opaque type (reference semantics for identity/mutation tracking)
register_custom_class(OpaqueCounter, typ="symbolic")


# Define a wrapper class that holds an opaque object as an attribute
class WrapperWithOpaque:
    # [CLASS DOC] HELPER: Plain wrapper that keeps an opaque counter and a small float tensor as
    # attributes. Not used by the test in this file.
    """A wrapper class that contains an opaque object."""

    def __init__(self, counter: OpaqueCounter):
        # [FUNCTION DOC] HELPER: Stores the counter and creates a tensor of three
        # floats as data.
        self.counter = counter
        self.data = torch.tensor([1.0, 2.0, 3.0])


class TestOpaqueInfrastructure(TestCase):
    # [CLASS DOC] Test case for tracing infrastructure around opaque objects such
    # as process groups.
    """
    Test opaque object descriptor and tracking infrastructure.

    This test validates infrastructure for tracking and passing opaque objects
    (like ProcessGroups) through AOTAutograd compilation without unwrapping/wrapping them.
    """

    def test_opaque_object_in_traced_graph(self):
        # [FUNCTION DOC] Traces a function that returns its tensor input plus one together with
        # the opaque object, using make_fx. Checks the graph exists, that running the traced
        # module returns the incremented tensor and the identical opaque object, that there are
        # two placeholder nodes, and that the second one is named starting with opaque_obj.
        """Test that opaque objects can be traced into FX graphs."""

        counter = OpaqueCounter(42)

        def fn(x, opaque_obj):
            # [FUNCTION DOC] HELPER: Returns the input plus one together with the
            # opaque object unchanged.
            # Just pass the opaque object through and return the tensor
            # The opaque object should appear in the graph
            return x + 1, opaque_obj

        # Trace the function with make_fx
        x = torch.randn(3, 3)
        traced = make_fx(fn)(x, counter)

        # Verify the graph was created
        self.assertIsNotNone(traced.graph)

        # Run the traced function and verify opaque object is passed through
        result_tensor, result_opaque = traced(x, counter)
        self.assertTrue(torch.allclose(result_tensor, x + 1))
        self.assertIs(result_opaque, counter)

        # Verify the opaque object appears as an input placeholder in the graph
        placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
        self.assertEqual(len(placeholders), 2)  # x and opaque_obj

        # The second placeholder should be for the opaque object
        opaque_placeholder = placeholders[1]
        self.assertTrue(opaque_placeholder.name.startswith("opaque_obj"))


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
