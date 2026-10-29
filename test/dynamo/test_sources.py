# Owner(s): ["module: dynamo"]
# [MODULE DOC]
# Tests for Dynamo source objects and a few source-related tracing behaviors: classifying
# whether a source derives from a local variable, compiling through a property whose getter
# is a closure, and exporting a module that reads the pytree registry of supported nodes.
# Defines a small placeholder class used as a pytree node type. Running the file as a script
# invokes the Dynamo test runner. No import-time side effects beyond imports.

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.nn as nn
from torch._dynamo.source import (
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)


class CausalLMOutputWithPast:
    # [CLASS DOC] HELPER: Empty stand-in for a model output type. It only carries a class
    # attribute and is registered as a pytree node by test_supported_nodes.
    value = 5


class SourceTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case grouping checks on Dynamo source classification and on tracing
    # constructs that depend on how sources are built (properties, pytree registry lookups).
    def test_is_local(self):
        # [FUNCTION DOC] Builds an attribute source on top of a local-variable source and
        # another on top of a global source, then checks that the local-source predicate is
        # true for the first and false for the second.
        x_src = LocalSource("x")
        y_src = GlobalSource("y")

        attr_x_a = AttrSource(x_src, "a")
        attr_y_b = AttrSource(y_src, "b")

        self.assertTrue(is_from_local_source(attr_x_a))
        self.assertEqual(is_from_local_source(attr_y_b), False)

    def test_property_closure(self):
        # [FUNCTION DOC] Defines a class whose property getter is a closure over a constant
        # seven, compiles a function branching on that property with the eager backend and
        # full graph mode, and checks the compiled result equals the eager result.
        def external_property():
            # [FUNCTION DOC] HELPER: Factory returning a getter function that ignores its
            # instance argument and returns the captured constant seven.
            closed_value = 7

            def internal_function(self):
                # [FUNCTION DOC] HELPER: Getter body; returns the closed-over constant
                # value regardless of the instance.
                return closed_value

            return internal_function

        class Elements:
            # [CLASS DOC] HELPER: Holder of a single property built from the closure
            # getter above.
            myprop = property(external_property())

        def func(elements):
            # [FUNCTION DOC] HELPER: Returns a tensor of one-to-three when the property
            # is falsy, otherwise a tensor of four-to-six; the property is truthy here.
            if not elements.myprop:
                return torch.tensor([1, 2, 3])
            else:
                return torch.tensor([4, 5, 6])

        e = Elements()
        a = func(e)
        b = torch.compile(func, backend="eager", fullgraph=True)(e)
        self.assertEqual(a, b)

    def test_supported_nodes(self):
        # [FUNCTION DOC] Defines a module whose forward chooses sin or cos depending on
        # whether the registered pytree node type for the placeholder output class is int.
        # Registers the placeholder class as a pytree node (global side effect on the
        # pytree registry that is never undone), then exports the module in strict mode.
        # The test passes if export completes without raising.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module holding a random ten by ten tensor; forward reads
            # the pytree registry during tracing to pick its branch.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor and stores a random
                # ten by ten tensor as an attribute.
                super().__init__()
                self.x = torch.randn(10, 10)

            def forward(self):
                # [FUNCTION DOC] HELPER: Looks up the placeholder class in the pytree
                # supported-nodes mapping; if its node type is int returns sin of the
                # stored tensor, else cos of it.
                if (
                    torch.utils._pytree.SUPPORTED_NODES[CausalLMOutputWithPast].type
                    is int
                ):
                    x = torch.sin(self.x)
                else:
                    x = torch.cos(self.x)
                return x

        torch.utils._pytree.register_pytree_node(
            CausalLMOutputWithPast,
            lambda x: ((), None),
            lambda x, _: CausalLMOutputWithPast(),
        )

        torch.export.export(Model(), (), strict=True)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
