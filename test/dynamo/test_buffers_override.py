# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests checking that an nn.Module which shadows its buffers or
# named_buffers methods with an instance attribute (an empty list) still compiles with the
# eager backend. No import-time side effects beyond imports.

import torch
import torch._dynamo.test_case
import torch.nn as nn


class TestBuffersOverride(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase with two tests that compile modules whose buffers /
    # named_buffers attribute has been overwritten by a list; stateless.
    def test_buffers_override(self):
        # [FUNCTION DOC] Builds a module with buffer "A" and an instance attribute named
        # buffers set to an empty list, compiles it with the eager backend, checks that the
        # compiled wrapper still exposes A equal to a 3x3 ones tensor, and runs forward once.
        # Passes if no exception is raised (Dynamo uses named_buffers for static marking).
        class SomeModel(nn.Module):
            # [CLASS DOC] HELPER: Module with one 3x3 ones buffer A whose buffers method is
            # shadowed by an empty list instance attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers buffer A (3x3 ones) and sets buffers to [].
                super().__init__()
                # Override buffers; should not cause breakage
                # this is because we use `named_buffers` for
                # static marking
                self.register_buffer("A", torch.ones(3, 3))
                self.buffers = []

            def forward(self):
                # [FUNCTION DOC] HELPER: Takes no input; returns A times a 1x1 zeros tensor
                # (a 3x3 zeros tensor via broadcasting).
                return self.A * torch.zeros(1, 1)

        model = SomeModel().to(torch.device("cpu"))
        compiled_model = torch.compile(model, backend="eager")
        self.assertEqual(compiled_model.A, torch.ones(3, 3))
        compiled_model()

    def test_named_buffers_override(self):
        # [FUNCTION DOC] Same as the buffers override test but the module shadows
        # named_buffers with an empty list and registers buffer "B". Compiles with the eager
        # backend, checks B equals 3x3 ones on the wrapper, and runs forward once; the test
        # only requires that compilation and execution do not raise.
        class SomeModel(nn.Module):
            # [CLASS DOC] HELPER: Module with one 3x3 ones buffer B whose named_buffers
            # method is shadowed by an empty list instance attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers buffer B (3x3 ones), sets named_buffers to [].
                super().__init__()
                # Override buffers; should not cause breakage
                # but skip the marking static here since
                # named_buffers is overridden
                self.register_buffer("B", torch.ones(3, 3))
                self.named_buffers = []

            def forward(self):
                # [FUNCTION DOC] HELPER: Takes no input; returns B times a 1x1 zeros tensor.
                return self.B * torch.zeros(1, 1)

        model = SomeModel().to(torch.device("cpu"))
        compiled_model = torch.compile(model, backend="eager")
        self.assertEqual(compiled_model.B, torch.ones(3, 3))
        compiled_model()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
