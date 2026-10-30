# Owner(s): ["oncall: jit"]

# [MODULE DOC] Regression test that scripting a module which refers to a missing submodule
# attribute fails with an error naming that attribute. Defines one unittest TestCase. Running
# the file directly is rejected and redirects to the main JIT test entry point.
import unittest

import torch
from torch.testing._internal.common_utils import raise_on_run_directly


class TestConcreteModuleTypeFindSubmodule(unittest.TestCase):
    # [CLASS DOC] unittest TestCase with a single test about the error text produced when the
    # TorchScript compiler cannot find a submodule on a concrete module type.
    def test_error_message_includes_submodule_name(self):
        # [FUNCTION DOC] Script a valid parent/child module pair and check the child attribute
        # is reachable. Then script a module whose forward calls a nonexistent attribute on a
        # nested parent module and require a RuntimeError whose lower-cased message contains
        # the missing attribute name.
        class ChildModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module owning one Linear layer from 5 to 3 features;
            # forward just applies it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the child module and its linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(5, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the output of the linear layer on the input.
                return self.linear(x)

        class ParentModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that holds one ChildModule as
            # the attribute existing_child.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the module and register the child module.
                super().__init__()
                self.existing_child = ChildModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the output of the existing child applied to
                # the input.
                return self.existing_child(x)

        module = ParentModule()
        scripted_module = torch.jit.script(module)

        self.assertIsNotNone(scripted_module.existing_child)

        # Now try to trigger the error by accessing a non-existent submodule
        # through the internal ConcreteModuleType mechanism. This happens
        # when the TorchScript compiler tries to resolve submodule references.
        class BrokenModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that holds a ParentModule and whose forward uses an
            # attribute that the parent does not define, so scripting it must fail.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the module and store a ParentModule instance.
                super().__init__()
                self.parent = ParentModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Call the missing attribute on the parent; never runs
                # because the compile step rejects it.
                return self.parent.missing_submodule(x)

        broken_module = BrokenModule()

        with self.assertRaises(RuntimeError) as context:
            torch.jit.script(broken_module)

        error_msg = str(context.exception)
        self.assertIn("missing_submodule", error_msg.lower())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
