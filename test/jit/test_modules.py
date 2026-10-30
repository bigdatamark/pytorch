# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT test for scripting nn.Module subclasses whose constants declaration is a
# list. Defines one JitTestCase; running the file directly raises via raise_on_run_directly.
# Import time: appends the test directory to sys.path so shared helpers can be imported.

import os
import sys

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestModules(JitTestCase):
    # [CLASS DOC] JitTestCase holding the module-scripting test for list-typed constants.
    def test_script_module_with_constants_list(self):
        # [FUNCTION DOC] Builds a Linear subclass with a Final integer attribute (the parent
        # keeps its constants declaration as a list, not a set) and checks that checkModule can
        # script it and that scripted and eager outputs match on a random vector input. Fails if
        # scripting raises.
        """
        Test that a module that has __constants__ set to something
        that is not a set can be scripted.
        """

        # torch.nn.Linear has a __constants__ attribute defined
        # and initialized to a list.
        class Net(torch.nn.Linear):
            # [CLASS DOC] HELPER: Linear layer subclass (5 inputs, 10 outputs) declaring a Final
            # int annotation and setting it to zero.
            x: torch.jit.Final[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the parent Linear with 5 inputs and 10
                # outputs and sets the integer attribute to zero.
                super().__init__(5, 10)
                self.x = 0

        self.checkModule(Net(), (torch.randn(5),))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
