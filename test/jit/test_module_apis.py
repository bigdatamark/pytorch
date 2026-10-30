# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT tests for state-dict APIs on scripted modules: default state_dict and
# load_state_dict, and user-customized save/load hooks (also on submodules). Import time:
# appends the test directory to sys.path so shared helpers can be imported.

import os
import sys
from typing import Any, Dict, List

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestModuleAPIs(JitTestCase):
    # [CLASS DOC] JitTestCase with tests of state dict methods on ScriptModules.
    def test_default_state_dict_methods(self):
        # [FUNCTION DOC] Scripts two modules of the same class, takes the state dict of the
        # first and loads it into the second. Passes if nothing raises (no value comparison
        # is made).
        """Tests that default state dict methods are automatically available"""

        class DefaultStateDictModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv2d and a Linear layer using default state
            # dict behavior.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a convolution (6 to 16 channels, kernel
                # five) and a linear layer (400 to 120).
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv layer then the linear layer and
                # returns the result.
                x = self.conv(x)
                x = self.fc(x)
                return x

        m1 = torch.jit.script(DefaultStateDictModule())
        m2 = torch.jit.script(DefaultStateDictModule())
        state_dict = m1.state_dict()
        m2.load_state_dict(state_dict)

    def test_customized_state_dict_methods(self):
        # [FUNCTION DOC] Scripts a module with exported custom save and load state dict hooks.
        # Checks the save flag is false before state_dict is called and true after, and that the
        # load flag on a second scripted instance flips only after load_state_dict is called.
        """Tests that customized state dict methods are in effect"""

        class CustomStateDictModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with conv and linear layers plus boolean flags
            # recording whether its custom save and load hooks ran.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers and initializes both hook-called
                # flags to false.
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv layer then the linear layer and
                # returns the result.
                x = self.conv(x)
                x = self.fc(x)
                return x

            @torch.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool
            ):
                # [FUNCTION DOC] HELPER: Exported custom save hook; sets the save-called flag
                # and returns a dictionary with a single one-element tensor under a dummy key,
                # ignoring the destination.
                self.customized_save_state_dict_called = True
                return {"dummy": torch.ones(1)}

            @torch.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                # [FUNCTION DOC] HELPER: Exported custom load hook; sets the load-called flag
                # and returns without loading anything. Other arguments are required by the
                # signature but unused.
                self.customized_load_state_dict_called = True
                return

        m1 = torch.jit.script(CustomStateDictModule())
        self.assertFalse(m1.customized_save_state_dict_called)
        state_dict = m1.state_dict()
        self.assertTrue(m1.customized_save_state_dict_called)

        m2 = torch.jit.script(CustomStateDictModule())
        self.assertFalse(m2.customized_load_state_dict_called)
        m2.load_state_dict(state_dict)
        self.assertTrue(m2.customized_load_state_dict_called)

    def test_submodule_customized_state_dict_methods(self):
        # [FUNCTION DOC] Same as the custom hook test but the hooks live on a submodule of a
        # parent module. Checks the submodule save flag flips after state_dict on the scripted
        # parent and the load flag of a second instance flips after load_state_dict.
        """Tests that customized state dict methods on submodules are in effect"""

        class CustomStateDictModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Same custom-hook module as in the previous test, used as
            # a submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers and initializes both hook-called
                # flags to false.
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv layer then the linear layer and
                # returns the result.
                x = self.conv(x)
                x = self.fc(x)
                return x

            @torch.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool
            ):
                # [FUNCTION DOC] HELPER: Exported custom save hook; sets the save-called flag
                # and returns a dictionary with a single one-element tensor under a dummy key,
                # ignoring the destination.
                self.customized_save_state_dict_called = True
                return {"dummy": torch.ones(1)}

            @torch.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                # [FUNCTION DOC] HELPER: Exported custom load hook; sets the load-called flag
                # and returns without loading anything. Other arguments are required by the
                # signature but unused.
                self.customized_load_state_dict_called = True
                return

        class ParentModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module that holds a single custom-hook submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the custom-hook submodule.
                super().__init__()
                self.sub = CustomStateDictModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forwards the input through the submodule and returns
                # its output.
                return self.sub(x)

        m1 = torch.jit.script(ParentModule())
        self.assertFalse(m1.sub.customized_save_state_dict_called)
        state_dict = m1.state_dict()
        self.assertTrue(m1.sub.customized_save_state_dict_called)

        m2 = torch.jit.script(ParentModule())
        self.assertFalse(m2.sub.customized_load_state_dict_called)
        m2.load_state_dict(state_dict)
        self.assertTrue(m2.sub.customized_load_state_dict_called)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
