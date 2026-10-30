# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for forward hooks and forward pre-hooks on TorchScript modules. Fixture
# modules come from test_hooks_modules. Most tests script a prepared module and compare with
# eager execution; others check the error messages raised when hooks have wrong signatures,
# clashing names, or are bypassed by calling forward directly. Not runnable directly: running
# the file as a script raises an error that points to the umbrella test runner.

import os
import sys
import unittest
from typing import Tuple

import torch
from jit.test_hooks_modules import (
    create_forward_tuple_input,
    create_module_forward_multiple_inputs,
    create_module_forward_single_input,
    create_module_hook_return_nothing,
    create_module_multiple_hooks_multiple_inputs,
    create_module_multiple_hooks_single_input,
    create_module_no_forward_input,
    create_module_same_hook_repeated,
    create_submodule_forward_multiple_inputs,
    create_submodule_forward_single_input,
    create_submodule_forward_single_input_return_not_tupled,
    create_submodule_hook_return_nothing,
    create_submodule_multiple_hooks_multiple_inputs,
    create_submodule_multiple_hooks_single_input,
    create_submodule_no_forward_input,
    create_submodule_same_hook_repeated,
    create_submodule_to_call_directly_with_hooks,
    ModuleDirectforwardSubmodCall,
    ModuleForwardSingleInput,
    ModuleForwardTupleInput,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests for JIT forward hooks and pre-hooks
class TestHooks(JitTestCase):
    # [CLASS DOC] JitTestCase with tests of module-level and submodule-level hooks and
    # pre-hooks.
    def test_module_no_forward_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Module hooks
        # with a forward that takes no inputs.
        self.checkModule(create_module_no_forward_input(), ())

    def test_submodule_no_forward_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Submodule hooks
        # with a forward that takes no inputs.
        self.checkModule(create_submodule_no_forward_input(), ())

    def test_module_forward_multiple_inputs(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Module-level
        # hooks, forward with a list and a string input.
        self.checkModule(
            create_module_forward_multiple_inputs(), (["a"], "no_pre_hook")
        )

    def test_module_multiple_hooks_multiple_inputs(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Two pre-hooks
        # and two hooks on the outer module with list and string inputs.
        self.checkModule(
            create_module_multiple_hooks_multiple_inputs(), (["a"], "no_pre_hook")
        )

    def test_module_forward_single_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Module-level
        # hooks, forward with one string input.
        self.checkModule(create_module_forward_single_input(), ("a",))

    def test_module_same_hook_repeated(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. The same hook
        # function registered twice on the outer module.
        self.checkModule(create_module_same_hook_repeated(), ("a",))

    def test_module_hook_return_nothing(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Module hooks
        # that return nothing.
        self.checkModule(create_module_hook_return_nothing(), ("a",))

    def test_module_multiple_hooks_single_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Several hooks on
        # the outer module with a single string input.
        self.checkModule(create_module_multiple_hooks_single_input(), ("a",))

    def test_submodule_forward_multiple_inputs(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Submodule hooks,
        # forward with a list and a string input.
        self.checkModule(
            create_submodule_forward_multiple_inputs(), (["a"], "no_pre_hook")
        )

    def test_submodule_multiple_hooks_multiple_inputs(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Several
        # submodule hooks with multiple forward inputs.
        self.checkModule(
            create_submodule_multiple_hooks_multiple_inputs(),
            (["a"], "no_pre_hook"),
        )

    def test_submodule_forward_single_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Submodule hooks,
        # forward with a single string input.
        self.checkModule(create_submodule_forward_single_input(), ("a",))

    def test_submodule_called_directly_with_hooks(self):
        # [FUNCTION DOC] Scripts a module whose submodule has hooks, then asserts that calling
        # the original submodule directly and calling the scripted submodule directly with the
        # same string give equal results (hooks run in both).
        module = create_submodule_to_call_directly_with_hooks()
        module_scripted = torch.jit.script(module)

        submodule = module.submodule
        scripted_submodule = module_scripted.submodule

        self.assertEqual(submodule("a"), scripted_submodule("a"))

    def test_submodule_same_hook_repeated(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. The same hook
        # function registered twice on the submodule.
        self.checkModule(create_submodule_same_hook_repeated(), ("a",))

    def test_submodule_hook_return_nothing(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Submodule hooks
        # that return nothing.
        self.checkModule(create_submodule_hook_return_nothing(), ("a",))

    def test_submodule_multiple_hooks_single_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Several
        # submodule hooks with a single input (the argument is just a list holding one string).
        self.checkModule(create_submodule_multiple_hooks_single_input(), (["a"]))

    def test_forward_tuple_input(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Forward takes
        # one tuple input, which makes pre-hook return wrapping special.
        self.checkModule(create_forward_tuple_input(), ((3,),))

    def test_submodule_forward_single_input_return_not_tupled(self):
        # [FUNCTION DOC] Uses checkModule, which scripts the module, runs eager and scripted
        # versions on the arguments, compares outputs and checks export/import. Submodule pre-
        # hook returning a bare string rather than a tuple.
        self.checkModule(
            create_submodule_forward_single_input_return_not_tupled(), ("a",)
        )

    def test_hook_method_name_collision(self):
        # [FUNCTION DOC] Registers on the submodule a pre-hook whose name equals an existing
        # method of the submodule class, then asserts that scripting raises a RuntimeError
        # saying a method or hook with that name already exists.
        # Hooks can't have the same name as methods.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def foo(self, input: Tuple[str]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook named like a submodule method. Asserts the
            # submodule name and first input, returns a one-string tuple. Never executed because
            # scripting fails first.
            assert self.name == "inner_mod_name"  # noqa: S101
            assert input[0] == "a_outermod"  # noqa: S101
            return ("pre_hook_override_name",)

        m.submodule.register_forward_pre_hook(foo)

        with self.assertRaisesRegex(
            RuntimeError,
            "Can't define hook: foo on class: .+ "
            "because a method or hook with that name already exists.",
        ):
            torch.jit.script(m)

    def test_hook_hook_name_collision(self):
        # [FUNCTION DOC] Registers two different pre-hook functions that share one name, expects
        # scripting to raise a RuntimeError about two differing python definitions. Repeats this
        # for forward hooks whose definitions differ in their parameters.
        # Test edge case of two hooks sharing name but not python definition
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def prehook(self, input: Tuple[str]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: First pre-hook sharing the name; returns a fixed string
            # (annotated as a tuple of one string). Never run.
            return "This is the first hook"

        m.submodule.register_forward_pre_hook(prehook)

        def prehook(self, input: Tuple[str]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Second pre-hook with the same name but different body,
            # returning a different fixed string. Never run.
            return "This is the second hook"

        m.submodule.register_forward_pre_hook(prehook)

        with self.assertRaisesRegex(
            RuntimeError,
            "Pre-hook '.+' on .+ has at least two different python "
            "definitions. Please use unique names for all hooks.",
        ):
            torch.jit.script(m)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def hook(self, input: Tuple[str], output: str):
            # [FUNCTION DOC] HELPER: First forward hook sharing the name; takes input and output
            # and returns a fixed string. Never run.
            return "This is the first hook"

        m.submodule.register_forward_hook(hook)

        def hook(self, input: Tuple[str]):
            # [FUNCTION DOC] HELPER: Second forward hook with the same name but without the
            # output parameter. Never run.
            return "This is the second hook"

        m.submodule.register_forward_hook(hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "Hook '.+' on .+ has at least two different python "
            "definitions. Please use unique names for all hooks.",
        ):
            torch.jit.script(m)

    def test_module_direct_forward_invocation(self):
        # [FUNCTION DOC] Registers a pre-hook and a forward hook on the outer module, scripts it
        # and asserts that eager forward called directly equals scripted forward called directly
        # (no hooks run), while calling the scripted module normally gives a different result.
        # Test that hooks are only invoked when the module is
        # called directly and not when forward is called.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook replacing the input with a fixed one-string tuple.
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            # [FUNCTION DOC] HELPER: Forward hook asserting the module name and the replaced
            # input, then appending a suffix to the output string and returning it.
            assert self.name == "outer_mod_name"  # noqa: S101
            assert input == ("pre_hook_override_name",)  # noqa: S101
            output = output + "_fh"
            return output

        m.register_forward_pre_hook(pre_hook)
        m.register_forward_hook(forward_hook)

        m_scripted = torch.jit.script(m)

        self.assertEqual(m.forward("a"), m_scripted.forward("a"))
        self.assertNotEqual(m_scripted("a"), m_scripted.forward("a"))

    def test_submodule_direct_forward_invocation(self):
        # [FUNCTION DOC] Builds one outer module that calls its submodule's forward method
        # directly and one that calls the submodule normally, registers the same pre-hook and
        # forward hook on both submodules, and scripts both. Asserts the direct-forward module
        # matches eager (no hooks) and differs from the normally-called one (hooks ran).
        m_submod_forward_call = ModuleDirectforwardSubmodCall(
            "outer_mod_name", "inner_mod_name"
        )
        m_submod_call = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook replacing the input with a fixed one-string tuple.
            return ("pre_hook_override_name",)

        def forward_hook(self, input: Tuple[str], output: str):
            # [FUNCTION DOC] HELPER: Forward hook asserting the replaced input and returning the
            # output with a suffix added.
            assert input == ("pre_hook_override_name",)  # noqa: S101
            return output + "_fh"

        m_submod_forward_call.submodule.register_forward_pre_hook(pre_hook)
        m_submod_forward_call.submodule.register_forward_hook(forward_hook)
        m_submod_call.submodule.register_forward_pre_hook(pre_hook)
        m_submod_call.submodule.register_forward_hook(forward_hook)

        m_submod_forward_call_scripted = torch.jit.script(m_submod_forward_call)
        m_submod_call_scripted = torch.jit.script(m_submod_call)

        self.assertEqual(
            m_submod_forward_call_scripted("a"), m_submod_forward_call("a")
        )
        self.assertNotEqual(
            m_submod_forward_call_scripted("a"), m_submod_call_scripted("a")
        )

    # TODO: add this test back once figured out how to print error msg
    @unittest.skip
    def test_hook_compilation_hint(self):
        # [FUNCTION DOC] Registers on the outer module a pre-hook that indexes out of range and
        # expects scripting to raise an error mentioning the pre-hook name. Skipped.
        # KNOWN ISSUE: skipped with a TODO because the hook error hint is not printed yet, so
        # this test never runs.
        # Tests if hook error message is printed out if erroring after schema check.
        # Useful for when user is scripting hooks while not aware of it.
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook that asserts the module name and an out-of-range
            # tuple element and returns a one-string tuple. Never run.
            assert self.name == "outer_mod_name"  # noqa: S101
            assert input[4] == "a"  # noqa: S101 out of bounds tuple range
            return ("pre_hook_override_name",)

        m.register_forward_pre_hook(pre_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "This error occurred while scripting the forward pre-hook 'pre_hook'",
        ):
            torch.jit.script(m)

    def test_wrong_pre_hook_signatures(self):
        # [FUNCTION DOC] For each of six defective pre-hooks, registers it on a fresh module and
        # asserts scripting raises a RuntimeError with the matching message: wrong inner tuple
        # type, an extra parameter, a non-tuple input type, wrong return type, missing return
        # annotation, and an invalid return for a module whose forward takes a single tuple.
        # correct signature: pre_hook_c(self, input: Tuple[str])
        def pre_hook_wrong_input1(self, input: Tuple[None]) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook whose input is annotated as a tuple of None
            # instead of the string forward takes; returns a one-string tuple. Never run.
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the input tuple argument",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input2(self, input: Tuple[str], input2: str) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook with an extra second parameter; returns a one-
            # string tuple. Never run.
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "was expected to only have exactly 2 inputs but it had 3 inputs",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_input3(self, input: int) -> Tuple[str]:
            # [FUNCTION DOC] HELPER: Pre-hook whose input is annotated as a plain int instead of
            # a tuple. Never run.
            return ("hello",)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the input argument to be typed as a Tuple but"
            " found type: 'int' instead",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_output(self, input: Tuple[str]) -> int:
            # [FUNCTION DOC] HELPER: Pre-hook annotated to return an int, which is not an
            # allowed pre-hook return type; returns the number one. Never run.
            return 1  # expecting Tuple[str], str, or None

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_output)

        with self.assertRaisesRegex(
            RuntimeError,
            "returned the wrong type of: 'int'",
        ):
            torch.jit.script(m)

        def pre_hook_no_output_annotation(self, input: Tuple[str]):
            # [FUNCTION DOC] HELPER: Pre-hook with no return annotation; returns the number one.
            # Never run.
            return 1  # expecting Tuple[str], str, or None

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_no_output_annotation)

        with self.assertRaisesRegex(
            RuntimeError,
            "is missing a return annotation. Return annotations"
            " are required, please add one.",
        ):
            torch.jit.script(m)

        def pre_hook_wrong_tuple_return(self, input: Tuple[Tuple[int]]) -> Tuple[int]:
            # [FUNCTION DOC] HELPER: Pre-hook for a module whose forward takes one tuple;
            # returns a flat one-int tuple instead of the required nested tuple. Never run.
            return (11,)  # doesn't work with eager, inner tuple lost

        m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_pre_hook(pre_hook_wrong_tuple_return)

        with self.assertRaisesRegex(
            RuntimeError,
            "When forward has a single tuple input argument, "
            "the return needs to be 'None' or a nested tuple containing "
            r"forward's input tuple argument as in: 'Tuple\[Tuple\[int\]\]'",
        ):
            torch.jit.script(m)

    def test_wrong_hook_signatures(self):
        # [FUNCTION DOC] For each of five defective forward hooks (input tuple with too many
        # items, non-tuple input, input tuple of None, output annotated with the wrong type, and
        # a hook after a hook that changed the output type), asserts scripting raises a
        # RuntimeError with the matching message.
        # correct signature:
        #   def forward_hook(self, input: Tuple[str], output: str)
        def forward_hook_wrong_input1(self, input: Tuple[str, str], output: str):
            # [FUNCTION DOC] HELPER: Hook whose input is annotated as a tuple of two strings;
            # returns the output. Never run.
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input1)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong number of contained types for the "
            r"input argument's Tuple. Received type: 'Tuple\[str, str\]'",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input2(self, input: str, output: str):
            # [FUNCTION DOC] HELPER: Hook whose input is annotated as a plain string; returns
            # the output. Never run.
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input2)

        with self.assertRaisesRegex(
            RuntimeError,
            "expected the input argument to be typed as a Tuple "
            "but found type: 'str' instead.",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_input3(self, input: Tuple[None], output: str):
            # [FUNCTION DOC] HELPER: Hook whose input is annotated as a tuple of None; returns
            # the output. Never run.
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_input3)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the input tuple"
            r" argument. Received type: 'Tuple\[NoneType\]'",
        ):
            torch.jit.script(m)

        def forward_hook_wrong_output(self, input: Tuple[str], output: Tuple[str]):
            # [FUNCTION DOC] HELPER: Hook whose output is annotated as a tuple of string though
            # forward returns a string; returns the output. Never run.
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_wrong_output)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. Received"
            r" type: 'Tuple\[str\]'. Expected type: 'str'",
        ):
            torch.jit.script(m)

        def forward_hook_correct(self, input: Tuple[str], output: str):
            # [FUNCTION DOC] HELPER: Valid hook that wraps the output string in a one-element
            # tuple; registered first so the next hook sees a tuple. Never run.
            return (output,)

        def forward_hook_wrong_output_from_prev_hook(
            self, input: Tuple[str], output: str
        ):
            # [FUNCTION DOC] HELPER: Second hook annotating the output as a string though the
            # previous hook now returns a tuple. Returns the output. Never run.
            return output

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        m.register_forward_hook(forward_hook_correct)
        m.register_forward_hook(forward_hook_wrong_output_from_prev_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. "
            r"Received type: 'str'. Expected type: 'Tuple\[str\]'",
        ):
            torch.jit.script(m)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
