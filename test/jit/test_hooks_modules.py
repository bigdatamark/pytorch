# Owner(s): ["oncall: jit"]
# [MODULE DOC] Library of fixture modules and factory functions for forward hook and pre-hook
# tests. Used by the Python hook tests and by test/jit_hooks/model.py. Each factory builds a
# small module, registers hooks on it or its submodule that assert expected names and inputs,
# and returns it. No import-time side effects; running the file directly raises a RuntimeError
# saying it is a utility collection.

from typing import List, Tuple

import torch


class SubmoduleNoForwardInputs(torch.nn.Module):
    # [CLASS DOC] HELPER: Submodule with a name attribute and a forward taking no inputs that
    # raises an AssertionError unless the name is the inner module name.
    def __init__(self, name):
        # [FUNCTION DOC] HELPER: Calls the base initializer and stores the given name.
        super().__init__()
        self.name = name

    def forward(self):
        # [FUNCTION DOC] HELPER: Raises an AssertionError unless the stored name is the inner
        # module name; returns nothing.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")


class ModuleNoForwardInputs(torch.nn.Module):
    # [CLASS DOC] HELPER: Outer module with a name and a submodule of the no-input kind; forward
    # takes no inputs and calls the submodule.
    def __init__(self, name: str, submodule_name: str):
        # [FUNCTION DOC] HELPER: Stores the outer name and builds the submodule from the
        # submodule name.
        super().__init__()
        self.name = name
        self.submodule = SubmoduleNoForwardInputs(submodule_name)

    def forward(self):
        # [FUNCTION DOC] HELPER: Calls the submodule with no arguments.
        self.submodule()


class SubmoduleForwardSingleInput(torch.nn.Module):
    # [CLASS DOC] HELPER: Submodule with a name whose forward transforms one string and also has
    # a plain method foo returning its argument.
    def __init__(self, name):
        # [FUNCTION DOC] HELPER: Calls the base initializer and stores the given name.
        super().__init__()
        self.name = name

    def foo(self, input: str):
        # [FUNCTION DOC] HELPER: Returns the given string unchanged.
        return input

    def forward(self, input: str):
        # [FUNCTION DOC] HELPER: Appends an inner-module suffix to the input string, passes it
        # through foo and returns the result.
        input = input + "_inner_mod"
        input = self.foo(input)
        return input


class ModuleForwardSingleInput(torch.nn.Module):
    # [CLASS DOC] HELPER: Outer module with a name and a single-input submodule; forward appends
    # an outer suffix to a string and returns the result of calling the submodule (so submodule
    # hooks fire).
    def __init__(self, name: str, submodule_name: str):
        # [FUNCTION DOC] HELPER: Stores the outer name and builds the single-input submodule
        # from the submodule name.
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        # [FUNCTION DOC] HELPER: Appends an outer suffix to the string and returns the submodule
        # called on it.
        input = input + "_outermod"
        return self.submodule(input)


class ModuleDirectforwardSubmodCall(torch.nn.Module):
    # [CLASS DOC] HELPER: Same as the single-input outer module but forward calls the
    # submodule's forward method directly, so submodule hooks do not run.
    def __init__(self, name: str, submodule_name: str):
        # [FUNCTION DOC] HELPER: Stores the outer name and builds the single-input submodule
        # from the submodule name.
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        # [FUNCTION DOC] HELPER: Appends an outer suffix to the string and returns the result of
        # the submodule's forward method.
        input = input + "_outermod"
        return self.submodule.forward(input)


class SuboduleForwardMultipleInputs(torch.nn.Module):
    # [CLASS DOC] HELPER: Submodule with a name whose forward takes a list of strings and a
    # string (class name is misspelled in the source).
    def __init__(self, name):
        # [FUNCTION DOC] HELPER: Calls the base initializer and stores the given name.
        super().__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        # [FUNCTION DOC] HELPER: Appends the module name to the list in place, builds the second
        # string with a trailing underscore, and returns the list and that string as a pair.
        input1.append(self.name)
        output2 = input2 + "_"
        return input1, output2


class ModuleForwardMultipleInputs(torch.nn.Module):
    # [CLASS DOC] HELPER: Outer module with a name and a multi-input submodule.
    def __init__(self, name: str, submodule_name: str):
        # [FUNCTION DOC] HELPER: Stores the outer name and builds the multi-input submodule.
        super().__init__()
        self.name = name
        self.submodule = SuboduleForwardMultipleInputs(submodule_name)

    def forward(self, input1: List[str], input2: str):
        # [FUNCTION DOC] HELPER: Appends the outer name to the list in place and returns the
        # submodule called on the list and the string.
        input1.append(self.name)
        return self.submodule(input1, input2)


class SubmoduleForwardTupleInput(torch.nn.Module):
    # [CLASS DOC] HELPER: Submodule with a name whose forward takes a one-int tuple.
    def __init__(self, name):
        # [FUNCTION DOC] HELPER: Calls the base initializer and stores the given name.
        super().__init__()
        self.name = name

    def forward(self, input: Tuple[int]):
        # [FUNCTION DOC] HELPER: Reads the first element of the input tuple (unused) and returns
        # a one-element tuple holding one.
        input_access = input[0]  # noqa: F841
        return (1,)


class ModuleForwardTupleInput(torch.nn.Module):
    # [CLASS DOC] HELPER: Outer module whose forward takes a one-int tuple and passes a new one-
    # element tuple to its submodule.
    def __init__(self, name: str, submodule_name: str):
        # [FUNCTION DOC] HELPER: Stores the outer name and builds the tuple-input submodule.
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardTupleInput(submodule_name)

    def forward(self, input: Tuple[int]):
        # [FUNCTION DOC] HELPER: Reads the first element of the input tuple (unused) and returns
        # the submodule called on a fresh one-element tuple holding one.
        input_access = input[0]  # noqa: F841
        return self.submodule((1,))


# Modules for JIT forward hook and pre-hooks python and cpp tests
def create_module_no_forward_input():
    # [FUNCTION DOC] HELPER: Builds the outer module whose forward and submodule take no inputs,
    # registers on it a pre-hook and a forward hook that raise an AssertionError unless the
    # module name is the outer name, and returns it.
    # Use to test module level hooks with no forward input
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        # [FUNCTION DOC] HELPER: Pre-hook taking the empty input tuple; checks the module name
        # is the outer name.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")

    def forward_hook(self, input: Tuple[()], output: None):
        # [FUNCTION DOC] HELPER: Forward hook taking the empty input tuple and a None output;
        # checks the module name is the outer name.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_submodule_no_forward_input():
    # [FUNCTION DOC] HELPER: Same fixture but the hooks are registered on the submodule and
    # check for the inner name.
    # Use to test submodule level hooks with no forward input
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        # [FUNCTION DOC] HELPER: Pre-hook taking the empty input tuple; checks the module name
        # is the inner name.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")

    def forward_hook(self, input: Tuple[()], output: None):
        # [FUNCTION DOC] HELPER: Forward hook taking the empty input tuple and None output;
        # checks the module name is the inner name.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_module_forward_multiple_inputs():
    # [FUNCTION DOC] HELPER: Builds the multi-input module and registers an outer pre-hook and
    # forward hook; returns it.
    # Use to test module level hooks with forward having multiple
    # inputs and returns
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # [FUNCTION DOC] HELPER: Pre-hook that checks the outer name and that the first input
        # list starts with the string a, then returns a replacement list and string.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "a":
            raise AssertionError(f"Expected input[0][0] 'a', got '{input[0][0]}'")
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        # [FUNCTION DOC] HELPER: Forward hook that checks the outer name and the replaced input
        # list, then returns the output list with a suffix added to the output string.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name', got '{input[0][0]}'"
            )
        output2 = output[1] + "fh"
        return output[0], output2

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_multiple_inputs():
    # [FUNCTION DOC] HELPER: Builds the multi-input module with two chained pre-hooks and two
    # chained forward hooks on the outer module, each checking what the earlier hook produced;
    # returns it.
    # Use to test that module level hooks with multiple inputs execute
    # in correct order and pass correct information between each other
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # [FUNCTION DOC] HELPER: First pre-hook; checks the outer name and original first input,
        # returns a replacement list and string.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "a":
            raise AssertionError(f"Expected input[0][0] 'a', got '{input[0][0]}'")
        return ["pre_hook_override_name"], "pre_hook_override"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # [FUNCTION DOC] HELPER: Second pre-hook; checks it sees the first pre-hook's
        # replacement and returns a further replacement.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name', got '{input[0][0]}'"
            )
        return ["pre_hook_override_name2"], "pre_hook_override"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        # [FUNCTION DOC] HELPER: First forward hook; checks the outer name and the second pre-
        # hook's input, then returns the output list and string with an added suffix.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name2":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name2', got '{input[0][0]}'"
            )
        output2 = output[1] + "fh1"
        return output[0], output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        # [FUNCTION DOC] HELPER: Second forward hook; checks the name, input and that the output
        # string carries the first hook's suffix, then adds another suffix.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name2":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name2', got '{input[0][0]}'"
            )
        if output[1] != "pre_hook_override_fh1":
            raise AssertionError(
                f"Expected output[1] 'pre_hook_override_fh1', got '{output[1]}'"
            )
        output2 = output[1] + "_fh2"
        return output[0], output2

    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)

    return m


def create_module_forward_single_input():
    # [FUNCTION DOC] HELPER: Builds the single-input module with an outer pre-hook and forward
    # hook; returns it.
    # Use to test module level hooks for forward with single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Pre-hook checking outer name and input a, returning a one-
        # string replacement tuple.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "a":
            raise AssertionError(f"Expected input[0] 'a', got '{input[0]}'")
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the outer name and the replaced input,
        # returning the output with a suffix.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_same_hook_repeated():
    # [FUNCTION DOC] HELPER: Builds the single-input module and registers the same pre-hook
    # twice and the same forward hook twice on the outer module; returns it.
    # Use to test module can run same hook multiple times
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Pre-hook checking the outer name and returning a one-string
        # tuple with a suffix appended, so two runs stack.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        input_change = input[0] + "_ph"
        return (input_change,)

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the outer name and that the input carries
        # two suffixes, returning the output with a suffix.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("a_ph_ph",):
            raise AssertionError(f"Expected input ('a_ph_ph',), got {input}")
        output = output + "_fh"
        return output

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_hook_return_nothing():
    # [FUNCTION DOC] HELPER: Builds the single-input module with outer hooks that only check
    # values and return nothing; returns it.
    # Use to test module level hooks that return nothing
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        # [FUNCTION DOC] HELPER: Pre-hook checking the outer name and input a; returns nothing.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "a":
            raise AssertionError(f"Expected input[0] 'a', got '{input[0]}'")

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the outer name and the unchanged input;
        # returns nothing.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("a",):
            raise AssertionError(f"Expected input ('a',), got {input}")

    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_single_input():
    # [FUNCTION DOC] HELPER: Builds the single-input module with two chained pre-hooks and two
    # chained forward hooks on the outer module; returns it.
    # Use to test that modules can run multiple hooks with single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: First pre-hook; checks the original input and returns a
        # replacement tuple.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "a":
            raise AssertionError(f"Expected input[0] 'a', got '{input[0]}'")
        return ("pre_hook_override_name1",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Second pre-hook; checks it sees the first replacement and
        # returns another.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input[0] != "pre_hook_override_name1":
            raise AssertionError(
                f"Expected input[0] 'pre_hook_override_name1', got '{input[0]}'"
            )
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: First forward hook; checks input and output strings and returns
        # a pair of the suffixed output.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output != "pre_hook_override_name2_outermod_inner_mod":
            raise AssertionError(
                f"Expected output 'pre_hook_override_name2_outermod_inner_mod', got '{output}'"
            )
        output = output + "_fh1"
        return output, output

    def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
        # [FUNCTION DOC] HELPER: Second forward hook; takes the pair output, checks the first
        # element and returns it with a further suffix as a single string.
        if self.name != "outer_mod_name":
            raise AssertionError(f"Expected name 'outer_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output[0] != "pre_hook_override_name2_outermod_inner_mod_fh1":
            raise AssertionError(
                f"Expected output[0] 'pre_hook_override_name2_outermod_inner_mod_fh1', got '{output[0]}'"
            )
        output = output[0] + "_fh2"
        return output

    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)

    return m


def create_submodule_forward_multiple_inputs():
    # [FUNCTION DOC] HELPER: Builds the multi-input module with a pre-hook and forward hook on
    # the submodule; returns it.
    # Use to test that submodules can run hooks that have multiple forward inputs
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # [FUNCTION DOC] HELPER: Pre-hook checking the inner name and that the list already
        # holds the outer name; returns a replacement list and string.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0][1] != "outer_mod_name":
            raise AssertionError(
                f"Expected input[0][1] 'outer_mod_name', got '{input[0][1]}'"
            )
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        # [FUNCTION DOC] HELPER: Forward hook checking the inner name and replaced input;
        # returns the output list and the output string with a suffix.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0][0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0][0] 'pre_hook_override_name', got '{input[0][0]}'"
            )
        output2 = output[1] + "fh"
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_multiple_inputs():
    # [FUNCTION DOC] HELPER: Builds the multi-input module with two chained pre-hooks and two
    # chained forward hooks on the submodule; returns it.
    # Use to test that submodules can run multiple hooks with multiple
    # forward inputs
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # [FUNCTION DOC] HELPER: First pre-hook; checks the inner name and the untouched string
        # input, returns replacements.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "no_pre_hook":
            raise AssertionError(f"Expected input[1] 'no_pre_hook', got '{input[1]}'")
        return ["pre_hook_override_name"], "pre_hook_override1"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # [FUNCTION DOC] HELPER: Second pre-hook; checks the first replacement and returns a
        # further one.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "pre_hook_override1":
            raise AssertionError(
                f"Expected input[1] 'pre_hook_override1', got '{input[1]}'"
            )
        return ["pre_hook_override_name"], "pre_hook_override2"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        # [FUNCTION DOC] HELPER: First forward hook; checks the name, input and output string,
        # and returns a triple of the output list and the suffixed string twice.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "pre_hook_override2":
            raise AssertionError(
                f"Expected input[1] 'pre_hook_override2', got '{input[1]}'"
            )
        if output[1] != "pre_hook_override2_":
            raise AssertionError(
                f"Expected output[1] 'pre_hook_override2_', got '{output[1]}'"
            )
        output2 = output[1] + "fh1"
        return output[0], output2, output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str, str]
    ):
        # [FUNCTION DOC] HELPER: Second forward hook; takes the triple output, checks the middle
        # element and returns a pair with a suffixed string.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[1] != "pre_hook_override2":
            raise AssertionError(
                f"Expected input[1] 'pre_hook_override2', got '{input[1]}'"
            )
        if output[1] != "pre_hook_override2_fh1":
            raise AssertionError(
                f"Expected output[1] 'pre_hook_override2_fh1', got '{output[1]}'"
            )
        output2 = output[1] + "_fh2"
        return output[0], output2

    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    return m


def create_submodule_forward_single_input():
    # [FUNCTION DOC] HELPER: Builds the single-input module with a pre-hook and forward hook on
    # the submodule; returns it.
    # Use to test that submodules can run hooks with a single argument
    # passed to forward
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Pre-hook checking the inner name and the input with the outer
        # suffix, returning a one-string tuple.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the inner name and replaced input;
        # returns the output unchanged.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        return output

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_to_call_directly_with_hooks():
    # [FUNCTION DOC] HELPER: Builds the module whose submodule hooks are meant to run when the
    # submodule is called directly; returns it.
    # Use to test that submodules have their hooks invoked when called
    # directly
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Pre-hook checking the inner name and returning a one-
        # string tuple.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        return ("pre_hook_override_name",)

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the inner name and replaced input;
        # returns the output with a suffix.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        return output + "_fh"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_same_hook_repeated():
    # [FUNCTION DOC] HELPER: Builds the single-input module with the same pre-hook and forward
    # hook each registered twice on the submodule; returns it.
    # Use to test that submodules can run same hooks multiple times
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Pre-hook checking the inner name and returning a one-string
        # tuple with a suffix appended.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        changed = input[0] + "_ph"
        return (changed,)

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the inner name and that the input has two
        # suffixes; returns the output with a suffix.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("a_outermod_ph_ph",):
            raise AssertionError(f"Expected input ('a_outermod_ph_ph',), got {input}")
        return output + "_fh"

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_hook_return_nothing():
    # [FUNCTION DOC] HELPER: Builds the single-input module with submodule hooks that only check
    # values and return nothing; returns it.
    # Use to test that submodules can run hooks that return nothing
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        # [FUNCTION DOC] HELPER: Pre-hook checking the inner name and input; returns nothing.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the inner name and input; returns
        # nothing.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("a_outermod",):
            raise AssertionError(f"Expected input ('a_outermod',), got {input}")

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_single_input():
    # [FUNCTION DOC] HELPER: Builds the single-input module with two pre-hooks and two forward
    # hooks chained on the submodule; returns it.
    # Use to test that submodules can run multiple hooks that have a single input
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: First pre-hook; checks the inner name and input, returns a
        # replacement tuple.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")
        return ("pre_hook_override_name",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        # [FUNCTION DOC] HELPER: Second pre-hook; checks the first replacement and
        # returns another.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "pre_hook_override_name":
            raise AssertionError(
                f"Expected input[0] 'pre_hook_override_name', got '{input[0]}'"
            )
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: First forward hook; checks input and output and returns the
        # output with a suffix.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output != "pre_hook_override_name2_inner_mod":
            raise AssertionError(
                f"Expected output 'pre_hook_override_name2_inner_mod', got '{output}'"
            )
        return output + "_fwh1"

    def forward_hook2(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Second forward hook; checks input and the suffixed output and
        # returns the output unchanged.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name2",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name2',), got {input}"
            )
        if output != "pre_hook_override_name2_inner_mod_fwh1":
            raise AssertionError(
                f"Expected output 'pre_hook_override_name2_inner_mod_fwh1', got '{output}'"
            )
        return output

    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    return m


def create_forward_tuple_input():
    # [FUNCTION DOC] HELPER: Builds the tuple-input module with pre-hooks returning a nested
    # tuple on both outer and inner modules and forward hooks on both; returns it.
    # The surrounding comment explains eager unwraps single tuple pre-hook returns, so a wrapper
    # tuple is required.
    # Use to test case where forward is passed a single tuple for input.
    # This is different because eager always wraps pre-hook return arguments
    # in a tuple when the returned pre-hook result isn't a tuple
    # (to allow the result to be passed to another pre-hook if needed).
    # The eager behavior doesn't wrap the single tuple input pre-hook return in a
    # tuple as it should. To get consistent behavior between single tuple inputs and
    # the rest of the possible forward inputs, pre-hooks need to
    # wrap single tuple inputs returns in another tuple. This is
    # enforced by the schema checker.
    m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")

    def pre_hook_outermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # [FUNCTION DOC] HELPER: Outer pre-hook returning a nested tuple holding a one-element
        # tuple of eleven.
        # 'return (11,)' doesn't work with eager, inner tuple lost
        return ((11,),)

    def pre_hook_innermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # [FUNCTION DOC] HELPER: Inner pre-hook returning a nested tuple holding a one-element
        # tuple of twenty-two.
        # 'return (22,)' doesn't work with eager, inner tuple lost
        return ((22,),)

    def forward_hook_outermod(self, input: Tuple[Tuple[int]], output: int):
        # [FUNCTION DOC] HELPER: Outer forward hook ignoring its inputs and returning a one-
        # element tuple of eleven.
        return (11,)

    def forward_hook_innermod(self, input: Tuple[Tuple[int]], output: Tuple[int]):
        # [FUNCTION DOC] HELPER: Inner forward hook ignoring its inputs and returning the
        # integer twenty-two.
        return 22

    m.register_forward_pre_hook(pre_hook_outermod)
    m.submodule.register_forward_pre_hook(pre_hook_innermod)
    m.register_forward_hook(forward_hook_outermod)
    m.submodule.register_forward_hook(forward_hook_innermod)

    return m


def create_submodule_forward_single_input_return_not_tupled():
    # [FUNCTION DOC] HELPER: Builds the single-input module whose submodule pre-hook returns a
    # bare string rather than a tuple, plus a forward hook; returns it.
    # Use to check that submodules can return modified inputs
    # that aren't wrapped in a tuple (to match eager behavior)
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> str:
        # [FUNCTION DOC] HELPER: Pre-hook checking the inner name and input, returning a bare
        # replacement string.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input[0] != "a_outermod":
            raise AssertionError(f"Expected input[0] 'a_outermod', got '{input[0]}'")
        # return is wrapped in tuple in other test cases
        return "pre_hook_override_name"

    def forward_hook(self, input: Tuple[str], output: str):
        # [FUNCTION DOC] HELPER: Forward hook checking the inner name and replaced input;
        # returns the output with a suffix.
        if self.name != "inner_mod_name":
            raise AssertionError(f"Expected name 'inner_mod_name', got '{self.name}'")
        if input != ("pre_hook_override_name",):
            raise AssertionError(
                f"Expected input ('pre_hook_override_name',), got {input}"
            )
        output = output + "_fh"
        return output

    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


if __name__ == "__main__":
    raise RuntimeError(
        "This file is a collection of utils, it should be imported not executed directly"
    )
