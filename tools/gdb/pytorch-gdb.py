# [MODULE DOC]
# GDB extension script for debugging PyTorch. Defines user commands torch-tensor-repr,
# torch-int-array-ref-repr and torch-dispatch-keyset-repr that call helper functions in the
# torch::gdb C++ namespace of the debugged process to print readable representations of
# tensors, IntArrayRef values and DispatchKeySet values. Import-time side effect: the
# three commands are registered with gdb (must be sourced from inside gdb).
import textwrap
from typing import Any

import gdb  # type: ignore[import]


class DisableBreakpoints:
    # [CLASS DOC] Context manager that disables all enabled gdb breakpoints on entry and
    # re-enables exactly those on exit, so evaluating helper calls cannot trigger
    # breakpoints. State: the list of breakpoints it disabled.
    """
    Context-manager to temporarily disable all gdb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        # [FUNCTION DOC] Records and disables every currently enabled breakpoint.
        self.disabled_breakpoints = []
        for b in gdb.breakpoints():
            if b.enabled:
                b.enabled = False
                self.disabled_breakpoints.append(b)

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        # [FUNCTION DOC] Re-enables the breakpoints disabled on entry. Does not suppress
        # exceptions raised inside the managed block.
        for b in self.disabled_breakpoints:
            b.enabled = True


class TensorRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC] gdb command torch-tensor-repr: prints the Python-level repr of an
    # at::Tensor expression by calling a helper in the debugged process. The class docstring
    # is re-assigned (dedented) as the gdb help text.
    """
    Print a human readable representation of the given at::Tensor.
    Usage: torch-tensor-repr EXP

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, torch-tensor-repr
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    """

    # pyrefly: ignore [bad-argument-type]
    __doc__ = textwrap.dedent(__doc__).strip()

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the torch-tensor-repr user command with gdb, with
        # expression completion.
        gdb.Command.__init__(
            self, "torch-tensor-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        # [FUNCTION DOC] Command handler. Expects exactly one argument, an expression naming
        # a tensor; otherwise prints a usage line and returns. With breakpoints disabled,
        # calls the torch::gdb tensor repr helper in the inferior, prints the returned string,
        # then frees the buffer that helper allocated. Runs code in the debugged process.
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-tensor-repr EXP")
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::tensor_repr({name})")
            print(f"Python-level repr of {name}:")
            print(res.string())
            # torch::gdb::tensor_repr returns a malloc()ed buffer, let's free it
            gdb.parse_and_eval(f"(void)free({int(res)})")


class IntArrayRefRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC] gdb command torch-int-array-ref-repr: prints a readable form of a
    # c10::IntArrayRef expression using a helper in the debugged process.
    """
    Print human readable representation of c10::IntArrayRef
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the torch-int-array-ref-repr user command with gdb.
        gdb.Command.__init__(
            self, "torch-int-array-ref-repr", gdb.COMMAND_USER, gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        # [FUNCTION DOC] Command handler. Needs exactly one expression argument, else prints
        # usage and returns. With breakpoints disabled, calls the torch::gdb int array ref
        # string helper in the inferior and prints the text between the first double quote
        # and the final character of gdb's string rendering of the result.
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-int-array-ref-repr EXP")
            return
        name = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::int_array_ref_string({name})")
            res = str(res)
            print(res[res.find('"') + 1 : -1])


class DispatchKeysetRepr(gdb.Command):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC] gdb command torch-dispatch-keyset-repr: prints a readable form of a
    # c10::DispatchKeySet expression using a helper in the debugged process.
    """
    Print human readable representation of c10::DispatchKeyset
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the torch-dispatch-keyset-repr user command with gdb.
        gdb.Command.__init__(
            self,
            "torch-dispatch-keyset-repr",
            gdb.COMMAND_USER,
            gdb.COMPLETE_EXPRESSION,
        )

    def invoke(self, args: str, from_tty: bool) -> None:
        # [FUNCTION DOC] Command handler. Needs exactly one expression argument, else prints
        # usage and returns. With breakpoints disabled, calls the torch::gdb dispatch keyset
        # string helper in the inferior and prints the text between the first double quote
        # and the final character of gdb's string rendering of the result.
        args = gdb.string_to_argv(args)
        if len(args) != 1:
            print("Usage: torch-dispatch-keyset-repr EXP")
            return
        keyset = args[0]
        with DisableBreakpoints():
            res = gdb.parse_and_eval(f"torch::gdb::dispatch_keyset_string({keyset})")
            res = str(res)
            print(res[res.find('"') + 1 : -1])


TensorRepr()
IntArrayRefRepr()
DispatchKeysetRepr()
