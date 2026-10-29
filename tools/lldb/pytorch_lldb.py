# [MODULE DOC]
# LLDB script providing pretty-printing type summaries for PyTorch C++ types (IntArrayRef,
# DispatchKeySet and Tensor). Summaries evaluate helper expressions of the torch gdb namespace
# inside the debugged process. Main contents: get_target, the DisableBreakpoints context
# manager, the three summary functions and the lldb init hook. Requires the lldb module (only
# available inside lldb). Loading it in lldb registers the summaries via the init hook.
from typing import Any

import lldb  # type: ignore[import]


def get_target() -> Any:
    # [FUNCTION DOC] Returns the currently selected lldb target, or None after printing an error
    # message if no target is selected.
    target = lldb.debugger.GetSelectedTarget()
    if not target:
        print("[-] error: no target available. please add a target to lldb.")
        return None
    return target


class DisableBreakpoints:
    # [CLASS DOC] Context manager that disables all breakpoints of the selected target on entry
    # and re-enables all of them on exit, printing an error line if either step fails. Stateless.
    # Re-enabling is unconditional, so breakpoints that were already disabled become enabled.
    """
    Context-manager to temporarily disable all lldb breakpoints, useful if
    there is a risk to hit one during the evaluation of one of our custom
    commands
    """

    def __enter__(self) -> None:
        # [FUNCTION DOC] Disables all breakpoints on the selected target; prints an error if lldb
        # reports failure.
        # KNOWN ISSUE: fails with an attribute error if no target is selected (None result).
        target = get_target()

        if target.DisableAllBreakpoints() is False:
            print("[-] error: failed to disable all breakpoints.")

    def __exit__(self, etype: Any, evalue: Any, tb: Any) -> None:
        # [FUNCTION DOC] Re-enables all breakpoints on the selected target; prints an error if
        # lldb reports failure. Does not suppress exceptions from the managed block.
        target = get_target()

        if target.EnableAllBreakpoints() is False:
            print("[-] error: failed to enable all breakpoints.")


def IntArrayRef_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    # [FUNCTION DOC] lldb summary provider for an IntArrayRef value. With breakpoints disabled,
    # evaluates the torch gdb helper that stringifies the variable (looked up by the value's
    # name) in the debuggee and returns the text between the first quote and the last character
    # of the result description. The internal_dict and options arguments are unused.
    """Print human readable representation of c10::IntArrayRef"""
    with DisableBreakpoints():
        target = get_target()
        tensor = valobj.GetName()
        result = target.EvaluateExpression(
            f"torch::gdb::int_array_ref_string({tensor})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def DispatchKeyset_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    # [FUNCTION DOC] lldb summary provider for a DispatchKeySet value. Same mechanism as the
    # IntArrayRef summary, using the torch gdb helper that stringifies a dispatch key set.
    # Returns the extracted text. The internal_dict and options arguments are unused.
    """Print human readable representation of c10::DispatchKeyset"""
    with DisableBreakpoints():
        target = get_target()
        keyset = valobj.GetName()
        result = target.EvaluateExpression(
            f"torch::gdb::dispatch_keyset_string({keyset})"
        )
        str_result = str(result)
        str_result = str_result[str_result.find('"') + 1 : -1]
        return str_result


def Tensor_summary(valobj: Any, internal_dict: Any, options: Any) -> str:
    # [FUNCTION DOC] lldb summary provider for an at::Tensor value. With breakpoints disabled,
    # evaluates the torch gdb helper that builds a Python repr string in the debuggee, frees the
    # returned buffer in the debuggee (side effect), and returns a newline followed by the text
    # starting at the word tensor. The internal_dict and options arguments are unused.
    """Print a human readable representation of the given at::Tensor.

    at::Tensor instances do not have a C++ implementation of a repr method: in
    pytorch, this is done by pure-Python code. As such, print <tensor>
    internally creates a Python wrapper for the given tensor and call repr()
    on it.
    Usage:
        print self
    """
    with DisableBreakpoints():
        target = get_target()
        tensor = valobj.GetName()
        result = target.EvaluateExpression(f"torch::gdb::tensor_repr({tensor})")
        str_result = str(result)
        target.EvaluateExpression(f"(void)free({result.GetValue()})")
        str_result = "\n" + str_result[str_result.find("tensor") : -1]
        return str_result


# And the initialization code to add your commands
def __lldb_init_module(debugger: Any, internal_dict: Any) -> Any:
    # [FUNCTION DOC] Hook called by lldb when this module is imported. Registers the three
    # summary providers (IntArrayRef, DispatchKeySet, Tensor) in the torch category and prints
    # an installation notice and usage hint. Returns nothing.
    debugger.HandleCommand(
        "type summary add c10::IntArrayRef -F pytorch_lldb.IntArrayRef_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add c10::DispatchKeySet -F pytorch_lldb.DispatchKeyset_summary -w torch"
    )
    debugger.HandleCommand(
        "type summary add at::Tensor -F pytorch_lldb.Tensor_summary -w torch"
    )
    print(
        "Pretty Printing lldb summary for PyTorch AT types has been installed and is ready for use. "
        "This category is enabled by default. To disable run: `type category disable torch`"
    )
    print(
        "Usage:\n\tprint <at::tensor>\n\tprint <c10::IntArrayRef>\n\tprint <c10::DispatchKeySet>"
    )
