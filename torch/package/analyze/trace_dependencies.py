# mypy: allow-untyped-defs
# [MODULE DOC] Defines trace_dependencies, which runs a callable under a profiling hook to
# discover the modules of the functions it calls.
import sys
from collections.abc import Callable, Iterable
from typing import Any


__all__ = ["trace_dependencies"]


def trace_dependencies(
    callable: Callable[[Any], Any], inputs: Iterable[tuple[Any, ...]]
) -> list[str]:
    # [FUNCTION DOC] Executes the callable once per entry of inputs (each entry is unpacked as
    # positional arguments) while a global profiler hook records the module of each called
    # Python function. Returns the de-duplicated list of module names (unordered). Side
    # effects: installs and then always clears the global profile hook (even on error, which
    # propagates), so any previously installed profiler is lost.
    """Trace the execution of a callable in order to determine which modules it uses.

    Args:
        callable: The callable to execute and trace.
        inputs: The input to use during tracing. The modules used by 'callable' when invoked by each set of inputs
            are union-ed to determine all modules used by the callable for the purposes of packaging.

    Returns: A list of the names of all modules used during callable execution.
    """
    modules_used = set()

    def record_used_modules(frame, event, arg):
        # [FUNCTION DOC] HELPER: Profiler hook. On call events, looks up the called function's
        # name in the frame globals, then locals, then as a method on the local self object,
        # and adds that object's module attribute to the enclosing set. Ignores other events.
        # KNOWN ISSUE: an object found by name that lacks a module attribute raises
        # AttributeError inside the hook; only the self case guards against missing values.
        # If the event being profiled is not a Python function
        # call, there is nothing to do.
        if event != "call":
            return

        # This is the name of the function that was called.
        name = frame.f_code.co_name
        module = None

        # Try to determine the name of the module that the function
        # is in:
        #   1) Check the global namespace of the frame.
        #   2) Check the local namespace of the frame.
        #   3) To handle class instance method calls, check
        #       the attribute named 'name' of the object
        #       in the local namespace corresponding to "self".
        if name in frame.f_globals:
            module = frame.f_globals[name].__module__
        elif name in frame.f_locals:
            module = frame.f_locals[name].__module__
        elif "self" in frame.f_locals:
            method = getattr(frame.f_locals["self"], name, None)
            module = method.__module__ if method else None

        # If a module was found, add it to the set of used modules.
        if module:
            modules_used.add(module)

    try:
        # Attach record_used_modules as the profiler function.
        sys.setprofile(record_used_modules)

        # Execute the callable with all inputs.
        for inp in inputs:
            callable(*inp)

    finally:
        # Detach the profiler function.
        sys.setprofile(None)

    return list(modules_used)
