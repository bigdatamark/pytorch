# Owner(s): ["module: dynamo"]
# [MODULE DOC] Shared helpers for the dynamo tests: a couple of module-level tensors used by
# export and import tests, small functions used as compile targets, a factory for a fake module
# registered in the module cache, and a context manager that swaps a guard testing hook.
# Importing it only creates the module-level tensors and the source text string.
import importlib
import os
import sys
import types
from contextlib import contextmanager

import torch
import torch._dynamo


g_tensor_export = torch.ones(10)


tensor_for_import_testing = torch.ones(10, 10)


def inner_func():
    # [FUNCTION DOC] HELPER: Return whether gradient mode is currently enabled, as a bool. Used
    # as a nested call target that dynamo must trace through.
    return torch.is_grad_enabled()


def outer_func(func):
    # [FUNCTION DOC] HELPER: Decorator-style wrapper factory. Takes a callable and returns a
    # wrapper function around it; the original function is not modified.
    def wrapped(*args):
        # [FUNCTION DOC] HELPER: Call the wrapped function with the given positional arguments,
        # force a graph break, then return a pair of the sine of the result plus one and the
        # result of the grad-mode helper. Requires the wrapped function to return a tensor.
        a = func(*args)
        torch._dynamo.graph_break()
        return torch.sin(a + 1), inner_func()

    return wrapped


# Create a dummy python module and function to test skipfiles rules.
module_code = """
def add(x):
    return x + 1
"""


def add(x):
    # [FUNCTION DOC] HELPER: Return the input plus one. Copied onto the dummy module so that its
    # code object has a real source file name.
    return x + 1


def break_it(x):
    # [FUNCTION DOC] HELPER: Sum the tensor and convert it to a Python number, which causes a
    # data dependent graph break under tracing.
    return x.sum().item()


def create_dummy_module_and_function():
    # [FUNCTION DOC] HELPER: Build a module object named dummy_module whose spec origin is
    # this file, execute the module source text in its namespace, and register it in the
    # interpreter module table (a global side effect that is never undone). Then replace the
    # module's add function with the one defined in this file, because functions created by
    # executing a string have no real file name, which the skip file rules need. Returns the
    # module and its add function.
    module = types.ModuleType("dummy_module")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "dummy_module", None, origin=os.path.abspath(__file__)
    )
    exec(module_code, module.__dict__)
    sys.modules["dummy_module"] = module
    # Need to override the original function since its __code__.co_filename is not a regular python file name,
    # and the skipfiles rules use filename when checking SKIP_DIRS.
    module.add = add
    return module, module.add


@contextmanager
def install_guard_manager_testing_hook(hook_fn):
    # [FUNCTION DOC] HELPER: Context manager that installs the given hook as the guard manager
    # testing hook in the dynamo guards module for the duration of the with block. The previous
    # hook is saved first and restored on exit, even when the body raises. Yields nothing.
    old_value = torch._dynamo.guards.guard_manager_testing_hook_fn
    try:
        torch._dynamo.guards.guard_manager_testing_hook_fn = hook_fn
        yield
    finally:
        torch._dynamo.guards.guard_manager_testing_hook_fn = old_value
