# mypy: allow-untyped-defs
# [MODULE DOC] Python-global state of the JIT: the enabled flag (EnabledProxy), the shared Python CompilationUnit, mappings between Python classes and script classes, and weak-key caches of compiled functions and overloads.
# Import-time side effects: reads the PYTORCH_JIT environment variable (may print a message or raise ValueError) and creates the global CompilationUnit.
"""JIT-related state.

This module stores various pieces of Python-global state relating to the JIT.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import os
import weakref
from typing import Any

import torch


class EnabledProxy:
    # [CLASS DOC] Mutable holder of the JIT-enabled boolean so it has reference semantics across importers. State: enabled, initialized from the PYTORCH_JIT environment variable (default True). Truthiness reflects enabled.
    """Stores whether the JIT is enabled or not.

    This is just a wrapper for a bool, so that we get reference semantics
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Sets enabled from the PYTORCH_JIT environment variable via parse_env, defaulting to True.
        self.enabled = self.parse_env(
            "PYTORCH_JIT", True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED"
        )

    def parse_env(self, name, default, true_message, false_message):
        # [FUNCTION DOC] Reads environment variable name. Returns default if unset; True for 1/true/yes, False for 0/false/no (case insensitive). The values 1v and 0v also return True/False after printing true_message or false_message.
        # Raises ValueError for any other value.
        value = os.environ.get(name)
        if value is None:
            return default
        if value.lower() in {"1", "true", "yes"}:
            return True
        elif value.lower() in {"0", "false", "no"}:
            return False
        if value == "1v":
            print(true_message)
            return True
        elif value == "0v":
            print(false_message)
            return False
        raise ValueError(f"Unknown setting of {name}. Try using 0 or 1.")

    def __bool__(self) -> bool:
        # [FUNCTION DOC] Returns whether the JIT is currently enabled.
        return self.enabled


_enabled = EnabledProxy()


def disable() -> None:
    # [FUNCTION DOC] Turns the global JIT-enabled flag off (mutates the shared proxy).
    _enabled.enabled = False


def enable() -> None:
    # [FUNCTION DOC] Turns the global JIT-enabled flag on (mutates the shared proxy).
    _enabled.enabled = True


# The Python CompilationUnit. All functions and modules defined in Python will
# live in here. It's defined in Python because doing in cpp creates static
# destruction order issues.
_python_cu = torch._C.CompilationUnit()


# python class => ScriptClass mapping
_script_classes: dict[type[Any], type[Any]] = {}
_name_to_pyclass: dict[str, type[Any]] = {}


def _add_script_class(python_class, script_class) -> None:
    # [FUNCTION DOC] HELPER: Registers a Python class to script class mapping and the reverse lookup from the script class qualified name to the Python class (global state).
    _script_classes[python_class] = script_class
    _name_to_pyclass[script_class.qualified_name()] = python_class


def _get_script_class(python_class):
    # [FUNCTION DOC] HELPER: Returns the script class registered for a Python class, or None. If the class defines a qualified-name override attribute, the Python class registered under that name is used instead.
    override = getattr(python_class, "_jit_override_qualname", None)
    if override is not None:
        python_class = _get_python_class(override)
    return _script_classes.get(python_class)


def _get_python_class(qualified_name):
    # [FUNCTION DOC] HELPER: Returns the Python class registered for a script class qualified name, or None.
    return _name_to_pyclass.get(qualified_name)


def _clear_class_state() -> None:
    # [FUNCTION DOC] HELPER: Empties both class registration mappings (global state).
    _script_classes.clear()
    _name_to_pyclass.clear()


# Caching: we currently cache compilation of free functions and overloaded functions.
# To cache free functions we hold a weak ref to the function object and
# map to the compiled fn's qualified name.
# To cache overloaded functions we hold a weak ref to the function obj and
# map to all of its overloaded compiled fns.
# In the future we could consider caching more types of objects so that
# aliasing is preserved across separate compilations of the same object.

_jit_caching_layer: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_jit_function_overload_caching: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _try_get_jit_cached_overloads(key):
    # [FUNCTION DOC] HELPER: Looks up cached overload names for a function key and returns the list of compiled functions found in the Python compilation unit, or None if not cached or empty.
    qual_names = _jit_function_overload_caching.get(key, None)
    if qual_names:
        return [_python_cu.find_function(qual_name) for qual_name in qual_names]
    else:
        return None


def _set_jit_overload_cache(key, compiled_fns) -> None:
    # [FUNCTION DOC] HELPER: Caches the qualified names of the compiled overload functions under the weakly held key.
    _jit_function_overload_caching[key] = [fn.qualified_name for fn in compiled_fns]


def _try_get_jit_cached_function(key):
    # [FUNCTION DOC] HELPER: Returns the cached compiled function for key from the compilation unit, or None if absent or if the key opts out through its disable-caching attribute set to True.
    if getattr(key, "__disable_jit_function_caching__", False) is True:
        return None
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        return _python_cu.find_function(qual_name)
    else:
        return None


def _set_jit_function_cache(key, value) -> None:
    # [FUNCTION DOC] HELPER: Caches a compiled free function (qualified name) under the weakly held key. Raises an assertion error if value is not a ScriptFunction.
    # only free functions currently supported
    if not isinstance(value, torch.jit.ScriptFunction):
        raise AssertionError(f"Expected ScriptFunction, got {type(value)}")
    _jit_caching_layer[key] = value.qualified_name
