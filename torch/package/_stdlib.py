# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Standard library detection for torch.package. Provides
# is_stdlib_module, which checks the top-level package name against the interpreter's
# list of standard library module names. No import-time side effects.
"""List of Python standard library modules.

Sadly, there is no reliable way to tell whether a module is part of the
standard library except by comparing to a canonical list.

This is taken from https://github.com/PyCQA/isort/tree/develop/isort/stdlibs,
which itself is sourced from the Python documentation.
"""

import sys


def is_stdlib_module(module: str) -> bool:
    # [FUNCTION DOC] Returns True if the top-level component of the dotted module name is a
    # standard library module, False otherwise. Pure; does not import anything.
    base_module = module.partition(".")[0]
    return base_module in _get_stdlib_modules()


def _get_stdlib_modules():
    # [FUNCTION DOC] HELPER: Returns the interpreter's frozenset of standard library module
    # names.
    return sys.stdlib_module_names
