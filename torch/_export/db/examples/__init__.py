# [PACKAGE DOC] Package of ExportDB example modules. Importing it auto-discovers
# every sibling example file, registers each as an ExportCase, and raises
# RuntimeError if any case name conflicts. Exports all_examples,
# filter_examples_by_support_level and get_rewrite_cases.
# mypy: allow-untyped-defs
import dataclasses
import glob
import inspect
from os.path import basename, dirname, isfile, join

import torch
from torch._export.db.case import (
    _EXAMPLE_CASES,
    _EXAMPLE_CONFLICT_CASES,
    _EXAMPLE_REWRITE_CASES,
    SupportLevel,
    export_case,
    ExportCase,
)


def _collect_examples():
    # [FUNCTION DOC] HELPER: import-time discovery. Lists every Python file next
    # to this package file except the package init, imports each as a relative
    # submodule, gathers those module-level variables whose names match
    # ExportCase fields (example_args, tags, model, ...), and registers a case
    # for the module's model via the export_case decorator. Side effects: imports
    # modules and mutates the global case registries.
    case_names = glob.glob(join(dirname(__file__), "*.py"))
    case_names = [
        basename(f)[:-3] for f in case_names if isfile(f) and not f.endswith("__init__.py")
    ]

    case_fields = {f.name for f in dataclasses.fields(ExportCase)}
    for case_name in case_names:
        case = __import__(case_name, globals(), locals(), [], 1)
        variables = [name for name in dir(case) if name in case_fields]
        export_case(**{v: getattr(case, v) for v in variables})(case.model)

_collect_examples()

def all_examples():
    # [FUNCTION DOC] Returns the global mapping from case name to ExportCase
    # (the live registry, not a copy).
    return _EXAMPLE_CASES


if len(_EXAMPLE_CONFLICT_CASES) > 0:

    def get_name(case):
        # [FUNCTION DOC] HELPER: returns the class name of the case's model
        # (using its type if the model is a module instance), for the conflict message.
        model = case.model
        if isinstance(model, torch.nn.Module):
            model = type(model)
        return model.__name__

    msg = "Error on conflict export case name.\n"
    for case_name, cases in _EXAMPLE_CONFLICT_CASES.items():
        msg += f"Case name {case_name} is associated with multiple cases:\n  "
        msg += f"[{','.join(map(get_name, cases))}]\n"

    raise RuntimeError(msg)


def filter_examples_by_support_level(support_level: SupportLevel):
    # [FUNCTION DOC] Returns a new mapping of case name to ExportCase containing
    # only registered cases whose support_level equals the given level.
    return {
        key: val
        for key, val in all_examples().items()
        if val.support_level == support_level
    }


def get_rewrite_cases(case):
    # [FUNCTION DOC] Returns the list of rewrite (fixed-variant) cases registered
    # for the given case's name, or an empty list if there are none.
    return _EXAMPLE_REWRITE_CASES.get(case.name, [])
