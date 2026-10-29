# Owner(s): ["oncall: export"]
# [MODULE DOC] Runs the main export test classes under a retraceability check: every export call is
# performed, then the resulting exported module is exported again with the same remaining
# arguments, and the second program is what the test sees. At import time it generates strict and
# non-strict variants of the listed test classes and registers them in the module globals.

import inspect


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export
from torch.export.dynamic_shapes import (
    _combine_args_for_tracing,
    _normalize_dynamic_shapes,
)


test_classes = {}


def _dynamic_shapes_for_retrace(mod, args, kwargs, dynamic_shapes):
    # [FUNCTION DOC] HELPER: Converts a dynamic shapes specification into the form needed when
    # re-exporting the program's module. Non-mapping specs are returned unchanged. A mapping is
    # normalized against the module forward signature. If forward has no variadic keyword
    # parameter, the normalized spec lacks it, or the actual bound value for it is not a mapping,
    # returns a tuple of the normalized per-argument specs. Otherwise the spec is combined for
    # tracing via the shared helper, and the result is returned as a tuple if it is a mapping, else
    # as is.
    if not isinstance(dynamic_shapes, dict):
        return dynamic_shapes

    dynamic_shapes = _normalize_dynamic_shapes(dynamic_shapes, mod, args, kwargs)
    signature = inspect.signature(mod.forward)
    var_keyword_name = None
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_name = name
            break
    if var_keyword_name is None or var_keyword_name not in dynamic_shapes:
        return tuple(dynamic_shapes.values())

    bound_args = signature.bind(*args, **(kwargs or {})).arguments
    var_kwargs = bound_args.get(var_keyword_name)
    if not isinstance(var_kwargs, dict):
        return tuple(dynamic_shapes.values())

    _, dynamic_shapes = _combine_args_for_tracing(mod, args, kwargs, dynamic_shapes)
    if isinstance(dynamic_shapes, dict):
        return tuple(dynamic_shapes.values())
    return dynamic_shapes


def _export_args_kwargs(args, kwargs):
    # [FUNCTION DOC] HELPER: Extracts the example positional and keyword inputs from the positional
    # or keyword arguments of an export call (arguments in position one and two, or named args and
    # kwargs). Returns the pair; the keyword inputs are none if absent and the positional inputs
    # default to an empty tuple.
    export_args = args[1] if len(args) > 1 else kwargs.get("args", ())
    export_kwargs = args[2] if len(args) > 2 else kwargs.get("kwargs")
    return export_args, export_kwargs


def mocked_retraceability_export_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function for strict retraceability. Exports in
    # strict mode unless a strict keyword was given, rewrites any dynamic shapes keyword for the
    # retrace (mutating the keyword mapping), then exports the first program's module again with
    # the remaining positional arguments, and returns that second program.
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=True)

    if "dynamic_shapes" in kwargs:
        export_args, export_kwargs = _export_args_kwargs(args, kwargs)
        kwargs["dynamic_shapes"] = _dynamic_shapes_for_retrace(
            args[0], export_args, export_kwargs, kwargs["dynamic_shapes"]
        )

    if "strict" in kwargs:
        ep = export(ep.module(), *(args[1:]), **kwargs)
    else:
        ep = export(ep.module(), *(args[1:]), **kwargs, strict=True)
    return ep


def mocked_retraceability_export_non_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function for non-strict retraceability. Exports
    # with the given arguments, rewrites any dynamic shapes keyword for the retrace (mutating the
    # keyword mapping), re-exports the resulting module with the remaining arguments and returns
    # the second program.
    ep = export(*args, **kwargs)
    if "dynamic_shapes" in kwargs:
        export_args, export_kwargs = _export_args_kwargs(args, kwargs)
        kwargs["dynamic_shapes"] = _dynamic_shapes_for_retrace(
            args[0], export_args, export_kwargs, kwargs["dynamic_shapes"]
        )

    ep = export(ep.module(), *(args[1:]), **kwargs)
    return ep


def make_dynamic_cls(cls, strict):
    # [FUNCTION DOC] HELPER: Builds a retrace variant of a test class with the strict or non-strict
    # mocked export, the matching name prefix, test suffix and expected-failure marker. Registers
    # it in the module mapping and module globals (needed for discovery), fixes its module
    # attribute and returns it.
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "RetraceExport",
            test_export.RETRACEABILITY_STRICT_SUFFIX,
            mocked_retraceability_export_strict,
            xfail_prop="_expected_failure_retrace",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "RetraceExportNonStrict",
            test_export.RETRACEABILITY_NON_STRICT_SUFFIX,
            mocked_retraceability_export_non_strict,
            xfail_prop="_expected_failure_retrace_non_strict",
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test, True)
    make_dynamic_cls(test, False)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
