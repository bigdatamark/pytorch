# Owner(s): ["oncall: export"]
# [MODULE DOC] Runs the main export test classes with every export call forced into strict
# (Dynamo-based) mode. At import time it creates, for each listed test class, a generated subclass
# whose tests patch the export function in the test_export module, registers them in the module
# globals so the runner finds them, and records them in a mapping.

try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_strict_export(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function. Forwards all arguments to the real
    # export, adding strict mode unless the caller already passed an explicit strict keyword.
    # Returns the exported program.
    # If user already specified strict, don't make it strict
    if "strict" in kwargs:
        return export(*args, **kwargs)
    return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    # [FUNCTION DOC] HELPER: Builds a strict-export variant of a test class via the testing helper
    # (class name prefix StrictExport, test name suffix from test_export, expected failures taken
    # from the strict xfail marker), stores it in the module-level mapping and module globals
    # (needed or tests will not run), fixes its module attribute and returns it.
    cls_prefix = "StrictExport"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.STRICT_SUFFIX,
        mocked_strict_export,
        xfail_prop="_expected_failure_strict",
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
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
