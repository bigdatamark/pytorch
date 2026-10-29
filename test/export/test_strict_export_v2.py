# Owner(s): ["oncall: export"]
# [MODULE DOC] Reruns the TestDynamismExpression and TestExport suites from test_export with the
# export entry point replaced by a strict-mode variant that disables legacy dynamo graph capture
# (the v2 strict path). Import time: builds the mocked test classes, registers them in a module
# level mapping and in module globals so the test runner discovers them; the loop variable is
# deleted afterwards. Running the file directly starts the dynamo test runner.

try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch._export import config
from torch.export import export


test_classes = {}


def mocked_strict_export_v2(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the export function. Calls export with the given
    # arguments inside a config patch that turns off legacy dynamo graph capture. If the caller
    # passed a strict keyword it is respected; otherwise strict mode is forced on. Returns the
    # exported program; exceptions from export propagate.
    # If user already specified strict, don't make it strict
    with config.patch(use_legacy_dynamo_graph_capture=False):
        if "strict" in kwargs:
            return export(*args, **kwargs)
        return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    # [FUNCTION DOC] HELPER: Builds a copy of the given test class (name prefixed with
    # StrictExportV2) whose test methods run with the mocked strict v2 export and carry the strict
    # v2 name suffix; tests flagged with the strict v2 expected-failure attribute become expected
    # failures. Registers the class in the module mapping and globals (required for discovery),
    # sets its module name to this module, and returns it.
    cls_prefix = "StrictExportV2"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        test_export.STRICT_EXPORT_V2_SUFFIX,
        mocked_strict_export_v2,
        xfail_prop="_expected_failure_strict_v2",
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
