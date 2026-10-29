# Owner(s): ["oncall: export"]
# [MODULE DOC] Reruns the export test classes with every export call replaced by an export
# followed by save to an in-memory buffer and load. Import time: builds a strict and a non-
# strict variant of each listed test class and registers them in this module's globals so the
# runner discovers them.

import io


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export, load, save


test_classes = {}


def mocked_serder_export_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement for the export function that round-trips through
    # serialization. Exports with strict mode on unless the caller passed a strict setting,
    # saves the program to an in-memory byte buffer, rewinds, loads it back and returns the
    # loaded program.
    if "strict" not in kwargs:
        ep = export(*args, **kwargs, strict=True)
    else:
        ep = export(*args, **kwargs)

    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def mocked_serder_export_non_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Same round trip as the strict variant, but forwards the arguments
    # to export unchanged (so the default strictness applies) before saving to an in-memory
    # buffer and loading the program back. Returns the loaded program.
    ep = export(*args, **kwargs)
    buffer = io.BytesIO()
    save(ep, buffer)
    buffer.seek(0)
    loaded_ep = load(buffer)
    return loaded_ep


def make_dynamic_cls(cls, strict):
    # [FUNCTION DOC] HELPER: Builds a copy of a test class whose export calls are replaced by
    # the serialize-and-reload variant (strict or non-strict, chosen by the flag), using the
    # matching name prefix, test suffix and expected-failure marker. Registers the new class in
    # the module level mapping of test classes and in the module globals so the test runner
    # finds it, and sets its module name to this module. Returns nothing.
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "SerDesExport",
            test_export.SERDES_STRICT_SUFFIX,
            mocked_serder_export_strict,
            xfail_prop="_expected_failure_serdes",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "SerDesExportNonStrict",
            test_export.SERDES_NON_STRICT_SUFFIX,
            mocked_serder_export_non_strict,
            xfail_prop="_expected_failure_serdes_non_strict",
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


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
