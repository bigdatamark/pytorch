# Owner(s): ["oncall: export"]
# [MODULE DOC] Runs the main export test classes with export followed by decomposition. At import
# time it generates, for each listed test class, a strict and a non-strict variant whose tests
# patch export so that every exported program is immediately passed through run_decompositions with
# an empty table, and registers them in the module globals.
import torch


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library

    import testing  # @manual=fbcode//caffe2/test:test_export-library


test_classes = {}


def mocked_training_ir_to_run_decomp_export_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function for the strict variant. Calls the real
    # export in strict mode unless the caller passed a strict keyword, then returns the program
    # after running decompositions with an empty decomposition table.
    if "strict" in kwargs:
        ep = torch.export.export(*args, **kwargs)
    else:
        ep = torch.export.export(*args, **kwargs, strict=True)
    return ep.run_decompositions({})


def mocked_training_ir_to_run_decomp_export_non_strict(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function for the non-strict variant. Calls the real
    # export with the given arguments (default strictness) and returns the result of running
    # decompositions with an empty table.
    ep = torch.export.export(*args, **kwargs)

    return ep.run_decompositions({})


def make_dynamic_cls(cls, strict):
    # [FUNCTION DOC] HELPER: Builds a generated test class from the given one, using either the
    # strict or non-strict mocked export. Class name prefix, test name suffix and expected-failure
    # marker differ by mode. Registers the class in the module-level mapping and module globals
    # (required for discovery), fixes its module attribute and returns it.
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExport",
            test_export.TRAINING_IR_DECOMP_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExportNonStrict",
            test_export.TRAINING_IR_DECOMP_NON_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_non_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp_non_strict",
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
