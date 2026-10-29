# [MODULE DOC] Test module that re-runs the unflatten tests in test_unflatten under a mocked
# export entry point that always uses strict export. At import time it builds one derived test
# class per base class listed in the tests list, registers it in the test_classes mapping and
# in the module globals so the runner discovers it, and then deletes the loop variable. Run
# directly, it calls the dynamo test_case runner.
# Owner(s): ["oncall: export"]


try:
    from . import test_unflatten, testing
except ImportError:
    import test_unflatten  # @manual=fbcode//caffe2/test:test_export-library
    import testing  # @manual=fbcode//caffe2/test:test_export-library

from torch.export import export


test_classes = {}


def mocked_training_ir_export(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement export function used by the generated classes.
    # Forwards all positional and keyword arguments to torch.export.export while forcing
    # strict mode on, and returns the resulting ExportedProgram. Passing strict explicitly in
    # the keyword arguments would raise a duplicate keyword TypeError.
    return export(*args, **kwargs, strict=True)


def make_dynamic_cls(cls):
    # [FUNCTION DOC] HELPER: Builds a derived test class from a base test class using
    # testing.make_test_cls_with_mocked_export, with name prefix TrainingIRUnflatten, suffix
    # _training_ir, the strict export mock, and the attribute _expected_failure_training_ir
    # used to mark expected failures. Side effects: stores the class in test_classes,
    # publishes it in module globals (required for test discovery), and sets its __module__ to
    # this module. Returns nothing.
    cls_prefix = "TrainingIRUnflatten"

    test_class = testing.make_test_cls_with_mocked_export(
        cls,
        cls_prefix,
        "_training_ir",
        mocked_training_ir_export,
        xfail_prop="_expected_failure_training_ir",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_unflatten.TestUnflatten,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
