# Owner(s): ["module: dynamo"]

# [MODULE DOC] Re-runs the export test class under a config where the dynamo option
# install_free_tensors is enabled. Builds a derived test class (prefix InlineAndInstall,
# test-name suffix _inline_and_install) and registers it in module globals so the runner
# discovers it. Tests flagged with the expected-failure attribute become expected failures.

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    from . import test_export
except ImportError:
    import test_export


test_classes = {}


def make_dynamic_cls(cls):
    # [FUNCTION DOC] HELPER: Creates a copy of the given test class whose tests run with
    # install_free_tensors patched to True, records it in test_classes and in module globals
    # (needed for test discovery), sets its module to this one, and returns it.
    suffix = "_inline_and_install"

    cls_prefix = "InlineAndInstall"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "install_free_tensors", True),
        xfail_prop="_expected_failure_inline_and_install",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.ExportTests,
]
for test in tests:
    make_dynamic_cls(test)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
