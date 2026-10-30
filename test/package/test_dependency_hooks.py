# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for the hooks that PackageExporter calls when a module is externed or
# mocked: single hook, multiple hooks and their order, removal through the returned handle, and
# independent extern and mock hooks. Falls back to a non-relative import of the shared base test
# case. Run as a script, it starts the test runner.

from io import BytesIO

from torch.package import PackageExporter
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDependencyHooks(PackageTestCase):
    # [CLASS DOC] Test case for registering extern and mock hooks on an exporter; each test
    # records module names in local sets through hook callbacks and compares the
    # sets afterwards.
    """Dependency management hooks API tests.
    - register_mock_hook()
    - register_extern_hook()
    """

    def test_single_hook(self):
        # [FUNCTION DOC] Registers one extern hook that records module names, externs two
        # modules and saves source importing one of them. Expects the recorded set to hold
        # exactly the imported one.
        buffer = BytesIO()

        my_externs = set()

        def my_extern_hook(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Extern hook; adds the module name to a set captured from
            # the enclosing test. The exporter argument is unused.
            my_externs.add(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])
            exporter.register_extern_hook(my_extern_hook)
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_externs, {"module_a"})

    def test_multiple_extern_hooks(self):
        # [FUNCTION DOC] Registers two extern hooks, the second removing the name the first
        # added. The removal would fail with a missing-key error if the hooks ran in the wrong
        # order, so this also checks call order; the final set must be empty.
        buffer = BytesIO()

        my_externs = set()

        def my_extern_hook(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Extern hook; adds the module name to a set captured from
            # the enclosing test.
            my_externs.add(module_name)

        # This also checks ordering, since `remove()` will fail if the value is not in the set.
        def my_extern_hook2(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Extern hook; removes the module name from the captured set,
            # raising a key error if it is absent (so it only works when run after the
            # first hook).
            my_externs.remove(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])
            exporter.register_extern_hook(my_extern_hook)
            exporter.register_extern_hook(my_extern_hook2)
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_externs, set())

    def test_multiple_mock_hooks(self):
        # [FUNCTION DOC] Mock-hook counterpart of the multiple extern hooks test: a second mock
        # hook removes the name that the first added, which checks call order; the final set
        # must be empty.
        buffer = BytesIO()

        my_mocks = set()

        def my_mock_hook(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Mock hook; adds the module name to a set captured from the
            # enclosing test.
            my_mocks.add(module_name)

        # This also checks ordering, since `remove()` will fail if the value is not in the set.
        def my_mock_hook2(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Mock hook; removes the module name from the captured set,
            # raising a key error if it is absent.
            my_mocks.remove(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.mock(["package_a.subpackage", "module_a"])
            exporter.register_mock_hook(my_mock_hook)
            exporter.register_mock_hook(my_mock_hook2)
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_mocks, set())

    def test_remove_hooks(self):
        # [FUNCTION DOC] Registers two extern hooks, then removes the first one through its
        # handle before saving. Expects only the second hook to have recorded the
        # externed module.
        buffer = BytesIO()

        my_externs = set()
        my_externs2 = set()

        def my_extern_hook(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Extern hook; adds the module name to the first captured
            # set. It is removed before use, so it should never run.
            my_externs.add(module_name)

        def my_extern_hook2(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Extern hook; adds the module name to the second
            # captured set.
            my_externs2.add(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])
            handle = exporter.register_extern_hook(my_extern_hook)
            exporter.register_extern_hook(my_extern_hook2)
            handle.remove()
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_externs, set())
        self.assertEqual(my_externs2, {"module_a"})

    def test_extern_and_mock_hook(self):
        # [FUNCTION DOC] Externs one module and mocks another, registers one hook of each kind,
        # and saves source importing both. Each hook must see only the name of its own kind
        # of module.
        buffer = BytesIO()

        my_externs = set()
        my_mocks = set()

        def my_extern_hook(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Extern hook; adds the module name to a captured set of
            # externed names.
            my_externs.add(module_name)

        def my_mock_hook(package_exporter, module_name):
            # [FUNCTION DOC] HELPER: Mock hook; adds the module name to a captured set of
            # mocked names.
            my_mocks.add(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern("module_a")
            exporter.mock("package_a")
            exporter.register_extern_hook(my_extern_hook)
            exporter.register_mock_hook(my_mock_hook)
            exporter.save_source_string("foo", "import module_a; import package_a")

        self.assertEqual(my_externs, {"module_a"})
        self.assertEqual(my_mocks, {"package_a"})


if __name__ == "__main__":
    run_tests()
