# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for the importer abstractions in torch.package: the system importer,
# OrderedImporter combinations with a PackageImporter, whichmodule resolution order, and the
# handling of objects without a module attribute. Falls back to a non-relative import of the
# shared base test case. Run as a script, it starts the test runner.

from io import BytesIO

import torch
from torch.package import (
    Importer,
    OrderedImporter,
    PackageExporter,
    PackageImporter,
    sys_importer,
)
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestImporter(PackageTestCase):
    # [CLASS DOC] Test case for Importer and its derived classes, using the fake packages in
    # this directory and in-memory buffers.
    """Tests for Importer and derived classes."""

    def test_sys_importer(self):
        # [FUNCTION DOC] Checks the system importer returns the identical module objects as the
        # regular import statement, for a package and its submodule.
        import package_a
        import package_a.subpackage

        self.assertIs(sys_importer.import_module("package_a"), package_a)
        self.assertIs(
            sys_importer.import_module("package_a.subpackage"), package_a.subpackage
        )

    def test_sys_importer_roundtrip(self):
        # [FUNCTION DOC] Asks the system importer for the module and type names of a class,
        # imports that module, and checks the named attribute is the same class object.
        import package_a
        import package_a.subpackage

        importer = sys_importer
        type_ = package_a.subpackage.PackageASubpackageObject
        module_name, type_name = importer.get_name(type_)

        module = importer.import_module(module_name)
        self.assertIs(getattr(module, type_name), type_)

    def test_single_ordered_importer(self):
        # [FUNCTION DOC] Packages one package, wraps its importer alone in an OrderedImporter,
        # and checks that the module comes from the package (identical to the importer's,
        # different from the regular one) and that a module not packaged
        # raises ModuleNotFoundError.
        import module_a  # noqa: F401
        import package_a

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_module(package_a.__name__)

        buffer.seek(0)
        importer = PackageImporter(buffer)

        # Construct an importer-only environment.
        ordered_importer = OrderedImporter(importer)

        # The module returned by this environment should be the same one that's
        # in the importer.
        self.assertIs(
            ordered_importer.import_module("package_a"),
            importer.import_module("package_a"),
        )
        # It should not be the one available in the outer Python environment.
        self.assertIsNot(ordered_importer.import_module("package_a"), package_a)

        # We didn't package this module, so it should not be available.
        with self.assertRaises(ModuleNotFoundError):
            ordered_importer.import_module("module_a")

    def test_ordered_importer_basic(self):
        # [FUNCTION DOC] Combines the system importer and a package importer in both orders and
        # checks that the first importer in the order wins for the shared module name.
        import package_a

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_module(package_a.__name__)

        buffer.seek(0)
        importer = PackageImporter(buffer)

        ordered_importer_sys_first = OrderedImporter(sys_importer, importer)
        self.assertIs(ordered_importer_sys_first.import_module("package_a"), package_a)

        ordered_importer_package_first = OrderedImporter(importer, sys_importer)
        self.assertIs(
            ordered_importer_package_first.import_module("package_a"),
            importer.import_module("package_a"),
        )

    def test_ordered_importer_whichmodule(self):
        # [FUNCTION DOC] Checks that whichmodule on an OrderedImporter asks the underlying
        # importers in order and takes the first answer, skipping an importer that reports the
        # main module name as a stand-in for not found.
        """OrderedImporter's implementation of whichmodule should try each
        underlying importer's whichmodule in order.
        """

        class DummyImporter(Importer):
            # [CLASS DOC] HELPER: Importer stub with a fixed answer to whichmodule;
            # import_module is not supported.
            def __init__(self, whichmodule_return):
                # [FUNCTION DOC] HELPER: Stores the fixed module name that whichmodule
                # will return.
                self._whichmodule_return = whichmodule_return

            def import_module(self, module_name):
                # [FUNCTION DOC] HELPER: Always raises NotImplementedError; this stub never
                # imports anything.
                raise NotImplementedError

            def whichmodule(self, obj, name):
                # [FUNCTION DOC] HELPER: Ignores its arguments and returns the stored
                # module name.
                return self._whichmodule_return

        class DummyClass:
            # [CLASS DOC] HELPER: Empty class whose instances are passed to whichmodule.
            pass

        dummy_importer_foo = DummyImporter("foo")
        dummy_importer_bar = DummyImporter("bar")
        dummy_importer_not_found = DummyImporter(
            "__main__"
        )  # __main__ is used as a proxy for "not found" by CPython

        foo_then_bar = OrderedImporter(dummy_importer_foo, dummy_importer_bar)
        self.assertEqual(foo_then_bar.whichmodule(DummyClass(), ""), "foo")

        bar_then_foo = OrderedImporter(dummy_importer_bar, dummy_importer_foo)
        self.assertEqual(bar_then_foo.whichmodule(DummyClass(), ""), "bar")

        notfound_then_foo = OrderedImporter(
            dummy_importer_not_found, dummy_importer_foo
        )
        self.assertEqual(notfound_then_foo.whichmodule(DummyClass(), ""), "foo")

    def test_package_importer_whichmodule_no_dunder_module(self):
        # [FUNCTION DOC] Pickles the float16 dtype object (which has no module attribute) into a
        # package, loads it back, repackages the loaded object using the first importer, and
        # loads again. Checks the loaded objects are identical to the original dtype, so the
        # package importer reproduces the default pickler's module search.
        """Exercise corner case where we try to pickle an object whose
        __module__ doesn't exist because it's from a C extension.
        """
        # torch.float16 is an example of such an object: it is a C extension
        # type for which there is no __module__ defined. The default pickler
        # finds it using special logic to traverse sys.modules and look up
        # `float16` on each module (see pickle.py:whichmodule).
        #
        # We must ensure that we emulate the same behavior from PackageImporter.
        my_dtype = torch.float16

        # Set up a PackageImporter which has a torch.float16 object pickled:
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.save_pickle("foo", "foo.pkl", my_dtype)
        buffer.seek(0)

        importer = PackageImporter(buffer)
        my_loaded_dtype = importer.load_pickle("foo", "foo.pkl")

        # Re-save a package with only our PackageImporter as the importer
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.save_pickle("foo", "foo.pkl", my_loaded_dtype)

        buffer2.seek(0)

        importer2 = PackageImporter(buffer2)
        my_loaded_dtype2 = importer2.load_pickle("foo", "foo.pkl")
        self.assertIs(my_dtype, my_loaded_dtype)
        self.assertIs(my_dtype, my_loaded_dtype2)


if __name__ == "__main__":
    run_tests()
