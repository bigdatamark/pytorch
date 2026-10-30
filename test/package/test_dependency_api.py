# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for the dependency-management actions of the torch.package exporter:
# extern, mock, deny and intern (by name and by glob), the empty-match check, error reporting
# for unhandled, C-extension and malformed dependencies, and repackaging. Falls back to a
# non-relative import of the shared base test case. Run as a script, it starts the test runner.

import importlib
from io import BytesIO
from textwrap import dedent
from unittest import skipIf

import torch.nn
from torch.package import EmptyMatchError, Importer, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDependencyAPI(PackageTestCase):
    # [CLASS DOC] Test case exercising the extern, mock and deny exporter actions plus the error
    # messages produced when dependencies are not handled. Fixtures come from the sibling fake
    # packages made importable by the base class.
    """Dependency management API tests.
    - mock()
    - extern()
    - deny()
    """

    def test_extern(self):
        # [FUNCTION DOC] Exports a source string that imports two externed modules, then loads
        # it. Checks that the externed top-level module is the very same object as the regular
        # import, while the parent package is a distinct object whose submodule is still the
        # regular one.
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.extern(["package_a.subpackage", "module_a"])
            he.save_source_string("foo", "import package_a.subpackage; import module_a")
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob(self):
        # [FUNCTION DOC] Same checks as the plain extern test but the extern patterns are globs,
        # with a whole package saved and a second source string importing both. Imported module
        # identity is shared with the regular import for the externed module and submodule, but
        # not for the package itself.
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.extern(["package_a.*", "module_*"])
            he.save_module("package_a")
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob_allow_empty(self):
        # [FUNCTION DOC] Externs a glob that matches nothing that is required, with the
        # empty-match allowance turned off, while saving an unrelated module. Expects the
        # exporter to raise EmptyMatchError whose message says the pattern matched no modules.
        """
        Test that an error is thrown when a extern glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        import package_a.subpackage  # noqa: F401

        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer) as exporter:
                exporter.extern(include=["package_b.*"], allow_empty=False)
                exporter.save_module("package_a.subpackage")

    def test_deny(self):
        # [FUNCTION DOC] Denies a submodule and a module, then saves source importing the denied
        # submodule. Expects a PackagingError whose message mentions the word denied.
        """
        Test marking packages as "deny" during export.
        """
        buffer = BytesIO()

        with self.assertRaisesRegex(PackagingError, "denied"):
            with PackageExporter(buffer) as exporter:
                exporter.deny(["package_a.subpackage", "module_a"])
                exporter.save_source_string("foo", "import package_a.subpackage")

    def test_deny_glob(self):
        # [FUNCTION DOC] Same as the deny test but with glob patterns and a source string
        # importing both matching modules; only checks that a PackagingError is raised.
        """
        Test marking packages as "deny" using globs instead of package names.
        """
        buffer = BytesIO()
        with self.assertRaises(PackagingError):
            with PackageExporter(buffer) as exporter:
                exporter.deny(["package_a.*", "module_*"])
                exporter.save_source_string(
                    "test_module",
                    dedent(
                        """\
                        import package_a.subpackage
                        import module_a
                        """
                    ),
                )

    def test_mock(self):
        # [FUNCTION DOC] Mocks a submodule and a module, exports source importing the submodule,
        # and loads it. The loaded submodule's result attribute is a mock object whose call
        # raises NotImplementedError saying it was mocked out.
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.mock(["package_a.subpackage", "module_a"])
            # Import something that dependso n package_a.subpackage
            he.save_source_string("foo", "import package_a.subpackage")
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage

        _ = package_a.subpackage
        import module_a

        _ = module_a

        m = hi.import_module("package_a.subpackage")
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    def test_mock_glob(self):
        # [FUNCTION DOC] Same as the mock test but patterns are globs and a whole package is
        # saved alongside a source string that imports both; calling the mocked result raises
        # NotImplementedError saying it was mocked out.
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.mock(["package_a.*", "module*"])
            he.save_module("package_a")
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage

        _ = package_a.subpackage
        import module_a

        _ = module_a

        m = hi.import_module("package_a.subpackage")
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    def test_mock_glob_allow_empty(self):
        # [FUNCTION DOC] Mocks a glob that matches no required module with the empty-match
        # allowance turned off while saving an unrelated module; expects EmptyMatchError saying
        # the pattern matched no modules.
        """
        Test that an error is thrown when a mock glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        import package_a.subpackage  # noqa: F401

        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer) as exporter:
                exporter.mock(include=["package_b.*"], allow_empty=False)
                exporter.save_module("package_a.subpackage")

    def test_pickle_mocked(self):
        # [FUNCTION DOC] Builds an object of a package_a class holding an object of a mocked
        # submodule class, then tries to pickle it with the submodule mocked and everything else
        # interned. Expects a PackagingError, since mocked modules cannot be pickled.
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with self.assertRaises(PackagingError):
            with PackageExporter(buffer) as he:
                he.mock(include="package_a.subpackage")
                he.intern("**")
                he.save_pickle("obj", "obj.pkl", obj2)

    def test_pickle_mocked_all(self):
        # [FUNCTION DOC] Pickles an object tree while interning the package_a modules and
        # mocking everything else. Expects no error; the test has no assertions beyond
        # completing the export.
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.intern(include="package_a.**")
            he.mock("**")
            he.save_pickle("obj", "obj.pkl", obj2)

    def test_allow_empty_with_error(self):
        # [FUNCTION DOC] Checks error precedence: an extern with the empty-match allowance off
        # is followed by saving a nonexistent module. The module-not-found error must surface,
        # not the empty-match error, because the with block exits through that exception.
        """If an error occurs during packaging, it should not be shadowed by the allow_empty error."""
        buffer = BytesIO()
        with self.assertRaises(ModuleNotFoundError):
            with PackageExporter(buffer) as pe:
                # Even though we did not extern a module that matches this
                # pattern, we want to show the save_module error, not the allow_empty error.

                pe.extern("foo", allow_empty=False)
                pe.save_module("aodoifjodisfj")  # will error

                # we never get here, so technically the allow_empty check
                # should raise an error. However, the error above is more
                # informative to what's actually going wrong with packaging.
                pe.save_source_string("bar", "import foo\n")

    def test_implicit_intern(self):
        # [FUNCTION DOC] Saves a package with no explicit actions configured and expects
        # success, showing that saving a module implicitly interns it.
        """The save_module APIs should implicitly intern the module being saved."""
        import package_a  # noqa: F401

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_module("package_a")

    def test_intern_error(self):
        # [FUNCTION DOC] Pickling an object whose modules have no action pattern must raise
        # PackagingError with an exact message listing both unmatched modules. Then retries
        # using the same buffer with the modules interned and expects success.
        """Failure to handle all dependencies should lead to an error."""
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as he:
                he.save_pickle("obj", "obj.pkl", obj2)

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Module did not match against any action pattern. Extern, mock, or intern it.
                    package_a
                    package_a.subpackage

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

        # Interning all dependencies should work
        with PackageExporter(buffer) as he:
            he.intern(["package_a", "package_a.subpackage"])
            he.save_pickle("obj", "obj.pkl", obj2)

    @skipIf(IS_WINDOWS, "extension modules have a different file extension on windows")
    def test_broken_dependency(self):
        # [FUNCTION DOC] Uses a custom importer whose modules look like compiled extension files
        # (names ending in the shared object suffix). Interning them and importing both from a
        # source string must raise PackagingError with an exact message saying they are C
        # extension modules. Skipped on Windows.
        """A unpackageable dependency should raise a PackagingError."""

        def create_module(name):
            # [FUNCTION DOC] HELPER: Builds an empty module object with a spec whose loader is
            # the test case and whose file attribute is the name with the shared object suffix,
            # mimicking a C extension. Returns the module.
            spec = importlib.machinery.ModuleSpec(name, self, is_package=False)  # type: ignore[arg-type]
            module = importlib.util.module_from_spec(spec)
            ns = module.__dict__
            ns["__spec__"] = spec
            ns["__loader__"] = self
            ns["__file__"] = f"{name}.so"
            ns["__cached__"] = None
            return module

        class BrokenImporter(Importer):
            # [CLASS DOC] HELPER: Importer serving two prebuilt fake extension modules, named
            # foo and bar, from an in-memory mapping.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the mapping from the names foo and bar to fake
                # extension modules.
                self.modules = {
                    "foo": create_module("foo"),
                    "bar": create_module("bar"),
                }

            def import_module(self, module_name):
                # [FUNCTION DOC] HELPER: Returns the stored fake module for the name; an unknown
                # name raises a key error.
                return self.modules[module_name]

        buffer = BytesIO()

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer, importer=BrokenImporter()) as exporter:
                exporter.intern(["foo", "bar"])
                exporter.save_source_string("my_module", "import foo; import bar")

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Module is a C extension module. torch.package supports Python modules only.
                    foo
                    bar

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

    def test_invalid_import(self):
        # [FUNCTION DOC] Saves source containing a relative import that goes beyond the
        # top-level package. Expects PackagingError with an exact message reporting failed
        # dependency resolution and the relative-import context.
        """An incorrectly-formed import should raise a PackagingError."""
        buffer = BytesIO()
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as exporter:
                # This import will fail to load.
                exporter.save_source_string("foo", "from ........ import lol")

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Dependency resolution failed.
                    foo
                      Context: attempted relative import beyond top-level package

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

    def test_repackage_mocked_module(self):
        # [FUNCTION DOC] Exports a package with a mocked module, loads it and checks calling
        # into the mock raises NotImplementedError. Then repackages using the loaded importer,
        # interning the previously mocked module and mocking the rest, and checks the module is
        # still mocked on reload.
        """Re-packaging a package that contains a mocked module should work correctly."""
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.mock("package_a")
            exporter.save_source_string("foo", "import package_a")

        buffer.seek(0)
        importer = PackageImporter(buffer)
        foo = importer.import_module("foo")

        # "package_a" should be mocked out.
        with self.assertRaises(NotImplementedError):
            foo.package_a.get_something()

        # Re-package the model, but intern the previously-mocked module and mock
        # everything else.
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.intern("package_a")
            exporter.mock("**")
            exporter.save_source_string("foo", "import package_a")

        buffer2.seek(0)
        importer2 = PackageImporter(buffer2)
        foo2 = importer2.import_module("foo")

        # "package_a" should still be mocked out.
        with self.assertRaises(NotImplementedError):
            foo2.package_a.get_something()

    def test_externing_c_extension(self):
        # [FUNCTION DOC] Pickles a transformer encoder layer with a gelu activation (whose
        # function lives in a C extension namespace) while externing torch and interning
        # everything else, then loads the pickle back. Passes if neither step raises.
        """Externing c extensions modules should allow us to still access them especially those found in torch._C."""

        buffer = BytesIO()
        # The C extension module in question is F.gelu which comes from torch._C._nn
        model = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=64,
            dropout=1.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        with PackageExporter(buffer) as e:
            e.extern("torch.**")
            e.intern("**")

            e.save_pickle("model", "model.pkl", model)
        buffer.seek(0)
        imp = PackageImporter(buffer)
        imp.load_pickle("model", "model.pkl")


if __name__ == "__main__":
    run_tests()
