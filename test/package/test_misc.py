# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Miscellaneous tests for torch.package: Directory file-structure views, custom
# importer loaders, python version metadata, exporter content lists, is_from_package, source
# inspection of packaged classes and the torch_package module attribute. Imports the test
# helper PackageTestCase from the sibling common module, or from a top-level import when the
# file is run directly. Runs the tests via run_tests when executed as a script.

import inspect
import os
import platform
import sys
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from unittest import skipIf

from torch.package import is_from_package, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_SANDCASTLE,
    run_tests,
    skipIfTorchDynamo,
)


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestMisc(PackageTestCase):
    # [CLASS DOC] Grab-bag test case deriving from PackageTestCase for one-off package
    # behaviors that do not fit the other package test modules. Holds no state of its own.
    """Tests for one-off or random functionality. Try not to add to this!"""

    def test_file_structure(self):
        # [FUNCTION DOC] Exports a module, a package, a pickled object and a text file into an
        # in-memory buffer, then checks the importer file_structure Directory. The printed
        # tree (minus its first line, which varies by platform) must match expected text for
        # no filter, for an include filter of subpackage and pickle files, and for an exclude
        # filter.
        """
        Tests package's Directory structure representation of a zip file. Ensures
        that the returned Directory prints what is expected and filters
        inputs/outputs correctly.
        """
        buffer = BytesIO()

        export_plain = dedent(
            """\
                \u251c\u2500\u2500 .data
                \u2502   \u251c\u2500\u2500 extern_modules
                \u2502   \u251c\u2500\u2500 python_version
                \u2502   \u251c\u2500\u2500 serialization_id
                \u2502   \u2514\u2500\u2500 version
                \u251c\u2500\u2500 main
                \u2502   \u2514\u2500\u2500 main
                \u251c\u2500\u2500 obj
                \u2502   \u2514\u2500\u2500 obj.pkl
                \u251c\u2500\u2500 package_a
                \u2502   \u251c\u2500\u2500 __init__.py
                \u2502   \u2514\u2500\u2500 subpackage.py
                \u251c\u2500\u2500 byteorder
                \u2514\u2500\u2500 module_a.py
            """
        )
        export_include = dedent(
            """\
                \u251c\u2500\u2500 obj
                \u2502   \u2514\u2500\u2500 obj.pkl
                \u2514\u2500\u2500 package_a
                    \u2514\u2500\u2500 subpackage.py
            """
        )
        import_exclude = dedent(
            """\
                \u251c\u2500\u2500 .data
                \u2502   \u251c\u2500\u2500 extern_modules
                \u2502   \u251c\u2500\u2500 python_version
                \u2502   \u251c\u2500\u2500 serialization_id
                \u2502   \u2514\u2500\u2500 version
                \u251c\u2500\u2500 main
                \u2502   \u2514\u2500\u2500 main
                \u251c\u2500\u2500 obj
                \u2502   \u2514\u2500\u2500 obj.pkl
                \u251c\u2500\u2500 package_a
                \u2502   \u251c\u2500\u2500 __init__.py
                \u2502   \u2514\u2500\u2500 subpackage.py
                \u251c\u2500\u2500 byteorder
                \u2514\u2500\u2500 module_a.py
            """
        )

        with PackageExporter(buffer) as he:
            import module_a
            import package_a
            import package_a.subpackage

            obj = package_a.subpackage.PackageASubpackageObject()
            he.intern("**")
            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
            he.save_pickle("obj", "obj.pkl", obj)
            he.save_text("main", "main", "my string")

        buffer.seek(0)
        hi = PackageImporter(buffer)

        file_structure = hi.file_structure()
        # remove first line from testing because WINDOW/iOS/Unix treat the buffer differently
        self.assertEqual(
            dedent("\n".join(str(file_structure).split("\n")[1:])),
            export_plain,
        )
        file_structure = hi.file_structure(include=["**/subpackage.py", "**/*.pkl"])
        self.assertEqual(
            dedent("\n".join(str(file_structure).split("\n")[1:])),
            export_include,
        )

        file_structure = hi.file_structure(exclude="**/*.storage")
        self.assertEqual(
            dedent("\n".join(str(file_structure).split("\n")[1:])),
            import_exclude,
        )

    def test_loaders_that_remap_files_work_ok(self):
        # [FUNCTION DOC] Checks that a source loader whose reported filename is remapped still
        # lets a module be packaged. Installs a custom meta path finder at the front of the
        # finder list, drops module_a from the module cache, exports it, and asserts the
        # packaged source came from the remapped path. The finally block removes the module
        # and the finder to avoid affecting other tests.
        from importlib.abc import MetaPathFinder
        from importlib.machinery import SourceFileLoader
        from importlib.util import spec_from_loader

        class LoaderThatRemapsModuleA(SourceFileLoader):
            # [CLASS DOC] HELPER: Source file loader that reports a different filename for
            # module_a (same directory, file named module_a_remapped_path) and defers to the
            # normal behavior for all other module names.
            def get_filename(self, name):
                # [FUNCTION DOC] HELPER: Returns the normal filename from the base loader, except
                # for module_a, where it returns a path in the same directory ending in the
                # remapped file name.
                result = super().get_filename(name)
                if name == "module_a":
                    return os.path.join(
                        os.path.dirname(result), "module_a_remapped_path.py"
                    )
                else:
                    return result

        class FinderThatRemapsModuleA(MetaPathFinder):
            # [CLASS DOC] HELPER: Meta path finder that handles only module_a. It locates the real
            # spec by asking the other finders on the path, then swaps its loader for the
            # remapping loader.
            def find_spec(self, fullname, path, target):
                # [FUNCTION DOC] HELPER: Returns None for any name other than module_a. Otherwise
                # walks the other meta path finders (using find_spec, or building a spec from the
                # loader for old-style finders) until one yields a spec. Raises AssertionError if
                # no spec was found or its loader is not a source file loader. Replaces the spec
                # loader with the remapping loader and returns the spec.
                """Try to find the original spec for module_a using all the
                remaining meta_path finders."""
                if fullname != "module_a":
                    return None
                spec = None
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    if hasattr(finder, "find_spec"):
                        spec = finder.find_spec(fullname, path, target=target)
                    elif hasattr(finder, "load_module"):
                        spec = spec_from_loader(fullname, finder)
                    if spec is not None:
                        break
                if spec is None or not isinstance(spec.loader, SourceFileLoader):
                    raise AssertionError(
                        f"Expected SourceFileLoader, got {type(spec.loader) if spec else None}"
                    )
                spec.loader = LoaderThatRemapsModuleA(
                    spec.loader.name, spec.loader.path
                )
                return spec

        sys.meta_path.insert(0, FinderThatRemapsModuleA())
        # clear it from sys.modules so that we use the custom finder next time
        # it gets imported
        sys.modules.pop("module_a", None)
        try:
            buffer = BytesIO()
            with PackageExporter(buffer) as he:
                import module_a

                he.intern("**")
                he.save_module(module_a.__name__)

            buffer.seek(0)
            hi = PackageImporter(buffer)
            self.assertTrue("remapped_path" in hi.get_source("module_a"))
        finally:
            # pop it again to ensure it does not mess up other tests
            sys.modules.pop("module_a", None)
            sys.meta_path.pop(0)

    def test_python_version(self):
        # [FUNCTION DOC] Exports a simple object, reloads the package and checks that the importer
        # reports the same python version string as the running interpreter.
        """
        Tests that the current python version is stored in the package and is available
        via PackageImporter's python_version() method.
        """
        buffer = BytesIO()

        with PackageExporter(buffer) as he:
            from package_a.test_module import SimpleTest

            he.intern("**")
            obj = SimpleTest()
            he.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        hi = PackageImporter(buffer)

        self.assertEqual(hi.python_version(), platform.python_version())

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_load_python_version_from_package(self):
        # [FUNCTION DOC] Loads a prebuilt package file from the package_e test data directory and
        # checks its recorded python version is 3.9.7. Skipped in fbcode and sandcastle because it
        # reads a file.
        """Tests loading a package with a python version embedded"""
        importer1 = PackageImporter(
            f"{Path(__file__).parent}/package_e/test_nn_module.pt"
        )
        self.assertEqual(importer1.python_version(), "3.9.7")

    def test_file_structure_has_file(self):
        # [FUNCTION DOC] Exports a pickled object and checks Directory has_file: true for a real
        # file path inside the package and false for a directory path.
        """
        Test Directory's has_file() method.
        """
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            import package_a.subpackage

            he.intern("**")
            obj = package_a.subpackage.PackageASubpackageObject()
            he.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)

        importer = PackageImporter(buffer)
        file_structure = importer.file_structure()
        self.assertTrue(file_structure.has_file("package_a/subpackage.py"))
        self.assertFalse(file_structure.has_file("package_a/subpackage"))

    def test_exporter_content_lists(self):
        # [FUNCTION DOC] Checks the exporter lists of extern, mock, intern and deny modules and
        # the reverse dependency query. In the first export, the extern/mock/intern lists and the
        # reverse dependencies of the mocked module are asserted. In the second, denying package_b
        # makes saving the pickle raise PackagingError, which is expected.
        """
        Test content list API for PackageExporter's contained modules.
        """

        with PackageExporter(BytesIO()) as he:
            import package_b

            he.extern("package_b.subpackage_1")
            he.mock("package_b.subpackage_2")
            he.intern("**")
            he.save_pickle("obj", "obj.pkl", package_b.PackageBObject(["a"]))
            self.assertEqual(he.externed_modules(), ["package_b.subpackage_1"])
            self.assertEqual(he.mocked_modules(), ["package_b.subpackage_2"])
            self.assertEqual(
                he.interned_modules(),
                ["package_b", "package_b.subpackage_0.subsubpackage_0"],
            )
            self.assertEqual(he.get_rdeps("package_b.subpackage_2"), ["package_b"])

        with self.assertRaises(PackagingError):
            with PackageExporter(BytesIO()) as he:
                import package_b

                he.deny("package_b")
                he.save_pickle("obj", "obj.pkl", package_b.PackageBObject(["a"]))
                self.assertEqual(he.denied_modules(), ["package_b"])

    def test_is_from_package(self):
        # [FUNCTION DOC] Verifies is_from_package is false for the regular module and a regular
        # object and true for a module imported from, and an object loaded from, a package.
        """is_from_package should work for objects and modules"""
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        mod = pi.import_module("package_a.subpackage")
        loaded_obj = pi.load_pickle("obj", "obj.pkl")

        self.assertFalse(is_from_package(package_a.subpackage))
        self.assertTrue(is_from_package(mod))

        self.assertFalse(is_from_package(obj))
        self.assertTrue(is_from_package(loaded_obj))

    def test_inspect_class(self):
        # [FUNCTION DOC] Verifies that source line retrieval for a class imported from a package
        # returns the same result as for the original class.
        """Should be able to retrieve source for a packaged class."""
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        packaged_class = pi.import_module(
            "package_a.subpackage"
        ).PackageASubpackageObject
        regular_class = package_a.subpackage.PackageASubpackageObject

        packaged_src = inspect.getsourcelines(packaged_class)
        regular_src = inspect.getsourcelines(regular_class)
        self.assertEqual(packaged_src, regular_src)

    def test_dunder_package_present(self):
        # [FUNCTION DOC] Verifies a module imported through PackageImporter has the torch_package
        # attribute.
        """
        The attribute '__torch_package__' should be populated on imported modules.
        """
        import package_a.subpackage

        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        mod = pi.import_module("package_a.subpackage")
        self.assertTrue(hasattr(mod, "__torch_package__"))

    def test_dunder_package_works_from_package(self):
        # [FUNCTION DOC] Verifies packaged code can see it is packaged: a helper in the
        # use_dunder_package module returns true when imported from a package and false in the
        # normal import.
        """
        The attribute '__torch_package__' should be accessible from within
        the module itself, so that packaged code can detect whether it's
        being used in a packaged context or not.
        """
        import package_a.use_dunder_package as mod

        buffer = BytesIO()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_module(mod.__name__)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        imported_mod = pi.import_module(mod.__name__)
        self.assertTrue(imported_mod.is_from_package())
        self.assertFalse(mod.is_from_package())

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_std_lib_sys_hackery_checks(self):
        # [FUNCTION DOC] Regression test for modules that reassign entries in the module table (as
        # the standard library does). Picks the module variant by python version (3.13 or later
        # uses a separate file), pickles an instance, loads it from the package and calls it.
        # Skipped under Dynamo.
        """
        The standard library performs sys.module assignment hackery which
        causes modules who do this hackery to fail on import. See
        https://github.com/pytorch/pytorch/issues/57490 for more information.
        """
        if sys.version_info < (3, 13):
            import package_a.std_sys_module_hacks as std_sys_module_hacks
        else:
            import package_a.std_sys_module_hacks_3_13 as std_sys_module_hacks

        buffer = BytesIO()
        mod = std_sys_module_hacks.Module()

        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", mod)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        mod = pi.load_pickle("obj", "obj.pkl")
        mod()


if __name__ == "__main__":
    run_tests()
