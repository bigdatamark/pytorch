# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Regression test for repackaging in torch.package: exporting a model whose
# module imports a name indirectly through a parent package, using a prior importer. Uses
# PackageTestCase from the sibling common module (or a top-level import when run directly).

from io import BytesIO

from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestRepackage(PackageTestCase):
    # [CLASS DOC] Test case deriving from PackageTestCase holding the repackaging scenario. No
    # state.
    """Tests for repackaging."""

    def test_repackage_import_indirectly_via_parent_module(self):
        # [FUNCTION DOC] Packages a model from package_d that imports directly from a
        # sub-subpackage and loads it. Then exports a second model whose module imports the
        # same class indirectly via the parent package, passing the first importer followed by
        # the system importer. The test passes if exporting completes without error; no output
        # is asserted.
        from package_d.imports_directly import ImportsDirectlyFromSubSubPackage
        from package_d.imports_indirectly import ImportsIndirectlyFromSubPackage

        model_a = ImportsDirectlyFromSubSubPackage()
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model.py", model_a)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        pi.load_pickle("default", "model.py")

        model_b = ImportsIndirectlyFromSubPackage()
        buffer = BytesIO()
        with PackageExporter(
            buffer,
            importer=(
                pi,
                sys_importer,
            ),
        ) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model_b.py", model_b)


if __name__ == "__main__":
    run_tests()
