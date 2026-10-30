# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for torch.package compatibility with torch.fx: packaging traced graph
# modules, tracing packaged modules, re-packaging modules that came from a package, graphs
# with leaf function dependencies, custom tracers, wrapped functions, and stack trace metadata
# preservation. At import time registers the builtin len as an fx wrapped function (twice, to
# check that repeated registration is harmless).

from io import BytesIO

import torch
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.package import (
    ObjMismatchError,
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

torch.fx.wrap("len")
# Do it twice to make sure it doesn't affect anything
torch.fx.wrap("len")


class TestPackageFX(PackageTestCase):
    # [CLASS DOC] Test case deriving from PackageTestCase that exercises fx and package
    # interplay. Holds no state of its own.
    """Tests for compatibility with FX."""

    def test_package_fx_simple(self):
        # [FUNCTION DOC] Symbolically traces a small relu module, packages the traced module,
        # reloads it and checks the loaded module output equals the traced module output.
        class SimpleTest(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module whose forward adds a constant to the input
            # and applies relu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the input plus the constant 3.0.
                return torch.relu(x + 3.0)

        st = SimpleTest()
        traced = symbolic_trace(st)

        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)

        f.seek(0)
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        input = torch.rand(2, 3)
        self.assertEqual(loaded_traced(input), traced(input))

    def test_package_then_fx(self):
        # [FUNCTION DOC] Packages a plain module from the package_a test data, loads it, then
        # symbolically traces the loaded module and checks the traced and loaded outputs
        # match.
        from package_a.test_module import SimpleTest

        model = SimpleTest()
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", model)

        f.seek(0)
        pi = PackageImporter(f)
        loaded = pi.load_pickle("model", "model.pkl")
        traced = symbolic_trace(loaded)
        input = torch.rand(2, 3)
        self.assertEqual(loaded(input), traced(input))

    def test_package_fx_package(self):
        # [FUNCTION DOC] Packages a module, loads it, traces the loaded module, and then tries
        # to package the traced module. Exporting without the original importer must raise
        # ObjMismatchError because the graph refers to globals only present in the package.
        # Exporting with the original package importer followed by the system importer
        # succeeds, and the reloaded result matches the first loaded module output.
        from package_a.test_module import SimpleTest

        model = SimpleTest()
        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", model)

        f.seek(0)
        pi = PackageImporter(f)
        loaded = pi.load_pickle("model", "model.pkl")
        traced = symbolic_trace(loaded)

        # re-save the package exporter
        f2 = BytesIO()
        # This should fail, because we are referencing some globals that are
        # only in the package.
        with self.assertRaises(ObjMismatchError):
            with PackageExporter(f2) as pe:
                pe.intern("**")
                pe.save_pickle("model", "model.pkl", traced)

        f2.seek(0)
        with PackageExporter(f2, importer=(pi, sys_importer)) as pe:
            # Make the package available to the exporter's environment.
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", traced)
        f2.seek(0)
        pi2 = PackageImporter(f2)
        loaded2 = pi2.load_pickle("model", "model.pkl")

        input = torch.rand(2, 3)
        self.assertEqual(loaded(input), loaded2(input))

    def test_package_fx_with_imports(self):
        # [FUNCTION DOC] Hand-builds a graph that calls a leaf function from
        # package_a.subpackage and then sine, wraps it in a graph module, packages and reloads
        # it. Checks numerical closeness against the original and that the packaged subpackage
        # module object differs from the one in the outer environment.
        import package_a.subpackage

        # Manually construct a graph that invokes a leaf function
        graph = Graph()
        a = graph.placeholder("x")
        b = graph.placeholder("y")
        c = graph.call_function(package_a.subpackage.leaf_function, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)

        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", gm)
        f.seek(0)

        pi = PackageImporter(f)
        loaded_gm = pi.load_pickle("model", "model.pkl")
        input_x = torch.rand(2, 3)
        input_y = torch.rand(2, 3)

        self.assertTrue(
            torch.allclose(loaded_gm(input_x, input_y), gm(input_x, input_y))
        )

        # Check that the packaged version of the leaf_function dependency is
        # not the same as in the outer env.
        packaged_dependency = pi.import_module("package_a.subpackage")
        self.assertTrue(packaged_dependency is not package_a.subpackage)

    def test_package_fx_custom_tracer(self):
        # [FUNCTION DOC] Traces a module with a custom tracer that treats all modules as
        # leaves, checks the tracer class is recorded on the graph, then wraps it in a graph
        # module subclass that carries an extra info attribute. Packages and reloads it,
        # checks the info attribute survives and that outputs match the original.
        from package_a.test_all_leaf_modules_tracer import TestAllLeafModulesTracer
        from package_a.test_module import ModWithTwoSubmodsAndTensor, SimpleTest

        class SpecialGraphModule(torch.fx.GraphModule):
            # [CLASS DOC] HELPER: Graph module subclass that stores one extra attribute, info,
            # in addition to the base graph module state.
            def __init__(self, root, graph, info):
                # [FUNCTION DOC] HELPER: Calls the base constructor with the root and graph
                # and stores the info argument on the instance.
                super().__init__(root, graph)
                self.info = info

        sub_module = SimpleTest()
        module = ModWithTwoSubmodsAndTensor(
            torch.ones(3),
            sub_module,
            sub_module,
        )
        tracer = TestAllLeafModulesTracer()
        graph = tracer.trace(module)

        self.assertEqual(graph._tracer_cls, TestAllLeafModulesTracer)

        gm = SpecialGraphModule(module, graph, "secret")
        self.assertEqual(gm._tracer_cls, TestAllLeafModulesTracer)

        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", gm)
        f.seek(0)

        pi = PackageImporter(f)
        loaded_gm = pi.load_pickle("model", "model.pkl")
        self.assertEqual(
            type(loaded_gm).__class__.__name__, SpecialGraphModule.__class__.__name__
        )
        self.assertEqual(loaded_gm.info, "secret")

        input_x = torch.randn(3)
        self.assertEqual(loaded_gm(input_x), gm(input_x))

    def test_package_fx_wrap(self):
        # [FUNCTION DOC] Traces a module whose forward calls the wrapped builtin len, packages
        # the traced module without interning, reloads it and checks outputs match.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no parameters whose forward returns the length
            # of its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base module constructor.
                super().__init__()

            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the length of the input via the wrapped
                # builtin.
                return len(a)

        traced = torch.fx.symbolic_trace(TestModule())

        f = BytesIO()
        with torch.package.PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)
        f.seek(0)

        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        input = torch.rand(2, 3)
        self.assertEqual(loaded_traced(input), traced(input))

    def test_package_gm_preserve_stack_trace(self):
        # [FUNCTION DOC] Traces a small module, sets a stack trace metadata string on every
        # graph node (derived from the node name), packages and reloads it, and checks every
        # loaded node still has the same stack trace string.
        class SimpleTest(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module whose forward adds a constant and applies
            # relu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the input plus the constant 3.0.
                return torch.relu(x + 3.0)

        st = SimpleTest()
        traced = symbolic_trace(st)

        for node in traced.graph.nodes:
            node.meta["stack_trace"] = f"test_{node.name}"

        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)

        f.seek(0)
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        for node in loaded_traced.graph.nodes:
            self.assertEqual(f"test_{node.name}", node.meta["stack_trace"])


if __name__ == "__main__":
    run_tests()
