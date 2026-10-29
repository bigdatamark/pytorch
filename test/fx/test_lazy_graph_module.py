# Owner(s): ["oncall: fx"]

# [MODULE DOC] Tests for the lazy graph module (torch.fx._lazy_graph_module): a GraphModule
# subclass that defers recompilation of generated code until the code or forward is first used.
# Covers recompile state tracking, recapture via make_fx, symbolic_trace and dynamo, pickling, the
# graph module factory, torch.package round trip, dynamo innermost function lookup and to_folder
# export. The main guard raises RuntimeError since the file is not run directly.
import contextlib
import importlib.util
import pickle
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch._export
from torch import fx
from torch.fx._lazy_graph_module import (
    _LazyGraphModule,
    _make_graph_module,
    _use_lazy_graph_module,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import TestCase


class TestLazyGraphModule(TestCase):
    # [CLASS DOC] Test case run with lazy graph modules enabled for the whole class: setUpClass
    # enters the lazy-mode context into a class level ExitStack (kept in exit_stack) so that
    # traced graph modules are lazy, and tearDownClass closes it. Neither hook calls the parent
    # versions.
    exit_stack = None

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Creates the class level ExitStack and enters the context that makes graph
        # module creation produce lazy graph modules.
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(_use_lazy_graph_module(True))

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Closes the class level ExitStack, restoring the previous lazy-module
        # setting.
        cls.exit_stack.close()

    @staticmethod
    def replace_sin_with_cos(gm):
        # [FUNCTION DOC] HELPER: Static method that mutates the graph in place, retargeting every
        # node whose target is the string sin to cos. It does not recompile the module.
        for n in gm.graph.nodes:
            if n.target == "sin":
                n.target = "cos"

    def test_replace_sin_with_cos(self):
        # [FUNCTION DOC] Retargets sin to cos on a traced lazy module, recompiles, and asserts the
        # output equals cosine of the input, the readable code mentions cos, and the module is
        # still lazy.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        x = torch.randn(2, 3)
        gm = fx.symbolic_trace(f)
        self.replace_sin_with_cos(gm)

        gm.recompile()
        expected = x.cos()
        actual = gm(x)

        self.assertTrue(torch.allclose(expected, actual))
        code = gm.print_readable(False)
        self.assertTrue("cos()" in code)
        self.assertTrue(isinstance(gm, _LazyGraphModule))

    def test_call_forward_directly(self):
        # [FUNCTION DOC] Same retargeting, then calls the forward method directly (not the module
        # call) and asserts it gives the cosine result, i.e. the lazy forward also recompiles on
        # demand.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        x = torch.randn(2, 3)
        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.replace_sin_with_cos(gm)
        gm.recompile()
        expected = x.cos()
        actual = gm.forward(x)

        self.assertTrue(torch.allclose(expected, actual))

    def test_needs_recompile(self):
        # [FUNCTION DOC] Asserts the needs-recompile flag is true right after tracing a lazy
        # module and false after the first call runs it.
        """
        Make sure needs_recompile() return the correct state.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        gm(torch.randn(2, 3))
        self.assertFalse(gm._needs_recompile())

    def test_multi_recompile(self):
        # [FUNCTION DOC] Walks through two recompilations: the first on the initial call (flag
        # clears), then retargeting alone leaves the flag clear, an explicit recompile call sets
        # it again, and the next call recompiles and clears it with the cosine result.
        """
        Cover the case that multiple recompilation happens.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        x = torch.randn(2, 3)
        # trigger the first recompilation
        self.assertTrue(torch.allclose(x.sin(), gm(x)))
        self.assertFalse(gm._needs_recompile())

        self.replace_sin_with_cos(gm)
        self.assertFalse(gm._needs_recompile())
        gm.recompile()
        self.assertTrue(gm._needs_recompile())
        # trigger the second recompilation
        self.assertTrue(torch.allclose(x.cos(), gm(x)))
        self.assertFalse(gm._needs_recompile())

    def test_accessing_code_cause_recompiling(self):
        # [FUNCTION DOC] Reading the code property of a not yet compiled lazy module triggers the
        # recompilation: the text contains sin and the needs-recompile flag becomes false.
        """
        Make sure we recompile if we have not done that yet when we access the code
        property of a GraphModule.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        # should trigger a recompilation
        code = gm.code
        self.assertTrue("sin" in code)
        self.assertFalse(gm._needs_recompile())

    def test_graph_module_str(self):
        # [FUNCTION DOC] Asserts a lazy module's string form includes sin and the module is lazy.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue("sin" in str(gm))

    def test_recapture_with_make_fx(self):
        # [FUNCTION DOC] Retraces a lazy module with make_fx. The new module is lazy and still
        # needs recompile, while the original's flag is cleared because make_fx runs its forward.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        gm2 = make_fx(gm)(torch.randn(2, 3))
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        self.assertTrue(gm2._needs_recompile())

        # make_fx will cal forward method of gm. That clears the _needs_recompile()
        # flag.
        self.assertFalse(gm._needs_recompile())

    def test_recapture_with_symbolic_trace(self):
        # [FUNCTION DOC] Retraces a lazy module with symbolic_trace. The original is recompiled up
        # front since symbolic_trace cannot trace lazy recompilation (flag false) and the new
        # module needs recompile.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        gm2 = fx.symbolic_trace(gm)

        # the lazy recompilcation is already realized. We realize the
        # recompilation in the beginning of symbolic_trace since symbolic_trace can not
        # handle the tracing of lazy recompilation.
        self.assertFalse(gm._needs_recompile())
        self.assertTrue(gm2._needs_recompile())

    def test_recapture_with_dynamo(self):
        # [FUNCTION DOC] Compiles a lazy module with torch.compile and runs it once; dynamo calls
        # forward with the eval hook installed, which performs the real recompilation, so the flag
        # clears.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        self.assertTrue(gm._needs_recompile())
        torch.compile(gm)(torch.rand(2, 3))

        # dynamo calls gm.forward with eval hook installed. That will trigger
        # the real recompilation.
        self.assertFalse(gm._needs_recompile())

    def test_save_lazy_foward(self):
        # [FUNCTION DOC] Patches the GraphModule recompile method with a counting wrapper, saves
        # the bound forward of a lazy module and calls it ten times. Asserts recompile ran exactly
        # once.
        """
        Save the lazy forward method and call it repeatedly. Make sure we
        don't recompile for each such call.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        orig_gm_recompile = fx.GraphModule.recompile
        recompile_count = 0

        def mock_gm_recompile(self):
            # [FUNCTION DOC] HELPER: Counting replacement for recompile: increments the enclosing
            # counter and delegates to the original unpatched recompile, returning its result.
            nonlocal recompile_count
            recompile_count += 1
            return orig_gm_recompile(self)

        with patch.object(fx.GraphModule, "recompile", mock_gm_recompile):
            gm = fx.symbolic_trace(f)
            self.assertTrue(isinstance(gm, _LazyGraphModule))
            saved_fwd = gm.forward

            x = torch.rand(2, 3)
            for _ in range(10):
                saved_fwd(x)

        self.assertEqual(recompile_count, 1)

    def test_pickle(self):
        # [FUNCTION DOC] Pickles a lazy module and unpickles it; asserts the copy is lazy and its
        # code contains sin. Needed because the FX graph cache pickles graph modules.
        """
        Fx graph cache need the ability to pickle GraphModule/_LazyGraphModule.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x via the tensor method.
            return x.sin()

        gm = fx.symbolic_trace(f)
        self.assertTrue(isinstance(gm, _LazyGraphModule))
        serialized = pickle.dumps(gm)
        gm2 = pickle.loads(serialized)
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        self.assertTrue("sin" in gm2.code)

    def test_make_graph_module(self):
        # [FUNCTION DOC] Checks the factory that builds graph modules. With an explicit class name
        # and plain GraphModule class the result is not lazy and has that class name; with
        # defaults it is lazy and the class name is GraphModule.
        gm = fx.symbolic_trace(lambda x: x.sin())
        self.assertTrue(isinstance(gm, _LazyGraphModule))

        gm1 = _make_graph_module(
            gm, gm.graph, class_name="MyGraphModule", graph_module_cls=fx.GraphModule
        )
        self.assertFalse(isinstance(gm1, _LazyGraphModule))
        self.assertTrue(gm1.__class__.__name__ == "MyGraphModule")

        gm2 = _make_graph_module(gm, gm.graph)
        self.assertTrue(isinstance(gm2, _LazyGraphModule))
        self.assertTrue(gm2.__class__.__name__ == "GraphModule")

    def test_package_fx_simple(self):
        # [FUNCTION DOC] Saves a traced lazy module into an in-memory torch.package archive, loads
        # it back and asserts the loaded module output equals the original on a random input.
        """
        Copied from test/package/test_package_fx.py to make sure LazyGraphModule
        works with torch.package.
        """

        class SimpleTest(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning relu of its input plus three.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of x plus a float constant three.
                return torch.relu(x + 3.0)

        st = SimpleTest()
        traced = fx.symbolic_trace(st)

        f = BytesIO()
        with PackageExporter(f) as pe:
            pe.save_pickle("model", "model.pkl", traced)

        f.seek(0)
        pi = PackageImporter(f)
        loaded_traced = pi.load_pickle("model", "model.pkl")
        input = torch.rand(2, 3)
        self.assertEqual(loaded_traced(input), traced(input))

    def test_dynamo_innermost_fn(self):
        # [FUNCTION DOC] Regression test for a dynamo innermost function lookup issue. Wraps the
        # forward of a regular and of a lazy module with dynamo disable and asserts the innermost
        # function found has the self attribute (bound method) in both cases.
        """
        Repro for https://github.com/pytorch/pytorch/issues/121198 .
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns x times two.
            return x * 2

        gm = torch.fx.symbolic_trace(f)
        lazy_gm = torch.fx._lazy_graph_module._LazyGraphModule.from_graphmodule(gm)

        wrapped_forward = torch._dynamo.disable(gm.forward)
        got_inner_forward = torch._dynamo.eval_frame.innermost_fn(wrapped_forward)
        if not hasattr(got_inner_forward, "__self__"):
            raise AssertionError("Expected got_inner_forward to have '__self__'")

        wrapped_lazy_forward = torch._dynamo.disable(lazy_gm.forward)
        got_lazy_inner_forward = torch._dynamo.eval_frame.innermost_fn(
            wrapped_lazy_forward
        )
        if not hasattr(got_lazy_inner_forward, "__self__"):
            raise AssertionError("Expected got_lazy_inner_forward to have '__self__'")

    def test_to_folder_class(self):
        # [FUNCTION DOC] Builds a small module with a sequential, a parameter, and a parameter
        # with the name zero, and validates the to_folder export round trip for it.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a sequential of linear and relu, a parameter a and a
            # second parameter stored under the attribute name zero.
            def __init__(self, features_in=10):
                # [FUNCTION DOC] HELPER: Creates the sequential and the two parameters, each
                # random with size features_in.
                super().__init__()

                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(features_in, features_in), torch.nn.ReLU()
                )
                self.a = torch.nn.Parameter(torch.randn(features_in))
                setattr(self, "0", torch.nn.Parameter(torch.randn(features_in)))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sequential output plus both parameters.
                return self.seq.forward(x) + self.a + getattr(self, "0")

        in_features = 10

        f = TestModule(in_features)

        self.validate_module(f, in_features=in_features)

    def test_to_folder_sequential(self):
        # [FUNCTION DOC] Validates the to_folder round trip for a sequential holding one linear
        # layer.
        in_features = 10

        f = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=in_features)
        )

        self.validate_module(f, in_features=in_features)

    def validate_module(
        self, module: torch.nn.Module, in_features: int, n_samples: int = 100
    ):
        # [FUNCTION DOC] HELPER: Traces the module, exports it with to_folder into a temporary
        # directory, imports the generated module file dynamically, instantiates the FxModule
        # class, deletes the imported module, and asserts the original and loaded modules agree on
        # a random batch of n_samples inputs with in_features features.
        gm = torch.fx.symbolic_trace(module)

        with TemporaryDirectory() as tmp:
            gm.to_folder(tmp)

            path = Path(tmp) / "module.py"

            spec = importlib.util.spec_from_file_location(
                "module.FxModule", path.absolute()
            )
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)

            loaded_module = foo.FxModule()

            del foo

        val = torch.randn(n_samples, in_features)

        self.assertEqual(module(val), loaded_module(val))


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
