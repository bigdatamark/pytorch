# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo compiler backends and the backend registry: running small
# models through built-in backends (eager, aot_eager, ts, cudagraphs, tvm, hpu), listing and
# looking up backends, custom backend registration, aot_autograd re-entry, ExplainWithBackend,
# default-backend setting. Import-time: TestOptimizations is instantiated per device type
# with instantiate_device_type_tests into this module's globals.
import importlib.util
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch._dynamo
import torch._dynamo.backends
import torch._dynamo.test_case
from torch._dynamo.backends.debugging import ExplainWithBackend
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.backends.tvm import has_tvm
from torch._dynamo.testing import same
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyHPU,
)
from torch.testing._internal.common_utils import skipIfHpu
from torch.testing._internal.triton_utils import requires_cuda_and_triton


class Seq(torch.nn.Module):
    # [CLASS DOC] HELPER: Small model: Linear(10,10), ReLU, Linear(10,10), Sigmoid in a
    # Sequential stored as layers.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the four-layer Sequential and stores it as layers.
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the output of the layers applied to x.
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    # [CLASS DOC] HELPER: Convolution (no bias), batch norm (eps 0.001) and ReLU in sequence.
    # Not used by any test in this file.
    def __init__(self, in_channels, out_channels, **kwargs):
        # [FUNCTION DOC] HELPER: Creates conv (extra kwargs go to Conv2d), bn and relu.
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns relu(bn(conv(x))).
        return self.relu(self.bn(self.conv(x)))


class TestOptimizations(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic TestCase (instantiated per device type) that runs the Seq
    # model through each named backend and compares forward/backward to eager; also has two
    # example-input tests and a backend-listing test. Stateless besides Dynamo resets.
    def test_example_inputs(self):
        # [FUNCTION DOC] Compiles a function with a custom compiler (optimize_assert) that runs
        # the graph once on the example inputs and stores the first output in r1. Asserts r1
        # has the same size, stride and dtype as both the eager result and the compiled result.
        def fn(a, bc, d):
            # [FUNCTION DOC] HELPER: Unpacks bc as (b, c); returns a / d - b / c.
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            # [FUNCTION DOC] HELPER: Backend; runs graph on the example inputs, stores the
            # first output in the enclosing r1, returns graph.forward unchanged.
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertEqual(r1.size(), r2.size())
        self.assertEqual(r1.stride(), r2.stride())
        self.assertEqual(r1.dtype, r2.dtype)

        self.assertEqual(r1.size(), r3.size())
        self.assertEqual(r1.stride(), r3.stride())
        self.assertEqual(r1.dtype, r3.dtype)

    def test_example_inputs_runtime_use(self):
        # [FUNCTION DOC] Compiles a function with a backend returning a wrapper that, on each
        # real call, runs graph.forward and stores its first output in r1. Asserts r1 is set
        # and numerically equal to both the eager and the compiled result.
        def fn(a, bc, d):
            # [FUNCTION DOC] HELPER: Unpacks bc as (b, c); returns a / d - b / c.
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            # [FUNCTION DOC] HELPER: Backend; returns the wrapper fwd below.
            def fwd(*args):
                # [FUNCTION DOC] HELPER: Runs graph.forward, records its first output in r1,
                # returns the full result.
                nonlocal r1
                r = graph.forward(*args)
                r1 = r[0]
                return r

            return fwd

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    def _check_backend_works(
        self, backend, device, boxed=True, options=None, backward=True
    ):
        # [FUNCTION DOC] HELPER: Compiles the Seq model (eval, moved to device) with the named
        # backend and options. If boxed, wraps the backend in a function that looks it up from
        # the registry and raises AssertionError unless the compiled function has a true
        # _boxed_call flag. Compares forward output to eager within 0.01 and, if backward is
        # true, compares input gradients too. Always resets Dynamo at the end (avoids false
        # leak reports).
        model = Seq().eval()
        model.to(device)

        if not boxed:
            compiled_model = torch.compile(model, backend=backend, options=options)
        else:

            def boxed_assert(gm, *example_args):
                # [FUNCTION DOC] HELPER: Backend wrapper; compiles gm with the looked-up named
                # backend, raises AssertionError if the result lacks a true _boxed_call.
                fn = lookup_backend(backend)(gm, *example_args)
                if not fn._boxed_call:
                    raise AssertionError("Expected fn._boxed_call to be True")
                return fn

            compiled_model = torch.compile(model, backend=boxed_assert, options=options)

        input1 = torch.randn(2, 10, device=device, requires_grad=True)
        input2 = input1.detach().clone().requires_grad_(True)

        r1 = model(input1)
        r2 = compiled_model(input2)
        self.assertTrue(same(r1, r2.float(), tol=0.01))

        if backward:
            r1.sum().backward()
            r2.sum().backward()
            self.assertTrue(same(input1.grad, input2.grad.float(), tol=0.01))

        # Clean up compilation state before test returns to avoid false positive
        # memory leak detection (leak check runs before tearDown)
        torch._dynamo.reset()

    def test_eager(self, device):
        # [FUNCTION DOC] Runs the eager backend (non-boxed) check on the device.
        self._check_backend_works("eager", device, boxed=False)

    def test_eager_noexcept(self, device):
        # [FUNCTION DOC] Runs the eager_noexcept backend (non-boxed) check on the device.
        self._check_backend_works("eager_noexcept", device, boxed=False)

    @skipIfHpu
    def test_torchscript(self, device):
        # [FUNCTION DOC] Runs the ts (TorchScript) backend, non-boxed; skipped on HPU.
        self._check_backend_works("ts", device, boxed=False)

    def test_aot_eager(self, device):
        # [FUNCTION DOC] Runs the aot_eager backend with the boxed-call assertion enabled.
        self._check_backend_works("aot_eager", device)

    def test_aot_eager_decomp_partition(self, device):
        # [FUNCTION DOC] Runs the aot_eager_decomp_partition backend with boxed-call check.
        self._check_backend_works("aot_eager_decomp_partition", device)

    @skipIfHpu
    def test_aot_ts(self, device):
        # [FUNCTION DOC] Runs the aot_ts backend with boxed-call check; skipped on HPU.
        self._check_backend_works("aot_ts", device)

    @requires_cuda_and_triton
    def test_aot_cudagraphs(self, device):
        # [FUNCTION DOC] Runs the cudagraphs backend with boxed-call check; requires CUDA and
        # triton.
        self._check_backend_works("cudagraphs", device)

    @unittest.skipIf(not has_tvm(), "requires tvm")
    def test_tvm(self, device):
        # [FUNCTION DOC] Runs the tvm backend, non-boxed and forward only; skipped without tvm.
        self._check_backend_works("tvm", device, boxed=False, backward=False)

    @unittest.skipIf(not has_tvm(), "requires tvm")
    def test_tvm_scalar_tensor_input(self, device):
        # [FUNCTION DOC] Compiles a module holding a scalar (0-dim) Parameter with the tvm
        # backend and asserts its output matches eager within 0.01. Skipped without tvm.
        class ScalarParam(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 0-dim parameter scale initialized to 3.0.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the scalar parameter scale.
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(3.0))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus the scalar parameter.
                return x + self.scale

        model = ScalarParam().eval().to(device)
        x = torch.randn(2, 10, device=device)
        expected = model(x)
        compiled = torch.compile(model, backend="tvm")
        self.assertTrue(same(expected, compiled(x), tol=0.01))

    @unittest.skipIf(not has_tvm(), "requires tvm")
    def test_tvm_relax_pipeline_option(self, device):
        # [FUNCTION DOC] Skips unless the tvm relax torch frontend is importable. For the
        # pipeline option given both as the string "zero" and as the object from
        # tvm.relax.get_pipeline("zero"), resets Dynamo, compiles Seq with the tvm backend and
        # asserts the output matches eager within 0.01.
        if importlib.util.find_spec("tvm.relax.frontend.torch") is None:
            self.skipTest("requires the tvm relax frontend")
        import tvm

        model = Seq().eval().to(device)
        x = torch.randn(2, 10, device=device)
        expected = model(x)
        for pipeline in ("zero", tvm.relax.get_pipeline("zero")):
            torch._dynamo.reset()
            compiled = torch.compile(
                model, backend="tvm", options={"pipeline": pipeline}
            )
            self.assertTrue(same(expected, compiled(x), tol=0.01))

    def test_tvm_missing_install_error(self, device):
        # [FUNCTION DOC] Traces a trivial add-one function with fx, blocks the tvm modules in
        # sys.modules (set to None) and asserts calling the tvm backend raises ImportError.
        from torch._dynamo.backends.tvm import tvm as tvm_backend

        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        with patch.dict(sys.modules, {"tvm": None, "tvm.relax.frontend.torch": None}):
            self.assertRaises(ImportError, tvm_backend, gm, [torch.randn(2)])

    @onlyHPU
    def test_intel_gaudi_backend(self, device):
        # [FUNCTION DOC] Runs the hpu_backend with boxed-call check; HPU devices only.
        self._check_backend_works("hpu_backend", device)

    def test_list_backends(self):
        # [FUNCTION DOC] Checks the exclude_tags behavior of list_backends: inductor is always
        # listed; eager (tagged debug) is hidden by default and with exclude_tags debug, but
        # shown with an empty exclude list.
        self.assertIn("inductor", torch._dynamo.list_backends())
        self.assertIn("inductor", torch._dynamo.list_backends(exclude_tags=None))
        self.assertNotIn("eager", torch._dynamo.list_backends())
        self.assertNotIn("eager", torch._dynamo.list_backends(exclude_tags=["debug"]))
        self.assertIn("eager", torch._dynamo.list_backends(exclude_tags=[]))


class NormalizeIRTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase with one aot_eager test of an in-place op across dtypes.
    def test_inplace_normalize(self):
        # [FUNCTION DOC] Compiles with aot_eager a function that adds a float64 tensor in place
        # to a float32 cosine result and applies sin, then asserts the output matches eager.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: x = cos(a); x += b in place; returns sin(x).
            x = torch.cos(a)
            x += b
            return torch.sin(x)

        a = torch.randn(10)
        b = torch.randn(10).to(torch.float64)

        ref = fn(a, b)

        optimized_fn = torch.compile(fn, backend="aot_eager")
        res = optimized_fn(a, b)
        self.assertTrue(same(ref, res))


class MPSSupportedTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase for the inductor backend on the MPS device; one test.
    @unittest.skipIf(not torch.backends.mps.is_available(), "requires mps")
    def test_mps_supported(self):
        # [FUNCTION DOC] Compiles the Seq model on mps with inductor and asserts the output
        # equals the eager output. Skipped when MPS is unavailable.
        model = Seq().to("mps")
        example_input = torch.randn(1, 10).to("mps")
        rc_eager = model(example_input)
        rc = torch.compile(model, backend="inductor")(example_input)
        self.assertEqual(rc, rc_eager)


class TestExplainWithBackend(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase for ExplainWithBackend, a backend wrapper that records
    # graph, break and op statistics while delegating compilation to another backend.
    def test_explain_with_backend(self):
        # [FUNCTION DOC] Wraps inductor in ExplainWithBackend and compiles a chain of nested
        # calls containing one explicit graph break. Asserts the output matches eager, the
        # printed explanation contains the Graph Count, Graph Break Count, Op Count and Break
        # Reasons fields, and the counts are 2 graphs, 1 break and 8 ops.
        def fn3(x):
            # [FUNCTION DOC] HELPER: sin, explicit graph break, sin again; returns result.
            x = torch.sin(x)
            torch._dynamo.graph_break()
            x = torch.sin(x)
            return x

        def fn2(x):
            # [FUNCTION DOC] HELPER: cos, then fn3, then cos; returns result.
            x = torch.cos(x)
            x = fn3(x)
            x = torch.cos(x)
            return x

        def fn1(x):
            # [FUNCTION DOC] HELPER: tan, then fn2, then tan; returns result.
            x = torch.tan(x)
            x = fn2(x)
            x = torch.tan(x)
            return x

        def fn(x):
            # [FUNCTION DOC] HELPER: sigmoid, then fn1, then sigmoid; returns result.
            x = torch.sigmoid(x)
            x = fn1(x)
            x = torch.sigmoid(x)
            return x

        # Wrap TorchInductor with explain backend
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        input_tensor = torch.randn(5)
        result = optimized_fn(input_tensor)

        # Check that fn still produces the same output when wrapped by ExplainWithBackend
        self.assertTrue(torch.allclose(result, fn(input_tensor)))

        # Verify ExplainOutput object contents, output might change but make sure these fields are present
        explain_output = eb.output()
        explain_str = str(explain_output)
        self.assertIn("Graph Count", explain_str)
        self.assertIn("Graph Break Count", explain_str)
        self.assertIn("Op Count", explain_str)
        self.assertIn("Break Reasons", explain_str)

        # Verify that for the given functions above, we report the correct number of graphs, graph breaks, and ops
        self.assertEqual(2, explain_output.graph_count)
        self.assertEqual(1, explain_output.graph_break_count)
        self.assertEqual(8, explain_output.op_count)


class TestCustomBackendAPI(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase for the public custom-backend APIs (register_backend,
    # aot_autograd, lookup_backend, list_backends, registry internals); tests clean up any
    # registry changes they make.
    """Test APIs documented by https://pytorch.org/docs/main/torch.compiler_custom_backends.html"""

    def test_register_backend_api(self):
        # [FUNCTION DOC] Registers a custom backend by name with register_backend, compiles a
        # relu function using that name and asserts the backend ran. A cleanup removes the name
        # from the registry's compiler, backend and tag tables.
        from torch._dynamo import register_backend
        from torch._dynamo.backends import registry as backend_registry

        backend_run = False
        backend_name = "my_custom_backend"

        def cleanup_backend():
            # [FUNCTION DOC] HELPER: Removes backend_name from the three registry tables
            # (ignoring absence).
            backend_registry._COMPILER_FNS.pop(backend_name, None)
            backend_registry._BACKENDS.pop(backend_name, None)
            backend_registry._BACKEND_TAGS.pop(backend_name, None)

        self.addCleanup(cleanup_backend)

        @register_backend(name=backend_name)
        def my_custom_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend; flags that it ran, returns gm.forward.
            nonlocal backend_run
            backend_run = True
            return gm.forward

        def f(x):
            # [FUNCTION DOC] HELPER: Returns relu(x).
            return torch.relu(x)

        opt_f = torch.compile(f, backend="my_custom_backend")
        opt_f(torch.randn(3, 3))
        self.assertTrue(backend_run)

    def test_aot_autograd_api(self):
        # [FUNCTION DOC] Builds a backend with aot_autograd using a custom forward compiler,
        # compiles a relu function with it and asserts the forward compiler ran.
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        backend_run = False

        def my_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler; flags it ran, returns boxed gm.forward.
            nonlocal backend_run
            backend_run = True
            return make_boxed_func(gm.forward)

        my_backend = aot_autograd(fw_compiler=my_compiler)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns relu(x).
            return torch.relu(x)

        opt_f = torch.compile(f, backend=my_backend)
        opt_f(torch.randn(3, 3))
        self.assertTrue(backend_run)

    def test_aot_autograd_reentrant_bw_compiler(self):
        # [FUNCTION DOC] Uses one aot_autograd backend for two successive compiled
        # forward/backward runs. After the first run, takes the bw_compiler stored in the
        # backend kwargs; after the second run asserts the same object is still stored (not
        # re-wrapped) and that it has no compiler_fn attribute.
        # re-entry with an already-wrapped bw_compiler must leave it untouched
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        def my_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Returns boxed gm.forward.
            return make_boxed_func(gm.forward)

        backend = aot_autograd(fw_compiler=my_compiler)

        torch.compile(lambda x: x.sin() + 1, backend=backend)(
            torch.randn(3, requires_grad=True)
        ).sum().backward()
        bw_compiler = backend.kwargs["bw_compiler"]
        torch.compile(lambda x: x.cos() * 2, backend=backend)(
            torch.randn(3, requires_grad=True)
        ).sum().backward()

        self.assertIs(backend.kwargs["bw_compiler"], bw_compiler)
        self.assertFalse(hasattr(bw_compiler, "compiler_fn"))

    def test_aot_autograd_reentrant_serializable_bw_compiler(self):
        # [FUNCTION DOC] Passes a SerializableAOTDispatchCompiler as bw_compiler to an
        # aot_autograd backend and runs two compiled forward/backward passes. Asserts the
        # compiler_fn recorded on it after the first pass is the identical object after the
        # second pass (no double wrapping).
        # re-entry must not re-wrap a SerializableAOTDispatchCompiler's compiler_fn
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch._aot_autograd.schemas import (
            SerializableAOTDispatchCompiler,
        )

        def my_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Returns boxed gm.forward.
            return make_boxed_func(gm.forward)

        bw_compiler = SerializableAOTDispatchCompiler(object, my_compiler)
        backend = aot_autograd(fw_compiler=my_compiler, bw_compiler=bw_compiler)

        torch.compile(lambda x: x.sin() + 1, backend=backend)(
            torch.randn(3, requires_grad=True)
        ).sum().backward()
        wrapped_fn = bw_compiler.compiler_fn
        torch.compile(lambda x: x.cos() * 2, backend=backend)(
            torch.randn(3, requires_grad=True)
        ).sum().backward()

        self.assertIs(bw_compiler.compiler_fn, wrapped_fn)

    def test_lookup_backend(self):
        # [FUNCTION DOC] Compiles a relu function with a custom backend that tries the
        # registered tensorrt backend, then inductor (swallowing any exception from each),
        # and finally falls back to gm.forward. Asserts only that the custom backend ran.
        from torch._dynamo import lookup_backend

        backend_run = False

        def my_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Flags it ran; returns the first non-None compiled result
            # from the tensorrt then inductor backends, else gm.forward.
            nonlocal backend_run
            backend_run = True
            try:
                trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
                if trt_compiled is not None:
                    return trt_compiled
            except Exception:
                pass
            # first backend failed, try something else...
            try:
                inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
                if inductor_compiled is not None:
                    return inductor_compiled
            except Exception:
                pass
            return gm.forward

        def f(x):
            # [FUNCTION DOC] HELPER: Returns relu(x).
            return torch.relu(x)

        opt_f = torch.compile(f, backend=my_compiler)
        opt_f(torch.randn(3, 3))
        self.assertTrue(backend_run)

    def test_device_and_dtype_from_inputs(self):
        # [FUNCTION DOC] Checks device_from_inputs and dtype_from_inputs: non-tensor objects
        # that merely have device/dtype attributes are skipped in favor of a real tensor's
        # values, and with no tensor present the defaults are cpu and float32.
        from torch._dynamo.backends.common import device_from_inputs, dtype_from_inputs

        class NotATensor:
            # [CLASS DOC] HELPER: Non-tensor with bogus device and dtype class attributes.
            device = "not-a-device"
            dtype = "not-a-dtype"

        tensor = torch.randn(3, dtype=torch.float64)
        self.assertEqual(device_from_inputs([NotATensor(), tensor]), tensor.device)
        self.assertEqual(dtype_from_inputs([NotATensor(), tensor]), torch.float64)
        self.assertEqual(device_from_inputs([NotATensor()]), torch.device("cpu"))
        self.assertEqual(dtype_from_inputs([NotATensor()]), torch.float32)

    def test_is_registered_backend(self):
        # [FUNCTION DOC] Checks _is_registered_backend: true for a looked-up eager backend, the
        # inductor wrapper and a compile wrapper around "eager"; false for an object that only
        # claims compiler_name inductor, a plain custom function, and a compile wrapper
        # around a custom function.
        from torch._dynamo.backends.registry import _is_registered_backend

        self.assertTrue(_is_registered_backend(lookup_backend("eager")))
        self.assertTrue(
            _is_registered_backend(torch._TorchCompileInductorWrapper(None, None, None))
        )
        self.assertTrue(
            _is_registered_backend(
                torch._TorchCompileWrapper("eager", None, None, None)
            )
        )

        class FakeBackend:
            # [CLASS DOC] HELPER: Fake with a compiler_name attribute equal to inductor.
            compiler_name = "inductor"

        self.assertFalse(_is_registered_backend(FakeBackend()))

        def my_custom_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Returns gm.forward.
            return gm.forward

        self.assertFalse(_is_registered_backend(my_custom_backend))
        self.assertFalse(
            _is_registered_backend(
                torch._TorchCompileWrapper(my_custom_backend, None, None, None)
            )
        )

    def test_lookup_backend_suggestion(self):
        # [FUNCTION DOC] Asserts lookup_backend raises InvalidBackend for an unknown name,
        # including a "did you mean" suggestion for a near miss of inductor and no suggestion
        # for an unrelated string.
        from torch._dynamo.backends.registry import lookup_backend
        from torch._dynamo.exc import InvalidBackend

        with self.assertRaisesRegex(InvalidBackend, "did you mean: 'inductor'"):
            lookup_backend("indutcor")

        with self.assertRaises(InvalidBackend) as cm:
            lookup_backend("zzzzzzzz")
        self.assertNotIn("did you mean", str(cm.exception))

    def test_lookup_custom_backend(self):
        # [FUNCTION DOC] Mocks importlib.metadata.entry_points to expose one backend name in the
        # torch_dynamo_backends group, clears the registry's lazy-import and entrypoint
        # discovery caches, and asserts list_backends then includes the name. A cleanup
        # restores the registry tables and caches.
        from torch._dynamo import list_backends

        backends_group = "torch_dynamo_backends"
        name = "mycustombackend"

        mock_3_10 = MagicMock()
        mock_3_10.load.return_value = lambda: "mocked 3.10"

        def mock_eps(group=None):
            # [FUNCTION DOC] HELPER: Stand-in for entry_points; raises AssertionError unless
            # group is torch_dynamo_backends, returns a mock group with the one name.
            if group != backends_group:
                raise AssertionError(f"Expected group {backends_group}, got {group}")
            mock_group = MagicMock()
            mock_group.names = [name]
            mock_group[name] = mock_3_10
            return mock_group

        with patch("importlib.metadata.entry_points", mock_eps):
            from torch._dynamo.backends import registry

            orig_backends = dict(registry._BACKENDS)
            orig_compiler_fns = dict(registry._COMPILER_FNS)
            orig_backend_tags = dict(registry._BACKEND_TAGS)

            def restore_registry():
                # [FUNCTION DOC] HELPER: Restores the three registry tables from the saved
                # copies and clears the lazy-import and entrypoint-discovery caches.
                registry._BACKENDS.clear()
                registry._BACKENDS.update(orig_backends)
                registry._COMPILER_FNS.clear()
                registry._COMPILER_FNS.update(orig_compiler_fns)
                registry._BACKEND_TAGS.clear()
                registry._BACKEND_TAGS.update(orig_backend_tags)
                registry._lazy_import.cache_clear()
                registry._discover_entrypoint_backends.cache_clear()

            self.addCleanup(restore_registry)

            registry._lazy_import.cache_clear()
            registry._discover_entrypoint_backends.cache_clear()

            backends = list_backends()
            if name not in backends:
                raise AssertionError(f"Expected {name} in backends, got {backends}")

    def test_backend_recompilation(self):
        # [FUNCTION DOC] Compiles a function with inductor and a testing option that disables
        # an injected error (runs fine), then compiles it again with the option enabled and
        # asserts running it raises BackendCompilerFailed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x + x.
            return x + x

        input = torch.tensor(2.0)

        opt_fn = torch.compile(
            fn, backend="inductor", options={"_raise_error_for_testing": False}
        )
        opt_fn(input)
        with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed):
            opt_fn = torch.compile(
                fn, backend="inductor", options={"_raise_error_for_testing": True}
            )
            opt_fn(input)

    def test_cudagraphs_backend_accepts_extra_kwargs(self):
        # [FUNCTION DOC] Patches the inner cudagraphs compiler to return a sentinel and calls a
        # CudagraphsBackend with an extra options kwarg. Asserts the sentinel is returned, the
        # inner compiler was called with only (gm, []), and a warning containing "ignoring
        # extra kwargs" was logged.
        # Issue #169939: torch.compile(backend="cudagraphs", options=...) used
        # to raise TypeError because CudagraphsBackend.__call__ did not accept
        # extra kwargs. The backend should ignore unknown kwargs with a warning,
        # matching eager and other compiler-fn-style backends.
        from torch._dynamo.backends.cudagraphs import CudagraphsBackend

        backend = CudagraphsBackend()
        gm = MagicMock()
        sentinel = object()
        with patch(
            "torch._dynamo.backends.cudagraphs.cudagraphs", return_value=sentinel
        ) as mock_cudagraphs:
            with self.assertLogs(
                "torch._dynamo.backends.cudagraphs", level="WARNING"
            ) as cm:
                result = backend(gm, [], options={"trace.enabled": True})

        self.assertIs(result, sentinel)
        # Extra kwargs are dropped, not forwarded to the inner compiler.
        mock_cudagraphs.assert_called_once_with(gm, [])
        self.assertTrue(any("ignoring extra kwargs" in m for m in cm.output))

    def test_backend_graph_freeze(self):
        # [FUNCTION DOC] Compiles a module whose forward is under the prepare_freezing config
        # using an aot_autograd forward compiler. Inside the compiler, when a tracing context
        # exists, asserts fw_metadata and params_flat are set and params_flat has 2 entries
        # (the two parameters). Finally asserts the compiler ran.
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        backend_run = False

        def my_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Forward compiler; checks the tracing context's metadata
            # when present, flags it ran, returns boxed gm.forward.
            nonlocal backend_run
            if tracing_context := torch._guards.TracingContext.try_get():
                fw_metadata = tracing_context.fw_metadata
                params_flat = tracing_context.params_flat
                self.assertTrue(fw_metadata is not None)
                self.assertTrue(params_flat is not None)
                self.assertTrue(len(params_flat) == 2)
            backend_run = True
            return make_boxed_func(gm.forward)

        my_backend = aot_autograd(fw_compiler=my_compiler)

        class MyClass(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two 2x3 random parameters p1 and p2.
            def __init__(self, *args, **kwargs) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters p1 and p2.
                super().__init__(*args, **kwargs)
                self.p1 = torch.nn.Parameter(torch.randn(2, 3))
                self.p2 = torch.nn.Parameter(torch.randn(2, 3))

            @torch._dynamo.config.patch("prepare_freezing", True)
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns (p1 + x) / p2, run with prepare_freezing on.
                t = self.p1 + x
                out = t / self.p2
                return out

        mod = MyClass()

        opt_mod = torch.compile(mod, backend=my_backend)
        opt_mod(torch.randn(2, 3))
        self.assertTrue(backend_run)


class TestDefaultBackend(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase for torch.compiler.set_default_backend / get_default_backend.
    def test_set_default_backend(self):
        # [FUNCTION DOC] Checks the default backend starts as inductor, can be set by name,
        # reset with None, and set to a callable (a compile counter). Compiling without an
        # explicit backend then uses the callable (frame count 1). Cleanup resets to None.
        self.addCleanup(torch.compiler.set_default_backend, None)

        self.assertEqual(torch.compiler.get_default_backend(), "inductor")

        torch.compiler.set_default_backend("eager")
        self.assertEqual(torch.compiler.get_default_backend(), "eager")

        torch.compiler.set_default_backend(None)
        self.assertEqual(torch.compiler.get_default_backend(), "inductor")

        cnt = torch._dynamo.testing.CompileCounter()
        torch.compiler.set_default_backend(cnt)
        self.assertIs(torch.compiler.get_default_backend(), cnt)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns relu(x).
            return torch.relu(x)

        opt_f = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND
        opt_f(torch.randn(3, 3))
        self.assertEqual(cnt.frame_count, 1)

    def test_set_default_backend_explicit_override(self):
        # [FUNCTION DOC] Sets a recording backend as default, then compiles with an explicit
        # compile-counter backend. Asserts the counter saw one frame and the recording default
        # captured no graphs. Cleanup resets the default to None.
        self.addCleanup(torch.compiler.set_default_backend, None)

        eager_and_record = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compiler.set_default_backend(eager_and_record)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns relu(x).
            return torch.relu(x)

        # Explicit backend= should override the default
        cnt = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnt)
        opt_f(torch.randn(3, 3))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(eager_and_record.graphs), 0)


instantiate_device_type_tests(TestOptimizations, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
