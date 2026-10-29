# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] torch.compile tests for the openreg backend (dynamo). Covers backend
# registration, graph capture, fake tensors, guards/recompilation, autograd, nn.Module,
# dynamic shapes, graph breaks, autocast, default device and the dynamo device
# interface. Uses the dynamo TestCase and run_tests. No import-time side effects.

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounterWithBackend


class TestBackendRegistration(TestCase):
    # [CLASS DOC] Checks that the "openreg" compile backend is registered with dynamo.
    def test_backend_in_list(self):
        # [FUNCTION DOC] Lists dynamo backends (without excluding any tags) and requires
        # that "openreg" is among them.
        from torch._dynamo.backends.registry import list_backends

        backends = list_backends(exclude_tags=())
        self.assertIn("openreg", backends)


class TestGraphCapture(TestCase):
    # [CLASS DOC] Tests that functions compile with the openreg backend and match eager results.
    def test_simple_compile(self):
        # [FUNCTION DOC] Compiles an add-one function and compares output with eager.
        @torch.compile(backend="openreg")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input plus one.
            return x + 1

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, x + 1)

    def test_compile_fullgraph_multiple_ops(self):
        # [FUNCTION DOC] Compiles add, multiply and relu in one full graph and compares
        # with the eager computation.
        @torch.compile(backend="openreg", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function computing relu of (x plus one) times two.
            y = x + 1
            z = y * 2
            return torch.relu(z)

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, torch.relu((x + 1) * 2))


class TestFakeTensor(TestCase):
    # [CLASS DOC] Tests that fake tensors and captured graphs preserve the openreg device.
    def test_device_metadata(self):
        # [FUNCTION DOC] Compiles with a custom capturing backend, then checks every
        # placeholder node whose fake value is a tensor reports device type openreg.
        captured_gm = None

        def capture_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Custom dynamo backend that stores the graph module in the
            # enclosing variable and returns its forward callable.
            nonlocal captured_gm
            captured_gm = gm
            return gm.forward

        @torch.compile(backend=capture_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input plus one.
            return x + 1

        x = torch.randn(4, device="openreg")
        fn(x)

        self.assertIsNotNone(captured_gm)
        for node in captured_gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                fake = node.meta["val"]
                if isinstance(fake, torch.Tensor):
                    self.assertEqual(fake.device.type, "openreg")

    def test_fake_tensor_mode(self):
        # [FUNCTION DOC] Under FakeTensorMode, creating an empty openreg tensor and adding
        # one keeps the openreg device type.
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(3, 3, device="openreg")
            self.assertEqual(x.device.type, "openreg")
            y = x + 1
            self.assertEqual(y.device.type, "openreg")


class TestGuards(TestCase):
    # [CLASS DOC] Tests guard behavior (caching and recompilation) with a counting backend.
    def test_backend_recompilation(self):
        # [FUNCTION DOC] Calling a compiled function twice with same-shaped input compiles
        # once (frame count stays one).
        counter = CompileCounterWithBackend("openreg")

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input plus one.
            return x + 1

        x = torch.randn(4, device="openreg")
        fn(x)
        self.assertEqual(counter.frame_count, 1)

        fn(x)
        self.assertEqual(counter.frame_count, 1)

    def test_recompilation_on_shape_change(self):
        # [FUNCTION DOC] Calling with shape 4 then shape 8 must trigger a second compile
        # (frame count goes from one to two).
        counter = CompileCounterWithBackend("openreg")

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input plus one.
            return x + 1

        x1 = torch.randn(4, device="openreg")
        fn(x1)
        self.assertEqual(counter.frame_count, 1)

        x2 = torch.randn(8, device="openreg")
        fn(x2)
        self.assertEqual(counter.frame_count, 2)


class TestExecution(TestCase):
    # [CLASS DOC] Tests running compiled code: autograd and nn.Module.
    def test_autograd(self):
        # [FUNCTION DOC] Compiles a scale-and-sum function, runs backward and checks the
        # gradient is filled with two.
        @torch.compile(backend="openreg")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the sum of the input times two.
            return (x * 2).sum()

        x = torch.randn(4, device="openreg", requires_grad=True)
        loss = fn(x)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad, torch.full_like(x, 2.0))

    def test_nn_module(self):
        # [FUNCTION DOC] Compiles a small module holding a linear layer and checks the
        # output device, shape and equality with the eager result.
        class SimpleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module with one 4-to-4 linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the module and its linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns it.
                return self.linear(x)

        m = SimpleModule().to("openreg")
        compiled_m = torch.compile(m, backend="openreg")

        x = torch.randn(2, 4, device="openreg")
        result = compiled_m(x)
        self.assertEqual(result.device.type, "openreg")
        self.assertEqual(result.shape, torch.Size([2, 4]))

        eager_result = m(x)
        self.assertEqual(result, eager_result)


class TestDynamicShapes(TestCase):
    # [CLASS DOC] Tests compilation with dynamic shapes enabled.
    def test_dynamic_shapes(self):
        # [FUNCTION DOC] Compiles with dynamic=True and checks results for two different
        # input sizes.
        @torch.compile(backend="openreg", dynamic=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning its input plus one.
            return x + 1

        x4 = torch.randn(4, device="openreg")
        x8 = torch.randn(8, device="openreg")

        self.assertEqual(fn(x4), x4 + 1)
        self.assertEqual(fn(x8), x8 + 1)


class TestGraphBreaks(TestCase):
    # [CLASS DOC] Tests tracing across device transfers and explicit graph breaks.
    def test_device_transfer_traces_fully(self):
        # [FUNCTION DOC] Compiles in full-graph mode a function that adds one, moves to the CPU
        # and doubles; checks values match eager and the result is on the CPU.
        @torch.compile(backend="openreg", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, copies the result to the CPU, multiplies by two.
            y = x + 1
            z = y.cpu()
            return z * 2

        x = torch.randn(4, device="openreg")
        result = fn(x)
        expected = (x + 1).cpu() * 2
        self.assertEqual(result, expected)
        self.assertEqual(result.device.type, "cpu")

    def test_graph_break_resilience(self):
        # [FUNCTION DOC] A function with an explicit graph break still computes the right
        # value and compiles into more than one frame.
        counter = CompileCounterWithBackend("openreg")

        @torch.compile(backend=counter)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, triggers a graph break, then multiplies by two.
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, (x + 1) * 2)
        self.assertGreater(counter.frame_count, 1)


class TestAutocast(TestCase):
    # [CLASS DOC] Tests autocast inside compiled functions on openreg.
    def test_compile_with_autocast(self):
        # [FUNCTION DOC] A compiled function that does a matmul inside a float16 autocast
        # region must return a float16 result.
        @torch.compile(backend="openreg")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matrix multiply of x and y under float16 autocast on openreg.
            with torch.autocast(device_type="openreg", dtype=torch.float16):
                return torch.mm(x, y)

        x = torch.randn(2, 3, device="openreg")
        y = torch.randn(3, 3, device="openreg")
        result = fn(x, y)
        self.assertEqual(result.dtype, torch.float16)


class TestDefaultDevice(TestCase):
    # [CLASS DOC] Tests compilation while the default device is set to openreg.
    def test_compile_with_default_device(self):
        # [FUNCTION DOC] Inside a default-device context of openreg, compiles a function that
        # allocates an uninitialized tensor and adds it; result must be on openreg.
        with torch.device("openreg"):

            @torch.compile(backend="openreg")
            def fn(x):
                # [FUNCTION DOC] HELPER: Allocates an empty tensor (on the default device) and
                # adds it to x.
                y = torch.empty(4)
                return x + y

            x = torch.randn(4, device="openreg")
            result = fn(x)
            self.assertEqual(result.device.type, "openreg")


class TestDeviceInterface(TestCase):
    # [CLASS DOC] Tests the dynamo device interface registered for openreg.
    def test_interface_registered(self):
        # [FUNCTION DOC] Looking up the device interface for "openreg" must return something.
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        self.assertIsNotNone(iface)

    def test_current_device(self):
        # [FUNCTION DOC] The interface's current_device must return an int.
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        device_idx = iface.current_device()
        self.assertIsInstance(device_idx, int)

    def test_device_count(self):
        # [FUNCTION DOC] The interface's device_count must be greater than zero.
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        self.assertGreater(iface.device_count(), 0)


if __name__ == "__main__":
    run_tests()
