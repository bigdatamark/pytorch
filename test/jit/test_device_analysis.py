# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for the device propagation pass: after input types get device
# annotations (and optionally shapes), the pass infers the device of the graph output, or
# leaves it unknown when inputs are missing or conflicting. Covers simple ops, device-setting
# ops, zero-dim tensors, control flow and a torchvision model. torchvision is imported
# optionally (models becomes None when missing). Running the file directly is refused in favor
# of test/test_jit.py.
import unittest
from itertools import product

import torch
from torch.jit._passes._property_propagation import apply_input_props_using_example
from torch.testing._internal.common_utils import raise_on_run_directly, TEST_CUDA
from torch.testing._internal.jit_utils import JitTestCase


try:
    from torchvision import models
except ImportError:
    models = None


class TestDeviceAnalysis(JitTestCase):
    # [CLASS DOC] Test case for device propagation. Class-level device objects (cpu, cuda,
    # vulkan, mkldnn and a list of the first three) are created once in setUpClass; helpers set
    # input devices on a scripted graph, run the pass and compare the single output device type
    # to expectations (None means unknown).
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class-level setup creating the shared device objects: cpu, cuda, vulkan
        # and mkldnn (mkldnn cannot be mixed with other devices), plus a list of device types
        # holding cpu, cuda and vulkan.
        cls.cpu = torch.device("cpu")
        cls.cuda = torch.device("cuda")
        cls.vulkan = torch.device("vulkan")
        cls.mkldnn = torch.device(
            "mkldnn"
        )  # MKLDNN can't mix with other device types at all
        cls.device_types = [cls.cpu, cls.cuda, cls.vulkan]

    @staticmethod
    def node_output_device(graph):
        # [FUNCTION DOC] HELPER: Static method returning the device recorded on the type of the
        # graph's only output; raises AssertionError if the graph does not have exactly one
        # output. May be None when unknown.
        graph_out = list(graph.outputs())
        if len(graph_out) != 1:
            raise AssertionError(f"Expected 1 graph output, got {len(graph_out)}")
        return graph_out[0].type().device()

    def prop_device_on_graph(self, graph, example_devices, in_shapes=None):
        # [FUNCTION DOC] HELPER: Runs device propagation on a graph. Erases existing shape info,
        # asserts one example device per graph input, sets each non-None device on the matching
        # input type, and when input shapes are given sets the non-None sizes and propagates
        # shapes first. Then runs the device propagation pass. Mutates the graph in place.
        graph_inputs = list(graph.inputs())
        torch._C._jit_pass_erase_shape_information(graph)

        self.assertEqual(len(graph_inputs), len(example_devices))
        for graph_i, device_i in zip(graph_inputs, example_devices):
            if device_i is not None:
                graph_i.setType(graph_i.type().with_device(device_i))

        if in_shapes:
            for graph_i, shapes_i in zip(graph_inputs, in_shapes):
                if shapes_i is not None:
                    graph_i.setType(graph_i.type().with_sizes(shapes_i))

            torch._C._jit_pass_propagate_shapes_on_graph(graph)

        torch._C._jit_pass_propagate_device(graph)

    def assert_device_equal(
        self, fn, in_devices, expected_device, in_shapes=None, subtest_str=""
    ):
        # [FUNCTION DOC] HELPER: Scripts a function, propagates the given input devices (and
        # optional shapes) and checks the output device inside a subtest labelled with the
        # inputs. If either the actual or expected device is None they must be equal; otherwise
        # only the device types are compared.
        with self.subTest(
            f"In device: {in_devices}, expected: {expected_device}, \n {subtest_str}"
        ):
            graph = torch.jit.script(fn).graph
            self.prop_device_on_graph(graph, in_devices, in_shapes)
            actual_device = self.node_output_device(graph)

            if expected_device is None or actual_device is None:
                self.assertEqual(actual_device, expected_device)
            else:
                self.assertEqual(
                    actual_device.type, expected_device.type, "Failed Verification"
                )

    def test_device_apply(self):
        # [FUNCTION DOC] Checks setting a device on a graph input type is reflected when reading
        # the type back: scripts a doubling function, annotates its input with cpu and asserts
        # the device equals cpu. No propagation is run.
        # Test if the device is properly applied to the input
        def add_self(x):
            # [FUNCTION DOC] HELPER: Returns the input added to itself.
            return x + x

        graph = torch.jit.script(add_self).graph
        graph_input = next(graph.inputs())
        graph_input.setType(graph_input.type().with_device(self.cpu))
        # self.prop_device_on_graph(graph, [self.cpu])
        self.assertEqual(graph_input.type().device(), self.cpu)

    @unittest.skipIf(models is None, "Requires torchvision")
    def test_mobilenet(self):
        # [FUNCTION DOC] Runs propagation on a frozen scripted mobilenet_v3_small (skipped
        # without torchvision) in eval mode. Input properties come from a random CPU example,
        # shapes and devices are propagated, and the output device type must be cpu.
        in_cpu = torch.randn(1, 3, 224, 224, device=self.cpu)
        in_example = in_cpu

        expected_device = self.cpu
        m = torch.jit.script(models.mobilenet_v3_small())
        m.eval()
        graph = torch.jit.freeze(m).graph
        # torch._C._jit_pass_erase_shape_information(graph)
        apply_input_props_using_example(graph, in_example)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        torch._C._jit_pass_propagate_device(graph)

        actual_device = self.node_output_device(graph)

        if expected_device is None or actual_device is None:
            self.assertEqual(actual_device, expected_device)
        else:
            self.assertEqual(
                actual_device.type, expected_device.type, "Failed Verification"
            )

    def test_simple(self):
        # [FUNCTION DOC] For each pair of a device from the standard list and a simple function
        # (add to self, in-place relu), checks the output device equals the input device.
        def add_self(x):
            # [FUNCTION DOC] HELPER: Returns the input added to itself.
            return x + x

        def relu_(x):
            # [FUNCTION DOC] HELPER: Applies in-place relu to the input and returns it.
            return torch.nn.functional.relu_(x)

        functions = [add_self, relu_]

        for in_device, fn in product(self.device_types, functions):
            self.assert_device_equal(fn, [in_device], in_device)

    def test_set_dtype(self):
        # [FUNCTION DOC] For each standard input device, a function that moves its input to cpu
        # by string must have a cpu output device.
        def set_device(x):
            # [FUNCTION DOC] HELPER: Moves the input to cpu using a device string.
            return x.to("cpu")

        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device], self.cpu)

    def test_device_arg(self):
        # [FUNCTION DOC] Checks that when the target device is a non-constant argument, no
        # output device is inferred: for each standard input device, the output device must
        # be unknown.
        # Test that no device gets propagated when arg is passed in
        def set_device(x, device_name: torch.device):
            # [FUNCTION DOC] HELPER: Moves the first argument to the device given as the
            # second argument.
            return x.to(device=device_name)

        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device, None], None)

    def test_tensor_as_fns(self):
        # [FUNCTION DOC] Checks device inference for ops taking a second tensor. For view_as,
        # expand_as and reshape_as the expected output device follows the first input (cpu with
        # cpu, cuda with unknown) and is unknown when only the second is a mkldnn tensor. For
        # type_as the expectation differs: cpu and cpu gives cpu, cuda with unknown gives
        # unknown, and unknown with mkldnn gives mkldnn.
        def view_as_fn(x, y):
            # [FUNCTION DOC] HELPER: Views the first tensor with the shape of the second.
            return x.view_as(y)

        def expand_as_fn(x, y):
            # [FUNCTION DOC] HELPER: Expands the first tensor to the shape of the second.
            return x.expand_as(y)

        def reshape_as_fn(x, y):
            # [FUNCTION DOC] HELPER: Reshapes the first tensor to the shape of the second.
            return x.reshape_as(y)

        for test_fn in [view_as_fn, expand_as_fn, reshape_as_fn]:
            self.assert_device_equal(test_fn, [self.cpu, self.cpu], self.cpu)
            self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
            self.assert_device_equal(test_fn, [None, self.mkldnn], None)

        def type_as_fn(x, y):
            # [FUNCTION DOC] HELPER: Casts the first tensor to the type of the second.
            return x.type_as(y)

        self.assert_device_equal(type_as_fn, [self.cpu, self.cpu], self.cpu)
        self.assert_device_equal(type_as_fn, [self.cuda, None], None)
        self.assert_device_equal(type_as_fn, [None, self.mkldnn], self.mkldnn)

    def zerodim_test_core(self, device_pairs):
        # [FUNCTION DOC] HELPER: Shared body for zero-dim tensor tests over a list of device
        # pairs. For multiply and add, three shape pairs (different rank, one zero-dim, both
        # zero-dim) and each device pair, creates random inputs and runs eager. If eager raises:
        # re-raises when a zero-dim input is on cpu or when both devices are equal; otherwise
        # expects unknown output device with shapes given and moves on. If eager succeeds,
        # expects the eager output device with shapes given, and without shapes expects the
        # result to be unknown or of the same device type.
        # Test the support of zerodim tensors with non-zerodim tensors
        def mul(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise product of the two inputs.
            return x * y

        def add(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two inputs.
            return x + y

        fns = [mul, add]

        input_shapes = [
            ((1, 2, 2), (2, 2)),  # Different dim, non-zerodim
            ((1, 2, 2), ()),  # one zerodim
            ((), ()),  # both zerodim
        ]

        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            subtest_str = f"{fn.__name__} \n shapes: {shapes}, \n devices: {devices}"
            in0 = torch.rand(shapes[0], device=devices[0])
            in1 = torch.rand(shapes[1], device=devices[1])

            try:
                out = fn(in0, in1)
            except Exception as e:
                # Don't expect eager failures for CPU zerodim tensors
                for i in range(len(devices)):
                    if shapes[i] == () and devices[i] == self.cpu:
                        raise e

                # only expect eager failures on different devices
                if devices[0] == devices[1]:
                    raise e

                # Expect result device to be None for the failure cases.
                self.assert_device_equal(fn, devices, None, shapes, subtest_str)
                continue

            self.assert_device_equal(fn, devices, out.device, shapes, subtest_str)

            # Test that without shapes, we either get the same device or None for the device
            # Aka that the code is convservative for tensor shapes.
            graph = torch.jit.script(fn).graph
            self.prop_device_on_graph(graph, devices)
            actual_device = self.node_output_device(graph)
            self.assertTrue(
                (actual_device is None) or (actual_device.type == out.device.type)
            )

    def test_zerodim_cpu(self):
        # [FUNCTION DOC] Local-friendly zero-dim test running the shared body with only a cpu
        # and cpu pair.
        # Allow for minimal testing locally
        self.zerodim_test_core([(self.cpu, self.cpu)])

    def test_zerodim_no_device(self):
        # [FUNCTION DOC] Checks that whenever either input device is unknown, the output device
        # is unknown, for multiply and add over three shape pairs and device pairs with unknown
        # entries (cpu with unknown, unknown with cpu, both unknown).
        # If device is missing, you should never be able to infer device type.
        def mul(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise product of the two inputs.
            return x * y

        def add(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two inputs.
            return x + y

        fns = [mul, add]

        device_pairs = [
            (self.cpu, None),
            (None, self.cpu),
            (None, None),
        ]

        input_shapes = [
            ((1, 2, 2), (2, 2)),  # Different dim, non-zerodim
            ((1, 2, 2), ()),  # one zerodim
            ((), ()),  # both zerodim
        ]

        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            self.assert_device_equal(fn, devices, None, shapes)

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_zerodim_gpu(self):
        # [FUNCTION DOC] Runs the shared zero-dim body over cpu/cuda, cuda/cpu and cuda/cuda
        # device pairs. Skipped when CUDA is unavailable.
        device_pairs = [
            (self.cpu, self.cuda),
            (self.cuda, self.cpu),
            (self.cuda, self.cuda),
        ]
        self.zerodim_test_core(device_pairs)

    def test_custom_device_op(self):
        # [FUNCTION DOC] For ops that set a device (cuda, cpu, to mkldnn), checks that for
        # every standard input device the output device is the target one: cuda, cpu and
        # mkldnn respectively.
        # Test both of the custom functions and check that the devicetype is
        # correctly applied
        def set_cuda(x):
            # [FUNCTION DOC] HELPER: Moves the input to the cuda device.
            return x.cuda()

        def set_cpu(x):
            # [FUNCTION DOC] HELPER: Moves the input to the cpu.
            return x.cpu()

        def set_mkldnn(x):
            # [FUNCTION DOC] HELPER: Converts the input to the mkldnn layout.
            return x.to_mkldnn()

        device_pairs = (
            (set_cuda, self.cuda),
            (set_cpu, self.cpu),
            (set_mkldnn, self.mkldnn),
        )

        for fn, out_device in device_pairs:
            for in_device in self.device_types:
                self.assert_device_equal(fn, [in_device], out_device)

    def test_device_if_propagation(self):
        # [FUNCTION DOC] Checks propagation through a conditional: both branch results must
        # agree. cpu and cpu gives cpu, mkldnn and mkldnn gives mkldnn, and cpu with cuda
        # gives unknown.
        def test_fn(x, y, z: bool):
            # [FUNCTION DOC] HELPER: Branches on a boolean argument: returns the first input
            # plus a constant, else the second input times a constant.
            if z:
                return x + 3
            else:
                return y * 2

        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.mkldnn, self.mkldnn, None], self.mkldnn)
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None], None)

    def test_loop_simple(self):
        # [FUNCTION DOC] Checks propagation through a loop that may or may not overwrite a
        # variable: cpu and cpu gives cpu, cpu with cuda gives unknown, and cpu with unknown
        # gives unknown.
        def test_fn(x, y, z: int):
            # [FUNCTION DOC] HELPER: Loops an integer number of times assigning the first input
            # to the second, then returns the second.
            for _ in range(z):
                y = x
            return y

        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None], None)
        self.assert_device_equal(test_fn, [self.cpu, None, None], None)

    def test_loop_device_change(self):
        # [FUNCTION DOC] Checks a loop that changes device: cpu input gives unknown (the loop
        # may run zero times), cuda input gives cuda, unknown input gives unknown.
        def test_fn(x, z: int):
            # [FUNCTION DOC] HELPER: Loops a given number of times, moving the input to cuda
            # each iteration, and returns it.
            for _ in range(z):
                x = x.cuda()
            return x

        self.assert_device_equal(test_fn, [self.cpu, None], None)
        self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_while_change(self):
        # [FUNCTION DOC] Same as the loop device change test but with a while loop: cpu input
        # gives unknown, cuda gives cuda, unknown gives unknown.
        def test_fn(x, z: int):
            # [FUNCTION DOC] HELPER: While the counter is positive, moves the input to cuda and
            # zeroes the counter; returns the input.
            while z > 0:
                x = x.cuda()
                z = 0
            return x

        self.assert_device_equal(test_fn, [self.cpu, None], None)
        self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_nested_loops(self):
        # [FUNCTION DOC] Checks nested loops where the outer loop moves to cpu: cpu input gives
        # cpu, cuda input gives unknown (outer loop might not run), unknown gives unknown.
        def test_fn(x, z: int):
            # [FUNCTION DOC] HELPER: Outer loop moves the input to cpu each iteration and runs
            # an inner loop adding one; returns the input.
            for i in range(z):
                x = x.cpu()
                for _ in range(i):
                    x = x + 1

            return x

        self.assert_device_equal(test_fn, [self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.cuda, None], None)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_if_loop_mix(self):
        # [FUNCTION DOC] Checks propagation through a while loop containing a conditional: cpu
        # and cpu gives cpu, mkldnn and mkldnn gives mkldnn, cpu with cuda gives unknown.
        def test_fn(x, y, z: bool, a: bool):
            # [FUNCTION DOC] HELPER: Starts with a copy of the first input; inside a while loop
            # on a flag, picks either the first input plus a constant or the second input times
            # a constant based on another flag, then clears the loop flag; returns the result.
            c = x
            while a:
                if z:
                    c = x + 3
                else:
                    c = y * 2
                a = False
            return c

        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None, None], self.cpu)
        self.assert_device_equal(
            test_fn, [self.mkldnn, self.mkldnn, None, None], self.mkldnn
        )
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None, None], None)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
