# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for the dtype propagation pass, which infers output tensor dtypes of
# a graph from input example tensors (it depends on symbolic shape analysis, still in beta).
# Part one tests rules derived from meta tensors (unary, binary, scalar mixes) and part two
# checks the hand-written custom rules over operator database entries, device-generic and CPU
# only. Defines module-level sets and lists of op names for ops that work, are expected to
# fail, or are not covered by the op database. Import time instantiates the device-specific
# test class (CPU only) into the module globals. Running the file directly is refused in favor
# of test/test_jit.py.
from itertools import product
from typing import Tuple
from unittest.case import expectedFailure

import torch
from torch import complex32, float32, float64, int32, int64
from torch.jit._passes import _property_propagation
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
    sample_inputs_adaptive_avg_pool2d,
    sample_inputs_conv2d,
    SampleInput,
)
from torch.testing._internal.common_utils import (
    first_sample,
    raise_on_run_directly,
    set_default_dtype,
)
from torch.testing._internal.jit_metaprogramming_utils import create_traced_fn
from torch.testing._internal.jit_utils import JitTestCase


"""
Dtype Analysis relies on symbolic shape analysis, which is still in beta
"""


custom_rules_works_list = {
    "nn.functional.adaptive_avg_pool1d",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool1d",
    "nn.functional.adaptive_max_pool2d",
    "avg_pool1d",
    "avg_pool3d",
    "conv_transpose2d",
    "conv1d",
    "conv2d",
    "hardswish",
    "avg_pool2d",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "nn.functional.prelu",
    "batch_norm",
}

custom_rules_expected_failure_list = {
    # create_traced_fn generates prim::NumToTensor nodes in graph (not supported yet)
    "nn.functional.adaptive_max_pool3d",
}

# These ops seem to not be in opinfos
custom_rules_not_tested_list = [
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose3d",
    "convolution",
    "_convolution",
    "max_unpool2d",
    "max_unpool3d",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
    "upsample_bilinear2d",
    "upsample_linear1d",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_nearest3d",
    "upsample_trilinear3d",
    "flatten",
]


class TestDtypeBase(JitTestCase):
    # [CLASS DOC] HELPER: Shared base for dtype propagation tests. Provides a marker constant
    # for scalar arguments, setup/teardown that put symbolic shape analysis into test mode,
    # and helpers that run the pass on a graph and compare the inferred output dtypes with an
    # eager run.
    SCALAR = "SCALAR"  # To mark unary vs 0 dim tensor

    def setUp(self):
        # [FUNCTION DOC] Per-test setup that deliberately skips the parent setup (which installs
        # emit hooks that cause segfaults at process exit). Records the torch function mode
        # stack length and state, remembers the previous symbolic shape test mode flag, and
        # turns symbolic shape test mode on.
        # KNOWN ISSUE: the in-code comment says tearDown checks the recorded torch function
        # stack baselines, but tearDown only restores the symbolic shape test mode flag; the two
        # function state attributes are never read.
        # Don't call super().setUp() — JitTestCase.setUp installs JIT emit
        # hooks that cause segfaults during process cleanup. Record state
        # baselines that tearDown checks for.
        self._prev_torch_function_mode_stack_len = torch._C._len_torch_function_stack()
        self._prev_torch_function_state = torch._C._get_torch_function_state()
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        # [FUNCTION DOC] Per-test cleanup restoring the symbolic shape test mode flag saved in
        # setup. Does not call the parent teardown.
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled
        )

    @staticmethod
    def node_output_dtypes(graph):
        # [FUNCTION DOC] HELPER: Static method returning a list with one entry per graph output:
        # the dtype if the output type is a tensor type, otherwise None.
        dtypes = []
        for out in graph.outputs():
            if isinstance(out.type(), torch._C.TensorType):
                dtypes.append(out.type().dtype())
            else:
                dtypes.append(None)
        return dtypes

    @staticmethod
    def node_output_dtype_single(graph):
        # [FUNCTION DOC] HELPER: Static method returning the dtype of the graph's only output;
        # raises AssertionError if the graph has a number of outputs other than one. The dtype
        # may be None for non-tensor output.
        dtypes = TestDtypeBase.node_output_dtypes(graph)
        if len(dtypes) != 1:
            raise AssertionError(f"Expected 1 output dtype, got {len(dtypes)}")
        return dtypes[0]

    def prop_dtype_on_graph(self, graph, example_inputs):
        # [FUNCTION DOC] HELPER: Runs dtype propagation on a graph using example inputs: erases
        # existing shape information first (scripting returns a cached graph on repeats),
        # applies input properties from the examples, propagates shapes, then dtypes. Mutates
        # the graph in place.
        # We need to clear shape information because torch.jit.script
        # will return a cached graph if the function is scripted twice.
        torch._C._jit_pass_erase_shape_information(graph)
        _property_propagation.apply_input_props_using_example(graph, example_inputs)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        torch._C._jit_pass_propagate_dtype(graph)

    def assert_dtype_equal(self, fn, in_shapes, in_dtypes):
        # [FUNCTION DOC] HELPER: Builds random inputs from parallel lists of shapes and dtypes
        # and delegates to the custom-args check. Any failure is re-raised as AssertionError
        # mentioning the shapes and dtypes, chained to the original error.
        inputs = [self.get_rand_tensor(s, d) for s, d in zip(in_shapes, in_dtypes)]
        try:
            self.assert_dtype_equal_custom_args(fn, inputs)
        except Exception as e:
            fail_text = f"Failed for shapes {in_shapes}, and dtypes {in_dtypes}"
            raise AssertionError(fail_text) from e

    def assert_dtype_equal_custom_args(self, fn, args):
        # [FUNCTION DOC] HELPER: Runs the function eagerly on the arguments; if eager raises
        # RuntimeError (unsupported combination) the check is silently skipped. Otherwise
        # scripts the function, propagates dtypes from the arguments and asserts the graph's
        # single output dtype equals the eager result dtype.
        try:
            # Eager execution
            expected_res = fn(*args)
        except RuntimeError:
            return

        expected_dtype = expected_res.dtype

        # Run the Dtype Analysis
        graph = torch.jit.script(fn).graph  # Note this is a cached graph
        self.prop_dtype_on_graph(graph, args)
        actual_dtype = self.node_output_dtype_single(graph)

        self.assertEqual(actual_dtype, expected_dtype, "Failed Verification")

    def get_rand_tensor(self, shape, dtype):
        # [FUNCTION DOC] HELPER: Returns a random value for a shape and dtype. For the scalar
        # marker returns a Python float for float32 or an int for int64 and raises RuntimeError
        # for other dtypes. For int32 and int64 returns small random integers, otherwise uniform
        # random floats; asserts the resulting dtype matches the request.
        if shape is self.SCALAR:
            if dtype is float32:
                return 1.1
            elif dtype is int64:
                return 2
            else:
                raise RuntimeError(
                    "Testing of scalars only supported for fp32 and int64"
                )

        if dtype in (int32, int64):
            rand_tensor = torch.randint(0, 10, shape, dtype=dtype)
        else:
            rand_tensor = torch.rand(shape, dtype=dtype)

        # Sanity check!

        self.assertEqual(rand_tensor.dtype, dtype)
        return rand_tensor


class TestDtypeAnalysis(TestDtypeBase):
    # [CLASS DOC] Tests of dtype propagation rules obtained from meta tensors (unary and binary
    # ops) and of custom rules for convolution and pooling, combining shapes of various ranks
    # including empty and zero-dim.
    def test_unary(self):
        # [FUNCTION DOC] Checks in-place relu and log over shapes with a regular, an empty and a
        # zero-dim tensor, and float32, int64 and complex dtypes; the inferred output dtype must
        # match eager (combinations that fail eagerly are skipped).
        # Testing the Unary Implementation that uses metatensors

        def relu_inplace(x):
            # [FUNCTION DOC] HELPER: Applies in-place relu to the input and returns it.
            return x.relu_()

        def log(x):
            # [FUNCTION DOC] HELPER: Returns the elementwise natural log of the input.
            return torch.log(x)

        functions = [relu_inplace, log]

        input_shapes = [
            ((2, 2),),  # Simple Case
            ((0, 2),),  # Size 0 Tensor
            ((),),  # zerodim
        ]

        input_dtypes = [
            (float32,),  # Simple Case
            (int64,),  # Test how some unary ops implicitly convert to float
            (complex32,),  # Show we can handle complex vals as well
        ]

        for fn, in_shapes, in_dtypes in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)

    def test_binary_tensors(self):
        # [FUNCTION DOC] Checks add and true divide on pairs of tensors with differing ranks,
        # zero-dim operands and an empty dimension, across dtype pairs that exercise type
        # promotion including complex; inferred dtype must match eager.
        # Testing using Metatensors
        def add(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
            return x + y

        def div(x, y):
            # [FUNCTION DOC] HELPER: Returns the true quotient of the two inputs.
            return x / y

        functions = [add, div]

        input_shapes = [
            ((1, 1, 2), (1, 2)),  # Different Dim, non-zerodim
            ((), (1, 2)),  # One zerodim
            ((1, 2), ()),  # Other zerodim
            ((2, 0, 3), (1, 3)),  # Test a tensor with a dim of 0
            ((), ()),  # both zerodim
        ]

        input_dtypes = [
            (float32, float32),  # Simple Case
            (int32, int64),  # Size Promotion (compliated case for 0dim tensors)
            (float32, int32),  # type Promotion
            (int64, float32),  # Type promotion with size change
            (float64, complex32),  # Show we can handle complex vals as well
        ]

        for fn, in_shapes, in_dtypes in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)

    def test_binary_scalar(self):
        # [FUNCTION DOC] Checks adding a tensor and a Python scalar under default dtype float32.
        # For each shape pair (regular or zero-dim tensor with a scalar) and dtype pair, defines
        # an add function whose scalar parameter is annotated float or int depending on the
        # second dtype, then compares inferred dtype to eager.
        # Test the mixing of scalar and non-scalar args

        input_shapes = [
            ((2, 2), self.SCALAR),  # Non-Zerodim vs scalar
            ((), self.SCALAR),  # Zerodim vs scalar
            # Scalar vs Scalar is automatically inferred.
        ]

        input_dtypes = [
            (float32, float32),  # Simple Case
            (int32, int64),  # Size Promotion (compliated case for 0dim tensors)
            (int32, float32),  # type Promotion
        ]

        with set_default_dtype(float32):
            for in_shapes, in_dtypes in product(input_shapes, input_dtypes):
                scalar_type = in_dtypes[1]

                if scalar_type == float32:

                    def add(x, y: float):
                        # [FUNCTION DOC] HELPER: Returns a tensor plus a float scalar argument.
                        return x + y

                else:

                    def add(x, y: int):
                        # [FUNCTION DOC] HELPER: Returns a tensor plus an integer
                        # scalar argument.
                        return x + y

                self.assert_dtype_equal(add, in_shapes, in_dtypes)

    def test_custom_rules(self):
        # [FUNCTION DOC] Checks custom dtype rules for ops not handled by meta tensors: for
        # conv2d and adaptive average pooling with int8 and float64, takes the last sample input
        # from the op database helper and compares inferred and eager dtypes.
        # Test some of the ops that are not covered by Metatensors

        # Note that unlike the Conv2d module, the function conv2d
        # does not take dtype/device arguments.

        def conv2d_fn(input, weight, bias):
            # [FUNCTION DOC] HELPER: Functional conv2d of input, weight and bias.
            return torch.nn.functional.conv2d(input, weight, bias)

        def adaptive_avg_pool2d_fn(input, output_size: Tuple[int]):
            # [FUNCTION DOC] HELPER: Adaptive average pooling over two dimensions with a given
            # output size.
            return torch._C._nn.adaptive_avg_pool2d(input, output_size)

        for fn, inputs_fn in (
            (conv2d_fn, sample_inputs_conv2d),
            (adaptive_avg_pool2d_fn, sample_inputs_adaptive_avg_pool2d),
        ):
            for dtype in (torch.int8, torch.float64):
                # Gets default version for conv2d
                sample_input: SampleInput = list(inputs_fn(None, "cpu", dtype, False))[
                    -1
                ]
                input_args = [sample_input.input, *sample_input.args]
                self.assert_dtype_equal_custom_args(fn, input_args)

    def test_conv_no_mixed_args(self):
        # [FUNCTION DOC] Checks conv2d does not accept mixed dtypes: with float input and a
        # long-typed weight the eager call must raise RuntimeError, and propagation on the
        # scripted graph must leave the output dtype unknown (None).
        def conv2d_fn(input, weight, bias):
            # [FUNCTION DOC] HELPER: Functional conv2d of input, weight and bias.
            return torch.nn.functional.conv2d(input, weight, bias)

        # Now make sure that conv2d doesn't support mixed args
        conv_ins = sample_inputs_conv2d(None, "cpu", torch.float, False)
        conv_in = list(conv_ins)[-1]
        weight, bias = conv_in.args
        weight = weight.type(torch.long)

        with self.assertRaises(RuntimeError):
            conv2d_fn(conv_in.input, weight, bias)

        # Check that we also don't propagate
        graph = torch.jit.script(conv2d_fn).graph  # Note this is a cached graph
        self.prop_dtype_on_graph(graph, [conv_in.input, weight, bias])
        actual_dtype = self.node_output_dtype_single(graph)
        self.assertEqual(actual_dtype, None)

    def test_combined(self):
        # [FUNCTION DOC] Checks a graph mixing custom rules and meta tensors: int8 conv2d output
        # added to a float32 one-element tensor, flattened from dimension two, and added again;
        # inferred dtype must match eager.
        # Test a case with both custom rules and metatensors

        def func(input, weight, bias, y):
            # [FUNCTION DOC] HELPER: Convolution of the inputs, adds y, flattens from dimension
            # two, adds y again and returns the result.
            conv_out = torch.nn.functional.conv2d(input, weight, bias)
            conv_2 = conv_out + y
            flattened = torch.flatten(conv_2, start_dim=2)
            add_res = flattened + y
            return add_res

        conv_ins = sample_inputs_conv2d(None, "cpu", torch.int8, False)
        conv_in = list(conv_ins)[-1]
        y_val = torch.rand((1,), dtype=torch.float32)
        input_args = [conv_in.input, *conv_in.args, y_val]
        self.assert_dtype_equal_custom_args(func, input_args)


class TestDtypeCustomRules(TestDtypeBase):
    # [CLASS DOC] Device-generic tests of the custom dtype rules over operator database entries.
    # Instantiated for CPU only at import time.
    def assert_output_dtype_equal(self, expected_res, prop_graph):
        # [FUNCTION DOC] HELPER: Compares eager results with the dtypes of the propagated graph
        # outputs. With one graph output the eager result is compared directly; otherwise
        # lengths must match and each pair is compared.
        actual_dtype = self.node_output_dtypes(prop_graph)
        if len(actual_dtype) == 1:
            # For len=1, there is no tuple packing for expected_res.
            self.assert_tensor_dtype_equal(expected_res, actual_dtype[0])
        else:
            self.assertEqual(len(expected_res), len(actual_dtype))
            for expected, actual in zip(expected_res, actual_dtype):
                self.assert_tensor_dtype_equal(expected, actual)

    def assert_tensor_dtype_equal(self, tensor_output, graph_dtype):
        # [FUNCTION DOC] HELPER: Asserts a tensor's dtype equals the graph dtype; non-tensor
        # eager outputs are ignored.
        if not isinstance(tensor_output, torch.Tensor):
            return
        self.assertEqual(tensor_output.dtype, graph_dtype)

    def custom_rules_test_base(self, device, dtype, op, allow_eager_fail=False):
        # [FUNCTION DOC] HELPER: Core of the custom rule tests. Takes the first sample input of
        # an operator for the device and dtype and runs the op eagerly (exceptions are swallowed
        # with an early return when eager failure is allowed, otherwise re-raised). Then builds
        # a traced function for the op, runs it once to record the trace, collects tensor
        # positional and keyword inputs, propagates dtypes on the traced graph and compares to
        # the eager outputs.
        try:
            samples = op.sample_inputs(device, dtype, requires_grad=False)
            sample_input = first_sample(self, samples)
            input_args = [sample_input.input, *sample_input.args]
            expected_res = op(*input_args, **sample_input.kwargs)

        except Exception as e:
            if allow_eager_fail:
                return
            else:
                raise e

        func = op.get_op()
        traced_fn = create_traced_fn(self, func)

        # Have to run the traced function to actually generate the trace
        traced_fn(sample_input.input, *sample_input.args, **sample_input.kwargs)

        # Run the Dtype Analysis
        graph = traced_fn.graph  # Note this is a cached graph
        input_tensors = [t for t in input_args if isinstance(t, torch.Tensor)]
        input_tensors += [
            v for v in sample_input.kwargs.values() if isinstance(v, torch.Tensor)
        ]
        self.prop_dtype_on_graph(graph, input_tensors)
        self.assert_output_dtype_equal(expected_res, graph)

    @ops([op for op in op_db if op.aten_name in custom_rules_works_list])
    def test_custom_rules(self, device, dtype, op):
        # [FUNCTION DOC] Runs the core check for every operator in the works list with the dtype
        # supplied by the test framework.
        self.custom_rules_test_base(device, dtype, op)

    @ops([op for op in op_db if op.aten_name in custom_rules_works_list])
    def test_custom_rules_ints(self, device, dtype, op):
        # [FUNCTION DOC] Runs the core check for the works-list operators with an integer dtype
        # (int32 if the framework dtype is float32, else int64), since the op database only runs
        # floats here; eager failures are tolerated because integer kernels may be missing.
        # This is done because opinfos currently only runs on floats.
        # Return fn, inputs_fn for all
        if dtype == torch.float32:
            dtype = torch.int32
        else:
            dtype = torch.int64

        # Because ints are not always implemented, we need to allow for eager to fail
        self.custom_rules_test_base(device, dtype, op, allow_eager_fail=True)

    @expectedFailure
    @ops([op for op in op_db if op.aten_name in custom_rules_expected_failure_list])
    def test_custom_rules_expected_failure(self, device, dtype, op):
        # [FUNCTION DOC] Marked as expected failure: runs the core check for the operator in the
        # expected failure list, which fails because tracing produces number-to-tensor nodes not
        # yet supported.
        self.custom_rules_test_base(device, dtype, op)


TestDtypeCustomRulesCPU = None
# This creates TestDtypeCustomRulesCPU
instantiate_device_type_tests(TestDtypeCustomRules, globals(), only_for=("cpu",))

if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
