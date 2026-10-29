# Owner(s): ["oncall: cpu inductor"]
# [MODULE DOC] Tests for the inductor mkldnn (oneDNN) CPU pattern matcher passes, which fuse
# convolution, linear, pointwise, binary and quantized patterns into mkldnn operators under
# freezing. Defines module-level tables of unary and binary operators with their expected match
# node counts, a helper counting expected generated kernels, a base test class that compiles a
# module and checks numerics against eager plus pattern-matcher counters, and generic, dynamic
# shape test classes; the generic ones are instantiated per device type (cpu, xpu) at the end.
import contextlib
import copy
import itertools
import unittest

import torch
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    is_mkldnn_bf16_supported,
    is_mkldnn_fp16_supported,
    run_and_get_code,
)
from torch.nn import functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.testing._internal.common_quantization import (
    skipIfNoDynamoSupport,
    skipIfNoONEDNN,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_LINUX,
    requires_mkl,
    skipIfXpu,
    TEST_ACL,
    xfailIfACL,
)
from torch.testing._internal.inductor_utils import (
    _check_has_dynamic_shape,
    clone_preserve_strides_offset,
    HAS_CPU,
)


# The dict value is match_nodes(computation_op+unary_op)

unary_list = {
    torch.nn.ReLU(): 2,
    torch.nn.Sigmoid(): 2,
    torch.nn.Tanh(): 2,
    torch.nn.Hardswish(): 6,
    torch.nn.LeakyReLU(0.1, inplace=False): 4,
    # Use floats for min/max, otherwise they can get converted to symints
    torch.nn.Hardtanh(min_val=-0.5, max_val=4.0, inplace=False): 3,
    torch.nn.Hardtanh(min_val=-0.5, max_val=float("inf"), inplace=False): 3,
    torch.nn.GELU(approximate="none"): 6,
    torch.nn.GELU(approximate="tanh"): 10,
    torch.nn.ReLU6(): 3,
    torch.nn.SiLU(): 5,
    torch.nn.Hardsigmoid(): 5,
}

non_decomposed_unary_list = [
    torch.nn.ReLU,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
]

# The dict value is (match_count, match_nodes, inplace)
binary_list = {
    lambda x, y: torch.add(x, y): (1, 2, False),  # call_function
    lambda x, y: torch.add(y, x): (1, 2, False),  # call_function
    lambda x, y: x.add(y): (1, 2, False),  # call_method
    lambda x, y: x.add_(y): (1, 2, True),  # call_method
    lambda x, y: torch.sub(x, y): (1, 2, False),  # call_function
    lambda x, y: x.sub(y): (1, 2, False),  # call_method
    lambda x, y: x.sub_(y): (1, 2, True),  # call_method
}

quantization_add_fn_list = [
    lambda x, y: torch.add(x, y),
    lambda x, y: x.add(y),
]

quantization_inplace_add_fn_list = [
    lambda x, y: x.add_(y),
]


def cal_conv_generated_kernel_number(mod, input, dtype, dim=4, device="cpu"):
    # [FUNCTION DOC] HELPER: Estimates how many kernels inductor should generate for a conv model.
    # Deep-copies the module and input to the device, runs the module once in no-grad mode (under
    # autocast for non-float32 dtypes) to get the eager output, then adds one kernel for the input
    # conversion when the input is contiguous, dtype is not float32, or ACL with 4D input, and one
    # kernel for the output when eager output is contiguous or ACL with bf16/half. Returns 0 to 2.
    # this function is to decide how many kernels are generated
    # while testing conv2d/3d/deconv2d
    # the assumption is:
    #   (1) There will be a to_dtype kernel for input for lp
    #   (2) inductor always use channel_last format, there will
    #       be a to_channel_last format for input
    #   (3) to_dtype and to_channel_last for input can be fused
    #   (4) inductor always get channel last format from mkldnn_conv_pointwise(binary),
    #       and force the output to have same stride with eager.
    #       So there will be a to_contiguous for output if eager output is contiguouse
    mod = copy.deepcopy(mod)
    mod = mod.to(device=device)
    input = input.clone()
    input = input.to(device)

    if dtype == torch.float32:
        maybe_autocast = contextlib.nullcontext()
    else:
        maybe_autocast = torch.amp.autocast(device_type=device, dtype=dtype)
    with torch.no_grad(), maybe_autocast:
        output = mod(input)
    input_kernel, output_kernel = 0, 0
    if (
        input.is_contiguous(memory_format=torch.contiguous_format)
        or dtype != torch.float32
        or (TEST_ACL and dim == 4)
    ):
        input_kernel = 1
    if output.is_contiguous(memory_format=torch.contiguous_format) or (
        TEST_ACL and (dtype == torch.bfloat16 or dtype == torch.half)
    ):
        output_kernel = 1

    return input_kernel + output_kernel


class TestPatternMatcherBase(TestCase):
    # [CLASS DOC] Base class for pattern matcher tests. Enables the freezing config for each test
    # through an exit stack stored on the instance, and provides helpers to clone inputs, compile
    # a module and compare against eager while running a matcher counter check, and to inspect the
    # generated source code for expected operators.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setup, creates an exit stack and enters a config
        # patch turning freezing on for the duration of the test.
        super().setUp()
        self.ctx_stack = contextlib.ExitStack()
        self.ctx_stack.enter_context(config.patch({"freezing": True}))

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the base teardown, then closes the exit stack, restoring the
        # config patched in setUp.
        TestCase.tearDown(self)
        self.ctx_stack.close()

    def _check_unary_is_decomposed(self, unary_fn):
        # [FUNCTION DOC] HELPER: Returns True if the given unary module is not one of
        # ReLU, Sigmoid or Tanh, i.e. if it is decomposed by inductor into several
        # nodes; False otherwise.
        return not any(
            isinstance(unary_fn, fn)
            for fn in [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]
        )

    def _clone_inputs(self, inputs):
        # [FUNCTION DOC] HELPER: Returns a tuple with a clone of each tensor in the inputs; non-
        # tensor inputs are passed through unchanged.
        def clone(x):
            # [FUNCTION DOC] HELPER: Returns x cloned if it is a tensor, otherwise x itself.
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_check_fn,
        atol=1e-5,
        rtol=1.3e-6,
        check_autocast=torch.float32,
        dtype=None,
        compile_options={},  # noqa: B006
    ):
        # [FUNCTION DOC] HELPER: Core check used by most tests. Picks the device (the test device
        # if set, else xpu if any input is an xpu tensor, else cpu), moves the module to it and
        # clones inputs for non-cpu devices preserving strides. Clears counters and resets
        # dynamo, builds an autocast context for bfloat16 or float16 when supported (then
        # tolerances become 5e-2) or a null context for float32, and raises AssertionError for an
        # unexpected autocast value. In no-grad mode runs eager and compiled (with
        # compile_options) on cloned inputs, compares with the test precision if nonzero, else
        # atol and rtol, then calls matcher_check_fn.
        if not hasattr(self, "device"):
            has_xpu = any(
                isinstance(input, torch.Tensor) and input.device.type == "xpu"
                for input in inputs
            )
            device = "xpu" if has_xpu else "cpu"
        else:
            device = self.device

        mod = mod.to(device=device)
        if device != "cpu":
            inputs = tuple(
                clone_preserve_strides_offset(x, device=device) for x in inputs
            )
        counters.clear()
        torch._dynamo.reset()
        if check_autocast == torch.bfloat16 and is_mkldnn_bf16_supported(device):
            maybe_autocast = torch.amp.autocast(
                device_type=device, dtype=torch.bfloat16
            )
            atol, rtol = 5e-2, 5e-2
        elif check_autocast == torch.float16 and (is_mkldnn_fp16_supported(device)):
            maybe_autocast = torch.amp.autocast(device_type=device, dtype=torch.float16)
            atol, rtol = 5e-2, 5e-2
        else:
            if check_autocast != torch.float32:
                raise AssertionError(
                    f"Expected check_autocast to be torch.float32, got {check_autocast}"
                )
            maybe_autocast = contextlib.nullcontext()
        with torch.no_grad(), maybe_autocast:
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual = torch.compile(mod, **compile_options)(*clone_inputs)
            if self.precision != 0:
                torch.testing.assert_close(
                    actual, expected, atol=self.precision, rtol=self.precision
                )
            else:
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            matcher_check_fn()

    def _test_code_common(
        self,
        mod,
        inputs,
        include_ops,
        exclude_ops,
        atol=1e-5,
        rtol=1.3e-6,
        check_dynamic=None,
        num_include_ops=None,
    ):
        # [FUNCTION DOC] HELPER: Compiles the module with fullgraph (and optional dynamic setting)
        # in no-grad mode, fetches the generated source, and removes lines mentioning
        # size/alignment asserts or tensor metadata asserts. Asserts every include_ops string
        # appears, that each count equals num_include_ops when given (raises AssertionError if the
        # list lengths differ), that no exclude_ops string appears, checks dynamic shapes appear
        # in code if check_dynamic is set, and compares the result with eager using atol and rtol.
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual, (source_code,) = run_and_get_code(
                torch.compile(mod, fullgraph=True, dynamic=check_dynamic),
                *clone_inputs,
            )
            assert_keywords = ["assert_size_stride", "assert_alignment"]
            filtered_lines = [
                line
                for line in source_code.splitlines()
                if not any(assert_key in line for assert_key in assert_keywords)
                and not line.strip().startswith("assert_tensor_metadata(")
            ]
            source_code = "\n".join(filtered_lines)

            for op in include_ops:
                self.assertIn(op, source_code)
            if num_include_ops is not None:
                if len(include_ops) != len(num_include_ops):
                    raise AssertionError(
                        f"len(include_ops)={len(include_ops)} != len(num_include_ops)={len(num_include_ops)}"
                    )
                for i in range(len(include_ops)):
                    self.assertEqual(
                        source_code.count(include_ops[i]), num_include_ops[i]
                    )
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestPatternMatcherGeneric(TestPatternMatcherBase):
    # [CLASS DOC] Device-generic pattern matcher tests for conv, conv transpose, linear, woq and
    # related fusions, instantiated for each device type. Tests set the device attribute from the
    # device argument before calling shared base methods.
    def _test_conv_unary_base(self, dim=4):
        # [FUNCTION DOC] HELPER: Conv followed by unary op test driver for 2D (dim 4) or 3D (dim
        # 5). Raises AssertionError for other dims. For every unary module in the unary table,
        # memory format (contiguous or channels last) and available dtype (float, plus bf16 and
        # fp16 if the device supports them; reduced precision tf32 skips non-float dtypes), resets
        # metrics, compiles a conv plus unary model and checks the unary fusion node count (zero
        # on ACL, plus 2 for decomposed unary ops under reduced precision autocast), one conv
        # weight pack, numerics, and the expected generated kernel count.
        if dim != 4 and dim != 5:
            raise AssertionError(f"Expected dim to be 4 or 5, got {dim}")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a conv layer (2D or 3D depending on the enclosing
            # dim) followed by the unary function module given at construction.
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Builds a 3 to 16 channel 3x3 conv (2D or 3D per enclosing
                # dim) and stores the unary function; extra keyword arguments are ignored.
                super().__init__()
                if dim == 4:
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.unary_fn = unary_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv and then the unary function to x
                # and returns it.
                x = self.conv(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list.keys(),
            [torch.contiguous_format, cl_format],
            dtypes,
        )

        for (
            unary_fn,
            memory_format,
            dtype,
        ) in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            mod = M(unary_fn).to(memory_format=memory_format).eval()

            v = (
                torch.randn(x_shape, dtype=torch.float32)
                .add(1)
                .to(memory_format=memory_format)
            )

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts the unary fusion matcher
                # node count equals the expected value from the unary table (adding 2 for
                # decomposed ops under fp16/bf16; zero on ACL) and that exactly one conv weight
                # pack match happened.
                match_nodes = unary_list[unary_fn]
                if dtype in (
                    torch.float16,
                    torch.bfloat16,
                ) and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off()
    def test_conv2d_unary(self, device):
        # [FUNCTION DOC] Sets the device and runs the conv unary driver for 2D conv. Skipped
        # without dynamo support or oneDNN; runs with reduced fp32 precision on and off.
        self.device = device
        self._test_conv_unary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off()
    def test_conv3d_unary(self, device):
        # [FUNCTION DOC] Sets the device and runs the conv unary driver for 3D conv. Skipped
        # without dynamo support or oneDNN; runs with reduced fp32 precision on and off.
        self.device = device
        self._test_conv_unary_base(dim=5)

    def _test_conv_transpose_unary_base(self, dim=4):
        # [FUNCTION DOC] HELPER: Conv transpose followed by unary op driver for 2D (dim 4) or 3D
        # (dim 5); raises AssertionError for other dims. Iterates over all unary modules, memory
        # formats and supported dtypes, builds the model, compiles it and checks unary fusion node
        # counts, one conv weight pack, numerics and the expected generated kernel count. Unlike
        # the conv driver it does not skip non-float dtypes under tf32.
        if dim != 4 and dim != 5:
            raise AssertionError(f"Expected dim to be 4 or 5, got {dim}")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a transposed conv layer (2D or 3D per
            # enclosing dim; 3 to 16 channels, kernel 3, stride 2, padding 1) followed by
            # the given unary function.
            def __init__(
                self,
                unary_fn,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Builds the transposed conv layer and stores the unary
                # function; extra keyword arguments are ignored.
                super().__init__()
                if dim == 4:
                    self.conv_transpose = torch.nn.ConvTranspose2d(
                        3, 16, 3, stride=2, padding=1
                    )
                else:
                    self.conv_transpose = torch.nn.ConvTranspose3d(
                        3, 16, 3, stride=2, padding=1
                    )
                self.unary_fn = unary_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the transposed conv then the unary function to x.
                x = self.conv_transpose(x)
                return self.unary_fn(x)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        options = itertools.product(
            unary_list,
            [torch.contiguous_format, cl_format],
            dtypes,
        )

        for unary_fn, memory_format, dtype in options:
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 28, 28)
            else:
                x_shape = (1, 3, 17, 28, 28)
            mod = M(unary_fn).eval()

            v = torch.randn(x_shape, dtype=torch.float32).to(
                memory_format=memory_format
            )

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts the unary fusion matcher
                # node count (adjusted as in the conv test; zero on ACL) and exactly one conv
                # weight pack match.
                match_nodes = unary_list[unary_fn]
                if dtype in (
                    torch.float16,
                    torch.bfloat16,
                ) and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfXpu(
        msg="The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
    )
    @reduced_f32_on_and_off()
    def test_conv_transpose2d_unary(self, device):
        # [FUNCTION DOC] Sets the device and runs the transposed conv unary driver for 2D. Skipped
        # without dynamo or oneDNN, and on XPU where the operator is unimplemented.
        self.device = device
        self._test_conv_transpose_unary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfXpu(
        msg="The operator 'mkldnn::_convolution_transpose_pointwise' is not currently implemented for the XPU device."
    )
    @reduced_f32_on_and_off()
    def test_conv_transpose3d_unary(self, device):
        # [FUNCTION DOC] Sets the device and runs the transposed conv unary driver for 3D. Skipped
        # without dynamo or oneDNN, and on XPU where the operator is unimplemented.
        self.device = device
        self._test_conv_transpose_unary_base(dim=5)

    def _test_conv_binary_base(self, dim=4):
        # [FUNCTION DOC] HELPER: Conv binary op driver for 2D or 3D; raises AssertionError
        # for other dims. Builds a model with two convs on the same input combined by a
        # binary function, with an optional trailing relu. Iterates over binary functions,
        # relu flag, memory formats and dtypes, checks the conv binary unary fusion node
        # count (plus 1 with relu; zero on ACL), two conv weight packs, numerics, and
        # expected generated kernel count.
        if dim != 4 and dim != 5:
            raise AssertionError(f"Expected dim to be 4 or 5, got {dim}")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two parallel convs on the input whose outputs are
            # combined by the binary function, optionally followed by relu.
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Builds two 3 to 16 channel 3x3 convs (2D or 3D per
                # enclosing dim) and stores the binary function and relu flag; extra keyword
                # arguments are ignored.
                super().__init__()
                if dim == 4:
                    self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv1 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                    self.conv2 = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both convs to x, combines them with the binary
                # function, and applies relu if the enclosing has_relu is set (the closure value,
                # not the attribute).
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        test_memory_format = [torch.contiguous_format, cl_format]
        options = itertools.product(
            binary_list,
            [True, False],
            test_memory_format,
            dtypes,
        )

        for (
            binary_fn,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            if dim == 4:
                x_shape = (1, 3, 56, 56)
            else:
                x_shape = (1, 3, 20, 56, 56)
            mod = M(binary_fn, has_relu).eval()
            v = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts the conv binary unary
                # fusion node count (table value plus 1 with relu; zero on ACL) and that two conv
                # weight packs matched.
                match_nodes = binary_list[binary_fn][1]
                if has_relu:
                    match_nodes += 1
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 2
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            generated_kernel_count = cal_conv_generated_kernel_number(
                mod, v, dtype, dim, self.device
            )
            self.assertEqual(metrics.generated_kernel_count, generated_kernel_count)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off(0.02)
    def test_conv2d_binary(self, device):
        # [FUNCTION DOC] Sets the device and runs the conv binary driver for 2D with reduced fp32
        # precision tolerance 0.02. Skipped without dynamo support or oneDNN.
        self.device = device
        self._test_conv_binary_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off(0.02)
    def test_conv3d_binary(self, device):
        # [FUNCTION DOC] Sets the device and runs the conv binary driver for 3D with reduced fp32
        # precision tolerance 0.02. Skipped without dynamo support or oneDNN.
        self.device = device
        self._test_conv_binary_base(dim=5)

    def _test_conv_binary_broadcast_shapes_base(self, dim=4):
        # [FUNCTION DOC] HELPER: Conv binary driver where the second operand is a tensor of
        # broadcast shape (singleton spatial dims, or channels only). Raises AssertionError
        # for dims other than 4 and 5, seeds the RNG, iterates over binary functions, shapes,
        # relu flag, memory formats and dtypes, and compiles a model with one conv and an
        # external tensor. Checks fusion node count and weight pack count and numerics. Does
        # not check kernel counts.
        if dim != 4 and dim != 5:
            raise AssertionError(f"Expected dim to be 4 or 5, got {dim}")
        torch.manual_seed(12345)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying a conv to the first input and combining
            # the result with a second input tensor via the binary function, optionally
            # followed by relu.
            def __init__(
                self,
                binary_fn,
                has_relu,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Builds one 3 to 16 channel 3x3 conv (2D or 3D per
                # enclosing dim) and stores the binary function and relu flag; extra keyword
                # arguments are ignored.
                super().__init__()
                if dim == 4:
                    self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                else:
                    self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x, x2):
                # [FUNCTION DOC] HELPER: Applies the conv to x and combines with x2 by the binary
                # function, with relu afterward if the enclosing has_relu is set.
                x1 = self.conv(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)

        dtypes = [
            torch.float,
        ]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        cl_format = torch.channels_last if dim == 4 else torch.channels_last_3d
        test_memory_format = [torch.contiguous_format, cl_format]
        if dim == 4:
            input_shapes = [
                [2, 3, 56, 56],
            ]
            other_shapes = [[2, 16, 1, 1], [1, 16, 1, 1], [1, 1, 1, 1]]
        else:
            input_shapes = [
                [2, 3, 20, 56, 56],
            ]
            other_shapes = [[2, 16, 1, 1, 1], [1, 16, 1, 1, 1], [1, 1, 1, 1, 1]]
        options = itertools.product(
            binary_list,
            input_shapes,
            other_shapes,
            [True, False],
            test_memory_format,
            dtypes,
        )

        for (
            binary_fn,
            x_shape,
            other_shape,
            has_relu,
            memory_format,
            dtype,
        ) in options:
            metrics.reset()
            mod = M(binary_fn, has_relu).eval()
            x = (
                torch.randn(x_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
            )
            other = (
                torch.randn(other_shape, dtype=torch.float32, requires_grad=True)
                .add(1)
                .to(memory_format=memory_format)
                .to(dtype)
            )

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts the conv binary unary
                # fusion node count (table value plus 1 with relu; zero on ACL) and the conv
                # weight pack matcher node counter equals 1.
                match_nodes = binary_list[binary_fn][1]
                if has_relu:
                    match_nodes += 1
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"], 1
                )

            self._test_common(mod, (x, other), matcher_check_fn, check_autocast=dtype)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off()
    def test_conv2d_binary_broadcast_shapes(self, device):
        # [FUNCTION DOC] Sets the device and runs the broadcast-shape conv binary driver for 2D.
        # Skipped without dynamo support or oneDNN.
        self.device = device
        self._test_conv_binary_broadcast_shapes_base(dim=4)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @reduced_f32_on_and_off(bf32_precision=5e-2)
    def test_conv3d_binary_broadcast_shapes(self, device):
        # [FUNCTION DOC] Sets the device and runs the broadcast-shape conv binary driver for 3D
        # with bf32 tolerance 5e-2. Skipped without dynamo support or oneDNN.
        self.device = device
        self._test_conv_binary_broadcast_shapes_base(dim=5)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @unittest.skipIf(IS_FBCODE, "Failing in fbcode")
    @reduced_f32_on_and_off()
    def test_conv2d_linear_add_broadcast_shapes(self, device):
        # [FUNCTION DOC] Compiles a model adding a conv output to a linear output expanded with
        # two trailing singleton dims, and checks (via the shared compile check) that the conv
        # binary fusion nodes (zero on ACL, else 2) and one conv weight pack node were counted.
        # Skipped in fbcode.
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model summing a conv of the first input and a linear of the
            # second input with singleton dims appended so the two broadcast.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a 3 to 16 channel 3x3 conv and a linear from 3
                # to 16 features.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.linear = torch.nn.Linear(3, 16)

            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Returns the conv of x1 plus the linear of x2 reshaped by
                # appending two singleton dims.
                return self.conv(x1) + self.linear(x2)[:, :, None, None]

        metrics.reset()
        mod = M().eval()
        x1 = torch.randn(2, 3, 56, 56)
        x2 = torch.randn(2, 3)

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check closure: asserts conv binary unary fusion node
            # count (zero on ACL, else 2) and the conv weight pack node count of 1.
            match_nodes = 0 if TEST_ACL else 2
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_nodes"],
                match_nodes,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"], 1
            )

        self._test_common(mod, (x1, x2), matcher_check_fn)

    @skipIfNoDynamoSupport
    def test_woq_int8(self, device):
        # [FUNCTION DOC] Weight-only int8 quantization pattern test on the given device. For two
        # input stride layouts (linear lowering to mm or to bmm) and with or without weight
        # permutation, builds a model that dequantizes an int8 weight by casting and scales the
        # result, runs it with bfloat16 input, int8 random weights and bfloat16 scales through the
        # shared compile check with atol 0.001 and rtol 0.07, and asserts exactly one woq matcher
        # match. Needs dynamo support.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model computing a scaled matmul with an int8 weight cast to the
            # input dtype, either via a transposed weight and matrix multiply with reshapes, or
            # via a linear call, chosen by the is_permute flag.
            def __init__(self, is_permute):
                # [FUNCTION DOC] HELPER: Stores the is_permute flag.
                super().__init__()
                self.is_permute = is_permute

            def forward(self, x, weight, scales):
                # [FUNCTION DOC] HELPER: Returns the weight-only-quantized matmul of x with
                # weight, times scales. With is_permute it transposes the weight, flattens x to
                # 2D, matmuls, scales, and restores the leading dims; otherwise uses the linear
                # function and scales directly.
                if self.is_permute:
                    weight = weight.t()
                    m = torch.mm(
                        x.reshape(-1, x.shape[-1]),
                        weight.to(x.dtype),
                    )
                    y = m * scales.to(m.dtype)
                    y = y.reshape(*x.shape[:-1], y.shape[-1])
                    return y
                else:
                    return (
                        torch.nn.functional.linear(x, weight.to(dtype=x.dtype)) * scales
                    )

        x_shape = (1, 1, 256)
        s_shape = 12
        x_strides = [
            (256, 256, 1),  # linear dispatching to mm
            (256, 32, 1),  # linear dispatching to bmm
        ]
        is_permutes = [False, True]
        for x_stride, is_permute in itertools.product(x_strides, is_permutes):
            mod = M(is_permute=is_permute).eval()
            x = (
                torch.randn(x_shape, dtype=torch.bfloat16)
                .as_strided(x_shape, x_stride)
                .to(device)
            )

            w_shape = (12, 256)
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8).to(device)
            s = torch.randn(s_shape, dtype=torch.bfloat16).to(device)

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure asserting exactly one
                # woq matcher match.
                self.assertEqual(counters["inductor"]["woq_matcher_count"], 1)

            self._test_common(
                mod,
                (x, w, s),
                matcher_check_fn,
                atol=0.001,
                rtol=0.07,
            )


class TestPatternMatcher(TestPatternMatcherBase):
    # [CLASS DOC] CPU pattern matcher tests: linear fusions (unary, binary, bias add), shared-
    # input linears, conv fallbacks and in-place binary fusion, and further CPU-only patterns.
    # Tests take a device argument defaulting to cpu and store it on the instance.
    # Note: tests containing the pattern *qconv2d* were removed in PR #169151 (PT2E migration to torchao).
    # See issue #168635 and its sub-issues.
    @reduced_f32_on_and_off()
    def test_linear_unary(self, device="cpu"):
        # [FUNCTION DOC] Linear followed by each unary op, with and without bias, for each lowp
        # dtype supported (bf16, fp16) plus float32 if reduced fp32 precision is bf16 or tf32
        # (tf32 skips non-float dtypes). Checks unary fusion node counts (zero on ACL; plus 2 for
        # decomposed ops under lowp), one linear weight pack, numerics, and exactly one generated
        # kernel (two on ACL) for the dtype cast, minus one when float32 input needs no cast.
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a linear layer (configurable features and bias)
            # followed by the given unary function.
            def __init__(
                self,
                unary_fn,
                in_features,
                out_features,
                bias,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Creates the linear layer, forwarding extra keyword
                # arguments to it, and stores the unary function.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features,
                    out_features,
                    bias,
                    **kwargs,
                )
                self.unary_fn = unary_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer then the unary function.
                x = self.linear(x)
                return self.unary_fn(x)

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        if torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]:
            dtypes.append(torch.float32)
        options = itertools.product(unary_list, [True, False], dtypes)
        for unary_fn, bias, dtype in options:
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue
            metrics.reset()
            mod = M(unary_fn, 10, 30, bias=bias).eval()
            # only fuse for linear when the dtype is bf16
            v = torch.randn(2, 10)

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts the unary fusion node
                # count (table value, plus 2 for decomposed ops in lowp; zero on ACL) and one
                # linear weight pack match.
                match_nodes = unary_list[unary_fn]
                if dtype != torch.float32 and self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            # only generated 1 kernel for "to_dtype"
            expected_kernel_count = 2 if TEST_ACL else 1
            if dtype == torch.float32:
                # In BF32, input is float32, will not generate kernel for "to_dtype"
                expected_kernel_count -= 1
            self.assertEqual(metrics.generated_kernel_count, expected_kernel_count)

    @reduced_f32_on_and_off()
    @requires_mkl
    def test_linear_fp32(self, device="cpu"):
        # [FUNCTION DOC] Float32 linear with and without bias on the given device (requires MKL):
        # compiles and checks numerics and that exactly one linear weight pack match was counted.
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a single 10 to 30 linear layer.
            def __init__(self, bias):
                # [FUNCTION DOC] HELPER: Creates the linear layer with or without bias.
                super().__init__()
                self.linear = torch.nn.Linear(10, 30, bias)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to x.
                return self.linear(x)

        for bias in [True, False]:
            mod = M(bias=bias).eval()
            v = torch.randn(2, 10)

            # packing pass.
            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure asserting one linear
                # weight pack match.
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            self._test_common(mod, (v,), matcher_check_fn)

    @requires_mkl
    def test_linear_input_non_contiguous_3D_wo_bias(self, device="cpu"):
        # [FUNCTION DOC] For float, and bf16/fp16 when supported, compiles a bias-free linear on a
        # permuted and reshaped non-contiguous 3D activation under autocast (for lowp dtypes) and
        # asserts the generated code calls the mkldnn linear pointwise op with autocast, or the
        # MKL linear op otherwise, and that results match eager within 1e-2. Requires MKL.
        self.device = device

        # Activation is 3D, non-contiguous and without Bias
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model permuting and reshaping a 4D input into a non-contiguous
            # 3D activation before a 4096 to 1024 bias-free linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the bias-free 4096 to 1024 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4096, 1024, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Permutes dims 1 and 2 of x, reshapes to 4 by 1 by 4096,
                # and applies the linear layer.
                x = torch.ops.aten.permute.default(x, [0, 2, 1, 3])
                x = torch.ops.aten.reshape.default(x, [4, 1, 4096])
                return self.linear(x)

        mod = M().eval()
        v = torch.randn(4, 32, 1, 128)

        dtypes = [torch.float]
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        for dtype in dtypes:
            torch._dynamo.reset()
            autocast_enabled = dtype in [torch.bfloat16, torch.float16]
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cpu",
                    enabled=autocast_enabled,
                    dtype=dtype,
                ),
            ):
                expected = mod(v)
                actual, (source_code,) = run_and_get_code(
                    torch.compile(mod, fullgraph=True),
                    v,
                )
                self.assertIn(
                    "torch.ops.mkldnn._linear_pointwise.default"
                    if autocast_enabled
                    else "torch.ops.mkl._mkl_linear.default",
                    source_code,
                )
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @skipIfXpu(
        msg="Different with CPU, two linears will be concat on XPU for better performance"
    )
    def test_linear_add_bias(self, device="cpu"):
        # [FUNCTION DOC] For each unary op and lowp dtype, compiles two bias-free linears sharing
        # an input, each added to a bias tensor then passed through the unary op. With biases cast
        # to the weight dtype, expects the bias folded: unary fusion count 2 (zero on ACL),
        # doubled node count, two weight packs and 1 generated kernel (3 on ACL). With biases left
        # float32, only checks two weight packs and 3 generated kernels. Skipped on XPU.
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two bias-free linears over the same input whose
            # outputs get separate bias tensors (stored as plain attributes) added, then a unary
            # function applied to both; returns both results.
            def __init__(self, device, dtype, unary_fn, cast_bias):
                # [FUNCTION DOC] HELPER: Creates two linears and two random bias tensors on the
                # device; when cast_bias is set, casts both biases to the given dtype. Stores
                # the unary function.
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 64, bias=False)
                self.bias1 = torch.randn(64, device=device)
                self.linear2 = torch.nn.Linear(10, 64, bias=False)
                self.bias2 = torch.randn(64, device=device)
                if cast_bias:
                    self.bias1 = self.bias1.to(dtype=dtype, device=device)
                    self.bias2 = self.bias2.to(dtype=dtype, device=device)
                self.unary_fn = unary_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the unary function applied to each linear output
                # plus its bias.
                a = self.linear1(x) + self.bias1
                b = self.linear2(x) + self.bias2
                return self.unary_fn(a), self.unary_fn(b)

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        options = itertools.product(unary_list, dtypes)
        for unary_fn, dtype in options:
            metrics.reset()
            fold_mod = M(self.device, dtype, unary_fn, cast_bias=True).eval()
            v = torch.randn(2, 10)

            def folder_matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure for the folded-bias case: asserts
                # two unary fusion matches, node count doubled from the table (plus 2 per
                # decomposed op), and two linear weight packs. Zero fusion counts on ACL.
                match_nodes = unary_list[unary_fn]
                if self._check_unary_is_decomposed(unary_fn):
                    # Has extra dtype conversion nodes for autocast.
                    match_nodes += 2
                # we have 2 linears, so we double the matcher_count/nodes
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                    0 if TEST_ACL else match_nodes * 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
                )

            self._test_common(
                fold_mod,
                (v,),
                folder_matcher_check_fn,
                check_autocast=dtype,
            )
            self.assertEqual(metrics.generated_kernel_count, 3 if TEST_ACL else 1)
            # we won't fold the bias if bias is not same dtype with weight
            # https://github.com/pytorch/pytorch/pull/129138
            metrics.reset()
            mod = M(self.device, dtype, unary_fn, cast_bias=False).eval()

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure asserting two linear
                # weight pack matches.
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)
            # 1 kernel for "to_lowp", 2 kernels for unary ops
            self.assertEqual(metrics.generated_kernel_count, 3)

    @skipIfXpu
    def test_linear_add_bias_float_constant(self, device="cpu"):
        # [FUNCTION DOC] Regression test (PR 183514) that the linear bias add analysis does not
        # crash when the added value is a Python float: compiles a model adding a float constant
        # to a SiLU plus linear output for each lowp dtype, and checks one linear weight pack
        # match and zero linear bias folds. Skipped on XPU.
        # Regression test: is_linear_add_bias should not crash when
        # add_node.args[1] is a Python float (e.g., from `1.0 + linear_output`).
        # See https://github.com/pytorch/pytorch/pull/183514
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning a float constant plus the output of a SiLU then
            # linear sequential block.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a sequential SiLU and 64 to 64 linear block.
                super().__init__()
                self.adaLN_modulation = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(64, 64, bias=True),
                )

            def forward(self, c):
                # [FUNCTION DOC] HELPER: Returns one plus the block output for c.
                return 1.0 + self.adaLN_modulation(c)

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        for dtype in dtypes:
            mod = M().eval()
            v = torch.randn(2, 64)

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: one linear weight pack match and
                # zero bias matcher matches (the float must not be treated as a bias).
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )
                # The float constant should NOT be folded as bias
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_bias_matcher_count"], 0
                )

            self._test_common(mod, (v,), matcher_check_fn, check_autocast=dtype)

    @reduced_f32_on_and_off()
    def test_linear_binary(self, device="cpu"):
        # [FUNCTION DOC] Linear followed by a binary op with a cloned second tensor, for 2D and 3D
        # inputs, with and without bias, over lowp dtypes (plus float32 under reduced precision;
        # tf32 skips non-float). Checks the binary fusion node count (zero on ACL, else 2),
        # reshape linear reshape node count (3 for 3D input, else 0), one weight pack, numerics,
        # and one generated kernel (two on ACL, one fewer for float32).
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying a linear layer to the first input and combining
            # the result with a clone of the second input through the binary function.
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the linear layer (forwarding extra keyword
                # arguments) and stores the binary function.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary_fn = binary_fn

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x, then the binary function
                # with a clone of y, and returns the result.
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        if torch.backends.mkldnn.matmul.fp32_precision in ["bf16", "tf32"]:
            dtypes.append(torch.float32)
        options = itertools.product(
            binary_list, [[2, 3, 10], [2, 10]], [True, False], dtypes
        )
        out_feature = 30

        for binary_fn, input_shape, bias, dtype in options:
            metrics.reset()
            if (
                dtype != torch.float32
                and torch.backends.mkldnn.matmul.fp32_precision == "tf32"
            ):
                continue

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts binary fusion nodes (zero
                # on ACL, else 2), reshape linear reshape nodes (3 for 3D input shape, else 0),
                # and one weight pack match.
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else 2,
                )
                reshape_linear_reshape_match_nodes = 3 if len(input_shape) == 3 else 0
                self.assertEqual(
                    counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"],
                    reshape_linear_reshape_match_nodes,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 1
                )

            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            v = torch.randn(input_shape)
            other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                matcher_check_fn,
                check_autocast=dtype,
            )
            # only generated 1 kernel for "to_dtype"
            expected_kernel_count = 2 if TEST_ACL else 1
            if dtype == torch.float32:
                # In BF32, input is float32, will not generate kernel for "to_dtype"
                expected_kernel_count -= 1
            self.assertEqual(metrics.generated_kernel_count, expected_kernel_count)

    def test_linear_binary_broadcast_shapes(self, device="cpu"):
        # [FUNCTION DOC] Same as the linear binary test but the second operand has a broadcast
        # shape (singleton leading dims) and only lowp dtypes are used. Checks reshape linear
        # reshape node count, binary fusion nodes, one linear weight pack node, numerics, and one
        # generated kernel (two on ACL).
        self.device = device

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying a linear layer to the first input and combining
            # the result with a clone of the second input through the binary function.
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                # [FUNCTION DOC] HELPER: Creates the linear layer (forwarding extra keyword
                # arguments) and stores the binary function.
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_channels, out_channels, bias=bias, **kwargs
                )
                self.binary_fn = binary_fn

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the linear layer to x, then the binary function
                # with a clone of y, and returns the result.
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)
        options = itertools.product(
            binary_list,
            (
                ([2, 3, 10], [1, 1, 30]),
                ([2, 10], [1, 30]),
            ),
            (True, False),
            dtypes,
        )
        out_feature = 30

        for binary_fn, (input_shape, other_shape), bias, dtype in options:
            metrics.reset()
            mod = M(binary_fn, input_shape[-1], out_feature, bias).eval()
            v = torch.randn(input_shape)
            other = torch.randn(other_shape).to(dtype)

            def matcher_check_fn():
                # [FUNCTION DOC] HELPER: Matcher check closure: asserts reshape linear reshape
                # node count (3 for 3D input, else 0), binary fusion nodes (zero on ACL, else 2)
                # and one linear weight pack node.
                reshape_linear_reshape_match_nodes = 3 if len(input_shape) == 3 else 0
                self.assertEqual(
                    counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"],
                    reshape_linear_reshape_match_nodes,
                )
                self.assertEqual(
                    counters["inductor"][
                        "mkldnn_conv_binary_unary_fusion_matcher_nodes"
                    ],
                    0 if TEST_ACL else 2,
                )
                self.assertEqual(
                    counters["inductor"]["mkldnn_linear_weight_pack_matcher_nodes"], 1
                )

            self._test_common(
                mod,
                (
                    v,
                    other,
                ),
                matcher_check_fn,
                check_autocast=dtype,
            )
            self.assertEqual(metrics.generated_kernel_count, 2 if TEST_ACL else 1)

    @skipIfXpu(
        msg="Different with CPU, two linears will be concat on XPU for better performance"
    )
    def test_multi_linear_share_same_input(self, device="cpu"):
        # [FUNCTION DOC] Llama-style pattern with two linears on one input, one through SiLU and
        # one through ReLU, multiplied. For each lowp dtype, expects 9 unary fusion nodes in 2
        # matches (zero on ACL), 6 reshape linear reshape nodes and 2 weight packs, with tolerance
        # 1e-2. Skipped on XPU.
        self.device = device

        # llama pattern.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model multiplying a SiLU of one bias-free 16 by 16 linear and a
            # ReLU of another linear applied to the same input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the two bias-free 16 to 16 linear layers.
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of silu of the first linear and relu
                # of the second linear, both applied to x.
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check closure: asserts the unary fusion node count
            # 9 and match count 2 (both zero on ACL), reshape linear reshape node count 6 and
            # two weight packs.
            # SiLU: 5 base nodes + 2 dtype conversion = 7
            # ReLU: 2 base nodes (non-decomposed, no dtype conversion)
            # Total: 7 + 2 = 9
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                0 if TEST_ACL else 9,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"], 6
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
            )

        for dtype in dtypes:
            mod = M().to(dtype).eval()
            v = torch.randn(2, 4, 16).to(dtype)
            self._test_common(mod, (v,), matcher_check_fn, rtol=1e-2, atol=1e-2)

    # https://github.com/pytorch/pytorch/issues/99841.
    def test_hardtanh_pattern_fallback(self):
        # [FUNCTION DOC] Regression test for issue 99841: a transposed conv followed by clamp
        # min and clamp max with scalar or tensor bounds where min exceeds max, checking the
        # hardtanh pattern falls back correctly (3 unary fusion nodes, zero on ACL, one conv
        # weight pack) and numerics.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying a transposed conv then a lower clamp then an
            # upper clamp with the bounds passed as inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 transposed conv
                # with padding 1.
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, min_value, max_value):
                # [FUNCTION DOC] HELPER: Returns the transposed conv of x clamped from below by
                # min_value then from above by max_value.
                conv_transpose_output = self.conv_transpose(x)
                clamp_min_output = torch.clamp_min(conv_transpose_output, min_value)
                clamp_max_output = torch.clamp_max(clamp_min_output, max_value)
                return clamp_max_output

        # check works for min_value > max_value.
        min_values = [3, torch.randn(1, 32, 28, 28)]
        max_values = [0, torch.randn(1, 32, 28, 28)]
        v = torch.randn(1, 3, 28, 28)

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check closure: 3 unary fusion nodes (zero on ACL) and
            # one conv weight pack match.
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                0 if TEST_ACL else 3,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
            )

        for min_value, max_value in zip(min_values, max_values):
            mod = Model().eval()
            self._test_common(mod, (v, min_value, max_value), matcher_check_fn)

    def test_leaky_relu_pattern_fallback(self):
        # [FUNCTION DOC] Checks the leaky relu pattern falls back correctly when the slope is a
        # float or a tensor: compiles a conv followed by a where-based leaky relu under no-grad
        # mode and checks 4 unary fusion nodes (zero on ACL), one conv weight pack, and numerics.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying a conv then a where-based leaky relu with the
            # negative slope supplied as an input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, negative_slope):
                # [FUNCTION DOC] HELPER: Returns the conv output where positive, else the conv
                # output times the negative slope.
                conv_out = self.conv(x)
                return torch.where(conv_out > 0, conv_out, conv_out * negative_slope)

        negative_slopes = [0.1, torch.randn(1, 32, 28, 28)]

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check closure: 4 unary fusion nodes (zero on ACL) and
            # one conv weight pack match.
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                0 if TEST_ACL else 4,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
            )

        with torch.no_grad():
            v = torch.randn(1, 3, 28, 28)
            for negative_slope in negative_slopes:
                mod = Model().eval()
                self._test_common(mod, (v, negative_slope), matcher_check_fn)

    # https://github.com/pytorch/pytorch/issues/99838.
    def test_conv2d_add_scalar(self):
        # [FUNCTION DOC] Regression test for issue 99838: a conv plus a scalar constant is folded
        # (binary folding counter 1) while the conv weight is packed once; numerics checked.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning a conv output plus the constant 1.0.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the conv of x plus 1.0.
                out_conv = self.conv(x)
                out = torch.add(out_conv, 1.0)
                return out

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check closure: binary folding counter equals 1 and
            # one conv weight pack match.
            self.assertEqual(counters["inductor"]["binary_folding"], 1)
            self.assertEqual(
                counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"], 1
            )

        with torch.no_grad():
            mod = Model().eval()
            v = torch.randn(1, 3, 28, 28)
            self._test_common(mod, (v,), matcher_check_fn)

    @xfailIfACL
    def test_conv2d_binary_inplace_fusion_pass_cpu(
        self, include_ops=None, exclude_ops=None
    ):
        # [FUNCTION DOC] Checks that conv plus add fuses into the in-place mkldnn conv binary op
        # on CPU when the written buffer is an intermediate: one model adds a relu of an extra
        # input to a conv output, the other adds a pow result to the output of two chained convs.
        # Both use channels last tensors. By default the generated code must include the in-place
        # binary op name and exclude the out-of-place one (the op name lists can be overridden
        # through arguments). Expected to fail on ACL.
        class Model_v1(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding the conv output to a relu of the second input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                # [FUNCTION DOC] HELPER: Returns the conv of x plus the relu of other.
                conv_out = self.conv(x)
                return torch.add(conv_out, other.relu())

        class Model_v2(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with three chained convs where the output of the first
            # conv's pow is both the input of the next conv and added to the last conv output; the
            # second argument is unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one 3 to 32 conv and two 32 to 32 convs, all 3x3
                # with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, _):
                # [FUNCTION DOC] HELPER: Squares the first conv output, passes it
                # through the second and third convs, and returns the last conv output
                # plus the squared tensor.
                conv_out1 = self.conv(x)
                pow_out = torch.pow(conv_out1, 2)
                conv_out2 = self.conv2(pow_out)
                conv_out3 = self.conv3(conv_out2)
                res = torch.add(conv_out3, pow_out)
                return res

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()

        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise_.binary"]
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise.binary"]

        for other, mod in zip(others, [mod_v1, mod_v2]):
            self._test_code_common(mod, (input, other), include_ops, exclude_ops)

    @xfailIfACL
    def test_conv2d_binary_inplace_fusion_failed_cpu(
        self, include_ops=None, exclude_ops=None
    ):
        # [FUNCTION DOC] Checks that in-place conv binary fusion is NOT used when it is unsafe:
        # five models (the written buffer is a graph input, an alias view, an intermediate with
        # extra users, a reinterpreted view from concat and permute, and a relu output reused
        # after the conv). By default the code must include the out-of-place binary op name and
        # exclude the in-place one (overridable by arguments). Uses channels last inputs. Expected
        # to fail on ACL.
        # Written buffer is graph input, we can't fuse inplace.
        class Model_v1(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding the conv output to the second input directly
            # (a graph input).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                # [FUNCTION DOC] HELPER: Returns the conv of x plus other.
                conv_out = self.conv(x)
                return torch.add(conv_out, other)

        # Written buffer is an alias tensor, we can't fuse inplace.
        class Model_v2(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a slice of the second input (an alias) to the conv
            # output and returning the sum and that input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other):
                # [FUNCTION DOC] HELPER: Returns the conv of x plus a one-row batch slice of
                # other, and other.
                conv_out = self.conv(x)
                return torch.add(conv_out, other[1:2, :, :, :]), other

        class Model_v3(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two convs where the squared first conv output is also
            # reused by a relu after the second conv; the second argument is unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 conv and a 32 to 32 conv, both 3x3
                # with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, _):
                # [FUNCTION DOC] HELPER: Squares the first conv output, applies relu to it,
                # applies the second conv to the squared tensor, and returns conv2 output plus
                # squared plus relu results.
                pow_out = torch.pow(self.conv(x), 2)
                other2 = F.relu(pow_out)
                conv_out2 = self.conv2(pow_out)
                res = torch.add(conv_out2, pow_out)
                res = res + other2
                return res

        # Written buffer is an ReinterpretView, we can't fuse inplace.
        class Model_v4(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a conv output of a relu input to a
            # linear/concat/permute derived tensor (a reinterpreted view as the written buffer).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 conv, a 896 to 896 linear and a relu.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 32, 3, padding=1, bias=True)
                self.linear = torch.nn.Linear(32 * 28, 32 * 28)
                self.relu = torch.nn.ReLU()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies relu and conv to x, applies the linear to y,
                # concatenates y with y plus 1 on dim 1, permutes and reshapes to one by 32 by 28
                # by 28, and returns the sum.
                x = self.conv(self.relu(x))
                y = self.linear(y)
                y = torch.cat((y, y + 1), 1)
                y = torch.ops.aten.permute.default(y, [0, 2, 1]).reshape(1, 32, 28, 28)
                return x + y

        class Model_v5(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a conv output to the relu output that was also the
            # conv input; the first argument is unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 32 to 32 conv and a relu.
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.relu = torch.nn.ReLU()

            def forward(self, _, x):
                # [FUNCTION DOC] HELPER: Applies relu to x and returns conv of the result plus
                # the relu result.
                x1 = self.relu(x)
                return self.conv(x1) + x1

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(2, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(1, 14, 32 * 28),
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
        ]
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()
        mod_v3 = Model_v3().to(memory_format=torch.channels_last).eval()
        mod_v4 = Model_v4().to(memory_format=torch.channels_last).eval()
        mod_v5 = Model_v5().to(memory_format=torch.channels_last).eval()

        if include_ops is None:
            include_ops = ["mkldnn._convolution_pointwise.binary"]
        if exclude_ops is None:
            exclude_ops = ["mkldnn._convolution_pointwise_.binary"]

        for other, mod in zip(others, [mod_v1, mod_v2, mod_v3, mod_v4, mod_v5]):
            self._test_code_common(mod, (input, other), include_ops, exclude_ops)

    def test_conv2d_binary_fusion_failed(self):
        # [FUNCTION DOC] Checks conv plus add is not fused into the mkldnn binary op in three
        # cases: add with alpha not 1 or an operand of different size, add of a tensor to itself
        # (issue 100802), and add of operands of different dtypes (issue 101374). The generated
        # code must contain the plain mkldnn conv pointwise op and neither binary variant.
        # we don't support alpha !=1 case or other has different size with conv's output.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a second input scaled by alpha to the conv output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 32 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x, other, alpha):
                # [FUNCTION DOC] HELPER: Returns the conv of x plus other with the given alpha.
                conv_out = self.conv(x)
                return torch.add(conv_out, other, alpha=alpha)

        # https://github.com/pytorch/pytorch/issues/100802.
        # we can't do the fusion when add's inputs are same tensor.
        class Model2(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding the conv output to itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 16 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the conv output added to itself.
                out = self.conv(x)
                out = torch.add(out, out)
                return out

        # https://github.com/pytorch/pytorch/issues/101374.
        # we can't do the fusion when add's inputs are mixed dtype.
        class Model3(torch.nn.Module):
            # [CLASS DOC] HELPER: Model adding a double precision ones tensor to the
            # float conv output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 16 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the conv output plus a double ones tensor of
                # the same shape.
                temp = self.conv(x)
                other = torch.ones(temp.shape, dtype=torch.double)
                out = torch.add(temp, other)
                return out

        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [
            torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last),
            torch.randn(32, 28, 28),
        ]
        include_ops = ["mkldnn._convolution_pointwise"]
        exclude_ops = [
            "mkldnn._convolution_pointwise.binary",
            "mkldnn._convolution_pointwise_.binary",
        ]

        # case1
        for other, alpha in zip(others, [0.1, 1.0]):
            mod = Model().to(memory_format=torch.channels_last).eval()
            self._test_code_common(mod, (input, other, alpha), include_ops, exclude_ops)
        # case2:
        mod = Model2().to(memory_format=torch.channels_last).eval()
        self._test_code_common(mod, (input,), include_ops, exclude_ops)
        # case3:
        mod = Model3().to(memory_format=torch.channels_last).eval()
        self._test_code_common(mod, (input,), include_ops, exclude_ops)

    @xfailIfACL
    def test_reproduce_99842_issue(self):
        # [FUNCTION DOC] Regression test for issue 99842: a conv followed by relu of the output
        # plus a ones tensor must use the in-place mkldnn binary op in the generated code.
        # Expected to fail on ACL.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying conv, adding a ones tensor and applying relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 3 to 64 channel 3x3 conv with padding 1.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            def forward(self, input_tensor):
                # [FUNCTION DOC] HELPER: Returns relu of the conv output plus a ones tensor of
                # the same size.
                x = self.conv(input_tensor)
                x = F.relu(x + torch.ones(x.size()))
                return x

        input = torch.randn(1, 3, 14, 14)
        mod = Model().eval()
        include_ops = ["mkldnn._convolution_pointwise_.binary"]
        self._test_code_common(mod, (input,), include_ops, [])

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_mkldnn_to_dense_input(self):
        # [FUNCTION DOC] Compiles a function converting an mkldnn layout tensor to dense and
        # adding one, with fullgraph and dynamic shapes, and asserts the result is strided and
        # equals eager. Needs dynamo support and oneDNN.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model converting its input to dense and adding one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dense version of x plus one.
                return x.to_dense() + 1

        x = torch.randn(4, 4).to_mkldnn()
        mod = Model().eval()
        expected = mod(x)
        actual = torch.compile(mod, fullgraph=True, dynamic=True)(x)

        self.assertEqual(actual.layout, torch.strided)
        self.assertEqual(actual, expected)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_mkldnn_to_dense_float8_dtype(self):
        # [FUNCTION DOC] Compiles a function converting an mkldnn tensor to dense with the float8
        # e4m3 dtype and checks the result is strided, has that dtype, and equals eager after
        # conversion to float. Needs dynamo support and oneDNN.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts the mkldnn tensor x to dense with the float8 dtype.
            return x.to_dense(torch.float8_e4m3fn)

        x = torch.randn(4, 4).to_mkldnn()
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(x)

        self.assertEqual(actual.layout, torch.strided)
        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertEqual(actual.float(), expected.float())

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_mkldnn_to_dense_input_backward(self):
        # [FUNCTION DOC] Checks backward through the mkldnn to dense conversion: compiled
        # and eager functions sum the dense tensor plus one, forward results are compared,
        # then gradients to the base tensors are compared after backward. Needs dynamo
        # support and oneDNN.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the dense version of x plus one.
            return (x.to_dense() + 1).sum()

        eager_base = torch.randn(4, 4, requires_grad=True)
        compiled_base = eager_base.detach().clone().requires_grad_()

        expected = fn(eager_base.to_mkldnn())
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(
            compiled_base.to_mkldnn()
        )
        self.assertEqual(actual, expected)

        expected.backward()
        actual.backward()
        self.assertEqual(compiled_base.grad, eager_base.grad)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    def test_mkldnn_to_dense_after_detach(self):
        # [FUNCTION DOC] Checks compile of converting a dense tensor to mkldnn, detaching,
        # converting back to dense and adding one: result must be strided and equal eager. Needs
        # dynamo support and oneDNN.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts x to mkldnn layout, detaches, converts to
            # dense, adds one.
            return x.to_mkldnn().detach().to_dense() + 1

        x = torch.randn(4, 4)
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(x)

        self.assertEqual(actual.layout, torch.strided)
        self.assertEqual(actual, expected)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @config.patch({"cpp_wrapper": True})
    def test_mkldnn_to_dense_cpp_wrapper(self):
        # [FUNCTION DOC] Same as the detach roundtrip test but with the cpp
        # wrapper config enabled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts x to mkldnn layout, detaches, converts to
            # dense, adds one.
            return x.to_mkldnn().detach().to_dense() + 1

        x = torch.randn(4, 4)
        expected = fn(x)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(x)

        self.assertEqual(actual.layout, torch.strided)
        self.assertEqual(actual, expected)

    def test_reproduce_113440_issue_1(self):
        # [FUNCTION DOC] Regression test for issue 113440 (first variant): compiles a model with
        # two parallel convs combined by an in-place add, relu, two more parallel convs, a
        # second in-place add and relu, then runs it twice. Passes if no exception is raised; no
        # values are checked.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two stages of parallel convs merged with the
            # given add function (in-place add in the test) followed by in-place relu when
            # use_relu is true.
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Creates six to six channel convs, relus and the add
                # functions; sets use_relu to true. Extra keyword arguments are ignored.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU(inplace=True)
                self.use_relu = True

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies two parallel convs and merges them with the
                # add function and optional relu, then repeats with two more convs, and
                # returns the result.
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res

        with torch.no_grad():
            example_inputs = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                    1
                ),
            )
            m = Mod(
                lambda x, y: x.add_(y),
            ).eval()
            om = torch.compile(m)
            om(*example_inputs)
            om(*example_inputs)

    def test_reproduce_113440_issue_2(self):
        # [FUNCTION DOC] Regression test for issue 113440 (second variant): compiles a model with
        # two parallel convs merged by in-place add and relu followed by a conv and in-place relu,
        # then runs it twice. Passes if no exception is raised; no values are checked.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with parallel convs merged by the add function, relu, then
            # one more conv and in-place relu; additional convs are created but unused.
            def __init__(
                self,
                add_fn,
                **kwargs,
            ):
                # [FUNCTION DOC] HELPER: Creates seven convs, three relus and three add function
                # references; use_relu is set to true. Extra keyword arguments are ignored.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU(inplace=True)

                self.conv5 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv6 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv7 = torch.nn.Conv2d(6, 6, kernel_size=1, stride=1)
                self.add_fn3 = add_fn
                self.relu3 = torch.nn.ReLU(inplace=True)

                self.use_relu = True

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Merges two parallel conv outputs with the add function,
                # applies optional relu, then conv3 and relu2, and returns the result.
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)

                tmp1 = self.conv3(tmp)
                res = self.relu2(tmp1)

                return res

        with torch.no_grad():
            example_inputs = (
                torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(
                    1
                ),
            )
            m = Mod(
                lambda x, y: x.add_(y),
            ).eval()
            om = torch.compile(m)
            om(*example_inputs)
            om(*example_inputs)

    @requires_mkl
    @xfailIfACL
    def test_reproduce_121253_issue_addmm_fusion_check(self):
        # [FUNCTION DOC] Regression test for issue 121253: addmm with various beta and alpha
        # values over float32 and bfloat16. When beta is 1 or 0 and alpha is 1 the generated code
        # must contain the mkl linear op (float) or the mkldnn linear pointwise op (bf16);
        # otherwise it must not. Requires MKL; expected to fail on ACL.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Model returning an addmm of fixed bias, input and weight with
            # given beta and alpha; those are stored as plain attributes.
            def __init__(self, weight, bias, beta, alpha):
                # [FUNCTION DOC] HELPER: Stores the weight, bias, beta and alpha.
                super().__init__()
                self.weight = weight
                self.bias = bias
                self.beta = beta
                self.alpha = alpha

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the addmm of the stored bias, x and the stored
                # weight using the stored beta and alpha.
                return torch.addmm(
                    self.bias, x, self.weight, beta=self.beta, alpha=self.alpha
                )

        dtypes = [torch.float32]
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            linear_op = (
                "mkl._mkl_linear"
                if dtype == torch.float32
                else "mkldnn._linear_pointwise"
            )
            for beta, alpha in zip([1.0, 0.1, 0.0], [1.0, 0.1, 1.0]):
                weight = torch.nn.Parameter(torch.randn(64, 64, dtype=dtype))
                bias = torch.nn.Parameter(torch.randn(64, dtype=dtype))
                mod = Mod(weight, bias, beta, alpha).to(dtype).eval()
                with torch.no_grad():
                    x = torch.randn(1, 64, dtype=dtype)
                    include_ops = []
                    exclude_ops = []
                    if (beta != 1.0 and beta != 0.0) or alpha != 1.0:
                        exclude_ops = [linear_op]
                    else:
                        include_ops = [linear_op]
                    self._test_code_common(mod, (x,), include_ops, exclude_ops)

    @skipIfNoDynamoSupport
    def test_woq_int4_cpu(self):
        # [FUNCTION DOC] Checks the int4 weight-only CPU pattern: for 2D and 3D inputs and several
        # feature and group sizes, builds a model calling the int4 pack matmul for CPU and
        # verifies the generated code uses the packed weight CPU op (the AOTI name under the cpp
        # wrapper) and does not contain the original aten op. Needs dynamo support.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model holding a random uint8 packed int4 weight and random
            # bfloat16 scales and zeros and applying the aten int4 pack matmul for CPU.
            def __init__(self, in_feature, out_feature, group_size):
                # [FUNCTION DOC] HELPER: Creates the random packed uint8 weight, stores the group
                # size and creates random bfloat16 scales and zeros.
                super().__init__()
                self.weight = torch.randint(
                    0, 255, (out_feature, in_feature // 2), dtype=torch.uint8
                )
                self.group_size = group_size
                self.qScaleAndZeros = torch.rand(
                    (in_feature // group_size, out_feature, 2), dtype=torch.bfloat16
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Flattens a 3D-or-higher input to 2D, runs the int4 pack
                # matmul, and reshapes the result; for 2D input runs the matmul directly.
                # KNOWN ISSUE: the final reshape uses the leading dims of the already flattened
                # input, so for 3D inputs the output stays 2D instead of being restored to the
                # original leading dims.
                if x.ndim > 2:
                    x = x.reshape(-1, x.shape[-1])
                    y = torch.ops.aten._weight_int4pack_mm_for_cpu.default(
                        x, self.weight, self.group_size, self.qScaleAndZeros
                    )
                    return y.reshape(*x.shape[:-1], y.shape[-1])
                return torch.ops.aten._weight_int4pack_mm_for_cpu.default(
                    x, self.weight, self.group_size, self.qScaleAndZeros
                )

        bs = 4
        seq = 8
        x_dim_list = [2, 3]
        in_feature_list = [256, 512]
        out_feature_list = [256, 512]
        group_size_list = [64, 128]
        cases = itertools.product(
            x_dim_list, in_feature_list, out_feature_list, group_size_list
        )
        for x_dim, in_feature, out_feature, group_size in cases:
            x_shape = (seq, in_feature) if x_dim == 2 else (bs, seq, in_feature)
            x = torch.randn(x_shape, dtype=torch.bfloat16)
            m = M(in_feature, out_feature, group_size).eval()

            include_ops = [
                "aoti_torch_cpu__weight_int4pack_mm_cpu_tensor"
                if torch._inductor.config.cpp_wrapper
                else "torch.ops.quantized.int4mm_packed_weight_cpu.default"
            ]
            self._test_code_common(
                m,
                (x,),
                include_ops,
                ["torch.ops.aten._weight_int4pack_mm_for_cpu.default"],
            )


class TestDynamicPatternMatcherGeneric(TestPatternMatcherBase):
    # [CLASS DOC] Runs a few of the generic pattern matcher tests with dynamic shape config
    # patched in setUp. Tests are aliases of methods from the generic test class, plus two new
    # tests. It is instantiated per device type.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setup and then enters a dynamo config patch
        # enabling dynamic shapes, assuming dynamic by default and specializing floats (to keep
        # kernel counts stable) for the test duration.
        super().setUp()
        self.ctx_stack.enter_context(
            # When testing kernel counts, unspecializing float causes wobbling of our tests because
            # we end up reusing the same compiled region across tests. Thus we purposely specialize floats
            # here since we primarily care about number of kernels generated in the absence of compile
            # caching.
            dynamo_config.patch(
                {
                    "dynamic_shapes": True,
                    "assume_static_by_default": False,
                    "specialize_float": True,
                }
            )
        )

    _test_conv_unary_base = TestPatternMatcherGeneric._test_conv_unary_base
    test_conv2d_unary_dynamic_shapes = TestPatternMatcherGeneric.test_conv2d_unary
    test_conv3d_unary_dynamic_shapes = TestPatternMatcherGeneric.test_conv3d_unary
    _test_conv_binary_base = TestPatternMatcherGeneric._test_conv_binary_base
    test_conv2d_binary_dynamic_shapes = TestPatternMatcherGeneric.test_conv2d_binary
    test_conv3d_binary_dynamic_shapes = TestPatternMatcherGeneric.test_conv3d_binary

    def test_conv_transpose2d_dynamic_shapes(self, device):
        # [FUNCTION DOC] Compiles a transposed conv 2D model with dynamic shapes; the matcher
        # check does nothing since this pattern is not supported, so only numerics are verified.
        self.device = device

        # We don't support conv_transpose2d for now.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying one 3 to 16 channel transposed conv 2D.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the transposed conv layer with kernel 3, stride
                # 2, padding 1.
                super().__init__()
                self.conv_transpose2d = torch.nn.ConvTranspose2d(
                    3, 16, 3, stride=2, padding=1
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the transposed conv of x.
                return self.conv_transpose2d(x)

        x_shape = (1, 3, 28, 28)
        mod = M().eval()
        v = torch.randn(x_shape, dtype=torch.float32)

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check that intentionally does nothing.
            return

        self._test_common(mod, (v,), matcher_check_fn)

    @skipIfXpu(
        msg="Different from CPU, two linears will be concat on XPU for better performance"
    )
    def test_multi_linear_share_same_input_dynamic(self, device):
        # [FUNCTION DOC] Dynamic shape version of the llama-style shared-input linear test: SiLU
        # and ReLU branches multiplied, for each lowp dtype, expecting 9 unary fusion nodes in 2
        # matches (zero on ACL), 6 and 2 reshape linear reshape nodes and matches, and 2 weight
        # packs. Skipped on XPU.
        self.device = device

        # llama pattern.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model multiplying a SiLU of one bias-free 16 by 16 linear and a
            # ReLU of another linear applied to the same input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the two bias-free 16 to 16 linear layers.
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of silu of the first linear and relu
                # of the second linear, both applied to x.
                return F.silu(self.w1(x)) * F.relu(self.w2(x))

        dtypes = []
        if is_mkldnn_bf16_supported(self.device):
            dtypes.append(torch.bfloat16)
        if is_mkldnn_fp16_supported(self.device):
            dtypes.append(torch.float16)

        def matcher_check_fn():
            # [FUNCTION DOC] HELPER: Matcher check closure: asserts unary fusion nodes 9 and
            # matches 2 (both zero on ACL), reshape linear reshape nodes 6 and matches 2, and
            # 2 weight packs.
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"],
                0 if TEST_ACL else 9,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_unary_fusion_matcher_count"],
                0 if TEST_ACL else 2,
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"], 6
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_count"], 2
            )
            self.assertEqual(
                counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"], 2
            )

        for dtype in dtypes:
            mod = M().to(dtype).eval()
            v = torch.randn(2, 4, 16).to(dtype)
            self._test_common(mod, (v,), matcher_check_fn, rtol=1e-2, atol=1e-2)


class TestDynamicPatternMatcher(TestPatternMatcherBase):
    # [CLASS DOC] Dynamic shape variants of CPU pattern matcher tests: linear unary and non-
    # contiguous linear tests are attached as aliases of the originals, with dynamic shapes
    # enabled in setUp.
    test_linear_unary_dynamic_shapes = TestPatternMatcher.test_linear_unary
    test_linear_input_non_contiguous_3D_wo_bias_dynamic_shapes = (
        TestPatternMatcher.test_linear_input_non_contiguous_3D_wo_bias
    )

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setup and then enters a dynamo config patch
        # enabling dynamic shapes, assuming dynamic by default and specializing floats, for
        # the test duration.
        super().setUp()
        self.ctx_stack.enter_context(
            # When testing kernel counts, unspecializing float causes wobbling of our tests because
            # we end up reusing the same compiled region across tests. Thus we purposely specialize floats
            # here since we primarily care about number of kernels generated in the absence of compile
            # caching.
            dynamo_config.patch(
                {
                    "dynamic_shapes": True,
                    "assume_static_by_default": False,
                    "specialize_float": True,
                }
            )
        )


instantiate_device_type_tests(
    TestPatternMatcherGeneric, globals(), allow_xpu=True, only_for=("cpu", "xpu")
)
instantiate_device_type_tests(
    TestDynamicPatternMatcherGeneric, globals(), allow_xpu=True, only_for=("cpu", "xpu")
)
instantiate_parametrized_tests(TestPatternMatcher)
if __name__ == "__main__":
    if IS_LINUX and (HAS_CPU) and torch.backends.mkldnn.is_available():
        run_tests()
