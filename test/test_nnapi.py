#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests for lowering traced PyTorch models to Android NNAPI.
# Each test builds a small module (float, quantized and/or NHWC channels-last
# inputs), traces it, converts it with convert_model_to_nnapi and, only if an
# NNAPI runtime library is available, compares eager and NNAPI outputs.
# Also defines the helpers qpt and nhwc and the TestNNAPI class (skipped unless
# the QNNPACK quantized engine is supported). Importing has no side effects.

import ctypes
import os
import unittest

import torch
from torch.backends._nnapi.prepare import convert_model_to_nnapi
from torch.testing._internal.common_quantized import supported_qengines
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


def qpt(t, scale, zero_point, dtype=torch.quint8):
    # [FUNCTION DOC] HELPER: Quantize a value per tensor. Converts the input to a
    # tensor, then quantizes it with the given scale, zero point and dtype
    # (default quint8). Returns the quantized tensor.
    t = torch.tensor(t)
    return torch.quantize_per_tensor(t, scale, zero_point, dtype)


def nhwc(t):
    # [FUNCTION DOC] HELPER: Return a channels-last contiguous clone of the tensor,
    # tagged with the attribute nnapi_nhwc set to true so the NNAPI converter treats
    # it as NHWC data. The input tensor is not modified.
    t = t.clone().contiguous(memory_format=torch.channels_last)
    t.nnapi_nhwc = True
    return t


@unittest.skipUnless(
    "qnnpack" in supported_qengines,
    "This Pytorch Build has not been built with or does not support QNNPACK",
)
class TestNNAPI(TestCase):
    # [CLASS DOC] Test case for NNAPI conversion. Class attribute hw_classification
    # marks it as generic hardware. Instance state: can_run_nnapi says whether an NNAPI
    # runtime was loaded in setUp. The check method is the common driver; subclasses
    # may override call_lowering_to_nnapi and set_can_run_nnapi to reuse the tests.
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        # [FUNCTION DOC] Per-test setup. Selects the qnnpack quantized engine (global
        # state in torch.backends.quantized). If the environment variable
        # LIBNEURALNETWORKS_PATH is set, loads that shared library, prints a notice and
        # sets can_run_nnapi true; otherwise sets can_run_nnapi false.
        super().setUp()
        # Avoid saturation in fbgemm
        torch.backends.quantized.engine = "qnnpack"

        libneuralnetworks_path = os.environ.get("LIBNEURALNETWORKS_PATH")
        if libneuralnetworks_path:
            ctypes.cdll.LoadLibrary(libneuralnetworks_path)
            print("Will attempt to run NNAPI models.")
            self.can_run_nnapi = True
        else:
            self.can_run_nnapi = False

    # Created for easy override by subclasses (eg TestNnapiBackend)
    def call_lowering_to_nnapi(self, traced_module, args):
        # [FUNCTION DOC] Convert a traced module to an NNAPI module using the given
        # example args via convert_model_to_nnapi and return the result. Exists as an
        # override point for subclasses.
        return convert_model_to_nnapi(traced_module, args)

    # Created for subclasses to set can_run_nnapi (eg TestNnapiBackend)
    def set_can_run_nnapi(self, can_run):
        # [FUNCTION DOC] Setter for can_run_nnapi, so subclasses can control whether the
        # NNAPI module is actually executed and compared.
        self.can_run_nnapi = can_run

    def check(
        self,
        module,
        arg_or_args,
        *,
        trace_args=None,
        convert_args=None,
        atol_rtol=None,
        limit=None,
        expected_memory_format=None,
    ):
        # [FUNCTION DOC] Common test driver. Under no-grad, puts the module in eval mode
        # and traces it with trace_args (falling back to args; a single tensor is wrapped
        # in a list). Lowers the traced module with convert_args (falling back to args),
        # which lets tests use zero-sized "flexible" dims for conversion. If NNAPI cannot
        # run, returns after conversion succeeded. Otherwise runs eager and NNAPI modules
        # on args and asserts equal outputs, using atol_rtol as (atol, rtol) if given. If
        # limit is given, outputs are treated as quantized: the integer representation
        # differences are counted and if more than limit differ, the comparison is
        # re-run with zero tolerance to produce a clear failure. If expected_memory_format
        # is given, asserts the NNAPI output is contiguous in that format.
        with torch.no_grad():
            if isinstance(arg_or_args, torch.Tensor):
                args = [arg_or_args]
            else:
                args = arg_or_args
            module.eval()
            traced = torch.jit.trace(module, trace_args or args)
            nnapi_module = self.call_lowering_to_nnapi(traced, convert_args or args)
            if not self.can_run_nnapi:
                # Only test that the model was converted successfully.
                return
            eager_output = module(*args)
            nnapi_output = nnapi_module(*args)
            kwargs = {}
            if atol_rtol is not None:
                kwargs["atol"] = atol_rtol[0]
                kwargs["rtol"] = atol_rtol[1]
            self.assertEqual(eager_output, nnapi_output, **kwargs)
            if limit is not None:
                mismatches = eager_output.int_repr().to(
                    torch.int32
                ) - nnapi_output.int_repr().to(torch.int32)
                if mismatches.count_nonzero() > limit:
                    # Too many mismatches.  Re-run the check with no tolerance
                    # to get a nice message.
                    self.assertEqual(eager_output, nnapi_output, atol=0, rtol=0)
            if expected_memory_format:
                self.assertTrue(
                    nnapi_output.is_contiguous(memory_format=expected_memory_format)
                )

    def float_and_quant_and_nhwc(self, inp_float, scale, zero_point):
        # [FUNCTION DOC] HELPER: Build the standard input variants for pooling and
        # upsample tests. Seeds the RNG with 29, quantizes inp_float with fixed scale 0.03
        # and zero point 128, and returns a list of four (name, tensor) pairs: float,
        # float-nhwc, quant, quant-nhwc.
        # KNOWN ISSUE: the scale and zero_point arguments are ignored; fixed values are used.
        torch.manual_seed(29)
        inp_quant = qpt(inp_float, 0.03, 128)
        return [
            ("float", inp_float),
            ("float-nhwc", nhwc(inp_float)),
            ("quant", inp_quant),
            ("quant-nhwc", nhwc(inp_quant)),
        ]

    def test_prelu(self):
        # [FUNCTION DOC] Checks PReLU lowering: single-parameter PReLU on a plain input,
        # a four-channel PReLU with set weights on an NHWC input, and the same module
        # traced with a fixed size but converted with zero-sized spatial dims.
        arg = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        single_a = torch.nn.PReLU()
        self.check(single_a, arg)
        multi_a = torch.nn.PReLU(4)
        with torch.no_grad():
            multi_a.weight.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.check(multi_a, nhwc(arg))

        # Test flexible size
        self.check(
            multi_a,
            arg,
            trace_args=[torch.zeros(1, 4, 3, 3)],
            convert_args=[nhwc(torch.zeros(1, 4, 0, 0))],
        )

    def test_quantize(self):
        # [FUNCTION DOC] Checks that the Quantize module (scale 0.25, zero point 2, quint8)
        # lowers correctly on a small NHWC float input.
        self.check(
            torch.ao.nn.quantized.Quantize(0.25, 2, torch.quint8),
            nhwc(torch.tensor([[[[1.0]], [[2.0]]]])),
        )

    def test_dequantize(self):
        # [FUNCTION DOC] Checks that the DeQuantize module lowers correctly on a small
        # NHWC quantized input.
        self.check(
            torch.ao.nn.quantized.DeQuantize(), nhwc(qpt([[[[1.0]], [[2.0]]]], 0.25, 2))
        )

    def test_unsqueeze(self):
        # [FUNCTION DOC] Checks unsqueeze lowering for dims -2, -1, 0, 1 and 2 on a rank-3
        # input, using a parameterized helper module.
        class UnsqueezeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that unsqueezes its input at a fixed dimension.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Store the dimension to unsqueeze.
                super().__init__()
                self.dim = dim

            def forward(self, arg):
                # [FUNCTION DOC] HELPER: Return the input unsqueezed at the stored dimension.
                return arg.unsqueeze(self.dim)

        self.check(UnsqueezeModule(-2), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(-1), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(0), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(1), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(2), torch.randn(4, 2, 2))

    def test_reshape(self):
        # [FUNCTION DOC] Checks reshape lowering: a contiguous input to a fixed shape, an
        # NHWC input to a shape with an inferred dim, and verifies that reshaping an NHWC
        # input to a shape that is incompatible raises an error mentioning "target size".
        class ReshapeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes its input to a stored target shape.
            def __init__(self, shape):
                # [FUNCTION DOC] HELPER: Store the target shape.
                super().__init__()
                self.shape = shape

            def forward(self, arg):
                # [FUNCTION DOC] HELPER: Return the input reshaped to the stored shape.
                return arg.reshape(self.shape)

        self.check(ReshapeModule((2, 4)), torch.randn(4, 2, 1, 1))

        self.check(ReshapeModule((8, -1)), nhwc(torch.randn(4, 2, 1, 1)))

        with self.assertRaisesRegex(Exception, "target size"):
            self.check(ReshapeModule((2, 4)), nhwc(torch.randn(4, 2, 1, 1)))

    def test_flatten(self):
        # [FUNCTION DOC] Checks Flatten lowering across several start/end dim settings,
        # with a flexible (zero-sized) batch dim at conversion, and with NHWC inputs.
        # Also asserts three conversion errors: flattening an NHWC input in an unsupported
        # case ("not supported on NHWC"), flattening flexible dims, and a flatten range
        # over more than one flexible dim ("Only 1 dim").
        for mod in [
            torch.nn.Flatten(),
            torch.nn.Flatten(start_dim=2, end_dim=3),
            torch.nn.Flatten(start_dim=2, end_dim=4),
            torch.nn.Flatten(start_dim=0, end_dim=-2),
            torch.nn.Flatten(start_dim=0, end_dim=4),
        ]:
            self.check(mod, torch.randn(4, 2, 1, 3, 7))

        # flex inputs
        self.check(
            torch.nn.Flatten(),
            torch.randn(4, 2, 1, 3, 7),
            convert_args=[torch.zeros(0, 2, 1, 3, 7)],
        )

        # channels last
        self.check(torch.nn.Flatten(), nhwc(torch.randn(2, 1, 4, 7)))
        self.check(torch.nn.Flatten(), nhwc(torch.randn(2, 3, 1, 1)))

        # Exceptions
        with self.assertRaisesRegex(Exception, "not supported on NHWC"):
            self.check(torch.nn.Flatten(), nhwc(torch.randn(1, 3, 4, 4)))
        with self.assertRaisesRegex(
            Exception, "Flattening flexible dims is not supported yet"
        ):
            self.check(torch.nn.Flatten(), torch.randn(4, 2, 0, 0, 7))
        with self.assertRaisesRegex(Exception, "Only 1 dim"):
            self.check(
                torch.nn.Flatten(start_dim=1, end_dim=-2), torch.randn(0, 2, 1, 3, 0)
            )

    def test_slice(self):
        # [FUNCTION DOC] Checks slice lowering with a start/stop/step module and a simple
        # tail-slice module, including a flexible last dim at conversion. Asserts that
        # slicing when the sliced dims are flexible raises an error ("slice with flexible
        # shape").
        class SliceModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that slices its input from index 1 on the first dim,
            # with stored start/stop/step on the second dim, and the full third dim.
            def __init__(self, start, stop, step):
                # [FUNCTION DOC] HELPER: Store the start, stop and step of the middle-dim slice.
                super().__init__()
                self.start = start
                self.stop = stop
                self.step = step

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Return the input sliced as described in the class doc.
                return t[1:, self.start : self.stop : self.step, :]

        class SliceModule2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that returns its input without the first three
            # elements of the first dim.
            def forward(self, t):
                # [FUNCTION DOC] HELPER: Return the tail slice of the input starting at index 3.
                return t[3:]

        self.check(SliceModule(1, 5, 2), torch.randn(4, 6, 2))
        self.check(SliceModule2(), torch.randn(5))

        # flex inputs
        self.check(
            SliceModule(1, 5, 2),
            torch.randn(4, 6, 2),
            convert_args=[torch.zeros(4, 6, 0)],
        )
        with self.assertRaisesRegex(Exception, "slice with flexible shape"):
            self.check(
                SliceModule(1, 5, 2),
                torch.randn(4, 6, 2),
                convert_args=[torch.zeros(0, 0, 0)],
            )

    def test_cat(self):
        # [FUNCTION DOC] Checks concatenation of two tensors along dim 0 and dim 1, with
        # plain and NHWC inputs, and with zero-sized flexible shapes at conversion.
        class CatModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that concatenates two inputs along a stored dim.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: Store the concatenation dim.
                super().__init__()
                self.dim = dim

            def forward(self, t1, t2):
                # [FUNCTION DOC] HELPER: Return both inputs concatenated along the stored dim.
                return torch.cat([t1, t2], self.dim)

        self.check(
            CatModule(0),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(2, 2, 3, 3),
            ],
        )

        self.check(
            CatModule(1),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(1, 4, 3, 3),
            ],
        )

        self.check(
            CatModule(1),
            [
                nhwc(torch.randn(1, 2, 3, 3)),
                nhwc(torch.randn(1, 4, 3, 3)),
            ],
        )

        self.check(
            CatModule(1),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(1, 4, 3, 3),
            ],
            convert_args=[torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)],
        )

    def test_pointwise_unary(self):
        # [FUNCTION DOC] For each of relu and sigmoid (as a subtest), checks lowering of the
        # op on a float input and on a quantized input with scale 1/256.
        for op in ["relu", "sigmoid"]:
            with self.subTest(op):

                class UnaryModule(torch.nn.Module):
                    # [CLASS DOC] HELPER: Module applying the unary op chosen by the enclosing loop
                    # variable (relu or sigmoid); other values raise a generic exception.
                    def forward(self, arg):
                        # [FUNCTION DOC] HELPER: Apply the selected unary op to the input;
                        # raises for an unknown op name.
                        if op == "relu":
                            return torch.nn.functional.relu(arg)
                        if op == "sigmoid":
                            return torch.sigmoid(arg)
                        raise Exception("Bad op")  # noqa: TRY002

                self.check(UnaryModule(), torch.tensor([-1.0, 1.0]))
                self.check(
                    UnaryModule(),
                    qpt(torch.tensor([-1.0, 1.0]), 1.0 / 256, 0),
                )

    def test_pointwise_binary(self):
        # [FUNCTION DOC] For each of add, sub, mul and div (as a subtest), checks lowering
        # with equal-shape inputs and with broadcast inputs of equal rank. Asserts that a
        # broadcast between different ranks fails conversion ("Non-equal-rank broadcast").
        for op in ["add", "sub", "mul", "div"]:
            with self.subTest(op):

                class BinaryModule(torch.nn.Module):
                    # [CLASS DOC] HELPER: Module applying the binary arithmetic op chosen by the
                    # enclosing loop variable to its two inputs. Unknown ops raise.
                    def forward(self, lhs, rhs):
                        # [FUNCTION DOC] HELPER: Apply the selected binary op to both inputs;
                        # raises for an unknown op name.
                        if op == "add":
                            return lhs + rhs
                        if op == "sub":
                            return lhs - rhs
                        if op == "mul":
                            return lhs * rhs
                        if op == "div":
                            return lhs / rhs
                        raise Exception("Bad op")  # noqa: TRY002

                self.check(
                    BinaryModule(),
                    [
                        torch.tensor([1.0, 2.0]),
                        torch.tensor([3.0, 4.0]),
                    ],
                )

                self.check(
                    BinaryModule(),
                    [
                        torch.tensor([[1.0, 2.0]]),
                        torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                    ],
                )

                with self.assertRaisesRegex(Exception, "Non-equal-rank broadcast"):
                    self.check(
                        BinaryModule(),
                        [
                            torch.tensor([1.0, 2.0]),
                            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                        ],
                    )

    def test_pointwise_binary_const(self):
        # [FUNCTION DOC] Checks addition of an input and a captured constant tensor, with
        # the constant on either side, for contiguous and NHWC inputs, and asserts the
        # output memory format matches the input layout.
        const = torch.randn(1, 4, 6, 6)

        class ArgPlusConst(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a captured constant tensor to its input.
            def forward(self, arg):
                # [FUNCTION DOC] HELPER: Return the input plus the captured constant.
                return arg + const

        class ConstPlusArg(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding its input to a captured constant tensor,
            # with the constant as the left operand.
            def forward(self, arg):
                # [FUNCTION DOC] HELPER: Return the captured constant plus the input.
                return const + arg

        arg_contig = torch.randn(2, 4, 6, 6)
        arg_nhwc = nhwc(torch.randn(2, 4, 6, 6))

        for mod_class in [ArgPlusConst, ConstPlusArg]:
            for use_nhwc in [False, True]:
                with self.subTest(mod_class=mod_class.__name__, use_nhwc=use_nhwc):
                    arg = arg_nhwc if use_nhwc else arg_contig
                    memory_format = (
                        torch.channels_last if use_nhwc else torch.contiguous_format
                    )
                    self.check(mod_class(), arg, expected_memory_format=memory_format)

    def test_hardtanh(self):
        # [FUNCTION DOC] Checks Hardtanh lowering with default limits and with limits 0 and 6,
        # and asserts that other limits (0 and 5) fail conversion ("hardtanh with args").
        inp = torch.tensor([-2.0, -0.5, 0.5, 2.0, 7.0])
        self.check(torch.nn.Hardtanh(), inp)
        self.check(torch.nn.Hardtanh(0.0, 6.0), inp)
        with self.assertRaisesRegex(Exception, "hardtanh with args"):
            self.check(torch.nn.Hardtanh(0.0, 5.0), inp)

    def test_softmax(self):
        # [FUNCTION DOC] Checks Softmax lowering with default dim, with dim 0, and with a
        # zero-sized flexible shape at conversion.
        inp = torch.tensor([[-2.0, -0.5], [0.5, 2.0]])
        self.check(torch.nn.Softmax(), inp)
        self.check(torch.nn.Softmax(dim=0), inp)
        # Test flexible size
        self.check(
            torch.nn.Softmax(),
            inp,
            convert_args=[torch.zeros(0, 0)],
        )

    def test_to(self):
        # [FUNCTION DOC] Checks lowering of a tensor move to cpu, followed by a PReLU layer
        # (because an input operand cannot also be an output), including a flexible shape.
        class ToCPU(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that moves its input to cpu and applies a PReLU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the PReLU submodule.
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Move the input to cpu and return the PReLU of it.
                y = x.to("cpu")
                # add prelu since input operand can't be output
                return self.prelu(y)

        arg = torch.randn(1, 2, 3, 3)
        self.check(ToCPU(), arg)
        # Test flexible size
        self.check(
            ToCPU(),
            arg,
            convert_args=[torch.zeros(1, 2, 0, 0)],
        )

    def test_detach(self):
        # [FUNCTION DOC] Checks lowering of a detach followed by a relu, with a fixed
        # shape and with a flexible spatial shape at conversion.
        class DetachModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that detaches its input and applies relu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return relu of the detached input.
                y = x.detach()
                return torch.nn.functional.relu(y)

        self.check(DetachModule(), torch.randn(1, 2, 3, 3))
        self.check(
            DetachModule(),
            torch.randn(1, 2, 3, 3),
            convert_args=[torch.zeros(1, 2, 0, 0)],
        )

    def test_log_softmax(self):
        # [FUNCTION DOC] Checks LogSoftmax lowering with the default dim and with dim 0.
        inp = torch.randn(3, 10)
        self.check(torch.nn.LogSoftmax(), inp)
        self.check(torch.nn.LogSoftmax(0), inp)

    def test_mean(self):
        # [FUNCTION DOC] Checks mean lowering over single dims, spatial dims (positive and
        # negative), with NHWC input, and with keepdim set.
        class MeanModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the mean over stored dim(s) with a stored
            # keepdim flag.
            def __init__(self, dim, keep=False):
                # [FUNCTION DOC] HELPER: Store the reduction dim(s) and the keepdim flag.
                super().__init__()
                self.dim = dim
                self.keep = keep

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Return the mean of the input using the stored settings.
                return torch.mean(t, dim=self.dim, keepdim=self.keep)

        self.check(MeanModule(0), torch.randn(2, 3))
        self.check(MeanModule(1), torch.randn(2, 3))
        self.check(MeanModule([2, 3]), torch.randn(2, 3, 6, 6))
        self.check(MeanModule([2, 3]), nhwc(torch.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2]), nhwc(torch.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2], keep=True), nhwc(torch.randn(2, 3, 6, 6)))

    def test_max_pool2d(self):
        # [FUNCTION DOC] Checks MaxPool2d lowering with several kernel and stride settings
        # for each input variant (float, NHWC, quantized, quantized NHWC), one subtest each.
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                self.check(torch.nn.MaxPool2d(2), inp)
                self.check(torch.nn.MaxPool2d((3, 4)), inp)
                self.check(torch.nn.MaxPool2d((3, 4), (1, 2)), inp)

    def test_avg_pool2d(self):
        # [FUNCTION DOC] Checks AvgPool2d lowering for several kernel/stride settings and
        # each input variant. For quantized variants allows an absolute difference of one
        # quantum on up to as many elements as the output has; also converts with a
        # zero-sized spatial flexible input of matching layout and quantization.
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                atol_rtol = None
                limit = None
                convert_dims = (2, 3, 0, 0)
                convert_arg = torch.zeros(*convert_dims)

                for model in (
                    torch.nn.AvgPool2d(2),
                    torch.nn.AvgPool2d((3, 4)),
                    torch.nn.AvgPool2d((3, 4), (1, 2)),
                ):
                    if "quant" in name:
                        atol_rtol = (1, 0)
                        limit = model(inp).numel()
                        convert_arg = qpt(torch.zeros(*convert_dims), 1.0 / 16, 128)
                    if "nhwc" in name:
                        convert_arg = nhwc(convert_arg)

                    self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                    self.check(
                        model,
                        inp,
                        convert_args=[convert_arg],
                        atol_rtol=atol_rtol,
                        limit=limit,
                    )

    def test_adaptive_avg_pool2d(self):
        # [FUNCTION DOC] Checks AdaptiveAvgPool2d lowering for each input variant with output
        # size 1 by 1, and asserts that output size 2 by 2 fails conversion ("with output size").
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                self.check(torch.nn.AdaptiveAvgPool2d((1, 1)), inp)
                with self.assertRaisesRegex(Exception, "with output size"):
                    self.check(torch.nn.AdaptiveAvgPool2d((2, 2)), inp)

    def test_upsample_nearest2d(self):
        # [FUNCTION DOC] Checks UpsamplingNearest2d lowering for each input variant, using
        # several fixed output sizes and scale factors, and for two settings also
        # converting with flexible zero-sized spatial dims of the matching variant.
        convert_args = dict(
            self.float_and_quant_and_nhwc(torch.randn(2, 3, 0, 0), 0.3, 128)
        )
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                self.check(torch.nn.UpsamplingNearest2d(size=(16, 20)), inp)
                self.check(torch.nn.UpsamplingNearest2d(size=(24, 32)), inp)
                self.check(torch.nn.UpsamplingNearest2d(size=(36, 48)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(1.5, 1.5)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(3.0, 3.0)), inp)

                self.check(
                    torch.nn.UpsamplingNearest2d(size=(24, 32)),
                    inp,
                    convert_args=[convert_args[name]],
                )
                self.check(
                    torch.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)),
                    inp,
                    convert_args=[convert_args[name]],
                )

    def test_linear(self):
        # [FUNCTION DOC] Checks Linear lowering (16 to 32 features) on a fixed batch, and
        # with a zero-sized flexible batch dim at conversion. Seeds the RNG with 29.
        torch.manual_seed(29)
        self.check(torch.nn.Linear(16, 32), torch.randn(2, 16))
        self.check(
            torch.nn.Linear(16, 32),
            torch.randn(2, 16),
            convert_args=[torch.zeros(0, 16)],
        )

    def test_conv2d(self):
        # [FUNCTION DOC] Checks Conv2d lowering over a table of configurations (kernel,
        # stride, padding, groups, bias or not) for each of float, NHWC, quantized and
        # quantized NHWC variants. Quantized variants are prepared and converted with the
        # qnnpack qconfig and compared with an absolute tolerance of one quantum, allowing
        # about three percent of outputs to differ. Each case also converts with flexible
        # zero-sized batch and spatial dims.
        cases = [
            # in_ch, out_ch, kernel, stride, padding, groups, bias, input_dim,      name
            (4, 8, (3, 3), 1, 0, 1, 1, (2, 4, 16, 16), "3x3"),  # noqa: E201,E241
            (4, 8, (3, 3), 1, 0, 1, 0, (2, 4, 16, 16), "3x3nobias"),  # noqa: E201,E241
            (4, 16, (3, 3), 1, 1, 1, 1, (2, 4, 16, 16), "3x3p1"),  # noqa: E201,E241
            (8, 8, (3, 3), 2, 0, 1, 1, (2, 8, 16, 16), "3x3s2"),  # noqa: E201,E241
            (4, 8, (5, 5), 1, 0, 1, 1, (2, 4, 16, 16), "5x5"),  # noqa: E201,E241
            (4, 4, (3, 3), 1, 0, 4, 1, (2, 4, 16, 16), "3x3dw"),  # noqa: E201,E241
            (8, 4, (1, 1), 1, 0, 1, 1, (2, 8, 16, 16), "1x1"),  # noqa: E201,E241
        ]

        for kind in ["float", "float-nhwc", "quant", "quant-nhwc"]:
            for case in cases:
                (
                    in_ch,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                    groups,
                    bias,
                    input_dim,
                    name,
                ) = case
                with self.subTest(f"{kind}-{name}"):
                    inp = torch.randn(input_dim)
                    model = torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel,
                        stride,
                        padding,
                        groups=groups,
                        bias=bool(bias),
                    )
                    output_size = model(inp).numel()
                    atol_rtol = None
                    limit = None
                    convert_dims = (0, in_ch, 0, 0)
                    convert_arg = torch.zeros(*convert_dims)

                    if "quant" in kind:
                        model = torch.nn.Sequential(model)
                        model.eval()
                        model.qconfig = torch.ao.quantization.get_default_qconfig(
                            "qnnpack"
                        )
                        model = torch.ao.quantization.prepare(model)
                        model(inp)
                        model = torch.ao.quantization.convert(model)
                        inp = qpt(inp, 1.0 / 16, 128)
                        # I've seen numerical differences between QNNPACK and NNAPI,
                        # but never more than 1 quantum, and never more than ~1% of
                        # the output in this test.
                        atol_rtol = (1, 0)
                        limit = output_size * 0.03
                        convert_arg = qpt(torch.zeros(*convert_dims), 1.0 / 16, 128)

                    if "nhwc" in kind:
                        inp = nhwc(inp)
                        convert_arg = nhwc(convert_arg)

                    self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                    self.check(
                        model,
                        inp,
                        convert_args=[convert_arg],
                        atol_rtol=atol_rtol,
                        limit=limit,
                    )

    def test_conv2d_transpose(self):
        # [FUNCTION DOC] Checks ConvTranspose2d lowering for float, NHWC, quantized and
        # quantized NHWC variants with a flexible zero-sized spatial conversion too. Float
        # uses a small tolerance; quantized allows one quantum and up to ten percent of
        # outputs to differ. Seeds the RNG with 29.
        torch.manual_seed(29)
        in_ch, out_ch, kernel = (5, 7, (2, 2))
        input_dim = (4, 5, 3, 3)
        convert_dims = input_dim[:2] + (0, 0)

        for kind in ["float", "float-nhwc", "quant", "quant-nhwc"]:
            with self.subTest(kind):
                inp = torch.randn(input_dim)
                model = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel)
                output_size = model(inp).numel()
                atol_rtol = (0.0002, 0)
                limit = None
                convert_arg = torch.zeros(*convert_dims)

                if "quant" in kind:
                    model = torch.ao.nn.quantized.ConvTranspose2d(in_ch, out_ch, kernel)
                    model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
                    inp = qpt(inp, 1.0 / 16, 128)
                    # I've seen numerical differences between QNNPACK and NNAPI,
                    # but never more than 1 quantum, and never more than ~10% of
                    # the output in this test.
                    atol_rtol = (1, 0)
                    limit = output_size * 0.1
                    convert_arg = qpt(convert_arg, 1.0 / 16, 128)

                if "nhwc" in kind:
                    inp = nhwc(inp)
                    convert_arg = nhwc(convert_arg)

                self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                self.check(
                    model,
                    inp,
                    convert_args=[convert_arg],
                    atol_rtol=atol_rtol,
                    limit=limit,
                )

    def test_qadd(self):
        # [FUNCTION DOC] Checks quantized functional add, add_relu and mul with fixed output
        # scale 0.5 and zero point 120, as a subtest each, using quantized inputs of
        # rank 1 and 2 and flexible conversion args where one or both inputs are zeros.
        func = torch.ao.nn.quantized.QFunctional()
        func.scale = 0.5
        func.zero_point = 120

        class AddMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the quantized add of two inputs.
            def forward(self, lhs, rhs):
                # [FUNCTION DOC] HELPER: Return the quantized add of both inputs.
                return func.add(lhs, rhs)

        class AddReluMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the quantized add-with-relu of two inputs.
            def forward(self, lhs, rhs):
                # [FUNCTION DOC] HELPER: Return the quantized add-with-relu of both inputs.
                return func.add_relu(lhs, rhs)

        class MulMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the quantized multiply of two inputs.
            def forward(self, lhs, rhs):
                # [FUNCTION DOC] HELPER: Return the quantized product of both inputs.
                return func.mul(lhs, rhs)

        for name, mod in [("add", AddMod), ("add_relu", AddReluMod), ("mul", MulMod)]:
            with self.subTest(name):
                self.check(
                    mod(),
                    [
                        qpt([1.0, 2.0], 0.25, 128),
                        qpt([3.0, 4.0], 0.25, 128),
                    ],
                )
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                    ],
                )
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                )
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                    ],
                )
                # NOTE: NNAPI qadd supports broadcast, but PT does not.

    def test_qlinear(self):
        # [FUNCTION DOC] Checks quantized Linear lowering: builds a 32 to 16 module with
        # a quantized int8 weight and float bias, and runs it on a quantized input.
        torch.manual_seed(29)
        weight = qpt(torch.randn(16, 32), 0.125, 0, torch.qint8)
        bias = torch.randn(16)
        mod = torch.ao.nn.quantized.Linear(32, 16)
        mod.set_weight_bias(weight, bias)
        inp = qpt(torch.randn(2, 32), 0.05, 130, torch.quint8)
        self.check(mod, inp)

    def test_seblock_mul(self):
        # [FUNCTION DOC] Checks a squeeze-and-excite style multiply of an NHWC tensor by a
        # per-channel tensor of spatial size one (a broadcast).
        class MulModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the elementwise product of two inputs.
            def forward(self, lhs, rhs):
                # [FUNCTION DOC] HELPER: Return the product of both inputs.
                return lhs * rhs

        self.check(
            MulModel(),
            [
                nhwc(torch.randn(2, 3, 4, 4)),
                torch.randn(1, 3, 1, 1),
            ],
        )

    def test_multi_output(self):
        # [FUNCTION DOC] Checks lowering of a model that returns two outputs (sum and
        # difference of its two inputs).
        class MultiModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a pair: the sum and the difference of
        # its two inputs.
            def forward(self, lhs, rhs) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Return the sum and difference of the inputs as a tuple.
                the_sum = lhs + rhs
                the_diff = lhs - rhs
                return the_sum, the_diff

        self.check(MultiModel(), [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])])


if __name__ == "__main__":
    run_tests()
