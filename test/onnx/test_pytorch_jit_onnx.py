# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests that hand-written TorchScript IR graphs convert to ONNX correctly and match
# the JIT interpreter when run in ONNX Runtime. Provides a helper converting an IR graph to an ONNX
# proto, a mixin class holding the tests, and a factory that builds one concrete test class per
# opset. Import-time side effect: creates the opset 14 test class at module level.
import onnxruntime
import pytorch_test_common
from pytorch_test_common import skipIfNoCuda

import torch
from torch.onnx._internal.torchscript_exporter import verification
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.onnx._internal.torchscript_exporter.utils import (
    _trigger_symbolic_function_registration,
)
from torch.testing._internal import common_utils


def _jit_graph_to_onnx_model(graph, operator_export_type, opset_version):
    # [FUNCTION DOC] HELPER: Convert a JIT graph object to a serialized ONNX model without a module
    # or tensor inputs. Sets the global exporter opset version (mutating global state), triggers
    # symbolic function registration, optimizes the graph with an empty parameter mapping, exports
    # it with default settings and returns the serialized model proto.
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    This function is for testing purpose.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """

    GLOBALS.export_onnx_opset_version = opset_version
    _trigger_symbolic_function_registration()
    graph = torch.onnx.utils._optimize_graph(
        graph, operator_export_type, params_dict={}
    )
    proto, _, _, _ = graph._export_onnx(
        {},
        opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
    return proto


class _TestJITIRToONNX:
    # [CLASS DOC] Abstract mixin (deliberately not a test case; MakeTestCase mixes it into a real
    # one) holding the IR-based tests. Class attributes: opset_version (-1, must be overridden),
    # ORT providers (CPU), shape/dtype checking flags, and ignore_none (true for tracing). The
    # run_test method performs the conversion and comparison.
    """Abstract base class for test cases.

    Intentionally not a sub-class of unittest.TestCase so that unittest / pytest
    don't run it directly. unitest.TestCase is mixed in as another base class when
    creating concrete sub-types. See MakeTestCase().
    """

    opset_version = -1  # Sub-classes must override
    ort_providers = ["CPUExecutionProvider"]
    check_shape = True
    check_dtype = True
    ignore_none = True  # True for tracing, and False for scripting

    def run_test(self, graph_ir, example_inputs, parse_tensor_constants=False):
        # [FUNCTION DOC] Parses IR text into a graph (optionally parsing tensor constants), runs it
        # in the JIT interpreter for reference outputs, converts it to ONNX at the class opset,
        # runs that in ONNX Runtime with the class providers, and compares outputs with relative
        # tolerance 1e-3 and absolute tolerance 1e-7 using the class shape, dtype and none-handling
        # flags. Raises on any mismatch or conversion failure.
        graph = torch._C.parse_ir(graph_ir, parse_tensor_constants)
        jit_outs = torch._C._jit_interpret_graph(graph, example_inputs)

        onnx_proto = _jit_graph_to_onnx_model(
            graph, torch.onnx.OperatorExportTypes.ONNX, self.opset_version
        )
        ort_sess = onnxruntime.InferenceSession(
            onnx_proto, providers=self.ort_providers
        )
        ort_outs = verification._run_onnx(ort_sess, example_inputs)

        options = verification.VerificationOptions(
            rtol=1e-3,
            atol=1e-7,
            check_shape=self.check_shape,
            check_dtype=self.check_dtype,
            ignore_none=self.ignore_none,
            acceptable_error_percentage=None,
        )
        verification._compare_onnx_pytorch_outputs(
            ort_outs,
            jit_outs,
            options,
        )

    def test_example_ir(self):
        # [FUNCTION DOC] Checks the add operator with an alpha constant of one on two 2x3
        # float inputs.
        graph_ir = """
        graph(%1 : Float(2, 3),
              %2 : Float(2, 3)):
          %3 : int = prim::Constant[value=1]()
          %4 : Float(2, 3) = aten::add(%1, %2, %3)
          return (%4)
        """
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        self.run_test(graph_ir, (a, b))

    def test_where_constants(self):
        # [FUNCTION DOC] Checks the where operator with a boolean condition, a float tensor and a
        # double scalar constant zero; tensor constants are parsed from the IR.
        graph_ir = """
        graph(%0 : Bool(8, device=cpu),
              %1 : Float(8, device=cpu)):
          %3 : Double(device=cpu) = prim::Constant[value={0.}]()
          %4 : Float(8) = aten::where(%0, %1, %3)
          return (%4)
        """
        a = torch.zeros(8, dtype=bool)
        b = torch.zeros(8)
        self.run_test(graph_ir, (a, b), parse_tensor_constants=True)

    def test_add_sub_with_graph_inputs(self):
        # [FUNCTION DOC] For each of add, sub and rsub, builds IR where the alpha value is a graph
        # input and checks conversion with alpha 2 on two 2x3 float inputs.
        for op in ["add", "sub", "rsub"]:
            graph_ir = f"""
            graph(%1 : Float(2, 3),
                  %2 : Float(2, 3),
                  %3 : int):
              %4 : Float(2, 3) = aten::{op}(%1, %2, %3)
              return (%4)
            """
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            self.run_test(graph_ir, (a, b, 2))

    def test_native_layer_norm(self):
        # [FUNCTION DOC] Checks native layer norm over the last two dims (3 and 2) with weight,
        # bias and epsilon 1e-5, returning its three outputs (result, mean, inverse std).
        graph_ir = """
        graph(%x : Float(2, 3, 2),
              %w : Float(3, 2),
              %b : Float(3, 2)):
          %5 : int = prim::Constant[value=3]()
          %6 : int = prim::Constant[value=2]()
          %7 : int[] = prim::ListConstruct(%5, %6)
          %10 : float = prim::Constant[value=1.0000000000000001e-05]()
          %11 : Float(2, 3, 2), %12 : Float(2, 1, 1), %13 : Float(2, 1, 1) = aten::native_layer_norm(%x, %7, %w, %b, %10)
          return (%11, %12, %13)
        """
        x = torch.randn(2, 3, 2)
        w = torch.randn(3, 2)
        b = torch.randn(3, 2)
        self.run_test(graph_ir, (x, w, b))

    def test_convolution(self):
        # [FUNCTION DOC] Checks the convolution operator with stride one, no padding, no dilation,
        # no bias, not transposed and one group, on an 8x1x5x5 input and 4x1x3x3 weight.
        graph_ir = """
        graph(%1 : Tensor,
              %2 : Tensor):
          %3 : NoneType = prim::Constant()
          %4 : int[] = prim::Constant[value=[1, 1]]()
          %5 : int[] = prim::Constant[value=[0, 0]]()
          %6 : bool = prim::Constant[value=0]()
          %7 : int = prim::Constant[value=1]()
          %8 : Tensor = aten::convolution(%1, %2, %3, %4, %5, %4, %6, %5, %7)
          return (%8)
        """
        x = torch.randn(8, 1, 5, 5)
        w = torch.randn(4, 1, 3, 3)
        self.run_test(graph_ir, (x, w))

    def test_log_softmax(self):
        # [FUNCTION DOC] Checks log softmax along dim 1 with the half-to-float flag false on a 5x2
        # float input.
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=0]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        x = torch.randn(5, 2)
        self.run_test(graph_ir, (x,))

    @skipIfNoCuda
    def test_log_softmax_half_to_float(self):
        # [FUNCTION DOC] Checks log softmax along dim 1 with the half-to-float flag true on a half
        # precision CUDA input; skipped when CUDA is unavailable.
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=1]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        x = torch.randn(5, 2).half().to("cuda")
        self.run_test(graph_ir, (x,))

    def test_native_dropout(self):
        # [FUNCTION DOC] Checks native dropout with probability zero in training mode (so output is
        # deterministic), returning both the output and the mask, on a 2x3 float input.
        graph_ir = """
        graph(%1 : Float(2, 3)):
          %2 : float = prim::Constant[value=0.0]()
          %training : bool = prim::Constant[value=1]()
          %3 : Tensor, %4 : Tensor = aten::native_dropout(%1, %2, %training)
          return (%3, %4)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))


def MakeTestCase(opset_version: int) -> type:
    # [FUNCTION DOC] Factory for concrete test classes. Returns a new type named for the opset that
    # derives from the shared export test case base and copies the attributes of the mixin class,
    # overriding opset_version with the given value.
    name = f"TestJITIRToONNX_opset{opset_version}"
    return type(
        str(name),
        (pytorch_test_common.ExportTestCase,),
        dict(_TestJITIRToONNX.__dict__, opset_version=opset_version),
    )


TestJITIRToONNX_opset14 = MakeTestCase(14)

if __name__ == "__main__":
    common_utils.run_tests()
