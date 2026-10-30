# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the TorchScript exporter utility functions in the ONNX utils module:
# constant folding of many operators, export modes, local function (module) export with scope
# names, custom operator and autograd function fallthrough, initializer handling and
# deduplication, conv/batch-norm fusion and others. Main contents are a scope-name helper, a base
# test case that converts a model to an ONNX graph, and one test class that is parameterized over
# every supported opset. Import-time effect: the class decorator creates one test class per opset
# from the base opset to the maximum TorchScript exporter opset.

import copy
import io
import re
import warnings

import onnx

import parameterized
import pytorch_test_common
import torchvision
from autograd_helper import CustomFunction as CustomFunction2
from pytorch_test_common import (
    skipIfNoCuda,
    skipIfUnsupportedMaxOpsetVersion,
    skipIfUnsupportedMinOpsetVersion,
)

import torch
import torch.onnx
import torch.utils.cpp_extension
from torch.onnx import _constants, OperatorExportTypes, TrainingMode, utils
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.onnx.symbolic_helper import _unpack_list, parse_args
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


def _remove_test_environment_prefix_from_scope_name(scope_name: str) -> str:
    # [FUNCTION DOC] HELPER: Normalizes a module scope name by removing the test-environment
    # prefixes (the test module name and the main module name), each followed by a dot and
    # optionally a locals marker segment, using regular-expression substitution on the string.
    # Returns the shortened scope name such as a bare class name.
    """Remove test environment prefix added to module.

    Remove prefix to normalize scope names, since different test environments add
    prefixes with slight differences.

    Example:

        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "test_utility_funs.M"
        >>> )
        "M"
        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "test_utility_funs.test_abc.<locals>.M"
        >>> )
        "M"
        >>> _remove_test_environment_prefix_from_scope_name(
        >>>     "__main__.M"
        >>> )
        "M"
    """
    prefixes_to_remove = ["test_utility_funs", "__main__"]
    for prefix in prefixes_to_remove:
        scope_name = re.sub(f"{prefix}\\.(.*?<locals>\\.)?", "", scope_name)
    return scope_name


class _BaseTestCase(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] HELPER: Base class for the utility function tests (based on the export test case
    # that cleans up after each test). Provides a wrapper that converts a model to a TorchScript
    # graph with the exporter internals.
    def _model_to_graph(
        self,
        model,
        input,
        do_constant_folding=True,
        training=TrainingMode.EVAL,
        operator_export_type=OperatorExportTypes.ONNX,
        input_names=None,
        dynamic_axes=None,
    ):
        # [FUNCTION DOC] HELPER: Converts a model and its input to an ONNX graph. Sets up the
        # trace module map, puts the model in train or eval mode when the training argument is one
        # of those (otherwise leaves the mode alone), validates the dynamic axes against the given
        # input names, then calls the exporter internal conversion with constant propagation of
        # torch disabled. Returns a tuple of graph, parameter dictionary and torch output. Mutates
        # the model mode.
        torch.onnx.utils._setup_trace_module_map(model, False)
        if training == torch.onnx.TrainingMode.TRAINING:
            model.train()
        elif training == torch.onnx.TrainingMode.EVAL:
            model.eval()
        utils._validate_dynamic_axes(dynamic_axes, model, None, None)
        graph, params_dict, torch_out = utils._model_to_graph(
            model,
            input,
            do_constant_folding=do_constant_folding,
            _disable_torch_constant_prop=True,
            operator_export_type=operator_export_type,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
        )
        return graph, params_dict, torch_out


@parameterized.parameterized_class(
    [
        {"opset_version": opset}
        for opset in range(
            _constants.ONNX_BASE_OPSET,
            _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET + 1,
        )
    ],
    class_name_func=lambda cls,
    num,
    params_dict: f"{cls.__name__}_opset_{params_dict['opset_version']}",
)
class TestUtilityFuns(_BaseTestCase):
    # [CLASS DOC] Tests of exporter utilities, one copy per opset version (the class attribute for
    # the opset is overwritten by the parameterization). Most tests set the global export opset
    # and operator export type, convert a small model to a graph and assert on the kinds and
    # number of nodes after constant folding or on scope names and attributes.
    opset_version = None

    def test_is_in_onnx_export(self):
        # [FUNCTION DOC] Exports a module whose forward asserts that the in-export flag is true
        # and then raises a value error. The export is expected to fail with that error, after
        # which the test asserts the flag has been reset to false.
        test_self = self

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward checks the in-export flag through the
            # enclosing test and then raises a value error (the final return is unreachable).
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Asserts the in-export flag is set, then raises
                # a value error.
                test_self.assertTrue(torch.onnx.is_in_onnx_export())
                raise ValueError
                return x + 1

        x = torch.randn(3, 4)
        f = io.BytesIO()
        try:
            torch.onnx.export(
                MyModule(), x, f, opset_version=self.opset_version, dynamo=False
            )
        except ValueError:
            self.assertFalse(torch.onnx.is_in_onnx_export())

    def test_validate_dynamic_axes_invalid_input_output_name(self):
        # [FUNCTION DOC] Calls the dynamic axes validation with two valid keys and two unknown
        # keys, capturing all warnings. Expects exactly two warnings, one for each unknown key,
        # with the text saying the key is not a valid input or output name.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils._validate_dynamic_axes(
                {"input1": {}, "output": {}, "invalid_name1": {}, "invalid_name2": {}},
                None,
                ["input1", "input2"],
                ["output"],
            )
            messages = [str(warning.message) for warning in w]
        self.assertIn(
            "Provided key invalid_name1 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertIn(
            "Provided key invalid_name2 for dynamic axes is not a valid input/output name",
            messages,
        )
        self.assertEqual(len(messages), 2)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_to_slice(self):
        # [FUNCTION DOC] Splits a tensor with sizes taken from the sizes of two inputs,
        # converts to a graph with dynamic axes and asserts that no split-to-sequence node
        # remains (the split is lowered to slices). Sets the global opset and export type.
        # Needs opset 11 or newer.
        class SplitModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module splitting its third input along dimension one into pieces
            # whose sizes come from the second dimension of the first two inputs.
            def forward(self, x, y, t):
                # [FUNCTION DOC] HELPER: Returns the two pieces of splitting t by the sizes of
                # x and y.
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        graph, _, _ = self._model_to_graph(
            SplitModule(),
            (x, y, t),
            input_names=["x", "y", "t"],
            dynamic_axes={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::SplitToSequence")

    def test_constant_fold_transpose(self):
        # [FUNCTION DOC] Constant folding of a transpose applied to a constant tensor. Asserts
        # that the graph has no transpose or cast node and has exactly two nodes.
        class TransposeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that transposes a constant 2x3 tensor and adds the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the transposed constant plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.transpose(a, 1, 0)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(3, 2)
        graph, _, __ = self._model_to_graph(
            TransposeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    @skipIfUnsupportedMaxOpsetVersion(17)
    def test_constant_fold_reduceL2(self):
        # [FUNCTION DOC] Constant folding of a L2 norm over a dimension of a constant tensor.
        # Asserts no ReduceL2 node remains. Only runs for opsets up to 17.
        class ReduceModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that takes the L2 norm of a constant 2x3 tensor over
            # dimension minus two and adds the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the folded norm plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=2, dim=-2, keepdim=False)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ReduceModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL2")

    @skipIfUnsupportedMaxOpsetVersion(17)
    def test_constant_fold_reduceL1(self):
        # [FUNCTION DOC] Constant folding of a L1 norm of a constant tensor. Asserts no ReduceL1
        # node remains. Only runs for opsets up to 17.
        class NormModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that takes the L1 norm of a constant 2x3 tensor over
            # dimension minus two and adds the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the norm of the constant plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.norm(a, p=1, dim=-2)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            NormModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::ReduceL1")

    def test_constant_fold_slice(self):
        # [FUNCTION DOC] Constant folding of a narrow (slice) of a constant tensor. Asserts no
        # slice or cast node remains and that exactly two nodes are left.
        class NarrowModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that narrows a constant 2x3 tensor to its first row and
            # adds the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the narrowed constant plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.narrow(a, 0, 0, 1)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            NarrowModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_index_exceeds_dim(self):
        # [FUNCTION DOC] Constant folding of a slice whose end index is larger than the dimension.
        # Asserts no slice or cast node remains and exactly two nodes are left.
        class SliceIndexExceedsDimModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that slices rows one to ten of a constant with two rows
            # and adds the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the over-long slice of the constant plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[1:10]  # index exceeds dimension
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceIndexExceedsDimModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_slice_negative_index(self):
        # [FUNCTION DOC] Constant folding of slices and selects using negative indices and
        # dimensions of a constant tensor. Asserts no slice or cast node remains in the graph.
        class SliceNegativeIndexModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a slice ending at minus one and two selects with
            # negative or positive dimension and index on a constant 2x3 tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple: the sliced constant plus x, and the sum
                # of two selected columns.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = a[0:-1]  # index relative to the end
                c = torch.select(a, dim=-1, index=-2)
                d = torch.select(a, dim=1, index=0)
                return b + x, c + d

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            SliceNegativeIndexModule(),
            (x,),
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Cast")

    def test_constant_fold_gather(self):
        # [FUNCTION DOC] Constant folding of a select with a negative index and an index select on
        # a constant tensor. Asserts no gather node remains.
        class GatherModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module using select and index select on a constant 2x3 tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple: the selected column plus one, and the
                # index-selected rows plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.select(a, dim=1, index=-2)
                c = torch.index_select(a, dim=-2, index=torch.tensor([0, 1]))
                return b + 1, c + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 3)
        graph, _, __ = self._model_to_graph(
            GatherModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Gather")

    def test_constant_fold_unsqueeze(self):
        # [FUNCTION DOC] Constant folding of unsqueeze at a negative dimension of a constant
        # tensor. Asserts no unsqueeze or cast node and exactly two nodes remain.
        class UnsqueezeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that unsqueezes a constant 2x3 tensor at dimension minus
            # two and adds the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the unsqueezed constant plus x.
                a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = torch.unsqueeze(a, -2)
                return b + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(1, 2, 3)
        graph, _, __ = self._model_to_graph(
            UnsqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1, 2]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_unsqueeze_multi_axies(self):
        # [FUNCTION DOC] Exports a module with a PReLU on a six-dimensional input plus a random
        # constant of the same shape. Asserts no unsqueeze or cast node remains (the multi-axis
        # unsqueeze of the slope is folded) and exactly five nodes are left.
        class PReluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a PReLU layer; forward adds a random constant of
            # shape 2x3x4x5x8x7 to the activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PReLU submodule.
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the PReLU of the input plus a freshly
                # drawn random tensor.
                a = torch.randn(2, 3, 4, 5, 8, 7)
                return self.prelu(x) + a

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(2, 3, 4, 5, 8, 7)
        graph, _, __ = self._model_to_graph(
            PReluModel(), x, input_names=["x"], dynamic_axes={"x": [0, 1, 2, 3, 4, 5]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 5)

    def test_constant_fold_squeeze_without_axes(self):
        # [FUNCTION DOC] Constant folding of squeeze without dimension arguments of a
        # constant tensor that is used twice. Asserts no squeeze or cast node and exactly
        # four nodes remain.
        class SqueezeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squeezing a constant 1x2x3 tensor (twice) and combining
            # with the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the squeezed constant plus x plus
                # the squeezed constant.
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a) + x + torch.squeeze(a)

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_squeeze_with_axes(self):
        # [FUNCTION DOC] Constant folding of squeeze with a dimension argument on a constant
        # tensor. Asserts no squeeze or cast node and exactly two nodes remain.
        class SqueezeAxesModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module squeezing dimension minus three of a constant 1x2x3
            # tensor and adding the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the squeezed constant plus x.
                a = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
                return torch.squeeze(a, dim=-3) + x

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            SqueezeAxesModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Squeeze")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_concat(self):
        # [FUNCTION DOC] Constant folding of a concatenation of two constant tensors. Asserts no
        # concat or cast node and exactly two nodes remain.
        class ConcatModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating two constant 1x3 float tensors, adding one
            # to the concatenation and adding the result to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds two constants (explicitly cast to float, see the
                # comment in the body), concatenates them along dimension zero, adds the second
                # constant and the input.
                # Why did I insert a Cast here?  There appears to be intentional
                # behavior in ONNX constant folding where constant tensors which
                # are not attached to any known to be foldable onnx
                # operations don't get extracted into the initializer graph.  So
                # without these casts, we will actually fail to pull out one of
                # the constants, thus failing constant folding.  I think the
                # test is wrong but I don't have time to write a more correct
                # test (I think the right way to go about the test is to setup
                # a predicate for what invariant graphs should hold after
                # constant folding, and then verify this predicate holds.
                # I think the asserts below are an attempt at this predicate,
                # but it is not right!)
                #
                # More commentary at
                # https://github.com/pytorch/pytorch/pull/18698/files#r340107552
                a = torch.tensor([[1.0, 2.0, 3.0]]).to(torch.float)
                b = torch.tensor([[4.0, 5.0, 6.0]]).to(torch.float)
                c = torch.cat((a, b), 0)
                d = b + c
                return x + d

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.ones(2, 3)
        graph, _, __ = self._model_to_graph(
            ConcatModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Concat")
            self.assertNotEqual(node.kind(), "onnx::Cast")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_lstm(self):
        # [FUNCTION DOC] Exports a single-layer unidirectional GRU (7 input, 3 hidden features)
        # with initial state and asserts that no slice, concat or unsqueeze node is left after
        # folding. The total node count is 3 for opsets up to 12 and 4 for later opsets.
        # KNOWN ISSUE: the test name says lstm but the model is a GRU.
        class GruNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a one-layer unidirectional GRU, input size 7 and
            # hidden size 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the GRU submodule.
                super().__init__()
                self.mygru = torch.nn.GRU(7, 3, 1, bidirectional=False)

            def forward(self, input, initial_state):
                # [FUNCTION DOC] HELPER: Returns the result of running the GRU on the input
                # and initial state.
                return self.mygru(input, initial_state)

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        input = torch.randn(5, 3, 7)
        h0 = torch.randn(1, 3, 3)
        graph, _, __ = self._model_to_graph(
            GruNet(),
            (input, h0),
            input_names=["input", "h0"],
            dynamic_axes={"input": [0, 1, 2], "h0": [0, 1, 2]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Slice")
            self.assertNotEqual(node.kind(), "onnx::Concat")
            self.assertNotEqual(node.kind(), "onnx::Unsqueeze")

        if self.opset_version <= 12:
            self.assertEqual(len(list(graph.nodes())), 3)
        else:
            # Unsqueeze op parameter "axes" as an input instead of as an attribute when opset version >= 13
            self.assertEqual(len(list(graph.nodes())), 4)

    def test_constant_fold_transpose_matmul(self):
        # [FUNCTION DOC] Constant folding of a transpose of a parameter feeding a matrix product.
        # Asserts no transpose node and exactly one node (the matrix product) remains.
        class MatMulNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 5x3 ones parameter that multiplies the input by
            # the transposed parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 5x3 parameter of ones.
                super().__init__()
                self.B = torch.nn.Parameter(torch.ones(5, 3))

            def forward(self, A):
                # [FUNCTION DOC] HELPER: Returns the matrix product of A with
                # the transposed parameter.
                return torch.matmul(A, torch.transpose(self.B, -1, -2))

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        A = torch.randn(2, 3)
        graph, _, __ = self._model_to_graph(
            MatMulNet(), (A,), input_names=["A"], dynamic_axes={"A": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Transpose")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_reshape(self):
        # [FUNCTION DOC] Constant folding of a reshape of a buffer. Asserts no reshape node and
        # exactly one node remains.
        class ReshapeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones that is reshaped to four
            # dims and multiplied with the input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times the reshaped buffer.
                b = self.weight.reshape(1, -1, 1, 1)
                return x * b

        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        x = torch.randn(4, 5)
        graph, _, __ = self._model_to_graph(
            ReshapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Reshape")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_div(self):
        # [FUNCTION DOC] Constant folding of a division of a buffer by a constant integer tensor.
        # Asserts no division node and exactly one node remains.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones that is divided by a
            # constant tensor and multiplied with the input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the buffer divided by the constant
                # tensor, times x.
                div = self.weight.div(torch.tensor([1, 2, 3, 4, 5]))
                return div * x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Div")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_mul(self):
        # [FUNCTION DOC] Constant folding of a multiplication of a buffer and a constant tensor.
        # Asserts no multiplication node and exactly one node remains.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones, multiplied by a constant
            # tensor and divided by the input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the buffer times the constant tensor,
                # divided by x.
                mul = self.weight.mul(torch.tensor([1, 2, 3, 4, 5]))
                return mul / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Mul")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_add(self):
        # [FUNCTION DOC] Constant folding of an addition of a buffer and a constant tensor.
        # Asserts no addition node and exactly one node remains, that exactly one parameter is
        # left, and that it holds the folded sum (2 to 6).
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones, added to a constant tensor
            # and reduced by the input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the buffer plus the constant, minus x.
                add = self.weight + torch.tensor([1, 2, 3, 4, 5])
                return add - x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        for node in graph.nodes():
            self.assertTrue(node.kind() != "onnx::Add")
        self.assertEqual(len(list(graph.nodes())), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        self.assertEqual(weight, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))

    def test_constant_fold_sub(self):
        # [FUNCTION DOC] Constant folding of a subtraction of a constant tensor from a buffer.
        # Asserts no subtraction node and exactly one node remains, that exactly one parameter is
        # left and that it holds the folded difference (0 down to minus 4).
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones from which a constant
            # tensor is subtracted, then the input is added.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the buffer minus the constant, plus x.
                sub = self.weight - torch.tensor([1, 2, 3, 4, 5])
                return sub + x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, params_dict, __ = self._model_to_graph(
            Module(),
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sub")
        self.assertEqual(len(list(graph.nodes())), 1)
        params = list(params_dict.values())
        self.assertEqual(len(params), 1)
        weight = params[0]
        self.assertEqual(weight, torch.tensor([0.0, -1.0, -2.0, -3.0, -4.0]))

    def test_constant_fold_sqrt(self):
        # [FUNCTION DOC] Constant folding of a square root of a buffer. Asserts no sqrt node and
        # exactly one node remains.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones whose square root is
            # divided by the input.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the square root of the buffer divided by x.
                sqrt = torch.sqrt(self.weight)
                return sqrt / x

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            Module(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Sqrt")
        self.assertEqual(len(list(graph.nodes())), 1)

    def test_constant_fold_shape(self):
        # [FUNCTION DOC] Constant folding of the size of a buffer. Asserts no shape node remains
        # and exactly two nodes are left.
        class ShapeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a buffer of five ones; forward adds the buffer
            # length to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the buffer of five ones.
                super().__init__()
                self.weight = torch.nn.Buffer(torch.ones(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus the first dimension size of the buffer.
                shape = self.weight.shape[0]
                return x + shape

        x = torch.randn(2, 5)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            ShapeModule(), (x,), input_names=["x"], dynamic_axes={"x": [0, 1]}
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::Shape")
        self.assertEqual(len(list(graph.nodes())), 2)

    def test_constant_fold_upsample_scale_fold_as_constant(self):
        # [FUNCTION DOC] Exports a bilinear upsample module with a constant scale factor (using
        # the legacy exporter) and asserts the saved model has no initializers, because the scale
        # is a constant and not a parameter.
        # upsample scale is a constant, not a model parameter,
        # therefore should not be added as initializer after constant folding.
        model = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.randn(1, 32, 224, 224)
        f = io.BytesIO()
        torch.onnx.export(model, x, f, dynamo=False)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(len(onnx_model.graph.initializer), 0)

    def test_verbose(self):
        # [FUNCTION DOC] Checks the verbose export flag. A nested helper exports a simple module
        # and tests whether the saved model equals a copy with all doc strings stripped. Asserts
        # that the default export (no verbose flag) is already stripped, and that exporting with
        # verbose true leaves doc strings in the model.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the exponential of the input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the exponential of the input.
                return torch.exp(input)

        x = torch.randn(3, 4)

        def is_model_stripped(f, verbose=None):
            # [FUNCTION DOC] HELPER: Exports the module (closing over the shared 3x4 input) to the
            # given in-memory file, passing the verbose flag only if it is not None. Loads the
            # saved model, copies it, strips doc strings from the copy, and returns True when the
            # two are equal (meaning the exported model had no doc strings).
            if verbose is None:
                torch.onnx.export(
                    MyModule(), x, f, opset_version=self.opset_version, dynamo=False
                )
            else:
                torch.onnx.export(
                    MyModule(),
                    x,
                    f,
                    verbose=verbose,
                    opset_version=self.opset_version,
                    dynamo=False,
                )
            model = onnx.load(io.BytesIO(f.getvalue()))
            model_strip = copy.copy(model)
            onnx.helper.strip_doc_string(model_strip)
            return model == model_strip

        # test verbose=False (default)
        self.assertTrue(is_model_stripped(io.BytesIO()))
        # test verbose=True
        self.assertFalse(is_model_stripped(io.BytesIO(), True))

    # NB: remove this test once DataParallel can be correctly handled
    def test_error_on_data_parallel(self):
        # [FUNCTION DOC] Wraps a reflection pad module in data parallel and asserts the export
        # raises a value error whose message says that data parallel is not supported and suggests
        # using the attribute module to unwrap it. The comment above says to remove the test once
        # data parallel is handled.
        model = torch.nn.DataParallel(torch.nn.ReflectionPad2d((1, 2, 3, 4)))
        x = torch.randn(1, 2, 3, 4)
        f = io.BytesIO()
        with self.assertRaisesRegex(
            ValueError,
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try ",
        ):
            torch.onnx.export(
                model, x, f, opset_version=self.opset_version, dynamo=False
            )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_sequence_dim(self):
        # [FUNCTION DOC] Exports a scripted module that returns a list of its two inputs
        # (scripting keeps the sequence output). With a dynamic second axis on one input, the
        # first graph output must be a float tensor sequence with shape 2 by unknown; without
        # dynamic axes the shape is 2 by 3. Needs opset 11 or newer.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning its two inputs in a list.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a list holding x and y.
                return [x, y]

        model = Module()
        # Export with scripting to keep output as Sequence type.
        # Tracing unpacks the list.
        script_model = torch.jit.script(model)
        x = torch.randn(2, 3)

        # Case 1: dynamic axis
        f = io.BytesIO()
        y = torch.randn(2, 3)
        torch.onnx.export(
            script_model,
            (x, y),
            f,
            opset_version=self.opset_version,
            input_names=["x", "y"],
            dynamic_axes={"y": [1]},
            dynamo=False,
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        loop_output_value_info_proto = onnx_model.graph.output[0]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, None]
        )
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

        # Case 2: no dynamic axes.
        f = io.BytesIO()
        y = torch.randn(2, 3)
        torch.onnx.export(
            script_model, (x, y), f, opset_version=self.opset_version, dynamo=False
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        loop_output_value_info_proto = onnx_model.graph.output[0]
        ref_value_info_proto = onnx.helper.make_tensor_sequence_value_info(
            loop_output_value_info_proto.name, 1, [2, 3]
        )
        self.assertEqual(loop_output_value_info_proto, ref_value_info_proto)

    def test_export_mode(self):
        # [FUNCTION DOC] Exports an eval-mode module with the training export mode set to
        # training, and then a train-mode module with the export mode set to eval. After each
        # export asserts the module training flag is the same as before the export.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the input plus one.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                y = x + 1
                return y

        model = MyModule()
        x = torch.randn(10, 3, 128, 128)
        f = io.BytesIO()

        # set mode to in inference mode and export in training mode
        model.eval()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamo=False,
        )
        # verify that the model state is preserved
        self.assertEqual(model.training, old_state)

        # set mode to training mode and export in inference mode
        model.train()
        old_state = model.training
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
        # verify that the model state is preserved
        self.assertEqual(model.training, old_state)

    def test_export_does_not_fail_on_frozen_scripted_module(self):
        # [FUNCTION DOC] Builds an outer module wrapping a scripted inner module with a data
        # dependent branch, traces the outer module in eval mode, freezes the traced module and
        # exports it. Passes if the export does not raise; there are no other assertions.
        class Inner(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the input if it is positive and
            # its square otherwise.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x when x is greater than zero, else x squared.
                if x > 0:
                    return x
                else:
                    return x * x

        class Outer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that holds a scripted copy of the inner module and
            # forwards to it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the attribute holding the scripted inner module.
                super().__init__()
                self.inner = torch.jit.script(Inner())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the inner scripted module applied to x.
                return self.inner(x)

        x = torch.zeros(1)
        # Freezing is only implemented in eval mode. So we need to call eval()
        outer_module = Outer().eval()
        module = torch.jit.trace_module(outer_module, {"forward": (x)})
        # jit.freeze removes the training attribute in the module
        module = torch.jit.freeze(module)

        torch.onnx.export(
            module, (x,), io.BytesIO(), opset_version=self.opset_version, dynamo=False
        )

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function(self):
        # [FUNCTION DOC] Exports a module with CELU activations, layer norms and a nested dropout
        # module with the option to export modules as local functions. With a set of three module
        # types, asserts one function each for CELU and layer norm with their module domains and
        # three attributes, two CELU nodes and three layer norm nodes in the graph (dropout is
        # absent in inference mode). With only CELU asserts one function; with an empty set
        # asserts no functions; with the all-modules option asserts three functions. Needs opset
        # 15 or newer.
        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a dropout layer with the given probability.
            def __init__(self, prob):
                # [FUNCTION DOC] HELPER: Creates the dropout layer.
                super().__init__()
                self.dropout = torch.nn.Dropout(prob)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the dropout of the input.
                return self.dropout(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a list of layer norms whose eps differ, two CELU
            # layers with different alpha and a nested dropout holder module.
            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Creates the layer norm list (one per layer), two CELU
                # activations and the nested dropout module; stores the layer count.
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)]
                )
                self.celu1 = torch.nn.CELU(1.0)
                self.celu2 = torch.nn.CELU(2.0)
                self.dropout = N(0.5)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Applies the two CELUs to x and y, passes z through every
                # layer norm in turn, and returns the sum of the CELU results and the dropout of
                # the normalized z.
                res1 = self.celu1(x)
                res2 = self.celu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.dropout(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        # Export specified modules. Test against specifying modules that won't
        # exist in the exported model.
        # Model export in inference mode will remove dropout node,
        # thus the dropout module no longer exist in graph.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={
                torch.nn.CELU,
                torch.nn.Dropout,
                torch.nn.LayerNorm,
            },
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))

        # Check function definition
        funcs = onnx_model.functions
        celu_funcs = [f for f in funcs if f.name == "CELU"]
        self.assertEqual(len(celu_funcs), 1)
        self.assertEqual(celu_funcs[0].domain, "torch.nn.modules.activation")
        self.assertEqual(len(celu_funcs[0].attribute), 3)
        ln_funcs = [f for f in funcs if f.name == "LayerNorm"]
        self.assertEqual(len(ln_funcs), 1)
        self.assertEqual(ln_funcs[0].domain, "torch.nn.modules.normalization")
        self.assertEqual(len(ln_funcs[0].attribute), 3)

        # Check local function nodes
        nodes = onnx_model.graph.node
        celu_ns = [n for n in nodes if n.op_type == "CELU"]
        ln_ns = [n for n in nodes if n.op_type == "LayerNorm"]
        self.assertEqual(len(celu_ns), 2)
        self.assertEqual(celu_ns[0].domain, "torch.nn.modules.activation")
        self.assertEqual(len(celu_ns[0].attribute), 3)
        self.assertEqual(len(ln_ns), 3)
        self.assertEqual(ln_ns[0].domain, "torch.nn.modules.normalization")
        self.assertEqual(len(ln_ns[0].attribute), 3)

        # Export specified modules.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={torch.nn.CELU},
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "CELU")

        # Export with empty specified modules. Normal export.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions=set(),
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 0)

        # Export all modules. Should contain {M, CELU, LayerNorm}.
        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions=True,
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 3)

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_overloads(self):
        # [FUNCTION DOC] Exports a module that calls one submodule three times with one, two and
        # three arguments, exporting that submodule type as a function. Asserts three functions
        # exist, named after the class and with numeric suffixes for the overloads. Needs opset
        # 15 or newer.
        class NWithOverloads(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with optional second and third arguments, returning
            # different results (one, two tuple elements) depending on which are provided.
            def forward(self, x, y=None, z=None):
                # [FUNCTION DOC] HELPER: Returns x plus one when y is missing, x plus y when z is
                # missing, otherwise a pair of x plus y and x plus z.
                if y is None:
                    return x + 1
                elif z is None:
                    return x + y
                else:
                    return x + y, x + z

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding one overloaded submodule and calling it with one,
            # two and three arguments.
            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Creates the overloaded submodule; the layer
                # count is unused.
                super().__init__()
                self.n = NWithOverloads()

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns a tuple of the three calls of the submodule.
                return self.n(x), self.n(x, y), self.n(x, y, z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        f = io.BytesIO()
        torch.onnx.export(
            M(3),
            (x, y, z),
            f,
            opset_version=self.opset_version,
            export_modules_as_functions={NWithOverloads},
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertEqual(len(funcs), 3)
        func_names = [f.name for f in funcs]
        self.assertIn("NWithOverloads", func_names)
        self.assertIn("NWithOverloads.1", func_names)
        self.assertIn("NWithOverloads.2", func_names)

    # Failing after ONNX 1.13.0
    @skipIfUnsupportedMaxOpsetVersion(1)
    def test_local_function_infer_scopes(self):
        # [FUNCTION DOC] Exports a module whose size arithmetic creates scalars without scope,
        # with all modules as functions and no constant folding, then asserts a function named
        # after the module exists.
        # KNOWN ISSUE: the decorator only allows opsets up to 1, so this test is always skipped
        # for the supported opsets (the comment above notes it fails after ONNX 1.13.0).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that reshapes the input by appending two size-one dims
            # and an inferred dim to the leading sizes.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input viewed with a shape built by
                # concatenating its leading sizes with scalar entries.
                # Concatenation of scalars inserts unscoped tensors in IR graph.
                new_tensor_shape = x.size()[:-1] + (1, 1, -1)
                tensor = x.view(*new_tensor_shape)
                return tensor

        x = torch.randn(4, 5)
        f = io.BytesIO()
        torch.onnx.export(
            M(),
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            do_constant_folding=False,
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        self.assertIn("M", [f.name for f in funcs])

    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_predefined_attributes(self):
        # [FUNCTION DOC] Exports a module that has an annotated integer attribute and layer norm
        # children, with modules as functions. Asserts the function of the module lists its
        # attribute names (only the layer count), the layer norm function lists eps and
        # elementwise affine, the model node for the module carries the layer count attribute
        # value, and the layer norm nodes inside the function carry attributes among the expected
        # eps and elementwise affine values. Needs opset 15 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an annotated layer count attribute and a list
            # of layer norms.
            num_layers: int

            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Stores the layer count and creates that many layer norms
                # over 3 features with a small epsilon.
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies all layer norms in sequence to x and
                # returns the result.
                for ln in self.lns:
                    x = ln(x)
                return x

        x = torch.randn(2, 3)
        f = io.BytesIO()
        model = M(3)
        torch.onnx.export(
            model,
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            dynamo=False,
        )

        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        funcs = onnx_model.functions
        m_funcs = [fn for fn in funcs if fn.name == "M"]
        self.assertEqual(m_funcs[0].attribute, ["num_layers"])
        ln_funcs = [fn for fn in funcs if fn.name == "LayerNorm"]
        self.assertEqual(ln_funcs[0].attribute, ["eps", "elementwise_affine"])

        from onnx import helper

        m_node = [n for n in onnx_model.graph.node if n.op_type == "M"]
        self.assertEqual(
            m_node[0].attribute[0],
            helper.make_attribute("num_layers", model.num_layers),
        )

        ln_nodes = [n for n in m_funcs[0].node if n.op_type == "LayerNorm"]
        expected_ln_attrs = [
            helper.make_attribute(
                "elementwise_affine", model.lns[0].elementwise_affine
            ),
            helper.make_attribute("eps", model.lns[0].eps),
        ]
        for ln_node in ln_nodes:
            self.assertIn(ln_node.attribute[0], expected_ln_attrs)
            self.assertIn(ln_node.attribute[1], expected_ln_attrs)

    # This test cases checks the issue where an object does not have an attribute.
    # When enabling `export_modules_as_functions = True`, the exporter could return an
    # AttributeError. With this test case, we check that the export passes successfully
    # without any AttributeError exceptions.
    # See https://github.com/pytorch/pytorch/pull/109759 for an example. The exception that
    # this test tries to avoid is `AttributeError: 'Embedding' object has no attribute 'freeze'`.
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_local_function_subset_of_predefined_attributes(self):
        # [FUNCTION DOC] Regression test for an attribute error raised when exporting modules as
        # functions with a module lacking some predefined attributes (an embedding layer).
        # Exports verbosely and passes when no exception is raised; no assertions. Needs opset
        # 15 or newer.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a pretrained embedding layer and a list
            # of layer norms.
            num_layers: int

            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Creates the embedding layer from a fixed 2x3 table,
                # stores the layer count and creates the layer norms.
                super().__init__()
                self.embed_layer = torch.nn.Embedding.from_pretrained(
                    torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
                )
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=1e-4) for _ in range(num_layers)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Looks up an embedding for a fixed index, applies
                # all layer norms to x in sequence and returns the pair of normalized x
                # and the embedding.
                e = self.embed_layer(torch.LongTensor([1]))
                for ln in self.lns:
                    x = ln(x)
                return x, e

        x = torch.randn(2, 3)
        f = io.BytesIO()
        model = M(3)
        torch.onnx.export(
            model,
            (x,),
            f,
            export_modules_as_functions=True,
            opset_version=self.opset_version,
            verbose=True,  # Allows the test case to print `Skipping module attribute 'freeze'`
            dynamo=False,
        )

    def test_node_scope(self):
        # [FUNCTION DOC] Converts a module with GELUs, layer norms and a nested relu module to a
        # graph, once traced and once scripted, and asserts that every node scope name, after
        # removing test prefixes, is in an expected set of scope names (module type and
        # attribute path strings).
        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ReLU layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ReLU layer.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the ReLU of the input.
                return self.relu(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two GELU layers, a list of layer norms and a nested
            # ReLU holder module.
            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Stores the layer count and creates the layer norms, two
                # GELUs and the nested ReLU holder.
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=float(i)) for i in range(num_layers)]
                )
                self.gelu1 = torch.nn.GELU()
                self.gelu2 = torch.nn.GELU()
                self.relu = N()

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Applies the GELUs to x and y, the layer norms to z in
                # turn, and returns the sum of GELU results with the ReLU of the normalized z.
                res1 = self.gelu1(x)
                res2 = self.gelu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.relu(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        model = M(3)
        expected_scope_names = {
            "M::/torch.nn.modules.activation.GELU::gelu1",
            "M::/torch.nn.modules.activation.GELU::gelu2",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.0",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.1",
            "M::/torch.nn.modules.normalization.LayerNorm::lns.2",
            "M::/N::relu/torch.nn.modules.activation.ReLU::relu",
            "M::",
        }

        graph, _, _ = self._model_to_graph(
            model, (x, y, z), input_names=[], dynamic_axes={}
        )
        for node in graph.nodes():
            self.assertIn(
                _remove_test_environment_prefix_from_scope_name(node.scopeName()),
                expected_scope_names,
            )

        graph, _, _ = self._model_to_graph(
            torch.jit.script(model), (x, y, z), input_names=[], dynamic_axes={}
        )
        for node in graph.nodes():
            self.assertIn(
                _remove_test_environment_prefix_from_scope_name(node.scopeName()),
                expected_scope_names,
            )

    def test_scope_of_constants_when_combined_by_cse_pass(self):
        # [FUNCTION DOC] Converts a stack of three layers that add the same constant tensor, and
        # checks the scope names of all constant nodes: because of duplication after scalar type
        # analysis, there are three constants whose scopes are the three layers under the root.
        # The comment in the test says the expectation changes if CSE improves.
        layer_num = 3

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Layer that adds a stored constant to its input.
            def __init__(self, constant):
                # [FUNCTION DOC] HELPER: Stores the constant.
                super().__init__()
                self.constant = constant

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus the stored constant.
                # 'self.constant' is designed to be the same for all layers,
                # hence it is common sub expression.
                return x + self.constant

        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a list of the layers sharing an equal constant.
            def __init__(self, layers: int = layer_num):
                # [FUNCTION DOC] HELPER: Creates the requested number of layers, each with a
                # tensor constant 1.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [M(constant=torch.tensor(1.0)) for i in range(layers)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes x through all layers in sequence.
                for layer in self.layers:
                    x = layer(x)
                return x

        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )

        # NOTE: Duplicated constants are populated due to implicit casting in scalar_type_analysis,
        #       so we expect 3 constants with different scopes. The 3 constants are for the 3 layers.
        #       If CSE in exporter is improved later, this test needs to be updated.
        #       It should expect 1 constant, with same scope as root.
        expected_root_scope_name = "N::"
        expected_layer_scope_name = "M::layers"
        expected_constant_scope_name = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        constant_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Constant":
                constant_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
        self.assertEqual(constant_scope_names, expected_constant_scope_name)

    def test_scope_of_nodes_when_combined_by_cse_pass(self):
        # [FUNCTION DOC] Converts three parallel layers computing the sum with a shared constant
        # then multiplying by a per-layer bias. Asserts the common add is combined and keeps only
        # the scope of the first layer, while the three multiplications keep the scopes of layers
        # zero to two.
        layer_num = 3

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Layer computing the sum of the input and a constant
            # times a bias.
            def __init__(self, constant, bias):
                # [FUNCTION DOC] HELPER: Stores the constant and bias.
                super().__init__()
                self.constant = constant
                self.bias = bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the product of the sum of x and the constant
                # with the bias.
                # 'constant' and 'x' is designed to be the same for all layers,
                # hence `x + self.constant` is common sub expression.
                # 'bias' is designed to be different for all layers,
                # hence `* self.bias` is not common sub expression.
                return (x + self.constant) * self.bias

        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding parallel layers with the same constant
            # but random biases.
            def __init__(self, layers: int = layer_num):
                # [FUNCTION DOC] HELPER: Creates the requested number of layers, each with the
                # same constant and its own random bias.
                super().__init__()

                self.layers = torch.nn.ModuleList(
                    [
                        M(constant=torch.tensor([1.0]), bias=torch.randn(1))
                        for i in range(layers)
                    ]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer to x separately and returns the first
                # three results as a tuple.
                y = []
                for layer in self.layers:
                    y.append(layer(x))
                return y[0], y[1], y[2]

        graph, _, _ = self._model_to_graph(
            N(), (torch.randn(2, 3)), input_names=[], dynamic_axes={}
        )
        expected_root_scope_name = "N::"
        expected_layer_scope_name = "M::layers"
        expected_add_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.0"
        ]
        expected_mul_scope_names = [
            f"{expected_root_scope_name}/{expected_layer_scope_name}.{i}"
            for i in range(layer_num)
        ]

        add_scope_names = []
        mul_scope_names = []
        for node in graph.nodes():
            if node.kind() == "onnx::Add":
                add_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
            elif node.kind() == "onnx::Mul":
                mul_scope_names.append(
                    _remove_test_environment_prefix_from_scope_name(node.scopeName())
                )
        self.assertEqual(add_scope_names, expected_add_scope_names)
        self.assertEqual(mul_scope_names, expected_mul_scope_names)

    def test_aten_fallthrough(self):
        # [FUNCTION DOC] Converts a module calling erfc with the fallthrough export type, which
        # keeps operators without a symbolic as native ops. Asserts the first node is the
        # native erfc op.
        # Test aten export of op with no symbolic
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the complementary error
            # function of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns erfc of x.
                return torch.erfc(x)

        x = torch.randn(2, 3, 4)
        GLOBALS.export_onnx_opset_version = self.opset_version
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::erfc")

    def test_custom_op_fallthrough(self):
        # [FUNCTION DOC] Compiles a small C++ extension inline (side effect: invokes a build,
        # verbose output) that registers a custom tensor addition operator in a custom namespace.
        # Converts a module calling it with the fallthrough export type and asserts the first
        # graph node is that custom operator.
        # Test custom op
        op_source = """
        #include <torch/script.h>

        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
          return self + other;
        }

        static auto registry =
          torch::RegisterOperators("custom_namespace::custom_op", &custom_add);
        """

        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )

        class FooModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the registered custom add operator.
            def forward(self, input, other):
                # [FUNCTION DOC] HELPER: Returns the custom operator applied to the two inputs.
                # Calling custom op
                return torch.ops.custom_namespace.custom_op(input, other)

        x = torch.randn(2, 3, 4, requires_grad=False)
        y = torch.randn(2, 3, 4, requires_grad=False)
        model = FooModel()
        graph, _, __ = self._model_to_graph(
            model,
            (x, y),
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1, 2], "y": [0, 1, 2]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "custom_namespace::custom_op")

    # gelu is exported as onnx::Gelu for opset >= 20
    @skipIfUnsupportedMaxOpsetVersion(19)
    def test_custom_opsets_gelu(self):
        # [FUNCTION DOC] Registers (opset 9) a custom symbolic for gelu that emits a Gelu node
        # in the Microsoft domain and copies the input type; exports a GELU module with that
        # custom opset domain at version 1. Asserts the first node is a Gelu, the first opset
        # import is the test opset and the second import is the Microsoft domain at version 1.
        # Registration is removed by cleanup. Only runs for opsets up to 19, since gelu is
        # native from opset 20.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::gelu", 9)

        def gelu(g, self, approximate):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a Microsoft-domain Gelu node whose
            # type is copied from the input; the approximate argument is ignored.
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("::gelu", gelu, 9)
        model = torch.nn.GELU(approximate="none")
        x = torch.randn(3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)

    # gelu is exported as onnx::Gelu for opset >= 20
    @skipIfUnsupportedMaxOpsetVersion(19)
    def test_register_aten_custom_op_symbolic(self):
        # [FUNCTION DOC] Same as the custom gelu test but registers the symbolic under the aten
        # namespace name and exports without passing custom opsets. Asserts the first node is Gelu
        # and the second opset import has the Microsoft domain. Registration is removed by
        # cleanup. Only runs for opsets up to 19.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "aten::gelu", 9)

        def gelu(g, self, approximate):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a Microsoft-domain Gelu node whose
            # type is copied from the input; the approximate argument is ignored.
            return g.op("com.microsoft::Gelu", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("aten::gelu", gelu, 9)
        model = torch.nn.GELU(approximate="none")
        x = torch.randn(3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model, (x,), f, opset_version=self.opset_version, dynamo=False
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))

        self.assertEqual(graph.graph.node[0].op_type, "Gelu")
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")

    @skipIfNoLapack
    def test_custom_opsets_inverse(self):
        # [FUNCTION DOC] Registers a custom symbolic for the inverse op that emits an Inverse node
        # in the Microsoft domain, exports a module computing the inverse plus the input, and
        # asserts the first node type, the first opset import version equal to the test opset, and
        # the second opset import (Microsoft domain, version 1). Skipped without LAPACK.
        # Registration is removed by cleanup.
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

        class CustomInverse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the matrix inverse of the input plus the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the inverse of x plus x.
                return torch.inverse(x) + x

        def linalg_inv(g, self):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a Microsoft-domain Inverse node
            # whose type is copied from the input.
            return g.op("com.microsoft::Inverse", self).setType(self.type())

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv, 9)
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            dynamo=False,
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.node[0].op_type, "Inverse")
        self.assertEqual(graph.opset_import[0].version, self.opset_version)
        self.assertEqual(graph.opset_import[1].domain, "com.microsoft")
        self.assertEqual(graph.opset_import[1].version, 1)

    def test_onnx_fallthrough(self):
        # [FUNCTION DOC] Converts a module calling digamma with the fallthrough export type and
        # asserts the first node is the native digamma op.
        # Test aten export of op with symbolic for aten
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the digamma of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns digamma of x.
                return torch.digamma(x)

        x = torch.randn(100, 128)
        graph, _, __ = self._model_to_graph(
            Module(),
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::digamma")

    # prim::ListConstruct is exported as onnx::SequenceConstruct for opset >= 11
    @skipIfUnsupportedMaxOpsetVersion(10)
    def test_prim_fallthrough(self):
        # [FUNCTION DOC] Converts a scripted module that returns the input list or wraps a
        # single tensor in a list, with the fallthrough export type, and asserts the first node
        # is the native list construct op. Only runs for opsets up to 10 (later opsets export it
        # as a sequence construct).
        # Test prim op
        class PrimModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module whose script method forward returns x unchanged if
            # it is a list, otherwise a list holding x.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method returning the list form of x.
                if isinstance(x, list):
                    y = x
                else:
                    y = [x]
                return y

        x = torch.tensor([2])
        model = PrimModule()
        model.eval()
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["x"],
            dynamic_axes={"x": [0]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::ListConstruct")

    def test_custom_layer_tuple(self):
        # [FUNCTION DOC] Converts a module using an autograd function that defines a symbolic
        # emitting a custom-namespace node with two outputs. Asserts the first graph node is
        # that custom node.
        class CustomFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function returning its input twice, with a static
            # symbolic method emitting a custom-namespace node that has two outputs.
            @staticmethod
            def symbolic(g, input):
                # [FUNCTION DOC] HELPER: Symbolic rule: emits the custom node with the input
                # and two outputs.
                return g.op("CustomNamespace::Custom", input, outputs=2)

            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Forward rule: returns the input twice as a tuple.
                return input, input

        class Custom(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom autograd function to its input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the result of applying the function to the input.
                return CustomFunction.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        graph, _, _ = self._model_to_graph(
            model, batch, input_names=["batch"], dynamic_axes={"batch": [0, 1]}
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "CustomNamespace::Custom")

    def test_autograd_onnx_fallthrough(self):
        # [FUNCTION DOC] Converts a module using an autograd function without a symbolic, with
        # the fallthrough export type, and asserts the first node is a Python op node (the
        # function is kept opaque).
        class CustomFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function implementing a relu-like clamp at zero.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Forward rule: saves the input for backward and returns it
                # clamped below at zero.
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward rule: returns a copy of the output gradient with
                # entries set to zero where the saved input was negative.
                (input,) = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

        class Custom(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the custom autograd function to its input.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the result of applying the function to the input.
                return CustomFunction.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        graph, _, _ = self._model_to_graph(
            model,
            batch,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["batch"],
            dynamic_axes={"batch": [0, 1]},
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")

    def test_autograd_module_name(self):
        # [FUNCTION DOC] Converts a module applying two autograd functions that share a class name
        # but come from different modules (one local, one imported). Asserts the first two nodes
        # are Python op nodes with different module attribute strings.
        class CustomFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function implementing a relu-like clamp at zero,
            # defined in this test.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Forward rule: saves the input for backward and returns it
                # clamped below at zero.
                ctx.save_for_backward(input)
                return input.clamp(min=0)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward rule: returns a copy of the output gradient with
                # entries set to zero where the saved input was negative.
                (input,) = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input[input < 0] = 0
                return grad_input

        class Custom(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing the outputs of two autograd functions with the
            # same class name from different modules.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the sum of the local function and the imported
                # function applied to the input.
                return CustomFunction.apply(input) + CustomFunction2.apply(input)

        model = Custom()
        batch = torch.FloatTensor(1, 3)

        graph, _, _ = self._model_to_graph(
            model,
            batch,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            input_names=["batch"],
            dynamic_axes={"batch": [0, 1]},
        )
        iter = graph.nodes()
        autograd1 = next(iter)
        autograd2 = next(iter)
        self.assertEqual(autograd1.kind(), "prim::PythonOp")
        self.assertEqual(autograd2.kind(), "prim::PythonOp")
        self.assertNotEqual(autograd1.s("module"), autograd2.s("module"))

    def test_unused_initializers(self):
        # [FUNCTION DOC] Converts a model with a transposed convolution and an unused linear layer
        # without constant folding. Asserts the parameter dictionary has exactly two entries, i.e.
        # only the used convolution weight and bias (the unused layer parameters are dropped).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a used transposed convolution and an
            # unused linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the transposed convolution (16 to 33 channels)
                # and the unused linear layer (5 to 5).
                super().__init__()
                self.conv2 = torch.nn.ConvTranspose2d(
                    16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(1, 1)
                )
                self.k_proj = torch.nn.Linear(5, 5, bias=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the transposed convolution of x.
                x = self.conv2(x)
                return x

        x = torch.randn(20, 16, 50, 100)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        _, params_dict, __ = self._model_to_graph(
            Model(),
            (x,),
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        self.assertEqual(len(params_dict), 2)

    def test_scripting_param(self):
        # [FUNCTION DOC] Scripts a convolution plus batch norm module and converts it to a graph
        # in training mode with constant folding. Asserts every named parameter of the scripted
        # model appears among the graph input names.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 to 16 channel convolution followed
            # by batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and the batch norm layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 16, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the batch norm of the convolution of x.
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        model = torch.jit.script(MyModule())
        x = torch.randn(10, 3, 128, 128)
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            training=torch.onnx.TrainingMode.TRAINING,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        graph_input_params = [param.debugName() for param in graph.inputs()]
        for item in dict(model.named_parameters()):
            self.assertIn(
                item,
                graph_input_params,
                "Graph parameter names does not match model parameters.",
            )

    def test_fuse_conv_bn(self):
        # [FUNCTION DOC] Converts a convolution followed by batch norm in eval mode. Asserts
        # every node is a Conv (no batch norm node) and the graph has exactly one node, i.e.
        # batch norm was fused.
        class Fuse(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 to 2 channel convolution followed by batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution and the batch norm layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=True
                )
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the batch norm of the convolution of x.
                out = self.conv(x)
                return self.bn(out)

        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        graph, _, __ = self._model_to_graph(
            Fuse(),
            (x,),
            training=TrainingMode.EVAL,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::BatchNormalization")
            self.assertEqual(node.kind(), "onnx::Conv")

        self.assertEqual(len(list(graph.nodes())), 1)

    def test_fuse_resnet18(self):
        # [FUNCTION DOC] Converts a torchvision resnet18 without pretrained weights in eval mode
        # and asserts that no batch normalization node remains in the graph.
        model = torchvision.models.resnet18(weights=None)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        graph, _, __ = self._model_to_graph(
            model,
            (x,),
            training=TrainingMode.EVAL,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "onnx::BatchNormalization")

    def test_onnx_function_substitution_pass(self):
        # [FUNCTION DOC] Converts a module calling a scripted function (difference then sum).
        # Asserts that the constant node standing for the scripted function is gone and the call
        # is inlined, leaving exactly two nodes (sub and add).
        @torch.jit.script
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function: computes the difference x minus y and
            # returns x plus that difference.
            z = x - y
            return x + z

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the scripted function on its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the scripted function applied to x and y.
                return f(x, y)

        input_1 = torch.tensor([11])
        input_2 = torch.tensor([12])
        GLOBALS.export_onnx_opset_version = self.opset_version
        GLOBALS.operator_export_type = OperatorExportTypes.ONNX
        graph, _, __ = self._model_to_graph(
            MyModule(),
            (input_1, input_2),
            do_constant_folding=True,
            operator_export_type=OperatorExportTypes.ONNX,
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": [0], "input_2": [0]},
        )
        # Check that the prim::Constant node in the graph for representing the
        # scripted function `f` is removed and the following prim::CallFunction
        # is replced by inline graph, with onnx::Sub and onnx::Add nodes.
        for node in graph.nodes():
            self.assertNotEqual(node.kind(), "prim::Constant")
        self.assertEqual(
            len(list(graph.nodes())), 2
        )  # onnx::Sub and onnx::Add nodes only.

    def test_onnx_value_name(self):
        # [FUNCTION DOC] Exports (eval mode, initializers kept as inputs) a linear computation
        # using sliced copies of two parameters. Asserts the second and third graph inputs have
        # the parameter names, so values keep their parameter names after slicing.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3x3 weight parameter and a bias parameter
            # of length 3.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the uninitialized weight and bias parameters.
                super().__init__()
                self.in_weight = torch.nn.Parameter(torch.Tensor(3, 3))
                self.in_bias = torch.nn.Parameter(torch.Tensor(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Slices the weight and bias from index zero to the end and
                # returns the linear function of x with them.
                start = 0
                end = None
                weight = self.in_weight
                bias = self.in_bias
                weight = weight[start:end, :]
                if bias is not None:
                    bias = bias[start:end]
                return torch.nn.functional.linear(x, weight, bias)

        model = MyModule()
        x = torch.randn(3, 3)
        f = io.BytesIO()

        model.eval()
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            keep_initializers_as_inputs=True,
            dynamo=False,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.input[1].name, "in_weight")
        self.assertEqual(graph.graph.input[2].name, "in_bias")

    def test_onnx_node_naming(self):
        # [FUNCTION DOC] Exports a module with four linear layers and two multiplications, once
        # traced and once scripted, and asserts every node name in the saved model is in the
        # reference list of module-path based names (per-layer Gemm names and Mul names).
        class MainModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with four 10 to 10 linear layers named with
            # a leading underscore.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the four linear layers.
                super().__init__()
                self._module_1 = torch.nn.Linear(10, 10)
                self._module_2 = torch.nn.Linear(10, 10)
                self._module_3 = torch.nn.Linear(10, 10)
                self._module_4 = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Chains the layers with two multiplications between
                # earlier outputs and returns the last output.
                y = self._module_1(x)
                z = self._module_2(y)
                z = self._module_3(y * z)
                z = self._module_4(y * z)
                return z

        module = MainModule()
        ref_node_names = [
            "/_module_1/Gemm",
            "/_module_2/Gemm",
            "/_module_3/Gemm",
            "/_module_4/Gemm",
            "/Mul",
            "/Mul_1",
        ]
        f = io.BytesIO()

        torch.onnx.export(
            module, torch.ones(1, 10), f, output_names=["y"], dynamo=False
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        for n in onnx_model.graph.node:
            self.assertIn(n.name, ref_node_names)

        torch.onnx.export(
            torch.jit.script(module),
            torch.ones(1, 10),
            f,
            output_names=["y"],
            dynamo=False,
        )
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        for n in onnx_model.graph.node:
            self.assertIn(n.name, ref_node_names)

    def _test_deduplicate_initializers(self, torchscript=False):
        # [FUNCTION DOC] HELPER: Shared body of the deduplication tests. Builds a module with
        # shared layers and parameters plus two equal-valued separate parameters, scripted if
        # requested. Exports in training and preserve modes and asserts the initializer names
        # equal the set of named parameters; exports in eval mode and asserts the equal-valued
        # second parameter has been removed from the initializers.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with layers and parameters shared by reuse, plus two
            # separate parameters with equal values.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two linear layers, aliases a third to
                # the first, shares the weight and bias between layers, and creates
                # two equal-valued parameters.
                super().__init__()
                self.layer1 = torch.nn.Linear(3, 3)
                self.layer2 = torch.nn.Linear(3, 3)

                # Reusing layers.
                self.layer3 = self.layer1

                # Reusing parameters.
                self.layer2.weight = self.layer1.weight
                self.layer1.bias = self.layer2.bias

                # Parameter with different tensors equal in value.
                self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
                self.param2 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes x through the three layer references and adds
                # both extra parameters.
                return (
                    self.layer3(self.layer2(self.layer1(x))) + self.param1 + self.param2
                )

        model = torch.jit.script(MyModule()) if torchscript else MyModule()

        x = torch.randn(3, 3)
        param_name_set = {k for k, _ in model.named_parameters()}

        # Test training mode.
        model.train()
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            training=TrainingMode.TRAINING,
            opset_version=self.opset_version,
            dynamo=False,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

        model.train()
        f = io.BytesIO()
        torch.onnx.export(
            model,
            (x,),
            f,
            training=TrainingMode.PRESERVE,
            opset_version=self.opset_version,
            dynamo=False,
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

        # Test eval mode.
        model.eval()
        f = io.BytesIO()
        torch.onnx.export(
            model, (x,), f, opset_version=self.opset_version, dynamo=False
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        param_name_set.remove("param2")
        self.assertSetEqual({i.name for i in graph.graph.initializer}, param_name_set)

    def test_deduplicate_initializers(self):
        # [FUNCTION DOC] Runs the shared deduplication check on the eager module.
        self._test_deduplicate_initializers(torchscript=False)

    def test_deduplicate_initializers_torchscript(self):
        # [FUNCTION DOC] Runs the shared deduplication check on the scripted module.
        self._test_deduplicate_initializers(torchscript=True)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        # [FUNCTION DOC] Exports a module whose two parameters hold equal values but live on CPU
        # and CUDA, and asserts only the CPU one remains as an initializer. Needs CUDA.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones parameter on CPU and another on CUDA.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the CPU and CUDA parameters of three ones.
                super().__init__()
                self.w_cpu = torch.nn.Parameter(
                    torch.ones(3, device=torch.device("cpu"))
                )
                self.w_cuda = torch.nn.Parameter(
                    torch.ones(3, device=torch.device("cuda"))
                )

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a pair: x plus the CPU parameter and y plus
                # the CUDA parameter.
                return x + self.w_cpu, y + self.w_cuda

        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("cuda"))
        f = io.BytesIO()
        torch.onnx.export(
            Model(), (x, y), f, opset_version=self.opset_version, dynamo=False
        )
        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertSetEqual({i.name for i in graph.graph.initializer}, {"w_cpu"})

    def test_duplicated_output_node(self):
        # [FUNCTION DOC] Exports a model returning repeated outputs (five outputs from two
        # distinct values) in training mode with named inputs, outputs and dynamic axes. Asserts
        # input and output names and that the node types in order are Gemm, Identity, Identity,
        # Gemm, Identity, i.e. duplicate outputs become identity nodes.
        class DuplicatedOutputNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer applied to two inputs.
            def __init__(self, input_size, num_classes):
                # [FUNCTION DOC] HELPER: Creates the linear layer from the input size to
                # the class count.
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, num_classes)

            def forward(self, input0, input1):
                # [FUNCTION DOC] HELPER: Returns five values: the first result three times in
                # positions one, two and four, and the second result in positions three and five.
                out1 = self.fc1(input0)
                out2 = self.fc1(input1)
                return out1, out1, out2, out1, out2

        N, D_in, D_out = 64, 784, 10
        pt_model = DuplicatedOutputNet(D_in, D_out)

        f = io.BytesIO()
        x = torch.randn(N, D_in)
        dynamic_axes = {
            "input0": {0: "input0_dim0", 1: "input0_dim1"},
            "input1": {0: "input1_dim0", 1: "input1_dim1"},
            "output-0": {0: "output-0_dim0", 1: "output-0_dim1"},
            "output-1": {0: "output-1_dim0", 1: "output-1_dim1"},
            "output-2": {0: "output-2_dim0", 1: "output-2_dim1"},
            "output-3": {0: "output-3_dim0", 1: "output-3_dim1"},
            "output-4": {0: "output-4_dim0", 1: "output-4_dim1"},
        }

        torch.onnx.export(
            pt_model,
            (x, x),
            f,
            input_names=["input0", "input1"],
            output_names=["output-0", "output-1", "output-2", "output-3", "output-4"],
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.TRAINING,
            dynamic_axes=dynamic_axes,
            verbose=True,
            keep_initializers_as_inputs=True,
            dynamo=False,
        )

        graph = onnx.load(io.BytesIO(f.getvalue()))
        self.assertEqual(graph.graph.input[0].name, "input0")
        self.assertEqual(graph.graph.input[1].name, "input1")
        for i in range(5):
            self.assertEqual(graph.graph.output[i].name, f"output-{i}")
        self.assertEqual(graph.graph.node[0].op_type, "Gemm")
        self.assertEqual(graph.graph.node[1].op_type, "Identity")
        self.assertEqual(graph.graph.node[2].op_type, "Identity")
        self.assertEqual(graph.graph.node[3].op_type, "Gemm")
        self.assertEqual(graph.graph.node[4].op_type, "Identity")

    def test_deduplicate_ignore_upsample_scale(self):
        # [FUNCTION DOC] Exports a module with two upsample layers using the same constant scale.
        # Asserts there are two resize nodes and each one takes its scales input from exactly one
        # constant node, i.e. scales are not treated as shared weights by the deduplication.
        # upsample scale is a constant, not a model parameter,
        # therefore should be ignored by shared weight deduplication.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two upsample layers, each with scale factor two.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two upsample layers.
                super().__init__()
                self.upsample_1 = torch.nn.Upsample(scale_factor=2)
                self.upsample_2 = torch.nn.Upsample(scale_factor=2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of the outputs of both upsample layers.
                return self.upsample_1(x), self.upsample_2(x)

        f = io.BytesIO()
        x = torch.randn(1, 32, 224, 224)
        torch.onnx.export(Model(), x, f, dynamo=False)
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # aten::upsample converts to onnx::resize
        resize_nodes = [n for n in onnx_model.graph.node if n.op_type == "Resize"]
        self.assertEqual(len(resize_nodes), 2)
        for resize_node in resize_nodes:
            scale_node = [
                n for n in onnx_model.graph.node if n.output[0] == resize_node.input[2]
            ]
            self.assertEqual(len(scale_node), 1)
            self.assertEqual(scale_node[0].op_type, "Constant")

    def test_bad_symbolic_registration(self):
        # [FUNCTION DOC] Registers a custom concat symbolic whose declared argument descriptors do
        # not match its parameters (one descriptor, two args), then asserts exporting a
        # concatenation at opset 9 raises an assertion error with the exact message about the
        # argument and descriptor count mismatch. Then unregisters the symbolic.
        # KNOWN ISSUE: the unregister is not done in cleanup, so a failing assertion leaves the
        # bad symbolic registered for later tests.
        _onnx_opset_version = 9

        @parse_args("v")
        def cat(g, tensor_list, dim):
            # [FUNCTION DOC] HELPER: Symbolic rule (with a single argument descriptor): unpacks
            # the list input and emits a Concat node with the given axis.
            tensors = _unpack_list(tensor_list)
            return g.op("Concat", *tensors, axis_i=dim)

        torch.onnx.register_custom_op_symbolic("::cat", cat, _onnx_opset_version)

        class CatModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module concatenating its input with itself three times
            # along dimension 0.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input concatenated three times
                # along dimension zero.
                return torch.cat((x, x, x), 0)

        model = CatModel()
        x = torch.randn(2, 3)
        f = io.BytesIO()
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: torch.onnx.export(
                model, (x,), f, opset_version=_onnx_opset_version, dynamo=False
            ),
            (
                "A mismatch between the number of arguments (2) and their descriptors (1) was found at symbolic function "
                "'cat'. If you believe this is not due to custom symbolic implementation within your code or an external "
                "library, please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to "
                "report this bug."
            ),
        )
        torch.onnx.unregister_custom_op_symbolic("::cat", _onnx_opset_version)


if __name__ == "__main__":
    common_utils.run_tests()
