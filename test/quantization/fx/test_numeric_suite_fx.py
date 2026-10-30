# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the FX graph mode Numeric Suite (torch.ao.ns), covering
# graph matching, weight extraction, activation logging, shadow logging and the
# "n shadows" workflow. Defines small test-only models, user-wrapped functions,
# a helper to collect all quantize patterns, and several test case classes.
# Importing it has no side effects beyond binding the quantized ops namespace
# alias; running the file directly is rejected via raise_on_run_directly.

import copy
import math
import operator
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    default_dynamic_qconfig,
    QConfigMapping,
    get_default_qconfig_mapping,
)
import torch.ao.nn.quantized as nnq
toq = torch.ops.quantized
from torch.ao.quantization.quantize_fx import (
    convert_fx,
    convert_to_reference_fx,
    prepare_fx,
    prepare_qat_fx,
)
from torch.testing._internal.common_quantization import (
    ConvBnModel,
    ConvBnReLUModel,
    ConvModel,
    QuantizationTestCase,
    skipIfNoFBGEMM,
    skipIfNoQNNPACK,
    withQNNPACKBackend,
    SingleLayerLinearDynamicModel,
    SingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    SparseNNModel,
    skip_if_no_torchvision,
    TwoLayerLinearModel
)
from torch.testing._internal.common_utils import raise_on_run_directly, skipIfTorchDynamo
from torch.ao.quantization.quantization_mappings import (
    get_default_static_quant_module_mappings,
    get_default_dynamic_quant_module_mappings,
    get_default_float_to_quantized_operator_mappings,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.ao.quantization.fx.pattern_utils import get_default_quant_patterns
import torch.ao.quantization.fx.quantize_handler as qh
from torch.ao.ns.fx.pattern_utils import (
    get_type_a_related_to_b,
)
from torch.ao.ns.fx.graph_matcher import (
    get_matching_subgraph_pairs,
    GraphMatchingException,
)
from torch.ao.ns.fx.utils import (
    compute_sqnr,
    compute_normalized_l2_error,
    compute_cosine_similarity,
)
from torch.ao.ns.fx.mappings import (
    get_node_type_to_io_type_map,
    get_unmatchable_types_map,
    get_base_name_to_sets_of_related_ops,
    get_base_name_for_op,
    add_op_to_sets_of_related_ops,
)
from torch.ao.ns.fx.weight_utils import (
    get_op_to_type_to_weight_extraction_fn,
)
from torch.ao.ns._numeric_suite_fx import (
    extract_weights,
    _extract_weights_impl,
    add_loggers,
    _add_loggers_impl,
    OutputLogger,
    add_shadow_loggers,
    _add_shadow_loggers_impl,
    extract_logger_info,
    extract_shadow_logger_info,
    extend_logger_results_with_comparison,
    prepare_n_shadows_model,
    convert_n_shadows_model,
    extract_results_n_shadows_model,
    OutputComparisonLogger,
    print_comparisons_n_shadows_model,
    loggers_set_enabled,
    loggers_set_save_activations,
    _prepare_n_shadows_add_loggers_model,
    _n_shadows_compare_weights,
)
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from torch.ao.quantization.backend_config import get_native_backend_config
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers


# Note: these models are not for use outside of this file. While it's good
# to reuse code, we also need to be able to iterate on tests
# quickly when debugging. If a test model has a large number of callsites
# across various different files, speed of debugging on individual test cases
# decreases.
class LinearReluFunctional(nn.Module):
    # [CLASS DOC] HELPER: Test model with one functional linear (4x4 weight, kaiming
    # initialized, zero bias) followed by a functional relu. Parameters w1 and b1.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the 4x4 weight w1 and zero bias b1 as parameters
        # and initializes w1 with kaiming uniform (a = sqrt of 5). Mutates only self.
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 4))
        self.b1 = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies functional linear with w1 and b1, then
        # functional relu; returns the result tensor.
        x = F.linear(x, self.w1, self.b1)
        x = F.relu(x)
        return x


class LinearFunctional(nn.Module):
    # [CLASS DOC] HELPER: Test model with a single functional linear using a 4x4
    # weight w1 and zero bias b1 (no activation).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates parameters w1 (4x4, kaiming uniform initialized)
        # and b1 (zeros of size 4).
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(4, 4))
        self.b1 = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns functional linear of the input with w1 and b1.
        x = F.linear(x, self.w1, self.b1)
        return x


class LinearReluLinearFunctional(nn.Module):
    # [CLASS DOC] HELPER: Test model applying functional linear, relu, then functional
    # linear again, reusing the same weight w and bias b both times.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates parameters w (4x4, from an uninitialized tensor,
        # then kaiming uniform initialized) and b (zeros of size 4).
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(4, 4))
        self.b = nn.Parameter(torch.zeros(4))
        torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs linear, relu, linear with shared w and b; returns
        # the output tensor.
        x = F.linear(x, self.w, self.b)
        x = F.relu(x)
        x = F.linear(x, self.w, self.b)
        return x


class AddMulFunctional(nn.Module):
    # [CLASS DOC] HELPER: Stateless test model that exercises add and multiply with
    # scalar operands on either side and with a second tensor input.
    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Takes tensors x and y. Adds and multiplies x by the scalar
        # one in both operand orders, then adds and multiplies by y; returns x.
        x = x + 1.0
        x = x * 1.0
        x = 1.0 + x
        x = 1.0 * x
        x = x + y
        x = x * y
        return x


class AllConvAndLinearFusionModules(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model holding conv1d, conv2d, conv3d and linear layers
    # in plain, relu-fused, batchnorm-fused (qat only) and batchnorm+relu-fused
    # (qat only) variants. Layers have 1 channel/feature and kernel size 1.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the convolution, batchnorm, relu and linear
        # submodules used by forward; each layer is a separate attribute.
        super().__init__()
        # conv1d
        self.conv1d_0 = nn.Conv1d(1, 1, 1)
        # conv1d - relu
        self.conv1d_1 = nn.Conv1d(1, 1, 1)
        self.relu_0 = nn.ReLU()
        # conv1d - bn (qat only)
        self.conv1d_2 = nn.Conv1d(1, 1, 1)
        self.bn1d_0 = nn.BatchNorm1d(1)
        # conv1d - bn - relu (qat only)
        self.conv1d_3 = nn.Conv1d(1, 1, 1)
        self.bn1d_1 = nn.BatchNorm1d(1)
        self.relu_4 = nn.ReLU()
        # conv2d
        self.conv2d_0 = nn.Conv2d(1, 1, 1)
        # conv2d - relu
        self.conv2d_1 = nn.Conv2d(1, 1, 1)
        self.relu_1 = nn.ReLU()
        # conv2d - bn (qat only)
        self.conv2d_2 = nn.Conv2d(1, 1, 1)
        self.bn2d_0 = nn.BatchNorm2d(1)
        # conv2d - bn - relu (qat only)
        self.conv2d_3 = nn.Conv2d(1, 1, 1)
        self.bn2d_1 = nn.BatchNorm2d(1)
        self.relu_5 = nn.ReLU()
        # conv3d
        self.conv3d_0 = nn.Conv3d(1, 1, 1)
        # conv3d - relu
        self.conv3d_1 = nn.Conv3d(1, 1, 1)
        self.relu_2 = nn.ReLU()
        # conv3d - bn (qat only)
        self.conv3d_2 = nn.Conv3d(1, 1, 1)
        self.bn3d_0 = nn.BatchNorm3d(1)
        # conv3d - bn - relu (qat only)
        self.conv3d_3 = nn.Conv3d(1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(1)
        self.relu_6 = nn.ReLU()
        # linear
        self.linear_0 = nn.Linear(1, 1)
        # linear - relu
        self.linear_1 = nn.Linear(1, 1)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the 1d conv group, reshapes to 4 dims for the 2d
        # group, reshapes to 5 dims for the 3d group, then to 2 dims for the linear
        # group; returns the final tensor. Expects a single-element input.
        # conv1d
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        x = self.relu_0(x)
        x = self.conv1d_2(x)
        x = self.bn1d_0(x)
        x = self.conv1d_3(x)
        x = self.bn1d_1(x)
        x = self.relu_4(x)
        # conv2d
        x = x.reshape(1, 1, 1, 1)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.bn2d_0(x)
        x = self.conv2d_3(x)
        x = self.bn2d_1(x)
        x = self.relu_5(x)
        # conv3d
        x = x.reshape(1, 1, 1, 1, 1)
        x = self.conv3d_0(x)
        x = self.conv3d_1(x)
        x = self.relu_2(x)
        x = self.conv3d_2(x)
        x = self.bn3d_0(x)
        x = self.conv3d_3(x)
        x = self.bn3d_1(x)
        x = self.relu_6(x)
        # linear
        x = x.reshape(1, 1)
        x = self.linear_0(x)
        x = self.linear_1(x)
        x = self.relu_3(x)
        return x


class AllConvFunctional(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model calling functional conv1d, conv2d and conv3d
    # twice each, with relu after each pair, using externally supplied weights and
    # biases stored as parameters plus fixed stride/padding/dilation/groups values.
    def __init__(self, weight1d, weight2d, weight3d, bias1d, bias2d, bias3d):
        # [FUNCTION DOC] HELPER: Wraps the six given weight and bias tensors in
        # parameters and records default stride, padding, dilation (all trivial) and one
        # group for the 1d, 2d and 3d convolutions.
        super().__init__()
        self.weight1d = torch.nn.Parameter(weight1d)
        self.weight2d = torch.nn.Parameter(weight2d)
        self.weight3d = torch.nn.Parameter(weight3d)
        self.bias1d = torch.nn.Parameter(bias1d)
        self.bias2d = torch.nn.Parameter(bias2d)
        self.bias3d = torch.nn.Parameter(bias3d)
        self.stride1d = 1
        self.padding1d = 0
        self.dilation1d = 1
        self.stride2d = (1, 1)
        self.padding2d = (0, 0)
        self.dilation2d = (1, 1)
        self.groups = 1
        self.stride3d = (1, 1, 1)
        self.padding3d = (0, 0, 0)
        self.dilation3d = (1, 1, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies conv1d twice, relu, conv2d twice, relu, conv3d
        # twice, relu, in that order, with the stored hyperparameters; returns the result.
        # Input shape must fit the supplied weights.
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.conv1d(
            x, self.weight1d, self.bias1d, self.stride1d, self.padding1d,
            self.dilation1d, self.groups)
        x = F.relu(x)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.conv2d(
            x, self.weight2d, self.bias2d, self.stride2d, self.padding2d,
            self.dilation2d, self.groups)
        x = F.relu(x)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.conv3d(
            x, self.weight3d, self.bias3d, self.stride3d, self.padding3d,
            self.dilation3d, self.groups)
        x = F.relu(x)
        return x

@torch.fx.wrap
def _wrapped_hardswish(x):
    # [FUNCTION DOC] HELPER: Traced as one opaque call (fx wrap) so user-defined
    # function matching can be tested; returns functional hardswish of the input.
    return F.hardswish(x)

@torch.fx.wrap
def _wrapped_hardswish_fp16(x):
    # [FUNCTION DOC] HELPER: Opaque (fx wrap) function that dequantizes the input,
    # applies hardswish and converts the result to float16; returns it.
    x = x.dequantize()
    x = F.hardswish(x)
    x = x.to(torch.float16)
    return x

@torch.fx.wrap
def _wrapped_sigmoid(x):
    # [FUNCTION DOC] HELPER: Opaque (fx wrap) function returning functional sigmoid
    # of the input.
    return F.sigmoid(x)

@torch.fx.wrap
def _wrapped_linear(x, w, b):
    # [FUNCTION DOC] HELPER: Opaque (fx wrap) function returning functional linear
    # with the given input, weight and bias.
    return F.linear(x, w, b)

def get_all_quant_patterns():
    # [FUNCTION DOC] HELPER: Returns the mapping of all quantize patterns to handler
    # classes: starts from the default quant patterns and adds (overwriting on
    # conflict) every pattern from the native backend config that has a handler.
    # Pure apart from building the mapping; no arguments.
    """ we are in the process to migrate the frontend of fx graph mode quant
    to use backend_config_dict, so some of the patterns are moved to backend_config_dict
    this function will include these patterns so that we can still have all the patterns
    """
    # TODO: we can remove this call, and get all patterns from backend_config_dict in
    # the future when the frontend refactor is done in fx graph mode quantization
    all_quant_patterns = get_default_quant_patterns()
    # some of the patterns are moved to (native) backend_config_dict so we need to
    # add them back here
    for pattern, quantize_handler in _get_pattern_to_quantize_handlers(get_native_backend_config()).items():
        all_quant_patterns[pattern] = quantize_handler
    return all_quant_patterns

class TestFXGraphMatcher(QuantizationTestCase):
    # [CLASS DOC] Tests numeric suite graph matching (get_matching_subgraph_pairs)
    # between float, prepared and quantized models, including expected failures and
    # completeness checks of the op relationship mapping. Most tests need FBGEMM.

    @skipIfNoFBGEMM
    def test_simple_mod(self):
        # [FUNCTION DOC] Matches a prepared single conv model with its quantized version
        # and checks the matched pair is the float conv module with an observer versus
        # the quantized conv module.
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'

        expected_types = {
            conv_name_0: ((nn.Conv2d, torch.ao.quantization.MinMaxObserver), (nnq.Conv2d, nnq.Conv2d)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_fun(self):
        # [FUNCTION DOC] Same as the module test but for a functional linear model;
        # expects the float linear with an observer matched against the quantized linear.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model applying functional linear with a 1x4 weight and
            # a one element bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters w (1x4, kaiming initialized) and b
                # (zeros of size 1).
                super().__init__()
                self.w = nn.Parameter(torch.empty(1, 4))
                self.b = nn.Parameter(torch.zeros(1))
                torch.nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns functional linear of the input with w and b.
                return F.linear(x, self.w, self.b)

        m = M().eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        expected_types = {
            linear_name_0:
                ((F.linear, torch.ao.quantization.MinMaxObserver), (toq.linear, toq.linear))
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_fusion(self):
        # [FUNCTION DOC] Checks that a functional linear followed by relu is matched as
        # one fused subgraph: float linear with observer versus the quantized fused
        # linear-relu op.
        m = LinearReluFunctional().eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(4, 4),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        linear_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.linear) + '_0'

        expected_types = {
            linear_name_0:
                ((F.linear, torch.ao.quantization.MinMaxObserver), (toq.linear_relu, toq.linear_relu)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_simple_mod_multi(self):
        # [FUNCTION DOC] Matches nested sequential conv models prepared and quantized;
        # passes if matching raises no exception (results are not inspected).
        m = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 1, 1),
            ),
            nn.Conv2d(1, 1, 1),
        ).eval()
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=(torch.randn(1, 1, 1, 1),))
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    def test_simple_tensor_ops(self):
        # [FUNCTION DOC] Matches a model that adds two tensor inputs, prepared and
        # quantized; passes if no exception is raised (results are not checked).
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model returning the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the elementwise sum of x and y.
                z = x + y
                return z

        m = M().eval()
        example_inputs = (torch.randn(1), torch.randn(1))
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    def test_matching_failure_node_count(self):
        # [FUNCTION DOC] Verifies matching raises the graph matching exception when two
        # models have the same node types but different numbers of matchable nodes (one
        # conv versus two convs).
        # verify that matching graphs with matching node types but
        # different counts of matchable nodes fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp1 = prepare_fx(m1, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp2 = prepare_fx(m2, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @skipIfNoFBGEMM
    def test_matching_failure_node_type(self):
        # [FUNCTION DOC] Verifies matching raises the graph matching exception when the
        # models have different node types (conv versus linear).
        # verify that matching graphs with non-matching node types fails
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        m2 = nn.Sequential(nn.Linear(1, 1)).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp1 = prepare_fx(m1, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        example_inputs = (torch.randn(1, 1),)
        mp2 = prepare_fx(m2, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        with self.assertRaises(GraphMatchingException) as ex:
            results = get_matching_subgraph_pairs(mp1, mp2)

    @skipIfNoFBGEMM
    def test_nodes_before_cat(self):
        # [FUNCTION DOC] Checks that the two add nodes feeding a concatenation, plus the
        # cat itself, are all matched, with adds mapping to quantized add and cat staying
        # the same function on both sides.
        # verify that nodes before cat get matched
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model adding one to the input twice and concatenating
            # the two results.
            def forward(self, x0):
                # [FUNCTION DOC] HELPER: Computes two add results from x and returns their
                # concatenation along the first dimension.
                x1 = torch.add(x0, 1.0)
                y1 = torch.add(x0, 1.0)
                x2 = torch.cat([x1, y1])
                return x2

        m = M().eval()
        example_inputs = (torch.randn(1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        cat_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.cat) + '_0'
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'

        expected_types = {
            cat_name_0: ((torch.cat, torch.cat), (torch.cat, torch.cat)),
            add_name_0: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_dict_return_type(self):
        # [FUNCTION DOC] Checks that three add nodes whose outputs are returned inside a
        # nested dictionary, tuple and list structure are all found and matched with
        # the quantized add.
        # verify that we can traverse up nodes which return dictionaries
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model returning a nested structure containing three
            # independent add results.
            def forward(self, x0):
                # [FUNCTION DOC] HELPER: Returns a dictionary holding three add outputs, nested
                # in tuple and list containers.
                x1 = torch.add(x0, 1.0)
                y1 = torch.add(x0, 1.0)
                z1 = torch.add(x0, 1.0)
                a1 = {'x1': x1, 'y1': (y1,), 'z1': [{'key': (z1,)}]}
                return a1

        m = M().eval()
        example_inputs = (torch.randn(1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_0'
        add_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_1'
        add_name_2 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.add) + '_2'

        expected_types = {
            add_name_0: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_1: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
            add_name_2: ((torch.add, torch.ao.quantization.MinMaxObserver), (toq.add, toq.add)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    @skipIfNoFBGEMM
    def test_nodes_with_equal_types_get_matched(self):
        # [FUNCTION DOC] Disables quantization for the second conv so that its float and
        # "quantized" module types are equal, and checks that such equal-typed nodes are
        # still matched, along with mul, relu and sigmoid.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of two convs followed by mul, sigmoid and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two 1x1 conv layers conv1 and conv2.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1, conv2, a self multiply, sigmoid and
                # relu in order; returns the result.
                x = self.conv1(x)
                x = self.conv2(x)
                x = torch.mul(x, x)
                x = torch.sigmoid(x)
                x = F.relu(x)
                return x

        m = M().eval()
        # prevent conv2 from getting quantized, so we can test
        # modules with equal types
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping().set_module_name("conv2", None)
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        conv_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_0'
        conv_name_1 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, nn.Conv2d) + '_1'
        mul_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.mul) + '_0'
        relu_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.relu) + '_0'
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'

        # all of these should be matched
        expected_types = {
            conv_name_1:
                ((nn.Conv2d, torch.ao.quantization.HistogramObserver), (nnq.Conv2d, nnq.Conv2d)),
            conv_name_0:
                ((nn.Conv2d, torch.ao.quantization.HistogramObserver), (nn.Conv2d, nn.Conv2d)),
            mul_name_0: ((torch.mul, torch.ao.quantization.HistogramObserver), (toq.mul, toq.mul)),
            relu_name_0: ((F.relu, torch.ao.quantization.FixedQParamsObserver), (F.relu, F.relu)),
            sigmoid_name_0:
                ((torch.sigmoid, torch.ao.quantization.FixedQParamsObserver), (torch.sigmoid, torch.sigmoid)),
        }
        self.assert_types_for_matched_subgraph_pairs(results, expected_types, mp, mq)

    def test_methods(self):
        # [FUNCTION DOC] Verifies matching works on tensor method calls: two prepared
        # copies of a sigmoid method model are matched with each other and the pair has
        # the method name with a fixed qparams observer on both sides.
        """
        Verify that graph matching works on methods
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model calling the sigmoid tensor method on its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sigmoid method applied to x.
                x = x.sigmoid()
                return x

        m1 = M().eval()
        m2 = M().eval()
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1),)
        m1p = prepare_fx(m1, qconfig_mapping, example_inputs=example_inputs)
        m2p = prepare_fx(m2, qconfig_mapping, example_inputs=example_inputs)
        results = get_matching_subgraph_pairs(m1p, m2p)
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        sigmoid_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, torch.sigmoid) + '_0'
        expected_types = {
            sigmoid_name_0:
                (('sigmoid', torch.ao.quantization.FixedQParamsObserver), ('sigmoid', torch.ao.quantization.FixedQParamsObserver)),
        }
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1p, m2p)

    def test_op_relationship_mapping(self):
        # [FUNCTION DOC] Checks that the relationship mapping used by the numeric suite
        # covers the quantization mappings: static module, static function and dynamic
        # module mappings (minus listed skips) must be related pairs, and ops of each
        # quantize handler in the full pattern table must appear in the related op sets
        # or be unmatchable, depending on the handler class. Raises an assertion
        # error when an unhandled handler/op combination is found.
        """
        Tests that the mapping of op relationships is complete.
        """
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        type_a_related_to_b = \
            get_type_a_related_to_b(base_name_to_sets_of_related_ops)

        # 1. check static quant module mappings
        static_quant_mod_mappings = get_default_static_quant_module_mappings()
        for fp32_type, int8_type in static_quant_mod_mappings.items():
            # skip quants and dequants, for the purposes of Numerical Suite
            types_to_skip = (
                torch.ao.quantization.QuantStub,
                torch.ao.quantization.DeQuantStub,
                nnq.FloatFunctional,
                # the ConvTranspose3d swap is not implemented in FX Graph
                # mode quantization yet
                nn.ConvTranspose3d,
                # the GroupNorm swap is not implemented in FX Graph
                # mode quantization yet
                nn.GroupNorm,
                # nnq.ReLU6 is no longer swapped, because nn.ReLU6 can
                # take quantized inputs
                nn.ReLU6,
            )
            if fp32_type in types_to_skip:
                continue

            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 2. check static quant op mappings
        static_quant_fun_mappings = get_default_float_to_quantized_operator_mappings()
        for fp32_type, int8_type in static_quant_fun_mappings.items():
            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 3. check dynamic quant mappings
        dynamic_quant_mappings = get_default_dynamic_quant_module_mappings()
        for fp32_type, int8_type in dynamic_quant_mappings.items():
            # TODO(future PR): enable correct weight extraction for these
            # and remove from this list.
            types_to_skip = (
                nn.GRUCell,
                nn.GRU,
                nn.LSTMCell,
                nn.RNNCell,
            )
            if fp32_type in types_to_skip:
                continue
            # verify relatedness
            in_type_a_related_to_b = \
                (fp32_type, int8_type) in type_a_related_to_b
            self.assertTrue(
                in_type_a_related_to_b,
                f"{fp32_type} and {int8_type} need a relationship mapping")

        # 4. go through the ops mapped to each QuantizeHandler type, and verify
        # correctness.
        def _op_in_base_sets_of_related_ops(op):
            # [FUNCTION DOC] HELPER: Returns true if the given op appears in any set of the
            # base name to related ops mapping, else false.
            for ops in base_name_to_sets_of_related_ops.values():
                if op in ops:
                    return True
            return False

        unmatchable_types_map = get_unmatchable_types_map()
        FUNS_UNMATCHABLE = unmatchable_types_map['funs_unmatchable']
        MODS_UNMATCHABLE = unmatchable_types_map['mods_unmatchable']
        METHS_UNMATCHABLE = unmatchable_types_map['meths_unmatchable']

        def _op_is_unmatchable(op):
            # [FUNCTION DOC] HELPER: Returns true if the op is listed as unmatchable among
            # functions, modules or methods.
            return (
                op in FUNS_UNMATCHABLE or
                op in MODS_UNMATCHABLE or
                op in METHS_UNMATCHABLE
            )

        default_quant_patterns = get_all_quant_patterns()
        for pattern, qhandler_cls in default_quant_patterns.items():
            base_op = None
            if isinstance(pattern, tuple):
                base_op = pattern[-1]
            elif isinstance(pattern, str):
                base_op = pattern
            else:
                base_op = pattern

            qhandler_cls_all_ops_quantizeable = [
                qh.CatQuantizeHandler,
                qh.ConvReluQuantizeHandler,
                qh.LinearReLUQuantizeHandler,
                qh.BatchNormQuantizeHandler,
                qh.EmbeddingQuantizeHandler,
                qh.RNNDynamicQuantizeHandler,
            ]

            qhandler_cls_quant_op_same_signature = [
                qh.FixedQParamsOpQuantizeHandler,
                qh.CopyNodeQuantizeHandler,
                qh.GeneralTensorShapeOpQuantizeHandler,
            ]

            if qhandler_cls == qh.BinaryOpQuantizeHandler:
                # these ops do not have quantized equivalents
                ops_to_skip = [
                    torch.bmm,
                    torch.div,
                    torch.sub,
                    operator.truediv,
                    operator.sub
                ]
                if base_op in ops_to_skip:
                    continue
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls == qh.RNNDynamicQuantizeHandler:
                # TODO(future PR): add support for all classes in
                # RNNDynamicQuantizeHandler
                pass
            elif qhandler_cls == qh.DefaultNodeQuantizeHandler:
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            elif qhandler_cls in qhandler_cls_quant_op_same_signature:
                # these ops use the same op signature for fp32 and quantized
                # tensors
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op) or
                    _op_is_unmatchable(base_op),
                    f"{base_op} not in sets of related ops or unmatchable")
            elif qhandler_cls in qhandler_cls_all_ops_quantizeable:
                self.assertTrue(
                    _op_in_base_sets_of_related_ops(base_op),
                    f"{base_op} not in sets of related ops")
            else:
                # torch.sum does not have quantized equivalents
                if base_op in [
                        torch.sum,
                        nn.GRUCell,
                        nn.GRU,
                        nn.LSTMCell,
                        nn.RNNCell,
                ]:
                    continue
                if isinstance(base_op, tuple):
                    # skip fusion patterns
                    continue
                # didn't match explicit quantize handler class, we can check if the
                # operator is in the related op set directly
                if not (_op_in_base_sets_of_related_ops(base_op) or _op_is_unmatchable(base_op)):
                    raise AssertionError(
                        f"handling for {qhandler_cls} for op {base_op} not implemented")

    @skipIfNoFBGEMM
    def test_user_defined_function(self):
        # [FUNCTION DOC] Verifies graph matching treats a wrapped user function as
        # related to the builtin hardswish after registering that relation; the pair
        # is matched with histogram observers.
        """
        Verify that graph matching works on user defined functions
        """
        class M1(nn.Module):
            # [CLASS DOC] HELPER: Model applying builtin functional hardswish.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns hardswish of x.
                x = F.hardswish(x)
                return x

        class M2(nn.Module):
            # [CLASS DOC] HELPER: Model applying the wrapped user hardswish function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the wrapped hardswish function applied to x.
                x = _wrapped_hardswish(x)
                return x

        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        m1 = prepare_fx(M1().eval(), qconfig_mapping, example_inputs=example_inputs)
        m2 = prepare_fx(M2().eval(), qconfig_mapping, example_inputs=example_inputs)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_hardswish, F.hardswish)

        results = get_matching_subgraph_pairs(
            m1, m2,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)

        hardswish_name_0 = 'base_op_' + get_base_name_for_op(
            base_name_to_sets_of_related_ops, F.hardswish) + '_0'

        expected_types = {
            hardswish_name_0:
                ((F.hardswish, torch.ao.quantization.HistogramObserver), (_wrapped_hardswish, _wrapped_hardswish)),
        }
        self.assert_types_for_matched_subgraph_pairs(
            results, expected_types, m1, m2)

    @skipIfNoFBGEMM
    def test_results_order(self):
        # [FUNCTION DOC] Checks that matching a conv followed by linear returns exactly
        # two results in graph order, with start nodes named by position (first node
        # then second node) in both graphs.
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Linear(1, 1),
        ).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        results = get_matching_subgraph_pairs(mp, mq)
        self.assertTrue(len(results) == 2)
        results_iter = iter(results.items())
        _, (subgraph_a_0, subgraph_b_0) = next(results_iter)
        self.assertTrue(subgraph_a_0.start_node.name == '_0' and
                        subgraph_b_0.start_node.name == '_0')
        _, (subgraph_a_1, subgraph_b_1) = next(results_iter)
        self.assertTrue(subgraph_a_1.start_node.name == '_1' and
                        subgraph_b_1.start_node.name == '_1')


class TestFXGraphMatcherModels(QuantizationTestCase):
    # [CLASS DOC] Graph matching tests on torchvision models (skipped without
    # torchvision).

    @skipIfTorchDynamo("too slow")
    @skipIfNoFBGEMM
    @skip_if_no_torchvision
    def test_mobilenet_v2(self):
        # [FUNCTION DOC] Verifies mobilenet v2 can be matched float versus prepared and
        # prepared versus quantized; passes if no exception is raised. Skipped under
        # dynamo as too slow.
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).eval().float()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_fx(copy.deepcopy(m), {'': torch.ao.quantization.default_qconfig}, example_inputs=example_inputs)
        # assume success if no exceptions
        results_m_mp = get_matching_subgraph_pairs(torch.fx.symbolic_trace(m), mp)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results_mp_mq = get_matching_subgraph_pairs(mp, mq)

    @skipIfNoFBGEMM
    @skip_if_no_torchvision
    def test_mobilenet_v2_qat(self):
        # [FUNCTION DOC] Same as the mobilenet test but with qat preparation using the
        # fbgemm qat qconfig; passes if matching raises no exception.
        # verify that mobilenetv2 graph is able to be matched
        import torchvision
        m = torchvision.models.__dict__['mobilenet_v2'](pretrained=False).float()
        example_inputs = (torch.randn(1, 3, 224, 224),)
        mp = prepare_qat_fx(
            copy.deepcopy(m),
            {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')},
            example_inputs=example_inputs)
        # assume success if no exceptions
        results_m_mp = get_matching_subgraph_pairs(torch.fx.symbolic_trace(m), mp)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        # assume success if no exceptions
        results_mp_mq = get_matching_subgraph_pairs(mp, mq)


class FXNumericSuiteQuantizationTestCase(QuantizationTestCase):
    # [CLASS DOC] HELPER: Base class for numeric suite end to end tests. Provides
    # shared drivers that prepare and convert a model, run weight extraction,
    # activation logging or shadow logging, validate result structure and attach
    # sqnr, l2 error and cosine similarity comparisons.
    def _test_extract_weights(
        self, m, example_inputs, results_len=0, qconfig_dict=None, prepare_fn=prepare_fx
    ):
        # [FUNCTION DOC] HELPER: Traces the model with fx, prepares and converts it, then
        # for both the public and internal weight extraction functions and for the pairs
        # float vs prepared and prepared vs quantized asserts the result length equals
        # results_len, validates the result dictionary and adds sqnr, l2 and cosine
        # comparisons. Uses the default qconfig when none is given.
        m = torch.fx.symbolic_trace(m)
        if qconfig_dict is None:
            qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=example_inputs)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        # test both the public API as well as the internal GraphModule API
        for extract_weights_fun in (extract_weights, _extract_weights_impl):
            # test both m vs mp and mp vs mq
            for m1, m2 in ((m, mp), (mp, mq)):
                results = extract_weights_fun('a', m1, 'b', m2)
                self.assertTrue(
                    len(results) == results_len,
                    f"expected len {results_len}, got len {len(results)}")
                self.assert_ns_compare_dict_valid(results)
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_sqnr, 'sqnr')
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_normalized_l2_error, 'l2_error')
                extend_logger_results_with_comparison(
                    results, 'a', 'b', compute_cosine_similarity,
                    'cosine_similarity')

    def _test_match_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=0,
        should_log_inputs=False,
        qconfig_dict=None,
        skip_scripting=False,
        prepare_fn=prepare_fx,
    ):
        # [FUNCTION DOC] HELPER: Prepares (eval mode for ptq, train mode otherwise),
        # calibrates and converts the model, adds output loggers to the float/prepared
        # and prepared/quantized pairs, optionally checks logger node counts, scripts
        # the graph modules unless skip_scripting, runs data, then extracts logger
        # info for both pairs, asserts the count equals results_len, validates and adds
        # comparisons. Returns a list of the two comparison dictionaries.
        if qconfig_dict is None:
            qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping()
        if prepare_fn is prepare_fx:
            m.eval()
        else:
            m.train()
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=data)
        mp(*data)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)

        m_ns, mp_ns2 = add_loggers(
            'a', m, 'b', copy.deepcopy(mp), OutputLogger,
            should_log_inputs=should_log_inputs)
        mp_ns, mq_ns = add_loggers(
            'a', mp, 'b', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        if prepared_expected_node_occurrence:
            self.checkGraphModuleNodes(
                m_ns, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_ns2, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_ns, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mq_ns, expected_node_occurrence=prepared_expected_node_occurrence)

        if not skip_scripting:
            m_ns = torch.jit.script(m_ns)
            mp_ns = torch.jit.script(mp_ns)
            mq_ns = torch.jit.script(mq_ns)

        # calibrate
        m_ns(*data)
        mp_ns2(*data)
        mp_ns(*data)
        mq_ns(*data)

        # check activation result correctness
        results = []
        for m1, m2 in ((m_ns, mp_ns2), (mp_ns, mq_ns)):
            act_compare_dict = extract_logger_info(
                m1, m2, OutputLogger, 'b')
            self.assertTrue(
                len(act_compare_dict) == results_len,
                f"expected len {results_len}, got len {len(act_compare_dict)}")
            self.assert_ns_compare_dict_valid(act_compare_dict)
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_normalized_l2_error, 'l2_error')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_cosine_similarity,
                'cosine_similarity')
            results.append(act_compare_dict)
        return results

    def _test_match_shadow_activations(
        self, m, data, prepared_expected_node_occurrence=None, results_len=None,
        should_log_inputs=False, qconfig_dict=None, skip_scripting=False,
        prepare_fn=prepare_fx, compare_fp32_vs_fp32_prepared=True,
    ):
        # [FUNCTION DOC] HELPER: Like the activation driver but adds shadow loggers
        # (optionally also for float vs prepared), prints the qconfig, prepared and
        # quantized models, optionally checks node counts, scripts unless skipped, runs
        # data and extracts shadow logger info. Checks length only when results_len is
        # given; validates and adds comparisons. Returns a list of result dictionaries.
        if qconfig_dict is None:
            qconfig_dict = torch.ao.quantization.get_default_qconfig_mapping()
        if prepare_fn is prepare_fx:
            m.eval()
        else:
            m.train()
        print("qconfig_dict:", qconfig_dict)
        mp = prepare_fn(copy.deepcopy(m), qconfig_dict, example_inputs=data)
        print("prepared:", mp)
        mp(*data)
        mp_copy = copy.deepcopy(mp)
        mq = convert_fx(mp_copy)
        print("quantized:", mq)

        if compare_fp32_vs_fp32_prepared:
            m_shadows_mp = add_shadow_loggers(
                'a', copy.deepcopy(m), 'b', copy.deepcopy(mp),
                OutputLogger, should_log_inputs=should_log_inputs)
        mp_shadows_mq = add_shadow_loggers(
            'a', mp, 'b', mq, OutputLogger,
            should_log_inputs=should_log_inputs)

        if prepared_expected_node_occurrence:
            if compare_fp32_vs_fp32_prepared:
                self.checkGraphModuleNodes(
                    m_shadows_mp, expected_node_occurrence=prepared_expected_node_occurrence)
            self.checkGraphModuleNodes(
                mp_shadows_mq, expected_node_occurrence=prepared_expected_node_occurrence)

        if not skip_scripting:
            if compare_fp32_vs_fp32_prepared:
                m_shadows_mp = torch.jit.script(m_shadows_mp)
            mp_shadows_mq = torch.jit.script(mp_shadows_mq)

        # calibrate
        if compare_fp32_vs_fp32_prepared:
            m_shadows_mp(*data)
        mp_shadows_mq(*data)

        # check activation result correctness
        results = []
        models = (m_shadows_mp, mp_shadows_mq) if \
            compare_fp32_vs_fp32_prepared else (mp_shadows_mq,)
        for model in models:
            act_compare_dict = extract_shadow_logger_info(
                model, OutputLogger, 'b')
            if results_len is not None:
                self.assertTrue(
                    len(act_compare_dict) == results_len,
                    f"expected len {results_len}, got len {len(act_compare_dict)}")
            self.assert_ns_compare_dict_valid(act_compare_dict)
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_normalized_l2_error, 'l2_error')
            extend_logger_results_with_comparison(
                act_compare_dict, 'a', 'b', compute_cosine_similarity,
                'cosine_similarity')
            results.append(act_compare_dict)
        return results


class TestFXNumericSuiteCoreAPIs(FXNumericSuiteQuantizationTestCase):
    # [CLASS DOC] Tests of the core numeric suite APIs: weight extraction, output
    # loggers, shadow loggers and result comparison, on toy models, user modules
    # and cuda. Most tests need FBGEMM.

    @skipIfNoFBGEMM
    def test_extract_weights_mod_ptq(self):
        # [FUNCTION DOC] Weight extraction on a ptq model with all conv and linear
        # fusion variants; expects 14 results.
        m = AllConvAndLinearFusionModules().eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(m, example_inputs, results_len=14)

    @skipIfNoFBGEMM
    def test_extract_weights_mod_qat(self):
        # [FUNCTION DOC] Same as the module ptq test but with qat preparation; expects
        # 14 results.
        m = AllConvAndLinearFusionModules().train()
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(
            m, example_inputs, results_len=14, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_linear_fun_ptq(self):
        # [FUNCTION DOC] Weight extraction for functional linear ptq model; expects 2
        # results.
        m = LinearReluLinearFunctional().eval()
        example_inputs = (torch.randn(1, 4),)
        self._test_extract_weights(m, example_inputs, results_len=2)

    @skipIfNoFBGEMM
    def test_extract_weights_linear_fun_qat(self):
        # [FUNCTION DOC] Weight extraction for functional linear with qat; expects 2
        # results.
        m = LinearReluLinearFunctional().train()
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 4),)
        self._test_extract_weights(
            m, example_inputs, results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_conv_fun_ptq(self):
        # [FUNCTION DOC] Weight extraction for the functional conv model (1d, 2d, 3d)
        # with ptq; expects 6 results.
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).eval()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(m, example_inputs, results_len=6)

    @skipIfNoFBGEMM
    def test_extract_weights_conv_fun_qat(self):
        # [FUNCTION DOC] Same as the functional conv ptq test but with qat preparation;
        # expects 6 results.
        w1d = torch.randn(1, 1, 1)
        w2d = torch.randn(1, 1, 1, 1)
        w3d = torch.randn(1, 1, 1, 1, 1)
        b1d = torch.randn(1)
        b2d = torch.randn(1)
        b3d = torch.randn(1)
        m = AllConvFunctional(w1d, w2d, w3d, b1d, b2d, b3d).train()
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        self._test_extract_weights(
            m, example_inputs, results_len=6, qconfig_dict=qconfig_dict, prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_extract_weights_dynamic(self):
        # [FUNCTION DOC] Weight extraction for a dynamically quantized linear; expects 1
        # result.
        # TODO(future PR): add Linear-ReLU, after #55393 is fixed.
        m = nn.Sequential(nn.Linear(1, 1)).eval()
        qconfig_dict = {
            'object_type': [
                (nn.Linear, default_dynamic_qconfig),
            ],
        }
        example_inputs = (torch.randn(1, 1),)
        self._test_extract_weights(m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_extract_weights_fqn(self):
        # [FUNCTION DOC] Checks that extracted weights are keyed by layer name and carry
        # the fully qualified module name ("0.0" and "1") for both models.
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        results = extract_weights('a', mp, 'b', mq)
        fqn_a_0 = results['_0_0']['weight']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['weight']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        fqn_a_1 = results['_1']['weight']['a'][0]['fqn']
        fqn_b_1 = results['_1']['weight']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    def _test_match_activations_mod_impl(self, prepare_fn=prepare_fx):
        # [FUNCTION DOC] HELPER: Runs the activation matching driver on a quant stub plus
        # two convs with expected two logger nodes and two results; uses the qat qconfig
        # when prepare_fn is the qat one.
        m = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = None
        if prepare_fn is prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(2, 1, 2, 2),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, qconfig_dict=qconfig_dict, prepare_fn=prepare_fn)

    @skipIfNoFBGEMM
    def test_match_activations_mod_ptq(self):
        # [FUNCTION DOC] Module activation matching with ptq preparation.
        self._test_match_activations_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_mod_qat(self):
        # [FUNCTION DOC] Module activation matching with qat preparation.
        self._test_match_activations_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_match_activations_fun_impl(self, prepare_fn=prepare_fx):
        # [FUNCTION DOC] HELPER: Runs the activation matching driver on the functional
        # linear-relu-linear model, expecting two logger nodes and two results.
        m = LinearReluLinearFunctional().eval()
        qconfig_dict = None
        if prepare_fn is prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self._test_match_activations(
            m, (torch.randn(4, 4),),
            prepared_expected_node_occurrence=expected_occurrence,
            results_len=2, prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_match_activations_fun_ptq(self):
        # [FUNCTION DOC] Function activation matching with ptq preparation.
        self._test_match_activations_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_match_activations_fun_qat(self):
        # [FUNCTION DOC] Function activation matching with qat preparation.
        self._test_match_activations_fun_impl(prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_match_activations_meth_ptq(self):
        # [FUNCTION DOC] Verifies activation loggers work on a tensor method call model;
        # expects one result.
        """
        Verify that add_loggers works on methods
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model calling the sigmoid tensor method.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sigmoid method applied to x.
                x = x.sigmoid()
                return x

        m = M().eval()
        res = self._test_match_activations(
            m, (torch.randn(4, 4),),
            results_len=1)

    @skipIfNoFBGEMM
    def test_match_activations_fqn(self):
        # [FUNCTION DOC] Checks output logger results carry fully qualified names for both
        # models, matching the module path ("0.0" and "1").
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        mp_ns, mq_ns = add_loggers('a', mp, 'b', mq, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        mp_ns(datum)
        mq_ns(datum)

        results = extract_logger_info(mp_ns, mq_ns, OutputLogger, 'b')
        fqn_a_0 = results['_0_0']['node_output']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        fqn_a_1 = results['_1']['node_output']['a'][0]['fqn']
        fqn_b_1 = results['_1']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    def _test_add_shadow_loggers_mod_impl(self, prepare_fn=prepare_fx):
        # [FUNCTION DOC] HELPER: Runs the shadow activation driver on a two conv model
        # expecting two results; qat qconfig used for the qat preparation function.
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_dict = None
        if prepare_fn is prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        res = self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),), results_len=2,
            prepare_fn=prepare_fn, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_ptq(self):
        # [FUNCTION DOC] Shadow loggers on modules with ptq.
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_mod_qat(self):
        # [FUNCTION DOC] Shadow loggers on modules with qat.
        self._test_add_shadow_loggers_mod_impl(prepare_fn=prepare_qat_fx)

    def _test_add_shadow_loggers_fun_impl(self, prepare_fn=prepare_fx):
        # [FUNCTION DOC] HELPER: Runs the shadow activation driver on the functional
        # linear-relu-linear model expecting two results.
        m = LinearReluLinearFunctional()
        qconfig_dict = None
        if prepare_fn is prepare_qat_fx:
            qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),), results_len=2, prepare_fn=prepare_fn,
            qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_ptq(self):
        # [FUNCTION DOC] Shadow loggers on functions with ptq.
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_fun_qat(self):
        # [FUNCTION DOC] Shadow loggers on functions with qat.
        self._test_add_shadow_loggers_fun_impl(prepare_fn=prepare_qat_fx)

    @skipIfNoFBGEMM
    def test_add_shadow_loggers_meth_ptq(self):
        # [FUNCTION DOC] Verifies shadow loggers do not crash on a method call model;
        # expects zero results since sigmoid shadowing is not supported yet.
        """
        Verify that add_loggers works on methods
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model calling the sigmoid tensor method.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sigmoid method applied to x.
                x = x.sigmoid()
                return x

        m = M().eval()
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            # For now, sigmoid is not supported for shadowing because the dtype
            # inference for it is not implemented yet. So, this is just testing
            # that shadowing models with method calls does not crash.
            results_len=0)

    @skipIfNoFBGEMM
    def test_shadow_activations_fqn(self):
        # [FUNCTION DOC] Checks shadow logger results carry the fully qualified module
        # names ("0.0" and "1") for both models.
        m = nn.Sequential(
            nn.Sequential(nn.Conv2d(1, 1, 1)),
            nn.Conv2d(1, 1, 1),
        ).eval()
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mq = convert_fx(copy.deepcopy(mp))
        mp_shadows_mq = add_shadow_loggers('a', mp, 'b', mq, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        mp_shadows_mq(datum)

        results = extract_shadow_logger_info(mp_shadows_mq, OutputLogger, 'b')
        fqn_a_0 = results['_0_0']['node_output']['a'][0]['fqn']
        fqn_b_0 = results['_0_0']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_0 == '0.0' and fqn_a_0 == fqn_b_0)
        fqn_a_1 = results['_1']['node_output']['a'][0]['fqn']
        fqn_b_1 = results['_1']['node_output']['b'][0]['fqn']
        self.assertTrue(fqn_a_1 == '1' and fqn_a_1 == fqn_b_1)

    @skipIfNoFBGEMM
    def test_logging_inputs(self):
        # [FUNCTION DOC] Verifies shadow logging with input logging enabled works on a
        # conv followed by cat model; expects one result.
        """
        Verifies that logging inputs works correctly
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model with a conv followed by concatenating the output
            # with itself.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 1x1 conv layer.
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv, then concatenates the result with
                # itself along the first dimension; returns it.
                x = self.conv(x)
                x = torch.cat([x, x], dim=0)
                return x

        m = M().eval()
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=1,
            should_log_inputs=True)

    @skipIfNoFBGEMM
    def test_ops_with_same_fp32_and_int8_signature(self):
        # [FUNCTION DOC] Verifies matching works for ops that share an op signature
        # for float and int8 tensors (max pool and relu); expects two results.
        """
        Verifies that we can match pairs of ops which have the same aten
        signature for fp32 and int8 tensors.
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of a 2d max pool followed by functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a max pooling layer with kernel size 2.
                super().__init__()
                self.max_pool_2d = nn.MaxPool2d(2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies max pool then relu; returns the result.
                x = self.max_pool_2d(x)
                x = F.relu(x)
                return x

        m = M().eval()
        self._test_match_activations(
            m, (torch.randn(1, 1, 2, 2),),
            results_len=2)

    @skipIfNoFBGEMM
    def test_add_mul_inputs_activations(self):
        # [FUNCTION DOC] Activation matching with input logging on the add/mul model
        # with scalar operands; expects six results.
        m = AddMulFunctional().eval()
        res = self._test_match_activations(
            m, (torch.randn(2, 2), torch.randn(2, 2)),
            results_len=6, should_log_inputs=True)

    @skipIfNoFBGEMM
    def test_linear_fp16_weights(self):
        # [FUNCTION DOC] Weight extraction for linear-relu with float16 static qconfig;
        # expects one result.
        qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
        m = LinearReluFunctional().eval()
        example_inputs = (torch.randn(1, 4),)
        self._test_extract_weights(m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_linear_fp16_activations(self):
        # [FUNCTION DOC] Activation matching with float16 static qconfig, with and
        # without input logging; expects one logger node without and two with input
        # logging, and one result.
        for should_log_inputs in (True, False):
            qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
            m = LinearReluFunctional().eval()
            num_loggers = 2 if should_log_inputs else 1
            expected_occurrence = {
                ns.call_module(OutputLogger): num_loggers,
            }
            res = self._test_match_activations(
                m, (torch.randn(4, 4),),
                prepared_expected_node_occurrence=expected_occurrence,
                results_len=1,
                qconfig_dict=qconfig_dict,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_linear_fp16_shadow_activations(self):
        # [FUNCTION DOC] Shadow activation matching with float16 static qconfig, with
        # and without input logging; expects two logger nodes without and four with
        # input logging, and one result.
        for should_log_inputs in (True, False):
            qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
            m = LinearReluFunctional().eval()
            num_loggers = 4 if should_log_inputs else 2
            expected_occurrence = {
                ns.call_module(OutputLogger): num_loggers,
            }
            res2 = self._test_match_shadow_activations(
                m, (torch.randn(4, 4),),
                prepared_expected_node_occurrence=expected_occurrence,
                results_len=1,
                qconfig_dict=qconfig_dict,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_linear_fp16_vs_linear_fp16_shadow_activations(self):
        # [FUNCTION DOC] Shadows one converted float16 linear with another converted
        # copy via the internal shadow function and checks one valid shadow result.
        m = LinearFunctional().eval()
        qconfig_dict = {'': torch.ao.quantization.float16_static_qconfig}
        example_inputs = (torch.randn(1, 4),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mq1 = convert_fx(copy.deepcopy(mp))
        mq2 = convert_fx(copy.deepcopy(mp))
        mq1_shadows_mq2 = _add_shadow_loggers_impl(
            'a', mq1, 'b', mq2, OutputLogger, should_log_inputs=False)
        mq1_shadows_mq2(torch.randn(4, 4))
        act_compare_dict = extract_shadow_logger_info(
            mq1_shadows_mq2, OutputLogger, 'b')
        self.assertTrue(len(act_compare_dict) == 1)
        self.assert_ns_compare_dict_valid(act_compare_dict)


    @skipIfNoFBGEMM
    def test_op_with_either_fp32_or_int8_input(self):
        # [FUNCTION DOC] Verifies shadowing does not crash with ops taking either fp32
        # or int8 inputs (relu module and function); expects zero results.
        """
        Verify that shadowing works with ops which accept either fp32 or
        int8 inputs.
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model applying a relu module then functional relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a relu module.
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the relu module then functional relu; returns
                # the result.
                x = self.relu(x)
                x = F.relu(x)
                return x

        m = M()
        res = self._test_match_shadow_activations(
            m, (torch.randn(4, 4),),
            # Note: shadowing relu by itself is currently not supported,
            # this test is just testing that it does not crash
            results_len=0)

    def _test_int8_shadows_int8_impl(self, m):
        # [FUNCTION DOC] HELPER: Prepares the given model with the default qconfig,
        # runs calibration, converts two copies to int8 and shadows one by the other,
        # runs data and checks exactly one valid shadow result. Consumes the model.
        """
        Verify that shadowing works where both modules are int8
        """
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(4, 1, 4, 4),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mq1 = convert_fx(copy.deepcopy(mp))
        mq2 = convert_fx(mp)
        mq1_shadows_mq2 = add_shadow_loggers('a', mq1, 'b', mq2, OutputLogger)
        mq1_shadows_mq2(torch.randn(4, 1, 4, 4))
        act_compare_dict = extract_shadow_logger_info(
            mq1_shadows_mq2, OutputLogger, 'b')
        self.assertTrue(len(act_compare_dict) == 1)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_int8_shadows_int8_mod(self):
        # [FUNCTION DOC] Int8 shadows int8 for a single conv module.
        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        self._test_int8_shadows_int8_impl(m)

    @skipIfNoFBGEMM
    def test_int8_shadows_int8_fun(self):
        # [FUNCTION DOC] Int8 shadows int8 for a functional linear model.
        m = LinearFunctional().eval()
        self._test_int8_shadows_int8_impl(m)

    @skipIfNoFBGEMM
    def test_user_module_scriptable(self):
        # [FUNCTION DOC] Verifies a prepared model containing a non-traceable user
        # module that returns a tuple can be instrumented with loggers (after marking
        # the type unmatchable) and that the graph modules can be scripted.
        # Logging of the output of this class is not supported, because it is
        # neither a tensor or an RNN return type.
        class M1(nn.Module):
            # [CLASS DOC] HELPER: User module returning a tuple of the input scaled by two
            # and by four; its output cannot be logged.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of x times two and x times four.
                x1 = x * 2
                x2 = x * 4
                return (x1, x2)

        class M2(nn.Module):
            # [CLASS DOC] HELPER: Model wrapping the tuple-returning user module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the wrapped inner user module.
                super().__init__()
                self.m1 = M1()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the inner module, unpacks and returns the two
                # outputs.
                x1, x2 = self.m1(x)
                return x1, x2

        m = M2().eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        prepare_custom_config_dict = {
            'non_traceable_module_class': [M1],
        }
        example_inputs = (torch.randn(1),)
        mp1 = prepare_fx(
            m,
            qconfig_dict,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        mp2 = copy.deepcopy(mp1)
        unmatchable_types_map = get_unmatchable_types_map()
        unmatchable_types_map['mods_unmatchable'].add(M1)
        mp1_ns, mp2_ns = _add_loggers_impl(
            'a', mp1, 'b', mp2, OutputLogger, should_log_inputs=False,
            unmatchable_types_map=unmatchable_types_map)

        # Scripting a model with loggers should succeed. If it fails because of
        # incorrect dtypes, we can blocklist the associated types from being instrumented.
        mp1_ns_scripted = torch.jit.script(mp1_ns)
        mp2_ns_scripted = torch.jit.script(mp2_ns)

    @skipIfNoFBGEMM
    def test_user_module(self):
        # [FUNCTION DOC] Checks that with a non-traceable user module in the model
        # weight extraction does not crash, that unshadowed loggers (including input
        # logging) are only added around the known linear (two loggers per model) and
        # shadowed loggers likewise only for known typed nodes (four loggers).
        """
        For user defined modules,
        1. weight extraction should not crash
        2. unshadowed activations should only have loggers for known types
        3. shadowed activations should only have loggers for known types with
             known dtypes
        """
        class UserModule(nn.Module):
            # [CLASS DOC] HELPER: Identity user module treated as non-traceable.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input unchanged.
                return x

        class M(nn.Module):
            # [CLASS DOC] HELPER: Model with a linear layer followed by the identity user
            # module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 1x1 linear layer and a user module.
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.user_module = UserModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and then the user module;
                # returns the result.
                x = self.linear(x)
                x = self.user_module(x)
                return x

        m = M().eval()

        # quantize without tracing through UserModule
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        prepare_custom_config_dict = {'non_traceable_module_name': ['user_module']}
        example_inputs = (torch.randn(1, 1, 1),)
        mp = prepare_fx(
            m,
            qconfig_dict,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config_dict)
        mp(*example_inputs)
        mq = convert_fx(copy.deepcopy(mp))

        # weight extraction should not crash
        weights = _extract_weights_impl('fp32_prepared', mp, 'int8', mq)

        # unshadowed activations should have loggers

        # add loggers, without retracing
        # note: converting again because we cannot copy a quantized linear
        mp_ns, mq_ns = _add_loggers_impl(
            'fp32_prepared', copy.deepcopy(mp), 'int8',
            convert_fx(copy.deepcopy(mp)), OutputLogger,
            should_log_inputs=True)
        # both fp32 and int8 models should have 2 loggers each, 2 for I/O
        # of linear, and 0 for I/O of user_module
        unshadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 2,
        }
        self.checkGraphModuleNodes(
            mp_ns, expected_node_occurrence=unshadowed_expected_occurrence)
        self.checkGraphModuleNodes(
            mq_ns, expected_node_occurrence=unshadowed_expected_occurrence)

        # shadowed activations should only have loggers for nodes where
        # the types are known and we can do a dtype cast

        # add shadow loggers, without retracing
        mp_shadows_mq_ns = _add_shadow_loggers_impl(
            'fp32_prepared', mp, 'int8', mq, OutputLogger,
            should_log_inputs=True)
        # 4 loggers for I/O of linear, 0 loggers for I/O of user_module
        shadowed_expected_occurrence = {
            ns.call_module(OutputLogger): 4,
        }
        self.checkGraphModuleNodes(
            mp_shadows_mq_ns, expected_node_occurrence=shadowed_expected_occurrence)

    def test_op_io_dtype_coverage(self):
        # [FUNCTION DOC] Checks that every op quantization handles has input and output
        # dtype handling in the io type maps: static module and function mappings,
        # dynamic module mappings (minus skips), and ops of each quantize handler class.
        # KNOWN ISSUE: the final else branch ends an assertTrue argument list with an
        # or of a message string, so the assertion can never fail for that case.
        # KNOWN ISSUE: several messages use an f-string with a stray f before the brace.
        """
        Tests that all the ops quantization cares about have input and output
        dtypes defined.
        """
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        type_a_related_to_b = \
            get_type_a_related_to_b(base_name_to_sets_of_related_ops)

        # TODO(future PR): clean this up
        node_type_to_io_type_map = get_node_type_to_io_type_map()
        FUNS_IO_TYPE_FP32 = node_type_to_io_type_map['funs_io_type_fp32']
        FUNS_IO_TYPE_INT8 = node_type_to_io_type_map['funs_io_type_int8']
        FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['funs_io_type_fp32_or_int8']
        MODS_IO_TYPE_FP32 = node_type_to_io_type_map['mods_io_type_fp32']
        MODS_IO_TYPE_INT8 = node_type_to_io_type_map['mods_io_type_int8']
        MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['mods_io_type_fp32_or_int8']
        METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map['meths_io_type_fp32_or_int8']

        unmatchable_types_map = get_unmatchable_types_map()
        FUNS_UNMATCHABLE = unmatchable_types_map['funs_unmatchable']
        MODS_UNMATCHABLE = unmatchable_types_map['mods_unmatchable']
        METHS_UNMATCHABLE = unmatchable_types_map['meths_unmatchable']

        # 1. check static quant module mappings
        static_quant_mod_mappings = get_default_static_quant_module_mappings()
        for fp32_type, int8_type in static_quant_mod_mappings.items():
            types_to_skip = (
                torch.ao.quantization.QuantStub,
                torch.ao.quantization.DeQuantStub,
                nnq.FloatFunctional,
                # TODO(future PR): look into whether shadowing embeddings
                # makes sense
                nn.Embedding,
                nn.EmbeddingBag,
                # the ConvTranspose3d swap is not implemented in FX Graph
                # mode quantization yet
                nn.ConvTranspose3d,
                # the GroupNorm swap is not implemented in FX Graph
                # mode quantization yet
                nn.GroupNorm,
                # nnq.ReLU6 is no longer swapped, because nn.ReLU6 can
                # take quantized inputs
                nn.ReLU6,
            )
            if fp32_type in types_to_skip:
                continue
            self.assertTrue(
                fp32_type in MODS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type}")
            self.assertTrue(
                int8_type in MODS_IO_TYPE_INT8,
                f"missing IO type handling for f{int8_type}")

        # 2. check static quant op mappings
        static_quant_fun_mappings = get_default_float_to_quantized_operator_mappings()
        for fp32_type, int8_type in static_quant_fun_mappings.items():
            self.assertTrue(
                fp32_type in FUNS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type}")
            self.assertTrue(
                int8_type in FUNS_IO_TYPE_INT8,
                f"missing IO type handling for f{int8_type}")

        # 3. check dynamic quant mappings
        dynamic_quant_mappings = get_default_dynamic_quant_module_mappings()
        for fp32_type1, fp32_type2 in dynamic_quant_mappings.items():
            # TODO(future PR): verify correct I/O for these and remove from
            # this list.
            types_to_skip = (
                nn.GRUCell,
                nn.GRU,
                nn.LSTMCell,
                nn.RNNCell,
                # TODO(future PR): look into whether shadowing embeddings
                # makes sense
                nn.Embedding,
                nn.EmbeddingBag,
            )
            if fp32_type1 in types_to_skip:
                continue
            self.assertTrue(
                fp32_type1 in MODS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type1}")
            self.assertTrue(
                fp32_type2 in MODS_IO_TYPE_FP32,
                f"missing IO type handling for f{fp32_type2}")

        # 4. go through the ops mapped to each QuantizeHandler type, and verify
        # correctness.
        default_quant_patterns = get_all_quant_patterns()
        for pattern, qhandler_cls in default_quant_patterns.items():
            base_op = None
            if isinstance(pattern, tuple):
                base_op = pattern[-1]
            elif isinstance(pattern, str):
                base_op = pattern
            else:
                base_op = pattern

            if (
                qhandler_cls in (
                    qh.BinaryOpQuantizeHandler,
                    qh.RNNDynamicQuantizeHandler,
                )
            ):
                # TODO(future PR): implement shadowing for binary ops
                # TODO(future PR): implement shadowing for RNN ops
                continue
            elif qhandler_cls == qh.CatQuantizeHandler:
                self.assertTrue(
                    base_op in FUNS_IO_TYPE_FP32_OR_INT8,
                    f"missing IO type handling for {base_op}")
            elif (
                qhandler_cls in (
                    qh.ConvReluQuantizeHandler,
                    qh.LinearReLUQuantizeHandler,
                    qh.BatchNormQuantizeHandler,
                    qh.DefaultNodeQuantizeHandler,
                )
            ):
                self.assertTrue(
                    (base_op in FUNS_IO_TYPE_FP32) or (base_op in MODS_IO_TYPE_FP32),
                    f"missing IO type handling for {base_op}")
            elif (
                qhandler_cls in (
                    qh.FixedQParamsOpQuantizeHandler,
                    qh.CopyNodeQuantizeHandler,
                    qh.GeneralTensorShapeOpQuantizeHandler,
                )
            ):
                if (
                    base_op in FUNS_UNMATCHABLE or
                    base_op in MODS_UNMATCHABLE or
                    base_op in METHS_UNMATCHABLE
                ):
                    continue

                self.assertTrue(
                    (base_op in FUNS_IO_TYPE_FP32_OR_INT8) or
                    (base_op in MODS_IO_TYPE_FP32_OR_INT8) or
                    (base_op in METHS_IO_TYPE_FP32_OR_INT8) or
                    # Softmax has a different signature for the quantized
                    # version, so it does not fit into the cases above.
                    (base_op is torch.nn.Softmax),
                    f"missing IO type handling for {base_op}")
            elif qhandler_cls == qh.EmbeddingQuantizeHandler:
                # embedding shadowing is not implemented, for now
                continue
            else:
                if (
                    base_op in FUNS_UNMATCHABLE or
                    base_op in MODS_UNMATCHABLE or
                    base_op in METHS_UNMATCHABLE
                ):
                    continue
                if qhandler_cls(None, {}).is_general_tensor_value_op():
                    self.assertTrue(
                        (base_op in FUNS_IO_TYPE_FP32_OR_INT8) or
                        (base_op in MODS_IO_TYPE_FP32_OR_INT8) or
                        (base_op in METHS_IO_TYPE_FP32_OR_INT8),
                        f"missing IO type handling for {base_op} using {qhandler_cls}")
                else:
                    self.assertTrue(
                        (base_op in FUNS_IO_TYPE_FP32_OR_INT8) or
                        (base_op in MODS_IO_TYPE_FP32_OR_INT8) or
                        (base_op in METHS_IO_TYPE_FP32_OR_INT8) or
                        (base_op in FUNS_IO_TYPE_FP32) or
                        (base_op in MODS_IO_TYPE_FP32) or
                        f"missing IO type handling for {base_op} using {qhandler_cls}")

    @skipIfNoFBGEMM
    def test_user_defined_function(self):
        # [FUNCTION DOC] Verifies weight extraction, unshadowed loggers (3 results) and
        # shadow loggers (2 results) work with user wrapped functions after registering
        # them as related ops, adding a weight extraction function and io type entries.
        """
        Verify that NS APIs work on user defined functions
        """
        class M1(nn.Module):
            # [CLASS DOC] HELPER: Model of builtin hardswish, sigmoid method and functional
            # linear with a 1x1 weight and bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters w1 (1x1, kaiming initialized) and b1.
                super().__init__()
                self.w1 = nn.Parameter(torch.empty(1, 1))
                self.b1 = nn.Parameter(torch.zeros(1))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies hardswish, sigmoid and linear; returns result.
                x = F.hardswish(x)
                x = x.sigmoid()
                x = F.linear(x, self.w1, self.b1)
                return x

        class M2(nn.Module):
            # [CLASS DOC] HELPER: Same as the builtin model but using wrapped user functions
            # for hardswish, sigmoid and linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters w1 (1x1, kaiming initialized) and b1.
                super().__init__()
                self.w1 = nn.Parameter(torch.empty(1, 1))
                self.b1 = nn.Parameter(torch.zeros(1))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the wrapped hardswish, sigmoid and linear
                # functions; returns the result.
                x = _wrapped_hardswish(x)
                x = _wrapped_sigmoid(x)
                x = _wrapped_linear(x, self.w1, self.b1)
                return x

        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping()
        example_inputs = (torch.randn(1, 1),)
        m1 = prepare_fx(M1().eval(), qconfig_mapping, example_inputs=example_inputs)
        m2 = prepare_fx(M2().eval(), qconfig_mapping, example_inputs=example_inputs)
        data = torch.randn(1, 1)

        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_hardswish, F.hardswish)
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_sigmoid, F.sigmoid)
        add_op_to_sets_of_related_ops(
            base_name_to_sets_of_related_ops, _wrapped_linear, F.linear)

        op_to_type_to_weight_extraction_fn = \
            get_op_to_type_to_weight_extraction_fn()
        op_to_type_to_weight_extraction_fn['call_function'][_wrapped_linear] = \
            torch.ao.ns.fx.weight_utils.get_linear_fun_weight

        # test compare weights
        results = extract_weights(
            'a', m1, 'b', m2,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
            op_to_type_to_weight_extraction_fn=op_to_type_to_weight_extraction_fn)
        self.assertTrue(len(results) == 1)
        self.assertTrue(len(results['_wrapped_linear']['weight']) == 2)

        # test unshadowed activations

        m1_ns, m2_ns = _add_loggers_impl(
            'a', copy.deepcopy(m1), 'b', copy.deepcopy(m2), OutputLogger,
            should_log_inputs=False,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops)

        # calibrate
        m1_ns(data)
        m2_ns(data)

        # check activation result correctness
        act_compare_dict = extract_logger_info(m1_ns, m2_ns, OutputLogger, 'b')
        self.assertTrue(len(act_compare_dict) == 3)
        self.assert_ns_compare_dict_valid(act_compare_dict)

        # test shadowed activations

        node_type_to_io_type_map = get_node_type_to_io_type_map()
        node_type_to_io_type_map['funs_io_type_fp32'].add(_wrapped_hardswish)
        node_type_to_io_type_map['funs_io_type_fp32'].add(_wrapped_sigmoid)

        m2_shadows_m1_ns = _add_shadow_loggers_impl(
            'a', m2, 'b', m1, OutputLogger,
            should_log_inputs=False,
            base_name_to_sets_of_related_ops=base_name_to_sets_of_related_ops,
            node_type_to_io_type_map=node_type_to_io_type_map)

        # calibrate
        m2_shadows_m1_ns(data)

        # check activation result correctness
        act_compare_dict = extract_shadow_logger_info(
            m2_shadows_m1_ns, OutputLogger, 'b')
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_layer_names(self):
        # [FUNCTION DOC] Checks that result keys of weight extraction, logger info and
        # shadow logger info are node names present in the quantized (or shadow) graph.
        m = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid(),
        ).eval()
        qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping("fbgemm")
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = torch.ao.quantization.quantize_fx.prepare_fx(m, qconfig_mapping, example_inputs=example_inputs)
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))

        # extract weights
        results = extract_weights('fp32', mp, 'int8', mq)
        mq_node_names = [node.name for node in mq.graph.nodes]
        for layer_name in results:
            self.assertTrue(layer_name in mq_node_names)

        # match activations
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        mp_ns, mq_ns = add_loggers(
            'fp32', copy.deepcopy(mp), 'int8', mq, OutputLogger)
        data = torch.randn(1, 1, 1, 1)
        mp_ns(data)
        mq_ns(data)
        results = extract_logger_info(mp_ns, mq_ns, OutputLogger, 'int8')
        mq_node_names = [node.name for node in mq_ns.graph.nodes]
        for layer_name in results:
            self.assertTrue(layer_name in mq_node_names)

        # match shadow activations
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        mp_shadows_mq = add_shadow_loggers(
            'fp32', mp, 'int8', mq, OutputLogger)
        mp_shadows_mq(data)
        results = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger, 'int8')
        mq_node_names = [node.name for node in mp_shadows_mq.graph.nodes]
        for layer_name in results:
            self.assertTrue(layer_name in mq_node_names)

    @skipIfNoFBGEMM
    def test_extend_logger_results_with_comparison(self):
        # [FUNCTION DOC] Extracts weights of a two conv model and adds sqnr, l2 error
        # and cosine similarity comparisons; raises an assertion error if any comparison
        # key is missing from the int8 results of a layer.
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = torch.ao.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs)
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))

        # extract weights
        results = extract_weights('fp32', mp, 'int8', mq)
        extend_logger_results_with_comparison(
            results, 'fp32', 'int8', compute_sqnr, 'sqnr_int8_vs_fp32')
        extend_logger_results_with_comparison(
            results, 'fp32', 'int8', compute_normalized_l2_error, 'l2_error_int8_vs_fp32')
        extend_logger_results_with_comparison(
            results, 'fp32', 'int8', compute_cosine_similarity,
            'cosine_similarity_int8_vs_fp32')

        for layer_results in results.values():
            if (
                'sqnr_int8_vs_fp32'
                not in layer_results['weight']['int8'][0]
            ):
                raise AssertionError(
                    f"'sqnr_int8_vs_fp32' not found in layer results: {layer_results['weight']['int8'][0].keys()}"
                )
            if (
                'l2_error_int8_vs_fp32'
                not in layer_results['weight']['int8'][0]
            ):
                raise AssertionError(
                    f"'l2_error_int8_vs_fp32' not found in layer results: {layer_results['weight']['int8'][0].keys()}"
                )
            if (
                'cosine_similarity_int8_vs_fp32'
                not in layer_results['weight']['int8'][0]
            ):
                raise AssertionError(
                    f"'cosine_similarity_int8_vs_fp32' not found in layer results: {layer_results['weight']['int8'][0].keys()}"
                )

    @skipIfNoFBGEMM
    def test_int8_shadows_fp32_simple(self):
        # [FUNCTION DOC] Shadows a quantized model with its float prepared version and
        # checks scale and zero point attributes saved on the shadow module equal those
        # of the quantized reference; also checks two valid results after a run.
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1), nn.ReLU()).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = torch.ao.quantization.quantize_fx.prepare_fx(
            m, qconfig_dict, example_inputs=example_inputs)
        mp(torch.randn(1, 1, 1, 1))
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        mq_ref = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        mp_shadows_mq = add_shadow_loggers(
            'int8', mq, 'fp32', mp, OutputLogger)

        # verify that scale and zp were extracted correctly

        # for the first op, the scale+zp live as attributes on the module
        scale_0 = mp_shadows_mq._0_input_scale_0
        scale_0_ref = getattr(mq_ref, '0_input_scale_0')
        self.assertEqual(scale_0, scale_0_ref)
        zp_0 = mp_shadows_mq._0_input_zero_point_0
        zp_0_ref = getattr(mq_ref, '0_input_zero_point_0')
        self.assertEqual(zp_0, zp_0_ref)

        # for the second op, the scale and zp of input to second op
        # must equal to scale and zp of output of first op
        scale_1 = mp_shadows_mq._1_input_scale_0
        scale_1_ref = getattr(mq_ref, '0').scale
        self.assertEqual(scale_1, scale_1_ref)
        zp_1 = mp_shadows_mq._1_input_zero_point_0
        zp_1_ref = getattr(mq_ref, '0').zero_point
        self.assertEqual(zp_1, zp_1_ref)

        # verify running data works
        mp_shadows_mq(torch.randn(1, 1, 1, 1))
        act_compare_dict = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger, 'fp32')
        self.assertTrue(len(act_compare_dict) == 2)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_int8_shadows_fp32_coverage(self):
        # [FUNCTION DOC] Smoke test of int8 shadows fp32 over adaptive pooling, conv,
        # mul, add and relu; expects three valid results.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with adaptive average pool, a conv used three times,
            # mul, add and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates an adaptive average pool and a 1x1 conv.
                super().__init__()
                self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)
                self.conv = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Pools, then applies the same conv, a self multiply, conv
                # again, self add, relu and conv a third time; returns the result.
                x = self.adaptive_avg_pool(x)
                # input qparams of conv will be input qparams of adaptive_avg_pool
                x = self.conv(x)
                x = torch.mul(x, x)
                x = self.conv(x)
                x = torch.add(x, x)
                x = F.relu(x)
                x = self.conv(x)
                return x

        m = M().eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mq = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        mq_ref = torch.ao.quantization.quantize_fx.convert_fx(copy.deepcopy(mp))
        mp_shadows_mq = add_shadow_loggers(
            'int8', mq, 'fp32', mp, OutputLogger)
        mp_shadows_mq(torch.randn(1, 1, 1, 1))
        act_compare_dict = extract_shadow_logger_info(
            mp_shadows_mq, OutputLogger, 'fp32')
        self.assertTrue(len(act_compare_dict) == 3)
        self.assert_ns_compare_dict_valid(act_compare_dict)

    @skipIfNoFBGEMM
    def test_loggers_preserve_qat_numerics(self):
        # [FUNCTION DOC] Checks that adding output loggers does not change the outputs
        # of qat prepared (observers disabled) and converted models.
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1))
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_qat_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mc = convert_fx(copy.deepcopy(mp))
        mp.apply(torch.ao.quantization.disable_observer)

        ref_fp32 = mp(*example_inputs)
        ref_int8 = mc(*example_inputs)

        mp_ns, mc_ns = add_loggers('fp32', mp, 'int8', mc, OutputLogger)
        ref_fp32_ns = mp_ns(*example_inputs)
        ref_int8_ns = mc_ns(*example_inputs)
        self.assertEqual(ref_fp32, ref_fp32_ns)
        self.assertEqual(ref_int8, ref_int8_ns)

    @skipIfNoFBGEMM
    def test_shadow_loggers_preserve_qat_numerics(self):
        # [FUNCTION DOC] Checks that adding shadow loggers does not change the output
        # of the qat prepared model that serves as the shadowed one.
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1))
        qconfig_dict = {'': torch.ao.quantization.get_default_qat_qconfig('fbgemm')}
        example_inputs = (torch.randn(1, 1, 1, 1),)
        mp = prepare_qat_fx(m, qconfig_dict, example_inputs=example_inputs)
        mp(*example_inputs)
        mc = convert_fx(copy.deepcopy(mp))
        mp.apply(torch.ao.quantization.disable_observer)

        ref_fp32 = mp(*example_inputs)
        ref_int8 = mc(*example_inputs)

        mc_shadows_mp = add_shadow_loggers('int8', mc, 'fp32', mp, OutputLogger)
        ref_shadow = mc_shadows_mp(*example_inputs)
        self.assertEqual(ref_fp32, ref_shadow)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_extract_weights_cuda(self):
        # [FUNCTION DOC] Weight extraction between two float cuda models with sqnr
        # comparison; skipped without cuda.
        # Note: this is not using quantization because quantized kernels do not
        # work on cuda yet.
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        results = extract_weights('a', m1, 'b', m2)
        extend_logger_results_with_comparison(
            results, 'a', 'b', compute_sqnr, 'sqnr')
        self.assert_ns_compare_dict_valid(results)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_add_loggers_cuda(self):
        # [FUNCTION DOC] Output loggers on two float cuda models with a cuda input and
        # sqnr comparison; skipped without cuda.
        # Note: this is not using quantization because quantized kernels do not
        # work on cuda yet.
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m1_ns, m2_ns = add_loggers('a', m1, 'b', m2, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        datum = datum.cuda()

        m1_ns(datum)
        m2_ns(datum)

        act_compare_dict = extract_logger_info(m1_ns, m2_ns, OutputLogger, 'b')
        extend_logger_results_with_comparison(
            act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_add_shadow_loggers_cuda(self):
        # [FUNCTION DOC] Shadow loggers on two float cuda models with a cuda input and
        # sqnr comparison; skipped without cuda.
        # Note: this is not using quantization because quantized kernels do not
        # work on cuda yet.
        m1 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m2 = nn.Sequential(nn.Conv2d(1, 1, 1)).cuda()
        m1_shadows_m2 = add_shadow_loggers('a', m1, 'b', m2, OutputLogger)
        datum = torch.randn(1, 1, 1, 1)
        datum = datum.cuda()

        m1_shadows_m2(datum)

        act_compare_dict = extract_shadow_logger_info(m1_shadows_m2, OutputLogger, 'b')
        extend_logger_results_with_comparison(
            act_compare_dict, 'a', 'b', compute_sqnr, 'sqnr')

    def test_fp16_shadows_fp32(self):
        # [FUNCTION DOC] Smoke test that a float16 reference model can shadow a float
        # model; passes if no exception is raised.
        m = LinearReluFunctional().eval()
        example_inputs = (torch.randn(1, 4),)
        qconfig_dict = {"": torch.ao.quantization.float16_static_qconfig}
        mp = prepare_fx(copy.deepcopy(m), qconfig_dict, example_inputs=example_inputs)
        mq = convert_to_reference_fx(mp)
        mq_shadows_m = add_shadow_loggers('a', mq, 'b', m, OutputLogger)

    def test_mul_add_cat_stack_skips_shadowing(self):
        # [FUNCTION DOC] Checks that mul, add, cat and stack are skipped by shadowing:
        # expects zero results.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model using mul, add, cat and stack in several forms.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies multiply and add in operator and function form,
                # then cat and stack of single element lists; returns the result.
                x = x * x
                x = torch.mul(x, x)
                x = x + x
                x = torch.add(x, x)
                x = torch.cat([x])
                x = torch.stack([x])
                return x

        m = M().eval()
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=0)

    def test_op_with_only_kwargs_skips_shadowing(self):
        # [FUNCTION DOC] Checks that cat and stack called only with keyword arguments are
        # skipped by shadowing; expects zero results.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model calling cat and stack with a keyword argument only.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies cat then stack using keyword-only arguments;
                # returns the result.
                x = torch.cat(tensors=[x])
                x = torch.stack(tensors=[x])
                return x

        m = M().eval()
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=0)

    def test_unsupported_op_copy_skips_shadowing(self):
        # [FUNCTION DOC] Checks that shadowing skips a call function node it cannot copy
        # (layer norm with a shape argument) without crashing; expects zero results.
        """
        Copying a `call_function` node is not implemented, test that this
        does not crash shadowing but instead skips the node.
        """
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model applying functional layer norm.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns layer norm of x over all dimensions after the
                # first.
                # the second argument leads to attempting to copy a
                # call_function node
                x = F.layer_norm(x, x.shape[1:])
                return x

        m = M().eval()
        self._test_match_shadow_activations(
            m, (torch.randn(1, 1, 4, 4),),
            results_len=0)

    def test_linear_kwargs_shadow(self):
        # [FUNCTION DOC] Shadows a float traced model with its copy where linear is
        # called with keyword arguments; expects one result.

        class M(nn.Module):
            # [CLASS DOC] HELPER: Model with a functional linear called with keywords.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters w1 (4x4, kaiming initialized) and b1.
                super().__init__()
                self.w1 = nn.Parameter(torch.empty(4, 4))
                self.b1 = nn.Parameter(torch.zeros(4))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns linear with keyword arguments input, weight,
                # bias.
                x = F.linear(input=x, weight=self.w1, bias=self.b1)
                return x

        # note: FX graph mode quantization does not have good support
        # for kwargs-only right now, so we pass in two unquantized
        # models
        m = M().eval()
        mt = torch.fx.symbolic_trace(m)
        mt_copy = copy.deepcopy(mt)

        mt_shadows_mt_copy = add_shadow_loggers(
            'a', mt, 'b', mt_copy, OutputLogger)

        mt_shadows_mt_copy(torch.randn(4, 4))
        act_compare_dict = extract_shadow_logger_info(
            mt_shadows_mt_copy, OutputLogger, 'b')
        self.assertTrue(len(act_compare_dict) == 1)

@skipIfNoQNNPACK
class TestFXNumericSuiteNShadows(FXNumericSuiteQuantizationTestCase):
    # [CLASS DOC] Tests the "n shadows" workflow (prepare, convert, extract and
    # print comparisons for several qconfig mappings), qconfig multi mapping
    # behaviors and adding loggers. Needs QNNPACK.
    """
    Tests the "n shadows" workflow.
    """

    def _test_impl(self, m, example_input, qconfig_mappings):
        # [FUNCTION DOC] HELPER: Runs the full n shadows flow: checks the model runs,
        # prepares with the native backend config, calibrates twice, converts, enables
        # loggers, runs once, extracts and prints the comparisons. Returns the converted
        # model.
        backend_config = get_native_backend_config()

        # test that input is valid
        _ = m(*example_input)

        msp = prepare_n_shadows_model(
            m, example_input, qconfig_mappings, backend_config)
        # print('msp', msp)

        for _ in range(2):
            msp(*example_input)

        msq = convert_n_shadows_model(msp)

        loggers_set_enabled(msq, True)
        msq(*example_input)

        results = extract_results_n_shadows_model(msq)
        print_comparisons_n_shadows_model(results)
        return msq

    @withQNNPACKBackend
    def test_linear_mod(self):
        # [FUNCTION DOC] Runs the n shadows flow for a single linear with a default
        # global qconfig.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model with one linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 2x2 linear layer fc1.
                super().__init__()
                self.fc1 = nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns fc1 applied to x.
                x = self.fc1(x)
                return x

        m = M().eval()
        example_input = (torch.randn(2, 2),)

        qconfig_mappings = \
            QConfigMultiMapping().set_global([torch.ao.quantization.default_qconfig])
        self._test_impl(m, example_input, qconfig_mappings)

    @withQNNPACKBackend
    def test_linear_relu_mod(self):
        # [FUNCTION DOC] Runs the n shadows flow on two linears and relu with static and
        # dynamic global qconfigs.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of two linear layers followed by relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layers fc1 and fc2 and a relu.
                super().__init__()
                self.fc1 = nn.Linear(2, 2)
                self.fc2 = nn.Linear(2, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, fc2 and relu; returns the result.
                x = self.fc1(x)
                x = self.fc2(x)
                x = self.relu(x)
                return x

        m = M().eval()
        example_input = (torch.randn(2, 2),)

        qconfig_mappings = (
            QConfigMultiMapping().set_global([
                torch.ao.quantization.default_qconfig,
                torch.ao.quantization.default_dynamic_qconfig
            ])
        )
        self._test_impl(m, example_input, qconfig_mappings)

    @withQNNPACKBackend
    def test_conv_bn_relu_mod(self):
        # [FUNCTION DOC] Runs the n shadows flow on a conv, batchnorm, relu model with
        # per tensor and per channel qconfigs.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of conv, batchnorm and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates conv, bn and relu submodules.
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 1)
                self.bn = nn.BatchNorm2d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, bn and relu; returns the result.
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        m = M().eval()
        example_input = (torch.randn(32, 1, 16, 16),)

        qconfig_mappings = QConfigMultiMapping() \
            .set_global([
                torch.ao.quantization.default_qconfig,
                torch.ao.quantization.default_per_channel_qconfig
            ])
        self._test_impl(m, example_input, qconfig_mappings)

    @withQNNPACKBackend
    def test_functions(self):
        # [FUNCTION DOC] Runs the n shadows flow on a model of many functions (sigmoid,
        # linear, relu, add, cat variants, matmul) with the default global qconfig.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model chaining sigmoid, linear (twice), relu, add, cat
            # (three call forms) and matmul.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters w1 (2x2, kaiming initialized) and b1.
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.zeros(2))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the listed functions in order and returns the
                # result.
                x = F.sigmoid(x)
                x = F.linear(x, self.w1, self.b1)
                x = F.linear(x, self.w1[:], self.b1)
                x = F.relu(x)
                x = x + x
                x = torch.cat([x])
                x = torch.cat((x,))
                x = torch.cat(tensors=[x])
                # TODO(future PR): enable layernorm
                # blocked on FX graph mode quant not inserting observer for
                # second arg, if the second arg is a module input
                # x = F.layer_norm(x, x.shape)
                # x = F.layer_norm(x, x.shape[1:])
                # x = x.reshape(1, -1) * 2
                # x = F.layer_norm(x.reshape(1, -1), x.shape[1:])
                x = torch.matmul(x, x.reshape(2, 2))
                x = torch.matmul(x.reshape(2, 2), x.reshape(2, 2))
                # TODO(future PR): enable below after FX graph mode quantization handles
                # it, currently this is not supported
                # x = F.linear(input=x, weight=self.w1, bias=self.b1)
                return x

        m = M().eval()
        example_input = (torch.randn(2, 2),)

        qconfig_mappings = QConfigMultiMapping() \
            .set_global([torch.ao.quantization.default_qconfig])
        self._test_impl(m, example_input, qconfig_mappings)

    @withQNNPACKBackend
    def test_partial_qconfig_mapping(self):
        # [FUNCTION DOC] Runs the n shadows flow with a qconfig mapping that only sets
        # object type qconfigs for linear and relu, leaving other ops unquantized.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of a linear module, functional linear, relu and add.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc, w1 and b1 (2x2 parameters).
                super().__init__()
                self.fc = nn.Linear(2, 2)
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc, functional linear, relu and self add;
                # returns the result.
                x = self.fc(x)
                x = F.linear(x, self.w1, self.b1)
                x = F.relu(x)
                x = x + x
                return x

        m = M().eval()
        example_input = (torch.randn(2, 2),)
        qconfig = torch.ao.quantization.default_qconfig

        qconfig_mappings = QConfigMultiMapping() \
            .set_object_type(F.linear, [qconfig]) \
            .set_object_type(F.relu, [qconfig])
        self._test_impl(m, example_input, qconfig_mappings)

    @withQNNPACKBackend
    def test_logger_enabled_and_save_activations_flags(self):
        # [FUNCTION DOC] Checks output logger and comparison logger counters in a
        # converted n shadows model with save_activations on and off: nothing is stored
        # before running; afterwards one stat and one comparison per logger when saving,
        # and zero stats with one comparison when not saving.
        m = nn.Sequential(nn.Linear(1, 1)).eval()
        example_input = (torch.randn(1, 1),)

        qconfig_mappings = QConfigMultiMapping() \
            .set_global([torch.ao.quantization.default_qconfig])
        backend_config = get_native_backend_config()

        msp = prepare_n_shadows_model(
            m, example_input, qconfig_mappings, backend_config)

        for _ in range(2):
            msp(*example_input)

        def _check_logger_count(model, exp_count_stats, exp_count_comparisons):
            # [FUNCTION DOC] HELPER: Walks all submodules of the model and asserts every
            # output logger has the expected number of stats and every comparison logger has
            # the expected number of comparisons.
            for mod in model.modules():
                if isinstance(mod, OutputLogger):
                    self.assertTrue(
                        len(mod.stats) == exp_count_stats,
                        f'stats: expected {len(mod.stats)} to equal {exp_count_stats}')
                    if isinstance(mod, OutputComparisonLogger):
                        self.assertTrue(
                            len(mod.comparisons) == exp_count_comparisons,
                            f'comparisons: expected {len(mod.comparisons)} to equal {exp_count_comparisons}')

        # check behavior with save_activations enabled
        msq = convert_n_shadows_model(copy.deepcopy(msp))
        loggers_set_enabled(msq, True)
        loggers_set_save_activations(msq, True)
        # after prepare calibration but before convert calibration, loggers
        # should not have anything saved
        _check_logger_count(msq, 0, 0)
        msq(*example_input)
        # loggers should save each item after calibration
        _check_logger_count(msq, 1, 1)

        # check behavior with save_activations disabled
        msq = convert_n_shadows_model(copy.deepcopy(msp))
        loggers_set_enabled(msq, True)
        loggers_set_save_activations(msq, False)
        # after prepare calibration but before convert calibration, loggers
        # should not have anything saved
        _check_logger_count(msq, 0, 0)
        msq(*example_input)
        # stats should be empty, but comparisons should be there
        _check_logger_count(msq, 0, 1)

    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    @withQNNPACKBackend
    def test_mobilenet_v2(self):
        # [FUNCTION DOC] Runs the n shadows flow on torchvision mobilenet v2 with static
        # and dynamic qconfigs; skipped under dynamo or without torchvision.
        import torchvision
        m = torchvision.models.quantization.mobilenet_v2(
            pretrained=False, quantize=False).eval()
        example_input = (torch.randn(1, 3, 224, 224),)

        qconfig_mappings = QConfigMultiMapping() \
            .set_global([torch.ao.quantization.default_qconfig, torch.ao.quantization.default_dynamic_qconfig])

        self._test_impl(m, example_input, qconfig_mappings)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_deduplication(self):
        # [FUNCTION DOC] Checks that setting duplicate qconfigs globally results in one
        # qconfig mapping in the multi mapping.
        # check that insertion deduplicates qconfigs
        qconfig_multi_mapping = QConfigMultiMapping().set_global(
            [torch.ao.quantization.default_qconfig, torch.ao.quantization.default_qconfig]
        )
        self.assertEqual(len(qconfig_multi_mapping.qconfig_mappings_list), 1)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_insert_padding(self):
        # [FUNCTION DOC] Checks that setting a style with fewer qconfigs than a previously
        # set global leaves empty (none) entries in the later mappings at that key.
        # test that inserting a higher priority qconfig style with fewer elements than a lower priority qconfig will
        # result in adding None to the extra QConfigMappings at that same style+key
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_object_type(torch.nn.Linear, [torch.ao.quantization.default_qconfig])
            .set_module_name_regex("fc", [torch.ao.quantization.default_qconfig])
            .set_module_name("fc2", [torch.ao.quantization.default_qconfig])
            .set_module_name_object_type_order(
                "", nn.Linear, 0, [torch.ao.quantization.default_qconfig]
            )
        )

        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].object_type_qconfigs[
                torch.nn.Linear
            ],
            None,
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_regex_qconfigs[
                "fc"
            ],
            None,
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[
                1
            ].module_name_object_type_order_qconfigs[("", nn.Linear, 0)],
            None,
        )

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_retroactive_padding(self):
        # [FUNCTION DOC] Checks that setting global qconfigs after other styles pads
        # the earlier style keys with empty entries in the extra mapping.
        # test that inserting a lower priority qconfig style with more elements thhan lower priority qconfig styles
        # will result in the new QConfigMapping having None at all previously existing styles+keys
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_object_type(torch.nn.Linear, [torch.ao.quantization.default_qconfig])
            .set_module_name_regex("fc", [torch.ao.quantization.default_qconfig])
            .set_module_name("fc2", [torch.ao.quantization.default_qconfig])
            .set_module_name_object_type_order(
                "", nn.Linear, 0, [torch.ao.quantization.default_qconfig]
            )
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
        )

        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].object_type_qconfigs[
                torch.nn.Linear
            ],
            None,
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_regex_qconfigs[
                "fc"
            ],
            None,
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[
                1
            ].module_name_object_type_order_qconfigs[("", nn.Linear, 0)],
            None,
        )

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_end_to_end(self):
        # [FUNCTION DOC] End to end flow with a multi mapping where one module has no
        # qconfig for the dynamic variant: verifies shadow wrappers are quantized or
        # dynamic quantized as expected and a missing wrapper raises an attribute error.
        # test that the prepare/convert_n_shadows_model works as expected
        # with qconfig_multi_mapping and avoids unwanted matches

        m = TwoLayerLinearModel().eval()
        example_input = m.get_example_inputs()

        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_module_name("fc2", [None, torch.ao.quantization.default_qconfig])
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )
        msq = self._test_impl(m, example_input, qconfig_multi_mapping)

        self.checkQuantizedLinear(msq.shadow_wrapper_0_1.mod_0)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_0_2.mod_0, torch.qint8)
        self.checkQuantizedLinear(msq.shadow_wrapper_1_1.mod_0)
        self.assertRaisesRegex(AttributeError, ".*", lambda: msq.shadow_wrapper_1_2)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_from_list(self):
        # [FUNCTION DOC] Same checks as the end to end test, but builds the multi mapping
        # from a list of single qconfig mappings.
        # test QConfigMultiMapping.from_list_qconfig_mapping works as expected

        m = TwoLayerLinearModel().eval()
        example_input = m.get_example_inputs()

        qconfig_mappings_list = [
            QConfigMapping().set_global(torch.ao.quantization.default_qconfig),
            QConfigMapping()
            .set_global(torch.ao.quantization.default_dynamic_qconfig)
            .set_module_name("fc2", torch.ao.quantization.default_qconfig),
        ]

        qconfig_multi_mapping = QConfigMultiMapping().from_list_qconfig_mapping(
            qconfig_mappings_list
        )
        self.assertEqual(
            qconfig_multi_mapping.qconfig_mappings_list[1].module_name_qconfigs["fc2"],
            None,
        )

        msq = self._test_impl(m, example_input, qconfig_multi_mapping)

        self.checkQuantizedLinear(msq.shadow_wrapper_0_1.mod_0)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_0_2.mod_0, torch.qint8)
        self.checkQuantizedLinear(msq.shadow_wrapper_1_1.mod_0)
        self.assertRaisesRegex(AttributeError, ".*", lambda: msq.shadow_wrapper_1_2)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_ordering(self):
        # [FUNCTION DOC] Checks that none entries are ignored when ordering shadow
        # wrappers: two mappings expected, module names assigned among the remaining
        # qconfigs, with quantized and dynamic quantized linears at the expected places.
        # test that the module ordering ignores None

        m = TwoLayerLinearModel().eval()
        example_input = m.get_example_inputs()
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_module_name(
                "fc2",
                [
                    None,
                    torch.ao.quantization.default_dynamic_qconfig,
                    torch.ao.quantization.default_qat_qconfig_v2,
                ],
            )
        )
        self.assertEqual(len(qconfig_multi_mapping.qconfig_mappings_list), 2)
        msq = self._test_impl(m, example_input, qconfig_multi_mapping)

        self.checkQuantizedLinear(msq.shadow_wrapper_0_1.mod_0)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_0_2.mod_0, torch.qint8)
        self.checkDynamicQuantizedLinear(msq.shadow_wrapper_1_1.mod_0, torch.qint8)
        self.checkQuantizedLinear(msq.shadow_wrapper_1_2.mod_0)

    @withQNNPACKBackend
    def test_qconfig_multi_mapping_repr(self):
        # [FUNCTION DOC] Checks that the multi mapping can be turned into a string.
        qconfig_multi_mapping = (
            QConfigMultiMapping()
            .set_global(
                [
                    torch.ao.quantization.default_qconfig,
                    torch.ao.quantization.default_dynamic_qconfig,
                ]
            )
            .set_module_name(
                "fc2",
                [
                    None,
                    torch.ao.quantization.default_dynamic_qconfig,
                    torch.ao.quantization.default_qat_qconfig_v2,
                ],
            )
        )
        self.assertTrue(isinstance(qconfig_multi_mapping.__repr__(), str))

    @withQNNPACKBackend
    def test_custom_functions_and_tracer(self):
        # [FUNCTION DOC] Runs the n shadows flow with a custom tracer (fc2 treated as a
        # leaf), the qat prepare function and a custom convert function that prints a
        # keyword argument; also prints the converted model and result comparisons.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model with two linear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layers fc1 and fc2.
                super().__init__()
                self.fc1 = nn.Linear(2, 2)
                self.fc2 = nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1 then fc2; returns the result.
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        m = M().eval()
        example_inputs = (torch.randn(2, 2),)

        qconfig_mappings = QConfigMultiMapping().set_global(
            [torch.ao.quantization.default_qat_qconfig]
        )

        custom_tracer = torch.ao.quantization.quantize_fx.QuantizationTracer(
            ["fc2"], []
        )

        custom_prepare_fn = torch.ao.quantization.quantize_fx.prepare_qat_fx

        def custom_convert_fn(module, to_print):
            # [FUNCTION DOC] HELPER: Prints the given string, then converts the module with
            # the standard fx convert and returns the converted module.
            print(to_print)
            mod = torch.ao.quantization.quantize_fx.convert_fx(module)
            return mod

        backend_config = get_native_backend_config()

        # test that input is valid
        _ = m(*example_inputs)

        kwargs = {"to_print": "working"}

        msp = prepare_n_shadows_model(
            m,
            example_inputs,
            qconfig_mappings,
            backend_config,
            custom_prepare_fn=custom_prepare_fn,
            custom_prepare_kwargs=None,
            custom_tracer=custom_tracer,
        )

        for _ in range(2):
            msp(*example_inputs)

        msq = convert_n_shadows_model(
            msp, custom_convert_fn=custom_convert_fn, custom_convert_kwargs=kwargs
        )
        print(msq)
        loggers_set_enabled(msq, True)
        msq(*example_inputs)

        results = extract_results_n_shadows_model(msq)
        print_comparisons_n_shadows_model(results)

    def _test_extract_weights_impl(self, m, example_input, qconfig_mapping):
        # [FUNCTION DOC] HELPER: Compares weights of the n shadows model for the given
        # mapping and prints the comparison table. Returns nothing.
        backend_config = get_native_backend_config()
        results = _n_shadows_compare_weights(
            m, example_input, qconfig_mapping, backend_config)
        print_comparisons_n_shadows_model(results)

    @withQNNPACKBackend
    def test_extract_weights_linear(self):
        # [FUNCTION DOC] Weight comparison for four linears with a mix of unquantized,
        # default and per tensor qconfig by position.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of four functional linears with separate weights and
            # biases and a relu after the second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates eight parameters (weights w1 to w4 and biases
                # b1 to b4), initializing each weight with kaiming uniform.
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
                self.w2 = nn.Parameter(torch.randn(2, 2))
                self.b2 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
                self.w3 = nn.Parameter(torch.randn(2, 2))
                self.b3 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
                self.w4 = nn.Parameter(torch.randn(2, 2))
                self.b4 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w4, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear 1, linear 2, relu, linear 3, linear 4;
                # returns the result.
                x = F.linear(x, self.w1, self.b1)
                x = F.linear(x, self.w2, self.b2)
                x = F.relu(x)
                x = F.linear(x, self.w3, self.b3)
                x = F.linear(x, self.w4, self.b4)
                return x

        per_tensor_qconfig = torch.ao.quantization.default_qconfig

        m = M().eval()
        example_input = (torch.randn(2, 2),)
        qconfig_mapping = get_default_qconfig_mapping()
        # test unquantized
        qconfig_mapping.set_module_name_object_type_order(
            '', F.linear, 2, None)
        # test per-tensor
        qconfig_mapping.set_module_name_object_type_order(
            '', F.linear, 3, per_tensor_qconfig)
        self._test_extract_weights_impl(m, example_input, qconfig_mapping)


    def _test_add_loggers_impl(self, m, example_input, qconfig_mapping):
        # [FUNCTION DOC] HELPER: Builds a model with loggers added only (no shadows),
        # calibrates, converts, runs, and compares the logged final output and the float
        # output against references computed with standard prepare and convert. Asserts
        # closeness; returns the converted model.
        backend_config = get_native_backend_config()
        m_copy = copy.deepcopy(m)

        # test that input is valid
        _ = m(*example_input)

        msp = _prepare_n_shadows_add_loggers_model(
            m, example_input, qconfig_mapping, backend_config)
        # print('msp', msp)

        msp(*example_input)

        msq = convert_n_shadows_model(msp)
        # print('msq', msq)

        loggers_set_enabled(msq, True)
        output_fp32 = msq(*example_input)

        results = extract_results_n_shadows_model(msq)
        # print(results)
        # print_comparisons_n_shadows_model(results)

        # get the last quantized output from results
        inner_results = results['model']['node_output']
        last_subgraph = list(inner_results.keys())[-1]
        output_shadow = inner_results[last_subgraph][0]['values'][-1]

        # verify that both fp32 and quantized output matches reference
        output_fp32_ref = m_copy(*example_input)
        mp_ref = prepare_fx(m_copy, qconfig_mapping, example_input)
        for _ in range(2):
            mp_ref(*example_input)
        mq_ref = convert_fx(mp_ref)
        output_shadow_ref = mq_ref(*example_input)
        self.assertTrue(
            torch.allclose(output_fp32, output_fp32_ref),
            f"fp32 comparison: {output_fp32} not close to {output_fp32_ref}")

        # print('shadow', output_shadow.shape, output_shadow)
        # print('shadow_ref', output_shadow_ref.shape, output_shadow_ref)

        self.assertTrue(
            torch.allclose(output_shadow, output_shadow_ref),
            f"shadow comparison: {output_shadow} not close to {output_shadow_ref}")

        return msq

    @withQNNPACKBackend
    def test_add_loggers_linear_mod_quant_quant(self):
        # [FUNCTION DOC] Adds loggers to two stacked linears, both quantized.
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)
        qconfig_mapping = get_default_qconfig_mapping()
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @withQNNPACKBackend
    def test_add_loggers_linear_mod_fp32_quant(self):
        # [FUNCTION DOC] Adds loggers to two linears, first left in float.
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)
        qconfig_mapping = get_default_qconfig_mapping()
        qconfig_mapping.set_module_name('0', None)
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @withQNNPACKBackend
    def test_add_loggers_linear_mod_quant_fp32(self):
        # [FUNCTION DOC] Adds loggers to two linears, second left in float.
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)
        qconfig_mapping = get_default_qconfig_mapping()
        qconfig_mapping.set_module_name('1', None)
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @withQNNPACKBackend
    def test_add_loggers_linear_mod_fp32_fp32(self):
        # [FUNCTION DOC] Adds loggers to two linears, both left in float.
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        example_input = (torch.randn(2, 2),)
        qconfig_mapping = get_default_qconfig_mapping()
        qconfig_mapping.set_module_name('0', None)
        qconfig_mapping.set_module_name('1', None)
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @withQNNPACKBackend
    def test_add_loggers_conv_bn_relu_fusion_quant(self):
        # [FUNCTION DOC] Adds loggers to a conv, batchnorm, relu model, fully quantized
        # fused.
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
        m.eval()
        example_input = (torch.randn(16, 1, 4, 4),)
        qconfig_mapping = get_default_qconfig_mapping()
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @withQNNPACKBackend
    def test_add_loggers_conv_bn_relu_fusion_fp32(self):
        # [FUNCTION DOC] Adds loggers to a conv, batchnorm, relu model with all three
        # left in float.
        m = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1), nn.ReLU())
        m.eval()
        example_input = (torch.randn(16, 1, 4, 4),)
        qconfig_mapping = get_default_qconfig_mapping()
        qconfig_mapping.set_module_name('0', None)
        qconfig_mapping.set_module_name('1', None)
        qconfig_mapping.set_module_name('2', None)
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @withQNNPACKBackend
    def test_add_loggers_functions(self):
        # [FUNCTION DOC] Adds loggers to a model of functions for both the default
        # qconfig mapping and an empty one.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Model of functional linear, relu, adds, cat, rrelu and a
            # second linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates parameters w1 (2x2, kaiming initialized) and b1.
                super().__init__()
                self.w1 = nn.Parameter(torch.randn(2, 2))
                self.b1 = nn.Parameter(torch.randn(2))
                torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the listed functions in order, including one
                # not matchable by quantization; returns the result.
                x = F.linear(x, self.w1, self.b1)
                x = F.relu(x)
                x = x + x
                x = x + 1
                # TODO(future PR): support first arg being a scalar
                # x = 1 + x
                x = torch.cat([x, x])
                x = torch.cat([x, x])
                x = torch.cat(tensors=[x, x])
                # function not matchable by quantization
                x = torch.nn.functional.rrelu(x)
                x = F.linear(x, self.w1, self.b1)
                return x

        m = M().eval()
        example_input = (torch.randn(16, 2),)
        for qconfig_mapping in (
            get_default_qconfig_mapping(),
            QConfigMapping(),
        ):
            self._test_add_loggers_impl(m, example_input, qconfig_mapping)

    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    @withQNNPACKBackend
    def test_add_loggers_mobilenet_v2(self):
        # [FUNCTION DOC] Adds loggers to torchvision mobilenet v2; skipped under dynamo
        # or without torchvision.
        import torchvision
        m = torchvision.models.quantization.mobilenet_v2(
            pretrained=False, quantize=False).eval()
        example_input = (torch.randn(8, 3, 224, 224),)
        qconfig_mapping = get_default_qconfig_mapping()
        self._test_add_loggers_impl(m, example_input, qconfig_mapping)


class TestFXNumericSuiteCoreAPIsModels(FXNumericSuiteQuantizationTestCase):
    # [CLASS DOC] Core API tests on larger standard test models (conv, linear, lstm,
    # sparse nn, and torchvision models).
    """
    Tests numeric suite core APIs on non-toy models.
    """

    @skipIfNoFBGEMM
    def test_compare_weights_conv(self):
        # [FUNCTION DOC] Weight extraction for conv, conv bn and conv bn relu models.
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            example_inputs = (torch.randn(1, 3, 5, 5),)
            self._test_extract_weights(m, example_inputs, results_len=1)

    @skipIfNoFBGEMM
    def test_compare_weights_linear(self):
        # [FUNCTION DOC] Weight extraction for static and dynamic single linear models.
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            example_inputs = (torch.randn(1, 3, 5, 5),)
            res = self._test_extract_weights(
                m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_weights_lstm_dynamic(self):
        # [FUNCTION DOC] Weight extraction for a dynamically quantized lstm with hidden
        # state input.
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        example_inputs = (lstm_input, lstm_hidden)
        m = LSTMwithHiddenDynamicModel().eval()
        res = self._test_extract_weights(
            m, example_inputs, results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_activations_conv(self):
        # [FUNCTION DOC] Activation matching for conv, conv bn and conv bn relu models.
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            res = self._test_match_activations(
                m, (torch.randn(1, 3, 4, 4),), results_len=1)

    @skipIfNoFBGEMM
    def test_compare_activations_linear(self):
        # [FUNCTION DOC] Activation matching for static and dynamic single linear models.
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            res = self._test_match_activations(
                m, (torch.randn(5, 5),), results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_activations_lstm_dynamic(self):
        # [FUNCTION DOC] Activation matching for a dynamic lstm; scripting is skipped as
        # the prepared lstm is not scriptable.
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        m = LSTMwithHiddenDynamicModel().eval()
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        # TODO(future PR): enable scripting (quant prepared LSTM not scriptable)
        res = self._test_match_activations(
            m, (lstm_input, lstm_hidden), results_len=1, qconfig_dict=qconfig_dict,
            skip_scripting=True)

    @skipIfNoFBGEMM
    def test_compare_shadow_activations_conv(self):
        # [FUNCTION DOC] Shadow activation matching for conv, conv bn and conv bn relu.
        test_cases = (
            (ConvModel(),),
            (ConvBnModel(),),
            (ConvBnReLUModel(),),
        )
        for m, in test_cases:
            m.eval()
            res = self._test_match_shadow_activations(
                m, (torch.randn(1, 3, 4, 4),), results_len=1)

    @skipIfNoFBGEMM
    def test_compare_shadow_activations_linear(self):
        # [FUNCTION DOC] Shadow activation matching for static and dynamic linear models.
        test_cases = (
            (SingleLayerLinearModel(), None),
            (
                SingleLayerLinearDynamicModel(),
                {"object_type": [(nn.Linear, default_dynamic_qconfig)]},
            ),
        )
        for m, qconfig_dict in test_cases:
            m.eval()
            res = self._test_match_shadow_activations(
                m, (torch.randn(5, 5),), results_len=1, qconfig_dict=qconfig_dict)

    @skipIfNoFBGEMM
    def test_compare_shadow_activations_lstm_dynamic(self):
        # [FUNCTION DOC] Shadow activation matching for a dynamic lstm; scripting skipped.
        qconfig_dict = {"object_type": [(nn.LSTM, default_dynamic_qconfig)]}
        m = LSTMwithHiddenDynamicModel().eval()
        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))
        # TODO(future PR): enable scripting (quant prepared LSTM not scriptable)
        res = self._test_match_shadow_activations(
            m, (lstm_input, lstm_hidden), results_len=1, qconfig_dict=qconfig_dict,
            skip_scripting=True)

    @skipIfNoFBGEMM
    def test_sparsenn_compare_activations(self):
        # [FUNCTION DOC] Activation matching on the sparse nn model with and without
        # input logging; expects five results.
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_activations(
                sparse_nn, (idx, offsets, x),
                results_len=5,
                should_log_inputs=should_log_inputs)

    @skipIfNoFBGEMM
    def test_sparsenn_shadow(self):
        # [FUNCTION DOC] Shadow matching on the sparse nn model with and without input
        # logging; expects three results.
        for should_log_inputs in (True, False):
            sparse_nn = SparseNNModel().eval()
            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            self._test_match_shadow_activations(
                sparse_nn, (idx, offsets, x),
                results_len=3,
                should_log_inputs=should_log_inputs)

    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    def test_resnet18(self):
        # [FUNCTION DOC] Shadow activation matching on quantizable resnet18; skipped
        # under dynamo or without torchvision.
        import torchvision
        m = torchvision.models.quantization.resnet18(pretrained=False, quantize=False).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        self._test_match_shadow_activations(
            m, (torch.randn(1, 3, 224, 224),),
            qconfig_dict=qconfig_dict,
            should_log_inputs=False)

    @skipIfTorchDynamo("too slow")
    @skip_if_no_torchvision
    @skipIfNoFBGEMM
    def test_mobilenet_v2(self):
        # [FUNCTION DOC] Shadow activation matching on quantizable mobilenet v2; skipped
        # under dynamo or without torchvision.
        import torchvision
        m = torchvision.models.quantization.mobilenet_v2(pretrained=False, quantize=False).eval()
        qconfig_dict = {'': torch.ao.quantization.default_qconfig}
        self._test_match_shadow_activations(
            m, (torch.randn(1, 3, 224, 224),),
            qconfig_dict=qconfig_dict,
            should_log_inputs=False)

if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
