# Owner(s): ["oncall: quantization"]
# [MODULE DOC] FX graph mode tests for input-weight equalization: the equalization observers and
# scale computation, the graph produced by prepare_fx with an equalization config, the reference
# conversion that folds equalization scales into weights and biases, the quantized graph node order
# after convert_fx, accuracy versus no equalization, and selective equalization chosen from layer
# SQNR. Defines shared qconfig mappings at import time (all-default, specific-object-type and
# equalization variants). Not runnable directly: the main guard calls the helper that reports the
# file must be run through the test_quantization driver.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
from torch.ao.quantization import default_qconfig
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx._equalize import (
    _InputEqualizationObserver,
    _WeightEqualizationObserver,
    calculate_equalization_scale,
    default_equalization_qconfig,
    _convert_equalization_ref,
    get_layer_sqnr_dict,
    get_equalization_qconfig_dict,
)

from torch.testing._internal.common_quantization import (
    NodeSpec as ns,
    QuantizationTestCase,
    SingleLayerLinearModel,
    TwoLayerLinearModel,
    LinearAddModel,
    SingleLayerFunctionalLinearModel,
    TwoLayerFunctionalLinearModel,
    FunctionalLinearAddModel,
    ConvModel,
    TwoLayerConvModel,
    SingleLayerFunctionalConvModel,
    TwoLayerFunctionalConvModel,
    skipIfNoFBGEMM,
    LinearReluModel,
    LinearReluLinearModel,
    LinearReluAddModel,
    FunctionalLinearReluModel,
    FunctionalLinearReluLinearModel,
    ConvReluModel,
    ConvReluConvModel,
    ConvReluAddModel,
    FunctionalConvReluModel,
    FunctionalConvReluConvModel,
)
from torch.testing._internal.common_utils import raise_on_run_directly

# Standard Libraries
import copy
import numpy as np

# Testing utils
from hypothesis import given
from hypothesis import strategies as st


default_qconfig_dict = {"": default_qconfig}

specific_qconfig_dict = {
    "": None,
    "object_type": [(nn.Linear, default_qconfig),
                    (F.linear, default_qconfig),
                    (nn.ReLU, default_qconfig),
                    (F.relu, default_qconfig),
                    (nn.Conv2d, default_qconfig),
                    (F.conv2d, default_qconfig)]
}

default_equalization_qconfig_dict = {
    "": None,
    "object_type": [(nn.Linear, default_equalization_qconfig),
                    (F.linear, default_equalization_qconfig),
                    (nn.ReLU, default_equalization_qconfig),
                    (F.relu, default_equalization_qconfig),
                    (nn.Conv2d, default_equalization_qconfig),
                    (F.conv2d, default_equalization_qconfig)]
}


class TestEqualizeFx(QuantizationTestCase):
    # [CLASS DOC] Tests for input-weight equalization with FX graph mode quantization. Holds
    # numpy-based reference helpers (min/max per channel, expected scales, weights, biases and
    # observer ranges) and the tests that compare library results with them for linear, conv and
    # relu model variants.
    def channel_minmax(self, input, axis=1):
        # [FUNCTION DOC] HELPER: Computes the per-channel minimum and maximum of a numpy array
        # along every dimension except the given channel axis (default one), by reducing the other
        # axes from the last to the first. Returns a pair of arrays (mins, maxs). Does not modify
        # the input.
        ''' Finds the min/max of inputs associated with a specific channel
        '''
        size_of_tensor_dim = input.ndim
        axis_list = list(range(size_of_tensor_dim))
        axis_list.remove(axis)
        axis_list.sort(reverse=True)

        mins = input.copy()
        maxs = input.copy()
        for a in axis_list:
            mins = mins.min(a)
            maxs = maxs.max(a)

        return (mins, maxs)

    @given(ndim=st.sampled_from((2, 3, 4, 5)),
           input_qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           input_qscheme=st.sampled_from((torch.per_tensor_affine, torch.per_tensor_symmetric)),
           weight_qdtype=st.sampled_from((torch.qint8, torch.quint8)),
           weight_qscheme=st.sampled_from((torch.per_channel_affine, torch.per_channel_symmetric,
                                           torch.per_channel_affine_float_qparams)))
    def test_input_weight_eq_observer(self, ndim, input_qdtype, input_qscheme, weight_qdtype, weight_qscheme):
        # [FUNCTION DOC] Hypothesis test over tensor rank two to five and input/weight dtype and
        # scheme choices. Builds random float arrays (rounded to two decimals) with a shared
        # channel dimension, feeds them through the input and weight equalization observers, and
        # checks the outputs pass through unchanged, that recorded input and weight column min/max
        # match numpy references, and that the equalization scale equals the square root of the
        # weight range over the input range. Then sets the scale on both observers and compares the
        # scaled input quantization parameters and the per-channel observer parameters on the
        # reciprocal-scaled weights with references for each scheme (symmetric, affine, float
        # qparams).
        # KNOWN ISSUE: the numpy clip call on the reference zero point discards its result, so the
        # reference zero point is never actually clipped to the dtype range.
        sizes = []
        for _ in range((ndim - 1) * 2):
            sizes.append(np.random.randint(2, 10))

        channel = np.random.randint(1, 10)
        if ndim == 2:
            x = np.random.random(size=(sizes[0], channel))
            w = np.random.random(size=(sizes[1], channel))
        elif ndim == 3:
            x = np.random.random(size=(sizes[0], channel, sizes[1]))
            w = np.random.random(size=(sizes[2], channel, sizes[3]))
        elif ndim == 4:
            x = np.random.random(size=(sizes[0], channel, sizes[1], sizes[2]))
            w = np.random.random(size=(sizes[3], channel, sizes[4], sizes[5]))
        elif ndim == 5:
            x = np.random.random(size=(sizes[0], channel, sizes[1], sizes[2], sizes[3]))
            w = np.random.random(size=(sizes[4], channel, sizes[5], sizes[6], sizes[7]))

        x = (x * 10).round(decimals=2).astype(np.float32)
        w = (w * 10).round(decimals=2).astype(np.float32)

        input_eq_obs = _InputEqualizationObserver(dtype=input_qdtype, qscheme=input_qscheme)
        weight_eq_obs = _WeightEqualizationObserver(dtype=weight_qdtype, qscheme=weight_qscheme)

        ret_x = input_eq_obs(torch.tensor(x))
        ret_w = weight_eq_obs(torch.tensor(w))
        self.assertEqual((ret_x, ret_w), (x, w))

        # Check the min/max input columns are correct
        ref_min_inputs, ref_max_inputs = self.channel_minmax(x)
        min_inputs, max_inputs = input_eq_obs.get_input_minmax()
        self.assertEqual(min_inputs, torch.tensor(ref_min_inputs, dtype=torch.float32))
        self.assertEqual(max_inputs, torch.tensor(ref_max_inputs, dtype=torch.float32))

        # Check the min/max weight columns are correct
        ref_min_weights_col, ref_max_weights_col = self.channel_minmax(w)
        min_weights_col, max_weights_col = weight_eq_obs.get_weight_col_minmax()
        self.assertEqual(min_weights_col, torch.tensor(ref_min_weights_col, dtype=torch.float32))
        self.assertEqual(max_weights_col, torch.tensor(ref_max_weights_col, dtype=torch.float32))

        # Check the equalization scale is correct
        equalization_scale = calculate_equalization_scale(input_eq_obs, weight_eq_obs)
        ref_equalization_scale = np.sqrt((ref_max_weights_col - ref_min_weights_col) /
                                         (ref_max_inputs - ref_min_inputs))
        self.assertEqual(equalization_scale, torch.tensor(ref_equalization_scale, dtype=torch.float32))

        input_eq_obs.set_equalization_scale(equalization_scale)
        weight_eq_obs.set_equalization_scale(equalization_scale)

        # Check the input scale/zero-point values
        min_input_scaled, max_input_scaled = input_eq_obs.calculate_scaled_minmax()
        input_quant_obs = MinMaxObserver(dtype=input_qdtype, qscheme=input_qscheme)
        input_quant_obs.min_val = min_input_scaled
        input_quant_obs.max_val = max_input_scaled
        input_qparams = input_quant_obs.calculate_qparams()

        ref_min_input_scaled = np.min(ref_min_inputs * ref_equalization_scale)
        ref_min_input_scaled = min(0, ref_min_input_scaled)
        ref_max_input_scaled = np.max(ref_max_inputs * ref_equalization_scale)
        ref_max_input_scaled = max(0, ref_max_input_scaled)

        if input_qscheme == torch.per_tensor_symmetric:
            ref_scale = 2 * max(abs(ref_min_input_scaled), ref_max_input_scaled) / 255
            ref_zero_point = 0 if input_qdtype is torch.qint8 else 128
        else:
            ref_scale = (ref_max_input_scaled - ref_min_input_scaled) / 255
            quant_min = -128 if input_qdtype is torch.qint8 else 0
            quant_max = 127 if input_qdtype is torch.qint8 else 255
            ref_zero_point = quant_min - np.round(ref_min_input_scaled / ref_scale)
            np.clip(ref_zero_point, quant_min, quant_max)

        self.assertEqual(input_qparams[0].item(), ref_scale, atol=1e-5, rtol=0)
        self.assertEqual(input_qparams[1].item(), ref_zero_point)

        # During input-weight equalization, we will scale the weights so that
        # the following weight quantized observer will have the correct scaled qparams
        # Check the weight scale/zero-point values of the quantized observer
        weight_quant_obs = PerChannelMinMaxObserver(ch_axis=1, dtype=weight_qdtype, qscheme=weight_qscheme)

        # Scale the weights for input-weight equalization
        new_shape = [1] * w.ndim
        new_shape[1] = w.shape[1]
        ref_w_scaled = w * np.reciprocal(ref_equalization_scale.reshape(tuple(new_shape)))

        w = torch.tensor(w)
        new_shape[1] = w.size(1)
        w_scaled = torch.mul(w, torch.reciprocal(equalization_scale.view(new_shape)))

        self.assertEqual(w_scaled, ref_w_scaled)

        # Call forward on the weight quantization observer
        weight_quant_obs(w_scaled)

        # Check the min/max weight rows are correct
        ref_min_weights_scaled, ref_max_weights_scaled = self.channel_minmax(ref_w_scaled)
        self.assertEqual(weight_quant_obs.min_val, torch.tensor(ref_min_weights_scaled, dtype=torch.float32))
        self.assertEqual(weight_quant_obs.max_val, torch.tensor(ref_max_weights_scaled, dtype=torch.float32))

        weight_qparams = weight_quant_obs.calculate_qparams()

        if weight_qscheme == torch.per_channel_symmetric:
            ref_min_weights_scaled = np.minimum(np.zeros(ref_min_weights_scaled.shape), ref_min_weights_scaled)
            ref_max_weights_scaled = np.maximum(np.zeros(ref_max_weights_scaled.shape), ref_max_weights_scaled)

            ref_scales = 2 * np.maximum(np.abs(ref_min_weights_scaled), ref_max_weights_scaled) / 255
            ref_zero_points = np.zeros_like(
                ref_scales) if weight_qdtype is torch.qint8 else np.ones_like(ref_scales) * 128
        elif weight_qscheme == torch.per_channel_affine_float_qparams:
            ref_scales = (ref_max_weights_scaled - ref_min_weights_scaled) / 255
            ref_scales = np.where(ref_scales > 1e-7, ref_scales, np.ones_like(ref_scales))
            ref_zero_points = -1 * ref_min_weights_scaled / ref_scales
        else:
            ref_min_weights_scaled = np.minimum(np.zeros_like(ref_min_weights_scaled), ref_min_weights_scaled)
            ref_max_weights_scaled = np.maximum(np.zeros_like(ref_max_weights_scaled), ref_max_weights_scaled)

            ref_scales = (ref_max_weights_scaled - ref_min_weights_scaled) / 255
            ref_zero_points = -128 if weight_qdtype is torch.qint8 else 0
            ref_zero_points = ref_zero_points - np.round(ref_min_weights_scaled / ref_scales)

        self.assertEqual(weight_qparams[0], torch.tensor(
            ref_scales, dtype=weight_qparams[0].dtype), rtol=1e-5, atol=0.0001)
        self.assertEqual(weight_qparams[1], torch.tensor(
            ref_zero_points, dtype=weight_qparams[1].dtype), rtol=1e-5, atol=1)

    def test_input_weight_equalization_prepare(self):
        # [FUNCTION DOC] Prepares many linear, conv and relu model variants with prepare_fx using
        # the specific object-type qconfig and the default equalization config, then checks the
        # expected count of input equalization observers, weight equalization observers and min/max
        # observers in the prepared graph for each.
        """ Tests that graphs created after prepare_fx is as expected
        """

        single_nn_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 1,
            ns.call_module(MinMaxObserver): 2,
        }

        two_nn_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 2,
            ns.call_module(MinMaxObserver): 3,
        }

        single_F_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 1,
            ns.call_module(_WeightEqualizationObserver): 1,
            ns.call_module(MinMaxObserver): 3,
        }

        two_F_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 2,
            ns.call_module(_WeightEqualizationObserver): 2,
            ns.call_module(MinMaxObserver): 5,
        }

        fp_F_layer_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 2,
            ns.call_module(_WeightEqualizationObserver): 2,
            ns.call_module(MinMaxObserver): 6,
        }

        tests = [(SingleLayerLinearModel, single_nn_layer_node_occurrence),
                 (TwoLayerLinearModel, two_nn_layer_node_occurrence),
                 (TwoLayerFunctionalLinearModel, two_F_layer_node_occurrence),
                 (FunctionalLinearAddModel, fp_F_layer_node_occurrence),
                 (LinearReluModel, single_nn_layer_node_occurrence),
                 (LinearReluLinearModel, two_nn_layer_node_occurrence),
                 (FunctionalLinearReluModel, single_F_layer_node_occurrence),
                 (FunctionalLinearReluLinearModel, two_F_layer_node_occurrence),
                 (ConvModel, single_nn_layer_node_occurrence),
                 (TwoLayerConvModel, two_nn_layer_node_occurrence),
                 (TwoLayerFunctionalConvModel, two_F_layer_node_occurrence),
                 (ConvReluModel, single_nn_layer_node_occurrence),
                 (ConvReluConvModel, two_nn_layer_node_occurrence),
                 (FunctionalConvReluModel, single_F_layer_node_occurrence),
                 (FunctionalConvReluConvModel, two_F_layer_node_occurrence)]

        for (M, node_occurrence) in tests:
            m = M().eval()
            example_inputs = m.get_example_inputs()
            prepared = prepare_fx(
                m,
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            self.checkGraphModuleNodes(prepared, expected_node_occurrence=node_occurrence)

    def test_input_weight_equalization_branching(self):
        # [FUNCTION DOC] Checks equalization observer insertion around branches. When both branch
        # heads are layers to be equalized, no input equalization observer is inserted (three
        # min/max observers only); when only one head is, one is inserted (two min/max observers).
        """ Tests that graphs containing branches are prepared correctly.
        Specifically, equalization observers should not be inserted in front of
        branches in which both initial layers in the branches plan to be
        quantized.
        """

        # Tests that we do not add an equalization observer due to both initial
        # nodes in the branch containing layers that need to be equalized.
        # Note that this should print out 2 warning messages for not being able
        # to equalize layers linear1 and linear1 because it is part of a branch
        class TestBranchingWithoutEqualizationModel(nn.Module):
            # [CLASS DOC] HELPER: Model where two linear layers consume the same input and the
            # results are added.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two linear layers of width five.
                super().__init__()
                self.linear1 = nn.Linear(5, 5)
                self.linear2 = nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both linear layers to the input and returns their
                # sum.
                y = self.linear1(x)
                z = self.linear2(x)
                return torch.add(y, z)

        no_eq_branching_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 0,
            ns.call_module(MinMaxObserver): 3,
        }

        m = TestBranchingWithoutEqualizationModel().eval()
        example_inputs = (torch.rand(1, 5),)
        prepared = prepare_fx(
            m, specific_qconfig_dict, example_inputs=example_inputs,
            _equalization_config=default_equalization_qconfig_dict)
        self.checkGraphModuleNodes(prepared, expected_node_occurrence=no_eq_branching_node_occurrence)

        # Tests that we will add an equalization observer because there is only
        # one initial node in the branch that needs to be equalized
        class TestBranchingWithEqualizationModel(nn.Module):
            # [CLASS DOC] HELPER: Model where one linear layer and a scalar add on the same input
            # are summed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one linear layer of width five.
                super().__init__()
                self.linear1 = nn.Linear(5, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds five to the input, applies the linear layer to the
                # input, and returns the sum.
                y = self.linear1(x)
                z = torch.add(x, 5)
                return torch.add(y, z)

        eq_branching_node_occurrence = {
            ns.call_module(_InputEqualizationObserver): 1,
            ns.call_module(MinMaxObserver): 2,
        }

        m = TestBranchingWithEqualizationModel().eval()
        example_inputs = (torch.randn(1, 5),)
        prepared = prepare_fx(
            m, specific_qconfig_dict, example_inputs=example_inputs,
            _equalization_config=default_equalization_qconfig_dict)
        self.checkGraphModuleNodes(prepared, expected_node_occurrence=eq_branching_node_occurrence)

    @skipIfNoFBGEMM
    def test_input_weight_equalization_convert(self):
        # [FUNCTION DOC] For each model variant (rank two or four inputs), prepares twice, with the
        # reference equalization conversion applied to the first and a normal quantized conversion
        # on the second. Asserts the equalized float graph output equals the prepared graph output,
        # and that the full conversion still compiles. Requires the FBGEMM backend.
        """ Tests that the modified model for equalization (before quantization)
        returns the same output as the original model
        """

        tests = [(SingleLayerLinearModel, 2), (LinearAddModel, 2), (TwoLayerLinearModel, 2),
                 (SingleLayerFunctionalLinearModel, 2), (FunctionalLinearAddModel, 2),
                 (TwoLayerFunctionalLinearModel, 2),
                 (LinearReluModel, 2), (LinearReluLinearModel, 2), (LinearReluAddModel, 2),
                 (FunctionalLinearReluModel, 2), (FunctionalLinearReluLinearModel, 2),
                 (ConvModel, 4), (TwoLayerConvModel, 4), (SingleLayerFunctionalConvModel, 4),
                 (TwoLayerFunctionalConvModel, 4),
                 (ConvReluModel, 4), (ConvReluConvModel, 4), (ConvReluAddModel, 4),
                 (FunctionalConvReluModel, 4), (FunctionalConvReluConvModel, 4)]

        for (M, ndim) in tests:
            m = M().eval()

            if ndim == 2:
                x = torch.rand((5, 5))
            elif ndim == 4:
                x = torch.rand((16, 3, 224, 224))

            example_inputs = (x,)
            prepared = prepare_fx(
                copy.deepcopy(m),
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict
            )
            output = prepared(x)

            convert_ref = _convert_equalization_ref(prepared)
            convert_ref_output = convert_ref(x)

            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            prepared(x)
            convert_fx(prepared)  # Check if compile
            self.assertEqual(output, convert_ref_output)

    def calculate_equalization_scale_ref(self, x, w):
        # [FUNCTION DOC] HELPER: Returns the numpy equalization scale for an input and weight: the
        # square root of the per-column weight range divided by the per-column input range (reduced
        # over rows).
        """ Calculates the equalization scale based on the input and weight
        """
        min_inputs = x.min(axis=0)
        max_inputs = x.max(axis=0)

        min_weights_col = w.min(axis=0)
        max_weights_col = w.max(axis=0)

        equalization_scale = np.sqrt((max_weights_col - min_weights_col) /
                                     (max_inputs - min_inputs))
        return equalization_scale

    def get_expected_eq_scales(self, model, x):
        # [FUNCTION DOC] HELPER: Walks the model child modules in order, computing the reference
        # equalization scale from the current activations and that child's weight, and propagating
        # the activations through the child's affine map. Returns the list of scales. Assumes every
        # child has a weight and a bias, i.e. only chains of linear layers.
        """ For each module in the graph, we want to calculate the equalization
        scale at that point. This only works for models containing single or
        connected linear layers.
        """
        exp_eq_scales = []
        for _, module in model.named_children():
            weight = module.weight.detach().numpy()
            bias = module.bias.detach().numpy()

            eq_scale = self.calculate_equalization_scale_ref(x, weight)
            exp_eq_scales.append(eq_scale)

            x = x @ weight.T + bias

        return exp_eq_scales

    def test_input_weight_equalization_equalization_scales(self):
        # [FUNCTION DOC] Runs prepare_fx and the reference equalization conversion on single and
        # two layer linear models, then checks every equalization scale buffer in the converted
        # graph equals the reference scale, in order.
        """ After applying the equalization functions, check if the equalization
        scales are the expected values
        """

        tests = [SingleLayerLinearModel, TwoLayerLinearModel,
                 SingleLayerFunctionalLinearModel, TwoLayerFunctionalLinearModel]

        x = torch.rand((5, 5))
        for M in tests:
            m = M().eval()
            exp_eq_scales = self.get_expected_eq_scales(m, x.detach().numpy())

            example_inputs = (x,)
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            prepared(*example_inputs)
            convert_ref = _convert_equalization_ref(prepared)
            convert_ref(x)

            counter = 0
            for node in convert_ref.graph.nodes:
                if 'equalization_scale' in node.name and node.op == 'get_attr':
                    self.assertEqual(convert_ref.get_buffer(str(node.target)).reshape(-1), exp_eq_scales[counter])
                    counter += 1

    def get_expected_weights_bias(self, model, x, exp_eq_scales):
        # [FUNCTION DOC] HELPER: For each child linear layer computes the expected equalized weight
        # (divided by its own scale, rows then multiplied by the next layer's scale when there is
        # one) and bias (scaled by the next scale). Returns the lists of expected weights and
        # biases.
        """ For each module in the graph, we want to calculate the expected
        scaled weight and bias values. This only works for models containing
        single or connected linear layers.
        """
        exp_weights = []
        exp_bias = []
        for i, (_, module) in enumerate(model.named_children()):
            weight = module.weight.detach().numpy()
            bias = module.bias.detach().numpy()

            scaled_weight = weight * np.reciprocal(exp_eq_scales[i])
            scaled_bias = bias
            if i + 1 < len(exp_eq_scales):
                scaled_weight = (scaled_weight.T * exp_eq_scales[i + 1]).T
                scaled_bias = (scaled_bias.T * exp_eq_scales[i + 1]).T

            exp_weights.append(scaled_weight)
            exp_bias.append(scaled_bias)

            x = x @ weight.T + bias

        return exp_weights, exp_bias

    def test_input_weight_equalization_weights_bias(self):
        # [FUNCTION DOC] Runs prepare_fx, calibration and the reference equalization conversion on
        # linear models and checks each linear module weight and bias in the converted graph equals
        # the expected scaled values, in order.
        """ After applying the equalization functions check if the weights and
        biases are as expected
        """

        tests = [SingleLayerLinearModel, TwoLayerLinearModel,
                 SingleLayerFunctionalLinearModel, TwoLayerFunctionalLinearModel]

        x = torch.rand((5, 5))
        for M in tests:
            m = M().eval()
            exp_eq_scales = self.get_expected_eq_scales(m, x.detach().numpy())
            exp_weights, exp_bias = self.get_expected_weights_bias(m, x.detach().numpy(), exp_eq_scales)

            example_inputs = (x,)
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            prepared(x)
            convert_ref = _convert_equalization_ref(prepared)
            convert_ref(x)

            modules = dict(convert_ref.named_modules(remove_duplicate=False))
            counter = 0
            for node in convert_ref.graph.nodes:
                if node.op == 'call_module' and isinstance(modules[str(node.target)], nn.Linear):
                    self.assertEqual(modules[str(node.target)].weight, exp_weights[counter])
                    self.assertEqual(modules[str(node.target)].bias, exp_bias[counter])
                    counter += 1

    def get_expected_inp_act_vals(self, model, x, exp_eq_scales, exp_weights, exp_bias):
        # [FUNCTION DOC] HELPER: Computes expected (min, max) pairs for each input activation of a
        # chain of linear layers: the input is multiplied by the first scale, and activations are
        # propagated through the expected scaled weights and biases. Returns the list, including
        # the final output.
        """ For each module in the graph, we want to calculate the expected
        min/max values for every input activation node. This only works for
        models containing only single or connected linear layers.
        """
        x = x * exp_eq_scales[0]

        exp_inp_activation_vals = []
        for i, _ in enumerate(model.named_children()):
            exp_inp_activation_vals.append((x.min(), x.max()))
            x = x @ exp_weights[i].T + exp_bias[i]

        exp_inp_activation_vals.append((x.min(), x.max()))
        return exp_inp_activation_vals

    def get_expected_weight_act_vals(self, exp_weights):
        # [FUNCTION DOC] HELPER: Returns a list of (min, max) pairs, one per expected scaled weight
        # array.
        """ For each module in the graph, we want to calculate the expected
        min/max values for every weight activation node. This is assuming that
        the weight observers are all MinMaxObservers.
        """

        exp_weight_activation_vals = []
        for w in exp_weights:
            exp_weight_activation_vals.append((w.min(), w.max()))

        return exp_weight_activation_vals

    def test_input_weight_equalization_activation_values(self):
        # [FUNCTION DOC] After equalization conversion, checks the min and max stored in the
        # min/max observers. Observers that feed only the weight argument of a functional linear
        # are compared with the expected weight ranges; all other observers with the expected input
        # activation ranges, in graph order. Tests single and two layer linear and single
        # functional linear models.
        """ After applying the equalization functions check if the input
        observer's min/max values are as expected
        """

        tests = [SingleLayerLinearModel, TwoLayerLinearModel, SingleLayerFunctionalLinearModel]

        x = torch.rand((5, 5))
        torch.manual_seed(0)
        for M in tests:
            m = M().eval()
            exp_eq_scales = self.get_expected_eq_scales(m, x.detach().numpy())
            exp_weights, exp_bias = self.get_expected_weights_bias(m, x.detach().numpy(), exp_eq_scales)
            exp_inp_act_vals = self.get_expected_inp_act_vals(m, x, exp_eq_scales, exp_weights, exp_bias)
            exp_weight_act_vals = self.get_expected_weight_act_vals(exp_weights)

            example_inputs = (x,)
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            prepared(x)
            convert_ref = _convert_equalization_ref(prepared)
            convert_ref(x)

            modules = dict(convert_ref.named_modules(remove_duplicate=False))
            inp_counter = 0
            weight_counter = 0
            for node in convert_ref.graph.nodes:
                users = list(node.users)
                if node.op == 'call_module' and isinstance(modules[str(node.target)], MinMaxObserver):
                    if len(users) == 1 and users[0].target == torch.nn.functional.linear and users[0].args[1] == node:
                        # Check min/max values of weight activation layers
                        exp_min_val, exp_max_val = exp_weight_act_vals[weight_counter]
                        self.assertEqual(modules[str(node.target)].min_val, exp_min_val)
                        self.assertEqual(modules[str(node.target)].max_val, exp_max_val)
                        weight_counter += 1
                    else:
                        # Check min/max values of input activation layers
                        exp_min_val, exp_max_val = exp_inp_act_vals[inp_counter]
                        self.assertEqual(modules[str(node.target)].min_val, exp_min_val)
                        self.assertEqual(modules[str(node.target)].max_val, exp_max_val)
                        inp_counter += 1


    def check_orig_and_eq_graphs(self, orig_model, eq_model):
        # [FUNCTION DOC] HELPER: Compares two graph modules node by node, skipping an equalization
        # scale attribute node followed by a multiply node in the equalized graph. Returns False
        # when a node operation differs, a called module has a different type, or a called function
        # differs; True otherwise.
        # KNOWN ISSUE: this helper is not called by any test in this file.
        """ Given a non-equalized model and an equalized model, check that the
        graphs are structured in the same way, except the equalized model has
        additional 'equalization_scale' and 'mul' nodes.
        """
        orig_idx = 0
        orig_nodes = list(orig_model.graph.nodes)
        orig_modules = dict(orig_model.named_modules(remove_duplicate=False))

        eq_idx = 0
        eq_nodes = list(eq_model.graph.nodes)
        eq_modules = dict(eq_model.named_modules(remove_duplicate=False))

        while orig_idx < len(orig_nodes) and eq_idx < len(eq_nodes):
            if 'equalization_scale' in eq_nodes[eq_idx].name and 'mul' in eq_nodes[eq_idx + 1].name:
                # Skip the equalization and mul nodes
                eq_idx += 2
                continue
            elif orig_nodes[orig_idx].op != eq_nodes[eq_idx].op:
                return False
            elif orig_nodes[orig_idx].op == 'call_module':
                # Check that the type of call_modules are the same (ex. nn.Linear, MinMaxObserver)
                orig_node = orig_nodes[orig_idx]
                eq_node = eq_nodes[eq_idx]
                if type(orig_modules[orig_node.target]) is not type(eq_modules[eq_node.target]):
                    return False
            elif orig_nodes[orig_idx].op == 'call_function':
                # Check that the call_functions are the same (ex. F.linear)
                orig_node = orig_nodes[orig_idx]
                eq_node = eq_nodes[eq_idx]
                if orig_node.target != eq_node.target:
                    return False

            eq_idx += 1
            orig_idx += 1

        return True

    @skipIfNoFBGEMM
    def test_input_weight_equalization_graphs(self):
        # [FUNCTION DOC] Prepares and converts each model variant with equalization and checks the
        # converted graph contains the expected ordered nodes: multiply by the scale, quantize, the
        # quantized layer (module or functional, with relu fused where applicable), then
        # dequantize. Requires FBGEMM.
        """ Tests that the modified model for equalization has the same graph
        structure as the model without equalization (before and after
        quantization).
        """

        linear_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        linearAdd_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize'),
            ns.call_function(torch.add),
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        linear2_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        functionalLinear_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize')
        ]

        functionalLinearAdd_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize'),
            ns.call_function(torch.add),
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize')
        ]

        functionalLinear2_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize')
        ]

        linearRelu_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_method('dequantize')
        ]

        linearReluLinear_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.LinearReLU),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        functionalLinearRelu_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear_relu),
            ns.call_method('dequantize')
        ]

        functionalLinearReluLinear_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.linear_relu),
            ns.call_function(torch.ops.quantized.linear),
            ns.call_method('dequantize')
        ]

        conv_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize')
        ]

        conv2_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize')
        ]

        functionalConv_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.conv2d),
            ns.call_method('dequantize')
        ]

        functionalConv2_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.conv2d),
            ns.call_function(torch.ops.quantized.conv2d),
            ns.call_method('dequantize')
        ]

        convRelu_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.ConvReLU2d),
            ns.call_method('dequantize')
        ]

        convReluConv_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nniq.ConvReLU2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method('dequantize')
        ]

        functionalConvRelu_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.conv2d_relu),
            ns.call_method('dequantize')
        ]

        functionalConvReluConv_node_list = [
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_function(torch.ops.quantized.conv2d_relu),
            ns.call_function(torch.ops.quantized.conv2d),
            ns.call_method('dequantize')
        ]

        tests = [(SingleLayerLinearModel, linear_node_list),
                 (LinearAddModel, linearAdd_node_list),
                 (TwoLayerLinearModel, linear2_node_list),
                 (SingleLayerFunctionalLinearModel, functionalLinear_node_list),
                 (FunctionalLinearAddModel, functionalLinearAdd_node_list),
                 (TwoLayerFunctionalLinearModel, functionalLinear2_node_list),
                 (LinearReluModel, linearRelu_node_list),
                 (LinearReluLinearModel, linearReluLinear_node_list),
                 (FunctionalLinearReluModel, functionalLinearRelu_node_list),
                 (FunctionalLinearReluLinearModel, functionalLinearReluLinear_node_list),
                 (ConvModel, conv_node_list),
                 (TwoLayerConvModel, conv2_node_list),
                 (SingleLayerFunctionalConvModel, functionalConv_node_list),
                 (TwoLayerFunctionalConvModel, functionalConv2_node_list),
                 (ConvReluModel, convRelu_node_list),
                 (ConvReluConvModel, convReluConv_node_list),
                 (FunctionalConvReluModel, functionalConvRelu_node_list),
                 (FunctionalConvReluConvModel, functionalConvReluConv_node_list)]

        for (M, node_list) in tests:
            m = M().eval()
            example_inputs = m.get_example_inputs()
            prepared = prepare_fx(
                m, specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict)
            equalized_quantized_model = convert_fx(prepared)

            # Check the order of nodes in the graph
            self.checkGraphModuleNodes(equalized_quantized_model, expected_node_list=node_list)

    @skipIfNoFBGEMM
    def test_input_weight_equalization_results(self):
        # [FUNCTION DOC] Compares outputs of quantized small linear models built without
        # equalization and with default equalization; asserts they agree within an absolute
        # tolerance of one tenth. Requires FBGEMM.
        """ Tests that for small models, the results of quantized models that
        have been equalized are very close to models that have not been equalized.
        """

        tests = [SingleLayerLinearModel, TwoLayerLinearModel, LinearAddModel,
                 SingleLayerFunctionalLinearModel, TwoLayerFunctionalLinearModel]

        x = torch.rand((5, 5))
        for M in tests:
            m = M().eval()

            # No equalization
            example_inputs = (x,)
            prepared = prepare_fx(
                copy.deepcopy(m),
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config={})
            prepared(x)
            quantized = convert_fx(prepared)  # Check if compile
            quantized_output = quantized(x)

            # With equalization
            prepared = prepare_fx(
                copy.deepcopy(m),
                specific_qconfig_dict,
                example_inputs=example_inputs,
                _equalization_config=default_equalization_qconfig_dict
            )
            prepared(x)
            equalized_and_quantized = convert_fx(prepared)  # Check if compile
            equalized_and_quantized_output = equalized_and_quantized(x)
            self.assertEqual(quantized_output, equalized_and_quantized_output, rtol=1e-5, atol=0.1)

    @skipIfNoFBGEMM
    def test_selective_equalization(self):
        # [FUNCTION DOC] Quantizes a two layer model, measures per-layer SQNR with the numeric
        # suite helper, builds an equalization config for the single worst layer, prepares and
        # converts with it, and checks the resulting graph has the scale multiply only before the
        # second quantization. Requires FBGEMM.
        """ Tests that we are able to run numeric suite on the equalized model
        and construct a valid equalization_config equalizing only the top
        4 layers with the highest quantization errors.
        """

        torch.manual_seed(1)

        class M(nn.Module):
            # [CLASS DOC] HELPER: Model with a bottom linear sequence, a scalar add of five, and a
            # top linear sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the bottom and top sequential modules, each with
                # one linear layer.
                super().__init__()
                self.bot = torch.nn.Sequential(torch.nn.Linear(5, 5))
                self.top = torch.nn.Sequential(torch.nn.Linear(5, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies bottom, adds five, then applies top and returns
                # the result.
                x = self.bot(x)
                x = torch.add(x, 5)
                x = self.top(x)
                return x

        float_model = M().eval()
        # Hard coded so that the top layer has a higher quantization error
        x = torch.tensor([[0.0642, 0.7824, 0.4255, 0.7106, 0.5957],
                          [0.8373, 0.8851, 0.8229, 0.0212, 0.8987],
                          [0.9077, 0.7538, 0.4530, 0.5772, 0.1376],
                          [0.0690, 0.9002, 0.7998, 0.2768, 0.8985],
                          [0.0282, 0.5068, 0.6725, 0.1829, 0.5480]])

        # Quantize the float model
        example_inputs = (x,)
        prepared_model = prepare_fx(
            copy.deepcopy(float_model),
            specific_qconfig_dict,
            example_inputs=example_inputs
        )
        prepared_model(x)
        quantized_model = convert_fx(copy.deepcopy(prepared_model))

        # Get the SQNR between the float and quantized model
        layer_to_sqnr_dict = get_layer_sqnr_dict(copy.deepcopy(prepared_model), quantized_model, x)

        # Construct the equalization_qconfig_dict equalizing layers with the highest
        # quantization errors
        selective_equalization_qconfig_dict = get_equalization_qconfig_dict(layer_to_sqnr_dict, 1)

        # Create the selectively equalized model
        prepared_model = prepare_fx(
            copy.deepcopy(float_model),
            specific_qconfig_dict,
            example_inputs=example_inputs,
            _equalization_config=selective_equalization_qconfig_dict,
        )
        prepared_model(x)
        equalized_model = convert_fx(prepared_model)

        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize'),
            ns.call_function(torch.add),
            ns.call_function(torch.mul),
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Linear),
            ns.call_method('dequantize')
        ]

        # Check the order of nodes in the graph
        self.checkGraphModuleNodes(equalized_model, expected_node_list=node_list)

if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
