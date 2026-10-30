# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
# [MODULE DOC] Eager-mode tests for the numeric suite (weight comparison, shadow-module based
# module comparison, whole-model output comparison, and the logger modules). Defines small fixture
# models (a submodule container and a functional-ops model) and a test case that quantizes models
# statically or dynamically across all engines, then checks the structure of the comparison
# dictionaries (keys, matching lengths, matching shapes). Includes torchvision mobilenet smoke
# tests that are skipped without torchvision or on ARM64. Running the file directly raises an
# error through the shared helper.

import unittest

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
from torch.ao.ns._numeric_suite import (
    compare_model_outputs,
    compare_model_stub,
    compare_weights,
    get_matching_activations,
    OutputLogger,
    prepare_model_outputs,
    Shadow,
    ShadowLogger,
)
from torch.ao.quantization import (
    convert,
    default_qconfig,
    DeQuantStub,
    prepare,
    quantize,
    quantize_dynamic,
    QuantStub,
)
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnReLUModel,
    AnnotatedConvModel,
    AnnotatedConvTransposeModel,
    AnnotatedSingleLayerLinearModel,
    AnnotatedTwoLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    QuantizationTestCase,
    SingleLayerLinearDynamicModel,
    skip_if_no_torchvision,
    test_only_eval_fn,
)
from torch.testing._internal.common_quantized import override_qengines
from torch.testing._internal.common_utils import IS_ARM64, raise_on_run_directly


class SubModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small fixture module with a default qconfig, a quant stub, a bias-free
    # float conv, a relu and a dequant stub, applied in that order. State: qconfig, mod1, mod2,
    # quant, dequant.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the stubs and layers listed in the class doc and sets the
        # default qconfig.
        super().__init__()
        self.qconfig = default_qconfig
        self.mod1 = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)
        self.mod2 = nn.ReLU()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantizes the input, applies conv and relu, dequantizes and
        # returns the result.
        x = self.quant(x)
        x = self.mod1(x)
        x = self.mod2(x)
        x = self.dequant(x)
        return x


class ModelWithSubModules(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture module with a nested SubModule child and a bias-free float conv
    # sibling that has no qconfig of its own, so it is left unquantized by the static flow.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the nested submodule and the extra conv.
        super().__init__()
        self.mod1 = SubModule()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the input through the submodule and then the conv, returning
        # the result.
        x = self.mod1(x)
        x = self.conv(x)
        return x


class ModelWithFunctionals(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture module holding six float functional modules (cat, add, mul,
    # add_relu, scalar add and scalar mul) plus quant and dequant stubs; used to test comparisons
    # of functional ops.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the six float functional attributes and the stubs.
        super().__init__()
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.mymul = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        self.my_scalar_add = nnq.FloatFunctional()
        self.my_scalar_mul = nnq.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Quantizes the input, then concatenates three copies, adds the
        # result to itself, multiplies by itself, applies add with relu, adds a scalar of minus
        # one half and multiplies by one half using the matching functional modules, and
        # dequantizes. Returns the final tensor.
        x = self.quant(x)
        x = self.mycat.cat([x, x, x])
        x = self.myadd.add(x, x)
        x = self.mymul.mul(x, x)
        x = self.myadd_relu.add_relu(x, x)
        w = self.my_scalar_add.add_scalar(x, -0.5)
        w = self.my_scalar_mul.mul_scalar(w, 0.5)

        w = self.dequant(w)
        return w


class TestNumericSuiteEager(QuantizationTestCase):
    # [CLASS DOC] Test case for the numeric suite APIs in eager mode: weight comparison,
    # module-level shadow comparison, full-model output comparison, logger classes and torchvision
    # model smoke tests. Most tests are run once per available quantization engine.
    @override_qengines
    def test_compare_weights_conv_static(self):
        # [FUNCTION DOC] Quantizes conv-based models statically (with fusion when the model
        # supports it) and checks that the weight comparison between float and quantized state
        # dicts yields exactly one entry whose float and quantized weight shapes match.
        r"""Compare the weights of float and static quantized conv layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Compares weights
            # and asserts one entry with equal shapes.
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for v in weight_dict.values():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_weights_linear_static(self):
        # [FUNCTION DOC] Same as the static conv weight test but for a single-layer linear model
        # quantized statically with the shared linear calibration data.
        r"""Compare the weights of float and static quantized linear layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Compares weights
            # and asserts one entry with equal shapes.
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for v in weight_dict.values():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_weights_linear_dynamic(self):
        # [FUNCTION DOC] Dynamically quantizes a single-layer linear model and checks that the
        # weight comparison has one entry whose float and quantized values are sequences of equal
        # length with matching per-element shapes.
        r"""Compare the weights of float and dynamic quantized linear layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Asserts one entry,
            # equal lengths of the float and quantized sequences, and equal per-element shapes.
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for v in weight_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_weights_lstm_dynamic(self):
        # [FUNCTION DOC] Dynamically quantizes an LSTM model with hidden state and runs the same
        # weight comparison checks as the dynamic linear test.
        r"""Compare the weights of float and dynamic quantized LSTM layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Asserts one entry,
            # equal lengths of the float and quantized sequences, and equal per-element shapes.
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for v in weight_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_model_stub_conv_static(self):
        # [FUNCTION DOC] Statically quantizes conv, conv-transpose (forced to the qnnpack engine
        # since it cannot use per-channel weights) and fused conv-bn-relu models, runs the
        # stub-based module comparison for a swap list of conv, fused conv-relu and conv-transpose
        # types, and checks one result entry with matching float and quantized output shapes.
        r"""Compare the output of static quantized conv layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Runs the stub
            # comparison on the given data and asserts one entry with equal lengths and shapes of
            # logged float and quantized outputs.
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        model_list = [
            AnnotatedConvModel(qengine),
            AnnotatedConvTransposeModel(
                "qnnpack"
            ),  # ConvT cannot use per channel weights
            AnnotatedConvBnReLUModel(qengine),
        ]
        module_swap_list = [
            nn.Conv2d,
            nn.intrinsic.modules.fused.ConvReLU2d,
            nn.ConvTranspose2d,
        ]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            compare_and_validate_results(
                model, q_model, module_swap_list, self.img_data_2d[0][0]
            )

    @override_qengines
    def test_compare_model_stub_linear_static(self):
        # [FUNCTION DOC] Same as the static conv stub test but for a single-layer linear model
        # with linear in the swap list, fed one linear input sample.
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Runs the stub
            # comparison and asserts one entry with equal lengths and shapes of logged outputs.
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]
        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_partial(self):
        # [FUNCTION DOC] Quantizes a two-layer linear model that has only partial quantization
        # annotation and runs the stub comparison with linear in the swap list, asserting one
        # entry with matching shapes.
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine
        # TODO: Rebase on top of PR to remove compare and validate results here

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Runs the stub
            # comparison and asserts one entry with equal lengths and shapes of logged outputs.
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]
        model_list = [AnnotatedTwoLayerLinearModel()]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_submodule_static(self):
        # [FUNCTION DOC] Quantizes a model containing a submodule and a non-quantized conv, runs
        # the stub comparison with the submodule type and conv in the swap list, and asserts the
        # quantized model now wraps the submodule in a shadow module while the unquantized conv is
        # not wrapped.
        r"""Compare the output of static quantized submodule and its float shadow module"""

        qengine = torch.backends.quantized.engine

        model = ModelWithSubModules().eval()
        q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
        module_swap_list = [SubModule, nn.Conv2d]
        ob_dict = compare_model_stub(
            model, q_model, module_swap_list, self.img_data_2d[0][0]
        )
        # Since conv is not quantized, we do not insert a shadow module
        # mod1 contains a linear that is quantized, so we insert a shadow module
        self.assertTrue(isinstance(q_model.mod1, Shadow))
        self.assertFalse(isinstance(q_model.conv, Shadow))

    @override_qengines
    def test_compare_model_stub_functional_static(self):
        # [FUNCTION DOC] Prepares and converts a model of float functional ops using the fbgemm
        # default qconfig, runs the stub comparison swapping functional modules, and asserts six
        # shadow modules were inserted and six entries were logged with matching float and
        # quantized shapes.
        r"""Compare the output of static quantized functional layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        model = ModelWithFunctionals().eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        q_model = prepare(model, inplace=False)
        q_model(self.img_data_2d[0][0])
        q_model = convert(q_model)
        module_swap_list = [nnq.FloatFunctional]
        ob_dict = compare_model_stub(
            model, q_model, module_swap_list, self.img_data_2d[0][0]
        )
        self.assertEqual(len(ob_dict), 6)
        self.assertTrue(isinstance(q_model.mycat, Shadow))
        self.assertTrue(isinstance(q_model.myadd, Shadow))
        self.assertTrue(isinstance(q_model.mymul, Shadow))
        self.assertTrue(isinstance(q_model.myadd_relu, Shadow))
        self.assertTrue(isinstance(q_model.my_scalar_add, Shadow))
        self.assertTrue(isinstance(q_model.my_scalar_mul, Shadow))
        for v in ob_dict.values():
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == val.shape)

    @override_qengines
    def test_compare_model_stub_linear_dynamic(self):
        # [FUNCTION DOC] Dynamically quantizes a single-layer linear model and runs the stub
        # comparison with linear and LSTM in the swap list, asserting one entry with matching
        # shapes.
        r"""Compare the output of dynamic quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Runs the stub
            # comparison and asserts one entry with equal lengths and shapes of logged outputs.
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        linear_data = self.calib_data[0][0]

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        module_swap_list = [nn.Linear, nn.LSTM]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_lstm_dynamic(self):
        # [FUNCTION DOC] Dynamically quantizes an LSTM-with-hidden model and runs the stub
        # comparison, passing a random input and a hidden state pair, asserting one entry with
        # matching shapes.
        r"""Compare the output of dynamic quantized LSTM layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(
            float_model, q_model, module_swap_list, input, hidden
        ):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Runs the stub
            # comparison with input and hidden state and asserts one entry with equal lengths and
            # shapes of logged outputs.
            ob_dict = compare_model_stub(
                float_model, q_model, module_swap_list, input, hidden
            )
            self.assertEqual(len(ob_dict), 1)
            for v in ob_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        module_swap_list = [nn.Linear, nn.LSTM]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(
                model, q_model, module_swap_list, lstm_input, lstm_hidden
            )

    @override_qengines
    def test_compare_model_outputs_conv_static(self):
        # [FUNCTION DOC] Statically quantizes conv models and compares whole-model module outputs,
        # asserting the result keys are exactly the conv and quant logger statistics and that the
        # first logged float and quantized outputs have equal shapes.
        r"""Compare the output of conv layer in stataic quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Compares model
            # outputs, asserts the exact set of keys and equal first-output shapes.
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            expected_act_compare_dict_keys = {"conv.stats", "quant.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                self.assertTrue(v["float"][0].shape == v["quantized"][0].shape)

        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            compare_and_validate_results(model, q_model, self.img_data_2d[0][0])

    @override_qengines
    def test_compare_model_outputs_linear_static(self):
        # [FUNCTION DOC] Statically quantizes a single-layer linear model and checks that the
        # whole-model output comparison produces exactly the keys for the first layer quantization
        # stub and the first layer wrapped module, with matching shapes.
        r"""Compare the output of linear layer in static quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Compares model
            # outputs, asserts the exact key set and equal lengths and shapes of logged outputs.
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            expected_act_compare_dict_keys = {"fc1.quant.stats", "fc1.module.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        linear_data = self.calib_data[0][0]
        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model, linear_data)

    @override_qengines
    def test_compare_model_outputs_functional_static(self):
        # [FUNCTION DOC] Prepares and converts the functional-ops model with the fbgemm default
        # qconfig, compares whole-model outputs on one sample and asserts five entries with the
        # expected logger keys (cat, add, mul, add_relu and quant) and matching shapes. The scalar
        # ops are not expected in the result.
        r"""Compare the output of functional layer in static quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        model = ModelWithFunctionals().eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        q_model = prepare(model, inplace=False)
        q_model(self.img_data_2d[0][0])
        q_model = convert(q_model)
        act_compare_dict = compare_model_outputs(model, q_model, self.img_data_2d[0][0])
        self.assertEqual(len(act_compare_dict), 5)
        expected_act_compare_dict_keys = {
            "mycat.stats",
            "myadd.stats",
            "mymul.stats",
            "myadd_relu.stats",
            "quant.stats",
        }
        self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
        for v in act_compare_dict.values():
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == val.shape)

    @override_qengines
    def test_compare_model_outputs_linear_dynamic(self):
        # [FUNCTION DOC] Dynamically quantizes a single-layer linear model and checks that the
        # whole-model output comparison has exactly the first layer key and matching output
        # shapes.
        r"""Compare the output of linear layer in dynamic quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Compares model
            # outputs, asserts the exact key set and equal lengths and shapes of logged outputs.
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            expected_act_compare_dict_keys = {"fc1.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == val.shape)

        linear_data = self.calib_data[0][0]

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model, linear_data)

    @override_qengines
    def test_compare_model_outputs_lstm_dynamic(self):
        # [FUNCTION DOC] Dynamically quantizes an LSTM-with-hidden model and compares whole-model
        # outputs with an input and hidden state pair, asserting the single lstm key and that each
        # logged entry matches in structure: the output element is compared by shape, and the
        # hidden state entry is compared by the shapes of both of its tensors.
        r"""Compare the output of LSTM layer in dynamic quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, input, hidden):
            # [FUNCTION DOC] HELPER: Nested checker used by the enclosing test. Compares outputs
            # with input and hidden state, asserts the key set, and compares nested shapes per
            # logged entry.
            act_compare_dict = compare_model_outputs(
                float_model, q_model, input, hidden
            )
            expected_act_compare_dict_keys = {"lstm.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for v in act_compare_dict.values():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(len(v["float"][i]) == len(val))
                    if i == 0:
                        self.assertTrue(v["float"][i][0].shape == val[0].shape)
                    else:
                        self.assertTrue(v["float"][i][0].shape == val[0].shape)
                        self.assertTrue(v["float"][i][1].shape == val[1].shape)

        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model, lstm_input, lstm_hidden)

    @override_qengines
    def test_output_logger(self):
        # [FUNCTION DOC] Feeds two random tensors of different shapes through an output logger and
        # asserts the logger's recorded tensor list equals the list of the inputs, in order.
        r"""Compare output from OutputLogger with the expected results"""
        x = torch.rand(2, 2)
        y = torch.rand(2, 1)

        l = []
        l.append(x)
        l.append(y)

        logger = OutputLogger()
        logger.forward(x)
        logger.forward(y)

        self.assertEqual(l, logger.stats["tensor_val"])

    @override_qengines
    def test_shadow_logger(self):
        # [FUNCTION DOC] Feeds two pairs of tensors through a shadow logger and asserts it
        # recorded two entries in each of the float and quantized lists. Values are not compared.
        r"""Compare output from ShawdowLogger with the expected results"""
        a_float = torch.rand(2, 2)
        a_quantized = torch.rand(2, 2)

        b_float = torch.rand(3, 2, 2)
        b_quantized = torch.rand(3, 2, 2)

        logger = ShadowLogger()
        logger.forward(a_float, a_quantized)
        logger.forward(b_float, b_quantized)

        self.assertEqual(len(logger.stats["float"]), 2)
        self.assertEqual(len(logger.stats["quantized"]), 2)

    @skip_if_no_torchvision
    def _test_vision_model(self, float_model):
        # [FUNCTION DOC] HELPER: Shared body for the vision model tests (skipped without
        # torchvision). Moves the float model to CPU, eval mode, fuses it, sets the default
        # qconfig, builds random image data and quantizes it with the default eval function. Then
        # compares weights, compares outputs on a sample while computing a decibel error per entry
        # (result is unused), prepares logging via the model-output preparation function, runs the
        # data through both models and gets matching activations. Only checks that nothing raises.
        float_model.to("cpu")
        float_model.eval()
        float_model.fuse_model()
        float_model.qconfig = torch.ao.quantization.default_qconfig
        img_data = [
            (
                torch.rand(2, 3, 224, 224, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            )
            for _ in range(2)
        ]
        qmodel = quantize(
            float_model,
            torch.ao.quantization.default_eval_fn,
            [img_data],
            inplace=False,
        )

        wt_compare_dict = compare_weights(float_model.state_dict(), qmodel.state_dict())

        def compute_error(x, y):
            # [FUNCTION DOC] HELPER: Computes the signal-to-noise ratio in decibels between a
            # reference tensor and an approximation as twenty times the base-ten log of the norm
            # ratio.
            Ps = torch.norm(x)
            Pn = torch.norm(x - y)
            return 20 * torch.log10(Ps / Pn)

        data = img_data[0][0]
        # Take in floating point and quantized model as well as input data, and returns a dict, with keys
        # corresponding to the quantized module names and each entry being a dictionary with two keys 'float' and
        # 'quantized', containing the activations of floating point and quantized model at matching locations.
        act_compare_dict = compare_model_outputs(float_model, qmodel, data)

        for key in act_compare_dict:
            compute_error(
                act_compare_dict[key]["float"][0],
                act_compare_dict[key]["quantized"][0].dequantize(),
            )

        prepare_model_outputs(float_model, qmodel)

        for data in img_data:
            float_model(data[0])
            qmodel(data[0])

        # Find the matching activation between floating point and quantized modules, and return a dict with key
        # corresponding to quantized module names and each entry being a dictionary with two keys 'float'
        # and 'quantized', containing the matching floating point and quantized activations logged by the logger
        act_compare_dict = get_matching_activations(float_model, qmodel)

    @skip_if_no_torchvision
    @unittest.skipIf(IS_ARM64, "Not working on arm right now")
    def test_mobilenet_v2(self):
        # [FUNCTION DOC] Smoke test: loads the pretrained torchvision quantizable mobilenet v2
        # without quantization and runs the shared vision model checks. Skipped without
        # torchvision or on ARM64.
        from torchvision.models.quantization import mobilenet_v2

        self._test_vision_model(mobilenet_v2(pretrained=True, quantize=False))

    @skip_if_no_torchvision
    @unittest.skipIf(IS_ARM64, "Not working on arm right now")
    def test_mobilenet_v3(self):
        # [FUNCTION DOC] Smoke test: loads the pretrained torchvision quantizable mobilenet v3
        # large without quantization and runs the shared vision model checks. Skipped without
        # torchvision or on ARM64.
        from torchvision.models.quantization import mobilenet_v3_large

        self._test_vision_model(mobilenet_v3_large(pretrained=True, quantize=False))


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
