# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
# [MODULE DOC] Eager-mode post-training quantization tests: reference-module equivalence for conv,
# conv-transpose and linear, activation ops, functional ops, static PTQ flows (prepare, calibrate,
# convert, one-line quantize) over flat, nested and custom-qconfig models, custom module classes,
# embedding and embedding-bag quantization, normalization, hook preservation, and dynamic PTQ
# (linear, nested models, RNN and RNN cell modules, embeddings). Import-time side effects: asserts
# that hypothesis deadlines are disabled via the hypothesis utilities helper, and imports numpy
# after that call. Running the file directly raises an error pointing to the quantization test
# driver.

from hypothesis import given, strategies as st

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.testing._internal.hypothesis_utils as hu
from torch.ao.quantization import (
    convert,
    default_dynamic_qconfig,
    default_dynamic_quant_observer,
    default_qconfig,
    default_weight_observer,
    DeQuantStub,
    FixedQParamsObserver,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    per_channel_dynamic_qconfig,
    PerChannelMinMaxObserver,
    prepare,
    prepare_qat,
    QConfig,
    quantize,
    quantize_dynamic,
    QuantStub,
    QuantWrapper,
)
from torch.nn.utils.rnn import PackedSequence

# annotated models
from torch.testing._internal.common_quantization import (
    ActivationsTestModel,
    AnnotatedCustomConfigNestedModel,
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    AnnotatedSkipQuantModel,
    AnnotatedSubNestedModel,
    AnnotatedTwoLayerLinearModel,
    convert_dynamic,
    EmbeddingBagModule,
    EmbeddingModule,
    EmbeddingWithStaticLinear,
    LinearReluLinearModel,
    ModelWithFunctionals,
    NestedModel,
    NormalizationTestModel,
    prepare_dynamic,
    QuantizationTestCase,
    QuantStubModel,
    ResNetBase,
    RNNCellDynamicModel,
    RNNDynamicModel,
    SingleLayerLinearDynamicModel,
    skipIfNoFBGEMM,
    test_only_eval_fn,
    TwoLayerLinearModel,
)
from torch.testing._internal.common_quantized import (
    override_qengines,
    override_quantized_engine,
    supported_qengines,
)


hu.assert_deadline_disabled()

# Standard library
import numpy as np


class TestQuantizeEagerOps(QuantizationTestCase):
    # [CLASS DOC] Tests of individual quantizable op types under eager-mode static quantization:
    # comparison of quantized modules against reference-quantized models, int16 reference modules,
    # activation modules and functional modules.
    @override_qengines
    def _test_reference_module_impl(
        self,
        float_module_class,
        quantized_module_class,
        extra_module_kwargs,
        input_size,
    ):
        # [FUNCTION DOC] Shared implementation (decorated to run per available quantization
        # engine) for the conv, conv-transpose and linear tests. Defines a float model with quant
        # stub, the given float module and a dequant stub, and a reference model with
        # quantize/dequantize pairs around the module and copies the weights. Skips (returns) for
        # unsupported engines and for qnnpack. Quantizes the first with prepare, one calibration
        # pass and convert, checks the stub, module and dequant types, then converts the reference
        # model with the reference option and asserts both outputs are equal.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Float model: quant stub, the module under test (built from the
            # enclosing arguments), dequant stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the quant stub, the module built from the
                # enclosing float class and keyword arguments, and the dequant stub.
                super().__init__()
                self.conv = float_module_class(**extra_module_kwargs)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: quantize stub, module, dequantize stub.
                x = self.quant(x)
                x = self.conv(x)
                x = self.dequant(x)
                return x

        class RefM(torch.nn.Module):
            # [CLASS DOC] HELPER: Reference model: the module under test sandwiched between
            # quantize-then-dequantize stub pairs on both its input and output, so that reference
            # quantization yields fake-quantized semantics.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module under test and two pairs of quant and
                # dequant stubs.
                super().__init__()
                self.conv = float_module_class(**extra_module_kwargs)
                self.quant1 = QuantStub()
                self.dequant1 = DeQuantStub()
                self.quant2 = QuantStub()
                self.dequant2 = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: first stub pair, module, second stub pair.
                x = self.quant1(x)
                x = self.dequant1(x)
                x = self.conv(x)
                x = self.quant2(x)
                x = self.dequant2(x)
                return x

        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines or qengine == "qnnpack":
            return  # qnnpack does not support nnq.ConvTranspose3d

        data = torch.randn(*input_size, dtype=torch.float)
        original_m = M()
        original_ref_m = RefM()

        original_ref_m.conv.weight = torch.nn.Parameter(original_m.conv.weight.detach())
        original_ref_m.conv.bias = torch.nn.Parameter(original_m.conv.bias.detach())

        original_m.qconfig = torch.ao.quantization.default_qconfig

        m = prepare(original_m)
        # calibration
        m(data)
        m = convert(m)
        # check if the module is properly quantized
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.conv), quantized_module_class)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)
        res = m(data)

        # quantize the reference model
        original_ref_m.eval()
        original_ref_m.qconfig = torch.ao.quantization.default_qconfig

        ref_m = prepare(original_ref_m)
        ref_m(data)
        ref_m = convert(ref_m, is_reference=True)
        ref_res = ref_m(data)
        self.assertEqual(res, ref_res)

    def test_conv_1d(self):
        # [FUNCTION DOC] Runs the reference comparison for 1d convolution with unit channels and
        # kernel on a small input.
        self._test_reference_module_impl(
            nn.Conv1d,
            nnq.Conv1d,
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1},
            (16, 1, 1),
        )

    def test_conv_2d(self):
        # [FUNCTION DOC] Runs the reference comparison for 2d convolution with unit channels and
        # kernel.
        self._test_reference_module_impl(
            nn.Conv2d,
            nnq.Conv2d,
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1},
            (16, 1, 10, 10),
        )

    def test_conv_3d(self):
        # [FUNCTION DOC] Runs the reference comparison for 3d convolution with unit channels and
        # kernel.
        self._test_reference_module_impl(
            nn.Conv3d,
            nnq.Conv3d,
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1},
            (16, 1, 10, 10, 10),
        )

    def test_conv_transpose_1d(self):
        # [FUNCTION DOC] Runs the reference comparison for 1d transposed convolution with unit
        # channels and kernel.
        self._test_reference_module_impl(
            nn.ConvTranspose1d,
            nnq.ConvTranspose1d,
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1},
            (16, 1, 1),
        )

    def test_conv_transpose_2d(self):
        # [FUNCTION DOC] Runs the reference comparison for 2d transposed convolution with unit
        # channels and kernel.
        self._test_reference_module_impl(
            nn.ConvTranspose2d,
            nnq.ConvTranspose2d,
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1},
            (16, 1, 10, 10),
        )

    def test_conv_transpose_3d(self):
        # [FUNCTION DOC] Runs the reference comparison for 3d transposed convolution; the shared
        # helper skips qnnpack since it lacks this module.
        self._test_reference_module_impl(
            nn.ConvTranspose3d,
            nnq.ConvTranspose3d,
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1},
            (16, 1, 10, 10, 10),
        )

    def test_linear(self):
        # [FUNCTION DOC] Runs the reference comparison for a linear layer with five inputs and ten
        # outputs.
        self._test_reference_module_impl(
            nn.Linear, nnq.Linear, {"in_features": 5, "out_features": 10}, (16, 5)
        )

    @override_qengines
    def test_int16_reference_module(self):
        # [FUNCTION DOC] Checks int16-style reference quantization of a transposed conv (decorated
        # to run per engine). Builds a reference model with random non-trainable weights and bias,
        # a custom qconfig using moving-average observers with the wide integer dtype restricted
        # to the 16-bit range (symmetric for weights), prepares, calibrates once and converts with
        # the reference option. Asserts the converted module's stored weight scale equals the
        # scale computed by an independently built observer fed the same weights. Returns early
        # for unsupported engines.
        class RefM(torch.nn.Module):
            # [CLASS DOC] HELPER: Reference model with a transposed conv between two
            # quantize-then-dequantize stub pairs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a transposed conv with one channel and unit
                # kernel plus two quant and dequant stub pairs.
                super().__init__()
                self.conv = nn.ConvTranspose2d(1, 1, 1)
                self.quant1 = QuantStub()
                self.dequant1 = DeQuantStub()
                self.quant2 = QuantStub()
                self.dequant2 = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: first stub pair, conv, second stub pair.
                x = self.quant1(x)
                x = self.dequant1(x)
                x = self.conv(x)
                x = self.quant2(x)
                x = self.dequant2(x)
                return x

        input_size = (16, 1, 10, 10)
        data = torch.randn(*input_size, dtype=torch.float)

        original_ref_m = RefM()
        rand_w = torch.randn_like(original_ref_m.conv.weight)
        rand_b = torch.randn_like(original_ref_m.conv.bias)
        original_ref_m.conv.weight = torch.nn.Parameter(rand_w, requires_grad=False)
        original_ref_m.conv.bias = torch.nn.Parameter(rand_b, requires_grad=False)

        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines:
            return
        from torch.ao.quantization.observer import MovingAverageMinMaxObserver

        weight_obs = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint32,
            # set qmin and qmax to represent qint16
            quant_min=-1 * (2**15),
            quant_max=(2**15) - 1,
            qscheme=torch.per_tensor_symmetric,
        )
        act_obs = MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint32,
            quant_min=-1 * (2**15),
            quant_max=(2**15) - 1,
        )
        custom_qconfig = QConfig(activation=act_obs, weight=weight_obs)

        # quantize the reference model
        original_ref_m.eval()
        original_ref_m.qconfig = custom_qconfig

        ref_m = prepare(original_ref_m)
        # calibration
        ref_m(torch.randn(*input_size, dtype=torch.float))

        ref_m = convert(ref_m, is_reference=True)

        myobs = MovingAverageMinMaxObserver(
            averaging_constant=0.5,
            dtype=torch.qint32,
            # set qmin and qmax to represent qint16
            quant_min=-1 * (2**15),
            quant_max=(2**15) - 1,
            qscheme=torch.per_tensor_symmetric,
        )
        result = myobs(rand_w)
        qparams = myobs.calculate_qparams()
        self.assertEqual(ref_m.conv.weight_scale, qparams[0])

    def _test_activation_op_impl(
        self, float_module_class, quantized_module_class, extra_module_kwargs
    ):
        # [FUNCTION DOC] HELPER: Shared implementation for activation-module tests (docstring
        # describes the extra keyword arguments used to build the float module). Wraps the
        # activation between quant and dequant stubs in eval mode, applies the default qconfig,
        # prepares, checks observers, converts and asserts the activation was swapped to the given
        # quantized class.
        """Implementation for testing common activation ops like leaky relu
        Args:
            extra_module_kwargs: keyword args to instantiate the float module
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Model: quant stub, the activation module under test, dequant
            # stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the activation from the enclosing class and
                # keyword arguments plus the stubs.
                super().__init__()
                self.activation_op = float_module_class(**extra_module_kwargs)
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: quantize stub, activation, dequantize stub.
                x = self.quant(x)
                x = self.activation_op(x)
                x = self.dequant(x)
                return x

        m = M().eval()
        m.qconfig = default_qconfig
        m = prepare(m)
        self.checkObservers(m)
        m = convert(m)
        self.assertEqual(type(m.activation_op), quantized_module_class)

    def test_leaky_relu(self):
        # [FUNCTION DOC] Checks the leaky relu activation converts to the quantized leaky relu
        # class.
        self._test_activation_op_impl(
            nn.LeakyReLU, nnq.LeakyReLU, {"negative_slope": 0.1, "inplace": False}
        )

    def test_relu(self):
        # [FUNCTION DOC] Checks relu is left as the regular relu class after conversion (the
        # quantized class given is the same type).
        self._test_activation_op_impl(nn.ReLU, nn.ReLU, {"inplace": False})

    # Histogram Observers are slow, so have no-deadline to ensure test doesn't time out
    @given(train_mode=st.booleans())
    def test_functional_module(self, train_mode):
        # [FUNCTION DOC] Hypothesis-parameterized (train or eval mode) test of a model containing
        # float functional modules. Checks scriptability, prepares with QAT or default qconfig
        # depending on the mode, asserts observers and no leftover prep modules, calibrates,
        # converts, and verifies add, cat, add-relu and matmul functionals became quantized
        # functional modules, no qconfig remains and the converted model is scriptable with a
        # quantized input.
        model = ModelWithFunctionals()
        x = torch.rand(10, 1, dtype=torch.float)
        xq = torch.quantize_per_tensor(x, 0.01, 30, torch.quint8)
        self.checkScriptable(model, [[x]], check_save_load=True)
        if train_mode:
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
            model = prepare_qat(model)
        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
            model = prepare(model)
        # Check if observers and quant/dequant nodes are inserted
        self.checkNoPrepModules(model)
        self.checkObservers(model)
        # Calibrate
        model(xq.dequantize())
        model = convert(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts the four functional attributes are
            # quantized functional modules, there are no prep modules and no qconfigs.
            self.checkNoPrepModules(model)
            self.assertEqual(type(model.myadd), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.mycat), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.myadd_relu), torch.ao.nn.quantized.QFunctional)
            self.assertEqual(type(model.mymatmul), torch.ao.nn.quantized.QFunctional)
            self.checkNoQconfig(model)

        checkQuantized(model)
        self.checkScriptable(model, [[xq]], check_save_load=True)


class TestQuantizeEagerPTQStatic(QuantizationTestCase):
    # [CLASS DOC] Static post-training quantization flow tests: prepare, calibrate, convert, and
    # the one-line quantize API (out of place and in place) on single-layer, two-layer, nested,
    # custom-qconfig, skip-quant, manual-stub, resnet, normalization, activation, state-dict,
    # hook, embedding, custom-module and misc models.
    def test_single_layer(self):
        # [FUNCTION DOC] For each supported engine, quantizes a single-linear model with the
        # engine's default qconfig. Verifies prep modules and observers after prepare, quantized
        # wrapped linear after convert, scriptability and no qconfig. Then repeats through the
        # one-line API on a fresh model (checking the original state dict keys are unchanged) and
        # the in-place API.
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                model = prepare(model)
                # Check if observers and quant/dequant nodes are inserted
                self.checkNoPrepModules(model)
                self.checkHasPrepModules(model.fc1)
                self.checkObservers(model)

                test_only_eval_fn(model, self.calib_data)
                model = convert(model)

                def checkQuantized(model):
                    # [FUNCTION DOC] HELPER: Nested checker. Asserts prep modules only on the
                    # first layer, a wrapped quantized linear, scriptability and no qconfig.
                    self.checkNoPrepModules(model)
                    self.checkHasPrepModules(model.fc1)
                    self.checkWrappedQuantizedLinear(model.fc1)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API - out of place version
                base = AnnotatedSingleLayerLinearModel(qengine)
                base.qconfig = qconfig
                keys_before = set(base.state_dict().keys())
                model = quantize(base, test_only_eval_fn, [self.calib_data])
                checkQuantized(model)
                keys_after = set(base.state_dict().keys())
                self.assertEqual(
                    keys_before, keys_after
                )  # simple check that nothing changed

                # in-place version
                model = AnnotatedSingleLayerLinearModel(qengine)
                model.qconfig = qconfig
                quantize(model, test_only_eval_fn, [self.calib_data], inplace=True)
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_two_layers(self):
        # [FUNCTION DOC] With the fbgemm engine, quantizes only the second linear layer of a
        # two-layer model: asserts prep modules only on it, that the first layer stays a regular
        # linear, then verifies the converted structure and the one-line API gives the same.
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        with override_quantized_engine("fbgemm"):
            model = AnnotatedTwoLayerLinearModel()
            model = prepare(model)

            self.checkNoPrepModules(model)
            self.checkObservers(model)
            self.checkNoPrepModules(model.fc1)
            self.checkHasPrepModules(model.fc2)

            test_only_eval_fn(model, self.calib_data)
            model = convert(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts only the second layer is a
                # wrapped quantized linear, the first is float, scriptability and no qconfig.
                self.checkNoPrepModules(model)
                self.checkNoPrepModules(model.fc1)
                self.checkHasPrepModules(model.fc2)
                self.assertEqual(type(model.fc1), torch.nn.Linear)
                self.checkWrappedQuantizedLinear(model.fc2)
                test_only_eval_fn(model, self.calib_data)
                self.checkScriptable(model, self.calib_data)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize(
                AnnotatedTwoLayerLinearModel(), test_only_eval_fn, [self.calib_data]
            )
            checkQuantized(model)

    def test_nested1(self):
        # [FUNCTION DOC] For each engine, quantizes a nested model where the top-level third layer
        # and the first layer of the second submodule are annotated, leaving the second layer of
        # that submodule float. Checks prep modules before and after conversion, then checks the
        # one-line API.
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = AnnotatedNestedModel(qengine)

                def checkPrepModules(model, before_calib=False):
                    # [FUNCTION DOC] HELPER: Asserts observers (only when requested, before
                    # calibration) and which nested modules have prepared state: only the
                    # annotated linears do.
                    if before_calib:
                        self.checkObservers(model)
                    self.checkNoPrepModules(model)
                    self.checkNoPrepModules(model.sub1)
                    self.checkNoPrepModules(model.sub1.fc)
                    self.checkNoPrepModules(model.sub1.relu)
                    self.checkNoPrepModules(model.sub2)
                    self.checkHasPrepModules(model.sub2.fc1)
                    self.checkNoPrepModules(model.sub2.fc2)
                    self.checkHasPrepModules(model.fc3)

                model = prepare(model)
                checkPrepModules(model, True)
                test_only_eval_fn(model, self.calib_data)
                model = convert(model)

                def checkQuantized(model):
                    # [FUNCTION DOC] HELPER: Nested checker. Asserts float layers stay linear,
                    # annotated layers are wrapped quantized linears, scriptability and no
                    # qconfig.
                    checkPrepModules(model)
                    self.checkLinear(model.sub1.fc)
                    self.checkWrappedQuantizedLinear(model.fc3)
                    self.checkWrappedQuantizedLinear(model.sub2.fc1)
                    self.checkLinear(model.sub2.fc2)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API
                model = quantize(
                    AnnotatedNestedModel(qengine), test_only_eval_fn, [self.calib_data]
                )
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_nested2(self):
        # [FUNCTION DOC] With the fbgemm engine, quantizes a nested model whose second submodule
        # is wrapped as a whole. Checks prep modules, then that converted submodule linears are
        # quantized (not wrapped) and the first submodule is float, plus the one-line API.
        model = AnnotatedSubNestedModel()
        model = prepare(model)

        def checkPrepModules(model, before_calib=False):
            # [FUNCTION DOC] HELPER: Asserts observers when requested and that only the wrapped
            # submodule and the top-level third layer have prep modules.
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkHasPrepModules(model.sub2)
            self.checkNoPrepModules(model.sub2.module.fc1)
            self.checkNoPrepModules(model.sub2.module.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(model, True)

        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts the float parts remain float, the
            # nested linears are quantized, the top layer is wrapped quantized, scriptability and
            # no qconfig.
            checkPrepModules(model)
            self.checkLinear(model.sub1.fc)
            self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
            self.checkQuantizedLinear(model.sub2.module.fc1)
            self.checkQuantizedLinear(model.sub2.module.fc2)
            self.checkWrappedQuantizedLinear(model.fc3)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)
            self.checkNoQconfig(model)

        checkQuantized(model)

        # test one line API
        model = quantize(
            AnnotatedSubNestedModel(), test_only_eval_fn, [self.calib_data]
        )
        checkQuantized(model)

    def test_nested3(self):
        # [FUNCTION DOC] For each engine, quantizes a nested model where a child qconfig overrides
        # the parent one, so the second submodule's two linears and the top-level layer all get
        # prepared. Verifies the converted wrapped quantized linears and the one-line API.
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = AnnotatedCustomConfigNestedModel()
                model = prepare(model)

                def checkPrepModules(model, before_calib=False):
                    # [FUNCTION DOC] HELPER: Asserts observers when requested and prep modules for
                    # both second-submodule linears and the top layer, none for the first
                    # submodule.
                    if before_calib:
                        self.checkObservers(model)
                    self.checkNoPrepModules(model)
                    self.checkNoPrepModules(model.sub1)
                    self.checkNoPrepModules(model.sub1.fc)
                    self.checkNoPrepModules(model.sub1.relu)
                    self.checkNoPrepModules(model.sub2)
                    self.checkHasPrepModules(model.sub2.fc1)
                    self.checkHasPrepModules(model.sub2.fc2)
                    self.checkHasPrepModules(model.fc3)

                checkPrepModules(model, True)

                test_only_eval_fn(model, self.calib_data)
                model = convert(model)

                def checkQuantized(model):
                    # [FUNCTION DOC] HELPER: Nested checker. Asserts the three annotated linears
                    # are wrapped quantized linears, scriptability and no qconfig.
                    checkPrepModules(model)
                    self.checkWrappedQuantizedLinear(model.sub2.fc1)
                    self.checkWrappedQuantizedLinear(model.sub2.fc2)
                    self.checkWrappedQuantizedLinear(model.fc3)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API
                model = quantize(
                    AnnotatedCustomConfigNestedModel(),
                    test_only_eval_fn,
                    [self.calib_data],
                )
                checkQuantized(model)

    def test_skip_quant(self):
        # [FUNCTION DOC] For each engine, quantizes a model with a float top-level layer and a
        # submodule wrapped with quant and dequant stubs. After convert, asserts the layer is
        # float, the submodule holds quant and dequant modules, its linears are quantized and its
        # relus stay float. Repeats with the one-line API.
        r"""The case when we want to skip quantizing some layers"""
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = AnnotatedSkipQuantModel(qengine)
                model = prepare(model)
                self.checkObservers(model)

                test_only_eval_fn(model, self.calib_data)
                model = convert(model)

                def checkQuantized(model):
                    # [FUNCTION DOC] HELPER: Nested checker. Asserts the layer-by-layer expected
                    # types after conversion, scriptability and no qconfig.
                    self.checkLinear(model.fc)
                    self.checkQuantDequant(model.sub)
                    self.checkQuantizedLinear(model.sub.module.fc1)
                    self.checkQuantizedLinear(model.sub.module.fc2)
                    self.assertEqual(type(model.sub.module.relu1), nn.ReLU)
                    self.assertEqual(type(model.sub.module.relu2), nn.ReLU)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                # test one line API
                model = quantize(
                    AnnotatedSkipQuantModel(qengine),
                    test_only_eval_fn,
                    [self.calib_data],
                )
                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_manual(self):
        # [FUNCTION DOC] With fbgemm, quantizes a model where the user placed quant and dequant
        # stubs by hand, with default configuration: prepare, observer check, calibrate, convert,
        # check the linear is quantized, then one-line API.
        r"""User inserts QuantStub and DeQuantStub in model code
        and call the quantization utility functions.
        """
        model = QuantStubModel()
        # propagate the qconfig of parents to children, model is changed
        # inplace
        model = prepare(model)
        self.checkObservers(model)

        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts the linear is quantized, runs
            # calibration data again, scriptability and no qconfig.
            self.assertEqual(type(model.fc), nnq.Linear)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)
            self.checkNoQconfig(model)

        checkQuantized(model)

        # test one line API
        model = quantize(QuantStubModel(), test_only_eval_fn, [self.calib_data])
        checkQuantized(model)

    def test_resnet_base(self):
        # [FUNCTION DOC] For each engine, fuses a resnet-style bottleneck model, wraps it in a
        # quantization wrapper with the default engine qconfig, prepares, calibrates on image data
        # and converts. Verifies fused conv-relu became quantized, the functional op became a
        # quantized functional module, average pooling stays float and the final linear is
        # quantized.
        r"""Test quantization for bottleneck topology used in resnet/resnext
        and add coverage for conversion of average pool and float functional
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                model = ResNetBase().float().eval()
                model.fuse_model()
                model = QuantWrapper(model)
                model.qconfig = qconfig
                model = prepare(model)
                self.checkObservers(model)
                test_only_eval_fn(model, self.img_data_2d)
                model = convert(model)

                def checkQuantized(model):
                    # [FUNCTION DOC] HELPER: Nested checker. Asserts the expected quantized and
                    # float types, runs image data and checks no qconfig remains.
                    self.assertEqual(
                        type(model.module.conv1), nn.intrinsic.quantized.ConvReLU2d
                    )
                    self.assertEqual(type(model.module.myop), nn.quantized.QFunctional)
                    self.assertEqual(type(model.module.avgpool), nn.AdaptiveAvgPool2d)
                    self.assertEqual(type(model.module.fc), nnq.Linear)

                    test_only_eval_fn(model, self.img_data_2d)
                    self.checkNoQconfig(model)

                checkQuantized(model)

    @skipIfNoFBGEMM
    def test_normalization(self):
        # [FUNCTION DOC] With fbgemm, quantizes a model with layer norm, group norm and 1d, 2d and
        # 3d instance norm: prepare in place, observers, calibrate, convert, then asserts each
        # norm became its quantized class. Also runs the one-line API on a fresh model, but the
        # final check is applied to the earlier model (the one-line result is not verified).
        r"""
        Test quantization of normalization layers
        """
        model = NormalizationTestModel()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        prepare(model, inplace=True)
        self.checkObservers(model)
        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts no prep modules on the norm layers,
            # their quantized classes, scriptability and no qconfig.
            self.checkNoPrepModules(model.layer_norm)
            self.checkNoPrepModules(model.group_norm)
            self.checkNoPrepModules(model.instance_norm1d)
            self.checkNoPrepModules(model.instance_norm2d)
            self.checkNoPrepModules(model.instance_norm3d)
            self.assertEqual(type(model.layer_norm), nnq.LayerNorm)
            self.assertEqual(type(model.group_norm), nnq.GroupNorm)
            self.assertEqual(type(model.instance_norm1d), nnq.InstanceNorm1d)
            self.assertEqual(type(model.instance_norm2d), nnq.InstanceNorm2d)
            self.assertEqual(type(model.instance_norm3d), nnq.InstanceNorm3d)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)
            self.checkNoQconfig(model)

        checkQuantized(model)

        model_oneline = quantize(
            NormalizationTestModel(), test_only_eval_fn, [self.calib_data]
        )
        checkQuantized(model)

    def test_save_load_state_dict(self):
        # [FUNCTION DOC] For each engine, quantizes a wrapped two-layer model, records its
        # reference output and quantized state dict, builds a fresh quantized model without
        # calibration, asserts the state dict keys match, loads the saved state and asserts the
        # output equals the reference.
        r"""Test PTQ flow of creating a model and quantizing it and saving the quantized state_dict
        Load the quantized state_dict for eval and compare results against original model
        """

        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)

                model = prepare(model)
                # calibrate
                test_only_eval_fn(model, self.calib_data)
                model = convert(model)
                x = torch.rand(2, 5, dtype=torch.float)
                ref = model(x)

                quant_state_dict = model.state_dict()

                # Create model again for eval
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                model = prepare(model)
                model = convert(model)
                new_state_dict = model.state_dict()

                # Check to make sure the state dict keys match original model after convert.
                self.assertEqual(
                    set(new_state_dict.keys()), set(quant_state_dict.keys())
                )

                model.load_state_dict(quant_state_dict)

                out = model(x)
                self.assertEqual(ref, out)

    @skipIfNoFBGEMM
    def test_activations(self):
        # [FUNCTION DOC] With fbgemm, quantizes a model with hardswish and elu: asserts observers,
        # quantized classes for both after convert, scriptability, then the one-line API result
        # passes the same check.
        r"""
        Test quantization of activations
        """
        model = ActivationsTestModel()
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        prepare(model, inplace=True)
        self.checkObservers(model)
        test_only_eval_fn(model, self.calib_data)
        model = convert(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts hardswish has no prep modules and
            # both activations are quantized classes, scriptability and no qconfig.
            self.checkNoPrepModules(model.hardswish)
            self.assertEqual(type(model.hardswish), nnq.Hardswish)
            self.assertEqual(type(model.elu), nnq.ELU)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data)
            self.checkNoQconfig(model)

        checkQuantized(model)

        # test one line API
        model_oneline = quantize(
            ActivationsTestModel(), test_only_eval_fn, [self.calib_data]
        )
        checkQuantized(model_oneline)

    @override_qengines
    def test_forward_hooks_preserved(self):
        # [FUNCTION DOC] For each engine, registers user forward pre and post hooks on the linear
        # of a manually stubbed model, prepares and verifies the hook counts (the observer hook
        # adds one post hook before conversion and the user hook stays last), calibrates, converts
        # in place and verifies only the user hooks remain.
        r"""Test post-training static quantization on preserving
        pre forward and post forward hooks of original model
        """
        qengine = torch.backends.quantized.engine
        model = QuantStubModel()
        counter = {
            "pre_forwards": 0,
            "forwards": 0,
        }

        def fw_pre_hook(h_module, input):
            # [FUNCTION DOC] HELPER: Pre-hook that increments the pre-forward counter in the
            # enclosing mapping.
            counter["pre_forwards"] += 1

        def fw_hook(h_module, input, output):
            # [FUNCTION DOC] HELPER: Post-hook that increments the forward counter in the
            # enclosing mapping.
            counter["forwards"] += 1

        model.fc.register_forward_pre_hook(fw_pre_hook)
        model.fc.register_forward_hook(fw_hook)

        model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        model = prepare(model)

        def checkHooksIsPresent(model, before_convert=True):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts the user hooks exist on the linear,
            # hook counts are one pre hook and a post hook count that is two before conversion and
            # one after, that the quant stub carries the observer hook before conversion, and that
            # the user hook is last.
            num_fwd_hooks = 1
            if before_convert:
                self.assertEqual(
                    len(model.quant._forward_hooks.values()),
                    1,
                    "Quantization observer hook has disappeared",
                )
                num_fwd_hooks = 2

            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            self.assertEqual(
                len(model.fc._forward_pre_hooks.values()),
                1,
                "Extra pre forward hooks have appeared on a layer",
            )
            # During static quantization non stub layers are provided with quantization observer hook too
            self.assertEqual(
                len(model.fc._forward_hooks.values()),
                num_fwd_hooks,
                "Extra post forward hooks have appeared on a layer",
            )
            # Implicitly check that fw_hook goes after _observer_forward_hook
            self.assertEqual(
                list(model.fc._forward_hooks.values())[-1],
                fw_hook,
                "_observer_forward_hook is not a first entry of the hooks list",
            )

        checkHooksIsPresent(model, True)
        test_only_eval_fn(model, self.calib_data)
        torch.ao.quantization.convert(model, inplace=True)
        checkHooksIsPresent(model, False)

    @skipIfNoFBGEMM
    def test_quantized_embedding(self):
        # [FUNCTION DOC] With fbgemm, for each of two float-qparams weight-only qconfigs,
        # quantizes an embedding model in place and checks the quantized embedding class, string
        # repr and scriptability with save/load. Then quantizes a model with an embedding and a
        # static linear (no qconfig assigned in the test), checks both quantized and runs it. The
        # weights tensor is unused.
        r"""Test the post-training quantization flow, serialization and scripting
        of embedding modules
        """

        for qconfig in [
            float_qparams_weight_only_qconfig,
            float_qparams_weight_only_qconfig_4bit,
        ]:
            model = EmbeddingModule().eval()
            indices = torch.tensor(
                [
                    9,
                    6,
                    5,
                    7,
                    8,
                    8,
                    9,
                    2,
                    8,
                    6,
                    6,
                    9,
                    1,
                    6,
                    8,
                    8,
                    3,
                    2,
                    3,
                    6,
                    3,
                    6,
                    5,
                    7,
                    0,
                    8,
                    4,
                    6,
                    5,
                    8,
                    2,
                    3,
                ]
            )
            weights = torch.randn(10, 12, dtype=torch.float32)
            model.qconfig = qconfig
            prepare(model, inplace=True)
            convert(model, inplace=True)
            self.assertTrue("QuantizedEmbedding" in str(model))
            self.assertEqual(type(model.emb), torch.ao.nn.quantized.Embedding)
            self.checkScriptable(model, [[indices]], check_save_load=True)

            idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
            offsets = torch.LongTensor([0, 4])
            x = torch.randn(2, 4)
            model = EmbeddingWithStaticLinear().eval()
            prepare(model, inplace=True)
            convert(model, inplace=True)
            self.assertTrue("QuantizedEmbedding" in str(model))
            self.assertTrue("QuantizedLinear" in str(model))
            self.checkQuantizedLinear(model.fc)
            model(idx, offsets, x)

    @skipIfNoFBGEMM
    def test_dequant_stub(self):
        # [FUNCTION DOC] With fbgemm, checks conversion turns quant, linear and dequant stubs into
        # their quantized forms, and that a dequant stub whose qconfig is cleared is left as the
        # float stub.
        m = QuantStubModel().eval()
        prepare(m, inplace=True)
        self.checkObservers(m)
        convert(m, inplace=True)
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.fc), nnq.Linear)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)

        # check DeQuantStub is not swapped when it doesn't have a qconfig
        m2 = QuantStubModel().eval()
        m2.dequant.qconfig = None
        prepare(m2, inplace=True)
        self.checkObservers(m2)
        convert(m2, inplace=True)
        self.assertEqual(type(m2.quant), nnq.Quantize)
        self.assertEqual(type(m2.fc), nnq.Linear)
        self.assertEqual(type(m2.dequant), DeQuantStub)

    def test_quantized_embedding_bag(self):
        # [FUNCTION DOC] Quantizes an embedding-bag model for two float-qparams observer dtypes
        # (8-bit and packed 4-bit) using per-channel float-qparams weights and a dynamic
        # activation observer. Checks the quantized repr, the dynamic quantized module check, and
        # scriptability with save/load using per-sample weights. Then quantizes only the embedding
        # bag of a model that also has a linear and checks the linear remains float.
        r"""Test the post-training quantization flow, serialization and scripting
        of embedding_bag modules
        """
        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        weights = torch.randn(10, 12, dtype=torch.float32)

        for dtype in [torch.quint8, torch.quint4x2]:
            model = EmbeddingBagModule().eval()
            float_qparams_observer = PerChannelMinMaxObserver.with_args(
                dtype=dtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0
            )
            float_qparams_qconfig = QConfig(
                activation=default_dynamic_quant_observer, weight=float_qparams_observer
            )
            model.qconfig = float_qparams_qconfig

            prepare(model, inplace=True)
            quantized_model = convert(model)

            per_sample_weights = torch.from_numpy(
                np.random.uniform(low=0.01, high=0.5, size=[len(indices)]).astype(
                    np.float32
                )
            )

            # Test to make sure module is quantized correctly.
            self.assertTrue("QuantizedEmbeddingBag" in str(quantized_model))
            self.checkDynamicQuantizedModule(
                quantized_model.emb, torch.ao.nn.quantized.EmbeddingBag, torch.quint8
            )
            self.checkScriptable(
                quantized_model,
                [[indices, offsets, per_sample_weights]],
                check_save_load=True,
            )

            class EmbeddingBagWithLinear(torch.nn.Module):
                # [CLASS DOC] HELPER: Model combining an embedding bag (sum mode, last offset
                # included) with a five by five linear; its forward returns the bag output with
                # per-sample weights and the linear output.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the embedding bag and the linear layer.
                    super().__init__()
                    self.emb = torch.nn.EmbeddingBag(
                        num_embeddings=10,
                        embedding_dim=12,
                        include_last_offset=True,
                        scale_grad_by_freq=False,
                        mode="sum",
                    )
                    self.fc = torch.nn.Linear(5, 5)

                def forward(self, indices, offsets, per_sample_weights, linear_in):
                    # [FUNCTION DOC] HELPER: Returns a pair: the embedding bag applied to indices,
                    # offsets and per-sample weights, and the linear applied to the extra input.
                    return self.emb(indices, offsets, per_sample_weights), self.fc(
                        linear_in
                    )

            # Test quantization of embedding_bag layer only
            model2 = EmbeddingBagWithLinear().eval()
            model2.emb.qconfig = float_qparams_qconfig
            prepare(model2, inplace=True)
            quantized_model = convert(model2)

            self.assertTrue("QuantizedEmbeddingBag" in str(quantized_model))
            self.checkLinear(model2.fc)
            self.checkDynamicQuantizedModule(
                quantized_model.emb, torch.ao.nn.quantized.EmbeddingBag, torch.quint8
            )

    @skipIfNoFBGEMM
    def test_custom_module_class(self):
        # [FUNCTION DOC] With fbgemm, tests custom module support: defines float, observed and
        # quantized custom module classes (convert hooks assert the qconfig and observer exist)
        # and registers them through prepare and convert custom config mappings. Quantizes a model
        # containing the custom module, checks observers and converted types, and asserts the
        # output equals a reference model made of regular convs quantized normally.
        class CustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Float custom module wrapping a one-channel conv.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the conv.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv.
                return self.conv(x)

        class ObservedCustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Observed form of the custom module; wraps a given conv and is
            # built from the float module by a class method.
            def __init__(self, conv):
                # [FUNCTION DOC] HELPER: Stores the given conv.
                super().__init__()
                self.conv = conv

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv.
                return self.conv(x)

            @classmethod
            def from_float(cls, float_module):
                # [FUNCTION DOC] HELPER: Class method creating an observed module from a float
                # module; raises an assertion error if the float module has no qconfig; reuses the
                # float conv and copies the qconfig.
                if not hasattr(float_module, "qconfig"):
                    raise AssertionError(
                        f"float_module {type(float_module).__name__} must have qconfig"
                    )
                observed = cls(float_module.conv)
                observed.qconfig = float_module.qconfig
                return observed

        class QuantizedCustomModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Quantized form of the custom module, wrapping a quantized conv.
            def __init__(self, conv):
                # [FUNCTION DOC] HELPER: Stores the given conv.
                super().__init__()
                self.conv = conv

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv.
                return self.conv(x)

            @classmethod
            def from_observed(cls, observed_module):
                # [FUNCTION DOC] HELPER: Class method creating a quantized module from an observed
                # one; raises an assertion error if the observed module lacks a qconfig or an
                # activation post-process; shares the post-process with the conv, converts the
                # conv to its quantized class and wraps it.
                if not hasattr(observed_module, "qconfig"):
                    raise AssertionError(
                        f"observed_module {type(observed_module).__name__} must have qconfig"
                    )
                if not hasattr(observed_module, "activation_post_process"):
                    raise AssertionError(
                        f"observed_module {type(observed_module).__name__} must have activation_post_process"
                    )
                observed_module.conv.activation_post_process = (
                    observed_module.activation_post_process
                )
                quantized = cls(nnq.Conv2d.from_float(observed_module.conv))
                return quantized

        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Container holding one custom module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the custom module.
                super().__init__()
                self.custom = CustomModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the custom module.
                return self.custom(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Top-level model: quant stub, conv, container with the custom
            # module, dequant stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the four children.
                super().__init__()
                self.quant = QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.sub = Sub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: quantize, conv, container, dequantize.
                x = self.quant(x)
                x = self.conv(x)
                x = self.sub(x)
                x = self.dequant(x)
                return x

        class RefM(torch.nn.Module):
            # [CLASS DOC] HELPER: Reference model equivalent to the top-level model but with plain
            # convs in place of the custom module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the stubs and two convs.
                super().__init__()
                self.quant = QuantStub()
                self.conv1 = torch.nn.Conv2d(1, 1, 1)
                self.conv2 = torch.nn.Conv2d(1, 1, 1)
                self.dequant = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: quantize, first conv, second conv,
                # dequantize.
                x = self.quant(x)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.dequant(x)
                return x

        data = torch.randn(1, 1, 1, 1)
        # instantiate M and RefM and align the parameters
        original_m = M()
        original_ref_m = RefM()
        original_ref_m.conv1.weight = torch.nn.Parameter(
            original_m.conv.weight.detach()
        )
        original_ref_m.conv1.bias = torch.nn.Parameter(original_m.conv.bias.detach())
        original_ref_m.conv2.weight = torch.nn.Parameter(
            original_m.sub.custom.conv.weight.detach()
        )
        original_ref_m.conv2.bias = torch.nn.Parameter(
            original_m.sub.custom.conv.bias.detach()
        )

        original_m.qconfig = default_qconfig
        prepare_custom_config_dict = {
            "float_to_observed_custom_module_class": {
                CustomModule: ObservedCustomModule
            }
        }
        convert_custom_config_dict = {
            "observed_to_quantized_custom_module_class": {
                ObservedCustomModule: QuantizedCustomModule
            }
        }
        m = prepare(original_m, prepare_custom_config_dict=prepare_custom_config_dict)
        self.checkObservers(m, None, prepare_custom_config_dict)
        # calibration
        m(data)
        # all activation observers are inserted in the top level module

        # check converted/quantized model
        m = convert(m, convert_custom_config_dict=convert_custom_config_dict)
        # check if the module is properly quantized
        self.assertEqual(type(m.quant), nnq.Quantize)
        self.assertEqual(type(m.conv), nnq.Conv2d)
        self.assertEqual(type(m.sub), Sub)
        self.assertEqual(type(m.sub.custom), QuantizedCustomModule)
        self.assertEqual(type(m.sub.custom.conv), nnq.Conv2d)
        self.assertEqual(type(m.dequant), nnq.DeQuantize)
        res = m(data)

        # quantize the reference model
        original_ref_m.eval()
        original_ref_m.qconfig = default_qconfig
        ref_m = prepare(original_ref_m)
        ref_m(data)
        ref_m = convert(ref_m)
        ref_res = ref_m(data)
        self.assertEqual(res, ref_res)

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_fails_early(self):
        # [FUNCTION DOC] With fbgemm, asserts that preparing a transposed conv that has a
        # per-channel weight observer qconfig raises an assertion error already at prepare time,
        # with a specific message saying per-channel is unsupported for transposed convolutions.
        r"""
        Verifies that attempting to quantize a ConvTranspose module with per-Channel
        weight observers fails in the prepare step, as opposed to the convert step.
        """
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        m.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        with self.assertRaises(AssertionError) as context:
            mp = torch.ao.quantization.prepare(m)
        self.assertTrue(
            str(context.exception)
            == "Per channel weight observer is not supported yet for ConvTranspose{n}d."
        )

    @skipIfNoFBGEMM
    def test_convtranspose_per_channel_qconfig_none(self):
        # [FUNCTION DOC] With fbgemm, asserts that prepare does not fail when the transposed conv
        # has no qconfig even though its parent has a per-channel one.
        r"""
        Verifies that having qconfig==None for conv transpose does not crash
        """
        m = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 1, 1))
        m.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        m[0].qconfig = None
        mp = torch.ao.quantization.prepare(m)

    @skipIfNoFBGEMM
    def test_quantwrapper_attaches_qconfig_to_dequant(self):
        # [FUNCTION DOC] With fbgemm, wraps a conv with a quantization wrapper after setting its
        # qconfig, prepares and converts, and asserts the wrapper's dequant child became the
        # quantized dequantize class.
        qconfig = torch.ao.quantization.default_qconfig

        m = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
        for i in range(len(m)):
            m[i].qconfig = qconfig
            m[i] = torch.ao.quantization.QuantWrapper(m[i])

        mp = torch.ao.quantization.prepare(m)
        mq = torch.ao.quantization.convert(mp)
        self.assertTrue(isinstance(mq[0].dequant, nnq.DeQuantize))

    def test_activations_in_non_leaf_module_list(self):
        # [FUNCTION DOC] Prepares a model of sigmoid, hardsigmoid, softmax and tanh with a
        # fixed-qparams activation observer qconfig and the four types passed as non-leaf observer
        # modules, then asserts each activation module's observer is the fixed-qparams observer
        # from the qconfig and not the default fake quant.
        """
        Ensure activations like `nn.Sigmoid` and `nn.Tanh` are properly handled in
        `non_leaf_module_list`.
        """

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model: quant stub, sigmoid, hardsigmoid, softmax, tanh, dequant
            # stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the six children.
                super().__init__()
                self.quant = QuantStub()
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.softmax = torch.nn.Softmax()
                self.tanh = torch.nn.Tanh()
                self.dequant = DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward pass: quantize, the four activations in order,
                # dequantize.
                x = self.quant(x)
                x = self.sigmoid(x)
                x = self.hardsigmoid(x)
                x = self.softmax(x)
                x = self.tanh(x)
                x = self.dequant(x)
                return x

        qconfig = QConfig(
            activation=FixedQParamsObserver.with_args(scale=123.0, zero_point=0),
            weight=default_weight_observer,
        )
        m = MyModel()
        m.qconfig = qconfig
        m = prepare(
            m,
            observer_non_leaf_module_list=[
                torch.nn.Sigmoid,
                torch.nn.Hardsigmoid,
                torch.nn.Softmax,
                torch.nn.Tanh,
            ],
        )

        # Should use the observer specified in the QConfig instead of the default (FixedQParamsFakeQuantize)
        self.assertTrue(
            isinstance(m.sigmoid.activation_post_process, FixedQParamsObserver)
        )
        self.assertTrue(
            isinstance(m.hardsigmoid.activation_post_process, FixedQParamsObserver)
        )
        self.assertTrue(
            isinstance(m.softmax.activation_post_process, FixedQParamsObserver)
        )
        self.assertTrue(
            isinstance(m.tanh.activation_post_process, FixedQParamsObserver)
        )

    @skipIfNoFBGEMM
    def test_mha_batch_first_attr_is_copied_in_prepare(self):
        # [FUNCTION DOC] With fbgemm, asserts prepare (out of place) preserves the batch-first
        # flag of a multihead attention module for both flag values.
        class TransformerDecoderLayer(nn.Module):
            # [CLASS DOC] HELPER: Decoder layer fixture holding one multihead attention module
            # with dropout 0.1 built from the given sizes and flag.
            def __init__(self, d_model, nhead, batch_first):
                # [FUNCTION DOC] HELPER: Creates the attention module with the given model size,
                # head count and batch-first flag.
                super().__init__()
                self.self_attn = nn.MultiheadAttention(
                    d_model, nhead, dropout=0.1, batch_first=batch_first
                )

        qengine = torch.backends.quantized.engine
        for batch_first in [True, False]:
            model = TransformerDecoderLayer(512, 8, batch_first)
            quantization_config = torch.ao.quantization.get_default_qconfig(qengine)
            model.qconfig = quantization_config
            prepared_model = torch.ao.quantization.prepare(model, inplace=False)
            self.assertTrue(
                prepared_model.self_attn.batch_first == model.self_attn.batch_first
            )


@skipIfNoFBGEMM
class TestQuantizeEagerPTQDynamic(QuantizationTestCase):
    # [CLASS DOC] Dynamic post-training quantization tests (skipped without fbgemm): linear layers
    # with int8 or float16 dynamic qconfigs, nested models, per-channel, linear-relu fusion, RNN
    # and RNN-cell modules, hooks, and embedding ops. Each test also verifies the one-line
    # quantize_dynamic API and that scripting works.
    def test_single_layer(self):
        # [FUNCTION DOC] For both dtypes, dynamically quantizes a single-linear model through the
        # prepare and convert helpers, then via the one-line API out of place (original state dict
        # keys unchanged), in place, and with a set of module types and dtype. Each result is
        # checked for a dynamic quantized linear of the right dtype.
        r"""Dynamic Quantize SingleLayerLinearDynamicModel which has one Linear module,
        make sure it is swapped to nnqd.Linear which is the quantized version of
        the module
        """
        for dtype in [torch.qint8, torch.float16]:
            model = SingleLayerLinearDynamicModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dict = {"fc1": qconfig}
            prepare_dynamic(model, qconfig_dict)
            convert_dynamic(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts the layer is a dynamic quantized
                # linear of the loop dtype, scriptability with save/load, and no qconfig.
                self.checkDynamicQuantizedLinear(model.fc1, dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API - out of place version
            base = SingleLayerLinearDynamicModel()
            keys_before = set(base.state_dict().keys())
            model = quantize_dynamic(base, qconfig_dict)
            checkQuantized(model)
            keys_after = set(base.state_dict().keys())
            self.assertEqual(
                keys_before, keys_after
            )  # simple check that nothing changed

            # in-place version
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, qconfig_dict, inplace=True)
            checkQuantized(model)

            # Test set qconfig
            model = SingleLayerLinearDynamicModel()
            quantize_dynamic(model, {nn.Linear}, inplace=True, dtype=dtype)
            checkQuantized(model)

    def test_two_layers(self):
        # [FUNCTION DOC] For both dtypes, dynamically quantizes only the second layer of a
        # two-layer model using a name-keyed qconfig mapping, the one-line API and a set of names
        # with dtype.
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        for dtype in [torch.qint8, torch.float16]:
            model = TwoLayerLinearModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dict = {"fc2": qconfig}
            prepare_dynamic(model, qconfig_dict)

            convert_dynamic(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts the first layer is float and the
                # second is dynamic quantized, scriptability and no qconfig.
                self.assertEqual(type(model.fc1), torch.nn.Linear)
                self.checkDynamicQuantizedLinear(model.fc2, dtype=dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(TwoLayerLinearModel().eval(), qconfig_dict)
            checkQuantized(model)

            # Test set API
            model = quantize_dynamic(TwoLayerLinearModel().eval(), {"fc2"}, dtype=dtype)
            checkQuantized(model)

    def test_nested1(self):
        # [FUNCTION DOC] For both dtypes, dynamically quantizes the top-level third layer and the
        # first layer of the second submodule of a nested model, via helpers, mapping API and
        # name-set API.
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dict = {"fc3": qconfig, "sub2.fc1": qconfig}

            prepare_dynamic(model, qconfig_dict)
            convert_dynamic(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts annotated linears are dynamic
                # quantized, others float, scriptability and no qconfig.
                self.checkLinear(model.sub1.fc)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkLinear(model.sub2.fc2)
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
            checkQuantized(model)

            model = quantize_dynamic(
                NestedModel().eval(), {"fc3", "sub2.fc1"}, dtype=dtype
            )
            checkQuantized(model)

    def test_nested2(self):
        # [FUNCTION DOC] For both dtypes, dynamically quantizes the whole second submodule and the
        # top-level third layer of a nested model, via helpers, mapping API and name-set API.
        r"""Another test case for quantized, we will quantize all submodules
        of submodule sub2
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dict = {"fc3": qconfig, "sub2": qconfig}
            prepare_dynamic(model, qconfig_dict)

            convert_dynamic(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts the first submodule stays float
                # (relu is a plain relu), the others are dynamic quantized, scriptability and no
                # qconfig.
                self.checkLinear(model.sub1.fc)
                self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            checkQuantized(model)

            # Test set API
            model = quantize_dynamic(NestedModel().eval(), {"fc3", "sub2"}, dtype=dtype)
            checkQuantized(model)

    def test_nested3(self):
        # [FUNCTION DOC] For both dtypes, uses a mapping where a child entry and its parent entry
        # both specify the qconfig, and checks the same layers are dynamic quantized through
        # helpers, mapping API and name-set API.
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dynamic_dict = {
                "fc3": qconfig,
                "sub2": qconfig,
                "sub2.fc1": qconfig,
            }
            prepare_dynamic(model, qconfig_dynamic_dict)

            convert_dynamic(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts the second submodule's linears
                # and the top layer are dynamic quantized, scriptability and no qconfig.
                self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                self.checkDynamicQuantizedLinear(model.fc3, dtype=dtype)
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dynamic_dict)
            checkQuantized(model)

            # Test set API
            model = quantize_dynamic(
                NestedModel().eval(), {"fc3", "sub2", "sub2.fc1"}, dtype=dtype
            )
            checkQuantized(model)

    def test_type_match_rule(self):
        # [FUNCTION DOC] For both dtypes, uses a mapping that disables two named layers and sets
        # the qconfig for the linear type, so all other linears are dynamic quantized and the
        # named ones stay float. Calibration data is run through before and after conversion.
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', All 'torch.nn.Linear' modules are quantized
        """
        for dtype in [torch.qint8, torch.float16]:
            model = NestedModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dict = {"fc3": None, "sub2.fc1": None, torch.nn.Linear: qconfig}

            prepare_dynamic(model, qconfig_dict)
            test_only_eval_fn(model, self.calib_data)
            convert_dynamic(model)

            def checkQuantized(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts quantized and float layers as
                # described, runs data, scriptability and no qconfig.
                self.checkDynamicQuantizedLinear(model.sub1.fc, dtype=dtype)
                self.checkLinear(model.fc3)
                self.checkLinear(model.sub2.fc1)
                self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=dtype)
                test_only_eval_fn(model, self.calib_data)
                self.checkScriptable(model, self.calib_data, check_save_load=True)
                self.checkNoQconfig(model)

            checkQuantized(model)

            # test one line API
            model = quantize_dynamic(NestedModel().eval(), qconfig_dict, dtype=dtype)
            checkQuantized(model)

    def test_per_channel_linear_quantize(self):
        # [FUNCTION DOC] Dynamically quantizes every linear of a nested model with the per-channel
        # dynamic qconfig and checks they are all int8 dynamic quantized linears; repeats via the
        # one-line API.
        r"""Test quantization for per_channel dynamic quantization"""
        model = NestedModel().eval()
        qconfig_dict = {torch.nn.Linear: per_channel_dynamic_qconfig}

        prepare_dynamic(model, qconfig_dict)
        test_only_eval_fn(model, self.calib_data)
        convert_dynamic(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts all four linears are int8 dynamic
            # quantized, runs data, scriptability and no qconfig.
            self.checkDynamicQuantizedLinear(model.sub1.fc, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.fc3, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.sub2.fc1, dtype=torch.qint8)
            self.checkDynamicQuantizedLinear(model.sub2.fc2, dtype=torch.qint8)
            test_only_eval_fn(model, self.calib_data)
            self.checkScriptable(model, self.calib_data, check_save_load=True)
            self.checkNoQconfig(model)

        checkQuantized(model)
        # test one line API
        model = quantize_dynamic(NestedModel().eval(), qconfig_dict)
        checkQuantized(model)

    def test_linear_relu_fusion(self):
        # [FUNCTION DOC] Fuses the first linear and relu of a model, applies the default dynamic
        # qconfig to the whole model with an empty-name key, and checks the fused layer became a
        # dynamic quantized linear-relu while the second linear is a dynamic quantized linear.
        dtype = torch.qint8
        model = LinearReluLinearModel().eval()
        qconfig = default_dynamic_qconfig
        qconfig_dict = {"": qconfig}
        torch.ao.quantization.fuse_modules(model, [["fc1", "relu"]], inplace=True)
        prepare_dynamic(model, qconfig_dict)
        convert_dynamic(model)

        def checkQuantized(model):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts the fused layer is a dynamic
            # linear-relu, the other layer a dynamic linear, scriptability and no qconfig.
            self.checkDynamicQuantizedLinearRelu(model.fc1, dtype)
            self.checkDynamicQuantizedLinear(model.fc2, dtype)
            self.checkScriptable(model, self.calib_data, check_save_load=True)
            self.checkNoQconfig(model)

        checkQuantized(model)

    @given(
        qconfig=st.sampled_from([per_channel_dynamic_qconfig, default_dynamic_qconfig]),
        dtype=st.sampled_from([torch.qint8, torch.float16]),
    )
    def test_quantized_rnn(self, qconfig, dtype):
        # [FUNCTION DOC] Hypothesis-parameterized by qconfig (per-channel or default dynamic) and
        # dtype. For LSTM and GRU wrappers, quantizes dynamically (without a qconfig spec for
        # float16), verifies the dynamic quantized module type and repr, scriptability, and also
        # scripts and runs a wrapper that takes a packed sequence input, comparing scripted
        # against eager through the shared script check helper (no tracing).
        r"""Test dynamic quantization, scriptability and serialization for dynamic quantized lstm modules on int8 and fp16"""
        niter = 10
        x = (
            torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)
            .unsqueeze(0)
            .repeat(niter, 1, 1)
        )
        qconfig_dict = {torch.nn.LSTM: qconfig, torch.nn.GRU: qconfig}

        def checkQuantized(model, module_type):
            # [FUNCTION DOC] HELPER: Nested checker. Asserts the repr string and module type of
            # the quantized recurrent module match the module type name. It reads the enclosing
            # quantized model rather than its own model argument.
            mod_type_map = {
                "LSTM": torch.ao.nn.quantized.dynamic.LSTM,
                "GRU": torch.ao.nn.quantized.dynamic.GRU,
            }
            mod_repr_map = {
                "LSTM": "DynamicQuantizedLSTM",
                "GRU": "DynamicQuantizedGRU",
            }
            self.assertTrue(mod_repr_map[module_type] in str(model_quantized))
            self.checkDynamicQuantizedModule(
                model_quantized.mod, mod_type_map[module_type], dtype
            )

        for module_type in ["LSTM", "GRU"]:
            model = RNNDynamicModel(module_type).eval()

            if dtype == torch.float16:
                model_quantized = quantize_dynamic(model=model, dtype=dtype)
            else:
                model_quantized = quantize_dynamic(
                    model=model, qconfig_spec=qconfig_dict, dtype=dtype
                )

            checkQuantized(model_quantized, module_type)
            self.checkScriptable(model_quantized, [[x]], check_save_load=True)

            class ScriptWrapperPackedLSTM(torch.nn.Module):
                # [CLASS DOC] HELPER: Script-compatible wrapper that applies a recurrent cell
                # module to a packed sequence input and returns the packed output and the pair of
                # final states.
                def __init__(self, cell):
                    # [FUNCTION DOC] HELPER: Stores the wrapped module.
                    super().__init__()
                    self.cell = cell

                def forward(
                    self, x: PackedSequence
                ) -> tuple[PackedSequence, tuple[torch.Tensor, torch.Tensor]]:
                    # [FUNCTION DOC] HELPER: Returns the wrapped module applied to the packed
                    # sequence.
                    return self.cell(x)

            class ScriptWrapperPackedGRU(torch.nn.Module):
                # [CLASS DOC] HELPER: Script-compatible wrapper like the LSTM wrapper but returns
                # the packed output and a single final state tensor.
                def __init__(self, cell):
                    # [FUNCTION DOC] HELPER: Stores the wrapped module.
                    super().__init__()
                    self.cell = cell

                def forward(
                    self, x: PackedSequence
                ) -> tuple[PackedSequence, torch.Tensor]:
                    # [FUNCTION DOC] HELPER: Returns the wrapped module applied to the packed
                    # sequence.
                    return self.cell(x)

            script_wrapper_map = {
                "LSTM": ScriptWrapperPackedLSTM,
                "GRU": ScriptWrapperPackedGRU,
            }
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                x, torch.tensor([10, 5, 2])
            )
            model_with_packed_input = script_wrapper_map[module_type](
                model_quantized.mod
            )
            model_with_packed_input(packed_input)
            scripted = torch.jit.script(model_with_packed_input)
            scripted(packed_input)
            # We cannot trace with input dtype being a packed sequence
            self._checkScriptable(
                model_with_packed_input, scripted, [[packed_input]], True
            )

    @given(
        qconfig=st.sampled_from([per_channel_dynamic_qconfig, default_dynamic_qconfig]),
        dtype=st.sampled_from([torch.qint8, torch.float16]),
    )
    def test_quantized_rnn_cell(self, qconfig, dtype):
        # [FUNCTION DOC] Hypothesis-parameterized by qconfig and dtype. For LSTM cell, GRU cell
        # and two vanilla RNN cell variants, dynamically quantizes the model (skipping float16
        # under qnnpack), then checks quantized class and repr and scriptability with save/load.
        r"""Test dynamic quantization, scriptability and serialization for dynamic quantized rnn cell modules on int8 and fp16"""
        qconfig_dict = {
            torch.nn.LSTMCell: qconfig,
            torch.nn.GRUCell: qconfig,
            torch.nn.RNNCell: qconfig,
        }

        for module_type in ["LSTMCell", "GRUCell", "RNNTanh", "RNNReLU"]:
            model = RNNCellDynamicModel(module_type).eval()
            x = torch.tensor([[100, -155], [-155, 100], [100, -155]], dtype=torch.float)

            if torch.backends.quantized.engine == "qnnpack" and dtype == torch.float16:
                continue
                # fp16 dynamic quant is not supported for qnnpack

            if dtype == torch.float16:
                model_quantized = quantize_dynamic(model=model, dtype=dtype)
            else:
                model_quantized = quantize_dynamic(
                    model=model, qconfig_spec=qconfig_dict, dtype=dtype
                )

            def checkQuantized(model, module_type):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts the repr string and quantized
                # module type for the cell kind (reads the enclosing quantized model for the repr
                # and module check) and that the passed model has no qconfig.
                mod_type_map = {
                    "LSTMCell": torch.ao.nn.quantized.dynamic.LSTMCell,
                    "GRUCell": torch.ao.nn.quantized.dynamic.GRUCell,
                    "RNNTanh": torch.ao.nn.quantized.dynamic.RNNCell,
                    "RNNReLU": torch.ao.nn.quantized.dynamic.RNNCell,
                }

                mod_repr_map = {
                    "LSTMCell": "DynamicQuantizedLSTMCell",
                    "GRUCell": "DynamicQuantizedGRUCell",
                    "RNNTanh": "DynamicQuantizedRNNCell",
                    "RNNReLU": "DynamicQuantizedRNNCell",
                }

                self.assertTrue(mod_repr_map[module_type] in str(model_quantized))
                self.checkDynamicQuantizedModule(
                    model_quantized.mod, mod_type_map[module_type], dtype
                )
                self.checkNoQconfig(model)

            # Smoke test extra reprs
            checkQuantized(model_quantized, module_type)
            self.checkScriptable(model_quantized, [[x]], check_save_load=True)

    def test_forward_hooks_preserved(self):
        # [FUNCTION DOC] For both dtypes, converts a linear model, registers user forward pre and
        # post hooks on its linear, prepares dynamically, checks the hooks are still the only
        # ones, runs data, converts again and re-checks.
        r"""Test post-training dynamic quantization on preserving
        pre forward and post forward hooks of original model
        """
        for dtype in [torch.qint8, torch.float16]:
            model = SingleLayerLinearDynamicModel().eval()
            qconfig = (
                float16_dynamic_qconfig
                if dtype == torch.float16
                else default_dynamic_qconfig
            )
            qconfig_dict = {"fc1": qconfig}
            convert_dynamic(model)

            counter = {
                "pre_forwards": 0,
                "forwards": 0,
            }

            def fw_pre_hook(h_module, input):
                # [FUNCTION DOC] HELPER: Pre-hook that increments the pre-forward counter in the
                # enclosing mapping.
                counter["pre_forwards"] += 1

            def fw_hook(h_module, input, output):
                # [FUNCTION DOC] HELPER: Post-hook that increments the forward counter in the
                # enclosing mapping.
                counter["forwards"] += 1

            model.fc1.register_forward_pre_hook(fw_pre_hook)
            model.fc1.register_forward_hook(fw_hook)
            prepare_dynamic(model, qconfig_dict)

            def checkHooksIsPresent(model):
                # [FUNCTION DOC] HELPER: Nested checker. Asserts the user hooks are present on the
                # linear and that exactly one pre hook and one post hook exist.
                self.assertObjectIn(fw_pre_hook, model.fc1._forward_pre_hooks.values())
                self.assertObjectIn(fw_hook, model.fc1._forward_hooks.values())
                self.assertEqual(
                    len(model.fc1._forward_pre_hooks.values()),
                    1,
                    "Extra pre forward hooks have appeared on a layer",
                )
                self.assertEqual(
                    len(model.fc1._forward_hooks.values()),
                    1,
                    "Extra post forward hooks have appeared on a layer",
                )

            checkHooksIsPresent(model)
            test_only_eval_fn(model, self.calib_data)
            convert_dynamic(model)
            checkHooksIsPresent(model)

    @skipIfNoFBGEMM
    def test_embedding_bag_dynamic(self):
        # [FUNCTION DOC] With fbgemm, dynamically quantizes a model with an embedding bag and a
        # linear using a type-keyed mapping (float-qparams for the bag, dynamic for linear), runs
        # it, and asserts the reprs show the quantized embedding bag and dynamic quantized linear.
        class EmbeddingBagWithLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with an embedding bag (sum mode, last offset included) and
            # a five by five linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the embedding bag and the linear.
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(
                    num_embeddings=10,
                    embedding_dim=12,
                    include_last_offset=True,
                    scale_grad_by_freq=False,
                    mode="sum",
                )
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, indices, offsets, linear_in):
                # [FUNCTION DOC] HELPER: Returns a pair: embedding bag applied to indices and
                # offsets, and linear applied to the extra input.
                return self.emb(indices, offsets), self.fc(linear_in)

        model = EmbeddingBagWithLinear().eval()

        qconfig_dict = {
            torch.nn.EmbeddingBag: float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig,
        }
        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        q_model = quantize_dynamic(model, qconfig_dict)

        q_model(indices, offsets, torch.randn(5, 5))
        self.assertTrue("QuantizedEmbeddingBag" in str(q_model.emb))
        self.assertTrue("DynamicQuantizedLinear" in str(q_model.fc))

    @skipIfNoFBGEMM
    def test_embedding_ops_dynamic(self):
        # [FUNCTION DOC] With fbgemm, dynamically quantizes a model with an embedding and a linear
        # using a type-keyed mapping, asserts the reprs show a quantized embedding and dynamic
        # quantized linear and runs a forward pass.
        class EmbeddingWithLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with an embedding of ten rows by twelve columns and a five
            # by five linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the embedding and the linear.
                super().__init__()
                self.emb = torch.nn.Embedding(
                    num_embeddings=10, embedding_dim=12, scale_grad_by_freq=False
                )
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, indices, linear_in):
                # [FUNCTION DOC] HELPER: Returns a pair: embedding applied to indices and linear
                # applied to the extra input.
                return self.emb(indices), self.fc(linear_in)

        model = EmbeddingWithLinear().eval()
        qconfig_dict = {
            torch.nn.Embedding: float_qparams_weight_only_qconfig,
            torch.nn.Linear: default_dynamic_qconfig,
        }
        indices = torch.tensor(
            [
                9,
                6,
                5,
                7,
                8,
                8,
                9,
                2,
                8,
                6,
                6,
                9,
                1,
                6,
                8,
                8,
                3,
                2,
                3,
                6,
                3,
                6,
                5,
                7,
                0,
                8,
                4,
                6,
                5,
                8,
                2,
                3,
            ]
        )
        q_model = quantize_dynamic(model, qconfig_dict)
        self.assertTrue("QuantizedEmbedding" in str(q_model.emb))
        self.assertTrue("DynamicQuantizedLinear" in str(q_model.fc))
        q_model(indices, torch.randn(5, 5))


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )
