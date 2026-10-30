# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Migration tests checking that the legacy nn.quantized, nn.quantizable, nn.qat and
# nn.intrinsic import paths still expose the same objects as their torch.ao counterparts. Each
# test passes a hard-coded name list to the base-class helper. Direct execution is refused via
# raise_on_run_directly pointing to test/test_quantization.py.

from torch.testing._internal.common_utils import raise_on_run_directly

from .common import AOMigrationTestCase


class TestAOMigrationNNQuantized(AOMigrationTestCase):
    # [CLASS DOC] Import-migration tests for torch.nn.quantized, torch.nn.quantizable and
    # torch.nn.qat versus their torch.ao locations. Stateless; every test calls the base-class
    # function-import helper.
    def test_functional_import(self):
        # [FUNCTION DOC] Compares 22 functional ops (pooling, convolutions, interpolate, linear,
        # activations, clamp, upsample variants) between the legacy
        # torch.nn.quantized.functional module and the torch.ao.nn.quantized.functional module
        # with the shared helper: each listed name must resolve to an equal object with equal
        # hash, else AssertionError.
        r"""Tests the migration of the torch.nn.quantized.functional"""
        function_list = [
            "avg_pool2d",
            "avg_pool3d",
            "adaptive_avg_pool2d",
            "adaptive_avg_pool3d",
            "conv1d",
            "conv2d",
            "conv3d",
            "interpolate",
            "linear",
            "max_pool1d",
            "max_pool2d",
            "celu",
            "leaky_relu",
            "hardtanh",
            "hardswish",
            "threshold",
            "elu",
            "hardsigmoid",
            "clamp",
            "upsample",
            "upsample_bilinear",
            "upsample_nearest",
        ]
        self._test_function_import("functional", function_list, base="nn.quantized")

    def test_modules_import(self):
        # [FUNCTION DOC] Compares quantized module classes (norms, conv, embedding, linear,
        # pooling, activations, quantize/dequantize) and the wrapper functional classes between
        # the legacy torch.nn.quantized.modules module and the torch.ao.nn.quantized.modules
        # module with the shared helper: each listed name must resolve to an equal object with
        # equal hash, else AssertionError.
        module_list = [
            # Modules
            "BatchNorm2d",
            "BatchNorm3d",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
            "ConvTranspose3d",
            "DeQuantize",
            "ELU",
            "Embedding",
            "EmbeddingBag",
            "GroupNorm",
            "Hardswish",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
            "LayerNorm",
            "LeakyReLU",
            "Linear",
            "MaxPool2d",
            "Quantize",
            "ReLU6",
            "Sigmoid",
            "Softmax",
            "Dropout",
            # Wrapper modules
            "FloatFunctional",
            "FXFloatFunctional",
            "QFunctional",
        ]
        self._test_function_import("modules", module_list, base="nn.quantized")

    def test_modules_activation(self):
        # [FUNCTION DOC] Compares the activation modules ReLU6, Hardswish, ELU, LeakyReLU,
        # Sigmoid and Softmax between the legacy torch.nn.quantized.modules.activation module
        # and the torch.ao.nn.quantized.modules.activation module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "ReLU6",
            "Hardswish",
            "ELU",
            "LeakyReLU",
            "Sigmoid",
            "Softmax",
        ]
        self._test_function_import(
            "activation", function_list, base="nn.quantized.modules"
        )

    def test_modules_batchnorm(self):
        # [FUNCTION DOC] Compares BatchNorm2d and BatchNorm3d between the legacy
        # torch.nn.quantized.modules.batchnorm module and the
        # torch.ao.nn.quantized.modules.batchnorm module with the shared helper: each listed
        # name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "BatchNorm2d",
            "BatchNorm3d",
        ]
        self._test_function_import(
            "batchnorm", function_list, base="nn.quantized.modules"
        )

    def test_modules_conv(self):
        # [FUNCTION DOC] Compares the reverse-repeat padding helper and the Conv and
        # ConvTranspose classes between the legacy torch.nn.quantized.modules.conv module and
        # the torch.ao.nn.quantized.modules.conv module with the shared helper: each listed name
        # must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "_reverse_repeat_padding",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
            "ConvTranspose3d",
        ]

        self._test_function_import("conv", function_list, base="nn.quantized.modules")

    def test_modules_dropout(self):
        # [FUNCTION DOC] Compares the Dropout class between the legacy
        # torch.nn.quantized.modules.dropout module and the
        # torch.ao.nn.quantized.modules.dropout module with the shared helper: each listed name
        # must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "Dropout",
        ]
        self._test_function_import(
            "dropout", function_list, base="nn.quantized.modules"
        )

    def test_modules_embedding_ops(self):
        # [FUNCTION DOC] Compares the embedding packed-params, Embedding and EmbeddingBag
        # classes between the legacy torch.nn.quantized.modules.embedding_ops module and the
        # torch.ao.nn.quantized.modules.embedding_ops module with the shared helper: each listed
        # name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "EmbeddingPackedParams",
            "Embedding",
            "EmbeddingBag",
        ]
        self._test_function_import(
            "embedding_ops", function_list, base="nn.quantized.modules"
        )

    def test_modules_functional_modules(self):
        # [FUNCTION DOC] Compares the functional wrapper classes between the legacy
        # torch.nn.quantized.modules.functional_modules module and the
        # torch.ao.nn.quantized.modules.functional_modules module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "FloatFunctional",
            "FXFloatFunctional",
            "QFunctional",
        ]
        self._test_function_import(
            "functional_modules", function_list, base="nn.quantized.modules"
        )

    def test_modules_linear(self):
        # [FUNCTION DOC] Compares the linear module classes between the legacy
        # torch.nn.quantized.modules.linear module and the torch.ao.nn.quantized.modules.linear
        # module with the shared helper: each listed name must resolve to an equal object with
        # equal hash, else AssertionError.
        function_list = [
            "Linear",
            "LinearPackedParams",
        ]
        self._test_function_import("linear", function_list, base="nn.quantized.modules")

    def test_modules_normalization(self):
        # [FUNCTION DOC] Compares the normalization module classes between the legacy
        # torch.nn.quantized.modules.normalization module and the
        # torch.ao.nn.quantized.modules.normalization module with the shared helper: each listed
        # name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "LayerNorm",
            "GroupNorm",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
        ]
        self._test_function_import(
            "normalization", function_list, base="nn.quantized.modules"
        )

    def test_modules_utils(self):
        # [FUNCTION DOC] Compares the helper names exported by the utils module between the
        # legacy torch.nn.quantized.modules.utils module and the
        # torch.ao.nn.quantized.modules.utils module with the shared helper: each listed name
        # must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "_ntuple_from_first",
            "_pair_from_first",
            "_quantize_weight",
            "_hide_packed_params_repr",
            "WeightedQuantizedModule",
        ]
        self._test_function_import("utils", function_list, base="nn.quantized.modules")

    def test_import_nn_quantized_dynamic_import(self):
        # [FUNCTION DOC] Compares the dynamic quantized module names between the legacy
        # torch.nn.quantized.dynamic module and the torch.ao.nn.quantized.dynamic module with
        # the shared helper: each listed name must resolve to an equal object with equal hash,
        # else AssertionError.
        module_list = [
            # Modules
            "Linear",
            "LSTM",
            "GRU",
            "LSTMCell",
            "RNNCell",
            "GRUCell",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
            "ConvTranspose3d",
        ]
        self._test_function_import("dynamic", module_list, base="nn.quantized")

    def test_import_nn_quantizable_activation(self):
        # [FUNCTION DOC] Compares the quantizable activation module names between the legacy
        # torch.nn.quantizable.modules.activation module and the
        # torch.ao.nn.quantizable.modules.activation module with the shared helper: each listed
        # name must resolve to an equal object with equal hash, else AssertionError.
        module_list = [
            # Modules
            "MultiheadAttention",
        ]
        self._test_function_import(
            "activation", module_list, base="nn.quantizable.modules"
        )

    def test_import_nn_quantizable_rnn(self):
        # [FUNCTION DOC] Compares the quantizable RNN module names between the legacy
        # torch.nn.quantizable.modules.rnn module and the torch.ao.nn.quantizable.modules.rnn
        # module with the shared helper: each listed name must resolve to an equal object with
        # equal hash, else AssertionError.
        module_list = [
            # Modules
            "LSTM",
            "LSTMCell",
        ]
        self._test_function_import("rnn", module_list, base="nn.quantizable.modules")

    def test_import_nn_qat_conv(self):
        # [FUNCTION DOC] Compares the QAT convolution module names between the legacy
        # torch.nn.qat.modules.conv module and the torch.ao.nn.qat.modules.conv module with the
        # shared helper: each listed name must resolve to an equal object with equal hash, else
        # AssertionError.
        module_list = [
            "Conv1d",
            "Conv2d",
            "Conv3d",
        ]
        self._test_function_import("conv", module_list, base="nn.qat.modules")

    def test_import_nn_qat_embedding_ops(self):
        # [FUNCTION DOC] Compares the QAT embedding module names between the legacy
        # torch.nn.qat.modules.embedding_ops module and the
        # torch.ao.nn.qat.modules.embedding_ops module with the shared helper: each listed name
        # must resolve to an equal object with equal hash, else AssertionError.
        module_list = [
            "Embedding",
            "EmbeddingBag",
        ]
        self._test_function_import("embedding_ops", module_list, base="nn.qat.modules")

    def test_import_nn_qat_linear(self):
        # [FUNCTION DOC] Compares the QAT linear module names between the legacy
        # torch.nn.qat.modules.linear module and the torch.ao.nn.qat.modules.linear module with
        # the shared helper: each listed name must resolve to an equal object with equal hash,
        # else AssertionError.
        module_list = [
            "Linear",
        ]
        self._test_function_import("linear", module_list, base="nn.qat.modules")

    def test_import_nn_qat_dynamic_linear(self):
        # [FUNCTION DOC] Compares the dynamic QAT linear module names between the legacy
        # torch.nn.qat.dynamic.modules.linear module and the
        # torch.ao.nn.qat.dynamic.modules.linear module with the shared helper: each listed name
        # must resolve to an equal object with equal hash, else AssertionError.
        module_list = [
            "Linear",
        ]
        self._test_function_import("linear", module_list, base="nn.qat.dynamic.modules")


class TestAOMigrationNNIntrinsic(AOMigrationTestCase):
    # [CLASS DOC] Import-migration tests for torch.nn.intrinsic (fused modules, QAT and
    # quantized variants) versus torch.ao.nn.intrinsic. Stateless; most tests call the
    # base-class function-import helper.
    def test_modules_import_nn_intrinsic(self):
        # [FUNCTION DOC] Compares the intrinsic module names between the legacy
        # torch.nn.intrinsic module and the torch.ao.nn.intrinsic module with the shared helper:
        # each listed name must resolve to an equal object with equal hash, else AssertionError.
        module_list = [
            # Modules
            "_FusedModule",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "LinearReLU",
            "BNReLU2d",
            "BNReLU3d",
            "LinearBn1d",
        ]
        self._test_function_import("intrinsic", module_list, base="nn")

    def test_modules_nn_intrinsic_fused(self):
        # [FUNCTION DOC] Compares the fused module names between the legacy
        # torch.nn.intrinsic.modules.fused module and the torch.ao.nn.intrinsic.modules.fused
        # module with the shared helper: each listed name must resolve to an equal object with
        # equal hash, else AssertionError.
        function_list = [
            "_FusedModule",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "LinearReLU",
            "BNReLU2d",
            "BNReLU3d",
            "LinearBn1d",
        ]
        self._test_function_import("fused", function_list, base="nn.intrinsic.modules")

    def test_modules_import_nn_intrinsic_qat(self):
        # [FUNCTION DOC] Compares the intrinsic QAT module names between the legacy
        # torch.nn.intrinsic.qat module and the torch.ao.nn.intrinsic.qat module with the shared
        # helper: each listed name must resolve to an equal object with equal hash, else
        # AssertionError.
        module_list = [
            "LinearReLU",
            "LinearBn1d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "ConvBn1d",
            "ConvBn2d",
            "ConvBn3d",
            "ConvBnReLU1d",
            "ConvBnReLU2d",
            "ConvBnReLU3d",
            "update_bn_stats",
            "freeze_bn_stats",
        ]
        self._test_function_import("qat", module_list, base="nn.intrinsic")

    def test_modules_intrinsic_qat_conv_fused(self):
        # [FUNCTION DOC] Compares the fused QAT convolution module names between the legacy
        # torch.nn.intrinsic.qat.modules.conv_fused module and the
        # torch.ao.nn.intrinsic.qat.modules.conv_fused module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "ConvBn1d",
            "ConvBnReLU1d",
            "ConvReLU1d",
            "ConvBn2d",
            "ConvBnReLU2d",
            "ConvReLU2d",
            "ConvBn3d",
            "ConvBnReLU3d",
            "ConvReLU3d",
            "update_bn_stats",
            "freeze_bn_stats",
        ]
        self._test_function_import(
            "conv_fused", function_list, base="nn.intrinsic.qat.modules"
        )

    def test_modules_intrinsic_qat_linear_fused(self):
        # [FUNCTION DOC] Compares the fused QAT linear module names between the legacy
        # torch.nn.intrinsic.qat.modules.linear_fused module and the
        # torch.ao.nn.intrinsic.qat.modules.linear_fused module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "LinearBn1d",
        ]
        self._test_function_import(
            "linear_fused", function_list, base="nn.intrinsic.qat.modules"
        )

    def test_modules_intrinsic_qat_linear_relu(self):
        # [FUNCTION DOC] Compares the QAT linear-ReLU module names between the legacy
        # torch.nn.intrinsic.qat.modules.linear_relu module and the
        # torch.ao.nn.intrinsic.qat.modules.linear_relu module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "LinearReLU",
        ]
        self._test_function_import(
            "linear_relu", function_list, base="nn.intrinsic.qat.modules"
        )

    def test_modules_import_nn_intrinsic_quantized(self):
        # [FUNCTION DOC] Compares the intrinsic quantized module names between the legacy
        # torch.nn.intrinsic.quantized module and the torch.ao.nn.intrinsic.quantized module
        # with the shared helper: each listed name must resolve to an equal object with equal
        # hash, else AssertionError.
        module_list = [
            "BNReLU2d",
            "BNReLU3d",
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
            "LinearReLU",
        ]
        self._test_function_import("quantized", module_list, base="nn.intrinsic")

    def test_modules_intrinsic_quantized_bn_relu(self):
        # [FUNCTION DOC] Compares the quantized batchnorm-ReLU module names between the legacy
        # torch.nn.intrinsic.quantized.modules.bn_relu module and the
        # torch.ao.nn.intrinsic.quantized.modules.bn_relu module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "BNReLU2d",
            "BNReLU3d",
        ]
        self._test_function_import(
            "bn_relu", function_list, base="nn.intrinsic.quantized.modules"
        )

    def test_modules_intrinsic_quantized_conv_relu(self):
        # [FUNCTION DOC] Compares the quantized conv-ReLU module names between the legacy
        # torch.nn.intrinsic.quantized.modules.conv_relu module and the
        # torch.ao.nn.intrinsic.quantized.modules.conv_relu module with the shared helper: each
        # listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "ConvReLU1d",
            "ConvReLU2d",
            "ConvReLU3d",
        ]
        self._test_function_import(
            "conv_relu", function_list, base="nn.intrinsic.quantized.modules"
        )

    def test_modules_intrinsic_quantized_linear_relu(self):
        # [FUNCTION DOC] Compares the quantized linear-ReLU module names between the legacy
        # torch.nn.intrinsic.quantized.modules.linear_relu module and the
        # torch.ao.nn.intrinsic.quantized.modules.linear_relu module with the shared helper:
        # each listed name must resolve to an equal object with equal hash, else AssertionError.
        function_list = [
            "LinearReLU",
        ]
        self._test_function_import(
            "linear_relu", function_list, base="nn.intrinsic.quantized.modules"
        )

    def test_modules_no_import_nn_intrinsic_quantized_dynamic(self):
        # [FUNCTION DOC] Smoke test: imports torch and accesses the dynamic subpackage of
        # intrinsic quantized under both torch.ao.nn and the legacy torch.nn path; passes if
        # neither access raises. Contains a TODO to generalize it.
        # TODO(future PR): generalize this
        import torch

        _ = torch.ao.nn.intrinsic.quantized.dynamic
        _ = torch.nn.intrinsic.quantized.dynamic


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
