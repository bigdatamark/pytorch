# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Unit tests for the backend configuration classes: DTypeConfig and
# DTypeWithConstraints conversion to and from dictionaries (including legacy dictionary form), the
# BackendPatternConfig setters and dictionary round trip including internal keys, and BackendConfig
# name, pattern registration and dictionary round trip. Importing it has no side effects; running
# it directly raises a RuntimeError pointing to the test runner.

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
from torch.testing._internal.common_quantization import QuantizationTestCase

from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from torch.ao.quantization.fuser_method_mappings import _sequential_wrapper2
from torch.ao.quantization.fx.quantize_handler import _default_root_node_getter


class TestBackendConfig(QuantizationTestCase):
    # [CLASS DOC] Test case for the backend config API. Class attributes hold shared fixtures:
    # three DTypeConfig objects (plain, dynamic, with quantization range constraints), their legacy
    # and current dictionary forms, a fuser method, and sample internal-option values. Helper
    # methods build matching pattern configs and dictionaries for a fused Linear and ReLU pattern
    # and for the add function.

    # =============
    #  DTypeConfig
    # =============

    dtype_config1 = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float
    )

    dtype_config2 = DTypeConfig(
        input_dtype=torch.float16,
        output_dtype=torch.float,
        is_dynamic=True
    )

    activation_dtype_with_constraints = DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12,
    )

    weight_dtype_with_constraints = DTypeWithConstraints(
        dtype=torch.qint8,
        quant_min_lower_bound=-128,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12,
    )

    dtype_config3 = DTypeConfig(
        input_dtype=activation_dtype_with_constraints,
        output_dtype=activation_dtype_with_constraints,
        weight_dtype=weight_dtype_with_constraints,
    )

    dtype_config_dict1_legacy = {
        "input_dtype": torch.quint8,
        "output_dtype": torch.quint8,
        "weight_dtype": torch.qint8,
        "bias_dtype": torch.float,
    }

    dtype_config_dict2_legacy = {
        "input_dtype": torch.float16,
        "output_dtype": torch.float,
        "is_dynamic": True,
    }

    dtype_config_dict1 = {
        "input_dtype": DTypeWithConstraints(dtype=torch.quint8),
        "output_dtype": DTypeWithConstraints(torch.quint8),
        "weight_dtype": DTypeWithConstraints(torch.qint8),
        "bias_dtype": torch.float,
    }

    dtype_config_dict2 = {
        "input_dtype": DTypeWithConstraints(dtype=torch.float16),
        "output_dtype": DTypeWithConstraints(dtype=torch.float),
        "is_dynamic": True,
    }

    dtype_config_dict3 = {
        "input_dtype": activation_dtype_with_constraints,
        "output_dtype": activation_dtype_with_constraints,
        "weight_dtype": weight_dtype_with_constraints,
    }

    def test_dtype_config_from_dict(self):
        # [FUNCTION DOC] Asserts that building a DTypeConfig from each dictionary fixture (legacy
        # plain and dynamic, current plain and dynamic, and the one with constraints) equals the
        # corresponding config fixture.
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict1_legacy), self.dtype_config1)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict2_legacy), self.dtype_config2)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict1), self.dtype_config1)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict2), self.dtype_config2)
        self.assertEqual(DTypeConfig.from_dict(self.dtype_config_dict3), self.dtype_config3)

    def test_dtype_config_to_dict(self):
        # [FUNCTION DOC] Asserts that converting each config fixture to a dictionary equals the
        # current-format dictionary fixture, where plain dtypes are wrapped in constraint objects.
        self.assertEqual(self.dtype_config1.to_dict(), self.dtype_config_dict1)
        self.assertEqual(self.dtype_config2.to_dict(), self.dtype_config_dict2)
        self.assertEqual(self.dtype_config3.to_dict(), self.dtype_config_dict3)

    # ======================
    #  BackendPatternConfig
    # ======================

    _fuser_method = _sequential_wrapper2(nni.LinearReLU)

    _num_tensor_args_to_observation_type = {
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    _input_type_to_index = {
        "bias": 0,
        "input": 1,
        "weight": 2,
    }

    def _extra_inputs_getter(self, p):
        # [FUNCTION DOC] HELPER: Dummy extra-inputs getter taking a pattern argument; ignores it
        # and returns a one-element tuple holding a random 3 by 3 tensor.
        return (torch.rand(3, 3),)

    def _get_backend_op_config1(self):
        # [FUNCTION DOC] HELPER: Builds a pattern config for the Linear and ReLU pair with a
        # different-observer observation type, both dtype configs, Linear as root module, the QAT
        # Linear, the reference quantized Linear, the fused LinearReLU module and the fuser method
        # fixture.
        return BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)) \
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
            .add_dtype_config(self.dtype_config1) \
            .add_dtype_config(self.dtype_config2) \
            .set_root_module(torch.nn.Linear) \
            .set_qat_module(nnqat.Linear) \
            .set_reference_quantized_module(nnqr.Linear) \
            .set_fused_module(nni.LinearReLU) \
            .set_fuser_method(self._fuser_method)

    def _get_backend_op_config2(self):
        # [FUNCTION DOC] HELPER: Builds a pattern config for the add function with the dynamic
        # dtype config and the internal root node getter, extra inputs getter, argument-count to
        # observation-type mapping and input-type to index mapping fixtures.
        return BackendPatternConfig(torch.add) \
            .add_dtype_config(self.dtype_config2) \
            ._set_root_node_getter(_default_root_node_getter) \
            ._set_extra_inputs_getter(self._extra_inputs_getter) \
            ._set_num_tensor_args_to_observation_type(self._num_tensor_args_to_observation_type) \
            ._set_input_type_to_index(self._input_type_to_index)

    def _get_backend_pattern_config_dict1(self):
        # [FUNCTION DOC] HELPER: Returns the dictionary form expected for the first pattern config:
        # pattern, observation type, current-format dtype dictionaries, root, QAT, reference
        # quantized and fused modules and the fuser method.
        return {
            "pattern": (torch.nn.Linear, torch.nn.ReLU),
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [self.dtype_config_dict1, self.dtype_config_dict2],
            "root_module": torch.nn.Linear,
            "qat_module": nnqat.Linear,
            "reference_quantized_module_for_root": nnqr.Linear,
            "fused_module": nni.LinearReLU,
            "fuser_method": self._fuser_method,
        }

    def _get_backend_pattern_config_dict2(self):
        # [FUNCTION DOC] HELPER: Returns the dictionary form expected for the second pattern
        # config: the add pattern, observation type, dynamic dtype dictionary and the four
        # internal-option entries.
        return {
            "pattern": torch.add,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [self.dtype_config_dict2],
            "root_node_getter": _default_root_node_getter,
            "extra_inputs_getter": self._extra_inputs_getter,
            "num_tensor_args_to_observation_type": self._num_tensor_args_to_observation_type,
            "input_type_to_index": self._input_type_to_index,
        }

    def test_backend_op_config_set_observation_type(self):
        # [FUNCTION DOC] Checks that a new pattern config defaults to the different-observer
        # observation type and that the setter changes it to the share-observer type.
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertEqual(conf.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        conf.set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        self.assertEqual(conf.observation_type, ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)

    def test_backend_op_config_add_dtype_config(self):
        # [FUNCTION DOC] Checks that a new pattern config has no dtype configs and that adding two
        # appends them in order.
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertEqual(len(conf.dtype_configs), 0)
        conf.add_dtype_config(self.dtype_config1)
        conf.add_dtype_config(self.dtype_config2)
        self.assertEqual(len(conf.dtype_configs), 2)
        self.assertEqual(conf.dtype_configs[0], self.dtype_config1)
        self.assertEqual(conf.dtype_configs[1], self.dtype_config2)

    def test_backend_op_config_set_root_module(self):
        # [FUNCTION DOC] Checks that the root module is initially unset and is stored after calling
        # the setter.
        conf = BackendPatternConfig(nni.LinearReLU)
        self.assertTrue(conf.root_module is None)
        conf.set_root_module(torch.nn.Linear)
        self.assertEqual(conf.root_module, torch.nn.Linear)

    def test_backend_op_config_set_qat_module(self):
        # [FUNCTION DOC] Checks that the QAT module is initially unset and is stored after calling
        # the setter.
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertTrue(conf.qat_module is None)
        conf.set_qat_module(nnqat.Linear)
        self.assertEqual(conf.qat_module, nnqat.Linear)

    def test_backend_op_config_set_reference_quantized_module(self):
        # [FUNCTION DOC] Checks that the reference quantized module is initially unset and is
        # stored after calling the setter.
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertTrue(conf.reference_quantized_module is None)
        conf.set_reference_quantized_module(nnqr.Linear)
        self.assertEqual(conf.reference_quantized_module, nnqr.Linear)

    def test_backend_op_config_set_fused_module(self):
        # [FUNCTION DOC] Checks that the fused module is initially unset and is stored after
        # calling the setter.
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        self.assertTrue(conf.fused_module is None)
        conf.set_fused_module(nni.LinearReLU)
        self.assertEqual(conf.fused_module, nni.LinearReLU)

    def test_backend_op_config_set_fuser_method(self):
        # [FUNCTION DOC] Checks that the fuser method is initially unset and is stored after
        # calling the setter.
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        self.assertTrue(conf.fuser_method is None)
        conf.set_fuser_method(self._fuser_method)
        self.assertEqual(conf.fuser_method, self._fuser_method)

    def test_backend_op_config_set_root_node_getter(self):
        # [FUNCTION DOC] Checks that the internal root node getter is initially unset and is stored
        # after calling its internal setter.
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
        self.assertTrue(conf._root_node_getter is None)
        conf._set_root_node_getter(_default_root_node_getter)
        self.assertEqual(conf._root_node_getter, _default_root_node_getter)

    def test_backend_op_config_set_extra_inputs_getter(self):
        # [FUNCTION DOC] Checks that the internal extra inputs getter is initially unset and is
        # stored after calling its internal setter.
        conf = BackendPatternConfig(torch.nn.Linear)
        self.assertTrue(conf._extra_inputs_getter is None)
        conf._set_extra_inputs_getter(self._extra_inputs_getter)
        self.assertEqual(conf._extra_inputs_getter, self._extra_inputs_getter)

    def test_backend_op_config_set_num_tensor_args_to_observation_type(self):
        # [FUNCTION DOC] Checks that the internal argument-count to observation-type mapping is
        # initially empty and equals the fixture after calling its internal setter.
        conf = BackendPatternConfig(torch.add)
        self.assertEqual(len(conf._num_tensor_args_to_observation_type), 0)
        conf._set_num_tensor_args_to_observation_type(self._num_tensor_args_to_observation_type)
        self.assertEqual(conf._num_tensor_args_to_observation_type, self._num_tensor_args_to_observation_type)

    def test_backend_op_config_set_input_type_to_index(self):
        # [FUNCTION DOC] Checks that the internal input-type to index mapping is initially empty
        # and equals the fixture after calling its internal setter.
        conf = BackendPatternConfig(torch.addmm)
        self.assertEqual(len(conf._input_type_to_index), 0)
        conf._set_input_type_to_index(self._input_type_to_index)
        self.assertEqual(conf._input_type_to_index, self._input_type_to_index)

    def test_backend_op_config_from_dict(self):
        # [FUNCTION DOC] Builds pattern configs from the two dictionary helpers and checks every
        # field. For the first, public fields match and the internal options stay unset or empty;
        # for the second, public module fields are unset and the internal getters and mappings
        # match the fixtures.
        conf_dict1 = self._get_backend_pattern_config_dict1()
        conf1 = BackendPatternConfig.from_dict(conf_dict1)
        self.assertEqual(conf1.pattern, (torch.nn.Linear, torch.nn.ReLU))
        self.assertEqual(conf1.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        self.assertEqual(conf1.root_module, torch.nn.Linear)
        self.assertEqual(conf1.qat_module, nnqat.Linear)
        self.assertEqual(conf1.reference_quantized_module, nnqr.Linear)
        self.assertEqual(conf1.fused_module, nni.LinearReLU)
        self.assertEqual(conf1.fuser_method, self._fuser_method)
        self.assertTrue(conf1._root_node_getter is None)
        self.assertTrue(conf1._extra_inputs_getter is None)
        self.assertEqual(len(conf1._num_tensor_args_to_observation_type), 0)
        self.assertEqual(len(conf1._input_type_to_index), 0)
        # Test temporary/internal keys
        conf_dict2 = self._get_backend_pattern_config_dict2()
        conf2 = BackendPatternConfig.from_dict(conf_dict2)
        self.assertEqual(conf2.pattern, torch.add)
        self.assertEqual(conf2.observation_type, ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
        self.assertTrue(conf2.root_module is None)
        self.assertTrue(conf2.qat_module is None)
        self.assertTrue(conf2.reference_quantized_module is None)
        self.assertTrue(conf2.fused_module is None)
        self.assertTrue(conf2.fuser_method is None)
        self.assertEqual(conf2._root_node_getter, _default_root_node_getter)
        self.assertEqual(conf2._extra_inputs_getter, self._extra_inputs_getter)
        self.assertEqual(conf2._num_tensor_args_to_observation_type, self._num_tensor_args_to_observation_type)
        self.assertEqual(conf2._input_type_to_index, self._input_type_to_index)

    def test_backend_op_config_to_dict(self):
        # [FUNCTION DOC] Asserts that converting each of the two helper pattern configs to a
        # dictionary equals the corresponding dictionary helper result.
        conf1 = self._get_backend_op_config1()
        conf2 = self._get_backend_op_config2()
        conf_dict1 = self._get_backend_pattern_config_dict1()
        conf_dict2 = self._get_backend_pattern_config_dict2()
        self.assertEqual(conf1.to_dict(), conf_dict1)
        self.assertEqual(conf2.to_dict(), conf_dict2)

    # ===============
    #  BackendConfig
    # ===============

    def test_backend_config_set_name(self):
        # [FUNCTION DOC] Checks that a backend config keeps the name given at construction and that
        # the name setter replaces it.
        conf = BackendConfig("name1")
        self.assertEqual(conf.name, "name1")
        conf.set_name("name2")
        self.assertEqual(conf.name, "name2")

    def test_backend_config_set_backend_pattern_config(self):
        # [FUNCTION DOC] Checks that a new backend config has no configs, and that registering two
        # pattern configs stores them in the internal mapping keyed by pattern, where the Linear
        # and ReLU pair is stored in reversed order (ReLU first) and the add function stays as is.
        conf = BackendConfig("name1")
        self.assertEqual(len(conf.configs), 0)
        backend_op_config1 = self._get_backend_op_config1()
        backend_op_config2 = self._get_backend_op_config2()
        conf.set_backend_pattern_config(backend_op_config1)
        self.assertEqual(conf._pattern_complex_format_to_config, {
            (torch.nn.ReLU, torch.nn.Linear): backend_op_config1,
        })
        conf.set_backend_pattern_config(backend_op_config2)
        self.assertEqual(conf._pattern_complex_format_to_config, {
            (torch.nn.ReLU, torch.nn.Linear): backend_op_config1,
            torch.add: backend_op_config2
        })

    def test_backend_config_from_dict(self):
        # [FUNCTION DOC] Builds a backend config from a dictionary with a name and the two pattern
        # dictionaries; checks the name, that two configs exist, that both the reversed Linear and
        # ReLU key and the add key are present, and that each stored config converts back to its
        # original dictionary.
        op_dict1 = self._get_backend_pattern_config_dict1()
        op_dict2 = self._get_backend_pattern_config_dict2()
        conf_dict = {
            "name": "name1",
            "configs": [op_dict1, op_dict2],
        }
        conf = BackendConfig.from_dict(conf_dict)
        self.assertEqual(conf.name, "name1")
        self.assertEqual(len(conf.configs), 2)
        key1 = (torch.nn.ReLU, torch.nn.Linear)
        key2 = torch.add
        self.assertTrue(key1 in conf._pattern_complex_format_to_config)
        self.assertTrue(key2 in conf._pattern_complex_format_to_config)
        self.assertEqual(conf._pattern_complex_format_to_config[key1].to_dict(), op_dict1)
        self.assertEqual(conf._pattern_complex_format_to_config[key2].to_dict(), op_dict2)

    def test_backend_config_to_dict(self):
        # [FUNCTION DOC] Builds a backend config by chaining registration of the two pattern
        # configs and asserts its dictionary form holds the name and the list of the two pattern
        # dictionaries in registration order.
        op1 = self._get_backend_op_config1()
        op2 = self._get_backend_op_config2()
        op_dict1 = self._get_backend_pattern_config_dict1()
        op_dict2 = self._get_backend_pattern_config_dict2()
        conf = BackendConfig("name1").set_backend_pattern_config(op1).set_backend_pattern_config(op2)
        conf_dict = {
            "name": "name1",
            "configs": [op_dict1, op_dict2],
        }
        self.assertEqual(conf.to_dict(), conf_dict)

if __name__ == '__main__':
    raise RuntimeError("This _test file is not meant to be run directly, use:\n\n"
                       "\tpython _test/_test_quantization.py TESTNAME\n\n"
                       "instead.")
