# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Smoke tests for the default observer and fake-quant factories exported by
# torch.ao.quantization, plus checks of the reduce_range setting in default qconfigs per backend.
# Importing has no side effects beyond imports. Running the file directly raises a RuntimeError
# because the tests are not wired into the test discovery list.

import torch
import torch.ao.quantization
from torch.testing._internal.common_utils import TestCase


class TestDefaultObservers(TestCase):
    # [CLASS DOC] Holds two lists of attribute names found in torch.ao.quantization: default
    # observer factories and default fake-quantize factories. Each test instantiates every named
    # factory with no arguments and runs the resulting module forward on one random tensor.
    observers = [
        "default_affine_fixed_qparams_observer",
        "default_debug_observer",
        "default_dynamic_quant_observer",
        "default_placeholder_observer",
        "default_fixed_qparams_range_0to1_observer",
        "default_fixed_qparams_range_neg1to1_observer",
        "default_float_qparams_observer",
        "default_float_qparams_observer_4bit",
        "default_histogram_observer",
        "default_observer",
        "default_per_channel_weight_observer",
        "default_reuse_input_observer",
        "default_symmetric_fixed_qparams_observer",
        "default_weight_observer",
        "per_channel_weight_observer_range_neg_127_to_127",
        "weight_observer_range_neg_127_to_127",
    ]

    fake_quants = [
        "default_affine_fixed_qparams_fake_quant",
        "default_dynamic_fake_quant",
        "default_embedding_fake_quant",
        "default_embedding_fake_quant_4bit",
        "default_fake_quant",
        "default_fixed_qparams_range_0to1_fake_quant",
        "default_fixed_qparams_range_neg1to1_fake_quant",
        "default_fused_act_fake_quant",
        "default_fused_per_channel_wt_fake_quant",
        "default_fused_wt_fake_quant",
        "default_histogram_fake_quant",
        "default_per_channel_weight_fake_quant",
        "default_symmetric_fixed_qparams_fake_quant",
        "default_weight_fake_quant",
        "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
        "fused_wt_fake_quant_range_neg_127_to_127",
    ]

    def _get_observer_ins(self, observer):
        # [FUNCTION DOC] HELPER: Look up the attribute with the given name in
        # torch.ao.quantization and call it with no arguments, returning the created observer or
        # fake-quant module. Raises AttributeError if the name does not exist.
        obs_func = getattr(torch.ao.quantization, observer)
        return obs_func()

    def test_observers(self) -> None:
        # [FUNCTION DOC] Builds every observer named in the observers list and runs its forward on
        # a random four dimensional float tensor. Only checks that nothing raises; no outputs are
        # compared.
        t = torch.rand(1, 2, 3, 4)
        for observer in self.observers:
            obs = self._get_observer_ins(observer)
            obs.forward(t)

    def test_fake_quants(self) -> None:
        # [FUNCTION DOC] Same smoke test as the observer test but over the fake_quants list: each
        # default fake-quantize module is created and run forward on a random float tensor, with no
        # assertions on the result beyond absence of exceptions.
        t = torch.rand(1, 2, 3, 4)
        for observer in self.fake_quants:
            obs = self._get_observer_ins(observer)
            obs.forward(t)


class TestQConfig(TestCase):
    # [CLASS DOC] Tests for the default qconfig helpers. REDUCE_RANGE_DICT maps each backend name
    # to a pair of booleans: the expected reduce_range of the activation observer and of the
    # weight observer.

    REDUCE_RANGE_DICT = {
        'fbgemm': (True, False),
        'qnnpack': (False, False),
        'onednn': (False, False),
        'x86': (True, False),
    }

    def test_reduce_range_qat(self) -> None:
        # [FUNCTION DOC] For each backend and for qat qconfig versions zero and one, builds the
        # default QAT qconfig, instantiates its activation and weight fake-quant modules and
        # asserts that the reduce_range flag on the inner activation_post_process equals the
        # expected pair entries.
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():
            for version in range(2):
                qconfig = torch.ao.quantization.get_default_qat_qconfig(backend, version)

                fake_quantize_activ = qconfig.activation()
                self.assertEqual(fake_quantize_activ.activation_post_process.reduce_range, reduce_ranges[0])

                fake_quantize_weight = qconfig.weight()
                self.assertEqual(fake_quantize_weight.activation_post_process.reduce_range, reduce_ranges[1])

    def test_reduce_range(self) -> None:
        # [FUNCTION DOC] For each backend and version zero only, builds the default non-QAT
        # qconfig, instantiates the activation and weight observers and asserts that their
        # reduce_range flags match the expected pair for that backend.
        for backend, reduce_ranges in self.REDUCE_RANGE_DICT.items():
            for version in range(1):
                qconfig = torch.ao.quantization.get_default_qconfig(backend, version)

                fake_quantize_activ = qconfig.activation()
                self.assertEqual(fake_quantize_activ.reduce_range, reduce_ranges[0])

                fake_quantize_weight = qconfig.weight()
                self.assertEqual(fake_quantize_weight.reduce_range, reduce_ranges[1])

if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
