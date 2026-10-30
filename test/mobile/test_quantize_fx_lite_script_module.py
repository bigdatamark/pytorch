# Owner(s): ["oncall: mobile"]
# [MODULE DOC] FX graph mode quantization tests for the lite interpreter. Each test quantizes
# a small model with prepare_fx and convert_fx, then uses the QuantizationLiteTestCase helper
# to compare the scripted module against its mobile (lite interpreter) round trip.
# KNOWN ISSUE: the main block calls run_tests without importing it (flagged noqa F821), so
# running this file directly raises NameError.

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.utils.bundled_inputs
from torch.ao.quantization import default_qconfig, float_qparams_weight_only_qconfig

# graph mode quantization based on fx
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.testing._internal.common_quantization import (
    LinearModelWithSubmodule,
    NodeSpec as ns,
    QuantizationLiteTestCase,
)


class TestLiteFuseFx(QuantizationLiteTestCase):
    # [CLASS DOC] Quantization lite-interpreter test case; tests mirror those in the FX
    # quantization test suite and only verify script/mobile output equivalence.
    # Tests from:
    # ./caffe2/test/quantization/fx/test_quantize_fx.py

    def test_embedding(self):
        # [FUNCTION DOC] Quantizes a model with a single embedding layer under three qconfigs
        # (float qparams weight-only, none, default) and compares scripted vs mobile outputs
        # on random integer indices. The expected node specs in the list are not asserted.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper module holding one Embedding (10 entries, dim 12).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the embedding submodule stored as emb.
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                # [FUNCTION DOC] HELPER: Looks up the given indices in the embedding layer.
                return self.emb(indices)

        model = M().eval()
        indices = torch.randint(low=0, high=10, size=(20,))

        ns.call_module(nnq.Embedding)
        configs = [
            (float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)),
            (None, ns.call_module(nn.Embedding)),
            (default_qconfig, ns.call_module(nn.Embedding)),
        ]

        for qconfig, _ in configs:
            qconfig_dict = {"": qconfig}
            m = prepare_fx(
                model,
                qconfig_dict,
                example_inputs=torch.randint(low=0, high=10, size=(20,)),
            )
            m = convert_fx(m)
            self._compare_script_and_mobile(m, input=indices)

    def test_conv2d(self):
        # [FUNCTION DOC] Quantizes a two-conv model with the default qconfig except that the
        # module named conv1 is excluded, so only the second conv is quantized, then compares
        # scripted and mobile outputs on a random 1x1x1x1 input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Two stacked 1x1 Conv2d layers (conv1, conv2), 1 channel each.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two convolution submodules.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv1 then conv2 to the input.
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        m = M().eval()
        qconfig_dict = {"": default_qconfig, "module_name": [("conv1", None)]}
        m = prepare_fx(m, qconfig_dict, example_inputs=torch.randn(1, 1, 1, 1))
        data = torch.randn(1, 1, 1, 1)
        m = convert_fx(m)
        # first conv is quantized, second conv is not quantized
        self._compare_script_and_mobile(m, input=data)

    def test_submodule(self):
        # [FUNCTION DOC] Quantizes LinearModelWithSubmodule with the qnnpack default qconfig
        # for three module-name configurations (everything, submodule subm skipped, layer fc
        # skipped) and compares scripted and mobile outputs on a random 5x5 input.
        # test quantizing complete module, submodule and linear layer
        configs = [
            {},
            {"module_name": [("subm", None)]},
            {"module_name": [("fc", None)]},
        ]
        for config in configs:
            model = LinearModelWithSubmodule().eval()
            qconfig_dict = {
                "": torch.ao.quantization.get_default_qconfig("qnnpack"),
                **config,
            }
            model = prepare_fx(
                model,
                qconfig_dict,
                example_inputs=torch.randn(5, 5),
            )
            quant = convert_fx(model)

            x = torch.randn(5, 5)
            self._compare_script_and_mobile(quant, input=x)


if __name__ == "__main__":
    run_tests()  # noqa: F821
