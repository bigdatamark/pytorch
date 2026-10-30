# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Test that a torchvision vision transformer, once exported, contains only core
# ATen operators (no scaled dot product attention op remains). The only test is skipped on
# Linux and macOS and marked as an expected failure. Imports pytest and the legacy export
# module at import time.
import copy
import unittest

import pytest

import torch
import torch._export as export
from torch.testing._internal.common_quantization import skip_if_no_torchvision
from torch.testing._internal.common_utils import IS_LINUX, IS_MACOS, TestCase


def _get_ops_list(m: torch.fx.GraphModule):
    # [FUNCTION DOC] HELPER: Walks the nodes of the graph of an FX graph module and returns
    # a list of the targets of every node whose op kind is call_function, in graph order.
    op_list = []
    for n in m.graph.nodes:
        if n.op == "call_function":
            op_list.append(n.target)
    return op_list


class TestQuantizePT2EModels(TestCase):
    # [CLASS DOC] Test case for PT2E style exported model checks; holds a single ViT export
    # test.
    @unittest.skipIf(
        IS_LINUX or IS_MACOS, "https://github.com/pytorch/pytorch/issues/113506"
    )
    @pytest.mark.xfail
    @skip_if_no_torchvision
    def test_vit_aten_export(self):
        # [FUNCTION DOC] Builds torchvision vit_b_16 with pretrained ImageNet weights, puts
        # it in eval mode, exports it with the strict torch.export API, runs the unwrapped
        # module once on a random 1x3x224x224 input, re-exports it with the legacy export
        # API and collects the call_function targets of the result. Asserts no target
        # mentions scaled dot product.
        # The decorators skip the test on Linux/macOS (issue 113506), mark it xfail, and
        # skip it when torchvision is unavailable, so it effectively never runs there.
        from torchvision.models import vit_b_16  # @manual

        m = vit_b_16(weights="IMAGENET1K_V1")
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        m = torch.export.export(m, copy.deepcopy(example_inputs), strict=True).module()
        m(*example_inputs)
        m = export.export(m, copy.deepcopy(example_inputs))
        ops = _get_ops_list(m.graph_module)
        non_core_aten_op_found = False
        for op in ops:
            if "scaled_dot_product" in str(op):
                non_core_aten_op_found = True
        self.assertFalse(non_core_aten_op_found)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
