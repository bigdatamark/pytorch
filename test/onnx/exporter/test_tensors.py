# Owner(s): ["module: onnx"]
# [MODULE DOC] Unit tests for the ONNX exporter symbolic tensor type. Defines one test class
# with a single hashability test; runs through the common_utils test runner when executed.
"""Unit tests for the _tensors module."""

from __future__ import annotations

import onnxscript

from torch.onnx._internal.exporter import _tensors
from torch.testing._internal import common_utils


class SymbolicTensorTest(common_utils.TestCase):
# [CLASS DOC] Test case for the SymbolicTensor class of the exporter _tensors module.
    def test_it_is_hashable(self):
    # [FUNCTION DOC] Checks that a SymbolicTensor built from an Opset can be hashed: the hash is
    # stable across two calls and the tensor is found in a set containing it.
        tensor = _tensors.SymbolicTensor(
            opset=onnxscript.values.Opset(domain="test", version=1)
        )
        self.assertEqual(hash(tensor), hash(tensor))
        self.assertIn(tensor, {tensor})


if __name__ == "__main__":
    common_utils.run_tests()
