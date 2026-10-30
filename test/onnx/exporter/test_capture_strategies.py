# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the graph capture strategies of the ONNX exporter: checks that the
# strict, non-strict and draft-export strategies capture small models into exported programs.
# No import-time side effects other than imports; the file runs its tests when executed
# directly.
"""Unit tests for the _capture_strategies module."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _capture_strategies
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class ExportStrategiesTest(common_utils.TestCase):
    # [CLASS DOC] Parametrized test case covering the torch.export based capture strategy
    # classes (strict, non-strict, draft export). Holds no state of its own.
    @common_utils.parametrize(
        "strategy_cls",
        [
            _capture_strategies.TorchExportStrictStrategy,
            _capture_strategies.TorchExportNonStrictStrategy,
            _capture_strategies.TorchExportDraftExportStrategy,
        ],
        name_fn=lambda strategy_cls: strategy_cls.__name__,
    )
    def test_jit_isinstance(self, strategy_cls):
        # [FUNCTION DOC] Captures a model that branches on the jit isinstance check using the
        # strategy class given by the parameter, then checks the exported program is present
        # and that running its module on two scalar tensors matches the eager model output.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the cosine of the first input if it
            # is a tensor according to the jit isinstance check, otherwise the sine of the second
            # input.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the cosine of a when a is a tensor, else the sine
                # of b.
                if torch.jit.isinstance(a, torch.Tensor):
                    return a.cos()
                return b.sin()

        model = Model()
        a = torch.tensor(0.0)
        b = torch.tensor(1.0)

        result = strategy_cls()(model, (a, b), kwargs=None, dynamic_shapes=None)
        ep = result.exported_program
        if ep is None:
            raise AssertionError("ep is None")
        torch.testing.assert_close(ep.module()(a, b), model(a, b))

    def test_draft_export_on_data_dependent_model(self):
        # [FUNCTION DOC] Captures a model with a data dependent branch using the draft export
        # strategy while log capture is active on the torch.export logger. Asserts that a warning
        # reporting one issue found during export is logged and that the exported program still
        # reproduces the eager output.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a data dependent branch on the sum of the first
            # input; returns the cosine of a if the sum is positive, otherwise the sine of b.
            # Draft export specializes the branch.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the cosine of a if its sum is positive, else the
                # sine of b.
                if a.sum() > 0:
                    return a.cos()
                # The branch is expected to be specialized and a warning is logged
                return b.sin()

        model = Model()
        a = torch.tensor(0.0)
        b = torch.tensor(1.0)

        strategy = _capture_strategies.TorchExportDraftExportStrategy()
        with self.assertLogs("torch.export", level="WARNING") as cm:
            result = strategy(model, (a, b), kwargs=None, dynamic_shapes=None)
            expected_warning = "1 issue(s) found during export, and it was not able to soundly produce a graph."
            self.assertIn(expected_warning, str(cm.output))
        ep = result.exported_program
        if ep is None:
            raise AssertionError("ep is None")
        torch.testing.assert_close(ep.module()(a, b), model(a, b))


if __name__ == "__main__":
    common_utils.run_tests()
