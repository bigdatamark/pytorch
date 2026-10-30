# Owner(s): ["module: onnx"]
# [MODULE DOC] Unit tests for the TorchScript exporter symbolic helper check on training mode
# consistency. Tests set the global export training mode and verify whether a warning is emitted.
# Import-time effect: the parametrized tests are instantiated at module level.
"""Unit tests on `torch.onnx.symbolic_helper`."""

import torch
from torch.onnx import symbolic_helper
from torch.onnx._internal.torchscript_exporter._globals import GLOBALS
from torch.testing._internal import common_utils


class TestHelperFunctions(common_utils.TestCase):
    # [CLASS DOC] Tests for the training-mode check helper. The setup saves the global export
    # training mode and the teardown restores it, so each test may freely override it.
    def setUp(self):
        # [FUNCTION DOC] Saves the current global training mode on the instance
        # for later restoration.
        super().setUp()
        self._initial_training_mode = GLOBALS.training_mode

    def tearDown(self):
        # [FUNCTION DOC] Restores the global training mode saved by setUp; does not call
        # the base teardown.
        GLOBALS.training_mode = self._initial_training_mode

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.PRESERVE], name="export_mode_is_preserve"
            ),
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.EVAL],
                name="modes_match_op_train_mode_0_export_mode_eval",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.TRAINING],
                name="modes_match_op_train_mode_1_export_mode_training",
            ),
        ],
    )
    def test_check_training_mode_does_not_warn_when(
        self, op_train_mode: int, export_mode: torch.onnx.TrainingMode
    ):
        # [FUNCTION DOC] Parametrized over three (operator train flag, export mode)
        # combinations: preserve mode with a train operator, eval mode with a non-train
        # operator, training mode with a train operator. Sets the global mode and asserts that
        # the check raises no warning.
        GLOBALS.training_mode = export_mode
        self.assertNotWarn(
            lambda: symbolic_helper.check_training_mode(op_train_mode, "testop")
        )

    @common_utils.parametrize(
        "op_train_mode,export_mode",
        [
            common_utils.subtest(
                [0, torch.onnx.TrainingMode.TRAINING],
                name="modes_do_not_match_op_train_mode_0_export_mode_training",
            ),
            common_utils.subtest(
                [1, torch.onnx.TrainingMode.EVAL],
                name="modes_do_not_match_op_train_mode_1_export_mode_eval",
            ),
        ],
    )
    def test_check_training_mode_warns_when(
        self,
        op_train_mode: int,
        export_mode: torch.onnx.TrainingMode,
    ):
        # [FUNCTION DOC] Parametrized over two mismatching combinations (non-train operator in
        # training export mode, train operator in eval export mode). Asserts the check emits a
        # user warning whose message starts with the export mode text.
        with self.assertWarnsRegex(
            UserWarning, f"ONNX export mode is set to {export_mode}"
        ):
            GLOBALS.training_mode = export_mode
            symbolic_helper.check_training_mode(op_train_mode, "testop")


common_utils.instantiate_parametrized_tests(TestHelperFunctions)


if __name__ == "__main__":
    common_utils.run_tests()
