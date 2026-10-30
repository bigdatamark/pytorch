# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the FX type promotion pass of the torch.onnx exporter: checks the
# generated elementwise type promotion rule set is current (ignoring pow and in-place pow), that
# the promotion table can be built, that overload lookup picks a schema-compatible overload, and
# that a where model with mixed scalars exports with the dynamo exporter.

import torch
from torch.onnx._internal.fx.passes import type_promotion
from torch.testing._internal import common_utils


# The following ops are ignored because we do not need these rules enabled for ONNX
IGNORED_OPS = {
    "pow",
    "pow_",
}


class TestGeneratedTypePromotionRuleSet(common_utils.TestCase):
    # [CLASS DOC] Tests that the checked-in generated type promotion rule set matches the one
    # generated from torch refs.
    def test_generated_rule_set_is_up_to_date(self):
        # [FUNCTION DOC] Regenerates the rule set from torch refs, drops rules for ops listed in
        # IGNORED_OPS, and asserts equality with the checked-in generated set. The failure hint
        # says to update the list in the type_promotion module.
        generated_set = type_promotion._GENERATED_ATEN_TYPE_PROMOTION_RULE_SET
        latest_set = type_promotion.ElementwiseTypePromotionRuleSetGenerator.generate_from_torch_refs()
        latest_set = {rule for rule in latest_set if rule.op_name not in IGNORED_OPS}

        # Please update the list in torch/onnx/_internal/fx/passes/type_promotion.py following the instruction
        # if this test fails
        self.assertEqual(generated_set, latest_set)

    def test_initialize_type_promotion_table_succeeds(self):
        # [FUNCTION DOC] Smoke test: constructing the type promotion table must not raise.
        type_promotion.TypePromotionTable()


class TestFindCompatibleOpOverload(common_utils.TestCase):
    # [CLASS DOC] Tests for finding a compatible op overload for given call arguments.
    def test_selects_schema_compatible_overload(self):
        # [FUNCTION DOC] Calls the overload finder for the where op with a boolean tensor, a
        # zero-dim tensor, and a float scalar, and asserts that the ScalarOther overload is
        # chosen.
        condition = torch.tensor([True, False])
        args = (condition, torch.tensor(0.0), -1000.0)

        overload = type_promotion.find_compatible_op_overload(
            torch.ops.aten.where, args, {}
        )

        self.assertEqual(overload, torch.ops.aten.where.ScalarOther)


class TestTypePromotionONNXExport(common_utils.TestCase):
    # [CLASS DOC] End-to-end export tests involving type promotion.
    def test_where_with_bool_tensor_and_mixed_scalars_exports(self):
        # [FUNCTION DOC] Exports a where model with a boolean condition and an int and a float
        # scalar using the dynamo-based exporter and asserts a non-None program is returned
        # (checks export does not raise).
        class WhereModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning where of the condition with scalar branches 0
            # and -1000.0.
            def forward(self, condition):
                # [FUNCTION DOC] HELPER: Returns the where result for the condition with scalar
                # branches 0 and -1000.0.
                return torch.where(condition, 0, -1000.0)

        model = WhereModel().eval()
        condition = torch.randn(1, 1, 4, 4) > 0

        onnx_program = torch.onnx.export(
            model,
            (condition,),
            dynamo=True,
            verbose=False,
        )

        self.assertIsNotNone(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()
