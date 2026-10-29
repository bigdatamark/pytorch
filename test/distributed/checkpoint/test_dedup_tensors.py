# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Unit test for torch.distributed.checkpoint._dedup_tensors.dedup_tensors:
# verifies that a shard item duplicated across rank save plans is retained only by the
# lowest rank. Defines a plan-building helper and one TestCase; no import-time side effects.

import dataclasses

import torch
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.planner import SavePlan, WriteItemType
from torch.distributed.checkpoint.planner_helpers import _create_write_item_for_tensor
from torch.testing._internal.common_utils import run_tests, TestCase


def create_plan(second_fqn: str) -> SavePlan:
    """
    Creates a SavePlan with two write items:

    1. A write item representing a shard of a tensor named "tensor_0".
    2. A write item representing another tensor identified by the provided second_fqn.

    Args:
        second_fqn (str): The fully qualified name for the second tensor.

    Returns:
        SavePlan: A plan that includes the two write items.
    """
    # [FUNCTION DOC] HELPER: Builds a save plan with two write items. The first is a random
    # 4-element tensor named tensor_0 whose item type is forced to SHARD (so every plan
    # contains the same duplicated shard); the second is a random 10-element tensor named
    # by second_fqn. Returns the SavePlan holding both items; no side effects.
    # the first write item is for a duplicated shard (that covers the whole tensor)
    write_item_1 = _create_write_item_for_tensor("tensor_0", torch.rand(4))
    write_item_1 = dataclasses.replace(write_item_1, type=WriteItemType.SHARD)

    # the second write item has different keys
    write_item_2 = _create_write_item_for_tensor(second_fqn, torch.rand(10))

    return SavePlan([write_item_1, write_item_2])


class TestDedupTensor(TestCase):
    # [CLASS DOC] Single-process test case for dedup_tensors; holds no state.
    """
    Test class for deduplication of tensor write items across different ranks.
    """

    def test_dedup_shards(self):
        # [FUNCTION DOC] Builds plans for two simulated ranks that share the duplicated
        # tensor_0 shard and each own a distinct tensor, runs dedup_tensors, and checks
        # that rank 0 keeps both items (tensor_0 and r0) while rank 1 keeps only r1.
        rank0 = create_plan("r0")
        rank1 = create_plan("r1")

        dedup_plans = dedup_tensors([rank0, rank1])

        self.assertEqual(2, len(dedup_plans[0].items))
        self.assertEqual(1, len(dedup_plans[1].items))

        self.assertIn("tensor_0", (item.index.fqn for item in dedup_plans[0].items))
        self.assertIn("r0", (item.index.fqn for item in dedup_plans[0].items))

        self.assertIn("r1", (item.index.fqn for item in dedup_plans[1].items))


if __name__ == "__main__":
    run_tests()
