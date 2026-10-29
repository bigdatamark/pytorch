# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for the functional assertion aten operators (async assert with message and
# symbolic range constraint) that take and return a dependency token.
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFuntionalAssertions(TestCase):
    # [CLASS DOC] Test case covering the functional variants of assertion operators.
    def test_functional_assert_async_msg(self) -> None:
        # [FUNCTION DOC] Checks that the functional async assert with message returns the
        # dependency token when the tensor is nonzero, and raises a RuntimeError containing the
        # message when the tensor is zero.
        dep_token = torch.ops.aten._make_dep_token()
        self.assertEqual(
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(1), "test msg", dep_token
            ),
            dep_token,
        )
        with self.assertRaisesRegex(RuntimeError, "test msg"):
            torch.ops.aten._functional_assert_async.msg(
                torch.tensor(0), "test msg", dep_token
            )

    def test_functional_sym_constrain_range(self) -> None:
        # [FUNCTION DOC] Checks that the functional symbolic range constraint operator, given an
        # in-range integer and a dep token, returns that same dep token.
        dep_token = torch.ops.aten._make_dep_token()
        self.assertEqual(
            torch.ops.aten._functional_sym_constrain_range(
                3, min=2, max=5, dep_token=dep_token
            ),
            dep_token,
        )


if __name__ == "__main__":
    run_tests()
