# Owner(s): ["oncall: distributed checkpointing"]

# [MODULE DOC] Tests for the experimental checkpoint types module: the RankInfo
# container and the STATE_DICT type alias. Pure unit tests with no I/O.

from torch.distributed.checkpoint._experimental.types import RankInfo, STATE_DICT
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRankInfo(TestCase):
    # [CLASS DOC] Tests construction of RankInfo and use of the STATE_DICT alias.
    def test_rank_info_initialization(self):
        # [FUNCTION DOC] Builds RankInfo with rank 0 and world size 4 and checks both
        # attributes are stored unchanged.
        """Test that RankInfo initializes correctly with all parameters."""
        # Create a RankInfo instance with all parameters
        rank_info = RankInfo(
            global_rank=0,
            global_world_size=4,
        )

        # Verify that all attributes are set correctly
        self.assertEqual(rank_info.global_rank, 0)
        self.assertEqual(rank_info.global_world_size, 4)

    def test_rank_info_default_initialization(self):
        # [FUNCTION DOC] Builds RankInfo with rank 0 and world size 1 and checks the stored
        # attributes.
        # KNOWN ISSUE: the name and docstring suggest default values, but both fields are
        # passed explicitly, so no defaults are exercised.
        """Test that RankInfo initializes correctly with default parameters."""
        # Create a RankInfo instance with only required parameters
        rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
        )

        # Verify that all attributes are set correctly
        self.assertEqual(rank_info.global_rank, 0)
        self.assertEqual(rank_info.global_world_size, 1)

    def test_state_dict_type_alias(self):
        # [FUNCTION DOC] Assigns a nested mapping to a variable annotated with STATE_DICT
        # and checks equality; only verifies the alias is importable and usable at runtime.
        """Test that STATE_DICT type alias works correctly."""
        # Create a state dictionary
        state_dict = {"model": {"weight": [1, 2, 3]}, "optimizer": {"lr": 0.01}}

        # Verify that it can be assigned to a variable of type STATE_DICT
        state_dict_var: STATE_DICT = state_dict
        self.assertEqual(state_dict_var, state_dict)


if __name__ == "__main__":
    run_tests()
