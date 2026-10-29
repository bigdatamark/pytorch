# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Minimal template for a multi-process distributed test using
# MultiProcContinuousTest; the sample tests only print the rank and world size. Runs via
# run_tests when executed as a script.

from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import run_tests


class TestTemplate(MultiProcContinuousTest):
    # [CLASS DOC] Skeleton test case on MultiProcContinuousTest, where each test method runs on
    # every rank of one persistent process group; exposes rank and world_size from the base.
    def testABC(self):
        # [FUNCTION DOC] Prints a line with this rank, the world size and the label ABC. No
        # assertions.
        print(f"rank {self.rank} of {self.world_size} testing ABC")

    def testDEF(self):
        # [FUNCTION DOC] Prints a line with this rank, the world size and the label DEF. No
        # assertions.
        print(f"rank {self.rank} of {self.world_size} testing DEF")


if __name__ == "__main__":
    run_tests()
