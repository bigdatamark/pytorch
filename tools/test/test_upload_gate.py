# [MODULE DOC] Tests for should_upload_full_test_run, the gate deciding by branch or tag name
# and repository whether a full test run's stats are uploaded.
import unittest

from tools.stats.upload_test_stats import should_upload_full_test_run


class TestUploadGate(unittest.TestCase):
    # [CLASS DOC] Test case for the upload gate; stateless.
    def test_main_branch_on_pytorch_repo(self) -> None:
        # [FUNCTION DOC] Checks the main branch of the pytorch repository is allowed.
        self.assertTrue(should_upload_full_test_run("main", "pytorch/pytorch"))

    def test_trunk_tag_valid_sha_on_pytorch_repo(self) -> None:
        # [FUNCTION DOC] Checks a trunk tag with a 40 character hex hash is allowed.
        sha = "a" * 40
        self.assertTrue(should_upload_full_test_run(f"trunk/{sha}", "pytorch/pytorch"))

    def test_trunk_tag_invalid_sha_on_pytorch_repo(self) -> None:
        # [FUNCTION DOC] Checks a trunk tag whose hash is not 40 hex characters is rejected.
        # Not 40 hex chars
        self.assertFalse(should_upload_full_test_run("trunk/12345", "pytorch/pytorch"))

    def test_non_main_branch_on_pytorch_repo(self) -> None:
        # [FUNCTION DOC] Checks a feature branch of the pytorch repository is rejected.
        self.assertFalse(
            should_upload_full_test_run("feature-branch", "pytorch/pytorch")
        )

    def test_main_branch_on_fork_repo(self) -> None:
        # [FUNCTION DOC] Checks the main branch of a fork repository is rejected.
        self.assertFalse(should_upload_full_test_run("main", "someone/fork"))


if __name__ == "__main__":
    unittest.main()
