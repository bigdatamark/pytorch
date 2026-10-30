# [MODULE DOC] Unit tests for the release notes CommitList. They build commit lists from the git
# history of the current repository between fixed revisions and compare against hard coded
# counts and hashes, so they need the full pytorch history, git and network access for commit
# data (a GitHub token through the common module).
# KNOWN ISSUE: the tests call a write_to_disk method that CommitList does not define (it has
#   write_result), so the read-write and update tests would fail with an attribute error.
import tempfile
import unittest

from commitlist import CommitList


class TestCommitList(unittest.TestCase):
    # [CLASS DOC] Tests for creating, saving, reloading and extending a commit list in a
    # temporary directory using fixed historic revisions of the repository.
    def test_create_new(self):
        # [FUNCTION DOC] Creates a list between two fixed revisions and checks the number of
        # commits, the first and last hashes and that the first and last titles start with the
        # expected text.
        with tempfile.TemporaryDirectory() as tempdir:
            commit_list_path = f"{tempdir}/commitlist.csv"
            commit_list = CommitList.create_new(
                commit_list_path, "v1.5.0", "6000dca5df"
            )
            self.assertEqual(len(commit_list.commits), 33)
            self.assertEqual(commit_list.commits[0].commit_hash, "7335f079abb")
            self.assertTrue(
                commit_list.commits[0].title.startswith("[pt][quant] qmul and qadd")
            )
            self.assertEqual(commit_list.commits[-1].commit_hash, "6000dca5df6")
            self.assertTrue(
                commit_list.commits[-1].title.startswith(
                    "[nomnigraph] Copy device option when customize "
                )
            )

    def test_read_write(self):
        # [FUNCTION DOC] Creates a list, saves it, reloads it, changes one category, saves
        # again, reloads and compares every commit with the modified list.
        # KNOWN ISSUE: relies on a missing write_to_disk method of CommitList, so it fails.
        with tempfile.TemporaryDirectory() as tempdir:
            commit_list_path = f"{tempdir}/commitlist.csv"
            initial = CommitList.create_new(commit_list_path, "v1.5.0", "7543e7e558")
            initial.write_to_disk()

            expected = CommitList.from_existing(commit_list_path)
            expected.commits[-2].category = "foobar"
            expected.write_to_disk()

            commit_list = CommitList.from_existing(commit_list_path)
            for commit, expected_commit in zip(commit_list.commits, expected.commits):
                self.assertEqual(commit, expected_commit)

    def test_update_to(self):
        # [FUNCTION DOC] Creates a list to a fixed revision, checks its size, saves it, reloads
        # it, extends it to a newer revision and checks that four commits were added and the
        # previous last commit is now fifth from the end.
        # KNOWN ISSUE: relies on a missing write_to_disk method of CommitList, so it fails.
        with tempfile.TemporaryDirectory() as tempdir:
            commit_list_path = f"{tempdir}/commitlist.csv"
            initial = CommitList.create_new(commit_list_path, "v1.5.0", "7543e7e558")
            initial.commits[-2].category = "foobar"
            self.assertEqual(len(initial.commits), 2143)
            initial.write_to_disk()

            commit_list = CommitList.from_existing(commit_list_path)
            commit_list.update_to("5702a28b26")
            self.assertEqual(len(commit_list.commits), 2143 + 4)
            self.assertEqual(commit_list.commits[-5], initial.commits[-1])


if __name__ == "__main__":
    unittest.main()
