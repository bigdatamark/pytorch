# [MODULE DOC] Unit tests for tryrebase: rebase_onto, rebase_ghstack_onto and
# additional_rebase_failure_info. Git and GitHub access is mocked (GraphQL via the cached mocks
# from test_trymerge, GitRepo._run_git and rev_parse, and the comment poster), so the tests assert
# on the exact git commands issued and the comments posted. Defines helper constants for branches
# and a fork point SHA. No import-time side effects beyond imports.

from typing import Any
from unittest import main, mock, TestCase

from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from test_trymerge import mocked_gh_graphql
from trymerge import GitHubPR
from tryrebase import additional_rebase_failure_info, rebase_ghstack_onto, rebase_onto


def mocked_rev_parse(branch: str) -> str:
    # [FUNCTION DOC] HELPER: Stand-in for rev_parse that returns the branch name unchanged.
    return branch


MAIN_BRANCH = "refs/remotes/origin/main"
VIABLE_STRICT_BRANCH = "refs/remotes/origin/viable/strict"
# A full 40-char hex SHA, as `git merge-base` actually outputs; value is opaque
# to the code (passed through verbatim to the rebase call).
FORK_POINT = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def make_mocked_run_git(push_result: str = "") -> Any:
    # [FUNCTION DOC] HELPER: Builds a replacement for the git runner. Takes the text a push should
    # return (default empty) and returns a function that answers merge-base commands with
    # FORK_POINT and every other git command with that text.
    """_run_git stub: merge-base yields a fork point, everything else push_result."""

    def run_git(*args: str) -> str:
        # [FUNCTION DOC] HELPER: Fake git runner; returns FORK_POINT for
        # merge-base, push_result otherwise.
        if args and args[0] == "merge-base":
            return FORK_POINT
        return push_result

    return run_git


class TestRebase(TestCase):
    # [CLASS DOC] Tests for rebase_onto, rebase_ghstack_onto and the failure-info helper against
    # PR 31093, with all git and GitHub calls mocked.
    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_rebase(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        # [FUNCTION DOC] Rebases onto the main branch and checks the sequence of git calls (fetch,
        # merge-base, rebase with onto and the fork point, forced push to the fork) and the
        # success comment posted.
        "Tests rebase successfully"
        mocked_run_git.side_effect = make_mocked_run_git()
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, MAIN_BRANCH)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mock.call("rebase", "--onto", MAIN_BRANCH, FORK_POINT, "pull/31093/head"),
            mock.call(
                "push",
                "-f",
                "https://github.com/mingxiaoh/pytorch.git",
                "pull/31093/head:master",
            ),
        ]
        mocked_run_git.assert_has_calls(calls)
        self.assertIn(
            f"Successfully rebased `master` onto `{MAIN_BRANCH}`",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_rebase_to_stable(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        # [FUNCTION DOC] Same as test_rebase but targets the viable/strict branch, with the flag
        # that disables the extra hint set to false; checks git calls and the success comment.
        "Tests rebase to viable/strict successfully"
        mocked_run_git.side_effect = make_mocked_run_git()
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH, False)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mock.call(
                "rebase", "--onto", VIABLE_STRICT_BRANCH, FORK_POINT, "pull/31093/head"
            ),
            mock.call(
                "push",
                "-f",
                "https://github.com/mingxiaoh/pytorch.git",
                "pull/31093/head:master",
            ),
        ]
        mocked_run_git.assert_has_calls(calls)
        self.assertIn(
            f"Successfully rebased `master` onto `{VIABLE_STRICT_BRANCH}`",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_no_need_to_rebase(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        # [FUNCTION DOC] Checks that when the push reports nothing to update, the comment says the
        # PR was already up to date and does not suggest rebasing against main.
        "Tests branch already up to date"
        mocked_run_git.side_effect = make_mocked_run_git("Everything up-to-date")
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, MAIN_BRANCH)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"
        calls = [
            mock.call("fetch", "origin", "pull/31093/head:pull/31093/head"),
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mock.call("rebase", "--onto", MAIN_BRANCH, FORK_POINT, "pull/31093/head"),
            mock.call(
                "push",
                "-f",
                "https://github.com/mingxiaoh/pytorch.git",
                "pull/31093/head:master",
            ),
        ]
        mocked_run_git.assert_has_calls(calls)
        self.assertIn(
            "Tried to rebase and push PR #31093, but it was already up to date",
            mocked_post_comment.call_args[0][3],
        )
        self.assertNotIn(
            "Try rebasing against [main]",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git", return_value="Everything up-to-date")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_no_need_to_rebase_try_main(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        # [FUNCTION DOC] Checks that when the push reports up to date while targeting
        # viable/strict, the comment suggests trying a rebase against main.
        "Tests branch already up to date again viable/strict"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH)
        self.assertIn(
            "Tried to rebase and push PR #31093, but it was already up to date. Try rebasing against [main]",
            mocked_post_comment.call_args[0][3],
        )

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=lambda branch: "same sha")
    @mock.patch("tryrebase.gh_post_comment")
    def test_same_sha(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        # [FUNCTION DOC] Checks both rebase_onto and rebase_ghstack_onto raise an error mentioning
        # the same sha when the rebased head equals the target branch.
        "Tests rebase results in same sha"
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        with self.assertRaisesRegex(Exception, "same sha as the target branch"):
            rebase_onto(pr, repo, MAIN_BRANCH)
        with self.assertRaisesRegex(Exception, "same sha as the target branch"):
            rebase_ghstack_onto(pr, repo, MAIN_BRANCH)

    def test_additional_rebase_failure_info(self) -> None:
        # [FUNCTION DOC] Checks that a permission-denied push error produces extra failure info
        # that explains the likely cause.
        error = (
            "Command `git -C /Users/csl/zzzzzzzz/pytorch push --dry-run -f "
            "https://github.com/Lightning-Sandbox/pytorch.git pull/106089/head:fix/spaces` returned non-zero exit code 128\n"
            "```\n"
            "remote: Permission to Lightning-Sandbox/pytorch.git denied to clee2000.\n"
            "fatal: unable to access 'https://github.com/Lightning-Sandbox/pytorch.git/': The requested URL returned error: 403\n"
            "```"
        )
        additional_msg = additional_rebase_failure_info(Exception(error))
        self.assertTrue("This is likely because" in additional_msg)

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("gitutils.GitRepo._run_git")
    @mock.patch("gitutils.GitRepo.rev_parse", side_effect=mocked_rev_parse)
    @mock.patch("tryrebase.gh_post_comment")
    def test_rebase_does_not_replay_trunk_commits(
        self,
        mocked_post_comment: Any,
        mocked_rp: Any,
        mocked_run_git: Any,
        mocked_gql: Any,
    ) -> None:
        # [FUNCTION DOC] Regression test: checks the rebase uses the onto form anchored on the
        # merge-base fork point, and that the only rebase call has exactly those arguments, so
        # trunk commits are not replayed onto the PR.
        """#187374: rebase must anchor on the fork point (--onto) and never use
        the 2-arg form, which grafts trunk commits onto the PR."""
        mocked_run_git.side_effect = make_mocked_run_git()
        pr = GitHubPR("pytorch", "pytorch", 31093)
        repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
        rebase_onto(pr, repo, VIABLE_STRICT_BRANCH)
        base_ref = f"refs/remotes/origin/{pr.base_ref()}"

        self.assertIn(
            mock.call("merge-base", base_ref, "pull/31093/head"),
            mocked_run_git.call_args_list,
        )
        rebase_calls = [
            c for c in mocked_run_git.call_args_list if c.args and c.args[0] == "rebase"
        ]
        self.assertEqual(
            rebase_calls,
            [
                mock.call(
                    "rebase",
                    "--onto",
                    VIABLE_STRICT_BRANCH,
                    FORK_POINT,
                    "pull/31093/head",
                )
            ],
        )


if __name__ == "__main__":
    main()
