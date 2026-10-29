# [MODULE DOC] Heuristic rating tests whose names appear in the commit messages, the PR
# body or the bodies of issues and PRs linked from them. Uses the GitHub API. No
# import-time side effects.

from __future__ import annotations

import re
from typing import Any

from tools.testing.target_determination.heuristics.interface import (
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import (
    get_git_commit_info,
    get_issue_or_pr_body,
    get_pr_number,
)
from tools.testing.test_run import TestRun


# This heuristic searches the PR body and commit titles, as well as issues/PRs
# mentioned in the PR body/commit title for test names (search depth of 1) and
# gives the test a rating of 1.  For example, if I mention "test_foo" in the PR
# body, test_foo will be rated 1.  If I mention #123 in the PR body, and #123
# mentions "test_foo", test_foo will be rated 1.
class MentionedInPR(HeuristicInterface):
    # [CLASS DOC] Heuristic that gives score 1 to tests mentioned (by plain substring match) in
    # the text gathered from the PR. Stateless apart from trial_mode.
    # KNOWN ISSUE: substring matching means a short test name also matches longer names that
    # contain it.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Forwards keyword arguments (e.g. trial_mode) to the base class.
        super().__init__(**kwargs)

    def _search_for_linked_issues(self, s: str) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the issue or PR numbers (as strings) referenced in the
        # text, by hash-number syntax or by pytorch repository URLs.
        return re.findall(r"#(\d+)", s) + re.findall(r"/pytorch/pytorch/.*/(\d+)", s)

    def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
        # [FUNCTION DOC] Collects commit messages and the PR body (errors are printed and treated
        # as empty text), fetches the bodies of linked issues (failures ignored), and gives score 1
        # to every test name found in any of that text. Returns a TestPrioritizations.
        # Side effects: git and GitHub API calls, prints on error.
        try:
            commit_messages = get_git_commit_info()
        except Exception as e:
            print(f"Can't get commit info due to {e}")
            commit_messages = ""
        try:
            pr_number = get_pr_number()
            if pr_number is not None:
                pr_body = get_issue_or_pr_body(pr_number)
            else:
                pr_body = ""
        except Exception as e:
            print(f"Can't get PR body due to {e}")
            pr_body = ""

        # Search for linked issues or PRs
        linked_issue_bodies: list[str] = []
        for issue in self._search_for_linked_issues(
            commit_messages
        ) + self._search_for_linked_issues(pr_body):
            try:
                linked_issue_bodies.append(get_issue_or_pr_body(int(issue)))
            except Exception:
                pass

        mentioned = []
        for test in tests:
            if (
                test in commit_messages
                or test in pr_body
                or any(test in body for body in linked_issue_bodies)
            ):
                mentioned.append(test)

        return TestPrioritizations(tests, {TestRun(test): 1 for test in mentioned})
