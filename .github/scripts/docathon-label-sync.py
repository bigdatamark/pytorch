# [MODULE DOC] Script that copies labels from the issue a pull request mentions onto that PR,
# but only when the issue carries the docathon label. Takes the PR number as the first command
# line argument and uses GITHUB_TOKEN for authentication via PyGithub. No import-time work
# besides imports; main runs when executed as a script.
import os
import re
import sys

from github import Github


def main() -> None:
    # [FUNCTION DOC] Reads the PR number from the command line, fetches the PR body and finds
    # the first issue reference in it (a hash followed by 1 to 6 digits). Returns silently if
    # the body is empty, has no reference, or the issue lacks the docathon label. Otherwise
    # adds to the PR every issue label not already on it, except the actionable label, and
    # prints a status line. Side effect: modifies PR labels on GitHub.
    # KNOWN ISSUE: the docathon label name is hardcoded to a specific year and must be updated.
    token = os.environ.get("GITHUB_TOKEN")

    repo_owner = "pytorch"
    repo_name = "pytorch"
    pull_request_number = int(sys.argv[1])

    g = Github(token)
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    pull_request = repo.get_pull(pull_request_number)
    pull_request_body = pull_request.body
    # PR without description
    if pull_request_body is None:
        return

    # get issue number from the PR body
    if not re.search(r"#\d{1,6}", pull_request_body):
        print("The pull request does not mention an issue.")
        return
    issue_number = int(re.findall(r"#(\d{1,6})", pull_request_body)[0])
    issue = repo.get_issue(issue_number)
    issue_labels = issue.labels
    docathon_label_present = any(
        label.name == "docathon-2026" for label in issue_labels
    )

    # if the issue has a docathon label, add all labels from the issue to the PR.
    if not docathon_label_present:
        print("The 'docathon-2026' label is not present in the issue.")
        return
    pull_request_labels = pull_request.get_labels()
    pull_request_label_names = [label.name for label in pull_request_labels]
    issue_label_names = [label.name for label in issue_labels]
    labels_to_add = [
        label
        for label in issue_label_names
        if label not in pull_request_label_names and label != "actionable"
    ]
    if not labels_to_add:
        print("The pull request already has the same labels.")
        return
    pull_request.add_to_labels(*labels_to_add)
    print("Labels added to the pull request!")


if __name__ == "__main__":
    main()
