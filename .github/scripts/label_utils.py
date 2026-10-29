# [MODULE DOC]
# GitHub label helpers used by the merge and labeling bots: fetching all labels of a repo
# through the REST API (paginated, cached), adding and removing labels on a PR, finding release
# notes labels, checking that a PR carries a release notes or not-user-facing label, and
# recognizing the bot comment that complains about a missing label. Defines the constants
# BOT_AUTHORS, NOT_USER_FACING_LABEL and the error message text. No import time side effects
# besides imports.
"""GitHub Label Utilities."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from github_utils import gh_fetch_url_and_headers, GitHubComment


# TODO: this is a temp workaround to avoid circular dependencies,
#       and should be removed once GitHubPR is refactored out of trymerge script.
if TYPE_CHECKING:
    from trymerge import GitHubPR

BOT_AUTHORS = ["github-actions", "pytorchmergebot", "pytorch-bot"]

# Label that marks a PR as excluded from release notes.
NOT_USER_FACING_LABEL = "topic: not user facing"

LABEL_ERR_MSG_TITLE = "This PR needs a `release notes:` label"
LABEL_ERR_MSG = f"""# {LABEL_ERR_MSG_TITLE}
If your changes are user facing and intended to be a part of release notes, please use a label starting with `release notes:`.

If not, please add the `topic: not user facing` label.

To add a label, you can comment to pytorchbot, for example
`@pytorchbot label "topic: not user facing"`

For more information, see
https://github.com/pytorch/pytorch/wiki/PyTorch-AutoLabel-Bot#why-categorize-for-release-notes-and-how-does-it-work.
"""


def request_for_labels(url: str) -> tuple[Any, Any]:
    # [FUNCTION DOC] Fetches one url from the GitHub API asking for the v3 json media type.
    # Returns the pair (response headers, body text decoded as utf-8) from the shared fetch
    # helper. Network access.
    headers = {"Accept": "application/vnd.github.v3+json"}
    return gh_fetch_url_and_headers(
        url, headers=headers, reader=lambda x: x.read().decode("utf-8")
    )


def update_labels(labels: list[str], info: str) -> None:
    # [FUNCTION DOC] HELPER: Parses a json list of label objects and appends each label name to
    # the given list (mutated in place). Raises a json error on invalid input.
    labels_json = json.loads(info)
    labels.extend([x["name"] for x in labels_json])


def get_last_page_num_from_header(header: Any) -> int:
    # [FUNCTION DOC] Returns the number of the last page of a paginated listing, read from the
    # link entry of the response headers (text between the final page parameter and the closing
    # bracket before rel). Returns 1 if the link value is None.
    # KNOWN ISSUE: a missing link key raises KeyError, so the None check only works if the
    #     headers object returns None for absent keys; with a single page of labels GitHub
    #     omits the header.
    # Link info looks like: <https://api.github.com/repositories/65600975/labels?per_page=100&page=2>;
    # rel="next", <https://api.github.com/repositories/65600975/labels?per_page=100&page=3>; rel="last"
    link_info = header["link"]
    # Docs does not specify that it should be present for projects with just few labels
    # And https://github.com/malfet/deleteme/actions/runs/7334565243/job/19971396887 it's not the case  # @lint-ignore
    if link_info is None:
        return 1
    prefix = "&page="
    suffix = ">;"
    return int(
        link_info[link_info.rindex(prefix) + len(prefix) : link_info.rindex(suffix)]
    )


@lru_cache
def gh_get_labels(org: str, repo: str) -> list[str]:
    # [FUNCTION DOC] Returns the names of all labels of org/repo, reading pages of 100 from the
    # GitHub API and following the last page number from the first response header. Raises
    # AssertionError if the last page number is not positive. Results are cached per (org, repo)
    # for the process lifetime.
    prefix = f"https://api.github.com/repos/{org}/{repo}/labels?per_page=100"
    header, info = request_for_labels(prefix + "&page=1")
    labels: list[str] = []
    update_labels(labels, info)

    last_page = get_last_page_num_from_header(header)
    if last_page <= 0:
        raise AssertionError(
            f"Error reading header info to determine total number of pages of labels, got last_page={last_page}"
        )
    for page_number in range(2, last_page + 1):  # skip page 1
        _, info = request_for_labels(prefix + f"&page={page_number}")
        update_labels(labels, info)

    return labels


def gh_add_labels(
    org: str, repo: str, pr_num: int, labels: str | list[str], dry_run: bool
) -> None:
    # [FUNCTION DOC] Adds the label(s) to the PR via the issues labels endpoint. In dry run mode
    # only prints what would be done and makes no request. Network side effect otherwise.
    if dry_run:
        print(f"Dryrun: Adding labels {labels} to PR {pr_num}")
        return
    gh_fetch_url_and_headers(
        url=f"https://api.github.com/repos/{org}/{repo}/issues/{pr_num}/labels",
        data={"labels": labels},
    )


def gh_remove_label(
    org: str, repo: str, pr_num: int, label: str, dry_run: bool
) -> None:
    # [FUNCTION DOC] Removes the label from the PR via the issues labels endpoint (HTTP DELETE).
    # In dry run mode only prints and makes no request.
    if dry_run:
        print(f"Dryrun: Removing {label} from PR {pr_num}")
        return
    gh_fetch_url_and_headers(
        url=f"https://api.github.com/repos/{org}/{repo}/issues/{pr_num}/labels/{label}",
        method="DELETE",
    )


def get_release_notes_labels(org: str, repo: str) -> list[str]:
    # [FUNCTION DOC] Returns the names of the repo labels that start with release notes: (after
    # leading whitespace is stripped). Uses the cached label listing.
    return [
        label
        for label in gh_get_labels(org, repo)
        if label.lstrip().startswith("release notes:")
    ]


def has_required_labels(pr: GitHubPR) -> bool:
    # [FUNCTION DOC] Returns True if the PR has the not user facing label or any label
    # (whitespace stripped) that is a release notes label of the PR's org and project. Network
    # access on first use.
    pr_labels = pr.get_labels()
    # Check if PR is not user facing
    is_not_user_facing_pr = any(
        label.strip() == NOT_USER_FACING_LABEL for label in pr_labels
    )
    return is_not_user_facing_pr or any(
        label.strip() in get_release_notes_labels(pr.org, pr.project)
        for label in pr_labels
    )


def is_label_err_comment(comment: GitHubComment) -> bool:
    # [FUNCTION DOC] Returns True if the comment looks like the bot's missing release notes
    # label error: the plain text (markup removed) after leading spaces and hash signs starts
    # with the error title and the author login is one of BOT_AUTHORS.
    # comment.body_text returns text without markdown
    no_format_title = LABEL_ERR_MSG_TITLE.replace("`", "")
    return (
        comment.body_text.lstrip(" #").startswith(no_format_title)
        and comment.author_login in BOT_AUTHORS
    )
