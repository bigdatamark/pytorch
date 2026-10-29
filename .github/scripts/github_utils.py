# [MODULE DOC] Thin GitHub REST and GraphQL client built on urllib, shared by CI scripts. It
# provides generic fetch helpers (with token injection for the API host and rate-limit
# reporting), JSON list and dict fetchers, a GraphQL runner, and wrappers to post PR and commit
# comments, close, merge and update PRs, delete comments, find a merge base and query issues by
# label. Also defines the GitHubComment record. The GITHUB_TOKEN environment variable is read at
# call time. No import-time side effects.
"""GitHub Utilities"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from collections.abc import Callable


GITHUB_API_URL = "https://api.github.com"


@dataclass
class GitHubComment:
    # [CLASS DOC] Plain data record for one GitHub comment: body text, creation time string,
    # author login and url, author association, optional editor login, database id and url.
    # Holds data only; no behavior.
    body_text: str
    created_at: str
    author_login: str
    author_url: str | None
    author_association: str
    editor_login: str | None
    database_id: int
    url: str


def gh_fetch_url_and_headers(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None | str = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> tuple[Any, Any]:
    # [FUNCTION DOC] Performs an HTTP request (method optional, body from data: a string is sent
    # as is, a mapping as JSON) and returns the pair of response headers and reader applied to
    # the response. If GITHUB_TOKEN is set and the url is on the GitHub API host, adds an
    # authorization header (mutating the given headers mapping). On an HTTP error it prints a
    # rate-limit report when the limit is exhausted, else a short error, and re-raises.
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith(f"{GITHUB_API_URL}/"):
        headers["Authorization"] = f"token {token}"

    data_ = None
    if data is not None:
        data_ = data.encode() if isinstance(data, str) else json.dumps(data).encode()

    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return conn.headers, reader(conn)
    except HTTPError as err:
        if (
            err.code == 403
            and all(
                key in err.headers
                for key in ["X-RateLimit-Limit", "X-RateLimit-Remaining"]
            )
            and int(err.headers["X-RateLimit-Remaining"]) == 0
        ):
            print(
                f"""{url}
                Rate limit exceeded:
                Used: {err.headers["X-RateLimit-Used"]}
                Limit: {err.headers["X-RateLimit-Limit"]}
                Remaining: {err.headers["X-RateLimit-Remaining"]}
                Resets at: {err.headers["x-RateLimit-Reset"]}"""
            )
        else:
            print(f"Error fetching {url} {err}")
        raise


def gh_fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None | str = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = json.load,
) -> Any:
    # [FUNCTION DOC] Like the headers-returning fetch but returns only the reader result, which
    # defaults to parsed JSON. Errors propagate.
    return gh_fetch_url_and_headers(
        url, headers=headers, data=data, reader=reader, method=method
    )[1]


def gh_fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Fetches JSON from url with the GitHub v3 accept header, appending params as
    # a URL-encoded query string, with optional JSON body data and HTTP method. Returns the
    # parsed value, typed as a list of mappings.
    # KNOWN ISSUE: the declared list type is not enforced; endpoints returning an object give a
    # mapping at runtime.
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    return cast(
        list[dict[str, Any]],
        gh_fetch_url(url, headers=headers, data=data, reader=json.load, method=method),
    )


def _gh_fetch_json_any(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    # [FUNCTION DOC] HELPER: Shared implementation for the list and dict JSON fetchers: adds the
    # accept header and encoded query parameters and returns the untyped parsed JSON. A body
    # (data) makes the request a POST by default of the underlying library.
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    return gh_fetch_url(url, headers=headers, data=data, reader=json.load)


def gh_fetch_json_list(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Fetches JSON from url (with optional query params and body) and returns it
    # typed as a list of mappings; no runtime check of the shape.
    return cast(list[dict[str, Any]], _gh_fetch_json_any(url, params, data))


def gh_fetch_json_dict(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] Fetches JSON from url (with optional query params and body) and returns it
    # typed as a mapping; no runtime check of the shape.
    return cast(dict[str, Any], _gh_fetch_json_any(url, params, data))


def gh_graphql(query: str, **kwargs: Any) -> dict[str, Any]:
    # [FUNCTION DOC] Sends the GraphQL query to the GitHub GraphQL endpoint with the keyword
    # arguments as query variables and returns the parsed response. Raises RuntimeError
    # containing the query, variables and errors if the response has an errors entry.
    rc = gh_fetch_url(
        "https://api.github.com/graphql",  # @lint-ignore
        data={"query": query, "variables": kwargs},
        reader=json.load,
    )
    if "errors" in rc:
        raise RuntimeError(
            f"GraphQL query {query}, args {kwargs} failed: {rc['errors']}"
        )
    return cast(dict[str, Any], rc)


def _gh_post_comment(
    url: str, comment: str, dry_run: bool = False
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Posts a comment body to a comments url. In dry run mode prints the
    # comment and returns an empty list without any request. Otherwise returns the API response
    # (typed as a list although the API returns the created comment object).
    if dry_run:
        print(comment)
        return []
    return gh_fetch_json_list(url, data={"body": comment})


def gh_post_pr_comment(
    org: str, repo: str, pr_num: int, comment: str, dry_run: bool = False
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Posts a comment on the pull request (issue comments endpoint) of org/repo;
    # dry run prints instead of posting. Returns the API response or an empty list.
    return _gh_post_comment(
        f"{GITHUB_API_URL}/repos/{org}/{repo}/issues/{pr_num}/comments",
        comment,
        dry_run,
    )


def gh_post_commit_comment(
    org: str, repo: str, sha: str, comment: str, dry_run: bool = False
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Posts a comment on the commit with the given sha in org/repo; dry run
    # prints instead of posting. Returns the API response or an empty list.
    return _gh_post_comment(
        f"{GITHUB_API_URL}/repos/{org}/{repo}/commits/{sha}/comments",
        comment,
        dry_run,
    )


def gh_close_pr(org: str, repo: str, pr_num: int, dry_run: bool = False) -> None:
    # [FUNCTION DOC] Closes the pull request by patching its state to closed; in dry run mode
    # only prints a message. Errors from the API propagate.
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/pulls/{pr_num}"
    if dry_run:
        print(f"Dry run closing PR {pr_num}")
    else:
        gh_fetch_url(url, method="PATCH", data={"state": "closed"})


def gh_merge_pr(
    org: str,
    repo: str,
    pr_num: int,
    *,
    merge_method: str = "squash",
    commit_title: str | None = None,
    commit_message: str | None = None,
    sha: str | None = None,
    dry_run: bool = False,
) -> str:
    # [FUNCTION DOC] Merges the pull request through the merge endpoint with the given method
    # (default squash) and optional commit title, message and expected head sha. Dry run prints
    # the request body and returns an empty string. Otherwise returns the merge commit sha from
    # the response; API errors (for example not mergeable) propagate.
    """Merge a PR via GitHub's merge API and return the resulting merge commit sha."""
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/pulls/{pr_num}/merge"
    data: dict[str, Any] = {"merge_method": merge_method}
    if commit_title is not None:
        data["commit_title"] = commit_title
    if commit_message is not None:
        data["commit_message"] = commit_message
    if sha is not None:
        data["sha"] = sha
    if dry_run:
        print(f"[dry_run] Merging PR {pr_num} via GitHub API with {data}")
        return ""
    headers = {"Accept": "application/vnd.github.v3+json"}
    resp = gh_fetch_url(url, headers=headers, data=data, method="PUT", reader=json.load)
    return cast(str, resp["sha"])


def gh_delete_comment(org: str, repo: str, comment_id: int) -> None:
    # [FUNCTION DOC] Deletes the issue or PR comment with the given id via the API. Errors
    # propagate; there is no dry run.
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/issues/comments/{comment_id}"
    gh_fetch_url(url, method="DELETE", reader=lambda x: x.read())


def gh_fetch_merge_base(org: str, repo: str, base: str, head: str) -> str:
    # [FUNCTION DOC] Uses the compare API to get the merge base commit sha of base and head.
    # Returns the sha, or an empty string (after a warning) on an empty response or any error.
    merge_base = ""
    # Get the merge base using the GitHub REST API. This is the same as using
    # git merge-base without the need to have git. The API doc can be found at
    # https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#compare-two-commits
    try:
        json_data = gh_fetch_url(
            f"{GITHUB_API_URL}/repos/{org}/{repo}/compare/{base}...{head}",
            headers={"Accept": "application/vnd.github.v3+json"},
            reader=json.load,
        )
        if json_data:
            merge_base = json_data.get("merge_base_commit", {}).get("sha", "")
        else:
            warnings.warn(
                f"Failed to get merge base for {base}...{head}: Empty response"
            )
    except Exception as error:
        warnings.warn(f"Failed to get merge base for {base}...{head}: {error}")

    return merge_base


def gh_update_pr_state(org: str, repo: str, pr_num: int, state: str = "open") -> None:
    # [FUNCTION DOC] Sets the pull request state (default open) by patching it. A 422 response
    # when reopening (head branch deleted) only warns; any other HTTP error is re-raised.
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/pulls/{pr_num}"
    try:
        gh_fetch_url(url, method="PATCH", data={"state": state})
    except HTTPError as err:
        # When trying to open the pull request, error 422 means that the branch
        # has been deleted and the API couldn't re-open it
        if err.code == 422 and state == "open":
            warnings.warn(
                f"Failed to open {pr_num} because its head branch has been deleted: {err}"
            )
        else:
            raise


def gh_query_issues_by_labels(
    org: str, repo: str, labels: list[str], state: str = "open"
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Lists issues of org/repo that carry all the given labels (sent as one
    # comma-joined query value) in the given state (default open). Returns the first page of
    # results only (no pagination is followed).
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/issues"
    return gh_fetch_json(
        url, method="GET", params={"labels": ",".join(labels), "state": state}
    )
