# [MODULE DOC]
# Small helpers for calling the GitHub REST API with the standard library: a generic URL
# fetcher that adds token authentication and reports rate-limit details, JSON fetch wrappers
# and a commit lookup. No import-time side effects; network access happens only when the
# functions are called. The GITHUB_TOKEN environment variable supplies credentials.
"""GitHub Utilities"""

from __future__ import annotations

import json
import os
from typing import Any, cast, TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from collections.abc import Callable


def gh_fetch_url_and_headers(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> tuple[Any, Any]:
    # [FUNCTION DOC] Performs an HTTP request and returns a pair of the response headers and
    # the result of applying reader to the open response. A JSON-encoded body is sent when
    # data is given. If GITHUB_TOKEN is set and the URL targets the GitHub API host, an
    # authorization header is added (mutating the caller's headers mapping, if one was passed).
    # On an HTTP error, prints rate-limit details when the 403 response carries rate-limit
    # headers, then re-raises the error. Network I/O.
    # KNOWN ISSUE: the rate-limit message reads the Remaining and Reset headers without
    # checking they exist, so a response lacking them raises a KeyError-style failure instead.
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"token {token}"
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return conn.headers, reader(conn)
    except HTTPError as err:
        if err.code == 403 and all(
            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]
        ):
            print(
                f"""Rate limit exceeded:
                Used: {err.headers["X-RateLimit-Used"]}
                Limit: {err.headers["X-RateLimit-Limit"]}
                Remaining: {err.headers["X-RateLimit-Remaining"]}
                Resets at: {err.headers["x-RateLimit-Reset"]}"""
            )
        raise


def gh_fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    method: str | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Any:
    # [FUNCTION DOC] Fetches a URL and returns only the parsed JSON body (headers dropped).
    # Arguments are forwarded to gh_fetch_url_and_headers.
    # KNOWN ISSUE: the reader argument is accepted but ignored; JSON parsing is always used.
    return gh_fetch_url_and_headers(
        url, headers=headers, data=data, reader=json.load, method=method
    )[1]


def _gh_fetch_json_any(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    # [FUNCTION DOC] HELPER: Fetches JSON from a GitHub API URL with the v3 accept header.
    # Non-empty query params are appended to the URL as a query string with percent-encoded
    # values; data, if given, is sent as a JSON body. Returns the parsed JSON of any shape.
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    return gh_fetch_url(url, headers=headers, data=data, reader=json.load)


def gh_fetch_json_dict(
    url: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] Fetches JSON from the URL and returns it typed as a mapping. The cast
    # is not checked at runtime, so a list response would be returned as is.
    return cast(dict[str, Any], _gh_fetch_json_any(url, params, data))


def gh_fetch_commit(org: str, repo: str, sha: str) -> dict[str, Any]:
    # [FUNCTION DOC] Returns the GitHub API JSON mapping describing commit sha in the
    # repository org/repo. Raises an HTTP error if the request fails.
    return gh_fetch_json_dict(
        f"https://api.github.com/repos/{org}/{repo}/commits/{sha}"
    )
