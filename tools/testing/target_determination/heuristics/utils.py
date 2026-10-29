# [MODULE DOC] Shared helpers for heuristics: GitHub API access, git merge base and diff
# queries for the changed files of the current PR (or of the last commit on the default
# branch), PR number discovery from the environment, text fetching for issues and PRs,
# rating normalization and loading of correlation files. Several results are cached for the
# process lifetime. REPO_ROOT is the repository root. No import-time side effects.

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import cast, TYPE_CHECKING
from urllib.parse import quote
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]


def _github_api_json(path: str) -> object:
    # [FUNCTION DOC] HELPER: GETs a path under the GitHub repository API (repository from
    # GITHUB_REPOSITORY, default pytorch/pytorch) with an optional GITHUB_TOKEN header and
    # returns the decoded JSON. Raises URL errors on network failure.
    github_repository = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    github_token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    url = f"https://api.github.com/repos/{github_repository}/{path}"
    with urlopen(Request(url, headers=headers)) as conn:
        return json.loads(conn.read().decode())


def _git_merge_base(base: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the merge base commit hash of the given ref and HEAD.
    # Raises CalledProcessError if git fails.
    return subprocess.check_output(["git", "merge-base", base, "HEAD"]).decode().strip()


def _git_head() -> str:
    # [FUNCTION DOC] HELPER: Returns the hash of HEAD from git.
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _get_pr_info(pr_number: int) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Fetches the pull request metadata mapping from the GitHub API.
    return cast(dict[str, object], _github_api_json(f"pulls/{pr_number}"))


def _get_pr_merge_base(pr_info: dict[str, object]) -> str:
    # [FUNCTION DOC] HELPER: Finds the merge base of a PR. Tries the remote base branch and the
    # recorded base sha with git; if both fail, asks the GitHub compare API using HEAD. Falls
    # back to the base sha, then re-raises the last git error, else raises RuntimeError.
    base_info = cast(dict[str, object], pr_info["base"])
    base_ref = cast(str, base_info["ref"])
    base_sha = cast("str | None", base_info.get("sha"))
    last_error: Exception | None = None

    for base in (f"origin/{base_ref}", base_sha):
        if not base:
            continue
        try:
            return _git_merge_base(base)
        except subprocess.CalledProcessError as e:
            last_error = e

    head = _git_head()
    compare = cast(
        dict[str, object],
        _github_api_json(f"compare/{quote(base_ref, safe='')}...{head}"),
    )
    merge_base_commit = cast(dict[str, object], compare.get("merge_base_commit", {}))
    merge_base = cast("str | None", merge_base_commit.get("sha"))
    if merge_base:
        return merge_base
    if base_sha:
        return base_sha
    if last_error:
        raise last_error
    raise RuntimeError(f"Unable to determine merge base for PR base {base_ref}")


def _query_changed_files_from_github(pr_number: int) -> list[str]:
    # [FUNCTION DOC] HELPER: Lists the changed file names of a PR from the GitHub API, paging
    # 100 files at a time until a short page is returned.
    changed_files: list[str] = []
    page = 1
    while True:
        files = cast(
            list[dict[str, object]],
            _github_api_json(f"pulls/{pr_number}/files?per_page=100&page={page}"),
        )
        changed_files.extend(cast(str, file["filename"]) for file in files)
        if len(files) < 100:
            return changed_files
        page += 1


def python_test_file_to_test_name(tests: set[str]) -> set[str]:
    # [FUNCTION DOC] Converts changed paths into test names: normalizes backslashes, keeps
    # paths under test/ ending in the python extension, and strips that prefix and extension.
    prefix = "test/"
    valid_tests = {
        f
        for f in (test.replace("\\", "/") for test in tests)
        if f.startswith(prefix) and f.endswith(".py")
    }
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests


@cache
def get_pr_number() -> int | None:
    # [FUNCTION DOC] Returns the PR number from the PR_NUMBER environment variable or from a
    # GITHUB_REF tag reference ending in a number; None if neither is available. Cached.
    pr_number = os.environ.get("PR_NUMBER", "")
    if pr_number == "":
        re_match = re.match(r"^refs/tags/.*/(\d+)$", os.environ.get("GITHUB_REF", ""))
        if re_match is not None:
            pr_number = re_match.group(1)
    if pr_number != "":
        return int(pr_number)
    return None


@cache
def get_merge_base() -> str:
    # [FUNCTION DOC] Returns the commit to diff against: the PR merge base if a PR number is
    # known, else the merge base with the default branch (GIT_DEFAULT_BRANCH or main); on the
    # default branch itself it returns the previous commit. Cached; runs git.
    pr_number = get_pr_number()
    if pr_number is not None:
        return _get_pr_merge_base(_get_pr_info(pr_number))
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = _git_merge_base(default_branch)

    head = _git_head()

    if merge_base == head:
        # We are on the default branch, so check for changes since the last commit
        merge_base = "HEAD^"
    return merge_base


def query_changed_files() -> list[str]:
    # [FUNCTION DOC] Returns the list of files changed between the merge base and HEAD using
    # git, printing the base and the files. If git fails and a PR number is known, falls back
    # to the GitHub API; otherwise re-raises. Side effects: subprocess calls and printing.
    # KNOWN ISSUE: empty git output yields a list containing one empty string.
    pr_number = get_pr_number()
    try:
        base_commit = get_merge_base()

        proc = subprocess.run(
            ["git", "diff", "--name-only", base_commit, "HEAD"],
            capture_output=True,
            check=False,
        )
        print(f"base_commit: {base_commit}")

        if proc.returncode != 0:
            raise RuntimeError("Unable to get changed files")
    except (subprocess.CalledProcessError, RuntimeError):
        if pr_number is None:
            raise
        lines = _query_changed_files_from_github(pr_number)
        print(f"Changed files from GitHub: {lines}")
        return lines

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    print(f"Changed files: {lines}")
    return lines


# File extensions that are documentation-only and don't require running tests
DOCS_ONLY_EXTENSIONS = frozenset({".rst", ".md"})


def is_docs_only_change(changed_files: list[str]) -> bool:
    # [FUNCTION DOC] Returns True if the list is non-empty and every non-empty entry has a
    # documentation-only extension (.rst or .md); False otherwise.
    # KNOWN ISSUE: a list consisting only of empty strings returns True.
    """
    Returns True if all changed files are documentation-only files
    (e.g., .rst, .md files) that don't require running tests.
    """
    if not changed_files:
        return False

    for f in changed_files:
        # Skip empty strings that might come from git diff output
        if not f:
            continue
        # Check if the file extension is in the docs-only set
        ext = os.path.splitext(f)[1].lower()
        if ext not in DOCS_ONLY_EXTENSIONS:
            return False

    return True


@cache
def get_git_commit_info() -> str:
    # [FUNCTION DOC] Returns the git log text of commits between the merge base and HEAD.
    # Cached; runs git and raises CalledProcessError on failure.
    """Gets the commit info since the last commit on the default branch."""
    base_commit = get_merge_base()

    return (
        subprocess.check_output(
            ["git", "log", f"{base_commit}..HEAD"],
        )
        .decode()
        .strip()
    )


@cache
def get_issue_or_pr_body(number: int) -> str:
    # [FUNCTION DOC] Returns the body text of an issue or PR (empty if none) from the GitHub
    # issues API. Cached. Raises URL errors on failure.
    # KNOWN ISSUE: the authorization header is always sent, so a missing GITHUB_TOKEN produces a
    # header with the text None.
    """Gets the body of an issue or PR"""
    github_token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    # Despite the 'issues' in the link, this also works for PRs
    url = f"https://api.github.com/repos/pytorch/pytorch/issues/{number}"
    with urlopen(Request(url, headers=headers)) as conn:
        body: str = json.loads(conn.read().decode())["body"] or ""
        return body


def normalize_ratings(
    ratings: dict[TestRun, float], max_value: float, min_value: float = 0
) -> dict[TestRun, float]:
    # [FUNCTION DOC] Returns a new mapping with ratings scaled so the maximum becomes max_value
    # and the rest proportionally, then shifted so that a zero rating maps to min_value.
    # Returns the input unchanged when empty. Raises AssertionError if the minimum or maximum
    # rating is not positive.
    # KNOWN ISSUE: the comment says ratings must be >= 0, but a zero rating raises an error.
    # Takse the ratings, makes the max value into max_value, and proportionally
    # distributes the rest of the ratings.
    # Ex [1,2,3,4] and max_value 8 gets converted to [2,4,6,8]
    # Assumes all rankings are >= 0
    # min_value is what 0 gets mapped to and shifts the values accordingly.  Ex
    # [1,2,3,4], min_value 1, max_value 5 gets converted to [2,3,4,5]
    # Don't modify in place
    if len(ratings) == 0:
        return ratings
    min_rating = min(ratings.values())
    if min_rating <= 0:
        raise AssertionError(f"min_rating must be > 0, got {min_rating}")
    max_rating = max(ratings.values())
    if max_rating <= 0:
        raise AssertionError(f"max_rating must be > 0, got {max_rating}")
    normalized_ratings = {}
    for tf, rank in ratings.items():
        normalized_ratings[tf] = rank / max_rating * (max_value - min_value) + min_value
    return normalized_ratings


def get_ratings_for_tests(file: str | Path) -> dict[str, float]:
    # [FUNCTION DOC] Loads a correlation JSON (changed file to test to score) located relative
    # to the repo root and sums, over the changed files of the PR, the scores per test. Returns
    # an empty mapping, after printing a message, if the file is missing.
    path = REPO_ROOT / file
    if not os.path.exists(path):
        print(f"could not find path {path}")
        return {}
    with open(path) as f:
        test_file_ratings = cast(dict[str, dict[str, float]], json.load(f))
    changed_files = query_changed_files()
    ratings: dict[str, float] = defaultdict(float)
    for file in changed_files:
        for test_file, score in test_file_ratings.get(file, {}).items():
            ratings[test_file] += score
    return ratings


def get_correlated_tests(file: str | Path) -> list[str]:
    # [FUNCTION DOC] Returns the correlated tests of the file sorted by descending total score.
    ratings = get_ratings_for_tests(file)
    prioritize = sorted(ratings, key=lambda x: -ratings[x])
    return prioritize
