#!/usr/bin/env python3
# [MODULE DOC]
# Core of the PyTorch merge bot (trymerge). Fetches PR data through the GitHub GraphQL API
# (GitHubPR), loads merge_rules.yaml, evaluates mergeability (approvals, required checks,
# flaky / broken-trunk classification, labels, docker-build constraints), then merges,
# reverts, or only checks mergeability, posting status comments and labels along the way.
# Main contents: GraphQL query strings, regexes, check-state types, GitHubPR, merge rule
# data classes and matching, check classification helpers, and the main CLI entry point.
# No significant import-time side effects (queries and regexes are compiled/built only).

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from re import Pattern
from typing import Any, cast, NamedTuple, TYPE_CHECKING
from warnings import warn

import yaml
from github_utils import (
    gh_close_pr,
    gh_fetch_json_list,
    gh_fetch_merge_base,
    gh_fetch_url,
    gh_graphql,
    gh_merge_pr,
    gh_post_commit_comment,
    gh_post_pr_comment,
    gh_update_pr_state,
    GitHubComment,
)
from gitutils import (
    are_ghstack_branches_in_sync,
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
    patterns_to_regex,
    retries_decorator,
)
from label_utils import (
    gh_add_labels,
    gh_remove_label,
    has_required_labels,
    LABEL_ERR_MSG,
    NOT_USER_FACING_LABEL,
)
from trymerge_explainer import get_revert_message, TryMergeExplainer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# labels
MERGE_IN_PROGRESS_LABEL = "merging"
MERGE_COMPLETE_LABEL = "merged"


class JobCheckState(NamedTuple):
    # [CLASS DOC]
    # Immutable record of one CI job / check run: name, details url, conclusion status
    # (may be None), classification label (e.g. flaky / broken trunk, None if unclassified),
    # GitHub job id, and optional check run title and summary text.
    name: str
    url: str
    status: str | None
    classification: str | None
    job_id: int | None
    title: str | None
    summary: str | None


JobNameToStateDict = dict[str, JobCheckState]


class WorkflowCheckState:
    # [CLASS DOC]
    # Mutable record of one workflow run: name, url, run id, conclusion status, and a
    # mapping of check run name to JobCheckState for the jobs seen in that run.
    def __init__(self, name: str, url: str, run_id: int, status: str | None):
        # [FUNCTION DOC]
        # Stores the arguments and starts with an empty jobs mapping.
        self.name: str = name
        self.url: str = url
        self.run_id: int = run_id
        self.status: str | None = status
        self.jobs: JobNameToStateDict = {}


GH_PR_REVIEWS_FRAGMENT = """
fragment PRReviews on PullRequestReviewConnection {
  nodes {
    author {
      login
    }
    bodyText
    createdAt
    authorAssociation
    editor {
      login
    }
    databaseId
    url
    state
  }
  pageInfo {
    startCursor
    hasPreviousPage
  }
}
"""

GH_CHECKSUITES_FRAGMENT = """
fragment PRCheckSuites on CheckSuiteConnection {
  edges {
    node {
      workflowRun {
        workflow {
          name
          databaseId
        }
        databaseId
        url
      }
      checkRuns(first: 50) {
        nodes {
          name
          conclusion
          detailsUrl
          databaseId
          title
          summary
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      conclusion
    }
    cursor
  }
  pageInfo {
    hasNextPage
  }
}
"""

GH_COMMIT_AUTHORS_FRAGMENT = """
fragment CommitAuthors on PullRequestCommitConnection {
  nodes {
    commit {
      authors(first: 2) {
        nodes {
          user {
            login
          }
          email
          name
        }
      }
      oid
    }
  }
  pageInfo {
    endCursor
    hasNextPage
  }
}
"""

GH_GET_PR_INFO_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + GH_CHECKSUITES_FRAGMENT
    + GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closed
      isCrossRepository
      author {
        login
      }
      title
      body
      headRefName
      headRepository {
        nameWithOwner
      }
      baseRefName
      baseRefOid
      baseRepository {
        nameWithOwner
        isPrivate
        defaultBranchRef {
          name
        }
      }
      mergeCommit {
        oid
      }
      commits_with_authors: commits(first: 100) {
        ...CommitAuthors
        totalCount
      }
      commits(last: 1) {
        nodes {
          commit {
            checkSuites(first: 10) {
              ...PRCheckSuites
            }
            status {
              contexts {
                context
                state
                targetUrl
              }
            }
            oid
          }
        }
      }
      changedFiles
      files(first: 100) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
      reviews(last: 100) {
        ...PRReviews
      }
      comments(last: 5) {
        nodes {
          bodyText
          createdAt
          author {
            login
            url
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
      labels(first: 100) {
        edges {
          node {
            name
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_FILES_QUERY = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      files(first: 100, after: $cursor) {
        nodes {
          path
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_CHECKSUITES = (
    GH_CHECKSUITES_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 10, after: $cursor) {
              ...PRCheckSuites
            }
          }
        }
      }
    }
  }
}
"""
)

GH_GET_PR_NEXT_CHECK_RUNS = """
query ($owner: String!, $name: String!, $number: Int!, $cs_cursor: String, $cr_cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits(last: 1) {
        nodes {
          commit {
            oid
            checkSuites(first: 1, after: $cs_cursor) {
              nodes {
                checkRuns(first: 100, after: $cr_cursor) {
                  nodes {
                    name
                    conclusion
                    detailsUrl
                    databaseId
                    title
                    summary
                  }
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

GH_GET_PR_PREV_COMMENTS = """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      comments(last: 100, before: $cursor) {
        nodes {
          bodyText
          createdAt
          author {
            login
          }
          authorAssociation
          editor {
            login
          }
          databaseId
          url
        }
        pageInfo {
          startCursor
          hasPreviousPage
        }
      }
    }
  }
}
"""

# This query needs read-org permission
GH_GET_TEAM_MEMBERS_QUERY = """
query($org: String!, $name: String!, $cursor: String) {
  organization(login: $org) {
    team(slug: $name) {
      members(first: 100, after: $cursor) {
        nodes {
          login
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

GH_GET_PR_NEXT_AUTHORS_QUERY = (
    GH_COMMIT_AUTHORS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      commits_with_authors: commits(first: 100, after: $cursor) {
        ...CommitAuthors
      }
    }
  }
}
"""
)

GH_GET_PR_PREV_REVIEWS_QUERY = (
    GH_PR_REVIEWS_FRAGMENT
    + """
query ($owner: String!, $name: String!, $number: Int!, $cursor: String!) {
  repository(name: $name, owner: $owner) {
    pullRequest(number: $number) {
      reviews(last: 100, before: $cursor) {
        ...PRReviews
      }
    }
  }
}
"""
)

GH_GET_REPO_SUBMODULES = """
query ($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    submodules(first: 100) {
      nodes {
        path
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
"""

RE_GHSTACK_HEAD_REF = re.compile(r"^(gh/[^/]+/[0-9]+/)head$")
RE_GHSTACK_DESC = re.compile(r"Stack.*:\r?\n(\* [^\r\n]+\r?\n)+", re.MULTILINE)
RE_PULL_REQUEST_RESOLVED = re.compile(
    r"(Pull Request resolved|Pull-Request-resolved|Pull-Request): "
    r"https://github.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>[0-9]+)",
    re.MULTILINE,
)
RE_PR_CC_LINE = re.compile(r"^cc:? @\w+.*\r?\n?$", re.MULTILINE)
RE_DIFF_REV = re.compile(r"^Differential Revision:.+?(D[0-9]+)", re.MULTILINE)
CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")
MERGE_RULE_PATH = Path(".github") / "merge_rules.yaml"
REMOTE_MAIN_BRANCH = "origin/main"
DRCI_CHECKRUN_NAME = "Dr.CI"
INTERNAL_CHANGES_CHECKRUN_NAME = "Meta Internal-Only Changes Check"
HAS_NO_CONNECTED_DIFF_TITLE = (
    "There is no internal Diff connected, this can be merged now"
)
# This could be set to -1 to ignore all flaky and broken trunk failures. On the
# other hand, using a large value like 10 here might be useful in sev situation
IGNORABLE_FAILED_CHECKS_THESHOLD = 10

# CI docker images are keyed by the git tree hash of the .ci/docker directory
# (see .github/workflows/docker-builds.yml, which tags the images it builds with
# `git rev-parse HEAD:.ci/docker`).  A PR that touches these files must have its
# images pre-built and pushed to ECR by the docker-builds (ciflow/docker)
# workflow, and the .ci/docker tree hash of the merge commit must match the tree
# hash that was actually built.  If another docker-affecting PR lands in the
# skew window, the merge commit would reference an image tag that was never
# built, leaving every downstream trunk job unable to find its image (see the
# #190927 / #186302 land race).
DOCKER_CI_PATH = ".ci/docker"
DOCKER_BUILDS_WORKFLOW_NAME = "docker-builds"


def iter_issue_timeline_until_comment(
    org: str, repo: str, issue_number: int, target_comment_id: int, max_pages: int = 200
) -> Any:
    # [FUNCTION DOC]
    # Generator over the issue timeline events fetched page by page (100 per page, via the
    # REST timeline endpoint), yielding each event in order and stopping right after the
    # "commented" event whose id equals target_comment_id. Stops early on an empty or short
    # page. Raises RuntimeError if the comment is not found within max_pages pages.
    # KNOWN ISSUE: if a page is exactly the last one and no match, the error path reuses the
    # url variable, which is undefined when max_pages is less than 1.
    """
    Yield timeline entries in order until (and including) the entry whose id == target_comment_id
    for a 'commented' event. Stops once the target comment is encountered.
    """
    page = 1

    while page <= max_pages:
        url = (
            f"https://api.github.com/repos/{org}/{repo}/issues/{issue_number}/timeline"
        )
        params = {"per_page": 100, "page": page}

        batch = gh_fetch_json_list(url, params)

        if not batch:
            return
        for ev in batch:
            # The target is the issue comment row with event == "commented" and id == issue_comment_id
            if ev.get("event") == "commented" and ev.get("id") == target_comment_id:
                yield ev  # nothing in the timeline after this matters, so stop early
                return
            yield ev
        if len(batch) < 100:
            return
        page += 1

    # If we got here without finding the comment, then we either hit a bug or some github PR
    # has a _really_ long timeline.
    # The max # of pages found on any pytorch/pytorch PR at the time of this change was 41
    raise RuntimeError(
        f"Could not find a merge commit in the first {max_pages} pages of the timeline at url {url}."
        f"This is most likely a bug, please report it to the @pytorch/pytorch-dev-infra team."
    )


def sha_from_committed_event(ev: dict[str, Any]) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the sha field of a timeline committed event, or None.
    """Extract SHA from committed event in timeline"""
    return ev.get("sha")


def sha_from_force_push_after(ev: dict[str, Any]) -> str | None:
    # [FUNCTION DOC] HELPER: Extracts the post-force-push head sha from a timeline event.
    # Prefers the commit_id field (current API); otherwise falls back to legacy fields
    # (after / after_commit objects with sha or oid, then after_sha / head_sha). May be None.
    """Extract SHA from force push event in timeline"""
    # The current GitHub API format
    commit_id = ev.get("commit_id")
    if commit_id:
        return str(commit_id)

    # Legacy format
    after = ev.get("after") or ev.get("after_commit") or {}
    if isinstance(after, dict):
        return after.get("sha") or after.get("oid")
    return ev.get("after_sha") or ev.get("head_sha")


def gh_get_pr_info(org: str, proj: str, pr_no: int) -> Any:
    # [FUNCTION DOC]
    # Runs the main PR info GraphQL query and returns the pullRequest object (nested
    # mappings with reviews, commits, check suites, files, comments, labels). Network call.
    rc = gh_graphql(GH_GET_PR_INFO_QUERY, name=proj, owner=org, number=pr_no)
    return rc["data"]["repository"]["pullRequest"]


@cache
def gh_get_team_members(org: str, name: str) -> list[str]:
    # [FUNCTION DOC]
    # Returns all member logins of a GitHub org team, paging through GraphQL results.
    # Returns an empty list (and emits a warning) if the team does not exist. Results are
    # cached per (org, name) for the process lifetime. Needs read-org permission.
    rc: list[str] = []
    team_members: dict[str, Any] = {
        "pageInfo": {"hasNextPage": "true", "endCursor": None}
    }
    while bool(team_members["pageInfo"]["hasNextPage"]):
        query = gh_graphql(
            GH_GET_TEAM_MEMBERS_QUERY,
            org=org,
            name=name,
            cursor=team_members["pageInfo"]["endCursor"],
        )
        team = query["data"]["organization"]["team"]
        if team is None:
            warn(f"Requested non-existing team {org}/{name}")
            return []
        team_members = team["members"]
        rc += [member["login"] for member in team_members["nodes"]]
    return rc


def get_check_run_name_prefix(workflow_run: Any) -> str:
    # [FUNCTION DOC] HELPER: Returns the workflow name followed by a slash separator to
    # prefix check run names, or an empty string when there is no workflow run.
    if workflow_run is None:
        return ""
    else:
        return f"{workflow_run['workflow']['name']} / "


def is_passing_status(status: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True if the status is non-None and, case-insensitively, one of
    # success, skipped or neutral.
    return status is not None and status.upper() in ["SUCCESS", "SKIPPED", "NEUTRAL"]


def is_docker_affecting_files(files: Iterable[str]) -> bool:
    # [FUNCTION DOC]
    # True if any path equals or lies under the CI docker directory (these change the
    # docker image tag).
    """Whether any of the given files change the CI docker image tree hash.

    The docker image tag is derived purely from the .ci/docker tree, so only
    changes under that directory matter (changes to docker-builds.yml or
    .lintrunner.toml re-trigger the build workflow but do not change the tag).
    """
    return any(f == DOCKER_CI_PATH or f.startswith(f"{DOCKER_CI_PATH}/") for f in files)


def get_docker_build_checks(checks: JobNameToStateDict) -> JobNameToStateDict:
    # [FUNCTION DOC]
    # Filters the checks mapping to entries whose name is the docker-builds workflow or
    # starts with that workflow name plus a slash separator.
    """Return the subset of checks that belong to the docker-builds workflow."""
    return {
        name: check
        for name, check in checks.items()
        if name == DOCKER_BUILDS_WORKFLOW_NAME
        or name.startswith(f"{DOCKER_BUILDS_WORKFLOW_NAME} / ")
    }


def add_workflow_conclusions(
    checksuites: Any,
    get_next_checkruns_page: Callable[[list[dict[str, dict[str, Any]]], int, Any], Any],
    get_next_checksuites: Callable[[Any], Any],
) -> JobNameToStateDict:
    # [FUNCTION DOC]
    # Converts GraphQL check suite data into a mapping of job name to JobCheckState.
    # Pages through extra check suites and check runs using the two callbacks, keeps only
    # the newest run (largest run id) per workflow, ignores cancelled runs that would
    # override an existing workflow, and keeps a non-passing duplicate job over a passing
    # one. Workflows with jobs contribute their jobs only; empty workflows contribute
    # themselves; jobs without a workflow are added last.
    # graphql seems to favor the most recent workflow run, so in theory we
    # shouldn't need to account for reruns, but do it just in case

    # workflow -> job -> job info
    workflows: dict[str, WorkflowCheckState] = {}

    # for the jobs that don't have a workflow
    no_workflow_obj: WorkflowCheckState = WorkflowCheckState("", "", 0, None)

    def add_conclusions(edges: Any) -> None:
        # [FUNCTION DOC] HELPER: Walks check suite edges, registers each workflow run and
        # records its check runs as jobs (following check run pagination via the callback).
        # Mutates the enclosing workflows mapping and the no-workflow holder.
        # KNOWN ISSUE: the cancelled-run test looks up workflow_name in a mapping keyed by
        # workflow_id, so it never matches.
        for edge_idx, edge in enumerate(edges):
            node = edge["node"]
            workflow_run = node["workflowRun"]
            checkruns = node["checkRuns"]

            workflow_obj: WorkflowCheckState = no_workflow_obj

            if workflow_run is not None:
                # This is the usual workflow run ID we see on GitHub
                workflow_run_id = workflow_run["databaseId"]
                # While this is the metadata name and ID of the workflow itself
                workflow_name = workflow_run["workflow"]["name"]
                workflow_id = workflow_run["workflow"]["databaseId"]

                workflow_conclusion = node["conclusion"]
                # Do not override existing status with cancelled
                if workflow_conclusion == "CANCELLED" and workflow_name in workflows:
                    continue

                # Only keep the latest workflow run for each workflow, heuristically,
                # it's the run with largest run ID
                if (
                    workflow_id not in workflows
                    or workflows[workflow_id].run_id < workflow_run_id
                ):
                    workflows[workflow_id] = WorkflowCheckState(
                        name=workflow_name,
                        status=workflow_conclusion,
                        url=workflow_run["url"],
                        run_id=workflow_run_id,
                    )
                workflow_obj = workflows[workflow_id]

            while checkruns is not None:
                for checkrun_node in checkruns["nodes"]:
                    if not isinstance(checkrun_node, dict):
                        warn(f"Expected dictionary, but got {type(checkrun_node)}")
                        continue
                    checkrun_name = f"{get_check_run_name_prefix(workflow_run)}{checkrun_node['name']}"
                    existing_checkrun = workflow_obj.jobs.get(checkrun_name)
                    if existing_checkrun is None or not is_passing_status(
                        existing_checkrun.status
                    ):
                        workflow_obj.jobs[checkrun_name] = JobCheckState(
                            checkrun_name,
                            checkrun_node["detailsUrl"],
                            checkrun_node["conclusion"],
                            classification=None,
                            job_id=checkrun_node["databaseId"],
                            title=checkrun_node["title"],
                            summary=checkrun_node["summary"],
                        )

                if bool(checkruns["pageInfo"]["hasNextPage"]):
                    checkruns = get_next_checkruns_page(edges, edge_idx, checkruns)
                else:
                    checkruns = None

    all_edges = checksuites["edges"].copy()
    while bool(checksuites["pageInfo"]["hasNextPage"]):
        checksuites = get_next_checksuites(checksuites)
        all_edges.extend(checksuites["edges"])

    add_conclusions(all_edges)

    # Flatten the dictionaries.  If there exists jobs in the workflow run, put
    # the jobs in but don't put the workflow in.  We care more about the jobs in
    # the workflow that ran than the container workflow.
    res: JobNameToStateDict = {}
    for workflow in workflows.values():
        if len(workflow.jobs) > 0:
            for job_name, job in workflow.jobs.items():
                res[job_name] = job
        else:
            res[workflow.name] = JobCheckState(
                workflow.name,
                workflow.url,
                workflow.status,
                classification=None,
                job_id=None,
                title=None,
                summary=None,
            )
    for job_name, job in no_workflow_obj.jobs.items():
        res[job_name] = job
    return res


def parse_args() -> Any:
    # [FUNCTION DOC] HELPER: Parses the command line: positional pr_num, flags --dry-run,
    # --revert, --force, --ignore-current, --check-mergeability, and options --comment-id
    # (int) and --reason (str). Returns the argparse namespace.
    from argparse import ArgumentParser

    parser = ArgumentParser("Merge PR into default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ignore-current", action="store_true")
    parser.add_argument("--check-mergeability", action="store_true")
    parser.add_argument("--comment-id", type=int)
    parser.add_argument("--reason", type=str)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def _unedited_comment_author(
    pr: GitHubPR, comment_id: int | None
) -> tuple[str, str | None] | None:
    # [FUNCTION DOC] HELPER: Looks up the triggering comment and returns its author login
    # and author url, or None if there is no comment id or the comment was edited.
    """(author_login, author_url) of the triggering comment.

    None when there is no comment or it has been edited, so a caller can never
    act on an identity someone may have rewritten.
    """
    if comment_id is None:
        return None
    comment = pr.get_comment_by_id(comment_id)
    if comment.editor_login is not None:
        return None
    return comment.author_login, comment.author_url


def can_skip_internal_checks(pr: GitHubPR, comment_id: int | None = None) -> bool:
    # [FUNCTION DOC]
    # True if the unedited triggering comment was written by the facebook-github-bot user
    # or by the facebook-github-tools GitHub App (matched by app url); such merges may skip
    # the internal-diff checks. False when the comment is missing or edited.
    author = _unedited_comment_author(pr, comment_id)
    if author is None:
        return False
    author_login, author_url = author
    if author_login == "facebook-github-bot":
        return True
    # facebook-github-tools is a GitHub App; identify by its app URL.
    return author_url == "https://github.com/apps/facebook-github-tools"


# Bot identities that auto-merge a co-dev PR once its internal Phabricator diff
# has landed. `meta-codesync` took this over from `facebook-github-tools`, so
# both are listed: the older one still appears on long-lived PRs.
#
# Matched on login rather than on the App URL that can_skip_internal_checks
# uses. Only `comments(last: 5)` selects `author { url }` — GH_GET_PR_PREV_COMMENTS
# and the reviews fragment select `login` alone, so author_url is silently None
# for any comment older than the prefetched window (see _comment_from_node's
# `.get("url", None)`). `login` is selected everywhere, and a `[bot]` suffix is
# unforgeable because GitHub usernames cannot contain brackets.
#
# Deliberately NOT reusing can_skip_internal_checks' allowlist. That predicate
# also waives the "must be landed via Phabricator" guard, so adding an identity
# to it is a trust decision; this list only decides whether a missing
# release-notes label is auto-filled.
CODEV_MERGE_BOT_LOGINS = frozenset(
    {
        "facebook-github-bot",
        "facebook-github-tools[bot]",
        "meta-codesync[bot]",
    }
)
# Same identities by App URL, for the queries that do select it.
CODEV_MERGE_BOT_APP_URLS = frozenset(
    {
        "https://github.com/apps/facebook-github-tools",
        "https://github.com/apps/meta-codesync",
    }
)


def is_bot_initiated_codev_merge(pr: GitHubPR, comment_id: int | None) -> bool:
    # [FUNCTION DOC]
    # True when the PR has a Differential Revision line and the unedited triggering comment
    # was authored by a known co-dev merge bot (by login or app url). Cheap local check first.
    # A co-dev merge is initiated by one of the Meta export bots once the
    # internal Phabricator diff has landed. Such a PR always carries a
    # "Differential Revision:" line (get_diff_revision); the local check runs
    # first so a human merge does not pay a comment fetch just to be rejected.
    if pr.get_diff_revision() is None:
        return False
    author = _unedited_comment_author(pr, comment_id)
    if author is None:
        return False
    author_login, author_url = author
    return (
        author_login in CODEV_MERGE_BOT_LOGINS or author_url in CODEV_MERGE_BOT_APP_URLS
    )


def ensure_mergeable_labels(
    pr: GitHubPR, comment_id: int | None, dry_run: bool
) -> None:
    # [FUNCTION DOC]
    # Ensures the PR has the required release-notes labels. If missing and the merge was
    # initiated by a co-dev bot, posts an audit comment and then adds the not-user-facing
    # label (comment first by design). Otherwise raises RuntimeError with the label error
    # message. Side effects: PR comment and label change (skipped when dry_run).
    if has_required_labels(pr):
        return
    # A co-dev merge runs after the internal diff already landed, so the author
    # can no longer act on a missing release-notes label. Auto-apply
    # "topic: not user facing" to unblock (as maintainers do manually today) and
    # leave a comment so the choice is auditable and can be corrected if the
    # change is in fact user facing. Human-initiated merges still fail.
    if is_bot_initiated_codev_merge(pr, comment_id):
        # Comment BEFORE labeling. If the label went on first and the comment
        # then failed, the merge would abort with the PR already satisfying
        # has_required_labels — so the retry would take the early return above
        # and the promised audit trail would never be written.
        gh_post_pr_comment(
            pr.org,
            pr.project,
            pr.pr_num,
            f"Adding the `{NOT_USER_FACING_LABEL}` label automatically: this co-dev "
            "PR is being merged after its internal diff landed, without a "
            "release-notes label. If this change is user facing, please replace "
            "the label with the appropriate `release notes: ...` one.",
            dry_run,
        )
        gh_add_labels(pr.org, pr.project, pr.pr_num, [NOT_USER_FACING_LABEL], dry_run)
    else:
        raise RuntimeError(LABEL_ERR_MSG.lstrip(" #"))


def _revlist_to_prs(
    repo: GitRepo,
    pr: GitHubPR,
    rev_list: Iterable[str],
    should_skip: Callable[[int, GitHubPR], bool] | None = None,
) -> list[tuple[GitHubPR, str]]:
    # [FUNCTION DOC] HELPER: Maps commit revisions to (GitHubPR, revision) pairs by parsing
    # the single "Pull Request resolved" url in each commit message. Reuses the given pr
    # object for its own number; skips entries for which should_skip(index, pr) is true.
    # Raises RuntimeError if a commit mentions zero or several PRs or a different repo.
    rc: list[tuple[GitHubPR, str]] = []
    for idx, rev in enumerate(rev_list):
        msg = repo.commit_message(rev)
        # findall doesn't return named captures, so we need to use finditer
        all_matches = list(RE_PULL_REQUEST_RESOLVED.finditer(msg))
        if len(all_matches) != 1:
            raise RuntimeError(
                f"Found an unexpected number of PRs mentioned in commit {rev}: "
                f"{len(all_matches)}.  This is probably because you are using an "
                "old version of ghstack.  Please update ghstack and resubmit "
                "your PRs"
            )

        m = all_matches[0]
        if pr.org != m.group("owner") or pr.project != m.group("repo"):
            raise RuntimeError(
                f"PR {m.group('number')} resolved to wrong owner/repo pair"
            )
        pr_num = int(m.group("number"))
        candidate = GitHubPR(pr.org, pr.project, pr_num) if pr_num != pr.pr_num else pr
        if should_skip is not None and should_skip(idx, candidate):
            continue
        rc.append((candidate, rev))
    return rc


def get_ghstack_prs(
    repo: GitRepo, pr: GitHubPR, open_only: bool = True
) -> list[tuple[GitHubPR, str]]:
    # [FUNCTION DOC]
    # Returns the ghstack PRs at and below this PR as (pr, commit revision) pairs, oldest
    # first, from the commits between the default branch and the origin orig ref. Closed
    # PRs are skipped when open_only. Raises AssertionError if the PR is not ghstack, and
    # RuntimeError if any open PR's head branch is out of sync with its orig revision.
    # Prints progress.
    """
    Get the PRs in the stack that are below this PR (inclusive).  Throws error if any of the open PRs are out of sync.
    @:param open_only: Only return open PRs
    """
    # For ghstack, cherry-pick commits based from origin
    orig_ref = f"{repo.remote}/{pr.get_ghstack_orig_ref()}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")

    def skip_func(idx: int, candidate: GitHubPR) -> bool:
        # [FUNCTION DOC] HELPER: True (with a printed note) when only open PRs are wanted and
        # the candidate is already closed.
        if not open_only or not candidate.is_closed():
            return False
        print(
            f"Skipping {idx + 1} of {len(rev_list)} PR (#{candidate.pr_num}) as its already been merged"
        )
        return True

    if not pr.is_ghstack_pr():
        raise AssertionError(
            f"Expected PR #{pr.pr_num} to be a ghstack PR, but head_ref is {pr.head_ref()}"
        )
    entire_stack = _revlist_to_prs(repo, pr, reversed(rev_list), skip_func)
    print(
        f"Found {len(entire_stack)} PRs in the stack for {pr.pr_num}: {[x[0].pr_num for x in entire_stack]}"
    )

    for stacked_pr, rev in entire_stack:
        if stacked_pr.is_closed():
            continue
        base_ref = stacked_pr.base_ref()
        if base_ref == pr.default_branch():
            base_ref = repo.get_merge_base(
                f"{repo.remote}/{base_ref}", f"{repo.remote}/{stacked_pr.head_ref()}"
            )
        if not are_ghstack_branches_in_sync(repo, stacked_pr.head_ref(), base_ref):
            raise RuntimeError(
                f"PR {stacked_pr.pr_num} is out of sync with the corresponding revision {rev} on "
                + f"branch {stacked_pr.get_ghstack_orig_ref()} that would be merged into {stacked_pr.default_branch()}.  "
                + "This usually happens because there is a non ghstack change in the PR.  "
                + f"Please sync them and try again (ex. make the changes on {orig_ref} and run ghstack)."
            )
    return entire_stack


class GitHubPR:
    # [CLASS DOC]
    # Wrapper around one pull request. State: org, project, pr_num, the raw GraphQL info
    # mapping fetched at construction, and lazily filled caches (changed files, labels, CI
    # conclusions, comments, authors, reviews, merge base, submodules). Provides accessors
    # for refs and ghstack status, comment / review / author retrieval, check status
    # gathering, merge rule evaluation (find_matching_merge_rule), commit message
    # construction, and merge / close helpers.
    def __init__(self, org: str, project: str, pr_num: int) -> None:
        # [FUNCTION DOC]
        # Fetches PR info from GitHub and initializes empty caches. Raises AssertionError if
        # pr_num is not an int. Network call.
        if not isinstance(pr_num, int):
            raise AssertionError(
                f"pr_num must be an int, got {type(pr_num).__name__}: {pr_num}"
            )
        self.org = org
        self.project = project
        self.pr_num = pr_num
        self.info = gh_get_pr_info(org, project, pr_num)
        self.changed_files: list[str] | None = None
        self.labels: list[str] | None = None
        self.conclusions: JobNameToStateDict | None = None
        self.comments: list[GitHubComment] | None = None
        self._authors: list[tuple[str, str]] | None = None
        self._reviews: list[tuple[str, str]] | None = None
        self.merge_base: str | None = None
        self.submodules: list[str] | None = None

    def is_closed(self) -> bool:
        # [FUNCTION DOC] True if the PR is closed (merged or abandoned).
        return bool(self.info["closed"])

    def is_cross_repo(self) -> bool:
        # [FUNCTION DOC] True if the PR head lives in a different repository (a fork).
        return bool(self.info["isCrossRepository"])

    def base_ref(self) -> str:
        # [FUNCTION DOC] Returns the name of the PR base branch.
        return cast(str, self.info["baseRefName"])

    def default_branch(self) -> str:
        # [FUNCTION DOC] Returns the default branch name of the base repository.
        return cast(str, self.info["baseRepository"]["defaultBranchRef"]["name"])

    def head_ref(self) -> str:
        # [FUNCTION DOC] Returns the name of the PR head branch.
        return cast(str, self.info["headRefName"])

    def is_ghstack_pr(self) -> bool:
        # [FUNCTION DOC] True if the head branch name looks like a ghstack head branch.
        return RE_GHSTACK_HEAD_REF.match(self.head_ref()) is not None

    def get_ghstack_orig_ref(self) -> str:
        # [FUNCTION DOC] Returns the ghstack orig branch name derived from the head branch
        # (head suffix replaced by orig). Raises AssertionError for non-ghstack PRs.
        if not self.is_ghstack_pr():
            raise AssertionError(
                f"get_ghstack_orig_ref called on non-ghstack PR #{self.pr_num}"
            )
        return re.sub(r"/head$", "/orig", self.head_ref())

    def is_base_repo_private(self) -> bool:
        # [FUNCTION DOC] True if the base repository is private.
        return bool(self.info["baseRepository"]["isPrivate"])

    def get_changed_files_count(self) -> int:
        # [FUNCTION DOC] Returns the changed file count reported by GitHub.
        return int(self.info["changedFiles"])

    def last_commit(self) -> Any:
        # [FUNCTION DOC] Returns the commit object of the PR's last commit from the info.
        return self.info["commits"]["nodes"][-1]["commit"]

    def last_commit_sha(self, default: str | None = None) -> str:
        # [FUNCTION DOC] Returns the sha of the last commit. With no default, a missing oid
        # raises KeyError; with a default, that value is returned when the oid is absent.
        # for commits, the oid is the sha

        if default is None:
            return str(self.last_commit()["oid"])

        return str(self.last_commit().get("oid", default))

    def get_merge_base(self) -> str:
        # [FUNCTION DOC]
        # Returns (and caches) the merge base of the last commit against the default branch
        # via the GitHub API, falling back to the PR's base ref oid when the call yields
        # nothing (e.g. rate limiting).
        if self.merge_base:
            return self.merge_base

        last_commit_sha = self.last_commit_sha()
        # NB: We could use self.base_ref() here for regular PR, however, that doesn't
        # work for ghstack where the base is the custom branch, i.e. gh/USER/ID/base,
        # so let's just use main instead
        self.merge_base = gh_fetch_merge_base(
            self.org, self.project, last_commit_sha, self.default_branch()
        )

        # Fallback to baseRefOid if the API call fails, i.e. rate limit. Note that baseRefOid
        # points to the base ref associated with the PR or, in other words, the head of main
        # when the PR is created or rebased. This is not necessarily the merge base commit,
        # but it could serve as a fallback in most cases and it's readily available as part
        # of the PR info
        if not self.merge_base:
            self.merge_base = cast(str, self.info["baseRefOid"])

        return self.merge_base

    def get_changed_files(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns (and caches) the unique changed file paths, paging through GraphQL file
        # results up to 100 pages. Raises RuntimeError if the number collected differs from
        # the count GitHub reports.
        if self.changed_files is None:
            info = self.info
            unique_changed_files = set()
            # Do not try to fetch more than 10K files
            for _ in range(100):
                unique_changed_files.update([x["path"] for x in info["files"]["nodes"]])
                if not info["files"]["pageInfo"]["hasNextPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_NEXT_FILES_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["files"]["pageInfo"]["endCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
            self.changed_files = list(unique_changed_files)

        if len(self.changed_files) != self.get_changed_files_count():
            raise RuntimeError("Changed file count mismatch")
        return self.changed_files

    def get_submodules(self) -> list[str]:
        # [FUNCTION DOC] Returns (and caches) the submodule paths of the repository.
        if self.submodules is None:
            rc = gh_graphql(GH_GET_REPO_SUBMODULES, name=self.project, owner=self.org)
            info = rc["data"]["repository"]["submodules"]
            self.submodules = [s["path"] for s in info["nodes"]]
        return self.submodules

    def get_changed_submodules(self) -> list[str]:
        # [FUNCTION DOC] Returns the changed files that are submodule paths.
        submodules = self.get_submodules()
        return [f for f in self.get_changed_files() if f in submodules]

    def is_docker_affecting(self) -> bool:
        # [FUNCTION DOC] True if any changed file affects the CI docker image tag.
        """Whether this PR modifies files that change the CI docker image tag."""
        return is_docker_affecting_files(self.get_changed_files())

    def has_invalid_submodule_updates(self) -> bool:
        # [FUNCTION DOC]
        # True if the PR changes any submodule but the word submodule appears in none of the
        # title, body or labels (case-insensitive for title/body).
        """Submodule updates in PR are invalid if submodule keyword
        is not mentioned in neither the title nor body/description
        nor in any of the labels.
        """
        return (
            len(self.get_changed_submodules()) > 0
            and "submodule" not in self.get_title().lower()
            and "submodule" not in self.get_body().lower()
            and all("submodule" not in label for label in self.get_labels())
        )

    def _get_reviews(self) -> list[tuple[str, str]]:
        # [FUNCTION DOC] HELPER: Collects (author login, state) review pairs in chronological
        # order (paging backwards through up to 100 pages) and caches the raw list. Returns the
        # latest non-comment review state per author as a list of pairs.
        if self._reviews is None:
            self._reviews = []
            info = self.info
            for _ in range(100):
                nodes = info["reviews"]["nodes"]
                self._reviews = [
                    (node["author"]["login"], node["state"]) for node in nodes
                ] + self._reviews
                if not info["reviews"]["pageInfo"]["hasPreviousPage"]:
                    break
                rc = gh_graphql(
                    GH_GET_PR_PREV_REVIEWS_QUERY,
                    name=self.project,
                    owner=self.org,
                    number=self.pr_num,
                    cursor=info["reviews"]["pageInfo"]["startCursor"],
                )
                info = rc["data"]["repository"]["pullRequest"]
        reviews = {
            author: state for author, state in self._reviews if state != "COMMENTED"
        }
        return list(reviews.items())

    def get_approved_by(self) -> list[str]:
        # [FUNCTION DOC] Returns logins whose latest non-comment review state is approved.
        return [login for (login, state) in self._get_reviews() if state == "APPROVED"]

    def get_commit_count(self) -> int:
        # [FUNCTION DOC] Returns the total number of commits in the PR as reported by GitHub.
        return int(self.info["commits_with_authors"]["totalCount"])

    def get_commit_sha_at_comment(self, comment_id: int) -> str | None:
        # [FUNCTION DOC]
        # Replays the issue timeline up to the given comment, tracking the latest head sha
        # from committed and force-push events, and returns the head at the comment time.
        # Returns None if the comment is not found or any error occurs (a warning is printed).
        # Prints progress lines.
        # KNOWN ISSUE: the final print uses the variable sha from the last head-changing
        # event, which is unbound if none occurred; the resulting NameError is swallowed by
        # the broad exception handler, which then returns None.
        """
        Get the PR head commit SHA that was present when a specific comment was posted.
        This ensures we only merge the state of the PR at the time the merge command was issued,
        not any subsequent commits that may have been pushed after.

        Returns None if no head-changing events found before the comment or if the comment was not found.
        """
        head = None

        try:
            for event in iter_issue_timeline_until_comment(
                self.org, self.project, self.pr_num, comment_id
            ):
                etype = event.get("event")
                if etype == "committed":
                    sha = sha_from_committed_event(event)
                    if sha:
                        head = sha
                        print(f"Timeline: Found commit event for SHA {sha}")
                elif etype == "head_ref_force_pushed":
                    sha = sha_from_force_push_after(event)
                    if sha:
                        head = sha
                        print(f"Timeline: Found force push event for SHA {sha}")
                elif etype == "commented":
                    if event.get("id") == comment_id:
                        print(f"Timeline: Found final comment with sha {sha}")
                        return head
        except Exception as e:
            print(
                f"Warning: Failed to reconstruct timeline for comment {comment_id}: {e}"
            )
            return None

        print(f"Did not find comment with id {comment_id} in the PR timeline")
        return None

    def get_pr_creator_login(self) -> str:
        # [FUNCTION DOC] Returns the login of the user who opened the PR.
        return cast(str, self.info["author"]["login"])

    def is_dependabot_pr(self) -> bool:
        # [FUNCTION DOC] True if the PR was opened by the dependabot bot account.
        return self.get_pr_creator_login() == "dependabot[bot]"

    def _fetch_authors(self) -> list[tuple[str, str]]:
        # [FUNCTION DOC] HELPER: Returns (and caches) a list of (login, "Name <email>") pairs
        # for the authors of all PR commits, paging up to 100 pages. Login is empty when the
        # commit author is not a GitHub user.
        if self._authors is not None:
            return self._authors
        authors: list[tuple[str, str]] = []

        def add_authors(info: dict[str, Any]) -> None:
            # [FUNCTION DOC] HELPER: Appends (login, formatted author) pairs from one page of
            # commit author data to the enclosing authors list.
            for node in info["commits_with_authors"]["nodes"]:
                for author_node in node["commit"]["authors"]["nodes"]:
                    user_node = author_node["user"]
                    author = f"{author_node['name']} <{author_node['email']}>"
                    if user_node is None:
                        # If author is not github user, user node will be null
                        authors.append(("", author))
                    else:
                        authors.append((cast(str, user_node["login"]), author))

        info = self.info
        for _ in range(100):
            add_authors(info)
            if not info["commits_with_authors"]["pageInfo"]["hasNextPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_NEXT_AUTHORS_QUERY,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["commits_with_authors"]["pageInfo"]["endCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
        self._authors = authors
        return authors

    def get_committer_login(self, num: int = 0) -> str:
        # [FUNCTION DOC] Returns the GitHub login of the num-th commit author (may be empty).
        return self._fetch_authors()[num][0]

    def get_committer_author(self, num: int = 0) -> str:
        # [FUNCTION DOC] Returns the "Name <email>" string of the num-th commit author.
        return self._fetch_authors()[num][1]

    def get_labels(self) -> list[str]:
        # [FUNCTION DOC] Returns (and caches) the label names from the fetched PR info,
        # or an empty list if the info has no labels field.
        if self.labels is not None:
            return self.labels
        labels = (
            [node["node"]["name"] for node in self.info["labels"]["edges"]]
            if "labels" in self.info
            else []
        )
        self.labels = labels
        return self.labels

    def get_checkrun_conclusions(self) -> JobNameToStateDict:
        # [FUNCTION DOC]
        # Returns (and caches) the mapping of job name to JobCheckState for the last commit,
        # built from check suites (with pagination callbacks) plus legacy commit statuses.
        # Removes the Claude Code check, which would block merges. Raises RuntimeError if the
        # last commit changes during pagination. Network calls.
        """Returns dict of checkrun -> [conclusion, url]"""
        if self.conclusions is not None:
            return self.conclusions
        orig_last_commit = self.last_commit()

        def get_pr_next_check_runs(
            edges: list[dict[str, dict[str, Any]]], edge_idx: int, checkruns: Any
        ) -> Any:
            # [FUNCTION DOC] HELPER: Fetches the next page of check runs for the check suite at
            # edge_idx (using the previous edge's cursor) and returns its checkRuns object.
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECK_RUNS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cs_cursor=edges[edge_idx - 1]["cursor"] if edge_idx > 0 else None,
                cr_cursor=checkruns["pageInfo"]["endCursor"],
            )
            last_commit = rc["data"]["repository"]["pullRequest"]["commits"]["nodes"][
                -1
            ]["commit"]
            checkruns = last_commit["checkSuites"]["nodes"][-1]["checkRuns"]
            return checkruns

        def get_pr_next_checksuites(checksuites: Any) -> Any:
            # [FUNCTION DOC] HELPER: Fetches the next page of check suites after the last edge
            # cursor. Raises RuntimeError if the PR head commit changed meanwhile.
            rc = gh_graphql(
                GH_GET_PR_NEXT_CHECKSUITES,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=checksuites["edges"][-1]["cursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]
            last_commit = info["commits"]["nodes"][-1]["commit"]
            if last_commit["oid"] != orig_last_commit["oid"]:
                raise RuntimeError("Last commit changed on PR")
            return last_commit["checkSuites"]

        checksuites = orig_last_commit["checkSuites"]

        self.conclusions = add_workflow_conclusions(
            checksuites, get_pr_next_check_runs, get_pr_next_checksuites
        )

        # Append old style statuses(like ones populated by CircleCI or EasyCLA) to conclusions
        if orig_last_commit["status"] and orig_last_commit["status"]["contexts"]:
            for status in orig_last_commit["status"]["contexts"]:
                name = status["context"]
                self.conclusions[name] = JobCheckState(
                    name,
                    status["targetUrl"],
                    status["state"],
                    classification=None,
                    job_id=None,
                    title=None,
                    summary=None,
                )

        # Same issue for Claude Code: triggered by comment events, uses a
        # protected environment (bedrock), resulting in ACTION_REQUIRED
        # check suite conclusions that block merges
        self.conclusions.pop("Claude Code", None)

        return self.conclusions

    def get_authors(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns a mapping of author login to "Name <email>" over all commit
        # authors (later commits overwrite earlier ones for the same login).
        rc = {}
        for idx in range(len(self._fetch_authors())):
            rc[self.get_committer_login(idx)] = self.get_committer_author(idx)

        return rc

    def get_author(self) -> str:
        # [FUNCTION DOC] Returns the "Name <email>" of the main PR author: the only author if
        # unique, else the PR creator's entry, else the first commit author.
        authors = self.get_authors()
        if len(authors) == 1:
            return next(iter(authors.values()))
        creator = self.get_pr_creator_login()
        # If PR creator is not among authors
        # Assume it was authored by first commit author
        if creator not in authors:
            return self.get_committer_author(0)
        return authors[creator]

    def get_title(self) -> str:
        # [FUNCTION DOC] Returns the PR title.
        return cast(str, self.info["title"])

    def get_body(self) -> str:
        # [FUNCTION DOC] Returns the PR description text.
        return cast(str, self.info["body"])

    def get_merge_commit(self) -> str | None:
        # [FUNCTION DOC] Returns the merge commit sha if the PR has one, else None.
        mc = self.info["mergeCommit"]
        return mc["oid"] if mc is not None else None

    def get_pr_url(self) -> str:
        # [FUNCTION DOC] Returns the web url of the PR.
        return f"https://github.com/{self.org}/{self.project}/pull/{self.pr_num}"

    @staticmethod
    def _comment_from_node(node: Any) -> GitHubComment:
        # [FUNCTION DOC] HELPER: Converts a GraphQL comment (or review) node into a
        # GitHubComment record; missing createdAt becomes empty, missing author url becomes
        # None, and editor login is None when there is no editor.
        editor = node["editor"]
        return GitHubComment(
            body_text=node["bodyText"],
            created_at=node.get("createdAt", ""),
            author_login=node["author"]["login"],
            author_url=node["author"].get("url", None),
            author_association=node["authorAssociation"],
            editor_login=editor["login"] if editor else None,
            database_id=node["databaseId"],
            url=node["url"],
        )

    def get_comments(self) -> list[GitHubComment]:
        # [FUNCTION DOC] Returns (and caches) all PR comments in chronological order, paging
        # backwards from the prefetched last comments, up to 100 pages.
        if self.comments is not None:
            return self.comments
        self.comments = []
        info = self.info["comments"]
        # Do not try to fetch more than 10K comments
        for _ in range(100):
            self.comments = [
                self._comment_from_node(node) for node in info["nodes"]
            ] + self.comments
            if not info["pageInfo"]["hasPreviousPage"]:
                break
            rc = gh_graphql(
                GH_GET_PR_PREV_COMMENTS,
                name=self.project,
                owner=self.org,
                number=self.pr_num,
                cursor=info["pageInfo"]["startCursor"],
            )
            info = rc["data"]["repository"]["pullRequest"]["comments"]
        return self.comments

    def get_last_comment(self) -> GitHubComment:
        # [FUNCTION DOC] Returns the newest comment from the prefetched comment window.
        return self._comment_from_node(self.info["comments"]["nodes"][-1])

    def get_comment_by_id(self, database_id: int) -> GitHubComment:
        # [FUNCTION DOC] Finds a comment by database id: first in the prefetched window (if
        # comments are not yet fully loaded), then in all comments, then in review bodies.
        # Raises RuntimeError if not found.
        if self.comments is None:
            # Fastpath - try searching in partial prefetched comments
            for node in self.info["comments"]["nodes"]:
                comment = self._comment_from_node(node)
                if comment.database_id == database_id:
                    return comment

        for comment in self.get_comments():
            if comment.database_id == database_id:
                return comment

        # The comment could have actually been a review left on the PR (the message written alongside the review).
        # (This is generally done to trigger the merge right when a comment is left)
        # Check those review comments to see if one of those was the comment in question.
        for node in self.info["reviews"]["nodes"]:
            # These review comments contain all the fields regular comments need
            comment = self._comment_from_node(node)
            if comment.database_id == database_id:
                return comment

        raise RuntimeError(f"Comment with id {database_id} not found")

    def get_diff_revision(self) -> str | None:
        # [FUNCTION DOC] Returns the internal Phabricator diff id (D followed by digits) from
        # the PR body's Differential Revision line, or None.
        rc = RE_DIFF_REV.search(self.get_body())
        return rc.group(1) if rc is not None else None

    def has_internal_changes(self) -> bool:
        # [FUNCTION DOC] True if the PR has a connected internal diff and the Meta
        # internal-only changes check exists but did not succeed.
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        if self.get_diff_revision() is None:
            return False
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].status != "SUCCESS"

    def has_no_connected_diff(self) -> bool:
        # [FUNCTION DOC] True if the internal changes check reports the "no internal diff
        # connected" title.
        checkrun_name = INTERNAL_CHANGES_CHECKRUN_NAME
        checks = self.get_checkrun_conclusions()
        if checks is None or checkrun_name not in checks:
            return False
        return checks[checkrun_name].title == HAS_NO_CONNECTED_DIFF_TITLE

    def merge_ghstack_into(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool,
        comment_id: int | None = None,
        skip_all_rule_checks: bool = False,
        ghstack_prs: list[tuple[GitHubPR, str]] | None = None,
    ) -> list[GitHubPR]:
        # [FUNCTION DOC]
        # Cherry-picks every open PR of the ghstack (bottom to top) onto the current local
        # branch, amending each commit message with the generated message (ghstack info
        # filtered, closed PRs listed as dependencies). Non-self PRs are first checked against
        # merge rules unless skip_all_rule_checks; a failure is re-raised with the stacked PR
        # number. Returns the list of open PRs merged. Raises AssertionError if not ghstack.
        # Side effects: local git commits.
        if not self.is_ghstack_pr():
            raise AssertionError(
                f"merge_ghstack_into called on non-ghstack PR #{self.pr_num}"
            )
        if ghstack_prs is None:
            ghstack_prs = get_ghstack_prs(
                repo, self, open_only=False
            )  # raises error if out of sync
        pr_dependencies = []
        for pr, rev in ghstack_prs:
            if pr.is_closed():
                pr_dependencies.append(pr)
                continue

            commit_msg = pr.gen_commit_message(
                filter_ghstack=True, ghstack_deps=pr_dependencies
            )
            if pr.pr_num != self.pr_num and not skip_all_rule_checks:
                try:
                    find_matching_merge_rule(
                        pr,
                        repo,
                        skip_mandatory_checks=skip_mandatory_checks,
                        skip_internal_checks=can_skip_internal_checks(self, comment_id),
                    )
                except MergeRuleFailedError as ex:
                    raise type(ex)(
                        f"Merge rule check failed for stacked PR #{pr.pr_num}:\n\n{ex}",
                        ex.rule,
                    ) from ex
            repo.cherry_pick(rev)
            repo.amend_commit_message(commit_msg)
            pr_dependencies.append(pr)
        return [x for x, _ in ghstack_prs if not x.is_closed()]

    def gen_commit_message(
        self,
        filter_ghstack: bool = False,
        ghstack_deps: list[GitHubPR] | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Builds the squash commit message: title with PR number, body without cc lines (and
        # without the ghstack stack section if filter_ghstack), the PR url, approver profile
        # urls, optional ghstack dependency numbers, and Co-authored-by lines for authors
        # other than the PR creator.
        """Fetches title and body from PR description
        adds reviewed by, pull request resolved and optionally
        filters out ghstack info"""
        # Adding the url here makes it clickable within the Github UI
        approved_by_urls = ", ".join(
            prefix_with_github_url(login) for login in self.get_approved_by()
        )
        # Remove "cc: " line from the message body
        msg_body = re.sub(RE_PR_CC_LINE, "", self.get_body())
        if filter_ghstack:
            msg_body = re.sub(RE_GHSTACK_DESC, "", msg_body)
        msg = self.get_title() + f" (#{self.pr_num})\n\n"
        msg += msg_body

        msg += f"\nPull Request resolved: {self.get_pr_url()}\n"
        msg += f"Approved by: {approved_by_urls}\n"
        if ghstack_deps:
            msg += f"ghstack dependencies: {', '.join([f'#{pr.pr_num}' for pr in ghstack_deps])}\n"

        # Mention PR co-authors, which should be at the end of the message
        # And separated from the body by two newlines
        first_coauthor = True
        for author_login, author_name in self.get_authors().items():
            if author_login != self.get_pr_creator_login():
                if first_coauthor:
                    msg, first_coauthor = (msg + "\n", False)
                msg += f"\nCo-authored-by: {author_name}"

        return msg

    def add_numbered_label(self, label_base: str, dry_run: bool) -> None:
        # [FUNCTION DOC] Adds a label named from label_base; if cached labels already contain
        # the base, the new name gets an X plus running count suffix. Side effect: label API
        # call (skipped when dry_run).
        labels = self.get_labels() if self.labels is not None else []
        full_label = label_base
        count = 0
        for label in labels:
            if label_base in label:
                count += 1
                full_label = f"{label_base}X{count}"
        self.add_label(full_label, dry_run)

    def add_label(self, label: str, dry_run: bool) -> None:
        # [FUNCTION DOC] Adds one label to the PR through the API (no-op when dry_run).
        gh_add_labels(self.org, self.project, self.pr_num, [label], dry_run)

    def merge_into(
        self,
        repo: GitRepo,
        *,
        skip_mandatory_checks: bool = False,
        dry_run: bool = False,
        comment_id: int,
        ignore_current_checks: list[str] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Full merge flow: finds a matching merge rule (raises if none), gathers the ghstack
        # PRs, verifies docker image builds are ready for docker-affecting changes, then merges
        # (dependabot via the GitHub API, otherwise local squash / cherry-pick and push to the
        # default branch). Afterwards adds the merged label, uploads a merge record when a
        # comment id exists, sleeps 60 seconds, and closes still-open merged PRs manually.
        # Side effects: git push, labels, record upload, PR state change (dry_run limits some).
        # Raises exception if matching rule is not found
        (
            merge_rule,
            pending_checks,
            failed_checks,
            ignorable_checks,
        ) = find_matching_merge_rule(
            self,
            repo,
            skip_mandatory_checks=skip_mandatory_checks,
            skip_internal_checks=can_skip_internal_checks(self, comment_id),
            ignore_current_checks=ignore_current_checks,
        )
        ghstack_prs: list[tuple[GitHubPR, str]] | None = None
        prs_to_merge = [self]
        if self.is_ghstack_pr():
            ghstack_prs = get_ghstack_prs(repo, self, open_only=False)
            prs_to_merge = [pr for pr, _ in ghstack_prs if not pr.is_closed()]

        # A ghstack merge lands all open PRs below this one. Use the topmost
        # docker-affecting PR because its cumulative head has the final docker
        # tree for which images must have been built. Enforced even on force
        # merges.
        docker_pr = get_topmost_docker_pr(prs_to_merge)
        if docker_pr is not None:
            check_docker_builds_ready(docker_pr)

        # Dependabot commits are authored/signed by the bot; merge them through
        # GitHub's squash+merge API so that signature is preserved and dependabot
        # can track the merge, instead of re-authoring a squash commit locally.
        if self.is_dependabot_pr():
            additional_merged_prs: list[GitHubPR] = []
            merge_commit_sha = self.merge_via_github_api(dry_run)
        else:
            additional_merged_prs = self.merge_changes_locally(
                repo,
                skip_mandatory_checks,
                comment_id,
                ghstack_prs=ghstack_prs,
            )

            # Log, but do not block on, a docker land race.
            if docker_pr is not None:
                warn_on_docker_merge_skew(repo, docker_pr)

            repo.push(self.default_branch(), dry_run)
            # When the merge process reaches this part, we can assume that the
            # commit has been successfully pushed to trunk
            merge_commit_sha = repo.rev_parse(name=self.default_branch())
        if not dry_run:
            self.add_numbered_label(MERGE_COMPLETE_LABEL, dry_run)
            for pr in additional_merged_prs:
                pr.add_numbered_label(MERGE_COMPLETE_LABEL, dry_run)

        if comment_id and self.pr_num:
            # Finally, upload the record to s3. The list of pending and failed
            # checks are at the time of the merge
            save_merge_record(
                comment_id=comment_id,
                pr_num=self.pr_num,
                owner=self.org,
                project=self.project,
                author=self.get_author(),
                pending_checks=pending_checks,
                failed_checks=failed_checks,
                ignore_current_checks=ignorable_checks.get("IGNORE_CURRENT_CHECK", []),
                broken_trunk_checks=ignorable_checks.get("BROKEN_TRUNK", []),
                flaky_checks=ignorable_checks.get("FLAKY", []),
                unstable_checks=ignorable_checks.get("UNSTABLE", []),
                last_commit_sha=self.last_commit_sha(default=""),
                merge_base_sha=self.get_merge_base(),
                merge_commit_sha=merge_commit_sha,
                is_failed=False,
                skip_mandatory_checks=skip_mandatory_checks,
                ignore_current=bool(ignore_current_checks),
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to s3")

        # Usually Github will see that the commit has "resolves <pr_num>" in the
        # commit message and close the PR, but sometimes it doesn't, leading to
        # confusion.  When it doesn't, we close it manually.
        time.sleep(60)  # Give Github some time to close the PR
        manually_close_merged_pr(
            pr=self,
            additional_merged_prs=additional_merged_prs,
            merge_commit_sha=merge_commit_sha,
            dry_run=dry_run,
        )

    def merge_via_github_api(self, dry_run: bool = False) -> str:
        # [FUNCTION DOC]
        # Squash merges through the GitHub merge API, splitting the generated commit message
        # into title and body and pinning to the last commit sha. Returns the merge commit sha.
        """Squash-and-merge this PR through GitHub's merge API, pinned to the
        commit that was reviewed, and return the resulting merge commit sha."""
        msg = self.gen_commit_message()
        title, _, body = msg.partition("\n\n")
        return gh_merge_pr(
            self.org,
            self.project,
            self.pr_num,
            merge_method="squash",
            commit_title=title,
            commit_message=body,
            sha=self.last_commit_sha(),
            dry_run=dry_run,
        )

    def merge_changes_locally(
        self,
        repo: GitRepo,
        skip_mandatory_checks: bool = False,
        comment_id: int | None = None,
        branch: str | None = None,
        skip_all_rule_checks: bool = False,
        ghstack_prs: list[tuple[GitHubPR, str]] | None = None,
    ) -> list[GitHubPR]:
        # [FUNCTION DOC]
        # Applies the PR onto the local target branch (default branch unless given). Ghstack
        # PRs delegate to merge_ghstack_into and return the merged PRs. For regular PRs a
        # comment_id is required (ValueError otherwise); the head commit at that comment time
        # must equal the current PR head (RuntimeError otherwise); the commit is fetched,
        # squash merged and committed with the PR author, then the PR is re-fetched and
        # RuntimeError is raised if it changed. Returns an empty list for regular PRs.
        """
        :param skip_all_rule_checks: If true, skips all rule checks on ghstack PRs, useful for dry-running merge locally
        """
        branch_to_merge_into = self.default_branch() if branch is None else branch
        if repo.current_branch() != branch_to_merge_into:
            repo.checkout(branch_to_merge_into)

        # It's okay to skip the commit SHA check for ghstack PRs since
        # authoring requires write access to the repo.
        if self.is_ghstack_pr():
            return self.merge_ghstack_into(
                repo,
                skip_mandatory_checks,
                comment_id=comment_id,
                skip_all_rule_checks=skip_all_rule_checks,
                ghstack_prs=ghstack_prs,
            )

        msg = self.gen_commit_message()
        pr_branch_name = f"__pull-request-{self.pr_num}__init__"

        # Determine which commit SHA to merge
        commit_to_merge = None
        if not comment_id:
            raise ValueError("Must provide --comment-id when merging regular PRs")

        # Get the commit SHA that was present when the comment was made
        commit_to_merge = self.get_commit_sha_at_comment(comment_id)
        if not commit_to_merge:
            raise RuntimeError(
                f"Could not find commit that was pushed before comment {comment_id}"
            )

        # Validate that this commit is the latest commit on the PR
        latest_commit = self.last_commit_sha()
        if commit_to_merge != latest_commit:
            raise RuntimeError(
                f"Commit {commit_to_merge} was HEAD when comment {comment_id} was posted "
                f"but now the latest commit on the PR is {latest_commit}. "
                f"Please re-issue the merge command to merge the latest commit."
            )

        print(f"Merging commit {commit_to_merge} locally")

        repo.fetch(commit_to_merge, pr_branch_name)
        repo._run_git("merge", "--squash", pr_branch_name)
        repo._run_git("commit", f'--author="{self.get_author()}"', "-m", msg)

        # Did the PR change since we started the merge?
        pulled_sha = repo.show_ref(pr_branch_name)
        latest_pr_status = GitHubPR(self.org, self.project, self.pr_num)
        if (
            pulled_sha != latest_pr_status.last_commit_sha()
            or pulled_sha != commit_to_merge
        ):
            raise RuntimeError(
                "PR has been updated since CI checks last passed. Please rerun the merge command."
            )
        return []


class MergeRuleFailedError(RuntimeError):
    # [CLASS DOC]
    # Error raised when a merge rule check fails; carries the rule that failed (or the
    # closest rule) in the rule attribute, which may be None.
    def __init__(self, message: str, rule: MergeRule | None = None) -> None:
        # [FUNCTION DOC] Initializes the error with its message and stores the optional rule.
        super().__init__(message)
        self.rule = rule


class MandatoryChecksMissingError(MergeRuleFailedError):
    # [CLASS DOC] Merge rule failure specifically due to missing or unfinished mandatory checks.
    pass


class PostCommentError(Exception):
    # [CLASS DOC] Error signalling that posting a comment to GitHub failed (message only).
    pass


@dataclass
class MergeRule:
    # [CLASS DOC]
    # One entry of merge_rules.yaml: rule name, file patterns it covers (a leading dash
    # marks an exclusion), logins allowed to approve, mandatory check names (or None), and
    # whether flaky failures may be ignored (default true).
    name: str
    patterns: list[str]
    approved_by: list[str]
    mandatory_checks_name: list[str] | None
    ignore_flaky_failures: bool = True


def gen_new_issue_link(
    org: str, project: str, labels: list[str], template: str = "bug-report.yml"
) -> str:
    # [FUNCTION DOC] HELPER: Builds a prefilled new-issue url for the repo with url-quoted
    # comma-joined labels and the issue template name.
    labels_str = ",".join(labels)
    return (
        f"https://github.com/{org}/{project}/issues/new?"
        f"labels={urllib.parse.quote(labels_str)}&"
        f"template={urllib.parse.quote(template)}"
    )


def read_merge_rules(repo: GitRepo | None, org: str, project: str) -> list[MergeRule]:
    # [FUNCTION DOC]
    # Loads merge rules as MergeRule objects. Without a repo, downloads and decodes the rules
    # file through the GitHub contents API; with a repo, reads it from the checkout (prints
    # a note and returns an empty list if the file is absent). Raises on bad YAML or fields.
    """Returns the list of all merge rules for the repo or project.

    NB: this function is used in Meta-internal workflows, see the comment
    at the top of this file for details.
    """
    repo_relative_rules_path = MERGE_RULE_PATH
    if repo is None:
        json_data = gh_fetch_url(
            f"https://api.github.com/repos/{org}/{project}/contents/{repo_relative_rules_path}",
            headers={"Accept": "application/vnd.github.v3+json"},
            reader=json.load,
        )
        content = base64.b64decode(json_data["content"])
        return [MergeRule(**x) for x in yaml.safe_load(content)]
    else:
        rules_path = Path(repo.repo_dir) / repo_relative_rules_path
        if not rules_path.exists():
            print(f"{rules_path} does not exist, returning empty rules")
            return []
        with open(rules_path) as fp:
            rc = yaml.safe_load(fp)
        return [MergeRule(**x) for x in rc]


def _find_non_matching_files(patterns: list[str], files: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns files not covered by the rule patterns, i.e. not
    # matching any positive pattern or matching an exclusion (dash-prefixed) pattern.
    """Return files that do not match the given patterns.

    Patterns prefixed with '-' are exclusions: a file matches the rule if it
    matches at least one positive pattern AND does not match any negative one.
    """
    positive = [p for p in patterns if not p.startswith("-")]
    negative = [p[1:] for p in patterns if p.startswith("-")]
    patterns_re = patterns_to_regex(positive)
    exclude_re = patterns_to_regex(negative) if negative else None
    return [
        f
        for f in files
        if not patterns_re.match(f) or (exclude_re is not None and exclude_re.match(f))
    ]


def find_matching_merge_rule(
    pr: GitHubPR,
    repo: GitRepo | None = None,
    skip_mandatory_checks: bool = False,
    skip_internal_checks: bool = False,
    ignore_current_checks: list[str] | None = None,
) -> tuple[
    MergeRule,
    list[tuple[str, str | None, int | None]],
    list[tuple[str, str | None, int | None]],
    dict[str, list[Any]],
]:
    # [FUNCTION DOC]
    # Finds the first merge rule the PR satisfies and returns (rule, pending checks, failed
    # checks, ignorable checks by category). A rule passes when all changed files match its
    # patterns, an allowed approver (users or org/team members) approved, mandatory checks
    # (all, or only CLA checks when skip_mandatory_checks) are neither failed nor pending, and
    # internal changes are absent unless skip_internal_checks. Raises RuntimeError when no
    # rules exist or internal changes block; otherwise raises MandatoryChecksMissingError
    # (best score is pending checks) or MergeRuleFailedError with the most relevant reason.
    # Network calls via the PR object, rule loading, and classification.
    """
    Returns merge rule matching to this pr together with the list of associated pending
    and failing jobs OR raises an exception.

    NB: this function is used in Meta-internal workflows, see the comment at the top of
    this file for details.
    """
    changed_files = pr.get_changed_files()
    approved_by = set(pr.get_approved_by())

    issue_link = gen_new_issue_link(
        org=pr.org,
        project=pr.project,
        labels=["module: ci"],
    )
    reject_reason = f"No rule found to match PR. Please [report]{issue_link} this issue to DevX team."

    rules = read_merge_rules(repo, pr.org, pr.project)
    if not rules:
        reject_reason = f"Rejecting the merge as no rules are defined for the repository in {MERGE_RULE_PATH}"
        raise RuntimeError(reject_reason)

    checks = pr.get_checkrun_conclusions()
    checks = get_classifications(
        pr.pr_num,
        pr.project,
        checks,
        ignore_current_checks=ignore_current_checks,
    )

    # This keeps the list of all approvers that could stamp the change
    all_rule_approvers = {}

    # PRs can fail multiple merge rules, but it only needs to pass one rule to be approved.
    # If it fails all rules, we need to find the rule that it came closest to passing and report
    # that to the dev.
    #
    # reject_reason_score ranks rules by relevancy. The higher the score, the more relevant the
    # rule & rejection reason, and we only care about the most relevant rule/reason
    #
    # reject_reason_score intrepretation:
    # Score 0 to 10K - how many files rule matched
    # Score 10K - matched all files, but no overlapping approvers
    # Score 20K - matched all files and approvers, but mandatory checks are pending
    # Score 30k - Matched all files and approvers, but mandatory checks failed
    reject_reason_score = 0
    for rule in rules:
        rule_name = rule.name
        non_matching_files = _find_non_matching_files(rule.patterns, changed_files)
        if len(non_matching_files) > 0:
            num_matching_files = len(changed_files) - len(non_matching_files)
            if num_matching_files > reject_reason_score:
                reject_reason_score = num_matching_files
                reject_reason = "\n".join(
                    (
                        f"Not all files match rule `{rule_name}`.",
                        f"{num_matching_files} files matched, but there are still non-matching files:",
                        f"{','.join(non_matching_files[:5])}{', ...' if len(non_matching_files) > 5 else ''}",
                    )
                )
            continue

        # If rule needs approvers but PR has not been reviewed, skip it
        if len(rule.approved_by) > 0 and len(approved_by) == 0:
            if reject_reason_score < 10000:
                reject_reason_score = 10000
                reject_reason = f"PR #{pr.pr_num} has not been reviewed yet"
            continue

        # Does the PR have the required approvals for this rule?
        rule_approvers = set()
        for approver in rule.approved_by:
            if "/" in approver:
                org, name = approver.split("/")
                rule_approvers.update(gh_get_team_members(org, name))
            else:
                rule_approvers.add(approver)
        approvers_intersection = approved_by.intersection(rule_approvers)
        # If rule requires approvers but they aren't the ones that reviewed PR
        if len(approvers_intersection) == 0 and len(rule_approvers) > 0:
            # Less than or equal is intentionally used here to gather all potential
            # approvers
            if reject_reason_score <= 10000:
                reject_reason_score = 10000

                all_rule_approvers[rule.name] = rule.approved_by
                # Prepare the reject reason
                all_rule_approvers_msg = [
                    f"- {name} ({', '.join(approved_by[:5])}{', ...' if len(approved_by) > 5 else ''})"
                    for name, approved_by in all_rule_approvers.items()
                ]

                reject_reason = "Approvers from one of the following sets are needed:\n"
                reject_reason += "\n".join(all_rule_approvers_msg)

            continue

        # Does the PR pass the checks required by this rule?
        mandatory_checks = (
            rule.mandatory_checks_name if rule.mandatory_checks_name is not None else []
        )
        required_checks = list(
            filter(
                lambda x: ("EasyCLA" in x)
                or ("Facebook CLA Check" in x)
                or not skip_mandatory_checks,
                mandatory_checks,
            )
        )
        pending_checks, failed_checks, _ = categorize_checks(
            checks,
            required_checks,
            ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD
            if rule.ignore_flaky_failures
            else 0,
        )

        # categorize_checks assumes all tests are required if required_checks is empty.
        # this is a workaround as we want to keep that behavior for categorize_checks
        # generally.
        if not required_checks:
            pending_checks = []
            failed_checks = []

        hud_link = f"https://hud.pytorch.org/{pr.org}/{pr.project}/commit/{pr.last_commit_sha()}"
        if len(failed_checks) > 0:
            if reject_reason_score < 30000:
                reject_reason_score = 30000
                reject_reason = "\n".join(
                    (
                        f"{len(failed_checks)} mandatory check(s) failed.  The first few are:",
                        *checks_to_markdown_bullets(failed_checks),
                        "",
                        f"Dig deeper by [viewing the failures on hud]({hud_link})",
                    )
                )
            continue
        elif len(pending_checks) > 0:
            if reject_reason_score < 20000:
                reject_reason_score = 20000
                reject_reason = "\n".join(
                    (
                        f"{len(pending_checks)} mandatory check(s) are pending/not yet run.  The first few are:",
                        *checks_to_markdown_bullets(pending_checks),
                        "",
                        f"Dig deeper by [viewing the pending checks on hud]({hud_link})",
                    )
                )
            continue

        if not skip_internal_checks and pr.has_internal_changes():
            raise RuntimeError(
                "This PR has internal changes and must be landed via Phabricator! Please try reimporting/rexporting the PR!"
            )

        # Categorize all checks when skip_mandatory_checks (force merge) is set. Do it here
        # where the list of checks is readily available. These records will be saved into
        # s3 merge records
        (
            pending_mandatory_checks,
            failed_mandatory_checks,
            ignorable_checks,
        ) = categorize_checks(
            checks,
            [],
            ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD,
        )
        return (
            rule,
            pending_mandatory_checks,
            failed_mandatory_checks,
            ignorable_checks,
        )

    if reject_reason_score == 20000:
        raise MandatoryChecksMissingError(reject_reason, rule)
    raise MergeRuleFailedError(reject_reason, rule)


def get_topmost_docker_pr(prs: list[GitHubPR]) -> GitHubPR | None:
    # [FUNCTION DOC]
    # Returns the last docker-affecting PR in the bottom-to-top list, or None if none are.
    """Find the highest docker-affecting PR in a bottom-to-top stack."""
    return next((pr for pr in reversed(prs) if pr.is_docker_affecting()), None)


def check_docker_builds_ready(pr: GitHubPR) -> None:
    # [FUNCTION DOC]
    # No-op for PRs not touching docker files. Otherwise inspects the docker-builds checks:
    # raises MergeRuleFailedError if none ran or any failed, and MandatoryChecksMissingError
    # (retryable) if any are still pending. Applies even to force merges.
    """Block merge of a docker-affecting PR unless its docker images have been
    pre-built.

    PRs that change .ci/docker must run the docker-builds (ciflow/docker)
    workflow so the images are built and pushed to ECR before landing.  If they
    aren't, every trunk job that needs one of those images fails because it
    can't find the image (see the #190927 / #186302 land race).  This gate is
    enforced even for force merges, since -f is exactly what bypassed it before.
    """
    if not pr.is_docker_affecting():
        return

    docker_checks = get_docker_build_checks(pr.get_checkrun_conclusions())

    if not docker_checks:
        raise MergeRuleFailedError(
            f"This PR changes files under {DOCKER_CI_PATH}/, but the "
            f"`{DOCKER_BUILDS_WORKFLOW_NAME}` workflow has not run on it. The CI "
            "docker images must be pre-built and pushed to ECR before this lands, "
            "otherwise trunk jobs will not be able to find the image they need. "
            "Please add the `ciflow/docker` label to this PR, wait for the docker "
            "builds to finish, and then re-issue the merge command."
        )

    pending = sorted(name for name, c in docker_checks.items() if c.status is None)
    failed = sorted(
        name
        for name, c in docker_checks.items()
        if c.status is not None and not is_passing_status(c.status)
    )

    if pending:
        # Raise MandatoryChecksMissingError so that a normal (non-force) merge
        # keeps retrying until the docker builds finish, mirroring how other
        # mandatory checks are waited on.
        raise MandatoryChecksMissingError(
            f"This PR changes files under {DOCKER_CI_PATH}/, so the "
            f"`{DOCKER_BUILDS_WORKFLOW_NAME}` builds must finish before merging. "
            f"Still waiting for {len(pending)} docker build job(s), the first few "
            f"are: {', '.join(pending[:5])}"
        )

    if failed:
        raise MergeRuleFailedError(
            f"This PR changes files under {DOCKER_CI_PATH}/, so the "
            f"`{DOCKER_BUILDS_WORKFLOW_NAME}` builds must all pass before merging, "
            f"but {len(failed)} of them failed, the first few are: "
            f"{', '.join(failed[:5])}. The docker images could not be built, so "
            "trunk jobs would be unable to find them. Please fix the docker build "
            "and re-run `ciflow/docker` before merging."
        )


def warn_on_docker_merge_skew(repo: GitRepo, pr: GitHubPR) -> None:
    # [FUNCTION DOC]
    # For docker-affecting PRs, compares the docker directory tree hash of the local merge
    # commit (HEAD) with that of the PR head (fetched first) and prints a warning if they
    # differ. Never blocks the merge. Errors from git lookups propagate.
    """Log when a docker-affecting PR raced with another docker change.

    The CI docker images are tagged by the git tree hash of .ci/docker, so a
    docker change landing on the base after this PR's images were built leaves
    the merge commit asking for an untested tree. This used to refuse the merge
    (#191508); it now only warns, since docker-builds.yml publishes the merged
    tree's images on push and jobs wait for them.

    Must be called after the merge commit has been created locally.
    """
    if not pr.is_docker_affecting():
        return

    # HEAD is the freshly created (squash/cherry-picked) merge commit.
    # A wholesale deletion of the directory intentionally fails this lookup.
    merge_commit_tree = repo.rev_parse(f"HEAD:{DOCKER_CI_PATH}")

    # Compare against the tree that CI actually built and tested on the PR head.
    pr_head_sha = pr.last_commit_sha()
    # The PR head commit may not be present locally (for example, for a fork
    # PR), so make sure we have the object before reading its tree.
    repo.fetch(pr_head_sha)
    pr_head_tree = repo.rev_parse(f"{pr_head_sha}:{DOCKER_CI_PATH}")

    if merge_commit_tree != pr_head_tree:
        print(
            f"WARNING: docker land race on PR #{pr.pr_num}: the {DOCKER_CI_PATH} "
            f"tree of the merge commit ({merge_commit_tree}) does not match the "
            f"tree that ciflow/docker built and tested on the PR head "
            f"({pr_head_tree}). Another docker-affecting change landed on the base "
            "branch after these images were built, so the first trunk jobs after "
            "this merge may wait for docker-builds to publish the merged tree's "
            "images, or rebuild them locally. Merging anyway."
        )


def checks_to_str(checks: list[tuple[str, str | None]]) -> str:
    # [FUNCTION DOC] HELPER: Renders (name, url) pairs as a comma separated string of markdown
    # links, or plain names when the url is None.
    return ", ".join(f"[{c[0]}]({c[1]})" if c[1] is not None else c[0] for c in checks)


def checks_to_markdown_bullets(
    checks: list[tuple[str, str | None, int | None]],
) -> list[str]:
    # [FUNCTION DOC] HELPER: Renders at most the first five (name, url, id) checks as markdown
    # bullet lines, linking the url when present.
    return [
        f"- [{c[0]}]({c[1]})" if c[1] is not None else f"- {c[0]}" for c in checks[:5]
    ]


def post_starting_merge_comment(
    repo: GitRepo,
    pr: GitHubPR,
    explainer: TryMergeExplainer,
    dry_run: bool,
    ignore_current_checks_info: list[tuple[str, str | None, int | None]] | None = None,
) -> None:
    # [FUNCTION DOC]
    # Posts the "merge started" explainer comment on the PR and, for ghstack PRs, a short
    # "starting merge as part of stack" comment on each other open PR in the stack.
    # Side effects: PR comments (skipped when dry_run).
    """Post the initial merge starting message on the PR. Also post a short
    message on all PRs in the stack."""
    gh_post_pr_comment(
        pr.org,
        pr.project,
        pr.pr_num,
        explainer.get_merge_message(ignore_current_checks_info),
        dry_run=dry_run,
    )
    if pr.is_ghstack_pr():
        for additional_prs, _ in get_ghstack_prs(repo, pr):
            if additional_prs.pr_num != pr.pr_num:
                gh_post_pr_comment(
                    additional_prs.org,
                    additional_prs.project,
                    additional_prs.pr_num,
                    f"Starting merge as part of PR stack under #{pr.pr_num}",
                    dry_run=dry_run,
                )


def manually_close_merged_pr(
    pr: GitHubPR,
    additional_merged_prs: list[GitHubPR],
    merge_commit_sha: str,
    dry_run: bool,
) -> None:
    # [FUNCTION DOC]
    # Safety net for when GitHub leaves a merged PR open: for the PR and each additional
    # stacked PR, refreshes its state and, if still open, comments explaining the merge commit
    # and closes it. Prints a completion line. Side effects: comments and closing (dry_run aware).
    def _comment_and_close(pr: GitHubPR, comment: str) -> None:
        # [FUNCTION DOC] HELPER: Re-fetches the PR and, if still open, posts the comment and
        # closes it.
        pr = GitHubPR(pr.org, pr.project, pr.pr_num)  # Refresh the PR
        if not pr.is_closed():
            gh_post_pr_comment(pr.org, pr.project, pr.pr_num, comment, dry_run)
            gh_close_pr(pr.org, pr.project, pr.pr_num, dry_run)

    message = (
        f"This PR (#{pr.pr_num}) was merged in {merge_commit_sha} but it is still open, likely due to a Github bug, "
        "so mergebot is closing it manually.  If you think this is a mistake, please feel free to reopen and contact Dev Infra."
    )
    _comment_and_close(pr, message)
    for additional_pr in additional_merged_prs:
        message = (
            f"This PR (#{additional_pr.pr_num}) was merged as part of PR #{pr.pr_num} in the stack under {merge_commit_sha} "
            "but it is still open, likely due to a Github bug, so mergebot is closing it manually. "
            "If you think this is a mistake, please feel free to reopen and contact Dev Infra."
        )
        _comment_and_close(additional_pr, message)

    print(f"PR {pr.pr_num} and all additional PRs in the stack have been closed.")


@retries_decorator()
def save_merge_record(
    comment_id: int,
    pr_num: int,
    owner: str,
    project: str,
    author: str,
    pending_checks: list[tuple[str, str | None, int | None]],
    failed_checks: list[tuple[str, str | None, int | None]],
    ignore_current_checks: list[tuple[str, str | None, int | None]],
    broken_trunk_checks: list[tuple[str, str | None, int | None]],
    flaky_checks: list[tuple[str, str | None, int | None]],
    unstable_checks: list[tuple[str, str | None, int | None]],
    last_commit_sha: str,
    merge_base_sha: str,
    merge_commit_sha: str = "",
    is_failed: bool = False,
    skip_mandatory_checks: bool = False,
    ignore_current: bool = False,
    error: str = "",
) -> None:
    # [FUNCTION DOC]
    # Writes a one-element JSON list describing the merge attempt (ids, author, check
    # categories, shas, flags, error text, and a dedup id built from project, PR, comment and
    # the GITHUB_RUN_ID env var) to merge_record.json at the repository root, for a later
    # upload step. Retried on failure by the decorator.
    # KNOWN ISSUE: the docstring and caller comments speak of uploading to s3, but this only
    # writes a local file.
    """
    This saves the merge records as a json, which can later be uploaded to s3
    """

    # Prepare the record to be written into s3
    data = [
        {
            "comment_id": comment_id,
            "pr_num": pr_num,
            "owner": owner,
            "project": project,
            "author": author,
            "pending_checks": pending_checks,
            "failed_checks": failed_checks,
            "ignore_current_checks": ignore_current_checks,
            "broken_trunk_checks": broken_trunk_checks,
            "flaky_checks": flaky_checks,
            "unstable_checks": unstable_checks,
            "last_commit_sha": last_commit_sha,
            "merge_base_sha": merge_base_sha,
            "merge_commit_sha": merge_commit_sha,
            "is_failed": is_failed,
            "skip_mandatory_checks": skip_mandatory_checks,
            "ignore_current": ignore_current,
            "error": error,
            # This is a unique identifier for the record for deduping purposes
            # in Rockset.  Any unique string would work.  This will not be used
            # after we migrate off Rockset
            "_id": f"{project}-{pr_num}-{comment_id}-{os.environ.get('GITHUB_RUN_ID')}",
        }
    ]
    repo_root = Path(__file__).resolve().parent.parent.parent

    with open(repo_root / "merge_record.json", "w") as f:
        json.dump(data, f)


@retries_decorator()
def get_drci_classifications(pr_num: int, project: str = "pytorch") -> Any:
    # [FUNCTION DOC]
    # Calls the HUD Dr.CI API (POST, auth keys from env vars DRCI_BOT_KEY and HUD_API_TOKEN)
    # and returns the failure classification mapping for the PR (category -> list of jobs), or
    # an empty mapping if none. Retried by the decorator. Network call.
    """
    Query HUD API to find similar failures to decide if they are flaky
    """
    # NB: This doesn't work internally atm because this requires making an
    # external API call to HUD
    failures = gh_fetch_url(
        f"https://hud.pytorch.org/api/drci/drci?prNumber={pr_num}",
        data=f"repo={project}",
        headers={
            "Authorization": os.getenv("DRCI_BOT_KEY", ""),
            "Accept": "application/vnd.github.v3+json",
            "x-hud-internal-bot": os.getenv("HUD_API_TOKEN", ""),
        },
        method="POST",
        reader=json.load,
    )

    return failures.get(str(pr_num), {}) if failures else {}


REMOVE_JOB_NAME_SUFFIX_REGEX = re.compile(r", [0-9]+, [0-9]+, .+\)$")


def remove_job_name_suffix(name: str, replacement: str = ")") -> str:
    # [FUNCTION DOC] HELPER: Replaces a trailing shard suffix (comma, two numbers, comma,
    # rest, closing paren) in a job name with the replacement text.
    return re.sub(REMOVE_JOB_NAME_SUFFIX_REGEX, replacement, name)


def is_broken_trunk(
    check: JobCheckState,
    drci_classifications: Any,
) -> bool:
    # [FUNCTION DOC] True if Dr.CI lists the check (by name or job id) under broken trunk
    # failures. False when the check or classifications are empty.
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    # Consult the list of broken trunk failures from Dr.CI
    return any(
        (name == broken_trunk["name"]) or (job_id and job_id == broken_trunk["id"])
        for broken_trunk in drci_classifications.get("BROKEN_TRUNK", [])
    )


def is_unstable(
    check: JobCheckState,
    drci_classifications: Any,
) -> bool:
    # [FUNCTION DOC] True if the job name contains the unstable keyword or Dr.CI lists the
    # check (by name or job id) as unstable. False when the check or classifications are empty.
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    # The job name has the unstable keyword. This is the original way to mark a job
    # as unstable on HUD, Dr.CI, and trymerge
    if "unstable" in name:
        return True

    # Consult the list of unstable failures from Dr.CI
    return any(
        (name == unstable["name"] or (job_id and job_id == unstable["id"]))
        for unstable in drci_classifications.get("UNSTABLE", [])
    )


def is_flaky(
    check: JobCheckState,
    drci_classifications: Any,
) -> bool:
    # [FUNCTION DOC] True if Dr.CI lists the check (by name or job id) under flaky failures.
    # False when the check or classifications are empty.
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    # Consult the list of flaky failures from Dr.CI
    return any(
        (name == flaky["name"] or (job_id and job_id == flaky["id"]))
        for flaky in drci_classifications.get("FLAKY", [])
    )


def is_invalid_cancel(
    name: str,
    conclusion: str | None,
    drci_classifications: Any,
) -> bool:
    # [FUNCTION DOC]
    # True if the job concluded as cancelled and Dr.CI lists it neither as a failure nor as
    # an unknown-status failure, meaning the cancel is an invalid signal to ignore. False when
    # any input is empty or the conclusion is not cancelled.
    """
    After https://github.com/pytorch/test-infra/pull/4579, invalid cancelled
    signals have been removed from HUD and Dr.CI. The same needs to be done
    here for consistency
    """
    if (
        not name
        or not drci_classifications
        or not conclusion
        or conclusion.upper() != "CANCELLED"
    ):
        return False

    # If a job is cancelled and not listed as a failure or unclassified failure
    # by Dr.CI, it's an invalid signal and can be ignored. UNKNOWN covers jobs
    # whose workflow did not run on the merge base, so Dr.CI has no signal to
    # tell whether the failure is pre-existing -- those should still block merge.
    return all(
        name != failure["name"]
        for failure in drci_classifications.get("FAILED", [])
        + drci_classifications.get("UNKNOWN", [])
    )


def is_crcr_l3(check: JobCheckState, drci_classifications: Any) -> bool:
    # [FUNCTION DOC]
    # True if Dr.CI lists the check (by name or job id) under the cross-repo CI L3 category,
    # whose failures do not block merging. False for empty inputs.
    """Return True if this check is a non-blocking CRCR L3 failure.

    Dr.CI is the classification authority. A check is L3 non-blocking when
    Dr.CI returns it under the ``CRCR_L3`` category. CRCR (cross-repo CI
    relay) L3 check runs are named ``crcr/<owner>/<repo>/<workflow>`` and will be
    classified as CRCR_L3 by Dr.CI when their downstream_repo_level is L3.
    """
    if not check or not drci_classifications:
        return False

    name = check.name
    job_id = check.job_id

    return any(
        name == crcr["name"] or (job_id and job_id == crcr["id"])
        for crcr in drci_classifications.get("CRCR_L3", [])
    )


def get_classifications(
    pr_num: int,
    project: str,
    checks: dict[str, JobCheckState],
    ignore_current_checks: list[str] | None,
) -> dict[str, JobCheckState]:
    # [FUNCTION DOC]
    # Returns a copy of checks where each non-successful check gets a classification string:
    # UNSTABLE, BROKEN_TRUNK, FLAKY, INVALID_CANCEL, CRCR_L3 (first match wins, in that order,
    # from Dr.CI data) or IGNORE_CURRENT_CHECK when its name is in ignore_current_checks.
    # Dr.CI data comes from the API, falling back to the Dr.CI check run summary parsed as
    # JSON (invalid JSON warns and yields no classifications). Prints the Dr.CI results.
    # Get the failure classification from Dr.CI, which is the source of truth
    # going forward. It's preferable to try calling Dr.CI API directly first
    # to get the latest results as well as update Dr.CI PR comment
    drci_classifications = get_drci_classifications(pr_num=pr_num, project=project)

    def get_readable_drci_results(drci_classifications: Any) -> str:
        # [FUNCTION DOC] HELPER: Formats the Dr.CI classification mapping as an indented text
        # listing (job id and name per category), or a JSON dump if the shape is unexpected.
        try:
            s = f"From Dr.CI API ({pr_num}):\n"
            for classification, jobs in drci_classifications.items():
                s += f"  {classification}: \n"
                for job in jobs:
                    s += f"    {job['id']} {job['name']}\n"
            return s
        except Exception:
            return f"From Dr.CI API: {json.dumps(drci_classifications)}"

    print(get_readable_drci_results(drci_classifications))

    # NB: if the latest results from Dr.CI is not available, i.e. when calling from
    # SandCastle, we fallback to any results we can find on Dr.CI check run summary
    if (
        not drci_classifications
        and DRCI_CHECKRUN_NAME in checks
        and checks[DRCI_CHECKRUN_NAME]
        and checks[DRCI_CHECKRUN_NAME].summary
    ):
        drci_summary = checks[DRCI_CHECKRUN_NAME].summary
        try:
            print(f"From Dr.CI checkrun summary: {drci_summary}")
            drci_classifications = json.loads(str(drci_summary))
        except json.JSONDecodeError:
            warn("Invalid Dr.CI checkrun summary")
            drci_classifications = {}

    checks_with_classifications = checks.copy()
    for name, check in checks.items():
        if check.status == "SUCCESS" or check.status == "NEUTRAL":
            continue

        if is_unstable(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "UNSTABLE",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        # NB: It's important to note that when it comes to ghstack and broken trunk classification,
        # Dr.CI uses the base of the whole stack
        if is_broken_trunk(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "BROKEN_TRUNK",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        elif is_flaky(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "FLAKY",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        elif is_invalid_cancel(name, check.status, drci_classifications):
            # NB: Create a new category here for invalid cancelled signals because
            # there are usually many of them when they happen. So, they shouldn't
            # be counted toward ignorable failures threshold
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "INVALID_CANCEL",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        elif is_crcr_l3(check, drci_classifications):
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "CRCR_L3",
                check.job_id,
                check.title,
                check.summary,
            )
            continue

        if ignore_current_checks is not None and name in ignore_current_checks:
            checks_with_classifications[name] = JobCheckState(
                check.name,
                check.url,
                check.status,
                "IGNORE_CURRENT_CHECK",
                check.job_id,
                check.title,
                check.summary,
            )

    return checks_with_classifications


def filter_checks_with_lambda(
    checks: JobNameToStateDict, status_filter: Callable[[str | None], bool]
) -> list[JobCheckState]:
    # [FUNCTION DOC] HELPER: Returns the checks whose status satisfies the predicate.
    return [check for check in checks.values() if status_filter(check.status)]


def get_pr_commit_sha(repo: GitRepo, pr: GitHubPR) -> str:
    # [FUNCTION DOC] Returns the PR merge commit sha from GitHub, or else the first local
    # commit resolving the PR. Raises PostCommentError if none is found.
    commit_sha = pr.get_merge_commit()
    if commit_sha is not None:
        return commit_sha
    commits = repo.commits_resolving_gh_pr(pr.pr_num)
    if len(commits) == 0:
        raise PostCommentError("Can't find any commits resolving PR")
    return commits[0]


def validate_revert(
    repo: GitRepo, pr: GitHubPR, *, comment_id: int | None = None
) -> tuple[str, str]:
    # [FUNCTION DOC]
    # Validates a revert request: the revert comment (given id or the last comment) must be
    # unedited and written by an allowed author association (collaborator, member, owner, plus
    # contributor for private repos, or one of a few GitHub Apps). Returns (author login,
    # commit sha to revert). Raises PostCommentError with a user-facing message on failure.
    comment = (
        pr.get_last_comment()
        if comment_id is None
        else pr.get_comment_by_id(comment_id)
    )
    if comment.editor_login is not None:
        raise PostCommentError(
            "Halting the revert as the revert comment has been edited."
        )
    author_association = comment.author_association
    author_login = comment.author_login
    allowed_reverters = ["COLLABORATOR", "MEMBER", "OWNER"]
    # For some reason, one can not be a member of private repo, only CONTRIBUTOR
    if pr.is_base_repo_private():
        allowed_reverters.append("CONTRIBUTOR")
    # Special case GitHub Apps that don't have a repo association
    # but should be able to issue revert commands
    allowed_apps = {
        "https://github.com/apps/pytorch-auto-revert",
        "https://github.com/apps/facebook-github-tools",
        "https://github.com/apps/meta-codesync",
    }
    if comment.author_url in allowed_apps:
        allowed_reverters.append("NONE")

    if author_association not in allowed_reverters:
        raise PostCommentError(
            f"Will not revert as @{author_login} is not one of "
            f"[{', '.join(allowed_reverters)}], but instead is {author_association}."
        )

    commit_sha = get_pr_commit_sha(repo, pr)
    return (author_login, commit_sha)


def get_ghstack_dependent_prs(
    repo: GitRepo, pr: GitHubPR, only_closed: bool = True
) -> list[tuple[str, GitHubPR]]:
    # [FUNCTION DOC]
    # Returns (merge commit sha, PR) pairs for the PRs in the ghstack from this PR upward.
    # Open PRs are skipped, or included with an empty sha when only_closed is false. Verifies
    # that all branches containing the orig ref are consistent prefixes of the longest
    # revision list. Raises AssertionError for non-ghstack PRs and RuntimeError when no
    # revisions exist or branches diverge.
    """
    Get the PRs in the stack that are above this PR (inclusive).
    Throws error if stack have branched or original branches are gone
    """
    if not pr.is_ghstack_pr():
        raise AssertionError(
            f"get_ghstack_dependent_prs called on non-ghstack PR #{pr.pr_num}"
        )
    orig_ref = f"{repo.remote}/{pr.get_ghstack_orig_ref()}"
    rev_list = repo.revlist(f"{pr.default_branch()}..{orig_ref}")
    if len(rev_list) == 0:
        raise RuntimeError(
            f"PR {pr.pr_num} does not have any revisions associated with it"
        )
    skip_len = len(rev_list) - 1
    for branch in repo.branches_containing_ref(orig_ref):
        candidate = repo.revlist(f"{pr.default_branch()}..{branch}")
        # Pick longest candidate
        if len(candidate) > len(rev_list):
            candidate, rev_list = rev_list, candidate
        # Validate that candidate always ends rev-list
        if rev_list[-len(candidate) :] != candidate:
            raise RuntimeError(
                f"Branch {branch} revlist {', '.join(candidate)} is not a subset of {', '.join(rev_list)}"
            )
    # Remove commits original PR depends on
    if skip_len > 0:
        rev_list = rev_list[:-skip_len]
    rc: list[tuple[str, GitHubPR]] = []
    for pr_, sha in _revlist_to_prs(repo, pr, rev_list):
        if not pr_.is_closed():
            if not only_closed:
                rc.append(("", pr_))
            continue
        commit_sha = get_pr_commit_sha(repo, pr_)
        rc.append((commit_sha, pr_))
    return rc


def do_revert_prs(
    repo: GitRepo,
    original_pr: GitHubPR,
    shas_and_prs: list[tuple[str, GitHubPR]],
    *,
    author_login: str,
    extra_msg: str = "",
    skip_internal_checks: bool = False,
    dry_run: bool = False,
) -> None:
    # [FUNCTION DOC]
    # Reverts the given merged commits on the default branch (one revert commit each, with the
    # PR-resolved line removed and a "Reverted ... on behalf of" note appended), pushes them,
    # then for each PR posts a comment (with an internal-changes warning when applicable),
    # adds the reverted and ci-no-td labels, comments on the commit and updates the PR state.
    # Side effects: git commits and push, PR comments and labels (dry_run limits some).
    # Prepare and push revert commits
    for commit_sha, pr in shas_and_prs:
        revert_msg = f"\nReverted {pr.get_pr_url()} on behalf of {prefix_with_github_url(author_login)}"
        revert_msg += extra_msg
        repo.checkout(pr.default_branch())
        repo.revert(commit_sha)
        msg = repo.commit_message("HEAD")
        msg = re.sub(RE_PULL_REQUEST_RESOLVED, "", msg)
        msg += revert_msg
        repo.amend_commit_message(msg)
    repo.push(shas_and_prs[0][1].default_branch(), dry_run)

    # Comment/reopen PRs
    for commit_sha, pr in shas_and_prs:
        revert_message = ""
        if pr.pr_num == original_pr.pr_num:
            revert_message += (
                f"@{pr.get_pr_creator_login()} your PR has been successfully reverted."
            )
        else:
            revert_message += (
                f"@{pr.get_pr_creator_login()} your PR has been reverted as part of the stack under "
                f"#{original_pr.pr_num}.\n"
            )
        if (
            pr.has_internal_changes()
            and not pr.has_no_connected_diff()
            and not skip_internal_checks
        ):
            revert_message += "\n:warning: This PR might contain internal changes"
            revert_message += "\ncc: @pytorch/pytorch-dev-infra"
        gh_post_pr_comment(
            pr.org, pr.project, pr.pr_num, revert_message, dry_run=dry_run
        )

        pr.add_numbered_label("reverted", dry_run)
        pr.add_label("ci-no-td", dry_run)
        if not dry_run:
            gh_post_commit_comment(pr.org, pr.project, commit_sha, revert_msg)
            gh_update_pr_state(pr.org, pr.project, pr.pr_num)


def try_revert(
    repo: GitRepo,
    pr: GitHubPR,
    *,
    dry_run: bool = False,
    comment_id: int | None = None,
    reason: str | None = None,
) -> None:
    # [FUNCTION DOC]
    # Entry for the revert command: validates the request (posting the error as a PR comment
    # and returning on failure), builds the explanatory message (reason and comment link),
    # expands to dependent stacked PRs for ghstack (falling back to a single revert on any
    # error), and performs the revert. Raises RuntimeError if the stack yields nothing.
    try:
        author_login, commit_sha = validate_revert(repo, pr, comment_id=comment_id)
    except PostCommentError as e:
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, str(e), dry_run=dry_run)
        return

    extra_msg = f" due to {reason}" if reason is not None else ""
    extra_msg += (
        f" ([comment]({pr.get_comment_by_id(comment_id).url}))\n"
        if comment_id is not None
        else "\n"
    )
    shas_and_prs = [(commit_sha, pr)]
    if pr.is_ghstack_pr():
        try:
            shas_and_prs = get_ghstack_dependent_prs(repo, pr)
            prs_to_revert = " ".join([t[1].get_pr_url() for t in shas_and_prs])
            print(f"About to stack of PRs: {prs_to_revert}")
        except Exception as e:
            print(
                f"Failed to fetch dependent PRs: {str(e)}, fall over to single revert"
            )

    if not shas_and_prs:
        raise RuntimeError(
            f"No revertable PRs found in ghstack for #{pr.pr_num}. "
            f"This typically means the PR is still open (not merged) or "
            f"its GitHub state is inconsistent. Only closed/merged PRs can be reverted."
        )

    do_revert_prs(
        repo,
        pr,
        shas_and_prs,
        author_login=author_login,
        extra_msg=extra_msg,
        dry_run=dry_run,
        skip_internal_checks=can_skip_internal_checks(pr, comment_id),
    )


def prefix_with_github_url(suffix_str: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the GitHub site url followed by the given path or login.
    return f"https://github.com/{suffix_str}"


def check_for_sev(org: str, project: str, skip_mandatory_checks: bool) -> None:
    # [FUNCTION DOC]
    # Unless mandatory checks are skipped (force), searches for open issues labeled as both
    # a CI sev and merge blocking in the repo; raises RuntimeError naming the first one found.
    # Network call.
    if skip_mandatory_checks:
        return
    response = cast(
        dict[str, Any],
        gh_fetch_json_list(
            "https://api.github.com/search/issues",  # @lint-ignore
            # Having two label: queries is an AND operation
            params={
                "q": f'repo:{org}/{project} is:open is:issue label:"ci: sev" label:"merge blocking"'
            },
        ),
    )
    if response["total_count"] != 0:
        raise RuntimeError(
            "Not merging any PRs at the moment because there is a "
            + "merge blocking https://github.com/pytorch/pytorch/labels/ci:%20sev issue open at: \n"
            + f"{response['items'][0]['html_url']}"
        )
    return


def has_label(labels: list[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    # [FUNCTION DOC] True if any label matches the regex at its start (default: ciflow labels).
    return len(list(filter(pattern.match, labels))) > 0


def categorize_checks(
    check_runs: JobNameToStateDict,
    required_checks: list[str],
    ok_failed_checks_threshold: int | None = None,
) -> tuple[
    list[tuple[str, str | None, int | None]],
    list[tuple[str, str | None, int | None]],
    dict[str, list[Any]],
]:
    # [FUNCTION DOC]
    # Splits checks into pending and failed lists of (name, url, job id), plus a mapping of
    # ignorable failure categories. Only checks whose name contains a required name are
    # considered (all if none required); missing required checks count as pending; unstable
    # pending jobs and invalid cancels are skipped; failures classified as ignore-current,
    # broken trunk, flaky, unstable or cross-repo L3 are put in the category mapping. If the
    # flaky plus broken-trunk count exceeds ok_failed_checks_threshold (when set) they are
    # also added to failed. Emits a warning listing flaky/broken failures.
    """
    Categories all jobs into the list of pending and failing jobs. All known flaky
    failures and broken trunk are ignored by defaults when ok_failed_checks_threshold
    is not set (unlimited)
    """
    pending_checks: list[tuple[str, str | None, int | None]] = []
    failed_checks: list[tuple[str, str | None, int | None]] = []

    # failed_checks_categorization is used to keep track of all ignorable failures when saving the merge record on s3
    failed_checks_categorization: dict[str, list[Any]] = defaultdict(list)

    # If required_checks is not set or empty, consider all names are relevant
    relevant_checknames = [
        name
        for name in check_runs
        if not required_checks or any(x in name for x in required_checks)
    ]

    for checkname in required_checks:
        if all(checkname not in x for x in check_runs):
            pending_checks.append((checkname, None, None))

    for checkname in relevant_checknames:
        status = check_runs[checkname].status
        url = check_runs[checkname].url
        classification = check_runs[checkname].classification
        job_id = check_runs[checkname].job_id

        if status is None and classification != "UNSTABLE":
            # NB: No need to wait if the job classification is unstable as it would be
            # ignored anyway. This is useful to not need to wait for scarce resources
            # like ROCm, which is also frequently in unstable mode
            pending_checks.append((checkname, url, job_id))
        elif classification == "INVALID_CANCEL":
            continue
        elif not is_passing_status(check_runs[checkname].status):
            target = (
                failed_checks_categorization[classification]
                if classification
                in (
                    "IGNORE_CURRENT_CHECK",
                    "BROKEN_TRUNK",
                    "FLAKY",
                    "UNSTABLE",
                    "CRCR_L3",
                )
                else failed_checks
            )
            target.append((checkname, url, job_id))

    flaky_or_broken_trunk = (
        failed_checks_categorization["BROKEN_TRUNK"]
        + failed_checks_categorization["FLAKY"]
    )

    if flaky_or_broken_trunk:
        warn(
            f"The following {len(flaky_or_broken_trunk)} checks failed but were likely due flakiness or broken trunk: "
            + ", ".join([x[0] for x in flaky_or_broken_trunk])
            + (
                f" but this is greater than the threshold of {ok_failed_checks_threshold} so merge will fail"
                if ok_failed_checks_threshold is not None
                and len(flaky_or_broken_trunk) > ok_failed_checks_threshold
                else ""
            )
        )

    if (
        ok_failed_checks_threshold is not None
        and len(flaky_or_broken_trunk) > ok_failed_checks_threshold
    ):
        failed_checks = failed_checks + flaky_or_broken_trunk

    # The list of failed_checks_categorization is returned so that it can be saved into the s3 merge record
    return (pending_checks, failed_checks, failed_checks_categorization)


def merge(
    pr: GitHubPR,
    repo: GitRepo,
    comment_id: int,
    dry_run: bool = False,
    skip_mandatory_checks: bool = False,
    timeout_minutes: int = 400,
    stale_pr_days: int = 3,
    ignore_current: bool = False,
) -> None:
    # [FUNCTION DOC]
    # Drives a merge request: adds the merging label, checks for blocking sevs, and for force
    # merges posts the start comment and merges immediately. Otherwise verifies approvals and
    # labels, optionally snapshots currently failing checks to ignore, posts the start comment,
    # then polls (retrying every 5 minutes on pending mandatory checks, up to timeout_minutes)
    # re-fetching the PR and merging once rules pass. Raises RuntimeError on new commits pushed,
    # startup failures, failed jobs, or timeout (after adding the land-failed label).
    # KNOWN ISSUE: stale_pr_days is accepted but never used.
    initial_commit_sha = pr.last_commit_sha()
    pr_link = f"https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num}"
    print(f"Attempting merge of {initial_commit_sha} ({pr_link})")

    if MERGE_IN_PROGRESS_LABEL not in pr.get_labels():
        gh_add_labels(pr.org, pr.project, pr.pr_num, [MERGE_IN_PROGRESS_LABEL], dry_run)

    explainer = TryMergeExplainer(
        skip_mandatory_checks,
        pr.get_labels(),
        pr.pr_num,
        pr.org,
        pr.project,
        ignore_current,
    )

    # probably a bad name, but this is a list of current checks that should be
    # ignored and is toggled by the --ignore-current flag
    ignore_current_checks_info = []

    if pr.is_ghstack_pr():
        get_ghstack_prs(repo, pr)  # raises error if out of sync

    check_for_sev(pr.org, pr.project, skip_mandatory_checks)

    if skip_mandatory_checks:
        post_starting_merge_comment(repo, pr, explainer, dry_run)
        return pr.merge_into(
            repo,
            dry_run=dry_run,
            skip_mandatory_checks=skip_mandatory_checks,
            comment_id=comment_id,
        )

    # Check for approvals
    find_matching_merge_rule(pr, repo, skip_mandatory_checks=True)

    ensure_mergeable_labels(pr, comment_id, dry_run)

    if ignore_current:
        checks = pr.get_checkrun_conclusions()
        _, failing, _ = categorize_checks(
            checks,
            list(checks.keys()),
            ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD,
        )
        ignore_current_checks_info = failing

    post_starting_merge_comment(
        repo,
        pr,
        explainer,
        dry_run,
        ignore_current_checks_info=ignore_current_checks_info,
    )

    start_time = time.time()
    last_exception = ""
    elapsed_time = 0.0
    ignore_current_checks = [
        x[0] for x in ignore_current_checks_info
    ]  # convert to List[str] for convenience
    while elapsed_time < timeout_minutes * 60:
        check_for_sev(pr.org, pr.project, skip_mandatory_checks)
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(
            f"Attempting merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} ({elapsed_time / 60} minutes elapsed)"
        )
        pr = GitHubPR(pr.org, pr.project, pr.pr_num)
        if initial_commit_sha != pr.last_commit_sha():
            raise RuntimeError(
                "New commits were pushed while merging. Please rerun the merge command."
            )
        try:
            required_checks = []
            failed_rule_message = None
            ignore_flaky_failures = True
            try:
                find_matching_merge_rule(
                    pr, repo, ignore_current_checks=ignore_current_checks
                )
            except MandatoryChecksMissingError as ex:
                if ex.rule is not None:
                    ignore_flaky_failures = ex.rule.ignore_flaky_failures
                    if ex.rule.mandatory_checks_name is not None:
                        required_checks = ex.rule.mandatory_checks_name
                failed_rule_message = ex

            checks = pr.get_checkrun_conclusions()
            checks = get_classifications(
                pr.pr_num,
                pr.project,
                checks,
                ignore_current_checks=ignore_current_checks,
            )
            pending, failing, _ = categorize_checks(
                checks,
                required_checks + [x for x in checks if x not in required_checks],
                ok_failed_checks_threshold=IGNORABLE_FAILED_CHECKS_THESHOLD
                if ignore_flaky_failures
                else 0,
            )
            # HACK until GitHub will be better about surfacing those
            startup_failures = filter_checks_with_lambda(
                checks, lambda status: status == "STARTUP_FAILURE"
            )
            if len(startup_failures) > 0:
                raise RuntimeError(
                    f"{len(startup_failures)} STARTUP failures reported, please check workflows syntax! "
                    + ", ".join(f"[{x.name}]({x.url})" for x in startup_failures[:5])
                )
            # END of HACK

            if len(failing) > 0:
                raise RuntimeError(
                    f"{len(failing)} jobs have failed, first few of them are: "
                    + ", ".join(f"[{x[0]}]({x[1]})" for x in failing[:5])
                )
            if len(pending) > 0:
                if failed_rule_message is not None:
                    raise failed_rule_message
                else:
                    raise MandatoryChecksMissingError(
                        f"Still waiting for {len(pending)} jobs to finish, "
                        + f"first few of them are: {', '.join(x[0] for x in pending[:5])}"
                    )

            return pr.merge_into(
                repo,
                dry_run=dry_run,
                skip_mandatory_checks=skip_mandatory_checks,
                comment_id=comment_id,
                ignore_current_checks=ignore_current_checks,
            )
        except MandatoryChecksMissingError as ex:
            last_exception = str(ex)
            print(
                f"Merge of https://github.com/{pr.org}/{pr.project}/pull/{pr.pr_num} failed due to: {ex}. Retrying in 5 min",
                flush=True,
            )
            time.sleep(5 * 60)
    # Finally report timeout back
    msg = f"Merged timed out after {timeout_minutes} minutes. Please contact the pytorch_dev_infra team."
    msg += f"The last exception was: {last_exception}"
    gh_add_labels(pr.org, pr.project, pr.pr_num, ["land-failed"], dry_run)
    raise RuntimeError(msg)


def main() -> None:
    # [FUNCTION DOC]
    # CLI entry point. Builds the repo and PR objects, then handles: revert (posts a start
    # comment and calls try_revert), closed PR, unsupported cross-repo ghstack, wrong base
    # branch, mergeability check only (local dry merge), unexplained submodule updates, and
    # the normal merge. Failures are posted as PR comments and recorded; the merging label is
    # removed at the end unless only checking mergeability. Most problems are reported via PR
    # comments rather than exit codes.
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    def handle_exception(e: Exception, title: str = "Merge failed") -> None:
        # [FUNCTION DOC] HELPER: Posts a failure comment with the reason and, when GH_RUN_URL
        # is set, a collapsed details block naming the workflow run and failing merge rule.
        # Also prints the current traceback.
        exception = f"**Reason**: {e}"

        failing_rule = None
        if isinstance(e, MergeRuleFailedError):
            failing_rule = e.rule.name if e.rule else None

        internal_debugging = ""
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            # Hide this behind a collapsed bullet since it's not helpful to most devs
            internal_debugging = "\n".join(
                line
                for line in (
                    "<details><summary>Details for Dev Infra team</summary>",
                    f'Raised by <a href="{run_url}">workflow job</a>\n',
                    f"Failing merge rule: {failing_rule}" if failing_rule else "",
                    "</details>",
                )
                if line
            )  # ignore empty lines during the join

        msg = "\n".join((f"## {title}", f"{exception}", "", f"{internal_debugging}"))

        gh_post_pr_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)
        import traceback

        traceback.print_exc()

    if args.revert:
        try:
            gh_post_pr_comment(
                org,
                project,
                args.pr_num,
                get_revert_message(org, project, pr.pr_num),
                args.dry_run,
            )
            try_revert(
                repo,
                pr,
                dry_run=args.dry_run,
                comment_id=args.comment_id,
                reason=args.reason,
            )
        except Exception as e:
            handle_exception(e, f"Reverting PR {args.pr_num} failed")
        return

    if pr.is_closed():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            f"Can't merge closed PR #{args.pr_num}",
            dry_run=args.dry_run,
        )
        return

    if pr.is_cross_repo() and pr.is_ghstack_pr():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            "Cross-repo ghstack merges are not supported",
            dry_run=args.dry_run,
        )
        return
    if not pr.is_ghstack_pr() and pr.base_ref() != pr.default_branch():
        gh_post_pr_comment(
            org,
            project,
            args.pr_num,
            f"PR targets {pr.base_ref()} rather than {pr.default_branch()}, refusing merge request",
            dry_run=args.dry_run,
        )
        return

    if args.check_mergeability:
        if pr.is_ghstack_pr():
            get_ghstack_prs(repo, pr)  # raises error if out of sync
        pr.merge_changes_locally(
            repo,
            skip_mandatory_checks=True,
            skip_all_rule_checks=True,
        )
        return

    if not args.force and pr.has_invalid_submodule_updates():
        message = (
            f"This PR updates submodules {', '.join(pr.get_changed_submodules())}\n"
        )
        message += '\nIf those updates are intentional, please add "submodule" keyword to PR title/description.'
        gh_post_pr_comment(org, project, args.pr_num, message, dry_run=args.dry_run)
        return
    try:
        # Ensure comment id is set, else fail
        if not args.comment_id:
            raise ValueError(
                "Comment ID is required for merging PRs, please provide it using --comment-id"
            )

        merge(
            pr,
            repo,
            comment_id=args.comment_id,
            dry_run=args.dry_run,
            skip_mandatory_checks=args.force,
            ignore_current=args.ignore_current,
        )
    except Exception as e:
        handle_exception(e)

        if args.comment_id and args.pr_num:
            # Finally, upload the record to s3, we don't have access to the
            # list of pending and failed checks here, but they are not really
            # needed at the moment
            save_merge_record(
                comment_id=args.comment_id,
                pr_num=args.pr_num,
                owner=org,
                project=project,
                author=pr.get_author(),
                pending_checks=[],
                failed_checks=[],
                ignore_current_checks=[],
                broken_trunk_checks=[],
                flaky_checks=[],
                unstable_checks=[],
                last_commit_sha=pr.last_commit_sha(default=""),
                merge_base_sha=pr.get_merge_base(),
                is_failed=True,
                skip_mandatory_checks=args.force,
                ignore_current=args.ignore_current,
                error=str(e),
            )
        else:
            print("Missing comment ID or PR number, couldn't upload to s3")
    finally:
        if not args.check_mergeability:
            gh_remove_label(
                org, project, args.pr_num, MERGE_IN_PROGRESS_LABEL, args.dry_run
            )


if __name__ == "__main__":
    main()
