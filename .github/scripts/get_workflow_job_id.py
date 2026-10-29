# [MODULE DOC] Script that finds the id and name of the GitHub Actions job currently running on
# this runner. It lists the jobs of the given workflow run through the GitHub API (following
# pagination links, with retries on HTTP errors), picks the most recently started job whose
# runner name matches, and publishes job-id and job-name as step outputs. On any failure it
# prints the error to standard error and a fallback workflow-<run id> string to standard output.
# Needs the GITHUB_TOKEN environment variable (and optionally GITHUB_REPOSITORY).
# Helper to get the id of the currently running job in a GitHub Actions
# workflow. GitHub does not provide this information to workflow runs, so we
# need to figure it out based on what they *do* provide.
from __future__ import annotations

import argparse
import json
import operator
import os
import re
import sys
import time
import urllib
import urllib.parse
from typing import Any, TYPE_CHECKING
from urllib.request import Request, urlopen


if TYPE_CHECKING:
    from collections.abc import Callable


def parse_json_and_links(conn: Any) -> tuple[Any, dict[str, dict[str, str]]]:
    # [FUNCTION DOC] HELPER: Reader for an open HTTP response. Parses the Link header (if any)
    # into a mapping from relation name (such as next) to a mapping of its parameters plus the
    # decoded url, skipping malformed elements, and then parses the response body as JSON.
    # Returns the pair of parsed JSON and links mapping.
    links = {}
    # Extract links which GH uses for pagination
    # see https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Link
    if "Link" in conn.headers:
        for elem in re.split(", *<", conn.headers["Link"]):
            try:
                url, params_ = elem.split(";", 1)
            except ValueError:
                continue
            url = urllib.parse.unquote(url.strip("<> "))
            qparams = urllib.parse.parse_qs(params_.strip(), separator=";")
            params = {
                k: v[0].strip('"')
                for k, v in qparams.items()
                if type(v) is list and len(v) > 0
            }
            params["url"] = url
            if "rel" in params:
                links[params["rel"]] = params

    return json.load(conn), links


def fetch_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
    retries: int | None = 3,
    backoff_timeout: float = 0.5,
) -> Any:
    # [FUNCTION DOC] Opens the URL with the given headers and returns reader applied to the
    # response (raw bytes by default). On an HTTP error, retries recursively up to retries more
    # times, sleeping backoff_timeout seconds each time; when exhausted raises RuntimeError whose
    # message is a tuple of text pieces describing the status. Other errors propagate.
    # KNOWN ISSUE: the error message passed to RuntimeError is a tuple rather than a string, and
    # the module does not import urllib.error explicitly (it relies on urlopen importing it).
    if headers is None:
        headers = {}
    try:
        with urlopen(Request(url, headers=headers)) as conn:
            return reader(conn)
    except urllib.error.HTTPError as err:
        if isinstance(retries, (int, float)) and retries > 0:
            time.sleep(backoff_timeout)
            return fetch_url(
                url,
                headers=headers,
                reader=reader,
                retries=retries - 1,
                backoff_timeout=backoff_timeout,
            )
        exception_message = (
            "Is github alright?",
            f"Received status code '{err.code}' when attempting to retrieve {url}:\n",
            f"{err.reason}\n\nheaders={err.headers}",
        )
        raise RuntimeError(exception_message) from err


def parse_args() -> Any:
    # [FUNCTION DOC] HELPER: Parses the two positional arguments, the workflow run id and the
    # runner name, and returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workflow_run_id", help="The id of the workflow run, should be GITHUB_RUN_ID"
    )
    parser.add_argument(
        "runner_name",
        help="The name of the runner to retrieve the job id, should be RUNNER_NAME",
    )

    return parser.parse_args()


def fetch_jobs(url: str, headers: dict[str, str]) -> list[dict[str, str]]:
    # [FUNCTION DOC] Fetches the jobs list at url and follows next-page links until exhausted,
    # concatenating all pages, and returns the combined list of job mappings. Raises an error if
    # the jobs value of the first page is not a list.
    response, links = fetch_url(url, headers=headers, reader=parse_json_and_links)
    jobs = response["jobs"]
    if type(jobs) is not list:
        raise AssertionError(f"Expected jobs to be a list, got {type(jobs).__name__}")
    while "next" in links:
        response, links = fetch_url(
            links["next"]["url"], headers=headers, reader=parse_json_and_links
        )
        jobs.extend(response["jobs"])

    return jobs


# Our strategy is to retrieve the parent workflow run, then filter its jobs on
# RUNNER_NAME to figure out which job we're currently running.
#
# Why RUNNER_NAME? Because it's the only thing that uniquely identifies a job within a workflow.
# GITHUB_JOB doesn't work, as it corresponds to the job yaml id
# (https://bit.ly/37e78oI), which has two problems:
# 1. It's not present in the workflow job JSON object, so we can't use it as a filter.
# 2. It isn't unique; for matrix jobs the job yaml id is the same for all jobs in the matrix.
#
# RUNNER_NAME on the other hand is unique across the pool of runners. Also,
# since only one job can be scheduled on a runner at a time, we know that
# looking for RUNNER_NAME will uniquely identify the job we're currently
# running.


def find_job_id_name(args: Any) -> tuple[str, str]:
    # [FUNCTION DOC] Lists all jobs of the workflow run named by args (repo from
    # GITHUB_REPOSITORY, default pytorch/pytorch; token from GITHUB_TOKEN, required), sorts them
    # newest-started first and returns the id and name of the first job whose runner name equals
    # the runner name argument. Raises RuntimeError if none matches.
    # KNOWN ISSUE: the return annotation says the id is a string but the API gives an integer.
    # From https://docs.github.com/en/actions/learn-github-actions/environment-variables
    PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
    PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    REQUEST_HEADERS = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + GITHUB_TOKEN,
    }

    url = f"{PYTORCH_GITHUB_API}/actions/runs/{args.workflow_run_id}/jobs?per_page=100"
    jobs = fetch_jobs(url, REQUEST_HEADERS)

    # Sort the jobs list by start time, in descending order. We want to get the most
    # recently scheduled job on the runner.
    jobs.sort(key=operator.itemgetter("started_at"), reverse=True)

    for job in jobs:
        if job["runner_name"] == args.runner_name:
            return (job["id"], job["name"])

    raise RuntimeError(f"Can't find job id for runner {args.runner_name}")


def set_output(name: str, val: Any) -> None:
    # [FUNCTION DOC] Prints the output assignment and publishes it as a GitHub Actions output:
    # appends a line to the file named by GITHUB_OUTPUT when set, else prints the legacy
    # set-output command.
    print(f"Setting output {name}={val}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def main() -> None:
    # [FUNCTION DOC] Entry point: finds the current job and sets the job-id and job-name outputs.
    # Any exception is caught: its representation goes to standard error and a fallback name
    # made of workflow- plus the run id is printed to standard output. Never exits non-zero.
    args = parse_args()
    try:
        # Get both the job ID and job name because we have already spent a request
        # here to get the job info
        job_id, job_name = find_job_id_name(args)
        set_output("job-id", job_id)
        set_output("job-name", job_name)
    except Exception as e:
        print(repr(e), file=sys.stderr)
        print(f"workflow-{args.workflow_run_id}")


if __name__ == "__main__":
    main()
