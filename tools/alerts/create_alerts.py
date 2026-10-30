#!/usr/bin/env python3
# [MODULE DOC] Command-line script that reads the PyTorch HUD CI grid for a repo/branch and prints
# a JSON list of alerts for jobs that are recurrently failing. Defines status constants, the
# JobStatus classifier and helpers to fetch, group, filter and classify jobs. Importing has no
# side effects; running as a script performs an HTTP request to hud.pytorch.org and prints the
# JSON to stdout (nothing is posted).

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import requests


ALL_SKIPPED_THRESHOLD = 100
SIMILARITY_THRESHOLD = 0.75
FAILURE_CHAIN_THRESHOLD = 2
MAX_CONCURRENT_ALERTS = 1
FAILED_JOB_PATTERN = (
    r"^- \[(.*)\]\(.*\) failed consecutively starting with commit \[.*\]\(.*\)$"
)

PENDING = "pending"
NEUTRAL = "neutral"
SKIPPED = "skipped"
SUCCESS = "success"
FAILURE = "failure"
CANCELED = "canceled"

ISSUES_WITH_LABEL_QUERY = """
query ($owner: String!, $name: String!, $labels: [String!]) {
  repository(owner: $owner, name: $name, followRenames: false) {
    issues(last: 10, labels: $labels, states: [OPEN]) {
      nodes {
        id
        title
        closed
        number
        body
        createdAt
        comments(first: 100) {
          nodes {
            bodyText
            databaseId
          }
        }
      }
    }
  }
}
"""

NUM_ISSUES_QUERY = """
query ($query: String!) {
  search(type: ISSUE, query: $query) {
    issueCount
  }
}
"""

DISABLED_ALERTS = [
    "rerun_disabled_tests",
    "unstable",
]


class JobStatus:
    # [CLASS DOC] Per-job-name view of CI history. Constructed from a job name and its list of
    # per-commit status records (newest first, as ordered by HUD). On construction it derives
    # filtered_statuses (skipped/neutral/unknown results removed), current_status (newest
    # non-pending status), failure_chain (most recent run of failed jobs) and flaky_jobs. The
    # class-level list attributes are only defaults; __init__ sets the instance values. The jobs
    # attribute is never used. should_alert decides whether the job warrants an alert.
    job_name: str = ""
    jobs: list[Any] = []
    current_status: Any = None
    job_statuses: list[Any] = []
    filtered_statuses: list[Any] = []
    failure_chain: list[Any] = []
    flaky_jobs: list[Any] = []

    def __init__(self, job_name: str, job_statuses: list[Any]) -> None:
        # [FUNCTION DOC] Stores the name and raw statuses, then computes the filtered statuses,
        # current status, failure chain and flaky job list in that order (flaky detection depends
        # on the current status). No I/O.
        self.job_name = job_name
        self.job_statuses = job_statuses

        self.filtered_statuses = list(
            filter(lambda j: not is_job_skipped(j), job_statuses)
        )
        self.current_status = self.get_current_status()
        self.failure_chain = self.get_most_recent_failure_chain()
        self.flaky_jobs = self.get_flaky_jobs()

    def get_current_status(self) -> Any:
        # [FUNCTION DOC] Returns the first status in the filtered list whose conclusion is not
        # pending, or None when every filtered status is pending or the list is empty.
        """
        When getting the current status, we want the latest status which is not pending,
        be it success or failure
        """
        for status in self.filtered_statuses:
            if status["conclusion"] != PENDING:
                return status
        return None

    def get_unique_failures(self, jobs: list[Any]) -> dict[str, list[Any]]:
        # [FUNCTION DOC] Groups the failed jobs in the given list by similar failure text. Only
        # jobs whose conclusion is failure are considered. The failure captures (a list in the HUD
        # data) are joined with spaces and compared to existing group keys using a
        # sequence-matcher ratio above SIMILARITY_THRESHOLD; the first similar key wins, otherwise
        # a new group is started. Jobs lacking failure captures go in the group keyed
        # unclassified.
        # KNOWN ISSUE: each such job replaces the whole unclassified group instead of being
        # appended, so earlier unclassified jobs are dropped.
        # Returns a mapping from group key to list of jobs.
        """
        Returns list of jobs grouped by failureCaptures from the input list
        """
        failures = defaultdict(list)
        for job in jobs:
            if job["conclusion"] == "failure":
                found_similar_failure = False
                if "failureCaptures" not in job:
                    failures["unclassified"] = [job]
                    continue

                # This is now a list returned by HUD API, not a string
                failureCaptures = " ".join(job["failureCaptures"])

                for failure in failures:
                    seq = SequenceMatcher(None, failureCaptures, failure)
                    if seq.ratio() > SIMILARITY_THRESHOLD:
                        failures[failure].append(job)
                        found_similar_failure = True
                        break
                if not found_similar_failure:
                    failures[failureCaptures] = [job]

        return failures

    # A flaky job is if it's the only job that has that failureCapture and is not the most recent job
    def get_flaky_jobs(self) -> list[Any]:
        # [FUNCTION DOC] HELPER: Returns the failed jobs that form a group of exactly one job (a
        # failure signature seen once) and whose commit sha differs from the current status sha,
        # treating them as flaky. Raises TypeError if the current status is None while a singleton
        # group exists.
        unique_failures = self.get_unique_failures(self.filtered_statuses)
        flaky_jobs = []
        for failure in unique_failures:
            failure_list = unique_failures[failure]
            if (
                len(failure_list) == 1
                and failure_list[0]["sha"] != self.current_status["sha"]
            ):
                flaky_jobs.append(failure_list[0])
        return flaky_jobs

    # The most recent failure chain is an array of jobs that have the same-ish failures.
    # A success in the middle of the chain will terminate the chain.
    def get_most_recent_failure_chain(self) -> list[Any]:
        # [FUNCTION DOC] HELPER: Walks the filtered statuses in order and collects failed jobs
        # starting at the first failure, stopping at the first non-failed job after that. Leading
        # non-failed jobs are skipped. Returns the (possibly empty) list of failed jobs.
        failures = []
        found_most_recent_failure = False

        for job in self.filtered_statuses:
            if is_job_failed(job):
                failures.append(job)
                found_most_recent_failure = True
            if found_most_recent_failure and not is_job_failed(job):
                break

        return failures

    def should_alert(self) -> bool:
        # [FUNCTION DOC] Decides whether this job should raise an alert. True only if there is a
        # current status that is not a success, some group of similar failures within the failure
        # chain has at least FAILURE_CHAIN_THRESHOLD jobs, and the job name does not contain any
        # entry of DISABLED_ALERTS. Pure computation, no side effects.
        # Group jobs by their failures. The length of the failure chain is used
        # to raise the alert, so we can do a simple tweak here to use the length
        # of the longest unique chain
        unique_failures = self.get_unique_failures(self.failure_chain)

        return (
            self.current_status is not None
            and self.current_status["conclusion"] != SUCCESS
            and any(
                len(failure_chain) >= FAILURE_CHAIN_THRESHOLD
                for failure_chain in unique_failures.values()
            )
            and all(
                disabled_alert not in self.job_name
                for disabled_alert in DISABLED_ALERTS
            )
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a short string showing the job name, used when printing
        # or debugging.
        return f"jobName: {self.job_name}"


def fetch_hud_data(repo: str, branch: str) -> Any:
    # [FUNCTION DOC] Fetches the HUD grid for the given repo and branch with an HTTP GET to
    # hud.pytorch.org (page 0), raising an HTTP error exception on a bad status. Returns a tuple
    # of the job-name list and the sha grid parsed from the JSON response.
    response = requests.get(f"https://hud.pytorch.org/api/hud/{repo}/{branch}/0")
    response.raise_for_status()
    hud_data = json.loads(response.text)
    return (hud_data["jobNames"], hud_data["shaGrid"])


# Creates a Dict of Job Name -> [JobData]. Essentially a Column in HUD
def map_job_data(jobNames: Any, shaGrid: Any) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Transposes the HUD grid into a mapping from job name to list of that
    # job's records, one per commit row, in grid order. Job names are matched to each row's jobs
    # by position.
    jobData = defaultdict(list)
    for sha in shaGrid:
        for ind, job in enumerate(sha["jobs"]):
            jobData[jobNames[ind]].append(job)
    return jobData


def is_job_failed(job: Any) -> bool:
    # [FUNCTION DOC] HELPER: True when the job has a conclusion that is neither success nor
    # pending (missing conclusion counts as not failed).
    conclusion = job.get("conclusion", None)
    return conclusion is not None and conclusion != SUCCESS and conclusion != PENDING


def is_job_skipped(job: Any) -> bool:
    # [FUNCTION DOC] HELPER: True when the job's conclusion is neutral or skipped, or is
    # missing/None; such jobs are ignored when computing status.
    conclusion = job.get("conclusion", None)
    return conclusion in (NEUTRAL, SKIPPED) or conclusion is None


def get_failed_jobs(job_data: list[Any]) -> list[Any]:
    # [FUNCTION DOC] HELPER: Returns the jobs from the list whose conclusion is exactly failure.
    # Raises KeyError for a job without a conclusion field. Not used elsewhere in this file.
    return [job for job in job_data if job["conclusion"] == "failure"]


def classify_jobs(
    all_job_names: list[str], sha_grid: Any, filtered_jobs_names: set[str]
) -> tuple[list[JobStatus], list[Any]]:
    # [FUNCTION DOC] Builds a JobStatus for every job name in the HUD data, then keeps only those
    # named in filtered_jobs_names. Returns a tuple: the JobStatus objects that should alert, and
    # a flat list of the flaky jobs from all considered names. Arguments are the list of all job
    # names, the sha grid and the set of names to consider.
    """
    Creates Job Statuses which has the logic for if need to alert or if there's flaky jobs.
    Classifies jobs into jobs to alert on and flaky jobs.
    :param all_job_names: list of all job names as returned by the HUD
    :param sha_grid: list of all job data as returned by the HUD (parallel index to all_job_names)
    :param filtered_jobs_names: set of job names to actually consider
    :return:
    """
    job_data = map_job_data(all_job_names, sha_grid)
    job_statuses: list[JobStatus] = []
    for job in job_data:
        job_statuses.append(JobStatus(job, job_data[job]))

    jobs_to_alert_on = []
    flaky_jobs = []

    for job_status in job_statuses:
        if job_status.job_name not in filtered_jobs_names:
            continue
        if job_status.should_alert():
            jobs_to_alert_on.append(job_status)
        flaky_jobs.extend(job_status.flaky_jobs)

    return jobs_to_alert_on, flaky_jobs


# filter job names that don't match the regex
def filter_job_names(job_names: list[str], job_name_regex: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the job names that match the given regex at the start of the
    # name; when the regex is empty returns the names unchanged.
    if job_name_regex:
        return [
            job_name for job_name in job_names if re.match(job_name_regex, job_name)
        ]
    return job_names


def get_recurrently_failing_jobs_alerts(
    repo: str, branch: str, job_name_regex: str
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Fetches HUD data, filters the job names by the regex and prints the filter
    # summary to stdout when a regex is given. Classifies the jobs and returns one alert mapping
    # per recurrently failing job with keys for alert type, object (job name), empty oncall lists
    # and flags, the sha of the last job in its failure chain, and the branch. Network errors from
    # the fetch propagate.
    job_names, sha_grid = fetch_hud_data(repo=repo, branch=branch)

    filtered_job_names = set(filter_job_names(job_names, job_name_regex))
    if job_name_regex:
        print()
        print(f"Filtered to {len(filtered_job_names)} jobs:")
        if len(filtered_job_names) == 0:
            print("No jobs matched the regex")
        elif len(filtered_job_names) == len(job_names):
            print("All jobs matched the regex")
        else:
            print("\n".join(filtered_job_names))

    (recurrently_failing_jobs, flaky_jobs) = classify_jobs(
        job_names, sha_grid, filtered_job_names
    )

    alerts = []
    for job in recurrently_failing_jobs:
        entry = {
            "AlertType": "Recurrently Failing Job",
            "AlertObject": job.job_name,
            "OncallTeams": [],
            "OncallIndividuals": [],
            "Flags": [],
            "sha": job.failure_chain[-1]["sha"],
            "branch": branch,
        }
        alerts.append(entry)
    return alerts


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC] Defines and parses the command line: repo, branch, job name regex, flaky-test
    # alert flag and dry-run flag, each defaulting from an environment variable (strtobool is
    # imported lazily here). Returns the namespace. Note that the flaky-test and dry-run flags are
    # parsed but not used by the script entry point.
    # Imported here so library users that pull in filter_job_names / JobStatus
    # do not pay for the strtobool import path.
    from tools.strtobool import strtobool

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        help="Repository to do checks for",
        type=str,
        default=os.getenv("REPO_TO_CHECK", "pytorch/pytorch"),
    )
    parser.add_argument(
        "--branch",
        help="Branch to do checks for",
        type=str,
        default=os.getenv("BRANCH_TO_CHECK", "main"),
    )
    parser.add_argument(
        "--job-name-regex",
        help="Consider only job names matching given regex (if omitted, all jobs are matched)",
        type=str,
        default=os.getenv("JOB_NAME_REGEX", ""),
    )
    parser.add_argument(
        "--with-flaky-test-alert",
        help="Run this script with the flaky test alerting",
        type=strtobool,
        default=os.getenv("WITH_FLAKY_TEST_ALERT", "YES"),
    )
    parser.add_argument(
        "--dry-run",
        help="Whether or not to actually post issues",
        type=strtobool,
        default=os.getenv("DRY_RUN", "YES"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = json.dumps(
        get_recurrently_failing_jobs_alerts(args.repo, args.branch, args.job_name_regex)
    )

    print(data)
