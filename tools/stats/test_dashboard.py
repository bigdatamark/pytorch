# [MODULE DOC]
# Utilities that build additional dashboard info for a workflow run: look up CI job
# names via the GitHub API, derive build names and test configs from them, collect
# test-discovery exclusion files from S3 artifacts, and upload the grouped result to
# S3. Defines REGEX_JOB_INFO; no import-time side effects.

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import requests

from tools.stats.upload_stats_lib import (
    _get_request_headers,
    download_s3_artifacts,
    get_job_id,
    get_s3_resource,
    unzip,
    upload_workflow_stats_to_s3,
)


REGEX_JOB_INFO = r"(.*) \/ .*test \(([^,]*), .*\)"


@lru_cache(maxsize=1000)
def get_job_name(job_id: int) -> str:
    # [FUNCTION DOC]
    # Queries the GitHub Actions API for the job with the given id and returns its
    # name. Results are memoized. On any failure prints a message and returns the
    # placeholder NoJobName instead of raising.
    try:
        return cast(
            str,
            requests.get(
                f"https://api.github.com/repos/pytorch/pytorch/actions/jobs/{job_id}",
                headers=_get_request_headers(),
            ).json()["name"],
        )
    except Exception as e:
        print(f"Failed to get job name for job id {job_id}: {e}")
        return "NoJobName"


@lru_cache(maxsize=1000)
def get_build_name(job_name: str) -> str:
    # [FUNCTION DOC]
    # Extracts the build name (first group of the job info pattern) from a test job
    # name. Memoized. Returns NoBuildEnv and prints a message if the name does not
    # match.
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(1)  # type: ignore[union-attr]
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoBuildEnv"


@lru_cache(maxsize=1000)
def get_test_config(job_name: str) -> str:
    # [FUNCTION DOC]
    # Extracts the test config (second group of the job info pattern) from a test job
    # name. Memoized. Returns NoTestConfig and prints a message if there is no match.
    try:
        return re.match(REGEX_JOB_INFO, job_name).group(2)  # type: ignore[union-attr]
    except AttributeError:
        print(f"Failed to match job name: {job_name}")
        return "NoTestConfig"


def get_td_exclusions(
    workflow_run_id: int, workflow_run_attempt: int
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Downloads and unzips the test-jsons artifacts of the given workflow run attempt
    # into a temporary directory, reads every test-discovery exclusions json file, and
    # returns a mapping from build name to test config to the sorted list of excluded
    # test files. Side effect: changes the process working directory to the temporary
    # directory and never restores it.
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        s3_paths = download_s3_artifacts(
            "test-jsons", workflow_run_id, workflow_run_attempt
        )
        for path in s3_paths:
            unzip(path)

        grouped_tests: dict[str, Any] = defaultdict(lambda: defaultdict(set))
        for td_exclusions in Path(".").glob("**/td_exclusions*.json"):
            with open(td_exclusions) as f:
                exclusions = json.load(f)
                for exclusion in exclusions["excluded"]:
                    job_id = get_job_id(td_exclusions)
                    job_name = get_job_name(job_id)
                    build_name = get_build_name(job_name)
                    test_config = get_test_config(job_name)
                    grouped_tests[build_name][test_config].add(exclusion["test_file"])

        for build_name, build in grouped_tests.items():
            for test_config, test_files in build.items():
                grouped_tests[build_name][test_config] = sorted(test_files)
        return grouped_tests


def get_all_run_attempts(workflow_run_id: int) -> list[int]:
    # [FUNCTION DOC]
    # Lists the objects stored in the gha-artifacts bucket under the workflow run's
    # prefix and returns the sorted distinct run attempt numbers found as the first
    # path component; entries that are not integers are skipped.
    # Returns all run attempts for a given workflow run id that have test
    # artifacts
    bucket = get_s3_resource().Bucket("gha-artifacts")
    prefix = f"pytorch/pytorch/{workflow_run_id}/"
    objs = bucket.objects.filter(Prefix=prefix)
    run_attempts = set()
    for obj in objs:
        no_prefix = obj.key[len(prefix) :]
        try:
            run_attempt = int(no_prefix.split("/")[0])
            run_attempts.add(run_attempt)
        except ValueError:
            continue
    return sorted(run_attempts)


def upload_additional_info(workflow_run_id: int, workflow_run_attempt: int) -> None:
    # [FUNCTION DOC]
    # Computes the test-discovery exclusions for the run attempt and uploads them as
    # a single document to S3 under the additional_info/td_exclusions type.
    exclusions = get_td_exclusions(workflow_run_id, workflow_run_attempt)

    upload_workflow_stats_to_s3(
        workflow_run_id,
        workflow_run_attempt,
        "additional_info/td_exclusions",
        [exclusions],
    )
