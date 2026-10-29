# [MODULE DOC] Downloads Dynamo/eager CI test report archives for a commit from GitHub Actions
# artifacts: uses the gh CLI to find the pull workflow run and its jobs, the HUD artifact
# listing to get zip URLs, and wget and unzip to fetch and extract them. CONFIGS maps config
# names to the CI job names whose reports are needed. No import-time side effects.
import json
import os
import pprint
import re
import subprocess

import requests


CONFIGS = {
    "dynamo39": {
        "linux-jammy-py3.10-clang12 / test (dynamo_wrapped, 1, 3, linux.2xlarge)",
        "linux-jammy-py3.10-clang12 / test (dynamo_wrapped, 2, 3, linux.2xlarge)",
        "linux-jammy-py3.10-clang12 / test (dynamo_wrapped, 3, 3, linux.2xlarge)",
    },
    "dynamo313": {
        "linux-jammy-py3.13-clang12 / test (dynamo_wrapped, 1, 3, linux.2xlarge)",
        "linux-jammy-py3.13-clang12 / test (dynamo_wrapped, 2, 3, linux.2xlarge)",
        "linux-jammy-py3.13-clang12 / test (dynamo_wrapped, 3, 3, linux.2xlarge)",
    },
    "eager313": {
        "linux-jammy-py3.13-clang12 / test (default, 1, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 2, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 3, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 4, 5, linux.4xlarge)",
        "linux-jammy-py3.13-clang12 / test (default, 5, 5, linux.4xlarge)",
    },
}


def download_reports(commit_sha, configs=("dynamo39", "dynamo313", "eager313")):
    # [FUNCTION DOC] Ensures test reports for the given configs exist locally under a
    # directory named tmp_test_reports plus the commit, one subdirectory per config. If all
    # exist, returns without downloading. Otherwise looks up the workflow run via gh, checks
    # all required jobs are present (assertion error if not), lists artifacts from the HUD
    # API, and downloads and unzips each missing job's report. Returns the list of
    # per-config directory paths. Side effects: network, subprocesses, directory creation;
    # unknown config names fail an assertion.
    log_dir = "tmp_test_reports_" + commit_sha

    def subdir_path(config):
        # [FUNCTION DOC] HELPER: Path of the local directory for the given config's reports.
        return f"{log_dir}/{config}"

    for config in configs:
        assert config in CONFIGS.keys(), config
    subdir_paths = [subdir_path(config) for config in configs]

    # See which configs we haven't downloaded logs for yet
    missing_configs = []
    for config, path in zip(configs, subdir_paths):
        if os.path.exists(path):
            continue
        missing_configs.append(config)
    if len(missing_configs) == 0:
        print(
            f"All required logs appear to exist, not downloading again. Run `rm -rf {log_dir}` if this is not the case"
        )
        return subdir_paths

    output = subprocess.check_output(
        ["gh", "run", "list", "-c", commit_sha, "-w", "pull", "--json", "databaseId"]
    ).decode()
    workflow_run_id = str(json.loads(output)[0]["databaseId"])
    output = subprocess.check_output(["gh", "run", "view", workflow_run_id])
    workflow_jobs = parse_workflow_jobs(output)
    print("found the following workflow jobs:")
    pprint.pprint(workflow_jobs)

    # Figure out which jobs we need to download logs for
    required_jobs = []
    for config in configs:
        required_jobs.extend(list(CONFIGS[config]))
    for job in required_jobs:
        assert job in workflow_jobs, (
            f"{job} not found, is the commit_sha correct? has the job finished running? The GitHub API may take a couple minutes to update."
        )

    # This page lists all artifacts.
    listings = requests.get(
        f"https://hud.pytorch.org/api/artifacts/s3/{workflow_run_id}"
    ).json()

    def download_report(job_name, subdir):
        # [FUNCTION DOC] HELPER: Finds the artifact whose name starts with test-reports- and
        # ends with the job's id plus .zip, downloads it into subdir with wget, and unzips it
        # into a directory named like the zip without extension. Raises AssertionError if no
        # artifact matches; subprocess failures raise.
        job_id = workflow_jobs[job_name]
        for listing in listings:
            name = listing["name"]
            if not name.startswith("test-reports-"):
                continue
            if name.endswith(f"_{job_id}.zip"):
                url = listing["url"]
                subprocess.run(["wget", "-P", subdir, url], check=True)
                path_to_zip = f"{subdir}/{name}"
                dir_name = path_to_zip[:-4]
                subprocess.run(["unzip", path_to_zip, "-d", dir_name], check=True)
                return
        raise AssertionError("should not be hit")

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for config in set(configs) - set(missing_configs):
        print(
            f"Logs for {config} already exist, not downloading again. Run `rm -rf {subdir_path(config)}` if this is not the case."
        )
    for config in missing_configs:
        subdir = subdir_path(config)
        os.mkdir(subdir)
        job_names = CONFIGS[config]
        for job_name in job_names:
            download_report(job_name, subdir)

    return subdir_paths


def parse_workflow_jobs(output):
    # [FUNCTION DOC] Parses the bytes output of the gh run view command, returning a mapping of
    # job name to job id string for each line of the form name in duration (ID number).
    # Other lines are ignored.
    result = {}
    lines = output.decode().split("\n")
    for line in lines:
        match = re.search(r"(\S+ / .*) in .* \(ID (\d+)\)", line)
        if match is None:
            continue
        result[match.group(1)] = match.group(2)
    return result
