# [MODULE DOC] CI script deciding whether a build can reuse a wheel built earlier on the
# merge base commit. Checks branch name, changed files, a force-rebuild PR label, a global
# config issue and the availability of an old artifact on S3; if reusable it downloads the
# artifact, swaps in the current python sources and version strings, repackages the wheel and
# sets a GitHub Actions output. Always emits a metric. Import-time: temporarily edits the
# module search path to import the metrics helper.
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.stats.upload_metrics import emit_metric


sys.path.remove(str(REPO_ROOT))  # Clean up sys.path after import


FORCE_REBUILD_LABEL = "ci-force-rebuild"


@lru_cache
def get_merge_base() -> str:
    # [FUNCTION DOC] Returns (cached) the git merge base commit hash of HEAD and origin main.
    # If that equals HEAD itself, it uses the parent of HEAD instead. Prints the result.
    # Raises CalledProcessError if a git command fails.
    merge_base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "origin/main"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    # Remove this when we turn this off for the main branch
    if merge_base == get_head_sha():
        print("Merge base is the same as HEAD, using HEAD^")
        merge_base = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    print(f"Merge base: {merge_base}")
    return merge_base


@lru_cache
def get_head_sha() -> str:
    # [FUNCTION DOC] HELPER: Returns (cached) the full commit hash of HEAD from git.
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return sha


def is_main_branch() -> bool:
    # [FUNCTION DOC] HELPER: Intended to tell whether HEAD is the merge base (main branch).
    # KNOWN ISSUE: always returns false (the real check is commented out) and the function
    # is not called anywhere in this file.
    return False
    # Testing on main branch for now
    # print(
    #     f"Checking if we are on main branch: merge base {get_merge_base()}, head {get_head_sha()}"
    # )
    # return get_merge_base() == get_head_sha()


def query_github_api(url: str) -> Any:
    # [FUNCTION DOC] HELPER: Sends an authenticated GET to the url using the token from the
    # GITHUB_TOKEN environment variable and returns the decoded JSON body. Raises KeyError
    # if the token is unset. The HTTP status is not checked, so error bodies are returned.
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    }
    response = requests.get(url, headers=headers)
    return response.json()


@lru_cache
def check_labels_for_pr() -> bool:
    # [FUNCTION DOC] Returns (cached) true if any pull request associated with the HEAD
    # commit carries the force-rebuild label, looked up through the GitHub API. Prints what
    # it finds.
    # Check if the current commit is part of a PR and if it has the
    # FORCE_REBUILD_LABEL
    head_sha = get_head_sha()
    url = f"https://api.github.com/repos/pytorch/pytorch/commits/{head_sha}/pulls"
    response = query_github_api(url)

    print(
        f"Found {len(response)} PRs for commit {head_sha}: {[pr['number'] for pr in response]}"
    )
    for pr in response:
        labels = pr.get("labels", [])
        for label in labels:
            if label["name"] == FORCE_REBUILD_LABEL:
                print(f"Found label {FORCE_REBUILD_LABEL} in PR {pr['number']}.")
                return True
    return False


def check_issue_open() -> bool:
    # [FUNCTION DOC] Returns true if the CI config issue (number 153759) is open, which is
    # the switch to force everyone to rebuild. Queries the GitHub API and prints the state.
    # Check if issue #153759 is open.  This is the config issue for quickly
    # forcing everyone to build
    url = "https://api.github.com/repos/pytorch/pytorch/issues/153759"
    response = query_github_api(url)
    if response.get("state") == "open":
        print("Issue #153759 is open.")
        return True
    else:
        print("Issue #153759 is not open.")
        return False


def get_workflow_id(run_id: str) -> str | None:
    # [FUNCTION DOC] Looks up the workflow run with the given id through the GitHub API and
    # returns its workflow id, or None if the response has no such field.
    # KNOWN ISSUE: the annotation says string but the API returns a number.
    # Get the workflow ID that corresponds to the file for the run ID
    url = f"https://api.github.com/repos/pytorch/pytorch/actions/runs/{run_id}"
    response = query_github_api(url)
    if "workflow_id" in response:
        print(f"Found workflow ID for run ID {run_id}: {response['workflow_id']}")
        return cast(str, response["workflow_id"])
    else:
        print("No workflow ID found.")
        return None


def ok_changed_file(file: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if a changed path is safe for wheel reuse: a python
    # file under torch (excluding the C++ sources directory), a python file under test, or a
    # markdown or rst file under docs. Pure function.
    # Return true if the file is in the list of allowed files to be changed to
    # reuse the old whl
    if (
        file.startswith("torch/")
        and file.endswith(".py")
        and not file.startswith("torch/csrc/")
    ):
        return True
    if file.startswith("test/") and file.endswith(".py"):
        return True
    if file.startswith("docs/") and file.endswith((".md", ".rst")):
        return True
    return False


def check_changed_files(sha: str) -> bool:
    # [FUNCTION DOC] Compares the given commit with HEAD using git. Returns false if any file
    # under torch was deleted or any changed file is not allowed by the file filter; true
    # otherwise. Prints each decision. Raises CalledProcessError if git fails.
    # Return true if all the changed files are in the list of allowed files to
    # be changed to reuse the old whl

    # Removing files in the torch folder is not allowed since rsync will not
    # remove files
    removed_files = (
        subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                sha,
                "HEAD",
                "--diff-filter=D",
                "--no-renames",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        .strip()
        .split()
    )
    if any(file.startswith("torch/") for file in removed_files):
        print(
            f"Removed files between {sha} and HEAD: {removed_files}, cannot reuse old whl"
        )
        return False

    changed_files = (
        subprocess.check_output(
            ["git", "diff", "--name-only", sha, "HEAD", "--no-renames"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        .strip()
        .split()
    )
    print(f"Checking changed files between {sha} and HEAD:")
    for file in changed_files:
        if not ok_changed_file(file):
            print(f"  File {file} is not allowed to be changed.")
            return False
        else:
            print(f"  File {file} is allowed to be changed.")
    return True


def find_old_whl(workflow_id: str, build_environment: str, sha: str) -> bool:
    # [FUNCTION DOC] Lists main-branch runs of the workflow for the given commit and, for
    # each run, tries to download its build artifact archive for the build environment from
    # S3. On the first success writes it to the file artifacts.zip in the current directory
    # and returns true. Returns false if the environment is unset, no runs exist, or none
    # has the archive. Network errors are printed and skipped.
    # Find the old whl on s3 and download it to artifacts.zip
    if build_environment is None:
        print("BUILD_ENVIRONMENT is not set.")
        return False
    print(f"SHA: {sha}, workflow_id: {workflow_id}")

    workflow_runs = query_github_api(
        f"https://api.github.com/repos/pytorch/pytorch/actions/workflows/{workflow_id}/runs?head_sha={sha}&branch=main&per_page=100"
    )
    if workflow_runs.get("total_count", 0) == 0:
        print("No workflow runs found.")
        return False
    for run in workflow_runs.get("workflow_runs", []):
        # Look in s3 for the old whl
        run_id = run["id"]
        try:
            url = f"https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/{run_id}/{build_environment}/artifacts.zip"
            print(f"Checking for old whl at {url}")
            response = requests.get(
                url,
            )
            if response.status_code == 200:
                with open("artifacts.zip", "wb") as f:
                    f.write(response.content)
                    print(f"Found old whl file from s3: {url}")
                    return True
        except requests.RequestException as e:
            print(f"Error checking for old whl: {e}")
            continue
    return False


def unzip_artifact_and_replace_files() -> None:
    # [FUNCTION DOC] Unpacks artifacts.zip into an artifacts directory and, for each wheel
    # there, unpacks it, copies the current torch python tree over it, rewrites the old git
    # version suffix to the HEAD short hash in the version file, metadata files and names,
    # then rebuilds the wheel and finally recreates artifacts.zip in the current directory.
    # Runs external unzip, zip, mv and rm tools; failures raise CalledProcessError.
    # Unzip the artifact and replace files
    subprocess.check_output(
        ["unzip", "-o", "artifacts.zip", "-d", "artifacts"],
    )
    os.remove("artifacts.zip")

    head_sha = get_head_sha()

    # Rename wheel into zip
    wheel_path = Path("artifacts/dist").glob("*.whl")
    for path in wheel_path:
        # Should be of the form torch-2.0.0+git1234567-cp37-etc.whl
        # Should usually be the merge base sha but for the ones that didn't do
        # the replacement, it won't be.  Can probably change it to just be merge
        # base later
        old_version = f"+git{path.stem.split('+')[1].split('-')[0][3:]}"
        new_version = f"+git{head_sha[:7]}"

        def rename_to_new_version(file: str | Path) -> None:
            # [FUNCTION DOC] HELPER: Moves the file or directory to the same path with the
            # old version suffix replaced by the new one (uses the enclosing loop values).
            # Rename file with old_version to new_version
            subprocess.check_output(
                ["mv", file, str(file).replace(old_version, new_version)]
            )

        def change_content_to_new_version(file: str | Path) -> None:
            # [FUNCTION DOC] HELPER: Rewrites a text file in place replacing the old version
            # suffix with the new one; directories are skipped. Non-text files would raise a
            # decoding error.
            # Check if is a file
            if os.path.isdir(file):
                return
            # Replace the old version in the file with the new version
            with open(file) as f:
                content = f.read()
                content = content.replace(old_version, new_version)
            with open(file, "w") as f:
                f.write(content)

        zip_path = path.with_suffix(".zip")
        os.rename(path, zip_path)
        old_stem = zip_path.stem
        # Unzip the wheel
        subprocess.check_output(
            ["unzip", "-o", zip_path, "-d", f"artifacts/dist/{old_stem}"],
        )

        # Remove the old wheel (which is now a zip file)
        os.remove(zip_path)

        # Copy python files into the artifact. Use shutil rather than rsync so
        # the script works on container images that don't ship rsync (e.g. the
        # OSDC ARC builder image).
        shutil.copytree("torch", f"artifacts/dist/{old_stem}/torch", dirs_exist_ok=True)

        change_content_to_new_version(f"artifacts/dist/{old_stem}/torch/version.py")

        for file in Path(f"artifacts/dist/{old_stem}").glob(
            "*.dist-info/*",
        ):
            change_content_to_new_version(file)

        rename_to_new_version(f"artifacts/dist/{old_stem}")
        new_stem = old_stem.replace(old_version, new_version)

        for file in Path(f"artifacts/dist/{new_stem}").glob(
            "*.dist-info",
        ):
            rename_to_new_version(file)

        # Zip the wheel back
        subprocess.check_output(
            ["zip", "-r", f"{new_stem}.zip", "."],
            cwd=f"artifacts/dist/{new_stem}",
        )

        subprocess.check_output(
            [
                "mv",
                f"artifacts/dist/{new_stem}/{new_stem}.zip",
                f"artifacts/dist/{new_stem}.whl",
            ],
        )

        # Remove the extracted folder
        subprocess.check_output(
            ["rm", "-rf", f"artifacts/dist/{new_stem}"],
        )

    # Rezip the artifact
    subprocess.check_output(["zip", "-r", "artifacts.zip", "."], cwd="artifacts")
    subprocess.check_output(
        ["mv", "artifacts/artifacts.zip", "."],
    )
    return None


def set_output() -> None:
    # [FUNCTION DOC] Publishes the GitHub Actions output reuse=true by appending to the file
    # named by GITHUB_OUTPUT, or by printing the legacy set-output command if unset.
    print("Setting output reuse=true")
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print("reuse=true", file=env)
    else:
        print("::set-output name=reuse::true")


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Parses command-line options: required run id and build
    # environment, optional github ref. Returns the namespace.
    # KNOWN ISSUE: the help text for the run id option says "Workflow ID".
    parser = argparse.ArgumentParser(description="Check for old whl files.")
    parser.add_argument("--run-id", type=str, required=True, help="Workflow ID")
    parser.add_argument(
        "--build-environment", type=str, required=True, help="Build environment"
    )
    parser.add_argument(
        "--github-ref",
        type=str,
    )
    return parser.parse_args()


def can_reuse_whl(args: argparse.Namespace) -> tuple[bool, str]:
    # [FUNCTION DOC] Decides whether an old wheel may be reused. Returns (false, reason) for
    # release, tag or nightly refs, disallowed changed files, a force-rebuild label, the open
    # config issue, a missing workflow id, or no old artifact found; else (true, reason) after
    # downloading the artifact as a side effect of the lookup.
    if args.github_ref and any(
        args.github_ref.startswith(x)
        for x in [
            "refs/heads/release",
            "refs/tags/v",
            "refs/heads/nightly",
        ]
    ):
        print("Release branch, rebuild whl")
        return (False, "Release branch")

    if not check_changed_files(get_merge_base()):
        print("Cannot use old whl due to the changed files, rebuild whl")
        return (False, "Changed files not allowed")

    if check_labels_for_pr():
        print(f"Found {FORCE_REBUILD_LABEL} label on PR, rebuild whl")
        return (False, "Found FORCE_REBUILD_LABEL on PR")

    if check_issue_open():
        print("Issue #153759 is open, rebuild whl")
        return (False, "Issue #153759 is open")

    workflow_id = get_workflow_id(args.run_id)
    if workflow_id is None:
        print("No workflow ID found, rebuild whl")
        return (False, "No workflow ID found")

    if not find_old_whl(workflow_id, args.build_environment, get_merge_base()):
        print("No old whl found, rebuild whl")
        return (False, "No old whl found")
        # TODO: go backwards from merge base to find more runs

    return (True, "Found old whl")


if __name__ == "__main__":
    # Script entry: parses arguments, decides on reuse, and if reusable rewrites the
    # downloaded artifact and sets the output; always emits a metric with the decision.
    args = parse_args()

    reuse_whl, reason = can_reuse_whl(args)

    if reuse_whl:
        print("Reusing old whl")
        unzip_artifact_and_replace_files()
        set_output()

    emit_metric(
        "reuse_old_whl",
        {
            "reuse_whl": reuse_whl,
            "reason": reason,
            "build_environment": args.build_environment,
            "merge_base": get_merge_base(),
            "head_sha": get_head_sha(),
        },
    )
