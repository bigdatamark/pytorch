# [MODULE DOC]
# Command line entry point to upload or download a job's pytest cache to or from S3 through the
# helpers in pytest_caching_utils. Defines the fallback temp directory TEMP_DIR. Run as a
# script; no import time side effects besides imports.
import argparse
import sys
from pathlib import Path

from pytest_caching_utils import (
    download_pytest_cache,
    GithubRepo,
    PRIdentifier,
    upload_pytest_cache,
)


TEMP_DIR = "./tmp"  # a backup location in case one isn't provided


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC] Builds and runs the argument parser. Exactly one of upload and download is
    # required, plus cache_dir, pr_identifier and job_identifier; sha, test_config and shard are
    # required only when upload appears in the process arguments; repo, temp_dir and bucket are
    # optional. Returns the parsed namespace.
    parser = argparse.ArgumentParser(
        description="Upload this job's the pytest cache to S3"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--upload", action="store_true", help="Upload the pytest cache to S3"
    )
    mode.add_argument(
        "--download",
        action="store_true",
        help="Download the pytest cache from S3, merging it with any local cache",
    )

    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Path to the folder pytest uses for its cache",
    )
    parser.add_argument("--pr_identifier", required=True, help="A unique PR identifier")
    parser.add_argument(
        "--job_identifier",
        required=True,
        help="A unique job identifier that should be the same for all runs of job",
    )
    parser.add_argument(
        "--sha", required="--upload" in sys.argv, help="SHA of the commit"
    )  # Only required for upload
    parser.add_argument(
        "--test_config", required="--upload" in sys.argv, help="The test config"
    )  # Only required for upload
    parser.add_argument(
        "--shard", required="--upload" in sys.argv, help="The shard id"
    )  # Only required for upload

    parser.add_argument(
        "--repo",
        required=False,
        help="The github repository we're running in, in the format 'owner/repo-name'",
    )
    parser.add_argument(
        "--temp_dir", required=False, help="Directory to store temp files"
    )
    parser.add_argument(
        "--bucket", required=False, help="The S3 bucket to upload the cache to"
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # [FUNCTION DOC] Entry point. Builds the PR identifier and GithubRepo from the arguments and
    # picks the temp dir (argument or TEMP_DIR). In upload mode skips with a message if the
    # cache dir does not exist, otherwise uploads the cache. In download mode downloads into the
    # cache dir. Prints the arguments; network and filesystem side effects.
    # KNOWN ISSUE: repo is optional but is parsed unconditionally; behavior without it depends
    #     on GithubRepo.from_string handling of None.
    args = parse_args()

    pr_identifier = PRIdentifier(args.pr_identifier)
    print(f"PR identifier for `{args.pr_identifier}` is `{pr_identifier}`")

    repo = GithubRepo.from_string(args.repo)
    cache_dir = Path(args.cache_dir)
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = Path(TEMP_DIR)

    if args.upload:
        print(f"Uploading cache with args {args}")

        # verify the cache dir exists
        if not cache_dir.exists():
            print(f"The pytest cache dir `{cache_dir}` does not exist. Skipping upload")
            return

        upload_pytest_cache(
            pr_identifier=pr_identifier,
            repo=repo,
            job_identifier=args.job_identifier,
            sha=args.sha,
            test_config=args.test_config,
            shard=args.shard,
            cache_dir=cache_dir,
            bucket=args.bucket,
            temp_dir=temp_dir,
        )

    if args.download:
        print(f"Downloading cache with args {args}")
        download_pytest_cache(
            pr_identifier=pr_identifier,
            repo=repo,
            job_identifier=args.job_identifier,
            dest_cache_dir=cache_dir,
            bucket=args.bucket,
            temp_dir=temp_dir,
        )


if __name__ == "__main__":
    main()
