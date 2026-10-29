# [MODULE DOC]
# Helpers to upload and download the pytest cache of a CI job to or from S3 so that tests that
# failed in earlier runs are run first. Contains the PRIdentifier and GithubRepo types, S3 key
# construction, and merging of the lastfailed file and the additional previous failures list
# across shard caches. Constants: bucket gha-artifacts, cache key prefix, temp folder names.
# Import time side effects: none besides imports from file_io_utils.
import hashlib
import os
from pathlib import Path
from typing import NamedTuple

from file_io_utils import (
    copy_file,
    download_s3_objects_with_prefix,
    load_json_file,
    sanitize_for_s3,
    unzip_folder,
    upload_file_to_s3,
    write_json_file,
    zip_folder,
)


PYTEST_CACHE_KEY_PREFIX = "pytest_cache"
PYTEST_CACHE_DIR_NAME = ".pytest_cache"
BUCKET = "gha-artifacts"
LASTFAILED_FILE_PATH = Path("v/cache/lastfailed")
TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL = "previous_failures_additional.json"

# Temp folders
ZIP_UPLOAD = "zip-upload"
CACHE_ZIP_DOWNLOADS = "cache-zip-downloads"
UNZIPPED_CACHES = "unzipped-caches"


# Since the pr identifier can be based on include user defined text (like a branch name)
# we hash it to sanitize the input and avoid corner cases
class PRIdentifier(str):
    # [CLASS DOC]
    # String subclass holding the md5 hex digest of the value given at construction, so that
    # user controlled text (such as a branch name) becomes a safe fixed-form identifier.
    # Immutable, no extra attributes.
    __slots__ = ()

    def __new__(cls, value: str) -> "PRIdentifier":
        # [FUNCTION DOC] Returns a new instance whose string value is the md5 hex digest of the
        # utf-8 encoded input.
        md5 = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
        return super().__new__(cls, md5)


class GithubRepo(NamedTuple):
    # [CLASS DOC]
    # Named tuple (owner, name) identifying a GitHub repository; prints as owner/name.
    owner: str
    name: str

    # Create a Repo from a string like "owner/repo"
    @classmethod
    def from_string(cls, repo_string: str) -> "GithubRepo":
        # [FUNCTION DOC] Creates a GithubRepo from text of the form owner/name. Raises
        # ValueError if there is no slash (and, through tuple unpacking, if there is
        # more than one).
        if "/" not in repo_string:
            raise ValueError(
                f"repo_string must be of the form 'owner/repo', not {repo_string}"
            )

        owner, name = repo_string.split("/")
        return cls(owner, name)

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the repository as owner/name text.
        return f"{self.owner}/{self.name}"


def upload_pytest_cache(
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    sha: str,
    test_config: str,
    shard: str,
    cache_dir: Path,
    temp_dir: Path,
    bucket: str = BUCKET,
) -> None:
    # [FUNCTION DOC] Zips the pytest cache dir and uploads it to S3 under a key built from the
    # repo, PR identifier, job identifier, sha, test config and shard, keeping the zip file
    # extension. An empty bucket falls back to the default bucket. Raises ValueError if
    # pr_identifier is not a PRIdentifier. Side effects: writes the zip under the temp dir and
    # uploads to S3.
    # KNOWN ISSUE: the docstring describes merging with previous caches and a job argument, but
    #     this function only zips and uploads; merging happens on download.
    """
    Uploads the pytest cache to S3, merging it with any previous caches from previous runs of the same job.
    In particular, this keeps all the failed tests across all runs of this job in the cache, so that
    future jobs that download this cache will prioritize running tests that have failed in the past.

    Args:
        pr_identifier: A unique, human readable identifier for the PR
        job: The name of the job that is uploading the cache
    """

    if not isinstance(pr_identifier, PRIdentifier):
        raise ValueError(
            f"pr_identifier must be of type PRIdentifier, not {type(pr_identifier)}"
        )

    if not bucket:
        bucket = BUCKET

    # Upload the cache
    obj_key_prefix = _get_s3_key_prefix(
        pr_identifier, repo, job_identifier, sha, test_config, shard
    )
    zip_file_path = zip_folder(cache_dir, temp_dir / ZIP_UPLOAD / obj_key_prefix)
    obj_key = f"{obj_key_prefix}{os.path.splitext(zip_file_path)[1]}"  # Keep the new file extension
    upload_file_to_s3(zip_file_path, bucket, obj_key)


def download_pytest_cache(
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    dest_cache_dir: Path,
    temp_dir: Path,
    bucket: str = BUCKET,
) -> None:
    # [FUNCTION DOC] Downloads every cache zip stored under the key prefix for this PR and job
    # (all shards), unzips each into a random folder under the temp dir and merges it into
    # dest_cache_dir. An empty bucket falls back to the default bucket. Raises ValueError if
    # pr_identifier is not a PRIdentifier. Side effects: network access, files written under the
    # temp dir and dest dir.
    """
    Downloads the pytest cache from S3. The goal is to detect any tests that have failed in the past
    and run them first, so that the dev can get faster feedback on them.

    We merge the cache from all shards since tests can get shuffled around from one shard to another
    (based on when we last updated our stats on how long each test takes to run). This ensures that
    even if a test moves to a different shard, that shard will know to run it first if had failed previously.
    """
    if not bucket:
        bucket = BUCKET

    if not isinstance(pr_identifier, PRIdentifier):
        raise ValueError(
            f"pr_identifier must be of type PRIdentifier, not {type(pr_identifier)}"
        )

    obj_key_prefix = _get_s3_key_prefix(pr_identifier, repo, job_identifier)

    zip_download_dir = temp_dir / CACHE_ZIP_DOWNLOADS / obj_key_prefix

    # downloads the cache zips for all shards
    downloads = download_s3_objects_with_prefix(
        bucket, obj_key_prefix, zip_download_dir
    )

    for downloaded_zip in downloads:
        # Unzip into random folder, then merge with the current cache
        cache_dir_for_shard = (
            temp_dir / UNZIPPED_CACHES / os.urandom(16).hex() / PYTEST_CACHE_DIR_NAME
        )

        unzip_folder(downloaded_zip, cache_dir_for_shard)
        print(f"Merging cache from {downloaded_zip}")
        _merge_pytest_caches(cache_dir_for_shard, dest_cache_dir)


def _get_s3_key_prefix(
    pr_identifier: PRIdentifier,
    repo: GithubRepo,
    job_identifier: str,
    sha: str = "",
    test_config: str = "",
    shard: str = "",
) -> str:
    # [FUNCTION DOC] HELPER: Returns the S3 key prefix (no extension) of the form cache prefix /
    # owner / name / PR identifier / sanitized job identifier, followed by the sha, sanitized
    # test config and shard for each of those that is non-empty.
    """
    The prefix to any S3 object key for a pytest cache. It's only a prefix though, not a full path to an object.
    For example, it won't include the file extension.
    """
    prefix = f"{PYTEST_CACHE_KEY_PREFIX}/{repo.owner}/{repo.name}/{pr_identifier}/{sanitize_for_s3(job_identifier)}"

    if sha:
        prefix += f"/{sha}"
    if test_config:
        prefix += f"/{sanitize_for_s3(test_config)}"
    if shard:
        prefix += f"/{shard}"

    return prefix


def _merge_pytest_caches(
    pytest_cache_dir_to_merge_from: Path, pytest_cache_dir_to_merge_into: Path
) -> None:
    # [FUNCTION DOC] HELPER: Merges one unzipped pytest cache into another. Copies the static
    # support files (gitignore, cache dir tag, readme) only if present in the source and missing
    # in the destination, then merges the lastfailed file and the additional failures file.
    # Mutates the destination folder.
    # LASTFAILED_FILE_PATH is the only file we actually care about in the cache
    # since it contains all the tests that failed.
    #
    # The remaining files are static supporting files that don't really matter. They
    # make the cache folder play nice with other tools devs tend to use (e.g. git).
    # But since pytest doesn't recreate these files if the .pytest_cache folder already exists,
    # we'll copy them over as a way to protect against future bugs where a certain tool
    # may need those files to exist to work properly (their combined file size is negligible)
    static_files_to_copy = [
        ".gitignore",
        "CACHEDIR.TAG",
        "README.md",
    ]

    # Copy over the static files. These files never change, so only copy them
    # if they don't already exist in the new cache
    for static_file in static_files_to_copy:
        source_file = pytest_cache_dir_to_merge_from / static_file
        if not source_file.is_file():
            continue

        dest_file = pytest_cache_dir_to_merge_into / static_file
        if not dest_file.exists():
            copy_file(source_file, dest_file)

    # Handle the v/cache/lastfailed file
    _merge_lastfailed_files(
        pytest_cache_dir_to_merge_from, pytest_cache_dir_to_merge_into
    )

    _merge_additional_failures_files(
        pytest_cache_dir_to_merge_from, pytest_cache_dir_to_merge_into
    )


def _merge_lastfailed_files(source_pytest_cache: Path, dest_pytest_cache: Path) -> None:
    # [FUNCTION DOC] HELPER: Merges the lastfailed json file of the source cache into the
    # destination. Does nothing if the source file is missing; copies it if the destination file
    # is missing; otherwise writes the merged content to the destination file.
    # Simple cases where one of the files doesn't exist
    source_lastfailed_file = source_pytest_cache / LASTFAILED_FILE_PATH
    dest_lastfailed_file = dest_pytest_cache / LASTFAILED_FILE_PATH

    if not source_lastfailed_file.exists():
        return
    if not dest_lastfailed_file.exists():
        copy_file(source_lastfailed_file, dest_lastfailed_file)
        return

    # Both files exist, so we need to merge them
    from_lastfailed = load_json_file(source_lastfailed_file)
    to_lastfailed = load_json_file(dest_lastfailed_file)
    merged_content = _merged_lastfailed_content(from_lastfailed, to_lastfailed)

    # Save the results
    write_json_file(dest_lastfailed_file, merged_content)


def _merged_lastfailed_content(
    from_lastfailed: dict[str, bool], to_lastfailed: dict[str, bool]
) -> dict[str, bool]:
    # [FUNCTION DOC] HELPER: Returns to_lastfailed with all keys of from_lastfailed added
    # (existing values win). Removes the empty-string placeholder key when more than one entry
    # remains. Mutates and returns the to_lastfailed mapping.
    """
    The lastfailed files are dictionaries where the key is the test identifier.
    Each entry's value appears to always be `true`, but let's not count on that.
    An empty dictionary is represented with a single value with an empty string as the key.
    """

    # If an entry in from_lastfailed doesn't exist in to_lastfailed, add it and it's value
    for key in from_lastfailed:
        if key not in to_lastfailed:
            to_lastfailed[key] = from_lastfailed[key]

    if len(to_lastfailed) > 1:
        # Remove the empty entry if it exists since we have actual entries now
        if "" in to_lastfailed:
            del to_lastfailed[""]

    return to_lastfailed


def _merge_additional_failures_files(
    source_pytest_cache: Path, dest_pytest_cache: Path
) -> None:
    # [FUNCTION DOC] HELPER: Merges the previous failures additional json file (a list) of the
    # source cache into the destination: skipped if the source is missing, copied if the
    # destination is missing, otherwise the destination becomes the de-duplicated union (order
    # not preserved).
    # Simple cases where one of the files doesn't exist
    source_lastfailed_file = (
        source_pytest_cache / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL
    )
    dest_lastfailed_file = dest_pytest_cache / TD_HEURISTIC_PREVIOUSLY_FAILED_ADDITIONAL

    if not source_lastfailed_file.exists():
        return
    if not dest_lastfailed_file.exists():
        copy_file(source_lastfailed_file, dest_lastfailed_file)
        return

    # Both files exist, so we need to merge them
    from_lastfailed = load_json_file(source_lastfailed_file)
    to_lastfailed = load_json_file(dest_lastfailed_file)
    merged_content = list(set(from_lastfailed + to_lastfailed))

    # Save the results
    write_json_file(dest_lastfailed_file, merged_content)
