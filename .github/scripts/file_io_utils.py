# [MODULE DOC] Small file and S3 helpers shared by CI scripts: zip and unzip a folder, create
# directories, copy files, read and write JSON files, sanitize text into an S3-safe key, upload
# a file to S3 and download all S3 objects under a prefix. Requires boto3 at import time (the
# import fails if it is missing); AWS credentials are only needed when the S3 functions run.
import json
import re
import shutil
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import]


def zip_folder(folder_to_zip: Path, dest_file_base_name: Path) -> Path:
    # [FUNCTION DOC] Archives folder_to_zip into a zip file whose name is dest_file_base_name
    # with the zip extension (a trailing zip suffix on the argument is dropped first so it is
    # not doubled). Creates the destination parent directory, prints progress, and returns the
    # path of the created archive. Raises on filesystem errors.
    """
    Returns the path to the resulting zip file, with the appropriate extension added if needed
    """
    # shutil.make_archive will append .zip to the dest_file_name, so we need to remove it if it's already there
    if dest_file_base_name.suffix == ".zip":
        dest_file_base_name = dest_file_base_name.with_suffix("")

    ensure_dir_exists(dest_file_base_name.parent)

    print(f"Zipping {folder_to_zip}\n     to {dest_file_base_name}")
    # Convert to string because shutil.make_archive doesn't like Path objects
    return Path(shutil.make_archive(str(dest_file_base_name), "zip", folder_to_zip))


def unzip_folder(zip_file_path: Path, unzip_to_folder: Path) -> None:
    # [FUNCTION DOC] Extracts the zip archive into unzip_to_folder (created by the unpack
    # step if needed) and prints progress. Returns nothing.
    # KNOWN ISSUE: the docstring claims it returns the path to the unzipped folder but it
    # returns None.
    """
    Returns the path to the unzipped folder
    """
    print(f"Unzipping {zip_file_path}")
    print(f"       to {unzip_to_folder}")
    shutil.unpack_archive(zip_file_path, unzip_to_folder, "zip")


def ensure_dir_exists(dir: Path) -> None:
    # [FUNCTION DOC] HELPER: Creates the directory and any missing parents; no error if it
    # already exists.
    dir.mkdir(parents=True, exist_ok=True)


def copy_file(source_file: Path, dest_file: Path) -> None:
    # [FUNCTION DOC] Copies the contents of source_file to dest_file, creating the destination
    # parent directory first. Overwrites an existing destination; raises if the source is
    # missing.
    ensure_dir_exists(dest_file.parent)
    shutil.copyfile(source_file, dest_file)


def load_json_file(file_path: Path) -> Any:
    # [FUNCTION DOC] Reads the file and returns the parsed JSON value; raises if the file is
    # missing or not valid JSON.
    """
    Returns the deserialized json object
    """
    with open(file_path) as f:
        return json.load(f)


def write_json_file(file_path: Path, content: Any) -> None:
    # [FUNCTION DOC] Serializes content as JSON with indent 2 into file_path, creating parent
    # directories and overwriting any existing file. Raises if content is not serializable.
    dir = file_path.parent
    ensure_dir_exists(dir)

    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)


def sanitize_for_s3(text: str) -> str:
    # [FUNCTION DOC] Returns text with every character other than ASCII letters, digits,
    # underscore and dash replaced by an underscore. Pure function.
    """
    S3 keys can only contain alphanumeric characters, underscores, and dashes.
    This function replaces all other characters with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def upload_file_to_s3(file_name: Path, bucket: str, key: str) -> None:
    # [FUNCTION DOC] Uploads the local file to the given S3 bucket and key using a default boto3
    # client (credentials from the environment) and prints progress. Raises on AWS errors.
    print(f"Uploading {file_name}")
    print(f"       to s3://{bucket}/{key}")

    boto3.client("s3").upload_file(
        str(file_name),
        bucket,
        key,
    )


def download_s3_objects_with_prefix(
    bucket_name: str, prefix: str, download_folder: Path
) -> list[Path]:
    # [FUNCTION DOC] Downloads every object whose key starts with prefix from the bucket into
    # download_folder, preserving the key as a relative path (parent directories are created).
    # Uses unsigned anonymous requests when no AWS credentials are available. Prints progress
    # and a note if nothing matched. Returns the list of local paths downloaded (possibly empty).
    # The artifacts bucket is public-read. If no credentials are available (e.g.
    # fork PRs that can't assume an OIDC role), fall back to anonymous/unsigned
    # requests so the (best-effort) download still works.
    if boto3.Session().get_credentials() is None:
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    else:
        s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    downloads = []

    for obj in bucket.objects.filter(Prefix=prefix):
        download_path = download_folder / obj.key

        ensure_dir_exists(download_path.parent)
        print(f"Downloading s3://{bucket.name}/{obj.key}")
        print(f"         to {download_path}")

        s3.Object(bucket.name, obj.key).download_file(str(download_path))
        downloads.append(download_path)

    if len(downloads) == 0:
        print(
            f"There were no files matching the prefix `{prefix}` in bucket `{bucket.name}`"
        )

    return downloads
