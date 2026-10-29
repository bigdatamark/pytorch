# [MODULE DOC] Helpers that upload test artifacts from a CI job to S3 while tests run:
# zipped reports, logs and JSON files, concatenated logs for the log classifier, per-xml
# JSON conversions guarded by file locks, and ad hoc failure records. Rate limited by the
# module-level LAST_UPDATED timestamp (20 minutes). No import-time side effects.

import glob
import gzip
import json
import os
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from tools.stats.upload_test_stats import parse_xml_report


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAST_UPDATED = 0.0
LOG_BUCKET_PREFIX = "temp_logs"


@lru_cache(maxsize=1)
def get_s3_resource() -> Any:
    # [FUNCTION DOC] Returns a cached boto3 S3 client, importing boto3 lazily.
    import boto3  # type: ignore[import]

    return boto3.client("s3")


def zip_artifact(file_name: str, paths: list[str]) -> None:
    # [FUNCTION DOC] Writes a zip file containing files matching each glob pattern (relative to
    # REPO_ROOT, recursive), stored under their path relative to REPO_ROOT.
    """Zip the files in the paths listed into file_name. The paths will be used
    in a glob and should be relative to REPO_ROOT."""

    with zipfile.ZipFile(file_name, "w") as f:
        for path in paths:
            for file in glob.glob(f"{REPO_ROOT}/{path}", recursive=True):
                f.write(file, os.path.relpath(file, REPO_ROOT))


def concated_logs() -> str:
    # [FUNCTION DOC] Reads every log under the test-reports folder and returns one string with
    # a header line per file and each log line prefixed by a fake timestamp for the classifier.
    """Concatenate all the logs in the test-reports directory into a single string."""
    logs = []
    for log_file in glob.glob(
        f"{REPO_ROOT}/test/test-reports/**/*.log", recursive=True
    ):
        logs.append(f"=== {log_file} ===")
        with open(log_file) as f:
            # For every line, prefix with fake timestamp for log classifier
            for line in f:
                line = line.rstrip("\n")  # Remove any trailing newline
                logs.append(f"2020-01-01T00:00:00.0000000Z {line}")
    return "\n".join(logs)


def upload_to_s3_artifacts(failed: bool) -> None:
    # [FUNCTION DOC] Zips xml and csv reports, logs and json reports and uploads them to the
    # artifacts bucket under a prefix built from the run id, attempt and file suffix, then
    # writes a marker object. Does nothing (prints a message) if required environment
    # variables are missing. If failed and a job id is set, also uploads gzipped concatenated
    # logs. Side effects: files, S3 uploads.
    # KNOWN ISSUE: the json archive upload passes the logs zip path as the file, so the logs
    # archive is uploaded twice and the json archive is never uploaded.
    """Upload the file to S3."""
    workflow_id = os.environ.get("GITHUB_RUN_ID")
    workflow_run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    file_suffix = os.environ.get("ARTIFACTS_FILE_SUFFIX")
    job_id = os.environ.get("JOB_ID")
    if not workflow_id or not workflow_run_attempt or not file_suffix:
        print(
            "GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, or ARTIFACTS_FILE_SUFFIX not set, not uploading"
        )
        return

    test_reports_zip_path = f"{REPO_ROOT}/test-reports-{file_suffix}.zip"
    zip_artifact(
        test_reports_zip_path,
        ["test/test-reports/**/*.xml", "test/test-reports/**/*.csv"],
    )
    test_logs_zip_path = f"{REPO_ROOT}/logs-{file_suffix}.zip"
    zip_artifact(test_logs_zip_path, ["test/test-reports/**/*.log"])
    jsons_zip_path = f"{REPO_ROOT}/test-jsons-{file_suffix}.zip"
    zip_artifact(jsons_zip_path, ["test/test-reports/**/*.json"])

    s3_prefix = f"pytorch/pytorch/{workflow_id}/{workflow_run_attempt}/artifact"
    get_s3_resource().upload_file(
        test_reports_zip_path,
        "gha-artifacts",
        f"{s3_prefix}/{Path(test_reports_zip_path).name}",
    )
    get_s3_resource().upload_file(
        test_logs_zip_path,
        "gha-artifacts",
        f"{s3_prefix}/{Path(test_logs_zip_path).name}",
    )
    get_s3_resource().upload_file(
        test_logs_zip_path,
        "gha-artifacts",
        f"{s3_prefix}/{Path(jsons_zip_path).name}",
    )
    get_s3_resource().put_object(
        Body=b"",
        Bucket="gha-artifacts",
        Key=f"workflows_failing_pending_upload/{workflow_id}.txt",
    )
    if job_id and failed:
        logs = concated_logs()
        # Put logs into bucket so log classifier can access them. We cannot get
        # the actual GH logs so this will have to be a proxy.
        print(f"Uploading logs for {job_id} to S3")
        get_s3_resource().put_object(
            Body=gzip.compress(logs.encode("utf-8")),
            Bucket="gha-artifacts",
            Key=f"{LOG_BUCKET_PREFIX}/{job_id}",
            ContentType="text/plain",
            ContentEncoding="gzip",
        )


def zip_and_upload_artifacts(failed: bool) -> None:
    # [FUNCTION DOC] Uploads artifacts if a test failed or at least 20 minutes passed since
    # the last successful upload. Errors are printed rather than raised; prints elapsed time.
    # Updates global LAST_UPDATED only on success.
    # not thread safe but correctness of the LAST_UPDATED var doesn't really
    # matter for this
    # Upload if a test failed or every 20 minutes
    global LAST_UPDATED

    if failed or time.time() - LAST_UPDATED > 20 * 60:
        start = time.time()
        try:
            upload_to_s3_artifacts(failed=failed)
            LAST_UPDATED = time.time()
        except Exception as e:
            print(f"Failed to upload artifacts: {e}")
        print(f"Uploading artifacts took {time.time() - start:.2f} seconds")


def trigger_upload_test_stats_intermediate_workflow() -> None:
    # [FUNCTION DOC] Dispatches the intermediate test stats workflow through the GitHub API
    # using the GITHUB_TOKEN and run environment variables, and prints the response.
    # KNOWN ISSUE: per the inline comment, the token cannot trigger workflows, so this is
    # currently unused.
    import requests

    # The GITHUB_TOKEN cannot trigger workflow so this isn't used for now
    print("Triggering upload_test_stats_intermediate workflow")
    x = requests.post(
        "https://api.github.com/repos/pytorch/pytorch/actions/workflows/upload_test_stats_intermediate.yml/dispatches",  # @lint-ignore
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
        },
        json={
            "ref": "main",
            "inputs": {
                "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
                "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            },
        },
    )
    print(x.text)


def parse_xml_and_upload_json() -> None:
    # [FUNCTION DOC] For each xml report without a sibling json file: takes a non-blocking
    # file lock, parses the xml into test case records, uploads them gzipped as line-delimited
    # json to S3 and writes the json locally to mark it done. Skips files being processed by
    # another process. Errors are printed, not raised.
    """
    Parse xml test reports that do not yet have a corresponding json report
    uploaded to s3, and upload the json reports to s3. Use filelock to avoid
    uploading the same file from multiple processes.
    """
    try:
        job_id: int | None = int(os.environ.get("JOB_ID", 0))
        if job_id == 0:
            job_id = None
    except (ValueError, TypeError):
        job_id = None

    try:
        for xml_file in glob.glob(
            f"{REPO_ROOT}/test/test-reports/**/*.xml", recursive=True
        ):
            xml_path = Path(xml_file)
            json_file = xml_path.with_suffix(".json")
            lock = FileLock(str(json_file) + ".lock")

            try:
                lock.acquire(timeout=0)  # immediately fails if already locked
                if json_file.exists():
                    continue  # already uploaded
                test_cases = parse_xml_report(
                    "testcase",
                    xml_path,
                    int(os.environ.get("GITHUB_RUN_ID", "0")),
                    int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
                    job_id,
                )
                line_by_line_jsons = "\n".join([json.dumps(tc) for tc in test_cases])

                gzipped = gzip.compress(line_by_line_jsons.encode("utf-8"))
                s3_key = (
                    json_file.relative_to(REPO_ROOT / "test/test-reports")
                    .as_posix()
                    .replace("/", "_")
                )

                get_s3_resource().put_object(
                    Body=gzipped,
                    Bucket="gha-artifacts",
                    Key=f"test_jsons_while_running/{os.environ.get('GITHUB_RUN_ID')}/{job_id}/{s3_key}",
                    ContentType="application/json",
                    ContentEncoding="gzip",
                )

                # We don't need to save the json file locally, but doing so lets us
                # track which ones have been uploaded already. We could probably also
                # check S3
                with open(json_file, "w") as f:
                    f.write(line_by_line_jsons)
            except Timeout:
                continue  # another process is working on this file
            finally:
                if lock.is_locked:
                    lock.release()
    except Exception as e:
        print(f"Failed to parse and upload json test reports: {e}")


def upload_adhoc_failure_json(invoking_file: str, current_failure: str) -> None:
    # [FUNCTION DOC] Uploads a gzipped json record to S3 stating that a whole test file failed
    # without xml (likely a crash). Splits the failure string into class and test name. Needs
    # JOB_ID and GITHUB_RUN_ID; prints and returns if missing.
    """
    manually upload a json to s3 indicating that the entire test file failed
    since xml was probably not generated in this case
    """
    try:
        job_id = int(os.environ["JOB_ID"])
        workflow_id = int(os.environ["GITHUB_RUN_ID"])
    except Exception as e:
        print(f"Failed to get job_id or workflow_id: {e}")
        return

    split_failure = current_failure.split("::")
    if len(split_failure) >= 2:
        className = split_failure[-2]
        testName = split_failure[-1]
    else:
        testName = current_failure
        className = ""

    message = "The test file failed but pytest did not generate xml.  The most likely cause is a segfault"
    j = {
        "invoking_file": invoking_file,
        "file": f"{invoking_file}.py",
        "name": testName,
        "classname": className,
        "workflow_id": workflow_id,
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "job_id": job_id,
        "failure": {"message": message, "text": message},
    }
    gzipped = gzip.compress(json.dumps(j).encode("utf-8"))
    s3_key = f"{invoking_file.replace('/', '_')}_{os.urandom(8).hex()}.json"
    get_s3_resource().put_object(
        Body=gzipped,
        Bucket="gha-artifacts",
        Key=f"test_jsons_while_running/{workflow_id}/{job_id}/{s3_key}",
        ContentType="application/json",
        ContentEncoding="gzip",
    )
