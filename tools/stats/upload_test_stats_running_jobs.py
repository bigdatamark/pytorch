# [MODULE DOC]
# Script that polls an S3 queue folder of workflow ids whose test stats upload is
# pending (marker objects under a prefix), moves each to a pending folder while
# uploading its additional info, and removes the marker afterwards. First retries
# the pending folder, then keeps polling the main folder forever.

import sys
import time
from functools import cache
from typing import Any

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_stats_lib import get_s3_resource


BUCKET_PREFIX = "workflows_failing_pending_upload"


@cache
def get_bucket() -> Any:
    # [FUNCTION DOC] HELPER: Returns the cached gha-artifacts S3 bucket object.
    return get_s3_resource().Bucket("gha-artifacts")


def delete_obj(key: str) -> None:
    # [FUNCTION DOC] HELPER: Deletes the object with the given key from the bucket
    # quietly; a missing key is not an error.
    # Does not raise error if key does not exist
    get_bucket().delete_objects(
        Delete={
            "Objects": [{"Key": key}],
            "Quiet": True,
        }
    )


def put_object(key: str) -> None:
    # [FUNCTION DOC] HELPER: Creates an empty marker object with the given key.
    get_bucket().put_object(
        Key=key,
        Body=b"",
    )


def do_upload(workflow_id: int) -> None:
    # [FUNCTION DOC] HELPER: Flushes stdout and uploads the additional info for
    # attempt 1 of the workflow run; errors propagate to the caller.
    workflow_attempt = 1
    # Flush stdout so that any errors in upload show up last in the logs.
    sys.stdout.flush()
    upload_additional_info(workflow_id, workflow_attempt)


def get_workflow_ids(pending: bool = False) -> list[int]:
    # [FUNCTION DOC] HELPER: Lists marker objects under the queue prefix (the pending
    # sub-folder if pending is set) and returns the workflow ids parsed from their
    # file names. Raises if a name is not numeric.
    # KNOWN ISSUE: the main prefix also matches objects in the pending sub-folder.
    prefix = f"{BUCKET_PREFIX}/{'pending/' if pending else ''}"
    objs = get_bucket().objects.filter(Prefix=prefix)
    return [int(obj.key.split("/")[-1].split(".")[0]) for obj in objs]


def read_s3(pending: bool = False) -> None:
    # [FUNCTION DOC] HELPER: Processing loop. For each queued workflow id it marks it
    # pending, removes the queue marker, uploads its info (failures are printed, not
    # raised) and removes the pending marker. With pending set it stops once the
    # folder is empty; otherwise it sleeps 60 seconds when empty and runs forever.
    while True:
        workflows = get_workflow_ids(pending)
        if not workflows:
            if pending:
                break
            # Wait for more stuff to show up
            print("Sleeping for 60 seconds")
            time.sleep(60)
        for workflow_id in workflows:
            print(f"Processing {workflow_id}")
            put_object(f"{BUCKET_PREFIX}/pending/{workflow_id}.txt")
            delete_obj(f"{BUCKET_PREFIX}/{workflow_id}.txt")
            try:
                do_upload(workflow_id)
            except Exception as e:
                print(f"Failed to upload {workflow_id}: {e}")
            delete_obj(f"{BUCKET_PREFIX}/pending/{workflow_id}.txt")


if __name__ == "__main__":
    # Workflows in the pending folder were previously in progress of uploading
    # but failed to complete, so we need to retry them.
    read_s3(pending=True)
    read_s3()
