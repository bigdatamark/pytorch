# [MODULE DOC] CI maintenance script that closes "disabled test" GitHub issues whose test no
# longer exists. It downloads the condensed disabled-tests JSON, fetches logs of recent
# viable/strict test jobs from ClickHouse/S3, and checks each disabled test against those logs
# and a 7-day test-run table. Tests found in neither are reported and (unless --dry-run) the
# issue gets a closing comment and is closed via the GitHub REST API (needs GITHUB_TOKEN).
# Import-time side effects: temporarily modifies sys.path to import tools.testing.clickhouse.
# Run as a script; exits non-zero if any issue failed to close.
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests
from gitutils import retries_decorator


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.clickhouse import query_clickhouse


sys.path.pop(0)


LOGS_QUERY = """
with
    shas as (
        SELECT
            distinct
            push.head_commit.id as sha
        FROM
            -- Not bothering with final here
            default.push
        WHERE
            push.ref = 'refs/heads/viable/strict'
            AND push.repository.'full_name' = 'pytorch/pytorch'
        ORDER BY
            push.head_commit.'timestamp' desc
        LIMIT
            5
    )
select
    id,
    name
from
    default.workflow_job j final
    join shas on shas.sha = j.head_sha
where
    j.id in (select id from materialized_views.workflow_job_by_head_sha where head_sha in (select sha from shas))
    and j.name like '% / test%'
    and j.name not like '%rerun_disabled_tests%'
    and j.name not like '%mem_leak_check%'
"""

TEST_EXISTS_QUERY = """
select
    name
from
    default.test_run_s3
where
    name::String like {name: String}
    and classname like {classname: String}
    and time_inserted > CURRENT_TIMESTAMP() - INTERVAL 7 DAY
limit 1
"""

CLOSING_COMMENT = (
    "I cannot find any mention of this test in the database for the past 7 days "
    "or in the logs for the past 5 commits on viable/strict.  Closing this "
    "issue as it is highly likely that this test has either been renamed or "
    "removed.  If you think this is a false positive, please feel free to "
    "re-open this issue."
)

DISABLED_TESTS_JSON = (
    "https://ossci-metrics.s3.amazonaws.com/disabled-tests-condensed.json"
)


@retries_decorator()
def query_db(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Runs a parametrized ClickHouse query and returns the result rows as
    # a list of mappings. Retried on failure by the retries decorator; errors propagate after
    # retries are exhausted.
    return query_clickhouse(query, params)


def parse_args() -> Any:
    # [FUNCTION DOC] HELPER: Parses command line arguments; only supports a --dry-run flag that
    # makes the script list candidate tests without closing any issue. Returns the namespace.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the tests.",
    )
    return parser.parse_args()


def download_log_worker(temp_dir: str, id: int, name: str) -> None:
    # [FUNCTION DOC] HELPER: Pool worker that downloads the raw log of job id from the public S3
    # bucket and writes it to a new file in temp_dir named after the job name (slashes replaced
    # by underscores) and id. The file is opened in exclusive-create mode, so an existing file
    # raises an error. Returns nothing; no HTTP status check is done on the download.
    url = f"https://ossci-raw-job-status.s3.amazonaws.com/log/{id}"
    data = requests.get(url).text
    with open(f"{temp_dir}/{name.replace('/', '_')} {id}.txt", "x") as f:
        f.write(data)


def printer(item: tuple[str, tuple[int, str, list[Any]]], extra: str) -> None:
    # [FUNCTION DOC] HELPER: Prints one line for a disabled-test entry (test name mapped to
    # issue number, issue link, extra data): the link, the test name in fixed-width columns,
    # then the extra text such as the reason the test was or was not found.
    test, (_, link, _) = item
    print(f"{link:<55} {test:<120} {extra}")


def close_issue(num: int) -> None:
    # [FUNCTION DOC] Posts the standard closing comment on pytorch/pytorch issue num and then
    # sets the issue state to closed, using the GITHUB_TOKEN environment variable for auth.
    # Raises RuntimeError if the comment call does not return 201 or the close call does not
    # return 200 (a missing token raises a key error). Side effect: mutates the GitHub issue.
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
    }
    response = requests.post(
        f"https://api.github.com/repos/pytorch/pytorch/issues/{num}/comments",
        data=json.dumps({"body": CLOSING_COMMENT}),
        headers=headers,
    )
    if response.status_code != 201:
        raise RuntimeError(f"Failed to comment on issue {num}: {response.text}")
    response = requests.patch(
        f"https://api.github.com/repos/pytorch/pytorch/issues/{num}",
        data=json.dumps({"state": "closed"}),
        headers=headers,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Failed to close issue {num}: {response.text}")


def check_if_exists(
    item: tuple[str, tuple[int, str, list[str]]], all_logs: list[str]
) -> tuple[bool, str]:
    # [FUNCTION DOC] Decides whether a disabled test still exists. Parses the test name into
    # test function name and class name (last dotted component); if the format is wrong returns
    # (False, "poorly formed"). Searches all log texts for the issue link or the class and test
    # pair; if found returns (True, "found in logs"). Otherwise queries the DB for a recent run
    # of the test: (True, "found in DB") or (False, "not found"). Performs a network query.
    test, (_, link, _) = item
    # Test names should look like `test_a (module.path.classname)`
    reg = re.match(r"(\S+) \((\S*)\)", test)
    if reg is None:
        return False, "poorly formed"

    name = reg[1]
    classname = reg[2].split(".")[-1]

    # Check if there is any mention of the link or the test name in the logs.
    # The link usually shows up in the skip reason.
    present = False
    for log in all_logs:
        if link in log:
            present = True
            break
        if f"{classname}::{name}" in log:
            present = True
            break
    if present:
        return True, "found in logs"

    # Query DB to see if the test is there
    count = query_db(
        TEST_EXISTS_QUERY, {"name": f"{name}%", "classname": f"{classname}%"}
    )
    if len(count) == 0:
        return False, "not found"
    return True, "found in DB"


if __name__ == "__main__":
    args = parse_args()
    # Entry point flow: load disabled tests, download logs of recent jobs with a process pool,
    # sanity check the log count (must exceed 200 and equal job count, else error), classify each
    # disabled test, print the ones to close, then close them unless dry run.
    disabled_tests_json = json.loads(requests.get(DISABLED_TESTS_JSON).text)

    all_logs = []
    jobs = query_db(LOGS_QUERY, {})
    with tempfile.TemporaryDirectory() as temp_dir:
        pool = mp.Pool(20)
        for job in jobs:
            id = job["id"]
            name = job["name"]
            pool.apply_async(download_log_worker, args=(temp_dir, id, name))
        pool.close()
        pool.join()

        for filename in os.listdir(temp_dir):
            with open(f"{temp_dir}/{filename}") as f:
                all_logs.append(f.read())

    # If its less than 200 something definitely went wrong.
    if len(all_logs) <= 200:
        raise AssertionError(
            f"Expected more than 200 logs, got {len(all_logs)}. Something went wrong."
        )
    if len(all_logs) != len(jobs):
        raise AssertionError(
            f"Number of logs ({len(all_logs)}) does not match number of jobs ({len(jobs)})"
        )

    to_be_closed = []
    for item in disabled_tests_json.items():
        exists, reason = check_if_exists(item, all_logs)
        printer(item, reason)
        if not exists:
            to_be_closed.append(item)

    print(f"There are {len(to_be_closed)} issues that will be closed:")
    for item in to_be_closed:
        printer(item, "")

    if args.dry_run:
        print("dry run, not actually closing")
    else:
        failed = False
        for item in to_be_closed:
            _, (num, _, _) = item
            try:
                close_issue(num)
            except RuntimeError as e:
                print(e)
                failed = True
        if failed:
            sys.exit(1)
