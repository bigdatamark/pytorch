# [MODULE DOC] Tests for tools.alerts.create_alerts: whether JobStatus decides to alert for
# consecutive failures, and regex-based job name filtering. Defines mock job data constants.
from __future__ import annotations

from typing import Any
from unittest import main, TestCase

from tools.alerts.create_alerts import filter_job_names, JobStatus


JOB_NAME = "periodic / linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck / test (default, 2, 2, linux.4xlarge.nvidia.gpu)"
MOCK_TEST_DATA = [
    {
        "sha": "f02f3046571d21b48af3067e308a1e0f29b43af9",
        "id": 7819529276,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7819529276?check_suite_focus=true",  # @lint-ignore
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7819529276",
        "durationS": 14876,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 83818,
        "repo": "pytorch/pytorch",
    },
    {
        "sha": "d0d6b1f2222bf90f478796d84a525869898f55b6",
        "id": 7818399623,
        "conclusion": "failure",
        "htmlUrl": "https://github.com/pytorch/pytorch/runs/7818399623?check_suite_focus=true",  # @lint-ignore
        "logUrl": "https://ossci-raw-job-status.s3.amazonaws.com/log/7818399623",
        "durationS": 14882,
        "failureLine": "##[error]The action has timed out.",
        "failureContext": "",
        "failureCaptures": ["##[error]The action has timed out."],
        "failureLineNumber": 72821,
        "repo": "pytorch/pytorch",
    },
]


class TestGitHubPR(TestCase):
    # [CLASS DOC] Test case for the alert-creation helpers. Stateless.
    # Should fail when jobs are ? ? Fail Fail
    def test_alert(self) -> None:
        # [FUNCTION DOC] Builds a job status from two empty entries followed by two failed
        # runs and asserts it decides to alert.
        modified_data: list[Any] = [{}]
        modified_data.append({})
        modified_data.extend(MOCK_TEST_DATA)
        status = JobStatus(JOB_NAME, modified_data)
        self.assertTrue(status.should_alert())

    # test filter job names
    def test_job_filter(self) -> None:
        # [FUNCTION DOC] Asserts filter_job_names keeps all names for an empty or match-all
        # regex, keeps only matching names for specific patterns, returns nothing when none
        # match, and raises an error for a malformed regex.
        job_names = [
            "pytorch_linux_xenial_py3_6_gcc5_4_test",
            "pytorch_linux_xenial_py3_6_gcc5_4_test2",
        ]
        self.assertListEqual(
            filter_job_names(job_names, ""),
            job_names,
            "empty regex should match all jobs",
        )
        self.assertListEqual(filter_job_names(job_names, ".*"), job_names)
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*"), job_names)
        self.assertListEqual(
            filter_job_names(job_names, ".*xenial.*test2"),
            ["pytorch_linux_xenial_py3_6_gcc5_4_test2"],
        )
        self.assertListEqual(filter_job_names(job_names, ".*xenial.*test3"), [])
        self.assertRaises(
            Exception,
            lambda: filter_job_names(job_names, "["),
            msg="malformed regex should throw exception",
        )


if __name__ == "__main__":
    main()
