# [MODULE DOC] Tests for tools.stats.upload_metrics (emit_metric, add_global_metric) and
# remove_nan_inf from upload_stats_lib, with boto3 resource creation mocked so emitted metrics
# are captured in memory. Import time: temporarily puts the repo root first on sys.path for
# the imports, then removes it, and defines default CI environment constants.
from __future__ import annotations

import gzip
import inspect
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.upload_metrics import add_global_metric, emit_metric, global_metrics
from tools.stats.upload_stats_lib import get_s3_resource, remove_nan_inf


sys.path.remove(str(REPO_ROOT))

# default values
REPO = "some/repo"
BUILD_ENV = "cuda-10.2"
TEST_CONFIG = "test-config"
WORKFLOW = "some-workflow"
JOB = "some-job"
RUN_ID = 56
RUN_NUMBER = 123
RUN_ATTEMPT = 3
PR_NUMBER = 6789
JOB_ID = 234
JOB_NAME = "some-job-name"


@mock.patch("boto3.resource")
class TestUploadStats(unittest.TestCase):
    # [CLASS DOC] Test case for metric emission. The class decorator patches the boto3 resource
    # factory for every test, passing the mock as an extra argument. State: emitted_metric
    # holds the last decoded uploaded payload, or a marker mapping saying nothing was emitted.
    emitted_metric: dict[str, Any] = {"did_not_emit": True}

    def mock_put_item(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Stand-in for the storage object put call. Decompresses and
        # decodes the gzip JSON payload given as the body keyword and stores it in emitted_metric.
        # Utility for mocking putting items into s3.  THis will save the emitted
        # metric so tests can check it
        self.emitted_metric = json.loads(
            gzip.decompress(kwargs["Body"]).decode("utf-8")
        )

    # Before each test, set the env vars to their default values
    def setUp(self) -> None:
        # [FUNCTION DOC] Clears the cached storage resource and global metrics, then patches
        # the process environment to a clean set of CI variables (repo, workflow, job, run ids).
        # KNOWN ISSUE: the environment patcher is started but never stopped or registered for
        # cleanup, so the patched environment leaks into later tests in the process.
        get_s3_resource.cache_clear()
        global_metrics.clear()

        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
                "TEST_CONFIG": TEST_CONFIG,
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_WORKFLOW": WORKFLOW,
                "GITHUB_JOB": JOB,
                "GITHUB_RUN_ID": str(RUN_ID),
                "GITHUB_RUN_NUMBER": str(RUN_NUMBER),
                "GITHUB_RUN_ATTEMPT": str(RUN_ATTEMPT),
                "JOB_ID": str(JOB_ID),
                "JOB_NAME": str(JOB_NAME),
            },
            clear=True,  # Don't read any preset env vars
        ).start()

    def test_emits_default_and_given_metrics(self, mock_resource: Any) -> None:
        # [FUNCTION DOC] Emits a metric and checks the uploaded payload contains the caller
        # file, module and function names, the CI environment fields and the given info.
        metric = {
            "some_number": 123,
            "float_number": 32.34,
        }

        # Querying for this instead of hard coding it b/c this will change
        # based on whether we run this test directly from python or from
        # pytest
        current_module = inspect.getmodule(inspect.currentframe()).__name__  # type: ignore[union-attr]

        emit_should_include = {
            "metric_name": "metric_name",
            "calling_file": "test_upload_stats_lib.py",
            "calling_module": current_module,
            "calling_function": "test_emits_default_and_given_metrics",
            "repo": REPO,
            "workflow": WORKFLOW,
            "build_environment": BUILD_ENV,
            "job": JOB,
            "test_config": TEST_CONFIG,
            "run_id": RUN_ID,
            "run_number": RUN_NUMBER,
            "run_attempt": RUN_ATTEMPT,
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
            "info": metric,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, **emit_should_include},
        )

    def test_when_global_metric_specified_then_it_emits_it(
        self, mock_resource: Any
    ) -> None:
        # [FUNCTION DOC] Checks a registered global metric is merged into the emitted info.
        metric = {
            "some_number": 123,
        }

        global_metric_name = "global_metric"
        global_metric_value = "global_value"

        add_global_metric(global_metric_name, global_metric_value)

        emit_should_include = {
            **metric,
            global_metric_name: global_metric_value,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
        )

    def test_when_local_and_global_metric_specified_then_global_is_overridden(
        self, mock_resource: Any
    ) -> None:
        # [FUNCTION DOC] Checks a per-call metric value overrides a global metric of the same
        # name in the emitted info.
        global_metric_name = "global_metric"
        global_metric_value = "global_value"
        local_override = "local_override"

        add_global_metric(global_metric_name, global_metric_value)

        metric = {
            "some_number": 123,
            global_metric_name: local_override,
        }

        emit_should_include = {
            **metric,
            global_metric_name: local_override,
        }

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
        )

    def test_when_optional_envvar_set_to_actual_value_then_emit_vars_emits_it(
        self, mock_resource: Any
    ) -> None:
        # [FUNCTION DOC] Sets the optional pull request number variable and checks it appears
        # in the emitted payload.
        metric = {
            "some_number": 123,
        }

        emit_should_include = {
            "info": {**metric},
            "pr_number": PR_NUMBER,
        }

        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": str(PR_NUMBER),
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, **emit_should_include},
        )

    def test_when_optional_envvar_set_to_a_empty_str_then_emit_vars_ignores_it(
        self, mock_resource: Any
    ) -> None:
        # [FUNCTION DOC] Sets the optional pull request number variable to an empty string
        # and checks the metric is still emitted but without a pull request number field.
        metric = {"some_number": 123}

        emit_should_include: dict[str, Any] = metric.copy()

        # Github Actions defaults some env vars to an empty string
        default_val = ""
        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": default_val,
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertEqual(
            self.emitted_metric,
            {**self.emitted_metric, "info": emit_should_include},
            f"Metrics should be emitted when an option parameter is set to '{default_val}'",
        )
        self.assertFalse(
            self.emitted_metric.get("pr_number"),
            f"Metrics should not include optional item 'pr_number' when it's envvar is set to '{default_val}'",
        )

    def test_no_metrics_emitted_if_required_env_var_not_set(
        self, mock_resource: Any
    ) -> None:
        # [FUNCTION DOC] Replaces the environment with only a few variables and checks nothing
        # is uploaded because required ones are missing.
        metric = {"some_number": 123}

        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
            },
            clear=True,
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertTrue(self.emitted_metric["did_not_emit"])

    def test_no_metrics_emitted_if_required_env_var_set_to_empty_string(
        self, mock_resource: Any
    ) -> None:
        # [FUNCTION DOC] Sets a required variable (job) to an empty string and checks nothing
        # is uploaded.
        metric = {"some_number": 123}

        mock.patch.dict(
            "os.environ",
            {
                "GITHUB_JOB": "",
            },
        ).start()

        mock_resource.return_value.Object.return_value.put = self.mock_put_item

        emit_metric("metric_name", metric)

        self.assertTrue(self.emitted_metric["did_not_emit"])

    def test_remove_nan_inf(self, _mocked_resource: Any) -> None:
        # [FUNCTION DOC] Checks remove_nan_inf turns infinite and not-a-number floats, also
        # nested in mappings and lists, into the strings inf and nan so JSON output is strict,
        # while plain JSON dumping of the raw input yields non-standard Infinity and NaN.
        checks = [
            (float("inf"), '"inf"', "Infinity"),
            (float("nan"), '"nan"', "NaN"),
            ({1: float("inf")}, '{"1": "inf"}', '{"1": Infinity}'),
            ([float("nan")], '["nan"]', "[NaN]"),
            ({1: [float("nan")]}, '{"1": ["nan"]}', '{"1": [NaN]}'),
        ]

        for input, clean, unclean in checks:
            clean_output = json.dumps(remove_nan_inf(input))
            unclean_output = json.dumps(input)
            self.assertEqual(
                clean_output,
                clean,
                f"Expected {clean} when input is {unclean}, got {clean_output}",
            )
            self.assertEqual(
                unclean_output,
                unclean,
                f"Expected {unclean} when input is {unclean}, got {unclean_output}",
            )


if __name__ == "__main__":
    unittest.main()
