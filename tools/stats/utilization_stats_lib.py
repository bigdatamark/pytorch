# [MODULE DOC]
# Data models shared by the utilization monitor and the utilization upload script:
# dataclasses for raw monitor output (stats, metadata, GPU usage, records), the
# database row models for metadata, segments and time series, a workflow info
# holder, and two small helpers for the data model version and current timestamp.
# Requires the dataclasses_json package at import time.

from dataclasses import dataclass, field
from datetime import datetime

#  pyrefly: ignore [missing-import]
from dataclasses_json import DataClassJsonMixin  # type: ignore[import-not-found]


_DATA_MODEL_VERSION = 1.5


# data model for test log usage
@dataclass
class UtilizationStats:
    # [CLASS DOC]
    # Aggregate of one measured quantity over an interval: average, maximum and the
    # raw sample list; all fields are optional and empty by default.
    avg: float | None = None
    max: float | None = None
    raw: list[float] | None = None


@dataclass
class UtilizationMetadata(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC]
    # First record emitted by the monitor (level metadata): workflow and job
    # identifiers and names, collect interval, data model version, start time, and
    # optional gpu count, cpu count, gpu type and error text. Serializable to json.
    level: str
    workflow_id: str
    job_id: str
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: float
    start_at: int
    gpu_count: int | None = None
    cpu_count: int | None = None
    gpu_type: str | None = None
    error: str | None = None


@dataclass
class GpuUsage(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC]
    # Aggregated usage of one GPU over an interval: uuid, stats for utilization,
    # memory utilization, allocated memory percent and value, and total memory.
    uuid: str | None = None
    util_percent: UtilizationStats | None = None
    mem_util_percent: UtilizationStats | None = None
    allocated_mem_percent: UtilizationStats | None = None
    allocated_mem_value: UtilizationStats | None = None
    total_mem_value: float | None = None


@dataclass
class RecordData(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC]
    # Payload of a monitor record: cpu stats, memory stats and an optional list of
    # per-GPU usage.
    cpu: UtilizationStats | None = None
    memory: UtilizationStats | None = None
    gpu_usage: list[GpuUsage] | None = None


@dataclass
class UtilizationRecord(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC]
    # One periodic monitor record (level record): timestamp, optional data payload,
    # names of observed commands, error text, how long collection took and logs.
    level: str
    timestamp: int
    data: RecordData | None = None
    cmd_names: list[str] | None = None
    error: str | None = None
    log_duration: str | None = None
    logs: list[str] | None = None


# the db schema related to this is:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_metadata_schema.sql
@dataclass
class OssCiSegmentV1(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    # [CLASS DOC]
    # A named time segment (for example a test run) within a job, with start and end
    # times and a string mapping of extra info; stored in the metadata row.
    level: str
    name: str
    start_at: int
    end_at: int
    extra_info: dict[str, str]


@dataclass
class OssCiUtilizationMetadataV1:
    # [CLASS DOC]
    # Database row model for a job's utilization metadata: creation time, repo,
    # workflow and job identity, collect interval, data model version, hardware
    # counts and type, start and end times, the segments and tags.
    # KNOWN ISSUE: data_model_version is a string here but a float in the monitor's
    # metadata record.
    created_at: int
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: str
    gpu_count: int
    cpu_count: int
    gpu_type: str
    start_at: int
    end_at: int
    segments: list[OssCiSegmentV1]
    tags: list[str] = field(default_factory=list)


# this data model is for the time series data:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_time_series_schema.sql
@dataclass
class OssCiUtilizationTimeSeriesV1:
    # [CLASS DOC]
    # Database row model for one time series data point: creation time, type, tags,
    # time stamp, repo, workflow and job identity and the payload as a json string.
    created_at: int
    type: str
    tags: list[str]
    time_stamp: int
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    json_data: str


def getDataModelVersion() -> float:
    # [FUNCTION DOC] Returns the current data model version constant.
    return _DATA_MODEL_VERSION


def getTsNow() -> int:
    # [FUNCTION DOC] Returns the current time as whole seconds since the epoch.
    ts = datetime.now().timestamp()
    return int(ts)


@dataclass
class WorkflowInfo:
    # [CLASS DOC]
    # Identity of a CI job: workflow run id and name, job id and name, run attempt
    # and repo (defaults to pytorch/pytorch).
    workflow_run_id: int
    workflow_name: str
    job_id: int
    run_attempt: int
    job_name: str
    repo: str = "pytorch/pytorch"
