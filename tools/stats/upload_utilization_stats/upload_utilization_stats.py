#!/usr/bin/env python3

# [MODULE DOC]
# Script that converts a job's raw utilization log (produced by the monitor; first
# line metadata, later lines records) into database models and uploads them to S3.
# SegmentGenerator derives python command segments, UtilizationDbConverter builds the
# metadata and time series rows, UploadUtilizationData orchestrates reading (from S3
# artifacts or a local file), conversion and upload, with helper file readers and
# argument parsing. Import-time side effect: edits sys.path; needs pandas.

import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import argparse
import json
import zipfile
from dataclasses import asdict
from typing import Any

import pandas as pd  # type: ignore[import]
from tools.stats.upload_stats_lib import download_s3_artifacts, upload_to_s3
from tools.stats.utilization_stats_lib import (
    getDataModelVersion,
    getTsNow,
    OssCiSegmentV1,
    OssCiUtilizationMetadataV1,
    OssCiUtilizationTimeSeriesV1,
    UtilizationMetadata,
    UtilizationRecord,
    WorkflowInfo,
)


TEST_USAGE_LOG_FILENAME = "usage_log.txt"
CMD_PYTHON_LEVEL = "CMD_PYTHON"
UTILIZATION_BUCKET = "ossci-utilization"
PYTORCH_REPO = "pytorch/pytorch"
JOB_TEST_ARTIFACT_PREFIX = "logs-test"


class SegmentGenerator:
    # [CLASS DOC]
    # Stateless helper that turns utilization records into segments, one per
    # continuous run of a python command name across record timestamps.
    """
    generates test segment from utilization records, currently it only generate segments on python commands level
    segment_delta_threshold is the threshold to determine if a segment is continuous or not, default is 60 seconds.
    """

    def generate(
        self, records: list[UtilizationRecord], segment_delta_threshold: int = 60
    ) -> list[OssCiSegmentV1]:
        # [FUNCTION DOC]
        # Flattens the records into (time, command) rows, keeps the distinct commands
        # starting with python, and for each finds windows of timestamps whose gaps
        # are within segment_delta_threshold seconds. Returns one segment (level
        # python command, name the command, start and end in epoch seconds, empty
        # extra info) per window. An empty record list gives an empty result.
        # KNOWN ISSUE: if records exist but none lists a command, the frame has no
        # columns and the time conversion raises a key error.
        if len(records) == 0:
            return []

        cmd_col_name = "cmd"
        time_col_name = "time"

        # flatten time series with detected cmds
        df = pd.DataFrame(
            [
                {time_col_name: record.timestamp, cmd_col_name: process}
                for record in records
                for process in (record.cmd_names or [])
            ]
        )
        df[time_col_name] = pd.to_datetime(df[time_col_name], unit="s", utc=True)

        # get unique cmd names
        unique_cmds_df = pd.DataFrame(df[cmd_col_name].unique(), columns=[cmd_col_name])

        # get all detected python cmds
        cmd_list = unique_cmds_df[
            unique_cmds_df[cmd_col_name].str.startswith("python")
        ][cmd_col_name].tolist()

        # find segments by screening continuoues time series data
        segments: list[OssCiSegmentV1] = []
        for value in cmd_list:
            subset = df[df[cmd_col_name] == value].copy()

            continuous_segments = self._find_continuous_windows(
                segment_delta_threshold, time_col_name, subset
            )
            for row in continuous_segments:
                segment = OssCiSegmentV1(
                    level=CMD_PYTHON_LEVEL,
                    name=value,
                    start_at=int(row["start_time"].timestamp()),
                    end_at=int(row["end_time"].timestamp()),
                    extra_info={},
                )
                segments.append(segment)
        print(
            f"[Db Segments] detected pytest cmd: {len(cmd_list)}, generated segments: {len(segments)}"
        )
        return segments

    def _find_continuous_windows(
        self,
        threshold: int,
        time_column_name: str,
        df: Any,  # the lintrunner keep complaining about the type of df, but it's not a problem
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Sorts the frame by the time column, starts a new
        # window wherever the gap to the previous row exceeds threshold seconds, and
        # returns a list of mappings with the first and last time of each window.
        time_threshold = pd.Timedelta(seconds=threshold)
        df = df.sort_values(by=time_column_name).reset_index(drop=True)
        df["time_diff"] = df[time_column_name].diff()
        df["segment"] = (df["time_diff"] > time_threshold).cumsum()
        segments = (
            df.groupby("segment")
            .agg(
                start_time=(time_column_name, "first"),
                end_time=(time_column_name, "last"),
            )
            .reset_index(drop=True)
        )
        return segments[["start_time", "end_time"]].to_dict(orient="records")  # type: ignore[no-any-return]


class UtilizationDbConverter:
    # [CLASS DOC]
    # Converts parsed log models (metadata, records, segments) plus workflow info
    # into the database models: one metadata row and one time series row per record.
    # State: the inputs, a creation timestamp taken at construction and the end time
    # (latest record timestamp). Entry point is convert.
    """convert utilization log model to db model"""

    def __init__(
        self,
        info: WorkflowInfo,
        metadata: UtilizationMetadata,
        records: list[UtilizationRecord],
        segments: list[OssCiSegmentV1],
    ):
        # [FUNCTION DOC]
        # Stores the inputs, records the creation time as now and the end time as the
        # largest record timestamp. Raises ValueError if records is empty.
        self.metadata = metadata
        self.records = records
        self.segments = segments
        self.created_at = getTsNow()
        self.info = info
        end_time_stamp = max([record.timestamp for record in records])
        self.end_at = end_time_stamp

    def convert(
        self,
    ) -> tuple[OssCiUtilizationMetadataV1, list[OssCiUtilizationTimeSeriesV1]]:
        # [FUNCTION DOC] Returns the metadata row and the list of time series rows.
        db_metadata = self._to_oss_ci_metadata()
        timeseries = self._to_oss_ci_timeseries_list()
        return db_metadata, timeseries

    def _to_oss_ci_metadata(self) -> OssCiUtilizationMetadataV1:
        # [FUNCTION DOC] HELPER: Builds the metadata row from the workflow info and
        # log metadata; missing gpu count, cpu count or gpu type become 0, 0 and empty;
        # the data model version is stored as a string and tags are empty.
        return OssCiUtilizationMetadataV1(
            repo=self.info.repo,
            workflow_id=self.info.workflow_run_id,
            run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            workflow_name=self.info.workflow_name,
            job_name=self.info.job_name,
            usage_collect_interval=self.metadata.usage_collect_interval,
            data_model_version=str(self.metadata.data_model_version),
            created_at=self.created_at,
            gpu_count=self.metadata.gpu_count if self.metadata.gpu_count else 0,
            cpu_count=self.metadata.cpu_count if self.metadata.cpu_count else 0,
            gpu_type=self.metadata.gpu_type if self.metadata.gpu_type else "",
            start_at=self.metadata.start_at,
            end_at=self.end_at,
            segments=self.segments,
            tags=[],
        )

    def _to_oss_ci_timeseries_list(self) -> list[OssCiUtilizationTimeSeriesV1]:
        # [FUNCTION DOC] HELPER: Converts every record to a time series row of type
        # utilization tagged record.
        return [
            self._to_oss_ci_time_series(record, type="utilization", tags=["record"])
            for record in self.records
        ]

    def _to_oss_ci_time_series(
        self, record: UtilizationRecord, type: str, tags: list[str]
    ) -> OssCiUtilizationTimeSeriesV1:
        # [FUNCTION DOC] HELPER: Builds one time series row for a record with the given
        # type and tags; the payload is the string form of the record's data json
        # (or of an empty mapping when the record has no data).
        return OssCiUtilizationTimeSeriesV1(
            created_at=self.created_at,
            type=type,
            tags=tags,
            time_stamp=record.timestamp,
            repo=self.info.repo,
            workflow_id=self.info.workflow_run_id,
            run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            workflow_name=self.info.workflow_name,
            job_name=self.info.job_name,
            json_data=str(record.data.to_json() if record.data else {}),
        )


class UploadUtilizationData:
    # [CLASS DOC]
    # Orchestrates the pipeline for one job: read the raw log (local file or S3
    # artifact), parse it into log models, generate segments, convert to database
    # models and upload them. State: artifact prefix, workflow info, a segment
    # generator, and debug, dry-run and local-path settings. Entry point is start.
    """
    main class to handle utilization data conversion and s3 upload
    fetches raw log data from s3, convert to log model, then convert to db model, and upload to s3
    """

    def __init__(
        self,
        artifact_prefix: str,
        info: WorkflowInfo,
        dry_run: bool = False,
        debug: bool = False,
        local_path: str = "",
    ):
        # [FUNCTION DOC] Stores the settings and creates the segment generator.
        self.artifact_prefix = artifact_prefix
        self.info = info
        self.segment_generator = SegmentGenerator()
        self.debug_mode = debug
        self.dry_run = dry_run
        self.local_path = local_path

    def start(self) -> None:
        # [FUNCTION DOC]
        # Loads log data from the local path if set, otherwise from the job's S3
        # artifact. Returns early (printing a message) when metadata or valid records
        # are missing. Otherwise generates segments, converts to database models,
        # prints the first time series row, and unless in dry run mode uploads the
        # metadata and time series to S3 (collection names get a debug prefix in
        # debug mode, and keys include the data model version).
        if self.local_path:
            metadata, valid_records, _ = self.get_log_data_from_local(self.local_path)
        else:
            print(f"Search for test log in s3 bucket: {UTILIZATION_BUCKET}")
            metadata, valid_records, _ = self.get_log_data_from_s3(
                self.info.workflow_run_id,
                self.info.job_id,
                self.info.run_attempt,
                self.artifact_prefix,
            )

        if not metadata:
            print("[Log Model] Failed to process test log, metadata is None")
            return None

        if len(valid_records) == 0:
            print("[Log Model] Failed to process test log, no valid records")
            return None
        segments = self.segment_generator.generate(valid_records)

        db_metadata, db_records = UtilizationDbConverter(
            self.info, metadata, valid_records, segments
        ).convert()

        if len(db_records) > 0:
            print(
                f"[db model] Peek db timeseries \n:{json.dumps(asdict(db_records[0]), indent=4)}"
            )

        if self.dry_run:
            print("[dry-run-mode]: no upload in dry run mode")
            return

        version = f"v_{db_metadata.data_model_version}"
        metadata_collection = "util_metadata"
        ts_collection = "util_timeseries"
        if self.debug_mode:
            metadata_collection = f"debug_{metadata_collection}"
            ts_collection = f"debug_{ts_collection}"

        self._upload_utilization_data_to_s3(
            collection=metadata_collection,
            version=version,
            repo=self.info.repo,
            workflow_run_id=self.info.workflow_run_id,
            workflow_run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            file_name="metadata",
            docs=[asdict(db_metadata)],
        )

        self._upload_utilization_data_to_s3(
            collection=ts_collection,
            version=version,
            repo=self.info.repo,
            workflow_run_id=self.info.workflow_run_id,
            workflow_run_attempt=self.info.run_attempt,
            job_id=self.info.job_id,
            file_name="time_series",
            docs=[asdict(record) for record in db_records],
        )

    def _upload_utilization_data_to_s3(
        self,
        collection: str,
        version: str,
        repo: str,
        workflow_run_id: int,
        workflow_run_attempt: int,
        job_id: int,
        file_name: str,
        docs: list[dict[str, Any]],
    ) -> None:
        # [FUNCTION DOC] HELPER: Uploads docs to the utilization bucket under a key
        # built from collection, version, repo, run id, attempt, job id and file name.
        bucket_name = UTILIZATION_BUCKET
        key = f"{collection}/{version}/{repo}/{workflow_run_id}/{workflow_run_attempt}/{job_id}/{file_name}"
        upload_to_s3(bucket_name, key, docs)

    def get_log_data_from_local(
        self,
        file_path: str,
        artifact_prefix: str = "",
    ) -> tuple[
        UtilizationMetadata | None, list[UtilizationRecord], list[UtilizationRecord]
    ]:
        # [FUNCTION DOC]
        # Reads the local log file and parses it. Returns a triple of metadata (None on
        # failure), valid records and error records; failures give None and empty lists.
        # KNOWN ISSUE: the artifact_prefix parameter is unused.
        test_log_content = read_file(file_path)
        if not test_log_content:
            return None, [], []
        metadata, records, error_records = self.convert_to_log_models(test_log_content)
        if metadata is None:
            return None, [], []
        print(f"Converted Log Model: UtilizationMetadata:\n {metadata}")
        return metadata, records, error_records

    def get_log_data_from_s3(
        self,
        workflow_run_id: int,
        job_id: int,
        workflow_run_attempt: int,
        artifact_prefix: str = JOB_TEST_ARTIFACT_PREFIX,
    ) -> tuple[
        UtilizationMetadata | None, list[UtilizationRecord], list[UtilizationRecord]
    ]:
        # [FUNCTION DOC]
        # Downloads the job's artifacts with the given prefix from S3 (into the current
        # directory), requires exactly one, reads its log content (zip or text) and
        # parses it. Returns metadata, valid records and error records, or None with
        # empty lists when zero or several artifacts are found or parsing fails.
        artifact_paths = download_s3_artifacts(
            artifact_prefix, workflow_run_id, workflow_run_attempt, job_id
        )
        if len(artifact_paths) == 0:
            print(
                f"Failed to download artifacts for workflow {workflow_run_id} and job {job_id}"
            )
            return None, [], []
        elif len(artifact_paths) > 1:
            print(
                f"Found more than one artifact for workflow {workflow_run_id} and job {job_id}, {artifact_paths}"
            )
            return None, [], []

        p = artifact_paths[0]

        test_log_content = handle_file(p)
        if not test_log_content:
            return None, [], []

        metadata, records, error_records = self.convert_to_log_models(test_log_content)
        if metadata is None:
            return None, [], []

        print(f"Converted Log Model: UtilizationMetadata:\n {metadata}")
        return metadata, records, error_records

    def _process_raw_record(self, line: str) -> tuple[UtilizationRecord | None, bool]:
        # [FUNCTION DOC] HELPER: Parses one json line into a record. Returns the record
        # and True if it has no error text, the record and False if it has one, or
        # None and False if parsing fails (the failure is printed).
        try:
            record = UtilizationRecord.from_json(line)
            if record.error:
                return record, False
            return record, True
        except Exception as e:
            print(f"Failed to parse JSON line: {e}")
            return None, False

    def _process_utilization_records(
        self,
        lines: list[str],
    ) -> tuple[list[UtilizationRecord], list[UtilizationRecord]]:
        # [FUNCTION DOC] HELPER: Parses all lines after the first (the metadata line)
        # and splits them into valid records and records carrying an error; unparsable
        # lines are dropped.
        results = [self._process_raw_record(line) for line in lines[1:]]
        valid_records = [
            record for record, valid in results if valid and record is not None
        ]
        invalid_records = [
            record for record, valid in results if not valid and record is not None
        ]
        return valid_records, invalid_records

    def convert_to_log_models(
        self,
        content: str,
    ) -> tuple[
        UtilizationMetadata | None, list[UtilizationRecord], list[UtilizationRecord]
    ]:
        # [FUNCTION DOC]
        # Splits the log text into lines, parses the first as metadata and the rest as
        # records. Returns None with empty lists when the content is empty, has fewer
        # than two lines, the metadata cannot be parsed, or its data model version
        # differs from the current one; otherwise metadata, valid and error records.
        if not content:
            return None, [], []
        lines = content.splitlines()
        metadata = None
        if len(lines) < 2:
            print("Expected at least two records from log file")
            return None, [], []

        try:
            metadata = UtilizationMetadata.from_json(lines[0])
        except Exception as e:
            print(f":: warning Failed to parse metadata: {e} for data: {lines[0]}")
            return None, [], []

        if metadata.data_model_version != getDataModelVersion():
            print(
                f":: warning Data model version mismatch: {metadata.data_model_version} != {getDataModelVersion()}"
            )
            return None, [], []

        result_logs, error_logs = self._process_utilization_records(lines)
        return metadata, result_logs, error_logs


def handle_file(file_path: Path) -> str:
    # [FUNCTION DOC] Returns the usage log text from a zip archive (the usage log
    # entry) or a plain text file, chosen by file name pattern; prints a message and
    # returns an empty string for other file types.
    if file_path.match("*.zip"):
        print(f"extracting {TEST_USAGE_LOG_FILENAME} from zip file {file_path}")
        return unzip_file(file_path, TEST_USAGE_LOG_FILENAME)
    elif file_path.match("*.txt"):
        print(f"extracting {file_path}")
        return read_file(file_path)
    print(f"{file_path} is not a supported file type")
    return ""


def read_file(file_path: str | Path) -> str:
    # [FUNCTION DOC] Returns the text of the file, or an empty string (after printing
    # a warning annotation) if it does not exist, has an unsupported path type, or
    # reading fails. Never raises.
    try:
        if isinstance(file_path, Path):
            if file_path.is_file():
                with file_path.open("r") as f:
                    return f.read()
            else:
                print(f"::warning file {file_path} does not exist.")
        elif isinstance(file_path, str):
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    return f.read()
            else:
                print(f"::warning file {file_path} does not exist.")
        else:
            print(f"::warning unsupported file_path type: {type(file_path)}")
    except Exception as e:
        print(f"::warning trying to read file {file_path} failed by: {e}")
    return ""


def unzip_file(path: Path, file_name: str) -> str:
    # [FUNCTION DOC] Returns the decoded text of the named entry inside the zip file,
    # or an empty string after printing a warning if anything fails.
    # KNOWN ISSUE: the warning formats the Python builtin named object instead of
    # the file path.
    try:
        with zipfile.ZipFile(path) as zip_file:
            # Read the desired file from the zip archive
            return zip_file.read(name=file_name).decode()
    except Exception as e:
        print(f"::warning trying to download test log {object} failed by: {e}")
        return ""


def parse_args() -> argparse.Namespace:
    # [FUNCTION DOC]
    # Parses the command line: required workflow run id, attempt, workflow name, job
    # id and job name; optional repo, debug and dry-run flags, artifact prefix and
    # local log path. Exits the process on invalid arguments.
    # KNOWN ISSUE: several help texts (workflow name, job id, job name) wrongly say
    # they are the id of the workflow to get artifacts from.
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Upload test stats to s3")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--workflow-name",
        type=str,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=False,
        help="which GitHub repo this workflow run belongs to",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("--dry-run", action="store_true", help="Enable dry-run mode")

    parser.add_argument(
        "--artifact-prefix",
        type=str,
        required=False,
        help="artifact prefix to download raw utilizarion data from s3",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        required=False,
        help="path of the raw utilizarion data from local location",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Flush stdout so that any errors in the upload show up last in the logs.
    sys.stdout.flush()

    repo = PYTORCH_REPO
    if args.repo:
        repo = args.repo
    print(f"repo: {repo}")

    workflow_info = WorkflowInfo(
        workflow_run_id=args.workflow_run_id,
        run_attempt=args.workflow_run_attempt,
        job_id=args.job_id,
        workflow_name=args.workflow_name,
        job_name=args.job_name,
        repo=repo,
    )

    artifact_prefix = JOB_TEST_ARTIFACT_PREFIX
    if args.artifact_prefix:
        artifact_prefix = args.artifact_prefix
        print(f"args.artifact_prefix: {args.artifact_prefix}")
        print(f"artifact_prefix: {artifact_prefix}")

    ud = UploadUtilizationData(
        info=workflow_info,
        dry_run=args.dry_run,
        debug=args.debug,
        artifact_prefix=artifact_prefix,
        local_path=args.local_path,
    )
    ud.start()
