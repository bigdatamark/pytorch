# [MODULE DOC]
# Unit tests for SegmentGenerator, which builds command segments from utilization
# records. Defines fixed base timestamps and command names, a shared record builder,
# and two unused timestamp formatting helpers. Import-time side effect: edits sys.path.

import os
import sys
import unittest
from collections import Counter
from datetime import datetime, timedelta


# adding sys.path makes the monitor script able to import path tools.stats.utilization_stats_lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tools.stats.upload_utilization_stats.upload_utilization_stats import (
    SegmentGenerator,
)
from tools.stats.utilization_stats_lib import OssCiSegmentV1, UtilizationRecord


# datetimes from January 1, 2022 12:00:00
TEST_DT_BASE = datetime(2022, 1, 1, 12, 0, 0)
TEST_DT_PLUS_5S = TEST_DT_BASE + timedelta(seconds=5)
TEST_DT_PLUS_10S = TEST_DT_BASE + timedelta(seconds=10)
TEST_DT_PLUS_15S = TEST_DT_BASE + timedelta(seconds=15)
TEST_DT_PLUS_30S = TEST_DT_BASE + timedelta(seconds=30)
TEST_DT_PLUS_40S = TEST_DT_BASE + timedelta(seconds=40)

# timestamps from January 1, 2022 12:00:00
TEST_TS_BASE = int(TEST_DT_BASE.timestamp())
TEST_TS_PLUS_5S = int(TEST_DT_PLUS_5S.timestamp())
TEST_TS_PLUS_10S = int(TEST_DT_PLUS_10S.timestamp())
TEST_TS_PLUS_15S = int(TEST_DT_PLUS_15S.timestamp())
TEST_TS_PLUS_30S = int(TEST_DT_PLUS_30S.timestamp())
TEST_TS_PLUS_40S = int(TEST_DT_PLUS_40S.timestamp())


# test cmd names
PYTEST1_NAME = "python test1.py"
PYTEST2_NAME = "python test2.py"
PYPIP_INSTALL_NAME = "python pip install install1"


class TestSegmentGenerator(unittest.TestCase):
    # [CLASS DOC]
    # Tests of SegmentGenerator.generate covering empty input, single and multiple
    # records and commands, and the default and a customized gap threshold.
    def test_generate_empty_records(self) -> None:
        # [FUNCTION DOC] Checks that no records produce no segments.
        records: list[UtilizationRecord] = []

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(segments, [])

    def test_generate_single_record(self) -> None:
        # [FUNCTION DOC] Checks that one record with one command gives one segment.
        record = UtilizationRecord(
            timestamp=TEST_TS_BASE, cmd_names=[PYTEST1_NAME], level="PYTHON_CMD"
        )
        records = [record]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(len(segments), 1)

    def test_generate_single_record_with_multiple_cmds(self) -> None:
        # [FUNCTION DOC] Checks that one record listing two python commands gives two
        # segments.
        record = UtilizationRecord(
            timestamp=TEST_TS_BASE,
            cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
            level="PYTHON_CMD",
        )
        records = [record]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(len(segments), 2)

    def test_generate_multiple_records(self) -> None:
        # [FUNCTION DOC] Checks that the base records give one segment per command
        # with the expected start and end times.
        records = get_base_test_records()

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        self.assertEqual(len(segments), 2)
        self.validate_segment(segments[0], PYTEST1_NAME, TEST_TS_BASE, TEST_TS_PLUS_30S)
        self.validate_segment(
            segments[1], PYPIP_INSTALL_NAME, TEST_TS_PLUS_10S, TEST_TS_PLUS_15S
        )

    def test_generate_cmd_interval_larger_than_default_threshold_setting(self) -> None:
        # [FUNCTION DOC] Adds two records separated by more than the default 60
        # second threshold and checks the first command splits into two segments.
        records = get_base_test_records()

        # record has more than 1 minute gap than last default record
        test_gap_dt1 = TEST_DT_PLUS_30S + timedelta(seconds=80)
        test_gap_dt2 = TEST_DT_PLUS_30S + timedelta(seconds=85)
        record_gap_1 = UtilizationRecord(
            timestamp=int(test_gap_dt1.timestamp()),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        record_gap_2 = UtilizationRecord(
            timestamp=int(test_gap_dt2.timestamp()),
            cmd_names=[PYTEST1_NAME],
            level="PYTHON_CMD",
        )
        records += [record_gap_1, record_gap_2]

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records)

        # assert
        counter = Counter(seg.name for seg in segments)
        self.assertEqual(counter[PYTEST1_NAME], 2)
        self.assertEqual(counter[PYPIP_INSTALL_NAME], 1)
        self.assertEqual(len(segments), 3)

        self.validate_segment(segments[0], PYTEST1_NAME, TEST_TS_BASE, TEST_TS_PLUS_30S)
        self.validate_segment(
            segments[1],
            PYTEST1_NAME,
            test_gap_dt1.timestamp(),
            test_gap_dt2.timestamp(),
        )
        self.validate_segment(
            segments[2], PYPIP_INSTALL_NAME, TEST_TS_PLUS_10S, TEST_TS_PLUS_15S
        )

    def test_generate_multiple_segments_with_customized_threshold(self) -> None:
        # [FUNCTION DOC] Uses a 10 second threshold on the base records and checks the
        # first command splits into two segments.
        # set threshold to consider as continuous segment to 10 seconds
        test_threshold = 10

        records = get_base_test_records()

        # execute
        generator = SegmentGenerator()
        segments = generator.generate(records, test_threshold)

        # assert
        counter = Counter(seg.name for seg in segments)
        self.assertEqual(counter[PYTEST1_NAME], 2)
        self.assertEqual(counter[PYPIP_INSTALL_NAME], 1)
        self.assertEqual(len(segments), 3)

        self.validate_segment(segments[0], PYTEST1_NAME, TEST_TS_BASE, TEST_TS_PLUS_15S)
        self.validate_segment(
            segments[1], PYTEST1_NAME, TEST_TS_PLUS_30S, TEST_TS_PLUS_30S
        )
        self.validate_segment(
            segments[2], PYPIP_INSTALL_NAME, TEST_TS_PLUS_10S, TEST_TS_PLUS_15S
        )

    def validate_segment(
        self, segment: OssCiSegmentV1, name: str, start_at: float, end_at: float
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts the segment's name, start and end times.
        self.assertEqual(segment.name, name)
        self.assertEqual(segment.start_at, start_at)
        self.assertEqual(segment.end_at, end_at)


def get_base_test_records() -> list[UtilizationRecord]:
    # [FUNCTION DOC] HELPER: Builds six records at 0, 5, 10, 15, 30 and 40 seconds
    # from the base time: the first command at the first five, a pip install command
    # at 10 and 15 seconds, and no commands in the last.
    record1 = UtilizationRecord(
        timestamp=TEST_TS_BASE, cmd_names=[PYTEST1_NAME], level="PYTHON_CMD"
    )
    record2 = UtilizationRecord(
        timestamp=TEST_TS_PLUS_5S,
        cmd_names=[PYTEST1_NAME],
        level="PYTHON_CMD",
    )
    record3 = UtilizationRecord(
        timestamp=TEST_TS_PLUS_10S,
        cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
        level="PYTHON_CMD",
    )
    record4 = UtilizationRecord(
        timestamp=TEST_TS_PLUS_15S,
        cmd_names=[PYTEST1_NAME, PYPIP_INSTALL_NAME],
        level="PYTHON_CMD",
    )
    record5 = UtilizationRecord(
        timestamp=TEST_TS_PLUS_30S,
        cmd_names=[PYTEST1_NAME],
        level="PYTHON_CMD",
    )
    record6 = UtilizationRecord(
        timestamp=TEST_TS_PLUS_40S,
        cmd_names=[],
        level="PYTHON_CMD",
    )
    return [record1, record2, record3, record4, record5, record6]


if __name__ == "__main__":
    unittest.main()


def getTimestampStr(timestamp: float) -> str:
    # [FUNCTION DOC] HELPER: Formats a timestamp as a string with no decimals. Unused
    # here and defined after the main guard.
    return f"{timestamp:.0f}"


def getCurrentTimestampStr() -> str:
    # [FUNCTION DOC] HELPER: Returns the current time formatted by getTimestampStr.
    # Unused here.
    timestamp_now = datetime.now().timestamp()
    return getTimestampStr(timestamp_now)
