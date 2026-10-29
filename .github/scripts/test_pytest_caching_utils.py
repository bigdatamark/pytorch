# [MODULE DOC]
# Unit tests for the lastfailed merge helper in pytest_caching_utils (union of test ids from the
# source into the destination, dropping the empty placeholder key).
from unittest import main, TestCase

from pytest_caching_utils import _merged_lastfailed_content


class TestPytestCachingUtils(TestCase):
    # [CLASS DOC]
    # Tests of merged lastfailed content for overlapping, disjoint, and placeholder-only
    # (empty key) inputs.
    def test_merged_lastfailed_content_with_overlap(self) -> None:
        # [FUNCTION DOC] Checks that overlapping source and destination entries merge
        # into their union.
        last_failed_source = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
        }
        last_failed_dest = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_without_overlap(self) -> None:
        # [FUNCTION DOC] Checks that disjoint source and destination entries merge
        # into their union.
        last_failed_source = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
        }
        last_failed_dest = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_with_empty_source(self) -> None:
        # [FUNCTION DOC] Checks that a source holding only the empty placeholder key leaves the
        # destination entries unchanged.
        last_failed_source = {
            "": True,
        }
        last_failed_dest = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_with_empty_dest(self) -> None:
        # [FUNCTION DOC] Checks that an empty placeholder key in the destination is dropped when
        # the source adds real entries.
        last_failed_source = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_dest = {
            "": True,
        }
        last_failed_merged = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)


if __name__ == "__main__":
    main()
