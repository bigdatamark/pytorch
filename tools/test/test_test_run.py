# [MODULE DOC] Unit tests for TestRun set-like operations (union, subtraction, intersection,
# truthiness) and ShardedTest pytest argument generation from tools.testing.test_run.
# Import time: appends the repo root to sys.path and exits the process if imports fail.
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_run import ShardedTest, TestRun
except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


class TestTestRun(unittest.TestCase):
    # [CLASS DOC] Test case for TestRun algebra. A TestRun names a test file plus optional
    # included or excluded test names; a file name with a double colon suffix is an inclusion.
    def test_union_with_full_run(self) -> None:
        # [FUNCTION DOC] Checks that unioning a whole-file run with a narrower run of the same
        # file yields the whole-file run, in either order.
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    def test_union_with_inclusions(self) -> None:
        # [FUNCTION DOC] Checks that unioning two different single-test inclusions gives a
        # run including both tests.
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo::baz")

        expected = TestRun("foo", included=["bar", "baz"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_non_overlapping_exclusions(self) -> None:
        # [FUNCTION DOC] Checks that unioning runs with disjoint exclusions yields the full
        # run (nothing excluded).
        run1 = TestRun("foo", excluded=["bar"])
        run2 = TestRun("foo", excluded=["baz"])

        expected = TestRun("foo")

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_overlapping_exclusions(self) -> None:
        # [FUNCTION DOC] Checks that unioning exclusion runs keeps only the exclusions common
        # to both.
        run1 = TestRun("foo", excluded=["bar", "car"])
        run2 = TestRun("foo", excluded=["bar", "caz"])

        expected = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_mixed_inclusion_exclusions(self) -> None:
        # [FUNCTION DOC] Checks that unioning an exclusion run with an inclusion run removes
        # from the exclusions any test that is included.
        run1 = TestRun("foo", excluded=["baz", "car"])
        run2 = TestRun("foo", included=["baz"])

        expected = TestRun("foo", excluded=["car"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_mixed_files_fails(self) -> None:
        # [FUNCTION DOC] Checks that unioning runs of different files raises an assertion
        # error.
        run1 = TestRun("foo")
        run2 = TestRun("bar")

        with self.assertRaises(AssertionError):
            run1 | run2

    def test_union_with_empty_file_yields_orig_file(self) -> None:
        # [FUNCTION DOC] Checks that unioning with the empty run returns the other run.
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    def test_subtracting_full_run_fails(self) -> None:
        # [FUNCTION DOC] Checks that subtracting a whole-file run from an inclusion of the same
        # file yields the empty run.
        # KNOWN ISSUE: the name says fails, but the test expects a successful empty result and
        # no error is raised.
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo")

        self.assertEqual(run1 - run2, TestRun.empty())

    def test_subtracting_empty_file_yields_orig_file(self) -> None:
        # [FUNCTION DOC] Checks that subtracting the empty run changes nothing, and that
        # subtracting anything from the empty run gives the empty run.
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, TestRun.empty())

    def test_empty_is_falsey(self) -> None:
        # [FUNCTION DOC] Checks that the empty run evaluates as false.
        self.assertFalse(TestRun.empty())

    def test_subtracting_inclusion_from_full_run(self) -> None:
        # [FUNCTION DOC] Checks that removing an inclusion from a whole-file run excludes that
        # test.
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        expected = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 - run2, expected)

    def test_subtracting_inclusion_from_overlapping_inclusion(self) -> None:
        # [FUNCTION DOC] Checks that subtracting an overlapping inclusion removes that test
        # from the included set.
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo::baz")

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))

    def test_subtracting_inclusion_from_nonoverlapping_inclusion(self) -> None:
        # [FUNCTION DOC] Checks that subtracting a disjoint inclusion leaves the run unchanged.
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo", included=["car"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar", "baz"]))

    def test_subtracting_exclusion_from_full_run(self) -> None:
        # [FUNCTION DOC] Checks that subtracting an exclusion run from a whole-file run leaves
        # exactly the excluded tests as inclusions.
        run1 = TestRun("foo")
        run2 = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))

    def test_subtracting_exclusion_from_superset_exclusion(self) -> None:
        # [FUNCTION DOC] Checks subtraction between exclusion runs where one exclusion set
        # contains the other: the superset minus subset is empty, the reverse leaves the
        # extra excluded test as an inclusion.
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["baz"])

        self.assertEqual(run1 - run2, TestRun.empty())
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))

    def test_subtracting_exclusion_from_nonoverlapping_exclusion(self) -> None:
        # [FUNCTION DOC] Checks subtraction between exclusion runs with disjoint sets: the
        # result includes the subtrahend's excluded tests.
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["car"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["car"]))
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar", "baz"]))

    def test_subtracting_inclusion_from_exclusion_without_overlaps(self) -> None:
        # [FUNCTION DOC] Checks that subtracting between an exclusion run and a disjoint
        # inclusion run leaves each operand unchanged.
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", included=["bar"])

        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, run2)

    def test_subtracting_inclusion_from_exclusion_with_overlaps(self) -> None:
        # [FUNCTION DOC] Checks subtraction between an exclusion run and an overlapping
        # inclusion run: the exclusion gains the included tests; the inclusion loses the
        # excluded ones.
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", included=["bar", "car"])

        self.assertEqual(run1 - run2, TestRun("foo", excluded=["bar", "baz", "car"]))
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))

    def test_and(self) -> None:
        # [FUNCTION DOC] Checks that intersecting two inclusion runs keeps common tests.
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo", included=["bar", "car"])

        self.assertEqual(run1 & run2, TestRun("foo", included=["bar"]))

    def test_and_exclusions(self) -> None:
        # [FUNCTION DOC] Checks that intersecting two exclusion runs excludes the union of
        # their excluded tests.
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["bar", "car"])

        self.assertEqual(run1 & run2, TestRun("foo", excluded=["bar", "baz", "car"]))


class TestShardedTest(unittest.TestCase):
    # [CLASS DOC] Test case for ShardedTest behavior; stateless.
    def test_get_pytest_args(self) -> None:
        # [FUNCTION DOC] Checks that a sharded test with two included tests produces a pytest
        # keyword filter joining the names with or.
        test = TestRun("foo", included=["bar", "baz"])
        sharded_test = ShardedTest(test, 1, 1)

        expected_args = ["-k", "bar or baz"]

        self.assertListEqual(sharded_test.get_pytest_args(), expected_args)


if __name__ == "__main__":
    unittest.main()
