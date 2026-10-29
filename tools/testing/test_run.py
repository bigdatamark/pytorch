# [MODULE DOC] Defines TestRun, a set of tests of one test file described by included or
# excluded test classes with set algebra (union, subtraction, intersection), and
# ShardedTest, a TestRun assigned to one shard of several with an optional duration. No
# import-time side effects.

from __future__ import annotations

from copy import copy
from functools import total_ordering
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


class TestRun:
    # [CLASS DOC] Immutable description of tests to run from one file: test_file plus either a
    # frozenset of included classes (run only those) or of excluded classes (run all but
    # those); neither means the whole file. The empty file name is the empty run. __test__ is
    # False so pytest does not collect it. Supports union, subtraction and intersection.
    """
    TestRun defines the set of tests that should be run together in a single pytest invocation.
    It'll either be a whole test file or a subset of a test file.

    This class assumes that we won't always know the full set of TestClasses in a test file.
    So it's designed to include or exclude explicitly requested TestClasses, while having accepting
    that there will be an ambiguous set of "unknown" test classes that are not expliclty called out.
    Those manifest as tests that haven't been explicitly excluded.
    """

    test_file: str
    _excluded: frozenset[str]  # Tests that should be excluded from this test run
    _included: frozenset[
        str
    ]  # If non-empy, only these tests should be run in this test run

    # NB: Also the class is called TestRun, it's not a test class, so having this field set
    # will allow pytest to ignore this accordingly
    __test__ = False

    def __init__(
        self,
        name: str,
        excluded: Iterable[str] | None = None,
        included: Iterable[str] | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a run from a name and optional included or excluded class
        # collections. Raises ValueError if both are given. A name containing the class separator
        # means file and one included class; combining that with included or excluded
        # raises AssertionError.
        if excluded and included:
            raise ValueError("Can't specify both included and excluded")

        ins = set(included or [])
        exs = set(excluded or [])

        if "::" in name:
            if included or excluded:
                raise AssertionError(
                    "Can't specify included or excluded tests when specifying a test class in the file name"
                )
            self.test_file, test_class = name.split("::")
            ins.add(test_class)
        else:
            self.test_file = name

        self._excluded = frozenset(exs)
        self._included = frozenset(ins)

    @staticmethod
    def empty() -> TestRun:
        # [FUNCTION DOC] Returns the empty run, which stands for nothing to run.
        return TestRun("")

    def is_empty(self) -> bool:
        # [FUNCTION DOC] Returns True if the run has no test file name.
        # Lack of a test_file means that this is an empty run,
        # which means there is nothing to run. It's the zero.
        return not self.test_file

    def is_full_file(self) -> bool:
        # [FUNCTION DOC] Returns True if there are no included and no excluded classes.
        return not self._included and not self._excluded

    def included(self) -> frozenset[str]:
        # [FUNCTION DOC] Returns the included classes.
        return self._included

    def excluded(self) -> frozenset[str]:
        # [FUNCTION DOC] Returns the excluded classes.
        return self._excluded

    def get_pytest_filter(self) -> str:
        # [FUNCTION DOC] Returns the pytest keyword expression: included names joined by or, or the
        # negated expression of excluded names, or an empty string for a whole file. Names sorted.
        if self._included:
            return " or ".join(sorted(self._included))
        elif self._excluded:
            return f"not ({' or '.join(sorted(self._excluded))})"
        else:
            return ""

    def contains(self, test: TestRun) -> bool:
        # [FUNCTION DOC] Returns True if every test of the other run is also in this run. False for
        # different files. A whole-file run contains everything; otherwise containment is decided
        # by comparing the included and excluded sets.
        if self.test_file != test.test_file:
            return False

        if self.is_full_file():
            return True  # self contains all tests

        if test.is_full_file():
            return False  # test contains all tests, but self doesn't

        # Does self exclude a subset of what test excludes?
        if test._excluded:
            return test._excluded.issubset(self._excluded)

        # Does self include everything test includes?
        if self._included:
            return test._included.issubset(self._included)

        # Getting to here means that test includes and self excludes
        # Does self exclude anything test includes? If not, we're good
        return not self._excluded.intersection(test._included)

    def __copy__(self) -> TestRun:
        # [FUNCTION DOC] Returns an equal new run.
        return TestRun(self.test_file, excluded=self._excluded, included=self._included)

    def __bool__(self) -> bool:
        # [FUNCTION DOC] Truthy if the run is not empty.
        return not self.is_empty()

    def __repr__(self) -> str:
        # [FUNCTION DOC] Debug text with the file name and included or excluded sets.
        # KNOWN ISSUE: the text starts with RunTest rather than the class name TestRun.
        r: str = f"RunTest({self.test_file}"
        r += f", included: {self._included}" if self._included else ""
        r += f", excluded: {self._excluded}" if self._excluded else ""
        r += ")"
        return r

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns Empty for the empty run, else the file name followed by the pytest
        # filter if any.
        if self.is_empty():
            return "Empty"

        pytest_filter = self.get_pytest_filter()
        if pytest_filter:
            return self.test_file + ", " + pytest_filter
        return self.test_file

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when the other is a TestRun with the same file and the same
        # included and excluded sets.
        if not isinstance(other, TestRun):
            return False

        ret = self.test_file == other.test_file
        ret = ret and self._included == other._included
        ret = ret and self._excluded == other._excluded
        return ret

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash from file name, included set and excluded set.
        return hash((self.test_file, self._included, self._excluded))

    def __or__(self, other: TestRun) -> TestRun:
        # [FUNCTION DOC] Union of two runs (all tests in either). Empty runs act as identity.
        # Raises AssertionError if the files differ. Cases: any whole file gives the whole file;
        # two include sets union; two exclude sets intersect; mixed keeps the exclusions that are
        # not included.
        # KNOWN ISSUE: the error message wrongly says exclude instead of union.
        """
        To OR/Union test runs means to run all the tests that either of the two runs specify.
        """

        # Is any file empty?
        if self.is_empty():
            return other
        if other.is_empty():
            return copy(self)

        # If not, ensure we have the same file
        if self.test_file != other.test_file:
            raise AssertionError(
                f"Can't exclude {other} from {self} because they're not the same test file"
            )

        # 4 possible cases:

        # 1. Either file is the full file, so union is everything
        if self.is_full_file() or other.is_full_file():
            # The union is the whole file
            return TestRun(self.test_file)

        # 2. Both files only run what's in _included, so union is the union of the two sets
        if self._included and other._included:
            return TestRun(
                self.test_file, included=self._included.union(other._included)
            )

        # 3. Both files only exclude what's in _excluded, so union is the intersection of the two sets
        if self._excluded and other._excluded:
            return TestRun(
                self.test_file, excluded=self._excluded.intersection(other._excluded)
            )

        # 4. One file includes and the other excludes, so we then continue excluding the _excluded set minus
        #    whatever is in the _included set
        included = self._included | other._included
        excluded = self._excluded | other._excluded
        return TestRun(self.test_file, excluded=excluded - included)

    def __sub__(self, other: TestRun) -> TestRun:
        # [FUNCTION DOC] Difference: the tests in this run that are not in the other. Empty if this
        # run is empty or the other is a whole file; returns a copy if the other is empty or in
        # another file. Otherwise computes the new included or excluded set, returning the empty
        # run when nothing is left.
        """
        To subtract test runs means to run all the tests in the first run except for what the second run specifies.
        """

        # Is any file empty?
        if self.is_empty():
            return TestRun.empty()
        if other.is_empty():
            return copy(self)

        # Are you trying to subtract tests that don't even exist in this test run?
        if self.test_file != other.test_file:
            return copy(self)

        # You're subtracting everything?
        if other.is_full_file():
            return TestRun.empty()

        def return_inclusions_or_empty(inclusions: frozenset[str]) -> TestRun:
        # [FUNCTION DOC] HELPER: Returns a run including the given classes, or the empty run if
        # there are none.
            if inclusions:
                return TestRun(self.test_file, included=inclusions)
            return TestRun.empty()

        if other._included:
            if self._included:
                return return_inclusions_or_empty(self._included - other._included)
            else:
                return TestRun(
                    self.test_file, excluded=self._excluded | other._included
                )
        else:
            if self._included:
                return return_inclusions_or_empty(self._included & other._excluded)
            else:
                return return_inclusions_or_empty(other._excluded - self._excluded)

    def __and__(self, other: TestRun) -> TestRun:
        # [FUNCTION DOC] Intersection of two runs. Empty if the files differ; otherwise derived
        # from union minus both differences.
        if self.test_file != other.test_file:
            return TestRun.empty()

        return (self | other) - (self - other) - (other - self)

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a JSON-ready mapping with the file name and, if non-empty,
        # lists of included and excluded classes (list order is unspecified).
        r: dict[str, Any] = {
            "test_file": self.test_file,
        }
        if self._included:
            r["included"] = list(self._included)
        if self._excluded:
            r["excluded"] = list(self._excluded)
        return r

    @staticmethod
    def from_json(json: dict[str, Any]) -> TestRun:
        # [FUNCTION DOC] Builds a run from the mapping produced by to_json; missing lists default
        # to empty.
        return TestRun(
            json["test_file"],
            included=json.get("included", []),
            excluded=json.get("excluded", []),
        )


@total_ordering
class ShardedTest:
    # [CLASS DOC] A TestRun assigned to shard number shard out of num_shards, with an optional
    # estimated time in seconds. Ordered by name, shard, shard count then time (None smallest).
    test: TestRun
    shard: int
    num_shards: int
    time: float | None  # In seconds

    def __init__(
        self,
        test: TestRun | str,
        shard: int,
        num_shards: int,
        time: float | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the run (a string is converted to a TestRun), shard, shard count
        # and time.
        if isinstance(test, str):
            test = TestRun(test)
        self.test = test
        self.shard = shard
        self.num_shards = num_shards
        self.time = time

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Returns the test file name of the run.
        return self.test.test_file

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal when the other is a ShardedTest with equal run, shard, shard count
        # and time.
        # KNOWN ISSUE: defining equality without a hash method makes instances unhashable.
        if not isinstance(other, ShardedTest):
            return False
        return (
            self.test == other.test
            and self.shard == other.shard
            and self.num_shards == other.num_shards
            and self.time == other.time
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text with run, shard over count, and the time if set.
        ret = f"{self.test} {self.shard}/{self.num_shards}"
        if self.time:
            ret += f" ({self.time}s)"

        return ret

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC] Ordering used by sorting. Raises NotImplementedError for non-ShardedTest.
        # Compares name, shard, shard count, then time with None treated as smallest.
        # KNOWN ISSUE: if both times are None it reports less-than, so it is not irreflexive.
        if not isinstance(other, ShardedTest):
            raise NotImplementedError

        # This is how the list was implicitly sorted when it was a NamedTuple
        if self.name != other.name:
            return self.name < other.name
        if self.shard != other.shard:
            return self.shard < other.shard
        if self.num_shards != other.num_shards:
            return self.num_shards < other.num_shards

        # None is the smallest value
        if self.time is None:
            return True
        if other.time is None:
            return False
        return self.time < other.time

    def __str__(self) -> str:
        # [FUNCTION DOC] Text with run, shard over count.
        return f"{self.test} {self.shard}/{self.num_shards}"

    def get_time(self, default: float = 0) -> float:
        # [FUNCTION DOC] Returns the time, or the default when the time is unknown.
        return self.time if self.time is not None else default

    def get_pytest_args(self) -> list[str]:
        # [FUNCTION DOC] Returns the pytest arguments selecting this run's classes (a keyword
        # option followed by the filter), or an empty list for a whole file.
        filter = self.test.get_pytest_filter()
        if filter:
            return ["-k", self.test.get_pytest_filter()]
        return []
