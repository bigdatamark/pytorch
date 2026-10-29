#!/usr/bin/env python3
# [MODULE DOC]
# Unit tests for gitutils: the PeekableIterator, glob pattern to regex conversion, the retries
# decorator, and GitRepo diff and ghstack sync checks. The GitRepo tests need a real checkout of
# the repo and are skipped otherwise. Defines BASE_DIR, the directory of this file.

from pathlib import Path
from unittest import main, SkipTest, TestCase

from gitutils import (
    _shasum,
    are_ghstack_branches_in_sync,
    GitRepo,
    patterns_to_regex,
    PeekableIterator,
    retries_decorator,
)


BASE_DIR = Path(__file__).parent


class TestPeekableIterator(TestCase):
    # [CLASS DOC]
    # Tests of PeekableIterator iteration and peeking.
    def test_iterator(self, input_: str = "abcdef") -> None:
        # [FUNCTION DOC] Checks that iterating yields the input characters in order.
        iter_ = PeekableIterator(input_)
        for idx, c in enumerate(iter_):
            self.assertEqual(c, input_[idx])

    def test_is_iterable(self) -> None:
        # [FUNCTION DOC] Checks that an iterator instance is recognized as a
        # collections Iterator.
        from collections.abc import Iterator

        iter_ = PeekableIterator("")
        self.assertTrue(isinstance(iter_, Iterator))

    def test_peek(self, input_: str = "abcdef") -> None:
        # [FUNCTION DOC] Checks that peek returns the next character without consuming it, and
        # None at the end.
        iter_ = PeekableIterator(input_)
        for idx, c in enumerate(iter_):
            if idx + 1 < len(input_):
                self.assertEqual(iter_.peek(), input_[idx + 1])
            else:
                self.assertTrue(iter_.peek() is None)


class TestPattern(TestCase):
    # [CLASS DOC]
    # Tests of glob pattern conversion.
    def test_double_asterisks(self) -> None:
        # [FUNCTION DOC] Checks that a pattern with a double asterisk and a single asterisk
        # matches files in both a directory and a nested subdirectory.
        allowed_patterns = [
            "aten/src/ATen/native/**LinearAlgebra*",
        ]
        patterns_re = patterns_to_regex(allowed_patterns)
        fnames = [
            "aten/src/ATen/native/LinearAlgebra.cpp",
            "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp",
        ]
        for filename in fnames:
            self.assertTrue(patterns_re.match(filename))


class TestRetriesDecorator(TestCase):
    # [CLASS DOC]
    # Tests of the retries decorator.
    def test_simple(self) -> None:
        # [FUNCTION DOC] Checks that the decorated function returns its normal result
        # when it succeeds.
        @retries_decorator()
        def foo(x: int, y: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of x and y (function being decorated).
            return x + y

        self.assertEqual(foo(3, 4), 7)

    def test_fails(self) -> None:
        # [FUNCTION DOC] Checks that a function that always raises (wrong argument types) yields
        # the fallback value given as rc.
        @retries_decorator(rc=0)
        def foo(x: int, y: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of x and y (raises TypeError for
            # mismatched types).
            return x + y

        self.assertEqual(foo("a", 4), 0)


class TestGitRepo(TestCase):
    # [CLASS DOC]
    # Tests of GitRepo that run git in the real repo checkout; skipped if no .git directory.
    def setUp(self) -> None:
        # [FUNCTION DOC] Creates a GitRepo for the repo root above the scripts dir; raises
        # SkipTest if it has no git directory.
        repo_dir = BASE_DIR.absolute().parent.parent
        if not (repo_dir / ".git").is_dir():
            raise SkipTest(
                "Can't find git directory, make sure to run this test on real repo checkout"
            )
        self.repo = GitRepo(str(repo_dir))

    def _skip_if_ref_does_not_exist(self, ref: str) -> None:
        # [FUNCTION DOC] HELPER: Raises SkipTest if the ref does not exist locally (stale
        # branches get deleted).
        """Skip test if ref is missing as stale branches are deleted with time"""
        try:
            self.repo.show_ref(ref)
        except RuntimeError as e:
            raise SkipTest(f"Can't find head ref {ref} due to {str(e)}") from e

    def test_compute_diff(self) -> None:
        # [FUNCTION DOC] Checks that the diff of HEAD hashes to a 64 character sha256 digest.
        diff = self.repo.diff("HEAD")
        sha = _shasum(diff)
        self.assertEqual(len(sha), 64)

    def test_ghstack_branches_in_sync(self) -> None:
        # [FUNCTION DOC] Checks that a known ghstack head ref reports in sync; skipped if the
        # ref is missing.
        head_ref = "gh/SS-JIA/206/head"
        self._skip_if_ref_does_not_exist(head_ref)
        self.assertTrue(are_ghstack_branches_in_sync(self.repo, head_ref))

    def test_ghstack_branches_not_in_sync(self) -> None:
        # [FUNCTION DOC] Checks that a known out of sync ghstack head ref reports not in sync;
        # skipped if missing.
        head_ref = "gh/clee2000/1/head"
        self._skip_if_ref_does_not_exist(head_ref)
        self.assertFalse(are_ghstack_branches_in_sync(self.repo, head_ref))


if __name__ == "__main__":
    main()
