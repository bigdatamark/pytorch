# mypy: ignore-errors
# [MODULE DOC] Shared base test case for the lintrunner-style linters under tools/linter.
# Provides golden-file comparison and helpers that run a linter both from the command line
# and in lintrunner mode. No import-time side effects.
import io
import json
import os
from pathlib import Path
from unittest import mock, TestCase


class LinterTestCase(TestCase):
    # [CLASS DOC] Base class for linter tests. Subclasses set LinterClass to the linter
    # under test. The rewrite_expected flag is fixed at import time from the presence of
    # the REWRITE_EXPECTED environment variable and makes golden files get regenerated.
    LinterClass = None
    rewrite_expected = "REWRITE_EXPECTED" in os.environ

    def assertExpected(self, path: Path, actual: str, suffix: str) -> None:
        # [FUNCTION DOC] Golden-file check. The expected file is the given path with the
        # suffix appended. If it exists and rewriting is off, asserts its text equals
        # actual; otherwise writes actual to it (side effect: creates or overwrites it).
        expected_file = Path(f"{path}.{suffix}")
        if not self.rewrite_expected and expected_file.exists():
            self.assertEqual(expected_file.read_text(), actual)
        else:
            expected_file.write_text(actual)

    def replace(self, s: str):
        # [FUNCTION DOC] Runs the linter's replacement pass on an in-memory source string
        # (linter built with a dummy name) and returns only the replacement text, dropping
        # the lint results.
        linter = self.LinterClass("dummy")
        pf = self.LinterClass.make_file(contents=s)
        replacement, _results = linter._replace(pf)
        return replacement

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def lint_test(self, path, args, mock_stdout):
        # [FUNCTION DOC] Runs the shared lint scenario for a file with extra linter args
        # (stdout is captured by a mock) and returns the first two items: the replacement
        # text and the list of lint results.
        return self._lint_test(path, args, mock_stdout)[:2]

    @mock.patch("sys.stdout", new_callable=io.StringIO)
    def lint_fix_test(self, path, args, mock_stdout):
        # [FUNCTION DOC] Like lint_test but also validates the autofix: takes the last lint
        # result, asserts its original text equals the file on disk and its replacement
        # equals the computed replacement, compares the replacement to a golden python
        # file, and returns that result.
        rep, results, linter = self._lint_test(path, args, mock_stdout)
        r = results[-1]
        path = linter.paths[0]
        self.assertEqual(r.original, path.read_text())
        self.assertEqual(rep, r.replacement)
        self.assertExpected(path, r.replacement, "python")
        return r

    def _lint_test(self, path, args, mock_stdout):
        # [FUNCTION DOC] HELPER: Runs the linter twice in subtests. First from the command
        # line over all files, comparing captured stdout to a golden lintrunner file. Then
        # in lintrunner mode, computing the replacement and results and comparing the
        # sorted JSON rendering of the displayed messages to a golden json file. Returns
        # the replacement text, results and the lintrunner-mode linter (defaults are
        # placeholder strings if a subtest fails early).
        with self.subTest("from-command-line"):
            linter = self.LinterClass([str(path), *args])
            linter.lint_all()
            self.assertExpected(path, mock_stdout.getvalue(), "lintrunner")

        replacement, results = "(no replacement)", "(no results)"
        with self.subTest("from-lintrunner"):
            linter = self.LinterClass(["--lintrunner", str(path), *args])
            pf = self.LinterClass.make_file(path)
            replacement, results = linter._replace(pf)

            actual = [json.loads(d) for d in linter._display(pf, results)]
            actual = json.dumps(actual, indent=2, sort_keys=True) + "\n"
            self.assertExpected(path, actual, "json")

        return replacement, results, linter
