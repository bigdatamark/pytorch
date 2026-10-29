# [MODULE DOC] Unit tests for the header-only linter adapter. They exercise symbol matching
# against C++ test globs and file checking against fixture files in
# tools/test/header_only_linter_testdata. No import-time side effects besides imports.
import re
import unittest

from tools.linter.adapters.header_only_linter import (
    check_file,
    CPP_TEST_GLOBS,
    find_matched_symbols,
    LINTER_CODE,
    LintMessage,
    LintSeverity,
    REPO_ROOT,
)


class TestHeaderOnlyLinter(unittest.TestCase):
    # [CLASS DOC] Test case covering find_matched_symbols and check_file of the header-only
    # linter. Stateless; each test uses fixture data under the repo test data directory.
    """
    Test the header only linter functionality
    """

    def test_find_matched_symbols(self) -> None:
        # [FUNCTION DOC] Checks that only regex alternatives actually appearing as symbols in
        # the fixture cpp files are returned, as a set (expects symDef, symC and a).
        sample_regex = re.compile("symDef|symD|symC|bbb|a")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_matches = {"symDef", "symC", "a"}
        self.assertEqual(
            find_matched_symbols(sample_regex, test_globs), expected_matches
        )

    def test_find_matched_symbols_empty_regex(self) -> None:
        # [FUNCTION DOC] Checks that an empty regex yields an empty set of matched symbols.
        sample_regex = re.compile("")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_matches: set[str] = set()
        self.assertEqual(
            find_matched_symbols(sample_regex, test_globs), expected_matches
        )

    def test_check_file_no_issues(self) -> None:
        # [FUNCTION DOC] Checks that a fixture file whose symbols are all tested yields no
        # lint messages.
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/good.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]
        self.assertEqual(len(check_file(sample_txt, test_globs)), 0)

    def test_check_empty_file(self) -> None:
        # [FUNCTION DOC] Checks that an empty fixture file yields no lint messages.
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/empty.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]
        self.assertEqual(len(check_file(sample_txt, test_globs)), 0)

    def test_check_file_with_untested_symbols(self) -> None:
        # [FUNCTION DOC] Checks that symbols listed in the bad fixture but absent from the
        # test cpp files (bbb on line 7, symD on line 8) each produce an untested-symbol
        # error message; the comparison ignores ordering by using sets.
        sample_txt = str(REPO_ROOT / "tools/test/header_only_linter_testdata/bad.txt")
        test_globs = ["tools/test/header_only_linter_testdata/*.cpp"]

        expected_msgs = [
            LintMessage(
                path=sample_txt,
                line=7,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[untested-symbol]",
                original=None,
                replacement=None,
                description=(
                    f"bbb has been included as a header-only API "
                    "but is not tested in any of CPP_TEST_GLOBS, which "
                    f"contains {CPP_TEST_GLOBS}.\n"
                    "Please add a .cpp test using the symbol without "
                    "linking anything to verify that the symbol is in "
                    "fact header-only. If you already have a test but it's"
                    " not found, please add the .cpp file to CPP_TEST_GLOBS"
                    " in tools/linters/adapters/header_only_linter.py."
                ),
            ),
            LintMessage(
                path=sample_txt,
                line=8,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[untested-symbol]",
                original=None,
                replacement=None,
                description=(
                    f"symD has been included as a header-only API "
                    "but is not tested in any of CPP_TEST_GLOBS, which "
                    f"contains {CPP_TEST_GLOBS}.\n"
                    "Please add a .cpp test using the symbol without "
                    "linking anything to verify that the symbol is in "
                    "fact header-only. If you already have a test but it's"
                    " not found, please add the .cpp file to CPP_TEST_GLOBS"
                    " in tools/linters/adapters/header_only_linter.py."
                ),
            ),
        ]
        self.assertEqual(set(check_file(sample_txt, test_globs)), set(expected_msgs))


if __name__ == "__main__":
    unittest.main()
