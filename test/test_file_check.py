# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for the FileCheck utility from torch.testing: verifies the warning for an
# unrun instance and that the chained Python API matches the inline directive-string form.

from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFileCheck(TestCase):
    # [CLASS DOC] Test case holding the two FileCheck behavior tests; keeps no state.
    def test_not_run(self):
        # [FUNCTION DOC] Runs a subprocess that builds a FileCheck with one check and deletes it
        # without running it, then verifies the captured stdout reports the instance was never run
        # and lists the pending check line.
        stdout, _ = self.run_process_no_exception(
            """\
from torch.testing import FileCheck
file_check = FileCheck().check("not run")
del file_check
""",
        )
        FileCheck().check("You have not run this instance of FileCheck!").check_next(
            "FileCheck checks:"
        ).check_next("\tCHECK: not run").run(stdout)

    def test_all_python_api(self):
        # [FUNCTION DOC] Exercises every check kind via the chained API (check, check_not,
        # check_same, check_next, check_count, check_dag, check_source_highlighted, check_regex)
        # against a sample text, then runs the equivalent inline directive comment string against
        # the same text. Passes if neither run raises.
        test_string = """
check check_same
check_next
check_count
check_dag
check_source_highlighted
~~~~~~~~~~~~~~~~~~~~~~~~
check_regex
"""
        FileCheck().check("check").check_not("check_not").check_same(
            "check_same"
        ).check_next("check_next").check_count("check_count", 1).check_dag(
            "check_dag"
        ).check_source_highlighted("check_source_highlighted").check_regex(
            r"check_.+"
        ).run(test_string)

        FileCheck().run(
            """
# CHECK: check
# CHECK-NOT: check_not
# CHECK-SAME: check_same
# CHECK-NEXT: check_next
# CHECK-DAG: check_dag
# CHECK-SOURCE-HIGHLIGHTED: check_source_highlighted
# CHECK-REGEX: check_.+
        """,
            test_string,
        )


if __name__ == "__main__":
    run_tests()
