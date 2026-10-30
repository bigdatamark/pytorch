# Owner(s): ["module: spin"]

# [MODULE DOC] Test module for the spin command line tool. Defines the repository root path and
# one test case class; imports the spin test helper that invokes spin commands in process. Runs
# the test runner when executed as a program.
import ast
import os
import shutil
import tempfile
from pathlib import Path

from spin.tests.testutil import spin

from torch.testing._internal.common_utils import run_tests, TestCase


PYTORCH_ROOT = Path(__file__).parent.parent.parent


class TestSpin(TestCase):
    # [CLASS DOC] Tests for spin subcommands; currently only the pyrefly type inference command.
    def test_autotype(self):
        # [FUNCTION DOC] Copies the untyped fixture file into a temporary directory, changes to
        # the repository root, runs the spin pyrefly infer command on the copy, restores the
        # previous directory, then parses both the rewritten copy and the typed fixture and
        # asserts their syntax tree dumps are equal. The temporary directory is removed
        # afterwards.
        # KNOWN ISSUE: the working directory is restored only on success; if the spin call
        # raises, the process stays in the repository root.
        with tempfile.TemporaryDirectory() as tmp_dir:
            here = Path(__file__).parent
            untyped_file_name = "autotype_test_untyped.py"
            source = here / untyped_file_name
            dest = Path(tmp_dir) / untyped_file_name
            shutil.copy(source, dest)

            cwd = os.getcwd()
            os.chdir(PYTORCH_ROOT)
            spin("pyrefly", "infer", dest)
            # Post test code
            os.chdir(cwd)

            with open(dest) as f:
                retyped_contents = f.read()
            retyped_ast = ast.parse(retyped_contents)

            typed_file_name = "autotype_test_typed.py"
            with open(here / typed_file_name) as f:
                typed_contents = f.read()
            typed_ast = ast.parse(typed_contents)

            self.assertEqual(
                ast.dump(typed_ast),
                ast.dump(retyped_ast),
            )


if __name__ == "__main__":
    run_tests()
