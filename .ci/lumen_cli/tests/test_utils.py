# [MODULE DOC] Unit tests for the temp_environ and working_directory context managers in
# cli.lib.common.utils. Defines a base test case that isolates os.environ and the current
# directory per test (running each test inside a fresh temporary directory), plus two test
# classes covering set/restore, exception safety, empty-path no-op and missing-directory
# error. No import-time side effects beyond imports; runnable via unittest main.
import os
import tempfile
import unittest
from pathlib import Path

from cli.lib.common.utils import temp_environ, working_directory  # <-- replace import


class EnvIsolatedTestCase(unittest.TestCase):
    # [CLASS DOC] HELPER: Base test case giving each test a clean process environment. State:
    # a backup of os.environ, the original working directory and a per-test temporary
    # directory (which becomes the current directory). setUp creates them; tearDown restores
    # the directory, deletes the temporary directory and resets os.environ to the backup.
    """Base class that snapshots os.environ and CWD for isolation."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Back up os.environ, record the current directory (falling back
        # to the system temp dir, and switching to it, if the current one was deleted), then
        # create a fresh temporary directory and change into it. Mutates global process state.
        import os
        import tempfile

        self._env_backup = dict(os.environ)

        # Snapshot/repair CWD if it's gone
        try:
            self._cwd_backup = os.getcwd()
        except FileNotFoundError:
            # If CWD no longer exists, switch to a safe place and record that
            self._cwd_backup = tempfile.gettempdir()
            os.chdir(self._cwd_backup)

        # Create a temporary directory for the test to run in
        self._temp_dir = tempfile.mkdtemp()
        os.chdir(self._temp_dir)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Change back to the saved directory (or the system temp dir on
        # OS error), remove the temporary directory ignoring errors, then delete any env vars
        # added during the test and restore every backed-up value. Never raises on cleanup.
        import os
        import shutil
        import tempfile

        # Restore cwd first (before cleaning up temp dir)
        try:
            os.chdir(self._cwd_backup)
        except OSError:
            os.chdir(tempfile.gettempdir())

        # Clean up temporary directory
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

        # Restore env
        to_del = set(os.environ.keys()) - set(self._env_backup.keys())
        for k in to_del:
            os.environ.pop(k, None)
        for k, v in self._env_backup.items():
            os.environ[k] = v


class TestTempEnviron(EnvIsolatedTestCase):
    # [CLASS DOC] Tests for temp_environ: setting, overwriting, multi-variable handling and
    # restoration after normal exit and after an exception.
    def test_sets_and_restores_new_var(self):
        # [FUNCTION DOC] Verifies a new variable exists inside the context and is removed after.
        var = "TEST_TMP_ENV_NEW"
        self.assertNotIn(var, os.environ)

        with temp_environ({var: "123"}):
            self.assertEqual(os.environ[var], "123")

        self.assertNotIn(var, os.environ)  # removed after exit

    def test_overwrites_and_restores_existing_var(self):
        # [FUNCTION DOC] Verifies an existing variable is overridden inside the context and its
        # original value is restored afterwards.
        var = "TEST_TMP_ENV_OVERWRITE"
        os.environ[var] = "orig"

        with temp_environ({var: "override"}):
            self.assertEqual(os.environ[var], "override")

        self.assertEqual(os.environ[var], "orig")  # restored

    def test_multiple_vars_and_missing_cleanup(self):
        # [FUNCTION DOC] Verifies two variables at once: the newly added one is removed on exit
        # while the pre-existing one gets its original value back.
        v1, v2 = "TEST_ENV_V1", "TEST_ENV_V2"
        os.environ.pop(v1, None)
        os.environ[v2] = "keep"

        with temp_environ({v1: "a", v2: "b"}):
            self.assertEqual(os.environ[v1], "a")
            self.assertEqual(os.environ[v2], "b")

        self.assertNotIn(v1, os.environ)  # newly-added -> removed
        self.assertEqual(os.environ[v2], "keep")  # pre-existing -> restored

    def test_restores_even_on_exception(self):
        # [FUNCTION DOC] Verifies the variable is removed after the context even when the body
        # raises a runtime error, and the error propagates.
        var = "TEST_TMP_ENV_EXCEPTION"
        self.assertNotIn(var, os.environ)

        with self.assertRaises(RuntimeError):
            with temp_environ({var: "x"}):
                self.assertEqual(os.environ[var], "x")
                raise RuntimeError("boom")

        self.assertNotIn(var, os.environ)  # removed after exception


class TestWorkingDirectory(EnvIsolatedTestCase):
    # [CLASS DOC] Tests for working_directory: changing, restoring, empty-path no-op, restore
    # on exception and error for a missing directory.
    def test_changes_and_restores(self):
        # [FUNCTION DOC] Verifies the current directory equals the target inside the context and
        # returns to the starting directory afterwards.
        start = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "wd"
            target.mkdir()

            with working_directory(str(target)):
                self.assertEqual(Path.cwd().resolve(), target.resolve())

        self.assertEqual(Path.cwd(), start)

    def test_noop_when_empty_path(self):
        # [FUNCTION DOC] Verifies an empty path leaves the current directory unchanged.
        start = Path.cwd()
        with working_directory(""):
            self.assertEqual(Path.cwd(), start)
        self.assertEqual(Path.cwd(), start)

    def test_restores_on_exception(self):
        # [FUNCTION DOC] Verifies the starting directory is restored after the body raises a
        # value error (paths resolved to cope with symlinked temp roots).
        start = Path.cwd()

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "wd_exc"
            target.mkdir()

            with self.assertRaises(ValueError):
                with working_directory(str(target)):
                    # Normalize both sides to handle /var -> /private/var
                    self.assertEqual(Path.cwd().resolve(), target.resolve())
                    raise ValueError("boom")

        self.assertEqual(Path.cwd().resolve(), start.resolve())

    def test_raises_for_missing_dir(self):
        # [FUNCTION DOC] Verifies a missing target directory raises a file-not-found error and
        # the current directory is unchanged.
        start = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist"
            with self.assertRaises(FileNotFoundError):
                # os.chdir should raise before yielding
                with working_directory(str(missing)):
                    pass
        self.assertEqual(Path.cwd(), start)


if __name__ == "__main__":
    unittest.main(verbosity=2)
