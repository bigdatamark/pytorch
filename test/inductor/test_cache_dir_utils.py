# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for torch._inductor.runtime.cache_dir_utils: relative cache dir resolution
# and the fallback names of the default per-user cache dir when the user name cannot be
# determined.

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from torch._inductor.runtime import cache_dir_utils
from torch._inductor.test_case import run_tests, TestCase


class TestCacheDirUtils(TestCase):
    # [CLASS DOC] Test case covering cache_dir and default_cache_dir; each test patches the
    # environment, getpass, os or is_fbcode with mocks so no real user lookup is required.
    def test_cache_dir_resolves_relative_env(self):
        # [FUNCTION DOC] In a temporary working directory, sets the cache dir environment variable
        # to a relative path and checks that the cache dir function returns the absolute path,
        # rewrites the environment variable to that absolute path, and keeps returning it after
        # the working directory changes to the parent. The original working directory is restored
        # in a finally block.
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": "cache"}):
                    expected = os.path.abspath("cache")
                    self.assertEqual(cache_dir_utils.cache_dir(), expected)
                    self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], expected)

                    os.chdir(os.path.dirname(tmpdir))
                    self.assertEqual(cache_dir_utils.cache_dir(), expected)
            finally:
                os.chdir(old_cwd)

    def test_default_cache_dir_falls_back_to_uid(self):
        # [FUNCTION DOC] For each of three exceptions (KeyError, ModuleNotFoundError, OSError)
        # raised by the user name lookup, with uid 1001 and fbcode disabled, checks the default
        # cache dir is the system temp dir joined with a name built from uid_1001. Uses a subtest
        # per exception type.
        for exception in (
            KeyError("getpwuid(): uid not found: 1001"),
            ModuleNotFoundError("No module named 'pwd'"),
            OSError("getpwuid(): uid not found: 1001"),
        ):
            with (
                self.subTest(exception=type(exception)),
                mock.patch.object(
                    cache_dir_utils.getpass,
                    "getuser",
                    side_effect=exception,
                ),
                mock.patch.object(
                    cache_dir_utils.os, "getuid", return_value=1001, create=True
                ),
                mock.patch.object(cache_dir_utils, "is_fbcode", return_value=False),
            ):
                self.assertEqual(
                    cache_dir_utils.default_cache_dir(),
                    os.path.join(tempfile.gettempdir(), "torchinductor_uid_1001"),
                )

    def test_default_cache_dir_falls_back_without_getuid(self):
        # [FUNCTION DOC] Makes the user name lookup raise OSError and replaces the os module in
        # the module under test with a namespace that has only the path attribute (no getuid),
        # with fbcode disabled. Checks the default cache dir ends in the unknown_user name under
        # the system temp dir.
        with (
            mock.patch.object(
                cache_dir_utils.getpass,
                "getuser",
                side_effect=OSError("user unavailable"),
            ),
            mock.patch.object(cache_dir_utils, "os", SimpleNamespace(path=os.path)),
            mock.patch.object(cache_dir_utils, "is_fbcode", return_value=False),
        ):
            self.assertEqual(
                cache_dir_utils.default_cache_dir(),
                os.path.join(tempfile.gettempdir(), "torchinductor_unknown_user"),
            )


if __name__ == "__main__":
    run_tests()
