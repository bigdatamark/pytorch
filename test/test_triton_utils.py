# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that the triton backend loader in the triton utility module turns driver
# load failures into an ImportError that names the cache directory and hints at a noexec
# mount. Tests are skipped when triton is not installed.

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils import _triton as triton_utils


class TestTritonUtils(TestCase):
    # [CLASS DOC] Test case covering error reporting of the cached triton backend lookup. It
    # has no state; every test fakes the triton runtime driver.
    def _run_triton_backend_load_failure(self, exc, env):
        # [FUNCTION DOC] HELPER: Replaces the triton runtime driver with a fake whose current-
        # target query raises the given exception, patches the process environment with the
        # given mapping (removing the triton cache dir variable when the mapping lacks it),
        # clears the backend lru cache before and after, and asks the utility module for its
        # cache dir text. Then calls the cached backend lookup, expecting an ImportError.
        # Returns a pair of the error message string and the cache dir value. Fails if no
        # ImportError is raised.
        driver_mod = importlib.import_module("triton.runtime.driver")
        fake_driver = SimpleNamespace(
            active=SimpleNamespace(get_current_target=mock.Mock(side_effect=exc))
        )

        triton_utils.triton_backend.cache_clear()
        try:
            with (
                mock.patch.object(driver_mod, "driver", fake_driver),
                mock.patch.dict(os.environ, env),
            ):
                if "TRITON_CACHE_DIR" not in env:
                    os.environ.pop("TRITON_CACHE_DIR", None)
                cache_dir = triton_utils._triton_cache_dir_for_error_message()
                with self.assertRaises(ImportError) as cm:
                    triton_utils.triton_backend()
        finally:
            triton_utils.triton_backend.cache_clear()

        return str(cm.exception), cache_dir

    @unittest.skipUnless(triton_utils.has_triton_package(), "requires triton")
    def test_triton_backend_reports_noexec_cache_dir(self):
        # [FUNCTION DOC] For both ImportError and OSError raised by the fake driver with a
        # shared-object mapping failure text, and an explicit cache dir in the environment,
        # checks the resulting message mentions that cache dir, the word noexec and an
        # executable filesystem.
        for exc_type in (ImportError, OSError):
            with self.subTest(exc_type=exc_type):
                msg, _ = self._run_triton_backend_load_failure(
                    exc_type(
                        "/noexec/triton/hash/cuda_utils.so: "
                        "failed to map segment from shared object"
                    ),
                    {"TRITON_CACHE_DIR": "/noexec/triton"},
                )

                self.assertIn("TRITON_CACHE_DIR=/noexec/triton", msg)
                self.assertIn("noexec", msg)
                self.assertIn("executable filesystem", msg)

    @unittest.skipUnless(triton_utils.has_triton_package(), "requires triton")
    def test_triton_backend_reports_default_cache_dir(self):
        # [FUNCTION DOC] With no cache dir in the environment and an ImportError about a
        # failed segment mapping, checks the cache dir reported by the helper is not None and
        # appears in parentheses in the message, along with the noexec and executable
        # filesystem hints.
        msg, cache_dir = self._run_triton_backend_load_failure(
            ImportError(
                "/default/triton/hash/cuda_utils.so: "
                "failed to map segment from shared object"
            ),
            {},
        )

        self.assertIsNotNone(cache_dir)
        self.assertIn(f"({cache_dir})", msg)
        self.assertIn("noexec", msg)
        self.assertIn("executable filesystem", msg)

    @unittest.skipUnless(triton_utils.has_triton_package(), "requires triton")
    def test_triton_backend_reraises_other_load_errors(self):
        # [FUNCTION DOC] For ImportError and OSError with an unrelated message, installs a
        # fake driver and checks the backend lookup re-raises the same exception type with the
        # original message instead of the cache dir hint. Clears the lru cache before and
        # after each subtest.
        driver_mod = importlib.import_module("triton.runtime.driver")

        for exc_type in (ImportError, OSError):
            fake_driver = SimpleNamespace(
                active=SimpleNamespace(
                    get_current_target=mock.Mock(side_effect=exc_type("other failure"))
                )
            )

            triton_utils.triton_backend.cache_clear()
            try:
                with (
                    self.subTest(exc_type=exc_type),
                    mock.patch.object(driver_mod, "driver", fake_driver),
                ):
                    with self.assertRaisesRegex(exc_type, "other failure"):
                        triton_utils.triton_backend()
            finally:
                triton_utils.triton_backend.cache_clear()


if __name__ == "__main__":
    run_tests()
