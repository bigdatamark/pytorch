# Owner(s): ["module: unknown"]
# [MODULE DOC] Test that the portable inductor config snapshot is deterministic across machines.
# Defines a path regex plus the current username and hostname (looked up at import time),
# and a TestCase that scans every saved config value for absolute paths, the username or
# the hostname, which would make compile cache keys machine dependent.

import getpass
import re
import socket

from torch._inductor import config as inductor_config
from torch.testing._internal.common_utils import run_tests, TestCase


PATH_PATTERN = re.compile(
    r"(?:\/[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.\/]+)|"  # Unix
    r"(?:[a-zA-Z]:\\[a-zA-Z0-9_\-\.\\]+)"  # Windows
)


USERNAME = getpass.getuser()
HOSTNAME = socket.gethostname()


class TestConfigModule(TestCase):
    # [CLASS DOC] Checks inductor config values for machine-specific content.
    # KNOWN_PATH_CONFIGS lists config keys allowed to hold absolute paths and are skipped by
    # the whole-config scan. The helper check_deterministic does the recursive check.
    # Config keys that legitimately contain absolute paths, for example,
    # /opt/clang-15/bin/clang
    KNOWN_PATH_CONFIGS = {"cpp.cxx"}

    def check_deterministic(self, key: str, value: object):
        # [FUNCTION DOC] Recursively assert that one config value has no machine-specific data.
        # Numbers, booleans and None are accepted. Strings must not match the path pattern at the
        # start, nor contain the username or hostname (each only checked when non-empty). Lists and
        # tuples are checked per item and mappings per value, with the key text extended for
        # messages.
        # Fails the test through an assertion for violations or for any other value type.
        if isinstance(value, (int, float, bool)) or value is None:
            return
        elif isinstance(value, str):
            self.assertFalse(
                PATH_PATTERN.match(value),
                lambda msg: f"{msg}\nDetected path in config value '{value}', key='{key}', "
                "this may cause non-deterministic behavior in compile caching.",
            )
            if USERNAME:
                self.assertNotIn(
                    USERNAME,
                    value,
                    lambda msg: f"{msg}\nDetected username in config value '{value}', key='{key}', "
                    "this may cause non-deterministic behavior in compile caching.",
                )
            if HOSTNAME:
                self.assertNotIn(
                    HOSTNAME,
                    value,
                    lambda msg: f"{msg}\nDetected hostname in config value '{value}', key='{key}', "
                    "this may cause non-deterministic behavior in compile caching.",
                )
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self.check_deterministic(f"{key}[{i}]", item)
        elif isinstance(value, dict):
            for k, v in value.items():
                self.check_deterministic(f"{key}[{k}]", v)
        else:
            self.fail(f"Unexpected type: {type(value)}")

    def test_basic_detection(self):
        # [FUNCTION DOC] Self-test of check_deterministic on good and bad inputs.
        # Plain strings, numbers, booleans and simple containers must pass. A temp-directory path
        # string, a string embedding the username and one embedding the hostname must each raise an
        # assertion error with the matching message (the last two only when that name is
        # non-empty).
        self.check_deterministic("test", "test")
        self.check_deterministic("test", 1)
        self.check_deterministic("test", 1.0)
        self.check_deterministic("test", True)
        self.check_deterministic("test", False)
        self.check_deterministic("test", [1, 2, 3])
        self.check_deterministic("test", (1, 2, 3))
        self.check_deterministic("test", {"a": 1, "b": 2, "c": 3})
        with self.assertRaisesRegex(
            AssertionError, "Detected path in config value '.*', key='test'"
        ):
            self.check_deterministic("test", "/tmp/test")
        if USERNAME:
            with self.assertRaisesRegex(
                AssertionError, "Detected username in config value '.*', key='test'"
            ):
                self.check_deterministic("test", f"123_{USERNAME}")
        if HOSTNAME:
            with self.assertRaisesRegex(
                AssertionError, "Detected hostname in config value '.*', key='test'"
            ):
                self.check_deterministic("test", f"456-{HOSTNAME}")

    def test_inductor_config_hash_portable_deterministic(self):
        # [FUNCTION DOC] Verify the portable inductor config snapshot has no paths, user or host.
        # Takes the snapshot from save_config_portable and runs the determinism check on every
        # entry except keys listed in KNOWN_PATH_CONFIGS.
        torch_config = inductor_config.save_config_portable()

        for key, value in torch_config.items():
            if key in self.KNOWN_PATH_CONFIGS:
                continue
            self.check_deterministic(key, value)

    def test_inductor_config_hash_portable_without_ignore(self):
        # [FUNCTION DOC] Verify the ignore list keeps cutlass directories out of the snapshot.
        # For each of the cuda, xpu and cutlass cutlass_dir keys: temporarily remove it from the
        # inductor cache-config ignore prefix list, take the portable snapshot, and expect the
        # determinism scan to fail with a path-detected error for that key. The list entry is
        # reinserted at its original index in a finally block, so the global config is restored.
        for cutlass_key in ("cuda", "xpu", "cutlass"):
            cutlass_dir_key = f"{cutlass_key}.cutlass_dir"
            idx = inductor_config._cache_config_ignore_prefix.index(cutlass_dir_key)
            inductor_config._cache_config_ignore_prefix.remove(cutlass_dir_key)
            try:
                changed_torch_config = inductor_config.save_config_portable()
                with self.assertRaisesRegex(
                    AssertionError,
                    f"Detected path in config value '.*', key='{cutlass_dir_key}'",
                ):
                    for key, value in changed_torch_config.items():
                        if key in self.KNOWN_PATH_CONFIGS:
                            continue
                        self.check_deterministic(key, value)
            finally:
                inductor_config._cache_config_ignore_prefix.insert(idx, cutlass_dir_key)


if __name__ == "__main__":
    run_tests()
