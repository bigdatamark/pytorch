# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for torch._inductor.remote_cache: logging and failure handling in
# RemoteCache, dump_cache_stats robustness when stderr was swapped, and the local autotune cache
# (creation, corrupt entries treated as misses, atomic writes). Defines backend/cache fakes.
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from unittest import mock

from torch._inductor.remote_cache import (
    create_cache,
    dump_cache_stats,
    RemoteCache,
    RemoteCacheBackend,
    RemoteCachePassthroughSerde,
)
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.logging_utils import log_settings


class FailingBackend(RemoteCacheBackend):
    # [CLASS DOC] HELPER: Backend whose get and put always raise AssertionError with fixed
    # messages.
    def _get(self, key):
        # [FUNCTION DOC] HELPER: Always raises AssertionError with message testget.
        raise AssertionError("testget")

    def _put(self, key, data):
        # [FUNCTION DOC] HELPER: Always raises AssertionError with message testput.
        raise AssertionError("testput")


class NoopBackend(RemoteCacheBackend):
    # [CLASS DOC] HELPER: Backend that stores nothing; get returns None and put does nothing.
    def _get(self, key):
        # [FUNCTION DOC] HELPER: Always reports a cache miss by returning None.
        return None

    def _put(self, key, data):
        # [FUNCTION DOC] HELPER: Discards the data and returns None.
        return None


@dataclass
class _TestSample:
    # [CLASS DOC] HELPER: Minimal log sample dataclass with one optional string field named
    # fail.
    fail: str = None


class FakeCache(RemoteCache):
    # [CLASS DOC] HELPER: RemoteCache using the failing backend and passthrough serialization;
    # it captures the sample it logs so tests can inspect it in the sample attribute.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the base cache with FailingBackend and passthrough
        # serde.
        super().__init__(FailingBackend(), RemoteCachePassthroughSerde())

    def _create_sample(self):
        # [FUNCTION DOC] HELPER: Returns a fresh sample object for the base class to fill in.
        return _TestSample()

    def _log_sample(self, sample):
        # [FUNCTION DOC] HELPER: Stores the received sample on self instead of logging it.
        self.sample = sample


class TestRemoteCache(TestCase):
    # [CLASS DOC] Tests for RemoteCache failure logging, cache stats dumping and the local
    # autotune cache created through create_cache.
    def test_normal_logging(
        self,
    ) -> None:
        # [FUNCTION DOC] Checks put and get on a no-op backend complete without raising.
        c = RemoteCache(NoopBackend(), RemoteCachePassthroughSerde())
        c.put("test", "value")
        c.get("test")

    def test_dump_cache_stats_after_stderr_capture_closed(
        self,
    ) -> None:
        # [FUNCTION DOC] Replaces stderr with a stream that is then closed, and swaps in a live
        # stream before dumping cache stats under inductor logging; expects the stats text on
        # the live stream and no logging error text. Restores the original stderr afterwards.
        captured_stderr = io.StringIO()
        live_stderr = io.StringIO()
        old_stderr = sys.stderr

        try:
            sys.stderr = captured_stderr
            with log_settings("inductor"):
                captured_stderr.close()
                sys.stderr = live_stderr
                dump_cache_stats()
        finally:
            sys.stderr = old_stderr

        self.assertIn("Cache Metrics", live_stderr.getvalue())
        self.assertNotIn("Logging error", live_stderr.getvalue())

    def test_failure_no_sample(
        self,
    ) -> None:
        # [FUNCTION DOC] Checks backend errors propagate from put and get as AssertionError for
        # a plain RemoteCache (no sample subclass).
        c = RemoteCache(FailingBackend(), RemoteCachePassthroughSerde())
        with self.assertRaises(AssertionError):
            c.put("test", "value")
        with self.assertRaises(AssertionError):
            c.get("test")

    def test_failure_logging(
        self,
    ) -> None:
        # [FUNCTION DOC] Checks that backend failures still raise, and that the captured sample
        # records the failure reason (testput for put, testget for get).
        c = FakeCache()
        with self.assertRaises(AssertionError):
            c.put("test", "value")
        self.assertEqual(c.sample.fail_reason, "testput")
        with self.assertRaises(AssertionError):
            c.get("test")
        self.assertEqual(c.sample.fail_reason, "testget")

    def test_create_local_autotune_cache(
        self,
    ) -> None:
        # [FUNCTION DOC] Creates a local autotune cache in a temp directory, stores a dict under
        # a nested key path, and checks the file exists and get returns the same dict.
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            expected = {"value": 1}
            c.put(key, expected)

            self.assertTrue(os.path.exists(key))
            self.assertEqual(c.get(key), expected)

    def test_local_autotune_cache_corrupt_json_is_miss(
        self,
    ) -> None:
        # [FUNCTION DOC] Writes a file with two concatenated JSON objects, then expects get to
        # return None (miss) and emit a warning mentioning the corrupt entry, a JSON decode
        # error and extra data, without a traceback.
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            os.makedirs(os.path.dirname(key), exist_ok=True)
            with open(key, "wb") as fd:
                fd.write(b'{"value": 1}{"value": 2}')

            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            with self.assertLogs("torch._inductor.remote_cache", level="WARNING") as cm:
                self.assertIsNone(c.get(key))
            self.assertIn("Ignoring corrupt local cache entry", cm.output[0])
            self.assertIn("JSONDecodeError", cm.output[0])
            self.assertIn("Extra data", cm.output[0])
            self.assertNotIn("Traceback", cm.output[0])

    def test_local_autotune_cache_invalid_utf8_is_miss(
        self,
    ) -> None:
        # [FUNCTION DOC] Writes a file containing an invalid UTF-8 byte, then expects get to
        # return None and log a warning naming a Unicode decode error and invalid start byte, no
        # traceback.
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            os.makedirs(os.path.dirname(key), exist_ok=True)
            with open(key, "wb") as fd:
                fd.write(b"\xff")

            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            with self.assertLogs("torch._inductor.remote_cache", level="WARNING") as cm:
                self.assertIsNone(c.get(key))
            self.assertIn("Ignoring corrupt local cache entry", cm.output[0])
            self.assertIn("UnicodeDecodeError", cm.output[0])
            self.assertIn("invalid start byte", cm.output[0])
            self.assertNotIn("Traceback", cm.output[0])

    def test_local_autotune_cache_put_uses_atomic_write(
        self,
    ) -> None:
        # [FUNCTION DOC] Patches the atomic file writer; puts a dict and checks the writer is
        # called once with the key path and a JSON string that decodes to the stored dict.
        with tempfile.TemporaryDirectory() as tmpdir:
            key = os.path.join(tmpdir, "cache", "entry.best_config")
            c = create_cache("local-autotune", local_cache_cls="LocalAutotuneCache")
            if c is None:
                self.fail("Expected local autotune cache")

            expected = {"value": 1}
            with mock.patch("torch._inductor.codecache.write_atomic") as write_atomic:
                c.put(key, expected)

            write_atomic.assert_called_once()
            args, _kwargs = write_atomic.call_args
            self.assertEqual(args[0], key)
            self.assertEqual(json.loads(args[1]), expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
