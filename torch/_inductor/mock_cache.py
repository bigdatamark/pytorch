# mypy: ignore-errors

# [MODULE DOC] Test utilities that mock Inductor's remote caches. Provides hit/miss/put statistics, per-thread in-memory
# cache storage, a MockBackend that replaces RemoteCacheBackend, and a PatchCaches context manager that swaps the
# cache backends for mocks. No import-time side effects beyond creating the thread-local global_stats object.
from __future__ import annotations

import contextlib
import dataclasses
import sys
import threading
from typing import Any, TYPE_CHECKING
from typing_extensions import override, Self
from unittest.mock import patch

from torch._inductor import config
from torch._inductor.remote_cache import RemoteCacheBackend


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType


@dataclasses.dataclass
class Stats:
    # [CLASS DOC] Counters for cache puts, get hits and get misses. Supports in-place addition, reset and string formatting.
    num_put: int = 0
    num_get_hit: int = 0
    num_get_miss: int = 0

    def __iadd__(self, other: Stats) -> Self:
        # [FUNCTION DOC] Add another stats object's counters into this one in place and return self.
        self.num_put += other.num_put
        self.num_get_hit += other.num_get_hit
        self.num_get_miss += other.num_get_miss
        return self

    def reset(self) -> None:
        # [FUNCTION DOC] Zero all counters.
        self.num_put = 0
        self.num_get_hit = 0
        self.num_get_miss = 0

    def __str__(self) -> str:
        # [FUNCTION DOC] Format the counters as a string listing puts, misses and hits.
        return "".join(
            (
                f"puts: {self.num_put}, ",
                f"misses: {self.num_get_miss}, ",
                f"hits: {self.num_get_hit}, ",
            )
        )

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equality on the three counters, accepting Stats or _GlobalItemStats (the default dataclass equality would reject mixed types).
        # Returns false for other types.
        # Dataclass's default __eq__ checks that the types are the same so can't
        # be used with _GlobalItemStats.
        return (
            isinstance(other, (Stats, _GlobalItemStats))
            and self.num_put == other.num_put
            and self.num_get_hit == other.num_get_hit
            and self.num_get_miss == other.num_get_miss
        )


class _GlobalItemStats(Stats):
    # [CLASS DOC] HELPER: Stats plus a dictionary of cached key to value entries for one cache kind.
    cache: dict[str, object]

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initialize zeroed counters and an empty cache mapping.
        super().__init__()
        self.cache = {}

    def reset(self) -> None:
        # [FUNCTION DOC] HELPER: Zero counters and replace the cache with a new empty mapping.
        super().reset()
        self.cache = {}


# The cache states are thread-local so if we're running multiple tests at once
# they won't cross contaminate. However - it needs to be "global" because we
# allow code to create new cache clients which refer to the same cache (because
# it's a remote cache).


class _GlobalStats(threading.local):
    # [CLASS DOC] HELPER: Thread-local holder of one _GlobalItemStats per cache kind (autotune_local, autotune_remote, bundled_autotune,
    # fx_graph, triton, aot_autograd, dynamo_pgo). Thread-local so concurrent tests do not interfere.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create a fresh stats object for each cache kind (per thread).
        self.autotune_local = _GlobalItemStats()
        self.autotune_remote = _GlobalItemStats()
        self.bundled_autotune = _GlobalItemStats()
        self.fx_graph = _GlobalItemStats()
        self.triton = _GlobalItemStats()
        self.aot_autograd = _GlobalItemStats()
        self.dynamo_pgo = _GlobalItemStats()

    def reset(self) -> None:
        # [FUNCTION DOC] Reset counters and contents of all cache kinds.
        self.autotune_local.reset()
        self.autotune_remote.reset()
        self.bundled_autotune.reset()
        self.fx_graph.reset()
        self.triton.reset()
        self.aot_autograd.reset()
        self.dynamo_pgo.reset()

    def get_stat(self, name: str) -> _GlobalItemStats:
        # [FUNCTION DOC] Return the stats object for the given cache kind name. Raises an attribute error for unknown names.
        return getattr(self, name)

    def report(self):
        # [FUNCTION DOC] Print cache statistics and the entries of every non-empty cache to stderr; values are shown by repr truncated to 100 characters.
        subs = (
            ("autotune_local", self.autotune_local),
            ("autotune_remote", self.autotune_remote),
            ("bundled_autotune", self.bundled_autotune),
            ("fx_graph", self.fx_graph),
            ("triton", self.triton),
            ("aot_autograd", self.aot_autograd),
            ("dynamo_pgo", self.dynamo_pgo),
        )

        print("Cache Stats:", file=sys.stderr)
        for name, sub in subs:
            print(f"  {name}: {sub}", file=sys.stderr)

        print("Cache Entries:", file=sys.stderr)
        for name, sub in subs:
            if sub.cache:
                print(f"  {name}:", file=sys.stderr)
                for k, v in sorted(sub.cache.items()):
                    v = repr(v)
                    if len(v) > 100:
                        v = v[:100] + "..."
                    print(f"    {k!r}: {v}", file=sys.stderr)


global_stats = _GlobalStats()


class MockBackend(RemoteCacheBackend[Any]):
    # [CLASS DOC] HELPER: In-memory remote cache backend for tests. Reads and writes the global_stats entry named at construction,
    # counting hits, misses and puts.
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Remember the cache kind name.
        self._name = name

    @staticmethod
    def with_name(name: str) -> Callable[[], MockBackend]:
        # [FUNCTION DOC] HELPER: Return a zero-argument factory that builds a MockBackend with the given name.
        def wrapper() -> MockBackend:
            # [FUNCTION DOC] HELPER: Build a MockBackend for the captured name.
            return MockBackend(name)

        return wrapper

    @override
    def _get(self, key: str) -> Any | None:
        # [FUNCTION DOC] HELPER: Look up key in the named cache, counting a hit or miss; returns the stored value or None.
        stat = global_stats.get_stat(self._name)
        if key in stat.cache:
            stat += Stats(num_get_hit=1)
            return stat.cache.get(key)
        else:
            stat += Stats(num_get_miss=1)
            return None

    @override
    def _put(self, key: str, data: Any) -> None:
        # [FUNCTION DOC] HELPER: Count a put and store data under key in the named cache.
        stat = global_stats.get_stat(self._name)
        stat += Stats(num_put=1)
        stat.cache[key] = data


# List of configs for each cache
_CACHE_CONFIG_EN = (
    "fx_graph_cache",
    "fx_graph_remote_cache",
    "autotune_local_cache",
    "autotune_remote_cache",
    "bundled_autotune_remote_cache",
)


class PatchCaches(contextlib.AbstractContextManager):
    # [CLASS DOC] Context manager (and test setUp/tearDown helper) that patches every remote cache class's backend_override_cls
    # with MockBackend factories so tests can inspect global_stats. Entering resets stats; exiting undoes the patches.
    # Holds an ExitStack of active patches; fbcode-only classes are patched only when running in fbcode.
    @classmethod
    def setUp(cls):
        # [FUNCTION DOC] Class-level test setup: save the current values of the cache enable configs and set them all to false.
        # If this test is using PatchCaches then disable all the caches by
        # default, letting the tests turn them on explicitly. This is because
        # tests using PatchCaches will often want to check stats explicitly.
        cls._savedCacheState = {}
        for name in _CACHE_CONFIG_EN:
            if hasattr(config, name):
                cls._savedCacheState[name] = getattr(config, name)
            setattr(config, name, False)

    @classmethod
    def tearDown(cls):
        # [FUNCTION DOC] Class-level test teardown: delete each cache config override and restore any saved value.
        # KNOWN ISSUE: delattr raises an error if a config name was never set on the config module.
        # Restore cache defaults
        for name in _CACHE_CONFIG_EN:
            delattr(config, name)
            if name in cls._savedCacheState:
                setattr(config, name, cls._savedCacheState[name])

    def __init__(self) -> None:
        # [FUNCTION DOC] Create the ExitStack for patches.
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> Self:
        # [FUNCTION DOC] Reset global stats and install the mock backends via patches (including fbcode ones in fbcode). Returns self.
        global_stats.reset()
        self._stack.__enter__()

        ctx = patch(
            "torch._inductor.remote_cache.LocalAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune_local"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteAutotuneCache.backend_override_cls",
            MockBackend.with_name("autotune_remote"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteBundledAutotuneCache.backend_override_cls",
            MockBackend.with_name("bundled_autotune"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteFxGraphCache.backend_override_cls",
            MockBackend.with_name("fx_graph"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteAOTAutogradCache.backend_override_cls",
            MockBackend.with_name("aot_autograd"),
        )
        self._stack.enter_context(ctx)

        ctx = patch(
            "torch._inductor.remote_cache.RemoteDynamoPGOCache.backend_override_cls",
            MockBackend.with_name("dynamo_pgo"),
        )
        self._stack.enter_context(ctx)

        if config.is_fbcode():
            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAutotuneCache.backend_override_cls",
                MockBackend.with_name("autotune_remote"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteBundledAutotuneCache.backend_override_cls",
                MockBackend.with_name("bundled_autotune"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteFxGraphCache.backend_override_cls",
                MockBackend.with_name("fx_graph"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "triton.fb.fb_memcache.FbMemcacheRemoteKernelCache.backend_override_cls",
                MockBackend.with_name("triton"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteAOTAutogradCache.backend_override_cls",
                MockBackend.with_name("aot_autograd"),
            )
            self._stack.enter_context(ctx)

            ctx = patch(
                "torch._inductor.fb.remote_cache.FbRemoteDynamoPGOCache.backend_override_cls",
                MockBackend.with_name("dynamo_pgo"),
            )
            self._stack.enter_context(ctx)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Undo all patches by closing the ExitStack; exceptions are not suppressed.
        self._stack.__exit__(exc_type, exc_value, traceback)
