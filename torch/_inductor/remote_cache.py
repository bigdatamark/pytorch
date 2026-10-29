# [MODULE DOC] Remote/local cache abstractions for Inductor: byte-oriented backends (local filesystem,
# Redis), serializers (JSON, passthrough), the RemoteCache controller combining them, concrete cache
# types used for stats and test overrides, a create_cache factory, and simple hit/miss/put/exception
# statistics dumped at exit. Import-time side effects: optionally imports redis (None if missing),
# imports the internal Sample type in fbcode, defines timed-wrapper partials, creates the global
# cache_stats, and registers an atexit hook that logs stats.
from __future__ import annotations

import atexit
import collections
import dataclasses
import functools
import json
import logging
import os
import sys
import typing
from abc import abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar, Union
from typing_extensions import override

from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch.monitor import _WaitCounter


if typing.TYPE_CHECKING:
    from collections.abc import Callable


try:
    import redis
except ImportError:
    redis = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


if config.is_fbcode():
    from rfe.scubadata.scubadata_py3 import (  # type: ignore[import-not-found]
        Sample as Sample_,
    )

    Sample: TypeAlias = Sample_
else:
    Sample: TypeAlias = type[object]  # type: ignore[misc,no-redef]


_T = TypeVar("_T")
_U = TypeVar("_U")


remote_fx_cache_get_timed = functools.partial(
    dynamo_timed,
    "FbRemoteFxGraphCache.get",
    phase_name="remote_fx_graph_cache_get",
    log_pt2_compile_event=False,
    dynamo_compile_column_us="remote_fx_graph_cache_get_time_us",
    log_waitcounter=True,
)
remote_fx_cache_put_timed = functools.partial(
    dynamo_timed,
    "FbRemoteFxGraphCache.put",
    phase_name="remote_fx_graph_cache_put",
    log_pt2_compile_event=False,
    dynamo_compile_column_us="remote_fx_graph_cache_put_time_us",
    log_waitcounter=True,
)


class RemoteCacheBackend(Generic[_T]):
    """
    A backend implementation for cache storage. Most backends are remote, but
    local filesystem backends use the same bytes-oriented interface. For
    structured data use a RemoteCache.
    """

    # [CLASS DOC] Abstract base for cache storage backends keyed by string. Subclasses implement
    # _get and _put; the public get and put wrap them with statistics recording under a name
    # derived from the concrete class (attribute _name).
    def __init__(self) -> None:
        # [FUNCTION DOC] Sets the stats name to "backend:" plus the concrete class name.
        self._name = f"backend:{type(self).__name__}"

    @abstractmethod
    def _get(self, key: str) -> _T | None:
        # [FUNCTION DOC] HELPER: Abstract; subclasses return the stored value for key or None if absent.
        pass

    @abstractmethod
    def _put(self, key: str, data: _T) -> None:
        # [FUNCTION DOC] HELPER: Abstract; subclasses store data under key.
        pass

    def get(self, key: str) -> _T | None:
        # [FUNCTION DOC] Fetches the value for key via _get and records a hit or miss (a None result
        # is a miss). On any exception, records an exception stat and re-raises.
        try:
            value = self._get(key)
            cache_stats.get(self._name, value)
        except Exception:
            cache_stats.exception(self._name)
            raise
        return value

    def put(self, key: str, data: _T) -> None:
        # [FUNCTION DOC] Stores data under key via _put and records a put stat; on any exception
        # records an exception stat and re-raises.
        try:
            self._put(key, data)
            cache_stats.put(self._name)
        except Exception:
            cache_stats.exception(self._name)
            raise


# Serde that encodes from _T to _U and decodes from _U to _T.
class RemoteCacheSerde(Generic[_T, _U]):
    # [CLASS DOC] Abstract serializer/deserializer interface converting between the structured type
    # accepted by a RemoteCache and the type its backend stores.
    @abstractmethod
    def encode(self, data: _T) -> _U:
        # [FUNCTION DOC] Abstract; converts structured data to the backend representation.
        pass

    @abstractmethod
    def decode(self, data: _U) -> _T:
        # [FUNCTION DOC] Abstract; converts backend data back to structured data.
        pass


JsonDataTy: TypeAlias = Union[  # noqa: UP007
    int, float, str, bool, dict[str, "JsonDataTy"], list["JsonDataTy"], None
]


class RemoteCacheJsonSerde(RemoteCacheSerde[JsonDataTy, bytes]):
    # [CLASS DOC] Serde that maps JSON-compatible data to ASCII-encoded JSON bytes and back.
    def encode(self, data: JsonDataTy) -> bytes:
        # [FUNCTION DOC] Serializes data to JSON text and returns it as ASCII bytes.
        return bytes(json.dumps(data), "ascii")

    def decode(self, data: bytes) -> JsonDataTy:
        # [FUNCTION DOC] Parses JSON bytes into data; raises a JSON decode error on malformed input.
        return json.loads(data)


class RemoteCachePassthroughSerde(RemoteCacheSerde[_T, _T]):
    # [CLASS DOC] Identity serde: values are stored and returned unchanged.
    def encode(self, data: _T) -> _T:
        # [FUNCTION DOC] Returns data unchanged.
        return data

    def decode(self, data: _T) -> _T:
        # [FUNCTION DOC] Returns data unchanged.
        return data


class LocalCacheBackend(RemoteCacheBackend[bytes]):
    """
    A local filesystem implementation of the cache backend interface.
    """

    # [CLASS DOC] Backend that treats each key as a filesystem path and stores raw bytes in that file.
    @override
    def _get(self, key: str) -> bytes | None:
        # [FUNCTION DOC] HELPER: Reads and returns the bytes of the file at key; None if the file does not exist.
        try:
            with open(key, "rb") as fd:
                return fd.read()
        except FileNotFoundError:
            return None

    @override
    def _put(self, key: str, data: bytes) -> None:
        # [FUNCTION DOC] HELPER: Creates the parent directories of the key path and writes data to it
        # atomically using the codecache atomic writer.
        os.makedirs(os.path.dirname(key), exist_ok=True)
        from torch._inductor import codecache

        codecache.write_atomic(key, data)


# This class is the top of a RemoteCache. A RemoteCache is fundamentally made of
# three parts:
#
# 1. The controller (this class).
# 2. A serializer/deserializer (instance of RemoteCacheSerde).
# 3. A backend (instance of RemoteCacheBackend).
#
# To write (`put`), the RemoteCache takes data, uses the RemoteCacheSerde to
# convert it for the backend and passes it to the backend.
#
# Conversely when reading (`get`), the RemoteCache takes data from the backend,
# uses the RemoteCacheSerde to convert it and returns it.
#
# The RemoteCacheBackend is generic on _U - which is the type of data the
# backend can directly cache (usually `bytes`).
#
# The RemoteCacheSerde is responsible for converting between _T (the type of
# data the RemoteCache accepts in `put` and returns in `get`) and _U.
#
# When instantiating a RemoteCache you should override, not directly create a
# RemoteCache. The reason is that when logging cache use (`TORCH_LOGS=cache`) we
# use the concrete type of the RemoteCache as the reported cache. See
# RemoteFxGraphCache below as an example.
class RemoteCache(Generic[_T]):
    # [CLASS DOC] Controller combining a backend and a serde: put encodes then stores, get fetches then
    # decodes. Attributes: backend, serde. The class attribute backend_override_cls, when set on a
    # subclass, replaces the supplied backend (testing hook). Subclasses override the underscore
    # hooks (_get, _put, _encode, _decode, _create_sample, _log_sample) to customize behavior.
    backend_override_cls: Callable[[], RemoteCacheBackend[Any]] | None = None

    def __init__(
        self, backend: RemoteCacheBackend[_U], serde: RemoteCacheSerde[_T, _U]
    ) -> None:
        # [FUNCTION DOC] Stores the serde and the backend; the backend is instead built from the
        # class's backend_override_cls if one is set.
        # Support for testing to mock out the backend on a class-by-class basis.
        if (override_cls := self.__class__.backend_override_cls) is not None:
            self.backend = override_cls()
        else:
            self.backend = backend
        # pyrefly: ignore [invalid-type-var]
        self.serde = serde

    # See if the cache contains `key`. Returns `None` if the value is not
    # present in the cache.
    def get(self, key: str) -> _T | None:
        # [FUNCTION DOC] Returns the decoded value for key, or None if absent. Runs under a wait
        # counter, creates a log sample, records hit/miss stats under the concrete class name, and on
        # error records an exception stat, stores the failure reason in the sample, and re-raises.
        # The sample is always logged.
        with _WaitCounter("pytorch.remote_cache.get").guard():
            sample = self._create_sample()
            try:
                result = self._get(key, sample)
                cache_stats.get(type(self).__name__, result)
            except Exception as e:
                cache_stats.exception(type(self).__name__)
                if sample:
                    sample.fail_reason = str(e)
                raise
            finally:
                self._log_sample(sample)
            return result

    # Add `value` to the cache with the key `key`. Note that `None` is not a
    # valid value even if _T supports it (because you can't tell the difference
    # between `None` and a missing cache entry).
    def put(self, key: str, value: _T) -> None:
        # [FUNCTION DOC] Encodes and stores value under key. Raises AssertionError if value is None.
        # Runs under a wait counter with a log sample; records put or exception stats, records the
        # failure reason in the sample on error and re-raises; the sample is always logged.
        with _WaitCounter("pytorch.remote_cache.put").guard():
            if value is None:
                raise AssertionError("cannot put None into the cache")
            sample = self._create_sample()
            try:
                self._put(key, value, sample)
                cache_stats.put(type(self).__name__)
            except Exception as e:
                cache_stats.exception(type(self).__name__)
                if sample:
                    sample.fail_reason = str(e)
                raise
            finally:
                self._log_sample(sample)

    # Used to convert data from the cache into structured data.
    def _decode(self, data: _U, sample: Sample | None) -> _T:  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Decodes backend data with the serde (the sample is unused here).
        return self.serde.decode(data)  # type: ignore[arg-type]

    # Used to convert structured data into data for the cache.
    def _encode(self, value: _T, sample: Sample | None) -> object:  # returns _U
        # [FUNCTION DOC] HELPER: Encodes a structured value with the serde (the sample is unused here).
        return self.serde.encode(value)

    # Get structured data from the cache.
    # Separate from `get` so that it can be overridden.
    def _get(self, key: str, sample: Sample | None) -> _T | None:
        # [FUNCTION DOC] HELPER: Fetches raw data from the backend and decodes it; returns None if the backend has no (or empty) data.
        if data := self._backend_get(key):
            return self._decode(data, sample)
        return None

    # Get unstructured data from the cache.
    # Separate from `get` so that it can be overridden.
    # Returns _U - but we aren't actually generic on _U
    def _backend_get(self, key: str) -> object:
        # [FUNCTION DOC] HELPER: Returns the raw backend value for key (None if absent).
        return self.backend.get(key)

    # Put structured data into the cache.
    # Separate from `put` so that it can be overridden.
    def _put(self, key: str, value: _T, sample: Sample | None) -> None:
        # [FUNCTION DOC] HELPER: Encodes the value and writes the result to the backend.
        data = self._encode(value, sample)
        self._backend_put(key, data)

    # Put unstructured data into the cache.
    # Separate from `put` so that it can be overridden.
    # Takes data: _U - but we aren't actually generic on _U
    def _backend_put(self, key: str, data: object) -> None:
        # [FUNCTION DOC] HELPER: Writes already-encoded data to the backend under key.
        self.backend.put(key, data)

    # Create a logging Sample - used with internal loggers to monitor cache
    # effectiveness.
    def _create_sample(self) -> Sample | None:
        # [FUNCTION DOC] HELPER: Hook returning a logging sample; the default returns None (no logging).
        return None

    # Write the logging Sample to the logger.
    def _log_sample(self, sample: Sample | None) -> None:
        # [FUNCTION DOC] HELPER: Hook that writes the sample to a logger; the default does nothing.
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend[bytes]):
    """
    A Redis implementation of a remote/distributed cache.
    """

    # [CLASS DOC] Backend storing raw bytes in a Redis server. The client (attribute _redis) is
    # created at construction and set to None after a connection error, after which gets return
    # None and puts are silently dropped.
    # pyrefly: ignore [missing-attribute]
    _redis: redis.Redis | None = None

    def __init__(self, cache_id: str) -> None:
        # [FUNCTION DOC] Creates the Redis client from the TORCHINDUCTOR_REDIS_URL environment
        # variable if set, else from TORCHINDUCTOR_REDIS_HOST (default localhost) and
        # TORCHINDUCTOR_REDIS_PORT (default 6379). Raises RuntimeError if the redis package is missing.
        # KNOWN ISSUE: the cache_id argument is accepted but unused.
        super().__init__()
        if not redis:
            raise RuntimeError("redis not available but required for remote cache")

        if "TORCHINDUCTOR_REDIS_URL" in os.environ:
            self._redis = redis.Redis.from_url(os.environ["TORCHINDUCTOR_REDIS_URL"])
        else:
            self._redis = redis.Redis(
                host=os.environ.get("TORCHINDUCTOR_REDIS_HOST", "localhost"),
                port=int(os.environ.get("TORCHINDUCTOR_REDIS_PORT", 6379)),
            )

    @override
    def _get(self, key: str) -> bytes | None:
        # [FUNCTION DOC] HELPER: Returns the bytes stored at key, or None if absent or Redis is
        # unavailable. A connection error marks Redis unavailable and returns None. Raises
        # AssertionError if the server returns something other than bytes or None.
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return None

        try:
            # pyrefly: ignore [missing-attribute]
            value = self._redis.get(key)
        # pyrefly: ignore [missing-attribute]
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark it as unavailable now.
            self._redis = None
            return None

        # In theory redis.get() can return an Awaitable as well...
        if not (value is None or isinstance(value, bytes)):
            raise AssertionError(f"expected bytes or None, got {type(value)}")
        return value

    @override
    def _put(self, key: str, data: bytes) -> None:
        # [FUNCTION DOC] HELPER: Sets key to data in Redis; does nothing if Redis is unavailable, and
        # marks Redis unavailable on a connection error.
        if not self._redis:
            # Either redis wasn't found or we already had some trouble...
            return

        try:
            # pyrefly: ignore [missing-attribute]
            self._redis.set(key, data)
        # pyrefly: ignore [missing-attribute]
        except redis.exceptions.ConnectionError:
            # Redis is lazy and doesn't actually attempt to connect until the
            # first use. Mark it as unavailable now.
            self._redis = None


class RedisRemoteCache(RemoteCache[JsonDataTy]):
    # [CLASS DOC] JSON cache over a Redis backend. Keys are namespaced with the cache id and a
    # format version, stored in attribute _key_fmt. Subclasses below only provide distinct cache
    # type names for stats and test overrides.
    def __init__(self, cache_id: str) -> None:
        # [FUNCTION DOC] Builds a Redis backend (or a placeholder none when a test backend override
        # exists) with the JSON serde and computes the key format string from cache_id and version 1.
        # Special test handling: If we're just going to override the backend
        # anyway don't require redis
        if self.__class__.backend_override_cls:
            # This is totally bogus but it works for now...
            backend = typing.cast(RemoteCacheBackend[bytes], None)
        else:
            backend = RedisRemoteCacheBackend(cache_id)
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)
        version = 1  # consistency between various types of keys
        self._key_fmt = f"pt2:{cache_id}::{{key}}:c{version}"

    def _get_key(self, key: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the namespaced Redis key for a plain key.
        return self._key_fmt.format(key=key)

    @override
    def _get(self, key: str, sample: Sample | None) -> JsonDataTy | None:
        # [FUNCTION DOC] HELPER: Looks up the namespaced key via the base implementation.
        key = self._get_key(key)
        return super()._get(key, sample)

    @override
    def _put(self, key: str, value: JsonDataTy, sample: Sample | None) -> None:
        # [FUNCTION DOC] HELPER: Stores the value under the namespaced key via the base implementation.
        key = self._get_key(key)
        super()._put(key, value, sample)


class LocalCache(RemoteCache[JsonDataTy]):
    # [CLASS DOC] JSON cache over the local filesystem backend; each key is used directly as a file path.
    def __init__(self, cache_id: str) -> None:
        # [FUNCTION DOC] Builds the local filesystem backend with the JSON serde. The cache_id is
        # accepted only for factory compatibility and is unused.
        # Local caches use the per-operation key as the filesystem path. Accept
        # cache_id so they can be constructed through create_cache like remote
        # caches.
        backend = LocalCacheBackend()
        serde = RemoteCacheJsonSerde()
        super().__init__(backend, serde)

    @override
    def _get(self, key: str, sample: Sample | None) -> JsonDataTy | None:
        # [FUNCTION DOC] HELPER: Reads via the base implementation; a corrupt entry (JSON or Unicode
        # decode error) is logged as a warning and treated as a miss (None).
        try:
            return super()._get(key, sample)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning(
                "Ignoring corrupt local cache entry %s: %s: %s",
                key,
                type(exc).__name__,
                exc,
            )
            return None


class LocalAutotuneCache(LocalCache):
    # [CLASS DOC] Local-file cache type for autotune results; adds no behavior.
    # Keep a distinct cache type for cache stats and test backend overrides.
    pass


class RemoteAutotuneCache(RedisRemoteCache):
    # [CLASS DOC] Redis cache type for autotune results; adds no behavior (distinct type for stats/overrides).
    pass


class RemoteBundledAutotuneCache(RedisRemoteCache):
    # [CLASS DOC] Redis cache type for bundled autotune results; adds no behavior (distinct type for stats/overrides).
    pass


class RemoteFxGraphCache(RedisRemoteCache):
    # [CLASS DOC] Redis cache type for FX graph cache entries; adds no behavior (distinct type for stats/overrides).
    pass


class RemoteAOTAutogradCache(RedisRemoteCache):
    # [CLASS DOC] Redis cache type for AOT autograd cache entries; adds no behavior (distinct type for stats/overrides).
    pass


class RemoteDynamoPGOCache(RedisRemoteCache):
    # [CLASS DOC] Redis cache type for Dynamo profile-guided-optimization data; adds no behavior (distinct type for stats/overrides).
    pass


def create_cache(
    key: str,
    is_fbcode: bool = False,
    fb_cache_cls: str | None = None,
    oss_cache_cls: str | None = None,
    *,
    local_cache_cls: str | None = None,
) -> RemoteCache[JsonDataTy] | None:
    # [FUNCTION DOC] Factory that instantiates a cache with the given key/id. Picks the class by name
    # from this module when local_cache_cls is given; else, in fbcode, from the internal fb remote_cache
    # module using fb_cache_cls (required); else from this module using oss_cache_cls (required).
    # Any failure, including missing class names, is logged as a warning and yields None.
    try:
        this_module = sys.modules[__name__]
        if local_cache_cls is not None:
            cache_cls = getattr(this_module, local_cache_cls)
            return cache_cls(key)
        elif is_fbcode:
            if fb_cache_cls is None:
                raise AssertionError("fb_cache_cls must not be None in fbcode")
            import torch._inductor.fb.remote_cache

            cache_cls = getattr(torch._inductor.fb.remote_cache, fb_cache_cls)
            return cache_cls(key)
        else:
            if oss_cache_cls is None:
                raise AssertionError("oss_cache_cls must not be None")
            cache_cls = getattr(this_module, oss_cache_cls)
            return cache_cls(key)

    except Exception:
        log.warning("Unable to create cache", exc_info=True)
        return None


# Some simple stat capture
@dataclasses.dataclass
class _CacheStat:
    # [CLASS DOC] HELPER: Dataclass of counters (miss, hit, put, exception) for one cache name.
    miss: int = 0
    hit: int = 0
    put: int = 0
    exception: int = 0

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Formats the four counters as a brace-delimited summary string.
        return f"{{hit: {self.hit}, miss: {self.miss}, put: {self.put}, exception: {self.exception}}}"


class _CacheStats:
    # [CLASS DOC] HELPER: Collector of per-cache-name statistics held in a mapping from name to
    # _CacheStat (created on demand). A single global instance, cache_stats, is used by the caches.
    _stats: dict[str, _CacheStat]

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes an empty on-demand mapping of statistics.
        self._stats = collections.defaultdict(_CacheStat)

    def miss(self, name: str, count: int = 1) -> None:
        # [FUNCTION DOC] HELPER: Adds count misses for the named cache.
        self._stats[name].miss += count

    def hit(self, name: str, count: int = 1) -> None:
        # [FUNCTION DOC] HELPER: Adds count hits for the named cache.
        self._stats[name].hit += count

    def get(self, name: str, value: object | None) -> None:
        # [FUNCTION DOC] HELPER: Records a miss if the fetched value is None, otherwise a hit.
        if value is None:
            self.miss(name)
        else:
            self.hit(name)

    def put(self, name: str, count: int = 1) -> None:
        # [FUNCTION DOC] HELPER: Adds count puts for the named cache.
        self._stats[name].put += count

    def exception(self, name: str, count: int = 1) -> None:
        # [FUNCTION DOC] HELPER: Adds count exceptions for the named cache.
        self._stats[name].exception += count


cache_stats = _CacheStats()


@atexit.register
def dump_cache_stats() -> None:
    # [FUNCTION DOC] Registered to run at interpreter exit. If INFO logging is enabled for this
    # module, logs a "Cache Metrics" report listing each cache name and its counters sorted by name
    # (or "None" when no stats were recorded). Otherwise does nothing.
    if not log.isEnabledFor(logging.INFO):
        return

    import io

    out = io.StringIO()

    if not cache_stats._stats:
        print(" None", file=out)
    else:
        print(file=out)
        for k, v in sorted(cache_stats._stats.items()):
            print(f"  {k}: {v}", file=out)

    log.info("Cache Metrics:%s", out.getvalue())
