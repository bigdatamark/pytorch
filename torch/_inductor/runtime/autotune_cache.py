# [MODULE DOC] Autotune config caching for Inductor Triton kernels: AutotuneCache coordinates local
# file and remote (Redis/internal) lookups of the best kernel config, AutotuneCacheBundler batches
# local entries into one remote bundle, plus helpers to apply cached configs to candidate configs.
# Import-time side effects: registers AutotuneCacheArtifact with the cache artifact factory.
"""
PyTorch Inductor Autotuning Cache System

This module implements a caching system for autotuning configurations in PyTorch's Inductor compiler.
It provides mechanisms to store and retrieve optimal kernel configurations both locally and remotely,
which significantly speeds up compilation by reusing previously discovered optimal parameters.

The caching system includes:
- Local filesystem caching for individual machine reuse
- Remote caching for sharing optimizations across machines
- Bundled caching to efficiently store multiple related configurations
- Cache invalidation based on PyTorch versions and backend changes
- Serialization/deserialization support for worker processes

Key components:
- AutotuneCache: Main class for managing cache access and storage
- AutotuneCacheBundler: Bundles multiple cache entries for efficient storage
- LocalCacheBackend: Low-level file operations for cache storage
- AutotuneCacheArtifact: Integration with PyTorch's artifact system

This caching system is critical for performance as it eliminates the need to re-run
expensive autotuning operations when the same kernels are compiled multiple times.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import os.path
import re
import threading
import weakref
from typing import Any
from typing_extensions import override

import torch
from torch._guards import CompileContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)
from torch.utils._triton import has_triton

from ..cache_key import AUTOTUNE_CACHE_KEY_STRATEGY
from ..remote_cache import (
    create_cache,
    JsonDataTy,
    LocalAutotuneCache,
    LocalCacheBackend,
    RemoteCache,
    RemoteCacheJsonSerde,
)
from .hints import InductorMeta
from .triton_compat import Config, HAS_WARP_SPEC


log = logging.getLogger(__name__)


_InductorMetaTy = InductorMeta


def inductor_meta_from_config() -> _InductorMetaTy:
    # [FUNCTION DOC] Builds the inductor metadata mapping from current inductor config: local/remote/
    # bundled autotune cache flags, coordinate descent tuning, fbcode flag, ROCm flag (true on HIP,
    # else None) and the Triton backend hash (None if Triton is missing or driver lookup raises a RuntimeError).
    from torch._inductor import config

    backend_hash = None
    if has_triton():
        try:
            backend_hash = torch.utils._triton.triton_hash_with_backend()
        except RuntimeError:
            # This can get the error:
            #   RuntimeError: 0 active drivers ([]). There should only be one.
            pass

    is_hip = None
    if torch.version.hip is not None:
        is_hip = True

    return {
        "autotune_local_cache": config.autotune_local_cache,
        "autotune_remote_cache": config.autotune_remote_cache,
        "backend_hash": backend_hash,
        "bundled_autotune_remote_cache": config.bundled_autotune_remote_cache,
        "coordinate_descent_tuning": config.coordinate_descent_tuning,
        "is_fbcode": config.is_fbcode(),
        "is_hip": is_hip,
    }


@CacheArtifactFactory.register
class AutotuneCacheArtifact(CacheArtifact):
    # [CLASS DOC] Cache artifact type "autotune": represents a best-config JSON entry that can be
    # serialized into an artifact bundle and written back into the local Inductor cache directory.
    @override
    def populate_cache(self) -> None:
        # [FUNCTION DOC] Writes the artifact content as a file named by its key under the Inductor cache directory.
        autotune_cache = LocalCacheBackend()
        key = os.path.join(cache_dir(), self.key)
        autotune_cache._put(key, self.content)

    @override
    @staticmethod
    def type() -> str:
        # [FUNCTION DOC] Returns the artifact type name, "autotune".
        return "autotune"

    @override
    @staticmethod
    def encode(content: JsonDataTy) -> bytes:
        # [FUNCTION DOC] Serializes JSON-compatible content to JSON bytes. Raises AssertionError if
        # the content is already bytes or the serde does not produce bytes.
        if isinstance(content, bytes):
            raise AssertionError("content must not be bytes before encoding")
        serde = RemoteCacheJsonSerde()
        content_bytes = serde.encode(content)
        if not isinstance(content_bytes, bytes):
            raise AssertionError(
                f"Expected bytes after encoding, got {type(content_bytes)}"
            )
        return content_bytes


@dataclasses.dataclass
class AutotuneCache:
    """Coordinates local and remote autotune cache lookups for one kernel."""

    # [CLASS DOC] Per-kernel autotune cache handle. Fields: configs_hash (identifies the candidate
    # config set), local_cache and remote_cache (each a cache plus its key, or None), and an
    # artifact_recorder. Supports pickling for async compile workers (the remote cache is rebuilt
    # on unpickling). Key methods: create, read_best, save.
    configs_hash: str
    local_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    remote_cache: tuple[RemoteCache[JsonDataTy], str] | None = None
    artifact_recorder: CacheArtifactRecorder | None = None

    # Create a AutotuneCache. Returns None if none of the caches can be used.
    @staticmethod
    def create(
        inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
    ) -> AutotuneCache | None:
        # [FUNCTION DOC] Builds an AutotuneCache for the kernel source file, computing the cache key
        # and a local key beside the file, creating an artifact recorder, and setting up local and
        # remote caches according to inductor_meta. Returns None if neither cache is usable.
        cache = AutotuneCache(configs_hash)
        key = AutotuneCache._prepare_key(filename)
        local_cache_key = AutotuneCache._make_local_cache_key(
            os.path.dirname(filename), key
        )
        cache.artifact_recorder = CacheArtifactRecorder(
            AutotuneCacheArtifact.type(),
            AutotuneCache._artifact_key_from_local_cache_key(local_cache_key),
        )

        cache._setup_local_cache(inductor_meta, local_cache_key)
        cache._setup_remote_autotune_cache(inductor_meta, key)
        if cache.local_cache or cache.remote_cache:
            return cache
        else:
            return None

    @staticmethod
    def _prepare_key(filename: str) -> str:
        # [FUNCTION DOC] HELPER: Derives the base cache key from the file's base name (already a
        # source hash) plus the configured cache key tag, passed through the autotune key strategy.
        from torch.compiler import config as cconfig

        # base of filename is already sha256 hash the source contents
        key = f"{os.path.basename(filename)}:{cconfig.cache_key_tag}"
        return AUTOTUNE_CACHE_KEY_STRATEGY.key(key)

    @staticmethod
    def _make_local_cache_key(dirname: str, cache_key: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the local cache file path: the directory joined with a hash of
        # the key combined with the torch key (so torch changes invalidate entries) and a .best_config suffix.
        """
        [Note: torch_key in autotune cache key]
        Include torch_key() in the cache key so that different versions
        of torch result in cache invalidation. This is important in case
        of changes to the best_config format or other code changes that
        are not backward compatible w.r.t. the cache.
        """
        from ..codecache import torch_key

        updated_cache_key = AUTOTUNE_CACHE_KEY_STRATEGY.key(cache_key, torch_key())
        return os.path.join(dirname, f"{updated_cache_key}.best_config")

    @staticmethod
    def _artifact_key_from_local_cache_key(local_cache_key: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the last two path components (directory and file) of the local cache key as the artifact key.
        return os.path.join(*local_cache_key.split(os.sep)[-2:])

    def _record_artifact(self, data: JsonDataTy) -> None:
        # [FUNCTION DOC] HELPER: Records the data with the artifact recorder if one exists (older pickled instances may lack it).
        # Older pickled AutotuneCache instances may not have this field.
        if recorder := getattr(self, "artifact_recorder", None):
            recorder.record(data)

    # Read the best config options from the most local cache and return it.
    def _read(self) -> dict[str, JsonDataTy] | None:
        # [FUNCTION DOC] HELPER: Returns the cached best-config mapping, trying the local cache first
        # (after syncing the bundler; a hit is re-added to the bundler and recorded as an artifact;
        # an empty mapping falls through) then the remote cache. Returns None on miss. Raises
        # AssertionError if the local entry is not a mapping.
        if local_cache := self.local_cache:
            cache, key = local_cache
            AutotuneCacheBundler.sync()
            best_config = cache.get(key)
            if best_config is not None:
                if not isinstance(best_config, dict):
                    raise AssertionError(
                        f"Expected dict for best_config, got {type(best_config)}"
                    )
                # Imagine we have a new model that reuses some existing kernels that
                # have already been compiled. If we didn't put() here on cache hit,
                # then the new model would only bundle newly compiled kernels, not
                # existing kernels that were already compiled and cached.
                AutotuneCacheBundler.put(key, best_config)
                self._record_artifact(best_config)
                if best_config:
                    return best_config

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            if best_config := cache.get(key):
                if isinstance(best_config, dict):
                    self._record_artifact(best_config)
                    return best_config

        return None

    # Read the best config options from the most local cache and figure out
    # which `configs` represents that option.
    def read_best(
        self, inductor_meta: _InductorMetaTy, configs: list[Config]
    ) -> Config | None:
        # [FUNCTION DOC] Reads the cached best config and maps it to one of the candidate configs
        # (or a new config built from it) via the cached-autotuning loader. Returns None on a miss.
        if best := self._read():
            return _load_cached_autotuning(
                best, self.configs_hash, configs, inductor_meta
            )
        return None

    # Set up local filesystem caching information
    def _setup_local_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Unless autotune_local_cache is disabled in the metadata (default
        # enabled), creates the local autotune cache and stores it with the given key.
        if not inductor_meta.get("autotune_local_cache", True):
            return

        local_cache = create_cache(
            "local-autotune",
            local_cache_cls=LocalAutotuneCache.__name__,
        )
        if local_cache is None:
            return
        self.local_cache = (local_cache, cache_key)

    # Set up remote caching information
    def _setup_remote_autotune_cache(
        self, inductor_meta: _InductorMetaTy, cache_key: str
    ) -> None:
        # [FUNCTION DOC] HELPER: If the remote autotune cache is enabled and a string backend_hash is
        # in the metadata (missing hash: logged and skipped; non-string: AssertionError), builds a
        # remote cache key from the torch key, backend hash, configs hash and a salt, creates the
        # cache, and stores it along with the full key and fbcode flag (kept for pickling).
        if not _should_use_remote_autotune_cache(inductor_meta):
            return

        if (backend_hash := inductor_meta.get("backend_hash", None)) is None:
            log.debug(
                "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
            )
            return
        if not isinstance(backend_hash, str):
            raise AssertionError(
                f"Expected str for backend_hash, got {type(backend_hash)}"
            )

        from ..codecache import torch_key

        is_fbcode = bool(inductor_meta.get("is_fbcode", False))

        salt = "autotune-best-config-v2"
        # re: torch_key - see [Note: torch_key in autotune cache key]
        key = AUTOTUNE_CACHE_KEY_STRATEGY.key(
            torch_key().hex(), backend_hash, self.configs_hash, salt
        )

        remote_cache = create_cache(
            key,
            is_fbcode,
            "FbRemoteAutotuneCache",
            "RemoteAutotuneCache",
        )
        if not remote_cache:
            return

        # Save the args passed to create_cache
        # in case AutotuneCache needs to be pickled
        self.remote_cache_full_key = key
        self.is_fbcode = is_fbcode
        self.remote_cache = (remote_cache, cache_key)

    # The AutotuneCache may be serialized/deserialized if we're using
    # AsyncCompile worker processes to run triton compilation.
    # This is because AutotuneCache instances are created on the worker
    # process, but we need to run AutotuneCache.save on the parent process
    # when actually doing autotuning.
    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the pickle state: the instance dict with the remote cache handle
        # replaced by only its key string (handles may not be serializable).
        # The remote cache handles themselves may not be serializable
        # So clear it and reconstruct it on setstate
        remote_cache = getattr(self, "remote_cache", None)
        return {
            **self.__dict__,
            # Save the cache_key portion
            "remote_cache": remote_cache and remote_cache[1],
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restores state and, if a remote cache key was pickled, recreates the remote
        # cache from the saved full key and fbcode flag; logs a warning and clears it on failure.
        # Raises AssertionError if the pickled fields are inconsistent or missing.
        # Reconstruct the remote cache on the parent class
        self.__dict__.update(state)
        if self.remote_cache is not None:
            if not isinstance(self.remote_cache, str):
                raise AssertionError(
                    f"Expected str for remote_cache after deserialization, got {type(self.remote_cache)}"
                )
            if not hasattr(self, "remote_cache_full_key"):
                raise AssertionError(
                    "Missing remote_cache_full_key attribute after deserialization"
                )
            if not hasattr(self, "is_fbcode"):
                raise AssertionError(
                    "Missing is_fbcode attribute after deserialization"
                )
            cache_key = self.remote_cache
            remote_cache = create_cache(
                self.remote_cache_full_key,
                self.is_fbcode,
                "FbRemoteAutotuneCache",
                "RemoteAutotuneCache",
            )
            if remote_cache is not None:
                self.remote_cache = (remote_cache, cache_key)
            else:
                log.warning("Warning, failed to recreate remote cache after pickling")
                self.remote_cache = None

    # Save the config in the caches
    def save(
        self,
        config: Config,
        time_taken_ns: int,
        found_by_coordesc: bool = False,
        triton_cache_hash: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Persists a chosen Triton config: builds a JSON record (config kwargs, warps,
        # stages, configs hash, coordinate-descent flag, time in ms, triton cache hash, optional extra
        # options and warp-specialization fields), records it as an artifact, writes it to the local
        # cache and the bundler, and writes it to the remote cache. Cache write errors propagate.
        data: dict[str, JsonDataTy] = {
            # pyrefly: ignore [missing-attribute]
            **config.kwargs,
            # pyrefly: ignore [missing-attribute]
            "num_warps": config.num_warps,
            # pyrefly: ignore [missing-attribute]
            "num_stages": config.num_stages,
            "configs_hash": self.configs_hash,
            "found_by_coordesc": found_by_coordesc,
            "time_taken_ms": time_taken_ns // 1000000,  # Convert from NS to MS
            "triton_cache_hash": triton_cache_hash,
        }
        # Save extra_options if present on the config. This allows third-party
        # backends to store custom tuned options alongside the standard config.
        if extra_options := getattr(config, "extra_options", None):
            data["extra_options"] = extra_options
        if HAS_WARP_SPEC:
            data.update(
                {
                    "num_consumer_groups": getattr(config, "num_consumer_groups", 0),
                    "num_buffers_warp_spec": getattr(
                        config, "num_buffers_warp_spec", 0
                    ),
                }
            )

        self._record_artifact(data)

        if local_cache := self.local_cache:
            cache, key = local_cache
            # pyrefly: ignore [bad-argument-type]
            cache.put(key, data)
            # pyrefly: ignore [bad-argument-type]
            AutotuneCacheBundler.put(key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, key)

        if remote_cache := self.remote_cache:
            cache, key = remote_cache
            # pyrefly: ignore [bad-argument-type]
            cache.put(key, data)


class _AutotuneCacheBundlerImpl:
    """
    Caches a set of local autotune entries together in a single cache.
    """

    # [CLASS DOC] HELPER: Collects local autotune cache entries (mapping from file basename to JSON
    # data) for one compilation and writes them as one remote entry under a single key; on a remote
    # hit it expands the bundle back into the local autotune cache. Attributes: _key, _cache, _entries.

    _key: str
    _cache: RemoteCache[JsonDataTy]

    # All known entries from local autotune cache writes.
    _entries: dict[str, JsonDataTy]

    def end_compile(self) -> None:
        # [FUNCTION DOC] HELPER: Writes all collected entries to the remote cache under the bundle key, if there are any.
        # TODO: Do we need to compute time_taken_ms and encode that somehow?
        if self._entries:
            self._cache.put(self._key, self._entries)

    def put(self, basename: str, data: JsonDataTy) -> None:
        # [FUNCTION DOC] HELPER: Records (or overwrites) the entry for the given file basename.
        # Do we need to worry about duplicates? We only have a single local fs
        # entry - so probably not.
        self._entries[basename] = data

    def __init__(self, key: str, cache: RemoteCache[JsonDataTy]) -> None:
        # [FUNCTION DOC] HELPER: Stores the bundle key and remote cache and starts with no entries.
        self._key = key
        self._cache = cache
        self._entries = {}

    def sync(self) -> None:
        # [FUNCTION DOC] HELPER: Currently a no-op placeholder for waiting on an asynchronous cache load.
        # We don't currently use this - but we could async load starting at
        # `begin_compile` and wait for the load to be finished here.
        pass

    @classmethod
    def _should_use_bundled_autotune_remote_cache(
        cls, inductor_meta: _InductorMetaTy
    ) -> bool:
        # [FUNCTION DOC] HELPER: Decides whether bundled remote autotune caching applies. False if the
        # local cache is disabled; otherwise the explicit bundled_autotune_remote_cache setting if
        # present; otherwise only in fbcode, outside unit tests and non-HIP, when the internal cache
        # version meets the JustKnobs version (False if the internal module is absent).
        # The bundled autotune cache is only available if you've also got local
        # caching enabled (because we feed the bundled data to the local cache).
        if not inductor_meta.get("autotune_local_cache", True):
            return False

        # Check if we're enabled via config
        if (
            bundled_autotune_remote_cache := inductor_meta.get(
                "bundled_autotune_remote_cache"
            )
        ) is not None:
            return bool(bundled_autotune_remote_cache)

        if not cls._get_is_fbcode(inductor_meta):
            return False
        if torch._utils_internal.is_fb_unit_test():
            return False
        if inductor_meta.get("is_hip"):
            return False

        try:
            from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
        except ModuleNotFoundError:
            return False

        jk = torch._utils_internal.justknobs_getval_int(
            "pytorch/remote_cache:bundled_autotune_remote_cache_version"
        )
        return REMOTE_CACHE_VERSION >= jk

    def _load_cache(self) -> bool:
        # [FUNCTION DOC] HELPER: Fetches the bundle from the remote cache; returns False if missing or
        # not a mapping. Otherwise writes each entry into the local autotune cache at its
        # reconstructed path, sums any saved-time values and extends the distributed timeout by that
        # total, and returns True.
        from torch._inductor import codecache

        # The single key is defined on construction of the cache.
        entries = self._cache.get(self._key)
        if entries is None or not isinstance(entries, dict):
            # We couldn't load the cache - so mark _entries as non-None so we
            # store local cache values.
            return False

        # Go through the entries we got from the cache and save them locally.
        time_saved_ns = 0
        local_cache = create_cache(
            "local-autotune",
            local_cache_cls=LocalAutotuneCache.__name__,
        )
        for basename, data in entries.items():
            # Reconstruct the final filename (see put())
            root, ext = _splitext_nodot(basename)
            _, _, filename = codecache.get_path(root, ext)
            if isinstance(data, dict) and (tsns := data.get("time_saved_ns")):
                time_saved_ns += int(tsns)  # type: ignore[arg-type]
            if local_cache is not None:
                local_cache.put(filename, data)

        codecache.add_ephemeral_timeout_increase_for_distributed(time_saved_ns)

        return True

    @staticmethod
    def _get_is_fbcode(inductor_meta: _InductorMetaTy) -> bool:
        # [FUNCTION DOC] HELPER: Returns the is_fbcode flag from the metadata as a bool (default false).
        return bool(inductor_meta.get("is_fbcode", False))

    @staticmethod
    def _get_backend_hash(inductor_meta: _InductorMetaTy) -> str:
        # [FUNCTION DOC] HELPER: Returns backend_hash from the metadata; raises KeyError if absent and AssertionError if not a string.
        backend_hash = inductor_meta["backend_hash"]
        if not isinstance(backend_hash, str):
            raise AssertionError(
                f"Expected str for backend_hash, got {type(backend_hash)}"
            )
        return backend_hash


class AutotuneCacheBundler:
    """Manages one bundled autotune cache write scope for a compile context."""

    # [CLASS DOC] Public facade over bundled autotune caching. Keeps a class-level weak mapping (with
    # a lock) from CompileContext to an AutotuneCacheBundler instance, each holding an optional
    # active bundler implementation. Class methods begin_compile, end_compile, sync and put operate
    # on the bundler of the current compile context and are no-ops when none is active.

    _context_bundlers: weakref.WeakKeyDictionary[
        CompileContext, AutotuneCacheBundler
    ] = weakref.WeakKeyDictionary()
    _context_bundlers_lock = threading.Lock()

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates a holder with no active bundler implementation.
        self._bundler: _AutotuneCacheBundlerImpl | None = None

    @classmethod
    def _get_context_bundler(
        cls,
        ctx: CompileContext | None = None,
        *,
        create: bool,
    ) -> AutotuneCacheBundler | None:
        # [FUNCTION DOC] HELPER: Under the lock, returns the bundler holder registered for the compile
        # context, creating one when create is true and none exists. Returns None if ctx is None or
        # nothing exists and create is false. Raises AssertionError on an unexpected stored type.
        if ctx is None:
            return None

        with cls._context_bundlers_lock:
            bundler = cls._context_bundlers.get(ctx)
            if bundler is None and create:
                bundler = cls()
                cls._context_bundlers[ctx] = bundler
            if bundler is not None and not isinstance(bundler, cls):
                raise AssertionError(
                    f"Expected AutotuneCacheBundler or None, got {type(bundler)}"
                )
            return bundler

    @classmethod
    def has_active_compile(cls, compile_context: CompileContext | None) -> bool:
        # [FUNCTION DOC] Returns true if the given compile context has an active bundler between begin_compile and end_compile.
        context_bundler = cls._get_context_bundler(compile_context, create=False)
        return context_bundler is not None and context_bundler._bundler is not None

    # Call this before we start any autotune computation for an inductor python
    # file. On a cache hit it copies the individual results into the local
    # autotune caches.
    @classmethod
    def begin_compile(
        cls,
        inductor_meta: _InductorMetaTy,
        *,
        code: str | None = None,
        code_hash: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Starts bundling for the current compile context. Exactly one of code (hashed
        # after stripping comments) or code_hash is required, else AssertionError; also raises
        # AssertionError if a bundler is already active. Returns silently if bundled caching is not
        # enabled, no compile context exists, or the cache cannot be created. Otherwise tries to load the
        # bundle for the key (code hash, backend hash, salt) into the local cache; only on a miss does
        # it keep the bundler active to collect entries.
        if code is not None:
            if code_hash is not None:
                raise AssertionError("Cannot specify both code and code_hash")
            code_hash = _comment_stripped_hash(code)
        if code_hash is None:
            raise AssertionError("Either code or code_hash must be provided")

        if not _AutotuneCacheBundlerImpl._should_use_bundled_autotune_remote_cache(
            inductor_meta
        ):
            return

        context_bundler = cls._get_context_bundler(
            CompileContext.try_get(), create=True
        )
        if context_bundler is None:
            log.debug(
                "Skipping bundled autotune cache because compile_context is not set"
            )
            return
        if context_bundler._bundler is not None:
            raise AssertionError(
                "begin_compile called while a bundler is already active"
            )

        cache = create_cache(
            "bundled-autotune-v1",
            _AutotuneCacheBundlerImpl._get_is_fbcode(inductor_meta),
            "FbRemoteBundledAutotuneCache",
            "RemoteBundledAutotuneCache",
        )
        if not cache:
            return

        # We're starting a compilation phase. We have a cache key for the code
        # we're compiling. We'll get the individual autotune bundles later (via
        # self.put()). For now create the AutotuneCacheBundler and try to load
        # from the cache.

        salt = "bundled-autotune-best-configs-v1"
        backend_hash = _AutotuneCacheBundlerImpl._get_backend_hash(inductor_meta)
        # TODO: The autotune cache includes configs_hash in the key. The problem
        # is that the configs_hash includes info from the individual pointwise()
        # calls (size_hints, for example) which we can't know yet. I *think*
        # that info is basically present in the `code_hash` (since it's a
        # parameter to the pointwise decorator) - but is there other info we
        # need to include from inductor_meta?
        key = AUTOTUNE_CACHE_KEY_STRATEGY.key(code_hash, backend_hash, salt)

        bundler = _AutotuneCacheBundlerImpl(key, cache)
        if not bundler._load_cache():
            # We couldn't load from the cache - so save the data so we can store
            # the saved autotunes.
            context_bundler._bundler = bundler

        # If we get a cache hit don't bother saving any of the individual
        # autotune results.

    # Call this after all individual autotune results are finished for a
    # inductor python file. If we gathered any individual results then we bundle
    # those and put it into the cache.
    @classmethod
    def end_compile(cls) -> None:
        # [FUNCTION DOC] Finishes bundling for the current compile context: detaches the active
        # bundler (if any) and writes its collected entries to the remote cache.
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            context_bundler._bundler = None
            bundler.end_compile()

    @classmethod
    def sync(cls) -> None:
        # [FUNCTION DOC] Calls sync on the current compile context's active bundler, if any.
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            bundler.sync()

    @classmethod
    def put(cls, filename: str, data: JsonDataTy) -> None:
        # [FUNCTION DOC] Adds a local autotune cache entry to the active bundler of the current
        # compile context, keyed by the file's base name; does nothing if no bundler is active.
        if not (
            context_bundler := cls._get_context_bundler(
                CompileContext.try_get(), create=False
            )
        ):
            return
        if bundler := context_bundler._bundler:
            # The filename comes in as something like
            # "/tmp/tmp{random}/{aa}/{basename}.py" (where aa is
            # basename[1:3]). Strip it down and make sure that it looks like a path
            # we could reconstruct (because it's possible for the caller to
            # customize the path).
            basename = os.path.basename(filename)

            # TODO: check cache_dir() vs filename, then strip dirname
            bundler.put(basename, data)


# Remove the comments from the code (which include things like run ids and file
# paths) and then hash the result.
def _comment_stripped_hash(code: str) -> str:
    # [FUNCTION DOC] HELPER: Removes hash-style comments from each line of source and returns the codecache code hash of the result.
    code = re.sub(r"#.*$", "", code, count=0, flags=re.MULTILINE)
    return torch._inductor.codecache.code_hash(code)


def _should_use_remote_autotune_cache(inductor_meta: _InductorMetaTy) -> bool:
    # [FUNCTION DOC] HELPER: Decides if the remote autotune cache applies: the explicit
    # autotune_remote_cache setting wins if present; otherwise only in fbcode, outside unit tests and
    # non-HIP, when the internal cache version meets the JustKnobs version (False if the internal module is missing).
    if (config := inductor_meta.get("autotune_remote_cache")) is not None:
        return bool(config)
    if not inductor_meta.get("is_fbcode"):
        return False
    if torch._utils_internal.is_fb_unit_test():
        return False
    if inductor_meta.get("is_hip"):
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:autotune_memcache_version"
    )


def _reconstruct_triton_config(
    best_config: dict[str, Any],
    extra_options: JsonDataTy | None,
) -> Config:
    # [FUNCTION DOC] HELPER: Builds a Triton Config from a cached mapping: removes num_warps and
    # num_stages (plus warp-specialization fields when supported) from the mapping (mutating it),
    # uses the remaining entries as kernel kwargs, and attaches the extra options. Raises KeyError if
    # num_warps or num_stages is missing.
    num_warps = best_config.pop("num_warps")
    num_stages = best_config.pop("num_stages")
    config_args: dict[str, Any] = {
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    if HAS_WARP_SPEC:
        config_args.update(
            {
                "num_consumer_groups": best_config.pop("num_consumer_groups", 0),
                "num_buffers_warp_spec": best_config.pop("num_buffers_warp_spec", 0),
            }
        )
    # pyrefly: ignore [bad-argument-count, unexpected-keyword]
    triton_config = Config(best_config, **config_args)
    # pyrefly: ignore [missing-attribute]
    triton_config.extra_options = extra_options
    return triton_config


def _load_cached_autotuning(
    best_config: dict[str, JsonDataTy],
    configs_hash: str,
    configs: list[Config],
    inductor_meta: _InductorMetaTy,
) -> Config | None:
    # [FUNCTION DOC] Converts a cached best-config mapping into a Config. Returns None if the mapping
    # is None or its stored configs hash differs from configs_hash. Strips bookkeeping fields (time,
    # triton cache hash, extra options, coordinate-descent flag; mutating the mapping). Unless found by
    # coordinate descent, returns the single candidate config whose kwargs, warps and stages match
    # (with extra options attached); otherwise reconstructs a new Config, flagging coordinate-descent results.
    if best_config is None:
        return None
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    # Remove time taken for comparison
    best_config.pop("time_taken_ms", None)

    best_config.pop("triton_cache_hash", None)

    # Extract extra_options if present. This allows third-party backends
    # to restore custom tuned options from the cache.
    extra_options = best_config.pop("extra_options", None)

    found_by_coordesc = inductor_meta.get(
        "coordinate_descent_tuning"
    ) and best_config.pop("found_by_coordesc", False)

    if not found_by_coordesc:
        matching_configs = [
            cfg
            for cfg in configs
            # pyrefly: ignore [missing-attribute]
            if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
            # pyrefly: ignore [missing-attribute]
            and cfg.num_warps == best_config.get("num_warps")
            # pyrefly: ignore [missing-attribute]
            and cfg.num_stages == best_config.get("num_stages")
        ]
        if len(matching_configs) == 1:
            matched_config = matching_configs[0]
            # pyrefly: ignore [missing-attribute]
            matched_config.extra_options = extra_options
            return matched_config

    # Reconstruct Config from cached data. This handles both coordesc
    # configs and dynamically added configs (e.g. _dynamic_scale_rblock)
    # that aren't in the original config list.
    best_config.pop("found_by_coordesc", None)
    triton_config = _reconstruct_triton_config(best_config, extra_options)
    if found_by_coordesc:
        # pyrefly: ignore [missing-attribute]
        triton_config.found_by_coordesc = True
    return triton_config


def _splitext_nodot(basename: str) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Splits a file name into root and extension, with the extension returned without its leading dot.
    root, ext = os.path.splitext(basename)
    if ext:
        ext = ext[1:]
    return root, ext
