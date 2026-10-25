# [MODULE DOC] Precompile support: defines the backend cache artifact base class, the
# eager artifact, a bypass exception, and PrecompileContext, a process-global registry that
# collects Dynamo cache entries and backend artifacts and stitches them into
# PrecompileCacheEntry objects written to DynamoCache. Import-time side effect: creates the
# module logger only.
import copy
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch._dynamo.package import (
    _BackendId,
    _DynamoCacheEntry,
    DynamoCache,
    PrecompileCacheEntry,
)


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class BackendCacheArtifact(Generic[T]):
    """
    Represents a single serializable backend artifact from a dynamo backend.
    Each BackendCacheArtifact has a key associated with it along with some
    serializable content.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """
    # [CLASS DOC] Abstract dataclass for one serializable backend artifact from a Dynamo
    # backend. State: key (identifier used to match artifacts to cache entries) and content
    # (serializable payload, usually raw bytes). Subclasses implement after_deserialization;
    # edit_contents lets callers transform the content in place.

    key: str
    content: Any

    @abstractmethod
    def after_deserialization(self) -> T:
        # [FUNCTION DOC] Abstract: converts the stored raw content back into the original
        # backend object after reading from disk. Returns that object (type T).
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Replaces the artifact content with the result of calling edit_fn on
        # the current content (mutates self; returns nothing).
        """
        Edit the contents of the artifact.
        """
        self.content = edit_fn(self.content)


class EagerCacheArtifact(BackendCacheArtifact[Any]):
    # [CLASS DOC] Backend artifact for the eager backend: the content is already in its
    # final form, so no deserialization work is needed.
    def after_deserialization(self) -> Any:
        # [FUNCTION DOC] Returns the stored content unchanged.
        return self.content


class BypassDynamoCacheEntry(Exception):
    # [CLASS DOC] Exception type signalling that a Dynamo cache entry should be bypassed
    # (not used or built). Carries no extra state.
    pass


class PrecompileContext:
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    PrecompileContext has two main portions: dynamo_cache_entries and backend_cache_artifacts.
    When saving, PrecompileContext.serialize() will serialize all dynamo cache entries along with any PrecompileCacheArtifacts that
    are needed to save those dynamo cache entries.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    """
    # [CLASS DOC] Global (class-level state only, never instantiated) registry for
    # precompilation. State: _backend_artifacts_by_key (backend artifacts keyed by
    # _BackendId, guarded by the compile lock) and _dynamo_cache_entries (Dynamo cache
    # entries keyed by string). Key methods: record_artifact, record_dynamo_cache_entry,
    # edit_artifact, create_cache_entries, save_to_dynamo_cache, clear.

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # Each object here must be serializable
    _backend_artifacts_by_key: dict[_BackendId, BackendCacheArtifact[Any]] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

    @classmethod
    def clear(cls) -> None:
        # [FUNCTION DOC] Empties both the backend artifact registry and the Dynamo cache
        # entry registry (mutates global class state).
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact: BackendCacheArtifact[Any],
    ) -> None:
        # [FUNCTION DOC] Stores a deep copy of the artifact in the backend registry under its
        # key, replacing any prior one. Dispatch modes (e.g. fake tensor mode) are disabled
        # during the copy so fake or meta tensors can be cloned.
        """
        Records a backend artifact to be used with dynamo cache entries
        """
        # Temporarily disable all dispatch modes (including FakeTensorMode) during
        # deepcopy to avoid issues with cloning fake tensors (e.g., device mesh
        # with meta tensors that fail when cloning due to device mismatches)
        from torch.utils._mode_utils import no_dispatch

        with no_dispatch():
            cls._backend_artifacts_by_key[_BackendId(artifact.key)] = copy.deepcopy(
                artifact
            )

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        # [FUNCTION DOC] Registers a Dynamo cache entry under the given key, overwriting any
        # existing entry for that key.
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Applies edit_fn to the content of the recorded artifact with the
        # given key, in place. Raises AssertionError if no artifact has that key.
        """
        Edit the content of an existing artifact
        """
        if key not in cls._backend_artifacts_by_key:
            raise AssertionError(f"Key {key} not found in artifacts")
        artifact = cls._backend_artifacts_by_key[_BackendId(key)]
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> BackendCacheArtifact[Any] | None:
        # [FUNCTION DOC] Looks up and returns the recorded backend artifact for the key, or
        # None if absent. Does not serialize anything despite the name.
        """
        Return the backend cache artifact with the associated key
        """
        return cls._backend_artifacts_by_key.get(_BackendId(key), None)

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[_BackendId, BackendCacheArtifact[Any]],
    ) -> dict[str, Any]:
        # [FUNCTION DOC] Builds a mapping with a "dynamo" list (each entry's debug_info plus
        # its key) and a "backends" list (keys of the backend artifacts). Pure; returns the
        # mapping.
        """
        Return a JSON serializable debug dump of all entries in the precompile context
        Called in serialize before serialization, and in populate_caches after deserialization
        """
        # Print debug information
        debug_info: defaultdict[str, list[Any]] = defaultdict(list)
        for key, cache_entry in dynamo_entries.items():
            info = cache_entry.debug_info()
            info["key"] = key
            debug_info["dynamo"].append(info)

        for artifact in backend_artifacts.values():
            debug_info["backends"].append(artifact.key)

        return debug_info

    @classmethod
    def save_to_dynamo_cache(cls) -> dict[str, Any]:
        # [FUNCTION DOC] Builds full precompile cache entries from the registries and writes
        # each to DynamoCache under its key (disk I/O side effect). Returns the debug info
        # mapping from entry creation.
        precompile_cache_entries, debug_info = cls.create_cache_entries()
        for key, entry in precompile_cache_entries.items():
            DynamoCache.write(entry, key)
        return debug_info

    @classmethod
    def create_cache_entries(
        cls,
    ) -> tuple[dict[str, PrecompileCacheEntry], dict[str, Any]]:
        # [FUNCTION DOC] Combines each Dynamo cache entry with the backend artifacts into a
        # PrecompileCacheEntry. Emits a structured-trace artifact with a JSON debug summary.
        # Entries whose conversion returns nothing are skipped; entries that raise are logged
        # as warnings, reported via a structured-trace artifact, and skipped. Returns the
        # mapping of key to entry and the debug info mapping.
        """
        Grabs all the cache entries in the precompile context and
        stitches them together into full PrecompileCacheEntries.
        """
        dynamo_entries = cls._dynamo_cache_entries
        backend_artifacts = cls._backend_artifacts_by_key

        num_artifacts = len(dynamo_entries)

        debug_info = PrecompileContext.dump_debug_info(
            dynamo_entries, backend_artifacts
        )
        debug_str = json.dumps(
            {
                "num_entries": num_artifacts,
                "artifacts": debug_info,
            },
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_entries",
                "encoding": "json",
            },
            payload_fn=lambda: debug_str,
            expect_trace_id=False,
        )

        precompile_cache_entries = {}

        for key, cache_entry in dynamo_entries.items():
            try:
                result = PrecompileCacheEntry.from_cache_entry(
                    cache_entry, backend_artifacts
                )
                if result is not None:
                    precompile_cache_entries[key] = result
            except Exception as e:
                logger.warning("Failed to create cache entry %s", key, exc_info=True)

                error = e
                data = json.dumps(
                    {
                        "key": key,
                        "error": str(error),
                    }
                )
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "dynamo_cache_exception",
                        "encoding": "json",
                    },
                    payload_fn=lambda: data,
                )
                continue
        return precompile_cache_entries, debug_info
