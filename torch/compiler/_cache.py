# [MODULE DOC] Infrastructure for the compiler mega-cache: recording, serializing and hot-loading cache
# artifacts (inductor, autotune, aot_autograd, PGO, precompile) so they can be moved between processes.
# Defines CacheArtifact (abstract), CacheArtifactFactory (type registry), CacheInfo (summary of keys),
# CacheArtifactRecorder and CacheArtifactManager (class-level global state). Only creates a module logger
# and class-level state on import.
import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from itertools import chain
from typing import Any

from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CacheArtifact(ABC):
    # [CLASS DOC] Immutable abstract record of one cached item: a string key plus bytes content (hidden from
    # repr). Subclasses must define a unique type name and implement populate_cache to write the content back
    # into its native cache. Provides static helpers to serialize/deserialize via byte writer/reader and to
    # encode raw content to bytes.
    """
    Data for each cache artifact that will be serialized and deserialized
    """

    key: str
    content: bytes = dataclasses.field(repr=False)  # Do not display potential binary

    @staticmethod
    def serialize(writer: BytesWriter, cls: "CacheArtifact") -> None:
        # [FUNCTION DOC] Static: writes the artifact key as a string and its content as bytes to the writer.
        writer.write_str(cls.key)
        writer.write_bytes(cls.content)

    @staticmethod
    def deserialize(artifact_type: str, reader: BytesReader) -> "CacheArtifact":
        # [FUNCTION DOC] Static: reads a key and content from the reader and builds an artifact of the given
        # registered artifact_type via the factory. Raises an error if the type is not registered.
        key = reader.read_str()
        content = reader.read_bytes()
        return CacheArtifactFactory.create(artifact_type, key, content)

    @staticmethod
    def encode(content: Any) -> bytes:
        # [FUNCTION DOC] Static: default encoder that requires content to already be bytes and returns it;
        # raises an assertion error otherwise. Subclasses may override to convert content.
        if not isinstance(content, bytes):
            raise AssertionError(f"Expected bytes, got {type(content)}")
        return content

    @abstractmethod
    def populate_cache(self) -> None:
        # [FUNCTION DOC] Abstract: subclasses write this artifact's content into the corresponding real cache (side effect).
        pass

    @staticmethod
    def type() -> str:
        # [FUNCTION DOC] Static: returns the unique type name of the artifact class; the base implementation raises
        # a runtime error because the base class is abstract.
        """
        Returns the type of the artifact. Must be unique across all CacheArtifact classes.

        CacheArtifactFactory.register will add property method to CacheInfo based on this (def {type}_artifacts)
        that returns all artifacts for specific cache.
        """
        raise RuntimeError("CacheArtifact is an abstract class, please use a subclass")


class CacheArtifactFactory:
    # [CLASS DOC] Registry mapping artifact type names to CacheArtifact subclasses, held in a class-level
    # mapping. Used to register subclasses and to create artifact instances from a type name, key and content.
    """
    Factory for creating CacheArtifact objects based on their type
    """

    _artifact_types: dict[str, type[CacheArtifact]] = {}

    @classmethod
    def register(cls, artifact_cls: type[CacheArtifact]) -> type[CacheArtifact]:
        # [FUNCTION DOC] Registers artifact_cls under its type name (raises an assertion error if already registered),
        # and dynamically adds a read-only property named after the type plus _artifacts on CacheInfo returning that
        # type's recorded keys. Returns the class so it can be used as a decorator. Mutates global state.
        artifact_type_key = artifact_cls.type()
        if artifact_cls.type() in cls._artifact_types:
            raise AssertionError(
                f"Artifact of type={artifact_type_key} already registered in mega-cache artifact factory"
            )
        cls._artifact_types[artifact_type_key] = artifact_cls
        setattr(
            CacheInfo,
            f"{artifact_type_key}_artifacts",
            property(lambda self: self.artifacts[artifact_type_key]),
        )
        return artifact_cls

    @classmethod
    def _get_artifact_type(cls, artifact_type_key: str) -> type[CacheArtifact]:
        # [FUNCTION DOC] HELPER: Looks up the class registered for the type name; raises an assertion error if absent.
        if artifact_type_key not in cls._artifact_types:
            raise AssertionError(
                f"Artifact of type={artifact_type_key} not registered in mega-cache artifact factory"
            )
        return cls._artifact_types[artifact_type_key]

    @classmethod
    def create(cls, artifact_type_key: str, key: str, content: bytes) -> CacheArtifact:
        # [FUNCTION DOC] Creates an artifact of the given type from key and already-encoded bytes content.
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        # pyrefly: ignore [bad-instantiation]
        return artifact_cls(key, content)

    @classmethod
    def encode_create(
        cls, artifact_type_key: str, key: str, content: Any
    ) -> CacheArtifact:
        # [FUNCTION DOC] Creates an artifact of the given type, first passing raw content through that class's encode.
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        # pyrefly: ignore [bad-instantiation]
        return artifact_cls(key, artifact_cls.encode(content))


@dataclasses.dataclass
class CacheInfo:
    # [CLASS DOC] Instrumentation summary returned by serialize/deserialize: a mapping from artifact type name to
    # list of artifact keys. The per-type properties (inductor, autotune, aot_autograd, pgo, precompile) are
    # empty stubs here; real implementations are attached by CacheArtifactFactory.register.
    """
    Return value of serialization and deserialization for the purpose of
    instrumentation
    """

    artifacts: defaultdict[str, list[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    # Methods set by CacheArtifactFactory.register based on CacheArtifact.type()
    @property
    def inductor_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def autotune_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def aot_autograd_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def pgo_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def precompile_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    def add(self, artifact: CacheArtifact) -> None:
        # [FUNCTION DOC] Appends the artifact's key under its type name.
        self.artifacts[artifact.type()].append(artifact.key)

    def clear(self) -> None:
        # [FUNCTION DOC] Removes all recorded keys.
        self.artifacts.clear()

    def empty(self) -> bool:
        # [FUNCTION DOC] Returns true when no artifacts are recorded.
        return not self.artifacts


def _serialize_single_cache(
    writer: BytesWriter, cls: "tuple[str, list[CacheArtifact]]"
) -> None:
    # [FUNCTION DOC] HELPER: Serializer callback for one cache group: writes the type name, the artifact count and
    # then each artifact. The argument is a pair of type name and artifact list.
    writer.write_str(cls[0])
    writer.write_uint64(len(cls[1]))
    for artifact in cls[1]:
        CacheArtifact.serialize(writer, artifact)


def _deserialize_single_cache(
    reader: BytesReader,
) -> "tuple[str, list[CacheArtifact]]":
    # [FUNCTION DOC] HELPER: Inverse of the single-cache serializer: reads the type name, the count and that many
    # artifacts; returns the pair of type name and list of artifacts.
    artifacts = []
    artifact_type_key = reader.read_str()
    num_artifacts = reader.read_uint64()
    for _ in range(num_artifacts):
        artifacts.append(CacheArtifact.deserialize(artifact_type_key, reader))

    return artifact_type_key, artifacts


CacheArtifactsResult = dict[str, list[CacheArtifact]]


@dataclasses.dataclass(frozen=True)
class CacheArtifactRecorder:
    # [CLASS DOC] Frozen helper bound to one artifact type and key, letting cache implementations record content
    # into the global CacheArtifactManager without knowing about it.
    """
    Shared entry point for cache implementations to record artifacts.
    """

    artifact_type: str
    key: str

    def record(self, content: Any) -> None:
        # [FUNCTION DOC] Records the content for this recorder's type and key with the manager.
        CacheArtifactManager.record_artifact(self.artifact_type, self.key, content)

    def record_if_present(self, content: Any | None) -> None:
        # [FUNCTION DOC] Records the content only if it is not none.
        if content is not None:
            self.record(content)


class CacheArtifactManager:
    # [CLASS DOC] Process-global (class-level state, no instances) collector of cache artifacts. Holds newly recorded
    # artifacts by type, a set of already-seen artifacts to avoid duplicates, an appending byte serializer and a
    # CacheInfo. Lifecycle: record during compilation, serialize to portable bytes, deserialize and populate in
    # another process. No format compatibility guarantees across versions.
    """
    Lightweight manager class for collecting and processing cache artifacts for
    hot loading

    Intended Lifecycle:
    - Execute code via torch.compile, this will call
        CacheArtifactRecorder.record on each cache artifact
    - Call CacheArtifactManager.serialize to convert all the cache artifacts
        to portable format
    - Call CacheArtifactManager.deserialize to hot load the cache artifacts on
        a potentially different process

    NOTE: There's no FB/FC guarantees, results of cache artifacts will not be
          used unless code version matches.
    """

    # Protected by the compile_lock
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    # Keep a separate seen artifacts list to avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]] = (
        AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    )
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        # [FUNCTION DOC] Clears the new artifacts, seen set, serializer and cache info (global state reset).
        cls._new_cache_artifacts.clear()
        cls._seen_artifacts.clear()
        cls._serializer.clear()
        cls._cache_info.clear()

    @classmethod
    @contextmanager
    def with_fresh_cache(cls) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that swaps in empty manager state for the duration of the block and restores
        # the original state afterwards, even on error.
        original_new_cache_artifacts = cls._new_cache_artifacts
        original_seen_artifacts = cls._seen_artifacts
        original_serializer = cls._serializer
        original_cache_info = cls._cache_info

        cls._new_cache_artifacts = defaultdict(list)
        cls._seen_artifacts = OrderedSet()
        cls._serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        cls._cache_info = cls._cache_info.__class__()
        try:
            yield
        finally:
            cls._new_cache_artifacts = original_new_cache_artifacts
            cls._seen_artifacts = original_seen_artifacts
            cls._serializer = original_serializer
            cls._cache_info = original_cache_info

    @classmethod
    def record_artifact(
        cls,
        artifact_type: str,
        key: str,
        content: Any,
    ) -> None:
        # [FUNCTION DOC] Encodes content into an artifact of artifact_type and key via the factory and, unless an equal
        # artifact was already seen, logs it, appends it to the new-artifacts list and marks it seen.
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
        artifact = CacheArtifactFactory.encode_create(artifact_type, key, content)
        if artifact in cls._seen_artifacts:
            return
        log.debug("Recording %s", artifact)
        cls._new_cache_artifacts[artifact_type].append(artifact)
        cls._seen_artifacts.add(artifact)

    @classmethod
    def need_serialize(cls) -> bool:
        # [FUNCTION DOC] Returns true if any new artifacts have been recorded since the last successful serialize.
        """
        Have we seen new artifacts since last serialize call?
        """
        return len(cls._new_cache_artifacts) != 0

    @classmethod
    def serialize(cls) -> tuple[bytes, CacheInfo] | None:
        # [FUNCTION DOC] Adds all new artifacts to the cache info; returns none if there is nothing recorded. Otherwise
        # appends the new artifacts to the serializer, and returns a pair of the serialized bytes and a deep copy of
        # the cache info, clearing the new artifacts. On any error logs a warning and returns none.
        # KNOWN ISSUE: on failure the new artifacts are kept but already added to cache info, so a retry can count them twice.
        """
        Converts the "mega" list into portable format
        """
        for artifact in chain(*cls._new_cache_artifacts.values()):
            log.debug("saving: %s", artifact)
            cls._cache_info.add(artifact)

        if cls._cache_info.empty():
            # If there are no artifacts, don't just return bytes with
            # version.
            return None

        try:
            # We deep copy cls._cache_info since later compilations
            # can keep adding to cache_info
            info = copy.deepcopy(cls._cache_info)
            cls._serializer.extend(cls._new_cache_artifacts.items())
            artifact_bytes = cls._serializer.to_bytes()
            cls._new_cache_artifacts.clear()
            return artifact_bytes, info
        except Exception:
            log.warning("Failed to pickle cache artifacts", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> CacheArtifactsResult | None:
        # [FUNCTION DOC] Static: ensures all artifact classes are registered, then parses the serialized bytes into a
        # mapping of type name to artifact list. Returns none (after logging a warning) on any failure.
        """
        Converts the portable format back into CacheArtifacts
        """
        try:
            CacheArtifactManager._ensure_cache_artifacts_registered()
            artifacts = dict(
                AppendingByteSerializer.to_list(
                    serialized_artifacts,
                    deserialize_fn=_deserialize_single_cache,
                )
            )
        except Exception:
            log.warning("Failed to un-pickle cache artifacts", exc_info=True)
            return None

        return artifacts

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        # [FUNCTION DOC] Static: for every artifact in the mapping, records its key in a new CacheInfo and calls
        # populate_cache to write it into the real cache. Returns the CacheInfo.
        info = CacheInfo()
        for artifact in chain(*artifacts.values()):
            log.debug("writing: %s", artifact)
            info.add(artifact)
            artifact.populate_cache()

        return info

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        # [FUNCTION DOC] HELPER: Imports the modules defining the concrete artifact classes so they register themselves
        # with the factory in a fresh process (import side effect only).
        """When deserializing caches in fresh process, we need to ensure that all
        cache artifacts are registered in the cache registry. This is done by
        simply importing all the cache artifacts already wrapped with register call.
        """
        from torch._dynamo.package import PrecompileCacheArtifact  # noqa: F401
        from torch._dynamo.pgo import PGOCacheArtifact  # noqa: F401
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            AOTAutogradCacheArtifact,
        )
        from torch._inductor.codecache import InductorCacheArtifact  # noqa: F401
        from torch._inductor.runtime.autotune_cache import (  # noqa: F401
            AutotuneCacheArtifact,
        )
