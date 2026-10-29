# [MODULE DOC] Concrete cache backends behind the _CacheImpl interface: in-memory, on-disk (files with a
# version header, guarded by a file lock) and a remote fallback that raises NotImplementedError
# unless an internal fb module provides a real one. Also defines the Hit wrapper and the
# miss sentinel. Import-time side effect: tries to import the optional fb remote implementation.
"""Cache implementation classes

This module provides concrete implementations of caching backends including
in-memory, on-disk, and remote caching strategies. Each implementation follows
the abstract _CacheImpl interface and provides thread-safe operations with
appropriate locking mechanisms.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BufferedReader, BufferedWriter
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Generic, TypeVar
from typing_extensions import override

from filelock import BaseFileLock, FileLock

from . import locks


# Type variable for cache value types
_V = TypeVar("_V")


@dataclass
class Hit(Generic[_V]):
    # [CLASS DOC] Dataclass wrapping a cached value so a hit on a stored None differs from a miss (None).
    """Result wrapper for hits on cache get operations.

    Allows distinguishing between a cache miss and a cached None value.

    Attributes:
        value: The cached value. The type depends on the cache implementation:
              - InMemoryCache: object (any Python object)
              - OnDiskCache: bytes
              - RemoteCache: bytes
    """

    value: _V


class Miss:
    # [CLASS DOC] Sentinel type for a cache miss; a single shared instance named miss exists.
    """Sentinel class representing a cache miss.

    Used to distinguish between a cached None value and a cache miss
    when None is a valid cached value.
    """


# Singleton instance for cache miss sentinel
miss = Miss()


class _CacheImpl(ABC, Generic[_V]):
    # [CLASS DOC] Abstract base for cache backends. Holds a threading lock exposed through the lock
    # property as a timeout-taking context manager factory. Subclasses implement get (returns a
    # Hit or None) and insert (returns whether stored). Callers, not the backend, decide when to lock.
    """Abstract base class for cache implementations.

    This class defines the interface that all cache implementations must follow.
    It provides thread-safe operations through a locking mechanism and supports
    both get and insert operations.

    Type Parameters:
        _V: The type of values stored in the cache. Each implementation specifies
           its own value type (e.g., object for in-memory, bytes for on-disk).
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the threading lock used by the default lock property.
        """Initialize the cache implementation with a threading lock."""
        self._lock: Lock = Lock()

    @property
    def lock(self) -> locks._LockProtocol:
        # [FUNCTION DOC] Property returning a function that, given an optional timeout, yields a context manager
        # holding the threading lock (raising LockTimeoutError on timeout). The lock is not taken by get or insert.
        """Get a context manager for acquiring the cache lock.

        Locking of the cache is not done by the implementation itself, but by the
        interface that uses it. The interface may want to hold the lock for longer
        than a single cache operation, for example when dealing with multiple
        cache implementations at once, so we leave that decision up to the interface.

        Args:
            timeout: Optional timeout in seconds (float) for acquiring the lock.

        Returns:
            A callable that returns a context manager for the lock.
        """

        def _lock_with_timeout(
            timeout: float | None = None,
        ) -> locks._LockContextManager:
            # [FUNCTION DOC] HELPER: Returns the context manager acquiring the instance threading lock with the timeout.
            return locks._acquire_lock_with_timeout(self._lock, timeout)

        return _lock_with_timeout

    @abstractmethod
    def get(self, key: str) -> Hit[_V] | None:
        # [FUNCTION DOC] Abstract: look up the key and return a Hit with the value, or None on a miss.
        """Retrieve a value from the cache.

        Args:
            key: The key to look up in the cache (must be str).

        Returns:
            A Hit object on cache hit where Hit.value is the cached value,
            or None on cache miss.
        """

    @abstractmethod
    def insert(self, key: str, value: _V) -> bool:
        # [FUNCTION DOC] Abstract: store the value under the key; return True if stored, False if not.
        """Insert a key-value pair into the cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key.

        Returns:
            True if the insertion was successful, False if not inserted.
        """


class _InMemoryCacheImpl(_CacheImpl[_V], Generic[_V]):
    # [CLASS DOC] Backend storing values in a plain dictionary keyed by string. Fast but limited to the
    # process lifetime. State: the _memory dictionary. Existing keys are never overwritten.
    """In-memory cache implementation using a dictionary.

    This implementation stores key-value pairs in a Python dictionary,
    with keys being pickled for consistent hashing. It provides fast
    access but is limited by available memory and process lifetime.

    Type Parameters:
        _V: The type of values stored in the cache.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the base lock and an empty dictionary.
        """Initialize the in-memory cache with an empty dictionary."""
        super().__init__()
        self._memory: dict[str, _V] = {}

    @override
    def get(self, key: str) -> Hit[_V] | None:
        # [FUNCTION DOC] Returns a Hit holding the stored value if the key is present, otherwise None.
        """Retrieve a value from the in-memory cache.

        Args:
            key: The key to look up (must be str).

        Returns:
            A Hit object on cache hit where Hit.value is the cached value,
            or None on cache miss.
        """
        if key in self._memory:
            return Hit(value=self._memory[key])
        return None

    @override
    def insert(self, key: str, value: _V) -> bool:
        # [FUNCTION DOC] Stores the value only if the key is absent; returns True when stored, False when the key already exists (first write wins).
        """Insert a key-value pair into the in-memory cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key.

        Returns:
            True if the insertion was successful (key was new),
            False if not inserted (key already existed).
        """
        if key not in self._memory:
            self._memory[key] = value
            return True
        return False


class _OnDiskCacheImpl(_CacheImpl[bytes]):
    # [CLASS DOC] Backend storing each entry as a file named by the key inside a cache sub-directory.
    # Each file starts with a short version header (hash of the class version number); entries
    # with a mismatching header are treated as stale and removed. Uses a cross-process file lock
    # (dir.lock) instead of the thread lock. Values are bytes. First write wins.
    """On-disk cache implementation using file system storage.

    This implementation stores cached data as files on disk, with version
    headers to handle cache invalidation. It uses file locking to ensure
    thread safety across processes and provides persistent storage that
    survives process restarts.

    The value type is 'bytes' since data must be serialized to bytes for disk storage.

    Attributes:
        _version: _Version number for cache format compatibility.
        _version_header_length: Length of the version header in bytes.
    """

    _version: int = 0
    _version_header_length: int = 4

    def __init__(self, sub_dir: PathLike[str] | None = None) -> None:
        # [FUNCTION DOC] Sets the cache directory to the default cache directory plus the cache folder and the optional
        # sub-directory, and creates the file lock object for the directory. No directories are created here.
        # KNOWN ISSUE: does not call the base initializer, so the inherited threading lock is never created (unused here since lock is overridden).
        """Initialize the on-disk cache with a specified subdirectory.

        Args:
            sub_dir: Subdirectory name within the cache directory.
                    Defaults to empty string if not specified.
        """
        self._cache_dir: Path = self._base_dir / (sub_dir or "")
        self._flock: BaseFileLock = FileLock(str(self._cache_dir / "dir.lock"))

    @property
    def _base_dir(self) -> Path:
        # [FUNCTION DOC] HELPER: Property returning the root on-disk cache path: the default inductor cache directory joined with the cache folder.
        """Get the base directory for cache storage.

        Returns:
            Path to the cache directory based on the default cache dir
            and the specified subdirectory.
        """
        from torch._inductor.runtime.runtime_utils import default_cache_dir

        return Path(default_cache_dir(), "cache")

    def _fpath_from_key(self, key: str) -> Path:
        # [FUNCTION DOC] HELPER: Returns the file path for a key inside the cache directory. The key is used as the file name unsanitized.
        """Generate a file path from a cache key.

        Args:
            key: The cache key to convert to a file path (must be str).

        Returns:
            A Path object representing the file location for this key.
        """
        return self._cache_dir / key

    @classmethod
    def _version_header(cls) -> bytes:
        # [FUNCTION DOC] HELPER: Class method returning the leading bytes (header length) of the SHA-256 digest of the class version number.
        """Generate the version header bytes.

        Returns:
            A byte string representing the current cache version header.
        """
        return sha256(str(cls._version).encode()).digest()[: cls._version_header_length]

    def _version_header_matches(self, fp: BufferedReader) -> bool:
        # [FUNCTION DOC] HELPER: Reads the header length from the file's current position and returns whether it equals the current version header.
        """Check if the file's version header matches the current version.

        Args:
            fp: File pointer positioned at the start of the file.

        Returns:
            True if the version header matches, False otherwise.
        """
        return fp.read(self._version_header_length) == self._version_header()

    def _write_version_header(self, fp: BufferedWriter) -> None:
        # [FUNCTION DOC] HELPER: Writes the version header bytes to the open file.
        """Write the version header to a file.

        Args:
            fp: File pointer where the version header should be written.
        """
        fp.write(self._version_header())

    @override
    @property
    def lock(self) -> locks._LockProtocol:
        # [FUNCTION DOC] Property returning a function that, given an optional timeout, yields a context manager
        # holding the directory file lock (raising FileLockTimeoutError on timeout).
        """Get a context manager for acquiring the file lock.

        Uses file locking to ensure thread safety across processes.

        Args:
            timeout: Optional timeout in seconds (float) for acquiring the file lock.

        Returns:
            A callable that returns a context manager for the file lock.
        """

        def _lock_with_timeout(
            timeout: float | None = None,
        ) -> locks._LockContextManager:
            # [FUNCTION DOC] HELPER: Returns the context manager acquiring the instance file lock with the timeout.
            return locks._acquire_flock_with_timeout(self._flock, timeout)

        return _lock_with_timeout

    @override
    def get(self, key: str) -> Hit[bytes] | None:
        # [FUNCTION DOC] Returns a Hit with the file contents after the header, or None. Returns None if the file does not exist.
        # If the header mismatches (or the payload is empty) the file is deleted and None is returned. Reads and may delete files.
        # KNOWN ISSUE: an entry with an empty payload is also treated as stale and deleted.
        """Retrieve a value from the on-disk cache.

        Args:
            key: The key to look up in the cache (must be str).

        Returns:
            A Hit object on cache hit where Hit.value is the cached value (bytes),
            or None on cache miss or version mismatch.
        """
        fpath: Path = self._fpath_from_key(key)

        if not fpath.is_file():
            return None

        pickled_value: bytes | None = None
        with open(fpath, "rb") as fp:
            if self._version_header_matches(fp):
                pickled_value = fp.read()

        if not pickled_value:
            # if pickled_value is still None, even though the file exists, then
            # we know that the version header did not match. in this case implementation
            # is up to preference, we choose to remove entries that do not match
            # the version header so that the key can be re-cached later with the correct
            # version header
            fpath.unlink()
            return None

        return Hit(value=pickled_value)

    @override
    def insert(self, key: str, value: bytes) -> bool:
        # [FUNCTION DOC] Writes the header and value to a new file named by the key, creating parent directories. Uses exclusive creation:
        # if the file exists with a current header, nothing is written and False is returned; if it exists with a stale header it is
        # deleted and rewritten. Returns True when written. Side effects: creates, deletes and writes files; callers should hold the lock.
        """Insert a key-value pair into the on-disk cache.

        Args:
            key: The key to insert (must be str).
            value: The value to associate with the key (must be bytes).

        Returns:
            True if successfully inserted, False if the key already exists
            with a valid version.
        """
        fpath: Path = self._fpath_from_key(key)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        r_fp, w_fp, inserted = None, None, False
        try:
            w_fp = open(fpath, "xb")  # noqa: SIM115
        except FileExistsError:
            is_stale: bool = False
            with open(fpath, "rb") as r_fp:
                is_stale = not self._version_header_matches(r_fp)

            if is_stale:
                # same story as above, in this case the version header doesn't
                # match so we choose to remove the old entry so that the new
                # k/v pair can be cached
                fpath.unlink()
                w_fp = open(fpath, "xb")  # noqa: SIM115
            else:
                w_fp = None
        finally:
            if w_fp:
                try:
                    self._write_version_header(w_fp)
                    w_fp.write(value)
                    inserted = True
                finally:
                    w_fp.close()

        return inserted


try:
    from .fb.implementations import _RemoteCacheImpl
except ModuleNotFoundError:

    class _RemoteCacheImpl(_CacheImpl[bytes]):
        # [CLASS DOC] Fallback remote backend used when the internal fb implementation is missing. Every get or insert
        # raises NotImplementedError. Class attributes: version number and has_strong_consistency (False). Its lock is a no-op.
        """Fallback remote cache implementation for non-Facebook environments.

        This is a no-op implementation that always raises NotImplementedError.
        The actual remote cache implementation is provided in the `.fb` module
        for Facebook-specific environments.

        The value type is 'bytes' for consistency with the FB implementation.

        Attributes:
            _version: _Version number for cache format compatibility.
            has_strong_consistency: Whether the remote cache provides strong
                                   consistency guarantees.
        """

        _version: int = 0
        has_strong_consistency: bool = False

        def __init__(self) -> None:
            # [FUNCTION DOC] Does nothing; no state or lock is created.
            """Initialize the fallback remote cache implementation.

            Note: We don't need to initialize any form of lock since this
            implementation provides a pseudo-lock context manager.
            """

        @override
        @property
        def lock(self) -> locks._LockProtocol:
            # [FUNCTION DOC] Property returning a function that, given an optional timeout (ignored), yields a no-op context manager.
            """Get a pseudo lock that does nothing.

            Most remote cache implementations don't have an ability to implement
            any form of locking, so we provide a no-op pseudo-lock for consistency
            with the interface.

            Args:
                timeout: Optional timeout in seconds (float). Ignored in this

            Returns:
                A callable that returns a no-op context manager.
            """

            @contextmanager
            def pseudo_lock(
                timeout: float | None = None,
            ) -> Generator[None, None, None]:
                # [FUNCTION DOC] HELPER: No-op context manager that yields immediately and ignores the timeout.
                yield

            return pseudo_lock

        @override
        def get(self, key: str) -> Hit[bytes] | None:
            # [FUNCTION DOC] Always raises NotImplementedError.
            """Raise NotImplementedError for remote cache get operations.

            Args:
                key: The key to look up (must be str, ignored).

            Raises:
                NotImplementedError: Always raised as this is a fallback implementation.
            """
            raise NotImplementedError

        @override
        def insert(self, key: str, value: bytes) -> bool:
            # [FUNCTION DOC] Always raises NotImplementedError.
            """Raise NotImplementedError for remote cache insert operations.

            Args:
                key: The key to insert (must be str, ignored).
                value: The value to insert (must be bytes, ignored).

            Raises:
                NotImplementedError: Always raised as this is a fallback implementation.
            """
            raise NotImplementedError
