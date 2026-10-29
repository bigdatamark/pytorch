# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Implements a c10d Store on top of etcd (EtcdStore), the store returned by the legacy EtcdRendezvous
# and by the etcd-v2 backend. Keys and values are base64 encoded so arbitrary bytes can be stored. Falls back to a stub etcd module
# when the etcd package is missing. Also defines cas_delay, a small random sleep used to reduce compare-and-swap contention.
import datetime
import random
import time
from base64 import b64decode, b64encode

# pyre-ignore[21]: Could not find name `Store` in `torch.distributed`.
from torch.distributed import Store


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    from . import _etcd_stub as etcd


# Delay (sleep) for a small random amount to reduce CAS failures.
# This does not affect correctness, but will reduce requests to etcd server.
def cas_delay():
    # [FUNCTION DOC] HELPER: Sleeps for a random duration between 0 and 0.1 seconds to reduce contention on retried etcd operations.
    time.sleep(random.uniform(0, 0.1))


# pyre-fixme[11]: Annotation `Store` is not defined as a type.
class EtcdStore(Store):
    # [CLASS DOC] A Store implementation backed by an etcd directory. All keys live under a prefix (a trailing slash is
    # ensured), and keys/values are base64 encoded. Holds the etcd client and prefix; the optional timeout is applied via
    # set_timeout. Methods: set, get, add (atomic via compare-and-swap), wait, check; blocking waits use etcd watches.
    """
    Implement a c10 Store interface by piggybacking on the rendezvous etcd instance.

    This is the store object returned by ``EtcdRendezvous``.
    """

    def __init__(
        self,
        etcd_client,
        etcd_store_prefix,
        # Default timeout same as in c10d/Store.hpp
        timeout: datetime.timedelta | None = None,
    ):
        # [FUNCTION DOC] Initializes the base Store, saves the client and prefix (appending a slash if missing) and sets the store timeout if one is given.
        super().__init__()  # required for pybind trampoline.

        self.client = etcd_client
        self.prefix = etcd_store_prefix

        if timeout is not None:
            self.set_timeout(timeout)

        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def set(self, key, value):
        # [FUNCTION DOC] Writes the key/value (str or bytes, both base64 encoded) under the prefix in etcd. Raises ValueError for other types.
        """
        Write a key/value pair into ``EtcdStore``.

        Both key and value may be either Python ``str`` or ``bytes``.
        """
        self.client.set(key=self.prefix + self._encode(key), value=self._encode(value))

    def get(self, key) -> bytes:
        # [FUNCTION DOC] Returns the value (bytes) for the key, waiting up to the store timeout for it to be published.
        # Raises LookupError if the key is still missing after the timeout, ValueError for an unsupported key type.
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most ``timeout`` duration or until the key is published.


        Returns:
            value ``(bytes)``

        Raises:
            LookupError - If key still not published after timeout
        """
        b64_key = self.prefix + self._encode(key)
        kvs = self._try_wait_get([b64_key])

        if kvs is None:
            raise LookupError(f"Key {key} not found in EtcdStore")

        return self._decode(kvs[b64_key])

    def add(self, key, num: int) -> int:
        # [FUNCTION DOC] Atomically adds an integer to the decimal-string value at the key and returns the new value. Creates the key with the
        # amount if absent; otherwise loops reading the value and doing compare-and-swap, sleeping a random delay on conflicts.
        """
        Atomically increment a value by an integer amount.

        The integer is represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value


        """
        b64_key = self._encode(key)
        # c10d Store assumes value is an integer represented as a decimal string
        try:
            # Assume default value "0", if this key didn't yet:
            node = self.client.write(
                key=self.prefix + b64_key,
                value=self._encode(str(num)),  # i.e. 0 + num
                prevExist=False,
            )
            return int(self._decode(node.value))
        except etcd.EtcdAlreadyExist:
            pass

        while True:
            # Note: c10d Store does not have a method to delete keys, so we
            # can be sure it's still there.
            node = self.client.get(key=self.prefix + b64_key)
            new_value = self._encode(str(int(self._decode(node.value)) + num))
            try:
                node = self.client.test_and_set(
                    key=node.key, value=new_value, prev_value=node.value
                )
                return int(self._decode(node.value))
            except etcd.EtcdCompareFailed:
                cas_delay()

    # pyrefly: ignore [bad-override]
    def wait(self, keys, override_timeout: datetime.timedelta | None = None):
        # [FUNCTION DOC] Blocks until all keys are present or the timeout (override_timeout if given, else the store timeout) expires.
        # Returns nothing. Raises LookupError on timeout.
        """
        Wait until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError("Timeout while waiting for keys in EtcdStore")
        # No return value on success

    def check(self, keys) -> bool:
        # [FUNCTION DOC] Returns true if all keys are currently present, using a near-zero wait; never blocks meaningfully.
        """Check if all of the keys are immediately present (without waiting)."""
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(
            b64_keys,
            override_timeout=datetime.timedelta(microseconds=1),  # as if no wait
        )
        return kvs is not None

    #
    # Encode key/value data in base64, so we can store arbitrary binary data
    # in EtcdStore. Input can be `str` or `bytes`.
    # In case of `str`, utf-8 encoding is assumed.
    #
    def _encode(self, value) -> str:
        # [FUNCTION DOC] HELPER: Base64-encodes a str (as UTF-8) or bytes into a str. Raises ValueError for any other type.
        if type(value) is bytes:
            return b64encode(value).decode()
        elif type(value) is str:
            return b64encode(value.encode()).decode()
        raise ValueError("Value must be of type str or bytes")

    #
    # Decode a base64 string (of type `str` or `bytes`).
    # Return type is `bytes`, which is more convenient with the Store interface.
    #
    def _decode(self, value) -> bytes:
        # [FUNCTION DOC] HELPER: Base64-decodes a str or bytes into bytes. Raises ValueError for any other type.
        if type(value) is bytes:
            return b64decode(value)
        elif type(value) is str:
            return b64decode(value.encode())
        raise ValueError("Value must be of type str or bytes")

    #
    # Get all of the (base64-encoded) etcd keys at once, or wait until all the keys
    # are published or timeout occurs.
    # This is a helper method for the public interface methods.
    #
    # On success, a dictionary of {etcd key -> etcd value} is returned.
    # On timeout, None is returned.
    #
    def _try_wait_get(self, b64_keys, override_timeout=None):
        # [FUNCTION DOC] HELPER: Fetches the prefix directory from etcd and checks whether all the given encoded keys exist. If so returns a
        # mapping from etcd key to value; otherwise watches the directory for changes (from the last seen etcd index) and retries until the
        # deadline, returning None on timeout. Missing directory and cleared-event-index conditions are retried.
        timeout = self.timeout if override_timeout is None else override_timeout  # type: ignore[attr-defined]
        deadline = time.time() + timeout.total_seconds()

        while True:
            # Read whole directory (of keys), filter only the ones waited for
            all_nodes = None
            try:
                all_nodes = self.client.get(key=self.prefix)
                req_nodes = {
                    node.key: node.value
                    for node in all_nodes.children
                    if node.key in b64_keys
                }

                if len(req_nodes) == len(b64_keys):
                    # All keys are available
                    return req_nodes
            except etcd.EtcdKeyNotFound:
                pass

            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None

            try:
                index = all_nodes.etcd_index + 1 if all_nodes else 0
                self.client.watch(
                    key=self.prefix,
                    recursive=True,
                    timeout=watch_timeout,
                    index=index,
                )
            except etcd.EtcdWatchTimedOut:
                if time.time() >= deadline:
                    return None
                else:
                    continue
            except etcd.EtcdEventIndexCleared:
                continue
