# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Etcd-backed RendezvousBackend used by the dynamic rendezvous ("etcd-v2"). Holds the serialized rendezvous state
# in a single etcd key with compare-and-swap semantics, and provides create_backend which builds the backend plus an EtcdStore
# from RendezvousParameters. Falls back to a stub etcd module if the etcd package is not installed.
import binascii
from base64 import b64decode, b64encode
from typing import cast

import urllib3.exceptions  # type: ignore[import]


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    from . import _etcd_stub as etcd

from torch.distributed import Store

from .api import RendezvousConnectionError, RendezvousParameters, RendezvousStateError
from .dynamic_rendezvous import RendezvousBackend, Token
from .etcd_store import EtcdStore
from .utils import parse_rendezvous_endpoint


class EtcdRendezvousBackend(RendezvousBackend):
    # [CLASS DOC] Rendezvous backend that stores the opaque rendezvous state (base64 text) under one etcd key
    # (optional key prefix, slash, run id) with a TTL (default 7200 seconds when unset or non-positive).
    # The etcd modified index serves as the version token for optimistic concurrency. Key methods: name, get_state, set_state.
    """Represents an etcd-based rendezvous backend.

    Args:
        client:
            The ``etcd.Client`` instance to use to communicate with etcd.
        run_id:
            The run id of the rendezvous.
        key_prefix:
            The path under which to store the rendezvous state in etcd.
        ttl:
            The TTL of the rendezvous state. If not specified, defaults to two hours.
    """

    _DEFAULT_TTL = 7200  # 2 hours

    _client: etcd.Client
    _key: str
    _ttl: int

    def __init__(
        self,
        client: etcd.Client,
        run_id: str,
        key_prefix: str | None = None,
        ttl: int | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the etcd client, builds the state key from the optional key_prefix and run_id, and picks the TTL
        # (given value if positive, else the default). Raises ValueError if run_id is empty.
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._client = client

        if key_prefix:
            self._key = key_prefix + "/" + run_id
        else:
            self._key = run_id

        if ttl and ttl > 0:
            self._ttl = ttl
        else:
            self._ttl = self._DEFAULT_TTL

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Property returning the backend name "etcd-v2".
        """See base class."""
        return "etcd-v2"

    def get_state(self) -> tuple[bytes, Token] | None:
        # [FUNCTION DOC] Reads the state key from etcd. Returns None if the key does not exist, otherwise a (state bytes, token) pair
        # where the token is the etcd modified index. Raises RendezvousConnectionError on etcd or timeout errors and
        # RendezvousStateError if the stored value is not valid base64.
        """See base class."""
        try:
            result = self._client.read(self._key)
        except etcd.EtcdKeyNotFound:
            return None
        except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        return self._decode_state(result)

    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        # [FUNCTION DOC] Compare-and-set of the state. Without a token the write only succeeds if the key does not exist; with a token
        # it only succeeds if the key's modified index equals the token. A token that cannot be converted to an integer is treated as stale.
        # Returns (state, token, True) when the write was applied; on a lost race (already exists / compare failed / stale token)
        # returns the current state as (state, token, False), or None if the key is gone. Raises RendezvousConnectionError
        # on etcd or timeout errors. Writes to etcd with the TTL.
        """See base class."""
        base64_state = b64encode(state).decode()

        kwargs = {}

        def get_state():
            # [FUNCTION DOC] HELPER: Reads current state and returns it with a False "applied" flag, or None if absent.
            result = self.get_state()
            if result is not None:
                return *result, False
            return None

        if token:
            try:
                token = int(token)
            except ValueError:
                return get_state()

        if token:
            kwargs["prevIndex"] = token
        else:
            kwargs["prevExist"] = False

        try:
            result = self._client.write(self._key, base64_state, self._ttl, **kwargs)
        except (etcd.EtcdAlreadyExist, etcd.EtcdCompareFailed):
            result = None
        except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
            raise RendezvousConnectionError(
                "The connection to etcd has failed. See inner exception for details."
            ) from exc

        if result is None:
            return get_state()

        tmp = *self._decode_state(result), True
        return tmp

    def _decode_state(self, result: etcd.EtcdResult) -> tuple[bytes, Token]:
        # [FUNCTION DOC] HELPER: Decodes the base64 value of an etcd result into bytes and returns it with the modified index as token.
        # Raises RendezvousStateError if the value is corrupt.
        # pyrefly: ignore [missing-attribute]
        base64_state = result.value.encode()

        try:
            state = b64decode(base64_state)
        except binascii.Error as exc:
            raise RendezvousStateError(
                "The state object is corrupt. See inner exception for details."
            ) from exc

        # pyrefly: ignore [missing-attribute]
        return state, result.modifiedIndex


def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    # [FUNCTION DOC] HELPER: Builds an etcd client from the endpoint (default port 2379) and params: read_timeout (default 60, must be positive),
    # protocol (http or https, default http), optional ssl_cert (paired with ssl_cert_key into a tuple) and ca_cert; reconnect allowed.
    # Raises ValueError for invalid timeout or protocol and RendezvousConnectionError if the client cannot be created.
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)

    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")

    # The communication protocol
    protocol = params.get("protocol", "http").strip().lower()
    if protocol != "http" and protocol != "https":
        raise ValueError("The protocol must be HTTP or HTTPS.")

    # The SSL client certificate
    ssl_cert = params.get("ssl_cert")
    if ssl_cert:
        ssl_cert_key = params.get("ssl_cert_key")
        if ssl_cert_key:
            # The etcd client expects the certificate key as the second element
            # of the `cert` tuple.
            ssl_cert = (ssl_cert, ssl_cert_key)

    # The root certificate
    ca_cert = params.get("ca_cert")

    try:
        return etcd.Client(
            host,
            port,
            read_timeout=read_timeout,
            protocol=protocol,
            cert=ssl_cert,
            ca_cert=ca_cert,
            allow_reconnect=True,
        )
    except (etcd.EtcdException, urllib3.exceptions.TimeoutError) as exc:
        raise RendezvousConnectionError(
            "The connection to etcd has failed. See inner exception for details."
        ) from exc


def create_backend(params: RendezvousParameters) -> tuple[EtcdRendezvousBackend, Store]:
    # [FUNCTION DOC] Creates the etcd client from params, an EtcdRendezvousBackend keyed under the rendezvous prefix with the run id,
    # and an EtcdStore under the store prefix. Returns the (backend, store) pair. Errors propagate from client creation.
    """Create a new :py:class:`EtcdRendezvousBackend` from the specified parameters.

    +--------------+-----------------------------------------------------------+
    | Parameter    | Description                                               |
    +==============+===========================================================+
    | read_timeout | The read timeout, in seconds, for etcd operations.        |
    |              | Defaults to 60 seconds.                                   |
    +--------------+-----------------------------------------------------------+
    | protocol     | The protocol to use to communicate with etcd. Valid       |
    |              | values are "http" and "https". Defaults to "http".        |
    +--------------+-----------------------------------------------------------+
    | ssl_cert     | The path to the SSL client certificate to use along with  |
    |              | HTTPS. Defaults to ``None``.                              |
    +--------------+-----------------------------------------------------------+
    | ssl_cert_key | The path to the private key of the SSL client certificate |
    |              | to use along with HTTPS. Defaults to ``None``.            |
    +--------------+-----------------------------------------------------------+
    | ca_cert      | The path to the root SSL authority certificate. Defaults  |
    |              | to ``None``.                                              |
    +--------------+-----------------------------------------------------------+
    """
    client = _create_etcd_client(params)

    backend = EtcdRendezvousBackend(
        client, params.run_id, key_prefix="/torch/elastic/rendezvous"
    )

    store = EtcdStore(client, "/torch/elastic/store")

    return backend, store
