# mypy: allow-untyped-defs
# [MODULE DOC] Rendezvous: lets processes find each other and obtain a shared key/value
# store plus rank and world size from a URL. Holds a registry of scheme handlers and
# built-in tcp, env and file handlers. Import-time side effect: registers those three
# handlers (RuntimeError if already registered).
try:
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    raise ImportError(
        "urllib cannot be found, urlparse from python2 is no longer supported."
    ) from e

import numbers
import os
import sys
from collections.abc import Callable, Iterator
from datetime import timedelta

from torch.distributed import FileStore, Store, TCPStore

from .constants import default_pg_timeout


_rendezvous_handlers: dict[str, Callable[..., Iterator[tuple[Store, int, int]]]] = {}

__all__ = ["register_rendezvous_handler", "rendezvous"]


def register_rendezvous_handler(scheme, handler):
    # [FUNCTION DOC] Adds a generator-function handler for a URL scheme to the global
    # registry. Raises RuntimeError if the scheme is already registered.
    """
    Register a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Args:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        raise RuntimeError(f"Rendezvous handler for {scheme}:// already registered")
    _rendezvous_handlers[scheme] = handler


# Query will have format "rank=0&world_size=1" and is
# converted into {"rank": 0, "world_size": 1}
def _query_to_dict(query: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Parses an ampersand-separated key=value query string into a
    # mapping of strings, skipping empty segments. KNOWN ISSUE: a segment without exactly
    # one equals sign raises IndexError or leaves extra parts ignored; values are not decoded.
    return {
        pair[0]: pair[1]
        for pair in (pair.split("=") for pair in filter(None, query.split("&")))
    }


def _get_use_libuv_from_query_dict(query_dict: dict[str, str]) -> bool:
    # [FUNCTION DOC] HELPER: Decides whether TCPStore uses libuv: reads the use_libuv query
    # entry, falling back to the USE_LIBUV environment variable, then to default on (off on
    # Windows). True only when the value is the string "1".
    # libuv is the default backend for TCPStore. To enable the non-libuv backend,
    # user can explicitly specify ``use_libuv=0`` in the URL parameter.
    if sys.platform == "win32":
        #  PyTorch is built without libuv support on windows, so default to 0
        return query_dict.get("use_libuv", os.environ.get("USE_LIBUV", "0")) == "1"
    return query_dict.get("use_libuv", os.environ.get("USE_LIBUV", "1")) == "1"


def _rendezvous_helper(url: str, rank: int, world_size_opt: int | None, **kwargs):
    # [FUNCTION DOC] HELPER: Normalizes rank and world size into the URL query and dispatches
    # to the scheme handler, returning its generator. If world size is None (dynamic group) and
    # the scheme is env, rank and world size may come from RANK and WORLD_SIZE environment
    # variables. Raises AssertionError if the URL already has rank or world_size, and
    # RuntimeError if no handler exists for the scheme. Extra kwargs go to the handler.
    result = urlparse(url)
    if world_size_opt is None:
        world_size = -1
        if result.scheme == "env":
            rank = int(os.environ.get("RANK", rank))
            # If the world_size env variable is not present then it is a dynamic group
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
    else:
        world_size = world_size_opt
    if rank != -1 or world_size != -1 or world_size_opt is None:
        query_dict = _query_to_dict(result.query)
        if "rank" in query_dict or "world_size" in query_dict:
            raise AssertionError(
                f"The url: {url} has node-specific arguments(rank, world_size) already."
            )
        if rank != -1:
            query_dict["rank"] = str(rank)
        if world_size != -1 or world_size_opt is None:
            query_dict["world_size"] = str(world_size)
        result = result._replace(
            query=f"{'&'.join([f'{k}={v}' for k, v in query_dict.items()])}"
        )
        # pyrefly: ignore [bad-assignment]
        url = urlunparse(result)

    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError(f"No rendezvous handler for {result.scheme}://")
    return _rendezvous_handlers[result.scheme](url, **kwargs)


def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    # [FUNCTION DOC] Public entry: validates that url is a string or bytes and rank and
    # world_size are integers (RuntimeError otherwise), then returns the handler generator
    # yielding (store, rank, world_size). Rank or world size of -1 means unspecified.
    if not isinstance(url, (str, bytes)):
        raise RuntimeError(f"`url` must be a string. {type(url)}: {url}")

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError(f"`rank` must be an integer. {rank}")

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError(f"`world_size` must be an integer. {world_size}")

    return _rendezvous_helper(url, rank, world_size, **kwargs)


def _create_store_from_options(backend_options, rank):
    # [FUNCTION DOC] HELPER: Runs rendezvous on the options' init_method with unknown world
    # size (dynamic group) and returns only the store from the first yielded triple.
    store, _, _ = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store


def _rendezvous_error(msg):
    # [FUNCTION DOC] HELPER: Builds (does not raise) a ValueError with a standard prefix.
    return ValueError("Error initializing torch.distributed using " + msg)


def _file_rendezvous_handler(url: str, **kwargs):
    # [FUNCTION DOC] HELPER: Generator handler for file:// URLs. Extracts the path (converted
    # on Windows), requires rank and world_size query entries (ValueError otherwise), creates
    # a FileStore, and yields (store, rank, world_size). Resuming it again raises
    # RuntimeError since re-rendezvous is unsupported.
    def _error(msg):
        # [FUNCTION DOC] HELPER: Builds a file:// specific ValueError.
        return _rendezvous_error("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if sys.platform == "win32":
        import urllib.request

        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            # Normalizing an empty string produces ".", which is not expected.
            path = os.path.normpath(path)

    if not path:
        raise _error("path missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


def _torchelastic_use_agent_store() -> bool:
    # [FUNCTION DOC] HELPER: True if the TORCHELASTIC_USE_AGENT_STORE environment variable
    # equals the string "True".
    return os.environ.get("TORCHELASTIC_USE_AGENT_STORE", None) == str(True)


def _agent_store_serves(hostname, port) -> bool:
    # [FUNCTION DOC] HELPER: True only when the elastic agent store flag is set and the given
    # host and port match the MASTER_ADDR and MASTER_PORT environment variables.
    """Whether the elastic agent is the one serving a store at ``hostname:port``.

    ``TORCHELASTIC_USE_AGENT_STORE`` only promises a server at the agent's own
    ``MASTER_ADDR:MASTER_PORT``, which the agent exports alongside it. Honoring
    the flag for any other endpoint makes every rank a client of a store nobody
    hosts. ``TCPStore``'s C++ constructor applies the same address check before
    it forgives a failed server bind.
    """
    return (
        _torchelastic_use_agent_store()
        and os.environ.get("MASTER_ADDR") == hostname
        and os.environ.get("MASTER_PORT") == str(port)
    )


def _create_c10d_store(
    hostname, port, rank, world_size, timeout, use_libuv=True
) -> Store:
    # [FUNCTION DOC] HELPER: Creates a TCPStore for the given host and port. Raises ValueError
    # if the port is outside 0..65535. If the elastic agent hosts the store, every rank gets a
    # client (not master); otherwise rank 0 hosts the server (multi-tenant, optional libuv)
    # and other ranks are clients. Returns the store.
    """
    Smartly creates a c10d Store object on ``rank`` based on whether we need to reuse agent store.

    The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    By default, the TCPStore server uses the asynchronous implementation
    ``LibUVStoreDaemon`` which utilizes libuv.

    If ``torchelastic_use_agent_store()`` is ``True`` *and* ``hostname:port`` is
    the agent's own ``MASTER_ADDR:MASTER_PORT``, then it is assumed that the
    agent leader (node rank 0) hosts the TCPStore server there. Hence
    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).

    Otherwise rank 0 will host the TCPStore (with multi-tenancy) and it is
    assumed that rank 0's hostname and port are correctly passed via
    ``hostname`` and ``port``. All non-zero ranks will create and return a
    TCPStore client.
    """
    # check if port is uint16_t
    if not 0 <= port < 2**16:
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    if _agent_store_serves(hostname, port):
        # We create a new TCPStore for every retry so no need to add prefix for each attempt.
        return TCPStore(
            host_name=hostname,
            port=port,
            world_size=world_size,
            is_master=False,
            timeout=timeout,
        )
    else:
        start_daemon = rank == 0
        return TCPStore(
            host_name=hostname,
            port=port,
            world_size=world_size,
            is_master=start_daemon,
            timeout=timeout,
            multi_tenant=True,
            use_libuv=use_libuv,
        )


def _tcp_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    # [FUNCTION DOC] HELPER: Generator handler for tcp:// URLs. Needs host, port, rank and
    # world_size in the URL (ValueError if port, rank or world size missing; AssertionError if
    # no hostname). Builds a TCPStore via _create_c10d_store and yields (store, rank,
    # world_size); resuming it raises RuntimeError since re-rendezvous is unsupported.
    def _error(msg):
        # [FUNCTION DOC] HELPER: Builds a tcp:// specific ValueError.
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if result.port is None:
        raise _error("port number missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    if result.hostname is None:
        raise AssertionError("hostname cannot be None")

    store = _create_c10d_store(
        result.hostname, result.port, rank, world_size, timeout, use_libuv
    )

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using tcp:// method")


def _env_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    # [FUNCTION DOC] HELPER: Generator handler for env:// URLs. Rank and world size come from
    # the URL query or else the RANK and WORLD_SIZE environment variables; master address and
    # port always come from MASTER_ADDR and MASTER_PORT. Missing variables raise ValueError.
    # Yields (store, rank, world_size); resuming it raises RuntimeError.
    def _error(msg):
        # [FUNCTION DOC] HELPER: Builds an env:// specific ValueError.
        return _rendezvous_error("env:// rendezvous: " + msg)

    def _env_error(var):
        # [FUNCTION DOC] HELPER: Builds the error for an unset environment variable.
        return _error(f"environment variable {var} expected, but not set")

    def _get_env_or_raise(env_var: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the environment variable value; raises ValueError
        # (via _env_error) when it is unset or empty.
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val

    result = urlparse(url)
    query_dict = _query_to_dict(result.query)

    rank: int
    world_size: int
    master_port: int
    master_addr: str

    if "rank" in query_dict:
        rank = int(query_dict["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    if "world_size" in query_dict:
        world_size = int(query_dict["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))

    master_addr = _get_env_or_raise("MASTER_ADDR")
    master_port = int(_get_env_or_raise("MASTER_PORT"))
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    store = _create_c10d_store(
        master_addr, master_port, rank, world_size, timeout, use_libuv
    )

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using env:// method")


register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)
