# pylint: disable=useless-parent-delegation
# [MODULE DOC] MTIA graph capture API: graph_pool_handle (new memory pool token), MTIAGraph
# (Python subclass of the native graph object with capture/instantiate/replay/reset/pool) and
# the graph context manager that captures work into an MTIAGraph. Exports the names in __all__.
# No import-time side effects.
from __future__ import annotations

from typing_extensions import Self

import torch


_POOL_HANDLE = tuple[int, int]


def graph_pool_handle() -> _POOL_HANDLE:
    # [FUNCTION DOC] Returns an opaque pair of integers identifying a fresh graph memory pool.
    """
    Return an opaque token representing the id of a graph memory pool.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._mtia_graphPoolHandle()


class MTIAGraph(torch._C._MTIAGraph):
    # [CLASS DOC] Thin Python wrapper over the native MTIA graph. Lifecycle: capture_begin with a
    # pool handle, capture_end, instantiate, then replay any number of times; reset destroys the
    # captured graph. All behavior is delegated to the native base class.
    """
    Wrapper around a MTIA graph.
    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        # [FUNCTION DOC] Creates the native graph object; keep_graph asks it to retain the graph
        # after instantiation (semantics defined natively).
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool: _POOL_HANDLE) -> None:
        # [FUNCTION DOC] Starts capture on the current stream using the given memory pool handle.
        """
        Begin capturing a MTIA graph.
        """
        super().capture_begin(pool)

    def capture_end(self) -> None:
        # [FUNCTION DOC] Ends the capture started by capture_begin.
        """
        End the capture of a MTIA graph.
        """
        super().capture_end()

    def instantiate(self) -> None:
        # [FUNCTION DOC] Builds an executable form of the captured graph.
        """
        Instantiate the captured MTIA graph.
        """
        super().instantiate()

    def replay(self) -> None:
        # [FUNCTION DOC] Launches the captured work again.
        """
        Replay the captured MTIA graph.
        """
        super().replay()

    def reset(self) -> None:
        # [FUNCTION DOC] Destroys the captured graph and resets internal state.
        """
        Destroy the captured graph and reset the states.
        """
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        # [FUNCTION DOC] Returns the memory pool handle used by this graph.
        """
        Return an opaque token representing the id of this graph's memory pool
        """
        return super().pool()


class graph:
    # [CLASS DOC] Context manager that captures MTIA work into an MTIAGraph. State: pool (empty or
    # one-element tuple with the pool handle), capture_stream (given stream, else a class-level
    # default set lazily to the current stream on first use and shared by all instances),
    # stream_ctx (stream context for the capture stream) and mtia_graph (the graph being filled).
    default_capture_stream: torch.mtia.Stream | None = None

    def __init__(
        self,
        mtia_graph: MTIAGraph,
        pool: _POOL_HANDLE | None = None,
        stream: torch.mtia.Stream | None = None,
    ):
        # [FUNCTION DOC] Stores the graph, optional pool handle and capture stream; initializes the
        # shared default capture stream if unset. Raises AssertionError if no capture stream can
        # be determined. Builds the stream context for entering later.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.mtia.current_stream()

        self.pool: tuple[()] | tuple[_POOL_HANDLE] = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise AssertionError("capture_stream must not be None")
        self.stream_ctx = torch.mtia.stream(self.capture_stream)
        self.mtia_graph = mtia_graph

    def __enter__(self) -> None:
        # [FUNCTION DOC] Synchronizes the device and empties the allocator cache, enters the
        # capture stream context and begins capture with the pool handle (or a zero handle).
        torch.mtia.synchronize()
        torch.mtia.empty_cache()

        self.stream_ctx.__enter__()

        pool_arg = self.pool[0] if self.pool else (0, 0)
        self.mtia_graph.capture_begin(pool_arg)

    def __exit__(self, *args: object) -> None:
        # [FUNCTION DOC] Ends capture on the graph, then exits the capture stream context.
        self.mtia_graph.capture_end()
        self.stream_ctx.__exit__(*args)


__all__ = [
    "MTIAGraph",
    "graph",
    "graph_pool_handle",
]
