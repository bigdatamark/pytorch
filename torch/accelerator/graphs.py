# [MODULE DOC] Defines Graph, a Python wrapper over the C++ accelerator graph for capture and replay of operation sequences. Exports only Graph. No import-time side effects beyond importing the C++ base class.
import gc
from typing import Literal
from typing_extensions import Self

import torch
from torch._C import _acceleratorGraph


class Graph(_acceleratorGraph):
    # [CLASS DOC] Accelerator graph that can be captured (usable as a context manager) and replayed with lower launch overhead.
    # Subclasses the C++ _acceleratorGraph. State: graph_pool (optional memory pool id tuple) and capture_error_mode (default, global, thread_local, relaxed), both stored at construction and used on each capture_begin. keep_graph controls whether the underlying graph is kept after capture_end.
    # Methods mostly forward to the C++ base: capture_begin, capture_end, instantiate, replay, reset, pool, enable_debug_mode, debug_dump.
    r"""
    Wrapper around an :ref:`accelerator<accelerators>` graph that supports capture and replay.

    A graph captures a sequence of operations and their dependencies, allowing them to be
    replayed efficiently with reduced overhead. This class can be used as a context manager
    to automatically capture operations on the current stream.

    Arguments:
        keep_graph (bool, optional): If ``False``, the underlying graph is destroyed and the
            executable graph is instantiated on the GPU at the end of ``capture_end``.
            If ``True``, the underlying graph is preserved after ``capture_end``. In this case,
            the executable graph is not instantiated automatically; it must be explicitly created
            by calling ``instantiate``, or it will be instantiated on the first call to ``replay``.
            Defaults to ``False``.
        pool (tuple[int, int], optional): Memory pool identifier for this graph. Multiple graphs
            can share the same pool by passing the same identifier, which can reduce memory overhead.
            Defaults to ``None``.
        capture_error_mode (Literal["default", "global", "thread_local", "relaxed"], optional):
            Specifies the behavior of graph capture. The exact semantics are backend-specific.
            ``"default"``: backend-defined default capture behavior.
            ``"global"``: potentially unsafe API calls are prohibited. Errors may occur if capture
            in the current thread affects other threads.
            ``"thread_local"``: potentially unsafe API calls are prohibited. Errors occur only if
            capture in the current thread affects itself.
            ``"relaxed"``: the current thread is allowed to make potentially unsafe API calls, except
            for calls that inherently conflict with stream capture.
            Default: ``"default"``.

    Example::

        >>> # xdoctest: +SKIP
        >>> x = torch.zeros([2000], device=0)

        >>> stream = torch.Stream()
        >>> graph = torch.accelerator.Graph()
        >>> with stream, graph:
        ...     x += 1

        >>> graph.replay()
    """

    def __new__(
        cls,
        keep_graph: bool = False,
        *,
        pool: tuple[int, int] | None = None,
        capture_error_mode: Literal[
            "default", "global", "thread_local", "relaxed"
        ] = "default",
    ) -> Self:
        # [FUNCTION DOC] HELPER: Allocate the C++ object, passing only keep_graph to the base constructor; pool and capture_error_mode are accepted but handled in __init__.
        return super().__new__(cls, keep_graph)

    def __init__(
        self,
        keep_graph: bool = False,
        *,
        pool: tuple[int, int] | None = None,
        capture_error_mode: Literal[
            "default", "global", "thread_local", "relaxed"
        ] = "default",
    ) -> None:
        # [FUNCTION DOC] Initialize the base with keep_graph and record pool as graph_pool and capture_error_mode as attributes for later captures.
        super().__init__(keep_graph)
        self.graph_pool = pool
        self.capture_error_mode = capture_error_mode

    # pyrefly: ignore [bad-override]
    def capture_begin(self) -> None:
        # [FUNCTION DOC] Begin recording operations on the current stream into the graph, passing the stored pool and capture error mode to the base implementation.
        r"""
        Begin graph capture on the current stream.

        All operations on the current stream after this call will be recorded into the graph until
        ``capture_end`` is called, using the memory pool and capture error mode provided at construction time.
        """
        super().capture_begin(
            pool=self.graph_pool, capture_error_mode=self.capture_error_mode
        )

    def capture_end(self) -> None:
        # [FUNCTION DOC] End capture on the current stream/device; afterwards the graph can be replayed. Forwards to the base.
        r"""
        End graph capture on the current stream of the current device.

        After this call, the graph can be replayed via ``replay``.
        """
        super().capture_end()

    def instantiate(self) -> None:
        # [FUNCTION DOC] Build the executable graph from the captured one. Called automatically by capture_end (keep_graph false) or by replay (keep_graph true and not yet instantiated).
        r"""
        Instantiate the underlying graph. Will be called by ``capture_end``
        if ``keep_graph=False``, or by ``replay`` if ``keep_graph=True`` and
        ``instantiate`` has not already been explicitly called.
        """
        super().instantiate()

    def replay(self) -> None:
        # [FUNCTION DOC] Replay the captured work; forwards to the base.
        r"""Replay the work captured by this graph."""
        super().replay()

    def reset(self) -> None:
        # [FUNCTION DOC] Discard the held graph so the instance can be recaptured; graph_pool/capture_error_mode may be changed beforehand.
        r"""
        Delete the graph currently held by this instance.

        After this call, the graph can be recaptured. Set :attr:`graph_pool` or
        :attr:`capture_error_mode` beforehand to use different settings on the next capture.
        """
        super().reset()

    def pool(self) -> tuple[int, int]:
        # [FUNCTION DOC] Return an opaque (int, int) memory pool id for this graph, usable as the pool argument of another graph to share memory.
        r"""
        Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.

        Example::
            >>> # xdoctest: +SKIP
            >>> g1 = torch.accelerator.Graph()
            >>> g1.capture_begin()
            >>> # ... operations ...
            >>> g1.capture_end()

            >>> # Share g1's memory pool with a new graph
            >>> pool_id = g1.pool()
            >>> g2 = torch.accelerator.Graph(pool=pool_id)
        """
        return super().pool()

    def enable_debug_mode(self) -> None:
        # [FUNCTION DOC] Enable debug mode, required before debug_dump. Forwards to the base.
        r"""Enable debugging mode for ``debug_dump``."""
        return super().enable_debug_mode()

    def debug_dump(self, path: str) -> None:
        # [FUNCTION DOC] Write the captured graph to the file at path (debug mode must be enabled first). Forwards to the base.
        r"""
        Dump the captured graph to a file for debugging purposes if the debugging is
        enabled via ``enable_debug_mode``.

        Arguments:
            path (str): Path to dump the graph to.

        Example::
            >>> # xdoctest: +SKIP
            >>> s = torch.Stream()
            >>> g = torch.accelerator.Graph()
            >>> g.enable_debug_mode()

            >>> with s, g:
            >>> # ... operations ...

            >>> # Dump captured graph to a file "graph_dump.dot"
            >>> g.debug_dump("graph_dump.dot")
        """
        return super().debug_dump(path)

    def __enter__(self) -> None:
        # [FUNCTION DOC] Context entry: synchronize the accelerator, optionally run garbage collection (only when the compiler config flag force_cudagraph_gc is set), empty the device and host caches, then begin capture. Returns None.
        torch.accelerator.synchronize()
        if torch.compiler.config.force_cudagraph_gc:
            # We previously always ran garbage collection here. While this can help
            # reclaim accelerator device memory held by dead Python cycles, it is
            # very expensive, especially when performing multiple graph captures in sequence.
            gc.collect()
        torch.accelerator.empty_cache()
        torch.accelerator.empty_host_cache()
        self.capture_begin()

    def __exit__(self, *exc_info: object) -> None:
        # [FUNCTION DOC] Context exit: end capture. Does not suppress exceptions.
        self.capture_end()


__all__ = ["Graph"]
