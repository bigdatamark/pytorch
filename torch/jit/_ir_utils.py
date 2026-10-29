# [MODULE DOC]
# Helpers for manipulating TorchScript IR graphs from Python. Provides a context
# manager that temporarily changes a graph's insertion point. No import-time
# side effects.
from types import TracebackType

import torch


class _InsertPoint:
    # [CLASS DOC] HELPER: Context manager that sets a graph's insertion point to
    # a given node or block on entry and restores the previous insertion point
    # on exit. State: the graph, the target insert point, the saved previous
    # insert point (set on entry), and an unused guard attribute (always None).
    def __init__(
        self,
        insert_point_graph: torch._C.Graph,
        insert_point: torch._C.Node | torch._C.Block,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the graph and the target insert point
        # (node or block) and initializes the unused guard attribute to None.
        self.insert_point = insert_point
        self.g = insert_point_graph
        self.guard = None

    def __enter__(self) -> None:
        # [FUNCTION DOC] HELPER: Saves the graph's current insertion point and
        # sets the graph's insertion point to the target. Mutates the graph.
        self.prev_insert_point = self.g.insertPoint()
        self.g.setInsertPoint(self.insert_point)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Restores the insertion point saved on entry.
        # Exception details are ignored and exceptions are not suppressed.
        self.g.setInsertPoint(self.prev_insert_point)


def insert_point_guard(
    self: torch._C.Graph, insert_point: torch._C.Node | torch._C.Block
) -> _InsertPoint:
    # [FUNCTION DOC] Returns a context manager that sets the graph's insertion
    # point to the given node or block while active and restores the previous
    # one on exit. Intended to be attached as a method of the graph class (hence
    # the self parameter name).
    return _InsertPoint(self, insert_point)
