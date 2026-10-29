# [MODULE DOC] Defines the base interfaces for FX passes: PassResult (named tuple of resulting graph
# module and a modified flag) and PassBase (abstract callable pass with optional pre/post checks).
# No import-time side effects.
import abc
from collections import namedtuple

import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule


__all__ = ["PassResult", "PassBase"]


@compatibility(is_backward_compatible=False)
# pyrefly: ignore [invalid-inheritance]
class PassResult(namedtuple("PassResult", ["graph_module", "modified"])):
    # [CLASS DOC] Immutable pair (graph_module, modified) returned by passes; no extra slots or state.
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """

    __slots__: tuple[str, ...] = ()

    def __new__(cls, graph_module: nn.Module, modified: bool) -> "PassResult":
        # [FUNCTION DOC] Creates the result tuple from the module and the modified flag; no validation.
        return super().__new__(cls, graph_module, modified)


@compatibility(is_backward_compatible=False)
class PassBase(abc.ABC):
    # [CLASS DOC] Abstract base for passes. Subclasses must implement call; requires and ensures are
    # optional hooks (no-ops by default) for precondition and postcondition checks. Instances are callable.
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult | None:
        # [FUNCTION DOC] Runs requires, then call, then ensures on the given module and returns the
        # result of call (a PassResult or None). Exceptions from any step propagate. Note ensures
        # receives the input module, not the module in the returned result.
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """

        self.requires(graph_module)
        res = self.call(graph_module)
        self.ensures(graph_module)
        return res

    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> PassResult | None:
        # [FUNCTION DOC] Abstract: subclasses implement the transformation or analysis on the module
        # and return a PassResult (or None when nothing is reported).
        """
        The pass that is run through the given graph module. To implement a
        pass, it is required to implement this function.

        Args:
            graph_module: The graph module we will run a pass on
        """

    def requires(self, graph_module: GraphModule) -> None:  # noqa: B027
        # [FUNCTION DOC] Optional precondition hook run before the pass; default does nothing.
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """

    def ensures(self, graph_module: GraphModule) -> None:  # noqa: B027
        # [FUNCTION DOC] Optional postcondition hook run after the pass; default does nothing.
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
