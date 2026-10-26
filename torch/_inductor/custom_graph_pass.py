# [MODULE DOC] Abstract interfaces for user supplied Inductor custom passes (graph, graph module, scheduler,
# partitioner, runtime estimator, knapsack solver) with a uuid used for cache keys, plus type aliases and helpers.
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, TYPE_CHECKING, TypeAlias

import torch.fx.graph


if TYPE_CHECKING:
    from torch._functorch.partitioners import NodeInfo
    from torch._inductor.scheduler import BaseSchedulerNode


class CustomPassBase(ABC):
    # [CLASS DOC] Abstract base for custom passes. Subclasses implement uuid, returning a picklable identifier included in the
    # compiled graph cache key, or None to disable caching.
    """
    Implement this interface for custom passes:

    The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom pass's behavior depends on some external state, then
    you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def uuid(self) -> Any | None:
        # [FUNCTION DOC] Abstract: return an identifier uniquely identifying the pass implementation, or None to skip caching.
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


class CustomGraphPass(CustomPassBase):
    # [CLASS DOC] Abstract custom pass operating on an fx Graph in place; call takes the graph and returns None.
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        # [FUNCTION DOC] Abstract: apply the custom pass to the graph in place.
        """
        Implementation of the custom pass.
        """


class CustomGraphModulePass(CustomPassBase):
    # [CLASS DOC] Abstract custom pass operating on a GraphModule in place.
    """
    Implement this interface for custom passes that operate on GraphModule:

    1) The __call__() method contains the implementation of the custom pass.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.
    """

    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] Abstract: apply the custom pass to the graph module in place.
        """
        Implementation of the custom pass.
        """


class CustomSchedulerPass(CustomPassBase):
    # [CLASS DOC] Abstract custom pass over scheduler nodes; takes and returns a list of scheduler nodes.
    """
    Implement this interface for custom Scheduler passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.
    """

    @abstractmethod
    def __call__(self, nodes: list["BaseSchedulerNode"]) -> list["BaseSchedulerNode"]:
        # [FUNCTION DOC] Abstract: transform the list of scheduler nodes and return the new list.
        """
        Implementation of the custom pass.
        """


class CustomInferenceAwareGraphPass(CustomGraphPass):
    # [CLASS DOC] Abstract graph pass whose call also receives an is_inference flag.
    """
    Implement this interface for custom inference aware Graph passes.
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph, is_inference: bool) -> None:
        # [FUNCTION DOC] Abstract: apply the pass to the graph in place, knowing whether it is an inference graph.
        """
        Implementation of the custom pass.
        """


CustomGraphPassCallable: TypeAlias = (
    CustomGraphPass | Callable[[torch.fx.graph.Graph], None]
)
CustomGraphPassType: TypeAlias = (
    CustomGraphPassCallable
    | list[CustomGraphPassCallable]
    | tuple[CustomGraphPassCallable, ...]
    | None
)

CustomSchedulerPassCallable: TypeAlias = (
    CustomSchedulerPass
    | Callable[[list["BaseSchedulerNode"]], list["BaseSchedulerNode"]]
)


def get_custom_graph_passes(
    custom_pass: CustomGraphPassType,
) -> tuple[CustomGraphPassCallable, ...]:
    # [FUNCTION DOC] Normalize a custom pass setting into a tuple of passes: None gives empty, a list or tuple is converted
    # to a tuple, a single pass is wrapped in a one-element tuple.
    if custom_pass is None:
        return ()
    if isinstance(custom_pass, (list, tuple)):
        return tuple(custom_pass)
    return (custom_pass,)


@lru_cache(1)
def get_hash_for_files(paths: tuple[str, ...], extra: str = "") -> bytes:
    # [FUNCTION DOC] Compute a sha256 digest over the extra string and the bytes of each listed file, in order. Returns bytes.
    # Raises OSError if a file cannot be read. Result is memoized with a cache of size one.
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    """
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.digest()


class CustomPartitionerFn(CustomPassBase):
    # [CLASS DOC] Abstract custom partitioner: call takes a graph module and joint inputs (plus keyword args) and returns
    # a pair of graph modules (forward, backward); uuid identifies it for caching.
    """
    Implement this interface for custom partitioner:

    1) The __call__() method contains the implementation of the custom partitioner.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    from torch._inductor.custom_graph_pass import get_hash_for_files

    class MyCustomPartitionerFn(CustomPartitionerFn):
        def __call__(
            self,
            gm: torch.fx.GraphModule,
            joint_inputs: Sequence[object],
            **kwargs: Any
        ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
            # my custom partitioner implementation
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        # [FUNCTION DOC] Abstract: partition the joint graph module and return the two resulting graph modules.
        """
        Implementation of the custom partitioner.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        # [FUNCTION DOC] Abstract: return an identifier for this partitioner, or None to skip caching.
        """
        Return an ID to uniquely identify your custom partitioner implementation.
        Return None to skip inductor code caching entirely.
        """


CustomPartitionerFnType: TypeAlias = CustomPartitionerFn | None


class CustomRuntimeEstimator(CustomPassBase):
    # [CLASS DOC] Abstract custom runtime estimator: call maps an fx node to an estimated runtime as a float.
    """
    Implement this interface for custom runtime estimators:

    1) The __call__() method contains the implementation of the runtime estimation.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    class MyCustomRuntimeEstimator(CustomRuntimeEstimator):
        def __call__(self, node: fx.Node) -> float:
            # my custom runtime estimation logic
            return estimated_runtime

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))
    """

    @abstractmethod
    def __call__(self, node: "torch.fx.Node") -> float:
        # [FUNCTION DOC] Abstract: return the estimated runtime of the given node.
        """
        Implementation of the custom runtime estimator.

        Args:
            node: An fx.Node object whose runtime is to be estimated.

        Returns:
            float: The estimated runtime for the node.
        """


class CustomKnapsackSolver(CustomPassBase):
    # [CLASS DOC] Abstract custom knapsack solver for the partitioner's recompute decision; call returns two index lists
    # (saved nodes, recomputed nodes).
    """
    Implement this interface for custom knapsack solvers:

    1) The __call__() method contains the implementation of the knapsack solver.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    class MyCustomKnapsackSolver(CustomKnapsackSolver):
        def __call__(
            self,
            memory: list[float],
            joint_graph: fx.Graph,
            max_memory: float,
            node_info: NodeInfo,
            all_recomputable_banned_nodes: list[fx.Node],
        ) -> tuple[list[int], list[int]]:
            # my custom knapsack solver logic
            return saved_node_idx, recomp_node_idx

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))
    """

    @abstractmethod
    def __call__(
        self,
        memory: list[float],
        joint_graph: "torch.fx.Graph",
        max_memory: float,
        node_info: "NodeInfo",
        all_recomputable_banned_nodes: list["torch.fx.Node"],
    ) -> tuple[list[int], list[int]]:
        # [FUNCTION DOC] Abstract: given memory costs, joint graph, memory budget, node info and banned recomputable nodes, return
        # the indices of saved nodes and of recomputed nodes.
        """
        Implementation of the custom knapsack solver.
        """
