# [MODULE DOC] Provenance and metadata tracking for FX tracing. Holds process-global
#   tracing state (current_meta, the replay node, should-preserve flags, a thread-local
#   regional inductor subgraph name, and a registry of per-module FX metadata for the
#   memory profiler), the NodeSource/NodeSourceAction provenance records, context
#   managers and helpers that set or restore that state (preserve_node_meta, annotate,
#   set_current_meta, set_current_replay_node, seq_nr helpers), and utilities that
#   extract provenance, custom annotations and seq_nr groupings from graphs. Import-time
#   side effects: creates the module-level globals only.
import copy
import logging
import threading
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional, ParamSpec, TypeVar, Union

from torch._utils_internal import signpost_event

from ._compatibility import compatibility
from .graph import Graph
from .graph_module import GraphModule
from .node import Node


_P = ParamSpec("_P")
_R = TypeVar("_R")

log = logging.getLogger(__name__)

__all__ = [
    "annotate",
    "annotate_fn",
    "preserve_node_meta",
    "has_preserved_node_meta",
    "set_stack_trace",
    "set_grad_fn_seq_nr",
    "reset_grad_fn_seq_nr",
    "format_stack",
    "set_current_meta",
    "get_current_meta",
    "NodeSource",
    "NodeSourceAction",
    "get_graph_provenance_json",
    "set_current_replay_node",
    "get_current_replay_node",
]

current_meta: dict[str, Any] = {}
current_replay_node: Node | None = None
# Thread-local holder for the optional name of the subgraph being compiled by a
# nested/regional inductor compile (e.g. a regional inductor scooped-out region
# or an invoke_subgraph submodule). Used to disambiguate trace_structured
# records emitted from sub-compiles within a single frame. Used for debugging
# only!
_regional_inductor_subgraph_name_tls = threading.local()
# Preserve the node meta fields in torch.fx.proxy._COPY_META_FIELDS
should_preserve_node_meta = False
# Preserve the "seq_nr" node meta field
_should_preserve_node_meta = False

GRADIENT_ACC_SPECIAL_STACK = (
    "Gradient addition node due to multiple use of tensor around:"
)
# =============================================================================
# FX Metadata Registry for Memory Profiler
# =============================================================================
# Global in-memory registry for FX metadata
# Maps module_name -> metadata dict containing lineno_map and node_metadata
_FX_METADATA_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_fx_metadata(module_name: str, metadata: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Store the given metadata mapping in the global in-memory
    #   registry under module_name, replacing any previous entry. Mutates global state.
    """
    Register FX metadata in the global in-memory registry.

    This is called automatically during graph module compilation to store metadata
    for later use by memory profiler augmentation.

    Args:
        module_name: The module identifier (content-addressed filename)
        metadata: Metadata dict containing lineno_map, node_metadata, and source_code
    """
    # TODO: add logging to tlparse
    _FX_METADATA_REGISTRY[module_name] = metadata


@compatibility(is_backward_compatible=False)
class NodeSourceAction(Enum):
    # [CLASS DOC] Enum of actions that can produce a node in provenance tracking: CREATE
    #   and REPLACE (string values create and replace).
    """Enum representing the action taken to produce a node in provenance tracking."""

    CREATE = "create"
    REPLACE = "replace"


@compatibility(is_backward_compatible=False)
class NodeSource:
    # [CLASS DOC] Provenance record for an FX node. Attributes: pass_name, a list of
    #   NodeSourceAction, from_node (list of earlier NodeSource records, copied from the
    #   node meta), node_info (name, target string and graph id of the source node, or
    #   None), and two private caches for the action string and the dictionary form.
    #   Supports readable printing, dictionary conversion/deserialization, and value
    #   equality/hashing based on the dictionary form.
    # KNOWN ISSUE: the caches are never invalidated, so changes to attributes after the
    #   first to_dict call are not reflected in equality or hashing.
    """
    NodeSource is a data structure that contains the provenance information of a node.
    If node `a` is created from node `b`, then `a.meta["from_node"]` may contain NodeSource(b).
    """

    class NodeInfo:
        # [CLASS DOC] HELPER: Plain holder of a source node's name, target string and
        #   graph id.
        def __init__(self, name: str, target: str, graph_id: int) -> None:
            # [FUNCTION DOC] HELPER: Store name, target and graph_id as attributes.
            self.name = name
            self.target = target
            self.graph_id = graph_id

    pass_name: str
    action: list["NodeSourceAction"]
    from_node: list["NodeSource"]
    node_info: Optional["NodeInfo"]
    _dict: dict[str, Any] | None
    _action_string: str | None

    def __init__(
        self,
        node: Node | None,
        pass_name: str = "",
        action: Union["NodeSourceAction", list["NodeSourceAction"]] | None = None,
    ) -> None:
        # [FUNCTION DOC] Build a provenance record. node may be None. action may be
        #   None, a single NodeSourceAction or a list of them; it is normalized to a
        #   list and every element is checked, raising AssertionError for a wrong type.
        #   If node is given, record its name, string form of its target and the id of
        #   its graph, and deep-copy its existing from_node history from node.meta
        #   (empty if absent); otherwise node_info is None and history is empty.
        #   Initializes the caches to empty.
        self.pass_name = pass_name

        if action is None:
            action = []
        elif not isinstance(action, list):
            action = [action]
        for a in action:
            if not isinstance(a, NodeSourceAction):
                raise AssertionError(f"Expected NodeSourceAction, got {type(a)}")
        self.action = action
        if node:
            self.node_info = self.NodeInfo(
                name=node.name, target=str(node.target), graph_id=id(node.graph)
            )
            self.from_node = (
                copy.deepcopy(node.meta["from_node"])
                if "from_node" in node.meta
                else []
            )
        else:
            self.node_info = None
            self.from_node = []

        # cache the action string and dict representation for performance.
        self._action_string: str | None = None
        self._dict: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Property: the source node name, or an empty string when there
        #   is no node_info.
        return self.node_info.name if self.node_info else ""

    @property
    def target(self) -> str:
        # [FUNCTION DOC] Property: the source node target string, or an empty string
        #   when there is no node_info.
        return self.node_info.target if self.node_info else ""

    @property
    def graph_id(self) -> int:
        # [FUNCTION DOC] Property: the source graph id, or -1 when there is no
        #   node_info.
        return self.node_info.graph_id if self.node_info else -1

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return the multi-line readable form from print_readable.
        return self.print_readable()

    def _get_action_string(self) -> str:
        # [FUNCTION DOC] HELPER: Return (and cache) the actions joined by a plus sign in
        #   lower case, e.g. create+replace.
        if self._action_string is None:
            self._action_string = "+".join([a.name.lower() for a in self.action])
        return self._action_string

    def print_readable(self, indent: int = 0) -> str:
        # [FUNCTION DOC] Return a text rendering of this record and its history: one
        #   line per record showing name, pass_name, action string and graph_id,
        #   indented four spaces per nesting level, recursing into from_node. Returns an
        #   empty string when indent exceeds 9 (depth cutoff).
        if indent > 9:
            return ""
        result = ""
        action_string = self._get_action_string()
        result += (
            " " * indent * 4
            + f"(name={self.name}, pass_name={self.pass_name}, action={action_string}, graph_id={self.graph_id})\n"
        )
        for item in self.from_node:
            result += item.print_readable(indent + 1)
        return result

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return (and cache) a dictionary with name, target, graph_id,
        #   pass_name, action string and from_node (list of recursively converted
        #   dictionaries). The same cached dictionary object is returned on later calls.
        if self._dict is None:
            # Convert the object to a dictionary
            action_string = self._get_action_string()
            self._dict = {
                "name": self.name,
                "target": self.target,
                "graph_id": self.graph_id,
                "pass_name": self.pass_name,
                "action": action_string,
                "from_node": [node.to_dict() for node in self.from_node],
            }

        if self._dict is None:
            raise AssertionError("_dict is None after initialization")
        return self._dict

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal only to another NodeSource whose dictionary form is
        #   equal.
        if not isinstance(other, NodeSource):
            return False
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash derived from the dictionary form, converted recursively
        #   into nested tuples (dictionaries become sorted tuples of pairs, lists become
        #   tuples).
        # Create a hash based on the dictionary representation
        # We need to convert the dict to a hashable form
        def _make_hashable(obj: Any) -> Any:
            # [FUNCTION DOC] HELPER: Recursively convert dictionaries and lists into
            #   hashable sorted tuples, leaving other values unchanged.
            if isinstance(obj, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(_make_hashable(item) for item in obj)
            else:
                return obj

        return hash(_make_hashable(self.to_dict()))

    @classmethod
    def _from_dict(cls, d: dict[str, Any] | None) -> Optional["NodeSource"]:
        # [FUNCTION DOC] HELPER: Rebuild a NodeSource from its dictionary form without
        #   calling the constructor. Returns None for None input; raises AssertionError
        #   if the input is not a mapping. Restores pass_name, parses the action string
        #   into enum values (unknown action names are silently dropped), creates
        #   node_info only if name, target and graph_id keys are all present, and
        #   recursively rebuilds from_node entries, skipping those that deserialize to
        #   None.
        """
        Recursively deserialize from_node metadata from dictionary data.
        It is used to deserialize the from_node field from serialized metadata.
        Please use constructor NodeSource(node, ...) to create a NodeSource object.
        """
        if d is None:
            return None

        if not isinstance(d, dict):
            raise AssertionError(f"Expected a dict, got {type(d)}")

        # Create a NodeSource object directly without going through the constructor
        # to avoid issues with graph ID and node creation
        node_source = NodeSource.__new__(NodeSource)

        # Reset the cached properties
        node_source._action_string = None
        node_source._dict = None

        # Set the basic attributes
        node_source.pass_name = d.get("pass_name", "")

        # Parse action string back to NodeSourceAction enum list
        action_str = d.get("action", "")
        actions = []
        if action_str:
            for action_name in action_str.split("+"):
                if action_name.upper() == "CREATE":
                    actions.append(NodeSourceAction.CREATE)
                elif action_name.upper() == "REPLACE":
                    actions.append(NodeSourceAction.REPLACE)
        node_source.action = actions

        # Create the NodeInfo object directly
        if "name" in d and "target" in d and "graph_id" in d:
            node_info = NodeSource.NodeInfo(
                d.get("name", ""), d.get("target", ""), d.get("graph_id", -1)
            )
            node_source.node_info = node_info
        else:
            node_source.node_info = None

        # Recursively deserialize nested from_node
        if d.get("from_node", None) is not None:
            node_source.from_node = [
                result
                for fn in d.get("from_node", [])
                if (result := cls._from_dict(fn)) is not None
            ]
        else:
            node_source.from_node = []
        return node_source


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta(enable: bool = True) -> Iterator[None]:
    # [FUNCTION DOC] Context manager (decorated with compatibility): sets the global
    #   should_preserve_node_meta flag to enable for the body, after saving the flag and
    #   a shallow copy of current_meta. On exit, in a finally block, restores the flag
    #   and replaces current_meta with the saved copy.
    global should_preserve_node_meta
    global current_meta
    saved_should_preserve_node_meta = should_preserve_node_meta
    # Shallow copy is OK since fields of current_meta are not mutated
    saved_current_meta = current_meta.copy()
    try:
        should_preserve_node_meta = enable
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta
        current_meta = saved_current_meta


@contextmanager
def _preserve_node_seq_nr(preserve_seq_nr: bool = True) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that sets the private global flag
    #   controlling preservation of seq_nr node meta to preserve_seq_nr for the body and
    #   restores the previous value on exit.
    """
    Temporarily enables or disables the preservation of node.meta["seq_nr"] in the
    tracing context.
    """
    global _should_preserve_node_meta
    saved = _should_preserve_node_meta

    try:
        _should_preserve_node_meta = preserve_seq_nr
        yield
    finally:
        _should_preserve_node_meta = saved


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: list[str]) -> None:
    # [FUNCTION DOC] If node meta preservation is on: set current_meta stack_trace to
    #   the joined strings of stack when stack is non-empty, otherwise remove the
    #   stack_trace key. Does nothing when preservation is off. Mutates global
    #   current_meta.
    global current_meta

    if should_preserve_node_meta:
        if stack:
            current_meta["stack_trace"] = "".join(stack)
        else:
            # when the stack is empty, we explicitly clear the stack_trace to avoid
            # propagating it to future node.
            current_meta.pop("stack_trace", None)


@compatibility(is_backward_compatible=False)
@contextmanager
def annotate(annotation_dict: dict[str, Any]) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that merges annotation_dict into the custom entry
    #   of the global current_meta for the body, creating it if needed, so nodes created
    #   inside carry the annotations. On exit (even on error) it restores the previous
    #   custom dictionary, or removes the entry if there was none. Not usable with
    #   symbolic_trace or jit.trace; meant for PT2 tracers.
    """
    Temporarily adds custom annotations to the current tracing context.
    The fx_node produced from this tracing context will have the
    custom annotations in node.metadata["custom"] field.

    This context manager allows you to insert arbitrary metadata into the PT2
    tracing system by updating the global `current_meta["custom"]` dictionary.
    The annotations are automatically reverted after the context exits.

    Gradient accumulation nodes will not be annotated.

    This is intended for advanced users who need to attach additional metadata to the fx nodes
    (e.g., for debugging, analysis, or external tooling) during export tracing.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        annotation_dict (dict): A dictionary of custom key-value pairs to inject
            into the FX trace metadata.

    Example:
        After exiting the context, custom annotations are removed.

        >>> with annotate({"source": "custom_pass", "tag": 42}):
        ...     pass  # Your computation here
    """

    global current_meta

    has_custom = "custom" in current_meta
    old_custom = copy.copy(current_meta.get("custom", {}))

    try:
        if not has_custom:
            current_meta["custom"] = dict[str, Any]()

        # Update with all key-value pairs from the input dict
        current_meta["custom"].update(annotation_dict)
        yield
    finally:
        if has_custom:
            # Restore the original custom dict
            current_meta["custom"] = old_custom
        else:
            del current_meta["custom"]


# Key under node.meta["custom"] used to carry the activation-checkpointing budget
# set by ``torch.autograd.graph.region_activation_memory_budget``. Shared by that
# writer and the reader below so the partitioner / AOTAutograd cache read it the
# same way.
MEMORY_BUDGET_ANNOTATION_KEY = "_region_activation_memory_budget"


def _get_memory_budget_annotation(node: Node) -> float | None:
    # [FUNCTION DOC] HELPER: Return the activation memory budget stored under the budget
    #   annotation key in the node's custom meta, or None if custom is missing or not a
    #   mapping or lacks the key.
    """
    Read the ``region_activation_memory_budget`` annotation off an FX node,
    returning ``None`` if absent. The value is written only by
    ``torch.autograd.graph.region_activation_memory_budget``, which validates it
    and stores a ``float``.
    """
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    return custom.get(MEMORY_BUDGET_ANNOTATION_KEY)


@compatibility(is_backward_compatible=False)
def annotate_fn(
    annotation_dict: dict[str, Any],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # [FUNCTION DOC] Decorator factory: returns a decorator that wraps a function so
    #   each call runs inside the annotate context manager with annotation_dict. Imports
    #   the stack-trace anchor registrar lazily.
    """
    A decorator that wraps a function with the annotate context manager.
    Use this when you want to annotate an entire function instead of a specific code block.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        annotation_dict (dict): A dictionary of custom key-value pairs to inject
            into the FX trace metadata for all operations in the function.

    Example:
        All operations in my_function will have {"pp_stage": 1} in their metadata.

        >>> @annotate_fn({"pp_stage": 1})
        ... def my_function(x):
        ...     return x + 1
    """
    from functools import wraps

    from torch.fx.proxy import _register_stack_trace_anchor

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        # [FUNCTION DOC] HELPER: Decorator body. Registers func as a stack-trace anchor
        #   (so make_fx keeps its frame in stack_trace meta) and returns a wrapper
        #   preserving func metadata.
        # Register the wrapped function as a stack-trace anchor so make_fx
        # preserves its frame in node.meta["stack_trace"]. The user
        # explicitly opted into annotating this function, so its frame is a
        # meaningful attribution point even though it isn't an
        # ``nn.Module.forward``.
        _register_stack_trace_anchor(func)

        @wraps(func)
        # NB: Do not annotate with _P.args/_P.kwargs here. Dynamo guards on
        # the identity of ParamSpec annotation objects, causing guard failures.
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Call func with all arguments inside
            #   annotate(annotation_dict) and return its result.
            with annotate(annotation_dict):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def _set_autograd_backward(enable: bool = True) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that marks current_meta as being in
    #   autograd backward when enable is true, and on exit restores the earlier
    #   autograd_backward value if the key existed, otherwise removes it.
    # KNOWN ISSUE: when enable is false and the key was absent, exit still calls the
    #   reset helper, harmless; and when enable is false and the key existed the old
    #   value is restored.
    global current_meta

    had_autograd_backward = "autograd_backward" in current_meta
    old_autograd_backward = current_meta.get("autograd_backward", False)

    if enable:
        _mark_autograd_backward()
    try:
        yield
    finally:
        if had_autograd_backward:
            current_meta["autograd_backward"] = old_autograd_backward
        else:
            _reset_autograd_backward()


@compatibility(is_backward_compatible=False)
def _mark_autograd_backward() -> None:
    # [FUNCTION DOC] HELPER: Set the autograd_backward key in current_meta to true.
    global current_meta

    current_meta["autograd_backward"] = True


@compatibility(is_backward_compatible=False)
def _reset_autograd_backward() -> None:
    # [FUNCTION DOC] HELPER: Remove the autograd_backward key from current_meta if
    #   present.
    global current_meta

    current_meta.pop("autograd_backward", None)


@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr: int) -> None:
    # [FUNCTION DOC] If node meta preservation is on: append seq_nr to the
    #   grad_fn_seq_nr list in current_meta and increment the in_grad_fn nesting
    #   counter. Otherwise does nothing.
    global current_meta

    if should_preserve_node_meta:
        # The seq_nr is captured by eager mode in the grad_fn during forward
        current_meta["grad_fn_seq_nr"] = current_meta.get("grad_fn_seq_nr", []) + [
            seq_nr
        ]
        current_meta["in_grad_fn"] = current_meta.get("in_grad_fn", 0) + 1


@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr() -> None:
    # [FUNCTION DOC] If node meta preservation is on: undo one set_grad_fn_seq_nr.
    #   Raises AssertionError if the nesting counter is not positive. At level one,
    #   deletes both in_grad_fn and grad_fn_seq_nr; otherwise decrements the counter and
    #   drops the last seq_nr.
    # NB: reset state properly, this would be helpful towards supporting
    #     reentrant autograd if we actually wanted to do that.
    global current_meta
    if should_preserve_node_meta:
        current_level = current_meta.get("in_grad_fn", 0)
        if current_level <= 0:
            raise AssertionError(f"Expected current_level > 0, got {current_level}")
        if current_level == 1:
            del current_meta["in_grad_fn"]
            del current_meta["grad_fn_seq_nr"]
        else:
            current_meta["in_grad_fn"] = current_level - 1
            current_meta["grad_fn_seq_nr"] = current_meta["grad_fn_seq_nr"][:-1]


@compatibility(is_backward_compatible=False)
def format_stack() -> list[str]:
    # [FUNCTION DOC] Return a list of strings describing the call stack: when meta
    #   preservation is on, a one-element list holding current_meta stack_trace (or
    #   empty string); otherwise the formatted Python stack minus the current frame.
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    # [FUNCTION DOC] Return the global should_preserve_node_meta flag.
    return should_preserve_node_meta


def _is_preserving_node_seq_nr() -> bool:
    # [FUNCTION DOC] HELPER: Return the private flag controlling seq_nr preservation.
    return _should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node: Node, pass_name: str = "") -> Iterator[None]:
    # [FUNCTION DOC] Context manager: if preservation is on and node.meta is non-empty,
    #   temporarily replace global current_meta with a copy of node.meta whose from_node
    #   is a single new NodeSource (action CREATE, with pass_name) built from node, then
    #   restore the old current_meta on exit. Otherwise just yields with no change.
    global current_meta
    if should_preserve_node_meta and node.meta:
        saved_meta = current_meta
        try:
            current_meta = node.meta.copy()

            # Update the "from_node" field in current_meta for provenance tracking.
            # Instead of appending, overwrite the "from_node" field because current_meta
            # will be assigned to the new node. The new NodeSource(node, ...) will
            # include the information from the previous current_meta["from_node"].
            current_meta["from_node"] = [
                NodeSource(node, pass_name, NodeSourceAction.CREATE)
            ]
            yield
        finally:
            current_meta = saved_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> dict[str, Any]:
    # [FUNCTION DOC] Return the global current_meta dictionary itself (not a copy).
    return current_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_replay_node(node: Node | None) -> Iterator[None]:
    # [FUNCTION DOC] Context manager setting the global current_replay_node to node for
    #   the body (a non-None value means the node is being regenerated under
    #   FunctionalTensorMode) and restoring the previous value on exit.
    """
    Set the current replay node. If `current_replay_node` is not None,
    then we're re-generating the `current_replay_node` in FunctionalTensorMode.
    """
    # See [Note] annotation for more details.
    global current_replay_node
    saved_current_replay_node = current_replay_node
    try:
        current_replay_node = node
        yield
    finally:
        current_replay_node = saved_current_replay_node


@compatibility(is_backward_compatible=False)
def get_current_replay_node() -> Node | None:
    # [FUNCTION DOC] Return the global current replay node, or None.
    """
    Get the current replay node
    """
    return current_replay_node


@compatibility(is_backward_compatible=False)
@contextmanager
def _set_regional_inductor_subgraph_name(name: str | None) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager setting the thread-local regional inductor
    #   subgraph name to name for the body and restoring the previous value (default
    #   None) on exit. Used to tag structured trace records.
    """
    Set the name of the subgraph currently being sub-compiled by regional
    inductor (a scooped-out region or an invoke_subgraph submodule). Used to
    tag trace_structured records so artifacts emitted from different
    sub-compiles within the same frame can be distinguished.
    """
    saved = getattr(_regional_inductor_subgraph_name_tls, "value", None)
    try:
        _regional_inductor_subgraph_name_tls.value = name
        yield
    finally:
        _regional_inductor_subgraph_name_tls.value = saved


@compatibility(is_backward_compatible=False)
def _get_regional_inductor_subgraph_name() -> str | None:
    # [FUNCTION DOC] HELPER: Return the thread-local regional inductor subgraph name, or
    #   None if unset.
    """
    Get the name of the subgraph currently being sub-compiled by regional
    inductor, or None.
    """
    return getattr(_regional_inductor_subgraph_name_tls, "value", None)


@compatibility(is_backward_compatible=False)
def get_graph_provenance_json(graph: Graph) -> dict[str, Any]:
    # [FUNCTION DOC] Return a mapping from the name of each call_function node of graph
    #   to a list of the dictionary forms of its from_node history (empty list when
    #   absent). Any exception is swallowed: a signpost event is emitted with the
    #   function name, message and stack trace and an empty mapping is returned.
    """
    Given an fx.Graph, return a json that contains the provenance information of each node.
    """
    try:
        provenance_tracking_json = {}
        for node in graph.nodes:
            if node.op == "call_function":
                provenance_tracking_json[node.name] = (
                    [source.to_dict() for source in node.meta["from_node"]]
                    if "from_node" in node.meta
                    else []
                )
        return provenance_tracking_json
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "get_graph_provenance_json",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return {}


def _get_custom_metadata(gm: GraphModule) -> str:
    # [FUNCTION DOC] HELPER: Return a newline-joined string of custom metadata for a
    #   GraphModule, one entry per node having non-empty custom meta (tuple of op, name,
    #   custom), recursing into get_attr nodes that refer to GraphModule submodules.
    #   Raises AssertionError if gm is not a GraphModule.
    # KNOWN ISSUE: recursive results are appended as a nested list, so a submodule
    #   appears as one list entry instead of being flattened.
    if not isinstance(gm, GraphModule):
        raise AssertionError(f"Expected GraphModule, got {type(gm)}")

    def helper(gm: GraphModule) -> list[Any]:
        # [FUNCTION DOC] HELPER: Collect (op, name, custom) tuples for nodes of a
        #   GraphModule with custom meta, appending a nested list for each GraphModule
        #   attribute.
        custom_metadata = []
        for node in gm.graph.nodes:
            if hasattr(node, "meta") and node.meta.get("custom", None):
                custom_metadata.append((node.op, node.name, node.meta["custom"]))
            if node.op == "get_attr" and isinstance(
                getattr(gm, node.target), GraphModule
            ):
                custom_metadata.append(
                    # pyrefly: ignore[bad-argument-type]
                    helper(getattr(gm, node.target))
                )
        return custom_metadata

    return "\n".join(str(x) for x in helper(gm))


def _get_ordered_seq_nr_groups(
    gm: GraphModule | list[GraphModule],
) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Group names of call_function nodes by their seq_nr meta
    #   (ignoring nodes without one) across one GraphModule or a list of them. Returns a
    #   list of name lists, each sorted alphabetically, ordered by increasing seq_nr.
    """
    Group call_function nodes by seq_nr, order by seq_nr value,
    and return a list of lists of node names (sorted alphabetically).

    Args:
        gm: A single GraphModule or a list of GraphModules to process.
            When a list is provided, nodes from all graphs are grouped together.

    Returns:
        A list of lists, where each inner list contains node names that share the same seq_nr,
        sorted alphabetically. The outer list is ordered by seq_nr value.
    """
    # Normalize input to a list
    if isinstance(gm, GraphModule):
        gms = [gm]
    else:
        gms = gm

    seq_nr_dict: dict[int, list[str]] = defaultdict(list)
    for graph_module in gms:
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                seq_nr = node.meta.get("seq_nr")
                if seq_nr is not None:
                    seq_nr_dict[seq_nr].append(node.name)
    # Sort by seq_nr and return list of sorted lists
    return [sorted(seq_nr_dict[k]) for k in sorted(seq_nr_dict.keys())]
