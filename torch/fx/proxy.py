# [MODULE DOC] Core of FX symbolic tracing: the Proxy value classes and the TracerBase
#   node-creation machinery. Contents: Scope and ScopeContextManager (module path
#   tracking), TracerBase (creates graph nodes, copies metadata and stack traces, lowers
#   Python values to graph arguments), GraphAppendingTracer, TraceError, Proxy and
#   subclasses MetaProxy, Attribute, ParameterProxy, and helper functions for
#   stack-trace anchors and seq_nr lookup. Import-time side effects: creates module
#   loggers, installs an operator-forwarding method on Proxy for every name in
#   magic_methods and a reflected (r-prefixed) one for every name in
#   reflectable_magic_methods, and builds the fast-path table used by create_arg.
import collections
import copy
import dis
import enum
import inspect
import logging
import operator
import sys
import traceback
import types
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from typing import Any, cast, TypeVar
from typing_extensions import Never

import torch
import torch.fx.traceback as fx_traceback
from torch._C import _fx_map_arg as map_arg
from torch._library.opaque_object import is_opaque_constant_type
from torch._logging import getArtifactLogger
from torch.utils._pytree import tree_map_
from torch.utils._traceback import CapturedTraceback

from ._compatibility import compatibility
from .graph import Graph, magic_methods, reflectable_magic_methods
from .immutable_collections import immutable_dict, immutable_list
from .node import Argument, base_types, Node, Target
from .operator_schemas import check_for_mutable_operation


__all__ = [
    "TracerBase",
    "GraphAppendingTracer",
    "TraceError",
    "Proxy",
    "MetaProxy",
    "Attribute",
    "ParameterProxy",
    "Scope",
    "ScopeContextManager",
]


log = logging.getLogger(__name__)
annotation_log = getArtifactLogger(__name__, "annotation")


def _is_arbitrary_callable(obj: object) -> bool:
    # [FUNCTION DOC] HELPER: Return True if obj is a plain function, bound method,
    #   builtin function or method, or functools.partial. Such callables cannot be
    #   handled by the standard Proxy mechanism.
    """
    Returns True if obj is an arbitrary callable (function, lambda, method, etc.)
    that requires special tracing to handle. These cannot be symbolically traced
    using the standard Proxy mechanism.
    """
    import functools
    import types

    return isinstance(
        obj,
        (
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            functools.partial,
        ),
    )


def _find_arbitrary_callable(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    # [FUNCTION DOC] HELPER: Walk args and kwargs with a tree traversal and return the
    #   first arbitrary callable found (see _is_arbitrary_callable), or None if there is
    #   none.
    """
    Recursively searches args and kwargs for any arbitrary callable.
    Returns the first arbitrary callable found, or None if none exist.
    """
    found = None

    _T = TypeVar("_T")

    def check(obj: _T) -> _T:
        # [FUNCTION DOC] HELPER: Tree-traversal callback that records the first
        #   arbitrary callable seen in the enclosing variable and returns the object
        #   unchanged.
        nonlocal found
        if found is not None:
            return obj
        if _is_arbitrary_callable(obj):
            found = obj
        return obj

    tree_map_(check, args)
    tree_map_(check, kwargs)
    return found


@compatibility(is_backward_compatible=False)
class Scope:
    # [CLASS DOC] Records the module path (string) and module type of the module
    #   currently being traced; used to attribute each created Node to its containing
    #   module. Plain mutable holder with two attributes, module_path and module_type.
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example::

        class Sub(torch.nn.Module):
            def forward(self, x):
                # This will be a call_method Node in GraphModule,
                # scope for this would be (module_path="sub", module_type=Sub)
                return x.transpose(1, 2)


        class M(torch.nn.Module):
            def __init__(self) -> None:
                self.sub = Sub()

            def forward(self, x):
                # This will be a call_method Node as well,
                # scope for this would be (module_path="", None)
                x = x.transpose(1, 2)
                x = self.sub(x)
                return x

    """

    def __init__(self, module_path: str, module_type: Any) -> None:
        # [FUNCTION DOC] Store module_path and module_type as attributes.
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


@compatibility(is_backward_compatible=False)
class ScopeContextManager:
    # [CLASS DOC] Context manager that temporarily overwrites a shared Scope with
    #   another scope's path and type and restores the previous values on exit.
    #   Invariant: the shared Scope object is mutated in place, not replaced.
    """A context manager to track the Scope of Node during symbolic tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_scope: Scope,
    ) -> None:
        # [FUNCTION DOC] Save a copy of scope to restore later, then overwrite scope
        #   module_path and module_type with those of current_scope, and remember scope
        #   for enter/exit.
        super().__init__()
        # Keep a copy of prev scope to restore on exit
        self._prev_scope = copy.copy(scope)
        # Update scope to current scope
        scope.module_path = current_scope.module_path
        scope.module_type = current_scope.module_type
        # Save a reference so we can restore it
        self._scope = scope

    def __enter__(self) -> Scope:
        # [FUNCTION DOC] Return the (already updated) shared Scope.
        return self._scope

    def __exit__(self, *args: object) -> None:
        # [FUNCTION DOC] Restore the shared Scope module_path and module_type from the
        #   saved copy. Ignores exception arguments and does not suppress exceptions.
        self._scope.module_path = self._prev_scope.module_path
        self._scope.module_type = self._prev_scope.module_type
        return


_COPY_META_FIELDS = [
    "nn_module_stack",
    "torch_fn",
    "source_fn_stack",
    "original_aten",
    "recompute",
    "ac_graph_id",
    "has_backward_hook",
    "from_node",
    "quantization_tag",  # TODO deprecated
    "_numeric_debug_handle",  # TODO deprecated
    "custom",
    "partitioner_tag",
]


# Set of (filename, co_name) pairs registered as user-code anchors. Frames
# matching any entry are preserved by ``_filter_traceback_frames`` when
# ``_record_forward_stack_traces_only`` is True. Populated via
# ``_register_stack_trace_anchor`` (e.g. by ``annotate_fn``).
_STACK_TRACE_ANCHORS: set[tuple[str, str]] = set()


def _register_stack_trace_anchor(fn: Callable[..., Any]) -> None:
    # [FUNCTION DOC] HELPER: Register fn as a stack-trace anchor by adding its code file
    #   name and code name to the global anchor set; does nothing if fn has no code
    #   object.
    code = getattr(fn, "__code__", None)
    if code is not None:
        _STACK_TRACE_ANCHORS.add((code.co_filename, code.co_name))


@compatibility(is_backward_compatible=True)
class TracerBase:
    # [CLASS DOC] Base class of tracers. Class-level state: graph (target Graph), flags
    #   record_stack_traces, check_mutable_operations, trace_asserts and
    #   proxy_buffer_attributes, traced_func_name, scope (current module Scope),
    #   module_stack (ordered mapping of module call stack), node_name_to_scope (node
    #   name to module path and type). Subclasses must set graph, scope, module_stack
    #   and node_name_to_scope. Key methods: create_node, create_proxy, create_arg, and
    #   the overridable hooks to_bool, iter and keys.
    graph: Graph
    record_stack_traces: bool = False
    # When record_stack_traces is True, only reocrd stack traces
    # with forward function names.
    # This helps when we want stack trace back to model code
    _record_forward_stack_traces_only: bool = False
    # Feature flag for mutable schema checking
    # Enableby default in 1.12
    check_mutable_operations: bool = False
    # Feature flag for assert tracing
    trace_asserts: bool = False
    # Feature flag for proxying accesses to buffer values
    proxy_buffer_attributes: bool = False

    # Name of the function to be traced. It will only be used when
    # ``root`` is an instance of ``nn.Module``
    traced_func_name: str = "forward"

    # Maps the containing module's name to the operator name
    scope: Scope

    # Records the module call stack
    module_stack: OrderedDict[str, tuple[str, Any]]

    # Mapping of node name to module scope
    node_name_to_scope: dict[str, tuple[str, type]]

    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> Node:
        # [FUNCTION DOC] Create a Node in self.graph and return it. Optionally rejects
        #   mutable operations (when check_mutable_operations is set and kind is
        #   call_function). Records the current scope under the node name. When
        #   node-meta preservation is on, copies stack_trace, selected metadata fields
        #   (_COPY_META_FIELDS), a computed seq_nr, replay-node overrides
        #   (is_functional_regenerated, custom, stack_trace) and autograd_backward flags
        #   from the global current meta; otherwise copies the module_stack into
        #   nn_module_stack if non-empty. If record_stack_traces is set and the node
        #   still has no stack trace, captures, filters and formats the user stack. Logs
        #   at debug level.
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """

        if kind == "call_function" and self.check_mutable_operations:
            target_fn = cast(Callable[..., Any], target)
            check_for_mutable_operation(target_fn, args, kwargs)

        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        # TODO node_name_to_scope will be deprecated in favor of
        # node.meta['nn_module_stack']
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )

        # Optionally set stack trace on the created Node for debugging purposes
        if fx_traceback.has_preserved_node_meta():
            current_meta: dict[str, Any] = fx_traceback.get_current_meta()

            stack_trace = current_meta.get("stack_trace")
            if stack_trace:
                node.stack_trace = stack_trace

                if fx_traceback.GRADIENT_ACC_SPECIAL_STACK in stack_trace:
                    node.meta["is_gradient_acc"] = True
                    node.meta["autograd_backward"] = True

            # Explicitly set the stack_trace, nn_module_stack and source_fn on the node.meta
            # If other meta fields are needed, they can be added here
            for field in _COPY_META_FIELDS:
                if field in current_meta:
                    node.meta[field] = copy.copy(current_meta[field])

            new_seq_nr = _get_seq_nr(node.name)
            if new_seq_nr is not None:
                annotation_log.debug(
                    "Assigning new_seq_nr %s to %s", new_seq_nr, node.name
                )
                node.meta["seq_nr"] = new_seq_nr

            # See Note [Functionalization View Replay Annotation]
            # Overriding some node meta with the original node meta of the
            # regenerated node.
            replay_node: Node | None = fx_traceback.get_current_replay_node()
            if replay_node is not None:
                node.meta["is_functional_regenerated"] = True
                if "custom" in replay_node.meta:
                    node.meta["custom"] = replay_node.meta.get("custom")
                if "stack_trace" in replay_node.meta:
                    node.stack_trace = replay_node.meta.get("stack_trace")

            if current_meta.get("autograd_backward", False):
                node.meta["autograd_backward"] = True

        elif self.module_stack:
            node.meta["nn_module_stack"] = copy.copy(self.module_stack)

        if self.record_stack_traces and not node.stack_trace:
            user_stack_summary = CapturedTraceback.extract().summary()
            if user_stack_summary:
                user_stack_summary = self._filter_traceback_frames(user_stack_summary)
                if user_stack_summary:
                    node.stack_trace = "".join(user_stack_summary.format()).strip()

        log.debug("create_node %s", node)
        return node

    def _filter_traceback_frames(
        self, user_stack_summary: traceback.StackSummary
    ) -> traceback.StackSummary:
        # [FUNCTION DOC] Reduce a captured stack summary to user-relevant frames and
        #   return a new StackSummary. In forward-only mode keeps frames named forward,
        #   frames from torch/__init__.py and registered anchors; otherwise keeps frames
        #   from the first forward frame onward, or none if there is no forward frame.
        #   Then drops frames in files considered uninteresting by symbolic_shapes.
        #   Meant to be overridden.
        # KNOWN ISSUE: user_frames is re-annotated in the no-forward branch, which is
        #   redundant.
        # This method can be overridden to customize the frame filtering logic
        # for the recorded stack trace
        user_frames: list[traceback.FrameSummary] = []
        if self._record_forward_stack_traces_only:
            user_frames = [
                frame
                for frame in user_stack_summary
                if (
                    frame.name == "forward"
                    or frame.filename.endswith("torch/__init__.py")
                    or (frame.filename, frame.name) in _STACK_TRACE_ANCHORS
                )
            ]
        else:
            first_forward = -1
            for i, frame in enumerate(user_stack_summary):
                if frame.name == "forward":
                    user_frames = user_stack_summary[i:]
                    first_forward = i
                    break

            # Not having a "forward" call in the stacktrace implies the
            # stacktrace will probably be irrelevant
            if first_forward == -1:
                user_frames: list[traceback.FrameSummary] = []

        from torch.fx.experimental.symbolic_shapes import uninteresting_files

        user_frames = [
            frame
            for frame in user_frames
            if frame.filename not in uninteresting_files()
        ]

        return traceback.StackSummary.from_list(user_frames)

    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> "Proxy":
        # [FUNCTION DOC] Wrap node in a Proxy bound to this tracer.
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any | None = None,
        proxy_factory_fn: Callable[[Node], "Proxy"] | None = None,
    ) -> "Proxy":
        # [FUNCTION DOC] Lower args and kwargs to graph arguments with create_arg
        #   (raises AssertionError if the results are not a tuple and a dict), create
        #   the node with create_node, and wrap it with proxy_factory_fn if given, else
        #   with the default proxy method. Returns the proxy.
        """
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        """

        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        if not isinstance(args_, tuple):
            raise AssertionError(f"Expected args_ to be tuple, got {type(args_)}")
        if not isinstance(kwargs_, dict):
            raise AssertionError(f"Expected kwargs_ to be dict, got {type(kwargs_)}")

        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)

        if not proxy_factory_fn:
            proxy = self.proxy(node)
        else:
            proxy = proxy_factory_fn(node)

        return proxy

    def _find_user_frame(self) -> types.FrameType | None:
        # [FUNCTION DOC] HELPER: Walk up the call stack from the current frame and
        #   return the first frame whose file is not one of the listed PyTorch tracing
        #   files, or None if none exists.
        """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
        # We have to do a little dance here. Basically, walk up the callstack and
        # record the first frame not in the pytorch source. This is the frame executing
        # the user code during tracing.
        frame = inspect.currentframe()

        pt_files = [
            "torch/fx/proxy.py",
            "torch/fx/_symbolic_trace.py",
            "torch/fx/experimental/proxy_tensor.py",
            "torch/_ops.py",
            "torch/_tensor.py",
            "torch/utils/_python_dispatch.py",
            "torch/_prims_common/wrappers.py",
            "torch/_refs/__init__.py",
            "torch/_refs/nn/functional/__init__.py",
            "torch/utils/_stats.py",
        ]
        while frame:
            frame = frame.f_back
            if frame and all(
                not frame.f_code.co_filename.endswith(file) for file in pt_files
            ):
                break

        if not frame:
            return None

        return frame

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        # [FUNCTION DOC] Lower a Python value seen during tracing to an Argument
        #   storable in the graph. Fast path through a type table; then Proxy becomes
        #   its node, objects with a create-arg hook delegate to it, tuples (including
        #   named tuples), lists, dicts, slices and ranges are converted element-wise,
        #   operator overloads, higher-order operators, opaque constant types, enum
        #   members, basic types, None and Ellipsis pass through, and dataclass
        #   instances become a call_function node on their class with field arguments.
        #   Raises NotImplementedError for any other type. May create graph nodes
        #   (dataclasses).
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        # IMPORTANT: Are you here because you are trying to proxy a new type into
        # the graph? Please Please Please contact someone on the PyTorch Compiler team;
        # the considerations are subtle.
        #
        # 1) When you add a new type, all of the downstream consumers and pass writers
        # need to handle the new type. torch.fx is intended to be easy to write
        # passes for, so we will push back against new types.
        # 2) In torch.compile's IR, there are only specific operations that go
        # into the graph. In particular, Tensor operations should go into the graph,
        # but non-Tensor operations shouldn't. What that means is that constructors
        # for new types *SHOULD NOT* become nodes in the FX graph.
        handler = _create_arg_bypass.get(type(a))
        if handler is not None:
            # this is just a performance optimization and can be removed if needed
            # for common types, we have a fast path to avoid isinstance() overhead
            # this doesn't remove the checks below since we need to handle subclasses
            return handler(self, a)

        if isinstance(a, Proxy):
            return a.node  # most common arg type goes first
        elif hasattr(a, "__fx_create_arg__"):
            return a.__fx_create_arg__(self)
        # aggregates
        elif isinstance(a, tuple):
            if hasattr(a, "_fields"):
                # NamedTuple constructors don't seem to like getting a generator
                # expression as an argument to their constructor, so build this
                # intermediate tuple and unpack it into the NamedTuple constructor
                args = [self.create_arg(elem) for elem in a]
                return type(a)(*args)  # type: ignore[arg-type]
            return type(a)([self.create_arg(elem) for elem in a])
        elif isinstance(a, list):
            return [self.create_arg(elem) for elem in a]
        elif isinstance(a, dict):
            return _create_arg_dict(self, a)
        elif isinstance(a, slice):
            return slice(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        elif isinstance(a, range):
            return range(  # pyrefly: ignore[no-matching-overload]
                self.create_arg(a.start),  # pyrefly: ignore[bad-argument-type]
                self.create_arg(a.stop),  # pyrefly: ignore[bad-argument-type]
                self.create_arg(a.step),  # pyrefly: ignore[bad-argument-type]
            )

        elif isinstance(a, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            return a  # pyrefly: ignore[bad-return]

        elif is_opaque_constant_type(type(a)):
            return a

        elif is_dataclass(a):
            kwargs = {
                field.name: self.create_arg(getattr(a, field.name))
                for field in fields(a)
            }
            return self.create_node("call_function", a.__class__, (), kwargs)

        elif isinstance(a, (*base_types, enum.Enum)) or a is None or a is ...:
            return a  # pyrefly: ignore[bad-return]

        raise NotImplementedError(f"argument of type: {type(a)}")

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: "Proxy") -> bool:
        # [FUNCTION DOC] Hook called when a Proxy is converted to bool. Default raises
        #   TraceError because control flow cannot depend on traced values; custom
        #   tracers may return a value.
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        raise TraceError(
            "symbolically traced variables cannot be used as inputs to control flow"
        )

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: "Proxy") -> Iterator:  # pyrefly: ignore[implicit-any]
        # [FUNCTION DOC] Hook called when a Proxy is iterated. Default raises TraceError
        #   with an explanatory message; custom tracers may return an iterator.
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        raise TraceError(
            "Proxy object cannot be iterated. This can be "
            "attempted when the Proxy is used in a loop or"
            " as a *args or **kwargs function argument. "
            "See the torch.fx docs on pytorch.org for a "
            "more detailed explanation of what types of "
            "control flow can be traced, and check out the"
            " Proxy docstring for help troubleshooting "
            "Proxy iteration errors"
        )

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: "Proxy") -> "Proxy":
        # [FUNCTION DOC] Hook called when keys is requested on a Proxy (as with
        #   double-star unpacking). Default records a call to the keys method via an
        #   Attribute proxy and returns the resulting Proxy.
        """Called when a proxy object has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator if ** is supposed to work in your custom tracer.
        """
        return Attribute(obj, "keys")()


def _get_seq_nr(node_name: str = "") -> int | None:
    # [FUNCTION DOC] HELPER: Compute the seq_nr to attach to a node being created. If
    #   inside a grad_fn context, use the last grad_fn_seq_nr in the global meta
    #   (backward nodes); else if seq_nr preservation is on, reuse current meta seq_nr;
    #   else use the autograd sequence number minus one (forward nodes). Unless
    #   preservation is on, a replay node's seq_nr, if present, overrides. node_name is
    #   only used for debug logging. Returns an int or None.
    """
    Returns the seq_nr node meta for the current proxy node that we're creating.
    The seq_nr number in node meta is related to but not the same as the "sequence number"
    in autograd.
    We use the seq_nr to correlate forward and backward nodes in the traced FX graphs
    (e.g. in copy_fwd_metadata_to_bw_nodes).
    The corresponding forward and backward FX graph nodes should have the same seq_nr.

    The ordering of the seq_nr in the graph does not indicate when the node is executed.
    For example, the nodes in the invoke_subgraph HOP's subgraph may be called multiple
    times by different HOP nodes, but these nodes only have a single seq_nr, which may be
    smaller, the same, or larger than the calling HOP.

    `node_name` is the name of the node that we're creating. It is used for logging only.
    """
    current_meta: dict[str, Any] = fx_traceback.get_current_meta()
    new_seq_nr = None
    # The sequence_nr increments every time a new autograd Node
    # is created. During the FWD pass we store the sequence_nr
    # corresponding to the last autograd Node created on this fx
    # node's meta.  A single aten op can create multiple autograd
    # nodes as is the case with in-place foreach ops. During the
    # BWD pass we retrieve the sequence_nr stored on the current
    # executing autograd Node. See NOTE [ Sequence Number ].
    if current_meta.get("in_grad_fn", 0) > 0:
        # This branch is used to get seq_nr for backward nodes
        annotation_log.debug("%s: seq_nr from current_meta grad_fn_seq_nr", node_name)
        new_seq_nr = current_meta["grad_fn_seq_nr"][-1]

    elif torch.fx.traceback._is_preserving_node_seq_nr():
        # Special case where we preserve seq_nr from currently tracing node
        # Used to preserve seq_nr when re-tracing subgraphs in HOP
        annotation_log.debug("%s: seq_nr from current_meta seq_nr", node_name)
        new_seq_nr = current_meta.get("seq_nr")
    else:
        # Here we decrement to account for the sequence_nr having
        # just been incremented while tracing this lowered aten op.
        # This branch is used to get seq_nr for forward nodes
        new_seq_nr = torch.autograd._get_sequence_nr() - 1

    if not torch.fx.traceback._is_preserving_node_seq_nr():
        # See Note [Functionalization View Replay Annotation]
        # Overriding some node meta with the original node meta of the
        # regenerated node.
        replay_node: Node | None = fx_traceback.get_current_replay_node()
        if replay_node is not None:
            if "seq_nr" in replay_node.meta:
                annotation_log.debug("%s: seq_nr from replay_node", node_name)
                new_seq_nr = replay_node.meta["seq_nr"]

    return new_seq_nr


# used in Proxy object when just appending to the graph while not tracing.
@compatibility(is_backward_compatible=True)
class GraphAppendingTracer(TracerBase):
    # [CLASS DOC] Minimal tracer used when a Proxy is built around a raw Node outside of
    #   tracing; just appends to the given graph.
    def __init__(self, graph: Graph) -> None:
        # [FUNCTION DOC] Set graph, an empty root Scope, an empty ordered module_stack
        #   and an empty node_name_to_scope mapping.
        super().__init__()
        self.graph = graph
        self.scope = Scope("", None)
        self.module_stack: OrderedDict[str, tuple[str, Any]] = collections.OrderedDict()
        self.node_name_to_scope: dict[str, tuple[str, type]] = {}


@compatibility(is_backward_compatible=False)
def assert_fn(x: object) -> None:
    # [FUNCTION DOC] HELPER: Raise AssertionError if x is falsy; used as the graph
    #   target recorded for traced assert statements.
    if not x:
        raise AssertionError("Assertion failed")


@compatibility(is_backward_compatible=True)
class TraceError(ValueError):
    # [CLASS DOC] Error raised when symbolic tracing hits something it cannot handle.
    #   Subclass of ValueError with no extra behavior.
    pass


@compatibility(is_backward_compatible=True)
class Proxy:
    # [CLASS DOC] Node wrapper that flows through the program during symbolic tracing
    #   and records every operation applied to it into the graph of its tracer.
    #   Attributes: tracer and node. Cannot be iterated except in fixed-size unpacking,
    #   and its truth value and length are not available by default. Operator methods
    #   are installed dynamically at import time. Intercepts torch functions via the
    #   torch function protocol.
    """
    ``Proxy`` objects are ``Node`` wrappers that flow through the
    program during symbolic tracing and record all the operations
    (``torch`` function calls, method calls, operators) that they touch
    into the growing FX Graph.

    If you're doing graph transforms, you can wrap your own ``Proxy``
    method around a raw ``Node`` so that you can use the overloaded
    operators to add additional things to a ``Graph``.

    ``Proxy`` objects cannot be iterated. In other words, the symbolic
    tracer will throw an error if a ``Proxy`` is used in a loop or as
    an ``*args``/``**kwargs`` function argument.

    There are two main ways around this:
    1. Factor out the untraceable logic into a top-level function and
    use ``fx.wrap`` on it.
    2. If the control flow is static (i.e. the loop trip count is
    based on some hyperparameter), the code can be kept in its original
    position and refactored into something like::

        for i in range(self.some_hyperparameter):
            indexed_item = proxied_value[i]

    For a more detailed description into the Proxy internals, check out
    the "Proxy" section in `torch/fx/README.md`
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, node: Node, tracer: "TracerBase | None" = None) -> None:
        # [FUNCTION DOC] Bind the proxy to node and tracer; if tracer is None, create a
        #   GraphAppendingTracer on the node's graph.
        if tracer is None:
            # This allows you to create a Proxy object around a raw Node
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return a string showing Proxy with the node name.
        return f"Proxy({self.node.name})"

    def __getattr__(self, k: str) -> "Attribute":
        # [FUNCTION DOC] Return an Attribute proxy for attribute k of this proxy; no
        #   graph node is created until it is used as a value (method calls are
        #   peephole-optimized into call_method).
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the instance dictionary for pickling.
        return self.__dict__

    def __deepcopy__(self, memo: dict[int, Any]) -> "Proxy":
        # [FUNCTION DOC] Deep copy the proxy: copy each instance attribute, falling back
        #   to a shallow copy with a logged warning when deep copy fails. Raises
        #   AssertionError if node or tracer is missing. Builds a new Proxy and copies
        #   all attributes into it. Explicitly defined to avoid the attribute hook
        #   returning an Attribute for the copy hook.
        # We have to explicitly override this method, because otherwise deepcopy
        # will go to __getattr__(self, "__deepcopy__") and return a
        # Attribute(__deepcopy__), and may go into an infinite loop in some cases.
        import copy

        new_dict: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            try:
                new_obj = copy.deepcopy(v, memo)
            except Exception:
                log.warning(
                    "Shallow copy %s of Proxy because it cannot be deepcopied. "
                    "Proxy is created for node %s",
                    k,
                    self.node.name,
                )
                new_obj = copy.copy(v)
            new_dict[k] = new_obj
        if "node" not in new_dict:
            raise AssertionError("'node' not in new_dict during proxy unpickling")
        if "tracer" not in new_dict:
            raise AssertionError("'tracer' not in new_dict during proxy unpickling")
        new_proxy = Proxy(new_dict["node"], new_dict["tracer"])
        for k, v in new_dict.items():
            new_proxy.__dict__[k] = v
        return new_proxy

    def __setstate__(self, d: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state when unpickling by replacing the instance
        #   dictionary with d.
        # This is called when being unpickled/loaded.
        self.__dict__ = d

    def __call__(self, *args: Any, **kwargs: Any) -> "Proxy":
        # [FUNCTION DOC] Record a call_method node named __call__ with this proxy and
        #   the given arguments; returns the new Proxy.
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs
        )

    def __iter__(self) -> Iterator["Proxy"]:
        # [FUNCTION DOC] Iterate the proxy. Inspects the calling frame bytecode; if the
        #   caller is a fixed-size sequence unpacking, returns a generator of indexed
        #   proxies for the expected count; otherwise delegates to the tracer iter hook
        #   (which raises by default). Raises AssertionError if frames are unavailable.
        # KNOWN ISSUE: on Python before 3.11 the instruction index is guessed from the
        #   byte offset, which depends on bytecode layout.
        frame = inspect.currentframe()
        if frame is None:
            raise AssertionError("inspect.currentframe() returned None")
        calling_frame = frame.f_back
        if calling_frame is None:
            raise AssertionError("frame.f_back is None")
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left

            inst_idx = bisect_left(
                inst_list, calling_frame.f_lasti, key=lambda x: x.offset
            )
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == "UNPACK_SEQUENCE":
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    def __abs__(self) -> "Proxy":
        # [FUNCTION DOC] Record a call_function node for the abs operator on this proxy
        #   and return the new Proxy.
        return self.tracer.create_proxy("call_function", operator.abs, (self,), {})

    def __bool__(self) -> bool:
        # [FUNCTION DOC] Convert to bool. If assert tracing is enabled, inspects the
        #   caller bytecode to detect a conditional jump guarding a raise of
        #   AssertionError; if matched, records an assert_fn node and returns True.
        #   Otherwise delegates to the tracer to_bool hook (raises TraceError by
        #   default). Raises AssertionError if frames are unavailable or a jump
        #   instruction lacks an argument.
        # KNOWN ISSUE: the bytecode pattern uses a jump target index computed by halving
        #   an argument, which only matches older Python bytecode layouts and likely
        #   fails on newer versions.
        if self.tracer.trace_asserts:
            # check if this boolean is used in an assertion, bytecode pattern for assertions
            # is pretty stable for Python 3.7--3.9
            frame = inspect.currentframe()
            if frame is None:
                raise AssertionError("inspect.currentframe() returned None")
            calling_frame = frame.f_back
            if calling_frame is None:
                raise AssertionError("frame.f_back is None")
            insts = list(dis.get_instructions(calling_frame.f_code))
            if sys.version_info >= (3, 11):
                from bisect import bisect_left

                cur = bisect_left(insts, calling_frame.f_lasti, key=lambda x: x.offset)
            else:
                cur = calling_frame.f_lasti // 2
            inst = insts[cur]

            if inst.opname == "POP_JUMP_IF_TRUE":
                first = insts[cur + 1]
                if inst.arg is None:
                    raise AssertionError("inst.arg is None for POP_JUMP_IF_TRUE")
                last = insts[inst.arg // 2 - 1]
                starts_with_assert = (
                    first.opname == "LOAD_GLOBAL"
                    and first.argval == "AssertionError"
                    or first.opname == "LOAD_ASSERTION_ERROR"
                )
                if starts_with_assert and last.opname == "RAISE_VARARGS":
                    self.tracer.create_proxy("call_function", assert_fn, (self,), {})
                    return True

        return self.tracer.to_bool(self)

    @compatibility(is_backward_compatible=True)
    def keys(self) -> "Proxy":
        # [FUNCTION DOC] Delegate to the tracer keys hook and return its result.
        return self.tracer.keys(self)

    def __len__(self) -> int:
        # [FUNCTION DOC] Always raises RuntimeError saying length is not supported in
        #   symbolic tracing, suggesting wrapping len at module scope.
        raise RuntimeError(
            "'len' is not supported in symbolic tracing by default. If you want "
            "this call to be recorded, please call torch.fx.wrap('len') at "
            "module scope"
        )

    @classmethod
    def __torch_function__(
        cls,
        orig_method: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> "Proxy":
        # [FUNCTION DOC] Torch function protocol hook. Finds the single tracer among
        #   Proxy arguments (raises RuntimeError if several differ). Script methods
        #   become call_method nodes on their owner; tensor methods or properties become
        #   call_method nodes by name; other callables become call_function nodes named
        #   from the callable. For higher-order operators, raises RuntimeError if any
        #   argument is an arbitrary callable. Returns the resulting Proxy.
        # KNOWN ISSUE: if no Proxy argument is found, taking the first tracer raises
        #   StopIteration instead of a clear error.
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        tracers: dict[TracerBase, None] = {}

        def find_tracer(a: Any) -> None:
            # [FUNCTION DOC] HELPER: Tree-traversal callback that records the tracer of
            #   any argument that is an instance of the class, using an ordered mapping
            #   as a set.
            if isinstance(a, cls):
                tracers[a.tracer] = None

        tree_map_(find_tracer, args)
        tree_map_(find_tracer, kwargs)

        if len(tracers) > 1:
            raise RuntimeError(
                f"Found multiple different tracers {list(tracers.keys())} while "
                f"trying to trace operations {orig_method}"
            )
        tracer = next(iter(tracers.keys()))

        if isinstance(orig_method, torch._C.ScriptMethod):
            args = (orig_method.owner,) + args
            return tracer.create_proxy("call_method", orig_method.name, args, kwargs)
        if torch.overrides.is_tensor_method_or_property(orig_method):
            return tracer.create_proxy(
                "call_method", orig_method.__name__, args, kwargs
            )
        else:
            if isinstance(orig_method, torch._ops.HigherOrderOperator):
                bad_callable = _find_arbitrary_callable(args, kwargs)
                if bad_callable is not None:
                    raise RuntimeError(
                        f"Unable to symbolically trace the HigherOrderOperator "
                        f"{orig_method._name} because it received an arbitrary "
                        f"callable argument {bad_callable}. Use make_fx or dynamo "
                        f"tracing instead."
                    )
            return tracer.create_proxy(
                "call_function",
                orig_method,
                args,
                kwargs,
                name=tracer.graph._target_to_str(orig_method.__name__),
            )


@compatibility(is_backward_compatible=False)
class MetaProxy(Proxy):
    # [CLASS DOC] Proxy subclass that also evaluates the real operation on the stored
    #   meta values under a fake mode and saves the output in the node meta under val.
    #   Extra attribute: fake_mode.
    """
    A Proxy subclass that propagates metadata (meta['val']) during graph tracing.
    """

    def __init__(
        self, node: Node, tracer: "TracerBase | None" = None, fake_mode: Any = None
    ) -> None:
        # [FUNCTION DOC] Initialize as Proxy with node and tracer and store fake_mode.
        super().__init__(node, tracer)
        self.fake_mode = fake_mode

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return a string showing MetaProxy with the node name.
        return f"MetaProxy({self.node.name})"

    @classmethod
    def __torch_function__(
        cls,
        orig_method: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> "MetaProxy":
        # [FUNCTION DOC] Torch function hook: requires at least one MetaProxy positional
        #   argument (raises AssertionError otherwise). Records the node via the base
        #   class, then under that proxy's fake mode runs the original function on the
        #   meta val of each Proxy argument (other arguments unchanged) and stores the
        #   result in the new node meta under val. Returns a new MetaProxy with the same
        #   fake mode.
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        meta_proxy = None
        for arg in args:
            if isinstance(arg, MetaProxy):
                meta_proxy = arg
                break

        if meta_proxy is None:
            raise AssertionError(
                "No MetaProxy found in arguments, but one is expected."
            )

        proxy = super().__torch_function__(orig_method, types, args, kwargs)
        with meta_proxy.fake_mode:
            proxy.node.meta["val"] = orig_method(
                *[a.node.meta["val"] if isinstance(a, Proxy) else a for a in args],
                **kwargs,
            )
        return MetaProxy(proxy.node, proxy.tracer, meta_proxy.fake_mode)


@compatibility(is_backward_compatible=True)
class Attribute(Proxy):
    # [CLASS DOC] Lazy proxy for attribute access on another Proxy. Stores root, attr
    #   and tracer; the graph node (a getattr call) is created only when the node
    #   property is first read. Calling it records a method call instead, which avoids a
    #   getattr node for method calls.
    @compatibility(is_backward_compatible=True)
    def __init__(self, root: Proxy, attr: str) -> None:
        # [FUNCTION DOC] Store root, attr and root's tracer and set the cached node to
        #   None. Does not call the base initializer.
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node | None = None

    @property
    def node(self) -> Node:  # pyrefly: ignore[bad-override]
        # [FUNCTION DOC] Property: lazily create (once) and return the node for the
        #   getattr of attr on root.
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args: Any, **kwargs: Any) -> "Proxy":
        # [FUNCTION DOC] Record a call_method node named after attr with root as the
        #   first argument and the given arguments; returns the new Proxy.
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs
        )


@compatibility(is_backward_compatible=False)
class ParameterProxy(Proxy):
    # [CLASS DOC] Proxy for a module parameter that forwards shape, size, dim, ndim,
    #   numel and nelement to the real parameter so tracing-time conditionals on them
    #   work. Attributes: param and name.
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing
    """

    def __init__(
        self, tracer: TracerBase, node: Node, name: str, param: torch.nn.Parameter
    ) -> None:
        # [FUNCTION DOC] Initialize as Proxy; raises AssertionError if param is not a
        #   Parameter; stores param and name.
        super().__init__(node, tracer)
        if not isinstance(param, torch.nn.Parameter):
            raise AssertionError(f"Expected Parameter, got {type(param)}")
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return a string showing ParameterProxy with the name.
        return f"ParameterProxy({self.name})"

    @property
    def shape(self) -> torch.Size:
        # [FUNCTION DOC] Property: the real parameter shape.
        return self.param.shape

    def size(self) -> torch.Size:
        # [FUNCTION DOC] Return the real parameter size.
        return self.param.size()

    def dim(self) -> int:
        # [FUNCTION DOC] Return the real parameter number of dimensions.
        return self.param.dim()

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Property: the real parameter number of dimensions.
        return self.param.ndim

    def numel(self) -> int:
        # [FUNCTION DOC] Return the real parameter element count.
        return self.param.numel()

    def nelement(self) -> int:
        # [FUNCTION DOC] Return the real parameter element count.
        return self.param.nelement()


for method in magic_methods:

    def _scope(method: str) -> None:
        # [FUNCTION DOC] HELPER: Install on Proxy a method for the given magic method
        #   name (underscores stripped and re-wrapped in double underscores) that
        #   records a call_function node on the matching operator function.
        def impl(*args: Any, **kwargs: Any) -> "Proxy":
            # [FUNCTION DOC] HELPER: Generated Proxy magic method: takes the tracer from
            #   the first argument and records a call_function node for the operator
            #   with the given arguments; returns the Proxy.
            tracer = args[0].tracer
            target = getattr(operator, method)
            return tracer.create_proxy("call_function", target, args, kwargs)

        impl.__name__ = method
        as_magic = f"__{method.strip('_')}__"
        setattr(Proxy, as_magic, impl)

    _scope(method)


def _define_reflectable(orig_method_name: str) -> None:
    # [FUNCTION DOC] HELPER: Install on Proxy the reflected variant (name with r prefix)
    #   of an operator method; it records the operator called with the other operand
    #   first and the proxy second.
    method_name = f"__r{orig_method_name.strip('_')}__"

    def impl(self: "Proxy", rhs: Any) -> "Proxy":
        # [FUNCTION DOC] HELPER: Generated reflected method: record a call_function node
        #   for the operator with arguments (rhs, self) and return the Proxy.
        target = getattr(operator, orig_method_name)
        return self.tracer.create_proxy("call_function", target, (rhs, self), {})

    impl.__name__ = method_name
    impl.__qualname__ = method_name
    setattr(Proxy, method_name, impl)


for orig_method_name in reflectable_magic_methods:
    _define_reflectable(orig_method_name)


def _no_nodes_error(arg: Argument) -> Never:
    # [FUNCTION DOC] HELPER: Always raises RuntimeError saying dictionary keys used as
    #   arguments cannot contain a Node, naming arg.
    raise RuntimeError(
        "Keys for dictionaries used as an argument cannot contain a "
        f"Node. Got key: {arg}"
    )


def _create_arg_dict(self: TracerBase, a: dict[Any, Any]) -> dict[Any, Argument]:
    # [FUNCTION DOC] HELPER: Lower a dict for use as a graph argument: non-string keys
    #   are lowered with create_arg and checked with map_arg so that any node inside
    #   raises RuntimeError; values are lowered with create_arg. Returns a new dict.
    r: dict[Any, Argument] = {}
    for k, v in a.items():
        if not isinstance(k, str):
            # Check for invalid dict keys. We do not want a Proxy to appear
            # anywhere within the key. Since keys can be collection types,
            # we iterate through the key with map_arg
            k = self.create_arg(k)
            map_arg(k, _no_nodes_error)
        r[k] = self.create_arg(v)
    return r


_create_arg_bypass = {
    t: lambda self, a: a
    for t in [
        *base_types,
        type(None),
        type(...),
        torch._ops.OpOverload,
        torch._ops.HigherOrderOperator,
    ]
}
_create_arg_bypass[Proxy] = lambda self, a: a.node
_create_arg_bypass[tuple] = lambda self, a: tuple(self.create_arg(elem) for elem in a)
_create_arg_bypass[list] = lambda self, a: [self.create_arg(elem) for elem in a]
_create_arg_bypass[dict] = _create_arg_dict
_create_arg_bypass[immutable_list] = _create_arg_bypass[list]
_create_arg_bypass[immutable_dict] = _create_arg_bypass[dict]
