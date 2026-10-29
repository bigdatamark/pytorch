# [MODULE DOC] Defines Node, the unit of the fx intermediate representation, plus helpers: argument mapping (map_arg, map_aggregate),
# the has_side_effect registry used by dead code elimination, and printing helpers for types, targets and arguments.
# Import-time side effects: creates the module logger and builds the global set of side-effectful functions (including some optional inductor ops).
# Nodes represent a definition of a value in our graph of operators.
import builtins
import inspect
import logging
import operator
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import ParamSpec, TypeVar

import torch
from torch._C import _fx_map_aggregate, _fx_map_arg, _NodeBase
from torch.fx.operator_schemas import (
    ArgsKwargsPair,
    normalize_function,
    normalize_module,
)
from torch.utils._dtype_abbrs import dtype_abbrs

from .._ops import ops as _ops
from ._compatibility import compatibility


if TYPE_CHECKING:
    from .graph import Graph

__all__ = ["Node", "map_arg", "map_aggregate", "has_side_effect"]

log = logging.getLogger(__name__)

BaseArgumentTypes = Union[  # noqa: UP007
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    torch._ops.OpOverload,
    torch.SymInt,
    torch.SymBool,
    torch.SymFloat,
]
base_types = typing.get_args(BaseArgumentTypes)

Target: TypeAlias = Callable[..., Any] | str

Argument = Optional[  # noqa: UP045
    Union[
        tuple["Argument", ...],
        Sequence["Argument"],
        Mapping[str, "Argument"],
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        range,
        "Node",
        BaseArgumentTypes,
    ]
]
# pyrefly: ignore [invalid-annotation]
ArgumentT = TypeVar("ArgumentT", bound=Argument)
_P = ParamSpec("_P")
_R = TypeVar("_R")

_legal_ops = dict.fromkeys(
    [
        "placeholder",
        "call_method",
        "call_module",
        "call_function",
        "get_attr",
        "output",
        "root",
    ]
)

# Dynamo is unable to trace global set[Callable].__contains__.
# See https://github.com/pytorch/pytorch/issues/145761. Since we only have
# a handful of ops so switch to list of callables.
_side_effectful_need_to_be_preserved_pre_dispatch: list[Callable[..., Any]] = [
    torch._C._set_grad_enabled,
    torch.amp._enter_autocast,
    torch.amp._exit_autocast,
]

# TODO: Either refactor this into 2 functions 1 dce for functional graphs and 1 dce for all graphs,
# or add logic to correctly mark all inplace ops as side effectful.
#
# NOTE: For new operators, please do not add to this set!
# Instead, consider using the effects system via
# torch.library._register_effectful_op() for operators.
#
# This _side_effectful_functions set is only for:
# - Legacy functions that aren't operators (e.g., profiler ops, asserts)
# - Things that cannot be marked via the normal effects system
_side_effectful_functions: set[Callable[..., Any]] = {
    torch._assert,
    torch._assert_async,
    _ops.aten._assert_async.msg,
    _ops.aten._assert_scalar.default,
    _ops.aten._assert_tensor_metadata.default,
    _ops.aten.sym_constrain_range.default,
    _ops.aten.sym_constrain_range_for_size.default,
    _ops.profiler._record_function_enter,
    _ops.profiler._record_function_enter.default,
    _ops.profiler._record_function_enter_new,
    _ops.profiler._record_function_enter_new.default,
    _ops.profiler._record_function_exit,
    _ops.profiler._record_function_exit._RecordFunction,
    _ops.inductor.accumulate_grad_.default,
    operator.setitem,
    *_side_effectful_need_to_be_preserved_pre_dispatch,
}

if hasattr(_ops.inductor, "resize_storage_bytes_"):
    _side_effectful_functions.add(_ops.inductor.resize_storage_bytes_.default)


def _device_annotation(device: torch.device) -> str:
    # [FUNCTION DOC] HELPER: Returns the printable string for a device used in graph dumps. When the compiler config flag
    # compile_on_one_rank is set and the device has an index equal to the current accelerator's current device index (and the same type),
    # returns only the device type so output is identical across ranks; otherwise returns the full device string.
    # Reads global accelerator and compiler config state; no mutation.
    # Render a tensor's device for graph printing. Under compile-on-one-rank the device
    # index is rank-specific; drop it when it is the current default device for the
    # accelerator (e.g. "cuda:3" -> "cuda") so the printed graph is byte-identical across
    # ranks. A non-current index keeps its index, preserving debuggability.
    import torch.compiler.config as compiler_config

    if compiler_config.compile_on_one_rank and device.index is not None:
        acc = torch.accelerator.current_accelerator()
        if (
            acc is not None
            and device.type == acc.type
            and device.index == torch.accelerator.current_device_index()
        ):
            return device.type
    return str(device)


@compatibility(is_backward_compatible=False)
def has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Adds fn to the module-level set of side-effectful functions so dead code elimination keeps calls to it.
    # Returns fn unchanged so it can be used as a decorator. Side effect: mutates the global set.
    """
    Registers a function to not be dead code eliminated by
    fx.graph.eliminate_dead_code

    NOTE: For new operators, please do not add to this set!
    Instead, consider using the effects system via
    torch.library._register_effectful_op() for operators.

    This _side_effectful_functions set is only for:
    - Legacy functions that aren't operators (e.g., profiler ops, asserts)
    - Things that cannot be marked via the normal effects system
    """
    _side_effectful_functions.add(fn)
    return fn


# this is fixed on master, WAR for 1.5
def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    # [FUNCTION DOC] HELPER: Returns the name of the module that defines orig_method: its module attribute if set, otherwise the
    # first of torch or torch.nn.functional that has an attribute of the same name that is the same object.
    # Raises RuntimeError if no module is found.
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f"cannot find module for {orig_method}")


# Borrowed from CPython typing module
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
def _type_repr(obj: object) -> str:
    # [FUNCTION DOC] HELPER: Returns a short string for obj: module-qualified name for classes (bare qualified name for builtins),
    # three dots for the ellipsis, the plain name for functions, and the repr for everything else. Generic aliases use repr.
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # Extension: If we don't ignore GenericAlias then `list[int]` will print
    # simply "list".
    if isinstance(obj, type) and not isinstance(obj, types.GenericAlias):
        if obj.__module__ == "builtins":
            return obj.__qualname__
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is ...:
        return "..."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


def _get_qualified_name(func: Callable[..., Any]) -> str:
    # [FUNCTION DOC] HELPER: Returns a dotted importable name for func, e.g. torch.add or torch.Tensor.method.
    # Builtins give the bare name; Tensor methods give a torch.Tensor prefix; torch._ops is rewritten to torch.ops; lambdas are named
    # by reading their source assignment target. A few private names are fixed up (segment_reduce, scaling/swizzle type enums).
    # Raises RuntimeError if a lambda's source is unavailable or the defining module cannot be found.
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    # torch.Tensor.{fn}
    if (
        isinstance(func, (types.MethodDescriptorType, types.WrapperDescriptorType))
        and func is getattr(torch.Tensor, func.__name__, None)
    ) or (
        func.__module__ == torch._tensor.__name__
        and func.__qualname__ == f"Tensor.{func.__name__}"
    ):
        return f"torch.Tensor.{func.__name__}"
    name = func.__name__

    if name == "<lambda>":
        # For lambdas, try to get their defining name in the module
        try:
            name = inspect.getsource(func).split("=")[0].strip()
        except Exception as e:
            raise RuntimeError("Unable to represent lambda") from e
    module = _find_module_of_method(func)
    module = module.replace(
        "torch._ops", "torch.ops"
    )  # WAR for bug in how torch.ops assigns module
    # Fixup segment_reduce mismatch
    if module == "torch" and name == "segment_reduce":
        name = "_" + name
    if module == "torch.nn.functional" and name in ("_ScalingType", "_SwizzleType"):
        name = name.removeprefix("_")
    return f"{module}.{name}"


def _format_arg(arg: object, max_list_len: float = float("inf")) -> str:
    # [FUNCTION DOC] HELPER: Formats a node argument for printing. Uses a custom repr hook if the object has one; lists, tuples and dicts are
    # formatted recursively; Nodes print with a percent prefix; devices go through the device annotation helper; others use str.
    # Lists and tuples are truncated to max_list_len elements with a total-length marker.
    # KNOWN ISSUE: max_list_len is not propagated to nested elements, and dict contents are never truncated.
    if hasattr(arg, "_custom_fx_repr_fn"):
        return arg._custom_fx_repr_fn()
    elif isinstance(arg, list):
        items = ", ".join(
            _format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len
        )
        maybe_len = (
            "" if len(arg) < max_list_len + 1 else f", ...[total_len={len(arg)}]"
        )
        return f"[{items}{maybe_len}]"
    elif isinstance(arg, tuple):
        items = ", ".join(
            _format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len
        )
        maybe_len = (
            "" if len(arg) < max_list_len + 1 else f", ...[total_len={len(arg)}]"
        )
        maybe_comma = "," if len(arg) == 1 else ""
        return f"({items}{maybe_comma}{maybe_len})"
    elif isinstance(arg, dict):
        items_str = ", ".join(f"{k}: {_format_arg(v)}" for k, v in arg.items())
        return f"{{{items_str}}}"

    if isinstance(arg, Node):
        return "%" + str(arg)
    elif isinstance(arg, torch.device):
        return _device_annotation(arg)
    else:
        return str(arg)


@compatibility(is_backward_compatible=True)
class Node(_NodeBase):
    # [CLASS DOC] A single operation (a value definition) in an fx Graph, implemented on top of the C++ _NodeBase.
    # State: graph, name, op (one of placeholder, get_attr, call_function, call_method, call_module, output), target, type (optional
    # return annotation), args and kwargs (tuple and dict of Arguments), users (ordered set of nodes that use this one), the private
    # _input_nodes (nodes it uses), neighbors _prev and _next in the graph's doubly linked list, _erased, and meta (a free-form
    # mapping preserved across copies). Assigning args or kwargs keeps use/user accounting up to date. Key methods: prepend, append,
    # replace_all_uses_with, replace_input_with, update_arg, insert_arg, update_kwarg, format_node, is_impure, normalized_arguments.
    """
    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *excluding the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """

    _args: tuple["Argument", ...]
    _kwargs: dict[str, "Argument"]
    graph: "Graph"
    # unique name of value being created
    name: str
    # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
    op: str
    # for method/module/function, the name of the method/module/function/attr
    # being invoked, e.g add, layer1, or torch.add
    target: "Target"
    # All `Node`-valued inputs. Key is the Node, value is don't-care.
    # The public API for this is `all_input_nodes`, this private attribute
    # should not be accessed directly.
    _input_nodes: dict["Node", None]
    # All of the nodes that use the value produced by this Node
    # Note one user may correspond to several uses, e.g. the node for ``x + x``
    # would appear once here, but represents two uses.
    # Is a dict to act as an "ordered set". Keys are significant, value don't-care
    users: dict["Node", None]
    # Type expression representing the output value of this node.
    # This should contain the same class of Type objects that would appear
    # as type annotations for function inputs/outputs.
    #
    # For placeholder nodes, this value will be used to type-annotate the
    # generated function parameters.
    # For the return node, this value will be used to type-annotate the
    # generated function return type. (Note this is a special case. ``return``
    # does not produce a value, it's more of a notation. Thus, this value
    # describes the type of args[0] in the ``return`` node.
    # TODO: narrow this to TensorType | _DynType | None
    type: Any | None
    _sort_key: Any
    # If set, use this fn to print this node
    _repr_fn: Callable[["Node"], str] | None
    # Dictionary to store metadata passes need to do their
    # transformations. This metadata is preserved across node copies
    meta: dict[str, Any]

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: "Target",
        args: tuple["Argument", ...],
        kwargs: dict[str, "Argument"],
        return_type: Any | None = None,
    ) -> None:
        # [FUNCTION DOC] Creates a node belonging to graph with the given name, opcode, target, args and kwargs and optional type annotation.
        # Validates the target: a callable for call_function, otherwise op must be legal and target a string; raises ValueError for a wrong
        # target type and AssertionError for an illegal op. Initializes the base class and records args/kwargs (updating users).
        """
        Instantiate an instance of ``Node``. Note: most often, you want to use the
        Graph APIs, i.e. ``Graph.call_module``, ``Graph.call_method``, etc. rather
        than instantiating a ``Node`` directly.

        Args:
            graph (Graph): The ``Graph`` to which this ``Node`` should belong.

            name (str): The name to which the output of this ``Node`` should be assigned

            op (str): The opcode for this ``Node``. Can be one of 'placeholder',
                'call_method', 'call_module', 'call_function', 'get_attr',
                'output'

            target ('Target'): The target this op should call. See the broader
                ``Node`` docstring for more details.

            args (Tuple['Argument']): The args to be passed to ``target``

            kwargs (Dict[str, 'Argument']): The kwargs to be passed to ``target``

            return_type (Optional[Any]): The python type expression representing the
                type of the output of this node. This field can be used for
                annotation of values in the generated code or for other types
                of analyses.
        """
        if op == "call_function":
            if not callable(target):
                raise ValueError(
                    f"Node [graph = {graph}, name = '{name}'] target {target} has type {torch.typename(target)} "
                    "but a Callable is expected"
                )
        else:
            if op not in _legal_ops:
                raise AssertionError(f"op '{op}' is not in _legal_ops")
            if not isinstance(target, str):
                raise ValueError(
                    f"Node [graph = {graph}, name = '{name}'] target {target} has type {torch.typename(target)} "
                    "but a str is expected"
                )
        super().__init__(graph, name, op, target, return_type)
        self._update_args_kwargs(args, kwargs)

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Pickle support: returns a dict of the instance dict plus all C-level node fields (graph, name, op, target, type,
        # sort key, args, kwargs, erased flag, prev/next links, input nodes, users, repr function and meta).
        return {
            **self.__dict__,
            "graph": self.graph,
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "type": self.type,
            "_sort_key": self._sort_key,
            "_args": self._args,
            "_kwargs": self._kwargs,
            "_erased": self._erased,
            "_prev": self._prev,
            "_next": self._next,
            "_input_nodes": self._input_nodes,
            "users": self.users,
            "_repr_fn": self._repr_fn,
            "meta": self.meta,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC] Pickle support: restores every entry of state as an attribute on the node.
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def next(self) -> "Node":
        # [FUNCTION DOC] Property: the next node in the graph's linked list.
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next

    @property
    def prev(self) -> "Node":
        # [FUNCTION DOC] Property: the previous node in the graph's linked list.
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev

    @compatibility(is_backward_compatible=True)
    def prepend(self, x: "Node") -> None:
        # [FUNCTION DOC] Moves node x to just before this node in the graph's node list (delegates to the base class; x must be in the same graph).
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """

        self._prepend(x)

    @compatibility(is_backward_compatible=True)
    def append(self, x: "Node") -> None:
        # [FUNCTION DOC] Moves node x to just after this node, implemented as prepending x to the next node.
        """
        Insert ``x`` after this node in the list of nodes in the graph.
        Equivalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """

        self._next._prepend(x)

    @property
    def args(self) -> tuple[Argument, ...]:
        # [FUNCTION DOC] Property: the tuple of positional arguments of this node.
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._args

    @args.setter
    def args(self, a: tuple[Argument, ...]) -> None:
        # [FUNCTION DOC] Setter: replaces the positional arguments with a, keeping the existing kwargs and updating use/user bookkeeping.
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `_update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.args = new_args`
        self._update_args_kwargs(a, self._kwargs)

    @property
    def kwargs(self) -> dict[str, Argument]:
        # [FUNCTION DOC] Property: the mapping of keyword arguments of this node.
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k: dict[str, Argument]) -> None:
        # [FUNCTION DOC] Setter: replaces the keyword arguments with k, keeping the existing args and updating use/user bookkeeping.
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `_update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.kwargs = new_kwargs`
        self._update_args_kwargs(self._args, k)

    @property
    def all_input_nodes(self) -> list["Node"]:
        # [FUNCTION DOC] Property: list of distinct nodes used by this node's args and kwargs, in order of first appearance.
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
        return list(self._input_nodes.keys())

    @compatibility(is_backward_compatible=True)
    def update_arg(self, idx: int, arg: Argument) -> None:
        # [FUNCTION DOC] Replaces the positional argument at index idx with arg by assigning a new args tuple (so bookkeeping is updated).
        # Raises IndexError if idx is out of range.
        """
        Update an existing positional argument to contain the new value
        ``arg``. After calling, ``self.args[idx] == arg``.

        Args:

            idx (int): The index into ``self.args`` of the element to update
            arg (Argument): The new argument value to write into ``args``
        """
        args = list(self.args)
        args[idx] = arg
        self.args = tuple(args)

    @compatibility(is_backward_compatible=True)
    def insert_arg(self, idx: int, arg: Argument) -> None:
        # [FUNCTION DOC] Inserts arg into the positional arguments before index idx (0 up to the current length inclusive).
        # Raises AssertionError if idx is out of range. Updates _args directly and adds any new Node inputs to the input and user sets.
        # KNOWN ISSUE: bypasses the args setter and the base class update path, so it does not go through the normal update logic.
        """
        Insert a positional argument to the argument list with given index.

        Args:

            idx (int): The index of the element in ``self.args`` to be inserted before.
            arg (Argument): The new argument value to insert into ``args``
        """
        if not (0 <= idx <= len(self.args)):
            raise AssertionError(
                f"insert_args index must be between 0 and len(self.args), got {idx}"
            )
        args_left = self.args[:idx]
        args_right = self.args[idx:]

        self._args = args_left + (arg,) + args_right

        _new_input_nodes: dict[Node, None] = {}
        _fx_map_arg(arg, _new_input_nodes.setdefault)

        for new_use in _new_input_nodes:
            if new_use not in self._input_nodes:
                self._input_nodes.setdefault(new_use)
                new_use.users.setdefault(self)

    @compatibility(is_backward_compatible=True)
    def update_kwarg(self, key: str, arg: Argument) -> None:
        # [FUNCTION DOC] Sets keyword argument key to arg (adding it if absent) by assigning a new kwargs mapping, so bookkeeping is updated.
        """
        Update an existing keyword argument to contain the new value
        ``arg``. After calling, ``self.kwargs[key] == arg``.

        Args:

            key (str): The key in ``self.kwargs`` of the element to update
            arg (Argument): The new argument value to write into ``kwargs``
        """
        self.kwargs = {**self.kwargs, key: arg}

    @property
    def stack_trace(self) -> str | None:
        # [FUNCTION DOC] Property: the recorded Python stack trace string from meta, or None if absent.
        """
        Return the Python stack trace that was recorded during tracing, if any.
        When traced with fx.Tracer, this property is usually populated by
        `Tracer.create_proxy`. To record stack traces during tracing for debug purposes,
        set `record_stack_traces = True` on the `Tracer` instance.
        When traced with dynamo, this property will be populated by default by
        `OutputGraph.create_proxy`.

        stack_trace would have the innermost frame at the end of the string.
        """
        return self.meta.get("stack_trace", None)

    @stack_trace.setter
    def stack_trace(self, trace: str | None) -> None:
        # [FUNCTION DOC] Setter: stores trace in meta under the stack_trace key.
        self.meta["stack_trace"] = trace

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the result of the custom repr function when one is set, otherwise the node's name.
        if self._repr_fn:
            return self._repr_fn(self)
        return self.name

    @staticmethod
    def _pretty_print_target(target: object) -> str:
        # [FUNCTION DOC] HELPER: Static method giving a readable target string: strings as is, builtins and operator functions with builtins or
        # operator prefix, everything else via the qualified-name helper.
        # KNOWN ISSUE: when a target has a module attribute but no name attribute, it falls back to the qualified-name helper, which itself needs the name attribute (see the TODO).
        """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualified name, e.g. torch.add
        """
        if isinstance(target, str):
            return target
        if hasattr(target, "__module__"):
            name = getattr(target, "__name__", None)
            if name is None:
                # Just to be defensive, if we don't have `__name__`, get the
                # qualname. Not sure if this happens for any members of `operator`
                # or `builtins`. This fallback path is not as good, since e.g.
                # things in `operator` have `_operator` as their __module__.
                # TODO: THIS IS BROKEN: _get_qualified_name calls `__name__`
                return _get_qualified_name(target)  # type: ignore[arg-type]
            if target.__module__ == "builtins":
                return f"builtins.{name}"
            elif target.__module__ == "_operator":
                return f"operator.{name}"
        return _get_qualified_name(target)  # type: ignore[arg-type]

    @compatibility(is_backward_compatible=True)
    def format_node(
        self,
        placeholder_names: list[str] | None = None,
        maybe_return_typename: list[str] | None = None,
        *,
        include_tensor_metadata: bool = False,
    ) -> str | None:
        # [FUNCTION DOC] Returns a one-line description of this node, e.g. a percent-prefixed name with type, user count, opcode, target and args/kwargs.
        # For a placeholder, if placeholder_names is a non-empty list it appends a signature entry to it and returns None; for an output node, fills the
        # first slot of maybe_return_typename with a return annotation when given. With include_tensor_metadata, call nodes show dtype, shape, stride and device
        # taken from the tensor metadata. Mutates the passed lists as side effect. Raises AssertionError if a placeholder target is not a string.
        # KNOWN ISSUE: the placeholder signature entry doubles the argument name when a type is present, and a non-empty-list test means an empty list is never appended to.
        """
        Return a descriptive string representation of ``self``.

        This method can be used with no arguments as a debugging
        utility.

        This function is also used internally in the ``__str__`` method
        of ``Graph``. Together, the strings in ``placeholder_names``
        and ``maybe_return_typename`` make up the signature of the
        autogenerated ``forward`` function in this Graph's surrounding
        GraphModule. ``placeholder_names`` and ``maybe_return_typename``
        should not be used otherwise.

        Args:
            placeholder_names: A list that will store formatted strings
                representing the placeholders in the generated
                ``forward`` function. Internal use only.
            maybe_return_typename: A single-element list that will store
                a formatted string representing the output of the
                generated ``forward`` function. Internal use only.
            include_tensor_metadata: Whether to include tensor metadata

        Returns:
            str: If 1) we're using ``format_node`` as an internal helper
                in the ``__str__`` method of ``Graph``, and 2) ``self``
                is a placeholder Node, return ``None``. Otherwise,
                return a  descriptive string representation of the
                current Node.
        """
        if self.op == "placeholder":
            if not isinstance(self.target, str):
                raise AssertionError(
                    f"Expected target to be str for placeholder, got {type(self.target)}"
                )
            arg_str = self.target
            arg_str += arg_str + f": {_type_repr(self.type)}" if self.type else ""
            if placeholder_names:
                placeholder_names.append(arg_str)
                return None
            maybe_typename = f"{_type_repr(self.type)} " if self.type else ""
            default_val = "(default=" + str(self.args[0]) + ")" if self.args else ""
            return f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = {self.op}[target={self.target}]{default_val}"
        elif self.op == "get_attr":
            maybe_typename = (
                f"{_type_repr(self.type)} " if self.type is not None else ""
            )
            return (
                f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = "
                f"{self.op}[target={self._pretty_print_target(self.target)}]"
            )
        elif self.op == "output":
            if self.type and maybe_return_typename:
                maybe_return_typename[0] = f" -> {_type_repr(self.type)}"
            return f"return {self.args[0]}"
        else:

            def stringify_shape(shape: Iterable[Any]) -> str:
            # [FUNCTION DOC] HELPER: Formats shape as a bracketed comma-separated string.
                return f"[{', '.join([str(x) for x in shape])}]"

            meta_val = self.meta.get(
                "val",
                self.meta.get("tensor_meta", self.meta.get("example_value", None)),
            )
            type_annotation = ""
            if (
                include_tensor_metadata
                and isinstance(meta_val, torch.Tensor)
                and meta_val.layout
                not in (
                    torch.sparse_csc,
                    torch.sparse_csr,
                )
            ):
                stride_annotation = f"{stringify_shape(meta_val.stride())}"
                device_annotation = _device_annotation(meta_val.device)
                type_annotation = (
                    f'Tensor "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}'
                    f'{stride_annotation}{device_annotation}"'
                )
            else:
                type_annotation = (
                    f"{_type_repr(self.type)} " if self.type is not None else ""
                )
            return (
                f"%{self.name} : {type_annotation}[num_users={len(self.users)}] = "
                f"{self.op}[target={self._pretty_print_target(self.target)}]("
                f"args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})"
            )

    @compatibility(is_backward_compatible=True)
    def replace_all_uses_with(
        self,
        replace_with: "Node",
        delete_user_cb: Callable[["Node"], bool] | None = None,
        *,
        propagate_meta: bool = False,
    ) -> list["Node"]:
        # [FUNCTION DOC] Makes every user of this node use replace_with instead. delete_user_cb, if given, is called per user and users for which it returns false are skipped.
        # With propagate_meta, copies this node's meta onto replace_with (raising AssertionError if replace_with already has meta).
        # Calls registered replace hooks of the owning module for each changed user. Returns the list of users that were modified.
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.
            delete_user_cb (Callable): Callback that is called to determine
              whether a given user of the self node should be removed.
            propagate_meta (bool): Whether or not to copy all properties
              on the .meta field of the original node onto the replacement node.
              For safety, this is only valid to do if the replacement node
              doesn't already have an existing .meta field.

        Returns:

            The list of Nodes on which this change was made.
        """
        if propagate_meta:
            if len(replace_with.meta) != 0:
                raise AssertionError(
                    "Called node.replace_all_uses_with(replace_with, propagate_meta=True), "
                    "but replace_with already has .meta keys"
                )
            for k, v in self.meta.items():
                replace_with.meta[k] = v
        to_process = [*self.users]
        replace_hooks = getattr(self.graph.owning_module, "_replace_hooks", None)
        result = []
        for use_node in to_process:
            if delete_user_cb is not None and not delete_user_cb(use_node):
                continue
            result.append(use_node)
            if replace_hooks:
                for replace_hook in replace_hooks:
                    replace_hook(old=self, new=replace_with.name, user=use_node)

            use_node._replace_input_with(self, replace_with)
        return result

    @compatibility(is_backward_compatible=False)
    def is_impure(self, impure_random: bool = True) -> bool:
        # [FUNCTION DOC] Returns whether the node must not be removed by dead code elimination. Placeholders and outputs are always impure; a call_module is impure if
        # its submodule has a truthy impure marker; call_function defers to the library impurity check (impure_random controls random ops); others are pure.
        # Raises AssertionError if a call_module node has no owning module or the submodule is missing.
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Args:
            impure_random (bool): Whether to treat rand op as impure.

        Returns:

            bool: If the op is impure or not.
        """
        # Placeholders and outputs are always impure for DCE purposes
        if self.op in {"placeholder", "output"}:
            return True

        # Check if an impure module.
        if self.op == "call_module":
            if self.graph.owning_module is None:
                raise AssertionError(
                    "self.graph.owning_module not set for purity check"
                )
            if not isinstance(self.target, str):
                raise AssertionError(
                    f"Expected str target for call_module, got {type(self.target)}"
                )
            target_mod = self.graph.owning_module.get_submodule(self.target)
            if target_mod is None:
                raise AssertionError(
                    f"Did not find expected submodule target {self.target}"
                )
            # NOTE: here we can end up considering GraphModule submodules pure,
            # even if they contain impure ops. It may not be safe to change
            # because this function is used by graph.eliminate_dead_code,
            # and some users depend on current elimination behavior.
            return getattr(target_mod, "_is_impure", False)

        # For call_function, delegate to the unified has_side_effects function
        if self.op == "call_function":
            from torch._library.utils import is_impure

            return is_impure(
                self.target,  # pyrefly: ignore[bad-argument-type]
                args=self.args,
                kwargs=self.kwargs,
                impure_random=impure_random,
            )

        return False

    @compatibility(is_backward_compatible=False)
    def normalized_arguments(
        self,
        root: torch.nn.Module,
        arg_types: tuple[Any] | None = None,
        kwarg_types: dict[str, Any] | None = None,
        normalize_to_only_use_kwargs: bool = False,
    ) -> ArgsKwargsPair | None:
        # [FUNCTION DOC] For call_function and call_module nodes, matches args and kwargs to the target's signature (filling defaults) and returns the pair,
        # optionally using only kwargs. arg_types and kwarg_types may disambiguate overloads. Returns None for other opcodes. root resolves module targets.
        # Raises AssertionError for a wrongly typed target.
        """
        Returns normalized arguments to Python targets. This means that
        `args/kwargs` will be matched up to the module/functional's
        signature and return exclusively kwargs in positional order
        if `normalize_to_only_use_kwargs` is true.
        Also populates default values. Does not support positional-only
        parameters or varargs parameters.

        Supports module calls.

        May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

        Args:
            root (torch.nn.Module): Module upon which to resolve module targets.
            arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
            kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
            normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

        Returns:

            Returns NamedTuple ArgsKwargsPair, or `None` if not successful.
        """
        if self.op == "call_function":
            if not callable(self.target):
                raise AssertionError(
                    f"Expected callable target, got {type(self.target)}"
                )
            return normalize_function(
                self.target,
                self.args,  # type: ignore[arg-type]
                self.kwargs,
                arg_types,
                kwarg_types,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )
        elif self.op == "call_module":
            if not isinstance(self.target, str):
                raise AssertionError(
                    f"Expected str target for call_module, got {type(self.target)}"
                )
            return normalize_module(
                root,
                self.target,
                self.args,  # type: ignore[arg-type]
                self.kwargs,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )

        return None

    @compatibility(is_backward_compatible=True)
    def replace_input_with(self, old_input: "Node", new_input: "Node") -> None:
        # [FUNCTION DOC] Replaces every use of old_input among this node's inputs with new_input, calling replace hooks of the owning module first (if any).
        """
        Loop through input nodes of ``self``, and replace all instances of
        ``old_input`` with ``new_input``.

        Args:

            old_input (Node): The old input node to be replaced.
            new_input (Node): The new input node to replace ``old_input``.
        """

        m = self.graph.owning_module
        if getattr(m, "_replace_hooks", None):
            for replace_hook in m._replace_hooks:
                replace_hook(old=old_input, new=new_input.name, user=self)

        self._replace_input_with(old_input, new_input)

    def _rename(self, candidate: str) -> None:
        # [FUNCTION DOC] HELPER: Renames the node to a unique name derived from candidate using the graph namespace; does nothing if the name is unchanged.
        if candidate == self.name:
            return
        name = self.graph._graph_namespace.create_name(candidate, None)
        self.name = name
        self.graph._graph_namespace._rename_object(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # [FUNCTION DOC] Attribute assignment hook. On renaming an existing name, notifies owning-module replace hooks for each user (AssertionError if the
        # new name is not a string). If the node is in the graph's find-nodes lookup table, removes it before the assignment and reinserts afterwards so the table stays consistent.
        if name == "name" and hasattr(self, "name"):
            m = self.graph.owning_module
            if getattr(m, "_replace_hooks", None):
                if not isinstance(value, str):
                    raise AssertionError(f"Expected value to be str, got {type(value)}")
                for user in self.users:
                    for replace_hook in m._replace_hooks:
                        replace_hook(old=self, new=value, user=user)
        update = False
        if (
            hasattr(self, name)
            and hasattr(self.graph, "_find_nodes_lookup_table")
            and self in self.graph._find_nodes_lookup_table
        ):
            update = True
            self.graph._find_nodes_lookup_table.remove(self)
        object.__setattr__(self, name, value)
        if update:
            self.graph._find_nodes_lookup_table.insert(self)


@compatibility(is_backward_compatible=True)
def map_arg(a: ArgumentT, fn: Callable[[Node], Argument]) -> ArgumentT:
    # [FUNCTION DOC] Applies fn to every Node found recursively inside a (lists, tuples, slices and dicts with string keys), returning a structure of the same shape.
    # Raises AssertionError if fn is not callable.
    """
    Apply fn recursively to each Node appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    if not callable(fn):
        raise AssertionError("torch.fx.map_arg(a, fn): fn must be a callable")
    return _fx_map_arg(a, fn)


@compatibility(is_backward_compatible=True)
def map_aggregate(a: ArgumentT, fn: Callable[[Argument], Argument]) -> ArgumentT:
    # [FUNCTION DOC] Applies fn to every leaf object found recursively inside a (lists, tuples, slices and dicts), returning a structure of the same shape.
    """
    Apply fn recursively to each object appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    return _fx_map_aggregate(a, fn)
