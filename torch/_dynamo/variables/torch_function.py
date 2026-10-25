# [MODULE DOC] Dynamo support for __torch_function__: tensor subclasses with overrides
# (TensorWithTFOverrideVariable), torch function mode variables and the symbolic mode stack,
# and the dispatch logic that emulates eager ordering of subclass/mode __torch_function__ calls.
# Import-time side effects: builds the operator lists (bin_ops etc.) and computes banned_attrs
# from torch.overrides; creates the module-level torch_function_mode_stack_state_mgr singleton.
"""TorchDynamo support for __torch_function__ tensor subclasses.

This module implements support for tensor subclasses with __torch_function__ overrides.
A tensor subclass instance is represented as a TensorWithTFOverrideVariable, which handles
dispatching __torch_function__ on attribute accesses, method calls, and torch API calls.

Unsupported features:
- Triggering __torch_function__ on tensor subclass non-tensor custom attributes
- Graph breaking on mutating guardable tensor properties within a __torch_function__ context
  (can cause excessive recompiles in certain cases)
- Matching exact eager behavior of ignoring __torch_function__ objects in non-tensor
  argument positions of Torch API calls

Supported features:
- Static method implementations of __torch_function__ on custom objects (triggers on torch
  API calls with the object as any argument)
- Triggering __torch_function__ on torch API calls with tensor subclass arguments
- __torch_function__ calls on base tensor attribute access and method calls for tensor
  subclass instances
- Matches dispatch ordering behavior of eager __torch_function__ with subclass/object
  arguments in any position

See https://docs.google.com/document/d/1WBxBSvW3NXhRp9ncmtokJloMLCtF4AYNhJaffvHe8Kw/edit#heading=h.vacn73lozd9w
for more information on the design.
"""

import collections
import contextlib
import inspect
import operator
from collections.abc import Generator, Iterable
from types import TracebackType
from typing import Any, TYPE_CHECKING

import torch._C
import torch.utils._pytree as pytree
from torch._guards import Source
from torch.overrides import (
    _get_overloaded_args,
    get_default_nowrap_functions,
    TorchFunctionMode,
)
from torch.utils._device import DeviceContext

from .. import graph_break_hints
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..polyfills import NoEnterTorchFunctionMode
from ..source import AttrSource, GlobalSource, TorchFunctionModeStackSource, TypeSource
from ..utils import (
    class_has_getattribute,
    clear_torch_function_mode_stack,
    get_safe_global_name,
    has_torch_function,
    is_tensor_base_attr_getter,
    set_torch_function_mode_stack,
)
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import GenericContextWrappingVariable
from .functions import UserMethodVariable
from .lazy import LazyVariableTracker
from .lists import TupleVariable
from .tensor import TensorSubclassVariable, TensorVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


bin_ops = [
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.lt,
    operator.gt,
    operator.ge,
    operator.le,
    operator.ne,
    operator.eq,
    operator.sub,
    operator.ipow,
    operator.imul,
    operator.imatmul,
    operator.ifloordiv,
    operator.itruediv,
    operator.imod,
    operator.iadd,
    operator.isub,
]

bin_int_ops = [
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
]

un_int_ops = [operator.invert]

tensor_and_int_ops = [
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
]

un_ops = [
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
]


banned_attrs = [
    fn.__self__.__name__  # type: ignore[attr-defined]
    for fn in get_default_nowrap_functions()
    if is_tensor_base_attr_getter(fn)
]


class TorchFunctionModeVariable(GenericContextWrappingVariable):
    # [CLASS DOC] Variable wrapping a TorchFunctionMode instance (or a placeholder with no value
    # for a device context inserted during tracing). Holds value, cm_obj (same object, for
    # context-manager compatibility) and source. Enter/exit push/pop the real mode stack via
    # in-graph calls; the mode is dispatched to through call_torch_function.
    @staticmethod
    def is_supported_torch_function_mode(ty: type[TorchFunctionMode]) -> bool:
        # [FUNCTION DOC] Returns True if graph breaks under a mode of type ty are supported:
        # the type is a NoEnterTorchFunctionMode or DeviceContext subclass, or it has no custom
        # attribute-lookup hook and keeps the default TorchFunctionMode enter and exit methods.
        # Args: ty, the mode class. Pure; no side effects.
        # Supported in this sense means we can support graph breaks under the
        # context.
        # We are able to trace custom modes but if there are graph breaks under them
        # and they have a custom __enter__/__exit__ we don't handle this for the
        # same reason we don't handle generic context managers: there may be side effects
        # that are now affected by executing the function across two frames instead of one
        # Today we support the enter/exit of the default TorchFunctionMode as well as
        # DeviceContext (which is used for set_default_device)
        return issubclass(ty, (NoEnterTorchFunctionMode, DeviceContext)) or (
            not class_has_getattribute(ty)
            and inspect.getattr_static(ty, "__enter__") is TorchFunctionMode.__enter__
            and inspect.getattr_static(ty, "__exit__") is TorchFunctionMode.__exit__
        )

    def __init__(
        self,
        value: TorchFunctionMode | None,
        source: Source | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the mode value (may be None for a device-context placeholder),
        # its source, and aliases cm_obj to the value. Only calls the parent initializer when
        # value is not None; extra kwargs are forwarded to it.
        if value is not None:
            super().__init__(value, **kwargs)
        self.value = value
        # needed for BC with calling enter from CM code
        self.cm_obj = value  # type: ignore[assignment]
        self.source = source  # type: ignore[assignment]

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that reloads the mode from its source.
        # Raises an AssertionError if the variable has no source.
        # This shouldn't be called unless we have a source
        if not self.source:
            raise AssertionError(
                "TorchFunctionModeVariable requires a source for reconstruct"
            )
        self.source.reconstruct(codegen)

    def module_name(self) -> str:
        # [FUNCTION DOC] Returns the module name of the wrapped mode object.
        return self.value.__module__

    def fn_name(self) -> str:
        # [FUNCTION DOC] Returns the class name of the wrapped mode object.
        return type(self.value).__name__

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the class of the wrapped mode object.
        return type(self.value)

    def call_torch_function(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        types: TupleVariable,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        # [FUNCTION DOC] Invokes this mode's __torch_function__ symbolically with the given
        # function, types tuple, args and kwargs, via the module-level call_torch_function.
        # Returns the resulting VariableTracker.
        return call_torch_function(
            tx,
            get_torch_function_fn(tx, self),  # type: ignore[arg-type]
            fn,
            types,
            args,
            kwargs,
        )

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Symbolically enters the mode: pushes it onto the torch function stack
        # via an in-graph call (no-op for NoEnterTorchFunctionMode). Returns a None constant.
        from .torch import TorchInGraphFunctionVariable

        if isinstance(self.value, NoEnterTorchFunctionMode):
            return ConstantVariable.create(None)

        TorchInGraphFunctionVariable(
            torch._C._push_on_torch_function_stack
        ).call_function(tx, [self], {})
        return ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslatorBase", *args: Any) -> VariableTracker:
        # [FUNCTION DOC] Symbolically exits the mode by popping the torch function stack via an
        # in-graph call. Extra args (exception info) are ignored. Returns a None constant.
        # KNOWN ISSUE: pops unconditionally, even when enter was a no-op for NoEnterTorchFunctionMode.
        from .torch import TorchInGraphFunctionVariable

        TorchInGraphFunctionVariable(torch._C._pop_torch_function_stack).call_function(
            tx, [], {}
        )
        return ConstantVariable.create(None)

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode loading the NoEnterTorchFunctionMode class (looked up by
        # module import and attribute), regardless of the wrapped mode's actual type.
        ty = NoEnterTorchFunctionMode
        codegen(
            AttrSource(
                codegen.tx.import_source(ty.__module__),
                ty.__name__,
            )
        )

    def supports_graph_breaks(self) -> bool:
        # [FUNCTION DOC] Always True: graph breaks under this context are supported.
        return True

    def exit_on_graph_break(self) -> bool:
        # [FUNCTION DOC] Always False: the mode is not exited when a graph break occurs.
        return False


# Used to clear/restore the python torch function mode stack and temporarily restore it as needed
class TorchFunctionModeStackStateManager:
    # [CLASS DOC] Context manager that saves the current python torch function mode stack in
    # self.stack and clears the real stack on entry, restoring it on exit. Also offers
    # temp_restore_stack to temporarily reinstall the saved stack.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes with an empty saved stack.
        self.stack: list[Any] = []

    def __enter__(self) -> None:
        # [FUNCTION DOC] Saves the current mode stack and clears the global mode stack (global
        # side effect). Returns nothing.
        self.stack = torch.overrides._get_current_function_mode_stack()
        clear_torch_function_mode_stack()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Restores the saved mode stack as the global stack and forgets it.
        # Exception arguments are ignored; exceptions are not suppressed.
        set_torch_function_mode_stack(self.stack)
        self.stack = []

    @contextlib.contextmanager
    def temp_restore_stack(self) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager: installs the saved stack as the global mode stack for
        # the duration of the block, then reinstalls the previous stack even if an error occurs.
        prev = torch.overrides._get_current_function_mode_stack()
        set_torch_function_mode_stack(self.stack)
        try:
            yield
        finally:
            set_torch_function_mode_stack(prev)


torch_function_mode_stack_state_mgr = TorchFunctionModeStackStateManager()


class SymbolicTorchFunctionState:
    # [CLASS DOC] Dynamo's symbolic view of torch function state during tracing: whether
    # subclass dispatch and mode dispatch are enabled, the skip_next flag, the current mode
    # being inlined (cur_mode), and mode_stack, a deque of TorchFunctionModeVariable.
    def __init__(self, py_stack: Iterable[Any], skip_next: bool) -> None:
        # [FUNCTION DOC] Snapshots the C-level enablement flags, records skip_next, resets the
        # TorchFunctionModeStackVariable offset (global class state), and builds mode_stack from
        # py_stack as lazy variables each sourced by its stack index.
        # This is annoyingly complicated because of how the torch function subclass + mode C API was designed
        # There are two exposed C knobs here as contexts: torch._C.DisableTorchFunction and torch._C.DisableTorchFunctionSubclass
        # These are their definitions:
        # 1) torch._C._is_torch_function_enabled indicates that neither of the above knobs have been entered
        # (if either are entered, this will be False)
        # 2) torch._C._is_torch_function_mode_enabled indicates that either the torch mode stack is empty OR
        # torch._C.DisableTorchFunction has been entered
        # To disambiguate these and keep myself sane I added a C API to check whether all torch function
        # concepts (modes and subclasses) are enabled.
        # This only returns true iff we have not entered torch._C.DisableTorchFunction and allows us to separate
        # the stack length from the enablement state of torch function modes.
        # This is important because now if a mode is pushed while dynamo is tracing, we know whether
        # or not torch function modes are enabled and whether we should trace it.
        self.torch_function_subclass_enabled = torch._C._is_torch_function_enabled()

        # Mirrors PythonTorchFunctionTLS::skip_next in C++.
        # Set by _skip_one_hop_torch_function, consumed by has_torch_function.
        # The real TLS flag is consumed by save_global_state in
        # OutputGraph.__init__ (which runs before this) and passed in here.
        # On tracing failure, restore_graphstate restores the real TLS.
        self.skip_next = skip_next

        # This differs from the C API of the same name
        # this will only be false iff we have entered torch._C.DisableTorchFunction
        # and does not take into account the mode stack length, while the C API bundles these
        # two concepts
        self.torch_function_mode_enabled = (
            not torch._C._is_torch_function_all_disabled()
        )

        self.cur_mode = None

        TorchFunctionModeStackVariable.reset()

        self.mode_stack: collections.deque[TorchFunctionModeVariable] = (
            collections.deque()
        )

        for i, val in enumerate(py_stack):
            self.mode_stack.append(
                LazyVariableTracker.create(val, source=TorchFunctionModeStackSource(i))  # type: ignore[arg-type]
            )

    def in_torch_function_mode(self) -> bool:
        # [FUNCTION DOC] True if the symbolic mode stack is non-empty.
        return len(self.mode_stack) > 0

    def pop_torch_function_mode(self) -> TorchFunctionModeVariable:
        # [FUNCTION DOC] Removes and returns the top mode of the symbolic stack.
        # Raises an error if the stack is empty.
        return self.mode_stack.pop()

    def push_torch_function_mode(self, mode_var: TorchFunctionModeVariable) -> None:
        # [FUNCTION DOC] Pushes mode_var onto the top of the symbolic stack.
        self.mode_stack.append(mode_var)

    def call_torch_function_mode(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        types: TupleVariable,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        # [FUNCTION DOC] Dispatches to the top mode: temporarily pops it (so nested torch calls
        # see only the modes below it), calls its __torch_function__ symbolically, and returns
        # the result. The mode is pushed back afterwards, even on error.
        with self._pop_mode_for_inlining() as cur_mode:
            return cur_mode.call_torch_function(tx, fn, types, args, kwargs)

    @contextlib.contextmanager
    def _pop_mode_for_inlining(
        self,
    ) -> Generator[TorchFunctionModeVariable, None, None]:
        # [FUNCTION DOC] HELPER: context manager that pops the top mode, records it as cur_mode
        # (saving the previous cur_mode), yields it, then restores cur_mode and re-pushes the mode.
        old_mode = self.cur_mode
        self.cur_mode = self.pop_torch_function_mode()  # type: ignore[assignment]
        try:
            yield self.cur_mode  # type: ignore[misc]
        finally:
            mode = self.cur_mode
            self.cur_mode = old_mode
            self.push_torch_function_mode(mode)  # type: ignore[arg-type]


class TorchFunctionModeStackVariable(VariableTracker):
    # [CLASS DOC] Dummy variable standing for the global torch function mode stack in the side
    # effect tracker, so stack mutations get replayed. Holds a source and the symbolic stack.
    # Class-level state: stack_value_singleton (tracking key) and offset (index shift caused by
    # inserting or removing a device context at the stack bottom).
    """Fake VT to use as a dummy object, indicating the presence of torch function mode stack mutation"""

    # singleton value representing the global torch function mode stack
    # singleton (it exists in C++)
    stack_value_singleton = object()

    # offset is used to track if we have inserted/removed a
    # device context which is always placed at the bottom of the stack
    # if a device context is inserted, the graph will run this mutation
    # so when we want to reconstruct any other modes on the stack
    # their indices should be shifted right by 1 (+1)
    # Conversely, if there was a device context on the stack, and the graph
    # mutates the stack to remove that context (set default device to None)
    # each of the indices of other modes should be shifted left by 1 (-1)
    offset = 0

    def __init__(
        self,
        source: Source,
        symbolic_stack: collections.deque[TorchFunctionModeVariable],
    ) -> None:
        # [FUNCTION DOC] Stores the source and the symbolic stack reference. Does not call the
        # parent initializer.
        self.source = source
        self.symbolic_stack = symbolic_stack

    @classmethod
    def reset(cls) -> None:
        # [FUNCTION DOC] Resets the class-level device-context index offset to zero.
        cls.offset = 0

    @classmethod
    def register_mutation(cls, tx: "InstructionTranslatorBase") -> None:
        # [FUNCTION DOC] Ensures the mode stack is tracked as mutated in the output graph's side
        # effects: on first call creates the dummy variable, tracks it and marks it mutated;
        # later calls do nothing.
        if cls.stack_value_singleton not in tx.output.side_effects:
            var = cls(
                source=Source(),
                symbolic_stack=tx.symbolic_torch_function_state.mode_stack,
            )
            tx.output.side_effects.track_mutable(cls.stack_value_singleton, var)
            tx.output.side_effects.mutation(var)

    @classmethod
    def register_device_context_insertion(cls, tx: "InstructionTranslatorBase") -> None:
        # [FUNCTION DOC] Records a default-device context being inserted: unless the bottom of
        # the symbolic stack is already a device context, increments offset and inserts a
        # placeholder mode variable (no value) at the bottom, sourced by the negative offset.
        stack = tx.symbolic_torch_function_state.mode_stack
        if stack and cls.is_device_context(stack[0]):
            return
        else:
            cls.offset += 1
            stack.insert(
                0,
                TorchFunctionModeVariable(
                    None, source=TorchFunctionModeStackSource(-cls.offset)
                ),
            )

    @classmethod
    def clear_default_device(cls, tx: "InstructionTranslatorBase") -> None:
        # [FUNCTION DOC] If the bottom of the symbolic stack is a device context, removes it and
        # decrements offset; otherwise does nothing.
        stack = tx.symbolic_torch_function_state.mode_stack
        if stack and cls.is_device_context(stack[0]):
            stack.popleft()
            cls.offset -= 1

    @staticmethod
    def is_device_context(var: TorchFunctionModeVariable) -> bool:
        # [FUNCTION DOC] True if the mode variable wraps a DeviceContext or is a valueless
        # placeholder (which stands for an inserted device context).
        return isinstance(var.value, DeviceContext) or var.value is None

    @classmethod
    def get_mode_index(cls, ind: int) -> int:
        # [FUNCTION DOC] Returns the stack index ind shifted by the current device-context offset.
        return ind + cls.offset


def _get_all_args(
    args: Iterable[Any], kwargs: dict[str, Any]
) -> Iterable[VariableTracker]:
    # [FUNCTION DOC] HELPER: collects all leaf arguments from args and kwargs using a pytree
    # flatten, then expands list and dict variables via _flatten_vts. Returns a list of variables.
    return _flatten_vts(pytree.arg_tree_leaves(*args, **kwargs))


def _flatten_vts(vts: Iterable[VariableTracker]) -> list[VariableTracker]:
    # [FUNCTION DOC] HELPER: breadth-first flattening of variables. Lazy variables whose type is
    # a dict, list or tuple are realized first; realized list variables contribute their items
    # and constant-dict variables their values (recursively); everything else is output as is.
    # Returns the flat list; realizing lazy variables is a side effect.
    from collections import deque

    from .dicts import ConstDictVariable
    from .lists import ListVariable

    vts = deque(vts)
    output = []

    while vts:
        vt = vts.popleft()

        if not vt.is_realized() and vt.peek_type() in (dict, list, tuple):  # type: ignore[attr-defined]
            vt.realize()

        if vt.is_realized():
            if isinstance(vt, ListVariable):
                vts.extend(vt.items)
                continue
            elif isinstance(vt, ConstDictVariable):
                vts.extend(vt.items.values())
                continue

        output.append(vt)

    return output


def _get_subclass_type(var: VariableTracker) -> type:
    # [FUNCTION DOC] HELPER: returns the Python type of a tensor-with-override or user-defined
    # object variable. Raises an AssertionError for any other variable kind.
    if not isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable)):
        raise AssertionError(f"Unexpected type {type(var)}")
    return var.python_type()


def _get_subclass_type_var(
    tx: "InstructionTranslatorBase", var: VariableTracker
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: returns a variable representing the class of var. For tensor
    # subclass variables uses class_type_var; for user-defined objects builds a variable for
    # the type (with a type source when the object has a source). Raises AssertionError otherwise.
    if isinstance(var, TensorWithTFOverrideVariable):
        return var.class_type_var(tx)
    elif isinstance(var, UserDefinedObjectVariable):
        source = var.source and TypeSource(var.source)
        return VariableTracker.build(tx, var.python_type(), source)
    else:
        raise AssertionError(f"Unexpected type {type(var)}")


def _is_attr_overridden(
    tx: "InstructionTranslatorBase", var: VariableTracker, name: str
) -> bool:
    # [FUNCTION DOC] HELPER: True if the variable's class defines attribute name differently
    # from torch.Tensor (static lookup on the class compared to the Tensor attribute). Returns
    # False for other variable kinds or if the class lacks the attribute.
    # KNOWN ISSUE: if torch.Tensor itself lacks the attribute, the Tensor lookup raises an
    # AttributeError that is swallowed, yielding False even for a class-defined attribute.
    if not isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable)):
        return False
    import torch

    overridden = False
    try:
        attr_val = inspect.getattr_static(var.python_type(), name)
        overridden |= attr_val != getattr(torch.Tensor, name)
    except AttributeError:
        pass

    return overridden


def call_torch_function(
    tx: "InstructionTranslatorBase",
    torch_function_var: VariableTracker,
    fn: VariableTracker,
    types: TupleVariable,
    args: Iterable[Any],
    kwargs: dict[str, Any],
    *,
    is_subclass_dispatch: bool = False,
) -> Any:
    # [FUNCTION DOC] Symbolically calls a __torch_function__ implementation (torch_function_var)
    # with the function, types tuple, and args/kwargs packed as tuple and dict variables.
    # For subclass dispatch (is_subclass_dispatch), temporarily disables subclass dispatch in the
    # symbolic state during the call to prevent re-entrant dispatch; the flag is restored after,
    # even on error. Returns the variable produced by the call.
    # This emulates calling __torch_function__, which has a signature
    #   def __torch_function__(cls, func, types, args=(), kwargs=None):
    #
    # Also notice the `cls` is not explicitly passed in the reference
    # implementations:
    # 1. https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/csrc/utils/python_arg_parser.cpp#L368-L374
    # 2. https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/overrides.py#L1741-L1743
    tf_args = [
        fn,
        types,
        VariableTracker.build(tx, tuple(args)),
        VariableTracker.build(tx, kwargs),
    ]
    # Mirror the C++ THPModule_disable_torch_function behavior: disable
    # __torch_function__ subclass dispatch during the call to prevent
    # re-entrant dispatch on operations inside __torch_function__.
    # Only do this for subclass dispatch, not mode dispatch. Modes need
    # subclass dispatch to remain enabled because the mode's
    # __torch_function__ may re-dispatch to the subclass.
    tf_state = tx.symbolic_torch_function_state
    old_subclass_enabled = tf_state.torch_function_subclass_enabled
    if is_subclass_dispatch and old_subclass_enabled:
        tf_state.torch_function_subclass_enabled = False
    try:
        return torch_function_var.call_function(tx, tf_args, {})
    finally:
        tf_state.torch_function_subclass_enabled = old_subclass_enabled


def get_torch_function_fn(
    tx: "InstructionTranslatorBase", vt: VariableTracker
) -> VariableTracker:
    # [FUNCTION DOC] Returns a variable for the __torch_function__ attribute of vt, obtained by
    # symbolically calling the builtin attribute getter.
    # The underlying function could be a classmethod, staticmethod, regular
    # function or a function with C-implementation. It doesn't matter as long as
    # they satisfy the calling convention in `call_torch_function`.

    args = [vt, VariableTracker.build(tx, "__torch_function__")]
    func_vt = VariableTracker.build(tx, getattr).call_function(tx, args, {})
    return func_vt


def can_dispatch_torch_function(
    tx: "InstructionTranslatorBase", args: Iterable[Any], kwargs: dict[str, Any]
) -> bool:
    # [FUNCTION DOC] Decides whether a torch API call needs __torch_function__ dispatch: True if
    # any flattened argument has an override and subclass dispatch is enabled, or if mode dispatch
    # is enabled and the symbolic mode stack is non-empty. Pure query.
    has_overridden_args = any(
        has_torch_function(arg) for arg in _get_all_args(args, kwargs)
    )
    tf_state = tx.symbolic_torch_function_state
    return (has_overridden_args and tf_state.torch_function_subclass_enabled) or (
        tf_state.torch_function_mode_enabled and tf_state.in_torch_function_mode()
    )


def dispatch_torch_function(
    tx: "InstructionTranslatorBase",
    fn: VariableTracker,
    args: Iterable[Any],
    kwargs: dict[str, Any],
) -> Any:
    # [FUNCTION DOC] Emulates eager __torch_function__ dispatch for fn. Finds args with overrides,
    # orders them with the eager overload-ordering helper, and builds the types tuple. First tries
    # the top mode (if any); then each overloaded arg in order; the first result that is not
    # NotImplemented is returned (marking torch_function_subclass_inlined on the output graph for
    # subclass results). If all return NotImplemented, triggers a graph break via unimplemented.
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = _get_all_args(args, kwargs)
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if has_torch_function(arg)],
        _get_subclass_type,
    )

    types = TupleVariable([_get_subclass_type_var(tx, arg) for arg in overloaded_args])

    if tx.symbolic_torch_function_state.in_torch_function_mode():
        res = tx.symbolic_torch_function_state.call_torch_function_mode(
            tx, fn, types, args, kwargs
        )
        if not res.is_constant_match(NotImplemented):
            return res

    for arg in overloaded_args:
        res = arg.call_torch_function(
            tx,
            fn,
            types,
            args,
            kwargs,
        )

        if not res.is_constant_match(NotImplemented):
            tx.output.torch_function_subclass_inlined = True
            return res

    unimplemented(
        gb_type="All __torch_function__ overrides returned NotImplemented due to TypeError from user code",
        context=f"{fn=}, {args=}, {kwargs=}",
        explanation=f"All __torch_function__ overrides for function {fn} returned NotImplemented",
        hints=[
            *graph_break_hints.USER_ERROR,
        ],
    )


class TensorWithTFOverrideVariable(TensorVariable):
    # [CLASS DOC] TensorVariable for an instance of a tensor subclass with a __torch_function__
    # override. class_type is the subclass; attribute access and method calls dispatch through
    # __torch_function__, and the subclass is installed as a global for output rewrapping.
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    @classmethod
    def from_tensor_var(
        cls,
        tx: "InstructionTranslatorBase",
        tensor_var: VariableTracker,
        class_type: type,
        cls_source: Source | None,
    ) -> "TensorWithTFOverrideVariable":
        # [FUNCTION DOC] Converts a tensor variable into an override variable of class_type, by
        # realizing it, shallow-copying its fields (minus its class_type) into a new instance, and
        # installing the subclass as a global. Raises AssertionError if the original class is not a
        # tensor type. cls_source is unused.
        # [Note: __torch_function__] coerce `tensor_var` into a
        # TensorWithTFOverrideVariable. In eager, this is just a type change.
        import torch

        tensor_var = tensor_var.realize()
        # This simulates shallow-copying the tensor object.
        kwargs = dict(tensor_var.__dict__)
        input_tensor_type = kwargs.pop("class_type")
        if not (
            input_tensor_type in (torch.Tensor, torch.nn.Parameter)
            or issubclass(input_tensor_type, torch.Tensor)
        ):
            raise AssertionError(
                f"invalid class type {input_tensor_type} in TensorWithTFOverrideVariable.from_tensor_var"
            )
        var = cls(class_type=class_type, **kwargs)
        var.install_global(tx)
        return var

    def install_global(self, tx: "InstructionTranslatorBase") -> None:
        # [FUNCTION DOC] Ensures the subclass is available in the output graph's global scope
        # under its mangled name (installing it if absent), so compiled code can rewrap outputs.
        # stash the subclass type to rewrap an output tensor if needed
        # this is needed because the actual type needs to be available
        # each time the compiled artifact is run and outputs a wrapped tensor.
        if self.global_mangled_class_name(tx) not in tx.output.global_scope:
            # Safe because global_mangled_class_name figures it out
            tx.output.install_global_unsafe(
                self.global_mangled_class_name(tx), self.class_type
            )

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the tensor subclass.
        return self.class_type

    def class_type_var(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns a subclass variable for class_type, sourced from its installed global.
        return TensorSubclassVariable(
            self.class_type, source=GlobalSource(self.global_mangled_class_name(tx))
        )

    def global_mangled_class_name(self, tx: "InstructionTranslatorBase") -> str:
        # [FUNCTION DOC] Returns the safe global name used to store class_type in the output
        # graph's globals, derived from a prefix plus the class name.
        return get_safe_global_name(
            tx, f"__subclass_{self.class_type.__name__}", self.class_type
        )

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute lookup on the subclass instance. Graph breaks for banned
        # attributes (tensor base attribute getters) and for overridden attributes that are not a
        # function, property or classmethod. Non-overridden plain Tensor attributes dispatch via
        # __torch_function__ when dispatch is active. Overridden functions become bound user
        # methods (with a closure guard), properties inline the getter, classmethods bind the class.
        # Otherwise falls back to the TensorVariable implementation.
        # [Note: __torch_function__] We currently only support attributes that are defined on
        # base tensors, custom attribute accesses will graph break.
        import torch

        # I think only `_base` is breaking because we aren't modelling view
        # relationship perfectly in some scenarios.
        if name in banned_attrs:
            unimplemented(
                gb_type="Unsupported tensor subclass attribute access",
                context=f"{name}",
                explanation="`torch.compile` currently can't trace this",
                hints=[
                    f"Avoid accessing {name} of tensor subclass in torch.compile region",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Handle non-overridden attributes inherited from `torch.Tensor`.
        attr_is_overridden = _is_attr_overridden(tx, self, name)
        if (
            hasattr(torch.Tensor, name)
            and not attr_is_overridden
            and not inspect.ismethoddescriptor(getattr(torch.Tensor, name))
        ):
            args = [self]
            kwargs: dict[Any, Any] = {}
            if can_dispatch_torch_function(tx, args, kwargs):
                get_fn = VariableTracker.build(tx, getattr(torch.Tensor, name).__get__)

                return self.call_torch_function(
                    tx,
                    get_fn,
                    TupleVariable([self.class_type_var(tx)]),
                    args,
                    kwargs,
                )
        else:
            # `TensorVariable.tp_getattro_impl` doesn't handle user-defined
            # function/attribute well, so we explicitly handle them here.
            #
            # TODO move this logic into `TensorVariable`, or try to merge it
            # with similar logic in `UserDefinedObjectVariable`.
            try:
                attr = inspect.getattr_static(self.class_type, name)
            except AttributeError:
                pass
            else:
                import types

                cls_source = GlobalSource(self.global_mangled_class_name(tx))
                attr_source = AttrSource(cls_source, name)
                if isinstance(attr, types.FunctionType):
                    install_guard(attr_source.make_guard(GuardBuilder.CLOSURE_MATCH))
                    return UserMethodVariable(attr, self)

                elif isinstance(attr, property):
                    getter_source = AttrSource(attr_source, "fget")
                    getter = attr.fget
                    getter_var = VariableTracker.build(
                        tx, getter, source=getter_source, realize=True
                    )
                    return getter_var.call_function(tx, [self], {})

                elif isinstance(attr, classmethod):
                    return UserMethodVariable(
                        attr.__func__, self.class_type_var(tx), source=attr_source
                    )

                elif attr_is_overridden:
                    unimplemented(
                        gb_type="Unsupported tensor subclass overridden attribute access",
                        context=f"{name}",
                        explanation="`torch.compile` only support tracing certain types of overridden tensor subclass attributes",
                        hints=[
                            f"Avoid accessing {name} of tensor subclass in torch.compile region",
                            f"Renaming attribute `{name}` of type {self.class_type}",
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )

        return super().tp_getattro_impl(tx, name)

    def call_torch_function(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        types: TupleVariable,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        # [FUNCTION DOC] Calls this instance's __torch_function__ symbolically as subclass
        # dispatch. Caches the resolved __torch_function__ variable on the instance on first use
        # (assuming it is not modified during tracing). Returns the call result.
        # NOTE this assumes `__torch_function__` isn't modified during tracing.
        if not hasattr(self, "torch_function_fn"):
            self.torch_function_fn = get_torch_function_fn(tx, self)

        return call_torch_function(
            tx,
            self.torch_function_fn,
            fn,
            types,
            args,
            kwargs,
            is_subclass_dispatch=True,
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # [FUNCTION DOC] Method call on the instance. If torch function dispatch applies, graph
        # breaks when the method is overridden by the subclass; otherwise resolves the Tensor
        # implementation (class attribute guarded via source when available) and dispatches it
        # through dispatch_torch_function with self prepended to args. If dispatch does not apply,
        # defers to the parent implementation.
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        tf_args = [self] + list(args)
        if can_dispatch_torch_function(tx, tf_args, kwargs):
            import torch

            if _is_attr_overridden(tx, self, name):
                unimplemented(
                    gb_type="Tensor subclass overridden method call",
                    context=f"{name}",
                    explanation="`torch.compile` currently can't trace this",
                    hints=[
                        f"Avoid calling {name} of tensor subclass in torch.compile region",
                        f"Renaming method `{name}` of type {self.class_type}",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            # [Note: __torch_function__] Currently we only support methods that are defined on tensor
            # we will graph break in other cases this will need a bigger overhaul of extracting methods/comparing them for equality
            # We've established with the above check that the method is not overridden, so we guard that the method is the same
            # as the impl defined on tensor and retrieve it
            if self.source:
                source = AttrSource(AttrSource(self.source, "__class__"), name)
                value = inspect.getattr_static(self.python_type(), name)
            else:
                source = None
                value = getattr(torch.Tensor, name)
            func_var = VariableTracker.build(tx, value, source)
            return dispatch_torch_function(tx, func_var, tf_args, kwargs)
        else:
            return super().call_method(tx, name, args, kwargs)
