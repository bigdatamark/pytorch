# [MODULE DOC] Dynamo VariableTrackers for TorchScript / torchbind / opaque custom-class objects.
# CustomClassVariable represents the class itself (constructing it yields an object VT, rejecting
# reference-type opaque classes inside the compiled region). CustomClassObjectVariable represents
# an instance: it holds an FX proxy plus a real or fake value, dispatches registered members
# either by calling the real object (USE_REAL), by emitting graph call_method nodes (INLINED),
# or via the call_torchbind higher-order op, and graph-breaks on unregistered members. A
# decorator turns graph breaks in these paths into hard UnsafeScriptObjectError errors.
# No import-time side effects.
"""
This module implements variable tracking for TorchScript objects during Dynamo tracing.

The CustomClassObjectVariable class provides specialized handling for TorchScript
objects with strong safety guarantees by:
- Enforcing method-call-only access to prevent unsafe attribute manipulation
- Converting graph breaks into hard errors via _raise_hard_error_if_graph_break
- Proper proxy and source tracking for TorchScript method calls
- Integration with higher-order operators for method call handling

Key safety features:
- Strict validation that only method calls are allowed (no direct attribute access)
- Immediate error reporting for potentially unsafe operations
- Proper source tracking for debugging and guard installation
- Safe handling of TorchScript object method calls through torchbind

The module ensures that TorchScript objects are handled safely during tracing
by limiting operations to known-safe patterns and failing fast for unsafe usage.
"""

import enum
import functools
import inspect
import types
from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import (
    get_member_type,
    is_custom_class,
    is_opaque_constant_type,
    is_opaque_symbolic_type,
    MemberType,
    should_hoist,
)
from torch.fx.proxy import Proxy

from .. import graph_break_hints
from ..eval_frame import skip_code
from ..exc import (
    raise_observed_exception,
    unimplemented,
    UnsafeScriptObjectError,
    Unsupported,
)
from ..source import AttrSource
from ..utils import proxy_args_kwargs
from .base import VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import TupleVariable
from .misc import LambdaVariable
from .user_defined import UserDefinedObjectVariable, UserDefinedVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _raise_hard_error_if_graph_break(
    reason: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] HELPER: Decorator factory. The returned decorator wraps a function so that
    # an Unsupported (graph break) raised inside becomes an UnsafeScriptObjectError carrying the
    # original message. KNOWN ISSUE: the reason argument is accepted but never used.
    def deco(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: wrap fn (preserving its metadata) with the conversion described
        # on the enclosing factory.
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # [FUNCTION DOC] HELPER: call fn; convert Unsupported into UnsafeScriptObjectError
            # (chained from the original); any other exception propagates.
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                raise UnsafeScriptObjectError(e.msg) from e

        return graph_break_as_hard_error

    return deco


class CustomClassVariable(UserDefinedVariable):
    # [CLASS DOC] VT for an opaque/custom class object (the class, not an instance). State:
    # value is the real class. Not treated as a python constant so attribute access goes
    # through tp_getattro_impl validation. Calling it constructs the real object (outside
    # tracing) and wraps it in a CustomClassObjectVariable. Key methods: call_function,
    # tp_getattro_impl, nb_or_impl.
    """
    A variable that represents an opaque object class (not instance).
    Since UserDefinedClassVariable has some special handling for side effects,
    we have a separate class here which will directly return the object when
    __init__ is called.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Store the wrapped class. Raises AssertionError if value is an Enum class
        # (those must use UserDefinedClassVariable).
        if isinstance(value, type) and issubclass(value, enum.Enum):
            raise AssertionError(
                f"Enum class {value} should use UserDefinedClassVariable, "
                "not CustomClassVariable"
            )
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Return the wrapped real class.
        return self.value

    def is_python_constant(self) -> bool:
        # [FUNCTION DOC] Always False, to block constant folding of attribute accesses on opaque
        # classes so tp_getattro_impl validation runs.
        # prevents constant folding of attribute accesses on
        # opaque classes. this ensures tp_getattro_impl is called,
        # allowing for proper validation and error handling
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the wrapped class; returns (hash, False) since classes are
        # always hashable and the hash is real.
        # CustomClassVariable wraps the CLASS, not an instance.
        # Classes are always hashable in CPython (type.__hash__ = object.__hash__).
        return hash(self.value), False

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: "VariableTracker",
        reverse: bool = False,
    ) -> "VariableTracker":
        # [FUNCTION DOC] Class | other (type union). If other is not a python constant, returns
        # NotImplemented; otherwise calls the metaclass or-operator on the real values and wraps
        # the result (or NotImplemented) in a VT. The reverse flag is ignored.
        try:
            other_val = other.as_python_constant()
        except NotImplementedError:
            return VariableTracker.build(tx, NotImplemented)
        # pyrefly: ignore[bad-argument-count]
        result = type(self.value).__or__(self.value, other_val)
        if result is NotImplemented:
            return VariableTracker.build(tx, NotImplemented)
        return VariableTracker.build(tx, result)

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] The class itself serves as its own graph argument.
        return self.value

    def __repr__(self) -> str:
        # [FUNCTION DOC] Debug string: VT class name with the wrapped class.
        return f"{self.__class__.__name__}({self.value})"

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Class attribute access. Looks the name up statically (graph break if
        # missing). staticmethod and property are resolved; other descriptors are resolved
        # dynamically only if implemented in C, otherwise graph break. Literal results become
        # constants; others are built with an attribute source when this VT has a source.
        obj = None
        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            unimplemented(
                gb_type="Attribute not found on opaque class",
                context=f"class={self.value}, attr={name}",
                explanation=f"The attribute '{name}' does not exist on opaque class {self.value}.",
                hints=[
                    f"Ensure '{name}' is a valid attribute of {type(self.value)}.",
                ],
            )

        if isinstance(obj, staticmethod):
            obj = obj.__get__(self.value)
        elif isinstance(obj, property):
            obj = obj.__get__(None, self.value)  # pyrefly: ignore[no-matching-overload]
        elif hasattr(obj, "__get__"):
            if not isinstance(type(obj).__dict__.get("__get__"), types.FunctionType):
                # C-level descriptors are safe to resolve dynamically.
                obj = getattr(self.value, name)
            else:
                type_name = type(obj).__name__
                unimplemented(
                    gb_type="Unsupported descriptor on opaque class",
                    context=f"class={self.value}, attr={name}, descriptor={type_name}",
                    explanation=f"The attribute '{name}' is a descriptor of type '{type_name}' which is not supported.",
                    hints=[
                        "Only staticmethod, property, and pybind11_static_property are supported.",
                        "Consider accessing this attribute outside of the compiled region.",
                    ],
                )

        if ConstantVariable.is_literal(obj):
            return VariableTracker.build(tx, obj)

        source = AttrSource(self.source, name) if self.source else None
        return VariableTracker.build(tx, obj, source)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Construct an opaque object at trace time. Reference-type (symbolic)
        # opaque classes graph-break (and their __init__ code is marked skipped). Otherwise the
        # args are converted to python constants (guard install skipped for hoisted types),
        # the real object is built, wrapped as a fake object unless it is a constant type, and
        # returned as a CustomClassObjectVariable carrying the constructor args and their sources.
        # disallow creating reference-type opaque objects in the middle of the
        # program
        if is_opaque_symbolic_type(self.value):
            # Skip __init__ to prevent dynamo from tracing it during resume.
            # C extension types (e.g. torch._C.Generator) have wrapper_descriptor
            # __init__ without __code__, so guard the skip_code call.
            if hasattr(self.value.__init__, "__code__"):
                skip_code(self.value.__init__.__code__)

            unimplemented(
                gb_type="An opaque object was created in the middle of the program.",
                context=f"Opaque object type: {self.value}.",
                explanation=(
                    "Opaque objects cannot be created inside the torch.compile region. "
                    "They must be created before entering the compiled function."
                ),
                hints=[
                    "Please create the opaque object before calling torch.compile "
                    "and pass it in as an argument or as a global variable."
                ],
            )

        var_args = TupleVariable(list(args))
        var_kwargs = ConstDictVariable(
            {VariableTracker.build(tx, k): v for k, v in kwargs.items()}
        )
        if should_hoist(self.value):
            with tx.output.tracing_context.guards_context.skip_guard_install():
                constant_args = var_args.as_python_constant()
                constant_kwargs = var_kwargs.as_python_constant()
        else:
            constant_args = var_args.as_python_constant()
            constant_kwargs = var_kwargs.as_python_constant()
        opaque_obj = self.value(  # pyrefly: ignore[not-callable]
            *constant_args, **constant_kwargs
        )

        # Capture sources from the VT args so subgraph reuse can apply
        # source replacement to resolve new ctor arg values on stamp-out.
        ctor_arg_sources = tuple(getattr(a, "source", None) for a in args)

        if is_opaque_constant_type(type(opaque_obj)):
            fake_script_obj = opaque_obj
        else:
            fake_script_obj = torch._library.fake_class_registry.maybe_to_fake_obj(
                tx.output.fake_mode, opaque_obj
            )

        return CustomClassObjectVariable.create(
            opaque_obj,
            fake_script_obj,
            (constant_args, constant_kwargs),
            ctor_arg_sources=ctor_arg_sources,
            tx=tx,
        )


class CustomClassObjectVariable(UserDefinedObjectVariable):
    # [CLASS DOC] VT for an instance of a TorchScript/torchbind/opaque custom class. State: proxy
    # (FX proxy, or the real object for lazily hoisted constant types), value (real or fake
    # script object), source, ctor_args_kwargs (constant constructor args when sourceless) and
    # ctor_arg_sources. Member access is controlled by each member's registered MemberType;
    # unregistered members graph-break, and graph breaks inside the traced paths become hard
    # errors. Key methods: create, as_proxy, tp_getattro_impl, call_method.
    _fake_script_object_cache: dict[int, "CustomClassObjectVariable"] = {}

    @classmethod
    def is_matching_cls(cls, user_cls: type) -> bool:
        # [FUNCTION DOC] True if user_cls is a ScriptObject subclass, a registered custom class,
        # or a FakeScriptObject subclass.
        return (
            issubclass(user_cls, torch.ScriptObject)
            or is_custom_class(user_cls)
            or issubclass(user_cls, FakeScriptObject)
        )

    @staticmethod
    def create(
        proxy: Proxy,
        value: Any,
        ctor_args_kwargs: Any = None,
        ctor_arg_sources: tuple[Source | None, ...] | None = None,
        *,
        tx: "InstructionTranslatorBase | None" = None,
        **options: Any,
    ) -> "CustomClassObjectVariable":
        # [FUNCTION DOC] Factory. Raises AssertionError for Enum values. Builds the VT; if proxy is
        # a real FX Proxy that is not a graph input placeholder, requires tx (AssertionError if
        # missing) and records the VT as proxyable in the current tracer. Returns the VT.
        if isinstance(value, enum.Enum):
            raise AssertionError(
                f"Enum {type(value)} should use UserDefinedObjectVariable, not CustomClassObjectVariable"
            )
        out = CustomClassObjectVariable(
            proxy, value, ctor_args_kwargs, ctor_arg_sources=ctor_arg_sources, **options
        )
        if isinstance(proxy, torch.fx.Proxy) and proxy.node.op != "placeholder":
            if tx is None:
                raise AssertionError(
                    "tx must be provided to CustomClassObjectVariable.create "
                    "when proxy is a real Proxy"
                )
            tx.output.current_tracer.record_proxyable_vt(out)
        return out

    def __init__(
        self,
        proxy: Proxy,
        value: Any,
        ctor_args_kwargs: Any = None,
        source: Source | None = None,
        ctor_arg_sources: tuple[Source | None, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Store proxy, source and constructor info; if proxy is an FX Proxy, record
        # the value as the node's example_value metadata (mutates the node).
        super().__init__(value, **kwargs)
        self.proxy = proxy
        if isinstance(self.proxy, torch.fx.Proxy):
            self.proxy.node.meta["example_value"] = value
        self.source = source
        # If the OpaqueObject is sourceless, then this is
        # the constant (args, kwargs) that Dynamo used to construct it.
        self.ctor_args_kwargs = ctor_args_kwargs
        # Sources of the constructor args, used by subgraph reuse to
        # resolve new values via source replacement on stamp-out.
        self.ctor_arg_sources = ctor_arg_sources

    def as_proxy(self) -> Proxy:
        # [FUNCTION DOC] Return the FX proxy. If proxy is still a raw constant-type object, lazily
        # lift it to a synthetic graph input when its type is hoisted (using the current
        # translator; RuntimeError if constructed with kwargs), caching the new proxy. Raises
        # AssertionError if the raw object is not an opaque constant type. Mutates self.proxy
        # and the graph inputs.
        if not isinstance(self.proxy, torch.fx.Proxy):
            # If we have a hoisted value type, then lazily lift it to be a graph
            # input when as_proxy() is called.
            if not is_opaque_constant_type(type(self.proxy)):
                raise AssertionError(
                    f"Expected opaque constant type, got {type(self.proxy)}"
                )
            if should_hoist(type(self.proxy)):
                from torch._dynamo.symbolic_convert import InstructionTranslator

                tx = InstructionTranslator.current_tx()
                # if any kwargs (synthetic_graph_input doesn't support them yet)
                # not a graph break because hard error more explicit here
                # (and opaque objects are really just used for compile)
                if self.ctor_args_kwargs[1]:
                    raise RuntimeError(
                        "NYI: hoisted opaque objects that accept kwargs, please pass as args"
                    )
                hoisted_vt = tx.output.synthetic_graph_input(
                    type(self.proxy),
                    self.ctor_args_kwargs[0],
                    ctor_arg_sources=self.ctor_arg_sources,
                )
                self.proxy = hoisted_vt.as_proxy()

        return self.proxy

    def __str__(self) -> str:
        # [FUNCTION DOC] Debug string: VT class name with the real object (unwrapping a fake).
        value = (
            self.value.real_obj
            if isinstance(self.value, FakeScriptObject)
            else self.value
        )
        return f"{self.__class__.__name__}({value})"

    __repr__ = __str__

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: "VariableTracker",
        op: str,
    ) -> "VariableTracker":
        # [FUNCTION DOC] Rich comparison by value. If self is not a python constant, use identity
        # based object_richcompare. If other is not constant, return NotImplemented. Otherwise
        # apply the named comparison dunder of self's real type to the real values; any
        # exception yields NotImplemented. Returns a constant VT.
        from .constant import ConstantVariable
        from .object_protocol import object_richcompare

        # Try value-based comparison first. CustomClassObjectVariable wraps
        # pybind11 objects (e.g. Placement subclasses) whose C++ operator==
        # does value comparison. Falling through to object_richcompare would
        # use identity, returning wrong results (e.g. Shard(0) == Shard(0)
        # would be False instead of True).
        try:
            self_val = self.as_python_constant()
        except NotImplementedError:
            return object_richcompare(self, tx, other, op)
        try:
            other_val = other.as_python_constant()
        except NotImplementedError:
            return ConstantVariable.create(NotImplemented)
        try:
            result = getattr(type(self_val), op)(self_val, other_val)
            return ConstantVariable.create(result)
        except Exception:
            return ConstantVariable.create(NotImplemented)

    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute access on the object (graph breaks become hard errors). For
        # registered custom classes by MemberType: USE_REAL and INLINED methods become a lambda VT
        # that routes to call_method (INLINED only when sourceless), other values go through the
        # generic lookup; opaque constant types use generic lookup; missing __bool__/__len__
        # raise AttributeError; unregistered members graph-break. For non-custom script objects
        # the attribute must be a callable method, returned as a call_torchbind higher-order op
        # VT (requires a source, AssertionError otherwise); missing or non-callable graph-breaks.
        from torch._higher_order_ops.torchbind import call_torchbind

        from .higher_order_ops import TorchHigherOrderOperatorVariable

        real_obj = self.as_python_constant()
        real_obj_type = type(real_obj)
        if is_custom_class(real_obj_type):
            member_type = get_member_type(real_obj_type, name)

            if member_type == MemberType.USE_REAL:
                value = getattr(real_obj, name)
                if inspect.ismethod(value) or isinstance(
                    value, types.MethodWrapperType
                ):
                    return LambdaVariable(
                        lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
                    )
                else:
                    return super().tp_getattro_impl(tx, name)

            elif member_type == MemberType.INLINED:
                value = getattr(real_obj, name)
                if (
                    inspect.ismethod(value)
                    or isinstance(value, types.MethodWrapperType)
                ) and self.source is None:
                    # When we don't have a source, fall back to call_method
                    # which creates a proxy node.
                    return LambdaVariable(
                        lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
                    )
                return super().tp_getattro_impl(tx, name)

            elif is_opaque_constant_type(real_obj_type):
                return super().tp_getattro_impl(tx, name)

            elif name in ("__bool__", "__len__") and not hasattr(real_obj, name):
                # Special case: __bool__ and __len__ are used for truthiness checks.
                # If they're not registered and the real object doesn't have them,
                # raise ObservedAttributeError so the caller can fall back to
                # treating the object as truthy (Python default behavior
                raise_observed_exception(AttributeError, tx)

            else:
                unimplemented(
                    gb_type="Attempted to access unregistered member on an OpaqueObject",
                    context=f"value={real_obj}, attr={name}",
                    explanation=f"Member '{name}' is not registered for this opaque object type.",
                    hints=[
                        f"Register '{name}' with a MemberType in register_custom_class(members=...).",
                    ],
                )

        method = getattr(self.value, name, None)
        if method is None:
            unimplemented(
                gb_type="FakeScriptObject missing method implementation",
                context=f"value={self.value}, method={name}",
                explanation=f"TorchScript object {self.value} doesn't define the method {name}.",
                hints=[
                    f"Ensure the method {name} is implemented in {self.value}.",
                    *graph_break_hints.USER_ERROR,
                ],
            )

        if not callable(method):
            unimplemented(
                gb_type="Attempted to access non-callable attribute of TorchScript object",
                context=f"value={self.value}, method={name}",
                explanation="Attribute accesses of TorchScript objects to non-callable attributes are not supported.",
                hints=[
                    "Use method calls instead of attribute access.",
                ],
            )

        if self.source is None:
            raise AssertionError(
                "CustomClassObjectVariable requires a source for tp_getattro_impl"
            )
        return TorchHigherOrderOperatorVariable.make(
            call_torchbind,
            source=AttrSource(self.source, name),
            script_obj_var=self,
            method_name=name,
        )

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: "VariableTracker",
    ) -> "VariableTracker":
        # [FUNCTION DOC] obj[key]: delegates to this class's call_method for the item-get dunder.
        # Call call_method directly on this class to avoid the __getitem__ →
        # mp_subscript_impl loop in VariableTracker.call_method.
        return CustomClassObjectVariable.call_method(self, tx, "__getitem__", [key], {})

    # We only support method calls on script objects. Interpreting the bytecodes
    # should go through tp_getattro_impl then call_function instead of call_method.

    # However, it's possible for call_method to be used directly e.g. for __setattr__.
    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method call on the object (graph breaks become hard errors). For custom
        # classes by MemberType: USE_REAL calls the real method on constant args (graph break for
        # custom __getattr__ or if it returns reference-type opaque objects; the set-attribute
        # dunder returns the real object), result built as a VT; INLINED or opaque constant
        # types emit a call_method graph node on the proxy and wrap the output; unregistered
        # members graph-break. Non-custom objects graph-break ("weird method call").
        from .builder import wrap_fx_proxy

        real_obj = self.as_python_constant()
        real_obj_type = type(real_obj)
        if is_custom_class(real_obj_type):
            member_type = get_member_type(real_obj_type, name)

            if member_type == MemberType.USE_REAL:
                if (
                    inspect.getattr_static(real_obj_type, "__getattr__", None)
                    is not None
                ):
                    unimplemented(
                        gb_type="Opaque object with custom __getattr__ not supported",
                        context=f"{real_obj_type.__name__} with custom __getattr__",
                        explanation="Dynamo does not support opaque objects types with custom __getattr__ methods",
                        hints=[],
                    )

                args_const = [x.as_python_constant() for x in args]
                kwargs_const = {k: v.as_python_constant() for k, v in kwargs.items()}

                method = getattr(real_obj, name)

                if name == "__setattr__":
                    method(*args_const, **kwargs_const)
                    return real_obj  # pyrefly: ignore[bad-return]

                constant_val = method(*args_const, **kwargs_const)

                if any(
                    is_opaque_symbolic_type(type(r))
                    for r in pytree.tree_leaves(constant_val)
                ):
                    unimplemented(
                        gb_type="Opaque object member with method-type USE_REAL returned a reference-type opaque object.",
                        context=f"Opaque object type: {real_obj_type}. Method name: '{name}'",
                        explanation=(
                            "To properly guard reference-type opaque objects, "
                            "we must lift them as inputs to the graph. In order "
                            "to do this, they must all have a source, meaning they "
                            "come from a global value or are an attribute of an input."
                        ),
                        hints=[
                            f"Register member '{name}' with MemberType.INLINED in "
                            f"register_custom_class({real_obj_type}, members=...).",
                        ],
                    )

                return VariableTracker.build(tx, constant_val)

            elif member_type == MemberType.INLINED or is_opaque_constant_type(
                real_obj_type
            ):
                proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)

                proxy = tx.output.create_proxy(
                    "call_method",
                    name,
                    args=(self.proxy, *proxy_args),
                    kwargs=proxy_kwargs,
                )

                return wrap_fx_proxy(tx=tx, proxy=proxy)

            else:
                unimplemented(
                    gb_type="Attempted to access unregistered member on an OpaqueObject",
                    context=f"value={real_obj}, attr={name}",
                    explanation=f"Member '{name}' is not registered for this opaque object type.",
                    hints=[
                        f"Register '{name}' with a MemberType in register_custom_class(members=...).",
                    ],
                )

        unimplemented(
            gb_type="Weird method call on TorchScript object",
            context=f"value={self.value}, method={name}",
            explanation=(
                f"This particular method call ({name}) is not supported (e.g. calling `__setattr__`). "
                "Most method calls to TorchScript objects should be supported."
            ),
            hints=[
                "Avoid calling this method.",
            ],
        )

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Return the real object: unwrap a fake script object, return opaque
        # constant types and torch script objects as is, else defer to the base class.
        if isinstance(self.value, FakeScriptObject):
            return self.value.real_obj
        elif is_opaque_constant_type(type(self.value)):
            return self.value
        elif isinstance(self.value, torch.ScriptObject):
            return self.value
        return super().as_python_constant()

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the real object as (hash, False); raises an observed TypeError
        # if it is unhashable.
        from ..exc import raise_type_error

        real_obj = self.as_python_constant()
        try:
            return hash(real_obj), False
        except TypeError:
            raise_type_error(tx, f"unhashable type: '{type(real_obj).__name__}'")

    def get_real_value(self) -> Any:
        # [FUNCTION DOC] Same as as_python_constant: the underlying real object.
        return self.as_python_constant()
