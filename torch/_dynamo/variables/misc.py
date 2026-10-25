# [MODULE DOC] Miscellaneous VariableTracker subclasses used by Dynamo's symbolic
# execution: super(), tracebacks, exceptions, comptime, cells, autograd.Function
# and its context, attribute/method wrappers, python modules, typing objects,
# numpy functions and dtypes, string formatting, debugging/logging calls,
# context variables, random.Random and weakrefs. Each tracks a Python value during
# tracing, answers attribute/call/compare queries, and knows how to reconstruct
# itself as bytecode. Also holds numpy<->torch._numpy function maps. Import-time
# side effects: none beyond defining classes and the np_constant_collections_map
# table (a Python 3.15-only SentinelVariable is defined conditionally).
"""
This module contains miscellaneous variable tracker implementations for various Python types
and features used in Dynamo's symbolic execution. These classes help track and propagate
information about different kinds of variables during graph capture.

Key classes include:
- SuperVariable: Handles super() calls and method resolution
- ExceptionVariable: Tracks exception objects
- RandomVariable: Manages random number generators
- GetAttrVariable: Tracks attribute access
- MethodWrapperVariable: Handles method wrappers
- PythonModuleVariable: Tracks Python modules
- NumpyVariable: Handles numpy functions and types
- StringFormatVariable: Manages string formatting
- DebuggingVariable: Handles print and logging
"""

import builtins
import contextvars
import dataclasses
import enum
import functools
import inspect
import itertools
import logging
import random
import re
import sys
import traceback
import types
import weakref
from collections.abc import Callable, Sequence
from random import Random
from types import BuiltinFunctionType
from typing import Any, TYPE_CHECKING, TypeGuard, Union

import torch._C
import torch._numpy as tnp
import torch.utils._pytree as pytree
from torch._dynamo.variables.base import MutationType
from torch._dynamo.variables.lists import TupleVariable
from torch._guards import Source

from .. import config, graph_break_hints, trace_rules, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_function_ex,
    create_instruction,
)
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import (
    raise_observed_exception,
    raise_type_error,
    raise_value_error,
    unimplemented,
)
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import unpatched_nn_module_init
from ..source import (
    AttrSource,
    GenericAttrSource,
    GetItemSource,
    TypeMROSource,
    TypeSource,
    WeakRefCallSource,
)
from ..utils import (
    check_positional,
    check_unspec_or_constant_args,
    identity,
    istype,
    no_keywords,
    proxy_args_kwargs,
    raise_args_mismatch,
    unpack_iterable,
)
from .base import (
    AsPythonConstantNotImplementedError,
    getset_build,
    getset_read,
    Member,
    Method,
    NO_SUCH_SUBOBJ,
    VariableTracker,
)
from .constant import ConstantVariable
from .functions import NestedUserFunctionVariable, UserFunctionVariable
from .object_protocol import generic_str
from .user_defined import call_random_fn, is_standard_setattr, UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


class SuperVariable(VariableTracker):
    # [CLASS DOC] Tracks a super() object. State: typevar (the class passed as first
    # argument, or the defining class for zero-arg super) and objvar (the instance
    # or class the lookup starts from). Resolves attributes by walking the MRO past
    # typevar, and dispatches method calls on the resolved attribute (functions,
    # classmethods, staticmethods, object/nn.Module __init__, __new__, __delattr__,
    # __getattribute__, torch function disabling, builtins).
    # PySuper_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L11511
    _cpython_type = super

    _nonvar_fields = {
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        typevar: VariableTracker,
        objvar: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores typevar and the optional objvar (None for the
        # unsupported one-argument form); other kwargs go to VariableTracker.
        super().__init__(**kwargs)
        # typevar is the first argument to super(). In the case where no argument
        # is provided to super(), it is the __class__ object where
        # the super() function is being called
        self.typevar = typevar
        # objvar here must be an instance or subtype of typevar.
        # In the case where super() is called without arguments, it is the first argument
        # to the current function where super() is called from (self for regular method,
        # cls for a classmethod)
        self.objvar = objvar

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the builtin super type.
        return builtins.super

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that calls super with typevar and, when
        # present, objvar (two-argument form, otherwise one-argument form).
        codegen.add_push_null(lambda: codegen(variables.BuiltinVariable(super)))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            codegen.extend_output(create_call_function(2, False))
        else:
            codegen.extend_output(create_call_function(1, False))

    def _resolved_getattr_and_source(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> tuple[Any, AttrSource | None]:
        # [FUNCTION DOC] HELPER: Finds attribute `name` the way super() would: walks
        # the MRO of type(objvar) (or objvar itself when it is a class) starting just
        # after typevar and looks only in each class's own dict. Returns the raw
        # attribute (not bound) and an AttrSource built from the MRO position, or
        # None when no source is available. If typevar is not in the MRO, falls back
        # to a real super lookup with no source. Graph-breaks via unimplemented when
        # objvar is missing (one-arg super) or the attribute is not found.
        # KNOWN ISSUE: the walrus truthiness test skips attributes whose value is
        # falsy (0, None, empty containers), so such class attributes are not found.
        if not self.objvar:
            unimplemented(
                gb_type="1-arg super not implemented",
                context="",
                explanation=f"Dynamo failed to trace attribute `{name}` accessed "
                f"via `super()` (for type `{self.typevar}` and object `{self.objvar}`) "
                "because one-argument of super() is not supported.",
                hints=[
                    "Use two-argument super(type, object_or_type).",
                ],
            )
        if self.objvar is None:
            raise AssertionError("super() requires objvar to be set")
        search_type = self.typevar.as_python_constant()

        # The rest of this function does two things:
        #   - Walk the mro to find where the attribute comes from to be
        #     able to provide accurate source
        #   - Call the getattr to get the object

        # Find the class object, where the function lives.
        # When objvar is "self", use type(self), when objvar is "cls", use it as-is
        type_to_use = self.objvar.python_type()
        type_to_use_source: Source | None = (
            TypeSource(self.objvar.source) if self.objvar.source else None
        )
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value  # type: ignore[attr-defined]
            type_to_use_source = self.objvar.source

        source = None
        search_mro = type_to_use.__mro__

        try:
            start_index = search_mro.index(search_type) + 1
        except ValueError:
            # Corner case where the typevar is not in the mro of the objvar
            # https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L8843-L8844
            return getattr(super(search_type, type_to_use), name), None
        # Implemented based on https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L8812
        # super has its getattro implementation. The key point is that instead of calling getattr, it checks the
        # attribute in the class __dict__
        for index in range(start_index, len(search_mro)):
            # Don't call getattr, just check the __dict__ of the class
            if resolved_getattr := search_mro[index].__dict__.get(name, NO_SUCH_SUBOBJ):
                if resolved_getattr is not NO_SUCH_SUBOBJ:
                    # Equivalent of something like type(L['self']).__mro__[1].attr_name
                    if type_to_use_source:
                        source = AttrSource(
                            GetItemSource(TypeMROSource(type_to_use_source), index),
                            name,
                        )
                    return resolved_getattr, source

        unimplemented(
            gb_type="Unable to resolve super getattr",
            context="",
            explanation=f"Dynamo failed to trace attribute `{name}` accessed "
            f"via `super()` (for type `{self.typevar}` and object `{self.objvar}`) "
            "because the resolved attribute type is not supported.",
            hints=[
                "Ensure the attribute exists in the parent class.",
                "Check the arguments passed to `super()`.",
            ],
        )

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Resolves `name` through the super MRO lookup. Literal
        # constants are returned as ConstantVariable (guarded with CONSTANT_MATCH
        # when a source exists); anything else is wrapped in a GetAttrVariable so
        # the real work happens at call time.
        # Check if getattr is a constant. If not, delay the actual work by
        # wrapping the result in GetAttrVariable. Mostly super is called with a
        # method, so most of the work is delayed to call_function.
        #
        # We could have just implemented a const_getattr. However, super is
        # special when it comes to finding sources. Compared to other VTs, super
        # requires the attr name to walk the mro and find the actual source (and
        # not just AttrSource).
        value, source = self._resolved_getattr_and_source(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            return GetAttrVariable(self, name, py_type=type(value))
        if source:
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
        return variables.ConstantVariable.create(value, source=source)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls method `name` found via the super lookup. Picks a
        # strategy by the kind of resolved attribute: object.__init__ becomes a no-op
        # lambda; nn.Module.__init__ is traced only for newly created modules without
        # args; __new__, staticmethod, classmethod (binding cls from objvar or its
        # type), plain functions, bound methods, standard __setattr__, __delattr__
        # (records a deletion in side effects, needs a constant string name),
        # base-class methods of user objects, object.__getattribute__,
        # _disabled_torch_function_impl (temporarily disables torch function
        # subclass state), tensor method descriptors and builtins are all supported.
        # Anything else graph-breaks with unimplemented.
        inner_fn, source = self._resolved_getattr_and_source(tx, name)
        if self.objvar is None:
            raise AssertionError("super() requires objvar to be set for method calls")
        # This essentially simulates CPython's `super_getattro`:
        # https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/typeobject.c#L11138-L11168
        # where `inner_fn` is the VT for `res = _super_lookup_descr(...)`.
        #
        # However, `res`'s type needs to be checked for `tp_descr_get`, and
        # applied if it has one. We currently don't have polyfills for all the
        # relevant `tp_descr_get`, so we explicitly handle the cases we care
        # about here (e.g., note the staticmethod, classmethod cases).
        if inner_fn is object.__init__:
            return LambdaVariable(identity)
        elif inner_fn is torch.nn.Module.__init__:
            objvar = self.objvar
            from ..side_effects import AttributeMutationNew

            if (
                isinstance(objvar, variables.UserDefinedObjectVariable)
                and isinstance(objvar.mutation_type, AttributeMutationNew)
                and not (args or kwargs)
            ):
                with do_not_convert_to_tracable_parameter():
                    fn_vt = VariableTracker.build(
                        tx, unpatched_nn_module_init, source=source
                    )
                    return fn_vt.call_function(tx, [self.objvar] + args, kwargs)
            else:
                unimplemented(
                    gb_type="Unsupported super().__init__() call",
                    context=f"call_method {self} {name} {args} {kwargs}",
                    explanation="Dynamo encountered a super().__init__() call "
                    f"on {objvar} that resolved to a `torch.nn.Module.__init__()` "
                    "call that we cannot trace.",
                    hints=[*graph_break_hints.DIFFICULT],
                )
        elif (
            self.objvar.source
            and hasattr(inner_fn, "__name__")
            and inner_fn.__name__ == "__new__"
            and variables.UserDefinedClassVariable.is_supported_new_method(inner_fn)
        ):
            user_cls = inner_fn.__self__
            if hasattr(user_cls, "__module__") and user_cls.__module__ == "builtins":
                user_cls_vt: VariableTracker = VariableTracker.build(tx, user_cls)
            else:
                if source is None:
                    raise AssertionError(
                        "source must not be None for user-defined class"
                    )
                user_cls_source = source.member
                user_cls_vt = variables.UserDefinedClassVariable(
                    user_cls, source=user_cls_source
                )
            return user_cls_vt.call_method(tx, "__new__", args, kwargs)
        elif isinstance(inner_fn, staticmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            fn_vt = VariableTracker.build(
                tx, inner_fn.__func__, source=source, realize=True
            )
            return fn_vt.call_function(tx, args, kwargs)
        elif isinstance(inner_fn, classmethod) and isinstance(
            inner_fn.__func__, types.FunctionType
        ):
            if isinstance(self.objvar, variables.UserDefinedClassVariable):
                # super().classmethod is called from a classmethod itself. So,
                # super was converted to super(__class__, cls) in bytecode and
                # therefore we have to propagate the cls.
                cls_variable = self.objvar
            else:
                # current function is an instance method, therefore super was
                # converted to super(__class__, self). We have to find
                # type(self) to bind the cls to the parent classmethod.
                # Note that it can't be the self.typevar because __class__ is
                # the class where the method is defined, which could be
                # different from type(self) with polymorphism.
                cls_source = None
                if self.objvar.source:
                    cls_source = TypeSource(self.objvar.source)
                cls_variable = VariableTracker.build(
                    tx,
                    self.objvar.value_type,  # type: ignore[attr-defined]
                    cls_source,
                )
            if source is None:
                raise AssertionError(
                    "source must not be None for classmethod resolution"
                )
            fn_vt = VariableTracker.build(
                tx,
                inner_fn.__func__,
                source=AttrSource(source, "__func__"),
                realize=True,
            )
            return fn_vt.call_function(tx, [cls_variable, *args], kwargs)
        elif isinstance(inner_fn, types.FunctionType):
            fn_vt = VariableTracker.build(tx, inner_fn, source=source, realize=True)
            return fn_vt.call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(
                inner_fn.__func__, self.objvar, source=source
            ).call_function(tx, args, kwargs)
        elif is_standard_setattr(inner_fn) and isinstance(
            self.objvar, UserDefinedObjectVariable
        ):
            # type: ignore[arg-type]
            return self.objvar.method_setattr_standard(tx, *args, **kwargs)
        elif inner_fn is object.__delattr__:
            attr = args[0]
            try:
                attr = attr.as_python_constant()
            except NotImplementedError as exc:
                unimplemented(
                    gb_type="Non-constant attribute given to `super().__delattr__()`",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo requires the attribute name passed to "
                    "`super().__delattr__(...)` to be a constant (string).",
                    hints=[
                        "Ensure the attribute name is a string literal or a constant variable."
                    ],
                    from_exc=exc,
                )
            if not tx.output.side_effects.is_attribute_mutation(self.objvar):
                unimplemented(
                    gb_type="Attempted super().__delattr__() on an object without mutation tracking",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo needs to track mutations on an object "
                    "before `super().__delattr__` can be used on it. But the "
                    f"object ({self.objvar}) doesn't have attribute mutation "
                    "tracking enabled.",
                    hints=[
                        "Ensure the object is tracked by Dynamo's side effect system.",
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )
            if not isinstance(attr, str):
                raise AssertionError(f"attr must be a str, got {type(attr)}")
            tx.output.side_effects.store_attr(
                self.objvar, attr, variables.DeletedVariable()
            )
            return variables.ConstantVariable.create(None)
        elif (
            isinstance(self.objvar, variables.UserDefinedObjectVariable)
            and self.objvar._base_vt is not None
            and self.objvar._base_methods is not None
            and inner_fn in self.objvar._base_methods
        ):
            return self.objvar._base_vt.call_method(tx, name, args, kwargs)
        elif inner_fn is object.__getattribute__:
            attr_name = args[0].value  # type: ignore[attr-defined]
            # object.__getattribute__ IS PyObject_GenericGetAttr.  Delegate
            # to the shared implementation so that __dict__, __class__,
            # polyfilled C descriptors, etc. are all handled consistently.
            if isinstance(self.objvar, UserDefinedObjectVariable):
                return self.objvar.generic_getattr(tx, attr_name)

            attr_value = None
            try:
                attr_value = object.__getattribute__(
                    self.objvar.value,  # pyrefly: ignore[missing-attribute]
                    attr_name,
                )
            except AttributeError:
                raise_observed_exception(AttributeError, tx)

            attr_source = None
            if self.objvar.source is not None:
                attr_source = GenericAttrSource(self.objvar.source, attr_name)
            return VariableTracker.build(tx, attr_value, attr_source)
        elif inner_fn is torch._C._disabled_torch_function_impl:
            # See `THPModule_disable_torch_function` for the C impl.
            # The signature of _disabled_torch_function_impl is similar to
            # `__torch_function__`, just without the first `cls` argument:
            #  * (func, types, args, kwargs)
            func = args[0]
            # pyrefly: ignore [implicit-any]
            tf_kwargs = {}
            tf_args = args[2].items  # type: ignore[attr-defined]
            # type: ignore[attr-defined]
            for hash_key_vt, value_vt in args[3].items.items():
                key_str = hash_key_vt.vt.as_python_constant()
                tf_kwargs[key_str] = value_vt

            tx_old = tx.symbolic_torch_function_state.torch_function_subclass_enabled
            tx.symbolic_torch_function_state.torch_function_subclass_enabled = False
            try:
                return func.call_function(tx, tf_args, tf_kwargs)
            finally:
                tx.symbolic_torch_function_state.torch_function_subclass_enabled = (
                    tx_old
                )
        elif (
            isinstance(inner_fn, types.MethodDescriptorType)
            and inner_fn in trace_rules.get_tensor_method()
        ):
            # FunctionType but implementation is in C, we support some of these,
            # e.g., tensor ops like `torch.Tensor.to`.
            fn_var = VariableTracker.build(tx, inner_fn, source, realize=True)
            return fn_var.call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.BuiltinFunctionType):
            fn_vt = VariableTracker.build(tx, inner_fn, source=source, realize=True)
            return fn_vt.call_function(tx, args, kwargs)

        unimplemented(
            gb_type="Attempted to call a super() attribute that is "
            "not a function or method",
            context=f"call_method {self} {name}",
            explanation="Dynamo does not know how to trace the call "
            f"`super().{name}()` because `super().{name}` is not a "
            "function or method attribute.",
            hints=[
                "Ensure the attribute accessed via `super()` is a standard method or function.",
            ],
        )


class FrameSummaryVariable(VariableTracker):
    # [CLASS DOC] Wraps a real traceback.FrameSummary. Exposes lineno, filename,
    # name and line as read-only members via tp_members.
    def __init__(self, frame_summary: traceback.FrameSummary, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the wrapped frame summary.
        super().__init__(**kwargs)
        self.frame_summary = frame_summary

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns traceback.FrameSummary.
        return traceback.FrameSummary

    # traceback.FrameSummary is pure-Python with __slots__ (Lib/traceback.py);
    # each slot is exposed as a read-only member_descriptor.
    tp_members = {
        "lineno": Member(getset_build(lambda s: s.frame_summary.lineno)),
        "filename": Member(getset_build(lambda s: s.frame_summary.filename)),
        "name": Member(getset_build(lambda s: s.frame_summary.name)),
        "line": Member(getset_build(lambda s: s.frame_summary.line)),
    }


class TracebackVariable(VariableTracker):
    # [CLASS DOC] Models a traceback object as a linked chain. State: frame_summary
    # (a FrameSummaryVariable for this entry) and tb_next (the next
    # TracebackVariable, or a None constant at the end). Invariant: the chain has
    # no cycles (enforced when tb_next is assigned).
    def __init__(
        self,
        frame_summary: FrameSummaryVariable,
        tb_next: Union["TracebackVariable", ConstantVariable],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores frame_summary and tb_next. Raises AssertionError if
        # tb_next is None (a None constant must be used to end the chain).
        # The traceback holds four attributes:
        #  - tb_frame
        #  - tb_lineno
        #  - tb_lasti
        #  - tb_next

        super().__init__(**kwargs)
        self.frame_summary = frame_summary
        # the next traceback in the chain
        if tb_next is None:
            raise AssertionError("tb_next must not be None")
        self.tb_next = tb_next

    @classmethod
    def from_frame_summary(
        cls,
        frame_summary: traceback.FrameSummary,
        tb_next: Union["TracebackVariable", ConstantVariable],
    ) -> "TracebackVariable":
        # [FUNCTION DOC] Builds a TracebackVariable from a raw FrameSummary by
        # wrapping it in a FrameSummaryVariable, linked to tb_next.
        return cls(FrameSummaryVariable(frame_summary), tb_next=tb_next)

    @staticmethod
    def is_valid_traceback(obj: VariableTracker) -> bool:
        # [FUNCTION DOC] True if obj is exactly a TracebackVariable or a None constant.
        return istype(obj, TracebackVariable) or obj.is_constant_none()

    def extract_tb(self) -> list[traceback.FrameSummary | FrameSummaryVariable]:
        # [FUNCTION DOC] Returns the frame summaries of this entry followed by all
        # later entries in the chain, in order (recursive over tb_next).
        if istype(self.tb_next, ConstantVariable):
            return [self.frame_summary]
        return [self.frame_summary] + self.tb_next.extract_tb()

    def has_reference_cycle(self, tb: VariableTracker) -> bool:
        # [FUNCTION DOC] Returns True if tb appears (by identity) in the chain that
        # starts at this traceback, including this one.
        # checks if `tb` is in the chain of tb_next starting from `self`
        curr_tb: TracebackVariable | ConstantVariable = self
        while istype(curr_tb, TracebackVariable):
            if curr_tb is tb:
                return True
            curr_tb = curr_tb.tb_next
        return False

    def python_type(self) -> type[types.TracebackType]:
        # [FUNCTION DOC] Returns types.TracebackType.
        return types.TracebackType

    def call_setattr(
        self,
        tx: "InstructionTranslatorBase",
        name_var: VariableTracker,
        val: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements setting an attribute. Only tb_next is honored:
        # the value must be a traceback or None (else TypeError is raised in the
        # traced program) and must not create a cycle in either direction (else
        # ValueError). Returns a None constant.
        # KNOWN ISSUE: assignments to any other attribute name are silently ignored.
        name = name_var.as_python_constant()
        if name == "tb_next":
            if not self.is_valid_traceback(val):
                raise_observed_exception(TypeError, tx)
            if not isinstance(val, (TracebackVariable, ConstantVariable)):
                raise AssertionError(
                    f"tb_next val must be TracebackVariable or ConstantVariable, got {type(val)}"
                )
            if self.has_reference_cycle(val) or (
                istype(val, TracebackVariable) and val.has_reference_cycle(self)
            ):
                raise_observed_exception(ValueError, tx)
            self.tb_next = val
        return variables.ConstantVariable.create(None)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute read: tb_next, tb_lineno (from the frame summary
        # line number) and frame_summary are served directly; tb_lasti graph-breaks;
        # other names fall back to the base implementation.
        if name == "tb_next":
            return self.tb_next
        elif name == "tb_lineno":
            return self.frame_summary.tp_getattro_impl(tx, "lineno")
        elif name == "frame_summary":
            return self.frame_summary
        elif name == "tb_lasti":
            unimplemented(
                gb_type="traceback.tb_lasti not supported",
                context=f"{self} accessing 'tb_lasti'",
                explanation="Dynamo does not support accessing the tb_lasti attribute of traceback objects.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return super().tp_getattro_impl(tx, name)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Rich comparison using the default object semantics
        # (identity based) via object_richcompare.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handles __setattr__ by delegating to call_setattr; every
        # other method goes to the base class.
        if name == "__setattr__":
            return self.call_setattr(tx, *args)
        return super().call_method(tx, name, args, kwargs)


class ExceptionVariable(VariableTracker):
    # _PyExc_BaseException: https://github.com/python/cpython/blob/v3.13.0/Objects/exceptions.c
    _cpython_type = BaseException

    # [CLASS DOC] Models a BaseException instance. State: exc_type (the exception
    # class), args (list of VariableTrackers), the chaining attributes __context__,
    # __cause__, __suppress_context__ and __traceback__ (each a VariableTracker,
    # defaulting to None/False constants) and python_stack (user stack summary at
    # first raise, used when re-raising). Custom attributes live in the side effects
    # instance dict. Subclasses add extra constructor attributes.
    def __init__(
        self,
        exc_type: Any,
        args: list[VariableTracker],
        init_kwargs: dict[str, VariableTracker] | None = None,
        source: Source | None = None,
        mutation_type: MutationType | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores exc_type and args and initializes the chaining
        # attributes to defaults. Graph-breaks (unimplemented) if any keyword args
        # are given to the constructor.
        super().__init__(source=source, mutation_type=mutation_type)
        self.exc_type = exc_type
        self.args = args
        if init_kwargs:
            unimplemented(
                gb_type="Keyword args passed to exception constructor",
                context=f"{self} with kwargs {init_kwargs}",
                explanation="Dynamo does not know how to handle keyword args passed to an exception constructor",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # When raising a new exception while another exception is already being
        # handled, the new exception's __context__ attribute is automatically
        # set to the handled exception.
        self.__context__: VariableTracker = ConstantVariable.create(None)
        # Set when user raised an exception from another:
        # raise ... from ...
        self.__cause__: VariableTracker = ConstantVariable.create(None)
        # Boolean flag that controls whether the __context__ attribute is set
        self.__suppress_context__: VariableTracker = ConstantVariable.create(False)
        # Contains the call stack where the exception was raised.
        self.__traceback__: VariableTracker = ConstantVariable.create(None)
        # The user stack at the time this exception was first raised.
        # Used to preserve the original exception location when re-raising.
        self.python_stack: traceback.StackSummary | None = None

    def set_context(self, context: VariableTracker) -> None:
        # [FUNCTION DOC] Sets the implicit __context__ (exception being handled).
        self.__context__ = context

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that constructs the builtin exception class
        # with args, then restores __context__, __cause__ and __suppress_context__
        # when they are not plain None/True/False constants.
        # KNOWN ISSUE: __traceback__ and custom attributes are not restored here.
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", self.exc_type.__name__)
        )
        codegen.foreach(self.args)
        codegen.call_function(len(self.args), False)

        def codegen_attr(name: str) -> None:
            # [FUNCTION DOC] HELPER: For attribute `name`, if it is a constant, only
            # checks it is True/False/None (AssertionError otherwise; nothing emitted).
            # Otherwise emits code storing the attribute on the exception on the stack.
            attr = getattr(self, name)
            if istype(attr, ConstantVariable):
                if attr.value not in (True, False, None):
                    raise AssertionError(
                        f"attr.value must be True, False, or None, got {attr}"
                    )
            else:
                codegen.dup_top()
                codegen(attr)
                codegen.extend_output(codegen.rot_n(2))
                codegen.store_attr(name)

        codegen_attr("__context__")
        codegen_attr("__cause__")
        codegen_attr("__suppress_context__")

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the exception class.
        return self.exc_type

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Default identity-based object comparison.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Supports __setattr__ (validated updates of __context__,
        # __cause__ which also sets __suppress_context__, __suppress_context__,
        # __traceback__, args, or an arbitrary attribute stored via side effects),
        # __setstate__ (None is a no-op, a dict is applied through __setattr__) and
        # with_traceback (sets and returns self). Invalid values raise TypeError in
        # the traced program. Other methods use the base class.
        # KNOWN ISSUE: the TypeError text for a bad __suppress_context__ value
        # wrongly says "exception cause must be None or derive from BaseException".
        if name == "__setattr__":
            name = args[0].as_python_constant()
            val = args[1]
            if name == "__context__":
                # Constant can be either an Exceptior or None
                if not (
                    val.is_constant_none()
                    or isinstance(
                        val,
                        (
                            variables.ExceptionVariable,
                            variables.UserDefinedExceptionClassVariable,
                            variables.UserDefinedExceptionObjectVariable,
                        ),
                    )
                ):
                    raise_type_error(
                        tx,
                        "exception context must be None or derive from BaseException",
                    )
                self.set_context(val)
            elif name == "__cause__":
                if val.is_constant_none() or isinstance(
                    val,
                    (
                        variables.BuiltinVariable,
                        variables.ExceptionVariable,
                        variables.UserDefinedExceptionClassVariable,
                        variables.UserDefinedExceptionObjectVariable,
                    ),
                ):
                    self.__cause__ = val
                    self.__suppress_context__ = variables.ConstantVariable.create(True)
                else:
                    raise_type_error(
                        tx, "exception cause must be None or derive from BaseException"
                    )
            elif name == "__suppress_context__":
                if val.is_constant_match(True, False):
                    self.__suppress_context__ = val
                else:
                    raise_type_error(
                        tx, "exception cause must be None or derive from BaseException"
                    )
            elif name == "__traceback__":
                if not TracebackVariable.is_valid_traceback(val):
                    raise_type_error(tx, "__traceback__ must be a traceback or None")
                self.__traceback__ = val
            elif name == "args":
                # CPython coerces any iterable to a tuple (PySequence_Tuple).
                self.args = unpack_iterable(tx, val)
            else:
                # Arbitrary user attribute -> store in the instance __dict__
                # via the side effects table.
                se = tx.output.side_effects
                if not se.is_attribute_mutation(self):
                    se.track_attribute_mutation_new(self)
                se.store_instance_dict_attr(self, name, val)
            return variables.ConstantVariable.create(None)
        elif name == "__setstate__":
            if len(args) != 1:
                raise_type_error(
                    tx, f"__setstate__() takes exactly one argument ({len(args)} given)"
                )
            [state] = args
            # BaseException.__setstate__(None) is a documented no-op.
            if state.is_constant_none():
                return variables.ConstantVariable.create(None)
            if not isinstance(state, variables.ConstDictVariable):
                raise_type_error(tx, "state is not a dictionary")
            for key, value in state.keys_as_python_constant().items():
                self.call_method(
                    tx, "__setattr__", [ConstantVariable.create(key), value], {}
                )
            return variables.ConstantVariable.create(None)
        elif name == "with_traceback":
            if len(args) != 1:
                raise_type_error(
                    tx,
                    f"with_traceback() takes exactly one argument ({len(args)} given)",
                )
            [tb] = args
            if not TracebackVariable.is_valid_traceback(tb):
                raise_type_error(tx, "__traceback__ must be a traceback or None")
            self.__traceback__ = tb
            return self
        else:
            return super().call_method(tx, name, args, kwargs)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute read for __class__, __context__, __cause__,
        # __suppress_context__, __traceback__ and args (as a tuple); any other name
        # goes to the generic lookup and raises an observed AttributeError if absent.
        if name == "__class__":
            return VariableTracker.build(tx, self.exc_type)
        elif name == "__context__":
            return self.__context__
        elif name == "__cause__":
            return self.__cause__
        elif name == "__suppress_context__":
            return self.__suppress_context__
        elif name == "__traceback__":
            return self.__traceback__
        elif name == "args":
            return VariableTracker.build(
                tx,
                tuple(self.args),
                source=self.source and AttrSource(self.source, "args"),
            )
        try:
            # Custom attributes are stored in the side effects instance dict and
            # resolved by generic_getattr before reaching here, so a fall-through
            # to the generic lookup that finds nothing means the attribute is
            # genuinely absent -- match CPython's BaseException tp_getattro
            # (PyObject_GenericGetAttr) and raise AttributeError.
            return super().tp_getattro_impl(tx, name)
        except NotImplementedError:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[f"'{self.exc_type.__name__}' object has no attribute '{name}'"],
            )

    def tp_str_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] str() of the exception: empty string for no args, str of the
        # single arg, or str of the tuple of args otherwise (matches CPython).
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/exceptions.c#L118-L129
        if len(self.args) == 0:
            return VariableTracker.build(tx, "")
        elif len(self.args) == 1:
            return generic_str(tx, self.args[0])
        else:
            from . import TupleVariable

            tuple_var = TupleVariable(list(self.args))
            return generic_str(tx, tuple_var)

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the class name and exception type.
        return f"{self.__class__.__name__}({self.exc_type})"

    __repr__ = __str__

    @staticmethod
    def _debug_format_arg(arg: VariableTracker) -> str:
        # [FUNCTION DOC] HELPER: Text of arg: repr of its constant value if it has
        # one, else its debug_repr.
        try:
            return repr(arg.as_python_constant())
        except Exception:
            return arg.debug_repr()

    def debug_repr(self) -> str:
        # [FUNCTION DOC] Returns a Python-like rendering such as the type name
        # followed by comma-separated formatted args.
        args = ", ".join(self._debug_format_arg(arg) for arg in self.args)
        return f"{self.python_type_name()}({args})"

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] repr() of the exception, built from debug_repr as a string.
        # ref: BaseException_repr in https://github.com/python/cpython/blob/3.13/Objects/exceptions.c#L135-L142
        return VariableTracker.build(tx, self.debug_repr())


class StopIterationVariable(ExceptionVariable):
    # [CLASS DOC] ExceptionVariable for StopIteration; adds a value attribute that is
    # the first constructor arg (or a None constant).
    def __init__(
        self,
        exc_type: Any,
        args: list[VariableTracker],
        init_kwargs: dict[str, VariableTracker] | None = None,
        source: Source | None = None,
        mutation_type: MutationType | None = None,
    ) -> None:
        # [FUNCTION DOC] Records value from args, then runs the base constructor.
        self.value = args[0] if args else variables.ConstantVariable.create(None)
        super().__init__(exc_type, args, init_kwargs, source, mutation_type)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Returns value for the value attribute; else base lookup.
        if name == "value":
            return self.value
        return super().tp_getattro_impl(tx, name)


class _KwargAttrExceptionVariable(ExceptionVariable):
    # [CLASS DOC] HELPER: Base for
    # exceptions whose constructor accepts keyword-only attributes that
    # default to None (e.g. NameError's `name`, AttributeError's `name`/`obj`).
    # Subclasses list the attribute names in `_kwarg_attrs`; they are popped from
    # init_kwargs, exposed via getattr, and restored on reconstruct.
    _kwarg_attrs: tuple[str, ...] = ()

    def __init__(
        self,
        exc_type: Any,
        args: list[VariableTracker],
        init_kwargs: dict[str, VariableTracker] | None = None,
        source: Source | None = None,
        mutation_type: MutationType | None = None,
    ) -> None:
        # [FUNCTION DOC] Pops each name listed in _kwarg_attrs from a copy of
        # init_kwargs into the _attrs mapping (default None constant), then runs the
        # base constructor with the remaining kwargs (which graph-break if non-empty).
        init_kwargs = dict(init_kwargs) if init_kwargs else {}
        none = variables.ConstantVariable.create(None)
        self._attrs = {name: init_kwargs.pop(name, none) for name in self._kwarg_attrs}
        super().__init__(exc_type, args, init_kwargs, source, mutation_type)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Returns the stored keyword attribute if name is one, else
        # defers to the base lookup.
        if name in self._attrs:
            return self._attrs[name]
        return super().tp_getattro_impl(tx, name)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Reconstructs the exception, then stores every keyword
        # attribute whose value is not a None constant onto it.
        super().reconstruct(codegen)
        for name, val in self._attrs.items():
            if not (istype(val, ConstantVariable) and val.value is None):
                codegen.dup_top()
                codegen(val)
                codegen.extend_output(codegen.rot_n(2))
                codegen.store_attr(name)


class AttributeErrorVariable(_KwargAttrExceptionVariable):
    # [CLASS DOC] AttributeError with optional keyword attributes name and obj,
    # exposed as read-only members.
    # https://docs.python.org/3/library/exceptions.html#AttributeError
    _kwarg_attrs = ("name", "obj")
    tp_members = {
        "name": Member(getset_read(lambda s: s._attrs["name"])),
        "obj": Member(getset_read(lambda s: s._attrs["obj"])),
    }


class NameErrorVariable(_KwargAttrExceptionVariable):
    # [CLASS DOC] NameError with an optional keyword attribute name, exposed as a
    # read-only member.
    # https://docs.python.org/3/library/exceptions.html#NameError
    _kwarg_attrs = ("name",)
    tp_members = {"name": Member(getset_read(lambda s: s._attrs["name"]))}


class UnknownVariable(VariableTracker):
    # [CLASS DOC] Placeholder for a value of unknown nature; adds no behavior.
    """
    It could be anything!
    """


class DelayGraphBreakVariable(UnknownVariable):
    # [CLASS DOC] Stack placeholder that defers a graph break until it is called.
    # State: msg (reason text) and hints (list of hint strings).
    """
    Used to insert a dummy variable in the stack to do the graph break at CALL_FUNCTION.
    """

    def __init__(
        self,
        msg: str | None = None,
        hints: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the break reason message and hints (default empty).
        super().__init__(**kwargs)
        self.msg = msg
        self.hints = hints or []

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Always graph-breaks (unimplemented), reporting the source
        # name and stored message and hints. Never returns.
        name = "" if self.source is None else self.source.name
        unimplemented(
            gb_type="Unsupported function call (delayed)",
            context=f"source: {self.source}",
            explanation="Dynamo determined that a graph break should occur "
            f"when calling `{name}`. Reason: {self.msg}",
            hints=self.hints,
        )


class ComptimeVariable(VariableTracker):
    # [CLASS DOC] Represents the comptime special form; calling it runs a user
    # function at Dynamo compile time with a ComptimeContext. Needs a source.
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Always raises NotImplementedError (cannot be reconstructed).
        raise NotImplementedError("comptime is special form")

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Looks up attribute `name` on the comptime helper object and
        # wraps it with an attribute source. Raises AssertionError without a source.
        from ..comptime import comptime

        if self.source is None:
            raise AssertionError("ComptimeVariable requires a source")
        # To support the comptime.print_graph convenience accessors
        return VariableTracker.build(
            tx, getattr(comptime, name), source=AttrSource(self.source, name)
        )

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Runs the first argument (a user or nested user function)
        # immediately, passing a ComptimeContext for the current translator.
        # Nested functions must have no free variables (TypeError otherwise) and are
        # rebuilt as plain functions with their globals and defaults. More than two
        # args or any kwargs raises an args mismatch error; other argument kinds raise
        # RuntimeError. The second argument is ignored. Returns a None constant.
        from ..comptime import ComptimeContext

        # TODO: support an expression form as well
        # Second argument is runtime lambda, ignored
        if kwargs or len(args) > 2:
            raise_args_mismatch(
                tx,
                "comptime()",
                "at most 2 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        fn = args[0]
        if isinstance(fn, UserFunctionVariable):
            fn.get_function()(ComptimeContext(tx))
        elif isinstance(fn, NestedUserFunctionVariable):
            # We have to manually bind the freevars ourselves
            code = fn.get_code()
            if fn.closure:
                raise_type_error(
                    tx,
                    f"comptime function must not have free variables, but these variables were free: {code.co_freevars}",
                )
            func = types.FunctionType(
                code,
                fn.f_globals,
                fn.fn_name.as_python_constant(),
                # type: ignore[attr-defined]
                tuple(fn.defaults.items) if fn.defaults else None,
                # We could automatically promote free variables into
                # ComptimeVar but this is confusing if you access
                # a free variable that we actually DO have the runtime
                # value for
                # tuple(make_cell(ComptimeVar(i)) for i in fn.closure.items)
                (),
            )
            func(ComptimeContext(tx))
        else:
            raise RuntimeError(f"unsupported argument to comptime: {type(fn)}")

        return variables.ConstantVariable.create(None)


class CellVariable(VariableTracker):
    # [CLASS DOC] Models a closure cell. State: pre_existing_contents (tracker for
    # the cell value if the cell existed before tracing) and local_name (name usable
    # for LOAD/STORE_DEREF in the root frame). Mutations of cell contents are
    # buffered in SideEffects, not stored here.
    # PyCell_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/cellobject.c#L151
    _cpython_type = types.CellType

    # If the cell existed before Dynamo tracing started, this will be the
    # VariableTracker that represents the cell content.
    #
    # Note that all mutation to the cell (i.e., its content) will be buffered in
    # SideEffects, rather than being reflected here. One can think of
    # `CellVariable` as a special case for `UserDefinedObjectVariable`.
    pre_existing_contents: VariableTracker | None

    # This is set when this cell can be referenced via `LOAD/STORE_DEREF` in the
    # root frame via this name (e.g., the name is in `co_cellvars/co_freevars`).
    local_name: str | None = None

    def __init__(
        self, pre_existing_contents: VariableTracker | None = None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Stores the optional pre-existing cell contents tracker.
        super().__init__(**kwargs)
        self.pre_existing_contents = pre_existing_contents

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns types.CellType.
        return types.CellType


class NewGlobalVariable(VariableTracker):
    # [CLASS DOC] Marker tracker for a global variable newly created during tracing.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Plain pass-through to the base constructor.
        super().__init__(**kwargs)


def produce_trampoline_autograd_apply(fn_cls: Any) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Returns a function that forwards all arguments to the
    # apply method of the given autograd.Function class. The returned function has
    # an attribute _origin pointing back to this factory so it can be recognized.
    def trampoline_autograd_apply(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls apply on the captured class and returns its result.
        return fn_cls.apply(*args, **kwargs)

    # type: ignore[attr-defined]
    trampoline_autograd_apply._origin = produce_trampoline_autograd_apply
    return trampoline_autograd_apply


class AutogradFunctionVariable(VariableTracker):
    # [CLASS DOC] Represents a torch.autograd.Function subclass (fn_cls). Handles
    # apply (either as an opaque call for allowed callables, or by tracing forward
    # and backward through AutogradFunctionApplyVariable), backward, and static or
    # class methods. Calling the class itself yields another variable of this kind.
    """represents a torch.autograd.Function subclass"""

    _nonvar_fields = {
        "fn_cls",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, fn_cls: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the autograd.Function subclass.
        super().__init__(**kwargs)
        self.fn_cls = fn_cls

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the type metaclass (the value is a class).
        return type

    def _resolve_kwargs(
        self,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> list[VariableTracker] | None:
        # [FUNCTION DOC] HELPER: Converts keyword arguments of apply into extra
        # positional arguments using forward's parameter names (skipping the ctx
        # parameter when setup_context is not defined). Returns None if some keyword
        # is not a positional parameter name (e.g. keyword-only). Raises TypeError if
        # a keyword duplicates a positional arg or a needed middle argument is
        # missing. Gaps are not filled with defaults.
        """Resolve kwargs to positional args using forward().__code__.

        Uses co_varnames/co_argcount directly to match the C++
        resolve_kwargs_to_positional in python_function.cpp.
        Keyword-only args are not resolved; callers should graph break.
        """
        from torch.autograd.function import _is_setup_context_defined

        fn = self.fn_cls.forward
        code = fn.__code__
        has_ctx = not _is_setup_context_defined(self.fn_cls.setup_context)
        param_offset = 1 if has_ctx else 0
        param_names = list(code.co_varnames[param_offset : code.co_argcount])

        for name in kwargs:
            if name not in param_names:
                return None
            if param_names.index(name) < len(args):
                raise TypeError(f"forward() got multiple values for argument '{name}'")

        max_idx = max(
            (param_names.index(name) for name in kwargs),
            default=len(args) - 1,
        )

        result: list[VariableTracker] = list(args)
        for i in range(len(args), max_idx + 1):
            name = param_names[i]
            if name in kwargs:
                result.append(kwargs[name])
            else:
                raise TypeError(
                    f"forward() missing required argument: '{name}' (position {i})"
                )
        return result

    def call_apply(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces Function.apply. Kwargs are first resolved to positional
        # args (graph break if impossible). If any tensor arg requires grad (or an
        # nn module is training) and grad mode is on, builds an
        # AutogradFunctionApplyVariable over forward (rewritten together with
        # setup_context when defined) and backward, graph-breaking on custom vjp or
        # jvp, and installs closure guards on forward/setup_context. Otherwise
        # inlines forward directly with a fresh context object as first argument
        # (dropped if forward does not take it). Graph-breaks if forward is neither
        # a function nor a method.
        if kwargs:
            resolved = self._resolve_kwargs(args, kwargs)
            if resolved is None:
                unimplemented(
                    gb_type="autograd_function_kwonly_args",
                    context=f"forward() has keyword-only args: {set(kwargs) - set(self.fn_cls.forward.__code__.co_varnames)}",
                    explanation="autograd.Function.apply does not support keyword-only arguments in forward().",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
            args = resolved
            kwargs = {}

        requires_grad = False

        def visit(vt: VariableTracker) -> None:
            # [FUNCTION DOC] HELPER: Sets the enclosing requires_grad flag if vt is a
            # tensor that may require grad or a module in training mode.
            nonlocal requires_grad
            if vt.is_tensor():
                # type: ignore[attr-defined]
                if vt.requires_grad is not False:
                    requires_grad = True
            if isinstance(vt, variables.NNModuleVariable):
                if vt.is_training(tx):
                    requires_grad = True

        VariableTracker.visit(visit, (args, kwargs))

        if requires_grad and torch.is_grad_enabled():
            source = self.source

            from torch._functorch.autograd_function import (
                autograd_function_forward_rewritten,
            )
            from torch.autograd.function import _is_setup_context_defined

            forward_fn = self.fn_cls.forward

            is_setup_ctx_defined = _is_setup_context_defined(self.fn_cls.setup_context)
            if is_setup_ctx_defined:
                # If setup_context is defined, we generate a new forward function which includes
                # the original forward and setup_context function, and trace the new forward function.
                forward_fn = autograd_function_forward_rewritten(
                    self.fn_cls.forward, self.fn_cls.setup_context
                )
                # The forward points to a new function now, so we can't use the
                # old source. Later on, we guard specifically on
                # is_setup_ctx_defined
                source = None

            vjp_fn = self.fn_cls.vjp  # type: ignore[attr-defined]
            if vjp_fn is not torch.autograd.Function.vjp:
                unimplemented(
                    gb_type="Unsupported custom vjp",
                    context=f"call_apply {self} {args} {kwargs}",
                    explanation="Dynamo does not support tracing "
                    "`torch.autograd.Function` subclasses that define "
                    "a custom `vjp` method.",
                    hints=[
                        "Remove the custom `vjp` method if possible.",
                        "Use standard `backward` instead if applicable.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            jvp_fn = self.fn_cls.jvp  # type: ignore[attr-defined]
            if jvp_fn is not torch.autograd.Function.jvp:
                unimplemented(
                    gb_type="Unsupported custom jvp",
                    context=f"call_apply {self} {args} {kwargs}",
                    explanation="Dynamo does not support tracing "
                    "`torch.autograd.Function` subclasses that define "
                    "a custom `jvp` method.",
                    hints=[
                        "Remove the custom `jvp` method if possible.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            from .higher_order_ops import AutogradFunctionApplyVariable

            if source is None and not is_setup_ctx_defined:
                source = AttrSource(
                    tx.import_source(self.fn_cls.__module__), self.fn_cls.__name__
                )
            apply_source = source and AttrSource(source, member="apply")
            val = AutogradFunctionApplyVariable(
                forward_fn,
                self.fn_cls.backward,
                source,
                source=apply_source,
            ).call_function(tx, args, kwargs)
            if self.source and is_setup_ctx_defined:
                fwd_src = AttrSource(self.source, "forward")
                install_guard(fwd_src.make_guard(GuardBuilder.CLOSURE_MATCH))
                setup_ctx_src = AttrSource(self.source, "setup_context")
                install_guard(setup_ctx_src.make_guard(GuardBuilder.CLOSURE_MATCH))

            return val

        if self.source:
            source = AttrSource(self.source, "forward")
        else:
            source = None

        fn = self.fn_cls.forward
        ctx = AutogradFunctionContextVariable.create(tx, args, kwargs)
        args = [ctx, *args]
        if isinstance(fn, types.FunctionType):
            sig = inspect.signature(fn)
            if len(args) - 1 == len(sig.parameters):
                args = args[1:]  # Don't use context
            fn_vt = VariableTracker.build(tx, fn, source=source, realize=True)
            return fn_vt.call_function(tx, args, kwargs)
        elif isinstance(fn, types.MethodType):
            return variables.UserMethodVariable(
                fn.__func__,
                variables.UserDefinedClassVariable(self.fn_cls),
                source=source,
            ).call_function(tx, args, kwargs)
        else:
            unimplemented(
                gb_type="Non-function or method in subclass of torch.autograd.Function",
                context=f"call_apply {self} {args} {kwargs}",
                explanation="Dynamo requires the `forward` attribute of a "
                "`torch.autograd.Function` subclass to be a standard Python "
                f"function or method. Found type `{type(fn).__name__}` instead.",
                hints=[
                    "Ensure the `forward` method is defined as a regular "
                    "function or instance method."
                ],
            )

    def call_backward(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Inlines the class's backward function. Requires the first
        # argument to hold a FakeBackwardCFunction, backward to be a plain function
        # and a source on this variable (AssertionError otherwise).
        fn = self.fn_cls.backward
        if (
            type(args[0].value)  # type: ignore[attr-defined]
            is not torch._dynamo.external_utils.FakeBackwardCFunction
        ):
            raise AssertionError(
                f"Expected FakeBackwardCFunction, got {type(args[0].value)}"
            )
        if not isinstance(fn, types.FunctionType):
            raise AssertionError(f"Expected FunctionType, got {type(fn)}")
        if self.source is None:
            raise AssertionError("AutogradFunctionVariable requires a source")
        fn_source = AttrSource(self.source, "backward")
        fn_vt = VariableTracker.build(tx, fn, source=fn_source, realize=True)
        return fn_vt.call_function(tx, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "AutogradFunctionVariable":
        # [FUNCTION DOC] Calling the function class returns a new variable for the
        # same class (arguments are ignored).
        return AutogradFunctionVariable(self.fn_cls)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] apply: if the class is in the allowed-callables set, emits a
        # single graph call through a trampoline; otherwise traces via call_apply.
        # backward: call_backward. Other names: static methods are traced through
        # trace_rules, class methods through a bound user method; anything else
        # graph-breaks.
        from .builder import wrap_fx_proxy

        if name == "apply":
            if trace_rules.is_callable_allowed(self.fn_cls):
                trampoline_autograd_apply = produce_trampoline_autograd_apply(
                    self.fn_cls
                )
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        trampoline_autograd_apply,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
            else:
                return self.call_apply(tx, args, kwargs)

        elif name == "backward":
            return self.call_backward(tx, args, kwargs)
        else:
            source = AttrSource(self.source, name) if self.source is not None else None
            try:
                obj = inspect.getattr_static(self.fn_cls, name)
            except AttributeError:
                obj = None

            if isinstance(obj, staticmethod):
                func = obj.__get__(self.fn_cls)
                traced = trace_rules.lookup(func)
                if traced is None:
                    raise AssertionError(f"trace_rules.lookup returned None for {func}")
                if source is not None:
                    return (
                        # type: ignore[attr-defined]
                        traced.create_with_source(func, source=source).call_function(
                            tx, args, kwargs
                        )
                    )
                else:
                    # type: ignore[misc]
                    return traced(func).call_function(tx, args, kwargs)
            elif isinstance(obj, classmethod):
                return variables.UserMethodVariable(
                    obj.__func__, self, source=source
                ).call_function(tx, args, kwargs)
            else:
                unimplemented(
                    gb_type="Unsupported autograd.Function method",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo does not support calling the method "
                    f"`{name}` directly on the `torch.autograd.Function` "
                    "instance. Supported methods include `apply`, `backward`, "
                    "static methods, and class methods.",
                    hints=[
                        "Ensure the method is decorated with `@staticmethod` "
                        "or `@classmethod` if it's meant to be called on the class.",
                    ],
                )


@dataclasses.dataclass
class SavedTensorBox:
    # [CLASS DOC] HELPER: Mutable holder for the list of tensor trackers saved by
    # save_for_backward; shared so updates are visible across copies.
    tensors: list[VariableTracker] = dataclasses.field(default_factory=list)


class AutogradFunctionContextVariable(UserDefinedObjectVariable):
    # [CLASS DOC] Models the ctx object passed to autograd.Function forward. State:
    # inference flag, saved_tensors (SavedTensorBox), non_differentiable and
    # dirty_tensors markers. Supports save_for_backward, mark_dirty and
    # mark_non_differentiable; other attribute mutation uses side effects.
    """
    Tracks an autograd.Function() context using mutation tracking in side_effects.py
    """

    _nonvar_fields = {
        "proxy",
        "inference",
        "saved_tensors",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    def __init__(
        self,
        value: Any,
        value_type: type | None = None,
        inference: bool = False,
        saved_tensors: Any | None = None,
        non_differentiable: Any | None = None,
        dirty_tensors: list[VariableTracker] | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the ctx bookkeeping fields on top of the base object.
        super().__init__(value=value, value_type=value_type, **kwargs)
        self.inference = inference
        self.saved_tensors = saved_tensors
        self.non_differentiable = non_differentiable
        self.dirty_tensors = dirty_tensors

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker] | None = None,
        kwargs: dict[str, VariableTracker] | None = None,
    ) -> VariableTracker:
        # [FUNCTION DOC] Creates a new tracked ctx object (inference mode with an
        # empty SavedTensorBox) registered in side effects. When positional args
        # exist and no kwargs, stores needs_input_grad (tuple of booleans, true for
        # tensors requiring grad) as an instance attribute. Returns the new variable.
        out = tx.output.side_effects.track_object_new(
            None,
            torch.autograd.function.FunctionCtx,
            functools.partial(
                AutogradFunctionContextVariable,
                inference=True,
                saved_tensors=SavedTensorBox(),
            ),
            {},
        )
        if args and not kwargs:
            # The real apply() populates ctx.needs_input_grad; mirror it as a
            # regular attribute store so reads and user writes both flow
            # through the generic side_effects machinery.
            # pyrefly: ignore [missing-attribute]
            needs_input_grad = tuple(x.is_tensor() and x.requires_grad for x in args)
            tx.output.side_effects.store_instance_dict_attr(
                out, "needs_input_grad", ConstantVariable.create(needs_input_grad)
            )
        return out

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] Returns the stored proxy; graph-breaks if none was set.
        # type: ignore[attr-defined]
        if self.proxy is None:
            unimplemented(
                gb_type="proxy not set",
                context=f"as_proxy {self}",
                explanation="Dynamo requires the autograd.Function context "
                "to be initialized with a proxy.",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        # type: ignore[attr-defined]
        return self.proxy

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Supports __setattr__ (base behavior), mark_non_differentiable
        # (records proxied args), mark_dirty (records args; graph-breaks outside
        # apply) and save_for_backward (replaces previously saved tensors with args;
        # requires saved_tensors to exist; outside inference mode needs a source and
        # no kwargs and registers with side effects). Keyword args to mark_* raise
        # an args mismatch error. Other methods graph-break. Returns a None constant.
        if name == "__setattr__":
            return super().call_method(tx, name, args, kwargs)
        elif name == "mark_non_differentiable":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            self.non_differentiable = proxy_args_kwargs(args, {})[0]
            return variables.ConstantVariable.create(None)
        elif name == "mark_dirty":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            if getattr(self, "proxy", None) is None:
                unimplemented(
                    gb_type="Unsupported autograd.Function context `mark_dirty`",
                    context=f"call_method {self} {name}",
                    explanation="Dynamo only supports tracing ctx.mark_dirty "
                    "inside autograd.Function.apply.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
            self.dirty_tensors = args
            return variables.ConstantVariable.create(None)

        if name != "save_for_backward":
            unimplemented(
                gb_type="Unsupported autograd.Function context method",
                context=f"call_method {self} {name}",
                explanation="Dynamo does not support calling the method "
                f"`{name}` on `autograd.Function` context objects. Supported "
                "methods are `__setattr__`, `save_for_backward`, "
                "`mark_dirty` and `mark_non_differentiable`.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if self.saved_tensors is None:
            unimplemented(
                gb_type="Unsupported autograd.Function context `save_for_backward`",
                context=f"call_method {self} {name}",
                explanation="Dynamo requires the `saved_tensors` attribute "
                "to be initialized on the `autograd.Function` context object.",
                hints=[
                    "Ensure that the `saved_tensors` attribute is properly "
                    "initialized before calling `save_for_backward`. "
                    "`save_for_backward` only supported on a newly constructed `torch.autograd.function.FunctionCtx`.",
                ],
            )
        if self.saved_tensors is None:
            raise AssertionError(
                "saved_tensors must be initialized before save_for_backward"
            )
        if not self.inference:
            if kwargs or not self.source:
                raise_type_error(
                    tx, "save_for_backward() requires a source and no keyword arguments"
                )
            tx.output.side_effects.track_save_for_backward(self, args)

        # In eager mode, multiple calls to .save_for_backward() will overwrite previous calls.
        if len(self.saved_tensors.tensors) > 0:
            # pyrefly: ignore [implicit-any]
            self.saved_tensors.tensors = []
        for arg in args:
            self.saved_tensors.tensors.append(arg)
        return variables.ConstantVariable.create(None)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute read: the three ctx methods become lambda wrappers
        # calling call_method; dirty_tensors and saved_tensors are returned as tuple
        # variables (dirty_tensors None constant if unset); else base lookup.
        if name in ["save_for_backward", "mark_dirty", "mark_non_differentiable"]:
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, list(args), kwargs)
            )
        if name == "dirty_tensors":
            if self.dirty_tensors is None:
                return variables.ConstantVariable.create(None)
            return variables.TupleVariable(list(self.dirty_tensors))
        if name == "saved_tensors" and self.saved_tensors is not None:
            return variables.TupleVariable(list(self.saved_tensors.tensors))

        return super().tp_getattro_impl(tx, name)


class AutogradEngineVariable(UserDefinedObjectVariable):
    # [CLASS DOC] Wraps the autograd engine object; only queue_callback is supported,
    # and only inside a compiled autograd region.
    """
    Represents a torch._C._ImperativeEngine instance.
    """

    def __init__(
        self,
        value: torch._C._ImperativeEngine,
        value_type: type[torch._C._ImperativeEngine] | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Forwards to the user-defined object constructor.
        super().__init__(value=value, value_type=value_type, **kwargs)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] queue_callback inside compiled autograd (requires fullgraph
        # or error-on-graph-break, else AssertionError) is redirected to the fake
        # engine with the final-callbacks variable prepended; outside compiled
        # autograd, or for any other method name, it graph-breaks.
        if name == "queue_callback":
            if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
                if not (tx.one_graph or tx.error_on_graph_break):
                    raise AssertionError(
                        "queue_callback() is only supported when Compiled Autograd is enabled with fullgraph=True"
                    )
                # queue_callback is a method-wrapper, no need to insert a guard.
                fn_vt = VariableTracker.build(
                    tx,
                    torch._dynamo.external_utils.FakeCompiledAutogradEngine.queue_callback,
                )
                return fn_vt.call_function(
                    tx,
                    (tx.output.side_effects.get_ca_final_callbacks_var(), *args),
                    kwargs,
                )
            else:
                unimplemented(
                    gb_type="Unsupported torch._C._ImperativeEngine.queue_callback()",
                    context=f"call_method {self} {name}",
                    explanation="queue_callback() is only supported when "
                    "Compiled Autograd is enabled with fullgraph=True.",
                    hints=[],
                )
        else:
            unimplemented(
                gb_type="Unsupported torch._C._ImperativeEngine method",
                context=f"call_method {self} {name}",
                explanation="Dynamo only supports the `queue_callback` method "
                f"on a torch._C._ImperativeEngine instance, but found: `{name}`.",
                hints=[],
            )


class LambdaVariable(VariableTracker):
    # [CLASS DOC] Wraps a Python callable operating directly on VariableTrackers;
    # calling it invokes the callable with the args and kwargs (no translator).
    # TODO: change to Ts = TypeVarTuple("Ts") for py 3.11+
    def __init__(self, fn: Callable[..., VariableTracker], **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the callable.
        super().__init__(**kwargs)
        self.fn = fn

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports types.FunctionType.
        return types.FunctionType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the stored callable with args and kwargs; tx is unused.
        return self.fn(*args, **kwargs)


class GetAttrVariable(VariableTracker):
    # [CLASS DOC] Lazy attribute access obj.name that Dynamo could not resolve to a
    # concrete tracker. State: obj (base tracker), name, py_type (optional known type
    # of the result). Calling it calls the method `name` on obj; it can be turned
    # into an FX proxy or a constant when the base allows, and reconstructs as an
    # attribute load.
    _nonvar_fields = {
        "name",
        "py_type",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        obj: VariableTracker,
        name: str,
        py_type: type | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores obj, name and optional py_type. Raises AssertionError
        # if obj is not a VariableTracker or name is not a string.
        super().__init__(**kwargs)
        if not isinstance(obj, VariableTracker):
            raise AssertionError(f"obj must be a VariableTracker, got {type(obj)}")
        if not isinstance(name, str):
            raise AssertionError(f"name must be a str, got {type(name)}")
        self.obj = obj
        self.name = name
        self.py_type = py_type  # In some cases we know the type (ex. tensor methods)

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns py_type when known, else the base implementation.
        if self.py_type is not None:
            return self.py_type
        else:
            return super().python_type()

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with class name, base and attribute name.
        return f"{self.__class__.__name__}({self.obj}, {self.name})"

    @staticmethod
    def create_getattr_proxy(base_proxy: torch.fx.Proxy, attr: str) -> Any:
        # [FUNCTION DOC] Returns the proxy for attribute attr of base_proxy (an FX
        # getattr node created lazily by the proxy).
        return getattr(base_proxy, attr)

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] Returns an FX proxy for the attribute of the base's proxy.
        return GetAttrVariable.create_getattr_proxy(self.obj.as_proxy(), self.name)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Gets the attribute from the base's python constant. Raises
        # NotImplementedError if the base is not constant or lacks the attribute.
        constant = self.obj.as_python_constant()
        try:
            return getattr(constant, self.name)
        except AttributeError:
            raise NotImplementedError(f"{self} is not a constant") from None

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hashes the resolved constant attribute value; returns the
        # hash and False (value-based hash). Graph-breaks if the value is not constant.
        # GetAttrVariable can wrap various types (bound methods, descriptors,
        # etc.) with different C tp_hash.  Resolve to the actual value and hash.
        try:
            val = self.as_python_constant()
        except (AsPythonConstantNotImplementedError, NotImplementedError):
            from ..exc import unimplemented

            unimplemented(
                gb_type="Non-constant GetAttrVariable hash",
                context=f"hash_impl {self}",
                explanation=f"Cannot hash {self} because Dynamo doesn't know how to represent "
                "the type of the getattr() result, which is not a constant.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return hash(val), False

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "ConstantVariable":
        # [FUNCTION DOC] hasattr check on the attribute value. Special case: for the
        # apply method of an autograd.Function that generates a vmap rule, checks the
        # real apply function; otherwise uses the base implementation.
        if (
            isinstance(self.obj, AutogradFunctionVariable)
            and self.name == "apply"
            and getattr(self.obj.fn_cls, "generate_vmap_rule", False)
        ):
            return variables.ConstantVariable.create(
                hasattr(self.obj.fn_cls.apply, name)
            )
        return super().call_obj_hasattr(tx, name)

    def const_getattr(self, tx: "InstructionTranslatorBase", name: str) -> Any:
        # [FUNCTION DOC] Statically reads attribute `name` of the attribute of an
        # nn module (looked up without triggering descriptors). Raises
        # NotImplementedError unless the base is an NNModuleVariable and both lookups
        # are found in instance dicts.
        if not isinstance(self.obj, variables.NNModuleVariable):
            raise NotImplementedError
        step1 = tx.output.get_submodule(self.obj.module_key)
        if self.name not in step1.__dict__:
            raise NotImplementedError
        step2 = inspect.getattr_static(step1, self.name)
        if name not in step2.__dict__:
            raise NotImplementedError
        return inspect.getattr_static(step2, name)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits the base followed by a load of the attribute.
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.name))

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Resolves the attribute on the base (falling back to a real
        # lookup when the base is a python constant), then compares that resolved
        # value with other. Graph-breaks if it cannot be resolved.
        from .object_protocol import generic_richcompare

        try:
            resolved = self.obj.tp_getattro_impl(tx, self.name)
        except NotImplementedError:
            resolved = None
        if resolved is None or isinstance(resolved, GetAttrVariable):
            if self.obj.is_python_constant():
                val = getattr(self.obj.as_python_constant(), self.name)
                resolved = VariableTracker.build(tx, val)
            else:
                unimplemented(
                    gb_type="Unresolved GetAttrVariable comparison",
                    context=f"tp_richcompare_impl {self} {op}",
                    explanation=f"Cannot compare {self} because the attribute "
                    f"could not be resolved to a concrete value.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
        return generic_richcompare(tx, resolved, other, op)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the method named by this attribute on the base.
        return self.obj.call_method(tx, self.name, args, kwargs)

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Subscripting: for the __dict__ attribute of a base that can
        # produce a dict tracker, indexes that dict; otherwise base behavior.
        if self.name == "__dict__" and hasattr(self.obj, "get_dict_vt"):
            return self.obj.get_dict_vt(tx).mp_subscript_impl(tx, key)
        return super().mp_subscript_impl(tx, key)


class CallMethodVariable(VariableTracker):
    # [CLASS DOC] A bound method tracker: obj (receiver tracker) plus method_name.
    # Calling it forwards to call_method on the receiver; reconstructs as a load of
    # the method attribute.
    """A method bound to a VT instance.

    Returned by object_generic_getattr when the MRO walk finds a method
    on a VT that has custom call_method logic.
    """

    _nonvar_fields = {
        "method_name",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        obj: VariableTracker,
        method_name: str,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the receiver and the method name.
        super().__init__(**kwargs)
        self.obj = obj
        self.method_name = method_name

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with receiver and method name.
        return f"{self.__class__.__name__}({self.obj}, {self.method_name})"

    def python_type(self) -> type:
        # [FUNCTION DOC] Type of the real bound method if the receiver is constant,
        # otherwise types.MethodType.
        try:
            return type(self.as_python_constant())
        except (AsPythonConstantNotImplementedError, AttributeError):
            return types.MethodType

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Gets the real bound method from the receiver's constant value.
        return getattr(self.obj.as_python_constant(), self.method_name)

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the real method when constant (value hash, False);
        # otherwise identity-based id and True.
        try:
            return hash(self.as_python_constant()), False
        except AsPythonConstantNotImplementedError:
            return id(self), True

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        # [FUNCTION DOC] Default identity-based object comparison.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the named method on the receiver.
        return self.obj.call_method(tx, self.method_name, args, kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits the receiver then loads the method attribute.
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.method_name))


class PythonModuleVariable(VariableTracker):
    # [CLASS DOC] Tracks a Python module object. State: value (the module) and
    # is_torch (true for torch or torch submodules). Attribute reads honor pending
    # side-effect mutations, then look up the real module and wrap the result.
    # PyModule_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/moduleobject.c#L1203
    _cpython_type = types.ModuleType

    _nonvar_fields = {
        "value",
        "is_torch",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value: types.ModuleType, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the module and computes is_torch from its name.
        super().__init__(**kwargs)
        self.value = value
        self.is_torch = self.value is torch or self.value.__name__.startswith("torch.")

    def python_type(self) -> type[types.ModuleType]:
        # [FUNCTION DOC] Returns types.ModuleType.
        return types.ModuleType

    def as_python_constant(self) -> types.ModuleType:
        # [FUNCTION DOC] Returns the module itself.
        return self.value

    def get_real_python_backed_value(self) -> types.ModuleType:
        # [FUNCTION DOC] Returns the module itself.
        return self.value

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the module.
        return f"PythonModuleVariable({self.value})"

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Reads module attribute `name`. A pending mutation recorded in
        # side effects wins. Otherwise torch modules (or names missing from the module
        # dict) use a real getattr, raising an observed AttributeError if absent;
        # other names read the module dict directly. The value is wrapped with an
        # attribute source when the module has a source.
        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            return tx.output.side_effects.load_attr(self, name)

        attr_value = None
        if self.is_torch or name not in self.value.__dict__:
            try:
                attr_value = getattr(self.value, name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
        else:
            attr_value = self.value.__dict__[name]

        source = self.source and AttrSource(self.source, name)
        return VariableTracker.build(tx, attr_value, source)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Default identity-based object comparison.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)


class TypingVariable(VariableTracker):
    # [CLASS DOC] Tracks a typing construct (e.g. typing.List or a subscripted
    # alias). State: value, the real typing object. Subscripting with a constant key
    # builds a new TypingVariable; equality and union (or) operations are evaluated
    # on the real objects; method calls graph-break.
    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the typing object.
        super().__init__(**kwargs)
        self.value = value

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Subscripts the typing object with a constant key and wraps
        # the result in a new TypingVariable. Graph-breaks on non-constant keys.
        # e.g., List[int] → typing.List[int]
        if not key.is_python_constant():
            unimplemented(
                gb_type="non-constant typing subscript",
                context=f"TypingVariable[{key}]",
                explanation=f"Cannot subscript typing construct {self.value} with a non-constant key.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        new_typing = self.value[key.as_python_constant()]
        return TypingVariable(new_typing)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] For equality or inequality between two TypingVariables,
        # returns a constant boolean comparing the wrapped objects; otherwise
        # returns a NotImplemented constant.
        if op in ("__eq__", "__ne__"):
            if istype(other, TypingVariable):
                result = self.value == other.value
                if op == "__ne__":
                    result = not result
                return ConstantVariable.create(result)
            return ConstantVariable.create(NotImplemented)
        return ConstantVariable.create(NotImplemented)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Always graph-breaks; method calls on typing objects are
        # unsupported.
        unimplemented(
            gb_type="unsupported method call on `typing` variable",
            context=f"typing variable: {self.value}, method name: {name}, args: {args}, kwargs: {kwargs}",
            explanation=f"`torch.compile` does not support method call `{name}` on `typing` variable f{self.value}.",
            hints=[
                f"Avoid calling the {name} method on {self.value}.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the union operator for generic alias types. If
        # other is not a constant, or the real operation returns NotImplemented,
        # returns a NotImplemented tracker; else the wrapped result. The reverse
        # flag is accepted but unused.
        # GenericAlias types (e.g. Callable[[int], bool]) support __or__ for
        # type unions (e.g. Callable[[int], bool] | None).
        if not other.is_python_constant():
            return VariableTracker.build(tx, NotImplemented)
        other_val = other.as_python_constant()
        # pyrefly: ignore[bad-argument-count]
        result = type(self.value).__or__(self.value, other_val)
        if result is NotImplemented:
            return VariableTracker.build(tx, NotImplemented)
        return VariableTracker.build(tx, result)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Reads attribute `name` of the typing object (pending side
        # effect mutations win), wrapping it with a source-based builder if this
        # variable has a source, else a sourceless one. A missing attribute raises
        # the real AttributeError (not an observed one).
        from .builder import SourcelessBuilder, VariableBuilder

        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            return tx.output.side_effects.load_attr(self, name)

        value = getattr(self.value, name)
        if self.source:
            attr_source = AttrSource(self.source, name)
            return VariableBuilder(tx, attr_source)(value)
        else:
            return SourcelessBuilder.create(tx, value)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped typing object.
        return self.value

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the typing object; value-based (second item False).
        return hash(self.value), False

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped typing object.
        return self.value

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] For generic aliases, loads the alias as a constant (rebuilding
        # it from parts is unreliable); other values use the base reconstruction.
        if not isinstance(self.value, types.GenericAlias):
            return super().reconstruct(codegen)
        # We're just trying to load the type here. Reconstructing the type from
        # scratch is tricky - for a type like `typing.List[int]` we'd need to
        # deconstruct the origin and args.  The origin for `List[int]` is `list`
        # and the args is `(int,)`. When we recombine those we get the parts
        # back and need to emit code for:
        #
        #     `typing.List[int]`
        #
        # But it's # worse than that - what if `typing` isn't in the globals (or
        # was loaded like `import typing as _typing ; _typing.List[int]`?) so we
        # really need to do something like:
        #
        #   `sys.modules["typing"].List[int]`
        #
        # Argh - but what if they rewrote the global `int`?  So we have to do:
        #
        #   `sys.modules["typing"].List[sys.modules["builtins"].int]`
        #
        # But where do we get `sys`? What if they never imported it or have
        # something ELSE called `sys`?
        #
        # Let's skip all that noise and just emit it as a simple const.
        #
        codegen.append_output(codegen.create_load_const(self.value))


@functools.lru_cache(maxsize=1)
def get_np_to_tnp_map() -> dict[types.BuiltinFunctionType, types.FunctionType]:
    # [FUNCTION DOC] Builds (once, cached) a mapping from numpy functions to the
    # equivalent torch._numpy callables, by scanning each paired numpy and
    # torch._numpy module and keeping callables whose name exists in numpy.
    """
    This generates a mapping from numpy modules to their torch._numpy
    modules equivalents.
    """
    from ..utils import NP_TO_TNP_MODULE

    np_fn_to_tnp_fn = {}

    for np_mod, tnp_mod in NP_TO_TNP_MODULE.items():
        for fn_name, tnp_fn in tnp_mod.__dict__.items():
            if callable(tnp_fn):
                # some internal details do leak from tnp
                # which are not part of numpy API.
                if np_fn := getattr(np_mod, fn_name, None):
                    np_fn_to_tnp_fn[np_fn] = tnp_fn

    return np_fn_to_tnp_fn


@functools.lru_cache(maxsize=1)
def get_tnp_to_np_map() -> dict[types.FunctionType, types.BuiltinFunctionType]:
    # [FUNCTION DOC] Returns (cached) the inverse of the numpy-to-torch._numpy map:
    # torch._numpy function to numpy function.
    """
    This is just the reverse mapping of get_np_to_tnp_map() - mapping from
    torch._numpy modules to numpy equivalents.
    """
    m = get_np_to_tnp_map()
    return {v: k for k, v in m.items()}


class NumpyVariable(VariableTracker):
    # [CLASS DOC] Wraps a numpy module-level object (function, dtype class, ufunc,
    # constant). Calling a function maps it to its torch._numpy equivalent and
    # traces it into the graph (or constant-folds, or builds dtype/finfo/iinfo
    # collections). constant_fold_functions lists foldable torch._numpy functions.
    """
    Wrapper around `numpy.*`. Currently, is able to trace a small subset of numpy functions as well as numpy dtypes.
    """

    constant_fold_functions = (tnp.issubdtype,)

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the numpy object.
        super().__init__(**kwargs)
        self.value = value

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped numpy object.
        return self.value

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        # [FUNCTION DOC] For equality or inequality against a tracker that has a
        # value attribute, returns a constant comparing the wrapped values; any other
        # operator or tracker gives a NotImplemented constant.
        # NumpyVariable wraps singleton numpy module attrs (dtypes, ufuncs,
        # constants); compare by the backed value for eq/ne, else NotImplemented.
        if op not in ("__eq__", "__ne__") or not hasattr(other, "value"):
            return ConstantVariable.create(NotImplemented)
        result = (
            self.value == other.value if op == "__eq__" else self.value != other.value
        )
        return ConstantVariable.create(result)

    @classmethod
    def can_constant_fold_through(cls, fn: types.FunctionType) -> bool:
        # [FUNCTION DOC] True if fn is in the constant-fold set. Raises
        # AssertionError unless fn lives in a torch._numpy module.
        mod = fn.__module__.split(".")
        if len(mod) < 2:
            raise AssertionError(
                f"Expected module path with at least 2 parts, got {mod}"
            )
        if mod[:2] != ["torch", "_numpy"]:
            raise AssertionError(
                f"Expected torch._numpy module, got {'.'.join(mod[:2])}"
            )
        return fn in cls.constant_fold_functions

    @classmethod
    def get_constant_collection_for_func(cls, fn: types.FunctionType) -> Any:
        # [FUNCTION DOC] Returns the tracker class (from the module-level
        # np_constant_collections_map) used for results of fn when it produces a
        # constant collection like dtype/finfo/iinfo, or None. Raises AssertionError
        # unless fn lives in a torch._numpy module.
        mod = fn.__module__.split(".")
        if len(mod) < 2:
            raise AssertionError(
                f"Expected module path with at least 2 parts, got {mod}"
            )
        if mod[:2] != ["torch", "_numpy"]:
            raise AssertionError(
                f"Expected torch._numpy module, got {'.'.join(mod[:2])}"
            )
        return np_constant_collections_map.get(fn)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces a call to the numpy function. Graph-breaks when
        # trace_numpy is off, when no torch._numpy equivalent exists, for numpy
        # random functions with use_numpy_random_stream, or when collection-producing
        # functions get non-constant args. Dtype/finfo/iinfo calls run the real numpy
        # function on constants and wrap the result. Otherwise patches args, constant
        # folds if allowed and all args are constant, else adds a graph call of the
        # torch._numpy function (wrapped to convert numpy arrays) and returns an
        # ndarray tracker.
        if not config.trace_numpy:
            unimplemented(
                gb_type="attempted to trace numpy function with config.trace_numpy=False",
                context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs}",
                explanation=f"Attempted to trace numpy function {self.value} "
                "while `torch._dynamo.config.trace_numpy` was set to False.",
                hints=[
                    "Set `torch._dynamo.config.trace_numpy` to True to trace numpy functions.",
                ],
            )

        from ..utils import numpy_to_tensor_wrapper
        from .tensor import NumpyNdarrayVariable

        func = get_np_to_tnp_map().get(self.value)
        if func is None:
            unimplemented(
                gb_type="attempted to trace numpy function unsupported by PyTorch",
                context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs} (corresponding torch function: {func})",
                explanation=f"Can't find numpy numpy function {self.value} in torch._numpy.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # We are dealing with a function that produces a const collection type (np.dtype, np.iinfo/np.finfo)
        if func is None:
            raise AssertionError(
                f"Could not find torch._numpy equivalent for {self.value}"
            )
        if (
            collection_variable_typ := self.get_constant_collection_for_func(func)
        ) is not None:
            try:
                return collection_variable_typ(
                    self.value(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    )
                )
            except AsPythonConstantNotImplementedError:
                unimplemented(
                    gb_type="numpy function that produces a const collection type encountered non-const arguments",
                    context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs} (corresponding torch function: {func})",
                    explanation=f"numpy function {self.value} that produces a const collection type "
                    "(e.g. np.dtype, np.iinfo/np.finfo) "
                    "received arguments that are not constant.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
        else:
            if (
                func.__module__ == "torch._numpy.random"
                and config.use_numpy_random_stream
            ):
                unimplemented(
                    gb_type="attempted to trace torch._numpy.random function with config.use_numpy_random_stream=True",
                    context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs} (corresponding torch function: {func})",
                    explanation=f"Attempted to trace {self.value} when `torch._dynamo.config.use_numpy_random_stream` "
                    "is set to True.",
                    hints=[
                        "Set `torch._dynamo.config.use_numpy_random_stream` to False.",
                        f"Avoid calling {self.value}.",
                    ],
                )

            args, kwargs = NumpyNdarrayVariable.patch_args(func.__name__, args, kwargs)

            if self.can_constant_fold_through(func) and (
                check_unspec_or_constant_args(args, kwargs)
            ):
                # constant fold
                return VariableTracker.build(
                    tx,
                    self.as_python_constant()(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    ),
                )

            # TODO Add all the functions that go from constants to constants to can_constant_fold_through
            proxy = tx.output.create_proxy(
                "call_function",
                numpy_to_tensor_wrapper(func),
                *proxy_args_kwargs(args, kwargs),
            )
            return NumpyNdarrayVariable.create(tx, proxy)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Always graph-breaks; numpy functions cannot be called as methods.
        unimplemented(
            gb_type="attempted to trace numpy.* function as a method",
            context=f"numpy function: {self.value}, args: {args}, kwargs: {kwargs}",
            explanation="Tracing numpy.* functions as methods is not supported.",
            hints=[
                *graph_break_hints.DIFFICULT,
            ],
        )

    def as_python_constant(self) -> BuiltinFunctionType:
        # [FUNCTION DOC] Returns the wrapped numpy object.
        return self.value

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] With numpy tracing on: enum classes are returned as is; type
        # objects (numpy scalar dtypes) are returned as their name string so no numpy
        # object lands in the graph. Otherwise uses the base implementation.
        if config.trace_numpy:
            # Can replace with EnumType once we drop 3.10 support
            if isinstance(self.value, enum.EnumMeta):
                # This is mostly for np._CopyMode
                return self.value
            if isinstance(self.value, type):
                # This handles numpy dtype attributes such as np.float32
                # We return a string as we don't want to serialize non-PyTorch objects in the output FX graph
                # In torch/_numpy we normalize strings to their dtypes when the input is a dtype, as NumPy does
                return self.value.__name__

        return super().as_proxy()


# Used to keep track of NULLs pushed on the stack for Python 3.11 function calls
class NullVariable(VariableTracker):
    # [CLASS DOC] Stand-in for the NULL pushed on the stack before calls (Python
    # 3.11+); reconstructs as a push-null instruction.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Pass-through to the base constructor.
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Fixed debug string.
        return "NullVariable"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a push-null instruction; graph-breaks on Python older
        # than 3.11, where that instruction does not exist.
        if sys.version_info < (3, 11):
            unimplemented(
                gb_type="cannot reconstruct NullVariable in Python < 3.11",
                context="",
                explanation="Attempted to generate PUSH_NULL instruction in Python < 3.11; "
                "where this instruction does not exist.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        codegen.append_output(create_instruction("PUSH_NULL"))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        # [FUNCTION DOC] Returns the source text for a null as a None literal.
        return "None"


class DeletedVariable(VariableTracker):
    # [CLASS DOC] Marker value recorded in side effects to denote a deleted attribute.
    """Marker used to implement delattr()"""


class StringFormatVariable(VariableTracker):
    # [CLASS DOC] A deferred str.format call with non-constant arguments. State:
    # format_string, sym_args (list) and sym_kwargs (mapping). Reconstructs as the
    # real format call so formatting happens after the graph runs.
    """
    Represents a call to str.format(), we delay calling format until after the graph.
    """

    _nonvar_fields = {"format_string", *VariableTracker._nonvar_fields}

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns str.
        return str

    @classmethod
    def create(
        cls,
        format_string: str,
        sym_args: list[VariableTracker],
        sym_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Factory: if every arg and kwarg is a constant, formats now
        # and returns a constant string tracker; otherwise returns a deferred
        # StringFormatVariable holding copies of the args.
        if all(
            x.is_python_constant()
            for x in itertools.chain(sym_args, sym_kwargs.values())
        ):
            return variables.ConstantVariable.create(
                format_string.format(
                    *[v.as_python_constant() for v in sym_args],
                    **{k: v.as_python_constant() for k, v in sym_kwargs.items()},
                )
            )
        return cls(format_string, list(sym_args), dict(sym_kwargs))

    def __init__(
        self,
        format_string: str,
        sym_args: list[VariableTracker],
        sym_kwargs: dict[str, VariableTracker],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the format string and arguments. Raises
        # AssertionError if format_string is not a str.
        super().__init__(**kwargs)
        if not isinstance(format_string, str):
            raise AssertionError(
                f"format_string must be a str, got {type(format_string)}"
            )
        self.format_string = format_string
        self.sym_args = sym_args
        self.sym_kwargs = sym_kwargs

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with format string and arguments.
        return f"{self.__class__.__name__}({self.format_string!r}, {self.sym_args!r}, {self.sym_kwargs!r})"

    @staticmethod
    def _debug_format_arg(arg: VariableTracker) -> object:
        # [FUNCTION DOC] HELPER: The constant value of arg if it has one, else its
        # debug_repr text.
        try:
            return arg.as_python_constant()
        except Exception:
            return arg.debug_repr()

    def debug_repr(self) -> str:
        # [FUNCTION DOC] Applies the format string to debug renderings of the args
        # and returns the quoted result; falls back to repr if formatting fails.
        try:
            rendered = self.format_string.format(
                *[self._debug_format_arg(arg) for arg in self.sym_args],
                **{
                    key: self._debug_format_arg(value)
                    for key, value in self.sym_kwargs.items()
                },
            )
        except Exception:
            return repr(self)
        return repr(rendered)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call of the format method of the constant format
        # string, passing the positional args as a tuple and keyword args as a
        # dict via extended call syntax.
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_const(self.format_string),
                    codegen.create_load_attr("format"),
                ]
            ),
            call_function_ex=True,
        )
        codegen(variables.TupleVariable(list(self.sym_args)))
        kwargs = {
            variables.ConstantVariable.create(k): v for k, v in self.sym_kwargs.items()
        }
        codegen(variables.ConstDictVariable(kwargs))
        codegen.extend_output(create_call_function_ex(True, False))


class ObjectVariable(VariableTracker):
    # PyBaseObject_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L7243
    _cpython_type = object

    # [CLASS DOC] Placeholder for an opaque plain object; keeps the real value and
    # compares by default object semantics.
    # placeholder for unknown / opaque values
    def __init__(self, value: object, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the opaque value.
        super().__init__(**kwargs)
        self.value = value

    def get_real_python_backed_value(self) -> object:
        # [FUNCTION DOC] Returns the stored value.
        return self.value

    def python_type(self) -> type[object]:
        # [FUNCTION DOC] Returns the object type.
        return object

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Default identity-based object comparison.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)


if sys.version_info >= (3, 15):

    class SentinelVariable(VariableTracker):
        # [CLASS DOC] Defined only on Python 3.15+. Wraps a builtin sentinel object
        # (value); reports the sentinel type.
        # Use builtins.sentinel to avoid ruff errors
        _cpython_type = builtins.sentinel

        def __init__(self, value: builtins.sentinel, **kwargs: Any) -> None:
            # [FUNCTION DOC] Stores the sentinel.
            super().__init__(**kwargs)
            self.value = value

        def get_real_python_backed_value(self) -> builtins.sentinel:
            # [FUNCTION DOC] Returns the sentinel.
            return self.value

        def python_type(self) -> type[builtins.sentinel]:
            # [FUNCTION DOC] Returns the sentinel type.
            return self._cpython_type


class DebuggingVariable(VariableTracker):
    # [CLASS DOC] A print-like or registered reorderable logging function. Calls are
    # not executed during tracing but recorded in the translator's debug_locals list
    # and replayed after the graph runs. State: value (the function).
    """
    Represents a call to a debugging function like print(), or something
    registered to config.reorderable_logging_functions.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the wrapped function.
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        # [FUNCTION DOC] Type of the wrapped function.
        return type(self.value)

    @staticmethod
    def is_reorderable_logging_function(
        obj: Any,
    ) -> TypeGuard[types.FunctionType | types.BuiltinFunctionType]:
        # [FUNCTION DOC] True if obj is a plain or builtin function listed in the
        # reorderable_logging_functions config set.
        return (
            callable(obj)
            and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
            and obj in torch._dynamo.config.reorderable_logging_functions
        )

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] In export mode, does nothing. Otherwise checks the call can be
        # reordered (graph-breaks if not), appends the (self, args) pair to the
        # translator's debug_locals for later replay, and returns a None constant.
        if tx.export:
            # For export cases, we can just make debugging functions no-ops
            return ConstantVariable.create(None)

        if not self.can_reorder_logs(self.value, args, kwargs):
            unimplemented(
                gb_type="attempted to reorder a debugging function that can't actually be reordered",
                context=f"fn: {self.value}, args: {args}, kwargs: {kwargs}",
                explanation="`torch.compile` can only reorder functions that are called "
                "without keyword arguments and whose arguments are Tensors, constants, "
                "or string formatters.",
                hints=[
                    f"Avoid calling the logging function {self.value} with args that are not supported.",
                ],
            )

        tx.debug_locals.append((self, list(args)))
        return ConstantVariable.create(None)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Reloads the function from its source; raises AssertionError
        # if there is no source.
        if self.source is None:
            raise AssertionError(
                "DebugLocalVariable requires a source for reconstruction"
            )
        return self.source.reconstruct(codegen)

    @staticmethod
    def can_reorder_logs(fn: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> bool:
        # [FUNCTION DOC] True only if there are no kwargs and every flattened argument
        # is a tensor, constant or string-format tracker. fn is unused.
        """
        Run some additional checks for what sort of function calls can we
        actually reorder.
        """

        # kwargs are dropped by the replay codegen, so refuse rather than lose them
        if kwargs:
            return False

        allowed_input_types = (
            variables.TensorVariable,
            variables.ConstantVariable,
            StringFormatVariable,
        )

        flat_args = pytree.tree_leaves(args)
        for arg in flat_args:
            if not isinstance(arg, allowed_input_types):
                return False

        return True


class IgnoredFunctionVariable(VariableTracker):
    # [CLASS DOC] A function whose calls are skipped during tracing; calling it
    # returns a None constant without running it. State: value (the function).
    """
    Represents a call to an arbitrary function that should be ignored.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the ignored function.
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        # [FUNCTION DOC] Type of the wrapped function.
        return type(self.value)

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped function.
        return self.value

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Ignores the call and returns a None constant.
        return variables.ConstantVariable.create(None)


class LoggingLoggerVariable(VariableTracker):
    # [CLASS DOC] Wraps a logging.Logger. Method calls are no-ops in export mode or
    # when listed in the ignore config, are treated as reorderable debugging calls
    # when registered as such, and otherwise graph-break. State: value (the logger).
    """
    Represents a call to any logging.Logger methods.
    """

    def __init__(self, value: logging.Logger, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the logger.
        super().__init__(**kwargs)
        self.value = value

    def python_type(self) -> type:
        # [FUNCTION DOC] Type of the wrapped logger.
        return type(self.value)

    def get_real_python_backed_value(self) -> logging.Logger:
        # [FUNCTION DOC] Returns the wrapped logger.
        return self.value

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handles logger.<name>(...). Export mode: None constant. If the
        # bound method (or its underlying function) is in ignore_logging_functions:
        # None constant. If this logger has a source and the method is in
        # reorderable_logging_functions: delegates to a DebuggingVariable call.
        # Otherwise graph-breaks with hints on how to ignore or reorder the call.
        if tx.export:
            # For export cases, we can just make logging functions no-ops.
            return variables.ConstantVariable.create(None)

        method = getattr(self.value, name, None)
        function = getattr(method, "__func__", None)

        # Unified ignore set
        ignore_set = torch._dynamo.config.ignore_logging_functions

        if method in ignore_set or function in ignore_set:
            return variables.ConstantVariable.create(None)

        reorderable = torch._dynamo.config.reorderable_logging_functions
        if self.source and (method in reorderable or function in reorderable):
            fn_var = DebuggingVariable(method, source=AttrSource(self.source, name))
            return fn_var.call_function(tx, args, kwargs)

        logger_cls = type(self.value)
        logger_cls_name = f"{logger_cls.__module__}.{logger_cls.__qualname__}"
        unimplemented(
            gb_type="logging.Logger method not supported for non-export cases",
            context=f"method: {self.value}.{name}, args: {args}, kwargs: {kwargs}",
            explanation="For non-export cases, logging.Logger methods are only supported if the logger "
            "has a source and the method is registered as reorderable.",
            hints=[
                "If you do not need this logging side effect, add the exact method being called to `torch._dynamo.config.ignore_logging_functions`. Dynamo will skip the call and return `None`.",
                f"For example, for `logger.{name}(...)`, use `torch._dynamo.config.ignore_logging_functions.add(logger.{name})`. If `{name}` is defined on the logger class, add the class method `{logger_cls_name}.{name}` to ignore this method for all instances of that class.",
                f"Dynamo does not trace into logging.Logger method bodies, so only the method you call directly (`{name}`) is checked against the ignore set. Ignoring a method that `{name}` calls internally has no effect.",
                f"If you need the log side effect to run, then you can try one of (1) create the logger outside the compiled region and add the method to `torch._dynamo.config.reorderable_logging_functions` (e.g. `torch._dynamo.config.reorderable_logging_functions.add(logger.{name})`) so that it runs after the compiled region, as long as it is called without kwargs and its arguments are tensors, constants, or string formatters, (2) `torch._higher_order_ops.print(...)`, (3) wrap the logging call in a custom op (marked as mutable), or (4) preserve the logging contents and move the logging call outside the compiled region.",
            ],
        )


class ConstantLikeVariable(VariableTracker):
    # [CLASS DOC] Wraps a compile-time constant that is not a simple literal (e.g.
    # regex match, numpy finfo/iinfo). Method calls run for real when all args are
    # constant and results are literals or regex matches; attribute reads convert
    # numpy floats/dtypes/scalar types to the matching trackers. Class attributes
    # np_floating, np_dtype, np_generic are numpy types, or dummy types if numpy is
    # not installed (np_generic is then undefined).
    """self.value is a compile-time constant, but not a literal"""

    try:
        from numpy import (
            dtype as np_dtype,
            floating as np_floating,
            generic as np_generic,
        )
    except ImportError:
        # type: ignore[misc, assignment]
        np_floating = type("invalid_type", (), {})
        # type: ignore[misc, assignment]
        np_dtype = type("invalid_type", (), {})

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the constant value.
        super().__init__(**kwargs)
        self.value = value

    @property
    def _error_prefix(self) -> str:
        # [FUNCTION DOC] HELPER: Qualified type name of the value (module omitted for
        # builtins), used in graph break messages.
        """Dynamically compute the prefix from the value's type"""
        t = type(self.value)

        # For builtins (int, str, etc.), just return the name
        if t.__module__ == "builtins":
            return t.__qualname__

        return f"{t.__module__}.{t.__qualname__}"

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped value.
        return self.value

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Value hash; second item False (value-based).
        return hash(self.value), False

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Compares using the generic python-constant comparison.
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the real method with constant-converted args. Graph-breaks
        # if any arg is not constant. The result becomes a constant tracker if it is a
        # literal, a new ConstantLikeVariable if it is a regex match, else graph-breaks.
        # pyrefly: ignore [implicit-any]
        cargs, ckwargs = [], {}
        try:
            # we only support constant propagation for methods
            cargs = [x.as_python_constant() for x in args]
            ckwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            unimplemented(
                gb_type="constant-like method call with non-constant args",
                context=f"{self._error_prefix}.{name}(*{args}, **{kwargs})",
                explanation=f"Attempted to call {self._error_prefix}.{name} with non-constant args.",
                hints=[
                    "Ensure that the args to the method call are constant (int, str, etc.).",
                ],
            )

        result = getattr(self.value, name)(*cargs, **ckwargs)

        if variables.ConstantVariable.is_literal(result):
            return VariableTracker.build(tx, result)
        if isinstance(result, re.Match):
            return ConstantLikeVariable(result)

        unimplemented(
            gb_type="constant-like method call with unsupported return type",
            context=f"{self._error_prefix}.{name}(*{args}, **{kwargs}) returned {result}",
            explanation=f"Attempted to call {self._error_prefix}.{name}, got unsupported return value {result}.",
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Reads the real attribute: numpy floats become Python floats,
        # numpy dtypes become NumpyDTypeVariable, numpy scalar types become
        # NumpyVariable, literals become constant trackers; anything else is a lazy
        # GetAttrVariable. A missing attribute raises the real AttributeError.
        result = getattr(self.value, name)
        if isinstance(result, self.np_floating):
            result = float(result)
        if isinstance(result, self.np_dtype):
            return NumpyDTypeVariable(result)
        if isinstance(result, type) and issubclass(result, self.np_generic):
            # things like x.dtype.type
            return NumpyVariable(result)
        if variables.ConstantVariable.is_literal(result):
            return VariableTracker.build(tx, result)
        return GetAttrVariable(self, name, py_type=type(result))


class NumpyDTypeVariable(ConstantLikeVariable):
    # [CLASS DOC] A constant numpy dtype object; appears in graphs as the name of its
    # scalar type.
    def as_proxy(self) -> str:
        # [FUNCTION DOC] Returns the name of the dtype's scalar type as a string.
        """Similar to how numpy dtype descriptors (e.g. np.float32 ) are handled by NumpyVariable:

        np.dtype() objects are serialized as strings, torch._numpy wrappers will normalize to the torch dtype.
        This also handles unsupported things nicely (i.e. structured arrays and object arrays).
        """
        return self.value.type.__name__


np_constant_collections_map = {
    tnp.finfo: ConstantLikeVariable,
    tnp.iinfo: ConstantLikeVariable,
    tnp.dtype: NumpyDTypeVariable,
}


class ContextVarVariable(VariableTracker):
    # [CLASS DOC] Wraps a contextvars.ContextVar (cv_obj). get is resolved at trace
    # time using a source that makes a guard re-check the value; set and reset
    # graph-break; the name attribute is a constant. Requires a source for get.
    """Wraps a contextvars.ContextVar for Dynamo tracing.

    .get() is resolved at trace time with a guard that re-checks at cache time.
    .set() and .reset() graph-break in Phase 1.
    """

    _nonvar_fields = {
        "cv_obj",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, cv_obj: contextvars.ContextVar[Any], **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the context variable.
        super().__init__(**kwargs)
        self.cv_obj = cv_obj

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns contextvars.ContextVar.
        return contextvars.ContextVar

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # [FUNCTION DOC] get is handled by _handle_get; set and reset graph-break;
        # other methods use the base implementation.
        if name == "get":
            return self._handle_get(tx, args, kwargs)
        elif name in ("set", "reset"):
            unimplemented(
                gb_type="ContextVar mutation not supported",
                context=f"ContextVar('{self.cv_obj.name}').{name}()",
                explanation=(
                    f"ContextVar.{name}() is not yet supported inside "
                    f"torch.compile. Move the .{name}() call outside the "
                    f"compiled region."
                ),
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return super().call_method(tx, name, args, kwargs)

    def _handle_get(
        self,
        tx: "InstructionTranslatorBase",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # [FUNCTION DOC] HELPER: Implements get. Keyword args or more than one arg
        # raise an observed TypeError; a default must be a safe compile-time constant
        # (else graph-break). Reads the current value now and returns it wrapped with a
        # ContextVarGetSource (base source, default flag and value) so guards re-check
        # it. Raises AssertionError without a source.
        from ..source import ContextVarGetSource
        from ..utils import is_safe_constant
        from .base import NO_SUCH_SUBOBJ

        if kwargs:
            raise_observed_exception(
                TypeError, tx, args=["ContextVar.get() takes no keyword arguments"]
            )
        if len(args) > 1:
            raise_observed_exception(
                TypeError,
                tx,
                args=[f"get expected at most 1 argument, got {len(args)}"],
            )

        default_var = args[0] if args else None
        has_default = default_var is not None
        default_value = None

        if has_default:
            default_value = default_var.get_real_python_backed_value()
            if default_value is NO_SUCH_SUBOBJ or not is_safe_constant(default_value):
                unimplemented(
                    gb_type="ContextVar.get() with non-constant default",
                    context=f"ContextVar('{self.cv_obj.name}').get(...)",
                    explanation=(
                        "ContextVar.get() default argument must be a "
                        "compile-time constant inside torch.compile."
                    ),
                    hints=[*graph_break_hints.SUPPORTABLE],
                )

        value = self._get_value(tx, has_default, default_value)

        if not self.source:
            raise AssertionError("ContextVarVariable requires a source for .get()")
        value_source = ContextVarGetSource(
            base=self.source,
            has_default=has_default,
            default_value=default_value,
        )
        return VariableTracker.build(tx, value, source=value_source)

    def _get_value(
        self,
        tx: "InstructionTranslatorBase",
        has_default: bool,
        default_value: Any,
    ) -> Any:
        # [FUNCTION DOC] HELPER: Reads the real context variable, with the default if
        # has_default is set. A missing value without default raises an observed
        # LookupError.
        if has_default:
            return self.cv_obj.get(default_value)
        try:
            return self.cv_obj.get()
        except LookupError:
            raise_observed_exception(LookupError, tx, args=[f"{self.cv_obj!r}"])

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] The name attribute is a string constant; others use base lookup.
        if name == "name":
            return ConstantVariable.create(self.cv_obj.name)
        return super().tp_getattro_impl(tx, name)


class RandomClassVariable(VariableTracker):
    # [CLASS DOC] The random.Random class; calling it creates a RandomVariable.
    """random.Random"""

    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Pass-through to the base constructor.
        super().__init__(**kwargs)

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the type metaclass (the value is a class).
        return type

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "RandomVariable":
        # [FUNCTION DOC] Creates a RandomVariable seeded with the optional single
        # argument (None constant if absent); marked as newly created for mutation
        # tracking. Graph-breaks on more than one arg or any kwargs.
        if len(args) > 1 or kwargs:
            unimplemented(
                gb_type="random.Random() with improper arguments",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="random.Random() with > 1 arg or with kwargs is not supported.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
        seed = variables.ConstantVariable.create(None) if len(args) == 0 else args[0]
        return RandomVariable(
            seed=seed, mutation_type=variables.base.ValueMutationNew()
        )


class RandomVariable(VariableTracker):
    # [CLASS DOC] Tracks a random.Random instance by holding a private real Random
    # object (random) whose state is advanced during tracing. Supports seed,
    # getstate, setstate, shuffle, sample, random, randint, randrange and uniform
    # through tp_methods (_supported_fn_names lists the value-producing ones). Mutating
    # methods notify side effects; reconstruct recreates the object with its state.
    """random.Random()

    Implemented by wrapping a VariableTracker around a random.Random object.
    The supported methods for the random.Random object cannot be overridden.
    Assumes that random objects behave the same given a set seed or state.
    """

    _cpython_type = random.Random

    _nonvar_fields = {
        "random",
        *VariableTracker._nonvar_fields,
    }

    _supported_fn_names = {
        "random",
        "randint",
        "randrange",
        "uniform",
    }

    def __init__(
        self,
        rand: random.Random | None = None,
        seed: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Creates the private Random: copies the state of rand if given
        # (must be an unmodified random.Random, else AssertionError), otherwise seeds
        # a new one from seed (a constant-convertible tracker, or None).
        super().__init__(**kwargs)
        if rand is not None:
            if not self.is_supported_random_obj(rand):
                raise AssertionError(
                    "Unsupported random.Random object with overridden methods"
                )
            self.random = random.Random()
            self.random.setstate(rand.getstate())
        else:
            seed = seed.as_python_constant() if seed is not None else None
            self.random = random.Random(seed)

    def python_type(self) -> type[random.Random]:
        # [FUNCTION DOC] Returns random.Random.
        return random.Random

    def as_python_constant(self) -> random.Random:
        # [FUNCTION DOC] Returns the private real Random object (not a copy).
        return self.random

    @staticmethod
    def is_supported_random_obj(val: Random) -> bool:
        # [FUNCTION DOC] True only if val is exactly random.Random and each of the
        # supported methods plus seed, getstate and setstate is the stock
        # implementation (not overridden or monkey patched).
        if type(val) is not random.Random:
            return False
        for name in itertools.chain(
            RandomVariable._supported_fn_names, ("seed", "getstate", "setstate")
        ):
            if not hasattr(val, name):
                return False
            meth = getattr(val, name)
            if inspect.isbuiltin(meth):
                # e.g. random.Random.random
                if meth != getattr(random.Random, name).__get__(val):
                    return False
            else:
                if getattr(meth, "__func__", None) is not getattr(random.Random, name):
                    return False
        return True

    @staticmethod
    def check_state(state: tuple[int, tuple[int, ...], float | None]) -> None:
        # [FUNCTION DOC] HELPER: Validates a Random state: a tuple of an int, a tuple
        # of ints, and a float or None. Raises AssertionError otherwise.
        if type(state) is not tuple:
            raise AssertionError(f"state must be a tuple, got {type(state)}")
        if type(state[0]) is not int:
            raise AssertionError(f"state[0] must be an int, got {type(state[0])}")
        if type(state[1]) is not tuple:
            raise AssertionError(f"state[1] must be a tuple, got {type(state[1])}")
        if not all(type(x) is int for x in state[1]):
            raise AssertionError("all elements of state[1] must be int")
        if state[2] is not None and type(state[2]) is not float:
            raise AssertionError(
                f"state[2] must be None or float, got {type(state[2])}"
            )

    @staticmethod
    def wrap_state(state: tuple[int, tuple[int, ...], float | None]) -> TupleVariable:
        # [FUNCTION DOC] HELPER: Validates a real Random state and converts it to a
        # nested tuple tracker of constants.
        RandomVariable.check_state(state)
        return variables.TupleVariable(
            [
                variables.ConstantVariable.create(state[0]),
                variables.TupleVariable(
                    [variables.ConstantVariable.create(x) for x in state[1]]
                ),
                variables.ConstantVariable.create(state[2]),
            ]
        )

    @staticmethod
    def unwrap_state(
        state: VariableTracker,
    ) -> tuple[int, tuple[int, ...], float | None]:
        # [FUNCTION DOC] HELPER: Converts a constant state tracker to a real state
        # tuple and validates it (AssertionError if malformed).
        state_obj = state.as_python_constant()
        RandomVariable.check_state(state_obj)
        return state_obj

    def seed(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method seed: records the mutation, reseeds the private Random
        # with constant-converted args, returns a None constant.
        tx.output.side_effects.mutation(self)
        self.random.seed(
            *[x.as_python_constant() for x in args],
            **{key: val.as_python_constant() for key, val in kwargs.items()},
        )
        return variables.ConstantVariable.create(None)

    def getstate(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method getstate: returns the current state as a tuple tracker.
        return self.wrap_state(self.random.getstate())

    def setstate(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method setstate: records the mutation and installs the state
        # given by the first argument (must be a valid constant state); returns None.
        tx.output.side_effects.mutation(self)
        self.random.setstate(self.unwrap_state(args[0]))
        return variables.ConstantVariable.create(None)

    def shuffle(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method shuffle: takes exactly one positional list or tuple
        # tracker and no keywords. Advances the RNG by shuffling an index list of the
        # same length, then reorders the tracker's items in place (recording the
        # mutation). Returns a None constant; AssertionError if the arg has no items.
        name = "shuffle"
        check_positional(tx, name, len(args), 1, 1)
        no_keywords(tx, name, kwargs)
        seq = args[0].realize()
        tx.output.side_effects.mutation(self)
        # shuffle's permutation depends only on the sequence length and the
        # RNG state, not on the elements, so shuffle a list of indices to
        # both advance the symbolic RNG and obtain the permutation to apply.
        if not hasattr(seq, "items"):
            raise AssertionError("shuffle only supports ListVariable and TupleVariable")
        perm = list(range(len(seq.items)))
        self.random.shuffle(perm)
        tx.output.side_effects.mutation(seq)
        seq.items[:] = [seq.items[i] for i in perm]
        return variables.ConstantVariable.create(None)

    def sample(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method sample: exactly two positional args (population, count k;
        # no keywords). k must be a constant int in range (ValueError otherwise).
        # Samples indices from the RNG and returns a new list tracker of the chosen
        # elements, marked as newly created.
        name = "sample"
        check_positional(tx, name, len(args), 2, 2)
        no_keywords(tx, name, kwargs)
        elems = unpack_iterable(tx, args[0])
        k = args[1].as_python_constant()
        if not isinstance(k, int) or k < 0 or k > len(elems):
            raise_value_error(
                tx,
                "Sample larger than population or is negative",
            )
        tx.output.side_effects.mutation(self)
        # Like shuffle, sample's selected positions depend only on the
        # population length and RNG state, so sample over an index range to
        # advance the symbolic RNG and pick the population elements to keep.
        indices = self.random.sample(range(len(elems)), k)
        return variables.ListVariable(
            [elems[i] for i in indices],
            mutation_type=variables.base.ValueMutationNew(),
        )

    def _call_random(self, tx, name, args, kwargs):
        # [FUNCTION DOC] HELPER: Shared implementation of random, randint, randrange
        # and uniform. Records the mutation, advances the private Random by running
        # the real method on constant-converted args, and delegates to call_random_fn
        # with a function replaying the method from the pre-call state (so the result
        # may be computed with tensors or symbolic values).
        tx.output.side_effects.mutation(self)
        state = self.random.getstate()

        def call_random_meth(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Runs the named method on a fresh Random restored
            # to the saved pre-call state and returns the result.
            r = random.Random()
            r.setstate(state)
            return getattr(r, name)(*args, **kwargs)

        # self.random state not actually updated by call_random_meth, so update here
        # by calling the method
        getattr(self.random, name)(
            *[x.as_python_constant() for x in args],
            **{k: v.as_python_constant() for k, v in kwargs.items()},
        )

        return call_random_fn(tx, call_random_meth, args, kwargs)

    def _random(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Method random via _call_random.
        return self._call_random(tx, "random", args, kwargs)

    def _randint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Method randint via _call_random.
        return self._call_random(tx, "randint", args, kwargs)

    def _randrange(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Method randrange via _call_random.
        return self._call_random(tx, "randrange", args, kwargs)

    def _uniform(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Method uniform via _call_random.
        return self._call_random(tx, "uniform", args, kwargs)

    tp_methods = {
        "seed": Method(seed),
        "getstate": Method(getstate),
        "setstate": Method(setstate),
        "shuffle": Method(shuffle),
        "sample": Method(sample),
        "random": Method(_random),
        "randint": Method(_randint),
        "randrange": Method(_randrange),
        "uniform": Method(_uniform),
    }

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits code creating a new random.Random and restoring the
        # current state through setstate, leaving the object on the stack.
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(random),
                    codegen.create_load_attr("Random"),
                ]
            )
        )
        codegen.call_function(0, False)
        # NOTE using add_push_null may result in NULL being duplicated
        # so defer the push_null to call_function
        codegen.dup_top()
        codegen.load_attr("setstate")
        codegen(self.wrap_state(self.random.getstate()))
        codegen.call_function(1, True)
        codegen.pop_top()


class WeakRefVariable(VariableTracker):
    # [CLASS DOC] Tracks a weakref.ref. State: referent_vt (tracker of the referent,
    # returned when the ref is called) and callback_vt (tracker of the callback).
    # Hashing and equality follow the referent.
    def python_type(self) -> type:
        # [FUNCTION DOC] Returns weakref.ref.
        return weakref.ref

    @staticmethod
    # pyrefly: ignore [bad-override, bad-param-name-override]
    def build(
        tx: "InstructionTranslatorBase",
        weakref_value: weakref.ReferenceType[Any],
        source: Source | None,
        **options: Any,
    ) -> "WeakRefVariable":
        # [FUNCTION DOC] Builds a tracker for an existing weakref: wraps its callback
        # (attribute source) and its live referent (weakref-call source), then
        # creates the WeakRefVariable with that source. AssertionError if source is None.
        if source is None:
            raise AssertionError("WeakRefVariable.build requires a source")
        callback = weakref_value.__callback__
        callback_source = source and AttrSource(source, "__callback__")
        callback_vt = VariableTracker.build(tx, callback, callback_source)
        referent = weakref_value()
        source = source and WeakRefCallSource(source)
        referent_vt = VariableTracker.build(tx, referent, source)
        options["source"] = source
        return WeakRefVariable(referent_vt, callback_vt, **options)

    def __init__(
        self, referent_vt: VariableTracker, callback_vt: VariableTracker, **options: Any
    ) -> None:
        # [FUNCTION DOC] Stores the referent and callback trackers.
        super().__init__(**options)
        self.referent_vt = referent_vt
        self.callback_vt = callback_vt

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calling the ref returns the referent tracker (assumes alive).
        return self.referent_vt

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to weakref.ref with the referent and callback.
        codegen.add_push_null(lambda: codegen.load_import_from("weakref", "ref"))
        codegen(self.referent_vt)
        codegen(self.callback_vt)
        codegen.extend_output(create_call_function(2, False))

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the referent via the generic hash helper.
        # CPython weakref_hash: hash(referent)
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/weakrefobject.c#L186
        from .object_protocol import generic_hash_impl

        return generic_hash_impl(tx, self.referent_vt)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        # [FUNCTION DOC] Equality or inequality between two WeakRefVariables compares
        # their referents; other operators or other types give a NotImplemented constant.
        from .object_protocol import generic_richcompare

        # Weak references only support equality, not ordering. Two weak references
        # are equal if the underlying objects are equal. If the underlying object has
        # gone away, they are equal if they are identical.
        if op not in ("__eq__", "__ne__") or not isinstance(other, WeakRefVariable):
            return ConstantVariable.create(NotImplemented)
        return generic_richcompare(tx, self.referent_vt, other.referent_vt, op)
