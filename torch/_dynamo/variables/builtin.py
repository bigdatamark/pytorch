# [MODULE DOC] Variable trackers that let Dynamo symbolically execute Python builtins
# (len, isinstance, getattr, min/max, arithmetic operators, etc.). Main contents:
# BaseBuiltinVariable, BuiltinVariable (generic builtin/operator dispatcher), and
# specialized trackers for dict, iter, getattr, hasattr, setattr and list, plus operator
# tables used to map operators to Tensor methods and a few helpers. Import-time side
# effects: only builds module-level constant tables; the tensor-method map is filled lazily
# on first use, and an extra foldable entry is added for complex on Python 3.14+.
"""
Built-in function and type variable tracking for TorchDynamo's symbolic execution.

This module contains variable tracker classes for Python built-in functions, types,
and operations during graph compilation. It handles symbolic execution of:

- Built-in functions (len, getattr, isinstance, etc.)
- Type constructors (int, float, str, list, dict, etc.)
- Built-in operators and methods
- Special Python constructs (super, hasattr, etc.)

Key classes:
- BuiltinVariable: Tracks built-in functions and handles their execution
- TypeVariable: Manages type constructor calls and type checking
- SuperVariable: Handles super() calls in class hierarchies

These variable trackers ensure that built-in Python operations are correctly
handled during symbolic execution, either by executing them directly when safe
or by creating appropriate graph nodes when needed.
"""

import abc
import ast
import builtins
import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import sys
import types
import typing
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch.overrides import BaseTorchFunctionMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import graph_break_hints, polyfills, variables
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    ObservedTypeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    raise_value_error,
    unimplemented,
    Unsupported,
    UserError,
    UserErrorType,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    is_constant_source,
    LocalSource,
    Source,
    TypeSource,
)
from ..utils import (
    check_constant_args,
    check_numpy_ndarray_args,
    check_unspec_or_constant_args,
    check_unspec_python_args,
    dict_methods,
    extract_fake_example_value,
    get_fake_value,
    is_tensor_getset_descriptor,
    istype,
    numpy_operator_wrapper,
    proxy_args_kwargs,
    raise_args_mismatch,
    str_methods,
    tensortype_to_dtype,
    unpack_iterable,
)
from .base import (
    AsPythonConstantNotImplementedError,
    GetSet,
    Member,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable, FakeIdVariable
from .dicts import (
    ConstDictVariable,
    DictItemsVariable,
    DictKeysVariable,
    DictViewVariable,
    OrderedDictVariable,
)
from .hashable import is_hashable
from .lists import BaseListVariable, ListVariable, TupleIteratorVariable, TupleVariable
from .misc import CellVariable, NullVariable, StringFormatVariable
from .object_protocol import (
    _NO_DEFAULT,
    binary_iop,
    binary_op,
    generic_delitem,
    generic_getattr,
    generic_getitem,
    generic_getiter,
    generic_hash,
    generic_is_true,
    generic_issubclass,
    generic_repr,
    generic_richcompare,
    generic_setitem,
    generic_size,
    generic_str,
    maybe_get_python_type,
    pycallable_check,
    pyiter_check,
    pylong_from_base,
    pynumber_absolute,
    pynumber_add,
    pynumber_float,
    pynumber_index,
    pynumber_inplace_add,
    pynumber_inplace_matrix_multiply,
    pynumber_inplace_multiply,
    pynumber_int,
    pynumber_invert,
    pynumber_matrix_multiply,
    pynumber_multiply,
    pynumber_negative,
    pynumber_positive,
    pynumber_tobase,
    pysequence_check,
    pysequence_contains,
    python_constant_richcompare_impl,
    ternary_iop,
    ternary_op,
    type_implements_mp_length,
    type_implements_sq_length,
    vt_identity_compare,
)
from .sets import FrozensetVariable, SetVariable
from .tensor import (
    FakeItemVariable,
    supported_comparison_ops,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .user_defined import UserDefinedObjectVariable, UserDefinedVariable


if TYPE_CHECKING:
    # Cyclic dependency...
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

log = logging.getLogger(__name__)


IN_PLACE_DESUGARING_MAP = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.ifloordiv: operator.floordiv,
    operator.itruediv: operator.truediv,
    operator.imod: operator.mod,
    operator.imatmul: operator.imatmul,
    operator.ilshift: operator.lshift,
    operator.irshift: operator.rshift,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}

_BUILTIN_CONSTANT_FOLDABLE_METHODS: dict[type, frozenset[str]] = {
    int: frozenset({"__new__", "from_bytes"}),
    bool: frozenset({"__new__", "from_bytes"}),
    float: frozenset({"fromhex", "hex"}),
}
if sys.version_info >= (3, 14):
    _BUILTIN_CONSTANT_FOLDABLE_METHODS[complex] = frozenset({"from_number"})


_HandlerCallback = Callable[
    ["InstructionTranslatorBase", typing.Any, typing.Any], VariableTracker | None
]
_TrackersType = type[VariableTracker] | tuple[type[VariableTracker], ...]
_OPERATOR_TO_DUNDER: dict[Callable[..., Any], str] = {
    operator.eq: "__eq__",
    operator.ne: "__ne__",
    operator.lt: "__lt__",
    operator.le: "__le__",
    operator.gt: "__gt__",
    operator.ge: "__ge__",
}


bin_ops = (
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
)

bin_int_ops = (
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
)

un_int_ops = (operator.invert,)

tensor_and_int_ops = (
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
)

un_ops = (
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
)

_SET_LIKE_OP_SUPPORT: tuple[type[VariableTracker], ...] = (
    DictItemsVariable,
    DictKeysVariable,
    SetVariable,
    UserDefinedObjectVariable,
)

BUILTIN_TO_TENSOR_FN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# These functions represent the r* versions of the above ops
# Basically, if __add__(1, Tensor) is called, it is translated
# to __radd__(Tensor, 1).
# In the builtin var, we check if there is a tensor in the first args position,
# if not, we swap the args and use the r* version of the op.
BUILTIN_TO_TENSOR_RFN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# Sentinel for `inspect.getattr_static` lookups that must distinguish
# "attribute absent" from "attribute is None" (e.g. `__reversed__ = None`
# opt-out).
_MISSING_SENTINEL = object()

_COMPUTED_LAZY_CONSTANT_OPS: frozenset[Callable[..., Any]] = frozenset(
    [
        operator.add,
        operator.sub,
        operator.mul,
    ]
)


def _try_computed_lazy_constant(
    fn: Callable[..., Any], args: list[VariableTracker]
) -> VariableTracker | None:
    # [FUNCTION DOC] HELPER: Given an operator and its argument trackers, build a computed
    # lazy constant tracker when the operator is add/sub/mul (in-place forms desugared first),
    # there are exactly two args, all are constants or lazy constants, and at least one lazy
    # constant is still unrealized. Returns None in every other case so the caller falls back.
    """Build a ComputedLazyConstantVariable for fn(*args), or None to fall back."""
    from .lazy import ComputedLazyConstantVariable, LazyConstantVariable

    fn = IN_PLACE_DESUGARING_MAP.get(fn, fn)
    if fn not in _COMPUTED_LAZY_CONSTANT_OPS or len(args) != 2:
        return None
    any_unrealized = False
    for arg in args:
        if (
            isinstance(arg, (LazyConstantVariable, ComputedLazyConstantVariable))
            and not arg.is_realized()
        ):
            any_unrealized = True
        elif not isinstance(arg, ConstantVariable):
            return None
    if not any_unrealized:
        return None
    return ComputedLazyConstantVariable.create(fn, args)


def populate_builtin_to_tensor_fn_map() -> None:
    # [FUNCTION DOC] Fill the module-level maps from builtin operators to the Tensor
    # methods they dispatch to (forward map) and to the reflected Tensor methods (reverse
    # map, only when it differs from the forward one). It runs each operator on small
    # sample tensors under a torch-function mode that records the function invoked. Does
    # nothing if the forward map is already populated. Mutates global state; raises an
    # error if no function was recorded for an operator.
    global BUILTIN_TO_TENSOR_FN_MAP
    if len(BUILTIN_TO_TENSOR_FN_MAP) > 0:
        # Only populate once; after there are elements present no need to
        # repopulate
        return
    most_recent_func: Callable[..., Any] | None = None

    class GetMethodMode(BaseTorchFunctionMode):
        # [CLASS DOC] HELPER: Torch-function mode that remembers the most recent function
        # intercepted (stored in the enclosing closure variable) and then runs it normally.
        """
        Mode to extract the correct methods from torch function invocations
        (Used to get the correct torch.Tensor methods from builtins)
        """

        def __torch_function__(
            self,
            func: Callable[..., Any],
            types: Any,
            args: Sequence[Any] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            # [FUNCTION DOC] HELPER: Record func as the most recent function seen, then call
            # it with the given args and kwargs (missing kwargs treated as empty) and return
            # its result. Side effect: overwrites the enclosing recorded-function variable.
            kwargs = kwargs or {}
            nonlocal most_recent_func
            most_recent_func = func
            return func(*args, **kwargs)

    inp0 = torch.ones(1)
    inp1 = torch.ones(1)
    inp0_int = torch.ones(1, dtype=torch.int32)
    inp1_int = torch.ones(1, dtype=torch.int32)
    with GetMethodMode():
        setups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (lambda o: o(inp0), un_ops),
            (lambda o: o(inp0_int), un_int_ops),
            (lambda o: o(inp0, inp1), bin_ops),
            (lambda o: o(inp0_int, inp1_int), bin_int_ops),
            (lambda o: o(inp0_int, 0), tensor_and_int_ops),
        ]
        for setup_fn, op_list in setups_and_oplists:
            for op in op_list:
                setup_fn(op)
                if most_recent_func is None:
                    raise AssertionError(
                        f"most_recent_func is None after setup for op {op}"
                    )
                BUILTIN_TO_TENSOR_FN_MAP[op] = most_recent_func

        # gather the reverse functions
        rsetups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (
                lambda o: o(1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: o(1, inp1_int), bin_int_ops),
            (lambda o: o(0, inp0_int), tensor_and_int_ops),
        ]

        rskips = {operator.matmul, operator.imatmul, operator.getitem}
        for setup_fn, op_list in rsetups_and_oplists:
            for op in op_list:
                if op in rskips:
                    continue
                setup_fn(op)
                if most_recent_func is None:
                    raise AssertionError(
                        f"most_recent_func is None after setup for reverse op {op}"
                    )
                if most_recent_func != BUILTIN_TO_TENSOR_FN_MAP[op]:
                    BUILTIN_TO_TENSOR_RFN_MAP[op] = most_recent_func


class BaseBuiltinVariable(VariableTracker):
    # [CLASS DOC] Common base for trackers of builtin callables. Subclasses expose the
    # wrapped callable through the class attribute _fn (BuiltinVariable overrides this with
    # an instance attribute). Provides type-attribute readers (__bases__, __base__,
    # __flags__) registered in tp_getset/tp_members, a guard-installing factory, constant
    # conversion, reconstruction as a global load, hasattr, hashing, rich comparison, and
    # special handling of str/repr methods called on object or type.
    """
    Common base class for all builtin variable trackers (BuiltinVariable,
    DictBuiltinVariable, IterBuiltinVariable, and future specialized builtins).

    Provides shared implementations for guard installation, hasattr tracing,
    and Python-level hashability/equality.

    Specialized subclasses (e.g. DictBuiltinVariable) set `_fn` as a class
    attribute. BuiltinVariable stores the callable on the instance as `self.fn`
    and overrides as_python_constant / reconstruct / tp_getattro_impl accordingly.
    """

    _fn: Any = None

    # Type attribute readers shared by BaseBuiltinVariable and BuiltinVariable.
    # Each reader declines (returns None) when the wrapped callable is not a type,
    # letting getattro_impl fall through to its dynamic getattr handling.
    # CPython classification (Objects/typeobject.c):
    #   __bases__ -> type_getsets[] type_get_bases (getset)
    #   __base__  -> type_members[] {T_OBJECT, offsetof(tp_base), Py_READONLY}
    #   __flags__ -> type_members[] {T_ULONG, offsetof(tp_flags), Py_READONLY}
    def _type_get_bases(
        self: "BaseBuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the __bases__ attribute. If the wrapped callable
        # is a class, returns a tuple tracker of trackers for each base class (sourced via
        # attribute and index when this tracker has a source); otherwise returns None so the
        # caller falls through to dynamic attribute handling.
        fn = self.as_python_constant()
        if not isinstance(fn, type):
            return None
        source = self.source and AttrSource(self.source, "__bases__")
        items = [
            VariableTracker.build(tx, b, source and GetItemSource(source, i))
            for i, b in enumerate(fn.__bases__)
        ]
        return variables.TupleVariable(items, source=source)

    def _type_get_base(
        self: "BaseBuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the __base__ attribute. Returns a tracker for the
        # wrapped class's primary base class, or None if the wrapped callable is not a class.
        fn = self.as_python_constant()
        if not isinstance(fn, type):
            return None
        source = self.source and AttrSource(self.source, "__base__")
        return VariableTracker.build(tx, fn.__base__, source)

    def _type_get_flags(
        self: "BaseBuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the __flags__ attribute. Returns a tracker for the
        # wrapped class's type flags, or None if the wrapped callable is not a class.
        fn = self.as_python_constant()
        if not isinstance(fn, type):
            return None
        source = self.source and AttrSource(self.source, "__flags__")
        return VariableTracker.build(tx, fn.__flags__, source)

    tp_getset = {"__bases__": GetSet(_type_get_bases, None)}
    tp_members = {
        "__base__": Member(_type_get_base, None),
        "__flags__": Member(_type_get_flags, None),
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BaseBuiltinVariable":
        # [FUNCTION DOC] Build an instance of the calling class for a builtin found at the
        # given source. Installs a builtin-match guard on that source (side effect on the
        # guard set) and returns a new tracker carrying the source; value is ignored.
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(source=source)

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Return the wrapped builtin callable stored in the class attribute.
        return self._fn

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emit bytecode that loads the builtin by its name as a global.
        # Raises an assertion error if the frame's globals already define that name
        # (shadowing). Side effect: appends instructions to the codegen output.
        name = self.as_python_constant().__name__
        if name in codegen.tx.f_globals:
            raise AssertionError("shadowed global")
        codegen.append_output(codegen.create_load_global(name, add=True))

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute lookup on the wrapped builtin. Returns a generic
        # attribute-access tracker for the named attribute (with its Python type when the
        # real attribute exists on the wrapped callable and a source-derived source), never
        # evaluating the attribute itself.
        # Declarative type-attribute dispatch, mirroring the consultation at the
        # top of VariableTracker.getattro_impl. Inlined here because this
        # override keeps its own GetAttrVariable fallback instead of delegating
        # to super().
        fn = self.as_python_constant()
        source = self.source and AttrSource(self.source, name)
        attr = getattr(fn, name, None)
        return variables.GetAttrVariable(
            self, name, py_type=type(attr) if attr is not None else None, source=source
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        # [FUNCTION DOC] Answer hasattr on the wrapped builtin by checking the real callable
        # directly; returns a constant boolean tracker.
        return VariableTracker.build(tx, hasattr(self.as_python_constant(), name))  # type: ignore[return-value]

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Hash of the tracker: the hash of the wrapped callable, paired with
        # a false flag (meaning no user code was run to compute it).
        # CPython meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        return hash(self.as_python_constant()), False

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich comparison of the wrapped callable with another tracker,
        # delegating to the shared constant-comparison helper for the given operator name.
        return python_constant_richcompare_impl(self, tx, other, op)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Method call on the wrapped builtin. Special-cases the unbound
        # string and repr methods with one argument and no keywords when the wrapped
        # callable is object or type (using the real type-level formatting for user-defined
        # classes and constant types), and otherwise uses the generic string/repr helpers.
        # All other methods defer to the parent implementation.
        if name == "__str__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if self.as_python_constant() is object:
                return generic_repr(tx, arg)
            if self.as_python_constant() is type:
                if isinstance(arg, variables.UserDefinedClassVariable):
                    return VariableTracker.build(tx, type.__str__(arg.value))
                if arg.is_python_constant() and isinstance(
                    arg.as_python_constant(), type
                ):
                    return VariableTracker.build(
                        tx, type.__str__(arg.as_python_constant())
                    )
            return generic_str(tx, arg)
        if name == "__repr__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if self.as_python_constant() is object and isinstance(
                arg, variables.UserDefinedObjectVariable
            ):
                return VariableTracker.build(tx, object.__repr__(arg.value))
            if self.as_python_constant() is type:
                if isinstance(arg, variables.UserDefinedClassVariable):
                    return VariableTracker.build(tx, type.__repr__(arg.value))
                if arg.is_python_constant() and isinstance(
                    arg.as_python_constant(), type
                ):
                    return VariableTracker.build(
                        tx, type.__repr__(arg.as_python_constant())
                    )
            return generic_repr(tx, arg)
        return super().call_method(tx, name, args, kwargs)


def _uses_custom_classinfo_check(
    type_info: Any,
    *,
    use_instancecheck: bool = True,
    use_subclasscheck: bool = True,
) -> bool:
    # [FUNCTION DOC] HELPER: Decide whether an isinstance/issubclass class-info argument
    # relies on a user-customized instance or subclass check hook that Dynamo cannot trace
    # soundly. Recurses through tuples and unions (for typing unions, instance checks only
    # count on Python 3.12+). Plain classes, tensor dtype pseudo-types, and known parameter
    # metaclasses return false; classes whose metaclass overrides the hooks (other than
    # the default type/ABC versions) return true. Non-class inputs return false.
    if isinstance(type_info, tuple):
        return any(_uses_custom_classinfo_check(item) for item in type_info)
    if isinstance(type_info, types.UnionType):
        return any(
            _uses_custom_classinfo_check(item) for item in typing.get_args(type_info)
        )
    if typing.get_origin(type_info) is typing.Union:
        typing_union_uses_instancecheck = sys.version_info >= (3, 12)
        return any(
            _uses_custom_classinfo_check(
                item,
                use_instancecheck=typing_union_uses_instancecheck,
                use_subclasscheck=True,
            )
            for item in typing.get_args(type_info)
        )
    if isinstance(type_info, type):
        if type_info in tensortype_to_dtype:
            return False

        type_info_meta = type(type_info)
        instancecheck = getattr(type_info_meta, "__instancecheck__", None)
        subclasscheck = getattr(type_info_meta, "__subclasscheck__", None)
        if (
            instancecheck is type.__instancecheck__
            and subclasscheck is type.__subclasscheck__
        ):
            return False

        if issubclass(type_info, torch.Tensor) and (
            type_info_meta.__module__,
            type_info_meta.__qualname__,
        ) in {
            ("torch.nn.parameter", "_ParameterMeta"),
            ("torch.nn.parameter", "_BufferMeta"),
            ("torch.distributed.fsdp._flat_param", "_FlatParameterMeta"),
        }:
            return False

        return (
            use_instancecheck
            and instancecheck is not type.__instancecheck__
            and instancecheck is not abc.ABCMeta.__instancecheck__
        ) or (
            use_subclasscheck
            and subclasscheck is not type.__subclasscheck__
            and subclasscheck is not abc.ABCMeta.__subclasscheck__
        )
    return False


class BuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for a builtin function, type or operator function (for example
    # len, int, or the arithmetic and comparison functions from the operator module). The
    # wrapped callable lives in the instance attribute fn (not a tracked variable field).
    # call_function builds, caches by (callable, argument tracker types, kwargs flag), and
    # runs a handler chain: insert a graph op for tensor args, binary-op handlers, a
    # dedicated call_<name> method, constant folding, and finally a graph break. Also
    # implements many call_<name> handlers following CPython semantics, comparison helpers
    # for tensors and symbolic nodes, and special unbound-method handling in call_method.
    # Must not wrap builtins listed in MUST_USE_SPECIALIZED (dict, getattr, hasattr, iter,
    # list, setattr), which have their own tracker classes.
    """
    A VariableTracker that represents a built-in value (functions and operators).
    A lot of the code here assumes it will be a function object.

    The BuiltinVariable class wraps Python built-in functions (like len, isinstance, etc.)
    and operators (like +, -, *, etc.) to enable symbolic execution during tracing. This allows
    Dynamo to properly handle these operations when converting Python code to FX graphs while
    maintaining correct semantics and enabling optimizations.
    """

    _SENTINEL = object()
    _nonvar_fields = {
        "fn",
        *VariableTracker._nonvar_fields,
    }

    # __name__ -> type_getsets[] type_get_name (getset). Unlike the type-attribute
    # readers on BaseBuiltinVariable, BuiltinVariable exposes __name__ for any
    # wrapped callable (not just types), so it never declines.
    def _builtin_type_get_name(
        self: "BuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the __name__ attribute. Always returns a string
        # constant tracker holding the wrapped callable's name (never declines).
        source = self.source and AttrSource(self.source, "__name__")
        return VariableTracker.build(tx, self.fn.__name__, source)

    tp_getset = {
        "__name__": GetSet(_builtin_type_get_name, None),
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BuiltinVariable":
        # [FUNCTION DOC] Build a tracker wrapping the given builtin value found at the given
        # source. Installs a builtin-match guard on the source (side effect on the guard
        # set) and returns the new tracker carrying that source.
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.cache
    def _constant_fold_functions() -> set[Callable[..., Any]]:
        # [FUNCTION DOC] HELPER: Cached set of callables that may be evaluated at compile
        # time when all arguments are constants: pure builtins (abs, len, min, max, str,
        # round, ...), most operator-module functions, the supported comparison operators,
        # and all plain functions in the math module. Takes no arguments; the result is
        # computed once and shared.
        fns: set[Callable[..., Any]] = {
            abs,
            all,
            any,
            ascii,
            bin,
            bool,
            callable,
            chr,
            complex,
            divmod,
            float,
            format,
            getattr,
            hex,
            int,
            len,
            max,
            min,
            oct,
            ord,
            pow,
            repr,
            round,
            str,
            str.format,
            sum,
            type,
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.truth,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.sub,
            operator.getitem,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
            operator.index,
        }
        from .tensor import supported_comparison_ops

        fns.update(supported_comparison_ops.values())
        fns.update(x for x in math.__dict__.values() if isinstance(x, type(math.sqrt)))
        return fns

    def can_constant_fold_through(self) -> bool:
        # [FUNCTION DOC] Return whether the wrapped callable is in the constant-foldable set.
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.cache
    def _fx_graph_functions() -> set[Callable[..., Any]]:
        # [FUNCTION DOC] HELPER: Cached set of operator-module callables (arithmetic,
        # bitwise, comparison, in-place and getitem forms) that can be recorded directly as
        # call nodes in the FX graph when tensor arguments are involved.
        fns = {
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.invert,
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
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.getitem,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
        }
        return fns  # type: ignore[return-value]

    @staticmethod
    @functools.cache
    def _binops() -> dict[
        Callable[..., object], tuple[list[str], Callable[..., object]]
    ]:
        # [FUNCTION DOC] HELPER: Cached table meant to map a binary operator function to its
        # forward, reverse and in-place magic method names plus its in-place operator.
        # KNOWN ISSUE: the table is always empty (all entries are commented out), so the
        # user-defined and symbolic-node handlers built from it in _binop_handlers are dead.
        # function -> ([forward name, reverse name, in-place name], in-place op)
        fns: dict[Callable[..., object], tuple[list[str], Callable[..., object]]] = {
            # NB: The follow binary operators are not supported for now, since the
            # corresponding magic methods aren't defined on SymInt / SymFloat:
            # operator.matmul
            # divmod
            # operator.and_
            # operator.or_
        }
        return fns

    @staticmethod
    @functools.cache
    def _binop_handlers() -> dict[
        Callable[..., object],
        list[
            tuple[
                tuple[
                    type[VariableTracker],
                    _TrackersType,
                ],
                _HandlerCallback,
            ]
        ],
    ]:
        # [FUNCTION DOC] HELPER: Cached builder of the multiple-dispatch table mapping an
        # operator function to an ordered list of (pair of tracker type constraints,
        # handler) entries. Handlers take the translator and two trackers. Contents: for
        # each entry of the binops table, user-defined-object handlers (call forward or
        # reflected magic methods), and symbolic-node handlers that record a graph call
        # (in-place ops are desugared to their out-of-place form); for every supported
        # comparison operator, constant-by-value comparison, symbolic-node and tensor
        # comparison handlers, generic rich comparison, never-None shortcuts and identity
        # comparison for the is/is-not operators. Raises an error if a comparison operator
        # is not callable or already has handlers.
        # KNOWN ISSUE: the binops table is empty, so the first group of handlers is never created.
        # Multiple dispatch mechanism defining custom binop behavior for certain type
        # combinations. Handlers are attempted in order, and will be used if the type checks
        # match. They are expected to have the signature:
        # fn(tx, arg0: VariableTracker, arg1: VariableTracker) -> VariableTracker
        from .functions import BaseUserFunctionVariable
        from .nn_module import NNModuleVariable
        from .tensor import supported_const_comparison_ops
        from .torch import BaseTorchVariable

        # Override table contains: op_fn -> [list of handlers]
        op_handlers: dict[Any, list[Any]] = {}
        for (
            op,
            (magic_method_names, in_place_op),
        ) in BuiltinVariable._binops().items():
            op_handlers[op] = []
            op_handlers[in_place_op] = []

            forward_name, reverse_name, inplace_name = magic_method_names

            # User-defined args (highest precedence)
            def user_defined_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = forward_name,
                reverse_name: str = reverse_name,
            ) -> VariableTracker:
                # [FUNCTION DOC] HELPER: Dispatch a binary operation involving a
                # user-defined object: if the left operand is user-defined, call its forward
                # magic method with the right operand; otherwise call the right operand's
                # reflected method with the left. Returns the resulting tracker.
                # Manually handle reversing logic if needed (e.g. call __radd__)

                # TODO: If we expand this to handle tensor args, we need to manually
                # handle cases like this:
                #
                # class A(int):
                #     def __radd__(self, other):
                #         print("woof")
                # torch.randn(3) + A(3)
                #
                # In this example, A.__radd__() is not called -> nothing is printed, because
                # Tensor.__add__ only does a subtype test against int, ignoring the subclass.
                # To be fully correct, we should not call A.__radd__() here, and there may be
                # other cases to reason about and add exceptions for.
                if isinstance(a, UserDefinedVariable):
                    return a.call_method(tx, forward_name, [b], {})
                else:
                    return b.call_method(tx, reverse_name, [a], {})

            op_handlers[op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_handler)
            )

            def user_defined_inplace_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = inplace_name,
            ) -> VariableTracker:
                # [FUNCTION DOC] HELPER: In-place binary operation on a user-defined
                # operand: call the left operand's in-place magic method with the right one.
                return a.call_method(tx, forward_name, [b], {})

            op_handlers[in_place_op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_inplace_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_inplace_handler)
            )

            # Dynamic shape args
            def dynamic_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                fn: Callable[..., Any] = op,
            ) -> VariableTracker:
                # [FUNCTION DOC] HELPER: Record the operator as a graph call on the two
                # operands (at least one symbolic) and wrap the resulting proxy in a tracker.
                # Side effect: adds a node to the output graph.
                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function", fn, *proxy_args_kwargs([a, b], {})
                    ),
                )

            op_handlers[op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

            # NB: Prefer out-of-place op when calling in-place op to generate valid graph
            op_handlers[in_place_op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

        # Special cases - lower precedence but still prefer these over constant folding

        def create_cmp_op_handlers(
            op: Callable[..., Any],
        ) -> list[tuple[tuple[_TrackersType, _TrackersType], _HandlerCallback]]:
            # [FUNCTION DOC] HELPER: Build the ordered handler list for one comparison
            # operator. Always starts with a constant-vs-constant handler. For the six rich
            # comparisons it adds symbolic-node handlers and a generic rich-comparison
            # fallback and returns. For other operators (identity tests) it adds shortcuts
            # for types never equal to None, tensor and symbolic-node handlers, and, for
            # is/is-not, an identity-comparison handler.
            def compare_by_value(
                tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
            ) -> VariableTracker:
                # [FUNCTION DOC] HELPER: Apply the comparison operator to the two constants'
                # underlying values and return a constant tracker. A TypeError from the
                # comparison is re-raised as an observed exception inside the traced program.
                try:
                    return VariableTracker.build(tx, op(a.value, b.value))  # type: ignore[attr-defined]
                except TypeError as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(exc.args),
                    )

            result: list[
                tuple[
                    tuple[
                        _TrackersType,
                        _TrackersType,
                    ],
                    _HandlerCallback,
                ]
            ] = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in _OPERATOR_TO_DUNDER:
                # For constants, speedup the comparison instead of going
                # through generic_richcompare. Removing this line causes
                # major regression for pr time benchmark - add_loop_eager.
                result = [
                    ((ConstantVariable, ConstantVariable), compare_by_value),
                ]

                op_var = BuiltinVariable(op)
                # Special handling of SymNode variable
                result.extend(
                    [
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                # COMPARE_OP (a == b) dispatches through generic_richcompare,
                # which implements do_richcompare via tp_richcompare_impl slots.
                # See object_protocol.py for details.
                dunder = _OPERATOR_TO_DUNDER[op]

                def handler(
                    tx: "InstructionTranslatorBase",
                    a: VariableTracker,
                    b: VariableTracker,
                ) -> VariableTracker:
                    # [FUNCTION DOC] HELPER: Compare two trackers through the generic rich
                    # comparison protocol using the captured magic method name.
                    return generic_richcompare(tx, a, b, dunder)

                result.append(((VariableTracker, VariableTracker), handler))
                return result

            result = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in supported_const_comparison_ops.values() and op.__name__.startswith(
                "is_"
            ):
                # Tensor is None, List is not None, etc
                none_result = op(object(), None)

                def never(
                    tx: "InstructionTranslatorBase",
                    a: VariableTracker,
                    b: VariableTracker,
                ) -> VariableTracker:
                    # [FUNCTION DOC] HELPER: Return the precomputed constant result of
                    # comparing an arbitrary object with None for this identity operator.
                    return VariableTracker.build(tx, none_result)

                obj_op_none = never
                none_op_obj = never

                types_that_are_never_none = (
                    TensorVariable,
                    SymNodeVariable,
                    NNModuleVariable,
                    BaseListVariable,
                    UserDefinedVariable,
                    BaseUserFunctionVariable,
                    ConstDictVariable,
                    BaseTorchVariable,
                )
                result.extend(
                    [
                        (
                            (types_that_are_never_none, ConstantVariable),
                            obj_op_none,
                        ),
                        (
                            (ConstantVariable, types_that_are_never_none),
                            none_op_obj,
                        ),
                    ]
                )

                op_var = BuiltinVariable(op)
                result.extend(
                    [
                        (
                            (TensorVariable, VariableTracker),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (VariableTracker, TensorVariable),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                def handle_is(
                    tx: "InstructionTranslatorBase",
                    left: VariableTracker,
                    right: VariableTracker,
                ) -> VariableTracker | None:
                    # [FUNCTION DOC] HELPER: Handle is / is-not. Asks the identity-compare
                    # helper whether the trackers are the same object; returns None when it
                    # cannot decide, else a constant boolean (negated for is-not).
                    result = vt_identity_compare(left, right)
                    if result is None:
                        return None
                    is_same = result.as_python_constant()
                    return VariableTracker.build(
                        tx, is_same if op.__name__ == "is_" else not is_same
                    )

                result.append(((VariableTracker, VariableTracker), handle_is))  # type: ignore[arg-type]

            return result

        for op in supported_comparison_ops.values():
            if not callable(op):
                raise AssertionError(f"comparison op {op} is not callable")
            if op in op_handlers:
                raise AssertionError(f"duplicate handler for op {op}")
            op_handlers[op] = create_cmp_op_handlers(op)

        return op_handlers

    @staticmethod
    def _find_binop_handler(
        op: Callable[..., Any], a_type: type[VariableTracker], b_type: type
    ) -> list[_HandlerCallback] | None:
        # [FUNCTION DOC] HELPER: Look up handlers for an operator and a pair of tracker
        # classes. Returns None if the operator has no table entry, otherwise the list of
        # handlers (possibly empty) whose type constraints both operand classes subclass,
        # in registration order.
        handlers = BuiltinVariable._binop_handlers().get(op)
        if handlers is None:
            return None

        matches = []
        for (type1, type2), handler in handlers:
            if issubclass(a_type, type1) and issubclass(b_type, type2):
                matches.append(handler)
        return matches

    def can_insert_in_graph(self) -> bool:
        # [FUNCTION DOC] Return whether the wrapped callable can be recorded directly as an
        # FX graph call (is in the graph-capable operator set).
        return self.fn in self._fx_graph_functions()

    # Builtins that have been promoted to their own VT classes. Creating a
    # BuiltinVariable for these is a bug; use the specialized class instead.
    MUST_USE_SPECIALIZED: frozenset[Any] = frozenset(
        {dict, getattr, hasattr, iter, list, setattr}
    )

    def __init__(self, fn: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Store the wrapped callable fn; other keyword arguments (source,
        # mutation type, etc.) go to the base tracker. Raises an assertion error if fn is
        # one of the builtins that must use a specialized tracker class.
        if fn in self.MUST_USE_SPECIALIZED:
            raise AssertionError(
                f"Use the specialized VT class for {fn!r}, not BuiltinVariable. "
                f"E.g. DictBuiltinVariable for dict."
            )
        super().__init__(**kwargs)
        self.fn = fn

    def __repr__(self) -> str:
        # [FUNCTION DOC] Debug string showing the class name and the wrapped callable's name.
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__

        return f"{self.__class__.__name__}({name})"

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Return the wrapped callable stored on the instance.
        return self.fn

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Return the real Python object backing this tracker: the wrapped callable.
        return self.fn

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implement the bitwise-or slot for the wrapped callable, used for
        # type unions such as one class or another. If the wrapped callable is a class and
        # the other operand is a compile-time constant, performs the real type union and
        # returns it as a constant tracker; otherwise (or if the real operation reports
        # not-implemented) returns a tracker for the not-implemented singleton. The reverse
        # flag is accepted but unused.
        # BuiltinVariable wraps built-in types like list, tuple, dict.
        # type(self.fn).__or__(self.fn, other_val) delegates to CPython's
        # _Py_union_type_or for type unions (e.g. list | tuple).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L6028-L6030 (type_as_number.nb_or = _Py_union_type_or)
        # https://github.com/python/cpython/blob/3.13/Objects/unionobject.c#L162 (_Py_union_type_or)
        if not isinstance(self.fn, type):
            return VariableTracker.build(tx, NotImplemented)
        try:
            other_val = other.as_python_constant()
        except NotImplementedError:
            return VariableTracker.build(tx, NotImplemented)
        # pyrefly: ignore[bad-argument-count]
        result = type(self.fn).__or__(self.fn, other_val)
        if result is NotImplemented:
            return VariableTracker.build(tx, NotImplemented)
        return VariableTracker.build(tx, result)

    def as_proxy(self) -> Any:
        # [FUNCTION DOC] For the wrapped builtin types bool, int and float, return the
        # matching torch dtype (bool, int64, float64) so they can be used as dtype arguments
        # in graph calls; any other callable defers to the parent implementation.
        DTYPE = {
            bool: torch.bool,
            int: torch.int64,
            float: torch.float64,
        }
        if self.fn in DTYPE:
            return DTYPE[self.fn]
        return super().as_proxy()

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emit bytecode loading the builtin by name as a global. Raises an
        # assertion error if the callable is not from the builtins module or if the frame's
        # globals shadow its name. Side effect: appends instructions to the codegen output.
        name = self.fn.__name__
        if self.fn.__module__ != "builtins":
            raise AssertionError(f"Expected builtins module, got {self.fn.__module__}")
        if name in codegen.tx.f_globals:
            raise AssertionError("shadowed global")
        codegen.append_output(codegen.create_load_global(name, add=True))

    def constant_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        # [FUNCTION DOC] Return whether every positional and keyword argument is a constant.
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args: VariableTracker) -> bool:
        # [FUNCTION DOC] Return whether at least one argument is a tensor tracker. Returns
        # false immediately if any argument is a generic attribute-access tracker.
        any_tensor = False
        for arg in args:
            if isinstance(arg, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or arg.is_tensor()
        return any_tensor

    def tensor_args_type(self, arg_types: list[type]) -> bool:
        # [FUNCTION DOC] Type-level version of the tensor-argument check: true if any
        # tracker class is a tensor tracker class; false immediately if any is a generic
        # attribute-access tracker class.
        any_tensor = False
        for arg_type in arg_types:
            if issubclass(arg_type, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or issubclass(arg_type, variables.TensorVariable)
        return any_tensor

    def python_and_tensor_constant_only(
        self, *args: VariableTracker, **kwargs: VariableTracker
    ) -> bool:
        # [FUNCTION DOC] Return true when every tensor argument comes from a constant source
        # (a tensor that is effectively a compile-time constant; a tensor without a source
        # counts as non-constant) and all non-tensor arguments are constants.
        tensor_args = []
        non_tensor_args = []
        for i in itertools.chain(args, kwargs.values()):
            if i.is_tensor():
                tensor_args.append(i)
            else:
                non_tensor_args.append(i)
        return all(
            is_constant_source(t.source) if t.source is not None else False
            for t in tensor_args
        ) and self.constant_args(*non_tensor_args)

    @staticmethod
    def unwrap_unspec_args_kwargs(
        args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> tuple[list[Any], dict[str, Any]]:
        # [FUNCTION DOC] Convert positional and keyword argument trackers to their raw Python
        # values; returns a list and a mapping. May raise if a tracker has no constant value.
        return [x.as_python_constant() for x in args], {
            k: v.as_python_constant() for k, v in kwargs.items()
        }

    def has_constant_handler(
        self, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> bool:
        # [FUNCTION DOC] Return whether the wrapped callable is foldable and all arguments
        # are constants or unspecialized Python scalars, i.e. constant folding would work.
        return self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        )

    @staticmethod
    def _make_handler(
        fn: Callable[..., Any], arg_types: list[type], has_kwargs: bool
    ) -> Callable[
        [
            "InstructionTranslatorBase",
            list[VariableTracker],
            dict[str, VariableTracker],
        ],
        VariableTracker | None,
    ]:
        # [FUNCTION DOC] HELPER: Build the handler used by call_function for a given builtin
        # callable, a list of argument tracker classes and a flag saying whether keyword
        # arguments were passed. The returned handler takes (translator, args, kwargs) and
        # returns a tracker or None. Lazy non-constant args are realized and re-dispatched;
        # lazy constants are resolved by delegating to a handler built for constants. Exception
        # classes build exception trackers. Otherwise an ordered chain is assembled: graph
        # insertion for tensor args, binary-op handlers, the matching call_<name> method,
        # and constant folding; the first handler returning a truthy result wins, and if none
        # does a graph break "Failed to trace builtin operator" is raised.
        from .lazy import (
            ComputedLazyConstantVariable,
            LazyConstantVariable,
            LazyVariableTracker,
        )

        obj = BuiltinVariable(fn)
        handlers: list[_HandlerCallback] = []

        lazy_constant_types = (LazyConstantVariable, ComputedLazyConstantVariable)
        lazy_types = [t for t in arg_types if issubclass(t, LazyVariableTracker)]
        if lazy_types:
            if not all(issubclass(t, lazy_constant_types) for t in lazy_types):
                # Realize non-constant lazy args and re-dispatch.  Any
                # lazy constant args are kept and handled on the
                # second dispatch through the branch below.
                return lambda tx, args, kwargs: obj.call_function(
                    tx,
                    [
                        a.realize()
                        if isinstance(a, LazyVariableTracker)
                        and not isinstance(a, lazy_constant_types)
                        else a
                        for a in args
                    ],
                    kwargs,
                )

            # Only lazy constant types.  Install type guards
            # and resolve the dispatch type.  If the resolved type is
            # ConstantVariable (the common case), delegate to a handler
            # built for ConstantVariable.  Otherwise (e.g. specialize_int=
            # False turned the int into a SymNodeVariable), realize and
            # re-dispatch so the correct handler is used.
            inner_handler = BuiltinVariable._make_handler(
                fn,
                [
                    ConstantVariable if issubclass(t, lazy_constant_types) else t
                    for t in arg_types
                ],
                has_kwargs,
            )

            def lazy_constant_handler(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                # [FUNCTION DOC] HELPER: Handle calls whose args include lazy constants.
                # Without kwargs, first try to build a computed lazy constant. If any lazy
                # constant resolves to a non-constant tracker type (for example a symbolic
                # int), realize all lazy constants and redo the call; otherwise run the handler
                # built for plain constants.
                if not kwargs:
                    result = _try_computed_lazy_constant(fn, args)
                    if result is not None:
                        return result
                for a in args:
                    if isinstance(a, lazy_constant_types):
                        if a.get_handler_type_for_dispatch() is not ConstantVariable:
                            return obj.call_function(
                                tx,
                                [
                                    v.realize()
                                    if isinstance(v, lazy_constant_types)
                                    else v
                                    for v in args
                                ],
                                kwargs,
                            )
                return inner_handler(tx, args, kwargs)

            return lazy_constant_handler

        if inspect.isclass(fn) and (
            issubclass(fn, BaseException)
            # GeneratorExit doesn't inherit from Exception
            # >>> issubclass(GeneratorExit, Exception)
            # False
            or fn is GeneratorExit
        ):

            def create_exception_class_object(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker:
                # [FUNCTION DOC] HELPER: Instantiate an exception class in the traced
                # program. Graph-breaks if constructing AssertionError with any non-string
                # argument. Returns the matching specialized exception tracker for
                # StopIteration, AttributeError and NameError, or a generic exception tracker.
                if fn is AssertionError and not all(
                    x.is_python_constant() and isinstance(x.as_python_constant(), str)
                    for x in args
                ):
                    unimplemented(
                        gb_type="assert with non-string message",
                        context=str(args),
                        explanation="Dynamo only supports asserts with string messages",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

                if fn is StopIteration:
                    return variables.StopIterationVariable(fn, args, kwargs)
                elif fn is AttributeError:
                    return variables.AttributeErrorVariable(fn, args, kwargs)
                elif fn is NameError:
                    return variables.NameErrorVariable(fn, args, kwargs)
                return variables.ExceptionVariable(fn, args, kwargs)

            return create_exception_class_object

        if obj.can_insert_in_graph() and not (
            fn is operator.getitem
            and (
                len(arg_types) != 2
                or not issubclass(arg_types[0], variables.TensorVariable)
            )
        ):
            if obj.tensor_args_type(arg_types):
                return obj._handle_insert_op_in_graph
            elif has_kwargs:
                # need runtime check for kwargs
                handlers.append(obj._handle_insert_op_in_graph)

        # Handle binary ops (e.g. __add__ / __radd__, __iadd__, etc.)
        # NB: Tensor args are handled above and not here
        if len(arg_types) == 2 and not has_kwargs:
            # Try to find a handler for the arg types; otherwise, fall through to constant handler
            binop_handlers = BuiltinVariable._find_binop_handler(fn, *arg_types)
            if not binop_handlers:
                pass
            elif len(binop_handlers) == 1:
                (binop_handler,) = binop_handlers
                handlers.append(lambda tx, args, _: binop_handler(tx, *args))
            else:

                def call_binop_handlers(
                    tx: "InstructionTranslatorBase", args: Any, _: Any
                ) -> Any:
                    # [FUNCTION DOC] HELPER: Try each matching binary-op handler in order and
                    # return the first truthy result, or None if none produced one.
                    # pyrefly: ignore [not-iterable]
                    for fn in binop_handlers:
                        rv = fn(tx, *args)
                        if rv:
                            return rv
                    return None

                handlers.append(call_binop_handlers)

        self_handler = getattr(obj, f"call_{fn.__name__}", None)
        if self_handler:

            def call_self_handler(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                # [FUNCTION DOC] HELPER: Run the dedicated call_<name> method with the args.
                # On a TypeError, check whether the arguments fail to bind to the method; if so
                # and no constant-fold alternative exists, log a warning and graph-break with
                # "invalid call to builtin op handler"; if binding was fine, re-raise. If the
                # method itself graph-breaks but constant folding is possible, discard that
                # break (removing it from stats). Falls through returning None so later
                # handlers can try.
                try:
                    # pyrefly: ignore [not-callable]
                    return self_handler(tx, *args, **kwargs)
                except TypeError:
                    # Check if binding is bad. inspect signature bind is expensive.
                    # So check only when handler call fails.
                    try:
                        # pyrefly: ignore [bad-argument-type]
                        inspect.signature(self_handler).bind(tx, *args, **kwargs)
                    except TypeError as e:
                        has_constant_handler = obj.has_constant_handler(args, kwargs)
                        if not has_constant_handler:
                            log.warning(
                                "incorrect arg count %s %s and no constant handler",
                                self_handler,
                                e,
                            )
                            unimplemented(
                                gb_type="invalid call to builtin op handler",
                                context=f"invalid args to {self_handler}: {args} {kwargs}",
                                explanation=f"Encountered TypeError when trying to handle op {fn.__name__}",
                                hints=[*graph_break_hints.DIFFICULT],
                            )
                    else:
                        raise
                except Unsupported as exc:
                    has_constant_handler = obj.has_constant_handler(args, kwargs)
                    if not has_constant_handler:
                        raise
                    # Actually, we will handle this just fine
                    exc.remove_from_stats()
                return None

            handlers.append(call_self_handler)

        if obj.can_constant_fold_through():
            if (
                all(issubclass(x, ConstantVariable) for x in arg_types)
                and not has_kwargs
            ):

                def constant_fold_handler(
                    tx: "InstructionTranslatorBase",
                    args: list[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # [FUNCTION DOC] HELPER: Constant folding when all args are constant
                    # trackers and there are no kwargs. Calls the real function on the raw
                    # values and wraps the result. Exceptions raised by the function are
                    # re-raised as observed exceptions in the traced program; failure to get a
                    # raw value causes a "constant fold exception" graph break.
                    # KNOWN ISSUE: the clause for the constant-conversion error comes after the
                    # generic exception clause; if that error is an Exception subclass it is
                    # converted to an observed exception rather than a graph break.
                    # fast path
                    try:
                        res = fn(
                            *[x.as_python_constant() for x in args],
                        )
                    except Exception as exc:
                        raise_observed_exception(
                            type(exc),
                            tx,
                            args=list(exc.args),
                        )
                    except AsPythonConstantNotImplementedError as exc:
                        unimplemented(
                            gb_type="constant fold exception",
                            context=f"attempted to run function {fn} with arguments {args}",
                            explanation="Encountered exception when attempting to constant fold.",
                            hints=[*graph_break_hints.DYNAMO_BUG],
                            from_exc=exc,
                        )
                    return VariableTracker.build(tx, res)

            else:

                def constant_fold_handler(
                    tx: "InstructionTranslatorBase",
                    args: list[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # [FUNCTION DOC] HELPER: Constant folding with a runtime check: if every
                    # arg and kwarg is a constant or unspecialized scalar, call the real function
                    # on raw values and wrap the result (errors become observed exceptions, or a
                    # graph break if no raw value is available). Otherwise returns None.
                    # path with a runtime check
                    if check_unspec_or_constant_args(args, kwargs):
                        try:
                            res = fn(
                                *[x.as_python_constant() for x in args],
                                **{
                                    k: v.as_python_constant() for k, v in kwargs.items()
                                },
                            )
                        except AsPythonConstantNotImplementedError as exc:
                            unimplemented(
                                gb_type="constant fold exception",
                                context=f"attempted to run function {fn} with arguments {args}",
                                explanation="Encountered exception when attempting to constant fold.",
                                hints=[*graph_break_hints.DYNAMO_BUG],
                                from_exc=exc,
                            )
                        except Exception as exc:
                            raise_observed_exception(
                                type(exc),
                                tx,
                                args=list(exc.args),
                            )
                        return VariableTracker.build(tx, res)
                    return None

            handlers.append(constant_fold_handler)

        def call_unimplemented(args: list[VariableTracker]) -> None:
            # [FUNCTION DOC] HELPER: Always graph-breaks with "Failed to trace builtin
            # operator", reporting the builtin name, argument tracker types and hints.
            real_arg_types = [arg.python_type_name() for arg in args]
            unimplemented(
                gb_type="Failed to trace builtin operator",
                context=f"builtin {fn.__name__} {arg_types} {has_kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator `{fn.__name__}` "
                f"with argument types {real_arg_types} (has_kwargs {has_kwargs})",
                hints=[
                    f"Avoid calling builtin `{fn.__name__}` with argument types {real_arg_types}. "
                    f"Consider using an equivalent alternative function/method to `{fn.__name__}`.",
                    "If you are attempting to call a logging function (e.g. `print`), "
                    "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
                    "Please report an issue to PyTorch.",
                ],
            )

        if len(handlers) == 0:
            return lambda tx, args, kwargs: call_unimplemented(args)
        elif len(handlers) == 1:
            (handler,) = handlers

            def builtin_dispatch(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                # [FUNCTION DOC] HELPER: Dispatch with a single handler; return its result if
                # truthy, otherwise graph-break as an unsupported builtin call.
                rv = handler(tx, args, kwargs)
                if rv:
                    return rv
                call_unimplemented(args)
                return rv

        else:

            def builtin_dispatch(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                # [FUNCTION DOC] HELPER: Dispatch with several handlers; return the first truthy
                # result, otherwise graph-break as an unsupported builtin call.
                rv = None
                for fn in handlers:
                    rv = fn(tx, args, kwargs)
                    if rv:
                        return rv
                call_unimplemented(args)
                return rv

        return builtin_dispatch

    @staticmethod
    def _constant_eval_numeric_expr(node: ast.AST) -> bool:
        # [FUNCTION DOC] HELPER: Return whether a parsed expression tree consists only of
        # numeric constants (bool, int, float, complex) combined with unary plus/minus/invert
        # and the arithmetic and bitwise binary operators, so it is safe to evaluate.
        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.UnaryOp,
            ast.BinOp,
            ast.UAdd,
            ast.USub,
            ast.Invert,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
        )
        return all(
            isinstance(child, allowed_nodes)
            and (
                not isinstance(child, ast.Constant)
                or isinstance(child.value, (bool, int, float, complex))
            )
            for child in ast.walk(node)
        )

    @staticmethod
    def _constant_eval_result(
        tx: "InstructionTranslatorBase", tree: ast.Expression, filename: str
    ) -> VariableTracker | None:
        # [FUNCTION DOC] HELPER: Evaluate a parsed expression at compile time. Returns None
        # if it contains any call. Tries literal evaluation first; if that fails, evaluates
        # purely numeric expressions with empty builtins (errors become observed exceptions),
        # else returns None. On success returns a constant tracker.
        if any(isinstance(child, ast.Call) for child in ast.walk(tree)):
            return None
        ast.fix_missing_locations(tree)
        try:
            result = ast.literal_eval(tree)
        except ValueError:
            if not BuiltinVariable._constant_eval_numeric_expr(tree):
                return None
            try:
                result = eval(
                    compile(tree, filename, "eval"),
                    {"__builtins__": {}},
                    {},
                )
            except Exception as exc:
                raise_observed_exception(type(exc), tx, args=list(exc.args))
        return VariableTracker.build(tx, result)

    def call_eval(
        self,
        tx: "InstructionTranslatorBase",
        source: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the eval builtin. Only supports a single constant string
        # argument (no extra args or kwargs, else returns None). Parses it as an expression
        # (a syntax error becomes an observed SyntaxError) and constant-evaluates it when it
        # is a literal or simple numeric expression; returns None otherwise.
        if args or kwargs:
            return None
        if not source.is_python_constant():
            return None
        source_str = source.as_python_constant()
        if not isinstance(source_str, str):
            return None

        try:
            tree = ast.parse(source_str.strip(), mode="eval")
        except SyntaxError as exc:
            raise_observed_exception(SyntaxError, tx, args=[exc.msg])

        return self._constant_eval_result(tx, tree, "<torch._dynamo.eval>")

    def call_vars(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for vars. No arguments gives a snapshot of the current
        # frame's locals; one argument gives that object's attribute dictionary (a missing
        # one becomes an observed TypeError). Keyword arguments or more than one argument
        # raise an observed TypeError.
        if kwargs:
            raise_type_error(tx, "vars() takes no keyword arguments")
        if len(args) == 0:
            return self._call_frame_locals_snapshot(tx)
        if len(args) != 1:
            raise_type_error(tx, f"vars expected at most 1 argument, got {len(args)}")
        # vars(obj) is obj.__dict__ if __dict__ is present else TypeError
        try:
            return args[0].tp_getattro_impl(tx, "__dict__")
        except ObservedAttributeError:
            raise_observed_exception(TypeError, tx)

    def call_locals(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for locals. Takes no arguments (otherwise an observed
        # TypeError) and returns a snapshot dictionary of the current frame's locals.
        if len(args) != 0:
            raise_observed_exception(TypeError, tx)
        return self._call_frame_locals_snapshot(tx)

    @staticmethod
    def _call_frame_locals_snapshot(tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Build a new dictionary tracker (fresh mutable value) mapping
        # name constants to the current frame's local, cell and free variable trackers.
        # Cell contents are loaded through side effects; empty or deleted slots are skipped;
        # ordering of cells vs fast locals depends on Python version so that fast locals
        # shadow cells (reversed on 3.12). Locals pruned from the symbolic state but present
        # at runtime are rebuilt from their local source, installing guards as a side effect.
        from .builder import VariableBuilder

        frame_local_names = set(tx.f_code.co_varnames) | set(tx.cell_and_freevars())
        cell_and_freevars = set(tx.cell_and_freevars())
        frame_locals = {}
        # symbolic_cellvars registers all of the frame's cells. Cells are listed first so that a colliding fast local of
        # the same name shadows the cell, matching CPython (except for 3.12, which is backwards): the two share a name
        # but not a localsplus slot, and the fast slot wins while it holds a value (an empty one is skipped below,
        # leaving the cell contents visible).
        if sys.version_info[:2] == (3, 12):
            its = (tx.symbolic_locals.items(), tx.symbolic_cellvars.items())
        else:
            its = (tx.symbolic_cellvars.items(), tx.symbolic_locals.items())

        for name, value in itertools.chain(*its):
            if name not in frame_local_names:
                continue
            # Match on CellVariable, not name: a colliding fast local shares a
            # cell's name but is not itself a cell.
            if type.__instancecheck__(CellVariable, value):
                value = tx.output.side_effects.load_cell(value)
            if type.__instancecheck__(NullVariable, value) or isinstance(
                value, variables.DeletedVariable
            ):
                continue
            frame_locals[ConstantVariable.create(name)] = value
        # Include locals pruned from symbolic_locals by the unused-input optimisation.
        # locals()/vars() observes the entire namespace, so build VTs (installing
        # guards) for any pruned entry that still has a runtime value.
        for name in frame_local_names:
            if name in frame_locals or name in cell_and_freevars:
                continue
            if name not in tx.f_locals:
                continue
            vt = VariableBuilder(tx, LocalSource(name))(tx.f_locals[name])
            frame_locals[ConstantVariable.create(name)] = vt
        return ConstDictVariable(
            frame_locals,
            mutation_type=ValueMutationNew(),
        )

    def _handle_insert_op_in_graph(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        # [FUNCTION DOC] HELPER: Record the wrapped operator on tensor/symbolic arguments.
        # Returns None if kwargs are given without tensor args. Tries torch-function
        # dispatch first (remapping to Tensor methods, swapping operands to use the reflected
        # method when the first arg is not a tensor, unless exporting). Otherwise: constants
        # fold into stored constants; in-place ops on constant left operands are desugared;
        # getitem with a symbolic index becomes a select call; numpy-array args use a numpy
        # wrapper; non-tensor comparisons use generic rich comparison; else a graph call is
        # created and wrapped as a fake-item, unspecialized scalar, symbolic or tensor
        # tracker. A NotImplementedError becomes a graph break. Mutates the output graph.
        # KNOWN ISSUE: the in-place desugaring map sends the in-place matmul to itself instead of
        # to the plain matmul; also the truediv special case rebinds args after the graph call.
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        if kwargs and not self.tensor_args(*args, *kwargs.values()):
            return None

        # insert handling for torch function here
        from .builder import SourcelessBuilder
        from .torch_function import can_dispatch_torch_function, dispatch_torch_function

        global BUILTIN_TO_TENSOR_RFN_MAP, BUILTIN_TO_TENSOR_FN_MAP
        if can_dispatch_torch_function(tx, args, kwargs):
            # Only remap the fn to tensor methods if we aren't exporting
            # export serde does not handle method descriptors today
            if not tx.export:
                # Ensure the builtin maps are populated before accessing them
                populate_builtin_to_tensor_fn_map()
                # Use sourceless builder, we built the map ourselves
                if not args[0].is_tensor():
                    if self.fn in BUILTIN_TO_TENSOR_RFN_MAP:
                        func = BUILTIN_TO_TENSOR_RFN_MAP[self.fn]
                    else:
                        func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]

                    tmp = args[0]
                    # swap args and call reverse version of func
                    args[0] = args[1]  # type: ignore[index]
                    args[1] = tmp  # type: ignore[index]
                else:
                    func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]
            else:
                func = self.fn

            fn_var = SourcelessBuilder.create(tx, func)

            return dispatch_torch_function(tx, fn_var, args, kwargs)

        fn = self.fn
        try:
            # Constant fold for constant tensor and python constants
            if self.python_and_tensor_constant_only(*args, **kwargs):
                from ..bytecode_transformation import unique_id
                from .functions import invoke_and_store_as_constant

                return invoke_and_store_as_constant(
                    tx, fn, unique_id(fn.__name__), args, kwargs
                )

            if fn in IN_PLACE_DESUGARING_MAP and isinstance(
                args[0], variables.ConstantVariable
            ):
                # In-place operators like += usually mustate tensor
                # values, but in the edge case of immutable values they
                # re-bind the variable.
                #
                # The easiest way to keep the graph consistent in this
                # scenario is to de-sugar eagerly.
                fn = IN_PLACE_DESUGARING_MAP[fn]
                args = [args[0], args[1]]  # type: ignore[assignment]

            if fn is operator.getitem:
                if kwargs:
                    raise_type_error(
                        tx, "_operator.getitem() takes no keyword arguments"
                    )
                if len(args) != 2:
                    raise_type_error(
                        tx, f"getitem expected 2 arguments, got {len(args)}"
                    )
                if isinstance(args[1], SymNodeVariable):
                    # Standard indexing will force specialization due to
                    # __index__.  Rewrite as a regular torch op which will
                    # trace fine
                    fn = torch.select
                    args = [
                        args[0],
                        variables.VariableTracker.build(tx, 0),
                        args[1],
                    ]  # type: ignore[assignment]

            # Interaction between ndarray and tensors:
            #   We prefer the tensor op whenever there are tensors involved
            # NB: Use exact type check here - NumpyNdarrayVariable is a TensorVariable
            # subclass but should NOT trigger the tensor path
            if check_numpy_ndarray_args(args, kwargs) and not any(
                type(arg) is TensorVariable for arg in args
            ):
                proxy = tx.output.create_proxy(
                    "call_function",
                    numpy_operator_wrapper(fn),
                    *proxy_args_kwargs(args, kwargs),
                )

                return wrap_fx_proxy_cls(variables.NumpyNdarrayVariable, tx, proxy)

            if (
                fn in _OPERATOR_TO_DUNDER
                and len(args) == 2
                and any(
                    not isinstance(a, (variables.TensorVariable, SymNodeVariable))
                    for a in args
                )
            ):
                return generic_richcompare(
                    tx, args[0], args[1], _OPERATOR_TO_DUNDER[fn]
                )

            proxy = tx.output.create_proxy(
                "call_function",
                fn,
                *proxy_args_kwargs(args, kwargs),
            )
            if any(isinstance(arg, FakeItemVariable) for arg in args):
                return wrap_fx_proxy_cls(
                    FakeItemVariable,
                    tx,
                    proxy,
                )
            elif check_unspec_python_args(args, kwargs):
                _args, _kwargs = self.unwrap_unspec_args_kwargs(args, kwargs)
                raw_value = fn(*_args, **_kwargs)

                need_unwrap = any(
                    x.need_unwrap
                    for x in itertools.chain(args, kwargs.values())
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )

                return wrap_fx_proxy_cls(
                    UnspecializedPythonVariable,
                    tx,
                    proxy,
                    raw_value=raw_value,
                    need_unwrap=need_unwrap,
                )
            elif all(isinstance(x, SymNodeVariable) for x in args):
                return SymNodeVariable.create(tx, proxy, None)
            else:
                # Work around for vision_maskrcnn due to precision difference
                # specialize the dividend when float divide by tensor
                if fn is operator.truediv and isinstance(
                    args[0], variables.UnspecializedPythonVariable
                ):
                    args = list(args)
                    args[0] = args[0].as_python_constant()
                return wrap_fx_proxy(tx, proxy)

        except NotImplementedError:
            unimplemented(
                gb_type="unimplemented builtin op on tensor arguments",
                context=f"partial tensor op: {self} {args} {kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator {self.fn} with tensor arguments",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

    call_function_handler_cache: dict[
        tuple[object, ...],
        Callable[
            [
                "InstructionTranslatorBase",
                list[VariableTracker],
                dict[str, VariableTracker],
            ],
            VariableTracker,
        ],
    ] = {}

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Call the wrapped builtin with the given argument trackers. Calling
        # object with no arguments builds a fresh opaque object tracker. Otherwise looks up
        # (or creates and caches in a class-level mapping) a handler keyed by the callable
        # and the argument tracker classes (plus a kwargs marker; kwargs are realized first)
        # and runs it. Raises an assertion error if no handler is available.
        if self.fn is object and not args and not kwargs:
            # object() -> a fresh opaque instance, wrapped as ObjectVariable to
            # match how SourcelessBuilder wraps bare `object` instances. Falling
            # through to the constant handler cannot build a VT from object().
            from .builder import SourcelessBuilder

            return SourcelessBuilder.create(tx, object())

        key: tuple[object, ...]
        if kwargs:
            kwargs = {k: v.realize() for k, v in kwargs.items()}
            key = (self.fn, *(type(x) for x in args), True)
        else:
            key = (self.fn, *(type(x) for x in args))

        handler = self.call_function_handler_cache.get(key)
        if not handler:
            self.call_function_handler_cache[key] = handler = self._make_handler(  # type: ignore[assignment]
                self.fn, [type(x) for x in args], bool(kwargs)
            )
        if handler is None:
            raise AssertionError(
                f"No handler found for {self.fn} with args {[type(x) for x in args]}"
            )
        return handler(tx, args, kwargs)  # type: ignore[return-value]

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Call an attribute of the wrapped builtin as an unbound method.
        # Handles: object.__setattr__ on a tracked user object; __new__ for object and tuple
        # (tracking new user-defined objects); constant-foldable class methods on constant
        # args; object.__init__ (no-op); methods of set/frozenset/list/tuple forwarded to the
        # first argument; unbound rich comparisons on constants of the wrapped type;
        # str/float methods on constants; and unbound length, iteration, negation, positive,
        # absolute, invert and hash dunder methods mapped to the generic protocol helpers.
        # Anything else goes to the parent implementation.
        if self.fn is object and name == "__setattr__":
            if len(args) != 3:
                raise AssertionError(
                    f"object.__setattr__ expects 3 args, got {len(args)}"
                )
            if len(kwargs) != 0:
                raise AssertionError(
                    f"object.__setattr__ expects no kwargs, got {len(kwargs)}"
                )
            obj, name_var, val = args
            obj = obj.realize()
            if (
                isinstance(obj, UserDefinedObjectVariable)
                and tx.output.side_effects.is_attribute_mutation(obj)
                and name_var.is_python_constant()
            ):
                return obj.method_setattr_standard(tx, name_var, val)

        if name == "__new__":
            # Supported __new__ methods
            if self.fn is object and len(args) == 1:
                if len(kwargs) != 0:
                    raise AssertionError(
                        f"object.__new__ expects no kwargs, got {len(kwargs)}"
                    )
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

            if self.fn is tuple and len(args) == 2 and not kwargs:
                if isinstance(args[0], BuiltinVariable) and args[0].fn is tuple:
                    init_args = unpack_iterable(tx, args[1])
                    return variables.TupleVariable(
                        init_args, mutation_type=ValueMutationNew()
                    )

                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

        if name in _BUILTIN_CONSTANT_FOLDABLE_METHODS.get(self.fn, ()):
            if all(a.is_python_constant() for a in args) and all(
                v.is_python_constant() for v in kwargs.values()
            ):
                try:
                    fn = getattr(self.fn, name)
                    res = fn(
                        *(a.as_python_constant() for a in args),
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    )
                    return VariableTracker.build(tx, res)
                except Exception as e:
                    raise_observed_exception(
                        type(e),
                        tx,
                        args=list(e.args),
                    )

        if self.fn is object and name == "__init__":
            # object.__init__ is a no-op
            return variables.ConstantVariable.create(None)

        if self.fn in (set, frozenset, list, tuple):
            if isinstance(args[0], variables.UserDefinedObjectVariable):
                if args[0]._base_vt is None:
                    raise AssertionError(
                        "UserDefinedObjectVariable._base_vt must not be None"
                    )
                return args[0]._base_vt.call_method(tx, name, args[1:], kwargs)
            else:
                return args[0].call_method(tx, name, args[1:], kwargs)

        if (
            name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")
            and len(args) == 2
            and not kwargs
            and isinstance(self.fn, type)
            and args[0].is_python_constant()
            and args[1].is_python_constant()
        ):
            # Unbound rich-comparison dunder, e.g. complex.__eq__(1+1j, 2).
            # Invoke only the left type's slot on the constants, mirroring
            # CPython (returns NotImplemented rather than falling back to the
            # reflected operation when the operands are incompatible).
            lval = args[0].as_python_constant()
            if isinstance(lval, self.fn):
                return ConstantVariable.create(
                    getattr(self.fn, name)(lval, args[1].as_python_constant())
                )

        if self.fn is str and len(args) >= 1:
            resolved_fn = getattr(self.fn, name, None)
            if resolved_fn in str_methods:
                # Only delegate to ConstantVariable, not other types that happen to be constants
                if isinstance(args[0], ConstantVariable):
                    return args[0].call_method(tx, name, args[1:], kwargs)

        if self.fn is float and len(args) >= 1:
            # Only delegate to ConstantVariable, not other types that happen to be constants
            if isinstance(args[0], ConstantVariable):
                return VariableTracker.build(
                    tx, getattr(float, name)(args[0].as_python_constant())
                )

        if name == "__len__" and len(args) == 1 and not kwargs:
            # type.__len__(instance) → len(instance)
            # e.g. list.__len__(my_list) → len(my_list)
            return generic_size(tx, args[0])

        if name == "__str__" and len(args) == 1 and not kwargs:
            return super().call_method(tx, name, args, kwargs)

        if name == "__repr__" and len(args) == 1 and not kwargs:
            return super().call_method(tx, name, args, kwargs)

        if name == "__iter__" and len(args) == 1 and not kwargs:
            # type.__iter__(instance) → iter(instance)
            # e.g., tuple.__iter__(my_tuple) → iter(my_tuple)
            # For builtin types called on user-defined subclasses, use the base iterator
            return generic_getiter(tx, args[0])

        if name == "__neg__" and len(args) == 1 and not kwargs:
            # type.__neg__(instance) → neg(instance)
            # e.g., int.__neg__(4) → neg(4)
            return pynumber_negative(tx, args[0])

        if name == "__pos__" and len(args) == 1 and not kwargs:
            # type.__pos__(instance) → pos(instance)
            # e.g., int.__pos__(4) → pos(4)
            return pynumber_positive(tx, args[0])

        if name == "__abs__" and len(args) == 1 and not kwargs:
            # type.__abs__(instance) → abs(instance)
            # e.g., int.__abs__(-4) → abs(-4)
            return pynumber_absolute(tx, args[0])

        if name == "__invert__" and len(args) == 1 and not kwargs:
            # type.__invert__(instance) → ~instance
            # e.g., int.__invert__(4) → ~4
            return pynumber_invert(tx, args[0])

        if name == "__hash__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if (
                isinstance(arg, variables.UserDefinedConstantVariable)
                and arg._base_vt is not None
            ):
                return generic_hash(tx, arg._base_vt)

        return super().call_method(tx, name, args, kwargs)

    def call_int(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        base: VariableTracker | None = None,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for int: convert arg to an integer, using the numeric
        # conversion protocol, or parse with the given base when a base is supplied.
        if base is None:
            return pynumber_int(tx, arg)
        return pylong_from_base(tx, arg, base)

    def call_float(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for float: convert arg using the numeric float protocol.
        return pynumber_float(tx, arg)

    def call_bin(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for bin: format arg as a binary string via the base-conversion helper.
        return pynumber_tobase(tx, arg, 2)

    def call_oct(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for oct: format arg as an octal string via the base-conversion helper.
        return pynumber_tobase(tx, arg, 8)

    def call_hex(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for hex: format arg as a hexadecimal string via the base-conversion helper.
        return pynumber_tobase(tx, arg, 16)

    def call_bool(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for bool: truth value of arg via the generic truthiness protocol.
        # Emulate PyBool_Type.tp_vectorcall which boils down to PyObject_IsTrue.
        return generic_is_true(tx, arg)

    def call_hash(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for hash: hash of arg via the generic hash protocol.
        return generic_hash(tx, arg)

    def call_repr(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for repr: representation string of arg via the generic repr protocol.
        return generic_repr(tx, arg)

    def call_str(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for str: string conversion of arg via the generic str protocol.
        return generic_str(tx, arg)

    def call___build_class__(self, tx, *args, **kwargs):
        # [FUNCTION DOC] Handler for the class-definition builtin. Requires the config flag
        # enabling class-build tracing; the first argument must be a user function (class
        # body) whose real function can be obtained, and all remaining args and kwargs must be
        # constants. Then really builds the class and returns it as a tracker. Every other
        # case graph-breaks "Invalid call to __build_class__".
        def fail(args, kwargs) -> NoReturn:
            # [FUNCTION DOC] HELPER: Always graph-breaks with "Invalid call to __build_class__".
            unimplemented(
                gb_type="Invalid call to __build_class__",
                context=f"Non-constant args to __build_class__: {args} {kwargs}",
                explanation="Cannot trace class definition: the class body function is unsupported or the base class argument are not compile-time constants",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        if not torch._dynamo.config.enable_trace_load_build_class:
            fail(args, kwargs)

        try:
            if isinstance(args[0], variables.NestedUserFunctionVariable):
                fn = args[0].get_function(allow_sourced_cells=True)
            else:
                fn = args[0].get_function()
        except NotImplementedError:
            fail(args, kwargs)

        if check_constant_args(args[1:], kwargs):
            r = builtins.__build_class__(
                fn,  # type: ignore[possibly-undefined]
                *[a.as_python_constant() for a in args[1:]],
            )
            return VariableTracker.build(tx, r)
        else:
            fail(args, kwargs)

    def _call_min_max(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        key: VariableTracker | None = None,
        default: VariableTracker | None = None,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] HELPER: Shared handler for min and max (assigned to both call_min
        # and call_max). Accepts either several positional values or one iterable, plus
        # optional key (None means no key) and default (only valid for a single iterable).
        # Raises observed type errors for zero args or a default with several positionals, and
        # an observed value error for an empty iterable without default (which is returned if
        # given). Delegates to the plain or key-based reduction.
        name = self.fn.__name__
        if len(args) == 0:
            raise_type_error(tx, f"{name} expected at least 1 argument, got 0")

        # `default` is keyword-only and only valid with a single iterable arg.
        positional = len(args) > 1
        if positional and default is not None:
            raise_type_error(
                tx,
                f"Cannot specify a default for {name}() with multiple positional arguments",
            )

        if key is not None and key.is_constant_none():
            key = None

        items = list(args) if positional else unpack_iterable(tx, args[0])

        if len(items) == 0:
            if default is not None:
                return default
            raise_value_error(tx, f"{name}() iterable argument is empty")

        if key is None:
            return self._call_min_max_seq(tx, items)
        return self._call_min_max_seq_with_key(tx, items, key)

    def _call_min_max_seq(
        self, tx: "InstructionTranslatorBase", items: list[VariableTracker]
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Reduce a non-empty list of trackers to the min or max by
        # pairwise binary comparison; a single item is returned unchanged. Raises an
        # assertion error on an empty list.
        if len(items) <= 0:
            raise AssertionError("_call_min_max_seq requires at least one item")
        if len(items) == 1:
            return items[0]

        return functools.reduce(functools.partial(self._call_min_max_binary, tx), items)  # type: ignore[arg-type,return-value]

    def _call_min_max_seq_with_key(
        self,
        tx: "InstructionTranslatorBase",
        items: list[VariableTracker],
        key: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: min/max with a key function. Applies the key to every item,
        # then scans keeping the first extremal item (replacing only on strictly better
        # key), comparing keys with the less-than operator. The comparison result must be a
        # compile-time constant, otherwise graph-breaks "min/max with non-constant key".
        # Returns the original item, not the key.
        # Mirror CPython builtin_min_max: compare on key(item) but return the
        # original element, keeping the first extremal element on ties (only
        # replace on a strict comparison). Like list.sort, comparisons are done
        # through Dynamo and must fold to a compile-time constant, so Tensor and
        # SymInt keys graph break.
        keyvals = [key.call_function(tx, [item], {}) for item in items]
        best_item, best_key = items[0], keyvals[0]
        for item, keyval in zip(items[1:], keyvals[1:]):
            left, right = (best_key, keyval) if self.fn is max else (keyval, best_key)
            cmp = BuiltinVariable(operator.lt).call_function(tx, [left, right], {})
            if not cmp.is_python_constant():
                unimplemented(
                    gb_type="min/max with non-constant key",
                    context=str(keyval),
                    explanation=(
                        f"Cannot compute {self.fn.__name__}() whose key comparison is "
                        f"not a compile-time constant. Key type: {keyval.python_type()}. "
                        f"Most notably, Tensor or SymInt keys are unsupported, but ints work."
                    ),
                    hints=[
                        "Use something else as the key.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            if cmp.as_python_constant():
                best_item, best_key = item, keyval
        return best_item

    def _call_min_max_binary(
        self,
        tx: "InstructionTranslatorBase",
        a: VariableTracker | None,
        b: VariableTracker | None,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] HELPER: Binary min or max of two trackers. Returns None if either is
        # None. With tensor args: scalar-item results are converted to tensors; symbolic
        # operands become a graph call of the builtin; a constant second operand becomes a
        # clamp (or numpy clip); otherwise elementwise maximum/minimum. If both operands are
        # unspecialized scalars or constants the result is an unspecialized-scalar (or
        # fake-item) tracker carrying the raw computed value. Symbolic operands without
        # tensors use symbolic max/min; two constants are computed directly; any other
        # combination returns None.
        if a is None or b is None:
            # a or b could be none if we reduce and _call_min_max_binary failed
            # to return something
            return None
        if self.tensor_args(a, b):
            if not a.is_tensor():
                a, b = b, a
            if not a.is_tensor():
                raise AssertionError(
                    "Expected at least one tensor argument for min/max"
                )

            # result of an item call is a scalar convert to a tensor
            if isinstance(a, FakeItemVariable):
                a = variables.TorchInGraphFunctionVariable(torch.tensor).call_function(
                    tx, [a], {}
                )

            # Dynamic input does not get resolved, rather, gets stored as call_function
            if isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
                from .builder import wrap_fx_proxy_cls

                return wrap_fx_proxy_cls(
                    type(a),
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.fn,
                        *proxy_args_kwargs([a, b], {}),
                    ),
                )

            # convert min/max to torch ops
            if b.is_python_constant():
                fn: VariableTracker
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np

                    fn = variables.NumpyVariable(np.clip)
                else:
                    fn = variables.TorchInGraphFunctionVariable(torch.clamp)
                kwargs = {"min": b} if (self.fn is max) else {"max": b}
                result = fn.call_function(tx, [a], kwargs)
            else:
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np

                    np_fn = {max: np.maximum, min: np.minimum}[self.fn]
                    fn = variables.NumpyVariable(np_fn)
                else:
                    torch_fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                    fn = variables.TorchInGraphFunctionVariable(torch_fn)
                result = fn.call_function(tx, [a, b], {})

            # return unspec if both a, b are unspec or const
            if all(
                isinstance(
                    i,
                    (
                        variables.UnspecializedPythonVariable,
                        variables.ConstantVariable,
                    ),
                )
                for i in [a, b]
            ):
                if any(isinstance(val, FakeItemVariable) for val in [a, b]):
                    # type: ignore[arg-type]
                    return variables.FakeItemVariable.from_tensor_variable(result)

                if b.is_python_constant():
                    raw_b = b.as_python_constant()
                else:
                    raw_b = b.raw_value  # type: ignore[attr-defined]
                if self.fn is max:
                    raw_res = max(a.raw_value, raw_b)  # type: ignore[attr-defined]
                else:
                    raw_res = min(a.raw_value, raw_b)  # type: ignore[attr-defined]

                need_unwrap = any(
                    x.need_unwrap
                    for x in [a, b]
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )
                return variables.UnspecializedPythonVariable.from_tensor_variable(
                    result,  # type: ignore[arg-type]
                    raw_res,
                    need_unwrap,
                )
            # otherwise return tensor
            else:
                return result
        elif isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
            py_fn = torch.sym_max if self.fn is max else torch.sym_min
            proxy = tx.output.create_proxy(
                "call_function", py_fn, *proxy_args_kwargs([a, b], {})
            )
            return SymNodeVariable.create(tx, proxy, None)
        elif isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            value = self.fn(
                a.as_python_constant(),
                b.as_python_constant(),
            )
            return VariableTracker.build(tx, value)
        return None

    call_min = _call_min_max
    call_max = _call_min_max

    def call_abs(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for abs: absolute value via the numeric protocol helper.
        return pynumber_absolute(tx, arg)

    def call_pos(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the unary-plus operator function via the numeric protocol helper.
        return pynumber_positive(tx, arg)

    def call_index(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the index operator function: convert arg to an exact
        # integer via the index protocol (symbolic values are specialized to constants).
        # Specialize SymNodeVariable to a constant first, matching CPython's
        # PyNumber_Index which forces a concrete int.
        return pynumber_index(tx, arg)

    def call_round(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for round: looks up the rounding method on arg and calls it
        # with the remaining positional and keyword arguments; returns its result.
        from .builder import SourcelessBuilder

        # Call arg.__round__()
        round_method = SourcelessBuilder.create(tx, getattr).call_function(
            tx, [arg, VariableTracker.build(tx, "__round__")], {}
        )
        return round_method.call_function(tx, list(args), kwargs)

    def call_range(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for range. Raises an observed type error for keyword
        # arguments or an argument count outside one to three; converts each argument with
        # the index protocol and returns a range tracker.
        if kwargs:
            raise_type_error(tx, "range() takes no keyword arguments")
        if len(args) == 0:
            raise_type_error(tx, "range expected at least 1 argument, got 0")
        if len(args) > 3:
            raise_type_error(tx, f"range expected at most 3 arguments, got {len(args)}")
        args = tuple(pynumber_index(tx, arg) for arg in args)
        return variables.RangeVariable(list(args))

    def _dynamic_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        # [FUNCTION DOC] HELPER: Return whether any positional or keyword argument is a symbolic node tracker.
        return any(isinstance(x, SymNodeVariable) for x in args) or any(
            isinstance(x, SymNodeVariable) for x in kwargs.values()
        )

    def call_slice(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for slice: requires one to three arguments (otherwise an
        # observed type error) and returns a slice tracker built from them.
        if not 1 <= len(args) < 4:
            raise_type_error(tx, f"slice expected at least 1 argument, got {len(args)}")
        return variables.SliceVariable(list(args), tx)

    def _dyn_proxy(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Record the wrapped callable as a graph call on the given
        # arguments and wrap the proxy in a tracker. Side effect: adds a node to the graph.
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs(args, kwargs)
            ),
        )

    def call_tuple(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for tuple. Keyword arguments or more than one argument raise
        # an observed type error. No argument gives a new empty tuple tracker; an existing
        # exact tuple tracker is returned as is; anything else is unpacked into a new tuple tracker.
        # ref: https://github.com/python/cpython/blob/main/Objects/abstract.c#L2004-L2078
        if kwargs:
            raise_type_error(
                tx,
                f"{self.fn.__name__}() takes no keyword arguments",
            )
        if len(args) == 0:
            return TupleVariable([], mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"{self.fn.__name__} expected at most 1 argument, got {len(args)}",
            )

        obj = args[0]
        if isinstance(obj, TupleVariable) and obj.python_type() is tuple:
            return obj

        items = unpack_iterable(tx, args[0])
        return TupleVariable(items, mutation_type=ValueMutationNew())

    def call_callable(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for callable: constant boolean telling whether the argument's
        # Python type defines a call slot (graph-breaks if the type cannot be determined).
        # PyCallable_Check: callable(x) is type(x)->tp_call != NULL. Dispatch on
        # the arg's Python type slot rather than enumerating callable VTs.
        # callable() is the only handler for this builtin, so always return a
        # result (or graph break via maybe_get_python_type) -- never None, which
        # would signal "try another handler" that does not exist.
        return variables.ConstantVariable.create(
            pycallable_check(maybe_get_python_type(arg))
        )

    def call_cast(
        self, _: Any, *args: VariableTracker, **kwargs: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for typing.cast: with exactly two arguments returns the
        # second (the value) unchanged; otherwise graph-breaks "bad args to builtin cast()".
        if len(args) == 2:
            return args[1]

        unimplemented(
            gb_type="bad args to builtin cast()",
            context=f"got args {args} {kwargs}",
            explanation="Dynamo expects exactly 2 args to builtin cast().",
            hints=["Ensure your call to cast() has exactly 2 arguments."],
        )

    def call_dir(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for dir. Supports user-defined classes, wrapped builtins and
        # compile-time constants by computing the real name list and wrapping it; returns
        # None for any other tracker so other handlers can try.
        if isinstance(arg, variables.UserDefinedClassVariable):
            return VariableTracker.build(tx, dir(arg.value))
        if isinstance(arg, BuiltinVariable):
            return VariableTracker.build(tx, dir(arg.fn))
        # Enable specialized VTs for constants to work with dir()
        if arg.is_python_constant():
            return VariableTracker.build(tx, dir(arg.as_python_constant()))
        return None

    def call_set(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for set. Any keyword argument or more than one positional
        # raises an observed type error. No argument gives a new empty set tracker; one
        # argument gives a new set tracker filled by calling its update method with it.
        # ref: https://github.com/python/cpython/blob/main/Objects/setobject.c#L2708-L2735
        # CPython set_init rejects keywords before unpacking positional args, so
        # set(a=1) and set().__init__(a=1) both raise regardless of arg count.
        if kwargs:
            raise_type_error(
                tx,
                "set() takes no keyword arguments",
            )
        if len(args) == 0:
            return variables.SetVariable(set(), mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"set expected at most 1 argument, got {len(args)}",
            )

        s = SetVariable([], mutation_type=ValueMutationNew())
        s.call_method(tx, "update", [args[0]], kwargs)
        return s

    def call_frozenset(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for frozenset. Keywords or more than one positional raise an
        # observed type error. No argument gives an empty frozenset tracker; an existing exact
        # frozenset tracker is returned as is; a set or dict tracker reuses its existing
        # hashable keys; otherwise the argument is unpacked into items. Returns a new
        # frozenset tracker.
        if kwargs:
            raise_type_error(
                tx,
                "frozenset() takes no keyword arguments",
            )
        if len(args) == 0:
            return variables.FrozensetVariable(set(), mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"frozenset expected at most 1 argument, got {len(args)}",
            )

        if istype(args[0], variables.FrozensetVariable):
            # CPython: frozenset(existing_frozenset) returns the same object.
            return args[0]

        # Reuse existing HashableTracker keys from a set/frozenset/dict operand
        # instead of re-hashing, mirroring CPython's set_update_internal fast
        # path (do-not-rehash-dict-keys).
        if isinstance(args[0], (variables.SetVariable, variables.ConstDictVariable)):
            items = list(args[0].items.keys())
        else:
            items = unpack_iterable(tx, args[0])
        fs = FrozensetVariable(items, mutation_type=ValueMutationNew())
        return fs

    def call_zip(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for zip. Only the strict keyword is allowed (other kwargs
        # raise an argument-mismatch error). Gets an iterator for each argument and returns
        # a zip tracker over them, with the strictness flag taken from the keyword.
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L2822-L2887
        if kwargs:
            if not (len(kwargs) == 1 and "strict" in kwargs):
                raise_args_mismatch(
                    tx,
                    "zip",
                    "1 kwargs (`strict`)",
                    f"{len(kwargs)} kwargs",
                )
        strict = kwargs.pop("strict", ConstantVariable.create(False))
        items = []
        for arg in args:
            items.append(generic_getiter(tx, arg))
        iter_args = TupleVariable(items, mutation_type=ValueMutationNew())
        return variables.ZipVariable(
            iter_args,
            strict=strict.as_python_constant(),
            mutation_type=ValueMutationNew(),
        )

    def call_len(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for len: needs exactly one positional argument and no
        # keywords (else an observed type error); returns its length via the generic size protocol.
        if kwargs:
            raise_type_error(tx, "len() takes no keyword arguments")
        if len(args) != 1:
            raise_type_error(
                tx, f"len() takes exactly one argument ({len(args)} given)"
            )
        return generic_size(tx, args[0])

    def call_length_hint(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the length-hint operator function. Takes one object and
        # an optional default (zero if omitted); wrong arity or keywords raise an observed
        # type error. Uses the real length if the type has one; else uses the default when
        # the type lacks a length-hint method; else calls that method, returning the default
        # if it raises a type error.
        # ref: PyObject_LengthHint (Objects/abstract.c): try __len__, then
        # __length_hint__, falling back to the supplied default for either a
        # missing slot or a TypeError raised by the slot.
        if kwargs or not (1 <= len(args) <= 2):
            raise_type_error(
                tx, f"length_hint expected 1 or 2 arguments, got {len(args)}"
            )
        obj = args[0]
        default = args[1] if len(args) == 2 else ConstantVariable.create(0)

        obj_type = maybe_get_python_type(obj)

        if type_implements_sq_length(obj_type) or type_implements_mp_length(obj_type):
            return generic_size(tx, obj)

        if getattr(obj_type, "__length_hint__", None) is None:
            return default
        try:
            return obj.call_method(tx, "__length_hint__", [], {})
        except ObservedTypeError:
            handle_observed_exception(tx)
            return default

    def call_getitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the getitem operator function: exactly two positional
        # arguments (container, key), no keywords, else an observed type error; delegates
        # to the generic subscript protocol.
        if kwargs:
            raise_type_error(tx, "_operator.getitem() takes no keyword arguments")
        if len(args) != 2:
            raise_type_error(tx, f"getitem expected 2 arguments, got {len(args)}")
        return generic_getitem(tx, args[0], args[1])

    def call_setitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the setitem operator function: exactly three positional
        # arguments (container, key, value), no keywords, else an observed type error;
        # delegates to the generic item-assignment protocol.
        if kwargs:
            raise_type_error(tx, "_operator.setitem() takes no keyword arguments")
        if len(args) != 3:
            raise_type_error(tx, f"setitem expected 3 arguments, got {len(args)}")
        return generic_setitem(tx, args[0], args[1], args[2])

    def call_delitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the delitem operator function: exactly two positional
        # arguments (container, key), no keywords, else an observed type error; delegates
        # to the generic item-deletion protocol.
        if kwargs:
            raise_type_error(tx, "_operator.delitem() takes no keyword arguments")
        if len(args) != 2:
            raise_type_error(tx, f"delitem expected 2 arguments, got {len(args)}")
        return generic_delitem(tx, args[0], args[1])

    def call_isinstance(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        isinstance_type_var: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for isinstance, always producing a constant boolean (or a
        # graph break / observed error). Graph-breaks if the argument's Python type is
        # unknown. For sourced async-collective tensors, reinstalls a type guard when the
        # checked classes distinguish them from plain tensors. For tensors with a dtype:
        # graph-breaks on custom instance/subclass check hooks, otherwise answers using
        # dtype pseudo-types, the parameter metaclass hook for wrapper subclasses, or a
        # subclass test on the tensor's type. Graph-breaks for user-defined objects wrapping
        # C-extension member descriptors; honors a user-class instance-check hook;
        # handles user exception classes; normalizes the class-info (class, union, tuple,
        # or object with an instance-check hook) or raises an observed TypeError; finally
        # answers with a subclass test of the argument's type against the class-info tuple,
        # falling back to membership when that test raises a TypeError.
        try:
            arg_type = arg.python_type()
        except NotImplementedError:
            unimplemented(
                gb_type="builtin isinstance() cannot determine type of argument",
                context=f"isinstance({arg}, {isinstance_type_var})",
                explanation=f"Dynamo doesn't have a rule to determine the type of argument {arg}",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        isinstance_type = isinstance_type_var.as_python_constant()
        # An AsyncCollectiveTensor (ACT) input's tensor-class guard is relaxed
        # (see VariableBuilder.wrap_tensor): the cache entry does not discriminate
        # ACT from the resolved plain Tensor. isinstance() observes the class and
        # is constant-folded below, so reinstall the class guard -- but only when
        # the result would actually differ between the ACT and the resolved
        # Tensor. isinstance(w, ACT) discriminates and must recompile;
        # isinstance(w, torch.Tensor) is True for both and must stay reusable.
        if isinstance(arg, variables.TensorVariable) and arg.source is not None:
            from torch._functorch._aot_autograd.utils import (
                is_async_collective_tensor_type,
            )

            if is_async_collective_tensor_type(arg_type):
                check_types = (
                    isinstance_type
                    if isinstance(isinstance_type, tuple)
                    else (isinstance_type,)
                )
                # Guard if any checked type distinguishes ACT from torch.Tensor,
                # or conservatively if a checked entry is not a plain class.
                if any(
                    not isinstance(ty, type)
                    or issubclass(arg_type, ty) != issubclass(torch.Tensor, ty)
                    for ty in check_types
                ):
                    install_guard(arg.source.make_guard(GuardBuilder.TYPE_MATCH))
        if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:
            if _uses_custom_classinfo_check(isinstance_type):
                unimplemented(
                    gb_type="builtin isinstance() with custom type check on tensor",
                    context=f"isinstance({arg}, {isinstance_type})",
                    explanation="Dynamo cannot soundly trace arbitrary custom "
                    "__instancecheck__ or __subclasscheck__ hooks on tensor "
                    "values because the hook may read external mutable state.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )

            def _tensor_isinstance(
                tensor_var: VariableTracker, tensor_type: Any
            ) -> bool:
                # [FUNCTION DOC] HELPER: Evaluate whether the tensor matches a class or tuple
                # of classes (true if any matches), using the enclosing argument tracker.
                # KNOWN ISSUE: the tensor_var parameter is unused; the enclosing arg is read instead.
                def check_type(ty: Any) -> bool:
                    # [FUNCTION DOC] HELPER: Check one class against the tensor. Legacy
                    # tensor type names match on the tensor's dtype; the parameter class is
                    # checked directly on the example value for traceable wrapper
                    # subclasses (to run its metaclass hook); other classes use a subclass
                    # test on the tracker's Python type.
                    if ty not in tensortype_to_dtype:
                        example_val = arg.as_proxy().node.meta["example_value"]
                        if (
                            is_traceable_wrapper_subclass(example_val)
                            and ty is torch.nn.parameter.Parameter
                        ):
                            # N.B: we are calling isinstance directly on the example value.
                            # torch.nn.Parameter has a meta-class that overrides __isinstance__,
                            # the isinstance check here allows us to invoke that logic.
                            return isinstance(example_val, ty)
                        else:
                            return issubclass(arg.python_type(), ty)

                    dtypes = tensortype_to_dtype[ty]
                    # pyrefly: ignore [missing-attribute]
                    return arg.dtype in dtypes

                if type(tensor_type) is tuple:
                    return any(check_type(ty) for ty in tensor_type)
                else:
                    return check_type(tensor_type)

            return VariableTracker.build(tx, _tensor_isinstance(arg, isinstance_type))
        # UserDefinedObject with C extensions can have torch.Tensor attributes,
        # so break graph.
        if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(
            arg.value, types.MemberDescriptorType
        ):
            unimplemented(
                gb_type="isinstance() called on user defined object with C extensions",
                context=f"isinstance({arg}, {isinstance_type})",
                explanation="User-defined object with C extensions can have torch.Tensor "
                "attributes; intentionally graph breaking.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # handle __instancecheck__ defined in user class
        if (
            isinstance(arg, variables.UserDefinedObjectVariable)
            and "__instancecheck__" in isinstance_type.__class__.__dict__
        ):
            return VariableTracker.build(
                tx,
                isinstance_type.__class__.__instancecheck__(isinstance_type, arg.value),
            )

        if isinstance(arg, variables.UserDefinedExceptionClassVariable):
            # pyrefly: ignore [unbound-name]
            return VariableTracker.build(tx, isinstance(arg_type, isinstance_type))

        isinstance_type_tuple: tuple[type, ...]
        if isinstance(isinstance_type, type) or callable(
            # E.g. isinstance(obj, typing.Sequence)
            getattr(isinstance_type, "__instancecheck__", None)
        ):
            isinstance_type_tuple = (isinstance_type,)
        elif isinstance(isinstance_type, types.UnionType):
            isinstance_type_tuple = typing.get_args(isinstance_type)
        elif isinstance(isinstance_type, tuple) and all(
            isinstance(tp, type) or callable(getattr(tp, "__instancecheck__", None))
            for tp in isinstance_type
        ):
            isinstance_type_tuple = isinstance_type
        else:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    "isinstance() arg 2 must be a type, a tuple of types, or a union"
                ],
            )

        try:
            # NB: `isinstance()` does not call `__subclasscheck__` but use `__instancecheck__`.
            # But usually `isinstance(obj, type_info)` and `issubclass(type(obj), type_info)` gives
            # the same result.
            # WARNING: This might run arbitrary user code `__subclasscheck__` and we did not trace
            # through it. This is a limitation of the current implementation.
            # Usually `__subclasscheck__` and `__instancecheck__` can be constant fold through, it
            # might not be a big issue and we trade off it for performance.
            val = issubclass(arg_type, isinstance_type_tuple)
        except TypeError:
            val = arg_type in isinstance_type_tuple
        return VariableTracker.build(tx, val)

    def call_issubclass(
        self,
        tx: "InstructionTranslatorBase",
        left_ty: VariableTracker,
        right_ty: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for issubclass: delegates to the generic subclass-check protocol helper.
        """Checks if first arg is subclass of right arg"""
        return generic_issubclass(tx, left_ty, right_ty)

    def call_super(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for two-argument super: returns a super tracker built from the class and instance trackers.
        return variables.SuperVariable(a, b)

    def call_next(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for next. Needs one or two positional args and no keywords
        # (else an observed type error); the first must be an iterator type (else an observed
        # type error). Returns the iterator's next item; when it is exhausted, returns the
        # optional default if given, otherwise re-raises the observed stop-iteration.
        if kwargs:
            raise_type_error(tx, "next() takes no keyword arguments")
        if len(args) == 0:
            raise_type_error(tx, "next expected at least 1 argument, got 0")
        if len(args) > 2:
            raise_type_error(tx, f"next expected at most 2 arguments, got {len(args)}")
        arg = args[0]
        if not pyiter_check(maybe_get_python_type(arg)):
            raise_type_error(
                tx, f"'{arg.python_type_name()}' object is not an iterator"
            )
        try:
            return arg.next_variable(tx)
        except ObservedUserStopIteration:
            if len(args) == 2:
                return args[1]
            raise

    def call_map(
        self,
        tx: "InstructionTranslatorBase",
        fn: VariableTracker,
        *seqs: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for map: needs a function and at least one iterable (else an
        # observed type error). The strict keyword is accepted only on Python 3.14+; any
        # other keyword raises an argument-mismatch error. Returns a map tracker over
        # iterators of the inputs.
        if len(seqs) == 0:
            raise_observed_exception(
                TypeError,
                tx,
                args=["map() must have at least two arguments."],
            )

        strict = ConstantVariable.create(False)
        if kwargs:
            if sys.version_info >= (3, 14):
                if not (len(kwargs) == 1 and "strict" in kwargs):
                    raise_args_mismatch(
                        tx,
                        "map",
                        "1 kwargs (`strict`)",
                        f"{len(kwargs)} kwargs",
                    )
                strict = kwargs.pop("strict", ConstantVariable.create(False))
            else:
                raise_args_mismatch(
                    tx,
                    "map",
                    "0 kwargs",
                    f"{len(kwargs)} kwargs",
                )

        iterables = [generic_getiter(tx, seq) for seq in seqs]
        iter_args = TupleVariable(iterables, mutation_type=ValueMutationNew())
        return variables.MapVariable(
            fn,
            iter_args,
            strict=strict.as_python_constant(),
            mutation_type=ValueMutationNew(),
        )

    def call_filter(
        self, tx: "InstructionTranslatorBase", fn: VariableTracker, seq: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for filter: returns a filter tracker over the function and an iterator of the sequence.
        return variables.FilterVariable(
            fn,
            generic_getiter(tx, seq),
            mutation_type=ValueMutationNew(),
        )

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Attribute lookup on the wrapped builtin. When the wrapped callable
        # is object, reads the real attribute (a missing one raises an observed
        # AttributeError) and returns non-callable values as constants. Otherwise (and for
        # callables) returns a generic attribute-access tracker carrying the attribute's
        # Python type when known and an attribute source.
        # Declarative type-attribute dispatch (__name__, __bases__, __base__,
        # __flags__), mirroring the consultation at the top of
        # VariableTracker.getattro_impl. Inlined because this override keeps its
        # own object / GetAttrVariable handling below instead of delegating.
        source = self.source and AttrSource(self.source, name)
        if self.fn is object:
            # for object, we can just directly read the attribute
            try:
                value = getattr(self.fn, name)
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
            if not callable(value):
                return VariableTracker.build(tx, value, source)
        attr = getattr(self.fn, name, None)
        return variables.GetAttrVariable(
            self, name, py_type=type(attr) if attr is not None else None, source=source
        )

    def call_delattr(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        name_var: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for delattr: forwards to the object's attribute-deletion method with the name.
        return obj.call_method(tx, "__delattr__", [name_var], {})

    def call_type(
        self, tx: "InstructionTranslatorBase", obj: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for single-argument type. Gets the object's Python type (an
        # unknown type raises a user error of kind invalid input) and returns a tracker for
        # that class. The source is a type source derived from the object's source, or the
        # user-object class source; for plain tensors it is the torch module's Tensor
        # attribute, installing the torch module as a global (side effect).
        try:
            py_type = obj.python_type()
        except NotImplementedError as error:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                str(error),
                case_name="unknown_python_type",
            ) from None

        source = obj.source and TypeSource(obj.source)
        if (
            source is None
            and isinstance(obj, variables.UserDefinedObjectVariable)
            and obj.cls_source
        ):
            source = obj.cls_source
        if py_type is torch.Tensor:
            # In some cases torch isn't available in globals
            name = tx.output.install_global_by_id("", torch)
            source = AttrSource(GlobalSource(name), "Tensor")

        return VariableTracker.build(tx, py_type, source)

    def call_reversed(
        self, tx: "InstructionTranslatorBase", obj: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for reversed. If the object's type defines the reversed
        # hook as None, raises an observed type error; if it defines the hook, calls it. Else
        # requires the sequence protocol (otherwise an observed type error) and returns the
        # result of a polyfill reverse sequence iterator.
        # Mirrors CPython's builtin_reversed_impl (Python/enumobject.c)
        # https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Objects/enumobject.c#L353-L395
        # 1. Look up __reversed__ via _PyObject_LookupSpecial. If found, call it.
        # 2. Else require PySequence_Check (sq_item). If absent, TypeError.
        # 3. Else build a reverse sequence iterator over __len__ + __getitem__.

        obj_type = maybe_get_python_type(obj)

        # Type-level __reversed__ lookup, mirrors _PyObject_LookupSpecial.
        # getattr_static skips descriptors / metaclass. CPython treats
        # `__reversed__ = None` on the type as an explicit opt-out, raising
        # TypeError even if the sequence protocol would otherwise work.
        reversed_attr = inspect.getattr_static(
            obj_type, "__reversed__", _MISSING_SENTINEL
        )
        if reversed_attr is None:
            raise_type_error(tx, f"'{obj_type.__name__}' object is not reversible")
        if reversed_attr is not _MISSING_SENTINEL:
            return obj.call_method(tx, "__reversed__", [], {})

        if not pysequence_check(obj_type):
            raise_type_error(tx, "argument to reversed() must be a sequence")

        return variables.UserFunctionVariable(
            polyfills.builtins.reversed_sequence_iterator
        ).call_function(tx, [obj], {})

    def call_sorted(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for sorted. For non-tensor input, unpacks it into a new
        # list tracker and sorts it with the list sort method using the keyword arguments
        # (key, reverse); returns the list. Returns None for tensor input so other handlers
        # may handle it.
        if not isinstance(obj, variables.TensorVariable):
            list_var = variables.ListVariable(
                unpack_iterable(tx, obj),
                mutation_type=ValueMutationNew(),
            )
            list_var.call_method(tx, "sort", [], kwargs)
            return list_var
        return None

    def call_neg(
        self, tx: "InstructionTranslatorBase", a: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the negation operator function via the numeric protocol helper.
        return pynumber_negative(tx, a)

    def call_invert(
        self, tx: "InstructionTranslatorBase", a: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the bitwise-invert operator function via the numeric protocol helper.
        return pynumber_invert(tx, a)

    def call_format(
        self,
        tx: "InstructionTranslatorBase",
        _format_string: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the string format method used as a function: converts the
        # first argument (must be a compile-time constant) to a string and returns a lazy
        # string-format tracker holding the remaining positional and keyword trackers.
        format_string = _format_string.as_python_constant()
        format_string = str(format_string)
        return StringFormatVariable.create(format_string, list(args), kwargs)

    def call_id(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for id. Needs exactly one argument (else an observed type
        # error). If the tracker knows its real id, installs the id-appropriate guard on its
        # source (when it has one) and returns that id as a constant; otherwise returns a
        # fake-id tracker built from the tracker object's own identity.
        if len(args) != 1:
            raise_observed_exception(
                TypeError,
                tx,
                args=[f"id() takes exactly one argument ({len(args)} given)"],
            )
        arg = args[0]

        real_id = arg.get_id(tx)
        if real_id is not None:
            if arg.source:
                guard_type = arg.get_id_guard_type()
                if guard_type is not None:
                    install_guard(arg.source.make_guard(guard_type))
            return VariableTracker.build(tx, real_id)

        return FakeIdVariable(id(arg))

    def call_deepcopy(
        self, tx: "InstructionTranslatorBase", x: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for deepcopy: always graph-breaks with "copy.deepcopy()" (unsupported).
        unimplemented(
            gb_type="copy.deepcopy()",
            context=f"copy.deepcopy({x})",
            explanation="Dynamo does not support copy.deepcopy()",
            hints=[
                "Avoid calling copy.deepcopy()",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def _comparison_with_tensor(
        self,
        tx: "InstructionTranslatorBase",
        left: VariableTracker,
        right: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Comparison where at least one operand is a tensor. For
        # is / is-not, compares the identity of the underlying fake example values and returns
        # a constant. Other operators must be supported tensor comparisons (else graph break);
        # when both tensors have known, different sizes they must be broadcastable (else graph
        # break). Records a graph call and wraps it with the tensor tracker class of the
        # tensor operand (tensor or ndarray).
        from .builder import wrap_fx_proxy_cls
        from .tensor import supported_tensor_comparison_op_values

        op = self.fn

        if op in [operator.is_, operator.is_not]:
            is_result = (
                left.is_tensor()
                and right.is_tensor()
                and id(extract_fake_example_value(left.as_proxy().node))
                == id(extract_fake_example_value(right.as_proxy().node))
            )
            if op is operator.is_:
                return VariableTracker.build(tx, is_result)
            else:
                return VariableTracker.build(tx, not is_result)

        if op not in supported_tensor_comparison_op_values:
            unimplemented(
                gb_type="unsupported Tensor comparison op",
                context=f"{op.__name__}({left}, {right})",
                explanation=f"Dynamo does not support the comparison op {op.__name__} "
                f"with Tensor arguments {left}, {right}",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if (
            isinstance(left, TensorVariable)
            and isinstance(right, TensorVariable)
            and (left.size and right.size) is not None
            and left.size != right.size
        ):
            try:
                torch.broadcast_shapes(left.size, right.size)
            except RuntimeError:
                # not broadcastable, can't be compared
                unimplemented(
                    gb_type="failed to broadcast when attempting Tensor comparison op",
                    context=f"{op.__name__}({left}, {right})",
                    explanation=f"Dynamo was unable to broad cast the arguments {left}, {right} "
                    f"when attempting to trace the comparison op {op.__name__}.",
                    hints=[*graph_break_hints.USER_ERROR],
                )
        tensor_cls = left if left.is_tensor() else right
        proxy = tx.output.create_proxy(
            "call_function", op, (left.as_proxy(), right.as_proxy()), {}
        )
        return wrap_fx_proxy_cls(
            type(tensor_cls),  # handle Ndarrays and Tensors
            tx,
            proxy,
        )

    def _comparison_with_symnode(
        self,
        tx: "InstructionTranslatorBase",
        left: VariableTracker,
        right: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Comparison where at least one operand is a symbolic node.
        # Graph-breaks for unsupported operators. If the other operand is not symbolic,
        # constant or tensor, the answer is known at compile time and returned as a constant
        # from comparing two unrelated objects. Otherwise records a graph call and returns a
        # new symbolic node tracker.
        from .tensor import supported_tensor_comparison_op_values

        op = self.fn

        if op not in supported_tensor_comparison_op_values:
            unimplemented(
                gb_type="unsupported SymNode comparison op",
                context=f"{op.__name__}({left}, {right})",
                explanation=f"Dynamo does not support the comparison op {op.__name__} "
                f"with SymNode arguments {left}, {right}",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        # SymNodes are numeric (int/float/bool). The non-SymNode operand
        # must be a type that can participate in a traced numeric comparison.
        # Anything else (classes, DataPtrVariable, etc.) is a different type
        # entirely — the comparison result is known at compile time.
        non_symnode = right if isinstance(left, SymNodeVariable) else left
        if not isinstance(
            non_symnode, (SymNodeVariable, ConstantVariable, TensorVariable)
        ):
            # pyrefly: ignore [bad-argument-type]
            return VariableTracker.build(tx, op(object(), None))

        proxy = tx.output.create_proxy(
            "call_function", op, (left.as_proxy(), right.as_proxy()), {}
        )
        return SymNodeVariable.create(
            tx,
            proxy,
            sym_num=None,
        )

    def call_xor(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the bitwise-xor operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_xor", "^")

    def call_ixor(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place xor operator function via the in-place protocol helper (falls back to xor).
        return binary_iop(tx, a, b, "nb_inplace_xor", "nb_xor", "^=")

    def call_mul(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the multiplication operator function via the numeric multiply helper.
        return pynumber_multiply(tx, a, b)

    def call_imul(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place multiplication operator function via the in-place multiply helper.
        return pynumber_inplace_multiply(tx, a, b)

    def call_matmul(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the matrix-multiplication operator function via the matrix multiply helper.
        return pynumber_matrix_multiply(tx, a, b)

    def call_imatmul(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place matrix-multiplication operator function via the in-place helper.
        return pynumber_inplace_matrix_multiply(tx, a, b)

    def call_sub(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the subtraction operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_subtract", "-")

    def call_isub(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place subtraction operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_subtract", "nb_subtract", "-=")

    def call_add(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the addition operator function via the numeric add helper.
        return pynumber_add(tx, a, b)

    def call_iadd(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place addition operator function via the in-place add helper.
        return pynumber_inplace_add(tx, a, b)

    def call_and_(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the bitwise-and operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_and", "&")

    def call_iand(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place bitwise-and operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_and", "nb_and", "&=")

    def call_or_(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the bitwise-or operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_or", "|")

    def call_ior(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place bitwise-or operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_or", "nb_or", "|=")

    def call_lshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the left-shift operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_lshift", "<<")

    def call_ilshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place left-shift operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_lshift", "nb_lshift", "<<=")

    def call_rshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the right-shift operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_rshift", ">>")

    def call_irshift(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place right-shift operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_rshift", "nb_rshift", ">>=")

    def call_floordiv(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the floor-division operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_floor_divide", "//")

    def call_ifloordiv(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place floor-division operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_floor_divide", "nb_floor_divide", "//=")

    def call_truediv(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the true-division operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_true_divide", "/")

    def call_itruediv(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place true-division operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_true_divide", "nb_true_divide", "/=")

    def call_mod(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the modulo operator function via the binary-operation protocol helper.
        return binary_op(tx, a, b, "nb_remainder", "%")

    def call_imod(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place modulo operator function via the in-place protocol helper.
        return binary_iop(tx, a, b, "nb_inplace_remainder", "nb_remainder", "%=")

    def call_divmod(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for divmod via the binary-operation protocol helper (divmod slot, no in-place form).
        # PyNumber_Divmod dispatches through the nb_divmod slot with no
        # in-place form. https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1056
        return binary_op(tx, a, b, "nb_divmod", "divmod()")

    def call_pow(
        self,
        tx: "InstructionTranslatorBase",
        a: VariableTracker,
        b: VariableTracker,
        c: VariableTracker | None = None,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for pow: power of a by b with optional modulus c, via the ternary-operation protocol helper.
        return ternary_op(tx, a, b, c, "nb_power", "** or pow()")

    def call_ipow(
        self,
        tx: "InstructionTranslatorBase",
        a: VariableTracker,
        b: VariableTracker,
        c: VariableTracker | None = None,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the in-place power operator function via the ternary in-place protocol helper.
        return ternary_iop(tx, a, b, c, "nb_inplace_power", "nb_power", "**=")

    def call_not_(
        self, tx: "InstructionTranslatorBase", a: VariableTracker
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Handler for the logical-not operator function. A symbolic node
        # records a graph call and returns a symbolic result; dictionary views are unwrapped
        # to their dictionary; list, dict and set trackers give a constant (true if empty);
        # user-defined objects use their truth value; anything else returns None.
        if isinstance(a, SymNodeVariable):
            return SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function", operator.not_, *proxy_args_kwargs([a], {})
                ),
                sym_num=None,
            )

        # Unwrap the underlying ConstDictVariable
        if isinstance(a, DictViewVariable):
            a = a.dv_dict
        if isinstance(a, (ListVariable, ConstDictVariable, SetVariable)):
            return VariableTracker.build(tx, len(a.items) == 0)
        if isinstance(a, UserDefinedObjectVariable):
            bool_result = self.call_bool(tx, a)
            return VariableTracker.build(tx, not bool_result.value)  # type: ignore[missing-attribute]

        return None

    def call_contains(
        self, tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
    ) -> VariableTracker:
        # [FUNCTION DOC] Handler for the contains operator function: membership test via the sequence-contains helper.
        return pysequence_contains(tx, a, b)


class DictBuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for the dict type itself. Calling it builds a dict by inlining the
    # construct-dict polyfill; its methods (__new__, fromkeys, and unbound dict methods
    # applied to a dict tracker) are forwarded to the underlying dictionary trackers.
    # Exposes static helpers for building dicts of a given dict-like class and for fromkeys.
    """Variable tracker for the `dict` builtin constructor."""

    _fn = dict

    def __init__(self, value: type = dict, **kwargs: Any) -> None:
        # [FUNCTION DOC] Create the tracker. Raises an assertion error unless value is dict;
        # remaining keyword arguments go to the base tracker.
        if value is not dict:
            raise AssertionError(f"DictBuiltinVariable value must be dict, got {value}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Fixed debug string.
        return "DictBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handle a call to dict: builds the dictionary through the custom-dict helper with the plain dict class.
        return DictBuiltinVariable.call_custom_dict(tx, dict, *args, **kwargs)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Unbound method call on dict. For the allocation method with args:
        # returns a fresh empty dict tracker when the class argument is dict itself, else
        # tracks a new user-defined object of that subclass. The fromkeys method uses the
        # fromkeys helper. Other known dict methods are forwarded to the first argument's
        # underlying dictionary tracker (user dict subclass base tracker, or dict tracker);
        # anything else defers to the parent implementation.
        if name == "__new__":
            if args:
                # dict.__new__ (tp_new) ignores extra args — only the first
                # arg (the type) matters.  Pass init_args=[] so reconstruction
                # emits base_cls.__new__(cls) without extras.
                # https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4735-L4768
                dict_vt = ConstDictVariable({}, mutation_type=ValueMutationNew())
                if isinstance(args[0], DictBuiltinVariable):
                    return dict_vt
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    [],
                    tx=tx,
                )

        if name == "fromkeys":
            return DictBuiltinVariable.call_custom_dict_fromkeys(
                tx, dict, *args, **kwargs
            )

        resolved_fn = getattr(dict, name, None)
        if resolved_fn is not None and resolved_fn in dict_methods:
            if isinstance(args[0], variables.UserDefinedDictVariable):
                if args[0]._base_vt is None:
                    raise AssertionError(
                        "UserDefinedDictVariable._base_vt must not be None for dict method dispatch"
                    )
                return args[0]._base_vt.call_method(tx, name, args[1:], kwargs)
            elif isinstance(args[0], ConstDictVariable):
                return args[0].call_method(tx, name, args[1:], kwargs)

        return super().call_method(tx, name, args, kwargs)

    @staticmethod
    def call_custom_dict(
        tx: "InstructionTranslatorBase",
        user_cls: type,
        /,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Build a dict-like object of class user_cls by inlining the
        # construct-dict polyfill with the class and the given positional and keyword
        # argument trackers; returns the resulting tracker.
        args_list = list(args)
        return tx.inline_user_function_return(
            VariableTracker.build(tx, polyfills.construct_dict),
            [VariableTracker.build(tx, user_cls), *args_list],
            kwargs,
        )

    @staticmethod
    def call_custom_dict_fromkeys(
        tx: "InstructionTranslatorBase",
        user_cls: type,
        /,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] Implement fromkeys for dict, OrderedDict or defaultdict (any other
        # class graph-breaks). Accepts keys plus an optional value (default None); the value
        # may be a keyword only for OrderedDict; bad argument counts raise an
        # argument-mismatch error. Builds the result of the right class holding the keys
        # (reusing existing keys of a set or dict tracker, otherwise iterating the argument,
        # which must be all hashable) mapped to the shared value; else graph-breaks "failed to
        # call dict.fromkeys()".
        # KNOWN ISSUE: the branch testing whether the argument is a raw dictionary can never
        # be taken because arguments are trackers, not real dictionaries.
        if user_cls not in {dict, OrderedDict, defaultdict}:
            unimplemented(
                gb_type="Unsupported dict type for fromkeys()",
                context=f"{user_cls.__name__}.fromkeys(): {args} {kwargs}",
                explanation=f"Failed to call {user_cls.__name__}.fromkeys() because "
                f"{user_cls.__name__} is not any type of dict, OrderedDict, or defaultdict",
                hints=[
                    f"Ensure {user_cls.__name__} is a type of dict, OrderedDict, or defaultdict.",
                ],
            )
        if kwargs:
            # Only `OrderedDict.fromkeys` accepts `value` passed by keyword
            if (
                user_cls is not OrderedDict
                or len(args) != 1
                or len(kwargs) != 1
                or "value" not in kwargs
            ):
                raise_args_mismatch(
                    tx,
                    f"{user_cls.__name__}.fromkeys",
                    "1 args and 1 kwargs (`value`)",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            args = (*args, kwargs.pop("value"))
        if len(args) == 0:
            raise_args_mismatch(
                tx,
                f"{user_cls.__name__}.fromkeys",
                "at least 1 args",
                f"{len(args)} args",
            )
        if len(args) == 1:
            args = (*args, ConstantVariable.create(None))
        if len(args) != 2:
            raise_args_mismatch(
                tx,
                f"{user_cls.__name__}.fromkeys",
                "2 args",
                f"{len(args)} args",
            )

        arg, value = args

        def _make_result(
            items: dict[VariableTracker, VariableTracker],
        ) -> VariableTracker:
            # [FUNCTION DOC] HELPER: Wrap an item mapping as a new mutable tracker of the
            # requested class: ordered dict tracker, default-dict tracker (tracked as a new
            # user-defined object with its base dictionary set), or plain dict tracker.
            if user_cls is OrderedDict:
                return OrderedDictVariable(items, mutation_type=ValueMutationNew())
            elif user_cls is defaultdict:
                from .builder import SourcelessBuilder
                from .user_defined import DefaultDictVariable

                result = tx.output.side_effects.track_new_user_defined_object(
                    SourcelessBuilder.create(tx, dict),
                    SourcelessBuilder.create(tx, defaultdict),
                    [],
                    tx=tx,
                )
                if not isinstance(result, DefaultDictVariable):
                    raise AssertionError(
                        f"Expected DefaultDictVariable, got {type(result)}"
                    )
                result._base_vt = ConstDictVariable(
                    items, mutation_type=ValueMutationNew()
                )
                return result
            else:
                return ConstDictVariable(items, mutation_type=ValueMutationNew())

        # Reuse the operand's existing HashableTracker keys instead of
        # re-wrapping (and thus re-hashing) the underlying VTs, mirroring
        # CPython's do-not-rehash-dict-keys behavior when building a dict from
        # an existing set/frozenset/dict.
        if isinstance(arg, (variables.SetVariable, ConstDictVariable)):
            # HashableTracker keys are accepted by ConstDictVariable.__init__.
            return _make_result(dict.fromkeys(arg.items.keys(), value))  # type: ignore[arg-type]
        if isinstance(arg, dict):
            arg_list = [VariableTracker.build(tx, k) for k in arg]
            return _make_result(dict.fromkeys(arg_list, value))
        elif iterator := generic_getiter(tx, arg):
            keys = unpack_iterable(tx, iterator)
            if all(is_hashable(v) for v in keys):
                return _make_result(dict.fromkeys(keys, value))

        unimplemented(
            gb_type="failed to call dict.fromkeys()",
            context=f"{user_cls.__name__}.fromkeys(): {args} {kwargs}",
            explanation=f"Failed to call {user_cls.__name__}.fromkeys() because "
            "arguments could not be automatically converted to a list, "
            "or some dict key is not hashable.",
            hints=[
                "Manually convert the argument to a list.",
                "Ensure all keys are hashable.",
            ],
        )


class IterBuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for the iter builtin. Calling it with one argument gets that
    # object's iterator; with two (callable and sentinel) it builds a callable iterator
    # through a polyfill.
    """Variable tracker for the `iter` builtin."""

    _fn = iter

    def __init__(self, value: Any = iter, **kwargs: Any) -> None:
        # [FUNCTION DOC] Create the tracker. Raises an assertion error unless value is iter;
        # remaining keyword arguments go to the base tracker.
        if value is not iter:
            raise AssertionError(f"IterBuiltinVariable value must be iter, got {value}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Fixed debug string.
        return "IterBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handle a call to iter. No arguments raises an observed TypeError.
        # One argument returns its iterator via the generic iteration protocol. Otherwise
        # inlines the callable-iterator polyfill with the arguments. Keyword arguments are
        # ignored in the one-argument case.
        # ref: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L1666-L1682

        if not args:
            raise_observed_exception(
                TypeError,
                tx,
                args=["iter expected at least 1 argument, got 0"],
            )

        if len(args) == 1:
            return generic_getiter(tx, args[0])
        else:
            return variables.UserFunctionVariable(
                polyfills.builtins.callable_iterator
            ).call_function(tx, args, kwargs)


class GetAttrBuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for the getattr builtin. Realizes lazy args, resolves the attribute
    # through the generic attribute protocol, and falls back to evaluating the real getattr
    # on constant arguments if tracing raises a graph break.
    """Variable tracker for the `getattr` builtin."""

    _fn = getattr

    def __init__(self, value: Any = getattr, **kwargs: Any) -> None:
        # [FUNCTION DOC] Create the tracker. Raises an assertion error unless value is
        # getattr; remaining keyword arguments go to the base tracker.
        if value is not getattr:
            raise AssertionError(
                f"GetAttrBuiltinVariable value must be getattr, got {value}"
            )
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Fixed debug string.
        return "GetAttrBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handle a call to getattr. Realizes lazy argument trackers, then
        # resolves through the internal getattr helper. If that graph-breaks and all
        # arguments are constants or unspecialized scalars, evaluates the real getattr on the
        # raw values instead: a missing attribute becomes an observed AttributeError, other
        # errors become observed exceptions of the same type (a constant-conversion failure
        # is re-raised unchanged), and success returns a tracker for the value. Otherwise
        # re-raises the graph break.
        from .lazy import LazyVariableTracker

        if any(isinstance(a, LazyVariableTracker) for a in args):
            args = [
                a.realize() if isinstance(a, LazyVariableTracker) else a for a in args
            ]
        try:
            return self._call_getattr(tx, args, kwargs)
        except Unsupported:
            # Replicate the constant-fold fallback from BuiltinVariable._make_handler:
            # if all args are python constants, evaluate getattr() directly rather
            # than propagating a graph break from tp_getattro_impl.
            if not check_unspec_or_constant_args(args, kwargs):
                raise
            try:
                result = getattr(*[a.as_python_constant() for a in args])
            except AttributeError:
                raise_observed_exception(AttributeError, tx)
                raise
            except AsPythonConstantNotImplementedError:
                raise
            except Exception as exc:
                raise_observed_exception(type(exc), tx, args=list(exc.args))
                raise
            return VariableTracker.build(tx, result)

    def _call_getattr(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Core of getattr. Reads object, name and optional default
        # from the arguments (indexing assumes at least two are present). The name must be a
        # compile-time constant, otherwise graph-breaks "getattr() with non-constant name
        # argument". Delegates to the generic attribute helper with the default sentinel when
        # no default was passed.
        obj = args[0]
        name_var = args[1]
        default = args[2] if len(args) > 2 else _NO_DEFAULT

        if not name_var.is_python_constant():
            unimplemented(
                gb_type="getattr() with non-constant name argument",
                context=f"getattr({obj}, {name_var}, {default})",
                explanation="getattr() with non-constant name argument is not supported",
                hints=["Ensure the name argument of getattr() is a string"],
            )

        name = name_var.as_python_constant()
        return generic_getattr(tx, obj, name, default)


class HasAttrBuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for the hasattr builtin. Validates arguments and asks the object
    # tracker whether it has the named attribute.
    """Variable tracker for the `hasattr` builtin."""

    _fn = hasattr

    def __init__(self, value: Any = hasattr, **kwargs: Any) -> None:
        # [FUNCTION DOC] Create the tracker. Raises an assertion error unless value is
        # hasattr; remaining keyword arguments go to the base tracker.
        if value is not hasattr:
            raise AssertionError(
                f"HasAttrBuiltinVariable value must be hasattr, got {value}"
            )
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Fixed debug string.
        return "HasAttrBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handle a call to hasattr. Realizes lazy arguments; requires exactly
        # two positional arguments with a constant attribute name, else raises an observed
        # TypeError. Returns the object tracker's attribute-presence answer; graph-breaks
        # "hasattr() on unsupported type" if the tracker cannot answer.
        from .lazy import LazyVariableTracker

        if any(isinstance(a, LazyVariableTracker) for a in args):
            args = [
                a.realize() if isinstance(a, LazyVariableTracker) else a for a in args
            ]
        if len(args) != 2 or kwargs:
            raise_observed_exception(TypeError, tx)
        obj, attr = args
        if not attr.is_python_constant():
            raise_observed_exception(TypeError, tx)
        result = obj.call_obj_hasattr(tx, attr.as_python_constant())
        if result is None:
            unimplemented(
                gb_type="hasattr() on unsupported type",
                context=f"hasattr({obj}, {attr})",
                explanation=f"hasattr() is not supported on type {obj.python_type_name()}",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        return result


class SetAttrBuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for the setattr builtin. Validates arguments and applies attribute
    # assignment by delegating to the object's setattr method or by recording attribute
    # mutations in the side-effect tracker (with special handling for tensor attributes and
    # module attributes).
    """Variable tracker for the `setattr` builtin."""

    _fn = setattr

    def __init__(self, value: Any = setattr, **kwargs: Any) -> None:
        # [FUNCTION DOC] Create the tracker. Raises an assertion error unless value is
        # setattr; remaining keyword arguments go to the base tracker.
        if value is not setattr:
            raise AssertionError(
                f"SetAttrBuiltinVariable value must be setattr, got {value}"
            )
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Fixed debug string.
        return "SetAttrBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handle a call to setattr. Realizes lazy arguments; requires exactly
        # three positional arguments (object, name, value), else raises an observed TypeError.
        # Returns the helper's result when it handled the assignment, otherwise graph-breaks
        # "setattr() on unsupported type". Side effects: may record attribute mutations.
        from .lazy import LazyVariableTracker

        if any(isinstance(a, LazyVariableTracker) for a in args):
            args = [
                a.realize() if isinstance(a, LazyVariableTracker) else a for a in args
            ]
        if len(args) != 3 or kwargs:
            raise_observed_exception(TypeError, tx)
        obj, name_var, val = args
        result = self._call_setattr(tx, obj, name_var, val)
        if result is not None:
            return result
        unimplemented(
            gb_type="setattr() on unsupported type",
            context=f"setattr({obj}, {name_var}, {val})",
            explanation=f"setattr() is not supported on type {obj.python_type_name()}",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def _call_setattr(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        name_var: VariableTracker,
        val: VariableTracker,
    ) -> VariableTracker | None:
        # [FUNCTION DOC] HELPER: Perform attribute assignment, returning the assigned value
        # tracker or None when unsupported. User-defined objects, dicts with defaults, nested
        # functions, exceptions, tracebacks and deques forward to their own setattr method.
        # A sourced object not yet tracked, with a real attribute dictionary, is registered
        # as an existing tracked object and the attribute store is recorded. For objects with
        # tracked attribute mutation and a constant name: tensors get special handling
        # (requires_grad assignment graph-breaks; data assignment requires a sourced tensor of
        # equal dtype and, across devices, equal shape/strides, then replaces the tracked
        # fake entry and records a shallow-copy-data graph call under no-grad; grad names
        # are unified; getset descriptor names graph-break), then the store is recorded.
        # Module trackers graph-break when inside a higher-order op, return the existing
        # attribute when assigning the identical tensor, else are converted to the
        # unspecialized form. Mutates side-effect and output-graph state.
        if isinstance(
            obj,
            (
                variables.DefaultDictVariable,
                variables.UserDefinedObjectVariable,
                variables.NestedUserFunctionVariable,
                variables.ExceptionVariable,
                variables.TracebackVariable,
                variables.DequeVariable,
            ),
        ):
            return obj.call_method(tx, "__setattr__", [name_var, val], {})
        elif (
            not tx.output.side_effects.is_attribute_mutation(obj)
            and obj.source is not None
            and (underlying := obj.get_value_for_setattr()) is not None
            and hasattr(underlying, "__dict__")
            and name_var.is_python_constant()
        ):
            tx.output.side_effects.track_object_existing(underlying, obj)
            name = name_var.as_python_constant()
            tx.output.side_effects.store_attr(obj, name, val)
            return val
        elif (
            tx.output.side_effects.is_attribute_mutation(obj)
            and name_var.is_python_constant()
        ):
            name = name_var.as_python_constant()
            if obj.is_tensor():
                from .builder import wrap_fx_proxy

                if name == "requires_grad":
                    # TODO(azahed98): Make it work properly
                    unimplemented(
                        gb_type="setattr() on Tensor.requires_grad",
                        context=f"setattr({obj}, {name}, {val})",
                        explanation="setattr() on Tensor.requires_grad not supported. "
                        "Mutating requires_grad can introduce a new leaf from non-leaf or vice versa in "
                        "the middle of the graph, which AOTAutograd does not currently know how to handle.",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
                elif name == "data":
                    # [Note: set_data_on_scoped_tensor]
                    if obj.source is None:
                        unimplemented(
                            gb_type="Failed to mutate tensor data attribute",
                            context=f"setattr({obj}, {name}, {val})",
                            explanation="Dynamo only supports mutating `.data`"
                            " of tensor created outside `torch.compile` region",
                            hints=[
                                "Don't mutate `.data` on this tensor, or move "
                                "the mutation out of `torch.compile` region",
                            ],
                        )
                    elif obj.dtype != val.dtype:  # type: ignore[attr-defined]
                        unimplemented(
                            gb_type="Failed to mutate tensor data attribute to different dtype",
                            context=f"setattr({obj}, {name}, {val})",
                            explanation="Dynamo only supports mutating `.data`"
                            " of tensor to a new one with the same dtype",
                            hints=[
                                "Don't mutate `.data` on this tensor, or move "
                                "the mutation out of `torch.compile` region",
                            ],
                        )
                    elif obj.device != val.device:  # type: ignore[attr-defined]
                        obj_fake = get_fake_value(obj.as_proxy().node, tx)
                        val_fake = get_fake_value(val.as_proxy().node, tx)
                        if (
                            obj_fake.dtype != val_fake.dtype
                            or obj_fake.shape != val_fake.shape
                            or obj_fake.stride() != val_fake.stride()
                        ):
                            unimplemented(
                                gb_type="Failed to mutate tensor data attribute across devices with different shape/strides",
                                context=f"setattr({obj}, {name}, {val})",
                                explanation="Dynamo only supports cross-device `.data`"
                                " mutation when shape and strides match",
                                hints=[
                                    "Don't mutate `.data` on this tensor, or move "
                                    "the mutation out of `torch.compile` region",
                                ],
                            )

                    # Remove the old reference in tracked fakes - if we don't
                    # do this, .data value size/shape differences cause
                    # tracked fakes to produce incorrect guards. Sound
                    # because the TensorVariable from shallow_copy_data_
                    # below is new and gets installed in tracked fakes.
                    to_remove = [
                        tf for tf in tx.output.tracked_fakes if tf.source == obj.source
                    ]
                    for tf in to_remove:
                        tx.output.tracked_fakes.remove(tf)

                    # Snapshot the placeholder before
                    # shallow_copy_data_ mutates it. Record the node
                    # and snapshot so compile_and_call_fx_graph can
                    # restore the correct metadata before passing the
                    # graph to the backend.
                    input_node = obj.as_proxy().node
                    if input_node.op == "placeholder":
                        ev = input_node.meta.get("example_value")
                        if ev is not None and hasattr(ev, "fake_mode"):
                            from torch._subclasses.fake_impls import fast_detach

                            snapshot = fast_detach(ev.fake_mode, ev)
                            tx.output._shallow_copy_placeholder_snapshots.setdefault(
                                input_node, snapshot
                            )

                    with dynamo_disable_grad(tx), torch.no_grad():
                        out = wrap_fx_proxy(
                            tx,
                            tx.output.create_proxy(
                                "call_function",
                                torch.ops.aten.shallow_copy_data_,
                                *proxy_args_kwargs([obj, val], {}),
                            ),
                        )

                    return out
                elif name in ("_grad", "grad"):
                    # NOTE: [Tensor "grad" and "_grad" attr]
                    # _grad and grad share the same setter/getter, see
                    # THPVariable_properties, and here we make sure setting one
                    # enables reading `val` from the other, by routing all
                    # read/write to `grad`.
                    name = "grad"
                elif is_tensor_getset_descriptor(name):
                    # Attribute like `torch.Tensor.real` has special setters we
                    # don't yet support; it's not as simple adding an entry to
                    # the side effect mapping.
                    unimplemented(
                        gb_type="Failed to set tensor attribute",
                        context=f"setattr({obj}, {name}, {val})",
                        explanation="Dynamo doesn't support setting these tensor attributes",
                        hints=[
                            f"Don't mutate attribute '{name}' on tensors, or "
                            "move the mutation out of `torch.compile` region",
                        ],
                    )

            tx.output.side_effects.store_attr(obj, name, val)
            return val
        elif isinstance(obj, variables.NNModuleVariable):
            if not tx.output.is_root_tracer():
                unimplemented(
                    gb_type="nn.Module mutation in HigherOrderOp",
                    context=f"nn.Module: {obj}",
                    explanation="Inplace modifying nn.Module params/buffers inside HigherOrderOps is not allowed.",
                    hints=[
                        "Remove the mutation or move it outside of the HigherOrderOp.",
                        *graph_break_hints.FUNDAMENTAL,
                    ],
                )
            if name_var.is_python_constant() and isinstance(
                val, variables.TensorVariable
            ):
                assigning_fake_val = get_fake_value(val.as_proxy().node, tx)

                try:
                    getattr_var = obj.tp_getattro_impl(
                        tx, name_var.as_python_constant()
                    )
                except (AttributeError, ObservedAttributeError):
                    getattr_var = None

                if getattr_var is not None and getattr_var.is_tensor():
                    # get_fake_val will get the same fake tensor
                    existing_fake_attr = get_fake_value(getattr_var.as_proxy().node, tx)

                    # same tensor identity, setattr is a no-op
                    mod_setattr = inspect.getattr_static(obj.module_type, "__setattr__")
                    if (
                        existing_fake_attr is assigning_fake_val
                        and mod_setattr is torch.nn.Module.__setattr__
                    ):
                        return getattr_var

            obj.convert_to_unspecialized(tx)
        return None


class ListBuiltinVariable(BaseBuiltinVariable):
    # [CLASS DOC] Tracker for the list type itself. Calling it builds a new mutable list
    # tracker by extending an empty list with the argument, installing guards on sourced
    # arguments; its allocation method creates list or user-subclass objects.
    """Variable tracker for the `list` builtin constructor."""

    _fn = list

    def __init__(self, value: type = list, **kwargs: Any) -> None:
        # [FUNCTION DOC] Create the tracker. Raises an assertion error unless value is list;
        # remaining keyword arguments go to the base tracker.
        if value is not list:
            raise AssertionError(f"ListBuiltinVariable value must be list, got {value}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Fixed debug string.
        return "ListBuiltinVariable()"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Handle a call to list. Keyword arguments or more than one positional
        # raise an observed type error. No argument gives a new empty list tracker. For a
        # sourced, non-constant argument installs a guard so later runs see the same
        # contents: tuple-iterator length, key order for dicts and ordered sets, mapping
        # keys for mapping proxies, or sequence length for most others (none for plain
        # iterators, generators and unspecialized modules). Returns a new list tracker
        # extended with the argument's items.
        # ref: https://github.com/python/cpython/blob/3.13/Objects/listobject.c#L1265-L1287
        if kwargs:
            raise_type_error(
                tx,
                "list() takes no keyword arguments",
            )
        if len(args) == 0:
            return ListVariable([], mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"list expected at most 1 argument, got {len(args)}",
            )

        obj = args[0]
        if obj.source and not is_constant_source(obj.source):
            if isinstance(obj, TupleIteratorVariable):
                install_guard(obj.source.make_guard(GuardBuilder.TUPLE_ITERATOR_LEN))
            elif not isinstance(
                obj,
                (variables.IteratorVariable, variables.LocalGeneratorObjectVariable),
            ):
                if isinstance(
                    obj,
                    (
                        ConstDictVariable,
                        variables.OrderedSetVariable,
                        variables.DictKeySetVariable,
                    ),
                ):
                    tx.output.guard_on_key_order.add(obj.source)
                if isinstance(obj, variables.MappingProxyVariable):
                    install_guard(
                        obj.source.make_guard(GuardBuilder.MAPPING_KEYS_CHECK)
                    )
                elif not isinstance(obj, variables.UnspecializedNNModuleVariable):
                    install_guard(obj.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))

        lst = ListVariable([], mutation_type=ValueMutationNew())
        lst.call_method(tx, "extend", [args[0]], {})
        return lst

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Unbound method call on list. The allocation method with a single
        # class argument returns a fresh empty list tracker when the class is list itself,
        # else tracks a new user-defined object of that subclass. Everything else defers to
        # the parent implementation.
        if name == "__new__":
            if len(args) == 1 and not kwargs:
                list_vt = ListVariable([], mutation_type=ValueMutationNew())
                if isinstance(args[0], ListBuiltinVariable):
                    return list_vt
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

        return super().call_method(tx, name, args, kwargs)


# pyrefly: ignore [deprecated]
@contextlib.contextmanager
def dynamo_disable_grad(tx: "InstructionTranslatorBase") -> typing.Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that turns off gradient mode in the traced
    # program for the duration of the with-block. On entry creates and enters a grad-mode
    # tracker set to disabled (recording it for the output graph); on exit, even after an
    # error, exits that tracker to restore the previous mode.
    from . import GradModeVariable

    gmv = GradModeVariable.create(tx, False)
    try:
        gmv.enter(tx)
        yield
    finally:
        gmv.exit(tx)
