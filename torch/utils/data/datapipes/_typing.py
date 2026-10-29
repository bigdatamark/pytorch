# mypy: allow-untyped-defs
# [MODULE DOC] Lightweight type system for datapipes: subtype checks between typing
# constructs (issubtype, issubinstance), the _DataPipeType wrapper that stores a datapipe's
# declared item type, the metaclasses _DataPipeMeta and _IterDataPipeMeta that attach the
# type, validate the declared return annotation of the iteration method, and wrap reset
# and iteration with hooks, plus the Integer and Boolean placeholders and reinforce_type.
# Relies on private names from the typing module. No import-time side effects besides
# defining the default type object.
# Taking reference from official Python typing
# https://github.com/python/cpython/blob/master/Lib/typing.py

import collections
import functools
import numbers
import sys

# Please check [Note: TypeMeta and TypeAlias]
# In case of metaclass conflict due to ABCMeta or _ProtocolMeta
# For Python 3.9, only Protocol in typing uses metaclass
from abc import ABCMeta
from collections.abc import Iterator

# TODO: Use TypeAlias when Python 3.6 is deprecated
from typing import (
    _eval_type,  # pyrefly: ignore [missing-module-attribute]
    _GenericAlias,  # pyrefly: ignore [missing-module-attribute]
    _tp_cache,  # pyrefly: ignore [missing-module-attribute]
    _type_check,  # pyrefly: ignore [missing-module-attribute]
    _type_repr,
    Any,
    ForwardRef,
    Generic,
    get_args,
    get_type_hints,
    TypeVar,
    Union,
)

from torch.utils.data.datapipes._hook_iterator import _SnapshotState, hook_iterator


class GenericMeta(ABCMeta):  # type: ignore[no-redef]
    # [CLASS DOC] HELPER: Empty metaclass alias of ABCMeta, the base for datapipe
    # metaclasses (kept separate to avoid metaclass conflicts with typing Protocol).
    pass


class Integer(numbers.Integral):
    # [CLASS DOC] Abstract stand-in for the Python int type in subtype checks (target of
    # int in TYPE2ABC); declares no behavior.
    pass


class Boolean(numbers.Integral):
    # [CLASS DOC] Abstract stand-in for the Python bool type in subtype checks (target of
    # bool in TYPE2ABC); declares no behavior.
    # KNOWN ISSUE: Integer and Boolean are unrelated classes, so bool is not treated as a
    # subtype of int here, unlike in Python itself.
    pass


# Python 'type' object is not subscriptable
# Tuple[int, List, dict] -> valid
# tuple[int, list, dict] -> invalid
# Map Python 'type' to abstract base class
TYPE2ABC = {
    bool: Boolean,
    int: Integer,
    float: numbers.Real,
    complex: numbers.Complex,
    dict: dict,
    list: list,
    set: set,
    tuple: tuple,
    None: type(None),
}


def issubtype(left, right, recursive=True):
    # [FUNCTION DOC] Returns True if type left is a subtype of type right. Builtin types
    # are first mapped to abstract classes. Any as right, equal types, or a bare Generic
    # as right are accepted; None as right rejects everything else; a TypeVar or Union is
    # expanded and every left variant must match some right constraint. If recursive is
    # False, generic arguments are not compared. Pure function.
    r"""
    Check if the left-side type is a subtype of the right-side type.

    If any of type is a composite type like `Union` and `TypeVar` with
    bounds, it would be expanded into a list of types and check all
    of left-side types are subtypes of either one from right-side types.
    """
    left = TYPE2ABC.get(left, left)
    right = TYPE2ABC.get(right, right)

    if right is Any or left == right:
        return True

    if isinstance(right, _GenericAlias):
        if getattr(right, "__origin__", None) is Generic:
            return True

    if right is type(None):
        return False

    # Right-side type
    constraints = _decompose_type(right)

    if len(constraints) == 0 or Any in constraints:
        return True

    if left is Any:
        return False

    # Left-side type
    variants = _decompose_type(left)

    # all() will return True for empty variants
    if len(variants) == 0:
        return False

    return all(
        _issubtype_with_constraints(variant, constraints, recursive)
        for variant in variants
    )


def _decompose_type(t, to_list=True):
    # [FUNCTION DOC] HELPER: Expands a TypeVar (its bound, else its constraints) or a Union
    # (its arguments) into a list of types with builtins mapped to abstract classes. For any
    # other type returns a one-element list, or None if to_list is False.
    # KNOWN ISSUE: only typing.Union is recognized; the pipe-style union type has a
    # different origin and would be treated as a single opaque type.
    if isinstance(t, TypeVar):
        if t.__bound__ is not None:
            ts = [t.__bound__]
        else:
            # For T_co, __constraints__ is ()
            ts = list(t.__constraints__)
    elif hasattr(t, "__origin__") and t.__origin__ == Union:
        ts = get_args(t)
    else:
        if not to_list:
            return None
        ts = [t]
    # Ignored: Generator has incompatible item type "object"; expected "Type[Any]"
    ts = [TYPE2ABC.get(_t, _t) for _t in ts]  # type: ignore[misc]
    return ts


def _issubtype_with_constraints(variant, constraints, recursive=True):
    # [FUNCTION DOC] HELPER: Returns True if variant matches any constraint. Nested
    # TypeVar/Union values on either side are expanded. Generic aliases match when origins
    # are equal and, if recursive, argument counts match and every argument pair is a
    # subtype (a constraint without arguments matches any arguments); plain types match by
    # equality of origin.
    r"""
    Check if the variant is a subtype of either one from constraints.

    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
    if variant in constraints:
        return True

    # [Note: Subtype for Union and TypeVar]
    # Python typing is able to flatten Union[Union[...]] or Union[TypeVar].
    # But it couldn't flatten the following scenarios:
    #   - Union[int, TypeVar[Union[...]]]
    #   - TypeVar[TypeVar[...]]
    # So, variant and each constraint may be a TypeVar or a Union.
    # In these cases, all of inner types from the variant are required to be
    # extracted and verified as a subtype of any constraint. And, all of
    # inner types from any constraint being a TypeVar or a Union are
    # also required to be extracted and verified if the variant belongs to
    # any of them.

    # Variant
    vs = _decompose_type(variant, to_list=False)

    # Variant is TypeVar or Union
    if vs is not None:
        return all(_issubtype_with_constraints(v, constraints, recursive) for v in vs)

    # Variant is not TypeVar or Union
    if hasattr(variant, "__origin__") and variant.__origin__ is not None:
        v_origin = variant.__origin__
        v_args = get_args(variant) or None
    else:
        v_origin = variant
        v_args = None

    # Constraints
    for constraint in constraints:
        cs = _decompose_type(constraint, to_list=False)

        # Constraint is TypeVar or Union
        if cs is not None:
            if _issubtype_with_constraints(variant, cs, recursive):
                return True
        # Constraint is not TypeVar or Union
        else:
            # __origin__ can be None for plain list, tuple, ... in Python 3.6
            if hasattr(constraint, "__origin__") and constraint.__origin__ is not None:
                c_origin = constraint.__origin__
                if v_origin == c_origin:
                    if not recursive:
                        return True
                    c_args = get_args(constraint) or None
                    if c_args is None or len(c_args) == 0:
                        return True
                    if (
                        v_args is not None
                        and len(v_args) == len(c_args)
                        and all(
                            issubtype(v_arg, c_arg)
                            for v_arg, c_arg in zip(v_args, c_args, strict=True)
                        )
                    ):
                        return True
            # Tuple[int] -> Tuple
            else:
                if v_origin == constraint:
                    return True

    return False


def issubinstance(data, data_type):
    # [FUNCTION DOC] Returns True if the runtime value data conforms to the type data_type.
    # First compares the shallow type of data; then, for tuples (arity must match), lists
    # and sets (all elements against the first argument) and dicts (keys and values), checks
    # contents recursively. Other values pass once the shallow check passes.
    # KNOWN ISSUE: a dict check unpacks exactly two type arguments, so a one-argument
    # annotation would raise an unpacking error.
    if not issubtype(type(data), data_type, recursive=False):
        return False

    dt_args = get_args(data_type) or None
    if isinstance(data, tuple):
        if dt_args is None or len(dt_args) == 0:
            return True
        if len(dt_args) != len(data):
            return False
        return all(issubinstance(d, t) for d, t in zip(data, dt_args, strict=True))
    elif isinstance(data, (list, set)):
        if dt_args is None or len(dt_args) == 0:
            return True
        t = dt_args[0]
        return all(issubinstance(d, t) for d in data)
    elif isinstance(data, dict):
        if dt_args is None or len(dt_args) == 0:
            return True
        kt, vt = dt_args
        return all(
            issubinstance(k, kt) and issubinstance(v, vt) for k, v in data.items()
        )

    return True


# [Note: TypeMeta and TypeAlias]
# In order to keep compatibility for Python 3.6, use Meta for the typing.
# TODO: When PyTorch drops the support for Python 3.6, it can be converted
# into the Alias system and using `__class_getitem__` for DataPipe. The
# typing system will gain benefit of performance and resolving metaclass
# conflicts as elaborated in https://www.python.org/dev/peps/pep-0560/


class _DataPipeType:
    # [CLASS DOC] HELPER: Immutable-style wrapper around a type annotation (param) describing
    # a datapipe's item type, with equality and hash based on param and subtype helpers.
    r"""Save type annotation in `param`."""

    def __init__(self, param) -> None:
        # [FUNCTION DOC] Stores the wrapped type annotation.
        self.param = param

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the typing-style text of the wrapped type.
        return _type_repr(self.param)

    def __eq__(self, other):
        # [FUNCTION DOC] Equal to another _DataPipeType with an equal param; otherwise
        # returns NotImplemented.
        if isinstance(other, _DataPipeType):
            return self.param == other.param
        return NotImplemented

    def __hash__(self):
        # [FUNCTION DOC] Hash of the wrapped type, consistent with equality.
        return hash(self.param)

    def issubtype(self, other):
        # [FUNCTION DOC] Returns True if this type is a subtype of other (a _DataPipeType or
        # a plain type). A bare Generic as other's param always matches. Raises TypeError
        # for any other kind of other.
        # KNOWN ISSUE: the Generic check reads other.param before the type check, so a
        # plain type (no param attribute) raises AttributeError instead of being compared.
        if isinstance(other.param, _GenericAlias):
            if getattr(other.param, "__origin__", None) is Generic:
                return True
        if isinstance(other, _DataPipeType):
            return issubtype(self.param, other.param)
        if isinstance(other, type):
            return issubtype(self.param, other)
        raise TypeError(f"Expected '_DataPipeType' or 'type', but found {type(other)}")

    def issubtype_of_instance(self, other):
        # [FUNCTION DOC] Returns True if the value other conforms to the wrapped type.
        return issubinstance(other, self.param)


# Default type for DataPipe without annotation
_T_co = TypeVar("_T_co", covariant=True)
# pyrefly: ignore [invalid-annotation]
_DEFAULT_TYPE = _DataPipeType(Generic[_T_co])


class _DataPipeMeta(GenericMeta):
    # [CLASS DOC] HELPER: Metaclass of DataPipe classes. Annotated with a class-level type
    # attribute holding the item type. Currently it only forwards class creation; the
    # parameterization helpers below (_getitem_, _eq_, _hash_) are defined but not wired up
    # as the special methods.
    r"""
    Metaclass for `DataPipe`.

    Add `type` attribute and `__init_subclass__` based on the type, and validate the return hint of `__iter__`.

    Note that there is subclass `_IterDataPipeMeta` specifically for `IterDataPipe`.
    """

    type: _DataPipeType

    def __new__(cls, name, bases, namespace, **kwargs):
        # [FUNCTION DOC] Creates the class by delegating to the parent metaclass and
        # returns it immediately.
        # KNOWN ISSUE: all statements after the return are unreachable, so the default type
        # and subclass hook are never installed here.
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        # TODO: the statements below are not reachable by design as there is a bug and typing is low priority for now.
        cls.__origin__ = None
        if "type" in namespace:
            return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        namespace["__type_class__"] = False
        #  For plain derived class without annotation
        for base in bases:
            if isinstance(base, _DataPipeMeta):
                return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

        namespace.update(
            {"type": _DEFAULT_TYPE, "__init_subclass__": _dp_init_subclass}
        )
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]

    def __init__(self, name, bases, namespace, **kwargs) -> None:
        # [FUNCTION DOC] Delegates initialization to the parent metaclass.
        super().__init__(name, bases, namespace, **kwargs)  # type: ignore[call-overload]

    # TODO: Fix isinstance bug
    @_tp_cache
    def _getitem_(self, params):
        # [FUNCTION DOC] HELPER: Implements subscripting a datapipe class with an item type
        # (results cached). Strings become forward references; a single type or a tuple is
        # accepted, each checked as a valid type; raises TypeError for None, for more than
        # one parameter, or when the type is not a subtype of the current declared type.
        # Returns self if the type is unchanged, otherwise a new subclass named with the
        # type and flagged as a type class. If the current type is a parameterized generic,
        # its parameters are substituted instead.
        if params is None:
            raise TypeError(f"{self.__name__}[t]: t can not be None")
        if isinstance(params, str):
            params = ForwardRef(params)
        if not isinstance(params, tuple):
            params = (params,)

        msg = f"{self.__name__}[t]: t must be a type"
        params = tuple(_type_check(p, msg) for p in params)

        if isinstance(self.type.param, _GenericAlias):
            orig = getattr(self.type.param, "__origin__", None)
            if isinstance(orig, type) and orig is not Generic:
                p = self.type.param[params]  # type: ignore[index]
                t = _DataPipeType(p)
                l = len(str(self.type)) + 2
                name = self.__name__[:-l]
                name = name + "[" + str(t) + "]"
                bases = (self,) + self.__bases__
                return self.__class__(
                    name,
                    bases,
                    {
                        "__init_subclass__": _dp_init_subclass,
                        "type": t,
                        "__type_class__": True,
                    },
                )

        if len(params) > 1:
            raise TypeError(
                f"Too many parameters for {self} actual {len(params)}, expected 1"
            )

        t = _DataPipeType(params[0])

        if not t.issubtype(self.type):
            raise TypeError(
                f"Can not subclass a DataPipe[{t}] from DataPipe[{self.type}]"
            )

        # Types are equal, fast path for inheritance
        if self.type == t:
            return self

        name = self.__name__ + "[" + str(t) + "]"
        bases = (self,) + self.__bases__

        return self.__class__(
            name,
            bases,
            {"__init_subclass__": _dp_init_subclass, "__type_class__": True, "type": t},
        )

    # TODO: Fix isinstance bug
    def _eq_(self, other):
        # [FUNCTION DOC] HELPER: Intended class equality: NotImplemented for non-datapipe
        # classes; identity if either has no origin; else equal origin and type.
        # KNOWN ISSUE: not installed as the equality method (trailing-underscore name), and
        # the origin attribute is only set in unreachable code, so it is effectively unused.
        if not isinstance(other, _DataPipeMeta):
            return NotImplemented
        if self.__origin__ is None or other.__origin__ is None:  # type: ignore[has-type]
            return self is other
        return (
            self.__origin__ == other.__origin__  # type: ignore[has-type]
            and self.type == other.type
        )

    # TODO: Fix isinstance bug
    def _hash_(self):
        # [FUNCTION DOC] HELPER: Intended class hash from name and item type; not installed
        # as the hash method.
        return hash((self.__name__, self.type))


class _IterDataPipeMeta(_DataPipeMeta):
    # [CLASS DOC] HELPER: Metaclass of IterDataPipe classes. On class creation it wraps a
    # defined reset method so it runs only in certain snapshot states, and wraps a defined
    # iteration method with the iterator hook.
    r"""
    Metaclass for `IterDataPipe` and inherits from `_DataPipeMeta`.

    Add various functions for behaviors specific to `IterDataPipe`.
    """

    def __new__(cls, name, bases, namespace, **kwargs):
        # [FUNCTION DOC] Modifies the class namespace before creation: replaces reset with a
        # conditional wrapper, and applies hook_iterator to the namespace if it defines the
        # iteration method. Then creates the class via the parent metaclass.
        if "reset" in namespace:
            reset_func = namespace["reset"]

            @functools.wraps(reset_func)
            def conditional_reset(*args, **kwargs) -> None:
                # [FUNCTION DOC] HELPER: Wrapper around the class's reset. Only if the
                # datapipe (first argument) is in the Iterating or NotStarted snapshot
                # state does it zero the yielded-sample counter, clear the fast-forward
                # iterator and call the real reset. Always sets the state to Iterating.
                r"""
                Only execute DataPipe's `reset()` method if `_SnapshotState` is `Iterating` or `NotStarted`.

                This allows recently restored DataPipe to preserve its restored state during the initial `__iter__` call.
                """
                datapipe = args[0]
                if datapipe._snapshot_state in (
                    _SnapshotState.Iterating,
                    _SnapshotState.NotStarted,
                ):
                    # Reset `NotStarted` is necessary because the `source_datapipe` of a DataPipe might have
                    # already begun iterating.
                    datapipe._number_of_samples_yielded = 0
                    datapipe._fast_forward_iterator = None
                    reset_func(*args, **kwargs)
                datapipe._snapshot_state = _SnapshotState.Iterating

            namespace["reset"] = conditional_reset

        if "__iter__" in namespace:
            hook_iterator(namespace)
        return super().__new__(cls, name, bases, namespace, **kwargs)  # type: ignore[call-overload]


def _dp_init_subclass(sub_cls, *args, **kwargs) -> None:
    # [FUNCTION DOC] HELPER: Subclass-creation hook for datapipe classes. Adds the
    # reinforce_type method to the class, skips internal type classes, resolves a string
    # item type via the defining module's globals (raises TypeError if unsupported), and
    # validates the iteration method's return annotation: it must be Iterator (bare or
    # parameterized) whose item type is a subtype of the declared type, else raises
    # TypeError. Side effect: mutates sub_cls.
    # Add function for datapipe instance to reinforce the type
    sub_cls.reinforce_type = reinforce_type

    # TODO:
    # - add global switch for type checking at compile-time

    # Ignore internal type class
    if getattr(sub_cls, "__type_class__", False):
        return

    # Check if the string type is valid
    if isinstance(sub_cls.type.param, ForwardRef):
        base_globals = sys.modules[sub_cls.__module__].__dict__
        try:
            param = _eval_type(sub_cls.type.param, base_globals, locals())
            sub_cls.type.param = param
        except TypeError as e:
            raise TypeError(
                f"{sub_cls.type.param.__forward_arg__} is not supported by Python typing"
            ) from e

    if "__iter__" in sub_cls.__dict__:
        iter_fn = sub_cls.__dict__["__iter__"]
        hints = get_type_hints(iter_fn)
        if "return" in hints:
            return_hint = hints["return"]
            # Plain Return Hint for Python 3.6
            if return_hint == Iterator:
                return
            if not (
                hasattr(return_hint, "__origin__")
                and (
                    return_hint.__origin__ == Iterator
                    or return_hint.__origin__ == collections.abc.Iterator
                )
            ):
                raise TypeError(
                    "Expected 'Iterator' as the return annotation for `__iter__` of {}"
                    ", but found {}".format(
                        sub_cls.__name__, _type_repr(hints["return"])
                    )
                )
            data_type = get_args(return_hint)[0]
            if not issubtype(data_type, sub_cls.type.param):
                raise TypeError(
                    f"Expected return type of '__iter__' as a subtype of {sub_cls.type},"
                    f" but found {_type_repr(data_type)} for {sub_cls.__name__}"
                )


def reinforce_type(self, expected_type):
    # [FUNCTION DOC] Narrows the item type of a datapipe instance. A tuple of types is
    # turned into a tuple type; the value must be a valid type and a subtype of the current
    # type, else raises TypeError. Sets the instance type attribute and returns the same
    # instance.
    r"""
    Reinforce the type for DataPipe instance.

    And the 'expected_type' is required to be a subtype of the original type
    hint to restrict the type requirement of DataPipe instance.
    """
    if isinstance(expected_type, tuple):
        expected_type = tuple[expected_type]  # type: ignore[valid-type]
    _type_check(expected_type, msg="'expected_type' must be a type")

    if not issubtype(expected_type, self.type.param):
        raise TypeError(
            f"Expected 'expected_type' as subtype of {self.type}, but found {_type_repr(expected_type)}"
        )

    self.type = _DataPipeType(expected_type)
    return self
