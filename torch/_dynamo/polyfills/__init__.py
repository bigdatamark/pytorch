# [PACKAGE DOC]
# Traceable pure-Python replacements (polyfills) for builtins and library
# functions that Dynamo cannot trace natively. This __init__ defines common
# helpers (comparison, dict/mapping operations, match-statement support,
# foreach and grad accumulation helpers, a no-enter torch function mode).
# Submodules (builtins, functools, itertools, ...) are loaded by loader.py and
# must not be imported here, to avoid circular imports.
"""
Python polyfills for common builtins.
"""

# NOTE: 1. Please do not import any submodule in the directory here to avoid circular imports.
#       2. While adding a new polyfill module, also add it to POLYFILLED_MODULE_NAMES in loader.py.
#          Add it in the TYPE_CHECKING block below as well.

import types
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from itertools import repeat as _repeat
from operator import eq, ge, gt, le, lt, ne
from typing import Any, TYPE_CHECKING, TypeGuard, TypeVar
from typing_extensions import TypeIs

import torch


T = TypeVar("T")
U = TypeVar("U")
C = TypeVar("C")


if TYPE_CHECKING:
    from ..utils import dict_keys

    # Load by torch._dynamo.polyfills.loader
    # See also the POLYFILLED_MODULE_NAMES in torch/_dynamo/polyfills/loader.py
    # Put the submodules here to avoid circular imports
    from . import (
        _collections as _collections,
        builtins as builtins,
        functools as functools,
        io as io,
        itertools as itertools,
        operator as operator,
        os as os,
        pytree as pytree,
        struct as struct,
        sys as sys,
        torch_c_nn as torch_c_nn,
        traceback as traceback,
    )

from torch.overrides import BaseTorchFunctionMode


# These classes handle support for TorchFunctionModes across
# graph breaks
# Today the TorchFunctionMode enter (for the classes we support)
# simply pushes the mode onto the stack. Since after this occurs
# the stack is mutated, and we replay these mutations, we don't need
# any cleanup logic to be run once the graph break occurs, we simply replay
# these mutations to ensure at the graph break the torch function mode stack is correct
#  and reconstruct the torch function mode stack normally
# when we compile the resume function on the other side of the break.
# However, to ensure we exit properly
# in the resume function, we need to re-enter the contexts as we do other contexts.
# These contexts do nothing on enter, but provide the correct exit logic to ensure
# the stack state is correct.
class NoEnterTorchFunctionMode(BaseTorchFunctionMode):
    # [CLASS DOC] A torch function mode whose enter does nothing (it is already
    # on the mode stack through replayed mutations) but which keeps the normal
    # exit logic, so resume functions after a graph break restore the stack correctly.
    def __enter__(self) -> None:
        # [FUNCTION DOC] No-op enter; does not push the mode.
        pass


# Used by WrappedUserFunctionVariable and similar to inline decorated function
# calls with bytecode backing. Without this, the context enter/exit happens in
# Python-level VT code, so a nested graph break inside `fn` would skip applying
# the context in the compiled fn/resume. By inlining through this polyfill, the
# `with` statement has real bytecode that the resume function can continue from.
def _fn_with_ctx(ctx: Any, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    # [FUNCTION DOC] HELPER: Call fn with the given args inside a with block on
    # ctx and return its result, so the context has real bytecode that resume
    # functions can continue from after nested graph breaks.
    with ctx:
        return fn(*args, **kwargs)


def index(
    iterator: Iterator[T], item: T, start: int = 0, end: int | None = None
) -> int:
    # [FUNCTION DOC] Return the position of the first element of the iterator
    # (within start..end) that is identical or equal to item; consumes the
    # iterator. Raises ValueError if not found.
    from itertools import islice

    for i, elem in islice(enumerate(iterator), start, end):
        if elem is item or elem == item:
            return i
    # This will not run in dynamo
    raise ValueError(f"{item} is not in {type(iterator)}")


def repeat(item: T, count: int) -> Iterator[T]:
    # [FUNCTION DOC] Generator yielding item count times.
    for _ in range(count):
        yield item


def radians(x: float) -> float:
    # [FUNCTION DOC] Convert degrees to radians.
    import math

    return math.pi / 180.0 * x


def impl_IS_MAPPING(a: object) -> TypeIs[Mapping[Any, Any]]:
    # [FUNCTION DOC] Mapping check used for the match-statement mapping opcode: True if a is a Mapping.
    return isinstance(a, Mapping)


def impl_MATCH_SEQUENCE(a: object) -> TypeGuard[Sequence[Any]]:
    # [FUNCTION DOC] Match-statement sequence check: True for Sequences other than str, bytes, bytearray.
    return isinstance(a, Sequence) and not isinstance(a, (str, bytes, bytearray))


def _match_class_attr(obj: object, name: str, seen: set[str]) -> object:
    # [FUNCTION DOC] HELPER: Fetch attribute name from obj for class patterns,
    # recording it in the seen set. Raises TypeError if name was already seen;
    # propagates AttributeError if missing. Mutates seen.
    if name in seen:
        raise TypeError(f"{type(obj)} got multiple sub-patterns for attribute {name}")

    attr = getattr(obj, name)
    seen.add(name)
    return attr


def impl_MATCH_CLASS(
    subject: object, cls: type, nargs: int, kwargs: tuple[str, ...]
) -> tuple[object, ...] | None:
    # [FUNCTION DOC] Match-statement class pattern. Returns None if subject is
    # not an instance of cls; otherwise a tuple of the matched attribute values:
    # nargs positional ones via __match_args__ (or the subject itself when the
    # class defines none, assuming match-self), then the keyword-named ones.
    # Raises TypeError for non-class cls, bad __match_args__, duplicate
    # attributes, or too many positional sub-patterns.
    if not isinstance(cls, type):
        raise TypeError("called match pattern must be a class")

    if not isinstance(subject, cls):
        return None

    typ = type(subject)
    match_self = False
    match_args = ()

    attrs = []
    seen = set()

    if nargs:
        if hasattr(typ, "__match_args__"):
            match_args = typ.__match_args__

            if not isinstance(match_args, tuple):
                raise TypeError(
                    f"{typ}.__match_args__ must be a tuple, (got {type(match_args)})"
                )

            for name in match_args[:nargs]:
                if not isinstance(name, str):
                    raise TypeError(
                        f"__match_args__ elements must be strings (got {type(name)})"
                    )
                attrs.append(_match_class_attr(subject, name, seen))
        else:
            # We should somehow check if the type has TPFLAGS_MATCH_SELF set
            # match_self is only true if TPFLAGS_MATCH_SELF is set, but there is
            # no way to check for it directly in Python. So we assume it is set
            # if there are no __match_args__
            match_self = True
            attrs.append(subject)

        allowed = 1 if match_self else len(match_args)
        if allowed < nargs:
            raise TypeError(
                f"accepts {allowed} positional sub-patterns ({nargs} given)"
            )

    for name in kwargs:
        attrs.append(_match_class_attr(subject, name, seen))

    return tuple(attrs)


def impl_MATCH_KEYS(obj: Mapping[T, U], keys: tuple[T, ...]) -> tuple[U, ...] | None:
    # [FUNCTION DOC] Match-statement mapping keys: return a tuple of values for
    # the keys if all are present, else None. Raises AssertionError if obj is not a Mapping.
    if not isinstance(obj, Mapping):
        raise AssertionError(f"Expected a Mapping, got {type(obj)}")
    if all(key in obj for key in keys):
        return tuple(obj[key] for key in keys)
    else:
        return None


def impl_CONTAINS_OP_fallback(a: T, b: Iterable[T]) -> bool:
    # [FUNCTION DOC] Containment test fallback: iterate b if it supports
    # iteration, else probe item access with increasing integer indices until
    # IndexError. Returns True on first equal element. Raises TypeError if b
    # supports neither.
    # performs fallback "a in b"
    # CPython: PySequence_Contains → _PySequence_IterSearch → PyObject_GetIter
    # PyObject_GetIter itself falls back to PySequence_GetItem when tp_iter is NULL.
    if hasattr(b, "__iter__"):
        for x in b:
            if x == a:
                return True
        return False
    if hasattr(b, "__getitem__"):
        i = 0
        while True:
            try:
                if b.__getitem__(i) == a:
                    return True
                i += 1
            except IndexError:
                return False
    raise TypeError(f"argument of type {type(b)} is not iterable")


def accumulate_grad(
    x: torch.Tensor,
    variable_grad: torch.Tensor | None,
    new_grad: torch.Tensor | None,
) -> torch.Tensor | None:
    # [FUNCTION DOC] Accumulate a new gradient into an existing one following
    # the gradient layout contract. None new grad returns the existing grad. The
    # new grad is copied into a tensor shaped like x; with no existing grad that
    # copy is returned; sparse existing plus dense new gives an out-of-place
    # sum; with grad mode on, an out-of-place sum; otherwise the existing grad
    # is updated in place and returned (may alias it).
    # polyfills according to the Gradient Layout Contract
    if new_grad is None:
        return variable_grad
    new_grad_strided = torch.empty_like(x)
    new_grad_strided.copy_(new_grad)
    if variable_grad is None:
        return new_grad_strided
    elif variable_grad.is_sparse and not new_grad_strided.is_sparse:
        return new_grad_strided + variable_grad
    elif torch.is_grad_enabled():
        return variable_grad + new_grad_strided
    else:
        variable_grad.add_(new_grad_strided)
        return variable_grad


def accumulate_grad_no_alias(
    x: torch.Tensor,
    variable_grad: torch.Tensor | None,
    new_grad: torch.Tensor | None,
) -> torch.Tensor | None:
    # [FUNCTION DOC] Like accumulate_grad but the result never aliases an input:
    # a None new grad returns a clone of the existing grad (or None), and the
    # no-grad-mode case sums into a clone instead of updating in place.
    # Mirrors inductor::accumulate_grad_: same gradient layout logic as
    # accumulate_grad, but the returned grad must not alias any input.
    if new_grad is None:
        if variable_grad is None:
            return None
        return variable_grad.clone()
    new_grad_strided = torch.empty_like(x)
    new_grad_strided.copy_(new_grad)
    if variable_grad is None:
        return new_grad_strided
    elif variable_grad.is_sparse and not new_grad_strided.is_sparse:
        return new_grad_strided + variable_grad
    elif torch.is_grad_enabled():
        return variable_grad + new_grad_strided
    else:
        result = variable_grad.clone()
        result.add_(new_grad_strided)
        return result


# This mirrors
# https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/listobject.c#L3352-L3413
def list_cmp(
    op: Callable[[Any, Any], bool], left: Sequence[T], right: Sequence[T]
) -> bool:
    # [FUNCTION DOC] Lexicographic comparison of two sequences with a given
    # comparison operator, as CPython does for lists/tuples. For equality and
    # inequality, differing lengths decide immediately. Otherwise the operator
    # is applied to the first differing pair, else to the lengths. Returns a bool.
    """emulate `(1,2,3) > (1,2)` etc"""

    # Optimization: For equality, short-circuit if lengths differ
    # This avoids iterating through elements and triggering guards on SymInts
    left_len = len(left)
    right_len = len(right)

    if op is eq and left_len != right_len:
        return False
    if op is ne and left_len != right_len:
        return True

    # Apply `op` to the first pair that differ
    for a, b in zip(left, right):
        if a != b:
            return op(a, b)

    # No more pairs to compare, so compare sizes.
    return op(left_len, right_len)


def dict___eq__(d: dict[T, U], other: dict[T, U]) -> bool:
    # [FUNCTION DOC] Dict equality: same length and keys, then values compared
    # (identity shortcut first, as in CPython). Two OrderedDicts also require
    # the same item order. Returns a bool.
    if (len(d) != len(other)) or (d.keys() != other.keys()):
        return False

    if all(isinstance(a, OrderedDict) for a in (d, other)):
        return list(d.items()) == list(other.items())

    # CPython's dict_equal uses PyObject_RichCompareBool for value
    # comparison, which has an identity shortcut (if v is w, eq is True).
    # This matters for NaN: {k: nan} == {k: nan} is True when same nan.
    for k, v in d.items():
        ov = other[k]
        if v is not ov and v != ov:
            return False

    return True


def dictview_richcompare(
    op: Callable[[Any, Any], bool], self: Iterable[T], other: Iterable[T]
) -> bool:
    # [FUNCTION DOC] Rich comparison of a dict view against another set-like
    # iterable using the given operator (eq, ne, lt, le, gt, ge), via length
    # checks and membership tests (subset/superset semantics). Returns a bool.
    """Mirrors dictview_richcompare for dict_keys/dict_items vs set/frozenset.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L5952-L6010
    Uses len() and ``in`` so that Dynamo traces through sq_length/sq_contains.
    """
    len_self = len(self)  # type: ignore[arg-type]
    len_other = len(other)  # type: ignore[arg-type]

    if op is eq:
        if len_self != len_other:
            return False
        return all(item in other for item in self)
    if op is ne:
        if len_self != len_other:
            return True
        return not all(item in other for item in self)
    if op is lt:
        if len_self >= len_other:
            return False
        return all(item in other for item in self)
    if op is le:
        if len_self > len_other:
            return False
        return all(item in other for item in self)
    if op is gt:
        if len_self <= len_other:
            return False
        return all(item in self for item in other)
    # ge
    if len_self < len_other:
        return False
    return all(item in self for item in other)


def assert_dict_equal(
    self_: Any, d1: dict[T, U], d2: dict[T, U], msg: str | None = None
) -> None:
    # [FUNCTION DOC] Replacement for the unittest dict-equality assertion: asserts d1 equals d2 via the test case's truth assertion.
    self_.assertTrue(d1 == d2, msg)


def assert_multi_line_equal(
    self_: Any, first: T, second: T, msg: str | None = None
) -> None:
    # [FUNCTION DOC] Replacement for the unittest multi-line string equality assertion: simple equality check without diffing.
    return self_.assertTrue(first == second, msg)


# The original impl. uses difflib
def assert_sequence_equal(
    self_: Any,
    seq1: Sequence[T],
    seq2: Sequence[T],
    msg: str | None = None,
    seq_type: type[Any] | None = None,
) -> None:
    # [FUNCTION DOC] Replacement for the unittest sequence-equality assertion: simple equality check; seq_type is ignored.
    return self_.assertTrue(seq1 == seq2, msg)


def getattr_and_trace(*args: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC] Look up the attribute named by the second positional arg on
    # the first positional arg and call it with the remaining args and keywords; returns its result.
    wrapper_obj = args[0]
    attr_name = args[1]
    fn = getattr(wrapper_obj, attr_name)
    return fn(*args[2:], **kwargs)


def getattr_and_trace_no_nested_graph_breaks(*args: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC] Same as getattr_and_trace but run with nested graph breaks disabled.
    with torch._dynamo.disable_nested_graph_breaks():
        return getattr_and_trace(*args, **kwargs)


def mapping_get(obj: Mapping[T, U], key: T, value: U | None = None, /) -> U | None:
    # [FUNCTION DOC] Mapping get: return the item for key, or value (default
    # None) when item access raises KeyError.
    try:
        return obj.__getitem__(key)
    except KeyError:
        return value


def instantiate_user_defined_class_object(
    cls: type[T], /, *args: Any, **kwargs: Any
) -> T:
    # [FUNCTION DOC] Instantiate a user-defined class: call its new with the
    # args, then call init only if the result's type is a true subclass of cls
    # (without consulting metaclass instance checks). Returns the object.
    obj = cls.__new__(cls, *args, **kwargs)

    # Only call __init__ if the object's type is a subclass of cls.
    # CPython uses PyType_IsSubtype(Py_TYPE(obj), type) at the C level, which does NOT
    # go through metaclass __instancecheck__. Using isinstance() here would be wrong
    # for classes with custom __instancecheck__ (e.g. torch.ByteStorage).
    # Reference: https://github.com/python/cpython/blob/3.12/Objects/typeobject.c#L1670-L1673
    if issubclass(type(obj), cls):
        obj.__init__(*args, **kwargs)
    return obj


def mutable_mapping_update(
    self,
    data: Mapping[T, U] | Iterable[tuple[T, U]] = (),
    /,
    **kwargs: Any,
) -> None:
    # [FUNCTION DOC] Mutable-mapping update: copy items from a mapping, or from
    # an iterable of key/value pairs, into self, then apply keyword items.
    # Raises TypeError if data is neither mapping nor iterable. Mutates self.
    # Objects with only a keys method are intentionally unsupported.
    if isinstance(data, Mapping):
        # Merge standard mapping with PyMapping_Items
        for key, value in data.items():
            self[key] = value
    # FIXME: Enabling the `elif`-branch below needs too many `VariableClass.call_obj_hasattr` changes.
    #   >>> class Foo:
    #   ...     def __init__(self):
    #   ...         self.keys = lambda: ['a', 'b', 'c']  # not required to be a method
    #   ...
    #   ...     def __getitem__(self, key):
    #   ...         return 0
    #   ...
    #   >>> dict(Foo())
    #   {'a': 0, 'b': 0, 'c': 0}
    #
    # > This is a rare case, so we comment it out for now.
    #
    # elif hasattr(data, "keys"):
    #     # Merge mapping-like object with PyMapping_Keys + PyObject_GetItem
    #     for key in data.keys():
    #         self[key] = data[key]
    else:
        if not isinstance(data, Iterable):
            raise TypeError(f"{type(data).__name__!r} object is not iterable")
        # Likely a sequence of pairs
        for key, value in data:
            self[key] = value

    if kwargs:
        for key, value in kwargs.items():
            self[key] = value


# Used with something like dict(obj)
def construct_dict(
    cls: type[T],
    data: Mapping[object, object] | Iterable[tuple[object, object]] = (),
    /,
    **kwargs: Any,
) -> T:
    # [FUNCTION DOC] Build a dict-like instance of cls (no init call) filled
    # from data and keyword items via mutable_mapping_update; returns it.
    self = cls.__new__(cls)
    mutable_mapping_update(self, data, **kwargs)
    return self


def foreach_map_fn(*args: Any) -> Any:
    # [FUNCTION DOC] Apply an op (first arg) elementwise over the remaining
    # args; non-list/tuple args are broadcast by repetition. If none are
    # lists, the op is applied once to the args and its result returned;
    # otherwise returns a list of per-element results (zip truncates to the shortest).
    op = args[0]
    new_args: list[Any] = []
    at_least_one_list = False
    for arg in args[1:]:
        if not isinstance(arg, (list, tuple)):
            new_args.append(_repeat(arg))
        else:
            at_least_one_list = True
            new_args.append(arg)

    # Just apply op once to args if there are no lists
    if not at_least_one_list:
        return op(*args[1:])

    out = []
    for unpacked in zip(*new_args):
        out.append(op(*unpacked))

    return out


def foreach_lerp_inplace(
    self,
    end: list[torch.Tensor] | tuple[torch.Tensor, ...],
    weight: float | int | torch.Tensor,
) -> None:
    # [FUNCTION DOC] In-place foreach lerp of the tensor list self toward end by
    # weight (scalar or tensor), decomposed into sub and addcmul to match native
    # CUDA rounding (low formula when |weight| < 0.5, else the high formula from
    # end). Mutates the tensors in self and returns self.
    # KNOWN ISSUE: annotated as returning None but returns self.
    # Decompose lerp via addcmul_ for FMA.  Uses the same dual-formula
    # approach as CUDA's native lerp to get bitwise identical results:
    #   |w| <  0.5  (low):  fma(w, diff, start)
    #   |w| >= 0.5  (high): fma(-(1-w), diff, end)
    # For tensor weights (e.g. 0-dim tensor from tensor betas in Adam) the
    # low formula is always used because the native lerp_scalar lowering
    # would crash on float(weight) for symbolic expressions.
    diff = torch._foreach_sub(end, self)
    if isinstance(weight, torch.Tensor):
        # Select base and weight for the dual formula before a single addcmul:
        #   low  (|w| <  0.5): fma(w,      diff, self)
        #   high (|w| >= 0.5): fma(-(1-w), diff, end)
        mask = weight.abs() >= 0.5
        neg_omw = -(1.0 - weight)
        w = torch.where(mask, neg_omw, weight)
        bases = [torch.where(mask, e, s) for s, e in zip(self, end)]
        w_list = [w] * len(diff)
        torch._foreach_addcmul_(bases, w_list, diff)
        for s, b in zip(self, bases):
            s.copy_(b)
    else:
        abs_weight = weight if weight >= 0 else -weight
        if abs_weight >= 0.5:
            # High formula: end + (-(1-w)) * diff  →  fma(-(1-w), diff, end)
            # Compute 1-w in target dtype to match CUDA rounding.
            d0 = self[0]
            neg_omw = -(1.0 - torch.tensor(weight, dtype=d0.dtype, device=d0.device))
            neg_omw_list = [neg_omw] * len(diff)
            for s, e in zip(self, end):
                s.copy_(e)
            torch._foreach_addcmul_(self, neg_omw_list, diff)
        else:
            # Low formula: start + w * diff  →  fma(w, diff, start)
            weights = [torch.full_like(d, weight) for d in diff]
            torch._foreach_addcmul_(self, weights, diff)
    return self


def foreach_pow_scalar(
    scalar: Any, exps: Sequence[bool | complex | float | int]
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Foreach power with a scalar base: raise scalar to each exponent, returning the foreach result.
    return torch._foreach_pow([scalar for _ in exps], exps)


def predicate(obj: object) -> bool:
    # [FUNCTION DOC] Truthiness of obj as a bool, written as a conditional so Dynamo handles the branch itself.
    # This will cause the rest of dynamo to handle the if statement correctly, so we don't have to rewrite it here.
    # We can't just use bool() here since we can't trace into that in general.
    if obj:
        return True
    return False


def group_tensors_by_device_and_dtype(
    tensorlistlist: list[list[torch.Tensor | None]], with_indices: bool = False
) -> dict[tuple[torch.device, torch.dtype], tuple[list[list[Any]], list[int]]]:
    # [FUNCTION DOC] Group tensors by (device, dtype) of the first non-None
    # tensor at each index across the lists. Returns a mapping from that key to
    # a pair: per-list grouped tensors (None where a list lacks the index) and,
    # if with_indices, the original indices (else an empty list). Empty input
    # gives an empty mapping.
    """Pure Python implementation of torch._C._group_tensors_by_device_and_dtype.

    Groups tensors by their device and dtype. This is useful before sending
    tensors off to a foreach implementation, which requires tensors to be on
    one device and dtype.

    Args:
        tensorlistlist: A list of lists of tensors (tensors can be None).
        with_indices: If True, track original indices in the output.

    Returns:
        A dict mapping (device, dtype) tuples to (grouped_tensorlistlist, indices).
    """
    # Result dict: (device, dtype) -> (list of lists, indices)
    result: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Any]], list[int]]
    ] = {}

    if not tensorlistlist or not tensorlistlist[0]:
        return result

    num_lists = len(tensorlistlist)
    num_tensors = len(tensorlistlist[0])

    for idx in range(num_tensors):
        # Find the first non-None tensor at this index to get device and dtype
        first_tensor = None
        for tlist in tensorlistlist:
            if tlist is not None and idx < len(tlist) and tlist[idx] is not None:
                first_tensor = tlist[idx]
                break

        if first_tensor is None:
            # All tensors at this index are None, skip
            continue

        key = (first_tensor.device, first_tensor.dtype)

        if key not in result:
            # Initialize empty lists for each tensorlist
            result[key] = ([[] for _ in range(num_lists)], [])

        grouped_lists, indices = result[key]

        # Add tensors from each list at this index
        for list_idx, tlist in enumerate(tensorlistlist):
            if tlist is not None and idx < len(tlist):
                grouped_lists[list_idx].append(tlist[idx])
            else:
                grouped_lists[list_idx].append(None)

        if with_indices:
            indices.append(idx)

    return result
