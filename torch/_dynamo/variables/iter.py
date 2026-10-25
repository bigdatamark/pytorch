# [MODULE DOC] Iterator-related VariableTrackers for Dynamo. Defines trackers for
# itertools callables (ItertoolsVariable, which eagerly or lazily builds iterator
# trackers when called) and iterator objects: chain, repeat, count, zip, zip_longest,
# map, filter and dict/set view iterators. Each iterator tracker implements the
# next-step protocol on symbolic values, records mutation through the side-effects
# tracker, and can reconstruct itself as real Python bytecode. Also defines the
# iterator size limit constant and an exhaustion-probe helper. Import-time side
# effects: captures the chain.from_iterable method object and builds the repeat
# call signature used for argument binding.
"""
This module provides iterator-related variable tracking functionality for Dynamo.
It implements variable classes for handling Python iterators and itertools functions
during symbolic execution and tracing.

The module includes:
- Base iterator variable classes for tracking iterator state
- Implementations of built-in iterators (zip, map, filter)
- Support for itertools functions (product, groupby, count, etc.)
- Mutation tracking and reconstruction capabilities for iterator operations

These classes integrate with Dynamo's variable tracking system to enable proper
handling of iterator operations during code transformation and optimization.
"""

import inspect
import itertools
import operator
import sys
from typing import Any, TYPE_CHECKING

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_call_function,
    create_call_function_ex,
    create_instruction,
)
from ..exc import (
    handle_observed_exception,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    raise_value_error,
    unimplemented,
)
from ..utils import raise_args_mismatch, tracked_repr, unpack_iterable
from .base import GetSet, Method, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .hashable import HashableTracker
from .object_protocol import generic_getiter, pyiter_next


# chain.from_iterable is a method descriptor that creates a new object on each
# attribute access (a is b → False). Capture once at import time for stable
# identity comparisons in ItertoolsVariable.call_function.
_CHAIN_FROM_ITERABLE = itertools.chain.from_iterable

# repeat(object, times=-1); itertools.repeat has no introspectable signature, so
# declare one to bind args/kwargs. `times` defaults to None to mark the unbounded
# form (CPython's cnt == -1).
_REPEAT_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter("object", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "times", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
    ]
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


MAX_ITERATOR_LIMIT = 100 * 1024  # 100k


def is_iterator_exhausted(
    tx: "InstructionTranslatorBase", iterator: VariableTracker
) -> bool:
    # [FUNCTION DOC] HELPER: Probes an iterator tracker by advancing it once. Returns
    # False if an item was produced (that item is consumed and discarded), or True
    # if it raised the observed stop-iteration (which is cleared from the translator).
    try:
        pyiter_next(tx, iterator)
        return False
    except ObservedUserStopIteration:
        handle_observed_exception(tx)
        return True


class ItertoolsVariable(VariableTracker):
    # [CLASS DOC] Tracker for an itertools callable (chain, zip_longest, product,
    # groupby, repeat, count, chain.from_iterable). State: value, the real callable.
    # Calling it produces iterator trackers or, for product and groupby, eagerly
    # materialized list iterators; unsupported callables fall back to generic calls.
    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the wrapped itertools callable in the value attribute.
        super().__init__(**kwargs)
        self.value = value

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich comparison: delegates to the constant-based comparison,
        # treating the wrapped callable as a python constant. Returns a tracker.
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string naming the wrapped callable.
        return f"ItertoolsVariable({self.value})"

    def as_python_constant(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped itertools callable as the python constant.
        return self.value

    def get_real_python_backed_value(self) -> Any:
        # [FUNCTION DOC] Returns the wrapped callable as the real python-backed value.
        return self.value

    def _get_from_iterable(
        self, tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        # [FUNCTION DOC] HELPER: Getter for the from_iterable attribute. Returns a tracker
        # for the saved chain.from_iterable method when the wrapped callable is chain,
        # otherwise None so generic attribute handling proceeds.
        # Only itertools.chain has from_iterable; declining (None) falls
        # through to the generic protocol for other itertools callables.
        if self.value is itertools.chain:
            return ItertoolsVariable(_CHAIN_FROM_ITERABLE)
        return None

    tp_getset = {
        "from_iterable": GetSet(_get_from_iterable),
    }

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list["VariableTracker"],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # [FUNCTION DOC] Symbolically calls the itertools callable. chain and
        # chain.from_iterable build a lazy ChainVariable; zip_longest builds a
        # ZipLongestVariable (optional fillvalue kwarg); product eagerly unpacks all
        # inputs and returns a list iterator of tuples (optional repeat kwarg);
        # groupby eagerly groups by constant keys (optional key function) into a list
        # iterator of key and sub-iterator pairs; repeat binds object and optional times
        # (negative clamped to 0, non-constant count uses a polyfill) and returns a
        # RepeatIteratorVariable; count returns a CountIteratorVariable. Unsupported
        # kwargs, key types or argument counts cause a graph break; a type error is
        # raised for a non-integer repeat count; other callables use the generic path.
        # See also: module `torch._dynamo.polyfills.itertools`

        if self.value is itertools.chain and not kwargs:
            # Wrap args in a ListIteratorVariable so sub-iterables are pulled lazily.
            # generic_getiter on each sub-iterable is deferred to tp_iternext_impl,
            # matching CPython's behavior (iter() on each arg is lazy, not at construction).
            source = variables.ListIteratorVariable(
                list(args), mutation_type=ValueMutationNew()
            )
            return ChainVariable(source, mutation_type=ValueMutationNew())
        elif self.value is _CHAIN_FROM_ITERABLE and not kwargs and len(args) == 1:
            # Convert outer iterable to iterator; each sub-iterable converted lazily.
            source = generic_getiter(tx, args[0])
            return ChainVariable(source, mutation_type=ValueMutationNew())
        elif self.value is itertools.zip_longest:
            fillvalue_vt = kwargs.pop("fillvalue", ConstantVariable.create(None))
            if kwargs:
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.zip_longest",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'fillvalue', but got "
                    f"{','.join(set(kwargs.keys()) - {'fillvalue'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )
            iterables = [generic_getiter(tx, arg) for arg in args]
            return ZipLongestVariable(
                iterables,
                fillvalue=fillvalue_vt,
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.product:
            if any(kw != "repeat" for kw in kwargs):
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.product",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'repeat', but got "
                    f"{','.join(set(kwargs.keys()) - {'repeat'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if "repeat" in kwargs:
                r = kwargs["repeat"].as_python_constant()
            else:
                r = 1
            seqs = [unpack_iterable(tx, arg) for arg in args]
            items = [
                variables.TupleVariable(list(item))
                for item in itertools.product(*seqs, repeat=r)
            ]
            return variables.ListIteratorVariable(
                items,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.groupby:
            if any(kw != "key" for kw in kwargs):
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'key', but got "
                    f"{','.join(set(kwargs.keys()) - {'key'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            def retrieve_const_key(key: VariableTracker) -> Any:
                # [FUNCTION DOC] HELPER: Converts a grouping key tracker to a real python
                # value (evaluating symbolic ints, or using a python constant).
                # Graph-breaks for non-constant keys.
                from ..utils import specialize_symnode

                # Unwrap LazyVariableTracker to get the underlying variable
                key = specialize_symnode(key)
                if isinstance(key, variables.SymNodeVariable):
                    return key.evaluate_expr()
                elif key.is_python_constant():
                    return key.as_python_constant()
                else:
                    unimplemented(
                        gb_type="Unsupported key type for itertools.groupby",
                        context=f"call_function {self} {args} {kwargs}",
                        explanation="Dynamo does not know how to trace "
                        f"itertools.groupby with key type: {str(type(key))}. "
                        "We only support grouping keys that are constants (int, float, str, etc.)",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

            if len(args) != 1:
                unimplemented(
                    gb_type="Unsupported arguments for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Dynamo does not know how to trace "
                    f"itertools.groupby with args: {args} and kwargs: {kwargs}. "
                    "itertools.groupby expects an iterable to group and an "
                    "optional key function to determine groupings.",
                    hints=[
                        "Make sure the arguments to itertools.groupby are correct.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            seq = unpack_iterable(tx, args[0])

            if "key" in kwargs:

                def keyfunc(x: VariableTracker) -> Any:
                    # [FUNCTION DOC] HELPER: Applies the user key function to an element
                    # symbolically and returns the resulting constant key value.
                    return retrieve_const_key(
                        kwargs.get("key").call_function(tx, [x], {})  # type: ignore[union-attr]
                    )

            else:

                def keyfunc(x: VariableTracker) -> Any:
                    # [FUNCTION DOC] HELPER: Identity key: the element itself as constant key.
                    return retrieve_const_key(x)

            result = []
            try:
                for k, v in itertools.groupby(seq, key=keyfunc):
                    result.append(
                        variables.TupleVariable(
                            [
                                (
                                    variables.ConstantVariable.create(k)
                                    if variables.ConstantVariable.is_literal(k)
                                    else k
                                ),
                                variables.ListIteratorVariable(
                                    list(v), mutation_type=ValueMutationNew()
                                ),
                            ],
                            mutation_type=ValueMutationNew(),
                        )
                    )
            except Exception as e:
                unimplemented(
                    gb_type="Unexpected failure during itertools.groupby() iteration",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Unexpected failure in invoking function during groupby",
                    hints=[*graph_break_hints.SUPPORTABLE],
                    from_exc=e,
                )
            return variables.ListIteratorVariable(
                result,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.repeat:
            # ref: repeat_new in itertoolsmodule.c. A user-supplied negative
            # `times` is clamped to 0; an absent `times` means unbounded.
            try:
                bound = _REPEAT_SIGNATURE.bind(*args, **kwargs)
            except TypeError:
                raise_args_mismatch(tx, "repeat")
            item = bound.arguments["object"]
            times = bound.arguments.get("times")
            if times is None:
                return RepeatIteratorVariable(item, mutation_type=ValueMutationNew())
            if not times.is_python_constant():
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.repeat),
                    [item, times],
                    {},
                )
            times_val = times.as_python_constant()
            if not isinstance(times_val, int):
                raise_type_error(
                    tx,
                    f"'{times.python_type_name()}' object cannot be interpreted "
                    "as an integer",
                )
            return RepeatIteratorVariable(
                item, times=max(times_val, 0), mutation_type=ValueMutationNew()
            )
        elif self.value is itertools.count:
            # count(start=0, step=1): let Python's own argument binding validate
            # the call. Anything it rejects (extra args, duplicate/unknown
            # kwargs) falls through to a graph break so eager raises the
            # CPython TypeError.
            def count_sig(start: Any = 0, step: Any = 1) -> tuple[Any, Any]:
                # [FUNCTION DOC] HELPER: Signature stand-in used only to bind and validate
                # count arguments; returns (start, step) with defaults 0 and 1.
                return start, step

            try:
                item, step = count_sig(*args, **kwargs)
            except TypeError:
                return super().call_function(tx, args, kwargs)
            return variables.CountIteratorVariable(
                item, step, mutation_type=ValueMutationNew()
            )
        else:
            return super().call_function(tx, args, kwargs)


class IteratorVariable(VariableTracker):
    # [CLASS DOC] Abstract base for all iterator trackers. Iterators are their own
    # iterator, report __iter__ and __next__ as present, and compare by object
    # identity semantics. Subclasses must override the next-item hook.
    def __init__(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Forwards keyword options to the base tracker; adds no state.
        super().__init__(**kwargs)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich comparison using the generic object comparison protocol.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Abstract next-item hook; the base version always graph-breaks
        # (marked as a Dynamo bug) because subclasses are expected to override it.
        unimplemented(
            gb_type="Unimplemented next() call",
            context=f"next({self})",
            explanation="This abstract method must be implemented",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "ConstantVariable":
        # [FUNCTION DOC] Answers hasattr queries: True for __iter__ and __next__,
        # otherwise defers to the base implementation.
        if name == "__iter__" or name == "__next__":
            return variables.ConstantVariable.create(True)
        return super().call_obj_hasattr(tx, name)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Returns the iterator itself (iter of an iterator is identity).
        """Iterators are their own iterator."""
        return self


class ChainVariable(IteratorVariable):
    # [CLASS DOC] Tracker for itertools.chain. State: source_iterator (iterator over raw
    # sub-iterables) and current (the active sub-iterator or None). Sub-iterables are
    # converted to iterators lazily at next time. Mutable; mutations are recorded in
    # side effects; reconstructs as chain of current plus chain.from_iterable of source.
    """
    Represents itertools.chain(*iterables) — yields one item per tp_iternext_impl call.

    Uses a source_iterator that yields raw sub-iterables (not yet converted to
    iterators). Conversion happens lazily in tp_iternext_impl so that non-iterable
    args raise TypeError at iteration time, matching CPython's behavior.
    """

    # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L225-L310
    _cpython_type = itertools.chain

    def __init__(
        self,
        source_iterator: "VariableTracker",
        current: "VariableTracker | None" = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the source iterator of raw sub-iterables and the optional
        # current sub-iterator.
        super().__init__(**kwargs)
        # source_iterator yields raw sub-iterables; converted lazily per tp_iternext_impl
        self.source_iterator = source_iterator
        # current sub-iterator (None = not started or just exhausted a sub-iterable)
        self.current = current

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the python type itertools.chain.
        return itertools.chain

    def _get_from_iterable(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] HELPER: Getter for from_iterable; always returns a tracker for
        # the saved chain.from_iterable method.
        return ItertoolsVariable(_CHAIN_FROM_ITERABLE)

    tp_getset = {
        "from_iterable": GetSet(_get_from_iterable),
    }

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Returns the next item. Loops: if no current sub-iterator, pulls
        # the next sub-iterable from the source (end of source raises observed
        # stop-iteration) and converts it to an iterator (non-iterables raise a type
        # error); then takes the next item from it, clearing current when exhausted.
        # Records mutation of this tracker; raises AssertionError if immutable.
        if not self.is_mutable():
            raise AssertionError("ChainVariable must be mutable for next()")
        while True:
            if self.current is None:
                # Pull next sub-iterable from source (source is always an iterator)
                try:
                    next_raw = pyiter_next(tx, self.source_iterator)
                except ObservedUserStopIteration:
                    handle_observed_exception(tx)
                    raise_observed_exception(StopIteration, tx)
                # Convert sub-iterable to iterator lazily — may raise TypeError
                it = generic_getiter(tx, next_raw)
                tx.output.side_effects.mutation(self)
                self.current = it
            try:
                return pyiter_next(tx, self.current)
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                tx.output.side_effects.mutation(self)
                self.current = None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode creating a real chain object from the current
        # sub-iterator (if any) followed by chain.from_iterable over the source iterator.
        # Reconstruct as itertools.chain(current, itertools.chain.from_iterable(source))
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("chain"),
                ]
            )
        )
        if self.current is not None:
            codegen(self.current)
        # chain.from_iterable(source_iterator) for remaining sub-iterables
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("chain"),
                    codegen.create_load_attr("from_iterable"),
                ]
            )
        )
        codegen(self.source_iterator)
        codegen.extend_output(create_call_function(1, False))
        n_args = (1 if self.current is not None else 0) + 1
        codegen.extend_output(create_call_function(n_args, False))


class RepeatIteratorVariable(IteratorVariable):
    # [CLASS DOC] Tracker for itertools.repeat. State: item (value yielded), times
    # (total count, None for unbounded), remaining (count left, None if unbounded).
    # Bounded instances are mutable and decrement on each next; reconstructs as repeat
    # with the remaining count.
    # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c
    # repeat carries a `cnt` count: -1 means unbounded; CPython's repeat_new
    # clamps a user-supplied negative `times` to 0. We model that with `times`
    # (None == unbounded) and a `remaining` counter consumed by next().
    _nonvar_fields = {
        "times",
        "remaining",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        item: VariableTracker,
        times: int | None = None,
        remaining: int | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores item, times and remaining; remaining defaults to times.
        super().__init__(**kwargs)
        self.item = item
        self.times = times
        self.remaining = times if remaining is None else remaining

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the python type itertools.repeat.
        return itertools.repeat

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns the repeated item. Unbounded: always the item, no state
        # change. Bounded: raises observed stop-iteration when the remaining count is
        # zero, otherwise records mutation, decrements remaining and returns the item.
        # Raises AssertionError if a bounded instance is immutable.
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c repeat_next
        if self.remaining is None:
            return self.item
        if not self.is_mutable():
            raise AssertionError("bounded RepeatIteratorVariable must be mutable")
        if self.remaining == 0:
            raise_observed_exception(StopIteration, tx)
        tx.output.side_effects.mutation(self)
        self.remaining -= 1
        return self.item

    def repeat_length_hint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Implements the length-hint method: returns a constant with the
        # remaining count; raises a type error for the unbounded form. The args and
        # kwargs are ignored.
        # ref: repeat_len in itertoolsmodule.c (exposed as __length_hint__);
        # raises TypeError for the unbounded form ("len() of unsized object").
        # Not a C-level slot, so it lives in tp_methods rather than call_method.
        if self.times is None:
            raise_type_error(tx, "len() of unsized object")
        return ConstantVariable.create(self.remaining)

    tp_methods = {
        "__length_hint__": Method(repeat_length_hint),
    }

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns a constant string repr: repeat of the item's repr, with
        # the remaining count appended for bounded iterators.
        item_repr = tracked_repr(tx, self.item)
        if self.times is None:
            return ConstantVariable.create(f"repeat({item_repr})")
        return ConstantVariable.create(f"repeat({item_repr}, {self.remaining})")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode creating a real repeat object with the item and,
        # if bounded, the remaining count.
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("repeat"),
                ]
            )
        )
        codegen(self.item)
        if self.times is None:
            codegen.extend_output(create_call_function(1, False))
        else:
            codegen(ConstantVariable.create(self.remaining))
            codegen.extend_output(create_call_function(2, False))


class CountIteratorVariable(IteratorVariable):
    # [CLASS DOC] Tracker for itertools.count. State: item (next value), step, and
    # advance_count (number of next calls during tracing, replayed on the real
    # iterator by side-effect handling). Mutable; reconstructs as count(item, step).
    # advance_count tracks how many next() calls were made during tracing,
    # used by side_effects.py to replay them on the real iterator post-execution.
    _nonvar_fields = {
        "advance_count",
        *IteratorVariable._nonvar_fields,
    }

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the python type itertools.count.
        return itertools.count

    def __init__(
        self,
        item: int | VariableTracker = 0,
        step: int | VariableTracker = 1,
        advance_count: int = 0,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores item and step (plain numbers are wrapped as constant
        # trackers) and the advance count.
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step
        self.advance_count = advance_count

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns the current value, then records mutation, advances the
        # value by adding step symbolically, and increments advance_count. Never
        # exhausts. Raises AssertionError if immutable.
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L4189-L4216
        if not self.is_mutable():
            raise AssertionError("CountIteratorVariable must be mutable for next()")
        old_item = self.item
        tx.output.side_effects.mutation(self)
        self.item = self.item.call_method(tx, "__add__", [self.step], {})
        self.advance_count += 1
        return old_item

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns a constant string repr like count(start) or
        # count(start, step) (step omitted when it is the integer one). Falls back to the
        # base repr if value or step is not a python constant.
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L4218-L4243
        if not (self.item.is_python_constant() and self.step.is_python_constant()):
            return super().tp_repr_impl(tx)
        cnt = self.item.as_python_constant()
        step = self.step.as_python_constant()
        # Suppress step in the repr when it is an integer equal to 1.
        if isinstance(step, int) and step == 1:
            result = f"count({cnt!r})"
        else:
            result = f"count({cnt!r}, {step!r})"
        return ConstantVariable.create(result)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode creating a real count object from the current
        # item and step.
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("count"),
                ]
            )
        )
        codegen(self.item)
        codegen(self.step)
        codegen.extend_output(create_call_function(2, False))


class ZipVariable(IteratorVariable):
    # [CLASS DOC] Tracker for builtin zip. State: iterable (tuple tracker of the
    # underlying iterators) and strict flag. Each next pulls one item from every
    # iterator into a tuple. Mutable; reconstructs as zip with a strict keyword.
    """
    Represents zip(*iterables)
    """

    # PyZip_Type: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L3011
    _cpython_type = zip

    _nonvar_fields = {
        "strict",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        iterable: VariableTracker,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the tuple tracker of iterators and the strict flag.
        # Raises AssertionError if the iterable is not a tuple tracker.
        super().__init__(**kwargs)
        if not isinstance(iterable, variables.TupleVariable):
            raise AssertionError(f"Expected a tuple of iterables, got {type(iterable)}")
        # can be list[Variable] or VariableTracker (with next_variable implemented)
        self.iterable = iterable
        self.strict = strict

    def python_type(self) -> type[zip]:  # type: ignore[type-arg]
        # [FUNCTION DOC] Returns the builtin zip type.
        return zip

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Returns a tuple tracker with the next item of each iterator. No
        # iterators: observed stop-iteration. A non-strict exhaustion propagates the
        # stop-iteration. Strict mode raises a value error if a later iterator is
        # shorter, or if the first is exhausted while another still yields; if all
        # are exhausted the stop-iteration propagates. Raises AssertionError if immutable.
        # KNOWN ISSUE: in strict mode, the shorter-than-previous value error is raised
        # without first clearing the pending observed stop-iteration, unlike the map
        # tracker which handles it first.
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L2906-L2994
        if not self.is_mutable():
            raise AssertionError("ZipVariable must be mutable for next()")
        tuplesize = len(self.iterable.items)

        if tuplesize == 0:
            raise_observed_exception(StopIteration, tx)

        items = []
        for i in range(tuplesize):
            it = self.iterable.items[i]
            try:
                items.append(pyiter_next(tx, it))
            except ObservedUserStopIteration:
                if not self.strict:
                    raise

                if i > 0:
                    raise_value_error(
                        tx, f"zip() argument {i} shorter than previous arguments"
                    )

                # In strict mode, if any iterable is exhausted, all must be exhausted
                for j in range(i + 1, tuplesize):
                    it_j = self.iterable.items[j]
                    if is_iterator_exhausted(tx, it_j):
                        continue
                    break
                else:
                    # all iterables exhausted, raise StopIteration
                    raise

                handle_observed_exception(tx)  # StopIteration
                raise_value_error(
                    tx, f"zip() argument {i} is longer than previous arguments"
                )

        return variables.TupleVariable(items)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode calling builtin zip with the iterator tuple
        # unpacked and the strict keyword set to the stored flag.
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "zip"), call_function_ex=True
        )
        codegen(self.iterable)
        codegen.extend_output(
            [
                codegen.create_load_const("strict"),
                codegen.create_load_const(self.strict),
                create_instruction("BUILD_MAP", arg=1),
                *create_call_function_ex(True, False),
            ]
        )


class ZipLongestVariable(IteratorVariable):
    # [CLASS DOC] Tracker for itertools.zip_longest. State: iterables (list of
    # iterators), fillvalue tracker, and exhausted (per-iterator flags). Exhausted
    # iterators contribute the fill value until all are exhausted. Mutable;
    # reconstructs with empty iterators substituted for exhausted ones.
    """
    Represents itertools.zip_longest(*iterables, fillvalue=None)
    """

    # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L2822-L2887
    _cpython_type = itertools.zip_longest

    _nonvar_fields = {
        "exhausted",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        iterables: "list[VariableTracker]",
        fillvalue: "VariableTracker",
        exhausted: "list[bool] | None" = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores iterators and fill value; exhausted flags default to all
        # False, one per iterator.
        super().__init__(**kwargs)
        self.iterables = iterables
        self.fillvalue = fillvalue
        self.exhausted = (
            exhausted if exhausted is not None else [False] * len(iterables)
        )

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the python type itertools.zip_longest.
        return itertools.zip_longest

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # [FUNCTION DOC] Returns a tuple tracker with one item per iterator, using the fill
        # value for exhausted ones. An iterator that raises stop-iteration is marked
        # exhausted (recording mutation) and replaced by the fill value. If all were
        # already exhausted, or none produced a real item this round, raises observed
        # stop-iteration. Raises AssertionError if immutable.
        # ref: https://github.com/python/cpython/blob/3.13/Modules/itertoolsmodule.c#L2737-L2808
        if not self.is_mutable():
            raise AssertionError("ZipLongestVariable must be mutable for next()")
        if all(self.exhausted):
            raise_observed_exception(StopIteration, tx)
        values = []
        # CPython: when the last active iterator exhausts, return without yielding.
        any_active = False
        for i, it in enumerate(self.iterables):
            if self.exhausted[i]:
                values.append(self.fillvalue)
            else:
                try:
                    values.append(pyiter_next(tx, it))
                    any_active = True
                except ObservedUserStopIteration:
                    handle_observed_exception(tx)
                    tx.output.side_effects.mutation(self)
                    self.exhausted[i] = True
                    values.append(self.fillvalue)
        if not any_active:
            raise_observed_exception(StopIteration, tx)
        return variables.TupleVariable(values)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode calling zip_longest with the live iterators (an
        # empty iterator stands in for each exhausted one) and the fillvalue keyword.
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("zip_longest"),
                ]
            ),
            call_function_ex=True,
        )
        for i, it in enumerate(self.iterables):
            if not self.exhausted[i]:
                codegen(it)
            else:
                codegen.add_push_null(
                    lambda: codegen.append_output(
                        codegen.create_load_python_module(iter)  # type: ignore[arg-type]
                    )
                )
                codegen.extend_output(
                    [
                        create_instruction("BUILD_TUPLE", arg=0),
                        *create_call_function(1, False),
                    ]
                )
        codegen.extend_output(
            [create_instruction("BUILD_TUPLE", arg=len(self.iterables))]
        )
        codegen.extend_output([codegen.create_load_const("fillvalue")])
        codegen(self.fillvalue)
        codegen.extend_output(
            [
                create_instruction("BUILD_MAP", arg=1),
                *create_call_function_ex(True, False),
            ]
        )


class MapVariable(IteratorVariable):
    # [CLASS DOC] Tracker for builtin map. State: fn (callable tracker), iterable (tuple
    # tracker of iterators) and strict flag (only reconstructible on Python 3.14+).
    # Each next pulls one item from every iterator and calls fn on them. Mutable.
    """
    Represents map(fn, *iterables)
    """

    # PyMap_Type: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L1484
    _cpython_type = map

    _nonvar_fields = {
        "strict",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn: VariableTracker,
        iterables: VariableTracker,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores fn, the tuple tracker of iterators and the strict flag.
        # Raises AssertionError if iterables is not a tuple tracker.
        super().__init__(**kwargs)
        self.fn = fn
        if not isinstance(iterables, variables.TupleVariable):
            raise AssertionError(
                f"Expected a tuple of iterables, got {type(iterables)}"
            )
        self.iterable = iterables
        self.strict = strict

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the builtin map type.
        return map

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Pulls the next item from each iterator and returns the result of
        # calling fn on them symbolically. Non-strict exhaustion propagates the
        # stop-iteration. Strict mode raises a value error if a later iterator is
        # shorter or a later one still has items after the first ended; otherwise
        # raises observed stop-iteration. Raises AssertionError if immutable.
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L1409-L1450
        if not self.is_mutable():
            raise AssertionError("MapVariable must be mutable for next()")

        tuplesize = len(self.iterable.items)

        items = []
        for i in range(tuplesize):
            it = self.iterable.items[i]
            try:
                items.append(pyiter_next(tx, it))
            except ObservedUserStopIteration:
                if not self.strict:
                    raise

                handle_observed_exception(tx)  # StopIteration

                if i:
                    raise_value_error(
                        tx,
                        f"map() argument {i + 1} shorter than argument {i}",
                    )

                # In strict mode, if any iterable is exhausted, all must be exhausted.
                for j in range(1, tuplesize):
                    it_j = self.iterable.items[j]
                    if not is_iterator_exhausted(tx, it_j):
                        raise_value_error(
                            tx, f"map() argument {j + 1} is longer than argument {j}"
                        )

                raise_observed_exception(StopIteration, tx)

        # type: ignore[attr-defined]
        return self.fn.call_function(tx, items, {})

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode calling builtin map with fn and the iterators. If
        # strict is set, adds the strict keyword (raises AssertionError before Python
        # 3.14, where map has no strict option).
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "map"), call_function_ex=True
        )
        codegen(variables.TupleVariable([self.fn, *self.iterable.items]))
        if self.strict:
            if sys.version_info < (3, 14):
                raise AssertionError(
                    "Unexpected bug: map(strict=True) requires Python 3.14+"
                )
            codegen.extend_output(
                [
                    codegen.create_load_const("strict"),
                    codegen.create_load_const(self.strict),
                    create_instruction("BUILD_MAP", arg=1),
                    *create_call_function_ex(True, False),
                ]
            )
        else:
            codegen.extend_output(create_call_function_ex(False, False))


class FilterVariable(IteratorVariable):
    # [CLASS DOC] Tracker for builtin filter. State: fn (predicate tracker, or None
    # constant for identity) and iterable (single iterator). Reconstructs as filter
    # of fn over the iterable.
    """
    Represents filter(fn, iterable)
    """

    # PyFilter_Type: https://github.com/python/cpython/blob/v3.13.0/Python/bltinmodule.c#L630
    _cpython_type = filter

    def __init__(
        self,
        fn: VariableTracker,
        iterable: VariableTracker,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the predicate and the iterable tracker.
        super().__init__(**kwargs)
        self.fn = fn
        self.iterable = iterable

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the builtin filter type.
        return filter

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Loops pulling items from the iterable until one is truthy. The
        # test value is the item itself when fn is None, else fn called on the item;
        # truthiness is evaluated through a polyfill and must be a python constant.
        # Iterator exhaustion propagates as stop-iteration. Does not record mutation.
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L573-L606
        # A do-while loop to find elements that make fn return true
        while True:
            item = pyiter_next(tx, self.iterable)
            if self.fn.is_constant_none():
                res = item
            else:
                res = self.fn.call_function(tx, [item], {})
            pred_res = variables.UserFunctionVariable(
                polyfills.predicate  # type: ignore[arg-type]
            ).call_function(tx, [res], {})
            if pred_res.as_python_constant():
                return item

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode calling builtin filter with fn and the iterable.
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "filter"))
        codegen(self.fn)
        codegen(self.iterable)
        codegen.extend_output(create_call_function(2, False))


class DictViewIterator(IteratorVariable):
    # [CLASS DOC] Base for iterators over a dict or set snapshot. The view_type class
    # attribute (keys, values or items) selects what is yielded. State: _iter, a real
    # Python iterator over the captured tracker items, so consumption during tracing
    # is reflected in length hints. Subclasses only set the CPython type and view_type.
    """Base class for dict view iterators (keys, values, or items)."""

    _nonvar_fields = {
        "view_type",
        *IteratorVariable._nonvar_fields,
    }

    view_type: str = "keys"

    def __init__(
        self,
        items: dict[HashableTracker, VariableTracker],
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC] Creates the underlying real iterator over the mapping's keys,
        # values or key-value pairs depending on view_type. Raises AssertionError for an
        # unknown view_type.
        super().__init__(**kwargs)
        if self.view_type == "keys":
            self._iter = iter(items)
        elif self.view_type == "values":
            self._iter = iter(items.values())  # type: ignore[bad-assignment]
        else:
            if self.view_type != "items":
                raise AssertionError(
                    f"Expected view_type 'items', got {self.view_type!r}"
                )
            self._iter = iter(items.items())  # type: ignore[bad-assignment]

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] Advances the real iterator and returns the key tracker, value
        # tracker, or a tuple tracker of both. A real StopIteration or RuntimeError
        # (for example dict changed size) is converted into the same observed
        # exception in the traced program.
        # dictiter_iternextitem: https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5538-L5578
        # dictiter_iternextkey: https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5125-L5144
        # dictiter_iternextvalue: https://github.com/python/cpython/blob/v3.13.3/Objects/dictobject.c#L5248-L5267
        try:
            item = next(self._iter)

            if self.view_type == "keys":
                return item.vt
            elif self.view_type == "values":
                return item  # type: ignore[bad-return]
            else:  # items
                k, v = item  # type: ignore[not-iterable]
                return VariableTracker.build(tx, (k.vt, v))
        except (RuntimeError, StopIteration) as e:
            raise_observed_exception(
                type(e),
                tx,
                args=[VariableTracker.build(tx, a) for a in e.args],
            )

    def dict_view_iter_length_hint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Length-hint method: returns a constant with the number of
        # not-yet-consumed elements computed from the real iterator. Args are ignored.
        # dictiter_len/setiter_len: __length_hint__ returns the number of
        # not-yet-consumed elements. self._iter is a live Python iterator over
        # the captured items, so its own length hint already reflects any
        # next() calls made during tracing.
        # Not a C-level slot, so it lives in tp_methods rather than call_method.
        return ConstantVariable.create(operator.length_hint(self._iter))

    tp_methods = {
        "__length_hint__": Method(dict_view_iter_length_hint),
    }

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the real iterator type matching view_type: dict key,
        # value or item iterator type. Set iterators (view_type keys) report the dict
        # key iterator type.
        if self.view_type == "keys":
            return type(iter({}))
        elif self.view_type == "values":
            return type(iter({}.values()))
        else:  # items
            return type(iter({}.items()))


class DictIterator(DictViewIterator):
    # [CLASS DOC] Iterator over a dict directly (yields keys).
    _cpython_type = type(iter({}))
    view_type = "keys"


class DictKeysIterator(DictViewIterator):
    # [CLASS DOC] Iterator over a dict keys view (yields keys).
    _cpython_type = type(iter({}.keys()))
    view_type = "keys"


class DictValuesIterator(DictViewIterator):
    # [CLASS DOC] Iterator over a dict values view (yields values).
    _cpython_type = type(iter({}.values()))
    view_type = "values"


class DictItemsIterator(DictViewIterator):
    # [CLASS DOC] Iterator over a dict items view (yields key-value tuples).
    _cpython_type = type(iter({}.items()))
    view_type = "items"


class SetIterator(DictViewIterator):
    # [CLASS DOC] Iterator over a set snapshot (yields the elements as keys).
    # KNOWN ISSUE: python_type inherited from the base reports the dict key iterator
    # type rather than the real set iterator type.
    _cpython_type = type(iter(set()))
    view_type = "keys"
