# [MODULE DOC] Lazy VariableTracker support for Dynamo. Defines LazyVariableTracker,
# which defers building the real tracker for a sourced value until it is used, plus
# cache holders (LazyCache, ComputedLazyCache), LazyConstantVariable (primitive
# constants with deferred guards: type guard first, value guard only on real use),
# ComputedLazyConstantVariable (results of ops over lazy constants, recomputed in
# output bytecode if never realized) and LazySymNodeFormatString. Import-time side
# effect: the populate call at the bottom installs forwarding methods on
# LazyVariableTracker so every other tracker method realizes first and then delegates.
from __future__ import annotations

import collections
import functools
import inspect
from typing import Any, TYPE_CHECKING

from .. import config
from ..utils import is_function_or_wrapper
from .base import SourceLocation, VariableTracker, VariableTrackerMeta


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing_extensions import Self

    from ..symbolic_convert import InstructionTranslatorBase
    from .tensor import SymNodeVariable


class LazyCache:
    # [CLASS DOC] Holder shared by a lazy tracker and its clones: the raw value, its
    # source, optional name hint and source location, and the realized tracker (vt,
    # None until realized). After realization the raw value, source and hints are
    # deleted, so only vt remains.
    """Container to cache the real VariableTracker"""

    def __init__(self, value: Any, source: Any) -> None:
        # [FUNCTION DOC] Stores the raw value and source with an unrealized state.
        # Raises AssertionError if source is empty and the value is not a lazy format string.
        if not isinstance(value, LazySymNodeFormatString):
            if not source:
                raise AssertionError(
                    "source is required for non-LazySymNodeFormatString values"
                )
        self.value = value
        self.source = source
        self.name_hint: str | None = None
        self.source_location: SourceLocation | None = None
        self.vt: VariableTracker | None = None

    def realize(self) -> None:
        # [FUNCTION DOC] Builds the real tracker for the current translator: a sourceless
        # build for lazy format strings, otherwise a sourced build with lazy constants
        # disabled (avoids infinite recursion). Applies the stored name hint and source
        # location, then deletes the raw value, source and hints. Raises AssertionError
        # if already realized.
        if self.vt is not None:
            raise AssertionError("LazyCache.realize() called but vt is already set")
        from ..symbolic_convert import InstructionTranslator
        from . import builder

        tx = InstructionTranslator.current_tx()

        if isinstance(self.value, LazySymNodeFormatString):
            self.vt = builder.SourcelessBuilder.create(tx, self.value)
        else:
            # Pass allow_lazy_constant=False to prevent VariableBuilder from
            # returning LazyConstantVariable, which would cause infinite recursion
            # when LazyVariableTracker.realize() returns LazyConstantVariable.
            self.vt = builder.VariableBuilder(
                tx, self.source, allow_lazy_constant=False
            )(self.value)

        if self.name_hint is not None:
            self.vt.set_name_hint(self.name_hint)

        if self.source_location is not None and self.vt.source_location is None:
            self.vt.set_source_location(self.source_location)

        del self.value
        del self.source
        del self.name_hint
        del self.source_location


class ComputedLazyCache:
    # [CLASS DOC] Cache for a sourceless constant computed from lazy operands. State:
    # the eagerly computed value, the lazy constant operands, the argument trackers,
    # the operator, a node count bounding chain depth, optional hints, and vt (the
    # realized tracker, None until realized). Unlike LazyCache, its fields are kept
    # after realization.
    """Like LazyCache, but for sourceless constants computed from lazy operands."""

    def __init__(
        self,
        value: Any,
        lazy_vars: list[LazyConstantVariable],
        args: list[VariableTracker],
        op: Callable[..., Any],
        num_nodes: int,
    ) -> None:
        # [FUNCTION DOC] Stores the precomputed value, operands, arguments, operator and
        # node count; starts unrealized.
        self.value = value
        self.lazy_vars = lazy_vars
        self.args = args
        self.op = op
        self.num_nodes = num_nodes
        self.name_hint: str | None = None
        self.source_location: SourceLocation | None = None
        self.vt: VariableTracker | None = None

    def realize(self) -> None:
        # [FUNCTION DOC] Realizes every lazy operand (installing their value guards). If
        # any operand became symbolic, recomputes the result by symbolically calling the
        # operator on realized arguments; otherwise wraps the precomputed value in a
        # constant tracker. Applies name hint and source location. Raises
        # AssertionError if already realized.
        if self.vt is not None:
            raise AssertionError(
                "ComputedLazyCache.realize() called but vt is already set"
            )
        from ..symbolic_convert import InstructionTranslator
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        any_symbolic = False
        for lazy_var in self.lazy_vars:
            if isinstance(lazy_var.realize(), SymNodeVariable):
                any_symbolic = True

        if any_symbolic:
            # The precomputed constant is stale; recompute symbolically
            from .builtin import BuiltinVariable

            tx = InstructionTranslator.current_tx()
            realized_args = [arg.realize() for arg in self.args]
            self.vt = BuiltinVariable(self.op).call_function(tx, realized_args, {})
        else:
            self.vt = ConstantVariable.create(self.value)

        if self.name_hint is not None:
            self.vt.set_name_hint(self.name_hint)
        if self.source_location is not None and self.vt.source_location is None:
            self.vt.set_source_location(self.source_location)


class LazyVariableTracker(VariableTracker, metaclass=VariableTrackerMeta):
    # [CLASS DOC] Placeholder tracker that defers building the real tracker until it
    # is needed. State: _cache (a shared LazyCache or ComputedLazyCache holding the
    # raw value and, once realized, the real tracker). Any tracker method not
    # explicitly defined realizes first and forwards (installed by the populate
    # helper); attribute access also forwards. Implicit realization in isinstance
    # checks is disabled via a class flag.
    """
    A structure that defers the creation of the actual VariableTracker
    for a given underlying value until it is accessed.

    The `realize` function invokes VariableTracker.build() to produce the real object.
    Once a LazyVariableTracker has been realized, internal bookkeeping will
    prevent double realization.

    This object should be utilized for processing containers, or objects that
    reference other objects where we may not want to take on creating all the
    VariableTrackers right away.
    """

    # Flag to prevent implicit realization in isinstance checks (inherited by subclasses)
    _no_implicit_realize = True
    _nonvar_fields = {"_cache", *VariableTracker._nonvar_fields}

    @staticmethod
    def create(
        value: Any,
        source: Any,
        *,
        tx: InstructionTranslatorBase | None = None,
        **options: Any,
    ) -> VariableTracker:
        # [FUNCTION DOC] Factory. Primitive int, float, bool or str values become lazy
        # constants. Otherwise, when a translator and source are given and no extra
        # options, returns the tracker cached per source in the output graph (creating
        # and caching one if missing); else makes a fresh lazy tracker with the options.
        if type(value) in LazyConstantVariable.supported_types:
            return LazyConstantVariable.create(value, source, **options)

        # Cache based on source when no extra options are passed
        if tx is not None and source is not None and not options:
            cache = tx.output.variable_tracker_cache
            cached = cache.get(source)
            if cached is not None:
                return cached
            vt = LazyVariableTracker(LazyCache(value, source), source=source)
            cache[source] = vt
            return vt

        return LazyVariableTracker(LazyCache(value, source), source=source, **options)

    def __init__(self, _cache: LazyCache | ComputedLazyCache, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the shared cache. Raises AssertionError if it is not one
        # of the two cache types.
        if not isinstance(_cache, (LazyCache, ComputedLazyCache)):
            raise AssertionError(
                f"_cache must be a LazyCache or ComputedLazyCache instance, got {type(_cache).__name__}"
            )
        super().__init__(**kwargs)
        self._cache = _cache

    def realize(self) -> VariableTracker:
        # [FUNCTION DOC] Returns the real tracker, building it through the cache on first
        # use (with its guards and side effects). Idempotent afterwards.
        """Force construction of the real VariableTracker"""
        if self._cache.vt is None:
            self._cache.realize()
            if self._cache.vt is None:
                raise AssertionError("LazyCache.realize() did not set vt")
        return self._cache.vt

    def lazy_isinstance(self, cls: type) -> bool:
        # [FUNCTION DOC] Realizes the tracker and returns whether the real tracker is an
        # instance of cls (bypassing the metaclass hook).
        """Check isinstance after realizing, used by ImplicitRealizingVariableTrackerMeta"""
        return type.__instancecheck__(cls, self.realize())

    def unwrap(self) -> VariableTracker | Self:
        # [FUNCTION DOC] Returns the real tracker if already realized, else self; never
        # triggers realization.
        """Return the real VariableTracker if it already exists"""
        if self.is_realized():
            if self._cache.vt is None:
                raise AssertionError("is_realized() returned True but vt is None")
            return self._cache.vt
        return self

    def is_realized(self) -> bool:
        # [FUNCTION DOC] Returns True once the cache holds a real tracker.
        return self._cache.vt is not None

    def clone(self, **kwargs: Any) -> VariableTracker:
        # [FUNCTION DOC] Clones with overridden fields. Changing the cache is not allowed
        # (AssertionError). Changing the source forces realization; the clone is made
        # from the real tracker if realized, so the result is usually not lazy.
        if kwargs.get("_cache", self._cache) is not self._cache:
            raise AssertionError(
                "cannot clone LazyVariableTracker with a different _cache"
            )
        if kwargs.get("source", self.source) is not self.source:
            self.realize()
        return VariableTracker.clone(self.unwrap(), **kwargs)

    def peek_type(self) -> type[Any]:
        # [FUNCTION DOC] Returns the type of the raw underlying value without realizing.
        # Raises AssertionError if already realized (raw value is gone).
        if self.is_realized():
            raise AssertionError(
                "peek_type() called on an already realized LazyVariableTracker"
            )
        return type(self._cache.value)

    def peek_value(self) -> Any:
        # [FUNCTION DOC] Returns the raw underlying value without realizing. Raises
        # AssertionError if already realized.
        if self.is_realized():
            raise AssertionError(
                "peek_value() called on an already realized LazyVariableTracker"
            )
        return self._cache.value

    def set_name_hint(self, name: str) -> None:
        # [FUNCTION DOC] Sets the graph name hint: forwarded to the real tracker if
        # realized, otherwise stored in the cache to apply at realization.
        if self.is_realized():
            self._cache.vt.set_name_hint(name)  # type: ignore[union-attr]
        else:
            self._cache.name_hint = name

    def set_source_location(self, source_location: SourceLocation) -> None:
        # [FUNCTION DOC] Records the user source location on this tracker and either
        # forwards it to the real tracker (if realized) or stores it in the cache.
        self.source_location = source_location
        if self.is_realized():
            self._cache.vt.set_source_location(source_location)  # type: ignore[union-attr]
        else:
            self._cache.source_location = source_location

    def __str__(self) -> str:
        # [FUNCTION DOC] Debug string showing the real tracker repr when realized, or the
        # unrealized raw value type. Does not realize. Also used as repr.
        variable_info = "LazyVariableTracker("
        if self.is_realized():
            variable_info += f"realized: {repr(self.unwrap())})"
        else:
            variable_info += f"unrealized: {self.peek_type()})"

        return variable_info

    def __getattr__(self, item: str) -> Any:
        # [FUNCTION DOC] Fallback for missing attributes: realizes and returns the same
        # attribute from the real tracker.
        return getattr(self.realize(), item)

    def get_handler_type_for_dispatch(self) -> type:
        # [FUNCTION DOC] Returns the lazy tracker class itself, signalling builtin dispatch
        # that arguments must be realized before the handler runs.
        """Return the VariableTracker type to use for builtin handler dispatch.

        For regular LazyVariableTracker (not LazyConstantVariable), we return
        LazyVariableTracker itself so that _make_handler knows it needs to realize
        the arguments before calling the handler. LazyConstantVariable overrides
        this to return ConstantVariable since it can stay lazy.
        """
        return LazyVariableTracker

    # most methods are auto-generated below, these are the ones we want to exclude
    visit = VariableTracker.visit  # type: ignore[assignment]
    __repr__ = __str__

    @classmethod
    def realize_all(
        cls,
        value: Any,
        cache: dict[int, tuple[Any, Any]] | None = None,
        *,
        allow_lazy_constant: bool = False,
    ) -> Any:
        # [FUNCTION DOC] Recursively walks a value and returns it with all lazy trackers
        # realized. Handles lazy trackers (optionally keeping lazy constants lazy, and
        # for unrealized computed constants only ensuring operand type guards), trackers
        # (their non-excluded fields are replaced in place), lists, tuples and dicts
        # (rebuilt). Uses an id-keyed memo (also keeps values alive) to handle cycles
        # and sharing. Side effects: realization and in-place field updates.
        """
        Walk an object and realize all LazyVariableTrackers inside it.
        """
        if cache is None:
            cache = {}

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        value_cls = type(value)
        if issubclass(value_cls, LazyVariableTracker):
            # Allow LazyConstantVariable and ComputedLazyConstantVariable to
            # stay lazy when returning from a frame
            keep_lazy = allow_lazy_constant and isinstance(
                value, (LazyConstantVariable, ComputedLazyConstantVariable)
            )
            if keep_lazy:
                if isinstance(value, ComputedLazyConstantVariable):
                    if value.is_realized():
                        result = cls.realize_all(
                            value.realize(), cache, allow_lazy_constant=True
                        )
                    else:
                        # Output bytecode recomputes the value; operands only need TYPE_MATCH
                        for lazy_var in value._cache.lazy_vars:
                            lazy_var._ensure_type_guard()
                        result = value
                else:
                    result = value
            else:
                result = cls.realize_all(
                    value.realize(), cache, allow_lazy_constant=allow_lazy_constant
                )
        elif issubclass(value_cls, VariableTracker):
            # update value in-place
            result = value
            # update cache now to prevent infinite recursion
            cache[idx] = (result, value)
            value_dict = value.__dict__
            nonvars = value._nonvar_fields
            for key in value_dict:
                if key not in nonvars:
                    value_dict[key] = cls.realize_all(
                        value_dict[key], cache, allow_lazy_constant=allow_lazy_constant
                    )
        elif value_cls is list:
            result = [
                cls.realize_all(v, cache, allow_lazy_constant=allow_lazy_constant)
                for v in value
            ]
        elif value_cls is tuple:
            result = tuple(
                cls.realize_all(v, cache, allow_lazy_constant=allow_lazy_constant)
                for v in value
            )
        elif value_cls in (dict, collections.OrderedDict):
            result = {
                k: cls.realize_all(v, cache, allow_lazy_constant=allow_lazy_constant)
                for k, v in list(value.items())
            }
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result

    def is_hashable_lazy(self) -> bool:
        # [FUNCTION DOC] Fast hashability check on the raw value without realizing: true
        # for builtins, classes and functions (or tuples of only those); conservative
        # false otherwise. Raises AssertionError if already realized.
        # Checks that the underlying value is hashable without realizing the VT.
        # This is used by the is_hashable() function in hashable.py as a fast
        # path for unrealized LazyVTs.
        def _helper(value: Any) -> bool:
            # [FUNCTION DOC] HELPER: True if the value is a builtin, a class, or a
            # function-like object.
            # TODO: Add support for more types
            return (
                inspect.isbuiltin(value)
                or issubclass(type(value), type)
                or is_function_or_wrapper(value)
            )

        if self.is_realized():
            raise AssertionError(
                "is_hashable() called on an already realized LazyVariableTracker"
            )
        value = self._cache.value
        if isinstance(value, tuple):
            return all(_helper(v) for v in value)
        return _helper(value)

    def original_value(self) -> Any:
        # [FUNCTION DOC] Returns the raw underlying value without realizing. Raises
        # AssertionError if already realized.
        # Returns the value without realizing the VT.
        if self.is_realized():
            raise AssertionError(
                "original_value() called on an already realized LazyVariableTracker"
            )
        return self._cache.value

    def original_source(self) -> Any:
        # [FUNCTION DOC] Returns the source stored in the cache without realizing. Raises
        # AssertionError if already realized.
        # Returns the source without realizing the VT.
        if self.is_realized():
            raise AssertionError(
                "original_source() called on an already realized LazyVariableTracker"
            )
        # pyrefly: ignore[missing-attribute]
        return self._cache.source


class LazyConstantVariable(LazyVariableTracker):
    # [CLASS DOC] Lazy tracker for int, float, bool and str values with a source. Guards
    # are deferred: a type guard is installed on type queries, and the value guard
    # only on full realization (which replaces the type guard). State: whether a type
    # guard is installed, a reference to it, and a cached dispatch type. Passing the
    # value through unused avoids recompiles when it changes.
    """
    A lazy variable tracker for constants (int, float, bool, str) that defers
    guarding until the value is actually used in a way that requires it.

    This allows constants that are just passed through (e.g., returned without
    being used in control flow or math) to avoid unnecessary recompilation when
    their values change.

    Guards are installed lazily:
    - TYPE_MATCH guard is installed when type-based methods (python_type, is_tensor,
      lazy_isinstance) are called
    - CONSTANT_MATCH guard is installed on full realization (e.g., used in control
      flow or math), which subsumes any TYPE_MATCH guard
    """

    supported_types = (int, float, bool, str)
    _nonvar_fields = {
        "_type_guard_installed",
        "_type_guard_ref",
        "_handler_type",
        *LazyVariableTracker._nonvar_fields,
    }

    @staticmethod
    def create(
        value: Any,
        source: Any,
        **options: Any,
    ) -> VariableTracker:
        # [FUNCTION DOC] Factory for a lazy constant. Raises AssertionError for an
        # unsupported value type or a missing source. If the source cannot carry guards
        # (constant source), returns a plain constant tracker instead.
        from ..source import is_constant_source
        from .constant import ConstantVariable

        if type(value) not in LazyConstantVariable.supported_types:
            raise AssertionError(
                f"unsupported type {type(value).__name__} for LazyConstantVariable"
            )
        if source is None:
            raise AssertionError("source must not be None for LazyConstantVariable")

        # If the source doesn't support guards (e.g., ConstantSource), fall back
        # to creating a regular ConstantVariable directly
        if is_constant_source(source):
            return ConstantVariable.create(value, source=source, **options)

        return LazyConstantVariable(LazyCache(value, source), source=source, **options)

    def __init__(self, _cache: LazyCache, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the cache and initializes type-guard bookkeeping to
        # not installed.
        super().__init__(_cache, **kwargs)
        self._type_guard_installed = False
        self._type_guard_ref: Any = None
        self._handler_type: type | None = None

    def _ensure_type_guard(self) -> None:
        # [FUNCTION DOC] HELPER: Installs a type-match guard on the source once, unless
        # already installed or realized, and remembers the guard so realization can
        # drop it. Raises AssertionError if the source is missing. Installs a guard.
        """Install TYPE_MATCH guard if not already installed and not realized."""
        if self._type_guard_installed or self.is_realized():
            return

        from ..guards import GuardBuilder, install_guard

        if self.source is None:
            raise AssertionError("source must not be None when installing type guard")
        guard = self.source.make_guard(GuardBuilder.TYPE_MATCH)
        install_guard(guard)
        self._type_guard_installed = True
        self._type_guard_ref = guard

    def realize(self) -> VariableTracker:
        # [FUNCTION DOC] Realizes the constant. If the result is a constant tracker, removes
        # the earlier type guard from the guard set and installs a constant-value guard
        # (a symbolic result keeps the type guard). Returns the real tracker. Raises
        # AssertionError if the source is missing.
        """Force construction of the real VariableTracker."""
        if self.is_realized():
            return super().realize()

        from torch._guards import TracingContext

        from ..guards import GuardBuilder, install_guard
        from .constant import ConstantVariable

        if self.source is None:
            raise AssertionError(
                "source must not be None when realizing LazyConstantVariable"
            )

        result = super().realize()

        # Only remove TYPE_MATCH if we're installing CONSTANT_MATCH
        # (which subsumes it). For SymNodeVariable, keep TYPE_MATCH.
        if isinstance(result, ConstantVariable):
            if self._type_guard_ref is not None:
                tracing_context = TracingContext.get()
                tracing_context.guards_context.dynamo_guards.inner.discard(
                    self._type_guard_ref
                )
                self._type_guard_ref = None
            install_guard(self.source.make_guard(GuardBuilder.CONSTANT_MATCH))

        return result

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the python type of the value; if unrealized, installs only
        # the type guard instead of realizing.
        """Return the Python type without triggering realization."""
        if self.is_realized():
            return super().python_type()
        self._ensure_type_guard()
        return self.peek_type()

    def is_tensor(self) -> bool:
        # [FUNCTION DOC] Always False; installs the type guard first.
        """Primitive constants are never tensors."""
        self._ensure_type_guard()
        return False

    def is_constant_none(self) -> bool:
        # [FUNCTION DOC] Always False (supported primitives are never None); installs the
        # type guard first.
        self._ensure_type_guard()
        return False

    def _maybe_realize_for_type(self) -> type | None:
        # [FUNCTION DOC] HELPER: Finds the eventual tracker type. Already realized, or an
        # int/float that may become symbolic (when specialization is off): realizes and
        # returns the real tracker type. Otherwise installs the type guard and returns
        # None, meaning a constant tracker.
        """Check if we need to realize to determine the VariableTracker type.

        Returns None if we can determine the type without realization (and installs
        TYPE_MATCH guard). Returns the realized type if realization was needed.

        With specialize_int=False or specialize_float=False, ints/floats may become
        either ConstantVariable or SymNodeVariable, so we must realize to know.
        For bool/str, we always know it will be ConstantVariable.
        """
        if self.is_realized():
            return type(self.realize())

        value_type = self.peek_type()

        # When specialize_int/specialize_float is False, ints/floats may become
        # SymNodeVariable. Must realize to determine the actual type.
        if not config.specialize_int and value_type is int:
            return type(self.realize())
        if not config.specialize_float and value_type is float:
            return type(self.realize())

        # For bool/str, or when specializing ints/floats, we know it will be
        # ConstantVariable. Install TYPE_MATCH guard and return None.
        self._ensure_type_guard()
        return None

    def get_handler_type_for_dispatch(self) -> type:
        # [FUNCTION DOC] Returns the tracker type used for builtin dispatch (cached):
        # the realized type if realization was required, else the constant tracker
        # class, without realizing.
        """Return the VariableTracker type to use for builtin handler dispatch.

        This allows builtins like isinstance() and type() to find the correct
        handler without triggering full realization of the lazy constant.
        """
        if self._handler_type is not None:
            return self._handler_type

        from .constant import ConstantVariable

        realized_type = self._maybe_realize_for_type()
        self._handler_type = (
            realized_type if realized_type is not None else ConstantVariable
        )
        return self._handler_type

    def lazy_isinstance(self, cls: type) -> bool:
        # [FUNCTION DOC] Answers an instance check against a tracker class, avoiding
        # realization when possible. Realized: normal check. If cls cannot be a parent
        # of constant or symnode trackers: type guard and False. Otherwise resolves the
        # eventual type (possibly realizing for int/float) and checks against it.
        """Check isinstance without triggering realization when possible.

        LazyConstantVariable only wraps primitive types (int, float, bool, str)
        which always realize to ConstantVariable, so we can answer isinstance
        checks by checking if the target class is ConstantVariable or a parent.

        However, when specialize_int=False or specialize_float=False, integers
        and floats may realize to SymNodeVariable instead of ConstantVariable,
        so we must fall back to full realization for those cases.
        """
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # If already realized, delegate to the parent which does the regular check
        if self.is_realized():
            return super().lazy_isinstance(cls)

        # LazyConstantVariable can only realize to ConstantVariable or SymNodeVariable.
        # If cls is not a parent of either, we can answer False without realization.
        if not issubclass(ConstantVariable, cls) and not issubclass(
            SymNodeVariable, cls
        ):
            self._ensure_type_guard()
            return False

        realized_type = self._maybe_realize_for_type()
        if realized_type is not None:
            return issubclass(realized_type, cls)
        return issubclass(ConstantVariable, cls)


class ComputedLazyConstantVariable(LazyVariableTracker):
    # [CLASS DOC] Lazy tracker for the result of an operator over lazy constant operands.
    # The value is computed eagerly at trace time; guards are deferred. Unrealized, only
    # operand type guards are installed and output bytecode recomputes the value by
    # calling the operator; realization realizes the operands (value guards) and builds
    # a constant, or a symbolic result if any operand turned symbolic. Uses a computed cache.
    """
    Result of a supported op over lazy constant operands.  The value is computed
    eagerly at trace time but guards are deferred: if never realized, output
    bytecode recomputes it from the operands (TYPE_MATCH only); realization
    realizes the operands, installing their CONSTANT_MATCH guards.
    """

    _cache: ComputedLazyCache  # pyrefly: ignore[bad-override]

    @staticmethod
    def create(  # pyrefly: ignore[bad-param-name-override, bad-override]
        op: Callable[..., Any],
        args: list[VariableTracker],
    ) -> VariableTracker | None:
        # [FUNCTION DOC] Factory. Applies the operator to the operands' raw values and
        # returns a computed lazy tracker, a plain constant if no operand is lazy, or None
        # (caller falls back to realization) when the feature is disabled by config,
        # the expression chain is too deep, the operator raises, or the result is not
        # a base literal.
        """Compute op(*args) eagerly; returns None to fall back to realization."""
        from .constant import ConstantVariable

        max_nodes = config.computed_lazy_constant_max_nodes
        if max_nodes <= 0:
            return None

        lazy_vars: list[LazyConstantVariable] = []
        num_nodes = 1

        def get_value(arg: VariableTracker) -> Any:
            # [FUNCTION DOC] HELPER: Returns the raw value of an operand. Unrealized lazy
            # operands contribute their lazy constants and node counts to the enclosing
            # lists; other operands use their python constant.
            nonlocal num_nodes
            if isinstance(arg, ComputedLazyConstantVariable) and not arg.is_realized():
                lazy_vars.extend(arg._cache.lazy_vars)
                num_nodes += arg._cache.num_nodes
                return arg._cache.value
            if isinstance(arg, LazyConstantVariable) and not arg.is_realized():
                lazy_vars.append(arg)
                return arg.peek_value()
            return arg.as_python_constant()

        values = [get_value(arg) for arg in args]
        # realize/reconstruct recurse through args; unbounded chains overflow the stack
        if num_nodes > max_nodes:
            return None
        try:
            value = op(*values)
        except Exception:
            return None
        if not ConstantVariable.is_base_literal(value):
            return None
        if not lazy_vars:
            return ConstantVariable.create(value)
        return ComputedLazyConstantVariable(
            ComputedLazyCache(value, lazy_vars, list(args), op, num_nodes)
        )

    def __init__(self, _cache: ComputedLazyCache, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the cache. Raises AssertionError if it is not a computed cache.
        if not isinstance(_cache, ComputedLazyCache):
            raise AssertionError(
                f"_cache must be a ComputedLazyCache instance, got {type(_cache).__name__}"
            )
        super().__init__(_cache, **kwargs)

    def _ensure_operand_type_guards(self) -> None:
        # [FUNCTION DOC] HELPER: Installs type guards on all lazy operands.
        # Type queries on the result depend on the operand types
        for lazy_var in self._cache.lazy_vars:
            lazy_var._ensure_type_guard()

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the python type of the result; unrealized, installs
        # operand type guards and uses the precomputed value's type.
        if self.is_realized():
            return super().python_type()
        self._ensure_operand_type_guards()
        return type(self._cache.value)

    def is_tensor(self) -> bool:
        # [FUNCTION DOC] False; unrealized, installs operand type guards, else defers to
        # the real tracker.
        if self.is_realized():
            return super().is_tensor()
        self._ensure_operand_type_guards()
        return False

    def _might_realize_symbolic(self) -> bool:
        # [FUNCTION DOC] HELPER: True if any operand is (or may become, given the int/float
        # specialization config) symbolic, so the result might not be a plain constant.
        # Mirrors LazyConstantVariable._maybe_realize_for_type
        from .tensor import SymNodeVariable

        for lazy_var in self._cache.lazy_vars:
            if lazy_var.is_realized():
                if type.__instancecheck__(SymNodeVariable, lazy_var.realize()):
                    return True
            else:
                value_type = lazy_var.peek_type()
                if value_type is int and not config.specialize_int:
                    return True
                if value_type is float and not config.specialize_float:
                    return True
        return False

    def get_handler_type_for_dispatch(self) -> type:
        # [FUNCTION DOC] Returns the constant tracker class (installing operand type
        # guards) when unrealized and no operand may turn symbolic; otherwise realizes
        # and returns the real tracker type.
        from .constant import ConstantVariable

        if not self.is_realized() and not self._might_realize_symbolic():
            self._ensure_operand_type_guards()
            return ConstantVariable
        return type(self.realize())

    def lazy_isinstance(self, cls: type) -> bool:
        # [FUNCTION DOC] Instance check against a tracker class, avoiding realization when
        # the answer is knowable: realized uses the real tracker; a class unrelated to
        # constants and symnodes gives False; possibly symbolic operands force
        # realization; otherwise answers for the constant tracker class.
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        if self.is_realized():
            return type.__instancecheck__(cls, self.realize())
        if not issubclass(ConstantVariable, cls) and not issubclass(
            SymNodeVariable, cls
        ):
            self._ensure_operand_type_guards()
            return False
        if self._might_realize_symbolic():
            return type.__instancecheck__(cls, self.realize())
        self._ensure_operand_type_guards()
        return issubclass(ConstantVariable, cls)

    def reconstruct(self, codegen: Any) -> None:
        # [FUNCTION DOC] Emits bytecode that recomputes the value at runtime: loads the
        # operator function by name from the operator module, pushes each argument
        # tracker, and calls it.
        # KNOWN ISSUE: relies on the operator's name existing in the operator module.
        from ..bytecode_transformation import create_call_function

        cache = self._cache
        codegen.add_push_null(
            lambda: codegen.load_import_from("operator", cache.op.__name__)
        )
        for arg in cache.args:
            codegen(arg)
        codegen.extend_output(create_call_function(len(cache.args), False))


class LazySymNodeFormatString:
    # [CLASS DOC] Deferred formatting of a symbolic number: holds the symnode tracker
    # and a constant tracker for the full format string built from a spec. Its string
    # form evaluates the symbolic expression only when rendered.
    def __init__(
        self, sym_node_variable: SymNodeVariable, fmt_spec_var: VariableTracker
    ) -> None:
        # [FUNCTION DOC] Stores the symnode tracker and builds a constant format string
        # wrapping the python-constant spec; raises if the spec is not a constant.
        from .constant import ConstantVariable

        self.sym_node_var = sym_node_variable
        self.fmt_var = ConstantVariable.create(
            "{:" + fmt_spec_var.as_python_constant() + "}"
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Evaluates the symbolic expression to a concrete value (may add
        # guards) and returns it rendered with the stored format.
        # KNOWN ISSUE: the evaluated value is converted to a string before formatting, so
        # numeric format specs (for example a float precision) would fail or misbehave.
        return str.format(
            self.fmt_var.as_python_constant(),
            str(self.sym_node_var.evaluate_expr()),
        )


def _create_realize_and_forward(
    name: str,
) -> Callable[[LazyVariableTracker, Any, Any], Any]:
    # [FUNCTION DOC] HELPER: Builds a method that realizes a lazy tracker and calls the
    # named base-tracker method on the real tracker, preserving the original metadata.
    @functools.wraps(getattr(VariableTracker, name))
    def realize_and_forward(
        self: LazyVariableTracker, *args: Any, **kwargs: Any
    ) -> Any:
        # [FUNCTION DOC] HELPER: Realizes self, then forwards the call with all arguments.
        return getattr(self.realize(), name)(*args, **kwargs)

    return realize_and_forward


def _populate() -> None:
    # [FUNCTION DOC] HELPER: Import-time setup. For every callable attribute of the base
    # tracker class not already defined on the lazy class, installs a realizing forwarder.
    # Mutates the lazy class.
    for name, value in VariableTracker.__dict__.items():
        if name not in LazyVariableTracker.__dict__:
            if callable(value):
                setattr(LazyVariableTracker, name, _create_realize_and_forward(name))


_populate()
