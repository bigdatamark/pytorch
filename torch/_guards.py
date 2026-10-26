# [MODULE DOC] Definitional home of general purpose guard and tracing-context structures
# shared by dynamo, AOT autograd, shape env and inductor: compile/trace ids, guard sources
# and Guard objects, checkpointable guard/module/global contexts, per-thread CompileContext
# and TracingContext (with the compile_context and tracing context managers), the HOP
# invoke_subgraph reuse caches, the Source base classes and fake mode detection helpers.
# Layering rule: no dynamo or guard-installation logic belongs here. Import time: creates
# the module logger, the compile id regexes and the per-thread storage object _TLS.
from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import logging
import re
import sys
import threading
import traceback
import unittest.mock
import weakref
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, overload, TYPE_CHECKING, TypeVar
from typing_extensions import dataclass_transform

import torch
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._traceback import CapturedTraceback, format_frame
from torch.utils.weak import WeakTensorKeyDictionary


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    import dis
    from collections.abc import Callable, Generator, Iterator
    from types import CodeType

    import sympy

    from torch._dynamo.backends.distributed import DDPOptimizerContext
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.guards import GuardCheckSpec
    from torch._dynamo.output_graph import CodeOptions
    from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta
    from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
    from torch._subclasses.fake_tensor import FakeTensorMode


"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""

COMPILE_ID_PATTERN = re.compile(r"^(?P<frame_id>\d+)/(?P<frame_compile_id>\d+)$")
CA_COMPILE_ID_PATTERN = re.compile(
    r"^!(?P<compiled_autograd_id>\d+)(?:/(?P<frame_id>\d+)/(?P<frame_compile_id>\d+))?$"
)

# [Note: Updating CompiledId]
#
# CompiledId represents a unique program-level identifier, and we want to keep that
# property as the codebase evolves. This property is relied on even outside of the pytorch
# repo, e.g. tlparse or other internal tooling. The in-memory format can be freely changed,
# as those dependencies only consume the string serialization.
#
# The string form should be:
# 1. Program-level uid: CompileId can uniquely identify a compiled graph.
# 2. Storage efficient: This object is logged in nearly every entry. We should elide symbols when possible.
# 3. Compact: The string form is directly displayed by some tools. Special symbols are okay.


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileId:
    # [CLASS DOC] Frozen identifier of one compilation: frame_id, per-frame
    # frame_compile_id, and an optional compiled_autograd_id for compiled autograd graphs.
    # Its string form (see __str__ and from_string) is a stable format consumed by external
    # tools such as tlparse, so it must stay in sync with them.
    frame_id: int | None
    # This id is per-frame, and counts how many times we've compiled this
    # frame.  This could have been a global id but having this be per-frame
    # gives you a better intuitive sense for how many recompiles have occurred
    # so far.
    frame_compile_id: int | None

    # torch.compiling a compiled autograd graph
    compiled_autograd_id: int | None = None

    # TODO: consider also tracking the recompilation count
    # See Note: Updating CompileId

    def __str__(self) -> str:
        # [FUNCTION DOC] Serializes the id. With a compiled autograd id the form is a bang
        # followed by that id, optionally followed by slash frame id slash compile id (both
        # frame fields must be set together, else AssertionError). Without it, returns
        # frame id slash frame compile id and raises AssertionError if either is None.
        # NOTE: Keep this in sync with both from_string and the tlparse repo
        if self.compiled_autograd_id is not None:
            if (self.frame_id is None) != (self.frame_compile_id is None):
                raise AssertionError(
                    f"frame_id and frame_compile_id must both be None or both be set, "
                    f"got frame_id={self.frame_id}, frame_compile_id={self.frame_compile_id}"
                )
            frame_str = ""
            if self.frame_id is not None:
                frame_str = f"/{self.frame_id}/{self.frame_compile_id}"

            return f"!{self.compiled_autograd_id}{frame_str}"
        else:
            if self.frame_id is None or self.frame_compile_id is None:
                raise AssertionError(
                    f"frame_id and frame_compile_id must not be None, "
                    f"got frame_id={self.frame_id}, frame_compile_id={self.frame_compile_id}"
                )
            return f"{self.frame_id}/{self.frame_compile_id}"

    @classmethod
    def from_string(cls, compile_id: str | None) -> CompileId | None:
        # [FUNCTION DOC] Parses the string form back into a CompileId; None input gives
        # None. Tries the plain and compiled-autograd patterns, converting captured numbers
        # to integers. Raises ValueError (chained) if no pattern matches or parsing fails.
        """
        Factory method that creates a CompileId from its string representation.
        Keep this in sync with the __str__ method.
        """
        if compile_id is None:
            return None
        try:
            for pattern in (COMPILE_ID_PATTERN, CA_COMPILE_ID_PATTERN):
                if match := pattern.match(compile_id):
                    groups = match.groupdict()
                    for k, v in groups.items():
                        if v is not None:
                            groups[k] = int(v)
                    return cls(**groups)  # type: ignore[arg-type]
            else:
                raise ValueError

        except Exception as e:
            raise ValueError(f"Invalid compile_id '{compile_id}'") from e


class TraceId(NamedTuple):
    # [CLASS DOC] Pair of a CompileId and an attempt counter that increments each time
    # analysis restarts for the same compile.
    compile_id: CompileId
    # This starts off as 0, and every time we restart analysis it goes
    # up by one
    attempt: int

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the compile id string for attempt zero, otherwise the
        # compile id string followed by an underscore and the attempt number.
        # Keep this in sync with tlparse repo
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f"{self.compile_id}_{self.attempt}"


class GuardSource(enum.Enum):
    # [CLASS DOC] Enumeration of where a guarded value originates (local or global scope,
    # specialized/unspecialized/FSDP nn modules, constants, shape env, etc.). Provides
    # predicate methods that classify members into groups.
    LOCAL = 0
    GLOBAL = 1
    LOCAL_SPECIALIZED_NN_MODULE = 2
    GLOBAL_SPECIALIZED_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8
    BACKWARD_STATE = 9
    EPHEMERAL = 10
    SYNTHETIC_LOCAL = 11
    LOCAL_UNSPECIALIZED_NN_MODULE = 12
    GLOBAL_UNSPECIALIZED_NN_MODULE = 13
    LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 14
    GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 15
    TEMP_LOCAL = 16

    def is_fsdp_module(self) -> bool:
        # [FUNCTION DOC] True if this is the global or local FSDP module source.
        return self in (GuardSource.GLOBAL_FSDP_MODULE, GuardSource.LOCAL_FSDP_MODULE)

    def is_specialized_nn_module(self) -> bool:
        # [FUNCTION DOC] True for global/local specialized nn module sources. If the dynamo
        # config flag to skip FSDP module guards is on, FSDP module sources also count.
        import torch._dynamo.config as config

        if config._unsafe_skip_fsdp_module_guards:
            return (
                self
                in (
                    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
                    GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
                )
                or self.is_fsdp_module()
            )
        return self in (
            GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
        )

    def is_unspecialized_nn_module(self) -> bool:
        # [FUNCTION DOC] True for global/local unspecialized nn module sources, including
        # the unspecialized builtin variants.
        return self in (
            GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )

    def is_unspecialized_builtin_nn_module(self) -> bool:
        # [FUNCTION DOC] True only for the global/local unspecialized builtin nn module sources.
        return self in (
            GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )

    def is_local(self) -> bool:
        # [FUNCTION DOC] True for local scope sources (plain, specialized nn module, FSDP,
        # unspecialized and unspecialized builtin nn module variants).
        # KNOWN ISSUE: SYNTHETIC_LOCAL and TEMP_LOCAL are not counted as local.
        return self in (
            GuardSource.LOCAL,
            GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_FSDP_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )


"""
Base class for a "GuardBuilder" role.

The GuardBuilderBase role is to represent a scope within which to build a guard. The name is a little
confusing, as it's not a builder, but for the sake of avoiding a lot of renames and keeping the original reference
to torchdynamo's GuardBuilder.

Note: create_fn is invoked with a GuardBuilderBase and a Guard. A GuardBuilder is chosen based
on GuardSource's select function.

There is value in keeping this GuardBuilderBase empty to keep layering clean.
"""


class GuardBuilderBase:
    # [CLASS DOC] Intentionally empty base class for the scope in which guards are built;
    # dynamo's GuardBuilder subclasses it. Kept empty to preserve layering.
    pass


@dataclasses.dataclass(frozen=True)
class SLoc:
    # [CLASS DOC] Frozen source location record: a framework location (frame summary, string
    # or None) and an optional user location string. Used to describe where a shape guard
    # came from.
    framework_loc: traceback.FrameSummary | str | None
    maybe_user_loc: str | None

    def __str__(self) -> str:
        # [FUNCTION DOC] Formats the location: the framework part (string as is, empty for
        # None, otherwise a formatted frame) in parentheses, prefixed by the user location
        # and a space when one exists.
        floc = (
            self.framework_loc
            if isinstance(self.framework_loc, str)
            else ""
            if self.framework_loc is None
            else format_frame(self.framework_loc)
        )
        if self.maybe_user_loc is not None:
            return f"{self.maybe_user_loc} ({floc})"
        else:
            return f"({floc})"


class ShapeGuard(NamedTuple):
    # [CLASS DOC] A symbolic shape guard: a sympy boolean expression, where it was created
    # (SLoc), and whether it was evaluated size-obliviously.
    expr: sympy.logic.boolalg.Boolean
    sloc: SLoc
    size_oblivious: bool


@dataclasses.dataclass(slots=True)
class Guard:
    # [CLASS DOC] A single guard: the originating Source, a create_fn (optionally a partial)
    # that emits the guard given a GuardBuilderBase and this guard, plus export-only data
    # (guard type names, code strings, weakrefs to the guarded object and class), creation
    # stack, user stack and a cached hash. Hash and ordering use name, source and create_fn.
    # originating_source is the source that called the make_guard method to
    # construct this guard object. The property name specifies what exactly it
    # is the guard is guarding on.  The meaning of the name is dependent on the
    # create_fn; you must look at the use-site inside create_fn to know what
    # name means.
    #
    # That being said, although you might think this is just a "name", name is
    # usually an arbitrary Python expression that will be evaluated with all
    # globals (and locals, if you create a LOCAL guard) to extract the Python
    # object that we want to perform guard tests on.  This evaluation
    # typically happens in GuardBuilder.eval.  In these cases, name is
    # typically produced by originating_source.name (not to be confused with
    # GuardSource - the property source).
    #
    # Occasionally, name is not a valid Python expression; sometimes
    # it is meaningless.  Example create_fns that are like this include
    # GRAD_MODE and SHAPE_ENV.
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]

    # Export only. These values are written to at time of guard check_fn creation.
    guard_types: list[str] | None = None
    code_list: list[str] | None = None
    obj_weakref: object | None = None
    guarded_class_weakref: weakref.ReferenceType[Any] | None = None

    stack: CapturedTraceback | None = None
    user_stack: traceback.StackSummary | None = None
    _hash: int | None = None
    _unserializable: bool = False
    _force_dict_keys_match: bool = False

    def __hash__(self) -> int:
        # [FUNCTION DOC] Computes and caches a hash of the name, guard source and identity of
        # the create function.
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self) -> tuple[bool, int, int, str, int]:
        # [FUNCTION DOC] Returns a tuple for deterministic ordering: duplicate-input guards
        # last, then guard source value (minus one if none), name length, name, and the first
        # line number of the underlying create function.
        # Put the duplicate input guards at the end. The duplicate guards have
        # two sources while guard.name only considers one source.

        is_duplicate_input = (
            isinstance(self.create_fn, functools.partial)
            and self.create_fn.func is torch._dynamo.guards.GuardBuilder.DUPLICATE_INPUT
        )
        return (
            is_duplicate_input,
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.inner_create_fn().__code__.co_firstlineno,
        )

    def __lt__(self, other: Guard) -> bool:
        # [FUNCTION DOC] Orders guards by their sort keys.
        return self.sort_key() < other.sort_key()

    def inner_create_fn(self) -> Callable[[GuardBuilderBase, Guard], Any]:
        # [FUNCTION DOC] Returns the wrapped function if create_fn is a partial, else create_fn.
        if isinstance(self.create_fn, functools.partial):
            return self.create_fn.func
        else:
            return self.create_fn

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Returns the name (expression string) of the originating source.
        return self.originating_source.name

    @property
    def source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind of the originating source.
        return self.originating_source.guard_source

    @staticmethod
    def weakref_to_str(obj_weakref: object) -> str:
        # [FUNCTION DOC] Safe string for a weakref (avoids str() on the weakref, which can
        # fail for objects overriding attribute access): live refs show the ref and target
        # class with addresses, dead refs say dead, non-weakrefs use plain str.
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f"<weakref at {hex(id(obj_weakref))}; dead>"
        else:
            return str(obj_weakref)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Multi-line debug representation: source, name, create function name,
        # and a block with guard types, code list, object weakref and guarded class.
        s = f"""
        {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.inner_create_fn().__name__}
        {{
            'guard_types': {self.guard_types},
            'code': {self.code_list},
            'obj_weakref': {self.weakref_to_str(self.obj_weakref)}
            'guarded_class': {self.guarded_class_weakref}
        }}
        """
        return s

    def __str__(self) -> str:
        # [FUNCTION DOC] Human readable multi-line description: name, source, create function,
        # guard types, code list and the weakrefs.
        output = f"Name:{repr(self.name)}\n"
        source = self.source.name.lower() if self.source else ""
        output += f"    Source: {source}\n"
        output += f"    Create Function: {self.inner_create_fn().__name__}\n"
        output += f"    Guard Types: {self.guard_types}\n"
        output += f"    Code List: {self.code_list}\n"
        output += f"    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n"
        output += f"    Guarded Class Weakref: {self.guarded_class_weakref}\n"
        return output

    def create(self, builder: GuardBuilderBase) -> Any:
        # [FUNCTION DOC] Runs create_fn with the builder and this guard and returns its result.
        # On any exception, logs the guard description and (if recorded) the last frames of
        # the creation stack, then re-raises.
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.exception("Error while creating guard:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise

    def is_specialized_nn_module(self) -> bool:
        # [FUNCTION DOC] Delegates to the guard source.
        return self.source.is_specialized_nn_module()

    def is_fsdp_module(self) -> bool:
        # [FUNCTION DOC] Delegates to the guard source.
        return self.source.is_fsdp_module()

    def is_local(self) -> bool:
        # [FUNCTION DOC] Delegates to the guard source.
        return self.source.is_local()

    def create_fn_name(self) -> str:
        # [FUNCTION DOC] Returns the __name__ of create_fn, looking through a partial.
        if isinstance(self.create_fn, functools.partial):
            create_fn = self.create_fn.func  # type: ignore[attr-defined]
        else:
            create_fn = self.create_fn
        return create_fn.__name__

    def set_export_info(
        self,
        guard_type: str,
        guarded_class: weakref.ReferenceType[Any] | None,
        code_list: list[str],
        obj_weakref: object,
    ) -> None:
        # [FUNCTION DOC] Records export information: appends guard_type, checks the guarded
        # class weakref matches any earlier one (or is unset), extends the code list, and
        # checks the object weakref matches any earlier one, is unset, or is dead (ephemeral
        # objects). Mutates this guard. Raises AssertionError on a class or object mismatch.
        if not self.guard_types:
            self.guard_types = []

        self.guard_types.append(guard_type)

        if self.guarded_class_weakref not in (guarded_class, None):
            raise AssertionError(
                f"Guarded class id must be identical, or None, "
                f"got {self.guarded_class_weakref} vs {guarded_class}"
            )
        self.guarded_class_weakref = guarded_class

        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)

        # Some objects are ephemeral, e.g., list[slice(1, 2)]. If we have
        # multiple guards on the same object, the weakref can die between the
        # invocation of set_export_info calls. So a dead weakref is also
        # acceptable.
        is_valid = (
            self.obj_weakref in (obj_weakref, None)
            or callable(self.obj_weakref)
            and self.obj_weakref() is None
        )
        if not is_valid:
            raise AssertionError(
                f"Guarded object must be identical, None or ephemeral (dead weakref), "
                f"got {self.obj_weakref} vs {obj_weakref}"
            )
        self.obj_weakref = obj_weakref


T = TypeVar("T")

"""
Parent structure for guard env expressions.
A GuardEnvExpr can have any subtype.
Note: All subtypes must be handled exhaustively in
torch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.
"""


@dataclasses.dataclass(frozen=True)
class GuardEnvExpr:
    # [CLASS DOC] Empty frozen base class for guard environment expressions (duplicate
    # inputs, storage overlap); all subtypes must be handled in dynamo's guard env parsing.
    pass


"""
A class representing a pair of duplicate inputs.
input_pos_a and input_pos_b are input positions we have deduped.
"""


@dataclasses.dataclass(frozen=True)
class DuplicateInputs(GuardEnvExpr):
    # [CLASS DOC] Records that two input sources were deduplicated because they are the same
    # object. Invariant: the two sources differ.
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates that the two sources differ; raises AssertionError otherwise.
        if self.input_source_a == self.input_source_b:
            raise AssertionError(
                f"input_source_a and input_source_b must be different, "
                f"got {self.input_source_a}"
            )


"""
A class representing storage overlap relations among inputs that aliases the same storage.

Given that a set of tensors alias the same storage, this guard checks whether they actually
have overlapping storages.

While non_overlapping_sources represent input tensors that definitely don't have any storage
overlapping with any other input, overlapping_sources represent tensors that either:

1. Do overlap some other input tensor
2. Might not overlap some other input tensor, but we are not sure
"""


@dataclasses.dataclass(frozen=True)
class StorageOverlap(GuardEnvExpr):
    # [CLASS DOC] Storage overlap facts for inputs sharing a storage: sources that overlap
    # (or might overlap) another input, and sources known not to overlap any other.
    overlapping_sources: list[Source]
    non_overlapping_sources: list[Source]


"""
Checkpointable is an interface for driving state snapshotting, left purposely vague for now.

copy_graphstate() -> T, a somewhat legacy name, is expected to emit a snapshot of any type that
can also be taken in at restore_graphstate(T) calls.

When to snapshot, is, at the moment, an implementation detail of upstream callers. Checkpointable
does not provide any guarantees around consistency, idempotency, or safety of calling its APIs, yet.

In the future, it will have a closer coupling to a generic Checkpoint management system.
"""


class Checkpointable(Generic[T]):
    # [CLASS DOC] Abstract interface for objects whose state can be snapshotted with
    # copy_graphstate (returns a state of type T) and restored with restore_graphstate.
    # No consistency guarantees are given.
    @abstractmethod
    def copy_graphstate(self) -> T: ...

    @abstractmethod
    def restore_graphstate(self, state: T) -> None: ...


class GuardsCheckpointState:
    """
    The GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext
    """
    # [CLASS DOC] Snapshot of a GuardsContext: an ordered set of dynamo guards. Equal to
    # another snapshot when it has no guards missing from the other.

    dynamo_guards: OrderedSet[Guard]

    def __init__(self, dynamo_guards: OrderedSet[Guard]) -> None:
        # [FUNCTION DOC] Stores the (not copied) set of guards.
        self.dynamo_guards = dynamo_guards

    def diff(self, other: GuardsCheckpointState) -> OrderedSet[Guard] | None:
        # [FUNCTION DOC] Returns guards present here but not in other, or None if none.
        """
        Produces a delta against another GuardsCheckpointState.

        Returns None if no delta is found, otherwise, return an OrderedSet() of mismatched
        Guard type objects.
        """
        r = self.dynamo_guards.difference(other.dynamo_guards)
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] False for other types; otherwise true when diff against other is empty.
        # KNOWN ISSUE: the comparison is one-directional; a snapshot that is a subset of
        # the other compares equal to it even though they differ.
        if not isinstance(other, GuardsCheckpointState):
            return False
        return self.diff(other) is None


class ModuleContextCheckpointState:
    # [CLASS DOC] Snapshot of a ModuleContext: mapping of names to nn modules. The class
    # level default mapping is shared but instances always assign their own.
    nn_modules: dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules: dict[str, torch.nn.Module]) -> None:
        # [FUNCTION DOC] Stores the (not copied) mapping.
        self.nn_modules = nn_modules

    def diff(self, other: ModuleContextCheckpointState) -> set[str] | None:
        # [FUNCTION DOC] Returns module names present here but not in other, or None if none.
        """
        Produces a delta against another ModuleContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        module key names.
        """
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] False for other types; otherwise true when no module names are
        # missing from other (compares keys only, one direction).
        if not isinstance(other, ModuleContextCheckpointState):
            return False
        return self.diff(other) is None


class ModuleContext(Checkpointable[ModuleContextCheckpointState]):
    # [CLASS DOC] Tracks the nn modules (by name) seen during tracing; checkpointable.
    def __init__(self) -> None:
        # [FUNCTION DOC] Starts with an empty name to module mapping.
        self.nn_modules: dict[str, Any] = {}

    def copy_graphstate(self) -> ModuleContextCheckpointState:
        # [FUNCTION DOC] Returns a snapshot holding a shallow copy of the module mapping.
        return ModuleContextCheckpointState(dict(self.nn_modules))

    def restore_graphstate(self, state: ModuleContextCheckpointState) -> None:
        # [FUNCTION DOC] Replaces the module mapping with the snapshot's mapping (not
        # copied). Raises AssertionError if state is not a ModuleContextCheckpointState.
        if not isinstance(state, ModuleContextCheckpointState):
            raise AssertionError(
                f"expected ModuleContextCheckpointState, got {type(state)}"
            )
        self.nn_modules = state.nn_modules


class GlobalContextCheckpointState:
    # [CLASS DOC] Snapshot of a GlobalContext: mapping from a global-state name to a
    # (setter function, saved argument) pair used to restore that state.
    global_state: dict[str, tuple[Callable, Any]] = {}

    def __init__(self, global_states: dict[str, tuple[Callable, Any]]) -> None:
        # [FUNCTION DOC] Stores the (not copied) mapping.
        self.global_state = global_states

    def diff(self, other: GlobalContextCheckpointState) -> set[str] | None:
        # [FUNCTION DOC] Returns state names present here but not in other, or None if none.
        """
        Produces a delta against another GlobalContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        global key names.
        """
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] False for other types; otherwise true when no state names are
        # missing from other (compares keys only, one direction).
        if not isinstance(other, GlobalContextCheckpointState):
            return False
        return self.diff(other) is None


class GlobalContext(Checkpointable[GlobalContextCheckpointState]):
    """
    This keeps track of the global torch state during tracing of a function.
    For example, torch.is_grad_enabled.
    """
    # [CLASS DOC] Holds saved global torch state (grad mode and autocast settings) as a
    # mapping of name to (setter, value). Only the names in _supported_global_states are
    # valid; restoring replays every setter.

    _supported_global_states = {
        "grad_enabled",
        "autocast_enabled",
        "autocast_cpu_enabled",
        "autocast_gpu_dtype",
        "autocast_cpu_dtype",
        "autocast_cache_enabled",
    }

    def __init__(self) -> None:
        # [FUNCTION DOC] Starts with an empty state mapping.
        self.global_state: dict[str, tuple[Callable, Any]] = {}

    def copy_graphstate(self) -> GlobalContextCheckpointState:
        # [FUNCTION DOC] Returns a snapshot sharing (not copying) the current mapping.
        return GlobalContextCheckpointState(self.global_state)

    def restore_graphstate(self, state: GlobalContextCheckpointState) -> None:
        # [FUNCTION DOC] Adopts the snapshot's mapping, checks its keys equal the supported
        # names, then calls each stored setter with its saved argument (side effect: changes
        # real global torch state). Raises AssertionError on a wrong type or key mismatch.
        if not isinstance(state, GlobalContextCheckpointState):
            raise AssertionError(
                f"expected GlobalContextCheckpointState, got {type(state)}"
            )
        self.global_state = state.global_state
        if not (
            len(self.global_state) == len(self._supported_global_states)
            and set(self.global_state.keys()) == self._supported_global_states
        ):
            raise AssertionError(
                f"Global state mismatch: got keys {set(self.global_state.keys())}, "
                f"expected {self._supported_global_states}"
            )
        for func, args in self.global_state.values():
            func(args)


# Like a Set[Guard] but will record the user stack on all guards at the
# time they were installed at their destination
class GuardsSet:
    # [CLASS DOC] Ordered set of guards that stamps creation and user stacks on guards when
    # added, and keeps an index from originating source to its guards. Supports iteration,
    # length, truthiness, set difference (returns a new GuardsSet), clear and removal by source.
    def __init__(self, inner: OrderedSet[Guard] | None = None) -> None:
        # [FUNCTION DOC] Uses the given ordered set (or a new empty one) and builds the
        # source to guards index from its current members.
        if inner is None:
            self.inner: OrderedSet[Guard] = OrderedSet()
        else:
            self.inner = inner
        # Map from source to list of guards with that source. Incrementally
        # built in `add` method.
        self.source_to_guards: defaultdict[Source, list[Guard]] = defaultdict(list)
        for guard in self.inner:
            self.track_guard_by_source(guard)

    def track_guard_by_source(self, guard: Guard) -> None:
        # [FUNCTION DOC] Appends the guard to the index list for its originating source.
        source = guard.originating_source
        self.source_to_guards[source].append(guard)

    def __iter__(self) -> Iterator[Guard]:
        # [FUNCTION DOC] Iterates guards in insertion order.
        return iter(self.inner)

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of guards.
        return len(self.inner)

    # Subtraction along with bool is typically used to determine the delta of
    # added guards between checkpoints for higher order ops
    def __sub__(self, other: GuardsSet) -> GuardsSet:
        # [FUNCTION DOC] Returns a new GuardsSet of guards here that are not in other.
        return GuardsSet(self.inner - other.inner)

    def __bool__(self) -> bool:
        # [FUNCTION DOC] True when there is at least one guard.
        return bool(self.inner)

    def clear(self) -> None:
        # [FUNCTION DOC] Removes all guards and resets the source index.
        self.inner = OrderedSet()
        self.source_to_guards = defaultdict(list)

    def add(
        self, guard: Guard, *, collect_debug_stack: bool = True, skip: int = 0
    ) -> None:
        # [FUNCTION DOC] Adds a guard unless already present. Mutates the guard: records a
        # captured creation stack (skipping skip extra frames) if requested and unset, and a
        # user stack from the current tracing context if unset. Also updates the source index.
        if guard in self.inner:
            return
        if collect_debug_stack:
            if guard.stack is None:
                guard.stack = CapturedTraceback.extract(skip=1 + skip)
        if guard.user_stack is None:
            guard.user_stack = TracingContext.extract_stack()
        self.inner.add(guard)
        self.track_guard_by_source(guard)

    def update(self, *others: set[Guard]) -> None:
        # [FUNCTION DOC] Adds every guard from each given collection via add.
        for o in others:
            for g in o:
                self.add(g, skip=1)

    def get_guards_for_source(self, source: Source) -> list[Guard]:
        # [FUNCTION DOC] Returns a copy of the list of guards indexed under the source.
        # Note: a lookup of a missing source inserts an empty entry (defaultdict).
        """Return all guards with the given originating_source."""
        return list(self.source_to_guards[source])

    def remove_guards_with_source(self, source: Source) -> None:
        # [FUNCTION DOC] Drops all guards whose originating source is derived from the given
        # source (via the chain of bases) and rebuilds the source index.
        """Delete all guards that contains a given source"""
        from ._dynamo.source import is_from_source

        self.inner = OrderedSet(
            g for g in self.inner if not is_from_source(g.originating_source, source)
        )
        # Rebuild the index since is_from_source walks the chain, so
        # multiple source keys may need removal.
        self.source_to_guards = defaultdict(list)
        for guard in self.inner:
            self.track_guard_by_source(guard)


"""
A GuardsContext is a checkpointable representation of all the guards in the current tracing
context. Its lifecycle is bound 1:1 to the tracing context, and it should never be instantiated
directly outside of it. For passing around internal state representations of this object,
prefer to extract them with copy_graphstate to produce a GuardsCheckpointState.
"""


class GuardsContext(Checkpointable[GuardsCheckpointState]):
    # [CLASS DOC] Checkpointable collection of all guards in a tracing context: dynamo guards
    # (GuardsSet), AOT autograd guard env expressions, and a flag to skip guard installation.
    def __init__(self) -> None:
        # [FUNCTION DOC] Starts with no guards, no env expressions and installation enabled.
        self.dynamo_guards: GuardsSet = GuardsSet()
        self.aotautograd_guards: list[GuardEnvExpr] = []
        self.skip_install: bool = False

    @contextlib.contextmanager
    def skip_guard_install(self) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that sets skip_install for its body and restores the
        # previous value on exit, even on error.
        old = self.skip_install
        self.skip_install = True
        try:
            yield
        finally:
            self.skip_install = old

    def copy_graphstate(self) -> GuardsCheckpointState:
        # [FUNCTION DOC] Returns a snapshot holding a copy of the current dynamo guard set.
        return GuardsCheckpointState(OrderedSet(self.dynamo_guards.inner))

    def restore_graphstate(self, state: GuardsCheckpointState) -> None:
        # [FUNCTION DOC] Replaces the dynamo guards with a new GuardsSet built on the
        # snapshot's set (which is taken over, not copied). Raises AssertionError if state
        # has the wrong type.
        # NB: "steals" the passed in state
        if not isinstance(state, GuardsCheckpointState):
            raise AssertionError(f"expected GuardsCheckpointState, got {type(state)}")
        self.dynamo_guards = GuardsSet(state.dynamo_guards)


class HopSubgraphCache:
    # [CLASS DOC] Abstract interface of a per-HOP cache used across tracing: installed
    # dynamo submodule names by code object, autograd and proxy dispatch entries by
    # identifier, functionalization schemas, and lazily built backward graphs keyed by
    # tangent metadata. Methods are abstract one-line stubs; InvokeSubgraphCache implements them.
    @abstractmethod
    def add_dynamo_installed_submodule(
        self, fn_code: CodeType, identifier: str
    ) -> None: ...

    @abstractmethod
    def get_dynamo_installed_submodules(self, fn_code: CodeType) -> list[str]: ...

    @abstractmethod
    def add_autograd_key_entry(self, identifier: str, key: Callable) -> None: ...

    @abstractmethod
    def get_autograd_key_entry(self, identifier: str) -> Callable | None: ...

    @abstractmethod
    def add_proxy_dispatch_entry(self, identifier: str, key: Callable) -> None: ...

    @abstractmethod
    def get_proxy_dispatch_entry(self, identifier: str) -> Callable | None: ...

    @abstractmethod
    def add_functionalize_schema_entry(
        self, key: object, schema: torch._C.FunctionSchema
    ) -> None: ...

    @abstractmethod
    def get_functionalize_schema_entry(
        self, key: object
    ) -> torch._C.FunctionSchema | None: ...

    @abstractmethod
    def add_lazy_bwd_entry(
        self,
        identifier: str,
        tangent_metadata: tuple[object],
        gmod: torch.fx.GraphModule,
    ) -> int: ...

    @abstractmethod
    def get_lazy_bwd_entry(
        self, identifier: str, tangent_metadata: tuple[object]
    ) -> tuple[torch.fx.GraphModule | None, int | None]: ...


@dataclass
class InvokeSubgraphReuseEntry:
    # [CLASS DOC] Cached result of tracing an invoke_subgraph body: graph name and module,
    # compile options, input mapping, single-output flag, cached output tensor metadata,
    # per-argument sources (for source rewriting on a cache hit) and the count of
    # user-visible outputs (graph may return extra side-effect values).
    body_name: str
    body_gmod: torch.fx.GraphModule
    config: NestedCompileRegionOptions | None
    subgraph_input_mapping: list[
        Any
    ]  # list[LiftedArgOrigin] - defined in invoke_subgraph.py
    single_tensor_output: bool
    # Per-output tensor metadata (shape, stride, dtype, device, requires_grad)
    # cached from the first trace so we can construct fresh FakeTensors on
    # cache hit without re-running body_gmod.
    output_metadata: list[
        tuple[torch.Size, tuple[int, ...], torch.dtype, torch.device, bool]
    ]
    # 1-1 mapping to flat_vts: source for each flattened arg/kwarg, or None if
    # the VT has no source. On cache hit, we build a source replacement mapping
    # (old arg sources → new arg sources) to rewrite captured variable sources
    # for the current invocation.
    arg_sources: list[Source | None]
    # Number of user-visible outputs (from the function return value).
    # The graph may have additional outputs from side-effect intermediates;
    # stamp_out_subgraph uses this to return only the user-visible slice.
    num_user_outputs: int = 0


@dataclass
class InvokeSubgraphReuseCondition:
    # [CLASS DOC] Conditions under which a cached subgraph trace may be reused: per-input
    # checks (tensor metadata, symbolic node, constant or object tags), guards captured during
    # the trace with their handlers and expected values, the expected call treespec, and all
    # sources touched while tracing (to detect mutation of captured variables).
    # Per flattened input VT: (InputTag, metadata).
    #   (InputTag.TENSOR, TensorMetadata)
    #   (InputTag.SYMNODE, sym_num — same object implies same symbol)
    #   (InputTag.CONSTANT, value)
    #   (InputTag.OBJECT, None)
    # Tensor metadata is checked here because TENSOR_MATCH guards for
    # subgraph inputs may already exist before tracing and thus won't
    # appear in the guard delta.
    input_checks: list[tuple[Any, object]]  # list[tuple[InputTag, object]]

    # Guards captured during the trace (delta + source-mapped).
    # Each entry: (source, handler, expected_value, guard)
    # handler is a pre-resolved GuardCheckSpec from GUARD_VALUE_DISPATCH.
    guards: list[tuple[Source, GuardCheckSpec, object, Guard]]

    # TreeSpec from pytree.tree_flatten of the (args, kwargs) structure.
    # On cache hit, we verify the new call has the same treespec.
    treespec: pytree.TreeSpec | None = None

    # All sources accessed via VariableBuilder during the subgraph trace.
    # On cache hit, we check if any modified VT's source is a base of one
    # of these to detect mutations on captured variables.
    traced_sources: OrderedSet[Source] = dataclasses.field(default_factory=OrderedSet)


class InvokeSubgraphCache(HopSubgraphCache):
    # [CLASS DOC] Concrete cache for the invoke_subgraph HOP. Holds mappings for autograd and
    # proxy dispatch callables, functionalization schemas, installed dynamo submodules, lazy
    # backward graphs, effect sets, and two reuse caches keyed by function code: a list of
    # (condition, entry) pairs searched in order, and a hash-key mapping for user-supplied hashes.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates all the cache mappings empty.
        self.autograd_cache: dict[str, Callable] = {}
        self.proxy_dispatch_cache: dict[str, Callable] = {}
        self.functionalize_schema_cache: dict[object, torch._C.FunctionSchema] = {}
        self.dynamo_installed_submodules: dict[CodeType, list[str]] = defaultdict(list)
        self.lazy_bwd_cache: dict[
            str, dict[tuple[object], tuple[torch.fx.GraphModule, int]]
        ] = defaultdict(dict)
        self.effects_cache: dict[
            str, set
        ] = {}  # Maps identifier -> set of effect types
        # fn.__code__ → list of (condition, cache_entry) pairs. Walked linearly
        # on lookup; first matching condition wins.
        self.subgraph_reuse_cache: dict[
            CodeType,
            list[tuple[InvokeSubgraphReuseCondition, InvokeSubgraphReuseEntry]],
        ] = defaultdict(list)
        # fn_code → {hash_key → cache_entry}. Used by user-provided
        # reuse_hash_fn for O(1) subgraph reuse lookup.
        self.subgraph_reuse_key_cache: dict[
            CodeType, dict[int, InvokeSubgraphReuseEntry]
        ] = defaultdict(dict)

    def add_dynamo_installed_submodule(
        self, fn_code: CodeType, identifier: str
    ) -> None:
        # [FUNCTION DOC] Records a submodule identifier installed by dynamo for fn_code.
        self.dynamo_installed_submodules[fn_code].append(identifier)

    def get_dynamo_installed_submodules(self, fn_code: CodeType) -> list[str]:
        # [FUNCTION DOC] Returns identifiers recorded for fn_code, or an empty list.
        return self.dynamo_installed_submodules.get(fn_code, [])

    def add_autograd_key_entry(self, identifier: str, key: Callable) -> None:
        # [FUNCTION DOC] Stores (overwrites) the autograd dispatch callable for identifier.
        self.autograd_cache[identifier] = key

    def get_autograd_key_entry(self, identifier: str) -> Callable | None:
        # [FUNCTION DOC] Returns the stored autograd callable or None.
        return self.autograd_cache.get(identifier, None)

    def add_proxy_dispatch_entry(self, identifier: str, key: Callable) -> None:
        # [FUNCTION DOC] Stores (overwrites) the proxy dispatch callable for identifier.
        self.proxy_dispatch_cache[identifier] = key

    def get_proxy_dispatch_entry(self, identifier: str) -> Callable | None:
        # [FUNCTION DOC] Returns the stored proxy dispatch callable or None.
        return self.proxy_dispatch_cache.get(identifier, None)

    def add_functionalize_schema_entry(
        self, key: object, schema: torch._C.FunctionSchema
    ) -> None:
        # [FUNCTION DOC] Stores (overwrites) the functionalization schema under key.
        self.functionalize_schema_cache[key] = schema

    def get_functionalize_schema_entry(
        self, key: object
    ) -> torch._C.FunctionSchema | None:
        # [FUNCTION DOC] Returns the stored schema for key or None.
        return self.functionalize_schema_cache.get(key, None)

    def add_lazy_bwd_entry(
        self,
        identifier: str,
        tangent_metadata: tuple[object],
        gmod: torch.fx.GraphModule,
    ) -> int:
        # [FUNCTION DOC] Stores a backward graph module for the identifier under the given
        # tangent metadata and returns its index (the count of entries stored before it),
        # used as a name suffix. Re-adding the same metadata overwrites but still gets a
        # new index based on the current count.
        # Save the number of existing graph modules in the dictionary to get the suffix
        num_gmods = len(self.lazy_bwd_cache[identifier])
        self.lazy_bwd_cache[identifier][tangent_metadata] = (gmod, num_gmods)
        return num_gmods

    def get_lazy_bwd_entry(
        self, identifier: str, tangent_metadata: tuple[object]
    ) -> tuple[torch.fx.GraphModule | None, int | None]:
        # [FUNCTION DOC] Looks up the stored (graph module, index) pair for the identifier and
        # tangent metadata; returns a pair of Nones when missing, without creating entries.
        if identifier not in self.lazy_bwd_cache:
            return (None, None)

        return self.lazy_bwd_cache[identifier].get(tangent_metadata, (None, None))

    def add_effects(self, identifier: str, effects: set) -> None:
        # [FUNCTION DOC] Records the effect set for identifier. If a non-empty set was
        # recorded before and differs, raises AssertionError; otherwise stores the new set.
        """Store the effect types for a given invoke_subgraph identifier."""
        if prev_effects := self.effects_cache.get(identifier, None):
            if effects != prev_effects:
                raise AssertionError(
                    "Different number of effects were found for invoke_subgraph "
                    f"call with identifier {identifier}. \n"
                    f"Previously we had the following effects: {prev_effects}.\n"
                    f"But now we have: {effects}."
                )
        self.effects_cache[identifier] = effects

    def get_effects(self, identifier: str) -> set | None:
        # [FUNCTION DOC] Returns the recorded effect set for identifier, or None.
        """Retrieve the effect types for a given invoke_subgraph identifier."""
        return self.effects_cache.get(identifier, None)

    def add_reuse_entry(
        self,
        fn_code: CodeType,
        condition: InvokeSubgraphReuseCondition,
        entry: InvokeSubgraphReuseEntry,
        max_reuse_entries: int = 8,
    ) -> None:
        # [FUNCTION DOC] Appends a (condition, entry) pair for fn_code. Raises RuntimeError
        # (with debugging advice) if there are already max_reuse_entries pairs for it.
        entries = self.subgraph_reuse_cache[fn_code]
        if len(entries) >= max_reuse_entries:
            raise RuntimeError(
                f"invoke_subgraph: exceeded maximum reuse entries "
                f"({max_reuse_entries}) for function code {fn_code}. "
                f"This most likely means a guard keeps failing on every "
                f"invocation, preventing subgraph reuse. "
                f"Set TORCH_LOGS='+hierarchical_compile' to identify which "
                f"guard is failing. If reuse is genuinely not possible and "
                f"you need more cache entries, increase the limit via the "
                f"max_reuse_entries argument to nested_compile_region()."
            )
        entries.append((condition, entry))

    def find_reuse_entry(
        self,
        fn_code: CodeType,
        evaluator: Callable[
            [InvokeSubgraphReuseCondition, InvokeSubgraphReuseEntry], bool
        ],
    ) -> InvokeSubgraphReuseEntry | None:
        # [FUNCTION DOC] Walks the pairs for fn_code in order, calling the evaluator on each
        # (condition, entry); returns the first entry for which it is true, after moving that
        # pair to the front of the list (most recently used first). Returns None if none match.
        entries = self.subgraph_reuse_cache.get(fn_code, [])
        for i, (condition, entry) in enumerate(entries):
            if evaluator(condition, entry):
                # MRU: move the hit entry to the front for faster future lookups
                if i > 0:
                    entries.insert(0, entries.pop(i))
                return entry
        return None

    def find_reuse_entry_by_key(
        self, fn_code: CodeType, hash_key: int
    ) -> InvokeSubgraphReuseEntry | None:
        # [FUNCTION DOC] Returns the entry stored for fn_code under hash_key, or None.
        return self.subgraph_reuse_key_cache.get(fn_code, {}).get(hash_key)

    def add_reuse_entry_by_key(
        self,
        fn_code: CodeType,
        hash_key: int,
        entry: InvokeSubgraphReuseEntry,
        max_reuse_entries: int = 8,
    ) -> None:
        # [FUNCTION DOC] Stores entry for fn_code under hash_key. Raises RuntimeError if the
        # cache already has max_reuse_entries keys and hash_key is new; replacing an existing
        # key is allowed.
        key_cache = self.subgraph_reuse_key_cache[fn_code]
        if len(key_cache) >= max_reuse_entries and hash_key not in key_cache:
            raise RuntimeError(
                f"invoke_subgraph: exceeded maximum reuse entries "
                f"({max_reuse_entries}) for function code {fn_code} (hash-key path). "
                f"Increase the limit via the max_reuse_entries argument to "
                f"nested_compile_region()."
            )
        key_cache[hash_key] = entry


class HopDispatchSetCache:
    # [CLASS DOC] Registry mapping supported higher-order ops to their subgraph cache
    # (currently only invoke_subgraph with an InvokeSubgraphCache).
    def __init__(self) -> None:
        # [FUNCTION DOC] Builds the op to cache mapping (imports invoke_subgraph lazily to
        # avoid a circular import).
        # Delayed import to avoid circular dependency
        from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

        self.hop_cache_map = {invoke_subgraph: InvokeSubgraphCache()}

    def get_cache(self, op: torch._ops.HigherOrderOperator) -> HopSubgraphCache | None:
        # [FUNCTION DOC] Returns the cache registered for the op, or None if unsupported.
        if op not in self.hop_cache_map:
            return None
        return self.hop_cache_map[op]  # type: ignore[index]


class _TLSStorage(threading.local):
    # [CLASS DOC] HELPER: Thread-local holder for the current tracing context and compile
    # context, both defaulting to None in every thread.
    # Default the hot-path attributes to None per thread so that
    # TracingContext.try_get() / CompileContext.try_get() -- called on every
    # torch.compile'd call -- hit a present attribute instead of paying for an
    # AttributeError raise+catch inside getattr(). Without this, a thread that
    # never ran compilation itself (e.g. a worker thread executing compiled code
    # compiled on another thread) takes the slow getattr-miss path on every call.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Runs per thread on first access; sets both contexts to None.
        self.tracing_context: TracingContext | None = None
        self.compile_context: CompileContext | None = None


_TLS = _TLSStorage()

"""
TracingContext is the source of truth for all currently accumulated information
needed to trace. Its lifecycle is kept 1:1 when using TorchDynamo, but other systems
are open to managing their own TracingContext with that in mind.

The purpose of TracingContext is not to be a dumping ground, or god object, but rather to avoid
having to plumb complex subsystems across multiple verticals.

Ex: A common example is guard accumulation between dynamo, shape_env, aot_autograd, and inductor.
Accessing the current tracing context via
TracingContext.get() allows users to accumulate their own guards for processing, without needing to know how
to plumb objects back up to where frame interpretation happened.

Note that you can end up with multiple TracingContext for a single compilation
of a frame, as we reset the TracingContext whenever we restart analysis.
CompileContext is a more overarching context that encompasses multiple restarts.
"""


class CompileContext:
    # [CLASS DOC] Per-thread context spanning all analysis restarts of one compile: optional
    # CompileId, attempt counter (starts at zero) and a list of verbose shape env guard strings.
    @staticmethod
    def get() -> CompileContext:
        # [FUNCTION DOC] Returns the thread's current compile context; raises AssertionError
        # if none is set.
        if _TLS.compile_context is None:
            raise AssertionError("compile_context is not set")
        return _TLS.compile_context

    @staticmethod
    def try_get() -> CompileContext | None:
        # [FUNCTION DOC] Returns the thread's current compile context, or None if unset.
        return getattr(_TLS, "compile_context", None)

    def __init__(self, compile_id: CompileId | None) -> None:
        # [FUNCTION DOC] Stores the compile id (None or CompileId, else AssertionError),
        # sets attempt to zero and the shape env guards list to empty.
        if compile_id is not None and not isinstance(compile_id, CompileId):
            raise AssertionError(
                f"compile_id must be None or CompileId, got {type(compile_id)}"
            )
        self.compile_id: CompileId | None = compile_id
        self.attempt = 0
        # Verbose ShapeEnv guards produced.
        self.shape_env_guards: list[str] = []

    @staticmethod
    def current_compile_id() -> CompileId | None:
        # [FUNCTION DOC] Returns the current context's compile id, or None if no context.
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id() -> TraceId | None:
        # [FUNCTION DOC] Returns a TraceId (compile id plus attempt) for the current context,
        # or None if there is no context or no compile id.
        self = CompileContext.try_get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)


@dataclass
class InlinedCodeCache:
    # [CLASS DOC] Record of data derived from a code object for inlining: its instruction
    # list, an instruction to index mapping, and the code options.
    """Cache for code-object-derived data used during inlining."""

    instructions: list[Any]
    indexof: dict[Any, int]
    code_options: CodeOptions


class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """
    # [CLASS DOC] Per-compile accumulator of tracing state shared across dynamo, shape env,
    # AOT autograd and inductor: guards, module and global contexts, fake mode, the frame
    # summary stack and current location (for user stacks), inlining caches, AOT metadata,
    # output strides, HOP caches and traced code objects. The active instance is installed
    # per thread by the tracing context manager and read via get or try_get.

    @staticmethod
    def try_get() -> TracingContext | None:
        # [FUNCTION DOC] Returns the thread's installed tracing context, or None.
        return getattr(_TLS, "tracing_context", None)

    @staticmethod
    def get() -> TracingContext:
        # [FUNCTION DOC] Returns the installed tracing context; raises RuntimeError if none.
        if ctx := TracingContext.try_get():
            return ctx
        raise RuntimeError(
            "TracingContext.get() must be called within an ongoing trace."
        )

    def __init__(self, fake_mode: FakeTensorMode | None) -> None:
        # [FUNCTION DOC] Creates fresh guards, module and global contexts, empty caches and
        # stacks, all optional metadata unset, flags false, and a new HOP dispatch cache;
        # stores the given fake mode.
        self.guards_context = GuardsContext()
        self.module_context = ModuleContext()
        self.global_context = GlobalContext()
        self.previously_inlined_functions: dict[Any, Any] = dict()
        self.previously_cleaned_instructions: dict[Any, Any] = dict()
        # Combined cache for inlined code data (instructions, indexof, code_options)
        self.inlined_code_cache: dict[Any, InlinedCodeCache] = dict()
        self.fake_mode: FakeTensorMode | None = fake_mode
        self.frame_summary_stack: list[traceback.FrameSummary] = []
        # This is morally part of frame_summary_stack, but it is kept separate
        # for clarity.  As we process a frame, this variable gets updated
        # to keep track of what line we are in the function.  We make a
        # function call, this gets cleared and the frame location is pushed
        # to frame_summary_stack (prepping this variable for the inner frame's
        # progress)
        self.loc_in_frame: tuple[str, int, str] | None = None
        self.loc_in_frame_positions: dis.Positions | None = None
        # this is only set after aot_autograd
        self.fw_metadata: ViewAndMutationMeta | None = None
        # this is only set when the DDPOptimizer is used
        self.ddp_optimizer_ctx: DDPOptimizerContext | None = None
        # this is only set after aot_autograd
        self.aot_graph_name: list[str] | None = None
        self.params_flat: list[Any] | None = None
        self.params_flat_unwrap_subclasses: list[Any] | None = None
        self.params_unwrapped_to_flat_index: list[Any] | None = None
        # this is for extended return calling convention from backend
        # compiler to aot_autograd
        # Per output, what the compiler specified stride of the output is,
        # or None if no stride is known.  This is always the HINT, it
        # is never a SymInt (it would be better if it was a SymInt, but
        # I can't conveniently get this from Inductor atm.  Also, be
        # careful not to accidentally induce guards on the SymInt if
        # you ever do change this in aot_autograd.py; you should check
        # on permutations preferentially.)
        self.output_strides: list[tuple[int, ...] | None] | None = None
        # When this is True, whenever we encounter an int in Dynamo tracing,
        # we will (1) force unspec it and (2) force it as a size-like unbacked
        # integer.  This is currently used when processing certain lists of
        # ints that are known to be size-like and may have 0/1 entries that we
        # must not specialize on.
        self.force_unspec_int_unbacked_size_like = False
        # See note [Tensor Fakification and Symbol Caching]
        self.tensor_to_context = WeakTensorKeyDictionary()

        # If this true, Aot Autograd will return output Fake Tensors with appropriate
        # meta on the first invocation
        # see note: [Returning Fake Tensors on First AOT Autograd Call]
        self.fakify_first_call = False
        self.hop_dispatch_set_cache = HopDispatchSetCache()
        # list of code objects for inlined functions
        self.traced_code: list[CodeType] = []
        self.cudagraph_annotation: Any = None

    def clear(self) -> None:
        # [FUNCTION DOC] Resets the saved global state mapping and clears the inlining
        # related caches (mutates in place). Other fields such as guards are untouched.
        # Look at the note in output_graph.py in function `save_global_state`
        # for the context on clearing global context.
        self.global_context.global_state = {}
        self.previously_inlined_functions.clear()
        self.previously_cleaned_instructions.clear()
        self.inlined_code_cache.clear()

    @staticmethod
    @contextmanager
    def patch(**kwargs: Any) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that temporarily sets the named attributes on the
        # current tracing context (raises RuntimeError if none, AttributeError for an unknown
        # name) and restores the prior values on exit, even on error.
        prior = {}
        ctx = TracingContext.get()

        for key in kwargs:
            # KeyError on invalid entry
            prior[key] = getattr(ctx, key)
        for key, val in kwargs.items():
            setattr(ctx, key, val)
        try:
            yield
        finally:
            for key, val in prior.items():
                setattr(ctx, key, val)

    @staticmethod
    def extract_stack() -> traceback.StackSummary:
        # [FUNCTION DOC] Returns the user-level stack as a StackSummary: the frame summary
        # stack plus the current location in the innermost frame if known. Empty when no
        # tracing context is installed.
        self = TracingContext.try_get()
        if self is None:
            return traceback.StackSummary()
        stack = self.frame_summary_stack
        if self.loc_in_frame is not None:
            stack = stack + [self._populate_loc_in_frame_summary()]
        return traceback.StackSummary.from_list(stack)

    def _populate_loc_in_frame_summary(self) -> traceback.FrameSummary:
        # [FUNCTION DOC] HELPER: Builds a FrameSummary (without reading source lines) from the
        # stored file, line and frame name, adding column positions on Python 3.11+ when
        # known. Raises AssertionError if no location is stored.
        if self.loc_in_frame is None:
            raise AssertionError("loc_in_frame must not be None")
        filename, lineno, frame_name = self.loc_in_frame
        # colno/end_colno kwargs were added to FrameSummary in 3.11
        kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 11) and self.loc_in_frame_positions is not None:
            kwargs["colno"] = self.loc_in_frame_positions.col_offset
            kwargs["end_colno"] = self.loc_in_frame_positions.end_col_offset
        return traceback.FrameSummary(
            filename,
            lineno,
            frame_name,
            lookup_line=False,
            **kwargs,
        )

    # Call this when you want to call into some code that isn't necessarily
    # associated with the current frame state
    @staticmethod
    @contextlib.contextmanager
    def clear_frame() -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager that temporarily empties the frame summary stack and
        # current location (patched on the current tracing context, which must exist). If the
        # body raises an exception without a real_stack attribute, sets it to None so outer
        # frames do not attach a misleading stack, then re-raises.
        tc = TracingContext.get()
        with (
            unittest.mock.patch.object(tc, "frame_summary_stack", []),
            unittest.mock.patch.object(tc, "loc_in_frame", None),
            unittest.mock.patch.object(tc, "loc_in_frame_positions", None),
        ):
            try:
                yield
            except Exception as e:
                # Prevent real_stack from getting attached
                #
                # The invariant is that if an Exception as real_stack, we've
                # appropriately attached a user stack and we no longer need to
                # attach anything. Because we cannot conveniently interpose
                # when an exception is thrown, we instead interpose everywhere
                # we set what the user stack is set (using the context
                # manager). However, our compiler stack does "tail calls"
                # (when it calls into user compiler), at which point the
                # parent exception frames would incorrectly attach an
                # incorrect frame.
                #
                # However, if, somehow, someone raised an exception with this
                # scope that had a stack (for example, because they are
                # restoring the user stack state appropriately as they process
                # node by node), we should respect it. Thus, we cannot
                # unconditionally set None.
                if not hasattr(e, "real_stack"):
                    e.real_stack = None  # type: ignore[attr-defined]
                raise

    @staticmethod
    @contextlib.contextmanager
    def current_frame(
        frame_summary: traceback.FrameSummary | None,
    ) -> Generator[None, None, None]:
        # [FUNCTION DOC] Context manager entering a user frame: pushes frame_summary on the
        # stack if given and clears the current location. On an exception lacking real_stack,
        # attaches the extracted stack, then re-raises. On exit pops the frame and restores
        # the previous location and positions. Requires an installed tracing context.
        # frame_summary can be None to solely take advantage of real_stack
        # attachment to thrown exceptions
        tc = TracingContext.get()
        if frame_summary is not None:
            tc.frame_summary_stack.append(frame_summary)
        old = tc.loc_in_frame
        old_positions = tc.loc_in_frame_positions
        tc.loc_in_frame = None
        tc.loc_in_frame_positions = None
        try:
            yield
        except Exception as e:
            if not hasattr(e, "real_stack"):
                e.real_stack = tc.extract_stack()  # type: ignore[attr-defined]
            raise
        finally:
            if frame_summary is not None:
                tc.frame_summary_stack.pop()
            tc.loc_in_frame = old
            tc.loc_in_frame_positions = old_positions

    @staticmethod
    @contextlib.contextmanager
    def report_output_strides() -> Generator[
        list[tuple[int, ...] | None] | None, None, None
    ]:
        # [FUNCTION DOC] Context manager that collects output strides reported during its
        # body. Yields None if there is no tracing context; otherwise installs a fresh list
        # as the context's output strides, yields it, and restores the previous value on exit.
        tc = TracingContext.try_get()
        if tc is None:
            yield None
            return
        old_output_strides = tc.output_strides
        tc.output_strides = []
        try:
            yield tc.output_strides
        finally:
            tc.output_strides = old_output_strides

    @staticmethod
    def set_current_loc(
        filename: str,
        lineno: int,
        frame_name: str,
        positions: dis.Positions | None = None,
    ) -> None:
        # [FUNCTION DOC] Records the current file, line and frame name (and optional column
        # positions) on the installed tracing context; raises RuntimeError if none.
        # Save the current location in the frame. Lazily generate the
        # framesummary.
        tc = TracingContext.get()
        tc.loc_in_frame = (filename, lineno, frame_name)
        tc.loc_in_frame_positions = positions

    @staticmethod
    def get_traced_code() -> list[CodeType] | None:
        # [FUNCTION DOC] Returns the code objects traced so far, or None without a context.
        tc = TracingContext.try_get()
        if tc is None:
            return None
        return tc.traced_code


@contextmanager
def compile_context(
    context: CompileContext | None,
) -> Generator[CompileContext | None, None, None]:
    # [FUNCTION DOC] Context manager installing the given compile context (may be None) on
    # the current thread, yielding it, and restoring the previous one on exit.
    old_context = getattr(_TLS, "compile_context", None)
    _TLS.compile_context = context
    try:
        yield context
    finally:
        _TLS.compile_context = old_context


@contextmanager
def tracing(
    context: TracingContext | None,
) -> Generator[TracingContext | None, None, None]:
    """
    This function installs the passed in tracing context as a dynamic scoped
    global variable.

    Calls to TracingContext.get() while not under a `with tracing()` context
    will return None.
    """
    # [FUNCTION DOC] Installs the given tracing context on the current thread for the body.
    # On an exception lacking real_stack (and with a context), attaches the extracted user
    # stack and re-raises. On exit, cleans up the context's shape env if it has one and
    # restores the previous context.
    old_context = getattr(_TLS, "tracing_context", None)
    _TLS.tracing_context = context
    try:
        yield context
    except Exception as e:
        if not hasattr(e, "real_stack") and context is not None:
            e.real_stack = context.extract_stack()  # type: ignore[attr-defined]
        raise
    finally:
        if (
            context is not None
            and context.fake_mode is not None
            and context.fake_mode.shape_env is not None
        ):
            context.fake_mode.shape_env.cleanup()
        _TLS.tracing_context = old_context


@overload
def dataclass_with_cached_hash(cls: type[T], **kwargs: Any) -> type[T]: ...


@overload
def dataclass_with_cached_hash(
    cls: None = None, **kwargs: Any
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform()
def dataclass_with_cached_hash(
    cls: type[T] | None = None, **kwargs: Any
) -> type[T] | Callable[[type[T]], type[T]]:
    # [FUNCTION DOC] Class decorator (usable with or without arguments) that applies the
    # standard dataclass decorator with the given options, then replaces the hash with one
    # that computes the original hash once and caches it on the instance, and replaces
    # pickling so the cached hash is excluded. Returns the decorated class, or a decorator
    # when called without a class.
    def wrap(cls_inner: type[T]) -> type[T]:
        # [FUNCTION DOC] HELPER: Applies the dataclass decorator and patches hash and
        # reduce onto the resulting class, which is returned.
        new_cls = dataclasses.dataclass(cls_inner, **kwargs)
        old_hash = cls_inner.__hash__

        def __hash__(self) -> int:
            # [FUNCTION DOC] HELPER: Returns the cached hash, computing and storing it (by
            # bypassing frozen attribute protection) on first use.
            if not hasattr(self, "_hash"):
                object.__setattr__(self, "_hash", old_hash(self))
            return self._hash

        def __reduce__(self):
            # [FUNCTION DOC] HELPER: Pickle support that rebuilds the object from the class
            # and its init field values only, leaving out the cached hash.
            # Exclude _hash from pickling to ensure deterministic cache keys.
            # The _hash is a cached value that can be nondeterministically computed
            # (e.g., based on id() of objects), so it should not affect pickling.
            fields = dataclasses.fields(self)
            field_values = tuple(getattr(self, f.name) for f in fields if f.init)
            return (self.__class__, field_values)

        new_cls.__hash__ = __hash__
        new_cls.__reduce__ = __reduce__
        return new_cls  # type: ignore[return-value]

    if cls is None:
        return wrap

    return wrap(cls)


# Subclasses can be found in torch/_dynamo/source.py
# TODO(voz): Consider a toplevel torch/_source.py
@dataclass_with_cached_hash(frozen=True)
class Source:
    # [CLASS DOC] Base class for descriptions of where a traced value comes from. Frozen with
    # cached hash. Subclasses (in dynamo source module) supply the name template, guard
    # source and code reconstruction; the base provides value lookup by evaluating the
    # name, guard creation, and cloning.
    def is_dict_key(self) -> bool:
        # [FUNCTION DOC] False by default; true for dictionary key sources in subclasses.
        return False

    def is_ephemeral(self) -> bool:
        # [FUNCTION DOC] False by default; true for sources that do not persist.
        return False

    def reconstruct(self, codegen: PyCodegen) -> None:
        # [FUNCTION DOC] Emits bytecode that rebuilds the source value; subclasses must
        # override. The base raises NotImplementedError.
        raise NotImplementedError

    def reconstruct_pycode(self, codegen: PyCodegen) -> str:
        # [FUNCTION DOC] Returns Python source text rebuilding the value; subclasses should
        # override. The base raises NotImplementedError.
        """
        Reconstructs the source into a string of Python code. This method should
        be eventually implemented for all subclasses of Source to achieve full
        coverage of python wrapper code generation.
        """
        raise NotImplementedError

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] The kind of guard source; subclasses must override (base raises
        # NotImplementedError).
        raise NotImplementedError

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Template string for the source expression; subclasses must
        # override (base raises NotImplementedError). Chained sources use a format string
        # with a placeholder for the base name.
        """
        A template for the name of the source. Used to prevent code duplication between
        `name` and `get_value`.

        For non-ChainedSources, `name` and `get_value` use the returned string directly.

        For ChainedSources, `name` and `get_value` expect the return to be a format string
        with `{0}` present - `name` and `get_value` will apply different values to this function's
        returned format string.
        """
        raise NotImplementedError

    @functools.cached_property
    def name(self) -> str:
        # [FUNCTION DOC] Cached expression string for this source (the template itself).
        return self._name_template

    def get_value(
        self,
        globals: dict[str, Any],
        locals: dict[str, Any],
        cache: dict[Source, Any],
    ) -> Any:
        # [FUNCTION DOC] Returns the value this source refers to by evaluating the name
        # template with the given globals and locals, memoizing in the cache mapping
        # (mutated). Evaluation errors from the expression propagate.
        if self in cache:
            return cache[self]
        value = eval(self._name_template, globals, locals)
        cache[self] = value
        return value

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        # [FUNCTION DOC] Creates a Guard for this source with the given create function.
        # Raises NotImplementedError for constant guard sources.
        if self.guard_source is GuardSource.CONSTANT:
            raise NotImplementedError
        return Guard(self, fn)

    def is_specialized_nn_module(self) -> bool:
        # [FUNCTION DOC] Delegates to the guard source.
        return self.guard_source.is_specialized_nn_module()

    def subguards_allowed(self) -> bool:
        # [FUNCTION DOC] True unless the guard source is synthetic local.
        """True if you can guard on attributes of this"""
        return self.guard_source != GuardSource.SYNTHETIC_LOCAL

    def clone(self, transform_fn: Callable[[Source], Source] | None = None) -> Source:
        # [FUNCTION DOC] Returns transform_fn applied to this source if given, else this
        # same (immutable) source.
        # Frozen dataclass, so returning self is effectively a clone.
        # Subclasses with mutable state should override.
        if transform_fn is not None:
            return transform_fn(self)
        return self


# Subclasses can be found in torch/_dynamo/source.py
@dataclass_with_cached_hash(frozen=True)
class ChainedSource(Source):
    # [CLASS DOC] Source derived from another source (attribute, index, etc.). Holds the base
    # source; guard source, dict-key and ephemeral properties come from the base, and the name
    # is the template formatted with the base name.
    base: Source

    def is_dict_key(self) -> bool:
        # [FUNCTION DOC] Delegates to the base source.
        # Recurse until you either hit a ConstDictKey or a Source
        return self.base.is_dict_key()

    def is_ephemeral(self) -> bool:
        # [FUNCTION DOC] Delegates to the base source.
        return self.base.is_ephemeral()

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Same guard source as the base.
        return self.base.guard_source

    def get_base(self) -> Source:
        # [FUNCTION DOC] Follows base links to the root, non-chained source and returns it.
        current: Source = self
        while isinstance(current, ChainedSource):
            current = current.base
        return current

    @functools.cached_property
    def name(self) -> str:
        # [FUNCTION DOC] Cached expression: the template formatted with the base's name.
        return self._name_template.format(self.base.name)

    def get_value(
        self,
        globals: dict[str, Any],
        locals: dict[str, Any],
        cache: dict[Source, Any],
    ) -> Any:
        # [FUNCTION DOC] Evaluates the base source's value (recursively, memoized in cache),
        # temporarily binds it to a fresh local variable name that does not clash with
        # existing locals, evaluates the template formatted with that name, removes the
        # temporary, caches and returns the value. Mutates locals only transiently.
        if self in cache:
            return cache[self]
        tmpvar = "tmp"
        counter = 0
        while tmpvar in locals:
            tmpvar = f"tmp{counter}"
            counter += 1
        locals[tmpvar] = self.base.get_value(globals, locals, cache)
        value = eval(self._name_template.format(tmpvar), globals, locals)
        del locals[tmpvar]
        cache[self] = value
        return value

    def clone(self, transform_fn: Callable[[Source], Source] | None = None) -> Source:
        # [FUNCTION DOC] Returns a copy with the base and any other Source-valued fields
        # cloned recursively with transform_fn; transform_fn is then applied to the result.
        cloned_fields: dict[str, Any] = {"base": self.base.clone(transform_fn)}
        for f in dataclasses.fields(self):
            if f.name == "base":
                continue
            val = getattr(self, f.name)
            if isinstance(val, Source):
                cloned_fields[f.name] = val.clone(transform_fn)
        result = dataclasses.replace(self, **cloned_fields)
        if transform_fn is not None:
            result = transform_fn(result)
        return result


def detect_fake_mode(inputs: Any = None) -> FakeTensorMode | None:
    # [FUNCTION DOC] Finds the fake mode in effect: the tracing context's fake mode wins if
    # set; else collects modes from active dispatch mode stack entries, from fake tensor
    # leaves of inputs (a pytree, flattened here) and from fake tensors inside traceable
    # wrapper subclasses. Returns the first, or None if none. Raises AssertionError if the
    # discovered modes are not all the same object.
    """
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    """
    from torch._subclasses.fake_tensor import (
        FakeTensor,
        FakeTensorMode,
        get_plain_tensors,
        is_fake_tensor,
        maybe_get_fake_mode,
    )

    # If TracingContext has a fake_mode, use it authoritatively.
    # This is the case when Dynamo is driving compilation - any fake tensors
    # from other modes in the inputs will be refakified by the caller.
    if context := TracingContext.try_get():
        fake_mode = context.fake_mode
        if fake_mode is not None:
            return fake_mode

    fake_modes = []

    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for i, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            fake_modes.append((m, "active fake mode", i))

    flat_inputs = pytree.tree_leaves(inputs)
    for i, flat_input in enumerate(flat_inputs):
        if is_fake_tensor(flat_input):
            fake_modes.append((maybe_get_fake_mode(flat_input), "fake tensor input", i))
        if is_traceable_wrapper_subclass(flat_input):
            out: list[torch.Tensor | int | torch.SymInt] = []
            get_plain_tensors(flat_input, out=out)  # type: ignore[arg-type]
            fake_tensors: list[FakeTensor] = [
                x
                for x in out
                if isinstance(x, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
            ]
            fake_modes.extend(
                [
                    (maybe_get_fake_mode(tensor), f"subclass input {i}", ix)
                    for ix, tensor in enumerate(fake_tensors)
                ]
            )

    if fake_modes:
        fake_mode, desc1, i1 = fake_modes[0]
        for m, desc2, i2 in fake_modes[1:]:
            if fake_mode is not m:
                raise AssertionError(
                    f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                    f"fake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\n"
                    f"fake mode from {desc2} {i2} allocated at:\n{m.stack}"
                )

        return fake_mode
    else:
        return None


def active_fake_mode() -> FakeTensorMode | None:
    # [FUNCTION DOC] Returns the innermost FakeTensorMode on the dispatch mode stack, or
    # None if none is active.
    """
    Inspects the dispatch mode stack for an active fake mode and returns it.
    Returns None if no fake mode is active.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for _, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            return m

    return None
