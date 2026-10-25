# [MODULE DOC] Defines the Source class hierarchy used by Dynamo to describe where a
# traced value originates (locals, globals, attributes, items, etc.). Sources drive
# guard generation (via guard_source and a name template that renders a guard-
# expression string) and code reconstruction (reconstruct emits bytecode that
# re-creates the value). Also holds the guard-source remapping tables for nn.Module
# specialization and FSDP, and many is_from_* / get_*_source_name chain-walking
# helpers (cached with lru_cache). No import-time side effects beyond defining
# classes/tables (the dataclass decorator generates frozen, hash-cached dataclasses).
"""
This module provides Source classes that track the origins of values in PyTorch Dynamo.
Sources represent where values come from (e.g. local variables, globals, attributes) and
are used for guard generation and code reconstruction during compilation.

The module includes specialized sources for:
- Local variables and synthetic locals
- Global variables and constants
- Object attributes and method calls
- NN module specialization (specialized vs unspecialized)
- Random values and tensor properties
- Default argument handling
- FSDP (Fully Sharded Data Parallel) modules

Sources play a key role in Dynamo's guard system by tracking value origins for
guard generation, and in code reconstruction by providing methods to rebuild
the code needed to recreate values.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from torch import device as device_type
from torch._guards import (
    ChainedSource,
    dataclass_with_cached_hash,
    Guard,
    GuardSource,
    Source,
)

from . import utils
from .bytecode_transformation import (
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
)


if TYPE_CHECKING:
    from .codegen import PyCodegen

# It shouldn't be supported to construct an NNModuleVariable inside an FSDP module,
# so those cases are omitted intentionally

# represents nn.Modules tracked with NNModuleVariable (specialized is implicit in the variable name)
_GUARD_SOURCE_SPECIALIZED_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

# represents nn.Modules tracked with UnspecializedNNModuleVariable
_GUARD_SOURCE_UNSPECIALIZED_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    # this happens for an UnspecializedNNModule submodule on a NNModuleVariable
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

# represents nn.Modules tracked with UnspecializedBuiltinNNModuleVariable
_GUARD_SOURCE_UNSPECIALIZED_BUILTIN_NN_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    # Just to ensure that guard_source() works
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}

_GUARD_SOURCE_FSDP_MODULE = {
    GuardSource.LOCAL: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_SPECIALIZED_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_FSDP_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_FSDP_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
    GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.LOCAL_FSDP_MODULE,
    GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE: GuardSource.GLOBAL_FSDP_MODULE,
}


def is_constant_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source is a ConstantSource instance or its
    # guard_source is the CONSTANT kind. A guard_source lookup that raises
    # NotImplementedError is treated as "not constant". Returns False otherwise.
    if isinstance(source, ConstantSource):
        return True
    try:
        if source.guard_source == GuardSource.CONSTANT:
            return True
    except NotImplementedError:
        pass

    return False


def _get_source_debug_name(source: Source | None) -> str:
    # [FUNCTION DOC] HELPER: Returns the source's name for debug output, or the
    # placeholder text for an unknown source if the source is missing or its name
    # lookup raises NotImplementedError.
    if source is None:
        return "<unknown source>"
    else:
        try:
            return source.name
        except NotImplementedError:
            return "<unknown source>"


def _esc_str(s: Any, apply_repr: bool = False) -> str:
    # [FUNCTION DOC] HELPER: Converts a value to a string (using repr when apply_repr
    # is set, str otherwise) and doubles every curly brace so the result can be
    # embedded safely inside a format-string template. Returns the escaped string.
    """
    Escapes curly brackets for format strings.
    e.g. "frozenset({0})" becomes "frozenset({{0}})".
    This is used by _name_template for example, because it's
    expected to return a format string, but we may wish to include
    strings that should not be accidentally formatted.
    """
    if apply_repr:
        s = repr(s)
    else:
        s = str(s)
    return s.replace("{", "{{").replace("}", "}}")


@dataclass_with_cached_hash(frozen=True)
class LocalSource(Source):
    # [CLASS DOC] Root source for a local variable of the frame being traced.
    # Frozen dataclass. Fields: local_name; is_input (local is an input of the root
    # frame); dynamism (optional frozenset describing known-dynamic input dims);
    # is_derefed_cell_contents (value is the content of a cell var/free var);
    # is_varargs / is_varkw (local is the *args / **kwargs parameter; hidden from
    # repr). Guard source is LOCAL; the name renders as a lookup in the locals map.
    local_name: str

    # Whether this local is an input to the root frame.
    is_input: bool = False

    # Whether we know this input is dynamic (based on example_inputs)
    # For non tensors, we simply look at the first index of the tuple
    dynamism: frozenset[str] | None = None

    # Whether the item at this source is the _content_ of a cell that is
    # dereferenced from the root frame, i.e., it's a part of the `co_cellvars`
    # or `co_freevars`.
    is_derefed_cell_contents: bool = False

    # Whether this local is the function's varargs (``*args``) parameter.
    # Set from ``co_flags & CO_VARARGS`` at frame-entry time. Element accesses
    # like ``args[N]`` produce a ``GetItemSource`` whose base has this flag.
    # Useful for distinguishing ``*args`` from a regular list-typed input.
    # ``repr=False`` so the vast majority of locals (which are not varargs) do
    # not get a noisy ``is_varargs=False`` in every debug string.
    is_varargs: bool = dataclasses.field(default=False, repr=False)

    # Whether this local is the function's varkw (``**kwargs``) parameter.
    # Set from ``co_flags & CO_VARKEYWORDS`` at frame-entry time. Element
    # accesses like ``kwargs["k"]`` produce a ``DictGetItemSource`` whose base
    # has this flag. ``repr=False`` for the same reason as ``is_varargs``.
    is_varkw: bool = dataclasses.field(default=False, repr=False)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that pushes this local onto the stack: a
        # deref load if the value is cell contents, otherwise a plain local load.
        # Side effect: appends instructions to the codegen output.
        if self.is_derefed_cell_contents:
            codegen.load_deref(self.local_name)
        else:
            codegen.append_output(codegen.create_load(self.local_name))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        # [FUNCTION DOC] Returns the Python expression text for this source, which is
        # just the local variable name. The codegen argument is unused.
        return self.local_name

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind LOCAL.
        return GuardSource.LOCAL

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the guard-name template, a lookup of the
        # repr of the local name in the locals map, with braces escaped. Cached.
        return f"L[{_esc_str(self.local_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class TempLocalSource(Source):
    # [CLASS DOC] Like LocalSource but for temporary locals that must never be guarded
    # on. Field: local_name. Can be reconstructed (loaded) but asking for its guard
    # name raises an error.
    # like LocalSource, but cannot be guarded on
    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a bytecode load of the named local onto the stack.
        codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind TEMP_LOCAL.
        return GuardSource.TEMP_LOCAL

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Always raises NotImplementedError, signalling an
        # internal Dynamo bug if a guard on this source is attempted.
        raise NotImplementedError(
            "Cannot create guard on TempLocalSource - this is an internal Dynamo bug. Please file an issue on GitHub."
        )


@dataclass_with_cached_hash(frozen=True)
class SyntheticLocalSource(Source):
    # [CLASS DOC] Source for a synthetic local variable created by Dynamo (not in the
    # user frame). Field: local_name. Guard source is SYNTHETIC_LOCAL.
    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a bytecode load of the named local onto the stack.
        codegen.append_output(codegen.create_load(self.local_name))

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind SYNTHETIC_LOCAL.
        return GuardSource.SYNTHETIC_LOCAL

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a name template of a synthetic-local lookup
        # using the repr of the local name, braces escaped. Cached.
        return f"SYNTHETIC_LOCAL[{_esc_str(self.local_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class RandomValueSource(Source):
    # [CLASS DOC] Source for a value produced by a random-number call during tracing.
    # Field: random_call_index, the position of the call in the recorded random
    # values list. Guard source is RANDOM_VALUE.
    random_call_index: int

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind RANDOM_VALUE.
        return GuardSource.RANDOM_VALUE

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that loads the output graph's random-values
        # variable, pushes the call index constant, and subscripts, yielding the
        # recorded random value.
        codegen.append_output(codegen.create_load(codegen.tx.output.random_values_var))
        codegen.append_output(codegen.create_load_const(self.random_call_index))
        codegen.append_output(create_binary_subscr())

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a name of the form random_value_<index>.
        return f"random_value_{_esc_str(self.random_call_index)}"


@dataclass_with_cached_hash(frozen=True)
class GlobalSource(Source):
    # [CLASS DOC] Root source for a global variable of the traced frame. Field:
    # global_name. Guard source is GLOBAL; name renders as a lookup in the globals map.
    global_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a global load of the name (registering it in the
        # codegen globals, add enabled) onto the stack.
        codegen.append_output(codegen.create_load_global(self.global_name, add=True))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        # [FUNCTION DOC] Returns the Python expression text, which is the global name.
        return self.global_name

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind GLOBAL.
        return GuardSource.GLOBAL

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the guard-name template, a globals-map
        # lookup of the repr of the global name, braces escaped. Cached.
        return f"G[{_esc_str(self.global_name, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class GlobalWeakRefSource(Source):
    # [CLASS DOC] Source for a global that holds a weak reference; the value is the
    # referent obtained by calling the weakref with no arguments. Field: global_name.
    # Guard source is GLOBAL.
    global_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that loads the global weakref (with a
        # push-null marker) and calls it with zero arguments to get the referent.
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_global(self.global_name, add=True)
            )
        )
        codegen.extend_output(create_call_function(0, False))

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind GLOBAL.
        return GuardSource.GLOBAL

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the global lookup name followed by a
        # zero-argument call, braces escaped. Cached.
        return f"G[{_esc_str(self.global_name, apply_repr=True)}]()"


@dataclass_with_cached_hash(frozen=True)
class WeakRefCallSource(ChainedSource):
    # [CLASS DOC] Chained source that calls the base value (a weakref) with no
    # arguments to obtain the referent.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that loads the base (with push-null marker)
        # and calls it with zero arguments.
        codegen.add_push_null(lambda: codegen(self.base))
        codegen.extend_output(create_call_function(0, False))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template that calls the base name with
        # no arguments.
        return "{0}()"


@dataclass_with_cached_hash(frozen=True)
class CallFunctionNoArgsSource(WeakRefCallSource):
    # [CLASS DOC] Alias-like subclass of WeakRefCallSource for a generic zero-argument
    # call of the base value; behavior is entirely inherited.
    pass


@dataclass_with_cached_hash(frozen=True)
class AttrSource(ChainedSource):
    # [CLASS DOC] Chained source for attribute access on the base value. Field:
    # member, a single attribute name (no dots; invariant checked at construction).
    # Names non-identifier members via the builtin attribute getter.
    member: str

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates construction. Raises an error if the base is
        # missing/falsy or the member name contains a dot (dotted paths must be built
        # with the output graph's chained attr source helper).
        if not self.base:
            raise AssertionError(
                "Can't construct an AttrSource without a valid base source"
            )
        if "." in self.member:
            raise AssertionError(
                f"AttrSource member must not contain '.', got {self.member!r}. "
                "Use OutputGraph.get_chained_attr_source() for dotted paths."
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base, then bytecode to load the
        # member attribute from it.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        # [FUNCTION DOC] Returns Python expression text: base text plus dotted member
        # if the member is a valid identifier, otherwise a getattr call text with the
        # member's repr.
        base = self.base.reconstruct_pycode(codegen)
        if self.member.isidentifier():
            return f"{base}.{self.member}"
        else:
            return f"getattr({base}, {self.member!r})"

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template with dotted access for identifier
        # members, or a getattr call template for other member names. Cached.
        if not self.member.isidentifier():
            return f"getattr({{0}}, {_esc_str(self.member, apply_repr=True)})"
        return f"{{0}}.{_esc_str(self.member)}"


@dataclass_with_cached_hash(frozen=True)
class CellContentsSource(AttrSource):
    # [CLASS DOC] AttrSource specialised for the cell_contents attribute of a closure
    # cell. Additionally stores freevar_name (default empty) so guard failure messages
    # can name the closure variable. Invariant: member must be cell_contents.
    """
    Source for closure cell contents that also stores the freevar name.
    This allows guard failure messages to show which variable the closure cell refers to.
    """

    freevar_name: str = dataclasses.field(default="")

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates construction. Raises an error if the base is
        # missing or the member is not cell_contents. Does not call the parent check
        # for dotted members (not needed given the fixed member).
        if not self.base:
            raise AssertionError(
                "Can't construct a CellContentsSource without a valid base source"
            )
        if self.member != "cell_contents":
            raise AssertionError(
                "CellContentsSource should only be used for cell_contents"
            )


@dataclass_with_cached_hash(frozen=True)
class GenericAttrSource(ChainedSource):
    # [CLASS DOC] Chained source for attribute access that bypasses user-overridden
    # attribute lookup (guard name uses the object-level attribute getter). Field:
    # member, a single attribute name without dots.
    member: str

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates construction. Raises an error if the base is
        # missing or the member contains a dot.
        if not self.base:
            raise AssertionError(
                "Can't construct a GenericAttrSource without a valid base source"
            )
        if "." in self.member:
            raise AssertionError(
                f"GenericAttrSource member must not contain '.', got {self.member!r}. "
                "Use OutputGraph.get_chained_attr_source() for dotted paths."
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then a plain attribute load of
        # member. KNOWN ISSUE: reconstruction uses ordinary attribute access, while
        # the guard name uses the object-level getter, so overridden getattr hooks
        # could differ between the two.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the object-level
        # attribute getter on the base with the member repr. Cached.
        return (
            f"object.__getattribute__({{0}}, {_esc_str(self.member, apply_repr=True)})"
        )


# Represents obj.__dict__ where obj is a type object
@dataclass_with_cached_hash(frozen=True)
class TypeDictSource(ChainedSource):
    # [CLASS DOC] Chained source for the class dictionary of a type object (base is
    # the type). Guard name wraps the access in a dict conversion.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then loads its __dict__ attribute.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__dict__"))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template that wraps the base's __dict__ in a
        # dict constructor so the guard manager sees a real dict, not a mapping proxy.
        # type(ob).__dict__ can return a proxy of the dict. But in the C++
        # guard accessor, we use type->tp_dict which is a dict. So,
        # forcefully pass a dict object to ensure that the GuardManager
        # registers that it's working on a dict object.
        return "dict({0}.__dict__)"


# Represents obj.__mro__ where object is type object
@dataclass_with_cached_hash(frozen=True)
class TypeMROSource(ChainedSource):
    # [CLASS DOC] Chained source for the method resolution order tuple of a type.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then loads its __mro__ attribute.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__mro__"))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the template for the base's __mro__ attribute.
        return "{0}.__mro__"


@dataclass_with_cached_hash(frozen=True)
class LocalCellSource(Source):
    # [CLASS DOC] Source for the cell object itself (implicitly created for captured
    # variables), as opposed to the cell's contents. Field: local_name. Only supports
    # reconstruction; guard-related members are intentionally unimplemented since a
    # cell object must never be guarded on.
    """
    Conceptually, this class is `LocalSource` for cell objects implicitly
    generated by Python (e.g., captured variables).
    """

    local_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a closure-load instruction for the named cell onto the
        # stack (always the closure-load form rather than a fast-local load).
        # Although `LOAD_FAST` and `LOAD_CLOSURE` have the same semantics,
        # Dynamo's bytecode transformation differentiates them slightly, so we
        # always emit `LOAD_CLOSURE` here.
        codegen.append_output(codegen.create_load_closure(self.local_name))

    # All the other methods are intentionally unimplemented because e.g., a
    # local cell object should never be used for guards.


# Represents obj.__code__ where object is type object
@dataclass_with_cached_hash(frozen=True)
class CodeSource(ChainedSource):
    # [CLASS DOC] Chained source for the code object of a function (base is the
    # function; the comment above says type object, which is imprecise).
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then loads its __code__ attribute.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__code__"))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the template for the base's __code__ attribute.
        return "{0}.__code__"


# Represents obj.__closure__ where object is type object
@dataclass_with_cached_hash(frozen=True)
class ClosureSource(ChainedSource):
    # [CLASS DOC] Chained source for the closure tuple of a function (base is the
    # function). is_from_closure_source detects chains containing this source.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then loads its __closure__ attribute.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("__closure__"))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the template for the base's __closure__ attribute.
        return "{0}.__closure__"


# Represents tensor.grad source. It could be represented by AttrSource as well.
# But, we could access grad field on tensor directly in C++ without going
# through the Python bytecodes. Therefore, we use a separate source for grad
# field.
@dataclass_with_cached_hash(frozen=True)
class GradSource(ChainedSource):
    # [CLASS DOC] Chained source for the gradient attribute of a tensor, kept distinct
    # from AttrSource so the C++ guards can read the field directly. Field: member
    # (default grad).
    member: str = "grad"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then loads the member attribute.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.member))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a dotted-attribute template for the member.
        return f"{{0}}.{_esc_str(self.member)}"


@dataclass_with_cached_hash(frozen=True)
class ParamBufferSource(AttrSource):
    # [CLASS DOC] AttrSource for a parameter or buffer of a specialized nn.Module.
    # Overrides guard_source so the base's guard source is mapped to its specialized
    # nn.Module equivalent.
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the base's guard source mapped through the
        # specialized-nn-module table. Raises KeyError for guard sources not in the
        # table. Cached.
        return _GUARD_SOURCE_SPECIALIZED_NN_MODULE[self.base.guard_source]


# Special AttrSource to differentiate module._buffers or module._parameters
@dataclass_with_cached_hash(frozen=True)
class UnspecializedParamBufferSource(AttrSource):
    # [CLASS DOC] Marker AttrSource (no behavior change) distinguishing access to a
    # module's parameters/buffers containers for unspecialized modules; detected by
    # is_from_unspecialized_param_buffer_source.
    pass


# This source is intended to be used in places where a source is needed but it is expected
# that the symbol will be simplified out later on. Symbols with ephemeral sources are
# prioritized to be simplified out when e.g. compared against a symbol without an ephemeral
# source. Guarding on this source is an error.
#
# Example: During subclass view fake-ification, any close-over ViewFunc state should be
# symbolicized / fake-ified to avoid invalid specialization during view replay. This source
# is useful for symbols utilized in the middle of the view chain that are not expected to be
# present within the final view shape metadata.
@dataclass_with_cached_hash(frozen=True)
class EphemeralSource(Source):
    # [CLASS DOC] Placeholder source for symbols expected to be simplified away later.
    # Field: desc, optional description shown in the name. Guarding on it is an
    # error; is_ephemeral reports True so such symbols are preferred for elimination.
    desc: str | None = None

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind EPHEMERAL.
        return GuardSource.EPHEMERAL

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns an angle-bracket name "ephemeral" with the
        # description appended after a colon when present. Cached.
        desc = ": " + self.desc if self.desc is not None else ""
        return f"<ephemeral{_esc_str(desc)}>"

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        # [FUNCTION DOC] Always raises NotImplementedError: guards cannot be made on
        # an ephemeral source.
        raise NotImplementedError

    def is_ephemeral(self) -> bool:
        # [FUNCTION DOC] Returns True, marking this source as ephemeral.
        return True


@dataclass_with_cached_hash(frozen=True)
class SkipGuardSource(ChainedSource):
    # [CLASS DOC] Transparent chained source marking a value whose guards should be
    # skipped; name and reconstruction forward to the base. Detected by
    # is_from_skip_guard_source.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Delegates reconstruction directly to the base source.
        self.base.reconstruct(codegen)

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the base name unchanged.
        return "{0}"


class TensorProperty(enum.Enum):
    # [CLASS DOC] Enumeration of tensor metadata kinds that can be a source: SIZE,
    # STRIDE, STORAGE_OFFSET.
    SIZE = 0
    STRIDE = 1
    STORAGE_OFFSET = 2

    def method_name(self) -> str:
        # [FUNCTION DOC] Returns the tensor method name for this property: size,
        # stride or storage_offset. Raises an error for an unhandled member.
        if self is TensorProperty.SIZE:
            return "size"
        elif self is TensorProperty.STRIDE:
            return "stride"
        elif self is TensorProperty.STORAGE_OFFSET:
            return "storage_offset"
        else:
            raise AssertionError(f"unhandled {_esc_str(self)}")


@dataclass_with_cached_hash(frozen=True)
class TensorPropertySource(ChainedSource):
    # [CLASS DOC] Chained source for a tensor's size, stride or storage offset.
    # Fields: prop (TensorProperty) and idx (dimension index; must be None exactly
    # when prop is STORAGE_OFFSET). Base is the tensor source.
    prop: TensorProperty
    idx: int | None = None  # None for STORAGE_OFFSET

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates construction. Raises an error if the base is
        # None, if idx is set for STORAGE_OFFSET, or if idx is missing for SIZE/STRIDE.
        if self.base is None:
            raise AssertionError("TensorPropertySource base must not be None")
        if self.prop is TensorProperty.STORAGE_OFFSET:
            if self.idx is not None:
                raise AssertionError(
                    "idx must be None for STORAGE_OFFSET TensorProperty"
                )
        else:
            if self.idx is None:
                raise AssertionError(
                    f"idx must not be None for {self.prop} TensorProperty"
                )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that calls the matching utils helper (for size,
        # stride or storage offset) with the tensor and, when idx is set, the index
        # constant.
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                utils.__name__, f"call_{_esc_str(self.prop.method_name())}"
            )
        )
        codegen(self.base)

        if self.idx is not None:
            codegen.append_output(codegen.create_load_const(self.idx))
        codegen.extend_output(
            create_call_function(2 if self.idx is not None else 1, False)
        )

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template like base.size()[idx],
        # base.stride()[idx] or base.storage_offset(). Raises an error if
        # STORAGE_OFFSET has an index or the property is unknown. Cached.
        if self.prop is TensorProperty.SIZE:
            return f"{{0}}.size()[{_esc_str(self.idx)}]"
        elif self.prop is TensorProperty.STRIDE:
            return f"{{0}}.stride()[{_esc_str(self.idx)}]"
        elif self.prop is TensorProperty.STORAGE_OFFSET:
            if self.idx is not None:
                raise AssertionError(
                    "idx must be None for STORAGE_OFFSET TensorProperty"
                )
            return "{0}.storage_offset()"
        else:
            raise AssertionError(f"unhandled {_esc_str(self.prop)}")


@dataclass_with_cached_hash(frozen=True)
class IndexedSource(ChainedSource):
    # [CLASS DOC] Chained source pairing an integer index with the base; naming only,
    # cannot be reconstructed. Field: idx.
    idx: int

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError("IndexedSource base must not be None")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a tuple-like template of the index and the
        # base name. Cached.
        return f"({_esc_str(self.idx)}, {{0}})"


@dataclass_with_cached_hash(frozen=True)
class NegateSource(ChainedSource):
    # [CLASS DOC] Chained source for the arithmetic negation of the base value;
    # naming only, cannot be reconstructed.
    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError("NegateSource base must not be None")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Not supported; always raises NotImplementedError.
        raise NotImplementedError

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the negation method on
        # the base (a method call so that function-stripping regexes work).
        # NB: use method call so that function stripping regexes work
        return "{0}.__neg__()"


@dataclass_with_cached_hash(frozen=True)
class ConvertIntSource(ChainedSource):
    # [CLASS DOC] Chained source for converting a symbolic bool to a symbolic int
    # without guarding; reconstructs as the base value itself.
    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError("ConvertIntSource base must not be None")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base only (no conversion emitted).
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the guardless
        # bool-to-int cast helper on the base name.
        return "cast_symbool_to_symint_guardless({0})"


@dataclass_with_cached_hash(frozen=True)
class DynamicScalarSource(ChainedSource):
    # [CLASS DOC] Chained source for a dynamic scalar value derived from the base.
    # Field: is_int (stored flag; not used by the methods here).
    is_int: bool

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError("DynamicScalarSource base must not be None")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode that loads the builtin int, then the base,
        # and calls int on it, so compiled regions return plain ints rather than
        # dynamic int wrappers.
        # Integer casting at reconstruction helps reduce the amount of DynamicInts returned
        # to the user, in favor of plain ints.
        # For example, a compiled region that only does int arithmetic could return a
        # DynamicInt without the casting here.
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "int"))
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template that wraps the base in an int
        # conversion.
        return "int({0})"


@dataclass_with_cached_hash(frozen=True)
class FlattenScriptObjectSource(ChainedSource):
    # [CLASS DOC] Chained source for the flattened form of a script (torchbind)
    # object, obtained via its flatten method. Reconstructs as the base itself.
    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError("FlattenScriptObjectSource base must not be None")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base only.
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the flatten method on the base.
        return "{0}.__obj_flatten__()"


@dataclass_with_cached_hash(frozen=True)
class ScriptObjectQualifiedNameSource(ChainedSource):
    # [CLASS DOC] Chained source for the qualified type name of a script object.
    # Reconstructs as the base itself.
    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError(
                "ScriptObjectQualifiedNameSource base must not be None"
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base only.
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template that reads the qualified name of
        # the base's type.
        return "{0}._type().qualified_name()"


class AttrProxySource(ChainedSource):
    # [CLASS DOC] Chained source for the underlying object behind an attribute proxy.
    # KNOWN ISSUE: unlike the other sources it is not decorated as a cached-hash
    # frozen dataclass.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base only.
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the proxy's base getter.
        return "{0}.get_base()"


@dataclass_with_cached_hash(frozen=True)
class DefaultsSource(ChainedSource):
    # [CLASS DOC] Chained source for a default argument of a function. Fields: idx_key
    # (int position for positional defaults, str name for keyword defaults), is_kw,
    # plus derived non-init fields field (defaults attribute name) and _name (name
    # template) computed at construction.
    idx_key: int | str
    is_kw: bool = False
    field: str = dataclasses.field(init=False, repr=False, compare=False)
    _name: str = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validates and fills derived fields. Raises an error if the
        # base is missing, or if idx_key has the wrong type (str for keyword defaults,
        # int for positional). Sets field to the kwdefaults or defaults attribute name
        # and builds the name template, bypassing frozen-ness.
        if not self.base:
            raise AssertionError(
                "Base must be a valid source in order to properly track and guard this Defaults to its origin."
            )
        if self.is_kw:
            if not isinstance(self.idx_key, str):
                raise AssertionError(
                    f"idx_key must be a str for keyword defaults, got {type(self.idx_key)}"
                )
            object.__setattr__(self, "field", "__kwdefaults__")
            object.__setattr__(
                self,
                "_name",
                f"{{0}}.{_esc_str(self.field)}['{_esc_str(self.idx_key)}']",
            )
        else:
            if not isinstance(self.idx_key, int):
                raise AssertionError(
                    f"idx_key must be an int for positional defaults, got {type(self.idx_key)}"
                )
            object.__setattr__(self, "field", "__defaults__")
            object.__setattr__(
                self, "_name", f"{{0}}.{_esc_str(self.field)}[{_esc_str(self.idx_key)}]"
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode: base, load the defaults attribute, push the
        # key constant, subscript.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs(self.field))
        codegen.append_output(codegen.create_load_const(self.idx_key))
        codegen.append_output(create_binary_subscr())

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the precomputed name template. Cached.
        return self._name


@dataclass_with_cached_hash(frozen=True)
class GetItemSource(ChainedSource):
    # [CLASS DOC] Chained source for subscripting the base with a constant index.
    # Fields: index (constant; a slice is stored in hashable reduced form) and
    # index_is_slice flag. Invariant: index is never itself a Source.
    index: Any
    index_is_slice: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None. If the index is a
        # slice, replaces it with its reduced tuple form and sets index_is_slice, so
        # the dataclass remains hashable (mutates the frozen instance).
        if self.base is None:
            raise AssertionError("GetItemSource base must not be None")
        if isinstance(self.index, slice):
            # store the hashable version of the slice so the whole GetItemSource is hashable
            super().__setattr__("index", self.index.__reduce__())
            super().__setattr__("index_is_slice", True)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode: base, constant index (rebuilt slice if the
        # index is a slice), then a subscript instruction.
        codegen(self.base)
        if self.index_is_slice:
            codegen.append_output(codegen.create_load_const(self.unpack_slice()))
        else:
            codegen.append_output(codegen.create_load_const(self.index))
        codegen.append_output(create_binary_subscr())

    def unpack_slice(self) -> slice:
        # [FUNCTION DOC] Rebuilds and returns the slice object from the stored
        # reduced form. Raises an error if the index is not a slice.
        if not self.index_is_slice:
            raise AssertionError("unpack_slice called but index is not a slice")
        slice_class, slice_args = self.index
        return slice_class(*slice_args)

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a subscript template with the repr of the
        # index (or of the rebuilt slice). Raises an error if the index is a Source.
        # Cached.
        # Index can be of following types
        # 1) index is a slice - example 1:4
        # 2) index is a constant - example string, integer
        if isinstance(self.index, Source):
            raise AssertionError(
                f"GetItemSource index must not be a Source, got {type(self.index)}"
            )
        if self.index_is_slice:
            return f"{{0}}[{_esc_str(self.unpack_slice(), apply_repr=True)}]"
        else:
            return f"{{0}}[{_esc_str(self.index, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class ConstDictKeySource(ChainedSource):
    # [CLASS DOC] Chained source for the key at a given position in a dictionary's
    # key order. Field: index (position). Reports itself as a dict key.
    index: Any

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils key-by-position helper with the
        # base dict and the index constant.
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "dict_keys_getitem")
        )
        codegen(self.base)
        codegen.append_output(codegen.create_load_const(self.index))
        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template that lists the dict's keys and
        # subscripts by the index repr (the list creation is deduplicated by a later
        # common-subexpression pass). Cached.
        # The list creation will be CSE'd by PyExprCSEPass
        return f"list(dict.keys({{0}}))[{_esc_str(self.index, apply_repr=True)}]"

    def is_dict_key(self) -> bool:
        # [FUNCTION DOC] Returns True: this source denotes a dictionary key.
        return True


@dataclass_with_cached_hash(frozen=True)
class NonSerializableSetGetItemSource(ChainedSource):
    # [CLASS DOC] Chained source for the element at a given iteration position of a
    # set (ordering may be unstable, hence non-serializable). Field: index, a literal
    # int. Provides its own value evaluation and is not a dict key.
    index: int

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the index is not a literal constant.
        from .variables import ConstantVariable

        if not ConstantVariable.is_literal(self.index):
            raise AssertionError(
                f"NonSerializableSetGetItemSource index must be a literal, got {self.index!r}"
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils set-item helper with the base set
        # and the index constant.
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "set_getitem")
        )
        codegen(self.base)
        codegen.append_output(codegen.create_load_const(self.index))
        codegen.extend_output(create_call_function(2, False))

    def get_value(
        self,
        globals: dict[str, Any],
        locals: dict[str, Any],
        cache: dict[Source, Any],
    ) -> Any:
        # [FUNCTION DOC] Evaluates the source: returns the cached value if present;
        # otherwise evaluates the base with the same globals/locals/cache, takes the
        # element at the index with the utils helper, stores it in the cache mapping
        # (mutation) and returns it.
        if self in cache:
            return cache[self]
        value = utils.set_getitem(
            self.base.get_value(globals, locals, cache), self.index
        )
        cache[self] = value
        return value

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the set-getitem helper
        # with the base and index repr. Cached.
        # set ordering might not be stable
        return f"___set_getitem({{0}}, {_esc_str(self.index, apply_repr=True)})"

    def is_dict_key(self) -> bool:
        # [FUNCTION DOC] Returns False: this source is not a dictionary key.
        return False


# Used to access an item from the dictionary
@dataclass_with_cached_hash(frozen=True)
class DictGetItemSource(ChainedSource):
    # [CLASS DOC] Chained source for looking up a key in a dictionary. Field: index,
    # either a ConstDictKeySource (key taken by position) or a literal constant.
    # Key to access in the dictionary. It can be one of the following types
    # 1) ConstDictKeySource
    # 2) constant - like string, integer
    index: Any

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error unless the index is a ConstDictKeySource or
        # a literal constant.
        from .variables import ConstantVariable

        if not isinstance(
            self.index, ConstDictKeySource
        ) and not ConstantVariable.is_literal(self.index):
            raise AssertionError(
                f"DictGetItemSource index must be a ConstDictKeySource or literal, got {self.index!r}"
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode: the dict, then the key (reconstructed from
        # the key source or loaded as a constant), then a subscript.
        # Load dict
        codegen(self.base)

        # Load key
        if isinstance(self.index, Source):
            codegen(self.index)
        else:
            codegen.append_output(codegen.create_load_const(self.index))
        codegen.append_output(create_binary_subscr())

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        # [FUNCTION DOC] Returns Python expression text: base text subscripted by the
        # key (the key source's text, or the repr of the constant).
        base = self.base.reconstruct_pycode(codegen)
        if isinstance(self.index, ConstDictKeySource):
            index = self.index.reconstruct_pycode(codegen)
        else:
            index = repr(self.index)
        return f"{base}[{index}]"

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a subscript template using the key source's
        # name or the repr of the constant, braces escaped. Cached.
        if isinstance(self.index, ConstDictKeySource):
            return f"{{0}}[{_esc_str(self.index.name)}]"
        else:
            return f"{{0}}[{_esc_str(self.index, apply_repr=True)}]"


# Same as DictGetItemSource but used for dict.__getitem__ calls to ensure that
# torch.compile does not run the overridden __getitem__ method
@dataclass_with_cached_hash(frozen=True)
class DictSubclassGetItemSource(ChainedSource):
    # [CLASS DOC] Like DictGetItemSource but for dict subclasses: reconstruction calls
    # the base dict getitem directly so an overridden item-access method is not run.
    # Field: index, a ConstDictKeySource or a literal constant.
    # Key to access in the dictionary. It can be one of the following types
    # 1) ConstDictKeySource
    # 2) constant - like string, integer
    index: Any

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error unless the index is a ConstDictKeySource or
        # a literal constant.
        from .variables import ConstantVariable

        if not isinstance(
            self.index, ConstDictKeySource
        ) and not ConstantVariable.is_literal(self.index):
            raise AssertionError(
                f"DictSubclassGetItemSource index must be a ConstDictKeySource or literal, got {self.index!r}"
            )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils dict getitem helper with the dict
        # and the key (key source reconstruction or constant).
        # reconstruct dict.__getitem__(dct, key)

        # Load dict.__getitem__
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "dict_getitem")
        )

        # Load dict
        codegen(self.base)

        # Load key
        if isinstance(self.index, Source):
            codegen(self.index)
        else:
            codegen.append_output(codegen.create_load_const(self.index))

        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a dict-getitem call template for a key
        # source index, or a plain subscript template for a constant. KNOWN ISSUE:
        # the constant case uses ordinary subscripting, which would run an overridden
        # item-access method, inconsistent with the class intent. Cached.
        if isinstance(self.index, ConstDictKeySource):
            return f"dict.__getitem__({{0}}, {_esc_str(self.index.name)})"
        else:
            return f"{{0}}[{_esc_str(self.index, apply_repr=True)}]"


@dataclass_with_cached_hash(frozen=True)
class ListGetItemSource(GetItemSource):
    # [CLASS DOC] GetItemSource variant for lists that bypasses overridden item
    # access by calling the list getitem helper. Slices are not supported.
    """
    Same as GetItemSource with reconstruct and name overridden to be list specific.
    """

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils list getitem helper with the list
        # and the index constant. Raises RuntimeError if the index is a slice, since
        # slicing a list yields a temporary without a source.
        # Reconstruct list.__getitem__(lst, index) to avoid any side effects
        # from possibly overridden __getitem__.

        # Load list.__getitem__
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "list_getitem")
        )

        # Load the list
        codegen(self.base)

        # Load the index
        if self.index_is_slice:
            raise RuntimeError(
                "List[slice] is a temporary object and should not have a source"
            )
        else:
            codegen.append_output(codegen.create_load_const(self.index))

        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the list getitem on the
        # base with the index repr. Raises an error if the index is a Source and
        # RuntimeError if it is a slice. Cached.
        # Index can be of following types
        # 1) index is a slice - example 1:4
        # 2) index is a constant - example string, integer
        if isinstance(self.index, Source):
            raise AssertionError(
                f"ListGetItemSource index must not be a Source, got {type(self.index)}"
            )
        if self.index_is_slice:
            raise RuntimeError(
                "List[slice] is a temporary object and should not have a source"
            )
        else:
            return f"list.__getitem__({{0}}, {_esc_str(self.index, apply_repr=True)})"


@dataclass_with_cached_hash(frozen=True)
class TupleIteratorGetItemSource(GetItemSource):
    # [CLASS DOC] GetItemSource variant for the remaining element at a position of a
    # tuple iterator, read through a utils helper.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils tuple-iterator getitem helper with
        # the base iterator and the index constant.
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "tuple_iterator_getitem")
        )
        codegen(self.base)
        codegen.append_output(codegen.create_load_const(self.index))
        codegen.extend_output(create_call_function(2, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the tuple-iterator
        # getitem helper with the base and index repr. Cached.
        return (
            f"___tuple_iterator_getitem({{0}}, {_esc_str(self.index, apply_repr=True)})"
        )


@dataclass_with_cached_hash(frozen=True)
class NamedTupleFieldsSource(ChainedSource):
    # [CLASS DOC] Chained source for the tuple of field names of a namedtuple
    # (base is the namedtuple type or instance).
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base then loads its _fields attribute.
        codegen(self.base)
        codegen.extend_output(codegen.create_load_attrs("_fields"))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the namedtuple-fields
        # guard helper on the base.
        return "___namedtuple_fields({0})"


@dataclass_with_cached_hash(frozen=True)
class DataclassFieldsSource(ChainedSource):
    # [CLASS DOC] Chained source for the fields of a dataclass (base is the
    # dataclass type or instance).
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils dataclass-fields helper with the base.
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "dataclass_fields")
        )
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the dataclass-fields
        # guard helper on the base.
        return "___dataclass_fields({0})"


@dataclass_with_cached_hash(frozen=True)
class TypeSource(ChainedSource):
    # [CLASS DOC] Chained source for the type of the base value.
    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises an error if the base is None.
        if self.base is None:
            raise AssertionError("TypeSource base must not be None")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the builtin type function on the base value.
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "type"))
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling type on the base name.
        return "type({0})"


@dataclass_with_cached_hash(frozen=True)
class OptimizerSource(ChainedSource):
    # [CLASS DOC] Transparent chained source marking a value that belongs to an
    # optimizer; detected by is_from_optimizer_source. Name and reconstruction forward
    # to the base.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base only.
        codegen(self.base)

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the base name unchanged.
        return "{0}"


@dataclass_with_cached_hash(frozen=True)
class NNModuleSource(ChainedSource):
    # [CLASS DOC] Transparent chained source for a specialized nn.Module. Name and
    # reconstruction forward to the base; guard_source maps the base's guard source to
    # the specialized-module kind. Subclasses select other mapping tables.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits bytecode for the base only.
        codegen(self.base)

    def reconstruct_pycode(self, codegen: "PyCodegen") -> str:
        # [FUNCTION DOC] Returns the base's Python expression text unchanged.
        return self.base.reconstruct_pycode(codegen)

    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the base guard source mapped through the
        # specialized-nn-module table; raises KeyError for unmapped kinds. Cached.
        return _GUARD_SOURCE_SPECIALIZED_NN_MODULE[self.base.guard_source]

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the base name unchanged.
        return "{0}"


@dataclass_with_cached_hash(frozen=True)
class UnspecializedNNModuleSource(NNModuleSource):
    # [CLASS DOC] NNModuleSource for an unspecialized nn.Module; only the guard source
    # mapping differs.
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the base guard source mapped through the
        # unspecialized-nn-module table; raises KeyError for unmapped kinds. Cached.
        return _GUARD_SOURCE_UNSPECIALIZED_NN_MODULE[self.base.guard_source]


@dataclass_with_cached_hash(frozen=True)
class UnspecializedBuiltinNNModuleSource(UnspecializedNNModuleSource):
    # [CLASS DOC] NNModuleSource for an unspecialized built-in (library-provided)
    # nn.Module; only the guard source mapping differs.
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the base guard source mapped through the
        # unspecialized-builtin-nn-module table; raises KeyError for unmapped kinds. Cached.
        return _GUARD_SOURCE_UNSPECIALIZED_BUILTIN_NN_MODULE[self.base.guard_source]


@dataclass_with_cached_hash(frozen=True)
class FSDPNNModuleSource(NNModuleSource):
    # [CLASS DOC] NNModuleSource for a module wrapped by FSDP; only the guard source
    # mapping differs.
    @functools.cached_property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the base guard source mapped through the FSDP-module
        # table; raises KeyError for unmapped kinds. Cached.
        return _GUARD_SOURCE_FSDP_MODULE[self.base.guard_source]


@dataclass_with_cached_hash(frozen=True)
class GlobalStateSource(Source):
    # [CLASS DOC] Synthetic root source tied to global-state guards (grad mode,
    # autocast, etc.). Has an empty name and guard source GLOBAL.
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the empty string.
        return ""

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind GLOBAL.
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class ImportSource(Source):
    # [CLASS DOC] Root source for an imported module, used instead of GlobalSource in
    # case the user shadowed the module name locally. Field: module_name. Guard
    # source is GLOBAL.
    """Points to an imported module - used instead of GlobalSource
    in case the user has overridden the module name in their local namespace"""

    module_name: str

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the builtin import
        # function with the module name. KNOWN ISSUE: the module name is inserted
        # without brace escaping or repr, unlike the other templates. Cached.
        return f"__import__('{self.module_name}')"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits the instructions of an import: level constant zero, an
        # empty from-list tuple, and the import-name instruction for the module.
        codegen.extend_output(
            [
                codegen.create_load_const(0),  # level
                create_build_tuple(0),  # fromlist
                codegen.create_import_name(self.module_name),
            ]
        )

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind GLOBAL.
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class TorchFunctionModeStackSource(Source):
    # [CLASS DOC] Root source for a torch-function mode on the mode stack. Field: ind,
    # the stack position as tracked by Dynamo, converted to a real stack index at use
    # time. Guard source is GLOBAL.
    ind: int

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the mode-stack-at guard
        # helper with the resolved stack index. Cached (index resolved once).
        return f"___get_torch_function_mode_stack_at({_esc_str(self._get_index())})"

    def _get_index(self) -> int:
        # [FUNCTION DOC] HELPER: Converts the stored position into the current mode
        # stack index via the torch function mode stack variable class and returns it.
        from .variables.torch_function import TorchFunctionModeStackVariable

        return TorchFunctionModeStackVariable.get_mode_index(self.ind)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils mode-stack-at helper with the
        # resolved index constant.
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                utils.__name__, "get_torch_function_mode_stack_at"
            )
        )
        codegen.extend_output([codegen.create_load_const(self._get_index())])
        codegen.extend_output(create_call_function(1, False))

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind GLOBAL.
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class ConstantSource(Source):
    # [CLASS DOC] Root source for a named constant that lives in the generated
    # code's globals. Field: source_name. Guard source is CONSTANT; making guards on
    # it is not supported.
    source_name: str

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a global load of the name without registering it.
        codegen.append_output(codegen.create_load_global(self.source_name, add=False))

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind CONSTANT.
        return GuardSource.CONSTANT

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the source name verbatim (not brace-escaped).
        # Cached.
        return self.source_name

    def make_guard(self, fn: Any) -> Any:
        # [FUNCTION DOC] Always raises NotImplementedError: constants are not guarded.
        raise NotImplementedError


@dataclass_with_cached_hash(frozen=True)
class NumpyTensorSource(ChainedSource):
    # [CLASS DOC] Chained source for a tensor obtained from a numpy array (base).
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the from-numpy guard
        # helper on the base.
        return "___from_numpy({0})"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the torch as-tensor function on the base.
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "as_tensor"))
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))


@dataclass_with_cached_hash(frozen=True)
class SubclassAttrListSource(ChainedSource):
    # [CLASS DOC] Chained source for the list of inner-tensor attribute names of a
    # tensor subclass (first element of its flatten result). Naming only; reconstruct
    # is inherited.
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template taking the first element of the
        # base's tensor-flatten result.
        return "{0}.__tensor_flatten__()[0]"


# Guard-only source that yields the inner tensor of an AsyncCollectiveTensor
# (ACT) and the base value unchanged for a plain Tensor. Used to guard on the
# unwrapped tensor so a graph traced on an ACT can be reused when the resolved
# plain Tensor is passed at runtime. See
# torch._dynamo.variables.builder.VariableBuilder.wrap_tensor.
@dataclass_with_cached_hash(frozen=True)
class UnwrapCollectiveTensorSource(ChainedSource):
    # [CLASS DOC] Guard-only chained source that unwraps an async collective tensor to
    # its inner tensor (and passes plain tensors through) so guards match both forms.
    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the guards-module unwrap helper on the base.
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                "torch._dynamo.guards", "unwrap_async_collective_tensor"
            )
        )
        codegen(self.base)
        codegen.extend_output(create_call_function(1, False))

    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the unwrap guard helper
        # on the base.
        return "___unwrap_async_collective_tensor({0})"


# NB: We don't expect you to actually ever generate guards against this
# source, it is ephemeral
@dataclass_with_cached_hash(frozen=True)
class FloatTensorSource(ChainedSource):
    # [CLASS DOC] Ephemeral chained source for a float converted to a tensor; not
    # expected to be guarded or reconstructed (reconstruct is inherited).
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the as-tensor guard
        # helper on the base.
        return "___as_tensor({0})"


@dataclass_with_cached_hash(frozen=True)
class CallMethodItemSource(ChainedSource):
    # [CLASS DOC] Chained source for the scalar obtained by calling the item method on
    # the base tensor. Naming only; reconstruct is inherited.
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling item on the base.
        return "{0}.item()"


@dataclass_with_cached_hash(frozen=True)
class ContextVarGetSource(ChainedSource):
    # [CLASS DOC] Chained source for the value from calling get on a context variable
    # (the base). Fields: has_default and default_value, the optional default passed
    # to get.
    has_default: bool = False
    default_value: Any = None

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call of the base's get method, passing the default
        # value constant as a single argument if has_default, else no arguments.
        def load_get_method():
            # [FUNCTION DOC] HELPER: Emits bytecode loading the base and then its
            # get attribute (the bound method to be called).
            codegen(self.base)
            codegen.extend_output(codegen.create_load_attrs("get"))

        codegen.add_push_null(load_get_method)
        if self.has_default:
            codegen.append_output(codegen.create_load_const(self.default_value))
            codegen.extend_output(create_call_function(1, False))
        else:
            codegen.extend_output(create_call_function(0, False))

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a get-call template, including the repr of
        # the default value when has_default is set. Cached.
        if self.has_default:
            return f"{{0}}.get({_esc_str(self.default_value, apply_repr=True)})"
        return "{0}.get()"


# This is a synthetic source that is associated with the singleton
# shape env guard we always register for all frames.  We get the actual
# guard contents from the ambient ShapeEnv
@dataclass_with_cached_hash(frozen=True)
class ShapeEnvSource(Source):
    # [CLASS DOC] Synthetic root source for the single shape-environment guard
    # registered for every frame; the guard contents come from the ambient ShapeEnv.
    # Empty name; guard source SHAPE_ENV.
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the empty string.
        return ""

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind SHAPE_ENV.
        return GuardSource.SHAPE_ENV


@dataclass_with_cached_hash(frozen=True)
class CurrentStreamSource(Source):
    # [CLASS DOC] Root source for the current stream of a device. Field: device (a
    # torch device, with type and optional index). Guard source is GLOBAL.
    device: device_type

    @functools.cached_property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a template calling the current-stream guard
        # helper with a device built from the device type and index. Cached.
        return f"___get_current_stream(torch.device('{_esc_str(self.device.type)}', {_esc_str(self.device.index)}))"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # [FUNCTION DOC] Emits a call to the utils current-stream helper whose argument
        # is a device constructed from the device type constant and, when the index is
        # not None, the index constant.
        num_args = 1
        codegen.add_push_null(
            lambda: codegen.load_import_from(utils.__name__, "get_current_stream")
        )
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "device"))
        codegen.extend_output([codegen.create_load_const(self.device.type)])
        if self.device.index is not None:
            num_args += 1
            codegen.extend_output([codegen.create_load_const(self.device.index)])
        codegen.extend_output(create_call_function(num_args, False))
        codegen.extend_output(create_call_function(1, False))

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind GLOBAL.
        return GuardSource.GLOBAL


@dataclass_with_cached_hash(frozen=True)
class BackwardStateSource(Source):
    # [CLASS DOC] Synthetic root source for the backward-state object of compiled
    # autograd/hooks. Empty name; guard source BACKWARD_STATE.
    @property
    def _name_template(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the empty string.
        return ""

    @property
    def guard_source(self) -> GuardSource:
        # [FUNCTION DOC] Returns the guard source kind BACKWARD_STATE.
        return GuardSource.BACKWARD_STATE


@functools.lru_cache
def get_local_source_name(
    source: Source, *, only_allow_input: bool = False
) -> str | None:
    # [FUNCTION DOC] Walks down the chain of bases to the root. Returns the local
    # variable name if the root is a LocalSource (and, when only_allow_input is set,
    # it is a root-frame input); otherwise returns None. Results are cached.
    if isinstance(source, ChainedSource):
        return get_local_source_name(source.base, only_allow_input=only_allow_input)
    if not isinstance(source, LocalSource):
        return None
    if only_allow_input and not source.is_input:
        return None
    return source.local_name


@functools.lru_cache
def is_from_local_source(source: Source, *, only_allow_input: bool = False) -> bool:
    # [FUNCTION DOC] Returns True if the source chain roots in a local variable (an
    # input local when only_allow_input is set). Cached.
    return get_local_source_name(source, only_allow_input=only_allow_input) is not None


@functools.lru_cache
def is_from_global_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source chain roots in a GlobalSource. Cached.
    return get_global_source_name(source) is not None


@functools.lru_cache
def get_global_source_name(source: Source | None) -> str | None:
    # [FUNCTION DOC] Walks down the chain of bases; returns the global variable name
    # if the root is a GlobalSource, otherwise None. Cached.
    if isinstance(source, ChainedSource):
        return get_global_source_name(source.base)
    if not isinstance(source, GlobalSource):
        return None
    return source.global_name


@functools.lru_cache
def is_from_nonlocal_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the chain roots in a LocalSource that is cell
    # contents (closure variable) and not a root-frame input. Cached.
    if isinstance(source, ChainedSource):
        return is_from_nonlocal_source(source.base)
    return (
        isinstance(source, LocalSource)
        and source.is_derefed_cell_contents
        and not source.is_input
    )


@functools.lru_cache
def is_from_closure_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is a
    # ClosureSource. Cached.
    if isinstance(source, ClosureSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_closure_source(source.base)
    return False


@functools.lru_cache
def is_from_source(source: Source, target: Source) -> bool:
    # [FUNCTION DOC] Returns True if the target equals the source or any base in its
    # chain. Cached.
    if source == target:
        return True
    if isinstance(source, ChainedSource):
        return is_from_source(source.base, target)
    return False


@functools.lru_cache
def is_from_unspecialized_nn_module_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is an
    # UnspecializedNNModuleSource (includes the builtin subclass). Cached.
    if isinstance(source, UnspecializedNNModuleSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_unspecialized_nn_module_source(source.base)
    return False


@functools.lru_cache
def is_from_unspecialized_builtin_nn_module_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is an
    # UnspecializedBuiltinNNModuleSource. Cached.
    if isinstance(source, UnspecializedBuiltinNNModuleSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_unspecialized_builtin_nn_module_source(source.base)
    return False


@functools.lru_cache
def is_from_unspecialized_param_buffer_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is an
    # UnspecializedParamBufferSource. Cached.
    if isinstance(source, UnspecializedParamBufferSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_unspecialized_param_buffer_source(source.base)
    return False


@functools.lru_cache
def is_from_flatten_script_object_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is a
    # FlattenScriptObjectSource. Cached.
    if isinstance(source, FlattenScriptObjectSource):
        return True
    elif isinstance(source, ChainedSource):
        return is_from_flatten_script_object_source(source.base)
    return False


@functools.lru_cache
def is_from_optimizer_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is an
    # OptimizerSource. Cached.
    if isinstance(source, OptimizerSource):
        return True
    if isinstance(source, ChainedSource):
        return is_from_optimizer_source(source.base)
    return False


# TODO: can probably write a generic "test this on everything in the chain"
# helper
@functools.lru_cache
def is_from_defaults(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if any source in the chain denotes a function
    # default argument: a DefaultsSource, a dict item of a keyword-defaults attribute,
    # or an item of a positional-defaults attribute. Cached.
    if isinstance(source, DefaultsSource):
        return True

    # Accessed with func.__kwdefaults__["foo"]
    if (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, AttrSource)
        and source.base.member == "__kwdefaults__"
    ):
        return True

    # Accessed with func.__defaults__[0]
    if (
        isinstance(source, GetItemSource)
        and isinstance(source.base, AttrSource)
        and source.base.member == "__defaults__"
    ):
        return True

    if isinstance(source, ChainedSource):
        return is_from_defaults(source.base)
    return False


@functools.lru_cache
def is_from_skip_guard_source(source: Source) -> bool:
    # [FUNCTION DOC] Returns True if the source or any base in its chain is a
    # SkipGuardSource. Cached.
    if isinstance(source, SkipGuardSource):
        return True

    if isinstance(source, ChainedSource):
        return is_from_skip_guard_source(source.base)

    return False
