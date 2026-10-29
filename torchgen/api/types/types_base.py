# [MODULE DOC] Implementation-agnostic base of the torchgen C++ type model: ArgName and
# SpecialArgName, BaseCppType and the basic scalar type constants, the CType hierarchy
# (base, const/mutable ref, vector, array, tuple), NamedCType, Binding and Expr.
# Defines constants at import time; no other side effects.
"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torchgen.model import Argument, SelfArgument, TensorOptionsArguments


# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occur.


class SpecialArgName(Enum):
    # [CLASS DOC] Enumeration of special argument names that may stand in for a string name;
    # currently only a redundant memory_format marker.
    possibly_redundant_memory_format = auto()


ArgName = str | SpecialArgName


# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    # [CLASS DOC] Frozen description of a basic C++ type: optional namespace ns and name. Create
    # singletons rather than ad hoc instances.
    ns: str | None
    name: str

    def __str__(self) -> str:
        # [FUNCTION DOC] Return the qualified name (namespace and name), or just the name when the
        # namespace is None or empty.
        if self.ns is None or self.ns == "":
            return self.name
        return f"{self.ns}::{self.name}"


# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType("", "uint8_t")
charT = BaseCppType("", "int8_t")
shortT = BaseCppType("", "int16_t")
# It would be more symmetric for this to be called intT, but it easy to mix
# this up with JIT int (which is int64_t in C++), so we intentionally don't
# define intT to make it obvious when you've stuffed it up
int32T = BaseCppType("", "int32_t")
longT = BaseCppType("", "int64_t")
doubleT = BaseCppType("", "double")
floatT = BaseCppType("", "float")
boolT = BaseCppType("", "bool")
voidT = BaseCppType("", "void")


class CType(ABC):
    # [CLASS DOC] Abstract base of the C++ type model. Subclasses produce a C++ spelling via
    # cpp_type and a version without const references via remove_const_ref.
    @abstractmethod
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Abstract: return the C++ spelling; strip_ref drops an outer reference.
        raise NotImplementedError

    @abstractmethod
    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Abstract: return the type without const reference wrappers (the default
        # implementation returns self).
        return self


@dataclass(frozen=True)
class BaseCType(CType):
    # [CLASS DOC] A plain C++ type wrapping a BaseCppType.
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the qualified name of the wrapped type.
        return str(self.type)

    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Return self; nothing to remove.
        return self


@dataclass(frozen=True)
class ConstRefCType(CType):
    # [CLASS DOC] A const reference to the element CType elem.
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the spelling of the const reference; with strip_ref return the
        # element spelling without the reference.
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f"const {self.elem.cpp_type()} &"

    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Return the element with its own const-refs removed (this wrapper dropped).
        return self.elem.remove_const_ref()


@dataclass(frozen=True)
class VectorCType(CType):
    # [CLASS DOC] A std vector of elem.
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the std vector spelling (strip_ref not forwarded to the element).
        # Do not pass `strip_ref` recursively.
        return f"::std::vector<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Return a VectorCType with const-ref removed from the element.
        return VectorCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayCType(CType):
    # [CLASS DOC] A std array of elem with a fixed integer size.
    elem: CType
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the std array spelling with element type and size (strip_ref not
        # forwarded).
        # Do not pass `strip_ref` recursively.
        return f"::std::array<{self.elem.cpp_type()},{self.size}>"

    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Return an ArrayCType with const-ref removed from the element, same size.
        return ArrayCType(self.elem.remove_const_ref(), self.size)


@dataclass(frozen=True)
class TupleCType(CType):
    # [CLASS DOC] A std tuple of the CTypes in elems.
    elems: list[CType]

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the std tuple spelling of the element types (strip_ref is not
        # forwarded).
        # Do not pass `strip_ref` recursively.
        return f"::std::tuple<{','.join([e.cpp_type() for e in self.elems])}>"

    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Return a TupleCType with const-ref removed from each element.
        return TupleCType([e.remove_const_ref() for e in self.elems])


@dataclass(frozen=True)
class MutRefCType(CType):
    # [CLASS DOC] A mutable (non-const) reference to elem.
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the reference spelling; with strip_ref return the element spelling.
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f"{self.elem.cpp_type()} &"

    def remove_const_ref(self) -> CType:
        # [FUNCTION DOC] Return the element with its own const-refs removed (this wrapper dropped).
        return self.elem.remove_const_ref()


# A NamedCType is short for Named C++ semantic type.  A NamedCType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#


@dataclass(frozen=True)
class NamedCType:
    # [CLASS DOC] A C++ type (CType) tagged with a semantic argument name, so that translate only
    # substitutes values of the right meaning. Frozen dataclass: name and type.
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # [FUNCTION DOC] Return the C++ spelling of the underlying type; strip_ref is forwarded.
        return self.type.cpp_type(strip_ref=strip_ref)

    def remove_const_ref(self) -> NamedCType:
        # [FUNCTION DOC] Return a NamedCType with the same name, const-ref removed from the type.
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> NamedCType:
        # [FUNCTION DOC] Return a NamedCType with the given name and the same type.
        return NamedCType(name, self.type)


# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See torchgen.api.translate)


@dataclass(frozen=True)
class Binding:
    # [CLASS DOC] A C++ binding site for a formal parameter: name, NamedCType nctype, the source
    # schema argument, and optional default value text. Immutable; the methods return changed
    # copies or render strings for declarations and definitions.
    name: str
    nctype: NamedCType
    argument: Argument | TensorOptionsArguments | SelfArgument
    # TODO: maybe don't represent default here
    default: str | None = None

    def rename(self, name: str) -> Binding:
        # [FUNCTION DOC] Return a copy of this binding with a new name.
        return Binding(
            name=name,
            nctype=self.nctype,
            argument=self.argument,
            default=self.default,
        )

    @property
    def type(self) -> str:
        # [FUNCTION DOC] Property: the C++ spelling of the binding type.
        return self.nctype.cpp_type()

    def no_default(self) -> Binding:
        # [FUNCTION DOC] Return a copy with the default value removed.
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        # [FUNCTION DOC] Render the parameter declaration: type, name and any default value text.
        # With func_ptr_cast only the type is returned (for function pointer casts).
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # casting only needs to know the type
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    def defn(self) -> str:
        # [FUNCTION DOC] Render the parameter definition: type and name, without default value.
        return f"{self.type} {self.name}"

    def with_name(self, name: str) -> Binding:
        # [FUNCTION DOC] Return a copy of this binding with a new name (same as rename).
        return Binding(
            name=name, nctype=self.nctype, argument=self.argument, default=self.default
        )


# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.


@dataclass(frozen=True)
class Expr:
    # [CLASS DOC] A C++ expression: expr holds source text and type the NamedCType it provides.
    expr: str
    type: NamedCType
