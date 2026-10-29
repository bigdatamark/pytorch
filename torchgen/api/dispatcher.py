# [MODULE DOC] Translates JIT schemas to the dispatcher API, the unboxed calling convention
# used when invoking operators through the dispatcher. Mostly delegates to the C++ API
# mapping, but flattens arguments so dtype, layout, device and pin_memory are separate.
# Provides name, type, jit_arguments and Binding helpers. No import-time side effects.
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing_extensions import assert_never

from torchgen.api import cpp
from torchgen.api.types import ArgName, Binding, CType, NamedCType
from torchgen.model import (
    Argument,
    FunctionSchema,
    Return,
    SelfArgument,
    TensorOptionsArguments,
    Type,
)
from torchgen.utils import concatMap


if TYPE_CHECKING:
    from collections.abc import Sequence


# This file describes the translation of JIT schema to the dispatcher
# API, the *unboxed* calling convention by which invocations through
# the dispatcher are made.  Historically, the dispatcher API matched
# the C++ API, but with the establishment of the boxed API, we've
# made changes to the dispatcher API to so that the unboxed API
# better aligns with the boxed API.  The dispatcher API hooks heavily
# into our template based boxing/unboxing machinery, so changes
# to this convention will usually need template updates too.
#
# Prominent characteristics of the dispatcher API:
#
#   - dtype, layout, device and pin_memory are represented as separate
#     arguments.
#


def name(func: FunctionSchema) -> str:
    # [FUNCTION DOC] Returns the dispatcher function name; same as the C++ API name.
    return cpp.name(func)


def argumenttype_type(
    t: Type,
    *,
    mutable: bool,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = True,
) -> NamedCType:
    # [FUNCTION DOC] Maps a schema type to a named C++ type by forwarding to the C++ API
    # mapping with the same flags (mutability, SymInt, optional removal of non-owning
    # reference types). Defaults to SymInt types. Pure.
    # This is a faux amis.  If it makes sense in the future to add
    # more special cases here, or invert things so cpp.argument_type
    # calls this, or just completely inline the function, please do
    # it.
    return cpp.argumenttype_type(
        t,
        mutable=mutable,
        binds=binds,
        symint=symint,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
    )


def argument_type(
    a: Argument,
    *,
    binds: ArgName,
    remove_non_owning_ref_types: bool = False,
    symint: bool = True,
) -> NamedCType:
    # [FUNCTION DOC] Returns the named C++ type for an Argument, treating it as mutable
    # when the argument is written to.
    return argumenttype_type(
        a.type,
        mutable=a.is_write,
        binds=binds,
        remove_non_owning_ref_types=remove_non_owning_ref_types,
        symint=symint,
    )


def returns_type(rs: Sequence[Return], *, symint: bool = True) -> CType:
    # [FUNCTION DOC] Returns the C++ return type for a sequence of returns; currently
    # identical to the C++ API result.
    # At present, there is no difference. But there could be!
    return cpp.returns_type(rs, symint=symint)


def jit_arguments(func: FunctionSchema) -> list[Argument]:
    # [FUNCTION DOC] Returns the flat list of plain Arguments of a schema in JIT order:
    # positional, then keyword-only, then out. Self arguments are unwrapped and tensor
    # options are expanded into dtype, layout, device and pin_memory.
    def to_argument(
        a: Argument | TensorOptionsArguments | SelfArgument,
    ) -> list[Argument]:
        # [FUNCTION DOC] HELPER: Expands one grouped argument into its plain Arguments.
        # Raises via assert_never on an unexpected kind.
        if isinstance(a, Argument):
            return [a]
        elif isinstance(a, SelfArgument):
            return [a.argument]
        elif isinstance(a, TensorOptionsArguments):
            return [a.dtype, a.layout, a.device, a.pin_memory]
        else:
            assert_never(a)

    return list(
        concatMap(
            to_argument,
            itertools.chain(
                func.arguments.positional, func.arguments.kwarg_only, func.arguments.out
            ),
        )
    )


def argument(
    a: Argument, *, remove_non_owning_ref_types: bool = False, symint: bool = True
) -> Binding:
    # [FUNCTION DOC] Builds a Binding (named C++ type, name, source argument) for one
    # Argument, with no default value.
    return Binding(
        nctype=argument_type(
            a,
            binds=a.name,
            remove_non_owning_ref_types=remove_non_owning_ref_types,
            symint=symint,
        ),
        name=a.name,
        argument=a,
    )


def arguments(func: FunctionSchema, *, symint: bool = True) -> list[Binding]:
    # [FUNCTION DOC] Returns Bindings for all flattened JIT arguments of a schema.
    return [argument(a, symint=symint) for a in jit_arguments(func)]
