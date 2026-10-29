# [MODULE DOC] Signature objects that render C++ declarations/definitions for a function schema
# under the different codegen APIs: public C++ API (CppSignature, CppSignatureGroup),
# dispatcher, native, view inverse and structured impl signatures, plus kernel_signature to
# choose between native and dispatcher. The torchgen.api submodule imports are at the very
# bottom of the file, deliberately after the class definitions, to avoid circular imports.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.api.types.types_base import Binding, CType, Expr


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torchgen.model import (
        BackendIndex,
        FunctionSchema,
        NativeFunction,
        NativeFunctionsGroup,
        NativeFunctionsViewGroup,
    )


@dataclass(frozen=True)
class CppSignature:
    # [CLASS DOC] One C++ API overload derived from a FunctionSchema. Frozen dataclass with fields
    # func, method, faithful, symint, cpp_no_default_args and fallback_binding. Methods compute
    # the argument bindings, the (possibly dispatch-prefixed) name, and declaration, definition,
    # function pointer and function type strings.
    """
    A CppSignature represents a single overload in the C++ API.  For
    any given function schema, there may be multiple CppSignatures
    corresponding to it, based on how we desugar to C++.  See also
    CppSignatureGroup.
    """

    # The schema this signature is derived from
    func: FunctionSchema

    # Is this a C++ signature for a method, i.e. Tensor::my_op(...)?
    method: bool

    # Is this a faithful C++ signature (i.e. following the JIT schema) or a convenience API
    # (i.e. with a potential TensorOptions argument and out arguments in the front)
    faithful: bool

    # Is this a symint C++ signature.  For BC reasons, functions that take
    # SymInts still present as int64_t in C++, and the SymInt variant is
    # offered at a different overload name
    #
    # NB: If a function RETURNS a SymInt, this is ALWAYS false
    symint: bool

    # The set of C++ arguments which should not have defaults applied to them
    cpp_no_default_args: set[str]

    # Is this a fallback C++ binding?  Fallback bindings are enabled by
    # manual_cpp_binding: True and are alternate, non-public API that
    # lets manual C++ binding implementers access the binding that would
    # have been automatically generated
    fallback_binding: bool = False

    # Return the unpacked argument structure of this signature,
    # discarding information about which arguments are semantically
    # related to each other.
    def arguments(self) -> Sequence[Binding]:
        # [FUNCTION DOC] Return the Binding list for this signature from cpp.arguments, honoring
        # the faithful, symint, method and cpp_no_default_args settings.
        return cpp.arguments(
            self.func.arguments,
            faithful=self.faithful,
            symint=self.symint,
            method=self.method,
            cpp_no_default_args=self.cpp_no_default_args,
        )

    def name(self, *, suppress_symint_suffix: bool = False) -> str:
        # [FUNCTION DOC] C++ function name from cpp.name. With suppress_symint_suffix the symint
        # overload name is not used. Fallback bindings get a __dispatch_ prefix.
        n = cpp.name(
            self.func,
            faithful_name_for_out_overloads=self.faithful,
            symint_overload=False if suppress_symint_suffix else self.symint,
        )
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # Render the C++ declaration for this signature
    def decl(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
        suppress_symint_suffix: bool = False,
    ) -> str:
        # [FUNCTION DOC] Render the declaration string: return type, name and arguments with
        # defaults. The name defaults to prefix plus the signature name. If is_redispatching_fn,
        # a leading DispatchKeySet parameter is prepended. Pure string building.
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        cpp_args = [a.decl() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        cpp_args_str = ", ".join(cpp_args)
        if name is None:
            name = prefix + self.name(suppress_symint_suffix=suppress_symint_suffix)
        return f"{returns_type} {name}({cpp_args_str})"

    # Render the C++ definition for this signature, not including
    # the body (with curly braces)
    def defn(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
    ) -> str:
        # [FUNCTION DOC] Render the definition head (no body): like decl but arguments are rendered
        # without default values and suppress_symint_suffix is not offered.
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        cpp_args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        cpp_args_str = ", ".join(cpp_args)
        if name is None:
            name = prefix + self.name()
        return f"{returns_type} {name}({cpp_args_str})"

    def ptr_type(self) -> str:
        # [FUNCTION DOC] Return the C++ function pointer type string, return type then (*) and the
        # argument types.
        args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_types_str})"

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        # [FUNCTION DOC] Return the C++ function type string: return type followed by the
        # argument types in parentheses.
        args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} ({args_types_str})"


# Represents group of all CppSignatures associated with a
# FunctionSchema.  Right now, that's the regular, user-visible
# signature, as well as a "faithful" signature which doesn't
# have grouping.
@dataclass(frozen=True)
class CppSignatureGroup:
    # [CLASS DOC] All CppSignatures of one FunctionSchema: the regular signature, an optional
    # faithful one (present when the schema has TensorOptions or out arguments) and optional
    # symint variants of each (present when the schema has SymInt). Frozen dataclass.
    func: FunctionSchema
    signature: CppSignature
    faithful_signature: CppSignature | None
    symint_signature: CppSignature | None
    symint_faithful_signature: CppSignature | None

    def most_faithful_signature(self) -> CppSignature:
        # [FUNCTION DOC] Return the faithful signature if present, else the regular signature.
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    def signatures(self, *, symint: bool = True) -> Iterator[CppSignature]:
        # [FUNCTION DOC] Generator yielding the regular signature, the faithful one if present and,
        # when symint is true, the symint and symint-faithful ones if present.
        yield self.signature
        if self.faithful_signature:
            yield self.faithful_signature
        if symint:
            if self.symint_signature:
                yield self.symint_signature
            if self.symint_faithful_signature:
                yield self.symint_faithful_signature

    @staticmethod
    def from_native_function(
        f: NativeFunction, *, method: bool, fallback_binding: bool = False
    ) -> CppSignatureGroup:
        # [FUNCTION DOC] Build the CppSignatureGroup for NativeFunction f. A faithful signature is
        # made only when the schema has tensor options or out arguments; symint signatures only
        # when the schema has SymInt. method and fallback_binding are copied to every signature.
        func = f.func

        def make_sig(*, faithful: bool, symint: bool) -> CppSignature:
            # [FUNCTION DOC] HELPER: Create one CppSignature for the faithful and symint flags.
            return CppSignature(
                func=func,
                faithful=faithful,
                symint=symint,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args,
            )

        def make_sigs(*, symint: bool) -> tuple[CppSignature, CppSignature | None]:
            # [FUNCTION DOC] HELPER: Return the (regular, faithful or None) pair for a symint flag.
            faithful_signature: CppSignature | None = None
            if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
                faithful_signature = make_sig(faithful=True, symint=symint)
            signature = make_sig(faithful=False, symint=symint)
            return signature, faithful_signature

        signature, faithful_signature = make_sigs(symint=False)
        symint_signature: CppSignature | None = None
        symint_faithful_signature: CppSignature | None = None
        if func.has_symint():
            symint_signature, symint_faithful_signature = make_sigs(symint=True)

        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
            symint_signature=symint_signature,
            symint_faithful_signature=symint_faithful_signature,
        )


@dataclass(frozen=True)
class DispatcherSignature:
    # [CLASS DOC] Signature of an operator under the dispatcher API: func, an optional name prefix
    # (to avoid collisions in generated wrappers) and a symint flag (default true).
    # The schema this signature is derived from
    func: FunctionSchema

    # Allows you to prepend an arbitrary prefix to the signature name.
    # This is useful for parts of the codegen that generate wrappers around kernels,
    # and need to avoid naming collisions.
    prefix: str = ""

    symint: bool = True

    def arguments(self) -> list[Binding]:
        # [FUNCTION DOC] Return the dispatcher Binding list for the schema.
        return dispatcher.arguments(self.func, symint=self.symint)

    def name(self) -> str:
        # [FUNCTION DOC] Return the prefix followed by the dispatcher name of the schema.
        return self.prefix + dispatcher.name(self.func)

    def decl(self, name: str | None = None) -> str:
        # [FUNCTION DOC] Render the declaration string (return type, name, arguments with
        # defaults). The name defaults to self.name().
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def defn(
        self, name: str | None = None, *, is_redispatching_fn: bool = False
    ) -> str:
        # [FUNCTION DOC] Render the definition head without defaults; optionally prepend a
        # DispatchKeySet parameter when is_redispatching_fn is true.
        args = [a.defn() for a in self.arguments()]
        if is_redispatching_fn:
            args = ["c10::DispatchKeySet dispatchKeySet"] + args
        args_str = ", ".join(args)
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    def exprs(self) -> list[Expr]:
        # [FUNCTION DOC] Return an Expr (name plus type) for each argument, to pass them onward.
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    def returns_type(self) -> CType:
        # [FUNCTION DOC] Return the dispatcher CType of the return value.
        return dispatcher.returns_type(self.func.returns, symint=self.symint)

    def ptr_type(self) -> str:
        # [FUNCTION DOC] Return the function pointer type string for this signature.
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})"

    # Return the C++ function type, e.g., something like int(bool)
    def type(self) -> str:
        # [FUNCTION DOC] Return the function type string, return type then argument types.
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} ({dispatcher_args_types_str})"

    @staticmethod
    def from_schema(
        func: FunctionSchema, *, prefix: str = "", symint: bool = True
    ) -> DispatcherSignature:
        # [FUNCTION DOC] Construct a DispatcherSignature from a schema, prefix and symint flag.
        return DispatcherSignature(func, prefix, symint)


@dataclass(frozen=True)
class NativeSignature:
    # [CLASS DOC] Signature of an operator under the native (kernel) API: func, symint (required)
    # and an optional name prefix. Methods render declaration/definition strings and can produce
    # the expressions needed to call the dispatcher API from native arguments.
    # The schema this signature is derived from
    func: FunctionSchema

    symint: bool

    prefix: str = ""

    def name(self) -> str:
        # [FUNCTION DOC] Return the prefix followed by the native name of the schema.
        return self.prefix + native.name(self.func)

    def decl(self, name: str | None = None) -> str:
        # [FUNCTION DOC] Render the declaration with default values; name defaults to self.name().
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def defn(self, name: str | None = None) -> str:
        # [FUNCTION DOC] Render the definition head (arguments without defaults).
        args_str = ", ".join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    def ptr_type(self) -> str:
        # [FUNCTION DOC] Return the function pointer type string (argument types, no defaults).
        # don't include defaults in type signature!
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})"

    def arguments(self) -> list[Binding]:
        # [FUNCTION DOC] Return the native Binding list for the schema.
        return native.arguments(self.func, symint=self.symint)

    def returns_type(self) -> CType:
        # [FUNCTION DOC] Return the native CType of the return value.
        return native.returns_type(self.func.returns, symint=self.symint)

    def dispatcher_exprs(self) -> list[Expr]:
        # [FUNCTION DOC] Translate the native arguments into expressions matching the dispatcher
        # arguments (using translate.translate). Raises UnsatError if impossible.
        return translate.translate(
            self.arguments(), dispatcher.arguments(self.func), method=False
        )


@dataclass(frozen=True)
class ViewInverseSignature:
    # [CLASS DOC] Signature of the inverse function of a view op, for a NativeFunctionsViewGroup g.
    # Used to declare the static reverse function in the functionalization codegen.
    g: NativeFunctionsViewGroup

    def name(self) -> str:
        # [FUNCTION DOC] Return the reverse function name without a namespace.
        return functionalization.reverse_name(self.g.view, include_namespace=False)

    def decl(self) -> str:
        # [FUNCTION DOC] Render a static declaration, ending in a semicolon, of the reverse
        # function using the reverse-direction argument list of the view operator.
        return_type = functionalization.returns_type(self.g.view.func)
        decls = [
            a.decl()
            for a in functionalization.op_arguments(self.g.view.func, is_reverse=True)
        ]
        return f"static {return_type.cpp_type()} {self.name()}({', '.join(decls)});"


@dataclass(frozen=True)
class StructuredImplSignature:
    # [CLASS DOC] Signature of a structured kernel impl function for NativeFunctionsGroup g, with
    # the function name stored in name. Rendered as a TORCH_IMPL_FUNC definition.
    g: NativeFunctionsGroup
    name: str

    def defn(self, name: str | None = None) -> str:
        # [FUNCTION DOC] Render the TORCH_IMPL_FUNC definition head with the impl argument list.
        # # KNOWN ISSUE: the name parameter is ignored; the stored self.name is always used.
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"TORCH_IMPL_FUNC({self.name})({args_str})"

    def arguments(self) -> list[Binding]:
        # [FUNCTION DOC] Return the impl Binding list from structured.impl_arguments for the group.
        return structured.impl_arguments(self.g)


# Helper functions


def kernel_signature(
    f: NativeFunction, backend_index: BackendIndex, *, prefix: str = ""
) -> NativeSignature | DispatcherSignature:
    # [FUNCTION DOC] Pick the signature for the kernel of f: NativeSignature for in-tree backends
    # or DispatcherSignature for external backends, with the given prefix. symint is true when the
    # backend kernel metadata says it supports symint. Raises AssertionError if symint is wanted
    # but the schema has no SymInt.
    # Note [External Backends Follow Dispatcher API]
    # Kernel signatures for in-tree backends follow the "native" API,
    # while kernels for out-of-tree backends follow the dispatcher API.
    # See the comments in `native.py` for details, but historically there have been
    # some small differences in schema convention between them and the Dispatcher API.
    # Any differences that require translating between the two will results in a runtime cost,
    # so we'd like to keep the differences as small as possible.
    # With external backends, we'd like to enforce that they write their kernels with schemas
    # that match the Dispatcher API directly, if they can.
    meta = backend_index.get_kernel(f)
    symint = meta is not None and meta.supports_symint()
    if symint:
        if not f.func.has_symint():
            raise AssertionError(
                f"attempted to define symint kernel for {backend_index.dispatch_key} "
                "without SymInt in schema"
            )
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix, symint=symint)
    else:
        return NativeSignature(f.func, prefix=prefix, symint=symint)


# Functions only, no types
from torchgen.api import (
    cpp,
    dispatcher,
    functionalization,
    native,
    structured,
    translate,
)
