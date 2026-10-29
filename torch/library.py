# mypy: allow-untyped-defs
# [MODULE DOC] Public Python API for registering operators and kernels with the dispatcher (torch.library).
# Provides the Library class (a wrapper over a C++ dispatcher library fragment, DEF, or IMPL), and
# functional registration helpers: define, impl, register_kernel, register_autocast, register_fake,
# register_autograd, register_torch_dispatch, register_vmap, get_ctx, get_kernel, opcheck. It also
# re-exports custom_op, triton_op, wrap_triton and infer_schema from torch._library.
# Module-level state tracks registered (namespace/op/dispatch key) kernels and defined ops so that
# Library cleanup can undo them, and keeps anonymous Library objects alive.
# Import-time side effects: none beyond creating the module-level sets and lists.
import contextlib
import functools
import inspect
import re
import sys
import weakref
from collections.abc import Callable, Sequence
from typing import Any, overload, TYPE_CHECKING, TypeVar, Union
from typing_extensions import deprecated, ParamSpec

import torch
import torch._library as _library
from torch._library.custom_ops import (
    _cast,
    _maybe_get_opdef,
    custom_op,
    CustomOpDef,
    device_types_t,
)
from torch._library.effects import EffectType
from torch._library.infer_schema import infer_schema
from torch._library.triton import triton_op, wrap_triton
from torch._ops import OpOverload
from torch.types import _dtype


__all__ = [
    "Library",
    "impl",
    "define",
    "fallthrough_kernel",
    "impl_abstract",
    "register_autocast",
    "register_fake",
    "register_torch_dispatch",
    "register_vmap",
    "get_ctx",
    "get_kernel",
    "custom_op",
    "triton_op",
    "wrap_triton",
    "infer_schema",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: set[str] = set()
_defs: set[str] = set()

# prim is reserved by TorchScript interpreter
_reserved_namespaces = ["prim"]


def fallthrough_kernel():
    # [FUNCTION DOC] Placeholder kernel passed to Library.impl to register a fallthrough. Takes no arguments and always raises NotImplementedError if actually called.
    """
    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.
    """
    raise NotImplementedError("fallthrough_kernel() should never be called.")


def _validate_out_schema(schema: "str | torch._C.FunctionSchema") -> None:
    # [FUNCTION DOC] HELPER: Check that a schema (string or parsed schema) can carry the out tag.
    # Requires at least one mutable argument, all mutable arguments keyword-only and Tensor typed, and
    # one return per mutable argument that is a mutable alias of the matching argument, in order.
    # Raises ValueError describing the first violated rule. No side effects.
    """Validate that a schema has valid out semantics, i.e., it can be tagged with torch.Tag.out.

    Requirements:
    - Must have at least one mutable argument
    - All returns must alias the mutable args in declaration order

    torchgen has equivalent checks (torchgen/model.py), but we reimplement them here
    because (1) it's simple and (2) torchgen uses a different schema object
    (torchgen.model.FunctionSchema vs torch._C.FunctionSchema) so it's difficult to
    share the function.
    """
    if isinstance(schema, str):
        schema = torch._C.parse_schema(schema)
    mutable_args = [
        arg
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    if not mutable_args:
        raise ValueError(
            f"Schema tagged with torch.Tag.out must have at least one mutable argument. "
            f"Got: {schema}"
        )
    positional_mutable = [arg for arg in mutable_args if not arg.kwarg_only]
    if positional_mutable:
        names = [a.name for a in positional_mutable]
        raise ValueError(
            f"Schema tagged with torch.Tag.out requires all mutable arguments to be "
            f"keyword-only (after the *). Found mutable positional args: {names}. "
            f"Got: {schema}"
        )
    unsupported_mutable = [
        arg for arg in mutable_args if not isinstance(arg.type, torch.TensorType)
    ]
    if unsupported_mutable:
        names = [a.name for a in unsupported_mutable]
        raise ValueError(
            f"Schema tagged with torch.Tag.out only supports Tensor mutable arguments. "
            f"Found unsupported mutable args: {names}. Got: {schema}"
        )
    returns = schema.returns
    if len(returns) != len(mutable_args):
        raise ValueError(
            f"Schema tagged with torch.Tag.out must return all mutable arguments "
            f"(got {len(mutable_args)} mutable args but {len(returns)} returns). "
            f"Got: {schema}"
        )
    for i, (ret, arg) in enumerate(zip(returns, mutable_args, strict=True)):
        arg_alias = arg.alias_info
        ret_alias = ret.alias_info
        if ret_alias is None:
            raise ValueError(
                f"Return {i} of schema tagged with torch.Tag.out must alias mutable arg '{arg.name}'. "
                f"Got: {schema}"
            )
        if not ret_alias.is_write:
            raise ValueError(
                f"Return {i} of schema tagged with torch.Tag.out must be a mutable alias "
                f"(e.g., Tensor(a!), not Tensor(a)) of arg '{arg.name}'. "
                f"Got: {schema}"
            )
        # arg_alias is guaranteed non-None by the mutable_args filter above
        if ret_alias.before_set != arg_alias.before_set:  # type: ignore[union-attr]
            raise ValueError(
                f"Return {i} of schema tagged with torch.Tag.out must alias mutable arg '{arg.name}' "
                f"(return aliases {ret_alias.before_set} but arg aliases {arg_alias.before_set}). "  # type: ignore[union-attr]
                f"Got: {schema}"
            )


def _validate_inplace_schema(schema: "str | torch._C.FunctionSchema") -> None:
    # [FUNCTION DOC] HELPER: Check that a schema (string or parsed schema) can carry the inplace tag.
    # Requires the first positional argument to be a mutable Tensor, exactly one return that is a
    # Tensor mutably aliasing it, and no other mutable arguments. Raises ValueError on the first
    # violated rule. No side effects.
    """Validate that a schema has valid inplace semantics, i.e., it can be tagged with torch.Tag.inplace.

    Requirements:
    - The first positional argument must be a mutable Tensor (Tensor(a!))
    - Must return exactly that Tensor
    - Only the first positional argument may be mutable

    Native inplace ops (defined in native_functions.yaml) have relaxed requirements:
    they may have Tensor[] self and multiple mutable arguments. This validation is
    stricter because it is only used for custom ops defined via torch.library.
    """
    if isinstance(schema, str):
        schema = torch._C.parse_schema(schema)
    args = schema.arguments
    if not args or args[0].kwarg_only:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must have at least one positional argument. "
            f"Got: {schema}"
        )
    first_arg = args[0]
    if first_arg.alias_info is None or not first_arg.alias_info.is_write:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace requires the first positional argument "
            f"to be mutable (e.g., Tensor(a!)). Got: {schema}"
        )
    if not isinstance(first_arg.type, torch.TensorType):
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace requires the first positional argument "
            f"to be a Tensor. Got type '{first_arg.type}' for arg '{first_arg.name}'. "
            f"Got: {schema}"
        )
    # Must return exactly one Tensor aliasing the first arg
    returns = schema.returns
    if len(returns) != 1:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return exactly one value "
            f"(the first argument). Got {len(returns)} returns. Got: {schema}"
        )
    ret = returns[0]
    if not isinstance(ret.type, torch.TensorType):
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return must be a Tensor, got type '{ret.type}'). Got: {schema}"
        )
    if ret.alias_info is None:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return must alias the first argument). Got: {schema}"
        )
    if not ret.alias_info.is_write:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return must be a mutable alias, e.g., Tensor(a!)). Got: {schema}"
        )
    if ret.alias_info.before_set != first_arg.alias_info.before_set:
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must return the first mutable argument "
            f"(return aliases {ret.alias_info.before_set} but first arg aliases "
            f"{first_arg.alias_info.before_set}). Got: {schema}"
        )
    other_mutable = [
        arg
        for arg in args[1:]
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    if other_mutable:
        names = [a.name for a in other_mutable]
        raise ValueError(
            f"Schema tagged with torch.Tag.inplace must only mutate the first positional argument. "
            f"Found additional mutable args: {names}. Got: {schema}"
        )


class Library:
    # [CLASS DOC] Wrapper around a C++ dispatcher library object for one namespace, used to define
    # operators and register kernels from Python. Created with a namespace, a kind (DEF, IMPL or
    # FRAGMENT) and an optional default dispatch key; DEF and FRAGMENT are rejected for reserved
    # namespaces. State: m (the C++ handle, None once destroyed), ns, kind, dispatch_key, the sets of
    # op definitions and kernel keys it registered, and a list of registration handles. A weakref
    # finalizer undoes all registrations when the object is collected; _destroy does the same eagerly.
    # Key methods: define, impl, fallback, register_symm_mem_args, and private fake/dispatch helpers.
    """
    A class to create libraries that can be used to register new operators or
    override operators in existing libraries from Python.
    A user can optionally pass in a dispatch keyname if they only want to register
    kernels corresponding to only one specific dispatch key.

    To create a library to override operators in an existing library (with name ns), set the kind to "IMPL".
    To create a new library (with name ns) to register new operators, set the kind to "DEF".
    To create a fragment of a possibly existing library to register operators (and bypass
    the limitation that there is only one library for a given namespace), set the kind to
    "FRAGMENT".

    Args:
        ns: library name
        kind: "DEF", "IMPL", "FRAGMENT"
        dispatch_key: PyTorch dispatch key (default: "")
    """

    def __init__(self, ns, kind, dispatch_key=""):
        # [FUNCTION DOC] Create the underlying C++ library for namespace ns with the given kind and dispatch key,
        # recording the calling file and line number. Raises ValueError for an unsupported kind or for
        # DEF/FRAGMENT on a reserved namespace. Initializes bookkeeping sets and registers a weakref
        # finalizer (not __del__) that unregisters everything and clears caches when the object dies.
        from torch.fx.operator_schemas import _SCHEMA_TO_SIGNATURE_CACHE

        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError(f"Unsupported kind: {kind}")

        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                f"{ns} is a reserved namespace. Please try creating a library with another name."
            )

        f = sys._getframe(1)
        filename, lineno = f.f_code.co_filename, f.f_lineno
        self.m: Any | None = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
        self.ns = ns
        self._op_defs: set[str] = set()
        self._op_impls: set[str] = set()
        self._registration_handles: list[torch._library.utils.RegistrationHandle] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        # Use a finalizer to setup the "destructor" instead of __del__.
        # Python __del__ can lead to weird things (globals and locals may already
        # be gone when __del__ actually gets called!). finalizers help the
        # situation because it lets us capture references and keeps them alive
        weakref.finalize(
            self,
            _del_library,
            _impls,
            self._op_impls,
            _defs,
            self._op_defs,
            self._registration_handles,
            self.m,
            _SCHEMA_TO_SIGNATURE_CACHE,
        )

    def __repr__(self):
        # [FUNCTION DOC] Return a string describing the library kind, namespace and dispatch key.
        # KNOWN ISSUE: the returned text ends with a stray closing angle bracket with no opening one.
        return f"Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>"

    def define(self, schema, alias_analysis="", *, tags=()):
        # [FUNCTION DOC] Define a new operator from a schema string in this library's namespace.
        # Validates alias_analysis (only empty, FROM_SCHEMA or CONSERVATIVE allowed, else RuntimeError),
        # normalizes tags to a tuple, and validates the schema for the out and inplace tags (ValueError).
        # Calls into C++ to define the op, refreshes an already existing torch.ops packet so the new
        # overload is visible, and records the qualified name in the library and global definition sets.
        # Returns the operator name from C++. Raises AssertionError if the library was destroyed.
        r"""Defines a new operator and its semantics in the ns namespace.

        Args:
            schema: function schema to define a new operator.
            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be
                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").
            tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
                                       operator. Tagging an operator changes the operator's behavior
                                       under various PyTorch subsystems; please read the docs for the
                                       torch.Tag carefully before applying it.

        Returns:
            name of the operator as inferred from the schema.

        Example::

            >>> my_lib = Library("mylib", "DEF")
            >>> my_lib.define("sum(Tensor self) -> Tensor")
        """

        # This is added because we also want to disallow PURE_FUNCTION alias analysis which is a valid
        # AliasAnalysis type in C++
        if alias_analysis not in ["", "FROM_SCHEMA", "CONSERVATIVE"]:
            raise RuntimeError(f"Invalid alias_analysis type {alias_analysis}")
        if self.m is None:
            raise AssertionError("Library object has been destroyed")
        if isinstance(tags, torch.Tag):
            tags = (tags,)

        name = schema.split("(")[0]
        packet_name = name.split(".")[0] if "." in name else name
        has_preexisting_packet = hasattr(torch.ops, self.ns) and hasattr(
            getattr(torch.ops, self.ns), packet_name
        )

        if torch.Tag.out in tags:
            _validate_out_schema(schema)
        if torch.Tag.inplace in tags:
            _validate_inplace_schema(schema)

        result = self.m.define(schema, alias_analysis, tuple(tags))
        name = schema.split("(")[0]
        # C++ accepts a name prefixed with the matching namespace ("ns::foo");
        # don't double-prepend the namespace in that case.
        qualname = name if "::" in name else f"{self.ns}::{name}"

        # If the OpOverloadPacket exists already, then this means we're adding a
        # new OpOverload for it. Refresh the packet to include the new OpOverload.
        if has_preexisting_packet:
            ns = getattr(torch.ops, self.ns)
            packet = getattr(ns, packet_name)
            torch._ops._refresh_packet(packet)

        self._op_defs.add(qualname)
        _defs.add(qualname)
        return result

    def _register_fake(self, op_name, fn, _stacklevel=1, *, allow_override=True):
        # [FUNCTION DOC] HELPER: Register a Python fake (meta) implementation for op_name in this library.
        # Finds the caller module via the stack level, and (except for torchvision callers) wraps the
        # function so a one-time check verifies the C++ set_python_module declaration matches the caller.
        # Registers on the op's registry entry and stores the handle for later cleanup.
        # Args include allow_override, which controls whether an existing fake impl may be replaced.
        r"""Registers the fake impl for an operator defined in the library."""

        source = torch._library.utils.get_source(_stacklevel + 1)
        frame = sys._getframe(_stacklevel)
        caller_module = inspect.getmodule(frame)
        # Can be none if you call register_fake from somewhere there isn't a module
        # (e.g. __main__)
        caller_module_name = None if caller_module is None else caller_module.__name__

        # TODO(rzou): We're gonna need to stage this change with torchvision,
        # since torchvision is github first.
        if caller_module_name is not None and caller_module_name.startswith(
            "torchvision."
        ):
            caller_module_name = None

        qualname = f"{self.ns}::{op_name}"
        entry = torch._library.simple_registry.singleton.find(qualname)
        if caller_module_name is not None:
            func_to_register = _check_pystubs_once(fn, qualname, caller_module_name)
        else:
            func_to_register = fn

        handle = entry.fake_impl.register(
            func_to_register, source, lib=self, allow_override=allow_override
        )
        self._registration_handles.append(handle)

    def _register_torch_dispatch_rule(self, op_name, torch_dispatch_class, fn):
        # [FUNCTION DOC] HELPER: Register a torch_dispatch rule for an operator and a dispatch class (a Tensor
        # subclass or a TorchDispatchMode) on the operator's registry entry, and keep the handle so it is
        # removed when the library is destroyed.
        r"""Registers a torch_dispatch rule for the given operator and torch_dispatch_class.

        This allows for open registration to specify the behavior between the operator
        and the torch_dispatch_class without needing to modify the torch_dispatch_class
        or the operator directly.

        The torch_dispatch_class is either a Tensor subclass with `__torch_dispatch__` or a
        TorchDispatchMode.

        If it is a Tensor subclass, we expect fn to have the following signature:
        (cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any

        If it is a TorchDispatchMode, we expect fn to have the following signature:
        (mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any
        """

        qualname = f"{self.ns}::{op_name}"
        entry = torch._library.simple_registry.singleton.find(qualname)
        handle = entry.torch_dispatch_rules.register(torch_dispatch_class, fn)
        self._registration_handles.append(handle)

    def _resolve_op_name(self, op_name, api_name):
        # [FUNCTION DOC] HELPER: Turn a string or OpOverload into an operator name string. For an OpOverload
        # the schema name plus a dot and the overload name (if non-empty) is returned. Raises RuntimeError
        # mentioning api_name for any other type.
        """Resolve op_name (str or OpOverload) to a name string."""
        if isinstance(op_name, str):
            return op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name:
                name = name + "." + overload_name
            return name
        else:
            raise RuntimeError(
                f"{api_name} should be passed either a name or an OpOverload object "
                f"as the first argument, got {type(op_name)}"
            )

    def _impl_with_aoti_compile(self, op_name, dispatch_key=""):
        # [FUNCTION DOC] HELPER: Register the AOTI-compiled implementation of an operator for a dense dispatch key.
        # Defaults the key to the library's key, requires it to include the Dense key set (AssertionError
        # otherwise), raises RuntimeError if a Python kernel was already registered for the same
        # namespace/op/key, then calls the C++ handle and records the key in the global and per-library sets.
        r"""Register the operator to use the AOTI-compiled implementation.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::

            >>> my_lib = Library("aten", "IMPL")
            >>> warnings.filterwarnings("ignore", message=".*other operators may also be overridden")  # docs: hide
            >>> my_lib._impl_with_aoti_compile("div.Tensor", "CPU")
        """

        if dispatch_key == "":
            dispatch_key = self.dispatch_key
        # pyrefly: ignore [bad-argument-type]
        if not torch.DispatchKeySet(dispatch_key).has(torch._C.DispatchKey.Dense):
            raise AssertionError(
                f"dispatch_key {dispatch_key} does not have Dense in its keyset"
            )

        name = self._resolve_op_name(op_name, "_impl_with_aoti_compile")

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when _impl_with_aoti_compile is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        if self.m is None:
            raise AssertionError("Library object has been destroyed")
        impl_fn: Callable = self.m.impl_with_aoti_compile
        impl_fn(self.ns, name.split("::")[-1], dispatch_key)

        _impls.add(key)
        self._op_impls.add(key)

    def impl(
        self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=True
    ):
        # [FUNCTION DOC] Register a Python function as the kernel for an operator and dispatch key.
        # Requires fn to be callable (TypeError); the key defaults to the library's key. If allow_override is
        # false and a Python kernel already exists for the same namespace/op/key, raises RuntimeError.
        # Refuses Meta kernels for ops that already have a CompositeImplicitAutograd kernel (RuntimeError).
        # Calls the C++ handle (empty key means CompositeImplicitAutograd), optionally passing the dispatch
        # keyset as the first argument, and records the key in the global and per-library sets.
        r"""Registers the function implementation for an operator defined in the library.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: function that's the operator implementation for the input dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.
            allow_override: Flag controlling if we want to override an
                         existing registered kernel implementation. This is on
                         by default; pass ``False`` to error if a kernel is
                         already registered for this dispatch key.

        Example::
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> my_lib = Library("aten", "IMPL")
            >>> def div_cpu(self, other):
            >>>     return self * (1 / other)
            >>> my_lib.impl("div.Tensor", div_cpu, "CPU")
        """

        if not callable(fn):
            raise TypeError(
                f"Input function is required to be a callable but found type {type(fn)}"
            )
        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        name = self._resolve_op_name(op_name, "impl")

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if (not allow_override) and key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when impl is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        if dispatch_key == "Meta":
            dispatcher_op_name = name
            if "::" not in dispatcher_op_name:
                dispatcher_op_name = f"{self.ns}::{dispatcher_op_name}"

            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            if torch._C._dispatch_has_kernel_for_dispatch_key(
                dispatcher_op_name, "CompositeImplicitAutograd"
            ):
                raise RuntimeError(
                    f"We should not register a meta kernel directly to the operator '{name}',"
                    " because it has a CompositeImplicitAutograd kernel in core."
                    " Instead we should let the operator decompose, and ensure that we have meta kernels"
                    " for the base ops that it decomposes into."
                )

        if self.m is None:
            raise AssertionError("Library object has been destroyed")
        self.m.impl(
            name,
            dispatch_key if dispatch_key != "" else "CompositeImplicitAutograd",
            fn,
            with_keyset,
        )

        _impls.add(key)
        self._op_impls.add(key)

    def register_symm_mem_args(self, op_name, arg_names):
        # [FUNCTION DOC] Declare which argument names of an operator need symmetric memory allocation, so that
        # Inductor can allocate them in peer-to-peer accessible memory. Resolves the qualified name, then
        # registers the argument names on the op's registry entry and stores the handle for cleanup.
        r"""Registers which arguments require symmetric memory allocation for an operator.

        This method allows operators to declaratively specify which arguments need
        symmetric memory treatment. When used with ``torch.compile``, Inductor
        automatically allocates these arguments in P2P-accessible NVLink memory
        (``empty_strided_p2p``), enabling zero-copy inter-GPU communication.

        The operator's schema should include a ``group_name`` (``str``) argument.
        Inductor's ``FallbackKernel._maybe_realize_symm_mem_args`` extracts
        ``group_name`` at compile time to allocate P2P buffers for the correct
        process group. Without ``group_name``, the automatic realization is skipped.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            arg_names: list of argument names that require symmetric memory allocation.

        Example::
            >>> # xdoctest: +SKIP(reason="illustrative example, not runnable")
            >>> my_lib = Library("symm_mem", "FRAGMENT")
            >>> my_lib.define("one_shot_all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor")
            >>> my_lib.register_symm_mem_args("one_shot_all_reduce", ["input"])
        """
        name = self._resolve_op_name(op_name, "register_symm_mem_args")
        if "::" in name:
            qualname = name
        else:
            qualname = f"{self.ns}::{name}"

        entry = torch._library.simple_registry.singleton.find(qualname)
        handle = entry.symm_mem_args.register(arg_names)
        self._registration_handles.append(handle)

    def fallback(self, fn, dispatch_key="", *, with_keyset=False):
        # [FUNCTION DOC] Register fn as the fallback kernel for a dispatch key. Only valid on a library for the
        # global namespace "_" (RuntimeError otherwise). The key defaults to the library's key and must be
        # non-empty; AssertionError if empty or if the library was destroyed. fn may be the fallthrough
        # kernel; with_keyset passes the dispatch keyset as the first argument.
        r"""Registers the function implementation as the fallback for the given key.

        This function only works for a library with global namespace ("_").

        Args:
            fn: function used as fallback for the given dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.

        Example::

            >>> my_lib = Library("_", "IMPL")
            >>> def fallback_kernel(op, *args, **kwargs):
            >>>     # Handle all autocast ops generically
            >>>     # ...
            >>> my_lib.fallback(fallback_kernel, "Autocast")
        """

        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        if self.ns != "_":
            raise RuntimeError(
                f"""Fallback can only be registered using library fragment on the global namespace "_" but it is {self.ns}"""
            )

        if dispatch_key == "":
            raise AssertionError("dispatch_key must not be empty for fallback")
        if self.m is None:
            raise AssertionError("Library object has been destroyed")

        self.m.fallback(dispatch_key, fn, with_keyset)

    def _register_effectful_op(self, op_name: str, effect: EffectType | None):
        # [FUNCTION DOC] HELPER: Register an effect type (or None) for an operator through the higher-order-ops
        # effects registry and keep the returned handle for cleanup.
        # KNOWN ISSUE: the docstring mentions OpOverload but the annotation says str only.
        """
        Registers an effect to an operator. This is used to register an op that
        has side effects that is not capturable by the schema.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            effect: The effect of the op.
        """
        from torch._higher_order_ops.effects import (
            _register_effectful_op as hoo_register_effect,
        )

        handle = hoo_register_effect(op_name, effect)
        self._registration_handles.append(handle)

    def _destroy(self):
        # [FUNCTION DOC] HELPER: Eagerly tear down the library: reset the C++ handle and clear it, destroy all
        # registration handles, remove this library's kernel keys from the global set, and clear cached
        # torch.ops entries for its definitions. After this, define/impl raise AssertionError.
        # KNOWN ISSUE: global definition set is not updated here, unlike in the finalizer path.
        if self.m is not None:
            self.m.reset()
        self.m = None
        for handle in self._registration_handles:
            handle.destroy()
        self._registration_handles.clear()
        global _impls
        _impls -= self._op_impls
        _clear_torch_ops_cache(self._op_defs)


def _clear_torch_ops_cache(op_defs):
    # [FUNCTION DOC] HELPER: For each qualified op name (namespace::name), delete the cached attribute from the
    # torch.ops namespace object and its directory list so it is recomputed on next access. Skips
    # malformed names, unknown namespaces and uncached names; avoids attribute lookups that could hit C++
    # during interpreter shutdown.
    # Delete the cached torch.ops.ns.foo if it was registered.
    # Otherwise, accessing it leads to a segfault, or to enumerations
    # over the namespace (e.g. _collect_all_valid_cia_ops) tripping on
    # an OpOverloadPacket whose underlying overload was just removed.
    # It's possible that we only registered an overload in this Library
    # and another library owns an alive overload.
    # That's OK - the next time torch.ops.ns.foo gets called, it'll be
    # recomputed to point at the right collection of overloads.
    for qualname in op_defs:
        splits = qualname.split("::")
        if len(splits) != 2:
            # Defense-in-depth: this runs in a shutdown-time finalizer that must
            # never raise, so tolerate any qualname that isn't a clean
            # "namespace::name" instead of unpacking blindly.
            continue
        ns, name_with_overload = splits
        name = name_with_overload.split(".")[0]
        if not hasattr(torch.ops, ns):
            continue
        namespace = getattr(torch.ops, ns)
        # Use vars() to check the instance dict directly, avoiding
        # __getattr__ which calls into C++ via _jit_get_operation.
        # During interpreter shutdown the C++ runtime may already be
        # torn down, causing UnicodeDecodeError or segfaults.
        if name not in vars(namespace):
            continue
        delattr(namespace, name)
        if name in namespace._dir:
            namespace._dir.remove(name)


def _del_library(
    captured_impls,
    op_impls,
    captured_defs,
    op_defs,
    registration_handles,
    m,
    schema_to_signature_cache,
):
    # [FUNCTION DOC] HELPER: Finalizer body for a Library. Drops signature-cache entries for its defined ops,
    # removes its kernel keys and definitions from the global sets, destroys registration handles, resets
    # the C++ handle if present, and clears cached torch.ops entries. Mutates the passed collections.
    # KNOWN ISSUE: op names are qualified (namespace::name) but cache keys are split on a dot only, so
    # the signature cache entries likely never match; an op name with two dots would raise ValueError.
    for op_def in op_defs:
        name = op_def
        overload_name = ""
        if "." in op_def:
            name, overload_name = op_def.split(".")
        if (
            name,
            overload_name,
        ) in schema_to_signature_cache:
            del schema_to_signature_cache[(name, overload_name)]

    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()

    if m is not None:
        m.reset()

    _clear_torch_ops_cache(op_defs)


@contextlib.contextmanager
def _scoped_library(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Context manager that creates a Library from the given arguments, yields it, and
    # always calls its _destroy on exit. Used for temporary registrations (mainly tests).
    # KNOWN ISSUE: if Library construction raises, the cleanup in the finally block fails on an unbound name.
    try:
        lib = Library(*args, **kwargs)
        yield lib
    finally:
        lib._destroy()


_keep_alive: list[Library] = []


NAMELESS_SCHEMA = re.compile(r"\(.*\) -> .*")


@functools.singledispatch
def define(qualname, schema, *, lib=None, tags=()):
    # [FUNCTION DOC] Define a new operator from a qualified name and a schema without the name (single
    # dispatch default for non-Library first arguments). Raises ValueError if qualname is not a string or
    # the schema does not look like arguments followed by an arrow and returns. Uses the given lib, or
    # creates a FRAGMENT library for the namespace and keeps it alive in a module-level list.
    # Applies the given tags. Returns None.
    r"""Defines a new operator.

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
    various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs, like :func:`torch.library.impl` or
    :func:`torch.library.register_fake`.

    Args:
        qualname (str): The qualified name for the operator. Should be
            a string that looks like "namespace::name", e.g. "aten::sin".
            Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        schema (str): The schema of the operator. E.g. "(Tensor x) -> Tensor"
            for an op that accepts one Tensor and returns one Tensor. It does
            not contain the operator name (that is passed in ``qualname``).
        lib (Optional[Library]): If provided, the lifetime of this operator
            will be tied to the lifetime of the Library object.
        tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
            operator. Tagging an operator changes the operator's behavior
            under various PyTorch subsystems; please read the docs for the
            torch.Tag carefully before applying it.

    Example::
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.sin(x)
        >>> assert torch.allclose(y, x.sin())

    """
    if not isinstance(qualname, str):
        raise ValueError(
            f"define(qualname, schema): expected qualname "
            f"to be instance of str, got {type(qualname)}"
        )
    namespace, name = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)
    if not NAMELESS_SCHEMA.fullmatch(schema):
        raise ValueError(
            f"define(qualname, schema, ...): expected schema "
            f'to look like e.g. "(Tensor x) -> Tensor" but '
            f'got "{schema}"'
        )
    lib.define(name + schema, alias_analysis="", tags=tags)


@define.register
def _(lib: Library, schema, alias_analysis=""):
    # [FUNCTION DOC] Legacy form of define, dispatched when the first argument is a Library. Returns a
    # decorator that defines the schema on the library and registers the decorated function as its
    # implementation, returning the function unchanged. Kept for backward compatibility.
    """The old torch.library.define.
    We're keeping this around for BC reasons
    """

    def wrap(f):
        # [FUNCTION DOC] HELPER: Decorator body: define the operator on the library, register f as its kernel, return f.
        name = lib.define(schema, alias_analysis)
        lib.impl(name, f)
        return f

    return wrap


@overload
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: None = None,
    *,
    lib: Library | None = None,
) -> Callable[[Callable[..., object]], None]: ...


@overload
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object],
    *,
    lib: Library | None = None,
) -> None: ...


# Deprecated BC API
@overload
def impl(
    lib: Library,
    name: str,
    dispatch_key: str = "",
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@functools.singledispatch
def impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[_P, _T] | None = None,
    *,
    lib: Library | None = None,
) -> object:
    # [FUNCTION DOC] Register an implementation for one or more device types (or dispatch keys) of an
    # operator named by qualname. Usable directly with func or as a decorator when func is None. The
    # type "default" registers for all devices. Optionally ties the registration to lib. Delegates to the
    # internal implementation without disabling dynamo tracing.
    """Register an implementation for a device type for this operator.

    You may pass "default" for ``types`` to register this implementation as the
    default implementation for ALL device types.
    Please only use this if the implementation truly supports all device types;
    for example, this is true if it is a composition of built-in PyTorch operators.

    This API may be used as a decorator. You can use nested decorators
    with this API provided they return a function and are placed inside
    this API (see Example 2).

    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".

    Args:
        qualname (str): Should be a string that looks like "namespace::operator_name".
        types (str | Sequence[str]): The device types to register an impl to.
        lib (Optional[Library]): If provided, the lifetime of this registration
            will be tied to the lifetime of the Library object.

    Examples:
        >>> # xdoctest: +SKIP("Requires Python <= 3.11")
        >>> import torch
        >>> import numpy as np
        >>> # Example 1: Register function.
        >>> # Define the operator
        >>> torch.library.define("mylib::mysin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the cpu device
        >>> @torch.library.impl("mylib::mysin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.mysin(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example 2: Register function with decorator.
        >>> def custom_decorator(func):
        >>>     def wrapper(*args, **kwargs):
        >>>         return func(*args, **kwargs) + 1
        >>>     return wrapper
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin_plus_one", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin_plus_one", "cpu")
        >>> @custom_decorator
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>>
        >>> y1 = torch.ops.mylib.sin_plus_one(x)
        >>> y2 = torch.sin(x) + 1
        >>> assert torch.allclose(y1, y2)
    """

    return _impl(qualname, types, func, lib=lib, disable_dynamo=False)


if not TYPE_CHECKING:

    @impl.register
    def _(
        lib: Library, name: str, dispatch_key: str = ""
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        # [FUNCTION DOC] Legacy form of impl dispatched when the first argument is a Library: returns a decorator
        # that registers the function on that library under the given name and dispatch key and returns it.
        """Legacy torch.library.impl API. Kept around for BC"""

        def wrap(f: Callable[_P, _T]) -> Callable[_P, _T]:
            # [FUNCTION DOC] HELPER: Decorator body: register f on the library and return it unchanged.
            lib.impl(name, f, dispatch_key)
            return f

        return wrap


@overload
def _impl(
    qualname: str,
    types: str | Sequence[str],
    func: None = None,
    *,
    lib: Library | None = None,
    disable_dynamo: bool = False,
) -> Callable[[Callable[..., object]], None]: ...


@overload
def _impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object],
    *,
    lib: Library | None = None,
    disable_dynamo: bool = False,
) -> None: ...


def _impl(
    qualname: str,
    types: str | Sequence[str],
    func: Callable[..., object] | None = None,
    *,
    lib: Library | None = None,
    disable_dynamo: bool = False,
) -> Callable[[Callable[..., object]], None] | None:
    # [FUNCTION DOC] HELPER: Shared implementation of impl and register_kernel. Normalizes types to a set of
    # dispatch keys (dispatch key names pass through, device types are mapped to keys). If func is given,
    # registers it immediately and returns None; otherwise returns a decorator doing the same. Registration
    # uses lib or a new kept-alive FRAGMENT library for the namespace. If disable_dynamo is true each
    # kernel is wrapped so dynamo does not trace into it.
    # See impl()
    if isinstance(types, str):
        types = (types,)
    keys = set({})
    for typ in types:
        is_dispatch_key = torch._C._parse_dispatch_key(typ)
        if is_dispatch_key:
            # We also support passing a DispatchKey to impl. Please prefer using
            # the higher-level torch.library APIs and only pass DispatchKey to
            # torch.library.impl with caution (or even better, don't use this
            # option and file an issue on GitHub for what you need).
            # We don't advertise this to users because
            # it is very easy to shoot yourself in the foot.
            keys.add(typ)
        else:
            keys.add(_device_type_to_key(typ))

    def register_(func: Callable[..., object]) -> None:
        # [FUNCTION DOC] HELPER: Register func for every computed key on the chosen library, optionally wrapped to disable dynamo.
        namespace, _ = torch._library.utils.parse_namespace(qualname)

        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        if disable_dynamo:

            @torch._disable_dynamo
            def func_no_dynamo(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped func with all arguments; marked so dynamo skips tracing it.
                return func(*args, **kwargs)

            for key in keys:
                use_lib.impl(qualname, func_no_dynamo, key)
        else:
            for key in keys:
                use_lib.impl(qualname, func, key)

    if func is None:
        return register_
    else:
        register_(func)
        return None


def _device_type_to_key(device_type: str) -> str:
    # [FUNCTION DOC] HELPER: Map a device type string to a dispatch key name. "default" maps to
    # CompositeExplicitAutograd (an approximation); others use the C++ device-to-key mapping.
    if device_type == "default":
        # This is technically not correct, because although all device_type
        # DispatchKeys are included in CompositeExplicitAutograd,
        # not everything in CompositeExplicitAutograd is associated with a
        # device_type. I don't really care that much about the difference.
        return "CompositeExplicitAutograd"
    return torch._C._dispatch_key_for_device(device_type)


@deprecated(
    "`torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that "
    "instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.",
    category=FutureWarning,
)
def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
    # [FUNCTION DOC] Deprecated alias of register_fake (emits a FutureWarning). Adjusts the stack level when
    # func is given so the fake impl records the right caller, then forwards all arguments.
    r"""This API was renamed to :func:`torch.library.register_fake` in PyTorch 2.4.
    Please use that instead.
    """
    if func is not None:
        _stacklevel = _stacklevel + 1
    return register_fake(qualname, func, lib=lib, _stacklevel=_stacklevel)


_op_identifier = Union[
    str, "torch._ops.OpOverload", "torch._library.custom_ops.CustomOpDef"
]


def _resolve_op_for_registration(
    op: _op_identifier, registration_name: str
) -> str | CustomOpDef:
    # [FUNCTION DOC] HELPER: Normalize an op argument for registration. Accepts a string, OpOverload or
    # CustomOpDef (ValueError otherwise); an OpOverload becomes its qualified name. Returns the CustomOpDef
    # if the name belongs to one, else the string.
    if not isinstance(op, (str, OpOverload, CustomOpDef)):
        raise ValueError(
            f"{registration_name}({op}): got unexpected type for op: {type(op)}"
        )
    if isinstance(op, OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef
    if not isinstance(op, str):
        raise AssertionError(f"op must be str at this point, got {type(op).__name__}")
    return op


def _library_for_registration(namespace: str, lib: Library | None) -> Library:
    # [FUNCTION DOC] HELPER: Return the given Library, or create a FRAGMENT library for the namespace and keep it alive.
    if lib is not None:
        return lib
    use_lib = Library(namespace, "FRAGMENT")
    _keep_alive.append(use_lib)
    return use_lib


def register_kernel(
    op: _op_identifier,
    device_types: device_types_t,
    func: Callable | None = None,
    /,
    *,
    lib: Library | None = None,
):
    # [FUNCTION DOC] Register an implementation of an op for given device types (None means all, mapped to
    # CompositeExplicitAutograd). Usable as a decorator when func is None. For a CustomOpDef, delegates
    # to its own register_kernel. Otherwise registers through the shared internal path with dynamo
    # disabled for the kernel. Returns the decorator, or None when func is supplied.
    # KNOWN ISSUE: the docstring text for lib is cut off mid-sentence.
    """Register an implementation for a device type for this operator.

    Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
    This API may be used as a decorator.

    Args:
        op (str | OpOverload): The operator to register an impl to.
        device_types (str | None | Sequence[str]): The device_types to register an impl to.
            If None, we will register to all device types -- please only use
            this option if your implementation is truly device-type-agnostic.
        func (Callable): The function to register as the implementation for
            the given device types.
        lib (Optional[Library]): If provided, the lifetime of this registration

    Examples::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> # Create a custom op that works on cpu
        >>> @custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> # Add implementations for the cuda device
        >>> @torch.library.register_kernel("mylib::numpy_sin", "cuda")
        >>> def _(x):
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x_cpu = torch.randn(3)
        >>> x_cuda = x_cpu.cuda()
        >>> assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
        >>> assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())

    """

    op = _resolve_op_for_registration(op, "register_kernel")
    if isinstance(op, CustomOpDef):
        return op.register_kernel(device_types, func)
    if device_types is None:
        device_types = "CompositeExplicitAutograd"

    return _impl(op, device_types, func, lib=lib, disable_dynamo=True)


def register_autocast(
    op: _op_identifier,
    device_type: str,
    cast_inputs: _dtype,
    /,
    *,
    lib: Library | None = None,
):
    # [FUNCTION DOC] Register an autocast rule for a custom op on a device type: under autocast, floating
    # point tensor inputs are cast to cast_inputs and the op runs with autocast disabled. Raises
    # ValueError if the device type does not support autocast. For a CustomOpDef, delegates. Otherwise
    # overrides any existing Python kernel for the autocast key and registers a keyset-aware kernel on the
    # library. Kwargs are not supported (AssertionError).
    r"""Register an autocast dispatch rule for this custom op.

    Valid ``device_type`` values include any device type that supports autocast.
    See :func:`torch.amp.is_autocast_available` for details.

    Args:
        op (str | OpOverload): The operator to register an autocast dispatch rule to.
        device_type(str):  Device type to use. 'cuda', 'cpu', 'xpu', or any other device type that supports autocast.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`torch.dtype`): When custom op runs in an autocast-enabled region,
            casts incoming floating-point Tensors to the target dtype (non-floating-point Tensors
            are not affected), then executes custom op with autocast disabled.
        lib (Optional[Library]): If provided, the lifetime of this registration

    Examples::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>>
        >>> # Create a custom op that works on cuda
        >>> @torch.library.custom_op("mylib::my_sin", mutates_args=())
        >>> def my_sin(x: Tensor) -> Tensor:
        >>>     return torch.sin(x)
        >>>
        >>> # Register autocast dispatch rule for the cuda device
        >>> torch.library.register_autocast("mylib::my_sin", "cuda", torch.float16)
        >>>
        >>> x = torch.randn(3, dtype=torch.float32, device="cuda")
        >>> with torch.autocast("cuda", dtype=torch.float16):
        >>>     y = torch.ops.mylib.my_sin(x)
        >>> assert y.dtype == torch.float16

    """
    op = _resolve_op_for_registration(op, "register_autocast")
    if not torch._C._is_autocast_available(device_type):
        raise ValueError(f"Device type '{device_type}' does not support autocast.")

    if isinstance(op, CustomOpDef):
        return op.register_autocast(device_type, cast_inputs)
    qualname = op
    _op = torch._library.utils.lookup_op(qualname)

    namespace, opname = torch._library.utils.parse_namespace(qualname)
    use_lib = _library_for_registration(namespace, lib)

    autocast_key = "Autocast" + torch._C._dispatch_key_for_device(device_type)
    autocast_dispatch_key = getattr(torch._C.DispatchKey, autocast_key)

    def _maybe_override_py_impl(op: torch._ops.OpOverload, dispatch_key):
        # [FUNCTION DOC] HELPER: Return a decorator that registers a Python kernel for the op and dispatch key,
        # first removing any existing Python kernel for that key.
        def inner(kernel):
            # [FUNCTION DOC] HELPER: Remove an existing Python kernel for the key, then register kernel via py_impl.
            if op.has_kernel_for_dispatch_key(dispatch_key):
                op.py_kernels.pop(dispatch_key)
            return op.py_impl(dispatch_key)(kernel)

        return inner

    @_maybe_override_py_impl(_op, autocast_dispatch_key)
    def _autocast_py_impl(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Python autocast kernel: rejects kwargs (AssertionError), excludes the autocast key
        # from dispatch, casts the positional args and calls the op.
        if len(kwargs) != 0:
            raise AssertionError("Custom ops do not support kwargs yet.")
        with torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(autocast_dispatch_key)
        ):
            return _op(*_cast(args, device_type, cast_inputs))

    def kernel(_, *args, **kwargs):
        # [FUNCTION DOC] HELPER: C++-registered autocast kernel; ignores the keyset argument, rejects kwargs
        # (AssertionError), and forwards positional args to the Python autocast implementation.
        if len(kwargs) != 0:
            raise AssertionError("Custom ops do not support kwargs yet.")
        return _autocast_py_impl(*args, **kwargs)

    return use_lib.impl(opname, kernel, autocast_key, with_keyset=True)


def register_fake(
    op: _op_identifier,
    func: Callable | None = None,
    /,
    *,
    lib: Library | None = None,
    _stacklevel: int = 1,
    allow_override: bool = True,
):
    # [FUNCTION DOC] Register a FakeTensor (meta) implementation for an operator. Usable as a decorator when
    # func is None. For a CustomOpDef, delegates to its register_fake. Otherwise registers via a library
    # (given or a new fragment), recording the caller's stack level and allow_override, and returns func.
    # KNOWN ISSUE: the docstring lists the first parameter as op_name but the parameter is op.
    r"""Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    An "FakeTensor implementation" specifies the behavior of this operator on
    Tensors that carry no data ("FakeTensor"). Given some input Tensors with
    certain properties (sizes/strides/storage_offset/device), it specifies
    what the properties of the output Tensors are.

    The FakeTensor implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write a FakeTensor
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The FakeTensor implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        func: Fake tensor implementation.
        lib (Optional[Library]): Library to register the fake tensor to.
        allow_override: Flag controlling if we want to override an
                        existing registered fake impl. This is on by default;
                        pass ``False`` to error if the operator already has a
                        fake impl. This also only applies if the custom operator
                        was not created via torch.library.custom_op, as
                        overriding an existing fake impl is already allowed.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_linear", mutates_args=())
        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        >>>     raise NotImplementedError("Implementation goes here")
        >>>
        >>> @torch.library.register_fake("mylib::custom_linear")
        >>> def _(x, weight, bias):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>     assert x.device == weight.device
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> with torch._subclasses.fake_tensor.FakeTensorMode():
        >>>     x = torch.randn(2, 3)
        >>>     w = torch.randn(3, 3)
        >>>     b = torch.randn(3)
        >>>     y = torch.ops.mylib.custom_linear(x, w, b)
        >>>
        >>> assert y.shape == (2, 3)
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_nonzero", mutates_args=())
        >>> def custom_nonzero(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     return torch.tensor(res, device=x.device)
        >>>
        >>> @torch.library.register_fake("mylib::custom_nonzero")
        >>> def _(x):
        >>> # Number of nonzero-elements is data-dependent.
        >>> # Since we cannot peek at the data in an fake impl,
        >>> # we use the ctx object to construct a new symint that
        >>> # represents the data-dependent size.
        >>>     ctx = torch.library.get_ctx()
        >>>     nnz = ctx.new_dynamic_size()
        >>>     shape = [nnz, x.dim()]
        >>>     result = x.new_empty(shape, dtype=torch.int64)
        >>>     return result
        >>>
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>>
        >>> x = torch.tensor([0, 1, 2, 3, 4, 0])
        >>> trace = make_fx(torch.ops.mylib.custom_nonzero, tracing_mode="symbolic")(x)
        >>> trace.print_readable()
        >>>
        >>> assert torch.allclose(trace(x), torch.ops.mylib.custom_nonzero(x))

    """
    op = _resolve_op_for_registration(op, "register_fake")
    if isinstance(op, CustomOpDef):
        if func is None:
            return op.register_fake
        else:
            return op.register_fake(func)

    stacklevel = _stacklevel

    def register(func):
        # [FUNCTION DOC] HELPER: Register func as the fake impl through the library and return func.
        namespace, op_name = torch._library.utils.parse_namespace(op)
        use_lib = _library_for_registration(namespace, lib)
        use_lib._register_fake(
            op_name, func, _stacklevel=stacklevel + 1, allow_override=allow_override
        )
        return func

    if func is None:
        return register
    else:
        stacklevel += 1
        return register(func)


def _register_effectful_op(
    op: _op_identifier,
    effect: EffectType | None,
    *,
    lib: Library | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Declare an effect type (or None for no effect) for an operator so compile passes
    # do not reorder it. Raises ValueError for an unsupported op type. Registers on the CustomOpDef if one
    # exists, and always also registers through a library (given or new fragment).
    # KNOWN ISSUE: the docstring names the parameter op_name but it is op, and lib is undocumented.
    r"""
    To specify that an operator has side-effects, we must register an effect
    type for the operator. This will prevent graph passes in torch.compile from
    reordering operations with the same effect type.

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        effect: Effect type to register. None means the operator is not effectful.
    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(
            f"register_effectful_op({op}): got unexpected type for op: {type(op)}"
        )

    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        opdef.register_effect(effect)
    if not isinstance(op, str):
        raise AssertionError(f"op must be str at this point, got {type(op).__name__}")

    namespace, _ = torch._library.utils.parse_namespace(op)
    if lib is None:
        use_lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(use_lib)
    else:
        use_lib = lib
    use_lib._register_effectful_op(op, effect)


def register_autograd(
    op: _op_identifier,
    backward: Callable,
    /,
    *,
    setup_context: Callable | None = None,
    lib=None,
) -> None:
    # [FUNCTION DOC] Register a backward formula (and optional setup_context) for an op. For a CustomOpDef,
    # delegates. Otherwise looks up the op and raises RuntimeError for out-tagged or non-functional ops and
    # NotImplementedError for ops with keyword-only Tensor args. Builds the autograd kernel and registers it
    # for the Autograd key on a library, passing the dispatch keyset. Returns None.
    r"""Register a backward formula for this custom op.

    In order for an operator to work with autograd, you need to register
    a backward formula:
    1. You must tell us how to compute gradients during the backward pass
    by providing us a "backward" function.
    2. If you need any values from the forward to compute gradients, you can
    use `setup_context` to save values for backward.

    ``backward`` runs during the backward pass. It accepts ``(ctx, *grads)``:
    - ``grads`` is one or more gradients. The number of gradients matches
    the number of outputs of the operator.
    The ``ctx`` object is `the same ctx object <context_method_mixins>`_ used by
    :class:`torch.autograd.Function`. The semantics of ``backward_fn`` are the
    same as :meth:`torch.autograd.Function.backward`.

    ``setup_context(ctx, inputs, output)`` runs during the forward pass.
    Please save quantities needed for backward onto the ``ctx`` object via
    either :meth:`torch.autograd.function.FunctionCtx.save_for_backward`
    or assigning them as attributes of ``ctx``. If your custom op has
    kwarg-only arguments, we expect the signature of ``setup_context``
    to be ``setup_context(ctx, inputs, keyword_only_inputs, output)``.

    Both ``setup_context_fn`` and ``backward_fn`` must be traceable. That is,
    they may not directly access :meth:`torch.Tensor.data_ptr` and they must
    not depend on or mutate global state. If you need a non-traceable backward,
    you can make it a separate custom_op that you call inside ``backward_fn``.

    If you need different autograd behavior on different devices, then we
    recommend creating two different custom operators, one for each device
    that needs different behavior, and switching between them at runtime.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> def setup_context(ctx, inputs, output) -> Tensor:
        >>>     x, = inputs
        >>>     ctx.save_for_backward(x)
        >>>
        >>> def backward(ctx, grad):
        >>>     x, = ctx.saved_tensors
        >>>     return grad * x.cos()
        >>>
        >>> torch.library.register_autograd(
        ...     "mylib::numpy_sin", backward, setup_context=setup_context
        ... )
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_sin(x)
        >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
        >>> assert torch.allclose(grad_x, x.cos())
        >>>
        >>> # Example with a keyword-only arg
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, *, val: float) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = x_np * val
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> def setup_context(ctx, inputs, keyword_only_inputs, output) -> Tensor:
        >>>     ctx.val = keyword_only_inputs["val"]
        >>>
        >>> def backward(ctx, grad):
        >>>     return grad * ctx.val
        >>>
        >>> torch.library.register_autograd(
        ...     "mylib::numpy_mul", backward, setup_context=setup_context
        ... )
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_mul(x, val=3.14)
        >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
        >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

    """
    op = _resolve_op_for_registration(op, "register_autograd")
    if isinstance(op, CustomOpDef):
        op.register_autograd(backward, setup_context=setup_context)
        return

    qualname = op
    op = torch._library.utils.lookup_op(qualname)
    schema = op._schema
    if _library.utils.is_out(op):
        raise RuntimeError(
            f"Cannot register autograd formula for operator tagged with "
            f"torch.Tag.out: {op}. Out variants do not support autograd."
        )
    if not _library.utils.is_functional_schema(schema):
        raise RuntimeError(
            f"Cannot register autograd formula for non-functional operator "
            f"{op} with schema {schema}. Please create "
            f"a functional operator and register an autograd formula for that."
        )
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_autograd with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    info = _library.autograd.Info(backward, setup_context)
    autograd_kernel = _library.autograd.make_autograd_impl(op, info)
    namespace, opname = torch._library.utils.parse_namespace(qualname)
    use_lib = _library_for_registration(namespace, lib)
    use_lib.impl(opname, autograd_kernel, "Autograd", with_keyset=True)


def register_torch_dispatch(
    op: _op_identifier,
    torch_dispatch_class: Any,
    func: Callable | None = None,
    /,
    *,
    lib: Library | None = None,
):
    # [FUNCTION DOC] Register a torch_dispatch rule for an op and a Tensor subclass or TorchDispatchMode.
    # Usable as a decorator when func is None. For a CustomOpDef, delegates. Otherwise registers the rule
    # via a library and returns func (or the decorator).
    r"""Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

    This allows for open registration to specify the behavior between the operator
    and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
    or the operator directly.

    The ``torch_dispatch_class`` is either a Tensor subclass with ``__torch_dispatch__`` or a
    TorchDispatchMode.

    If it is a Tensor subclass, we expect ``func`` to have the following signature:
    ``(cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    If it is a TorchDispatchMode, we expect ``func`` to have the following signature:
    ``(mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    ``args`` and ``kwargs`` will have been normalized the same way they are
    in ``__torch_dispatch__`` (see :ref:`torch-dispatch-calling-convention`).

    Examples:

        >>> import torch
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args={})
        >>> def foo(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.clone()
        >>>
        >>> class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
        >>>     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        >>>         return func(*args, **kwargs)
        >>>
        >>> @torch.library.register_torch_dispatch("mylib::foo", MyMode)
        >>> def _(mode, func, types, args, kwargs):
        >>>     x, = args
        >>>     return x + 1
        >>>
        >>> x = torch.randn(3)
        >>> y = foo(x)
        >>> assert torch.allclose(y, x)
        >>>
        >>> with MyMode():
        >>>     y = foo(x)
        >>> assert torch.allclose(y, x + 1)

    """
    op = _resolve_op_for_registration(op, "register_torch_dispatch")
    if isinstance(op, CustomOpDef):
        return op.register_torch_dispatch(torch_dispatch_class, func)

    def register(func):
        # [FUNCTION DOC] HELPER: Register func as the torch_dispatch rule and return func.
        namespace, op_name = torch._library.utils.parse_namespace(op)
        use_lib = _library_for_registration(namespace, lib)
        use_lib._register_torch_dispatch_rule(op_name, torch_dispatch_class, func)
        return func

    if func is None:
        return register
    else:
        return register(func)


def register_vmap(
    op: _op_identifier,
    func: Callable | None = None,
    /,
    *,
    lib=None,
):
    # [FUNCTION DOC] Register a vmap rule for a custom op so torch.vmap can batch it. For a CustomOpDef,
    # delegates. Otherwise raises NotImplementedError if the schema has keyword-only Tensor args, and
    # registers a wrapper for the FuncTorchBatched key that calls the vmap helper with the functorch
    # interpreter. Usable as a decorator when func is None.
    # KNOWN ISSUE: the inner register function returns None, so decorator use rebinds the decorated name to None.
    r"""Register a vmap implementation to support :func:`torch.vmap` for this custom op.

    This API may be used as a decorator (see examples).

    In order for an operator to work with :func:`torch.vmap`, you may need to register a
    vmap implementation in the following signature:

        ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

    where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.
    We do not support kwarg-only Tensor args.

    It specifies how do we compute the batched version of ``op`` given inputs with an additional
    dimension (specified by ``in_dims``).

    For each arg in ``args``, ``in_dims`` has a corresponding ``Optional[int]``. It is ``None``
    if the arg is not a Tensor or if the arg is not being vmapped over, otherwise, it is an integer
    specifying what dimension of the Tensor is being vmapped over.

    ``info`` is a collection of additional metadata that may be helpful:
    ``info.batch_size`` specifies the size of the dimension being vmapped over, while
    ``info.randomness`` is the ``randomness`` option that was passed to :func:`torch.vmap`.

    The return of the function ``func`` is a tuple of ``(output, out_dims)``. Similar to ``in_dims``,
    ``out_dims`` should be of the same structure as ``output`` and contain one ``out_dim``
    per output that specifies if the output has the vmapped dimension and what index it is in.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>> from typing import Tuple
        >>>
        >>> def to_numpy(tensor):
        >>>     return tensor.cpu().numpy()
        >>>
        >>> lib = torch.library.Library("mylib", "FRAGMENT")
        >>> @torch.library.custom_op("mylib::numpy_cube", mutates_args=())
        >>> def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
        >>>     x_np = to_numpy(x)
        >>>     dx = torch.tensor(3 * x_np ** 2, device=x.device)
        >>>     return torch.tensor(x_np ** 3, device=x.device), dx
        >>>
        >>> def numpy_cube_vmap(info, in_dims, x):
        >>>     result = numpy_cube(x)
        >>>     return result, (in_dims[0], in_dims[0])
        >>>
        >>> torch.library.register_vmap(numpy_cube, numpy_cube_vmap)
        >>>
        >>> x = torch.randn(3)
        >>> torch.vmap(numpy_cube)(x)
        >>>
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
        >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
        >>>
        >>> @torch.library.register_vmap("mylib::numpy_mul")
        >>> def numpy_mul_vmap(info, in_dims, x, y):
        >>>     x_bdim, y_bdim = in_dims
        >>>     x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
        >>>     y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
        >>>     result = x * y
        >>>     result = result.movedim(-1, 0)
        >>>     return result, 0
        >>>
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.randn(3)
        >>> torch.vmap(numpy_mul)(x, y)

    .. note::
        The vmap function should aim to preserve the semantics of the entire custom operator.
        That is, ``grad(vmap(op))`` should be replaceable with a ``grad(map(op))``.

        If your custom operator has any custom behavior in the backward pass, please
        keep this in mind.

    """
    op = _resolve_op_for_registration(op, "register_vmap")
    if isinstance(op, CustomOpDef):
        return op.register_vmap(func)
    qualname = op
    op_overload = torch._library.utils.lookup_op(qualname)
    schema = op_overload._schema
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_vmap with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    def register(func):
        # [FUNCTION DOC] HELPER: Create or reuse the library, then register a keyset-aware wrapper that runs func
        # under the current functorch interpreter. Returns None.
        nonlocal lib

        namespace, opname = torch._library.utils.parse_namespace(qualname)
        use_lib = _library_for_registration(namespace, lib)
        if lib is None:
            lib = use_lib

        from torch._functorch.autograd_function import custom_function_call_vmap_helper
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        def wrapped_func(keyset, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Kernel for the batched key: fetch the current functorch interpreter and call the vmap helper.
            interpreter = retrieve_current_functorch_interpreter()
            return custom_function_call_vmap_helper(
                # pyrefly: ignore[bad-argument-type]
                interpreter,
                func,
                op_overload,
                *args,
                **kwargs,
            )

        use_lib.impl(opname, wrapped_func, "FuncTorchBatched", with_keyset=True)

    if func is None:
        return register
    else:
        return register(func)


# If the op was defined in C++, then we want to make sure there was an
# m.set_python_module(module, ...) call and that the module is the
# same as the module that called torch.library.register_fake.
def _check_pystubs_once(func, qualname, actual_module_name):
    # [FUNCTION DOC] HELPER: Wrap a fake impl so that on its first call it verifies that an op defined in C++
    # has a matching set_python_module declaration naming the module that registered the fake impl.
    # Raises RuntimeError if the declaration is missing (when required) or names a different module.
    # Ops defined in Python skip the check. Later calls forward directly.
    checked = False

    def inner(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Run the one-time module check (see the enclosing function), then call func with the given arguments.
        nonlocal checked
        if checked:
            return func(*args, **kwargs)

        op = torch._library.utils.lookup_op(qualname)
        if op._defined_in_python:
            checked = True
            return func(*args, **kwargs)

        maybe_pystub = torch._C._dispatch_pystub(
            op._schema.name, op._schema.overload_name
        )
        if maybe_pystub is None:
            if torch._library.utils.requires_set_python_module():
                namespace = op.namespace
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' was defined in C++ and has a Python "
                    f"fake impl. In this situation, we require there to also be a "
                    f'companion C++ `m.set_python_module("{actual_module_name}")` '
                    f"call, but we could not find one. Please add that to "
                    f"to the top of the C++ TORCH_LIBRARY({namespace}, ...) block the "
                    f"operator was registered in ({cpp_filename})"
                )
        else:
            pystub_module = maybe_pystub[0]
            if actual_module_name != pystub_module:
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' specified that its python fake impl "
                    f"is in the Python module '{pystub_module}' but it was actually found "
                    f"in '{actual_module_name}'. Please either move the fake impl "
                    f"or correct the m.set_python_module call ({cpp_filename})"
                )
        checked = True
        return func(*args, **kwargs)

    return inner


# NOTE [ctx inside the fake implementation]
# If a user has an operator with data-dependent output shape, then when writing
# a fake implementation they must query the current ctx and use methods on the
# ctx to construct a new unbacked symint.
#
# This is done via us setting the global_ctx_getter function every time a fake
# implementation is invoked.
def get_ctx() -> "torch._library.fake_impl.FakeImplCtx":
    # [FUNCTION DOC] Return the current fake-impl context object from the global getter. Only valid while a
    # fake impl is running; used to create unbacked sizes for data-dependent shapes.
    """get_ctx() returns the current AbstractImplCtx object.

    Calling ``get_ctx()`` is only valid inside of an fake impl
    (see :func:`torch.library.register_fake` for more usage details.
    """
    return torch._library.fake_impl.global_ctx_getter()


def get_kernel(
    op: _op_identifier, dispatch_key: str | torch.DispatchKey
) -> torch._C._SafeKernelFunction:
    # [FUNCTION DOC] Return the computed kernel (a boxed callable wrapper) for an op and dispatch key.
    # Raises ValueError for an unsupported op type or an unknown dispatch key name; strings are looked up
    # among DispatchKey members and an OpOverload is converted to its name.
    # KNOWN ISSUE: the docstring says a CustomOpDef is accepted, but the code rejects it with ValueError.
    """Returns the computed kernel for a given operator and dispatch key.

    This function retrieves the kernel that would be executed for a given
    operator and dispatch key combination. The returned SafeKernelFunction
    can be used to call the kernel in a boxed fashion. The intended use
    case for this function is to retrieve the original kernel for a given
    dispatch key and then register another kernel to the same dispatch key
    that calls into the original kernel for certain cases.

    Args:
        op: Operator name (along with the overload) or OpOverload object
            Can be a string (e.g., "aten::add.Tensor"), an OpOverload, or a CustomOpDef.
        dispatch_key (str | torch.DispatchKey): The dispatch key to get the kernel for.
            Can be a string (e.g., "CPU", "CUDA") or a DispatchKey enum value.

    Returns:
        torch._C._SafeKernelFunction: A safe kernel function that can be used to
            call the kernel.

    Raises:
        RuntimeError: If the operator does not exist.

    Example:
        >>> # Get the CPU kernel for torch.add
        >>> kernel = torch.library.get_kernel("aten::add.Tensor", "CPU")
        >>>
        >>> # You can also use DispatchKey enum
        >>> kernel = torch.library.get_kernel("aten::add.Tensor", torch.DispatchKey.CPU)
        >>>
        >>> # Or use an OpOverload directly
        >>> kernel = torch.library.get_kernel(torch.ops.aten.add.Tensor, "CPU")
        >>>
        >>> # Example: Using get_kernel in a custom op with conditional dispatch
        >>> # Get the original kernel for torch.sin
        >>> original_sin_kernel = torch.library.get_kernel("aten::sin", "CPU")
        >>>
        >>> # If input has negative values, use original sin, otherwise return zeros
        >>> def conditional_sin_impl(dispatch_keys, x):
        >>>     if (x < 0).any():
        >>>         return original_sin_kernel.call_boxed(dispatch_keys, x)
        >>>     else:
        >>>         return torch.zeros_like(x)
        >>>
        >>> lib = torch.library.Library("aten", "IMPL")
        >>> # with_keyset=True so the first argument to the impl is the current DispatchKeySet
        >>> which needs to be the first argument to ``kernel.call_boxed``
        >>> lib.impl("sin", conditional_sin_impl, "CPU", with_keyset=True)
        >>>
        >>> # Test the conditional behavior
        >>> x_positive = torch.tensor([1.0, 2.0])
        >>> x_mixed = torch.tensor([-1.0, 2.0])
        >>> torch.sin(x_positive)
        tensor([0., 0.])
        >>> torch.sin(x_mixed)
        tensor([-0.8415, 0.9093])
    """
    if not isinstance(op, (str, torch._ops.OpOverload)):
        raise ValueError(f"get_kernel({op}): got unexpected type for op: {type(op)}")

    if isinstance(op, torch._ops.OpOverload):
        op = op._name

    if isinstance(dispatch_key, str):
        try:
            dispatch_key = torch._C.DispatchKey.__members__[dispatch_key]
        except KeyError:
            raise ValueError(f"Invalid dispatch key: {dispatch_key}") from None

    return torch._C._dispatch_get_computed_kernel_for_dispatch_key(op, dispatch_key)


_OPCHECK_DEFAULT_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def opcheck(
    op: torch._ops.OpOverload | torch._ops.OpOverloadPacket | CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    test_utils: str | Sequence[str] = _OPCHECK_DEFAULT_UTILS,
    raise_exception: bool = True,
    atol=None,
    rtol=None,
) -> dict[str, str]:
    # [FUNCTION DOC] Test that an operator is registered correctly by delegating to the testing helper in
    # torch.testing._internal.optests (imported lazily). Runs the chosen test utilities (schema, autograd
    # registration, faketensor, aot dispatch dynamic) on the given args and kwargs with optional tolerances.
    # Returns a mapping from test name to result; raises on the first failure if raise_exception is set.
    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    That is, when you use the torch.library/TORCH_LIBRARY APIs to create a
    custom op, you specified metadata (e.g. mutability info) about the custom op
    and these APIs require that the functions you pass them satisfy certain
    properties (e.g. no data pointer access in the fake/meta/abstract kernel)
    ``opcheck`` tests these metadata and properties.

    Concretely, we test the following:

    - test_schema: If the schema matches the implementation of
      the operator. For example: if the schema specifies a Tensor is mutated,
      then we check the implementation mutates the Tensor. If the schema
      specifies that we return a new Tensor, then we check that the
      implementation returns a new Tensor (instead of an existing one or
      a view of an existing one).
    - test_autograd_registration: If the operator supports training
      (autograd): we check that its autograd formula is registered via
      torch.library.register_autograd or a manual registration to one
      or more DispatchKey::Autograd keys. Any other DispatchKey-based
      registrations may lead to undefined behavior.
    - test_faketensor: If the operator has a FakeTensor kernel
      (and if it is correct). The FakeTensor kernel is necessary (
      but not sufficient) for the operator to work with PyTorch compilation
      APIs (torch.compile/export/FX). We check that a FakeTensor kernel
      (also sometimes known as a meta kernel) was registered for the
      operator and that it is correct. This test takes the result of
      running the operator on real tensors and the result of running
      the operator on FakeTensors and checks that they have the same
      Tensor metadata (sizes/strides/dtype/device/etc).
    - test_aot_dispatch_dynamic: If the operator has correct behavior
      with PyTorch compilation APIs (torch.compile/export/FX).
      This checks that the outputs (and gradients, if applicable) are the
      same under eager-mode PyTorch and torch.compile.
      This test is a superset of ``test_faketensor`` and is an e2e test;
      other things it tests are that the operator supports
      functionalization and that the backward pass (if it exists) also
      supports FakeTensor and functionalization.

    For best results, please call ``opcheck`` multiple times with a
    representative set of inputs. If your operator supports
    autograd, please use ``opcheck`` with inputs with ``requires_grad = True``;
    if your operator supports multiple devices (e.g. CPU and CUDA), please
    use ``opcheck`` with inputs on all supported devices.

    Args:
        op: The operator. Must either be a function decorated with
            :func:`torch.library.custom_op` or an OpOverload/OpOverloadPacket
            found in torch.ops.* (e.g. torch.ops.aten.sin, torch.ops.mylib.foo)
        args: The args to the operator
        kwargs: The kwargs to the operator
        test_utils: Tests that we should run. Default: all of them.
            Example: ("test_schema", "test_faketensor")
        raise_exception: If we should raise an exception on the first
            error. If False, we will return a dict with information
            on if each test passed or not.
        rtol (Optional[float]): Relative tolerance for floating point comparisons.
            If specified ``atol`` must also be specified.
            If omitted, default values based on the ``dtype`` are selected
            (see the table in :func:`torch.testing.assert_close`).
        atol (Optional[float]): Absolute tolerance for floating point comparisons.
            If specified ``rtol`` must also be specified.
            If omitted, default values based on the ``dtype`` are selected
            (see the table in :func:`torch.testing.assert_close`).

    .. warning::

        opcheck and :func:`torch.autograd.gradcheck` test different things;
        opcheck tests if your usage of torch.library APIs is correct while
        :func:`torch.autograd.gradcheck` tests if your autograd formula is
        mathematically correct. Use both to test custom ops that support
        gradient computation.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: float) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     z_np = x_np * y
        >>>     return torch.from_numpy(z_np).to(x.device)
        >>>
        >>> @numpy_mul.register_fake
        >>> def _(x, y):
        >>>     return torch.empty_like(x)
        >>>
        >>> def setup_context(ctx, inputs, output):
        >>>     y, = inputs
        >>>     ctx.y = y
        >>>
        >>> def backward(ctx, grad):
        >>>     return grad * ctx.y, None
        >>>
        >>> numpy_mul.register_autograd(backward, setup_context=setup_context)
        >>>
        >>> sample_inputs = [
        >>>     (torch.randn(3), 3.14),
        >>>     (torch.randn(2, 3, device='cuda'), 2.718),
        >>>     (torch.randn(1, 10, requires_grad=True), 1.234),
        >>>     (torch.randn(64, 64, device='cuda', requires_grad=True), 90.18),
        >>> ]
        >>>
        >>> for args in sample_inputs:
        >>>     torch.library.opcheck(numpy_mul, args)

    """
    import torch.testing._internal.optests as optests

    return optests.opcheck(
        op,
        args,
        kwargs,
        test_utils=test_utils,
        raise_exception=raise_exception,
        rtol=rtol,
        atol=atol,
    )
