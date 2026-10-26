# [MODULE DOC] Export tracing wrappers. Defines the _export_tracepoint higher-order op (an identity op
# that marks module call inputs/outputs in traced graphs), helpers to hook submodules with it
# and record their call specs, a strict-mode marker, and experimental decorators that make
# tensor-subclass constructors and custom autograd Function.apply traceable in the
# pre-dispatch export graph via flat_apply. Import-time side effects: instantiates the
# _export_tracepoint op and registers its proxy, fake, functionalize, Autograd and CPU impls.
import inspect
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export
from torch._higher_order_ops.flat_apply import (
    _ConstantFunction,
    flat_apply,
    to_graphable,
)
from torch._higher_order_ops.strict_mode import strict_mode
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    PreDispatchTorchFunctionMode,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
    from torch.fx.experimental.proxy_tensor import _ProxyTracer
    from torch.fx.proxy import Proxy
    from torch.utils.hooks import RemovableHandle


_P = ParamSpec("_P")
_R = TypeVar("_R")


class ExportTracepoint(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator named _export_tracepoint. It is an identity on its tensor
    # arguments; keyword arguments (kind, path) are metadata that survive in the exported graph.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register the operator under the name _export_tracepoint.
        super().__init__("_export_tracepoint")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Invoke the higher-order op dispatch with the given arguments and return its result.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(*args, **kwargs)


_export_tracepoint = ExportTracepoint()


@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(
    mode: ProxyTorchDispatchMode, *args: Any, **kwargs: Any
) -> Any:
    # [FUNCTION DOC] Proxy-mode implementation: unwraps proxies from the arguments, records a call to
    # the tracepoint op in the traced graph, and returns the original args tracked to that
    # proxy's outputs.
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
    proxy = mode.tracer.create_proxy(
        "call_function", _export_tracepoint, p_args, p_kwargs
    )
    return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)


@register_fake(_export_tracepoint, skip_cache=True)
def export_tracepoint_fake_tensor_mode(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    # [FUNCTION DOC] Fake-tensor implementation: returns the positional args unchanged as a tuple.
    return args


@_export_tracepoint.py_functionalize_impl
def export_tracepoint_functional(
    ctx: "BaseFunctionalizeAPI", *args: Any, **kwargs: Any
) -> tuple[Any, ...]:
    # [FUNCTION DOC] Functionalization implementation: unwraps functional tensors, redispatches the
    # tracepoint op to the next layer for its graph effect, and returns the original args.
    unwrapped_args = ctx.unwrap_tensors(args)
    # pyrefly: ignore[bad-argument-type]  # TODO unwrap_tensors accepts pytrees at runtime
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

    with ctx.redispatch_to_next():
        _export_tracepoint(*unwrapped_args, **unwrapped_kwargs)
        return args


_export_tracepoint.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_export_tracepoint, deferred_error=True)
)


@_export_tracepoint.py_impl(DispatchKey.CPU)
def export_tracepoint_cpu(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    # [FUNCTION DOC] Eager CPU implementation: returns the positional args unchanged as a tuple.
    return args


def _wrap_submodule(
    mod: torch.nn.Module,
    path: str,
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]],
) -> tuple["RemovableHandle", "RemovableHandle"]:
    # [FUNCTION DOC] HELPER: Look up the submodule at a dotted path under mod and register forward pre and
    # post hooks that route flattened inputs and outputs through the tracepoint op (kinds
    # module_call_inputs / module_call_outputs). The post hook records the input and output
    # tree specs for the path into module_call_specs. Returns the two removable handles.
    # Raises AssertionError if mod is not an nn.Module or path is empty.
    if not isinstance(mod, torch.nn.Module):
        raise AssertionError(f"expected torch.nn.Module, got {type(mod)}")
    if path == "":
        raise AssertionError("path must not be empty")
    submodule = torch.fx.graph_module._get_attr(mod, path)

    def update_module_call_signatures(
        path: str, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the in/out specs for path in module_call_specs; if the path was
        # already recorded, raises AssertionError when either spec differs from the stored one.
        if path in module_call_specs:
            if module_call_specs[path]["in_spec"] != in_spec:
                raise AssertionError(
                    f"in_spec mismatch for {path}: {module_call_specs[path]['in_spec']} != {in_spec}"
                )
            if module_call_specs[path]["out_spec"] != out_spec:
                raise AssertionError(
                    f"out_spec mismatch for {path}: {module_call_specs[path]['out_spec']} != {out_spec}"
                )
        module_call_specs[path] = {"in_spec": in_spec, "out_spec": out_spec}

    def check_flattened(flat_args: list[Any]) -> None:
        # [FUNCTION DOC] HELPER: Raises AssertionError unless every flattened leaf is a tensor, string,
        # int, float, bool or None.
        for a in flat_args:
            if not (isinstance(a, (torch.Tensor, str, int, float, bool)) or a is None):
                raise AssertionError(
                    f"Only Tensors or scalars are supported as pytree flattened inputs, got: {a}"
                )

    def pre_hook(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Forward pre-hook: flatten args and kwargs, validate leaves, pass them
        # through the tracepoint op with kind module_call_inputs, and return the rebuilt
        # (args, kwargs).
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        check_flattened(flat_args)
        flat_args = _export_tracepoint(*flat_args, kind="module_call_inputs", path=path)
        args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        return args, kwargs

    def post_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        res: Any,
    ) -> Any:
        # [FUNCTION DOC] HELPER: Forward post-hook: flatten the result, validate leaves, pass them through
        # the tracepoint op with kind module_call_outputs, record the call signature for the
        # path, and return the rebuilt result structure.
        _, in_spec = pytree.tree_flatten((args, kwargs))
        flat_res, out_spec = pytree.tree_flatten(res)
        check_flattened(flat_res)
        flat_res = _export_tracepoint(*flat_res, kind="module_call_outputs", path=path)
        update_module_call_signatures(path, in_spec, out_spec)
        return pytree.tree_unflatten(flat_res, out_spec)

    pre_handle = submodule.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = submodule.register_forward_hook(post_hook, with_kwargs=True)
    return pre_handle, post_handle


@contextmanager
def _wrap_submodules(
    f: torch.nn.Module | Callable[..., object],
    preserve_signature: Iterable[str],
    module_call_signatures: dict[str, dict[str, pytree.TreeSpec]],
) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that wraps each submodule path in preserve_signature of f
    # with the tracepoint hooks (recording specs into module_call_signatures) for the body
    # of the with block, then removes all hooks even if an error occurs.
    handles: list[RemovableHandle] = []

    try:
        for path in preserve_signature:
            # pyrefly: ignore[bad-argument-type]  # TODO f is an nn.Module at runtime (asserted in _wrap_submodule)
            handles.extend(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _mark_strict_experimental(cls: type[_R]) -> type[_R]:
    # [FUNCTION DOC] HELPER: Class decorator that replaces __call__ so calling an instance runs it under
    # the strict_mode higher-order op with the given positional args. Mutates and returns cls.
    def call(self: Any, *args: Any) -> Any:
        # [FUNCTION DOC] HELPER: Call the instance through strict_mode with the positional args.
        return strict_mode(self, args)

    cls.__call__ = call  # type: ignore[attr-defined]
    return cls


def _register_func_spec_proxy_in_tracer(
    tracer: "_ProxyTracer", name: str, spec: pytree.TreeSpec
) -> "Proxy":
    # [FUNCTION DOC] HELPER: Return a get_attr proxy for a pytree spec stored on the tracer root, reusing
    # the existing attribute when the cached name (name plus "0") is present, else
    # registering a fresh attribute. Raises AssertionError if the cached attribute holds a
    # different spec. Mutates the tracer root.
    # KNOWN ISSUE: the cache lookup assumes the first fresh qualname is name plus "0"; if the
    # tracer yields another name the cache never hits and duplicates are created.
    """
    This is a wrapper utility method on top of tracer to cache the
    already registered subclass spec attribute. This is useful because
    Subclass.__init__ will be same for each subclass. By default, fx will
    create multiple attributes/proxies for given attribute.
    """
    fx_name = name + "0"
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    if hasattr(tracer.root, fx_name):
        # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
        if getattr(tracer.root, fx_name) != spec:
            raise AssertionError(f"spec mismatch for {fx_name}")
        return tracer.create_proxy("get_attr", fx_name, (), {})

    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    qualname = tracer.get_fresh_qualname(name)
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    setattr(tracer.root, qualname, spec)
    return tracer.create_proxy("get_attr", qualname, (), {})


def _emit_flat_apply_call(
    *,
    tracer: "_ProxyTracer",
    spec_name: str,
    const_target_for_apply: Callable[..., object],
    graphable_args: pytree.PyTree,
    track_value: object,
    call_spec_cache_key: str,
) -> None:
    # [FUNCTION DOC] HELPER: Emit a flat_apply call node into the traced graph. Flattens graphable_args to
    # leaves plus a spec stored on the tracer root, obtains a cached spec for a constant
    # function wrapper of const_target_for_apply, creates the call_function node with proxied
    # leaves, and tracks track_value to the node's output. Returns nothing; mutates the graph.
    # Flatten to graphable form and record the spec on the FX root
    flat_args, in_spec = to_graphable(graphable_args)
    qualname = tracer.get_fresh_qualname(spec_name)  # type: ignore[union-attr]
    setattr(tracer.root, qualname, in_spec)  # type: ignore[union-attr]
    spec_proxy = tracer.create_proxy("get_attr", qualname, (), {})

    # Reuse/cached ConstantFunction spec on the root
    _, func_spec = pytree.tree_flatten(_ConstantFunction(const_target_for_apply))
    func_spec_proxy = _register_func_spec_proxy_in_tracer(
        tracer, f"{call_spec_cache_key}_const_func_spec", func_spec
    )

    # Map runtime args -> proxies (always via tracer.unwrap_proxy now)
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    flat_proxy_args = pytree.tree_map(tracer.unwrap_proxy, flat_args)

    # Emit flat_apply and track result structure
    out_proxy = tracer.create_proxy(
        "call_function", flat_apply, (func_spec_proxy, spec_proxy, *flat_proxy_args), {}
    )
    track_tensor_tree(track_value, out_proxy, constant=None, tracer=tracer)


def _is_init(fn: object) -> bool:
    # [FUNCTION DOC] HELPER: True if fn is callable and named __init__.
    return callable(fn) and fn.__name__ == "__init__"


def mark_subclass_constructor_exportable_experimental(
    constructor_subclass: Callable[_P, None],
) -> Callable[_P, None]:
    # [FUNCTION DOC] Decorator for a tensor subclass __init__. Raises RuntimeError at decoration time if
    # the function is not named __init__. The returned wrapper runs the original constructor;
    # while exporting it also (a) raises RuntimeError if the class is not a traceable wrapper
    # subclass, (b) returns quietly if no pre-dispatch torch-function mode is active, and (c)
    # otherwise records a flat_apply node that rebuilds the subclass from the constructor
    # arguments (excluding self) and tracks the instance to it.
    """
    Experimental decorator that makes subclass to be traceable in export
    with pre-dispatch IR. To make your subclass traceable in export, you need to:
        1. Implement __init__ method for your subclass (Look at DTensor implementation)
        2. Decorate your __init__ method with _mark_constructor_exportable_experimental
        3. Put torch._dynamo_disable decorator to prevent dynamo from peeking into its' impl

    Example:

    class FooTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, elem, *, requires_grad=False):
            # ...
            return torch.Tensor._make_subclass(cls, elem, requires_grad=requires_grad)

        @torch._dynamo_disable
        @mark_subclass_constructor_exportable_experimental
        def __init__(self, elem, ...):
            # ...
    """
    if not _is_init(constructor_subclass):
        raise RuntimeError(
            f"torch._export.wrappers.mark_constructor_exportable_experimental can only be applied on subclass tensor.__init__"
            f"But, you are adding it on {constructor_subclass.__name__} which is not supported. "
            f"If __init__ doesn't exist on your subclass, please add it. Look at DTensor.__init__ implementation for example"
        )

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        # [FUNCTION DOC] HELPER: Wrapper described by mark_subclass_constructor_exportable_experimental:
        # run the constructor, then during export emit the flat_apply constructor node.
        # Raises RuntimeError for non-traceable subclasses; AssertionError for unexpected
        # qualname or mode type.
        constructor_subclass(*args, **kwargs)

        if not torch.compiler.is_exporting():
            return

        if not is_traceable_wrapper_subclass_type(type(args[0])):
            if not constructor_subclass.__qualname__.endswith("__init__"):
                raise AssertionError(
                    f"expected __qualname__ to end with '__init__', got {constructor_subclass.__qualname__}"
                )
            obj_name = constructor_subclass.__qualname__[: -len("__init__")]
            raise RuntimeError(
                f"Can't intercept {obj_name} in export because this object is not a traceable "
                f"tensor subclass. Please look at DTensor.__init__ implementation as an example of proper usage of this API."
            )

        mode = _maybe_find_pre_dispatch_tf_mode_for_export()
        if mode is None:
            return

        if not isinstance(mode, PreDispatchTorchFunctionMode):
            raise AssertionError(
                f"expected PreDispatchTorchFunctionMode, got {type(mode)}"
            )

        tracer = mode.tracer
        subclass = args[0]
        graphable = (tuple(args[1:]), kwargs)

        spec_name = "_".join(constructor_subclass.__qualname__.lower().split("."))
        call_spec_cache_key = type(subclass).__name__.lower()

        _emit_flat_apply_call(
            tracer=tracer,
            spec_name=spec_name,
            const_target_for_apply=type(subclass),
            graphable_args=graphable,
            track_value=subclass,  # track the constructed subclass instance
            call_spec_cache_key=call_spec_cache_key,
        )
        return

    return wrapper


def allow_in_pre_dispatch_graph(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Decorator that lets an __init__ (delegated to the subclass-constructor decorator) or a
    # custom autograd Function apply method appear in the pre-dispatch export graph. Raises
    # RuntimeError at decoration time for any other function name. The wrapper behaves like
    # the original unless exporting a call on an autograd Function class under a pre-dispatch
    # proxy mode; then it runs the function with pre-dispatch excluded for the concrete
    # result and emits a flat_apply node calling the custom-autograd helper, tracking the output.
    # KNOWN ISSUE: the name check repeats _is_init after it was already handled, and the error
    # message says "__init_" (typo).
    """
    Experimental decorator that adds user function to export pre-dispatch graph. Note that
    we only support custom autograd function/subclass constructors today. To use this function:
        1. For subclasses:
            1. refer to instructions in mark_subclass_constructor_exportable_experimental
        2. Define apply method on your custom autograd function and apply this decorator.

    Example:

    class MyCoolCustomAutogradFunc(autograd.Function):
        @classmethod
        @torch._export.wrappers.allow_in_pre_dispatch_graph
        def apply(cls, *args, **kwargs):
            return super(MyCoolCustomAutogradFunc, cls).apply(*args, **kwargs)

    """
    if _is_init(func):
        # pyrefly: ignore[bad-return, bad-argument-type]  # TODO _is_init(func) implies func returns None
        return mark_subclass_constructor_exportable_experimental(func)

    if not (_is_init(func) or func.__name__ == "apply"):
        raise RuntimeError(
            f"torch._export.wrappers.allow_in_pre_dispatch_graph can only be applied on subclass tensor.__init_ "
            f"or custom_autograd_function.apply. "
            f"But, you are adding it on {func.__name__} which is not supported. "
            f"If __init__ doesn't exist on your subclass, please add it. Look at DTensor.__init__ implementation for example. "
            f"If you are adding it on custom autograd function, please add it on apply method. "
            f"If anything else, file an issue on github and we may consider extending our support. "
        )

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Wrapper described by allow_in_pre_dispatch_graph. Falls through to the plain
        # call unless exporting, args[0] is an autograd Function subclass, and a pre-dispatch
        # proxy mode exists. Raises AssertionError if that mode is not pre-dispatch.
        if not torch.compiler.is_exporting():
            return func(*args, **kwargs)

        if not inspect.isclass(args[0]):
            return func(*args, **kwargs)

        if not issubclass(args[0], torch.autograd.Function):
            return func(*args, **kwargs)

        from torch._ops import _get_dispatch_mode_pre_dispatch

        mode = _get_dispatch_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)
        if mode is None:
            return func(*args, **kwargs)

        # Sometimes custom autograd functions can call into HOPs that don't have proxy impl
        # at PreDispatch level, so we just dispatch it below to get the concrete result.
        include_to_set = torch._C._dispatch_tls_local_include_set().remove(
            torch._C.DispatchKey.PreDispatch
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set()
            | torch._C.DispatchKeySet(torch._C.DispatchKey.PreDispatch)
        )

        with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
            out = func(*args, **kwargs)

        if not mode.pre_dispatch:
            raise AssertionError("Should only do this in predispatch")
        tracer = mode.tracer

        function_cls_name = f"{args[0].__module__}.{args[0].__qualname__}"
        graphable = ((function_cls_name, *args[1:]), kwargs)

        from torch.export.custom_ops import (
            _call_custom_autograd_function_in_pre_dispatch,
        )

        spec_name = "_".join(function_cls_name.split("."))
        call_spec_cache_key = type(
            _call_custom_autograd_function_in_pre_dispatch
        ).__name__.lower()
        _emit_flat_apply_call(
            tracer=tracer,
            spec_name=spec_name,
            const_target_for_apply=_call_custom_autograd_function_in_pre_dispatch,
            graphable_args=graphable,
            track_value=out,
            call_spec_cache_key=call_spec_cache_key,
        )
        return out

    return wrapper
