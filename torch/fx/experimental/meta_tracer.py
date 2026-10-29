# [MODULE DOC]
# Experimental FX tracer that propagates meta-device tensors alongside symbolic
# tracing, so each traced node carries shape/dtype metadata (_tensor_meta) and
# size/dim/shape/dtype queries are resolved to concrete values during tracing.
# Main contents: manual override functions for ops that do not work on meta tensors
# (manual_meta_overrides), a constructor wrapper, MetaProxy / MetaAttribute /
# MetaDeviceAttribute proxy types, the MetaTracer class and a symbolic_trace entry
# point. No import-time side effects (torch functions are patched only while
# MetaTracer.trace runs).
import builtins
import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.fx
from torch.fx.node import Node
from torch.fx.proxy import Proxy


_C = TypeVar("_C", bound=Callable[..., Any])

__all__ = [
    "embedding_override",
    "functional_relu_override",
    "gen_constructor_wrapper",
    "manual_meta_overrides",
    "MetaAttribute",
    "MetaDeviceAttribute",
    "MetaProxy",
    "MetaTracer",
    "nn_layernorm_override",
    "proxys_to_metas",
    "symbolic_trace",
    "torch_abs_override",
    "torch_nn_relu_override",
    "torch_relu_override",
    "torch_where_override",
]


def embedding_override(self: torch.nn.Embedding, input: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Meta-propagation stand-in for an embedding module call: returns
    # an uninitialized meta tensor shaped like the input indices plus a trailing
    # dimension equal to the weight's embedding size. No side effects.
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta")


def nn_layernorm_override(
    self: torch.nn.LayerNorm, input: torch.Tensor
) -> torch.Tensor:
    # [FUNCTION DOC] Meta-propagation stand-in for a layer norm module call: output
    # metadata equals the input (returns the input unchanged).
    return input


def torch_relu_override(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Meta stand-in for the relu function: returns the input as is.
    return x


def torch_nn_relu_override(self: torch.nn.ReLU, x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Meta stand-in for a ReLU module call: returns the input as is.
    return x


def functional_relu_override(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] Meta stand-in for the functional relu: returns the input as is.
    # Raises AssertionError if inplace is true (unsupported for meta analysis).
    if inplace:
        raise AssertionError(
            "don't support inplace functional.relu for metatensor analysis"
        )
    return x


def torch_where_override(
    condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # [FUNCTION DOC] Meta stand-in for the three-argument where: moves condition, x
    # and y to the meta device and adds them, so the result has the broadcast shape
    # of all three. The dtype of the result follows addition promotion, not where.
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs_override(
    input: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] Meta stand-in for abs: returns the input as is. Raises
    # AssertionError if an out tensor is supplied.
    if out is not None:
        raise AssertionError("Don't support in-place abs for MetaTensor analysis")
    return input


manual_meta_overrides: dict[Callable[..., Any], Callable[..., Any]] = {
    torch.nn.Embedding: embedding_override,
    torch.nn.LayerNorm: nn_layernorm_override,
    torch.relu: torch_relu_override,
    torch.nn.functional.relu: functional_relu_override,
    torch.nn.ReLU: torch_nn_relu_override,
    torch.where: torch_where_override,
    torch.abs: torch_abs_override,
}


def gen_constructor_wrapper(
    target: _C,
) -> tuple[Callable[..., Any], _C]:
    # [FUNCTION DOC] Builds a replacement for a torch tensor constructor (target).
    # Returns a pair of the wrapper and the original target. When called, the
    # wrapper scans its args and kwargs for an fx Proxy; if one is found it records
    # a call_function node through that proxy's tracer and returns the new proxy,
    # otherwise it calls the original constructor directly.
    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Proxy-aware call of the wrapped constructor; see
        # gen_constructor_wrapper. Returns a Proxy if any argument is a Proxy, else
        # the real constructor result.
        proxy = None

        def check_has_proxy(v: Any) -> None:
            # [FUNCTION DOC] HELPER: If v is a Proxy, stores it in the enclosing
            # wrapper's proxy variable (last one seen wins); otherwise does nothing.
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


class MetaProxy(torch.fx.Proxy):
    # [CLASS DOC] fx Proxy that can carry a meta tensor (_tensor_meta, set via
    # install_tensor_meta). When metadata is present, size/dim/shape/dtype return
    # concrete values instead of recording graph nodes; otherwise they record nodes.
    # Unknown attribute access yields a lazy MetaAttribute; device yields a
    # MetaDeviceAttribute so device usage can be replaced by "meta".
    def install_tensor_meta(self, tensor_meta: torch.Tensor) -> None:
        # [FUNCTION DOC] Attaches the given meta tensor to this proxy (mutates it).
        self._tensor_meta = tensor_meta

    def size(self, dim: int | None = None) -> Any:
        # [FUNCTION DOC] Returns the concrete size (whole shape or one dimension) from
        # the installed meta tensor; without metadata, records a call_method size node
        # and returns its proxy.
        # KNOWN ISSUE: the dim check is by truthiness, so dim equal to zero is treated
        # as no dim and the whole size is returned.
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.size(*[dim] if dim else [])
        return self.tracer.create_proxy(
            "call_method", "size", (self, dim) if dim else (self,), {}
        )

    def dim(self) -> Any:
        # [FUNCTION DOC] Returns the rank from the meta tensor if installed, else
        # records a call_method dim node and returns its proxy.
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy("call_method", "dim", (self,), {})

    @property
    def shape(self) -> Any:
        # [FUNCTION DOC] Property. Concrete shape from the meta tensor if installed,
        # else a proxy for a recorded attribute read of shape.
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.shape
        return self.tracer.create_proxy(
            "call_function", builtins.getattr, (self, "shape"), {}
        )

    @property
    def dtype(self) -> Any:
        # [FUNCTION DOC] Property. Concrete dtype from the meta tensor if installed,
        # else a proxy for a recorded attribute read of dtype.
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dtype
        return self.tracer.create_proxy(
            "call_function", builtins.getattr, (self, "dtype"), {}
        )

    @property
    def device(self) -> "MetaDeviceAttribute":
        # [FUNCTION DOC] Property. Returns a MetaDeviceAttribute marker (no node is
        # recorded yet) so that proxys_to_metas can substitute the string "meta".
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __getattr__(self, k: str) -> Any:
        # [FUNCTION DOC] Called for attributes not found normally. Looking up the
        # private meta slot re-raises through normal lookup (AttributeError if unset);
        # any other name returns a lazy MetaAttribute on this proxy.
        if k == "_tensor_meta":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return MetaAttribute(self, k)


class MetaAttribute(MetaProxy):
    # [CLASS DOC] Lazy proxy for an attribute (root, attr) of a MetaProxy. Its graph
    # node is created only when the node is first needed (as an attribute read);
    # calling it records a call_method node instead, so method calls avoid the
    # intermediate attribute node.
    def __init__(self, root: MetaProxy, attr: str) -> None:
        # [FUNCTION DOC] Stores root, attr and the root's tracer; the node starts
        # unset. Does not call the Proxy base initializer.
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node | None = None

    @property
    def node(self):  # type: ignore[override]
        # [FUNCTION DOC] Property. On first access records a call_function node that
        # reads attribute attr from root, caches it and returns it (mutates the cache).
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Treats the attribute as a method: records a call_method node
        # named attr with root as receiver plus the given args and kwargs, and returns
        # the resulting proxy.
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs
        )


class MetaDeviceAttribute(MetaAttribute):
    # [CLASS DOC] Marker subtype of MetaAttribute for the device attribute; recognized
    # by proxys_to_metas, which replaces it with the constant string "meta".
    pass


def proxys_to_metas(v: Any) -> Any:
    # [FUNCTION DOC] Maps one value to its meta-propagation form: a device marker
    # becomes the string "meta"; a MetaProxy becomes its installed meta tensor; other
    # values pass through. Raises AssertionError for a non-Meta Proxy or a MetaProxy
    # without installed metadata.
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        if not isinstance(v, MetaProxy):
            raise AssertionError(f"Expected MetaProxy but got {type(v)}")
        if not hasattr(v, "_tensor_meta"):
            raise AssertionError("MetaProxy does not have an associated meta")
        return v._tensor_meta
    return v


class MetaTracer(torch.fx.Tracer):
    # [CLASS DOC] fx Tracer that, for every node it creates, also executes the op on
    # meta tensors and stores the result on the proxy. State: meta_args (placeholder
    # name to meta tensor), patched_torch_methods and orig_fns (constructors replaced
    # during tracing), orig_forward (last module forward seen), and the transient
    # flag _disable_module_getattr. Stateless modules not registered as submodules
    # may be auto-registered when allow_insert_stateless_mods is true.
    allow_insert_stateless_mods: bool = True

    _TORCH_METHODS_TO_PATCH = ["arange", "zeros", "ones", "full_like", "eye"]

    def create_proxy(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any = None,
        proxy_factory_fn: Callable[[Node], Proxy] | None = None,
    ) -> MetaProxy:
        # [FUNCTION DOC] Creates the node via the base tracer, then computes meta
        # output. Placeholders listed in meta_args get their meta tensor installed.
        # For patched constructors a device keyword is forced to "meta" in the kwargs.
        # call_function (with manual override lookup), call_method, call_module (with
        # module-type overrides, else the original forward) and get_attr (tensor moved
        # to meta) are evaluated on meta values and installed on the proxy. Any failure
        # (including non-proxy composite outputs) only emits a warning and leaves the
        # proxy without metadata. Other kinds return the proxy unchanged.
        # Side effect: toggles _disable_module_getattr temporarily.
        rv = super().create_proxy(
            kind,
            target,
            args,
            kwargs,
            name,
            type_expr,
            # pyrefly: ignore [bad-argument-type]
            proxy_factory_fn,
        )

        if kind == "placeholder" and target in self.meta_args:
            rv.install_tensor_meta(self.meta_args[target])
            return rv  # pyrefly: ignore [bad-return]

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)

            if kind == "call_function":
                # pyrefly: ignore [no-matching-overload]
                meta_target = manual_meta_overrides.get(target, target)

                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                meta_target = getattr(args_metas[0], target)  # type: ignore[index]
                meta_out = meta_target(*args_metas[1:], **kwargs_metas)  # type: ignore[index]
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AssertionError("orig_forward not set for call_module")
                self._disable_module_getattr = True
                try:
                    # pyrefly: ignore [bad-argument-type]
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](
                            mod, *args_metas, **kwargs_metas
                        )  # type: ignore[misc, arg-type]
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")  # pyrefly: ignore [missing-attribute]
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if not isinstance(attr_itr, torch.Tensor):
                        raise AssertionError(f"Expected Tensor, got {type(attr_itr)}")
                    meta_out = attr_itr.to(device="meta")
                finally:
                    self._disable_module_getattr = False
            else:
                return rv  # pyrefly: ignore [bad-return]

            # TODO
            if not isinstance(rv, torch.fx.Proxy):
                raise AssertionError("Don't support composite output yet")
            rv.install_tensor_meta(meta_out)
        except Exception as e:
            warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        return rv  # pyrefly: ignore [bad-return]

    def getattr(
        self, attr: str, attr_val: Any, parameter_proxy_cache: dict[str, Proxy]
    ) -> Any:
        # [FUNCTION DOC] Module attribute hook: while meta evaluation is running
        # (_disable_module_getattr set) returns the raw value; otherwise defers to the
        # base tracer, which may wrap parameters in proxies.
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:
            return super().getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # [FUNCTION DOC] Remembers the module's forward callable in orig_forward (used
        # later by create_proxy for meta evaluation), then defers to the base tracer.
        # Returns the base result.
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def _insert_module_as_submodule(self, mod: torch.nn.Module) -> str:
        # [FUNCTION DOC] HELPER: Registers mod on the traced root under a fresh name
        # built from its lowercase class name and a counter, and returns that name.
        # Mutates the root module.
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        while hasattr(self.root, path):
            path = f"{mod_name}_{idx}"
            idx += 1

        self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: torch.nn.Module) -> str:
        # [FUNCTION DOC] Returns the qualified name of mod in the root. If the base
        # lookup raises NameError and auto-insertion is allowed and the module has no
        # parameters or buffers, registers it as a new submodule, records the name in
        # prev_module and returns it; otherwise re-raises the NameError.
        try:
            return super().path_of_module(mod)
        except NameError:
            if (
                self.allow_insert_stateless_mods
                and len(list(mod.parameters())) == 0
                and len(list(mod.buffers())) == 0
            ):
                path = self._insert_module_as_submodule(mod)
                self.prev_module = path
                return path
            raise

    def proxy(self, node: torch.fx.Node) -> MetaProxy:
        # [FUNCTION DOC] Proxy factory: wraps a node in a MetaProxy bound to this tracer.
        return MetaProxy(node, self)

    def trace(self, root, meta_args: dict[str, torch.Tensor], concrete_args=None):  # type: ignore[override]
        # [FUNCTION DOC] Traces root with placeholder metadata given by meta_args
        # (mapping of argument name to meta tensor). Raises AssertionError if
        # meta_args is not a mapping. While tracing, temporarily replaces torch
        # constructors arange, zeros, ones, full_like and eye with proxy-aware
        # wrappers, and restores them afterwards even on error. Returns the Graph,
        # with its tracer extras recording meta_args.
        if not isinstance(meta_args, dict):
            raise AssertionError(f"Expected dict for meta_args, got {type(meta_args)}")
        self.meta_args = meta_args

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            graph = super().trace(root, concrete_args)
            graph._tracer_extras = {"meta_args": meta_args}
            return graph
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)


def symbolic_trace(
    root: torch.nn.Module | Callable[..., Any],
    meta_args: dict[str, torch.Tensor] | None = None,
    concrete_args: dict[str, Any] | None = None,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Traces a module or function with a MetaTracer and returns a
    # GraphModule named after the module's class or the function's name.
    # KNOWN ISSUE: meta_args defaults to no value, but the tracer rejects anything
    # that is not a mapping, so calling without meta_args raises AssertionError.
    tracer = MetaTracer()
    graph = tracer.trace(root, meta_args, concrete_args)  # type: ignore[arg-type]
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    gm = torch.fx.GraphModule(tracer.root, graph, name)
    return gm
