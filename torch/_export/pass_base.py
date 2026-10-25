# [MODULE DOC] Deprecated base class for interpreter-based export passes
# (_ExportPassBaseDeprecatedDoNotUse) plus the tracer and interpreter helpers that
# rebuild a graph while attaching fake-tensor metadata, and type aliases Argument,
# Value, Fn and PassType. Exports only the base class. Import-time side effects: builds
# the set of torch symbolic function targets.
# mypy: allow-untyped-defs
import operator
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._higher_order_ops.map import _unstack_pytree
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    is_fake_tensor,
    maybe_get_fake_constant,
    maybe_get_fake_mode,
)
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    PropagateUnbackedSymInts,
)
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree


__all__ = ["_ExportPassBaseDeprecatedDoNotUse"]


Argument = Any
Value = Any
Fn = Callable[..., Any]
PassType = Callable[[torch.fx.GraphModule], PassResult | None]


_TORCH_SYM_OPS: set[Callable] = {
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_not,
    torch.sym_sqrt,
}


class ExportPassBaseError(RuntimeError):
    # [CLASS DOC] Error type raised for unsupported or misused operations in the export
    # pass base class (plain RuntimeError subclass, no state).
    pass


class _ExportPassBaseDeprecatedDoNotUse(PassBase):
    # [CLASS DOC] HELPER: Deprecated base class for interpreter-based export passes.
    # Subclasses override hooks such as placeholder, call_operator, call_cond, call_map,
    # output and on_attr; the base class re-traces the graph through an interpreter plus
    # an ExportTracer so that every new node gets fake-tensor val and tensor_meta
    # metadata. State: interpreter (used to compute result values), tracer (builds the
    # new graph), fake_tensor_mode, node_debug_str (text of the node being processed)
    # and _initialized (set by __init__, checked by call). Key methods: call,
    # call_submodule, _fx, inputs.
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    @staticmethod
    def _create_dummy_node_metadata():
        # [FUNCTION DOC] HELPER: Static method returning a NodeMetadata with a
        # stack_trace string generated from the current call stack (one frame), for
        # nodes that have no original metadata.
        return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})

    class ExportTracer(PythonKeyTracer):
        # [CLASS DOC] HELPER: PythonKeyTracer subclass used to build the new graph
        # during a pass. State: callback (the owning pass), root (a plain Module holding
        # submodules and attributes), graph (the new Graph using the given codegen),
        # tensor_attrs, fake_tensor_mode and submodules (map from module to its
        # registered name submodule_N). Tracing from a function is unsupported; nodes
        # are created via create_proxy by the pass.
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", codegen: CodeGen
        ) -> None:
            # [FUNCTION DOC] HELPER: Create the tracer for the given pass callback, with
            # an empty root module, a new graph using the supplied codegen, and empty
            # attribute/submodule tables.
            super().__init__()
            self.callback = callback
            self.root = torch.nn.Module()
            self.graph = torch.fx.Graph()
            self.graph.set_codegen(codegen)
            self.tensor_attrs: dict[str, torch.Tensor] = {}  # type: ignore[assignment]
            self.fake_tensor_mode: FakeTensorMode | None = None
            self.submodules: dict[torch.nn.Module, str] = {}

        def trace(self) -> None:  # type: ignore[override]
            # [FUNCTION DOC] HELPER: Always raises ExportPassBaseError; this tracer
            # cannot trace a function.
            raise ExportPassBaseError("ExportTracer doesn't support trace().")

        def create_arg(self, a: Argument) -> torch.fx.Node:
            # [FUNCTION DOC] HELPER: Convert a Python argument into a graph argument.
            # Modules are registered once on the root module under a generated name.
            # Fake tensors must carry a real fake constant, which replaces them
            # (otherwise raises ExportPassBaseError). Tensors that become get_attr nodes
            # get metadata set and trigger the pass on_attr callback. Returns the
            # resulting Node or argument structure.
            if isinstance(a, torch.nn.Module):
                if a not in self.submodules:
                    name_submodule = f"submodule_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            elif is_fake_tensor(a):
                if maybe_get_fake_constant(a) is None:
                    raise ExportPassBaseError(f"Cannot add {a} to graph.")
                a = maybe_get_fake_constant(a)
            node = super().create_arg(a)
            if (
                isinstance(a, torch.Tensor)
                and isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
            ):
                self.set_metadata(node, a)
                self.callback.on_attr(ProxyValue(a, node))
            return node

        def set_metadata(
            self,
            node: torch.fx.Node,
            value: Argument,
        ) -> None:
            # [FUNCTION DOC] HELPER: Fill node meta val and tensor_meta from a (possibly
            # nested) value. val: fake tensors are kept; real tensors are converted to
            # fake tensors with the tracer fake mode (parameters with static shapes;
            # quantized tensors are dequantized first); symbolic and primitive scalars
            # are kept; anything else becomes None. Tensors that cannot be fakeified
            # (subclasses) print a message and get None as val plus an extracted
            # TensorMetadata entry in tensor_meta. Mutates node.meta. Raises
            # AssertionError if the fake tensor mode is unset when needed.
            # propagate the fake tensor or sym nodes
            def make_val(
                x: Argument,
            ) -> (
                FakeTensor
                | torch.SymInt
                | torch.SymFloat
                | torch.SymBool
                | int
                | float
                | bool
                | str
                | None
            ):
                # [FUNCTION DOC] HELPER: Map one leaf to its meta val as described by
                # set_metadata. Prints a notice on unsupported tensor subclasses.
                if isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
                    return x
                elif isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        if self.fake_tensor_mode is None:
                            raise AssertionError("fake_tensor_mode must not be None")
                        # TODO we should allocate static shapes
                        # for param/buffer values
                        if isinstance(x, torch.nn.Parameter):
                            fake_tensor = self.fake_tensor_mode.from_tensor(
                                x, static_shapes=True
                            )
                        else:
                            fake_tensor = self.fake_tensor_mode.from_tensor(x)
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        print(
                            "Fakeifying a Tensor subclass is not supported \
                            right now. Instead a TensorMetadata is used."
                        )
                        fake_tensor = None
                    return fake_tensor
                elif isinstance(
                    x,
                    (
                        torch.SymInt,
                        torch.SymFloat,
                        torch.SymBool,
                        int,
                        float,
                        bool,
                        str,
                    ),
                ):
                    return x
                else:
                    return None

            node.meta["val"] = pytree.tree_map(make_val, value)

            # Set the tensor_metadata for values that do not have a corresponding FakeTensor
            def make_tensor_meta(x: Argument) -> TensorMetadata | None:
                # [FUNCTION DOC] HELPER: For real tensors that cannot be fakeified,
                # return extracted TensorMetadata; return None for fake tensors,
                # fakeifiable tensors and non-tensors.
                if not is_fake_tensor(x) and isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        if self.fake_tensor_mode is None:
                            raise AssertionError("fake_tensor_mode must not be None")
                        _ = self.fake_tensor_mode.from_tensor(x)
                        tensor_meta = None
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        tensor_meta = _extract_tensor_metadata(x)
                    return tensor_meta
                else:
                    return None

            node.meta["tensor_meta"] = pytree.tree_map(make_tensor_meta, value)

    class ExportInterpreter(fx.Interpreter):
        # [CLASS DOC] HELPER: fx.Interpreter subclass that forwards each node kind to
        # the pass callbacks. State: callback (the pass) and node (the node currently
        # executing, used to supply original metadata). call_module and call_method are
        # rejected.
        def __init__(
            self, callback: "_ExportPassBaseDeprecatedDoNotUse", gm: fx.GraphModule
        ) -> None:
            # [FUNCTION DOC] HELPER: Initialize the interpreter on gm, remember the pass
            # and start with the first node as the current node.
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

        # pyrefly: ignore [bad-override]
        def placeholder(
            self,
            target: str,  # type: ignore[override]
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            # [FUNCTION DOC] HELPER: Run the default placeholder handling to get the
            # input value, then call the pass placeholder hook with the current node
            # metadata. Returns the ProxyValue.
            arg = super().placeholder(target, args, kwargs)
            return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

        def output(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            # [FUNCTION DOC] HELPER: Call the pass output hook with the output argument
            # and current node metadata, and return the data field of the resulting
            # ProxyValue.
            return self.callback.output(args[0], NodeMetadata(self.node.meta)).data  # type: ignore[return-value]

        def call_function(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> ProxyValue:
            # [FUNCTION DOC] HELPER: Dispatch a call_function node to the matching pass
            # hook: getitem to call_getitem; operator/builtins/math functions and torch
            # sym ops to call_sym; OpOverload or packets to call_operator; higher-order
            # cond to call_cond; map_impl to call_map; other HigherOrderOperators are
            # re-emitted directly through the _fx helper. Raises ExportPassBaseError for
            # any other target type.
            meta = NodeMetadata(self.node.meta)

            if target is operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif getattr(target, "__module__", None) in {
                "_operator",
                "builtins",
                "math",
            }:
                if not callable(target):
                    raise AssertionError(f"expected callable target, got {target}")
                return self.callback.call_sym(target, args, meta)
            elif target in _TORCH_SYM_OPS:
                if not callable(target):
                    raise AssertionError(f"expected callable target, got {target}")
                return self.callback.call_sym(target, args, meta)
            elif isinstance(
                target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
            ):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target is torch.ops.higher_order.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target is torch.ops.higher_order.map_impl:
                f, mapped_args, operands = args  # type: ignore[assignment]
                return self.callback.call_map(f, mapped_args, operands, meta)
            # For other unregistered HigherOrderOps, just interpret them blindly
            elif isinstance(target, torch._ops.HigherOrderOperator):
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )
            else:
                raise ExportPassBaseError(f"Unsupported target type: {target}")

        def get_attr(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> Argument:
            # [FUNCTION DOC] HELPER: Delegate to the default interpreter get_attr
            # behavior (no extra handling).
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: torch.fx.node.Target,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            # [FUNCTION DOC] HELPER: Always raises ExportPassBaseError (call_module
            # unsupported).
            raise ExportPassBaseError("call_module is not supported.")

        def call_method(  # type: ignore[override]
            self,
            target: str,
            args: tuple[Argument, ...],
            kwargs: dict[str, Argument],
        ) -> None:
            # [FUNCTION DOC] HELPER: Always raises ExportPassBaseError (call_method
            # unsupported).
            raise ExportPassBaseError("call_method is not supported.")

        def run_node(self, n: torch.fx.Node) -> Argument:
            # [FUNCTION DOC] HELPER: Record n as the current node and store its
            # formatted text on the pass for debugging, then run it with the default
            # behavior. Returns the node result.
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        # [FUNCTION DOC] Set up the pass: a PropagateUnbackedSymInts interpreter over an
        # empty graph module, a new ExportTracer with a default codegen, no fake tensor
        # mode yet, the initialized flag, and no debug string.
        self.interpreter = PropagateUnbackedSymInts(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())
        self.fake_tensor_mode: FakeTensorMode | None = None
        self._initialized = True
        self.node_debug_str: str | None = None

    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # [FUNCTION DOC] HELPER: Core node-emission helper. Runs the underlying
        # interpreter method of the given kind on the concrete data of args to compute
        # the result data, then creates a proxy for the node in the new graph using the
        # proxies of the args (op overloads get a name from the overload packet). Copies
        # meta into the node, records unbacked symbol bindings when a shape env exists,
        # sets val/tensor_meta from the result, and returns a ProxyValue pairing data
        # and proxy.
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        if isinstance(target, torch._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        res_proxy = self.tracer.create_proxy(
            kind, target, args_proxy, kwargs_proxy, name=name
        )
        res_proxy.node.meta.update(meta.data)
        if self.fake_tensor_mode and (shape_env := self.fake_tensor_mode.shape_env):
            if symbol_to_path := compute_unbacked_bindings(shape_env, res_data):
                res_proxy.node.meta["unbacked_bindings"] = symbol_to_path
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: torch.fx.GraphModule) -> list[Argument]:
        # [FUNCTION DOC] Compute example inputs for a graph module: use args stored in
        # the module meta if present; otherwise for each placeholder use its meta val
        # (preferring its fake constant), else build a fake tensor from tensor_meta,
        # else None for unused placeholders. Raises ExportPassBaseError if an input
        # cannot be constructed, AssertionError if a fake mode is required but unset.
        # Returns a list.
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: torch.fx.Node) -> FakeTensor | None:
            # [FUNCTION DOC] HELPER: Produce the example value for one placeholder node
            # as described by inputs.
            if "val" in node.meta:
                fake = node.meta["val"]
                if hasattr(fake, "constant") and fake.constant is not None:
                    return fake.constant
                return fake
            elif tensor_meta := node.meta.get("tensor_meta"):
                if self.fake_tensor_mode is None:
                    raise AssertionError("fake_tensor_mode must not be None")
                return FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    torch.device("cpu"),
                )
            elif len(node.users) == 0:
                return None
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        # [FUNCTION DOC] Hook called when a tensor attribute is added to the graph;
        # default does nothing.
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        # [FUNCTION DOC] Hook creating a placeholder node named name in the new graph
        # with the given metadata and a val derived from arg. Returns a ProxyValue of
        # arg and the new proxy.
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        arg_proxy.node.meta = meta.data
        self.tracer.set_metadata(arg_proxy.node, arg)
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # [FUNCTION DOC] Hook for a call to an operator overload: emit it through _fx
        # and return the ProxyValue. Subclasses override to rewrite ops.
        return self._fx("call_function", op, args, kwargs, meta)

    def call_sym(
        self,
        target: Fn,
        args: tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # [FUNCTION DOC] Hook for a symbolic or builtin function call with positional
        # args only; emitted through _fx.
        return self._fx("call_function", target, args, {}, meta)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: torch.fx.GraphModule,
        false_fn: torch.fx.GraphModule,
        inputs: list[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # [FUNCTION DOC] Hook for the cond higher-order op: re-traces both branch
        # submodules with the given inputs through call_submodule and emits a cond node
        # with the new branch modules. Raises AssertionError if a branch yields nothing.
        # Returns a ProxyValue.
        true_branch = self.call_submodule(true_fn, tuple(inputs))
        false_branch = self.call_submodule(false_fn, tuple(inputs))
        if true_branch is None:
            raise AssertionError("true_branch must not be None")
        if false_branch is None:
            raise AssertionError("false_branch must not be None")
        return self._fx(
            "call_function",
            torch.ops.higher_order.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)),
            {},
            meta,
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        mapped_args: list[ProxyValue],
        operands: list[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        # [FUNCTION DOC] Hook for map_impl: unstack the mapped arguments to build
        # example inputs, re-trace the body submodule via call_submodule with them and
        # the operands, then emit a map_impl node with the new body. Raises
        # AssertionError if retracing yields nothing.
        xs = _unstack_pytree([arg.data for arg in mapped_args])[0]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in operands]))
        if f_branch is None:
            raise AssertionError("f_branch must not be None")
        return self._fx(
            "call_function",
            torch.ops.higher_order.map_impl,
            (f_branch.graph_module, mapped_args, operands),
            {},
            meta,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        # [FUNCTION DOC] Hook for indexing a value by key; emitted through _fx as a
        # getitem call.
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        # [FUNCTION DOC] Hook emitting the graph output node for the results through
        # _fx.
        return self._fx("output", "output", (results,), {}, meta)

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: tuple[Argument, ...]
    ) -> PassResult:
        # [FUNCTION DOC] Re-trace a graph module with the given inputs using the pass
        # hooks. Temporarily swaps in a new tracer (sharing the fake tensor mode) and a
        # plain interpreter over an empty module, runs an ExportInterpreter with the
        # input data while preserving node meta, builds a new GraphModule from the
        # traced graph and restores the previous tracer and interpreter. Returns
        # PassResult with the new module and modified True.
        # KNOWN ISSUE: the return annotation is PassResult but callers treat the result
        # as possibly None; the previous tracer is not restored if the interpreter
        # raises.
        prev_tracer, self.tracer = (
            self.tracer,
            self.ExportTracer(self, graph_module.graph._codegen),
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        # pyrefly: ignore [bad-assignment]
        prev_interpreter, self.interpreter = (
            self.interpreter,
            torch.fx.Interpreter(  # type: ignore[assignment]
                torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
            ),
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        # [FUNCTION DOC] Entry point of the pass. Raises ExportPassBaseError if __init__
        # was not called. Builds example inputs, detects a single fake tensor mode among
        # them (raises AssertionError if several), otherwise creates a fresh
        # FakeTensorMode allowing non-fake inputs; enables the python dispatcher when a
        # fake mode exists. Then re-traces the module under that mode via call_submodule
        # and returns its PassResult.
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if is_fake_tensor(i):
                if (
                    fake_tensor_mode is not None
                    and fake_tensor_mode is not maybe_get_fake_mode(i)
                ):
                    raise AssertionError("Multiple fake tensor mode detected.")
                fake_tensor_mode = maybe_get_fake_mode(i)
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            fake_tensor_mode.allow_non_fake_inputs = True
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        return result
