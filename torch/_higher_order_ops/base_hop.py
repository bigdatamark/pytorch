# mypy: allow-untyped-defs

# [MODULE DOC] Defines BaseHOP, a base class for higher-order operators of the form
# hop(subgraph, *operands, **kwargs) whose semantics are just calling the subgraph on the
# operands, with default autograd, functionalization, proxy tracing, fake and dense
# implementations. Also defines BaseHOPFunction (autograd function used for backward) and
# FunctionWithNoFreeVars (marker wrapper for plain callables). No import-time side effects.
import abc

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.auto_functionalize import FunctionalCallableWithEpilogue
from torch._higher_order_ops.utils import (
    check_input_alias_and_mutation_return_outputs,
    HopInstance,
    materialize_as_graph,
    reenter_make_fx,
    register_fake,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class BaseHOP(HigherOrderOperator, abc.ABC):
    # [CLASS DOC] Abstract base for subgraph-calling higher-order operators. Subclasses give a
    # name via the constructor; the constructor registers default implementations
    # (autograd, functionalize, proxy mode, fake, dense) as bound methods, which subclasses may
    # override. Kwargs are config options and are never passed to the subgraph. Must only be
    # subclassed in tree.
    """
    This is the "Base" HOP implementation for a HOP that looks like:

        call_subgraph_hop(subgraph, *operands, **kwargs)

    That is:
    1) the HOP stays alive until Inductor
    2) the HOP's semantics are subgraph(*operands)
    3) kwargs may be some config options but aren't passed directly to the subgraph.

    To use this, please subclass this class and override methods as necessary:
    ```
    class InvokeQuant(BaseHOP):
        def __init__(self):
            return super().__init__("invoke_quant")


    invoke_quant = InvokeQuant()


    def g(x):
        return x.sin().cos()


    @torch.compile(backend="aot_eager")
    def f(x):
        return invoke_quant(g, x, scheme="nf4")
    ```

    NOTE: don't subclass BaseHOP out of tree! That is not allowed. All
    usages must be in tree.
    """

    def __init__(self, hop_name) -> None:
        # [FUNCTION DOC] Creates the operator with the given name and registers the default
        # autograd, functionalization, proxy-mode, fake-tensor (cache skipped) and
        # CompositeExplicitAutograd implementations on this instance.
        super().__init__(hop_name)

        # Set up the registrations
        # If you want to override any of these, override them in your subclass.
        self.py_autograd_impl(self._call_Autograd)
        self.py_functionalize_impl(self._call_Functionalize)
        self.py_impl(ProxyTorchDispatchMode)(self._call_ProxyTorchDispatchMode)
        register_fake(self, self._fake_impl, skip_cache=True)
        self.py_impl(DispatchKey.CompositeExplicitAutograd)(
            self._call_CompositeExplicitAutograd
        )

    def __call__(self, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] Entry point. Raises RuntimeError unless subgraph is a GraphModule,
        # a FunctionWithNoFreeVars, or a FunctionalCallableWithEpilogue; then dispatches via the
        # base operator and returns the subgraph's outputs.
        if not isinstance(
            subgraph,
            (
                torch.fx.GraphModule,
                FunctionWithNoFreeVars,
                FunctionalCallableWithEpilogue,
            ),
        ):
            raise RuntimeError(
                f"{self._name}: when calling this API without torch.compile, "
                f"we require that the subgraph be a torch.fx.GraphModule (or "
                f"a function we know doesn't have free variables)."
            )
        # pyrefly: ignore [missing-attribute]
        return super().__call__(subgraph, *operands, **kwargs)

    def _call_Autograd(self, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] HELPER: Autograd implementation: applies the custom autograd function
        # wrapping this hop, subgraph, kwargs and operands. Assumes no input mutation or
        # aliasing in the subgraph. Returns the hop outputs.
        # KNOWN ISSUE: the GraphModule check below is a no-op (its body is just pass).
        if isinstance(subgraph, torch.fx.GraphModule):
            pass

        # We assume the subgraph doesn't mutate inputs and there is no aliasing.
        # In the PT2 stack, this is Dynamo's responsibility to figure out.
        return BaseHOPFunction.apply(self, subgraph, kwargs, *operands)

    def _call_CompositeExplicitAutograd(self, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] HELPER: Dense implementation: asserts no dispatch mode is active
        # (AssertionError otherwise), then simply calls the subgraph on the operands and
        # returns its result. Kwargs are ignored.
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        if mode is not None:
            raise AssertionError("Mode should never be enabled for CPU/CUDA key")
        return subgraph(*operands)

    def _call_ProxyTorchDispatchMode(self, proxy_mode, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] HELPER: Proxy tracing implementation. Traces the subgraph into a new
        # graph module, registers it on the tracer root under a fresh name starting with
        # subgraph (side effect), emits a call_function node of this hop with the traced
        # graph, operands and kwargs, runs the hop for real outputs, and returns the outputs
        # tracked to the node. Raises AssertionError if the tracer is not an FX tracer.
        traced_graph = reenter_make_fx(subgraph)(*operands)
        if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
            raise AssertionError(
                f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
            )
        qualname = proxy_mode.tracer.get_fresh_qualname("subgraph")
        proxy_mode.tracer.root.register_module(qualname, traced_graph)

        node_args = (traced_graph, *operands)
        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
        proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", self, proxy_args, proxy_kwargs
        )

        out = self(subgraph, *operands, **kwargs)
        return track_tensor_tree(
            out,
            out_proxy,
            constant=None,
            tracer=proxy_mode.tracer,  # type: ignore[arg-type]
        )

    def _fake_impl(self, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] HELPER: Fake tensor implementation: calls the subgraph on the operands.
        return subgraph(*operands)

    # NOTE [Support input mutation of hops]
    # To support input mutation, hop's subgraph must be functionalized because many inductor passes are
    #   applied to subgraph recursively and only work on functional graph. However, we could inline an
    #   epilogue graph (i.e. the copy_) into the subgraph because this is how input mutation
    #   is implemented in the top-level graph when no hop is presented. All passes must have been and will be
    #   aware of the epilogue graph.
    #
    # Since we've supported input mutation for custom op with auto_functionalized, we share the infra for hops
    # The plan is:
    #   1. In hop's Functionalization key, it calls do_auto_functionalize_v2 if subgraph mutates input
    #   2. In do_auto_functionalize_v2:
    #       a. we functionalize the callables in hop's argument. This is to make the subgraphs functional so we
    #          could recursively run passes on them. Also the epilogue graph is inlined at the end.
    #       b. we call auto_functionalized_v2 and pass in an additional schema in order to properly invoke
    #          the hop with normalized kwargs.
    #   3. In inductor, we decompose the auto_functionalized hop by calling into the dense implementation, which
    #      copies the mutated inputs to the hop if necessary and call the hop.
    # After these steps, the rest of the inductor stack knows how to fuse the copy_ in subgraph with other ops.
    def _call_Functionalize(self, ctx, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] HELPER: Functionalization implementation. Except for the
        # invoke_quant_packed hop, builds a hop instance and, if it can be auto-functionalized
        # (subgraph mutates inputs), returns the auto-functionalize v2 result. Otherwise unwraps
        # the operands, functionalizes the subgraph and re-invokes this hop below
        # functionalization, returning the re-wrapped outputs. Assumes no mutation/aliasing in
        # that fallback path.
        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize_v2,
        )

        # invoke_quant has non-proxable argument of type InvokeQuant that
        # we cannot generate schema for.
        if self is not torch.ops.higher_order.invoke_quant_packed:
            hop_instance = HopInstance.create(self, subgraph, *operands, **kwargs)
            if can_auto_functionalize(hop_instance):
                return do_auto_functionalize_v2(
                    ctx.mode, hop_instance, (subgraph, *operands), kwargs
                )

        unwrapped_operands = ctx.unwrap_tensors(operands)
        with ctx.redispatch_to_next():
            # We assume the subgraph doesn't mutate inputs and there is no aliasing.
            # In the PT2 stack, this is Dynamo's responsibility to figure out.
            functionalized_subgraph = FunctionWithNoFreeVars(
                ctx.functionalize(subgraph)
            )
            out = self(functionalized_subgraph, *unwrapped_operands, **kwargs)
        return ctx.wrap_tensors(out)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, subgraph, *operands, **kwargs):
        # [FUNCTION DOC] Builds a HopSchema for a call. Materializes the subgraph as a graph,
        # analyses input/output aliasing and mutated input indices, and emits a warning (not
        # an error) if any aliasing exists. The schema has the subgraph argument, one
        # positional argument per operand (marked mutated when mutated), keyword-only
        # arguments for kwargs (defaulting to their given values), and the graph outputs.
        from .schema import HopSchemaGenerator

        subgraph = materialize_as_graph(subgraph, operands)
        (
            inp_inp_alias,
            inp_out_alias,
            out_out_alias,
            mutated_inp_idx,
            output,
        ) = check_input_alias_and_mutation_return_outputs(subgraph)

        if not (
            len(inp_inp_alias) == 0
            and len(inp_out_alias) == 0
            and len(out_out_alias) == 0
        ):
            # TODO: turn this into an error.
            # test_foreach_map_backward_binary_foreach_map_addrecip_op fails the alias test.
            import warnings

            warnings.warn(
                "Aliasing is not supported for HOP subgraph.\n"
                f"{subgraph.print_readable(print_output=False)}\n"
                f"Alias info: inp-inp alias: {inp_inp_alias}, inp-out alias: {inp_out_alias}, out-out alias{out_out_alias}"
                f"This may lead to silent incorrectness.",
                stacklevel=2,
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("subgraph", subgraph)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"arg{idx}", arg, is_mutated=idx in mutated_inp_idx)

        for name, arg in kwargs.items():
            schema_gen.add_arg(name, arg, default_value=arg, kw_only=True)

        for out in output:
            schema_gen.add_output(out)

        return schema_gen.gen_schema()


class BaseHOPFunction(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function used by BaseHOP. Forward stores the hop, subgraph,
    # kwargs and operands on the context and runs the hop below autograd; backward builds a
    # joint forward/backward graph for the subgraph and calls the same hop on it.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, hop, subgraph, kwargs, *operands):
        # [FUNCTION DOC] HELPER: Saves hop, operands, subgraph and kwargs on ctx (plain
        # attributes), then calls the hop with autograd disabled below this key; returns its outputs.
        ctx.hop = hop
        ctx.operands = operands
        ctx.subgraph = subgraph
        ctx.kwargs = kwargs

        with torch._C._AutoDispatchBelowAutograd():
            return hop(subgraph, *operands, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # [FUNCTION DOC] HELPER: Builds, with functionalization and proxy tracing disabled, a
        # joint forward/backward graph of the saved subgraph from the operands and incoming
        # gradients, wraps a function that runs it and keeps only the input gradients, and
        # calls the saved hop on that function with operands and grad outputs. Returns None for
        # the hop, subgraph and kwargs inputs followed by the operand gradients.
        subgraph = ctx.subgraph
        operands = ctx.operands
        kwargs = ctx.kwargs

        # TODO: Something special needs to happen with min cut partitioner
        with (
            suspend_functionalization(),
            disable_functional_mode(),
            torch.enable_grad(),
        ):
            with disable_proxy_modes_tracing():
                from .invoke_subgraph import create_fw_bw_graph
                from .utils import _from_fun

                fw_inputs = pytree.tree_map(_from_fun, operands)
                (
                    _,
                    joint_graph,
                    _,
                ) = create_fw_bw_graph(subgraph, fw_inputs, grad_outputs)

        # The joint graph returns (*grad_inputs, *fwd_outputs).
        # We only need the grad_inputs.
        def bwd_fn(*args):
            # [FUNCTION DOC] HELPER: Splits args into operands and grad outputs (by the count
            # of grad_outputs), runs the joint graph, and returns the result minus its trailing
            # forward outputs, i.e. the input gradients.
            operands = args[: -len(grad_outputs)]
            grad_outs = args[-len(grad_outputs) :]
            result = joint_graph(*operands, *grad_outs)
            grad_inputs = result[: -len(grad_outputs)]
            return grad_inputs

        return (
            None,
            None,
            None,
            *ctx.hop(
                FunctionWithNoFreeVars(bwd_fn), *operands, *grad_outputs, **kwargs
            ),
        )


class FunctionWithNoFreeVars:
    # [CLASS DOC] Thin wrapper marking a callable as having no free variables, so BaseHOP
    # accepts it as a subgraph. Holds the callable in attribute fn.
    def __init__(self, fn):
        # [FUNCTION DOC] Stores the wrapped callable.
        self.fn = fn

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] Calls the wrapped callable with the arguments and returns the result.
        return self.fn(*args, **kwargs)
