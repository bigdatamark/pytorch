# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the map higher order operator: apply a function to each slice along
# the first dimension of the mapped tensors (plus shared extra arguments) and
# stack the results. Contains the public map function, the map_impl operator,
# its autograd Function, and dense / proxy-tracing / fake-tensor / functionalize
# implementations, plus a reference eager implementation (_fake_map).
# Import-time side effects: creates the singleton map_impl and registers its
# dispatch implementations (dense, autograd, proxy, fake, functionalize).
import functools
from collections.abc import Callable, Sequence
from typing import Any
from typing_extensions import TypeVarTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.auto_functionalize import (
    can_auto_functionalize,
    do_auto_functionalize_v2,
)
from torch._higher_order_ops.utils import (
    _from_fun,
    _maybe_run_with_interpreter,
    _stack_pytree,
    _unstack_pytree,
    create_bw_fn,
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
    get_graph_output_example_values,
    HopInstance,
    materialize_as_graph,
    reenter_make_fx,
    register_fake,
    save_values_for_backward,
    saved_values,
    split_into_chunks,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class MapImpl(HigherOrderOperator):
    # [CLASS DOC] HigherOrderOperator "map_impl" taking a body function f, a sequence
    # of mapped tensors xs, additional positional args, and an optional keyword
    # string of comma-separated indices of mutated xs. Stateless.
    def __init__(self):
        # [FUNCTION DOC] Initializes the operator under the name "map_impl".
        super().__init__("map_impl")

    def __call__(
        self,
        f: Callable[..., Any],
        xs: Sequence[torch.Tensor],
        pos_args: Sequence[torch.fx.node.BaseArgumentTypes],
        *,
        mutated_arg_indices: str = "",
    ) -> Any:
        # [FUNCTION DOC] Dispatches the map: forwards f, xs, pos_args and the
        # mutated_arg_indices keyword to the HOP dispatcher and returns the
        # stacked outputs.
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            f, xs, pos_args, mutated_arg_indices=mutated_arg_indices
        )

    # pyrefly: ignore [bad-override]
    def gen_schema(self, f, xs, pos_args, mutated_arg_indices: str = ""):
        # [FUNCTION DOC] Builds the HOP function schema. Creates example inputs
        # (per-slice empty tensors with the stride and dtype of each mapped x, plus
        # the extra args), traces f into a graph, and declares: the body graph arg,
        # one arg per xs (marked mutated when its index is listed in
        # mutated_arg_indices, a comma-separated string), one per additional input
        # (never mutable, since sharing a mutable buffer would break iteration
        # independence), the body's outputs, and the pytree spec. Returns the schema.
        from torch._higher_order_ops.schema import HopSchemaGenerator

        all_inputs = tuple(
            [
                torch.empty_strided(
                    x.shape[1:],
                    x.stride()[1:],
                    dtype=x.dtype,
                    device=x.device,
                    requires_grad=x.requires_grad,
                )
                for x in xs
            ]
            + list(pos_args)
        )
        body_gm = materialize_as_graph(f, all_inputs)

        # Mutation semantics for map:
        # - xs is mutable: each iteration sees a storage-disjoint slice
        #   (xs[t] and xs[t+1] share no storage), so in-place writes are
        #   race-free regardless of evaluation order, preserving map's
        #   "iterations are independent" contract.
        # - pos_args is NOT mutable: every iteration sees the same
        #   tensor, so mutating it makes iterations depend on each
        #   other, breaking the independence contract and introducing a
        #   data race under any parallel lowering. Users who need a
        #   shared mutable buffer should use scan / while_loop, where
        #   sequential iteration is part of the contract.
        outputs = get_graph_output_example_values(body_gm)
        mutated_set = {int(i) for i in mutated_arg_indices.split(",") if i}

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("f", body_gm)

        for idx, x in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", x, is_mutated=idx in mutated_set)

        for idx, arg in enumerate(pos_args):
            schema_gen.add_arg(f"additional_input{idx}", arg)

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(f, xs, pos_args)
        return schema_gen.gen_schema()


map_impl = MapImpl()


def map(
    f: Callable[[pytree.PyTree, tuple[pytree.PyTree, ...]], pytree.PyTree],
    xs: pytree.PyTree | torch.Tensor,
    *args: TypeVarTuple,
):
    # [FUNCTION DOC] Public map API. Flattens xs and args with pytree, requires
    # every leaf of xs to be a tensor with the same non-zero leading dimension,
    # then calls map_impl on a wrapper that unflattens the leaves and invokes f
    # with the xs slice followed by args, optionally compiling it via the shared
    # compile-and-run helper. Returns the stacked outputs. Raises RuntimeError if
    # xs has non-tensor leaves, a zero leading dimension, or inconsistent
    # leading dimensions.
    r"""
    Performs a map of f with xs. Intuitively, you can think of the semantic being::

        out = []
        for idx in len(xs.size(0)):
            xs_sliced = xs.select(0, idx)
            out.append(f(xs_sliced, *args))
        torch.stack(out)

    .. warning::

        ``torch._higher_order_ops.map`` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype


    Args:
        f (Callable): a callable that takes an input x, that could either be a single Tensor
            or a nested dict, list of tensors and some additional inputs
        xs: the inputs that're to be mapped over. We'll iterate over the first dim of each x
            and perform f on each slice.

        *args: additional arguments provided to each step of f. They could also be omitted and
            map is able to automatically figure out the read dependency.

    Return:
        the stacked output for each step of f

    Example::

        def f(xs):
            return xs[0] + xs[1] + const1 + const2


        xs = [torch.randn(2, 3), torch.randn(2, 3)]
        const1 = torch.randn(2, 3)
        const2 = torch.randn(2, 3)
        # returns a tensor of shape [2, 2, 3]
        torch._higher_order_ops.map(f, xs)

    """
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    flat_args, args_spec = pytree.tree_flatten(args)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    def run_flattened_map(f, flat_xs, flat_args):
        # [FUNCTION DOC] HELPER: Binds f and the tree specs into a flat-argument
        # function and runs map_impl on the flat xs and flat extra args.
        def wrapped_fn(*flat_args, f, xs_tree_spec, args_tree_spec, num_xs):
            # [FUNCTION DOC] HELPER: Rebuilds xs (first num_xs leaves) and the extra
            # args from flat leaves using the saved specs and calls f with them.
            xs =pytree.tree_unflatten(flat_args[:num_xs], xs_tree_spec)
            args = pytree.tree_unflatten(flat_args[num_xs:], args_tree_spec)
            return f(xs, *args)

        inner_f = functools.partial(
            wrapped_fn,
            f=f,
            xs_tree_spec=xs_spec,
            args_tree_spec=args_spec,
            num_xs=len(flat_xs),
        )
        return map_impl(inner_f, flat_xs, flat_args)

    from torch._higher_order_ops.utils import _maybe_compile_and_run_fn

    return _maybe_compile_and_run_fn(run_flattened_map, f, flat_xs, flat_args)


class MapAutogradOp(torch.autograd.Function):
    # [CLASS DOC] Autograd Function for map_impl. Forward runs map_impl below
    # autograd and saves inputs plus the body function, counts and dispatch key
    # sets on ctx; backward traces a per-slice backward graph and runs it via
    # map_impl over the slices.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, f, num_mapped_args, *flat_args):
        # [FUNCTION DOC] Saves f, number of mapped and positional args, the current
        # dispatch TLS include/exclude key sets, and the inputs on ctx, then runs
        # map_impl with autograd disabled. Args: f body, num_mapped_args, flat_args
        # (mapped tensors followed by positional args). Returns a tuple of outputs.
        ctx._f = f
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args) - num_mapped_args

        # We snapshot the dispatch keys in forward for materializing
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_values_for_backward(ctx, flat_args)
        with torch._C._AutoDispatchBelowAutograd():
            return (
                *map_impl(f, flat_args[:num_mapped_args], flat_args[num_mapped_args:]),
            )

    @staticmethod
    def backward(ctx, *flat_grads):
        # [FUNCTION DOC] Computes gradients. Splits saved forward args into mapped and
        # positional ones, builds a per-slice backward function from f, wraps it so
        # arguments are reordered (mapped, grads, pos) and non-tensor gradients are
        # recorded as masks, traces it to a graph using first-slice example inputs
        # (functionalization and proxy tracing suspended, forward key sets reused,
        # grad enabled), and runs map_impl with that graph over mapped args and
        # gradients. Returns None for f and num_mapped_args followed by input
        # gradients with None restored via the masks.
        # KNOWN ISSUE: positional-arg gradients come out stacked per slice by
        # map_impl rather than reduced across iterations.
        fw_args = saved_values(ctx)
        num_mapped_args = ctx._num_mapped_args
        num_pos_args = ctx._num_pos_args
        num_grads = len(flat_grads)

        fw_mapped_args, pos_args = split_into_chunks(
            fw_args,
            [
                num_mapped_args,
                num_pos_args,
            ],
        )

        bw_f = create_bw_fn(ctx._f, fw_args)

        grads_tensor_masks = []

        # Create a wrapper around thefor the bw_f
        def bw_f_wrapper(*args):
            # [FUNCTION DOC] HELPER: Backward body for one slice. Takes mapped args,
            # upstream grads and pos args, reorders them to primals then tangents,
            # runs bw_f, records which gradient outputs are tensors in the
            # enclosing grads_tensor_masks, and returns only the masked-in grads.
            nonlocal grads_tensor_masks

            # Dissect args and re-order them for the ``ctx._bw_f``
            # args provided to the wrapper are composed of [*fw_mapped_args, *flat_grads, *pos_args]
            # The content of ``bw_f_tangents`` are the upstream gradients, i.e. flat_grads
            # The content of ``bw_f_primals`` are the fw_args, i.e., [*fw_mapped_args, *pos_args]
            # The bw_f requires *bw_f_primals, *bw_f_tangents
            fw_m_args, bw_f_tangents, pos_args = split_into_chunks(
                args, [num_mapped_args, num_grads, num_pos_args]
            )
            bw_f_primals = *fw_m_args, *pos_args
            gradients = bw_f(*bw_f_primals, *bw_f_tangents)
            grads_tensor_masks = [
                True if isinstance(out, torch.Tensor) else out for out in gradients
            ]
            return filter_with_masks(gradients, grads_tensor_masks)

        def construct_args_single_step_bw():
            # [FUNCTION DOC] HELPER: Builds example inputs for tracing one backward
            # step: first-slice copies of mapped args and of gradients (keeping
            # their requires_grad), and the positional args unwrapped from
            # functional tensors. Returns them as one tuple in the same order.
            unwrapped_mapped_xs = pytree.tree_map(_from_fun, fw_mapped_args)
            # Use first_slice_copy instead of _unstack_pytree to avoid
            # iterating over batch dim, which would guard on symbolic sizes.
            example_xs = [
                first_slice_copy(x).requires_grad_(x.requires_grad)
                for x in unwrapped_mapped_xs
            ]
            unwrapped_grads = pytree.tree_map(_from_fun, flat_grads)
            example_grads = [
                first_slice_copy(x).requires_grad_(x.requires_grad)
                for x in unwrapped_grads
            ]
            example_pos_args = [
                _from_fun(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in pos_args
            ]
            return *example_xs, *example_grads, *example_pos_args

        with suspend_functionalization(), disable_functional_mode():
            with disable_proxy_modes_tracing():
                args_single_step_bw = construct_args_single_step_bw()

            # TODO: we need to materialize the bw graphs because dynamo is unable to
            # trace through the joint function when torch.compile torch.autograd.grad.
            fn_bw_gm = materialize_as_graph(
                bw_f_wrapper,
                args_single_step_bw,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )

        grads = map_impl(fn_bw_gm, fw_mapped_args + flat_grads, pos_args)

        return None, None, *fill_none_with_masks(grads, grads_tensor_masks)


def _broadcast_to_batch(output, batch_size):
    # [FUNCTION DOC] HELPER: For each tensor in the output pytree, adds a leading
    # dimension of batch_size by expanding and cloning into contiguous memory;
    # non-tensor leaves are returned unchanged. Returns the same pytree structure.
    """Expand each tensor in output pytree to include batch dimension.

    Note: Although outputs are flattened in the compiled path (via torch.compile),
    users may call map_impl directly with pytree outputs, so we support pytrees
    for backward compatibility.
    """

    def expand_with_batch(t):
        # [FUNCTION DOC] HELPER: Leaf transform for tensors described above.
        if isinstance(t, torch.Tensor):
            # Use contiguous_format to match torch.stack behavior
            return (
                t.unsqueeze(0)
                .expand(batch_size, *t.shape)
                .clone(memory_format=torch.contiguous_format)
            )
        return t

    return pytree.tree_map(expand_with_batch, output)


def trace_map(
    proxy_mode,
    func_overload: HigherOrderOperator,
    f: Callable[..., Any],
    xs: Sequence[torch.Tensor],
    pos_args: Sequence[torch.fx.node.BaseArgumentTypes],
    mutated_arg_indices: str = "",
) -> Any:
    # [FUNCTION DOC] Proxy-tracing helper. With proxy tracing disabled, traces f on
    # first-slice example inputs plus pos_args into a body graph; registers that
    # graph as a fresh submodule (body_graph_N) on the tracer root; calls map_impl
    # on real/fake inputs to get outputs; creates a map_impl call_function node
    # (including the mutated_arg_indices keyword only when non-empty) and tracks
    # the outputs to that proxy. Returns the tracked outputs. Side effects:
    # registers a module and adds a node to the graph.
    with disable_proxy_modes_tracing():
        # Use first_slice_copy instead of _unstack_pytree to avoid
        # iterating over batch dim, which would guard on symbolic sizes.
        example_input =pytree.tree_map(first_slice_copy, xs)

        body_graph = reenter_make_fx(f)(*example_input, *pos_args)

    next_name = proxy_mode.tracer.get_fresh_qualname("body_graph_")
    proxy_mode.tracer.root.register_module(next_name, body_graph)

    fake_outs = map_impl(
        body_graph, xs, pos_args, mutated_arg_indices=mutated_arg_indices
    )

    node_args = (body_graph, list(xs), list(pos_args))
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        proxy_args,
        {"mutated_arg_indices": mutated_arg_indices} if mutated_arg_indices else {},
        name="map_impl",
    )
    return track_tensor_tree(
        fake_outs, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, xs, pos_args, mutated_arg_indices=""):
    # [FUNCTION DOC] Eager implementation: unstacks xs along the first dimension,
    # calls f for each slice with pos_args, and stacks the per-slice output
    # pytrees. Returns the stacked pytree.
    # KNOWN ISSUE: mutated_arg_indices is ignored; mutations simply happen in f.
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


@map_impl.py_autograd_impl
def map_autograd(f, xs, pos_args, mutated_arg_indices=""):
    # [FUNCTION DOC] Autograd-key implementation: applies MapAutogradOp to f, the
    # number of mapped args, xs and pos_args and returns its outputs.
    # KNOWN ISSUE: mutated_arg_indices is dropped and not forwarded.
    num_mapped_args = len(xs)
    flat_out = MapAutogradOp.apply(f, num_mapped_args, *xs, *pos_args)
    return flat_out


@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(mode, f, xs, pos_args, mutated_arg_indices=""):
    # [FUNCTION DOC] HELPER: Proxy-mode implementation; delegates to trace_map.
    return trace_map(
        mode, map_impl, f, xs, pos_args, mutated_arg_indices=mutated_arg_indices
    )


@register_fake(map_impl, skip_cache=True)
def map_fake_tensor_mode(f, xs, pos_args, mutated_arg_indices=""):
    # [FUNCTION DOC] Fake-tensor implementation (not cached): runs f on the first
    # slice of each x with pos_args and expands every output tensor with a leading
    # dimension equal to the batch size of the first flattened x. Returns the
    # resulting pytree.
    # Use first_slice_copy instead of _unstack_pytree to avoid
    # iterating over batch dim, which would guard on symbolic sizes.
    first_row = pytree.tree_map(first_slice_copy, xs)
    example_output = f(*first_row, *pos_args)

    flat_xs, _ = pytree.tree_flatten(xs)
    batch_size = flat_xs[0].shape[0]

    return _broadcast_to_batch(example_output, batch_size)


@map_impl.py_functionalize_impl
def map_functionalize(ctx, f, xs, pos_args, mutated_arg_indices=""):
    # [FUNCTION DOC] Functionalization implementation. If the context has a mode and
    # the map call can be auto-functionalized, delegates to the v2 auto
    # functionalization with flattened inputs. Otherwise unwraps tensors,
    # functionalizes f, checks (on first-slice example inputs) that f has no
    # disallowed aliasing or mutation, then redispatches map_impl and wraps the
    # result. Raises via the alias/mutation check when f violates it.
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    if hasattr(ctx, "mode"):
        hop_instance = HopInstance.create(
            map_impl,
            f,
            xs,
            pos_args,
            mutated_arg_indices=mutated_arg_indices,
        )
        if can_auto_functionalize(hop_instance):
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                tuple(pytree.tree_flatten((f, xs, pos_args))[0]),
                {},
            )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_args = ctx.unwrap_tensors(pos_args)
    wrapped_fn = ctx.functionalize(_maybe_run_with_interpreter(f))

    with ctx.redispatch_to_next():
        # Use first_slice_copy instead of _unstack_pytree to avoid
        # iterating over batch dim, which would guard on symbolic sizes.
        example_inputs = (
            *pytree.tree_map(first_slice_copy, unwrapped_xs),
            *unwrapped_args,
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(f, example_inputs, "map", pre_dispatch)
        map_return = map_impl(
            wrapped_fn,
            unwrapped_xs,
            unwrapped_args,
            mutated_arg_indices=mutated_arg_indices,
        )
        return ctx.wrap_tensors(map_return)


def _fake_map(f, x, *args):
    # [FUNCTION DOC] HELPER: Reference eager map using the control-flow helpers:
    # unstacks x, calls f on each slice with args, and stacks the results.
    from functorch.experimental.control_flow import _stack_pytree, _unstack_pytree

    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)
