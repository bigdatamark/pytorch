# mypy: allow-untyped-defs
# [MODULE DOC] Implements torch.scan (inclusive scan with carry) as the higher-order op scan_op.
# Contains the public scan function, helpers for flattening/stacking, a dense loop implementation
# (generic_scan), tracing (trace_scan), a partitioner-based autograd implementation
# (ScanAutogradOp / ScanAutogradImpl with intermediate handling policies), and proxy, fake tensor,
# functionalize and vmap rules, plus a test-only eager reference (_fake_scan).
# Import-time side effects: creates the scan_op singleton and registers all implementations.
import enum
import functools
import itertools
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.vmap import restore_vmap, unwrap_batched, wrap_batched
from torch._guards import detect_fake_mode
from torch._higher_order_ops.auto_functionalize import (
    can_auto_functionalize,
    do_auto_functionalize_v2,
)
from torch._higher_order_ops.partitioner import (
    _find_hop_subgraph_outputs,
    HopGraphMinCutPartitioner,
    HopPartitionedGraph,
)
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    check_meta_consistency,
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
    get_graph_output_example_values,
    get_tensor_mask,
    HopInstance,
    mask_list,
    materialize_as_graph,
    reenter_make_fx,
    split_into_chunks,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


logger: logging.Logger = logging.getLogger(__name__)
aten = torch._ops.ops.aten


def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    # [FUNCTION DOC] HELPER: Adapter turning a pytree-based combine_fn into one taking flat leaves.
    # Expects init leaves followed by xs leaves (AssertionError if the count is wrong), rebuilds
    # the carry and xs structures from their specs and returns combine_fn(carry, xs).
    if len(args) != (num_init_leaves + num_inp_leaves):
        raise AssertionError(
            f"combine_fn received wrong number of arguments, expected {num_init_leaves + num_inp_leaves}, but got {len(args)}"
        )
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    xs = pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
    return combine_fn(carry, xs)


def _extract_carry_and_out(flat_out: list[Any], num_carry: int):
    # [FUNCTION DOC] HELPER: Splits a flat output list into the first num_carry carry entries and
    # the remaining per-step outputs; returns the two chunks.
    return split_into_chunks(flat_out, [num_carry, len(flat_out) - num_carry])


# We also do a clone with contiguous_format. This is to be consistent with
# eager semantic of scan, which stacks the outputs. The result is contiguous
# as a result of the stack operation.
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor:
    # [FUNCTION DOC] Returns a contiguous tensor with y repeated scan_length times along a new
    # leading dimension; used to build fake/meta shapes of stacked scan outputs.
    return (
        y.unsqueeze(0)
        .repeat(*([scan_length] + [1] * y.ndim))
        .clone(memory_format=torch.contiguous_format)
    )


def call_operator(operator, *args):
    # [FUNCTION DOC] HELPER: Calls operator with args and returns the flattened list of leaves of
    # its result.
    return pytree.tree_leaves(operator(*args))


def _build_empty_output_for_length_zero(
    combine_fn: Callable, init: pytree.PyTree
) -> pytree.PyTree:
    # [FUNCTION DOC] HELPER: Builds the per-step output structure for a zero-length scan. Runs
    # combine_fn once on fake copies of init and a None x (under a detected or new fake mode) to
    # learn the output pytree, then replaces each tensor leaf by an empty tensor with a new
    # leading dimension of size 0 (same dtype/device); None leaves pass through; any other leaf
    # raises AssertionError.
    """Probe combine_fn(init, None) to learn the output structure for length==0."""
    fake_mode = detect_fake_mode(pytree.tree_leaves(init)) or FakeTensorMode()
    with fake_mode:
        fake_init = pytree.tree_map(
            lambda t: fake_mode.from_tensor(t)
            if isinstance(t, torch.Tensor) and not is_fake_tensor(t)
            else t,
            init,
        )
        _, sample_y = combine_fn(fake_init, None)

    def _empty_or_passthrough(l):
        # [FUNCTION DOC] HELPER: Maps a tensor leaf to an empty tensor with a zero-size leading
        # dim; returns None unchanged; raises AssertionError for other types.
        if isinstance(l, torch.Tensor):
            return torch.empty([0] + list(l.shape), dtype=l.dtype, device=l.device)
        if l is not None:
            raise AssertionError(f"Expected leaf to be a Tensor or None, got {type(l)}")
        return l

    return pytree.tree_map(_empty_or_passthrough, sample_y)


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
    length: int | None = None,
) -> tuple[pytree.PyTree, pytree.PyTree]:
    # [FUNCTION DOC] Public torch.scan. Flattens init and xs; with an explicit length and no xs
    # tensors it iterates length times using a fabricated counter input (combine_fn receives None
    # as x), returns (init, empty outputs) for length 0 in eager mode (RuntimeError under
    # compile), and without length and tensors returns (init, empty list). Validates arguments
    # (RuntimeError on a non-callable combine_fn, non-int dim, non-bool reverse, bad length,
    # length mismatch, empty or non-tensor init leaves, xs leaves of low rank or differing scan
    # size). Moves the scan dim to the front, optionally flips for reverse, runs scan_op on flat
    # leaves via the compile-and-run helper, then undoes the flip and dim move on the stacked
    # outputs. Returns (final carry, stacked outputs).
    r"""
    Performs an inclusive scan with a combine function.

    .. warning::

        ``torch.scan`` is a prototype feature in PyTorch. You may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> (Tensor, Tensor)``,
            or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor or pytree with tensor leaves): The initial scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
            of ``combine_fn``.
        xs (torch.Tensor or pytree with tensor leaves or None): The input tensor, or nested pytree of tensors.
            May be ``None`` when ``length`` is provided, in which case ``combine_fn`` receives ``None`` as ``x``
            each step (counter-loop mode).

    Keyword Args:
        dim (int): the dimension to scan over, default 0.
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        length (int or None): Optional number of scan iterations, default ``None``.
            When ``xs`` has tensor leaves, ``length`` is optional; if given it must equal
            ``xs.shape[dim]`` and serves only as a consistency check (no constraint when
            ``length`` is ``None``). When ``xs`` has no leaves (``None`` or empty pytree),
            ``length`` drives the number of iterations and ``combine_fn`` receives
            ``x=None`` each step. ``length=0`` with no xs tensors is supported in
            eager mode only; it is not supported under ``torch.compile``.

    Returns:
        final_carry (torch.Tensor or pytree with tensor leaves),
            the final carry of the scan operation with same pytree structure as init.
        out (torch.Tensor or pytree with tensor leaves),
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.
            If the scan dimension has size 0, ``final_carry`` equals ``init`` unchanged and each output leaf has
            size 0 along ``dim``. The gradient of ``final_carry`` with respect to ``init`` is the identity
            (not zero), since the body is never called and the carry passes through untouched.

    Restrictions:
        - The combine_fn shouldn't have any aliasing between input-input, input-output, and output-output. E.g. return a view
            or the same tensor as input is not supported. As a workaround, can clone the output to avoid aliasing.

        - The combine_fn shouldn't mutate any inputs. We'll remove the mutation restriction for inference soon. Please file an issue
            if you input mutation support for training is needed.

        - The combine_fn's init carry should match the next_carry in pytree structure and in tensor metadata.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            # clone the output to avoid output-output aliasing
            return next_carry, y.clone()


        i0 = torch.zeros(1)
        xs = torch.arange(5)
        # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    # The reason we flatten init and xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    # Determine whether xs carries any tensor data.
    xs_has_tensors = any(isinstance(l, torch.Tensor) for l in leaves_xs_orig)

    # short-cuts
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise RuntimeError(
                f"scan() length must be a non-negative integer, got {length!r}"
            )

        if not xs_has_tensors:
            if length == 0:
                if torch.compiler.is_dynamo_compiling():
                    # TODO: Resolve this in a follow-up PR
                    raise RuntimeError(
                        "scan() with length=0 and no xs tensors is not supported under torch.compile"
                    )
                return init, _build_empty_output_for_length_zero(combine_fn, init)

            # No real xs: fabricate a length-N dummy purely as an iteration counter.
            # the wrapped combine_fn below discards each slice and passes x=None to the body.
            leaves_xs_orig = [torch.zeros(length, dtype=torch.int64)]
            spec_xs = pytree.tree_structure(None)
            _user_combine_fn = combine_fn

            def combine_fn(carry, _ignored):  # noqa: E306
                # [FUNCTION DOC] HELPER: Wrapper that ignores the dummy counter slice and calls
                # the user combine function with None as x.
                return _user_combine_fn(carry, None)
    elif not xs_has_tensors:
        return init, []

    def _validate_input(cfn, lxs, linit, d, r, l):
        # [FUNCTION DOC] HELPER: Raises RuntimeError unless combine_fn is callable, dim is an int,
        # reverse is a bool, length (if given) matches xs size along dim, init is non-empty with
        # tensor leaves, and xs leaves are tensors with more than dim dimensions and equal size
        # along the scan dim.
        # Basic arguments check
        if not callable(cfn):
            raise RuntimeError(f"Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise RuntimeError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))

        if l is not None and xs_has_tensors and lxs[0].shape[d] != l:
            raise RuntimeError(
                f"scan() length={l} does not match xs size along dim={d}: "
                f"{lxs[0].shape[d]}"
            )

        # Checks for init
        if len(linit) == 0:
            raise RuntimeError("scan() operator requires init leaves.")
        for x in linit:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All init leaves must be a Tensor but got {x}")

        # Checks for xs
        for x in lxs:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All xs leaves must be a Tensor but got {x}")
        if any(x.ndim <= d for x in lxs):
            raise RuntimeError("All xs leaves must have at least 'dim + 1' dimensions")
        if any(x.shape[d] != lxs[0].shape[d] for x in lxs[1:]):
            raise RuntimeError("All xs leaves must have the same scan dimension size")

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, leaves_init, dim, reverse, length)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = []
    for elem in leaves_xs_orig:
        leaves_xs.append(torch.movedim(elem, dim, 0) if dim != 0 else elem)

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    # TODO: Support _inductor lowering
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

    combine_fn = functools.partial(
        wrap_combine_fn_flat,
        combine_fn=combine_fn,
        spec_init=spec_init,
        spec_xs=spec_xs,
        num_init_leaves=len(leaves_init),
        num_inp_leaves=len(leaves_xs),
    )

    def run_flattened_scan(combine_fn, leaves_init, leaves_xs):
        # [FUNCTION DOC] HELPER: Invokes scan_op on flat leaves with no additional inputs.
        return scan_op(combine_fn, leaves_init, leaves_xs, ())

    carry, out = _maybe_compile_and_run_fn(
        run_flattened_scan,
        combine_fn,
        leaves_init,
        leaves_xs,
    )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    # Move the scan dimension from 0 back to the user-specified `dim`.
    if dim != 0:
        out = pytree.tree_map(
            lambda elem: torch.movedim(elem, 0, dim) if dim < elem.ndim else elem,
            out,
        )

    return carry, out


class ScanOp(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator "scan" taking combine_fn, init leaves, xs leaves and
    # additional (loop-invariant) inputs, with optional mutated_arg_indices. Stateless; singleton
    # instance is scan_op.
    def __init__(self):
        # [FUNCTION DOC] Registers the operator under the name "scan".
        super().__init__("scan")

    def __call__(
        self,
        combine_fn,
        init,
        xs,
        additional_inputs,
        *,
        mutated_arg_indices: str = "",
    ):
        # [FUNCTION DOC] Validates that additional_inputs is a tuple or list (AssertionError
        # otherwise; a list is converted to a tuple), checks subgraph argument types, and forwards
        # to the generic dispatch, passing mutated_arg_indices (a comma separated string of
        # indices) only when non-empty.
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        if not isinstance(additional_inputs, (tuple, list)):
            raise AssertionError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        additional_inputs = (
            tuple(additional_inputs)
            if isinstance(additional_inputs, list)
            else additional_inputs
        )
        validate_subgraph_args_types(additional_inputs)
        kwargs = {}
        if mutated_arg_indices:
            kwargs["mutated_arg_indices"] = mutated_arg_indices
        # pyrefly: ignore [missing-attribute]
        return super().__call__(combine_fn, init, xs, additional_inputs, **kwargs)

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
    ):
        # [FUNCTION DOC] Builds the op's FunctionSchema for a concrete call. Materializes
        # combine_fn as a graph on sample inputs (init, one slice per xs, additional inputs), adds
        # args named init<i>, xs<i>, additional_input<i> (only additional inputs listed in
        # mutated_arg_indices are marked mutated; init and xs never are), adds the graph outputs,
        # registers the original structure as the tree spec and returns the schema.
        from torch._higher_order_ops.schema import HopSchemaGenerator

        all_inputs = tuple(
            list(init)
            + [
                torch.empty_strided(
                    x.shape[1:],
                    x.stride()[1:],
                    dtype=x.dtype,
                    device=x.device,
                    requires_grad=x.requires_grad,
                )
                for x in xs
            ]
            + list(additional_inputs)
        )
        combine_gm = materialize_as_graph(combine_fn, all_inputs)

        # Mutation semantics for scan:
        # - additional_inputs is mutable: loop-invariant tensor identity
        #   across sequential iterations, same semantics as while_loop's
        #   additional_inputs (CUDA-graph-friendly lifted / pre-allocated
        #   buffers such as KV caches or workspace scratch).
        # - init is NOT mutable: init is only the *initial* carry,
        #   thus an in-place update to init only affects step 0.
        # - xs is NOT mutable: each iteration sees a fresh, storage-disjoint
        #   slice (xs[t] and xs[t+1] share no storage), so a mutation on
        #   xs[t] cannot be observed by iteration t+1. The only externally-
        #   observable effect is "write-back to xs's t-th slice", which is
        #   already expressible via the ys output path at no extra cost.
        #   If xs-like in-place updates are required, pass the buffer via
        #   additional_inputs and index into it inside combine_fn.
        outputs = get_graph_output_example_values(combine_gm)
        mutated_set = (
            {int(i) for i in mutated_arg_indices.split(",") if i}
            if mutated_arg_indices
            else set()
        )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("combine_fn", combine_gm)

        for idx, arg in enumerate(init):
            schema_gen.add_arg(f"init{idx}", arg)

        for idx, arg in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", arg)

        offset = len(init) + len(xs)
        for idx, arg in enumerate(additional_inputs):
            schema_gen.add_arg(
                f"additional_input{idx}",
                arg,
                is_mutated=(offset + idx) in mutated_set,
            )

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(combine_fn, init, xs, additional_inputs)
        return schema_gen.gen_schema()


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, additional_inputs=()):
    # [FUNCTION DOC] Dense eager scan over flat tensor lists. Applies operator step by step along
    # dim, feeding the carry, the current slice of each xs and the additional inputs; returns the
    # final carry followed by outputs stacked along a new leading dimension (None for non-tensor
    # outputs). Output shapes are inferred from the first step's real result. See _scan.
    def _scan(init, xs):
        # [FUNCTION DOC] HELPER: Runs the loop. With empty xs returns (carry, empty list). Runs
        # step 0 to learn output shapes, preallocates output buffers sized by the number of
        # elements, writes each step's tensor outputs into its slot via scatter, and returns a flat
        # tuple of final carry then expanded outputs. If there are zero elements it returns init
        # plus the empty buffers.
        # KNOWN ISSUE: with zero elements xs[0] has size 0 yet step 0 is still run first, so the
        # operator is called on empty slices.
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        num_init_leaves = len(init)

        proto_xs = [first_slice_copy(x, dim) for x in xs]

        # Process element 0 to infer output shapes for pre-allocation
        # AND produce the first real result in a single call.  The previous
        # approach used first_slice_copy() for shape inference and then
        # re-processed element 0 in the main loop, calling the operator
        # num_elems+1 times.  That extra invocation is incorrect for
        # operators with side effects.
        carry, out_0 = _extract_carry_and_out(
            call_operator(
                operator,
                *carry,
                *proto_xs,
                *additional_inputs,
            ),
            num_init_leaves,
        )

        out_tensor_mask = get_tensor_mask(out_0)
        out_0_masked = mask_list(out_tensor_mask, out_0)

        # Pre-allocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        # out: (num_elems, M, N, ...)
        # idx: (1, M, N)
        outs = [
            torch.empty(
                [num_elems] + list(e.size()),
                dtype=e.dtype,
                device=e.device,
            )
            for e in out_0_masked
        ]
        idxs = [
            torch.ones_like(e, dtype=torch.int64).unsqueeze(0) for e in out_0_masked
        ]

        if num_elems == 0:
            outs_expanded = [
                outs.pop(0) if out_m else None for out_m in out_tensor_mask
            ]
            return (*init, *outs_expanded)

        def store_out_in_outs(out, ind):
            # [FUNCTION DOC] HELPER: Writes each tensor in out into slot ind of the matching
            # preallocated output buffer using scatter along dim 0.
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                # o: (num_elems, M, N ...)
                # x: (M, N, ...) -> (1, M, N)
                # ind * idx: (1, M, N,) with values to be ind
                # essentially: o[ind][n][k] = x[0][n][k]
                o.scatter_(0, ind * idx, x.unsqueeze(0))

        # Store element 0's result, then continue from element 1.
        store_out_in_outs(out_0_masked, 0)

        for i in range(1, num_elems):
            carry, out = _extract_carry_and_out(
                call_operator(
                    operator,
                    *carry,
                    *[elem.select(dim, i) for elem in xs],
                    *additional_inputs,
                ),
                num_init_leaves,
            )

            store_out_in_outs(mask_list(out_tensor_mask, out), i)

        # Expand outs with None depending on the tensor mask of the output
        outs_expanded = [outs.pop(0) if out_m else None for out_m in out_tensor_mask]

        return (*carry, *outs_expanded)

    scans = _scan(init, xs)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: list[torch.Tensor],
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
    mutated_arg_indices: str = "",
):
    # [FUNCTION DOC] Proxy tracing of scan. Traces combine_fn into a graph on cloned sample inputs
    # (init clones, first slice of each xs, additional inputs) with proxy tracing disabled, finds
    # the single output node (AssertionError if none or multiple), checks that init metadata
    # matches the carry outputs, registers the graph under a unique "scan_combine_graph" name on
    # the tracer root, emits a call_function node for func_overload (passing mutated_arg_indices if
    # non-empty), and builds fake outputs (carry plus outputs stacked to the scan length; outputs
    # must be tensors or None, else AssertionError). Returns them tracked to the node's proxy.
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_inits = [clone_input(x_init) for x_init in init]
        sample_inputs = [first_slice_copy(x) for x in xs]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_inits, *sample_inputs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            if outputs is not None:
                raise AssertionError("found multiple output nodes in combine_graph")
            if len(node.args) != 1:
                raise AssertionError(
                    f"expected output node to have 1 arg, got {len(node.args)}"
                )
            outputs = node.args[0]

    if outputs is None:
        raise AssertionError("no output node found in combine_graph")

    carry, output = _extract_carry_and_out(outputs, len(init))
    init_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        i.clone() for i in init
    ]
    carry_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in carry
    ]
    check_meta_consistency(
        init_fake_tensors, carry_fake_tensors, "init", "carry", include_contiguity=False
    )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, init, xs, additional_inputs)
    kwargs = {}
    if mutated_arg_indices:
        kwargs["mutated_arg_indices"] = mutated_arg_indices

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, kwargs, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[0]
        fake_carry, fake_outputs = _extract_carry_and_out(
            [o.meta["val"] if o is not None else None for o in outputs], len(init)
        )
        for t in fake_outputs:
            if not isinstance(t, torch.Tensor) and t is not None:
                raise AssertionError(
                    f"Expected leaf to be a Tensor or None, got {type(t)}"
                )
        out = (
            *fake_carry,
            *(
                stack_y(t, scan_length) if isinstance(t, torch.Tensor) else t
                for t in fake_outputs
            ),
        )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, additional_inputs, mutated_arg_indices=""):
    # [FUNCTION DOC] Eager implementation: requires no active dispatch mode (AssertionError) and
    # runs generic_scan with the given inputs; mutated_arg_indices is ignored.
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return generic_scan(combine_fn, init, xs, additional_inputs=additional_inputs)


class ScanAutogradOp(torch.autograd.Function):
    # [CLASS DOC] Autograd function for scan_op. Forward builds a ScanAutogradImpl from a
    # partitioned (forward/backward) graph and the operands, stores it on ctx and runs the forward
    # scan; backward runs the reverse scan and returns gradients (None for the four non-tensor
    # leading arguments).
    """
    NOTE: [scan partial grad handling]
    If any element of init, of xs, of the outputs or of the additional_inputs does not require gradients,
    i.e., requires_grad=False, there will be still gradients returned for those elements,
    but those gradients will be a tensor filled with zeros of the same shape as the element itself.

    A special case are additional_inputs that are not tensors. Such inputs can occur for example with symbolic tracing,
    where the shape symbol (SymInt) becomes an additional_input.
    For such cases, we compute a ``additional_inputs_tensor_mask``, which is True for elements of additional_inputs
    that are tensors and False otherwise. Gradients of additional_inputs are only accumulated if this mask is True,
    otherwise, the value of initial_g_additional_inputs is passed, which is None for non-Tensor values.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hop_partitioned_graph,
        n_init,
        n_xs,
        n_additional_inputs,
        *operands,
    ):
        # [FUNCTION DOC] Splits operands into init, xs and additional inputs by the given counts,
        # stores a new ScanAutogradImpl on ctx, and returns its forward outputs (carry then
        # stacked outputs), computed below autograd.
        init, xs, additional_inputs = split_into_chunks(
            operands, [n_init, n_xs, n_additional_inputs]
        )
        ctx._scan_impl = ScanAutogradImpl(
            hop_partitioned_graph, init, xs, additional_inputs
        )
        with torch._C._AutoDispatchBelowAutograd():
            return ctx._scan_impl.call_forward()

    @staticmethod
    def backward(ctx, *grad_fw_outputs):
        # [FUNCTION DOC] Returns four None entries for the non-tensor forward arguments followed by
        # the gradients computed by the stored impl's backward scan.
        return (
            None,
            None,
            None,
            None,
            *ctx._scan_impl.call_backward(*grad_fw_outputs),
        )


class ScanForwardIntermediatesHandlingPolicy(enum.Enum):
    # [CLASS DOC] Enumeration of how each forward-graph intermediate saved for backward is handled:
    # KEEP (stack as a per-step output), CLONE (a carried init value; clone then stack),
    # REMOVE_XS (an xs input; reuse the original xs) and REMOVE_ADDITIONAL_INPUTS (an additional
    # input; reuse it directly).
    """
    Partitioner can add intermediates to the output of original graph.
    These intermediates fall into 4 categories and we want to have different policies for handling them by
    modifying the graph:

    CLONE: we clone the intermediate when it is a carried input (i.e. init). In this case, this carry will be
        replaced with new values at each forward step so we need to clone the carry as part of return (i.e. ys)
        so as to remove the aliasing and that each step's intermediate will be stacked together and saved in backward.

    REMOVE_XS: we remove the intermediate from output when it is part of xs. Since xs is read-only, in this case,
        we can directly save them for backward to use.

    REMOVE_ADDITIONAL_INPUTS: we remove the intermediate from output when it is part of additional_inputs. additional_inputs
        are also read-only in each step, we can directly save them for backward to use. We differentiate XS and ADDITIONAL_INPUTS
        so that we could have different treatment for them in backward. In backward, we need to put xs intermediates in carry but
        put additional_inputs as backward scan's additional_inputs.

    KEEP: this corresponds to a real intermediate tensor operations' output. It varies at each forward step, we could just keep
        it as part of ys.

    """

    KEEP = 0
    CLONE = 1
    REMOVE_XS = 2
    REMOVE_ADDITIONAL_INPUTS = 3


class ScanAutogradImpl:
    # [CLASS DOC] Holds the state for one scan autograd call: the partitioned graph, the init, xs
    # and additional inputs, the per-intermediate handling policies, the saved xs / additional
    # inputs / intermediates for backward, and the pytree spec of the forward inputs. The
    # constructor rewrites the forward and backward graphs in place (mutates hop_partitioned_graph).
    # call_forward runs the forward scan once and saves intermediates; call_backward builds and
    # runs the reversed scan producing gradients.
    """
    Wraps over partitioned graph and encapsulates scan-specific implementation details
    """

    def __init__(
        self, hop_partitioned_graph: HopPartitionedGraph, init, xs, additional_inputs
    ):
        # [FUNCTION DOC] Stores the inputs, initializes empty policy/saved lists and the forward
        # input spec, then rewrites the forward graph's intermediates and breaks input/output
        # aliasing in the backward graph (both modify the graphs in place).
        self.hop_partitioned_graph = hop_partitioned_graph
        self.init = init
        self.xs = xs
        self.additional_inputs = additional_inputs
        self.forward_intermediates_handling_policies: list[
            ScanForwardIntermediatesHandlingPolicy
        ] = []
        self.saved_fw_xs: list[Any] = []
        self.saved_fw_additional_inputs: list[Any] = []
        self.saved_intermediates: list[Any] = []
        self.fw_spec = pytree.tree_flatten((init, xs, additional_inputs))[1]
        self._optimize_forward_intermediates()
        self._break_bw_input_output_aliasing()

    def _insert_clone(
        self, need_copy_node: torch.fx.Node, output_node: torch.fx.Node
    ) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Inserts a clone call node for need_copy_node just before
        # output_node in its graph, copies the original node's metadata to the clone and returns it.
        graph: torch.fx.Graph = output_node.graph
        with graph.inserting_before(output_node):
            clone_node = graph.call_function(
                torch.ops.aten.clone.default,
                args=(need_copy_node,),
            )
            clone_node.meta = (
                need_copy_node.meta.copy() if hasattr(need_copy_node, "meta") else {}
            )
        return clone_node

    def _break_bw_input_output_aliasing(self) -> None:
        # [FUNCTION DOC] HELPER: Rewrites the backward graph so no output is a placeholder or shares
        # storage with a placeholder's example value: such outputs are replaced by clones. If any
        # rewrite happened, the output node is updated, the graph linted and recompiled. Raises
        # AssertionError if the output node does not have exactly one argument.
        """
        The min-cut partitioner can produce a ``bw_gm`` whose output aliases
        an input placeholder. The most common case is a direct placeholder
        return (e.g. when an input doesn't require grad and its gradient is
        ``zeros_like(input)`` saved as a forward intermediate), but transitive
        aliases (views, ``_unsafe_view``, slices, ...) can hit the same path.

        When ``bw_gm`` is wrapped as the per-step backward inside ``scan_op``,
        any input returned as an output violates ``scan_op``'s aliasing-free
        invariant and surfaces as ``UncapturedHigherOrderOpError`` under
        dynamo. Clone bw_gm outputs that are direct placeholders or whose
        ``meta['val']`` shares storage with any placeholder.
        """
        from torch.multiprocessing.reductions import StorageWeakRef

        bw_gm = self.hop_partitioned_graph.bw_gm
        bw_output_node = next(iter(bw_gm.graph.find_nodes(op="output")))
        if len(bw_output_node.args) != 1:
            raise AssertionError(
                f"expected bw_gm output to have 1 arg, got {len(bw_output_node.args)}"
            )
        bw_outputs = bw_output_node.args[0]

        ph_storages: set = set()
        for ph in bw_gm.graph.find_nodes(op="placeholder"):
            val = ph.meta.get("val", None) if hasattr(ph, "meta") else None
            if isinstance(val, torch.Tensor):
                ph_storages.add(StorageWeakRef(val._typed_storage()))

        def _aliases_placeholder(node: torch.fx.Node) -> bool:
            # [FUNCTION DOC] HELPER: True if node is a placeholder or its example tensor value
            # shares storage with any placeholder's example tensor; else False.
            if node.op == "placeholder":
                return True
            val = node.meta.get("val", None) if hasattr(node, "meta") else None
            if isinstance(val, torch.Tensor):
                return StorageWeakRef(val._typed_storage()) in ph_storages
            return False

        new_bw_outputs = []
        rewrote = False
        for out in bw_outputs:
            if isinstance(out, torch.fx.Node) and _aliases_placeholder(out):
                new_bw_outputs.append(self._insert_clone(out, bw_output_node))
                rewrote = True
            else:
                new_bw_outputs.append(out)
        if rewrote:
            bw_output_node.args = (tuple(new_bw_outputs),)
            bw_gm.graph.lint()
            bw_gm.recompile()

    def _optimize_forward_intermediates(self):
        # [FUNCTION DOC] HELPER: Classifies each saved forward intermediate (outputs of the forward
        # graph beyond the real outputs) by which kind of placeholder it is: init (CLONE), xs
        # (REMOVE_XS), additional input (REMOVE_ADDITIONAL_INPUTS) or computed (KEEP). Fills the
        # policy list and the saved xs / additional inputs lists, then rewrites the forward output
        # node to keep the real outputs plus cloned and kept intermediates only, linting and
        # recompiling the graph. Raises AssertionError if state lists are not initially empty.
        # Side effect: mutates the forward graph module.
        """
        We optimize the forward intermediates by categorize forward intermediates into categories
        and construct a ScanForwardIntermediatesHandlingPolicy for them

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Need remove aliasing in fw_gm:\n%s",
                self.hop_partitioned_graph.fw_gm.print_readable(print_output=False),
            )

        fw_gm = self.hop_partitioned_graph.fw_gm
        fw_all_outputs = _find_hop_subgraph_outputs(fw_gm)
        phs = list(fw_gm.graph.find_nodes(op="placeholder"))
        fw_outputs = fw_all_outputs[: self.hop_partitioned_graph.n_fw_outputs]
        fw_intermediates = fw_all_outputs[self.hop_partitioned_graph.n_fw_outputs :]

        init_phs, xs_phs, additional_inputs_phs = pytree.tree_unflatten(
            phs, self.fw_spec
        )
        init_node_set, xs_node_set, addi_node_set = (
            set(init_phs),
            set(xs_phs),
            set(additional_inputs_phs),
        )

        if len(self.forward_intermediates_handling_policies) != 0:
            raise AssertionError(
                "forward_intermediates_handling_policies should be empty"
            )
        if len(self.saved_fw_xs) != 0:
            raise AssertionError("saved_fw_xs should be empty")
        if len(self.saved_fw_additional_inputs) != 0:
            raise AssertionError("saved_fw_additional_inputs should be empty")
        intermediate_idx_to_ph_idx = {}
        ph_idx = {ph: i for i, ph in enumerate(phs)}
        for i, out in enumerate(fw_intermediates):
            if out in init_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.CLONE
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            elif out in xs_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.REMOVE_XS
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            elif out in addi_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            else:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.KEEP
                )

        new_output_node = []
        real_graph_inputs = (
            list(self.init) + list(self.xs) + list(self.additional_inputs)
        )
        fw_output_node = next(iter(fw_gm.graph.find_nodes(op="output")))
        for intermediate_idx, (node, policy) in enumerate(
            zip(fw_intermediates, self.forward_intermediates_handling_policies)
        ):
            if policy == ScanForwardIntermediatesHandlingPolicy.CLONE:
                new_output_node.append(self._insert_clone(node, fw_output_node))
            elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                if intermediate_idx not in intermediate_idx_to_ph_idx:
                    raise AssertionError(
                        f"intermediate_idx {intermediate_idx} not in intermediate_idx_to_ph_idx"
                    )
                inp_idx = intermediate_idx_to_ph_idx[intermediate_idx]
                self.saved_fw_xs.append(real_graph_inputs[inp_idx])
            elif (
                policy
                == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
            ):
                if intermediate_idx not in intermediate_idx_to_ph_idx:
                    raise AssertionError(
                        f"intermediate_idx {intermediate_idx} not in intermediate_idx_to_ph_idx for REMOVE_ADDITIONAL_INPUTS"
                    )
                inp_idx = intermediate_idx_to_ph_idx[intermediate_idx]
                self.saved_fw_additional_inputs.append(real_graph_inputs[inp_idx])
            else:
                new_output_node.append(node)

        fw_output_node.args = (tuple(fw_outputs) + tuple(new_output_node),)
        fw_gm.graph.lint()
        fw_gm.recompile()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "after removing aliasing:\n%s", fw_gm.print_readable(print_output=False)
            )

    def call_forward(self):
        # [FUNCTION DOC] Runs scan_op on the forward graph with the stored inputs, splits the
        # result into real outputs and intermediates, saves the intermediates for backward
        # (AssertionError if some were already saved) and returns the real outputs as a tuple.
        fw_outputs_and_intermediates: tuple[Any] = scan_op(
            self.hop_partitioned_graph.fw_gm, self.init, self.xs, self.additional_inputs
        )  # type: ignore[return-type]
        fw_outs = fw_outputs_and_intermediates[
            : self.hop_partitioned_graph.n_fw_outputs
        ]
        saved_intermediates = fw_outputs_and_intermediates[
            self.hop_partitioned_graph.n_fw_outputs :
        ]
        if len(self.saved_intermediates) != 0:
            raise AssertionError(
                "saved_intermediates should be empty before call_forward"
            )
        self.saved_intermediates.extend(saved_intermediates)
        return tuple(fw_outs)

    def call_backward(self, *grad_fw_outputs):
        # [FUNCTION DOC] Computes gradients by running a reversed scan over the backward graph.
        # Carry gradients are carried, additional-input gradients are accumulated (only for tensor
        # additional inputs), and per-step xs gradients are stacked. Saved intermediates are
        # reassembled per step according to the handling policies. Inputs are flipped along the scan
        # dimension before and the xs gradients flipped back after. Returns init gradients, xs
        # gradients, then additional-input gradients (None for non-tensors).
        # Raises AssertionError if the gradient spec was never set.
        """
        Recall that fw_outputs = (*carry, *ys), bw_gm takes in (*fw_intermediates, *grad_carry, *grad_ys)
        and returns (*grad_init, *grad_xs, *grad_additional_inputs)
        The backward is a reversed scan that can be constructed as follows:

          grad_additional_inputs = torch.zeros_like(additional_inputs)
          bw_init = (grad_carry, grad_additional_inputs)
          bw_xs = (fw_intermediates, grad_ys)
          grad_init, grad_additional_inputs, grad_xs = scan(
            combine_fn,
            bw_init,
            bw_xs,
            reverse = True
          )
          , where combine_fn is defined as follows:

           def combine_fn(bw_init, bw_xs):
             grad_carry, grad_additional_inputs = bw_init
             fw_intermediates, grad_y = bw_xs
             nxt_grad_carry, grad_x, nxt_grad_additional_inputs = bw_gm(*fw_intermediates, *grad_carry, *grad_y)
             return (nxt_grad_carry, grad_additional_inputs + nxt_grad_additional_inputs), grad_x

          Note that grad_additional_inputs is accumulated with add, grad_carry is carried over to next iteration and
          grad_x is the ys output, which will be stacked together after the loop and will have the same shape as xs.
        """
        fw_policy = self.forward_intermediates_handling_policies
        saved_intermediates = self.saved_intermediates
        saved_fw_xs = self.saved_fw_xs
        saved_fw_additional_inputs = self.saved_fw_additional_inputs

        n_carry = len(self.init)

        grad_carry, grad_ys = grad_fw_outputs[:n_carry], grad_fw_outputs[n_carry:]
        additional_inputs_tensor_masks = [
            bool(isinstance(t, torch.Tensor)) for t in self.additional_inputs
        ]
        grad_additional_inputs = [
            torch.zeros_like(t)
            for t in filter_with_masks(
                self.additional_inputs, additional_inputs_tensor_masks
            )
        ]

        bw_init = [grad_carry, grad_additional_inputs]
        bw_xs = [
            grad_ys,
            saved_fw_xs,
            saved_intermediates,
        ]
        bw_additional_inputs = saved_fw_additional_inputs

        _, flat_spec = pytree.tree_flatten((bw_init, bw_xs, bw_additional_inputs))

        grad_spec = None

        def bw_single_step_wrapper(*args):
            # [FUNCTION DOC] HELPER: One backward step on flat args. Rebuilds the structured carry
            # grads, accumulated additional-input grads, per-step grads, saved xs and intermediates;
            # reorders intermediates per policy (RuntimeError for an unknown policy); calls the
            # backward graph; returns the flat list of next carry grads, updated accumulated
            # additional-input grads (previous plus current) and xs grads. Records the output spec
            # in the enclosing grad_spec variable.
            bw_init, bw_xs, bw_additional_inputs = pytree.tree_unflatten(
                args, flat_spec
            )
            grad_carry, grad_additional_inputs = bw_init
            grad_y, saved_fw_xs, saved_intermediates = bw_xs
            saved_fw_additional_inputs = bw_additional_inputs

            fw_intermediates = []
            xs_it = iter(saved_fw_xs)
            carry_it = iter(saved_intermediates)
            addi_it = iter(saved_fw_additional_inputs)
            for policy in fw_policy:
                if policy in (
                    ScanForwardIntermediatesHandlingPolicy.CLONE,
                    ScanForwardIntermediatesHandlingPolicy.KEEP,
                ):
                    fw_intermediates.append(next(carry_it))
                elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                    fw_intermediates.append(next(xs_it))
                elif (
                    policy
                    == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
                ):
                    fw_intermediates.append(next(addi_it))
                else:
                    raise RuntimeError(f"Unknown policy: {policy}")

            grad_fw_outputs = (*grad_carry, *grad_y)

            flat_out = self.hop_partitioned_graph.bw_gm(
                *fw_intermediates,
                *grad_fw_outputs,
            )

            next_grad_carry, grad_xs, grad_addi = split_into_chunks(
                flat_out,  # type: ignore[arg-type]
                [len(self.init), len(self.xs), len(self.additional_inputs)],
            )

            nonlocal grad_spec
            flat_grads, grad_spec = pytree.tree_flatten(
                (
                    next_grad_carry,
                    [
                        prev + cur
                        for prev, cur in zip(
                            grad_additional_inputs,
                            filter_with_masks(
                                grad_addi, additional_inputs_tensor_masks
                            ),
                        )
                    ],
                    grad_xs,
                )
            )
            return flat_grads

        single_step_bw_xs = pytree.tree_map(first_slice_copy, bw_xs)
        bw_single_step_gm = materialize_as_graph(
            bw_single_step_wrapper,
            tuple(
                pytree.tree_flatten((bw_init, single_step_bw_xs, bw_additional_inputs))[
                    0
                ]
            ),
        )

        flat_grads = scan_op(
            bw_single_step_gm,
            pytree.tree_flatten(bw_init)[0],
            # TODO: torch.flip copies the tensor, we should optimize it away
            [torch.flip(x, (0,)) for x in pytree.tree_flatten(bw_xs)[0]],
            pytree.tree_flatten(bw_additional_inputs)[0],
        )
        if grad_spec is None:
            raise AssertionError("grad_spec must not be None after scan_op")
        grad_init, grad_additional_inputs, grad_xs = pytree.tree_unflatten(
            flat_grads, grad_spec
        )
        return (
            *grad_init,
            *[torch.flip(elem, (0,)) for elem in grad_xs],
            *fill_none_with_masks(
                grad_additional_inputs, additional_inputs_tensor_masks
            ),
        )


@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs, mutated_arg_indices=""):
    # [FUNCTION DOC] Autograd key implementation. Temporarily turns on requires_grad for
    # floating/complex init tensors that lack it (so gradients flow through the carry), builds a
    # min-cut partitioned forward/backward graph of combine_fn, restores requires_grad, and applies
    # ScanAutogradOp. mutated_arg_indices is ignored. Returns the scan outputs.
    with disable_proxy_modes_tracing():
        # If init was passed in with requires_grad=False, AOT joint creation drops it from
        # grad_primals and zero-fills, severing the carry chain and silently
        # zeroing gradients that should reach closed-over additional_inputs from
        # earlier steps. Thus, we temporarily flip requires_grad on the init tensors
        # during tracing and restore it afterwards; this only affects the traced
        # graphs, not the operands passed to apply. init elements are leaves.
        flipped = [
            t
            for t in init
            if (t.dtype.is_floating_point or t.dtype.is_complex) and not t.requires_grad
        ]
        try:
            for t in flipped:
                t.requires_grad_(True)
            hop_partitioned_graph: HopPartitionedGraph = (
                HopGraphMinCutPartitioner.create_partitioned_graph(
                    combine_fn,
                    (*init, *[first_slice_copy(x) for x in xs], *additional_inputs),
                    always_recompute_complex_exprs=True,
                )
            )
        finally:
            for t in flipped:
                t.requires_grad_(False)

    return ScanAutogradOp.apply(
        hop_partitioned_graph,
        len(init),
        len(xs),
        len(additional_inputs),
        *init,
        *xs,
        *additional_inputs,
    )


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(
    mode, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    # [FUNCTION DOC] Proxy-mode implementation; delegates to trace_scan with scan_op.
    return trace_scan(
        mode,
        scan_op,
        combine_fn,
        init,
        xs,
        additional_inputs,
        mutated_arg_indices=mutated_arg_indices,
    )


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(
    mode, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    # [FUNCTION DOC] Fake tensor implementation: under the fake mode, runs combine_fn once on init,
    # the first slice of each xs and the additional inputs; outputs must be tensors or None
    # (AssertionError otherwise). Returns the carry followed by outputs stacked to the scan length.
    with mode:
        scan_length = xs[0].shape[0]
        carry, outputs = _extract_carry_and_out(
            combine_fn(
                *init,
                *[first_slice_copy(inp) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        for t in outputs:
            if not isinstance(t, torch.Tensor) and t is not None:
                raise AssertionError(
                    f"Expected leaf to be a Tensor or None, got {type(t)}"
                )
        out = (
            *carry,
            *(
                stack_y(t, scan_length) if isinstance(t, torch.Tensor) else t
                for t in outputs
            ),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(
    ctx, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    # [FUNCTION DOC] Functionalization implementation. If the context has a mode and the op can be
    # auto-functionalized, delegates to that path. Otherwise unwraps init, xs and additional
    # inputs, functionalizes combine_fn, checks combine_fn for input aliasing/mutation on sample
    # inputs, re-invokes scan_op below the functionalize key (forwarding mutated_arg_indices) and
    # wraps the outputs.
    from torch._higher_order_ops.utils import (
        _check_alias_and_mutation,
        _maybe_run_with_interpreter,
    )

    if hasattr(ctx, "mode"):
        hop_instance = HopInstance.create(
            scan_op,
            combine_fn,
            init,
            xs,
            additional_inputs,
            mutated_arg_indices=mutated_arg_indices,
        )
        if can_auto_functionalize(hop_instance):
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                tuple(
                    pytree.tree_flatten((combine_fn, init, xs, additional_inputs))[0]
                ),
                {},
            )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)

    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        sample_unwrapped_xs_sliced = (
            [first_slice_copy(inp) for inp in unwrapped_xs]
            if len(unwrapped_xs) > 0
            else [None]
        )
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(combine_fn, sample_inputs, "scan", pre_dispatch)
        ret = scan_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            unwrapped_additional_inputs,
            mutated_arg_indices=mutated_arg_indices,
        )
    return ctx.wrap_tensors(ret)


@scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
def scan_batch_rule(
    interpreter, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
):
    # [FUNCTION DOC] vmap batching rule. Unwraps batched inputs, moves every batch dimension to
    # the last position so it does not interfere with scan's leading dimension, runs scan_op one
    # level below with combine_fn wrapped to vmap-restore and move output batch dims last, then
    # rewraps outputs as batched using the recorded output dims. Raises AssertionError if output
    # dims were never recorded.
    unbatched_args, in_dims = unwrap_batched(
        (init, xs, additional_inputs), interpreter.level()
    )
    # move to last dim to not interfere with scan's batching
    unbatched_init, unbatched_xs, unbatched_additional_inputs = pytree.tree_map(
        lambda x, bdim: x.movedim(bdim, -1) if bdim is not None else x,
        unbatched_args,
        in_dims,
    )
    after_move_dims = tuple(
        pytree.tree_flatten(
            pytree.tree_map(lambda x: -1 if x is not None else None, in_dims)
        )[0]
    )

    with interpreter.lower():
        out_dims = None

        def wrapper(*args):
            # [FUNCTION DOC] HELPER: Runs combine_fn under restored vmap, moves batched outputs'
            # batch dim to last, records output dims (last or none) in the enclosing out_dims and
            # returns the outputs tuple.
            nonlocal out_dims
            outputs, per_slice_out_dims = restore_vmap(
                combine_fn,
                after_move_dims,
                interpreter.batch_size(),
                interpreter.randomness(),
            )(*args)
            # Note: outputs are not batched, we just move the batch dim to the end
            # this is to avoid it interfering with scan's batching
            outputs = tuple(
                pytree.tree_map(
                    lambda out, out_bdim: out.movedim(out_bdim, -1)
                    if out_bdim is not None
                    else out,
                    outputs,
                    per_slice_out_dims,
                )
            )
            out_dims = tuple(
                pytree.tree_map(
                    lambda out_bdim: -1 if out_bdim is not None else None,
                    per_slice_out_dims,
                )
            )
            return outputs

        op_kwargs = {}
        if mutated_arg_indices:
            op_kwargs["mutated_arg_indices"] = mutated_arg_indices
        unwrapped_out = scan_op(
            wrapper,
            unbatched_init,
            unbatched_xs,
            unbatched_additional_inputs,
            **op_kwargs,
        )

    if out_dims is None:
        raise AssertionError("out_dims must not be None after scan_op")
    batched_out = wrap_batched(unwrapped_out, out_dims, interpreter.level())
    return batched_out


# dense implementation for scan. Used for testing only.
def _fake_scan(combine_fn, init, xs=None, dim=0, reverse=False, length=None):
    # [FUNCTION DOC] HELPER: Pure-Python reference scan for tests. Supports a length counter mode
    # without xs tensors (length 0 returns init and empty outputs), returns (init, empty list) when
    # there are no xs tensors and no length, otherwise loops over the scan dim (reversed if asked),
    # calling combine_fn on the carry and each slice, and stacks per-step outputs along dim (empty
    # tensors for zero steps; None leaves stay None, other non-tensors raise AssertionError).
    # Returns (final carry, outputs).
    carry_leaves, carry_spec = pytree.tree_flatten(init)
    inp_leaves, inp_spec = pytree.tree_flatten(xs)
    xs_has_tensors = any(isinstance(l, torch.Tensor) for l in inp_leaves)

    if length is not None and not xs_has_tensors:
        if length == 0:
            return init, _build_empty_output_for_length_zero(combine_fn, init)

        _user_combine_fn = combine_fn

        def combine_fn(carry, _ignored):
            # [FUNCTION DOC] HELPER: Wrapper that ignores the counter slice and calls the user
            # combine function with None as x.
            return _user_combine_fn(carry, None)

        # Dummy length-N iteration counter; the wrapped combine_fn passes x=None.
        inp_leaves = [torch.zeros(length, dtype=torch.int64)]
        inp_spec = pytree.tree_structure(None)
        xs_has_tensors = True

    if not xs_has_tensors:
        return init, []
    result_flat = []
    carry = carry_leaves
    op = reversed if reverse else lambda x: x

    _, dummy_out = combine_fn(
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(
            [first_slice_copy(elem, dim) for elem in inp_leaves],
            inp_spec,
        ),
    )
    dummy_out_leaves, dummy_out_spec = pytree.tree_flatten(dummy_out)

    for ind in op(range(inp_leaves[0].size(dim))):
        xs = [elem.select(dim, ind) for elem in inp_leaves]

        carry, y = combine_fn(
            pytree.tree_unflatten(carry, carry_spec),
            pytree.tree_unflatten(xs, inp_spec),
        )
        carry, _ = pytree.tree_flatten(carry)
        y, _ = pytree.tree_flatten(y)
        result_flat.append(y)

    results: list[torch.Tensor | None] = []
    for leaf_idx, leaf in enumerate(dummy_out_leaves):
        if isinstance(leaf, torch.Tensor):
            if len(result_flat) == 0:
                stacked = torch.empty(
                    [0] + list(leaf.shape), dtype=leaf.dtype, device=leaf.device
                )
            else:
                stacked = torch.stack([e[leaf_idx] for e in op(result_flat)])
            results.append(
                torch.movedim(stacked, 0, dim) if dim < stacked.ndim else stacked
            )
        else:
            if leaf is not None:
                raise AssertionError(
                    f"Expected leaf to be a Tensor or None, got {type(leaf)}"
                )
            results.append(leaf)
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
