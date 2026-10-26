# mypy: allow-untyped-defs
# [MODULE DOC] Support for lowering torch.cond and torch.while_loop to CUDA graph
# conditional nodes. Provides two dispatch modes (one for capture, one for warming up
# control-flow subgraphs before capture) and the node-building helpers if_else_node and
# while_loop_node, plus small validation helpers. No import-time side effects.
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class CUDAGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode active during CUDA graph capture that intercepts the cond and
    # while_loop higher-order operators and builds CUDA graph conditional nodes for them
    # instead of executing them. All other ops pass through. Opts in to seeing higher-order
    # operators via the supports_higher_order_operators flag set in the constructor.
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # [FUNCTION DOC] Always returns True: the compiler should ignore this mode.
        return True

    def __init__(
        self,
    ) -> None:
        # [FUNCTION DOC] Sets the flag allowing higher-order operators to reach this mode,
        # then initializes the base dispatch mode.
        self.supports_higher_order_operators = True
        super().__init__()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        # [FUNCTION DOC] For the cond operator: rejects kwargs (RuntimeError), re-enters this
        # mode and builds an if/else conditional node from the positional args, returning the
        # branch output. For while_loop: same, building a while conditional node. Any other
        # func is simply called with args and kwargs.
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.higher_order.cond:
            # Re-enter the mode to support nested conditionals
            _check_no_cond_kwargs(kwargs)
            with self:
                return if_else_node(*args)
        if func is torch.ops.higher_order.while_loop:
            # Re-enter the mode to support nested control flow
            _check_no_while_loop_kwargs(kwargs)
            with self:
                return while_loop_node(*args)
        return func(*args, **kwargs)


class ControlFlowOpWarmupDispatchMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode used before CUDA graph capture to warm up cond branches and
    # while_loop bodies. State: capture_stream, a dedicated CUDA stream on which throwaway
    # relaxed-mode captures are recorded (the resulting graphs are discarded), and the flag
    # allowing higher-order operators. After warmup the real operator is executed normally.
    """Warm up control-flow subgraphs before CUDA graph capture.

    Data-dependent control flow does not necessarily execute every subgraph, so
    operations in an untaken torch.cond branch or a torch.while_loop body may
    not have been warmed up. This mode uses a relaxed stream capture, whose
    final CUDA graph is discarded, to warm up both cond branches and execute a
    while_loop body once. This works because stream capture does not execute GPU
    code and the branch and body functions are FX graphs without CPU side
    effects.
    """

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        # [FUNCTION DOC] Always returns True: the compiler should ignore this mode.
        return True

    def __init__(
        self,
    ) -> None:
        # [FUNCTION DOC] Initializes the base mode, sets the flag allowing higher-order
        # operators, and creates the dedicated CUDA stream used for warmup captures.
        super().__init__()
        self.supports_higher_order_operators = True
        self.capture_stream = torch.cuda.Stream()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        # [FUNCTION DOC] For cond and while_loop: rejects kwargs (RuntimeError). If the current
        # stream is already capturing (nested control flow) it re-enters the mode and builds
        # the conditional node directly, returning its result. Otherwise it performs a throwaway
        # relaxed-mode capture on capture_stream that builds the node (warming every cond
        # branch, or the while body once), discards the graph, then runs the real operator
        # and returns its result. Other ops are called unchanged.
        kwargs = {} if kwargs is None else kwargs

        # Warm up both sides of this torch.cond()
        if func is torch.ops.higher_order.cond:
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.cond() nested within another
                # control-flow function.
                _check_no_cond_kwargs(kwargs)
                with self:
                    # We re-enter the mode in case of nested calls to torch.cond()
                    return if_else_node(*args)
            else:
                _check_no_cond_kwargs(kwargs)
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    if_else_node(*args)

                return func(*args, **kwargs)
        elif func is torch.ops.higher_order.while_loop:
            _check_no_while_loop_kwargs(kwargs)
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.while_loop() nested within another
                # control-flow function.
                with self:
                    return while_loop_node(*args)
            else:
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    while_loop_node(*args)

                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


def _check_no_cond_kwargs(kwargs) -> None:
    # [FUNCTION DOC] HELPER: Raises RuntimeError if any keyword arguments were given to cond.
    if kwargs:
        raise RuntimeError("CUDA graph conditional torch.cond does not support kwargs")


def _check_no_while_loop_kwargs(kwargs) -> None:
    # [FUNCTION DOC] HELPER: Raises RuntimeError if any keyword arguments were given to while_loop.
    if kwargs:
        raise RuntimeError(
            "CUDA graph conditional torch.while_loop does not support kwargs"
        )


def _is_boolean_scalar_cuda_tensor(pred: object) -> bool:
    # [FUNCTION DOC] HELPER: True only if pred is a zero-dimensional bool tensor on a CUDA device.
    return (
        isinstance(pred, torch.Tensor)
        and pred.size() == torch.Size([])
        and pred.dtype == torch.bool
        and pred.is_cuda
    )


@contextmanager
def _if_body(pred: torch.Tensor) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager. On entry begins capturing into a new if
    # conditional node on the currently capturing CUDA graph, controlled by pred; on exit
    # (even on error) ends the capture into the conditional node.
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    # [FUNCTION DOC] Captures an if/else as two if conditional nodes (the second guarded by the
    # logical negation of pred). Raises ValueError if pred is not on CUDA. Each branch function
    # is called with the operands inside its node; the false branch outputs are copied in place
    # into the true branch outputs so later graph nodes see a single fixed output. Returns the
    # true branch output structure.
    if not pred.is_cuda:
        raise ValueError(
            "Conditions must be on a cuda device to use conditional node in cuda graphs"
        )
    # if-else is not supported until CUDA 12.8. Therefore, we use two
    # if conditions, where one evaluates !pred
    outs = []

    for lazy_pred, fn in [
        (lambda: pred, true_fn),
        (lambda: torch.logical_not(pred), false_fn),
    ]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))

            # The output of the else branch gets copied into the
            # output of the if branch. This is done because the rest
            # of the cudagraph after the conditional node has fixed
            # inputs, so we need to merge the two outputs into a
            # single output.
            if len(outs) == 2:
                for if_out, else_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[1])
                ):
                    if_out.copy_(else_out)
    return outs[0]


@contextmanager
def _while_body(pred: torch.Tensor):
    # [FUNCTION DOC] HELPER: Context manager. On entry begins capturing into a new while
    # conditional node on the currently capturing CUDA graph, controlled by pred, and yields
    # that graph; on exit (even on error) ends the capture into the conditional node.
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_while_node(pred)
    try:
        yield current_cuda_graph
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def while_loop_node(
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
):
    # [FUNCTION DOC] Captures a while loop as a while conditional node. Carried inputs must
    # all be tensors (RuntimeError otherwise); they are cloned into loop-carried buffers.
    # cond_fn is evaluated before the loop and again at the end of each body iteration and
    # must return a boolean scalar CUDA tensor (RuntimeError otherwise). body_fn must return
    # the same pytree structure of tensors as carried_inputs (RuntimeError otherwise); its
    # outputs are copied into the loop-carried buffers when they do not share memory, and
    # the updated predicate is set as the node's conditional handle. Returns the loop-carried
    # buffers (same structure as carried_inputs).
    flat_carried_inputs, carried_spec = pytree.tree_flatten(carried_inputs)
    if not all(isinstance(inp, torch.Tensor) for inp in flat_carried_inputs):
        raise RuntimeError(
            "CUDA graph while_loop conditional nodes only support tensor carried_inputs"
        )

    loop_carried = pytree.tree_map_only(
        torch.Tensor, lambda inp: inp.clone(), carried_inputs
    )
    flat_loop_carried = pytree.tree_leaves(loop_carried)

    pred = cond_fn(*loop_carried, *additional_inputs)
    if not _is_boolean_scalar_cuda_tensor(pred):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar CUDA tensor but got {pred}"
        )

    with _while_body(pred) as current_cuda_graph:
        body_out = body_fn(*loop_carried, *additional_inputs)
        flat_body_out, body_out_spec = pytree.tree_flatten(body_out)
        if body_out_spec != carried_spec:
            raise RuntimeError(
                "body_fn should return the same pytree structure as carried_inputs"
            )
        if not all(isinstance(out, torch.Tensor) for out in flat_body_out):
            raise RuntimeError(
                "CUDA graph while_loop conditional nodes only support tensor "
                "body_fn outputs"
            )

        for carried, out in zip(flat_loop_carried, flat_body_out):
            if carried.data_ptr() != out.data_ptr():
                carried.copy_(out)

        pred = cond_fn(*loop_carried, *additional_inputs)
        if not _is_boolean_scalar_cuda_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar CUDA tensor but got {pred}"
            )
        current_cuda_graph.set_conditional_handle_for_current_node(pred)

    return loop_carried
