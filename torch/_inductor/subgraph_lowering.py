# [MODULE DOC] Utilities to lower a subgraph of pointwise ops (used by higher order operators) into a
# single callable over an ops handler. Contains an FX interpreter that lowers each node via the
# lowering table while forbidding buffer creation, an ops handler that traces ops into an FX graph,
# and lower_pointwise_subgraph tying them together. No import-time side effects beyond type aliases.
"""Utilities for lowering subgraphs used by higher order operators"""

import functools
import operator
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, TypeVar
from typing_extensions import ParamSpec

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from . import ir
from .exc import SubgraphLoweringException
from .graph import GraphLowering
from .ops_handler import OpsHandler, SimpleCSEHandler
from .virtualized import ops, V, WrapperHandler


T = TypeVar("T")
_P = ParamSpec("_P")

OpOverload = torch._ops.OpOverload
LoweringDict = dict[OpOverload | str, Callable[..., Any]]
TargetType = Callable[..., Any] | str


class PointwiseSubgraphLowering(torch.fx.Interpreter):
    # [CLASS DOC] FX Interpreter that lowers a pointwise subgraph using the root GraphLowering. State:
    # graph_outputs (set by the output node), root_graph, allowed_mutations (ops permitted to mutate
    # buffers), additional_lowerings (override table), the op currently being processed, buffers and
    # mutated_buffers. Raises SubgraphLoweringException on unexpected buffer creation/mutation or on
    # ops lacking a lowering. Unknown attributes fall through to the root graph.
    """
    Lowers a pointwise subgraph to a single set of buffers with a separate
    lowering object. Errors if buffers are created unexpectedly
    """

    graph_outputs: list[ir.IRNode] | None
    root_graph: GraphLowering
    _current_op: TargetType | None
    # For backwards of buffer_grads with scatters we allow mutations
    allowed_mutations: OrderedSet[OpOverload] | None
    additional_lowerings: LoweringDict | None
    buffers: list[ir.Buffer]
    mutated_buffers: OrderedSet[str]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: GraphLowering,
        allowed_mutations: OrderedSet[OpOverload] | None = None,
        additional_lowerings: LoweringDict | None = None,
    ) -> None:
        # [FUNCTION DOC] Initialize the interpreter on the graph module and store the root lowering, allowed
        # mutation ops and extra lowerings; graph_outputs and current op start as None, buffer tracking
        # containers start empty.
        super().__init__(gm)
        self.graph_outputs = None
        self.root_graph = root_graph_lowering
        self.allowed_mutations = allowed_mutations
        self.additional_lowerings = additional_lowerings
        self._current_op = None

        # Used to track buffers created during lowering
        self.mutated_buffers = OrderedSet()
        self.buffers = []

    @contextmanager
    def _op_context(self, op: TargetType) -> Generator[None, None, None]:
        # [FUNCTION DOC] HELPER: Context manager recording op as the current op while processing a node and
        # restoring the previous one on exit (even on error).
        """Set which op is being processed in call function to know if we can mutate buffers"""
        previous = self._current_op
        self._current_op = op
        try:
            yield
        finally:
            self._current_op = previous

    def _approved_mutator(self) -> bool:
        # [FUNCTION DOC] HELPER: True if allowed mutations are configured and the current op is among them.
        return (
            self.allowed_mutations is not None
            and self._current_op in self.allowed_mutations
        )

    def mark_buffer_mutated(self, name: str) -> None:
        # [FUNCTION DOC] Record a buffer name as mutated if the current op is an approved mutator; otherwise
        # raise SubgraphLoweringException.
        if self._approved_mutator():
            self.mutated_buffers.add(name)
        else:
            raise SubgraphLoweringException(
                f"Buffer mutation detected during lowering of {self._current_op}. "
                "Buffer mutations are only allowed in approved mutation ops. "
                "This is an error in the lowering of the subgraph, please file a bug report."
            )

    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str:
        # [FUNCTION DOC] Register a buffer with the root graph and return its name, only when the current op is
        # an approved mutator (forwarding set_name); otherwise raises SubgraphLoweringException since
        # pointwise subgraphs must not create buffers.
        if self._approved_mutator():
            name = self.root_graph.register_buffer(buffer, set_name=set_name)
            return name
        else:
            raise SubgraphLoweringException(
                "Buffers cannot be created while lowering a pointwise subgraph. "
                "This could be for a good reason (e.g. you're calling an op we can't codegen as a pointwise op), "
                "but it could also be a bug. Please file a bug report if you think this should be supportable."
            )

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] Delegate any attribute not found on this object to the root graph lowering.
        return getattr(self.root_graph, name)

    def call_function(
        self,
        target: TargetType,
        args: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        # [FUNCTION DOC] Lower one call_function node. Sets the current op for the duration. A getitem on a
        # list/tuple/dict runs the default interpreter behavior; a target in additional_lowerings uses that
        # lowering (asserts the target is an OpOverload); otherwise uses the global lowerings table, raising
        # SubgraphLoweringException if there is no lowering. Returns the lowered result.
        from .lowering import lowerings

        with self._op_context(target):
            if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
                return super().call_function(target, args, kwargs)

            # These takes precedence over the main lowerings
            if self.additional_lowerings is not None:
                if target in self.additional_lowerings:
                    if not isinstance(target, OpOverload):
                        raise AssertionError(
                            f"expected target to be an OpOverload, got {type(target)}"
                        )
                    return self.additional_lowerings[target](*args, **kwargs)

            if target not in lowerings:
                raise SubgraphLoweringException(
                    f"{target} not supported in subgraph, (missing lowering)"
                )
            return lowerings[target](*args, **kwargs)

    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Handle the output node: asserts exactly one argument and stores it as graph_outputs.
        if len(args) != 1:
            raise AssertionError(f"expected exactly one output arg, got {len(args)}")
        self.graph_outputs = args[0]


@dataclass
class InputDescriptor:
    # [CLASS DOC] Describes a tensor input of a subgraph by dtype and device.
    dtype: torch.dtype
    device: torch.device


SubgraphInput = InputDescriptor | int | sympy.Basic


class TracingOpsHandler(WrapperHandler):
    # [CLASS DOC] Ops handler wrapper that records every op call into an FX graph via a tracer. Holds the
    # tracer and one placeholder proxy per subgraph input; wraps a proxy for the ops placeholder as its
    # parent handler. The output method emits the FX output node.
    def __init__(self, tracer: torch.fx.Tracer, num_inputs: int) -> None:
        # [FUNCTION DOC] Create the ops placeholder proxy as the wrapped handler, and one input placeholder
        # proxy per input (named input0, input1, ...).
        parent = tracer.create_proxy("placeholder", "ops", (), {})
        super().__init__(cast(OpsHandler[Any], parent))
        self.tracer = tracer

        self.placeholders = [
            self.tracer.create_proxy("placeholder", f"input{i}", (), {})
            for i in range(num_inputs)
        ]

    def placeholder(self, idx: int) -> torch.fx.Proxy:
        # [FUNCTION DOC] Return the placeholder proxy for input idx.
        return self.placeholders[idx]

    def output(self, *args: tuple[object]) -> None:
        # [FUNCTION DOC] Emit an FX output node whose arguments are the given values.
        self.tracer.create_node(
            "output", "output", (tuple(self.tracer.create_arg(a) for a in args),), {}
        )


def lower_pointwise_subgraph(
    subgraph: ir.Subgraph, inputs: list[SubgraphInput]
) -> Callable[_P, Any]:
    # [FUNCTION DOC] Lower an ir.Subgraph of pointwise ops to a callable. Builds Pointwise graph inputs for
    # tensor descriptors (others passed through), runs PointwiseSubgraphLowering with the current
    # graph, then traces each scalar Pointwise output through a CSE-wrapped tracing handler into one
    # FX GraphModule. Asserts outputs are scalar TensorBox/StorageBox/Pointwise. Returns a function
    # that executes the traced module against the current ops handler.
    # Lower subgraph to ir.Pointwise nodes
    def fake_inner_fn(loop_idx: int, input_idx: int) -> ir.Expr | ir.TensorBox | None:
        # [FUNCTION DOC] HELPER: Inner function for placeholder pointwise inputs; ignores the loop index and
        # returns the ops placeholder for the bound input index.
        return ops.placeholder(input_idx)

    graph_inputs = []
    for i, desc in enumerate(inputs):
        if isinstance(desc, InputDescriptor):
            graph_inputs.append(
                ir.Pointwise.create(
                    device=desc.device,
                    dtype=desc.dtype,
                    inner_fn=functools.partial(fake_inner_fn, input_idx=i),
                    ranges=[],
                )
            )
        else:
            graph_inputs.append(desc)
    gm = subgraph.graph_module
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        pw_subgraph.run(*graph_inputs)

    # Combine multiple pointwise computations into a single graph module
    # Do this by tracing through each individually and doing CSE
    tracer = torch.fx.Tracer()
    tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
    trace_ops = SimpleCSEHandler(TracingOpsHandler(tracer, len(inputs)))
    if pw_subgraph.graph_outputs is None:
        raise AssertionError("expected pw_subgraph.graph_outputs to be set")

    with V.set_ops_handler(trace_ops):
        output_irs = []

        for out_var in pw_subgraph.graph_outputs:
            if not isinstance(out_var, ir.TensorBox):
                raise AssertionError(type(out_var))
            if out_var.get_size() != []:
                raise AssertionError(
                    f"expected scalar output (empty size), got {out_var.get_size()}"
                )
            if not isinstance(out_var.data, ir.StorageBox):
                raise AssertionError(
                    f"expected out_var.data to be a StorageBox, got {type(out_var.data)}"
                )
            if not isinstance(out_var.data.data, ir.Pointwise):
                raise AssertionError(
                    f"expected out_var.data.data to be a Pointwise, got {type(out_var.data.data)}"
                )

            idx = ()
            ir_out = out_var.data.data.inner_fn(idx)

            output_irs.append(ir_out)

        ops.output(*output_irs)

    lowered_gm = torch.fx.GraphModule({}, tracer.graph)

    def inner_fn(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        # [FUNCTION DOC] HELPER: Run the lowered module with the current ops handler followed by the given args.
        return lowered_gm(V.get_ops_handler(), *args, **kwargs)

    return inner_fn
