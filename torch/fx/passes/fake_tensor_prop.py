# [MODULE DOC] Defines FakeTensorProp, an FX interpreter that runs a graph on fake tensors and records
# the resulting fake values (and unbacked symbol bindings) in node metadata. No import-time side effects.
from typing import Any

import torch.fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, is_fake_tensor
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import py_sym_types, snapshot_fake
from torch.fx.node import map_aggregate
from torch.utils._ordered_set import OrderedSet


__all__ = ["FakeTensorProp"]


@compatibility(is_backward_compatible=False)
class FakeTensorProp(torch.fx.Interpreter):
    # [CLASS DOC] Interpreter that executes each node under a fake tensor mode and stores snapshots of
    # the outputs in the val metadata entry of the node. State: _mode (the fake mode, created if not
    # supplied) and seen_subgraphs (names of invoke_subgraph bodies already propagated, to avoid redundant
    # work). Key methods: propagate (converts real tensor inputs to fakes), propagate_dont_convert_inputs,
    # run_node.
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """

    def __init__(
        self, module: torch.fx.GraphModule, mode: FakeTensorMode | None = None
    ) -> None:
        # [FUNCTION DOC] Initializes the interpreter for the module, creating a new fake mode when none
        # is given. Side effects: increments the epoch of the mode and resets its nested tensor id counter
        # (even for a mode shared with the caller). Starts with an empty set of seen subgraphs.
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        mode.epoch += 1
        mode.reset_nt_tensor_id_counter()
        self.seen_subgraphs: OrderedSet[str] = OrderedSet()

    def run_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Executes one node and records its result. For the first occurrence of an
        # invoke_subgraph call on a given subgraph name, it instead recursively propagates a fresh
        # FakeTensorProp (sharing the same mode) over the referenced submodule using the val metadata of
        # the operands, and returns that result without recording metadata on this node. Otherwise it
        # runs the node via the base class, rebinds unbacked symbols, converts the result tree to
        # snapshots of fake tensors (real tensors are first converted with static shapes; ints, floats,
        # bools and symbolic scalars are kept; other leaves become None) and, if not entirely None,
        # stores it as val, plus unbacked_bindings when the shape env yields any. Returns the raw result.
        # Raises AssertionError for malformed invoke_subgraph arguments or operands lacking val.
        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            rebind_unbacked,
        )

        if (
            n.op == "call_function"
            and n.target is torch.ops.higher_order.invoke_subgraph
            and n.args[1] not in self.seen_subgraphs
        ):
            # Prevent redundant fake tensor prop for invoke_subgraphs. Note that
            # there is also fake tensor caching for the entire subgraph. This
            # happens the next time we call `run_node` for the same subgraph,
            # which goes through super.run_node and caches the fake tensor prop.
            # Therefore, we are propagating fake tensor through the subgraphs
            # twice.
            if not isinstance(n.args[1], str):
                raise AssertionError(f"Expected str, got {type(n.args[1])}")
            if not (
                isinstance(n.args[0], torch.fx.Node)
                and n.args[0].op == "get_attr"
                and isinstance(n.args[0].target, str)
            ):
                raise AssertionError(
                    "Expected n.args[0] to be a get_attr Node with str target"
                )
            self.seen_subgraphs.add(n.args[1])
            operands = n.args[2:]
            example_inputs = []
            for operand in operands:
                if not (isinstance(operand, torch.fx.Node) and "val" in operand.meta):
                    raise AssertionError("Expected Node with 'val' in meta")
                example_inputs.append(operand.meta["val"])
            return FakeTensorProp(
                getattr(self.module, n.args[0].target), mode=self._mode
            ).propagate(*example_inputs)

        result = super().run_node(n)
        rebind_unbacked(self._mode.shape_env, n, result)

        def extract_val(obj: Any) -> Any:
            # [FUNCTION DOC] HELPER: Maps one result leaf to its metadata form: fake tensors are
            # snapshotted, real tensors are converted to static-shape fakes then snapshotted, symbolic
            # and plain numeric/bool values pass through, anything else yields none.
            if is_fake_tensor(obj):
                return snapshot_fake(obj)
            elif isinstance(obj, torch.Tensor):
                # TODO: How is it possible that we get a non fake tensor?  We
                # should be running under the mode...
                return snapshot_fake(self._mode.from_tensor(obj, static_shapes=True))
            elif isinstance(obj, (*py_sym_types, int, float, bool)):
                return obj
            else:
                return None

        meta = map_aggregate(result, extract_val)
        if meta is not None:
            n.meta["val"] = meta
            if (shape_env := self._mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, result)
            ):
                n.meta["unbacked_bindings"] = symbol_to_path

        return result

    def propagate(self, *args: object) -> Any:
        # [FUNCTION DOC] Converts every real tensor argument to a fake tensor using the mode (other
        # arguments are passed unchanged) and runs the graph via propagate_dont_convert_inputs.
        # Returns the output of the graph; annotates node metadata as a side effect.
        fake_args = [
            self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        return self.propagate_dont_convert_inputs(*fake_args)

    def propagate_dont_convert_inputs(self, *args: object) -> Any:
        # [FUNCTION DOC] Runs the interpreter with the given inputs (assumed already fake) inside the
        # fake mode context and returns the graph output. Saves the fake device of each fake tensor input
        # beforehand and restores it afterwards (even on error) because in-place ops may change it.
        # In-place ops like shallow_copy_data_ can mutate fake_device on
        # input FakeTensors during propagation. Save and restore so the
        # caller's inputs are not permanently corrupted.
        saved_devices = [
            (a, a.fake_device)
            for a in args
            if isinstance(a, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
        ]
        with self._mode:
            try:
                return super().run(*args)
            finally:
                for fake, device in saved_devices:
                    fake.fake_device = device
