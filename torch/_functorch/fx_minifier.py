# [MODULE DOC] FX graph minifier: shrinks a failing torch.fx GraphModule (and its inputs) to a
# smaller graph that still fails a user-supplied predicate. Contains ConcreteProp (an
# interpreter that records concrete per-node values, optionally offloaded to disk), helpers
# that turn nodes into placeholders or load_tensor calls, a ReproState dataclass, and the
# minifier driver with its strategies (truncate suffix, remove outputs, remove unused inputs,
# eliminate dead code, delta debugging, consolidate inputs). Progress is printed to stderr.
# No import-time side effects other than defining a module-level sentinel is_tuple.
from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._content_store import ContentStoreWriter

from .compile_utils import get_outputs, get_placeholders


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


is_tuple = object()


@dataclass
class LoadTensorMeta:
    # [CLASS DOC] HELPER: Lightweight record (size, stride, dtype, device) describing a tensor that
    # was offloaded to disk, stored in node metadata so it can be reloaded via load_tensor.
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class ConcreteProp(torch.fx.Interpreter):
    # [CLASS DOC] FX interpreter that runs a graph on real inputs and records each node's concrete
    # result in node.meta under concrete_value: the tensor itself, a LoadTensorMeta (if offloaded
    # through a ContentStoreWriter), None (aliasing tensors that cannot be offloaded) or the
    # is_tuple sentinel for non-tensor results. State: writer, skip_offload, the set of seen
    # storages (weak refs, used to detect aliasing) and a progress bar set during propagate.
    def __init__(
        self,
        mod: fx.GraphModule,
        *,
        writer: ContentStoreWriter | None = None,
        skip_offload: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initializes the interpreter for mod and stores the optional disk writer,
        # the skip_offload flag (assume tensors are already on disk), an empty seen-storage set
        # and an unset progress bar.
        super().__init__(mod)
        self.writer = writer
        self.skip_offload = skip_offload
        self.seen_storages: set[StorageWeakRef] = set()
        self.pbar: Any = None

    def run_node(self, n: fx.Node) -> Any:
        # [FUNCTION DOC] Executes one node, advances the progress bar, and records the result in
        # the node's metadata. Tensors: stored directly when there is no writer; with a writer,
        # tensors sharing storage with an already seen tensor get None (not offloaded, to keep
        # aliasing contracts), otherwise the tensor is written to disk under the eager directory
        # (unless skip_offload) and a LoadTensorMeta is stored. Non-tensors get the is_tuple
        # sentinel. Returns the real result. Requires self.pbar set by propagate.
        self.pbar.update(1)
        r = super().run_node(n)
        name = n.name

        if isinstance(r, torch.Tensor):
            if self.writer is None:
                n.meta["concrete_value"] = r
            else:
                if StorageWeakRef(r.untyped_storage()) in self.seen_storages:
                    # Refuse to offload tensors which alias other live
                    # tensors, because this will violate operator contracts
                    n.meta["concrete_value"] = None
                else:
                    if not self.skip_offload:
                        self.writer.write_tensor(os.path.join("eager", name), r)
                    n.meta["concrete_value"] = LoadTensorMeta(
                        r.size(), r.stride(), r.dtype, r.device
                    )
                    self.seen_storages.add(StorageWeakRef(r.untyped_storage()))
        else:
            n.meta["concrete_value"] = is_tuple

        return r

    def propagate(self, *args: Any) -> Any:
        # [FUNCTION DOC] Runs the whole module on args under a progress bar (disabled when there
        # is no writer), populating concrete values on every node. Raises an error if the module
        # is not a GraphModule. Returns the module output; updates the bar text after saving
        # unless offloading is skipped.
        mod = self.module
        if not isinstance(mod, fx.GraphModule):
            raise AssertionError(f"expected fx.GraphModule, got {type(mod)}")
        with tqdm(
            desc="Saving intermediates for delta debugging",
            total=len(mod.graph.nodes),
            disable=self.writer is None,
        ) as pbar:
            self.pbar = pbar
            r = super().run(*args)
            if not self.skip_offload:
                pbar.set_description(
                    "Saved!  To skip next time, run with --skip-saving-eager-intermediates"
                )
            return r


def is_load_tensor_node(node: fx.Node) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the node is a call to the debugprims load_tensor op.
    return (
        node.op == "call_function"
        and node.target is torch.ops.debugprims.load_tensor.default
    )


# inplace modifies node/inps
def _convert_node_to_placeholder(
    graph: fx.Graph, node: fx.Node, inps: list[torch.Tensor]
) -> bool:
    # [FUNCTION DOC] HELPER: Mutates node in place to cut the graph at it, based on its recorded
    # concrete value. Output, placeholder and load_tensor nodes are left alone (False). A tensor
    # value turns the node into a placeholder and appends the tensor to inps (True). A missing
    # value (None) does nothing. The tuple sentinel recurses into the node's users (node itself
    # is not erased to keep iteration valid), returning True if any user changed. A LoadTensorMeta
    # turns the node into a load_tensor call with the stored size/stride/device/dtype (True).
    if node.op == "output" or node.op == "placeholder":
        return False

    if is_load_tensor_node(node):
        return False

    concrete_val = node.meta.get("concrete_value", None)

    if isinstance(concrete_val, torch.Tensor):
        node.op = "placeholder"
        node.target = node.name
        node.args = ()
        node.kwargs = {}

        inps.append(concrete_val)
        return True

    elif concrete_val is None:
        return False

    elif concrete_val is is_tuple:
        r = False
        for tuple_user in list(node.users):
            r = _convert_node_to_placeholder(graph, tuple_user, inps) or r
        # NB: We must not erase the node at this point, because
        # we are iterating over the nodes and this would change
        # the iteration order
        # graph.erase_node(node)
        return r

    elif isinstance(concrete_val, LoadTensorMeta):
        node.op = "call_function"
        node.target = torch.ops.debugprims.load_tensor.default
        node.args = (
            os.path.join("eager", node.name),
            concrete_val.size,
            concrete_val.stride,
        )
        node.kwargs = {
            "device": concrete_val.device,
            "dtype": concrete_val.dtype,
        }
        return True

    return False


def create_minified_hlo_graph(
    minified_fx_graph: fx.GraphModule, inputs: Sequence[torch.Tensor]
) -> None:
    # [FUNCTION DOC] Converts a minified FX module to StableHLO via torch_xla and saves it into
    # an hlo_files directory under the current working directory (created if missing). Side
    # effects: filesystem writes; imports torch_xla lazily, so it raises ImportError if absent.
    """
    Takes minified FX graph as primary input, and ports it to HLO via StableHLO
    Provides minified HLO graph as output, and archive them to local directory
    """
    hlo_dir = f"{os.getcwd()}/hlo_files"
    os.makedirs(hlo_dir, exist_ok=True)

    from torch_xla.stablehlo import save_torch_model_as_stablehlo

    save_torch_model_as_stablehlo(minified_fx_graph, inputs, hlo_dir)


def dump_state(fx_g: fx.GraphModule, inps: Sequence[torch.Tensor]) -> None:
    # [FUNCTION DOC] Default state dumper: prints to stdout a runnable repro snippet containing the
    # node count, input shapes/dtypes/devices, code to create dummy inputs (zeros/ones) and the
    # graph's generated source. No return value.
    print(
        f"""
# Working Repro with {len(fx_g.graph.nodes)} nodes
inps = {[(i.shape, i.dtype, str(i.device)) for i in inps]}
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]
{fx_g.code}
"""
    )


def is_power_of_two(n: int) -> bool:
    # [FUNCTION DOC] HELPER: True if n is a positive power of two (zero gives False).
    if n == 0:
        return False
    return (n & (n - 1)) == 0


@dataclass
class ReproState:
    # [CLASS DOC] HELPER: A candidate minimization state: an FX graph plus its input tensors.
    # Invariant (checked on construction): one input per placeholder node.
    graph: fx.Graph
    inps: Sequence[torch.Tensor]

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Validates that the number of placeholders equals the number of
        # inputs; raises an assertion error otherwise.
        ph_nodes = get_placeholders(self.graph)
        if len(ph_nodes) != len(self.inps):
            raise AssertionError(
                f"len(ph_nodes)={len(ph_nodes)} != len(self.inps)={len(self.inps)}"
            )


def minifier(
    fail_f: fx.GraphModule,
    inps: Sequence[torch.Tensor],
    module_fails: Callable[[fx.GraphModule, Sequence[torch.Tensor]], bool],
    dump_state: Callable[[fx.GraphModule, Sequence[torch.Tensor]], None] = dump_state,
    *,
    save_dir: str | None = None,
    offload_to_disk: bool = False,
    skip_offload: bool = False,
    skip_sanity: bool = False,
    max_granularity: int | None = None,
) -> tuple[fx.GraphModule, Sequence[torch.Tensor]]:
    # [FUNCTION DOC] Minimizes a failing graph module while module_fails stays true. Steps: validate
    # max_granularity (power of two, else RuntimeError); optionally create a ContentStoreWriter in
    # save_dir (required when offload_to_disk); run ConcreteProp to record concrete values; unless
    # skip_sanity, check the original graph fails (else RuntimeError). Then loop: at granularity
    # equal to the largest power of two not above the node count (capped by max_granularity) try
    # strategies (remove outputs when outputs are many, dead code elimination, unused-input
    # removal, input consolidation, suffix truncation, delta debugging); on success restart,
    # otherwise retry at halved granularities without the non-granular strategies, finally try
    # removing outputs one at a time. Calls dump_state on every iteration and at the end. Raises
    # RuntimeError if the final graph no longer fails. If the XLA_HLO_DEBUG environment variable
    # is set, also writes minified HLO files. Prints progress and query counts to stderr.
    # Returns the minimized GraphModule and its inputs.
    # KNOWN ISSUE: the docstring example calls minimize, but the function is named minifier;
    # the final message claims output was written to repro.py though dump_state only prints.
    """
    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> # xdoctest: +SKIP(failing)
    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    """

    failing_graph = fail_f.graph
    cur_size = len(failing_graph.nodes)

    if max_granularity is not None and not is_power_of_two(max_granularity):
        raise RuntimeError(f"max_granularity {max_granularity} not power of two")

    num_queries = 0

    def deepcopy_fx_graph(fx_graph: fx.Graph) -> fx.Graph:
        # [FUNCTION DOC] HELPER: Returns an independent deep copy of an FX graph (via a temporary
        # GraphModule built on fail_f).
        return fx.GraphModule(fail_f, copy.deepcopy(fx_graph)).graph

    def graph_fails(graph: fx.Graph, inps: Sequence[torch.Tensor]) -> bool:
        # [FUNCTION DOC] HELPER: Oracle query. Deep-copies the graph, increments the query counter,
        # builds and lints a GraphModule, and returns module_fails on it with the given inputs.
        # Lint errors propagate.
        nonlocal num_queries
        graph = copy.deepcopy(graph)
        num_queries += 1
        mod = fx.GraphModule(fail_f, graph)
        mod.graph.lint()
        return module_fails(mod, inps)

    writer = None
    if offload_to_disk:
        if save_dir is None:
            raise AssertionError("save_dir must not be None when offload_to_disk=True")
        writer = ContentStoreWriter(save_dir)

    ConcreteProp(fail_f, writer=writer, skip_offload=skip_offload).propagate(*inps)
    if not skip_sanity and not graph_fails(failing_graph, inps):
        raise RuntimeError("Input graph did not fail the tester")
    print(f"Started off with {cur_size} nodes", file=sys.stderr)

    def _register_strategy(
        strategy: Callable[[fx.Graph, Sequence[torch.Tensor], int], ReproState | None],
        name: str,
    ) -> Callable[[ReproState, int], ReproState | None]:
        # [FUNCTION DOC] HELPER: Wraps a graph-level strategy into a state-level one that logs the
        # attempt, runs the strategy on a copy of the state, reports node/input/output reductions,
        # and re-verifies the result fails.
        @wraps(strategy)
        def new_func(old_state: ReproState, granularity: int = 1) -> ReproState | None:
            # [FUNCTION DOC] HELPER: Runs the strategy at the given granularity on a copy of
            # old_state. Returns None if the strategy found nothing, or if the new graph no longer
            # fails (warning printed). Raises RuntimeError if the strategy claimed success without
            # fewer nodes, more inputs or fewer outputs. Otherwise returns the new state.
            print(file=sys.stderr)
            print(
                f"Strategy: {name} (G: {granularity}) "
                f"({len(old_state.graph.nodes)} nodes, {len(old_state.inps)} inputs)",
                file=sys.stderr,
            )
            new_state = strategy(
                deepcopy_fx_graph(old_state.graph), list(old_state.inps), granularity
            )
            if new_state is not None:
                new_nodes = len(new_state.graph.nodes)
                old_nodes = len(old_state.graph.nodes)
                new_inps = len(new_state.inps)
                old_inps = len(old_state.inps)
                new_outs = len(get_outputs(new_state.graph))
                old_outs = len(get_outputs(old_state.graph))
                progress_made = False
                if new_nodes < old_nodes:
                    progress_made = True
                    print(
                        f"SUCCESS: Went from {old_nodes} to {new_nodes} nodes",
                        file=sys.stderr,
                    )
                if new_inps > old_inps:
                    progress_made = True
                    print(
                        f"SUCCESS: Went from {old_inps} to {new_inps} inputs",
                        file=sys.stderr,
                    )
                if new_outs < old_outs:
                    progress_made = True
                    print(
                        f"SUCCESS: Went from {old_outs} to {new_outs} outputs",
                        file=sys.stderr,
                    )

                if not progress_made:
                    raise RuntimeError("Success raised but no progress made?")

                if not graph_fails(new_state.graph, new_state.inps):
                    print(
                        "WARNING: Something went wrong, not applying this minification",
                        file=sys.stderr,
                    )
                    return None
                return new_state
            else:
                print(f"FAIL: {name}", file=sys.stderr)
            return None

        return new_func

    def register_strategy(
        name: str,
    ) -> Callable[
        [Callable[[fx.Graph, Sequence[torch.Tensor], int], ReproState | None]],
        Callable[[ReproState, int], ReproState | None],
    ]:
        # [FUNCTION DOC] HELPER: Returns a decorator that registers a strategy under the given name
        # by partially applying the strategy wrapper.
        return partial(_register_strategy, name=name)

    @register_strategy("Truncate suffix")
    def remove_suffix(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Strategy "Truncate suffix". Copies nodes one by one into a new graph;
        # at indices that are multiples of granularity but not of twice granularity (the latter
        # were tried at a coarser level), temporarily makes the current node the sole output and
        # checks whether the shorter graph still fails. Returns the first failing truncated state,
        # or None if none fails.
        tested: set[int] = set()
        new_graph = fx.Graph()
        env: dict[fx.Node, fx.Node] = {}
        for idx, node in enumerate(cur_graph.nodes):
            new_node = new_graph.node_copy(node, lambda x: env[x])
            if node.op not in ["placeholder", "output"]:
                # If idx is divisible by (granularity * 2), it would have been checked already.
                if (
                    idx % granularity == 0
                    and (idx % (granularity * 2) != 0)
                    and idx not in tested
                ):
                    output_node = new_graph.output((new_node,))
                    if len(new_graph.nodes) < len(cur_graph.nodes) and graph_fails(
                        new_graph, cur_inps
                    ):
                        return ReproState(new_graph, cur_inps)
                    else:
                        tested.add(idx)
                        new_graph.erase_node(output_node)
            env[node] = new_node
        return None

    @register_strategy("Remove outputs")
    def remove_outputs(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Strategy "Remove outputs". Works on graphs whose output is a
        # sequence of nodes. Sorts outputs by graph position and tries dropping chunks of size
        # half the granularity (at least one) from the output list, returning the first state
        # that still fails. Returns None if there is no output node, a single output, or no
        # removal keeps the failure. Mutates cur_graph (annotates node idx, rewrites output args).
        granularity = max(1, granularity // 2)
        output: fx.Node | None = None
        for idx, node in enumerate(cur_graph.nodes):
            node.idx = idx  # type: ignore[attr-defined]
            if node.op == "output":
                output = node
                break

        if output is None:
            return None

        if isinstance(output.args[0], fx.Node):
            return None

        # output.args[0] is a tuple/list of nodes when returning multiple outputs
        output_args_raw = output.args[0]
        if not isinstance(output_args_raw, (list, tuple)):
            raise AssertionError(
                f"expected output_args_raw to be list or tuple, got {type(output_args_raw)}"
            )
        output_args = sorted(
            output_args_raw,
            key=lambda x: x.idx if isinstance(x, fx.Node) else int(1e9),  # type: ignore[attr-defined]
        )
        if len(output_args) == 1:
            return None

        for idx in range(0, len(output_args), granularity):
            output.args = (output_args[:idx] + output_args[idx + granularity :],)
            if graph_fails(cur_graph, cur_inps):
                return ReproState(cur_graph, cur_inps)
        return None

    def remove_unused_inputs_unchecked(cur_state: ReproState) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Erases placeholder nodes with no users from the state's graph
        # (mutating it) and drops the matching inputs. Returns the new state if any input was
        # removed, otherwise None. Does not test the failure predicate. Raises an assertion
        # error if placeholder and input counts differ.
        cur_graph = cur_state.graph
        cur_inps = cur_state.inps
        ph_nodes = list(get_placeholders(cur_graph))
        if len(ph_nodes) != len(cur_inps):
            raise AssertionError(
                f"len(ph_nodes)={len(ph_nodes)} != len(cur_inps)={len(cur_inps)}"
            )

        new_inps: list[torch.Tensor] = []
        for idx in range(len(ph_nodes)):
            if len(ph_nodes[idx].users) == 0:
                cur_graph.erase_node(ph_nodes[idx])
            else:
                new_inps.append(cur_inps[idx])
        if len(new_inps) < len(cur_inps):
            return ReproState(cur_graph, new_inps)
        return None

    def remove_unused_inputs_checked(cur_state: ReproState) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Removes unused inputs and returns the new state only if the result
        # still fails the predicate; otherwise None.
        new_state = remove_unused_inputs_unchecked(cur_state)
        if new_state is not None and graph_fails(new_state.graph, new_state.inps):
            return new_state
        return None

    def _remove_unused_wrapper(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Adapter giving the checked unused-input removal the strategy
        # signature; granularity is ignored.
        return remove_unused_inputs_checked(ReproState(cur_graph, cur_inps))

    remove_unused_inputs = register_strategy("Remove unused inputs")(
        _remove_unused_wrapper
    )

    @register_strategy("Eliminate dead code")
    def eliminate_dead_code(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Strategy "Eliminate dead code". Removes dead nodes from the graph;
        # if anything was removed and the graph still fails, returns the new state, else None.
        # Granularity is ignored.
        if cur_graph.eliminate_dead_code() and graph_fails(cur_graph, cur_inps):
            return ReproState(cur_graph, cur_inps)
        return None

    def _consolidate_placeholders(
        cur_graph: fx.Graph, inps: list[torch.Tensor]
    ) -> fx.Graph:
        # [FUNCTION DOC] HELPER: Builds a new graph with all placeholders first. Leading load_tensor
        # nodes (before any other non-placeholder node) are converted to placeholders and the
        # tensors they load are appended to inps (mutated). The remaining nodes are copied after,
        # preserving order. Returns the new graph.
        new_graph = fx.Graph()
        env = {}
        seen_non_placeholder = False

        # Move all placeholders to the front; also, if any load_tensor
        # is at the front, convert it into an input (because it can be live
        # all the time)
        for node in cur_graph.nodes:
            if node.op == "placeholder":
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
            elif not seen_non_placeholder and is_load_tensor_node(node):
                new_node = new_graph.placeholder(node.name)
                env[node] = new_node
                inps.append(
                    torch.ops.debugprims.load_tensor.default(*node.args, **node.kwargs)
                )
            else:
                seen_non_placeholder = True

        # Move everyone else
        for node in cur_graph.nodes:
            if node not in env:
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
        return new_graph

    @register_strategy("Delta Debugging")
    def delta_debugging(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Strategy "Delta Debugging". For each window of granularity
        # consecutive node positions, copies the graph and turns window nodes into placeholders or
        # load_tensor nodes using recorded concrete values. If something changed, eliminates dead
        # code, consolidates placeholders, drops unused inputs, and returns the state if it still
        # fails. Returns None if no window works.
        num_nodes = len(cur_graph.nodes)
        for start_range in range(0, num_nodes, granularity):
            is_removing = False
            new_graph = deepcopy_fx_graph(cur_graph)
            new_inps = list(cur_inps[:])
            end_range = min(num_nodes, start_range + granularity)
            for idx in range(start_range, end_range):
                new_node = list(new_graph.nodes)[idx]
                if _convert_node_to_placeholder(new_graph, new_node, new_inps):
                    is_removing = True
            if not is_removing:
                continue
            new_graph.eliminate_dead_code()
            new_graph = _consolidate_placeholders(new_graph, new_inps)
            new_state = remove_unused_inputs_unchecked(ReproState(new_graph, new_inps))
            if new_state is None:
                new_state = ReproState(new_graph, new_inps)
            if graph_fails(new_state.graph, new_state.inps):
                return ReproState(new_state.graph, new_state.inps)

        return None

    @register_strategy("Consolidate Inputs")
    def consolidate_inputs(
        cur_graph: fx.Graph, cur_inps: Sequence[torch.Tensor], granularity: int
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Strategy "Consolidate Inputs". Moves placeholders to the front and
        # converts leading load_tensor nodes to inputs, returning the new state if it still fails.
        # KNOWN ISSUE: the growth check compares the length of the original input sequence with
        # itself (cur_inps, not the extended new_inps), so it is never true and this strategy
        # always returns None.
        old_len = len(cur_inps)
        new_inps = list(cur_inps[:])
        cur_graph = _consolidate_placeholders(cur_graph, new_inps)
        if len(cur_inps) > old_len and graph_fails(cur_graph, new_inps):
            return ReproState(cur_graph, new_inps)
        return None

    failing_state = ReproState(failing_graph, inps)

    def try_granularity(
        failing_state: ReproState, granularity: int, use_non_granular: bool
    ) -> ReproState | None:
        # [FUNCTION DOC] HELPER: Tries the strategy list at one granularity, in order: remove outputs
        # (only if outputs exceed half the node count), then, if use_non_granular, dead code
        # elimination, unused-input removal and input consolidation, then suffix truncation and
        # delta debugging. Returns the first successful new state, or None.
        print(f"Trying granularity {granularity}", file=sys.stderr)

        strategies = []
        num_nodes = len(failing_state.graph.nodes)
        num_outputs = len(get_outputs(failing_state.graph))
        if num_outputs > num_nodes // 2:
            strategies += [remove_outputs]

        if use_non_granular:
            strategies += [
                eliminate_dead_code,
                remove_unused_inputs,
                consolidate_inputs,
            ]

        strategies += [remove_suffix, delta_debugging]

        for strategy in strategies:
            new_state = strategy(failing_state, granularity)
            if new_state is not None:
                return new_state
        return None

    while True:
        dump_state(fx.GraphModule(fail_f, failing_state.graph), failing_state.inps)
        granularity = int(2 ** (math.floor(math.log2(len(failing_state.graph.nodes)))))
        if max_granularity is not None:
            granularity = min(max_granularity, granularity)
        new_state = try_granularity(failing_state, granularity, use_non_granular=True)
        if new_state is not None:
            failing_state = new_state
            continue

        granularity //= 2
        has_progress = False
        while granularity >= 1:
            new_state = try_granularity(
                failing_state, granularity, use_non_granular=False
            )
            if new_state is not None:
                failing_state = new_state
                has_progress = True
                break
            granularity //= 2
        if has_progress:
            continue

        new_state = remove_outputs(failing_state, 1)
        if new_state is not None:
            failing_state = new_state
            continue

        break

    if not graph_fails(failing_state.graph, failing_state.inps):
        raise RuntimeError("Uh oh, something went wrong :( Final graph is not failing")

    print(f"Made {num_queries} queries", file=sys.stderr)
    failing_fx = fx.GraphModule(fail_f, failing_state.graph)

    # If XLA debugging environment is enabled, create minified HLO graph as well
    if "XLA_HLO_DEBUG" in os.environ:
        create_minified_hlo_graph(failing_fx, failing_state.inps)

    dump_state(failing_fx, failing_state.inps)
    print("Wrote minimal repro out to repro.py", file=sys.stderr)
    return failing_fx, failing_state.inps
