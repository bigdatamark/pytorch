# [MODULE DOC] Base machinery for splitting an fx GraphModule into accelerator ("acc") and CPU submodules
# according to an operator-support oracle. Contains settings (command line plus defaults), a node event
# tracker for debugging, FxNetAccNodesFinder (decides which nodes may run on acc), Subgraph/SplitResult
# records, generate_inputs_for_submodules and the _SplitterBase driver. Import-time side effects: reads
# the environment variables FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH (dump file prefix, default
# _fx_net_tracker) and FX_NET_ACC_SPLITTER_TRACKER_MODE (0 to 3, default 0) into module constants.
import argparse
import copy
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, IO, Literal, NamedTuple

import torch
from torch._logging import trace_structured
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from torch.fx.passes.graph_manipulation import get_size_of_node

from .graph_drawer import FxGraphDrawer
from .operator_support import get_node_target, OperatorSupportBase
from .shape_prop import ShapeProp
from .split_utils import move_non_tensor_nodes_on_boundary, split_by_tags
from .tools_common import (
    CALLABLE_NODE_OPS,
    FxNetAccFusionsFinder,
    is_node_output_tensor,
    NodeList,
    NodeSet,
    Tensors,
)


__all__ = [
    "FxNetAccNodesFinder",
    "FxNetSplitterInternalError",
    "Subgraph",
    "SplitResult",
    "generate_inputs_for_submodules",
    "NodeEvent",
    "NodeEventTracker",
]

DEFAULT_MIN_ACC_MODULE_SIZE = 1
DEFAULT_SKIP_FUSION = False
DEFAULT_ALLOW_NON_TENSOR = False

# ENV var and constants for node tracker

TRACKER_DUMP_PATH = "_fx_net_tracker"
NODES_SUFFIX = "_nodes.txt"
ALL_SUFFIX = "_all.txt"

ENV_FX_NET_ACC_SPLITTER_TRACKER_MODE = "FX_NET_ACC_SPLITTER_TRACKER_MODE"
ENV_FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH = "FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH"
ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES = (
    "FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES"
)

DUMP_PREFIX = os.environ.get(
    ENV_FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH, TRACKER_DUMP_PATH
)

"""
Different modes of the event tracker for local debugging:
"0": No local dumps. Information available by setting breakpoints and visually inspect in pdb.
"1": Dump all events to DUMP_PREFIX_all.txt
"2": In addition to events dump, track nodes specified by ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES
     recursively and dump to DUMP_PREFIX_nodex.txt
"3": In addition to events dump, track all nodes with more than 1 event recursively and dump to DUMP_PREFIX_nodex.txt
In addition to the above local dumps, tracker is always enabled and dumps via trace_structured.
"""
# pyrefly: ignore [bad-assignment]
TRACKER_MODE: Literal["0", "1", "2", "3"] = os.environ.get(
    ENV_FX_NET_ACC_SPLITTER_TRACKER_MODE, "0"
)  # type: ignore[assignment]


class _SplitterSettingBase:
    # [CLASS DOC] HELPER: Plain settings holder for the splitter: min_acc_module_size, skip_fusion,
    # allow_non_tensor, max_acc_splits and move_non_tensor_nodes_on_boundary. Values come from command line
    # flags when given (truthy), else from constructor arguments.
    def __init__(
        self,
        min_acc_module_size: int = DEFAULT_MIN_ACC_MODULE_SIZE,
        skip_fusion: bool = DEFAULT_SKIP_FUSION,
        allow_non_tensor: bool = DEFAULT_ALLOW_NON_TENSOR,
        max_acc_splits: int = -1,
        move_non_tensor_nodes_on_boundary: bool = False,
    ) -> None:
        # [FUNCTION DOC] HELPER: Parses known command line flags from the process argument list (unknown ones are
        # ignored) and stores the settings, preferring a truthy flag value over the constructor argument.
        # Side effect: reads global process arguments.
        # KNOWN ISSUE: the max-acc-splits flag is defined but its parsed value is never used; the constructor
        # argument is always stored. A flag value of zero is also treated as unset.
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--min-acc-module-size",
            "--min_acc_module_size",
            required=False,
            type=int,
            help="Minimum size limit of an accelerator subgraph.",
        )
        parser.add_argument(
            "--max-acc-splits",
            "--max_acc_splits",
            required=False,
            type=int,
            help="Enforce a maximum number of split subgraphs.",
        )
        parser.add_argument(
            "--skip-fusion",
            "--skip_fusion",
            default=False,
            action="store_true",
            help="If true then no fusion groups. Fusion group is used to "
            "enforce no non-tensor data flow between submodules. If we don't "
            "have this constraint, setting this to false is recommended as it "
            "can reduce overhead.",
        )
        parser.add_argument(
            "--allow-non-tensor",
            "--allow_non_tensor",
            default=False,
            action="store_true",
            help="For some backends non-tensor data flow between cpu and them "
            "are not allowed. Therefore, if a node supported by accelerator but "
            "it has non-tensor inputs or outputs to a cpu node we would want to "
            "consider it as a cpu node during splitting. However, for some backends "
            "we might not care about non-tensor data flow and we can set this option "
            "to true to disable the functionality that prevent non-tensor data flow.",
        )
        parser.add_argument(
            "--move-non-tensor-nodes-on-boundary",
            "--move_non_tensor_nodes_on_boundary",
            required=False,
            action="store_true",
            help="AOTI does not support non-tensor nodes on acc->acc, acc->gpu and gpu->acc boundary. "
            "For non-tensor nodes on acc->acc boundary and acc->gpu, we move the nodes from upstream to downstream. "
            "For non-tensor nodes on gpu->acc boundary, it is handled by the pre-split process. "
            "(by method reduce_acc_nodes_non_tensor_input). ",
        )
        args, _unknown = parser.parse_known_args()

        self.min_acc_module_size: int = (
            args.min_acc_module_size
            if args.min_acc_module_size
            else min_acc_module_size
        )
        self.skip_fusion: bool = args.skip_fusion if args.skip_fusion else skip_fusion
        self.allow_non_tensor: bool = (
            args.allow_non_tensor if args.allow_non_tensor else allow_non_tensor
        )
        self.max_acc_splits: int = max_acc_splits
        self.move_non_tensor_nodes_on_boundary: bool = (
            args.move_non_tensor_nodes_on_boundary
            if args.move_non_tensor_nodes_on_boundary
            else move_non_tensor_nodes_on_boundary
        )


@compatibility(is_backward_compatible=False)
class NodeEvent:
    # [CLASS DOC] Record of one event on a node during splitting: source (the node), desc (text in the form
    # event type, bar, explanation) and dep (optional node that caused it). Immutable by convention.
    """
    An event in graph split that happened on a node.
    source: Subject of the event
    desc: readable description
    dep: Optional dependency, usually the node that caused the event.
    """

    def __init__(
        self, source: torch.fx.Node, desc: str, dep: torch.fx.Node | None = None
    ) -> None:
        # [FUNCTION DOC] Stores the source node, description and optional causing node.
        self.source = source
        self.desc = desc
        self.dep = dep

    def to_str(self) -> str:
        # [FUNCTION DOC] Returns the text form: source name, colon, description, then the name of the causing node
        # or a hash sign if none.
        # source: The name of the subject of the event.
        # desc: description of the event, in the format of <event_type>|<explanation>
        # dep: The name of the cause of this event, which is another node, or #
        # if it's caused by the subject node
        return f"{self.source.name}: {self.desc} {self.dep.name if self.dep else '#'}"


@compatibility(is_backward_compatible=False)
class NodeEventTracker:
    # [CLASS DOC] Collects NodeEvent records during node finding. State: tracker_mode (0 to 3), dump_prefix for
    # output files, events (ordered list), node_events (node name to indices into events) and writer (print by
    # default). Can print per node, print all, convert to a dict and dump to structured trace and local files.
    """
    Tracks node events during the splitter execution.
    """

    def __init__(self, tracker_mode: int, dump_prefix: str) -> None:
        # [FUNCTION DOC] Stores the mode and file prefix and initializes empty event storage with print as writer.
        self.tracker_mode = tracker_mode
        self.dump_prefix = dump_prefix
        # list of events
        self.events = []
        # dict from node name to event index
        self.node_events = {}
        self.writer = print

    def add(
        self, node: torch.fx.Node, desc: str, dep: torch.fx.Node | None = None
    ) -> None:
        # [FUNCTION DOC] Appends a new event for the node and records its index under the node name.
        """
        Add a new event to the tracker.
        """
        event = NodeEvent(node, desc, dep)
        self.events.append(event)
        if node.name not in self.node_events:
            self.node_events[node.name] = []
        self.node_events[node.name].append(len(self.events) - 1)

    def print_node(
        self,
        node_name: str,
        recursive: bool = False,
        tab: str = "",
        writer: Callable[[str], object] | None = None,
    ) -> None:
        # [FUNCTION DOC] Writes each event of the named node using the writer (default: the tracker writer), each
        # line prefixed by tab. When recursive, also prints events of the causing nodes with a deeper prefix.
        """
        Print a node and its events.
        @param recursive: if True, print nodes that caused the events on this current node.
        @param tab: Indentation for dependencies.
        @param writer: function to write to file. If None, use print.
        """
        if not writer:
            writer = self.writer
        for idx in self.node_events.get(node_name, []):
            event = self.events[idx]
            writer(tab + event.to_str())
            if recursive and event.dep is not None:
                self.print_node(
                    event.dep.name, recursive=True, tab="| " + tab, writer=writer
                )

    def to_dict(self) -> dict[str, list[str]]:
        # [FUNCTION DOC] Returns a mapping from node name to the list of its event strings, in recording order.
        """
        Create dict dump on all events.
        """
        ret: dict[str, list[str]] = {}
        for name in self.node_events:
            ret[name] = []
            for idx in self.node_events.get(name, []):
                event = self.events[idx]
                ret[name].append(event.to_str())
        return ret

    def print_all(self, writer: Callable[[str], object] | None = None) -> None:
        # [FUNCTION DOC] Writes a header line and the events (non-recursive, indented) of every tracked node using
        # the writer (default: the tracker writer).
        """
        Print all nodes in a list.
        @param writer: function to write to file. If None, use print.
        """
        if not writer:
            writer = self.writer
        for name in self.node_events:
            writer(f"Node: {name}:")
            self.print_node(name, recursive=False, tab="  ", writer=writer)

    def dump(self) -> None:
        # [FUNCTION DOC] Emits all events as a JSON artifact through structured tracing (always). Local file side
        # effects: mode 1 or higher writes all events to the dump prefix plus the all suffix; mode 2 also writes
        # recursive dumps for nodes named in the tracked-nodes environment variable (comma separated); mode 3
        # does so for all nodes with more than one event. Files are overwritten.
        """
        Function to be invoked at the end of the finder execution to printout tracked events specified by the mode.
        """
        # dump via trace_structured
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_net_acc_splitter_finder_events",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(self.to_dict()),
        )

        def make_writer(f: IO[str]) -> Callable[[str], int]:
            # [FUNCTION DOC] HELPER: Returns a function that writes a line (text plus newline) to the given file.
            def fn(x: str) -> int:
                # [FUNCTION DOC] HELPER: Writes the text and a newline to the captured file; returns the character count.
                return f.write(x + "\n")

            return fn

        # Mode 0: no local dump
        # Mode >=1: Dump all events to file
        if self.tracker_mode >= 1:
            with open(self.dump_prefix + ALL_SUFFIX, "w") as f:
                self.print_all(make_writer(f))

        def dump_selected_nodes(nodes: list[str]) -> None:
            # [FUNCTION DOC] HELPER: Opens the nodes dump file for writing and, for each node name, writes a banner,
            # the recursive event listing and a closing banner.
            with open(self.dump_prefix + NODES_SUFFIX, "w") as f:
                writer = make_writer(f)
                for node_name in nodes:
                    writer(f"===== Tracking node {node_name} =====")
                    self.print_node(node_name, recursive=True, tab="|-", writer=writer)
                    writer(f"===== End of tracking node {node_name} =====")

        # Mode 2: Dump specific nodes in recursive manner.
        # Mode 3: Dump all nodes with more than 1 event in recursive manner.
        if self.tracker_mode == 2 or self.tracker_mode == 3:
            nodes = (
                os.environ.get(ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES, "").split(
                    ","
                )
                if self.tracker_mode == 2
                else [
                    name for name, events in self.node_events.items() if len(events) > 1
                ]
            )
            dump_selected_nodes(nodes)


@compatibility(is_backward_compatible=False)
class FxNetAccNodesFinder:
    # [CLASS DOC] Computes the set of nodes that can run on the accelerator. State: module, operator_support,
    # allow_non_tensor, acc_nodes (current accepted set) and tracker (event recorder). Calling it starts from
    # callable nodes the operator-support oracle accepts and, unless non-tensor flow is allowed, removes nodes
    # with non-tensor data crossing to CPU nodes.
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: OperatorSupportBase,
        allow_non_tensor: bool,
    ) -> None:
        # [FUNCTION DOC] Stores the module, oracle and flag, creates an empty acc node set and a tracker using the
        # module level mode and dump prefix.
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor
        self.acc_nodes: NodeSet = set()

        self.tracker = NodeEventTracker(int(TRACKER_MODE), DUMP_PREFIX)

    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList) -> None:
        # [FUNCTION DOC] Worklist propagation: for each CPU node taken from the list (consumed, mutated), removes
        # its users from the acc set (recording events) and queues those users whose output is not a tensor, so
        # exclusion spreads downstream. Mutates acc_nodes.
        """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
        while cpu_worklist:
            node = cpu_worklist.pop(0)

            for user in node.users:
                if user in self.acc_nodes:
                    self.acc_nodes.remove(user)
                    self.tracker.add(user, "acc_del|user_of_new_cpu_node", node)
                    if not is_node_output_tensor(user):
                        self.tracker.add(user, "new_cpu_node|non_tensor_output")
                        cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self) -> None:
        # [FUNCTION DOC] Finds callable nodes outside the acc set whose output is not known to be a tensor, records
        # them as new CPU nodes and removes their downstream acc nodes transitively. Mutates acc_nodes.
        """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
        non_tensor_cpu_nodes: NodeList = []

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            self.tracker.add(node, "new_cpu_node|callable_non_tensor_input")
            non_tensor_cpu_nodes.append(node)

        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self) -> None:
        # [FUNCTION DOC] Repeatedly removes acc nodes that produce non-tensor output and have a user outside the
        # acc set, then propagates the exclusion downstream, until no more nodes are removed. Mutates acc_nodes.
        """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
        while True:
            new_cpu_nodes: NodeList = []

            for acc_node in self.acc_nodes:
                if is_node_output_tensor(acc_node):
                    continue
                for user in acc_node.users:
                    if user not in self.acc_nodes:
                        new_cpu_nodes.append(acc_node)
                        self.tracker.add(
                            acc_node, "acc_del|non_tensor_output_with_cpu_user", user
                        )
                        break

            if not new_cpu_nodes:
                break

            for new_cpu_node in new_cpu_nodes:
                self.acc_nodes.remove(new_cpu_node)

            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)

    def __call__(self) -> NodeSet:
        # [FUNCTION DOC] Computes and returns the acc node set: resets it, adds each callable node approved by the
        # oracle (recording events), applies the non-tensor reductions unless allowed, then dumps tracker output
        # (possible local file writes). Returns the set (also stored on the instance).
        submodules = dict(self.module.named_modules())
        self.acc_nodes = set()
        for n in self.module.graph.nodes:
            if n.op not in CALLABLE_NODE_OPS:
                self.tracker.add(n, "init_cpu|not_callable")
                continue
            if not self.operator_support.is_node_supported(submodules, n):
                self.tracker.add(n, "init_cpu|operator_support")
                continue

            self.tracker.add(n, "init_acc|callable_and_operator_supported")
            self.acc_nodes.add(n)

        if not self.allow_non_tensor:
            self.reduce_acc_nodes_non_tensor_input()
            self.reduce_acc_nodes_non_tensor_output()
        self.tracker.dump()
        return self.acc_nodes


@compatibility(is_backward_compatible=False)
class FxNetSplitterInternalError(Exception):
    # [CLASS DOC] Exception raised when the splitter reaches an inconsistent internal state (for example an
    # empty subgraph or already tagged node).
    pass


@compatibility(is_backward_compatible=False)
@dataclass
class Subgraph:
    # [CLASS DOC] Dataclass for one contiguous group of nodes: is_acc (runs on accelerator or CPU), nodes (list
    # in execution order) and optional device_ordinal.
    is_acc: bool
    nodes: NodeList
    device_ordinal: int | None = None


@compatibility(is_backward_compatible=False)
class SplitResult(NamedTuple):
    # [CLASS DOC] Named tuple result of splitting: split_module (the root module with submodules),
    # submodule_inputs (mapping of submodule name to captured example inputs) and non_acc_submodule_prefix.
    """
    Stores the results of the splitter.

    Attributes:
        split_module: root module after splitting.
        submodule_inputs: a dict that maps submodule name to its inputs.
        non_acc_submodule_prefix: the prefix for non acc submodules. For
            acc submodule the prefix is always "_run_on_acc_".
    """

    split_module: torch.fx.GraphModule
    submodule_inputs: dict[str, Any]
    non_acc_submodule_prefix: str


@compatibility(is_backward_compatible=False)
def generate_inputs_for_submodules(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    target_submodules: Iterable[str],
    deepcopy: bool = False,
) -> dict[str, Any]:
    # [FUNCTION DOC] Runs the model once under no-grad with forward pre-hooks on the named target submodules
    # (skipping scripted modules) to capture each submodule input tuple (deep-copied if requested). Returns a
    # mapping from submodule name to inputs. Hooks are always removed; exceptions from the run propagate.
    # KNOWN ISSUE: the lookup is by module object, so if two names share one module only one name is kept;
    # submodules never called are absent from the result.
    """
    Generate inputs for targeting submodules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """

    handles = []
    results: dict[str, Any] = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module: torch.nn.Module, module_inputs: tuple[Any, ...]) -> None:
        # [FUNCTION DOC] HELPER: Forward pre-hook that stores the (optionally deep-copied) inputs under the name of
        # the module.
        results[submodule_to_names[module]] = (
            copy.deepcopy(module_inputs) if deepcopy else module_inputs
        )

    for name, mod in model.named_modules():
        if name in target_submodules:
            if not isinstance(mod, torch.jit.ScriptModule):
                handles.append(mod.register_forward_pre_hook(pre_forward))

    def clean_up_handles() -> None:
        # [FUNCTION DOC] HELPER: Removes all registered hook handles.
        for h in handles:
            h.remove()

    try:
        with torch.no_grad():
            model(*inputs)
    except Exception as e:
        clean_up_handles()
        raise e

    clean_up_handles()
    return results


class _SplitterBase:
    # [CLASS DOC] Driver that splits a GraphModule into acc and CPU submodules. State: module, settings,
    # operator_support, sample_input, acc_nodes, fusions (node to fusion group set), deps (node to direct
    # dependencies), tags (one per subgraph), submodule naming and tuple-return option. Typical flow: build
    # subgraphs by traversal, drop small acc subgraphs, tag nodes, split by tags. Subclasses may override
    # lowering and culprit-finding hooks used by the preview.
    """
    Splits a GraphModule into sub-GraphModules for execution on CPU or the accelerator.
    Output is a GraphModule with supported and unsupported operators grouped into as few sub-GraphModules as possible.
    Assumes that only "call_module", "call_function" and "call_method" from FX IR can potentially be executed on the accelerator.

    Given the following graph:
          ==> b ==>
        //         \\
       a             d
        \\         //
          ==> c ==>

    class SimpleModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = b + c
            return d

    and providing "operator_support" that indicates that 'b' and 'c' can be executed on the accelerator,
    we will get the following split result:

    main:
    def forward(self, a):
        run_on_acc_0_0 = self._run_on_acc_0_0(a)
        getitem = run_on_acc_0_0[0]
        getitem_1 = run_on_acc_0_0[1]
        run_on_cpu_1_1 = self._run_on_cpu_1_1(getitem, getitem_1)
        return run_on_cpu_1_1

    _run_on_acc_0_0:
    def forward(self, a):
        sin_1 = torch.sin(a)
        cos_1 = torch.cos(a)
        return (sin_1, cos_1)

    _run_on_cpu_1_1:
    def forward(self, sin_1, cos_1):
        add_1 = sin_1 + cos_1
        return add_1
    """

    # PCIe bandwidth for the backend, default to 100 GB/s
    PCIe_BW = 100 * 2**30

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Sequence[Any],
        operator_support: OperatorSupportBase,
        settings: _SplitterSettingBase,
        non_acc_submodule_name: str = "_run_on_cpu_",
        return_tuple: bool = False,
        nodes_finder: FxNetAccNodesFinder | None = None,
    ) -> None:
        # [FUNCTION DOC] Checks module is a GraphModule (AssertionError otherwise), runs shape propagation on the
        # sample input, finds acc nodes (using the given finder or a default), finds fusion groups unless fusion is
        # skipped, optionally merges overlapping fusions, builds dependency maps with fusion adjustments and
        # initializes naming and tag state. Mutates node metadata of the module.
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected GraphModule, got {type(module)}")

        self.module = module
        ShapeProp(self.module).propagate(*sample_input)

        self.settings = settings
        self.operator_support = operator_support
        self.sample_input = sample_input
        if nodes_finder is None:
            nodes_finder = FxNetAccNodesFinder(
                self.module, self.operator_support, self.settings.allow_non_tensor
            )
        self.acc_nodes = nodes_finder()

        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            self.fusions = FxNetAccFusionsFinder(module, self.acc_nodes)()
            self._merge_overlapping_fusions()

        # Modify deps to add more deps for fused nodes
        self.deps = self.find_deps()
        self.update_deps_for_fusions()

        self.non_acc_submodule_name = non_acc_submodule_name
        self._node_submodule_map: dict[str, str] = {}
        self._return_tuple = return_tuple

        self.tags: list[str] = []

    # ===============================================================
    # Helpers for ctor and initial state
    # ===============================================================

    def get_node_submodule_map(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the mapping from node name to the submodule tag filled during tagging.
        """Returns a map from node name to submodule name, e.g.
        node: main_module_impl_impl_over_arch_unary_multiple_embedding
          _pooling_embedding_pooling_sparse_entity_equivalence_key
          _proxy_embedding_bag
        maps to submodule name of: _run_on_acc_1
        """
        return self._node_submodule_map

    def find_deps(self) -> dict[torch.fx.Node, NodeSet]:
        # [FUNCTION DOC] Builds a mapping from each callable node to the set of callable nodes it directly depends
        # on (non-callable producers and users that are output nodes are ignored).
        """
        Builds a graph of node dependencies. Leaf nodes don't have any
        dependencies and the "output" node doesn't have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        """
        deps: dict[torch.fx.Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            for user in node.users:
                if user.op != "output":
                    deps[user].add(node)
        return deps

    def update_deps_for_fusions(self) -> None:
        # [FUNCTION DOC] Mutates deps so that fused nodes share the outer dependencies of their group and outer
        # users of fused nodes depend on the fused node.
        """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
        for node in self.fusions:
            fusion = self.fusions[node]
            for fused_neighbor in fusion:
                self.deps[node].update(self.deps[fused_neighbor] - fusion)

                for user in fused_neighbor.users:
                    if user not in fusion:
                        self.deps[user].add(node)

    def _merge_overlapping_fusions(self) -> None:
        # [FUNCTION DOC] Merges fusion groups that share nodes using union-find and rewrites fusions so each node
        # points to its merged group. Only active when the environment variable
        # _SPLITTER_MERGE_OVERLAPPING_FUSIONS equals 1; otherwise returns immediately. Mutates fusions.
        """
        Merge fusion groups that share nodes.

        FxNetAccFusionsFinder can produce overlapping fusion groups when a
        node is absorbed into multiple groups during expansion. When
        update_deps_for_fusions() later propagates deps, shared nodes act
        as bridges between groups, creating bidirectional dependency cycles
        that crash the splitter with "Subgraph can't be empty".

        This method detects overlapping groups via union-find and merges
        them into single groups before dep propagation.
        """
        if os.environ.get("_SPLITTER_MERGE_OVERLAPPING_FUSIONS", "0") != "1":
            return

        if not self.fusions:
            return

        # Collect unique groups by identity.
        unique_groups: dict[int, NodeSet] = {}
        for group in self.fusions.values():
            unique_groups[id(group)] = group

        if len(unique_groups) <= 1:
            return

        # Map each node to all group IDs it belongs to.
        node_to_gids: dict[torch.fx.Node, list[int]] = defaultdict(list)
        for gid, group in unique_groups.items():
            for node in group:
                node_to_gids[node].append(gid)

        # Union-find: merge groups that share nodes.
        parent: dict[int, int] = {gid: gid for gid in unique_groups}

        def find(x: int) -> int:
            # [FUNCTION DOC] HELPER: Union-find root lookup with path halving.
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        needs_merge = False
        for gids in node_to_gids.values():
            if len(gids) > 1:
                root = find(gids[0])
                for i in range(1, len(gids)):
                    other = find(gids[i])
                    if other != root:
                        parent[other] = root
                        needs_merge = True

        if not needs_merge:
            return

        # Build merged groups.
        merged: dict[int, NodeSet] = defaultdict(set)
        for gid, group in unique_groups.items():
            merged[find(gid)].update(group)

        # Rebuild self.fusions so every node points to its merged group.
        for merged_group in merged.values():
            for node in merged_group:
                self.fusions[node] = merged_group

    # ===============================================================
    # Helpers for preview
    # ===============================================================

    def _lower_model_to_backend(
        self, mod: torch.fx.GraphModule, inputs: Tensors
    ) -> torch.nn.Module:
        # [FUNCTION DOC] Hook for subclasses: lowers a module to the backend. Default returns the module unchanged.
        """
        Lower the model to a backend.
        """

        return mod

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors) -> str:
        # [FUNCTION DOC] Hook for subclasses: locate the node responsible for a lowering or run failure. Default
        # returns a message saying it is not implemented.
        """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """

        return "Unable to find a culprit because _find_culprit() function is not implemented."

    def _draw_graph_based_on_node_support(
        self, mod: torch.fx.GraphModule, supported_nodes: NodeList
    ) -> None:
        # [FUNCTION DOC] HELPER: Writes node_support.dot in the current directory, a graph drawing in which
        # supported nodes are green, unsupported callable nodes red and others light blue. Side effect: file write.
        color_map = {
            "default": "AliceBlue",
            "supported": "chartreuse1",
            "unsupported": "crimson",
        }

        class CustomDrawer(FxGraphDrawer):
            # [CLASS DOC] HELPER: Graph drawer that colors nodes by support status using the enclosing color map.
            def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]:
                # [FUNCTION DOC] HELPER: Returns the base node style with its fill color set by support status.
                template = super()._get_node_style(node)
                if node in supported_nodes:
                    template["fillcolor"] = color_map["supported"]
                elif node.op in CALLABLE_NODE_OPS:
                    template["fillcolor"] = color_map["unsupported"]
                else:
                    template["fillcolor"] = color_map["default"]

                return template

        drawer = CustomDrawer(mod, "node_support", ignore_getattr=True)
        dot_graph = drawer.get_main_dot_graph()
        # pyre-fixme[16]: `pydot.Dot` has no attribute `write_raw`.
        dot_graph.write_raw("node_support.dot")  # type: ignore[attr-defined]

    def node_support_preview(self, dump_graph: bool = False) -> str:
        # [FUNCTION DOC] Builds and prints a report of supported and unsupported node target types with the dtypes
        # of their tensor arguments and keyword arguments. If dump_graph, also writes node_support.dot. Returns the
        # report text. Side effects: prints.
        submodules = dict(self.module.named_modules())

        supported_nodes: NodeList = []
        supported_node_types = defaultdict(set)
        unsupported_node_types = defaultdict(set)

        def get_dtype(arg: torch.fx.Node) -> torch.dtype | None:
            # [FUNCTION DOC] HELPER: Returns the dtype from the tensor_meta of the node, or None.
            tensor_meta = arg.meta.get("tensor_meta")
            return getattr(tensor_meta, "dtype", None)

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            target = get_node_target(submodules, node)

            # Store dtype of arg in node.args. If arg doesn't have dtype, i.e. not a tensor, we'll store None.
            arg_dtypes = [
                get_dtype(arg) if isinstance(arg, torch.fx.Node) else None
                for arg in node.args
            ]

            # Find last non-None element. If all elements are None, return max_len.
            last_index = len(arg_dtypes) - next(
                (
                    i
                    for i, dtype in enumerate(reversed(arg_dtypes))
                    if dtype is not None
                ),
                len(arg_dtypes),
            )

            # Strip None elements at the end.
            arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
            kwarg_dtypes_tuple = tuple(
                (k, get_dtype(arg))
                for k, arg in node.kwargs.items()
                if isinstance(arg, torch.fx.Node)
            )

            if self.operator_support.is_node_supported(submodules, node):
                supported_nodes.append(node)
                supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
            else:
                unsupported_node_types[target].add(
                    (arg_dtypes_tuple, kwarg_dtypes_tuple)
                )

        if dump_graph:
            self._draw_graph_based_on_node_support(self.module, supported_nodes)

        reports = "\nSupported node types in the model:\n"
        for t, dtypes in supported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        reports += "\nUnsupported node types in the model:\n"
        for t, dtypes in unsupported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        print(reports)

        # Return reports for testing purpose
        return reports

    def split_preview(self, dump_graph: bool = False) -> str:
        # [FUNCTION DOC] Previews the split: builds subgraphs, removes small acc ones, reports counts, tags and
        # splits the module (tags removed), optionally writes dot files per graph, then for each acc submodule
        # captures its inputs, estimates input and output bytes and PCIe bound qps, tries lowering and running it
        # and reports failures using the culprit hook. Returns and prints the report. Side effects: file writes and
        # tag mutation on nodes.
        reports = ""
        subgraphs = self.put_nodes_into_subgraphs()
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f"Before removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:"
        reports += f" {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n"

        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f"After removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:"
        reports += f" {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n"

        for i, subgraph in enumerate(subgraphs):
            reports += (
                f"_run_on_acc_{i}: "
                if subgraph.is_acc
                else f"{self.non_acc_submodule_name}{i}: "
            )
            reports += f"{len(subgraph.nodes)} node(s)\n"

        self.tag(subgraphs)
        split_mod = self.split(remove_tag=True)
        split_mod.eval()

        if dump_graph:
            drawer = FxGraphDrawer(split_mod, "preview", ignore_getattr=True)
            dot_graphs = drawer.get_all_dot_graphs()
            for name, dot_graph in dot_graphs.items():
                # pyre-fixme[16]: `pydot.Dot` has no attribute `write_raw`.
                dot_graph.write_raw(f"{name}.dot")  # type: ignore[attr-defined]

        max_qps: float = self.PCIe_BW
        bottleneck_module = ""

        for node in split_mod.graph.nodes:
            if node.op == "call_module" and "acc" in node.target:
                reports += f"\nProcessing acc submodule {node.target}\n"

                submod = getattr(split_mod, node.target)

                def get_submod_inputs(
                    main_mod: torch.fx.GraphModule,
                    submod: torch.fx.GraphModule,
                    example_inputs: Sequence[Any],
                ) -> tuple[Any, ...]:
                    # [FUNCTION DOC] HELPER: Captures the input tuple of the submodule by a forward pre-hook while running the
                    # main module on the example inputs. Raises AssertionError if nothing was captured. Returns the inputs.
                    sub_inputs: tuple[Any, ...] | None = None

                    def get_inputs(
                        self: torch.nn.Module, inputs: tuple[Any, ...]
                    ) -> None:
                        # [FUNCTION DOC] HELPER: Pre-hook that stores the inputs in the enclosing variable.
                        nonlocal sub_inputs
                        sub_inputs = inputs

                    handle = submod.register_forward_pre_hook(get_inputs)
                    main_mod(*example_inputs)
                    handle.remove()
                    if sub_inputs is None:
                        raise AssertionError("Forward pre-hook did not capture inputs")
                    return sub_inputs

                submod_inputs = get_submod_inputs(split_mod, submod, self.sample_input)
                ShapeProp(submod).propagate(*submod_inputs)

                total_input_bytes = 0
                total_output_bytes = 0

                reports += "Checking inputs...\n"
                for n in submod.graph.nodes:
                    if n.op == "placeholder":
                        if not is_node_output_tensor(n):
                            reports += f"Input {n.name} is not a tensor, this might cause problems during lowering!\n"
                        else:
                            total_input_bytes += get_size_of_node(submod, n)[0]
                    if n.op == "output":
                        output_node = n

                reports += "Checking outputs...\n"

                def get_bytes(node: torch.fx.Node) -> None:
                    # [FUNCTION DOC] HELPER: Adds the byte size of an output node to the output total, or a warning line to
                    # the report if it is not a tensor.
                    nonlocal total_output_bytes
                    nonlocal reports
                    if not is_node_output_tensor(node):
                        reports += f"Output {node.name} is not a tensor, this might cause problems during lowering!\n"
                    else:
                        total_output_bytes += get_size_of_node(submod, node)[0]

                map_arg(output_node.args, get_bytes)  # type: ignore[possibly-undefined]
                qps = self.PCIe_BW / max(total_input_bytes, total_output_bytes)
                reports += f"Total input size in bytes is {total_input_bytes}, total output size in bytes is {total_output_bytes},"
                reports += f" theoretical max qps (bounds by PCIe bandwidth) for this submodule is {qps}.\n"

                if qps < max_qps:
                    max_qps = qps
                    bottleneck_module = node.target

                try:
                    lowered_submod = self._lower_model_to_backend(submod, submod_inputs)
                except RuntimeError:
                    reports += "Run into an error during lowering!\n"
                    reports += self._find_culprit(submod, submod_inputs)
                    continue

                try:
                    lowered_submod(*submod_inputs)
                except RuntimeError:
                    reports += "Run into an error during inference!\n"
                    reports += self._find_culprit(submod, submod_inputs)
                else:
                    reports += "Lowering and running succeed!\n"

        reports += f"\nTheoretical max qps (bounds by PCIe bandwidth) for this model is {max_qps},"
        reports += f" bottleneck is submodule {bottleneck_module}."
        print(reports)

        # return the reports for testing purposes
        return reports

    # ===============================================================
    # Helpers for extend_acc_subgraph() method
    # ===============================================================

    def find_reverse_deps(
        self, tag_id: int | None = None
    ) -> dict[torch.fx.Node, NodeSet]:
        # [FUNCTION DOC] Builds a mapping from each callable node to the set of its callable users. If tag_id is
        # given, users whose tag number is not smaller than tag_id are ignored. Requires users to be tagged.
        """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
        result: dict[torch.fx.Node, NodeSet] = defaultdict(set)

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                if tag_id is None or (int(user.tag.split("_")[-1]) < tag_id):
                    result[node].add(user)

        return result

    def update_reverse_deps_for_fusions(
        self, deps: dict[torch.fx.Node, NodeSet]
    ) -> None:
        # [FUNCTION DOC] Adjusts a reverse-dependency mapping in place for fusion groups: members share the
        # union of group users (minus members) and producers outside a group gain the group as users.
        processed_node = set()

        for node, fusion in self.fusions.items():
            if node in processed_node:
                continue

            new_dep = set()

            # Create a new dependency set which include all the
            # dependencies of the nodes in the fusion group
            for n in fusion:
                new_dep.update(deps[n])

            # Exclude nodes in the fusion
            new_dep.difference_update(fusion)

            # Update dependency
            for n in fusion:
                deps[n] = new_dep

                for arg in n.all_input_nodes:
                    if arg not in fusion:
                        deps[arg].update(fusion)

                processed_node.add(n)

    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        # [FUNCTION DOC] Returns callable nodes that feed nodes tagged with tag but are themselves callable and
        # tagged differently (placeholders and get_attr are ignored).
        """
        Finds parent nodes of the `tag` subgraph.

        Traverse the inputs of nodes in the subgraph, if input doesn't belong to the subgraph
        and is not a placeholder, we consider it as the parent node of the subgraph.
        """
        parent_nodes = set()

        for node in self.module.graph.nodes:
            if node.op in CALLABLE_NODE_OPS and node.tag == tag:
                for arg in node.all_input_nodes:
                    if arg.op in CALLABLE_NODE_OPS and arg.tag != tag:
                        parent_nodes.add(arg)

        return parent_nodes

    def extend_acc_subgraph(self, tag: str) -> None:
        # [FUNCTION DOC] Grows the acc subgraph named tag backwards: repeatedly picks a parent node in the acc set
        # whose reverse dependencies are all already visited and assigns it the tag, queueing its fusion buddies
        # and callable inputs. Stops when none qualifies. Mutates node tags.
        """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
        # Dict that maps node to its users and ignore users that
        # are in the subgraph that has greater tag
        deps = self.find_reverse_deps(tag_id=int(tag.rsplit("_", maxsplit=1)[-1]))
        self.update_reverse_deps_for_fusions(deps)

        # Parent nodes of the subgraph
        parent_nodes = self.find_parent_nodes_of_subgraph(tag)

        visited_nodes: NodeSet = set()

        while parent_nodes:
            node = None

            # Find a acc node that depends on visited nodes only
            for n in parent_nodes:
                if deps[n] <= visited_nodes and n in self.acc_nodes:
                    node = n
                    break

            if node is None:
                break

            # Put the node into `tag` subgraph
            node.tag = tag  # type: ignore[attr-defined]
            parent_nodes.remove(node)
            visited_nodes.add(node)

            # If node is in a fusion group, add all fusion buddies to parent nodes
            if node in self.fusions:
                for fusion_node in self.fusions[node]:
                    if fusion_node not in visited_nodes:
                        parent_nodes.add(fusion_node)

            # Add inputs of the node to parent nodes
            for arg in node.all_input_nodes:
                if arg.op in CALLABLE_NODE_OPS and arg not in visited_nodes:
                    parent_nodes.add(arg)

    # ===============================================================
    # Helpers for split() method
    # ===============================================================

    def starter_nodes(self) -> tuple[NodeSet, NodeSet]:
        # [FUNCTION DOC] Returns a pair of sets (CPU starters, acc starters): nodes that consume placeholders or
        # get_attr nodes, plus call_function nodes with no inputs, split by membership in the acc set.
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()
        for node in self.module.graph.nodes:
            # edge case, call_function, but with no dependencies
            if node.op == "call_function" and len(node.all_input_nodes) == 0:
                if node in self.acc_nodes:
                    starter_acc_nodes.add(node)
                else:
                    starter_cpu_nodes.add(node)

            if node.op not in {"placeholder", "get_attr"}:
                continue

            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)

        return starter_cpu_nodes, starter_acc_nodes

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        # [FUNCTION DOC] Traverses the graph from the starter nodes, alternating between acc and CPU, to form
        # ordered subgraphs. A node joins the current subgraph only when its dependencies are visited; when none
        # is ready the subgraph is closed and the kind flips. Fusion buddies are queued with the node. Raises
        # FxNetSplitterInternalError for an empty subgraph or if no subgraphs were created. Returns the list.
        # We start graph traversal from leaf nodes
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        visited_nodes: NodeSet = set()

        # Determine which subgraph to start from based on which subgraph has
        # 0-dep node
        acc_subgraph: bool = not any(len(self.deps[n]) == 0 for n in current_cpu_nodes)

        current_subgraph_nodes: NodeList = []

        # Result accumulator
        subgraphs: list[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            # Find the first node that should belong to the current subgraph and has all dependencies resolved
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            node = next(
                (n for n in current_nodes if self.deps[n] <= visited_nodes),
                None,
            )

            # If nothing was found, then it's time to flip the mode and start a new subgraph
            if node is None:
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
                )
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue

            current_nodes.remove(node)
            visited_nodes.add(node)
            current_subgraph_nodes.append(node)

            # Add fusion buddies
            if node in self.fusions:
                if node in self.acc_nodes:
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)

            # Put depending nodes into the queue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # Add downstream nodes
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)

        # Check if the last subgraph was not created
        if current_subgraph_nodes:
            subgraphs.append(
                Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
            )

        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")

        return subgraphs

    def remove_small_acc_subgraphs(self, subgraphs: list[Subgraph]) -> list[Subgraph]:
        # [FUNCTION DOC] Merges acc subgraphs smaller than the minimum size into the previous subgraph (or turns
        # the first into a CPU one) and merges adjacent CPU subgraphs. Prints a message per elimination. Mutates
        # subgraph node lists. Returns the new list.
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
        result: list[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if len(subgraph.nodes) >= self.settings.min_acc_module_size:
                    result.append(subgraph)
                else:
                    print(
                        "Eliminating acc subgraph because it's smaller than the threshold: "
                        f"{len(subgraph.nodes)} < {self.settings.min_acc_module_size}"
                    )
                    if result:
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        subgraph.is_acc = False
                        result.append(subgraph)
            else:
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    result.append(subgraph)
        return result

    def tag(self, subgraphs: list[Subgraph]) -> None:
        # [FUNCTION DOC] Assigns each subgraph a tag (acc prefix or the non-acc name, plus index), stores tags in
        # the instance, sets a tag attribute on its nodes and records the node to tag mapping. Raises
        # FxNetSplitterInternalError if a node is already tagged. Replaces any previous tags list.
        self.tags = []
        for subgraph in subgraphs:
            tag = (
                f"_run_on_acc_{len(self.tags)}"
                if subgraph.is_acc
                else f"{self.non_acc_submodule_name}{len(self.tags)}"
            )
            self.tags.append(tag)
            for node in subgraph.nodes:
                if hasattr(node, "tag"):
                    raise FxNetSplitterInternalError(f"Node {node} was already tagged")

                node.tag = tag  # type: ignore[attr-defined]
                self._node_submodule_map[node.name] = tag

    def split(self, remove_tag: bool = False) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Splits the module by the stored tags into a module of submodules (optionally returning
        # tuples). If remove_tag, deletes the tag attribute from all nodes afterwards. Returns the split module.
        split_module = split_by_tags(
            self.module, self.tags, return_tuple=self._return_tuple
        )
        if remove_tag:
            for node in self.module.graph.nodes:
                if hasattr(node, "tag"):
                    del node.tag
        return split_module  # type: ignore[return-value]

    def __call__(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Runs the whole pipeline: builds subgraphs, optionally moves non-tensor boundary nodes,
        # removes small acc subgraphs, prints counts, tags and splits. Returns the split module.
        subgraphs = self.put_nodes_into_subgraphs()
        if self.settings.move_non_tensor_nodes_on_boundary:
            move_non_tensor_nodes_on_boundary(subgraphs)
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_count = len([s for s in subgraphs if s.is_acc])
        non_acc_subgraphs_count = len(subgraphs) - acc_subgraphs_count
        print(
            f"Got {acc_subgraphs_count} acc subgraphs and {non_acc_subgraphs_count} non-acc subgraphs"
        )
        self.tag(subgraphs)
        return self.split()

    def generate_split_results(self) -> SplitResult:
        # [FUNCTION DOC] Runs the split and gathers the names of child submodules. Raises ValueError if a positive
        # max_acc_splits limit is exceeded, then captures example inputs for each submodule. Returns a SplitResult.
        # KNOWN ISSUE: the limit is compared with the count of all child submodules (acc and CPU), not just acc.
        split_module = self()
        submodule_names: list[str] = []
        for name, _mod in split_module.named_children():
            submodule_names.append(name)
        if (
            self.settings.max_acc_splits > 0
            and len(submodule_names) > self.settings.max_acc_splits
        ):
            raise ValueError(
                "Cannot fulfill max_acc_splits limit. "
                "This may cause split fragmentation and "
                "result in performance issues."
            )

        submodule_inputs = generate_inputs_for_submodules(
            split_module, self.sample_input, submodule_names
        )
        return SplitResult(split_module, submodule_inputs, self.non_acc_submodule_name)
