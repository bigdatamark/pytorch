# [MODULE DOC] Data structures and helpers for building the optimization (MILP) inputs
# for activation checkpointing and FSDP planning. Defines typed dictionaries
# (ModOrder, ModRuntime, ModStats, ModuleInfo), a module tree Graph/Node with an
# ancestor-descendant matrix, an aggregator that merges stats from MemTracker,
# RuntimeEstimator and SACEstimator, and small name/byte-formatting and baseline-peak
# helpers. Requires numpy; no import-time side effects.
import copy
from collections import OrderedDict
from typing import cast, TypedDict

import numpy as np

import torch
from torch.distributed._tools.mem_tracker import (
    _MemRefType,
    _ModMemStats,
    _ModState,
    MemTracker,
)
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACTradeOffStats


class ModOrder(TypedDict):
    # [CLASS DOC] Typed dictionary holding module FQN orderings: forward and backward pre-
    # order and post-order lists.
    fw_pre_order: list[str]
    bw_pre_order: list[str]
    fw_post_order: list[str]
    bw_post_order: list[str]


class ModRuntime(TypedDict):
    # [CLASS DOC] Typed dictionary with the forward and backward runtime of a module (ms).
    fw: float
    bw: float


class ModStats(TypedDict):
    # [CLASS DOC] Typed dictionary of per-module statistics: parameter, gradient and
    # activation memory figures, input/output sizes, forward/backward runtimes, leaf flag,
    # and the SAC trade-off data (runtime, memory, piecewise-linear segment count, slopes,
    # intercepts, breakpoints, original curve). Memory in bytes, time in ms.
    fqn: str
    # per-module params
    param_per_module: int
    # per-module grads
    grad_per_module: int
    # total accumulated gradients up to and including this module
    grad_total: int
    # per module fw activation size (excluding input and output)
    act_fw_per_module: int
    # per module bw activation size during peak_bw
    act_bw_per_module: int
    # per module activation grad size during peak_bw
    act_grad_per_module: int
    # total activation size up to but excluding the current module
    # includes input of the current module (i.e., output of previous module)
    act_total: int
    # Inputs to the module
    input_per_module: int
    # Outputs of the module
    output_per_module: int
    # Total fw run-time of the module
    fw_runtime_per_module: float
    # Total bw run-time of the module
    bw_runtime_per_module: float
    # Is this module a leaf module
    is_leaf: bool
    # Total ac run-time of the module
    sac_runtime: float
    # Total ac_memory for the module
    sac_memory: int
    # Number of piecewise-linear functions used for approximating ac tradeoff curve
    n_segments: int
    # Slopes of the of piecewise-linear functions
    slopes: list[float]
    # Intercepts of the of piecewise-linear functions
    intercepts: list[float]
    # X breakpoints of the of piecewise-linear functions
    breakpoints: list[float]
    # Original trade-off curves
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    # [CLASS DOC] Typed dictionary combining the module ordering (mod_order) and the list
    # of per-module stats (mod_stats).
    mod_order: ModOrder
    mod_stats: list[ModStats]


def aggregate_stats(
    model: torch.nn.Module,
    mem_tracker: MemTracker,
    runtime_estimator: RuntimeEstimator,
    sac_estimator: SACEstimator,
    dev: torch.device,
) -> ModuleInfo:
    # [FUNCTION DOC] Collects per-module stats for a model into a ModuleInfo. Takes the
    # model, a MemTracker, a RuntimeEstimator, a SACEstimator and the device the model ran
    # on. Deep-copies the tracked memory stats, reads runtimes and orderings from the
    # runtime estimator, and calls the SAC estimator to fit piecewise-linear trade-off
    # curves (side effect on the estimator) before copying its stats. For each module with
    # memory stats, builds a ModStats: modules lacking SAC trade-off stats are marked leaf
    # with zeroed SAC fields, others non-leaf. Activation figures are computed from memory
    # snapshots on the given device and clamped at zero where noted. Returns the dict with
    # the ordering and the stats list. Raises KeyError if snapshots or runtimes are
    # missing for a module.
    # KNOWN ISSUE: grad_per_module is set to the module parameter memory, not a measured
    # gradient size (assumes grads match params).
    """
    Collect modulewise stats for a given model, including memory, runtime, and AC tradeoff stats.

    Args:
        model: nn.Module object
        runtime_estimator: RuntimeEstimator object with runtime stats
        mem_tracker: MemTracker object with memory stats
        sac_estimator: SACEstimator object with AC tradeoff stats
        dev: device the model was run on (used to extract memory stats from MemTracker)

    Returns:
        ModuleInfo: A dictionary with module order and module stats.
    """

    # Memory stats
    mod_mem_stats: dict[torch.nn.Module, _ModMemStats] = dict(
        copy.deepcopy(mem_tracker.memory_tracking)
    )

    # Runtime stats
    mod_runtime_stats: dict[str, ModRuntime] = {
        fqn: {"fw": v["fw"], "bw": v["bw"]}
        for fqn, v in runtime_estimator.mod_runtimes.items()
    }

    # Module order
    mod_order: ModOrder = {
        "fw_pre_order": list(runtime_estimator.mod_fw_pre_order),
        "bw_pre_order": list(runtime_estimator.mod_bw_pre_order),
        "fw_post_order": list(runtime_estimator.mod_fw_post_order),
        "bw_post_order": list(runtime_estimator.mod_bw_post_order),
    }

    # Selective Activation Checkpointing stats
    sac_estimator.pwlf_sac_tradeoff_curve()
    mod_sac_tradeoff_stats: dict[str, SACTradeOffStats] = copy.deepcopy(
        sac_estimator.sac_mod_tradeoff_stats
    )

    module_info: ModuleInfo = {
        "mod_order": mod_order,
        "mod_stats": [],
    }

    for mod in model.modules():
        if mod_mem_stat := mod_mem_stats.get(mod):
            if tradeoff_stats := mod_sac_tradeoff_stats.get(mod_mem_stat.mod_fqn):
                sac_runtime = tradeoff_stats.sac_runtime
                sac_memory = tradeoff_stats.sac_memory
                n_segments = tradeoff_stats.n_segments
                slopes = tradeoff_stats.slopes
                intercepts = tradeoff_stats.intercepts
                breakpoints = tradeoff_stats.fit_breaks
                tradeoff_curve = tradeoff_stats.tradeoff_curve
                is_leaf = False
            else:
                sac_runtime = sac_memory = n_segments = 0
                slopes = intercepts = breakpoints = []
                tradeoff_curve: OrderedDict[float, float] = OrderedDict()  # type: ignore[no-redef]
                is_leaf = True
            mod_stat: ModStats = {
                "fqn": mod_mem_stat.mod_fqn,
                "param_per_module": mod_mem_stat.parameter_mem,
                "grad_per_module": mod_mem_stat.parameter_mem,
                "grad_total": mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                    _MemRefType.GRAD
                ],
                "act_fw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.snapshots[_ModState.PRE_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.output_mem,
                ),
                "act_bw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.ACT],
                ),
                "act_grad_per_module": (
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP]
                    - mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                        _MemRefType.TEMP
                    ]
                ),
                "act_total": mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][
                    _MemRefType.ACT
                ],
                "input_per_module": mod_mem_stat.input_mem,
                "output_per_module": mod_mem_stat.output_mem,
                "fw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["fw"],
                "bw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["bw"],
                "is_leaf": is_leaf,
                "sac_runtime": sac_runtime,
                "sac_memory": sac_memory,
                "n_segments": n_segments,
                "slopes": slopes,
                "intercepts": intercepts,
                "breakpoints": breakpoints,
                "tradeoff_curve": tradeoff_curve,
            }
            module_info["mod_stats"].append(mod_stat)

    return module_info


class Node(ModStats):
    # [CLASS DOC] A ModStats record extended with the module index in forward pre-order
    # and its position in forward post-order.
    index: int  # index according to forward pre-order
    pos_fw_post_order: int  # index according to forward post-order


class Graph:
    # [CLASS DOC] Tree of modules for the MILP solver. State: nodes (list of Node in pre-
    # order), name2node (mapping from FQN to node), ad_matrix (square numpy array where
    # entry i,j is 1 if node j is node i or its descendant) and fw_post_order (list of
    # FQNs). add_node appends a node and indexes it by FQN.
    def __init__(self, n: int) -> None:
        # [FUNCTION DOC] Creates an empty graph sized for n nodes: empty node list and name
        # mapping, an n by n zero ancestor-descendant matrix and an empty post-order list.
        self.nodes: list[Node] = []
        self.name2node: dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))
        self.fw_post_order: list[str] = []

    def add_node(self, node: Node) -> None:
        # [FUNCTION DOC] Appends the node to the node list and registers it by its FQN
        # (overwriting a same-named entry).
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node


def parse_module_info(module_info: ModuleInfo) -> Graph:
    """
    Parse module info and create a graph (tree) of modules. The graph will be
    used by MILP solver to find optimal SAC and/or FSDP configurations.
    # [FUNCTION DOC] Builds a Graph from a ModuleInfo. Raises AssertionError if the number
    # of stats differs from the length of the forward pre-order. Records the forward post-
    # order, sorts the stats by forward pre-order position, gives each node its index and
    # post-order position, adds them to the graph, and fills the ancestor-descendant
    # matrix (row i marks consecutive following nodes that are the same or descendants of
    # node i, stopping at the first non-descendant). Returns the graph.
    # KNOWN ISSUE: the sorted list is stored back into the input dict but the loop still
    # iterates the original unsorted local list, so node indices follow the original
    # order, not pre-order, unless the input was already sorted. The input dict is also
    # mutated (mod_stats replaced, nodes get index fields).
    """
    mod_stats = module_info["mod_stats"]
    fw_pre_order = module_info["mod_order"]["fw_pre_order"]
    # assertion and number of nodes
    if len(mod_stats) != len(fw_pre_order):
        raise AssertionError
    n_nodes = len(mod_stats)

    # create graph
    g = Graph(n_nodes)
    g.fw_post_order = module_info["mod_order"]["fw_post_order"]

    # sort the modules by pre-order and add them to the graph
    module_info["mod_stats"] = sorted(
        mod_stats, key=lambda x: fw_pre_order.index(x["fqn"])
    )
    for i, one_mod_stats in enumerate(mod_stats):
        node: Node = cast(Node, one_mod_stats)
        node["index"] = i
        node["pos_fw_post_order"] = g.fw_post_order.index(node["fqn"])
        g.add_node(node)

    # set up ancestor-descendant matrix
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if is_self_or_submodule(g.nodes[j]["fqn"], g.nodes[i]["fqn"]):
                g.ad_matrix[i][j] = 1
            else:
                break

    return g


def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    # [FUNCTION DOC] Returns True if the first name equals the ancestor name or contains
    # the ancestor name followed by a dot, i.e. is the same module or a submodule by FQN
    # text. Uses substring containment, so it can also be true when the ancestor appears
    # mid-name rather than as a prefix.
    check if name_descendant is a submodule of name_ancestor, or if they are the same
    """
    return name_descendant == name_ancestor or name_ancestor + "." in name_descendant


def is_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    # [FUNCTION DOC] Returns True if the ancestor name followed by a dot occurs in the
    # descendant name (strict submodule test). Substring based, not strictly a prefix
    # check.
    if name_descendant is a submodule of name_ancestor, but not the same
    """
    return name_ancestor + "." in name_descendant


def display_bytes(b: int, unit: str = "MiB") -> str:
    """
    # [FUNCTION DOC] Formats a byte count as text with two decimals in KiB, MiB or GiB
    # depending on the unit argument (default MiB); any other unit gives plain bytes.
    return a string that represent the number of bytes in a desired unit
    """
    if unit == "KiB":
        return f"{b / 2**10:.2f} KiB"
    if unit == "MiB":
        return f"{b / 2**20:.2f} MiB"
    if unit == "GiB":
        return f"{b / 2**30:.2f} GiB"
    return f"{b:.2f} bytes"


def get_peak_memory_runtime_baseline(graph: Graph) -> tuple[int, float]:
    """
    # [FUNCTION DOC] Computes the baseline (no FSDP, no activation checkpointing) peak
    # memory and compute time from a graph. Peak memory is the maximum over nodes of
    # first-node parameter memory plus that node's total gradient, activation-gradient and
    # total activation memory. Compute time is the first (root) node's forward plus
    # backward runtime. Returns a pair (peak bytes, time in ms). Excludes optimizer state
    # and embeddings.
    Get the baseline peak memory and runtime.
    Baseline here means there is no FSDP or AC.
    Memory includes the parameters, gradients, activations, and activation gradients.
    Memory does not include e.g., optimizer states, embedding tables, etc.

    Returns:
        int: peak memory in bytes
        float: compute time in ms
    """
    P_1 = graph.nodes[0]["param_per_module"]
    num_nodes = len(graph.nodes)
    peak_mem = 0
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"]
        AG_i = graph.nodes[i]["act_grad_per_module"]
        TA_i = graph.nodes[i]["act_total"]
        peak_mem = max(peak_mem, P_1 + TG_i + AG_i + TA_i)
    compute_time = (
        graph.nodes[0]["fw_runtime_per_module"]
        + graph.nodes[0]["bw_runtime_per_module"]
    )
    return (peak_mem, compute_time)
