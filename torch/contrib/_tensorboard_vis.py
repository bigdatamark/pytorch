# mypy: allow-untyped-defs
# [MODULE DOC] Debugging tool that converts TorchScript graphs and graph executor states into a
# TensorBoard GraphDef protobuf and writes it as an event file. Hard-depends on TensorFlow: importing
# this module raises ImportError with an explanatory message when TensorFlow is missing.
import time
from collections import defaultdict
from functools import partial

import torch


# Unfortunately it doesn't seem as if there was any way to get TensorBoard to do
# anything without having TF installed, and so this file has a hard dependency on it
# as well. It really is a debugging tool, so it doesn't matter.
try:
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.util import event_pb2
    from tensorflow.python.summary.writer.writer import FileWriter
except ImportError:
    raise ImportError(
        "TensorBoard visualization of GraphExecutors requires having "
        "TensorFlow installed"
    ) from None


def dump_tensorboard_summary(graph_executor, logdir):
    # [FUNCTION DOC] Writes a TensorBoard event file in logdir containing the visualization of graph_executor.
    # Builds the protobuf with visualize, wraps it in an event stamped with the current time and adds it
    # through a FileWriter. Side effect: file I/O in logdir; returns nothing.
    with FileWriter(logdir) as w:
        pb_graph = visualize(graph_executor)
        evt = event_pb2.Event(
            wall_time=time.time(), graph_def=pb_graph.SerializeToString()
        )
        w.add_event(evt)


def visualize(graph, name_prefix="", pb_graph=None, executors_it=None):
    # [FUNCTION DOC] Builds (or appends to) a GraphDef for graph, which may be a plain graph or a graph executor
    # state. For an executor state it delegates to visualize_graph_executor and returns the protobuf.
    # For a plain graph it adds an input node, maps the parameter values to its outputs, recursively
    # adds all nodes, then adds an output node connected to the return values. name_prefix namespaces node
    # names; pb_graph is the graph to append to (created if absent); executors_it optionally supplies
    # nested executors in order. Returns the GraphDef.
    """Visualizes an independent graph, or a graph executor."""
    value_map = {}
    pb_graph = pb_graph or graph_pb2.GraphDef()

    if isinstance(graph, torch._C.GraphExecutorState):
        visualize_graph_executor(
            graph, name_prefix, pb_graph, partial(visualize, pb_graph=pb_graph)
        )
        return pb_graph

    # Set up an input node
    pb_graph.node.add(op="input", name=name_prefix + "input")
    for i, value in enumerate(graph.param_node().outputs()):
        value_map[value.unique()] = name_prefix + "input:" + str(i)

    visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it)

    # Gather all outputs
    return_node = pb_graph.node.add(op="output", name=name_prefix + "output")
    for value in graph.return_node().inputs():
        return_node.input.append(value_map[value.unique()])

    return pb_graph


def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    # [FUNCTION DOC] Adds the contents of one graph executor state to pb_graph: the autograd fallback graph (if
    # any) and each execution plan as independent subgraphs (with a disconnected node recording the
    # argument spec, plus an optional gradient executor subgraph). Finally calls inline_graph once on the
    # original graph under the original/ prefix and returns its result. Mutates pb_graph.
    """Append the state of a given GraphExecutor to the graph protobuf.

    Args:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (Callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
    if state.autograd_fallback_graph is not None:
        visualize(
            graph=state.autograd_fallback_graph,
            name_prefix=name_prefix + "autograd_fallback/",
            pb_graph=pb_graph,
            executors_it=iter(state.autograd_fallback.executors()),
        )

    for i, (arg_spec, plan) in enumerate(state.execution_plans.items()):
        subgraph_name = name_prefix + f"plan{i}/"

        # Create a disconnected node that will keep information regarding the input
        # types of this trace. This is unfortunately a bit too verbose to be included
        # in the subgraph name.
        input_kinds = pb_graph.node.add(op="INPUT_KIND", name=subgraph_name)
        input_kinds.attr["inputs"].s = repr(arg_spec).encode("ascii")

        visualize(plan.graph, subgraph_name, pb_graph, iter(plan.code.executors()))

        # Show gradient as an independent subgraph of this plan
        if plan.grad_executor is not None:
            grad_subgraph_name = subgraph_name + "grad/"
            visualize(plan.grad_executor, grad_subgraph_name, pb_graph)

    return inline_graph(state.graph, name_prefix + "original/")


def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    # [FUNCTION DOC] Recursive worker for visualize: adds a protobuf node for every node of graph, wiring inputs
    # through value_map (value id -> protobuf name, updated with outputs). Fusion groups and nested graph
    # executors are expanded inline as subgraphs. Mutates value_map and pb_graph. Returns nothing.
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""

    def inline_graph(subgraph, name, node):
        # [FUNCTION DOC] HELPER: Recursively visualizes subgraph under the given name, mapping its inputs to the
        # already-named values of node's inputs and then aliasing node's outputs to the subgraph's outputs.
        rec_value_map = {
            inp.unique(): value_map[val.unique()]
            for inp, val in zip(subgraph.inputs(), node.inputs())
        }
        visualize_rec(
            graph=subgraph, value_map=rec_value_map, name_prefix=name, pb_graph=pb_graph
        )
        for out, val in zip(subgraph.outputs(), node.outputs()):
            value_map[val.unique()] = rec_value_map[out.unique()]

    op_id_counter: defaultdict[str, int] = defaultdict(int)

    def name_for(node):
        # [FUNCTION DOC] HELPER: Returns the operator kind (namespace stripped) and a unique protobuf node name
        # built from the prefix, kind and a per-kind counter, incrementing the counter.
        kind = node.kind()[node.kind().index("::") + 2 :]
        op_id_counter[kind] += 1
        return kind, name_prefix + kind + "_" + str(op_id_counter[kind])

    def add_fusion_group(node):
        # [FUNCTION DOC] HELPER: Names a fusion group node and inlines its Subgraph attribute.
        op, name = name_for(node)
        inline_graph(node.g("Subgraph"), name + "/", node)

    def add_graph_executor(node):
        # [FUNCTION DOC] HELPER: Handles a graph executor node: if no nested executors iterator was given, adds it as a
        # plain node; otherwise takes the next executor and visualizes its state inlined at this node.
        op, name = name_for(node)
        if executors_it is None:
            add_node(node)
        else:
            ge = next(executors_it)
            visualize_graph_executor(
                ge, name + "/", pb_graph, partial(inline_graph, node=node)
            )

    def add_node(node):
        # [FUNCTION DOC] HELPER: Adds one node, dispatching fusion groups and graph executors to special handlers;
        # otherwise creates a protobuf node with its input wiring and records its outputs in value_map.
        # KNOWN ISSUE: node attributes are not exported (TODO in code).
        if node.kind() == "prim::FusionGroup":
            return add_fusion_group(node)
        elif node.kind() == "prim::GraphExecutor":
            return add_graph_executor(node)
        op, name = name_for(node)
        pb_node = pb_graph.node.add(op=op, name=name)
        for value in node.inputs():
            pb_node.input.append(value_map[value.unique()])
        # TODO: handle attrs
        for i, value in enumerate(node.outputs()):
            value_map[value.unique()] = name + ":" + str(i)

    for node in graph.nodes():
        add_node(node)
