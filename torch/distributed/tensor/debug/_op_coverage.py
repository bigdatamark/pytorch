# mypy: allow-untyped-defs
# [MODULE DOC] Utilities to measure which operators a model needs and whether DTensor has
# sharding rules for them. Traces a model's forward and backward with aot_module using inductor
# decompositions and prints a table of operator counts and support status.
# Import-time side effects: builds the inductor decomposition table and creates a module-level
# list that collects traced graphs (it is never cleared, so repeated tracing accumulates).
from operator import itemgetter

import torch
import torch.fx
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed.tensor import DTensor


inductor_decomps = select_decomp_table()

graphs: list[torch.fx.GraphModule] = []


def fwd_bwd_compiler(fx_g, _):
    # [FUNCTION DOC] HELPER: Compiler callback for aot_module. Appends the given graph module to
    # the module-level graphs list (mutation of global state) and returns it wrapped as a boxed
    # callable. The second argument (example inputs) is ignored.
    graphs.append(fx_g)
    return make_boxed_func(fx_g)


def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    # [FUNCTION DOC] Traces the model with aot_module (inductor decompositions) by running a
    # forward pass with the given positional and keyword inputs, reducing a non-scalar output by
    # summation, then running backward. Returns the module-level graphs list (forward graph then
    # backward graph). Raises an AssertionError if the list does not hold exactly two graphs.
    # KNOWN ISSUE: the global graphs list is never reset, so a second call in the same process
    # accumulates more than two graphs and fails the length check.
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    compiled_mod = aot_module(
        model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps
    )
    output = compiled_mod(*args, **kwargs)

    if output.ndim != 0:
        # if output is not a scalar tensor, by default sum it in order to
        # run backward
        output = output.sum()

    output.backward()

    # one fwd, one bwd graph
    if len(graphs) != 2:
        raise AssertionError
    return graphs


def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    # [FUNCTION DOC] Traces the model, counts each operator overload across forward and backward
    # graphs, and checks whether the DTensor sharding propagator has a rule for it. Prints a
    # table (operator, schema, total count, supported) sorted by count descending using the
    # tabulate package (must be installed, else import error). If output_csv is true, also writes
    # the same rows to a file named op_summary.csv in the current directory (overwrites it).
    # Returns nothing.
    """
    Util to print the operator coverage summary of a certain model with tabulate.

    Must have tabulate module installed.
    """
    # python module required for summary
    import csv

    from tabulate import tabulate

    fwd_graph, bwd_graph = get_inductor_decomp_graphs(model, args, kwargs)

    op_counts = {}

    for node in fwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    for node in bwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    op_infos = []

    for op, count in op_counts.items():
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        op_infos.append([op, str(op._schema), count, supported])

    # sort the op info base on the total count index
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)

    headers = ["Operator", "Schema", "Total Count", "Supported"]
    print(tabulate(op_infos, headers=headers))

    if output_csv:
        # Open a CSV file for writing
        with open("op_summary.csv", "w", newline="") as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(headers)
            # Write each table row to the CSV file
            for row in op_infos:
                csv_writer.writerow(row)
