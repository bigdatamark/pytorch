# [MODULE DOC] Small FX graph passes used by the exporter: decomposition with an ONNX registry,
# type promotion insertion, and removal of assertion nodes. All operate on torch.export / torch.fx
# objects. No import-time side effects.
from __future__ import annotations

import torch
import torch.export
import torch.fx
from torch.onnx._internal.exporter import _decomp, _registration
from torch.onnx._internal.fx import passes


def decompose_with_registry(
    exported_program: torch.export.ExportedProgram, registry: _registration.ONNXRegistry
) -> torch.export.ExportedProgram:
    # [FUNCTION DOC] Decomposes an ExportedProgram into ops the ONNX registry can handle. Builds a
    # decomposition table that skips ops with a registered ONNX implementation, then runs the
    # decompositions. Args: exported_program, registry (ONNXRegistry). Returns: a new
    # ExportedProgram. Exists as a separate function so it shows up in profiler results.
    """Decompose the exported program with the given registry.

    This function is needed so it shows clearly on the profiler results.
    """
    onnx_registered_ops = set(_decomp.get_onnx_implemented_overloads(registry))
    decomp_table = _decomp.create_onnx_friendly_decomposition_table(onnx_registered_ops)
    return exported_program.run_decompositions(decomp_table)


def insert_type_promotion_nodes(
    graph_module: torch.fx.GraphModule,
) -> None:
    # [FUNCTION DOC] Inserts explicit type promotion nodes in place, for the graph module and
    # every nested module. Args: graph_module. Returns: nothing (mutates). Raises an
    # AssertionError if a submodule is not a GraphModule.
    """Inplace pass to insert explicit type promotion nodes, recursively through nested modules."""
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected GraphModule, got {type(module)}")
        passes.InsertTypePromotion(module).run()


def remove_assertion_nodes(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Erases assertion/check call nodes (range constraints, async/scalar asserts,
    # tensor metadata asserts) from the graph module and all nested modules, then recompiles each.
    # Args: graph_module. Returns: the same, mutated, graph module.
    """Remove all assertion and check nodes from the FX graph"""
    aten_assertion_targets = {
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten._assert_tensor_metadata.default,
    }
    for gm in graph_module.modules():
        for node in gm.graph.nodes:  # type: ignore[union-attr]
            if node.op == "call_function" and node.target in aten_assertion_targets:
                gm.graph.erase_node(node)  # type: ignore[operator, union-attr]
        gm.recompile()  # type: ignore[operator]
    return graph_module
