# [MODULE DOC] Pass that compiles invoke_subgraph higher-order-op regions with
# user-specified Inductor compilers. Nodes carry nested_region_config in their
# custom metadata (forward or backward compiler). Each referenced subgraph is
# compiled to an ahead-of-time artifact and the invoke_subgraph call nodes are
# replaced with direct calls to it. Public entry: regional_inductor_invoke_subgraph.
# Import-time side effects: creates a module logger.
import copy
import logging
from collections import defaultdict

import torch
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.compiler._cache import CacheArtifactManager
from torch.fx._compatibility import compatibility
from torch.fx.passes.regional_inductor import (
    _disable_remat_for_regional_subcompile,
    _dummy_wrapper,
)


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor_invoke_subgraph"]


def _compile_submod(
    gm: torch.fx.GraphModule, subgraph: str, subgraph_users: list[torch.fx.Node]
) -> torch.fx.GraphModule:
    """
    Compiles subgraph submodule in gm. subgraph is used by subgraph_users.
    subgraph_users must all be  torch.ops.higher_order.invoke_subgraph HOP.
    """
    # [FUNCTION DOC] HELPER: Compile one subgraph submodule of gm and rewrite its
    # users. The first user supplies the compile config and picks the forward or
    # backward compiler (by partitioner tag) and the fake example inputs (the
    # inputs after the subgraph reference; raises RuntimeError if one lacks a fake
    # value). The submodule is deep-copied and compiled inside a fresh tracing
    # context, fresh cache manager, bundled autograd cache config and remat-disabled
    # scope; the result must be an AOTCompiledArtifact (AssertionError otherwise).
    # Each user node is replaced with a call to the compiled artifact on the
    # remaining args, copying metadata, and gm is recompiled. Mutates and returns gm.

    submod = getattr(gm, subgraph)

    compile_config = None
    fake_inputs = []

    # We use the first user for compile configs and inputs
    sub_node = subgraph_users[0]
    if not _needs_inductor_compile(sub_node):
        raise AssertionError("sub_node does not need inductor compile")
    compile_config = sub_node.meta["custom"]["nested_region_config"]
    if sub_node.meta.get("partitioner_tag") == "is_forward":
        compile_fn = compile_config.fw_compiler
    else:
        compile_fn = compile_config.bw_compiler

    for inp_node in sub_node.all_input_nodes[
        1:
    ]:  # exclude the graph module input to torch.ops.higher_order.invoke_subgraph
        if hasattr(inp_node, "meta") and "val" in inp_node.meta:
            fake_inputs.append(inp_node.meta["val"])
        else:
            raise RuntimeError(
                f"Partition is bad because non fake tensor value is seen {inp_node}"
            )

    # Log the options being used
    logger.info(
        "Compiling submodule %s with inductor options: %s",
        subgraph,
        compile_config,
    )

    def get_compiled_fn() -> AOTCompiledArtifact:
        # [FUNCTION DOC] HELPER: Run the selected compiler on a deep copy of the
        # submodule under a new tracing context built from the current fake mode.
        # Raises AssertionError if the current context has no fake mode.
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("context.fake_mode is None")

        context = torch._guards.TracingContext(context.fake_mode)

        with (
            torch._guards.tracing(context),
            CacheArtifactManager.with_fresh_cache(),
            torch._functorch.config.patch("bundled_autograd_cache", True),
            _disable_remat_for_regional_subcompile(),
            torch.fx.traceback._set_regional_inductor_subgraph_name(subgraph),
        ):
            # compile_fx can mutate gm
            gm = copy.deepcopy(submod)

            compiled_fn = compile_fn(gm, fake_inputs)
            return compiled_fn

    compiled_fn = get_compiled_fn()
    if not isinstance(compiled_fn, AOTCompiledArtifact):
        raise AssertionError(f"Expected AOTCompiledArtifact, got {type(compiled_fn)}")

    # _dummy_wrapper is to make call_function happy
    compiled_submod = _dummy_wrapper(compiled_fn)
    for node in subgraph_users:
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                # exclude graph nodes input args
                compiled_submod,
                args=node.args[2:],
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] HELPER: True if the node is not a placeholder or output, has
    # a nested_region_config in custom metadata, and that config has a forward
    # compiler (when the partitioner tag is not backward) or a backward compiler
    # (when the tag is backward). Pure read of metadata.
    # TODO: maybe we could change to check
    # node.meta.get("partitioner_tag") != "is_forward"
    # if the tag is relibable
    return bool(
        (
            node.op not in ("placeholder", "output")
            and hasattr(node, "meta")
            and node.meta.get("custom", None)
            and node.meta["custom"].get("nested_region_config", None)
            and node.meta["custom"]["nested_region_config"].fw_compiler
            and node.meta.get("partitioner_tag") != "is_backward"
        )
        or (
            node.op not in ("placeholder", "output")
            and hasattr(node, "meta")
            and node.meta.get("custom", None)
            and node.meta["custom"].get("nested_region_config", None)
            and node.meta["custom"]["nested_region_config"].bw_compiler
            and node.meta.get("partitioner_tag") == "is_backward"
        )
    )


def _compile_invoke_subgraph_nodes_with_inductor(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Find invoke_subgraph call nodes needing compile, group
    # them by the name of the referenced get_attr subgraph (AssertionError if the
    # first arg is not a get_attr of a string name), and compile each distinct
    # subgraph once for all its users. Returns the updated graph module.
    map_subgraph_to_nodes = defaultdict(list)
    subgraphs: set[str] = set()

    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        if not _needs_inductor_compile(node):
            continue
        if node.args[0].op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {node.args[0].op}")
        subgraph_name = node.args[0].target
        if not isinstance(subgraph_name, str):
            raise AssertionError(f"Expected str, got {type(subgraph_name)}")
        subgraphs.add(subgraph_name)
        map_subgraph_to_nodes[subgraph_name].append(node)

    for subgraph in subgraphs:
        gm = _compile_submod(gm, subgraph, map_subgraph_to_nodes[subgraph])

    return gm


def _recursive_compile_invoke_subgraph_nodes(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Depth-first handling of nested graph modules: first
    # recurse into get_attr submodules that are graph modules (skipping get_attr
    # nodes that are themselves marked for compile, to avoid nested regional
    # compiles), then compile this module's own invoke_subgraph nodes. Returns gm.
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we don't do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_invoke_subgraph_nodes(submod)

    return _compile_invoke_subgraph_nodes_with_inductor(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor_invoke_subgraph(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Compile invoke_subgraph nodes if they have custom compiler specified
    in node.meta["nested_region_config"].bw_compiler or fw_compiler
    """
    # [FUNCTION DOC] Pass entry point: compile every invoke_subgraph node that has a
    # forward or backward compiler configured, recursively through nested graph
    # modules, with node metadata preservation disabled while transforming. The
    # example args are unused. Returns the result of wrapping the compiled graph
    # module with the debugging boxed no-op backend (empty example inputs).
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        compiled_gm = _recursive_compile_invoke_subgraph_nodes(gm)
        # TODO: might not need this boxed_nop after we switch to _RegionCompiler
        # pyrefly: ignore [bad-return]
        return torch._dynamo.backends.debugging.boxed_nop(
            compiled_gm, example_inputs=[]
        )
