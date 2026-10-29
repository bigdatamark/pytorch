# [MODULE DOC] Regional Inductor pass. Finds FX nodes annotated in their custom
# metadata with a compile_with_inductor entry (optionally naming an inductor
# region and inductor config overrides), groups them into partitions that are
# fused into submodules (prefix __marked_inductor_submod), compiles each such
# submodule ahead of time with standalone Inductor, and swaps the submodule call
# for a call to the compiled artifact. Public entry: regional_inductor.
# Import-time side effects: creates a module logger.
import contextlib
import functools
import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Wrap a callable object in a plain function (keeping its
    # metadata) so graph call_function nodes accept it. Returns the wrapper.
    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Forward all arguments to the wrapped callable.
        return fn(*args, **kwargs)

    return inner


@contextlib.contextmanager
def _disable_remat_for_regional_subcompile() -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager that temporarily turns off the
    # functorch tag-based remat config option for the duration of a nested
    # subcompile, restoring it afterwards. Yields nothing.
    # In torch.compile, regional_inductor subcompiles run after the enclosing
    # non-strict full graph has already been partitioned, so any graph-SAC
    # remat pass has already run before we reach this nested compile.
    # Rerunning remat here can see stage-2-reordered backward nodes that
    # violate remat's contiguous-backward-region assumption.
    with torch._functorch.config.patch(remat_using_tags_for_fwd_loss_bwd_graph=False):
        yield


def _compile_submod(gm: torch.fx.GraphModule, prefix: str) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: For each call_module node whose target starts with the
    # prefix, gather fake example values from its input nodes (RuntimeError if an
    # input has none), read inductor config overrides from the first annotated node
    # in the submodule, and validate keys against the inductor config (ValueError
    # for an unknown key). Compile the submodule ahead of time under patched config
    # with remat disabled (AssertionError if the result is not an AOT artifact),
    # insert a call to the artifact after the node, copy node metadata, replace
    # uses, erase the node and delete the submodule. Recompiles and returns gm.
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # Get inductor configs from annotation
            # TODO we should change partition when there are multiple differently
            # annotated regions.
            inductor_options: dict[str, Any] = {}
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "compile_with_inductor" in custom:
                        compile_value = custom["compile_with_inductor"]
                        if (
                            isinstance(compile_value, dict)
                            and "inductor_configs" in compile_value
                        ):
                            inductor_options = compile_value["inductor_configs"]
                            break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                inductor_options,
            )

            # Apply config patches before compilation
            import torch._inductor.config as inductor_config

            # Validate that all config keys exist
            for key in inductor_options:
                if not hasattr(inductor_config, key):
                    raise ValueError(
                        f"Invalid inductor config key '{key}' in regional_inductor annotation. "
                        f"Available config keys can be found in torch._inductor.config"
                    )

            with (
                inductor_config.patch(inductor_options),
                _disable_remat_for_regional_subcompile(),
                torch.fx.traceback._set_regional_inductor_subgraph_name(node.target),
            ):
                compiled_fn = torch._inductor.standalone_compile(
                    submod,
                    fake_inputs,
                    dynamic_shapes="from_tracing_context",
                    aot=True,
                    donate_graph_module=True,
                )
            if not isinstance(compiled_fn, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(compiled_fn)}"
                )
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] HELPER: True if the node is not a placeholder or output and its
    # custom metadata contains a compile_with_inductor entry.
    return bool(
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


class _RegionScooper:
    """
    Scoops out the inductor marked regions. It does NOT compile them.
    """
    # [CLASS DOC] HELPER: Stateless pass object that fuses annotated nodes into
    # submodules. Static methods do the work; calling an instance runs the
    # recursive version with node metadata preservation disabled.

    @staticmethod
    def scoop_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Group annotated nodes by region id (the optional
        # inductor_region value; all unnamed ones share one default region). If none
        # are found, log and return gm unchanged. Otherwise run the capability-based
        # partitioner separately per region (single-node partitions allowed) so
        # partitions never span regions, then fuse all partitions into submodules
        # named with the marked-submodule prefix, always returning tuples. Returns
        # the fused graph module (mutates the graph).
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch.fx.passes.operator_support import create_op_support
        from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

        # Group tagged nodes by region ID.  The region ID comes from the
        # optional "inductor_region" key inside the compile_with_inductor
        # annotation. When absent, all tagged nodes share a single default region
        _DEFAULT_REGION = object()
        regions: dict[object, set[torch.fx.Node]] = {}
        for node in gm.graph.nodes:
            if _needs_inductor_compile(node):
                compile_value = node.meta["custom"]["compile_with_inductor"]
                if (
                    isinstance(compile_value, dict)
                    and "inductor_region" in compile_value
                ):
                    rid = compile_value["inductor_region"]
                else:
                    rid = _DEFAULT_REGION
                regions.setdefault(rid, set()).add(node)

        if not regions:
            logger.info("No inductor marked nodes found")
            return gm

        # Run CapabilityBasedPartitioner per region to get cycle-safe partitions
        # without merging across region boundaries.
        def _is_in_region(
            region_nodes: set[torch.fx.Node],
        ) -> Callable[[Mapping[str, torch.nn.Module], torch.fx.Node], bool]:
            # [FUNCTION DOC] HELPER: Build a support predicate that accepts exactly
            # the nodes in the given set.
            def is_node_supported(
                _submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
            ) -> bool:
                # [FUNCTION DOC] HELPER: True if the node belongs to the region set.
                return node in region_nodes

            return is_node_supported

        all_partitions: list[dict[torch.fx.Node, int | None]] = []
        for region_nodes in regions.values():
            support = create_op_support(_is_in_region(region_nodes))
            partitioner = CapabilityBasedPartitioner(
                gm, support, allows_single_node_partition=True
            )
            for partition in partitioner.propose_partitions():
                all_partitions.append(partition.nodes)

        return fuse_by_partitions(
            gm,
            all_partitions,
            prefix="__marked_inductor_submod",
            always_return_tuple=True,
        )

    @staticmethod
    def recursively_scoop_regions(
        gm: torch.fx.GraphModule, _processed: set[int] | None = None
    ) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Recurse into graph-module attributes (each
        # distinct module once, tracked by identity in the processed set; skipping
        # get_attr nodes themselves marked for compile), scooping their regions
        # first, then scoop this module. Returns the resulting graph module.
        if _processed is None:
            _processed = set()
        for node in gm.graph.find_nodes(op="get_attr"):
            if _needs_inductor_compile(node):
                # If the get_attr itself is marked for compile, the outer graph will
                # take care of it. If we don't do that, we end up with nested
                # regional inductor compiles that do not work well.
                continue
            submod = getattr(gm, node.target)
            # Track by id: multiple get_attr nodes may reference the same GraphModule
            if (
                isinstance(submod, torch.fx.GraphModule)
                and id(submod) not in _processed
            ):
                _processed.add(id(submod))
                _RegionScooper.recursively_scoop_regions(submod, _processed)

        return _RegionScooper.scoop_regions(gm)

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Recursively scoop regions with node metadata
        # preservation disabled; returns the graph module.
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionScooper.recursively_scoop_regions(gm)


class _RegionCompiler:
    """
    Compiles the scooped out regions.
    """
    # [CLASS DOC] HELPER: Stateless pass object that compiles the fused marked
    # submodules produced by the scooper. Static methods do the work; calling an
    # instance runs the recursive version with node metadata preservation disabled.

    @staticmethod
    def compile_region(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Compile the marked submodules of this graph module,
        # switch its graph to boxed-calling-convention codegen, and recompile.
        # Returns the module.
        from torch.fx.graph import _BoxedCodeGen

        gm = _compile_submod(gm, "__marked_inductor_submod")
        gm.graph.set_codegen(_BoxedCodeGen())
        gm.recompile()
        return gm

    @staticmethod
    def recursively_compile_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Detect whether this module directly calls a marked
        # submodule, recurse into all graph-module attributes, then compile this
        # module's regions if any were found. Returns the module.
        # Find if the graph module has a scooped out region
        found_region = False
        for node in gm.graph.find_nodes(op="call_module"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                if node.target.startswith("__marked_inductor_submod"):
                    found_region = True

        # Recurse through the subgraphs
        for node in gm.graph.find_nodes(op="get_attr"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _RegionCompiler.recursively_compile_regions(submod)

        if found_region:
            return _RegionCompiler.compile_region(gm)
        return gm

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Recursively compile regions with node metadata
        # preservation disabled; returns the module.
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionCompiler.recursively_compile_regions(gm)


def _create_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Run the region scooper on gm; returns the result.
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionScooper()(gm)


def _compile_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Run the region compiler on gm; returns the result.
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionCompiler()(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Scoops out inductor marked regions and compiles them with inductor.

    Inductor options should be provided via the annotation API::

        with fx_traceback.annotate(
            {
                "compile_with_inductor": {
                    "inductor_configs": {
                        "max_autotune": True,
                        "triton.cudagraphs": False,
                    }
                }
            }
        ):
            ...
    """
    # [FUNCTION DOC] Pass entry point: first fuse annotated nodes into marked
    # submodules, then compile them with Inductor, all with node metadata
    # preservation disabled. Example args are unused. Returns the compiled graph
    # module, or, when the functorch force_autograd_cache option is on, a
    # RegionalOutputCode wrapping it. Mutates the graph; may raise ValueError or
    # RuntimeError for invalid configs or missing fake values.

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues

    with torch.fx.traceback.preserve_node_meta(enable=False):
        gm = _create_inductor_marked_regions(gm)
        gm = _compile_inductor_marked_regions(gm)
        if torch._functorch.config.force_autograd_cache:
            from torch._inductor.output_code import RegionalOutputCode

            return RegionalOutputCode(gm)  # type: ignore[return-value]
        return gm
