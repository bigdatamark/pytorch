# [MODULE DOC] Graph capture for export built on Dynamo's full-graph capture. Provides the
# entry points that trace a callable/module into an FX GraphModule whose forward replays
# Dynamo bytecode for flattening/unflattening (dynamo_graph_capture_for_export), or a cleaner
# variant rebuilt by an FX transformer with pytree codegen (_dynamo_graph_capture_for_export).
# Also holds helpers that strip export-root naming artifacts, constraint-violation reporting,
# and the bytecode flatten/unflatten callables and custom code generator. Import-time side
# effects: creates a module logger only.
from __future__ import annotations

import inspect
import logging
import sys
import traceback
import types
from collections import namedtuple
from contextlib import nullcontext
from typing import Any, cast, TYPE_CHECKING, TypeVar

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.convert_frame import CaptureOutput, fullgraph_capture, get_traced_fn
from torch._dynamo.decorators import disable as dynamo_disable
from torch._dynamo.eval_frame import argument_names, check_user_input_output
from torch._dynamo.exc import UserErrorType
from torch._dynamo.source import GetItemSource
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._export.utils import _compiling_state_context
from torch._guards import detect_fake_mode, Source, TracingContext
from torch.export.dynamic_shapes import _RelaxedConstraint, Constraint
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._dynamo.output_graph import OutputReturnInfo
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.export._trace import _DynamicShapesInput
    from torch.fx import Node
    from torch.fx.node import Argument, Target

from torch.fx.experimental._spec_binding import _bind_spec_to_args
from torch.fx.experimental.dynamic_spec import (
    _coerce_to_shapes_spec,
    IntermediateSpec,
    ParamsSpec,
    ShapesSpec,
)


T = TypeVar("T")
log = logging.getLogger(__name__)


def post_process_error_msg(
    constraint_violation_error: ConstraintViolationError,
    func: Callable[..., Any],
    args: Any,
    kwargs: Any,
) -> ConstraintViolationError:
    # [FUNCTION DOC] Rewrites the message of a constraint violation error so that source names
    # refer to the user's original callable inputs (instead of the internal traced wrapper):
    # computes flat input paths for the call arguments from the callable signature and replaces
    # sources in the first error argument. Mutates and returns the same error object.
    """
    Because we trace a different callable, the sources are all messed up.
    Manually patch them so the error message looks correct.
    """
    from torch.export._unlift import _get_input_paths, _replace_sources

    orig_sig = inspect.signature(func)
    flat_input_paths = _get_input_paths((args, kwargs), orig_sig)
    if constraint_violation_error.args:
        constraint_violation_error.args = (
            _replace_sources(constraint_violation_error.args[0], flat_input_paths),
        )
    return constraint_violation_error


EXPORT_ROOT_REPLACEMENTS = [
    ("__export_root_", "_"),
    ("_export_root.", ""),
    ("._export_root", ""),
]


def clean_export_root_string(text: str) -> str:
    # [FUNCTION DOC] Returns text with the export-root naming artifacts removed or replaced,
    # applying each replacement rule in order. Pure.
    """Generic utility to clean export_root patterns from strings."""
    result = text
    for pattern, replacement in EXPORT_ROOT_REPLACEMENTS:
        result = result.replace(pattern, replacement)
    return result


def clean_nn_module_stack_and_source_fn(
    graph_module: torch.fx.GraphModule, is_inline_builtin: bool = False
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Mutates every node's module-stack and source-function-stack metadata, and the
    # graph module's flat-name-to-original-name metadata, to remove export-root artifacts.
    # Module stack: drops the root export entry, cleans keys and names, and (if
    # is_inline_builtin) drops entries that refer to self. Returns the same graph module.
    """
    Clean up nn_module_stack metadata by removing export_root references.

    Removes the _export_root module references from nn_module_stack metadata
    in graph nodes, which are artifacts from the export process. Fixes two patterns:

    1. Keys: Removes "__export_root_" and "__modules['_export_root']_" prefixes
       - Normal case: "L__self____export_root_child" -> "L__self__child"
       - inline_builtin case: Uses numeric ID strings like "140468831433840"

    2. Values: Removes "._export_root" and "._modules['_export_root']" from child names
       e.g., "L['self']._export_root.child" -> "L['self'].child"
       e.g., "L['self']._modules['_export_root'].child" -> "L['self'].child"

    Also removes the root export entry "L__self____export_root" entirely.

    Args:
        graph_module: The GraphModule to clean up
        is_inline_builtin: If True, keys are numeric ID strings and self references
                          (L['self']) are filtered out

    Returns:
        The cleaned GraphModule (modified in-place)
    """

    def _process_nn_module_stack(
        nn_module_stack: dict[str, tuple[str, T]],
    ) -> dict[str, tuple[str, T]]:
        # [FUNCTION DOC] HELPER: Given a mapping key -> (name, class), removes the root export
        # entry (mutating the input), cleans keys and names, skips self references when
        # inline-builtin mode is set, and returns a new mapping.
        if "L__self____export_root" in nn_module_stack:
            del nn_module_stack["L__self____export_root"]

        # Clean up remaining entries
        cleaned_stack = {}
        for key, (child_name, child_class) in nn_module_stack.items():
            # Clean key by removing export_root patterns
            clean_key = clean_export_root_string(key)

            # Clean child_name by removing export_root patterns
            clean_name = clean_export_root_string(child_name)

            # Skip self reference for inline builtin case
            if is_inline_builtin and clean_name == "L['self']":
                continue

            cleaned_stack[clean_key] = (clean_name, child_class)
        return cleaned_stack

    def _process_source_fn(source_fn_stack: Iterable[T]) -> Iterable[T]:
        # [FUNCTION DOC] HELPER: Returns a list where each (string name, class) pair has its name
        # cleaned; all other items are kept unchanged.
        cleaned_stack = []
        for item in source_fn_stack:
            if isinstance(item, tuple) and len(item) == 2:
                name, cls = item
                if isinstance(name, str):
                    clean_name = clean_export_root_string(name)
                    cleaned_stack.append((clean_name, cls))
                else:
                    cleaned_stack.append(item)
            else:
                # pyrefly: ignore [bad-argument-type]
                cleaned_stack.append(item)
        # pyrefly: ignore [bad-return]
        return cleaned_stack

    for node in graph_module.graph.nodes:
        if "nn_module_stack" in node.meta:
            node.meta["nn_module_stack"] = _process_nn_module_stack(
                node.meta["nn_module_stack"].copy()
            )

        source_fn_stack = node.meta.get("source_fn_stack", None)
        if source_fn_stack:
            node.meta["source_fn_stack"] = _process_source_fn(source_fn_stack.copy())

    if "dynamo_flat_name_to_original_fqn" in graph_module.meta:
        # Clean up flat name to original fqn mapping
        clean_name_to_original_fqn = {}
        for flat_name, original_fqn in graph_module.meta[
            "dynamo_flat_name_to_original_fqn"
        ].items():
            clean_name_to_original_fqn[clean_export_root_string(flat_name)] = (
                clean_export_root_string(original_fqn)
            )
        graph_module.meta["dynamo_flat_name_to_original_fqn"] = (
            clean_name_to_original_fqn
        )

    return graph_module


def clean_export_root(graph_module: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Renames, in place, attribute-fetch targets and called-submodule targets (and
    # node names of module calls) that contain export-root artifacts; the underlying
    # parameters or submodules are moved to the new names. A module called multiple times is
    # renamed once and all its call nodes updated. Raises AssertionError if an attribute to
    # move is missing or a target is not a string.
    """Remove export_root artifacts from FX graph in-place"""

    # Unlike getattr node, call_module can be invoked multiple times
    # In those cases, we should fix all invocations of call_module
    clean_named_module_map: dict[str, str] = {}

    # Update get_attr nodes in-place
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            old_target = node.target
            new_target = clean_export_root_string(old_target)
            if new_target != old_target:
                node.target = new_target
                if not hasattr(graph_module, old_target):
                    raise AssertionError(
                        f"graph_module missing attribute {old_target!r}"
                    )
                # Move the parameter to the new name
                param = torch.fx.graph_module._get_attr(graph_module, old_target)
                torch.fx.graph_module._assign_attr(param, graph_module, new_target)
                torch.fx.graph_module._del_attr(graph_module, old_target)
        # Dynamo will only have one nested level
        if node.op == "call_module":
            old_target = node.target
            if not isinstance(old_target, str):
                raise AssertionError(
                    f"expected old_target to be str, got {type(old_target)}"
                )
            new_target = clean_export_root_string(old_target)
            if not isinstance(new_target, str):
                raise AssertionError(
                    f"expected new_target to be str, got {type(new_target)}"
                )
            new_name = clean_export_root_string(node.name)
            if new_target == old_target:
                continue

            # if this module has already been cleaned before, just lookup from map.
            if old_target in clean_named_module_map:
                node.target = clean_named_module_map[old_target]
                node.name = new_name
                continue
            target = graph_module.get_submodule(old_target)
            graph_module.delete_submodule(old_target)
            graph_module.add_submodule(new_target, target)
            node.target = new_target
            node.name = new_name
            clean_named_module_map[old_target] = new_target


class ModuleToTrace(torch.nn.Module):
    # [CLASS DOC] Wrapper module traced instead of the user callable so inputs arrive flat.
    # State: _export_root (the wrapped user callable/module) and in_spec (pytree spec used to
    # rebuild the original args and kwargs from flat arguments).
    def __init__(self, foo: Any, in_spec: Any) -> None:
        # [FUNCTION DOC] Stores the wrapped callable under the export-root attribute and the
        # input pytree spec.
        super().__init__()
        self._export_root = foo
        self.in_spec = in_spec

    def forward(self, *flat_args: Any) -> ExportTracerOutput:
        # [FUNCTION DOC] Unflattens flat_args into the original args/kwargs, calls the wrapped
        # callable, flattens its result, and returns a named pair of (flat outputs, output spec).
        args, kwargs = pytree.tree_unflatten(flat_args, self.in_spec)
        res = self._export_root(*args, **kwargs)
        out_flat, out_spec = pytree.tree_flatten(res)
        return ExportTracerOutput(out_flat, out_spec)


ExportTracerOutput = namedtuple("ExportTracerOutput", ["flat_args", "out_spec"])


# mypy: disable-error-code="no-untyped-def,var-annotated,assignment,index,operator"
class DynamoGraphTransformer(torch.fx.Transformer):
    """Graph transformer for dynamo export that flattens inputs/outputs without complex matching."""

    # [CLASS DOC] FX transformer that rebuilds a Dynamo-captured graph so that its placeholders
    # are one per flattened user input (named arg_N) and its outputs follow the output map.
    # State: flat_inputs, per-input dynamic dims, graph_input_order (user input index to graph
    # placeholder index), graph_output_map (output index to kind and value), optional fake mode,
    # graph_inputs (graph placeholder index to source), the original placeholders and output
    # node, new_input_nodes (user index to new placeholder) and old_to_new_mapping. Inputs not
    # matched are turned into get_attr nodes when they are captured objects.
    def __init__(
        self,
        module: torch.fx.GraphModule,
        flat_inputs: list[Any],
        flat_args_dynamic_dims: list[set[int]],
        graph_input_order: dict[int, int],
        graph_output_map: dict[int, OutputReturnInfo],
        fake_mode: FakeTensorMode | None = None,
        graph_inputs: dict[int, Source | None] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the transformer over module and stores the inputs, dynamic
        # dims, order and output maps, fake mode and graph inputs (empty if None). Collects the
        # original placeholders and output node, creates the new flattened placeholders, and
        # builds the old-to-new placeholder mapping. Raises AssertionError if the dynamic dims
        # list length differs from the number of flat inputs, or StopIteration if the module
        # has no output node.
        super().__init__(module)

        if len(flat_args_dynamic_dims) != len(flat_inputs):
            raise AssertionError(
                f"flat_args_dynamic_dims length ({len(flat_args_dynamic_dims)}) "
                f"must match flat_inputs length ({len(flat_inputs)})"
            )

        self.flat_inputs = flat_inputs
        self.flat_args_dynamic_dims = flat_args_dynamic_dims
        self.graph_input_order = graph_input_order
        self.graph_output_map = graph_output_map
        self.fake_mode = fake_mode
        self.graph_inputs = graph_inputs or {}

        # Get original placeholders and output
        self.placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
        self.output_node = next(n for n in module.graph.nodes if n.op == "output")

        # Create new flattened input placeholders
        self.new_input_nodes: dict[int, torch.fx.Node] = {}
        self._create_flattened_inputs()

        # Iterator for replacing old placeholders
        self.old_to_new_mapping = {}
        self._create_placeholder_mapping()

    def _create_flattened_inputs(self) -> None:
        # [FUNCTION DOC] HELPER: Creates one new placeholder per flat input, recording it in
        # new_input_nodes. Copies non-value metadata from the matching original placeholder, and
        # sets the value metadata: a fake tensor with dynamic/static dims for tensors (when a
        # fake mode exists), the wrapped value for integer wrapper objects, else the raw input.
        """Create new placeholder nodes for flattened inputs with proper fake tensors."""
        for i in range(len(self.flat_inputs)):
            placeholder = super().placeholder(f"arg_{i}", (), {})

            # Check if this user input (index i) maps to a graph placeholder
            if i in self.graph_input_order:
                # graph_input_order[i] gives us which graph placeholder this user input corresponds to
                graph_placeholder_idx = self.graph_input_order[i]
                if graph_placeholder_idx < len(self.placeholders):
                    orig_placeholder = self.placeholders[graph_placeholder_idx]
                    # Copy other metadata but not "val" yet
                    for key, value in orig_placeholder.meta.items():
                        if key != "val":
                            placeholder.node.meta[key] = value

            # Always ensure we have proper "val" metadata from fake tensor
            if self.fake_mode is not None and isinstance(
                self.flat_inputs[i], torch.Tensor
            ):
                placeholder.node.meta["val"] = self.fake_mode.from_tensor(
                    self.flat_inputs[i],
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[
                            (
                                DimDynamic.DYNAMIC
                                if d in self.flat_args_dynamic_dims[i]
                                else DimDynamic.STATIC
                            )
                            for d in range(len(self.flat_inputs[i].shape))
                        ],
                        constraint_sizes=[None] * len(self.flat_inputs[i].shape),
                    ),
                )
            elif hasattr(self.flat_inputs[i], "val"):  # _IntWrapper case
                placeholder.node.meta["val"] = self.flat_inputs[i].val
            else:
                placeholder.node.meta["val"] = self.flat_inputs[i]

            # pyrefly: ignore [unsupported-operation]
            self.new_input_nodes[i] = placeholder

    def _create_placeholder_mapping(self) -> None:
        # [FUNCTION DOC] HELPER: Fills old_to_new_mapping from each original graph placeholder to
        # the new placeholder of the user input mapped to it, for in-range placeholder indices.
        """Create mapping from old placeholders to new ones."""
        # graph_input_order maps: user_input_index -> graph_placeholder_index
        # We need to create: old_graph_placeholder -> new_user_input_placeholder
        for user_input_idx, graph_placeholder_idx in self.graph_input_order.items():
            if graph_placeholder_idx < len(self.placeholders):
                old_placeholder = self.placeholders[graph_placeholder_idx]
                new_placeholder = self.new_input_nodes[user_input_idx]
                self.old_to_new_mapping[old_placeholder] = new_placeholder

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Transformer hook for placeholder nodes. If the current node was mapped,
        # returns its new placeholder after copying extra metadata (tensor dict, example value,
        # unbacked bindings, and the value metadata only if missing). Otherwise, if the
        # placeholder's source is a captured (non item-access) object with an example value,
        # registers it as a buffer (tensors) or plain attribute on the module and returns an
        # attribute-fetch proxy carrying copied metadata. Else falls back to the default
        # placeholder behavior.
        """Replace old placeholders with new flattened ones."""
        # Return the corresponding new placeholder
        if self.current_node in self.old_to_new_mapping:
            new_arg = self.old_to_new_mapping[self.current_node]

            # Copy over additional metadata from current node, but don't overwrite "val"
            for key in ["tensor_dict", "example_value", "unbacked_bindings"]:
                if key in self.current_node.meta:
                    new_arg.node.meta[key] = self.current_node.meta[key]

            # Only copy "val" if we don't already have a good one
            if "val" in self.current_node.meta and "val" not in new_arg.node.meta:
                new_arg.node.meta["val"] = self.current_node.meta["val"]

            return new_arg
        else:
            # Convert captured objects (e.g., opaque objects from closures) to
            # get_attr nodes
            placeholder_idx = self.placeholders.index(self.current_node)
            if placeholder_idx in self.graph_inputs:
                source = self.graph_inputs[placeholder_idx]
                if not isinstance(source, GetItemSource):
                    example_val = self.current_node.meta.get(
                        "val"
                    ) or self.current_node.meta.get("example_value")
                    if example_val is not None:
                        attr_name = f"_captured_{placeholder_idx}"
                        if isinstance(example_val, torch.Tensor):
                            self.module.register_buffer(attr_name, example_val)
                        else:
                            setattr(self.module, attr_name, example_val)
                        result = self.tracer.create_proxy("get_attr", attr_name, (), {})
                        result.node.meta = self.current_node.meta.copy()
                        result.node.meta["val"] = example_val
                        return result
            return super().placeholder(target, args, kwargs)

    def output(
        self, target: Target, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        # [FUNCTION DOC] Transformer hook for the output node. Builds the new output tuple by
        # visiting graph_output_map in sorted key order: graph outputs are taken from the
        # original outputs by index, input passthroughs from the new input placeholders, and
        # constants are inserted as is. Returns the output proxy for that tuple.
        """Transform output according to graph_output_map."""
        original_outputs = args[0]

        # Build new output list based on graph_output_map
        new_outputs = []
        for i in sorted(self.graph_output_map.keys()):
            output_type, val = self.graph_output_map[i]

            if output_type == "graph_out":
                new_outputs.append(original_outputs[cast(int, val)])
            elif output_type == "input":
                input_idx = cast(GetItemSource, val).index
                new_outputs.append(self.new_input_nodes[input_idx])
            elif output_type == "constant":
                new_outputs.append(val)

        return super().output(target, (tuple(new_outputs),), {})

    def run_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Records n as the current node, runs the default transformation, and if
        # the result is a different node, copies value, example value and unbacked binding
        # metadata from n and renames the result to n's name (except for output nodes).
        # Returns the result.
        """Run node transformation and preserve metadata."""
        self.current_node = n
        result = super().run_node(n)

        # Copy important metadata
        if hasattr(result, "node") and result.node is not n:
            for key in ["val", "example_value", "unbacked_bindings"]:
                if key in n.meta:
                    result.node.meta[key] = n.meta[key]

            # Preserve node names (except output)
            if n.op != "output" and hasattr(n, "name"):
                result.node._rename(n.name)

        return result

    def transform(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Runs the base transformation and copies the flat-name-to-original-name
        # and compile id entries from the source module's metadata (if present) to the result
        # module. Returns the new graph module.
        """Perform the graph transformation and copy module metadata."""
        result_gm = super().transform()

        # Copy module metadata like the original implementation
        if hasattr(self.module, "meta"):
            # pyrefly: ignore [unsupported-operation]
            if "dynamo_flat_name_to_original_fqn" in self.module.meta:
                # pyrefly: ignore [bad-index]
                result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[
                    # pyrefly: ignore [bad-index]
                    "dynamo_flat_name_to_original_fqn"
                ]
            # pyrefly: ignore [unsupported-operation]
            if "dynamo_compile_id" in self.module.meta:
                # pyrefly: ignore [bad-index]
                result_gm.meta["dynamo_compile_id"] = self.module.meta[
                    # pyrefly: ignore [bad-index]
                    "dynamo_compile_id"
                ]

        return result_gm


def _suggest_or_raise_constraint_violation(
    module_to_trace: torch.nn.Module,
    orig_callable: Callable[..., Any],
    fake_mode: FakeTensorMode | None,
    graph_capture_output: CaptureOutput,
    args: Any,
    kwargs: Any,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
) -> None:
    # [FUNCTION DOC] HELPER: After capture, builds guards to detect constraint violations, then
    # (if a shape environment exists and the traced forward is not an operator) solves dimension
    # constraints and produces a suggestion message. If there is a violation, appends the
    # message to it; if forced specializations exist, creates a violation error; otherwise logs
    # an info summary. Also creates an error if any shape variable range is for a static
    # integer. Raises the (post-processed) ConstraintViolationError when one exists; returns
    # nothing otherwise. Mutates shape environment state via solving.
    constraint_violation_error = None
    try:
        # Check if we have any constraint violations
        fn, _ = get_traced_fn(module_to_trace)
        graph_capture_output.graph_capture_output.build_guards(fn.__code__)
    except ConstraintViolationError as e:
        constraint_violation_error = e

    if (
        (shape_env := getattr(fake_mode, "shape_env", None)) is not None
        and (dim_constraints := shape_env.dim_constraints) is not None
        and not isinstance(
            module_to_trace.forward,
            torch._ops.OpOverloadPacket | torch._ops.OpOverload,
        )
    ):
        dim_constraints.solve()

        forced_specializations = dim_constraints.forced_specializations()

        msg = dim_constraints.prettify_results(
            inspect.signature(orig_callable),  # type: ignore[attr-defined]
            dynamic_shapes,
            constraint_violation_error,
            forced_specializations,
        )
        if constraint_violation_error:
            if constraint_violation_error.args:
                constraint_violation_error.args = (
                    constraint_violation_error.args[0] + msg,
                )
            else:
                constraint_violation_error.args = (msg,)
        else:
            if forced_specializations:
                constraint_violation_error = ConstraintViolationError(msg)
            else:
                log.info(
                    "Summary of dimension constraints:%s",
                    msg,
                )

        # Error if we have any constraints on static values

        for k in shape_env.var_to_range:
            if isinstance(k, sympy.Integer):
                constraint_violation_error = ConstraintViolationError(
                    f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                    "It appears that you're trying to set a constraint on a "
                    f"value which we evaluated to have a static value of {k}. "
                    'Set TORCH_LOGS="+export" for more information.'
                )
    if constraint_violation_error:
        constraint_violation_error = post_process_error_msg(
            constraint_violation_error, orig_callable, args, kwargs
        )
        raise constraint_violation_error


def _normalize_shuffle_graph(shuffle_gm: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] HELPER: Eliminates dead code from the graph, recompiles it, and re-registers
    # every buffer by deleting and re-setting its attribute. Mutates in place.
    shuffle_gm.graph.eliminate_dead_code()
    shuffle_gm.recompile()
    for name, buffer in list(shuffle_gm.named_buffers()):
        delattr(shuffle_gm, name)
        setattr(shuffle_gm, name, buffer)


def normalize_graph_module(gm: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] For each placeholder node, sets its value metadata to its example value
    # metadata (KeyError if absent). Mutates in place.
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = node.meta["example_value"]


class InputProcessor:
    # [CLASS DOC] Callable that turns a flat inputs tuple back into (args, kwargs) for the
    # original call. State: root (module or callable prepended as first argument, or None),
    # num_args (count of positional args) and kwarg_names (names for trailing values).
    def __init__(
        self,
        root: object,
        num_args: int,
        kwarg_names: list[str],
    ) -> None:
        # [FUNCTION DOC] Stores root, number of positional args and keyword names.
        self.root = root
        self.num_args = num_args
        self.kwarg_names = kwarg_names

    def __call__(
        self, inputs: tuple[object, ...]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        # [FUNCTION DOC] Splits inputs into positional args (first num_args) and keyword args
        # (remaining values zipped with kwarg_names). If a root exists it is prepended to the
        # positional args; when root is a graph module, first checks that its codegen is the
        # bytecode codegen whose flatten object refers back to this processor, raising
        # AssertionError otherwise. Returns (args, kwargs).
        args = inputs
        # pyrefly: ignore [implicit-any]
        kwargs = {}
        if len(args) > self.num_args:
            kwargs = dict(zip(self.kwarg_names, args[self.num_args :]))
            args = args[: self.num_args]
        if self.root is not None:
            if isinstance(self.root, torch.fx.GraphModule):
                if not isinstance(self.root.graph._codegen, _DynamoBytecodeCodeGen):
                    raise AssertionError(
                        f"expected _DynamoBytecodeCodeGen, got {type(self.root.graph._codegen)}"
                    )
                if not hasattr(
                    self.root.graph._codegen.dynamo_bytecode_flatten, "input_processor"
                ):
                    raise AssertionError(
                        "dynamo_bytecode_flatten missing input_processor attribute"
                    )
                if (
                    self.root.graph._codegen.dynamo_bytecode_flatten.input_processor
                    is not self
                ):
                    raise AssertionError(
                        "dynamo_bytecode_flatten.input_processor is not self"
                    )
            args = (self.root, *args)
        return args, kwargs


class Yield(Exception):
    # [CLASS DOC] HELPER: Sentinel exception raised by the dummy backend to abort the replayed
    # compiled call once the graph inputs have been captured.
    pass


class DynamoBytecodeFlatten:
    # [CLASS DOC] Callable that, given the user's inputs, replays the captured Dynamo bytecode up
    # to the graph call and returns the flat list of graph inputs. State: input_processor, the
    # capture output, globals for the replay, and gm_inputs (last captured graph inputs).
    def __init__(
        self,
        input_processor: InputProcessor,
        out: CaptureOutput,
        f_globals: dict[str, object],
    ) -> None:
        # [FUNCTION DOC] Stores the processor, capture output and globals; gm_inputs starts None.
        self.input_processor = input_processor
        self.out = out
        self.f_globals = f_globals
        self.gm_inputs: tuple[Any, ...] | None = None

    @dynamo_disable(reason="do not trace internal dynamo graph capture")  # type: ignore[misc]
    def __call__(self, *inputs: object) -> object:
        # [FUNCTION DOC] Rebuilds args/kwargs from inputs and runs the captured forward with a
        # dummy backend that records the graph inputs and aborts via the sentinel exception.
        # Returns the recorded graph inputs. Raises RuntimeError if the call finishes without
        # reaching the backend, AssertionError if nothing was recorded. Not traced by Dynamo.
        def backend_dummy(*example_inputs: object) -> None:
            # [FUNCTION DOC] HELPER: Records the example inputs on the flatten object and raises
            # the sentinel to stop execution.
            self.gm_inputs = example_inputs
            raise Yield

        args, kwargs = self.input_processor(inputs)
        try:
            self.out.forward_callable(
                compiled_fn=backend_dummy, extra_globals=self.f_globals
            )(*args, **kwargs)
        except Yield:
            if self.gm_inputs is None:
                raise AssertionError("gm_inputs must not be None after Yield") from None
            return self.gm_inputs
        raise RuntimeError


class DynamoBytecodeUnflatten:
    # [CLASS DOC] Callable that, given the flat graph outputs and the original user inputs,
    # replays the captured Dynamo bytecode with a backend that returns those outputs, producing
    # the user-visible result structure. State: input_processor, capture output, globals.
    def __init__(
        self,
        input_processor: InputProcessor,
        out: CaptureOutput,
        f_globals: dict[str, object],
    ) -> None:
        # [FUNCTION DOC] Stores the processor, capture output and globals.
        self.input_processor = input_processor
        self.out = out
        self.f_globals = f_globals

    @dynamo_disable(reason="do not trace internal dynamo graph capture")  # type: ignore[misc]
    def __call__(
        self, flat_outs: Sequence[object], inputs: tuple[object, ...]
    ) -> object:
        # [FUNCTION DOC] Rebuilds args/kwargs from inputs, then runs the captured forward with
        # torch dispatch disabled and a dummy backend that returns flat_outs; returns the
        # resulting user-level output. Not traced by Dynamo.
        def backend_dummy(*example_inputs: object) -> Sequence[object]:
            # [FUNCTION DOC] HELPER: Ignores its inputs and returns the supplied flat outputs.
            return flat_outs

        args, kwargs = self.input_processor(inputs)
        with torch._C._DisableTorchDispatch():
            results = self.out.forward_callable(
                compiled_fn=backend_dummy, extra_globals=self.f_globals
            )(*args, **kwargs)
        return results


def create_fx_graph_from_captured_output(
    out: CaptureOutput, mod: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Turns a capture output into the final export graph module whose forward
    # replays bytecode. Validates the flat user inputs, takes the captured graph module, and if
    # the traced root is an nn.Module copies its parameters, buffers, submodules, non-persistent
    # buffer set, extra instance attributes and hooks onto the graph module (with a custom
    # wrapped call that disables the hooks while preserving their state). Installs the bytecode
    # code generator plus flatten/unflatten objects as attributes, normalizes placeholder
    # values, removes the parameter-name-to-source attribute, recompiles, and stores module
    # call specs, fake mode (allowing non-fake inputs) and a tracing context in metadata.
    # Raises AssertionError if backend input is missing or attribute conflicts exist.
    if out.backend_input is None:
        raise AssertionError("out.backend_input must not be None")
    backend_input = out.backend_input

    _, root = torch._dynamo.convert_frame.get_traced_fn(mod)

    flat_real_args = pytree.tree_leaves((args, kwargs))
    torch._dynamo.eval_frame.check_user_input_output(
        flat_real_args, UserErrorType.INVALID_INPUT
    )
    f_globals = out.graph_capture_output.f_globals

    graph_module = backend_input.graph_module
    if isinstance(root, torch.nn.Module):
        graph_module._parameters = root._parameters
        graph_module._buffers = root._buffers
        if not all(not hasattr(graph_module, m) for m in root._modules):
            raise AssertionError(
                "graph_module already has attributes conflicting with root._modules"
            )
        graph_module._modules.update(root._modules)
        graph_module._non_persistent_buffers_set = root._non_persistent_buffers_set
        if sys.version_info >= (3, 14):
            import annotationlib  # added in 3.14

            annotations = annotationlib.get_annotations(torch.nn.Module)
        else:
            annotations = getattr(torch.nn.Module, "__annotations__", None)
        for name, value in root.__dict__.items():
            if annotations and name not in annotations:
                graph_module.__dict__[name] = value
        graph_module._forward_hooks = root._forward_hooks.copy()
        graph_module._forward_pre_hooks = root._forward_pre_hooks.copy()
        graph_module._backward_hooks = root._backward_hooks.copy()
        graph_module._backward_pre_hooks = root._backward_pre_hooks.copy()
        if graph_module._forward_hooks or graph_module._forward_pre_hooks:
            # Even forward hooks are traced through, they still capture a bunch
            # of state through closure. We need to make sure these data are
            # accessible through the captured module (but the hooks should be
            # disabled).
            if getattr(graph_module, "_wrapped_call", None) is None:
                raise AssertionError("graph_module._wrapped_call must not be None")
            if not isinstance(
                graph_module._wrapped_call, torch.fx.graph_module._WrappedCall
            ):
                raise AssertionError(
                    f"expected _WrappedCall, got {type(graph_module._wrapped_call)}"
                )
            if graph_module._wrapped_call.cls_call is not None:
                raise AssertionError("graph_module._wrapped_call.cls_call must be None")

            def dynamo_wrapped_call(self, *args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Replacement class-level call. Temporarily installs an
                # instance forward that restores the forward hooks only for the inner forward,
                # clears the hook dicts so the outer call machinery does not run them (they were
                # already traced), calls the parent call, and in a finally block removes the
                # instance forward and restores the hooks. Raises AssertionError if an instance
                # forward already exists.
                if "forward" in self.__dict__:
                    raise AssertionError("'forward' should not be in self.__dict__")

                fwd_hooks = self._forward_hooks
                fwd_pre_hooks = self._forward_pre_hooks
                original_forward = type(self).forward

                def patched_forward(self, *args: object, **kwargs: object) -> object:
                    # [FUNCTION DOC] HELPER: Restores the saved hooks on self, then calls the
                    # original class forward and returns its result.
                    self._forward_hooks = fwd_hooks
                    self._forward_pre_hooks = fwd_pre_hooks
                    return original_forward(self, *args, **kwargs)

                try:
                    self.forward = types.MethodType(patched_forward, self)
                    # pyrefly: ignore [implicit-any]
                    self._forward_hooks = {}
                    # pyrefly: ignore [implicit-any]
                    self._forward_pre_hooks = {}
                    # pyrefly: ignore [invalid-argument]
                    return super(type(self), self).__call__(*args, **kwargs)
                finally:
                    self.__dict__.pop("forward")
                    self._forward_hooks = fwd_hooks
                    self._forward_pre_hooks = fwd_pre_hooks

            # pyrefly: ignore [bad-assignment]
            graph_module._wrapped_call.cls_call = dynamo_wrapped_call

    root = graph_module if isinstance(root, torch.nn.Module) else root
    input_processor = InputProcessor(root, len(args), list(kwargs.keys()))
    dynamo_bytecode_flatten = DynamoBytecodeFlatten(input_processor, out, f_globals)
    dynamo_bytecode_unflatten = DynamoBytecodeUnflatten(input_processor, out, f_globals)

    graph_module.graph._codegen = _DynamoBytecodeCodeGen(
        argument_names(inspect.signature(mod), args, kwargs),
        dynamo_bytecode_flatten,
        dynamo_bytecode_unflatten,
    )  # type: ignore[attr-defined]
    normalize_graph_module(graph_module)
    if hasattr(graph_module, "_dynamo_bytecode_flatten"):
        raise AssertionError(
            "graph_module already has _dynamo_bytecode_flatten attribute"
        )
    if hasattr(graph_module, "_dynamo_bytecode_unflatten"):
        raise AssertionError(
            "graph_module already has _dynamo_bytecode_unflatten attribute"
        )
    # pyrefly: ignore [bad-argument-type]
    graph_module._dynamo_bytecode_flatten = dynamo_bytecode_flatten
    # pyrefly: ignore [bad-argument-type]
    graph_module._dynamo_bytecode_unflatten = dynamo_bytecode_unflatten
    delattr(graph_module, "_param_name_to_source")
    graph_module.recompile()
    graph_module.meta["module_call_specs"] = (
        out.graph_capture_output.output_graph.export_metadata.module_call_spec
    )
    if out.backend_input is None:
        raise AssertionError("out.backend_input must not be None for fake_mode")
    graph_module.meta["fake_mode"] = out.backend_input.fake_mode  # type: ignore[attr-defined]
    graph_module.meta["fake_mode"].allow_non_fake_inputs = True
    tracing_context = TracingContext(graph_module.meta["fake_mode"])
    tracing_context.tensor_to_context = out.backend_input.tensor_to_context  # type: ignore[attr-defined]
    graph_module.meta["tracing_context"] = tracing_context
    return graph_module


class _DynamoBytecodeCodeGen(torch.fx.graph.CodeGen):
    # [CLASS DOC] HELPER: FX code generator for graphs whose generated forward takes the user's
    # original arguments, flattens them by replaying Dynamo bytecode, and unflattens the
    # outputs the same way. State: orig_arg_names, the flatten and unflatten callables,
    # wrap_tuple flag (default False) and _inputs (inputs of the in-progress call).
    def __init__(
        self,
        orig_arg_names: list[str],
        dynamo_bytecode_flatten: DynamoBytecodeFlatten,
        dynamo_bytecode_unflatten: DynamoBytecodeUnflatten,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the argument names and the two callables; wrap_tuple is
        # False and no inputs are pending.
        super().__init__()
        self.orig_arg_names = orig_arg_names
        self.dynamo_bytecode_flatten = dynamo_bytecode_flatten
        self.dynamo_bytecode_unflatten = dynamo_bytecode_unflatten
        self.wrap_tuple = False
        self._inputs: tuple[Any, ...] | None = None

    def process_inputs(self, *inputs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Remembers the inputs for the later output step, flattens them via
        # the bytecode flatten callable, and returns the flat graph inputs. If a fake mode is
        # active and any resulting tensor is not fake, enables non-fake inputs on that mode
        # (mutation) so recovered module-attribute tensors can be used.
        self._inputs = inputs
        results = self.dynamo_bytecode_flatten(*inputs)
        fake_mode = detect_fake_mode()
        if fake_mode is not None and pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and not fake_mode.is_our_fake(x),
            results,
        ):
            # Bytecode replay can recover tensors captured from module
            # attributes as extra FX placeholder values. These tensors are not
            # visible in AOTAutograd's user-facing call signature, so allow the
            # active FakeTensorMode to convert them at dispatch time. This must
            # remain enabled for AOT's backward trace too, since autograd can
            # save the recovered tensors from the forward.
            fake_mode.allow_non_fake_inputs = True
        return results

    def process_outputs(self, outputs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Unflattens the graph outputs using the inputs saved by the
        # preceding input step, wraps the result in a one-element tuple if wrap_tuple is set,
        # clears the saved inputs and returns the result. Raises AssertionError if called before
        # inputs were processed.
        if self._inputs is None:
            raise AssertionError("process_outputs called before process_inputs")
        results = self.dynamo_bytecode_unflatten(outputs, self._inputs)
        if self.wrap_tuple:
            results = (results,)
        self._inputs = None
        return results

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        # [FUNCTION DOC] HELPER: Generates the source text of the forward definition using the
        # original argument names. If the first name is self it is treated as the receiver and
        # the free variable list gets self inserted first (mutating free_vars); any other
        # argument named self is renamed to a unique name to avoid duplicate parameters. Calls
        # the base generator with the adjusted names and, if free variables exist, appends the
        # variable-binding text. Returns the definition text.
        fn_args = self.orig_arg_names
        has_orig_self = (fn_args[0] == "self") if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, "self")
        # Rename any non-first `self` in fn_args to a unique name. The base
        # CodeGen.gen_fn_def prepends `"self"` for the GraphModule's bound-
        # method receiver whenever fn_args[0] != "self", which collides with
        # a schema param literally named `self` (e.g. `aten.where.self(Tensor
        # cond, Tensor self, Tensor other)` -> `def forward(self, cond, self,
        # other)` -> `SyntaxError: duplicate argument 'self' in function
        # definition`). Both the function-def arg list (via super().gen_fn_def)
        # and the body binding (via gen_var_bindings) reference fn_args, so
        # the rename is consistent end-to-end.
        fn_args = list(fn_args)
        first_pos = 1 if has_orig_self else 0
        for i in range(first_pos, len(fn_args)):
            if fn_args[i] == "self":
                new_name = "self_"
                while new_name in fn_args:
                    new_name += "_"
                fn_args[i] = new_name
        fn_definition = super().gen_fn_def(
            fn_args[:], maybe_return_annotation, expanded_def=expanded_def
        )

        if len(free_vars) > 0:  # pytree has placeholders in it
            fn_definition += self.gen_var_bindings(fn_args, free_vars, expanded_def)
        return fn_definition

    def gen_var_bindings(
        self, fn_args: list[str], free_vars: list[str], expanded_def: bool
    ) -> str:
        # [FUNCTION DOC] HELPER: Returns source text that packs the function arguments into a
        # tuple and unpacks the result of calling the module's flatten attribute on it into the
        # free variables (annotations and comments stripped from their names).
        without_annotation =[x.split(":")[0].split("#")[0] for x in free_vars]
        if len(fn_args) == 0:
            fn_signature = ""
        elif len(fn_args) == 1:
            fn_signature = f"{fn_args[0]}, "
        else:
            fn_signature = f"{', '.join(fn_args)}"
        return f"""
    _fn_args = ({fn_signature})
    {", ".join(without_annotation)}, = self._dynamo_bytecode_flatten(*_fn_args)"""

    def generate_output(
        self,
        output_args: torch.fx.node.Argument,
        *,
        descs: object | None = None,
        repr_fn: Any | None = None,
    ) -> str:
        # [FUNCTION DOC] HELPER: Returns the source text of the return statement: it calls the
        # module's unflatten attribute with the tuple of output values (rendered with repr_fn,
        # default repr) and the saved argument tuple, optionally wrapped in a one-element tuple
        # when wrap_tuple is set. The descs argument is unused.
        if repr_fn is None:
            repr_fn = repr
        # pyrefly: ignore [not-iterable]
        returned = f"self._dynamo_bytecode_unflatten(({', '.join([repr_fn(a) for a in output_args])},), _fn_args)"
        if self.wrap_tuple:
            returned = f"({returned},)"
        return f"return {returned}"


def dynamo_graph_capture_for_export(
    fn: Callable[..., Any],
    constraints: list[Constraint] | None = None,
) -> Callable[..., Any]:
    # [FUNCTION DOC] Returns a callable that, when invoked with example args and kwargs, captures
    # fn with full-graph Dynamo capture and returns an export-style graph module (see
    # create_fx_graph_from_captured_output). If fn is an operator overload, it is first wrapped
    # in a generated Python function with the operator's argument names and defaults
    # (executed from generated source). The returned callable runs under the compiling-state
    # context with side-effect replay disabled (warn policy), metrics and timing contexts, and
    # raises AssertionError if the config to install free tensors is enabled. The constraints
    # list is passed through to capture.
    if isinstance(fn, torch._ops.OpOverload):

        def default_annotation(arg: torch.Argument) -> str:
            # [FUNCTION DOC] HELPER: Returns the default-value suffix for an argument in
            # generated source text (equals sign plus repr of the default), or empty text.
            if arg.has_default_value():
                return f"={arg.default_value!r}"
            return ""

        has_kwarg_only = False
        arg_list = []
        for arg in fn._schema.arguments:
            if arg.kwarg_only and not has_kwarg_only:
                has_kwarg_only = True
                arg_list.append("*")
            arg_list.append(arg.name + default_annotation(arg))
        func_str = f"""
def op_overload_wrapper({", ".join(arg_list)}):
    return op({", ".join([f"{arg.name}={arg.name}" for arg in fn._schema.arguments])})
"""
        out = {}
        exec(func_str, {"op": fn}, out)
        fn = out["op_overload_wrapper"]  # type: ignore[assignment]

    def inner(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Captures fn on the given example inputs and returns the export
        # graph module. Raises AssertionError if install-free-tensors config is on.
        if torch._dynamo.config.install_free_tensors:
            raise AssertionError("install_free_tensors must be False")
        with (
            _compiling_state_context(),
            torch._dynamo.config.patch(
                replay_side_effects=False, side_effect_replay_policy="warn"
            ),
            get_metrics_context(),
            dynamo_timed("fullgraph_capture"),
        ):
            out = fullgraph_capture(
                fn,
                args,
                kwargs,
                constraints=constraints,
            )
        graph_module = create_fx_graph_from_captured_output(out, fn, args, kwargs)
        return graph_module

    return inner


def _dynamo_graph_capture_for_export(
    mod: Callable[..., Any],
    *,
    constraints: list[Constraint] | None = None,
    dynamic_shapes: _DynamicShapesInput = None,
) -> Callable[..., torch.fx.GraphModule]:
    # [FUNCTION DOC] Returns a callable that captures mod (module or function) on example args
    # and kwargs into an export-ready graph module with one placeholder per flattened input,
    # pytree-based input/output codegen and cleaned export-root names. Optional constraints
    # and dynamic_shapes control dynamic dimensions (spec objects use the newer spec API).
    # See the docstring for the overall steps. The returned callable resets Dynamo state,
    # patches Dynamo config (specialization, scalar/dynamic-output capture, free tensors) and
    # may raise a constraint violation error.
    """
    Improved dynamo graph capture using transformer approach with proper fake tensor handling.

    This function creates a capture instance that handles:
    1. PyTree flattening/unflattening with proper input ordering
    2. Dynamo graph capture with export-specific context
    3. FX graph transformation for export compatibility
    4. Proper fake tensor metadata preservation
    5. Dynamic dimension constraint handling

    Notable improvements over manual approach:
    - Uses FX Transformer for cleaner graph manipulation
    - Properly handles fake tensor metadata and dynamic dimensions
    - Preserves all necessary metadata for export
    - More robust error handling and edge case management

    TODO:
    1. Are we actually gonna run the bytecode?
    2. Need to attach guards
    """

    _constraints = constraints

    def inner(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Performs the capture for the given example inputs: flattens them
        # (honoring a user shapes spec), wraps mod in the flat-input module, resets Dynamo,
        # runs full-graph capture under the patched config, checks constraint violations
        # (unless a shapes spec is used), derives dynamic dims per input and the graph input
        # order from export metadata, rebuilds the graph with the transformer, installs pytree
        # codegen, recompiles, cleans export-root artifacts and stores module call specs and
        # fake mode in metadata. Returns the graph module. Side effects: resets global Dynamo
        # state and temporarily patches config.
        # This sets the is_exporting flag when building guards.
        with _compiling_state_context():
            # Only a ShapesSpec/ParamsSpec object opts into the new spec API.
            # A bare dict/tuple in export's `dynamic_shapes` is now bound to
            # the Dim export API.
            # TODO: inspect dictionary leaves and enable dict for ParamsSpec
            # here too, like make_fx and compile.
            user_spec: ShapesSpec | None = None
            if isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec)):
                user_spec = _coerce_to_shapes_spec(dynamic_shapes)
            flattened_spec: ShapesSpec | None = None
            if user_spec is not None:
                # _bind_spec_to_args also flattens (args, kwargs) for us.
                leaf_specs, flat_inputs, in_spec = _bind_spec_to_args(
                    mod, args, kwargs, user_spec
                )
                flattened_spec = ShapesSpec(
                    # leaf_specs is list[LeafSpec]; cast around list invariance
                    # (ParamsSpec wants list[IntermediateSpec]).
                    ParamsSpec({"*args": cast(list[IntermediateSpec], leaf_specs)}),
                    assumptions=user_spec._assumptions or None,
                )
            else:
                flat_inputs, in_spec = pytree.tree_flatten((args, kwargs))
            check_user_input_output(flat_inputs, UserErrorType.INVALID_INPUT)
            module_to_trace = ModuleToTrace(mod, in_spec)
            orig_callable = mod.forward if isinstance(mod, torch.nn.Module) else mod

            constraints: list[Constraint] | None = _constraints

            from . import reset  # type: ignore[attr-defined]

            reset()

            dynamo_config_ctx = torch._dynamo.config.patch(
                specialize_int=True,
                specialize_float=True,
                assume_static_by_default=True,
                automatic_dynamic_shapes=False,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
                constant_fold_autograd_profiler_enabled=True,
                log_graph_in_out_metadata=True,
                # install_free_tensors ensures that params and buffers are still
                # added as graph attributes, and makes Dynamo emits graphs that
                # follow export pytree-able input requirements In future, if we
                # fully rely on bytecode for the runtime, we can turn this flag
                # off.
                install_free_tensors=torch._dynamo.config.install_free_tensors_for_export,
            )

            # If a user spec was supplied, re-key it to the intermediate
            # ModuleToTrace.forward(*flat_args) layout and expose via
            # ``torch._dynamo.config._dynamic_shapes_spec`` for the variable
            # builder.
            shapes_spec_in_use = user_spec is not None
            shapes_spec_ctx = nullcontext()
            if flattened_spec is not None:
                shapes_spec_ctx = torch._dynamo.config.patch(
                    _dynamic_shapes_spec=flattened_spec
                )

            with (
                get_metrics_context(),
                dynamo_timed("fullgraph_capture"),
                dynamo_config_ctx,
                shapes_spec_ctx,
            ):
                out = fullgraph_capture(
                    module_to_trace,
                    tuple(flat_inputs),
                    constraints=_constraints,
                    _is_export_deprecated_do_not_use=True,
                )

                if out.graph_capture_output.output_graph is None:
                    raise AssertionError(
                        "output_graph must not be None after fullgraph_capture"
                    )

                example_inputs: list[Any] = []
                if out.backend_input is not None:
                    graph = out.backend_input.graph_module
                    fake_mode = out.backend_input.fake_mode
                    example_inputs = out.backend_input.example_inputs
                else:
                    graph = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
                    graph.graph.output(None)
                    graph.recompile()
                    fake_mode = None

                # ShapesSpec is unbacked based, constraint violations are not
                # relevant.
                if not shapes_spec_in_use:
                    _suggest_or_raise_constraint_violation(
                        module_to_trace,
                        orig_callable,
                        fake_mode,
                        out,
                        args,
                        kwargs,
                        dynamic_shapes,  # type: ignore[arg-type]
                    )

                # Extract export metadata from the new location
                export_metadata = out.graph_capture_output.output_graph.export_metadata
                graph_inputs = export_metadata.graph_input_idx_to_local_source
                graph_output_map = export_metadata.output_return_type
                out_spec = export_metadata.out_spec
                module_call_spec = export_metadata.module_call_spec

            # Compute dynamic dimensions for each input based on constraints
            flat_args_dynamic_dims = [
                {
                    c.dim
                    for c in (constraints or ())
                    if (
                        c.t_id == id(x)
                        and not isinstance(c, _RelaxedConstraint)
                        and c.constraint_range.vr.lower != c.constraint_range.vr.upper
                    )
                }
                for x in flat_inputs
            ]

            # Create input order mapping from dynamo's internal order to user order
            # Only process inputs that come from function arguments (GetItemSource).
            # Skip inputs that come from other sources like closures (e.g., captured
            # opaque objects like DeviceMesh).
            graph_input_order: dict[int, int] = {}
            for inp in graph_inputs:
                source = graph_inputs[inp]
                if isinstance(source, torch._dynamo.source.GetItemSource):
                    graph_input_order[source.index] = len(graph_input_order)

            for real_idx, graph_idx in graph_input_order.items():
                flat_inputs[real_idx] = example_inputs[graph_idx]

            # Use FX transformer to rebuild the graph cleanly
            transformed_graph = DynamoGraphTransformer(
                graph,
                flat_inputs,
                flat_args_dynamic_dims,
                graph_input_order,
                graph_output_map,
                fake_mode,
                graph_inputs,
            ).transform()

            # Set up PyTree codegen for proper input/output handling
            transformed_graph.graph._codegen = _PyTreeCodeGen(
                _PyTreeInfo(
                    argument_names(inspect.signature(orig_callable), args, kwargs),  # type: ignore[attr-defined, arg-type]
                    in_spec,
                    out_spec,
                )
            )
            transformed_graph.recompile()

            clean_nn_module_stack_and_source_fn(transformed_graph, True)
            clean_export_root(transformed_graph)

            transformed_graph.meta["module_call_specs"] = module_call_spec
            transformed_graph.meta["fake_mode"] = fake_mode

            return transformed_graph

    return inner
