# [MODULE DOC] Visualization of FX graphs as graphviz dot graphs through the optional pydot package.
# Defines color tables and the FxGraphDrawer class. Import-time side effect: tries to import pydot and
# defines the real drawer when it is available, otherwise defines a stub class whose constructor raises
# a runtime error asking to install pydot. Exports FxGraphDrawer.
import hashlib
from itertools import chain
from types import ModuleType
from typing import Any, TYPE_CHECKING

import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import _parse_stack_trace
from torch.fx.node import _format_arg, _get_qualified_name
from torch.fx.operator_schemas import normalize_function
from torch.fx.passes.shape_prop import TensorMetadata


if TYPE_CHECKING:
    import pydot

    HAS_PYDOT = True
else:
    pydot: ModuleType | None
    try:
        import pydot

        HAS_PYDOT = True
    except ModuleNotFoundError:
        HAS_PYDOT = False
        pydot = None


__all__ = ["FxGraphDrawer"]

_COLOR_MAP = {
    "placeholder": '"AliceBlue"',
    "call_module": "LemonChiffon1",
    "get_param": "Yellow2",
    "get_attr": "LightGrey",
    "output": "PowderBlue",
}

_HASH_COLOR_MAP = [
    "CadetBlue1",
    "Coral",
    "DarkOliveGreen1",
    "DarkSeaGreen1",
    "GhostWhite",
    "Khaki1",
    "LavenderBlush1",
    "LightSkyBlue",
    "MistyRose1",
    "MistyRose2",
    "PaleTurquoise2",
    "PeachPuff1",
    "Salmon",
    "Thistle1",
    "Thistle3",
    "Wheat1",
]

_WEIGHT_TEMPLATE = {
    "fillcolor": "Salmon",
    "style": '"filled,rounded"',
    "fontcolor": "#000000",
}

if HAS_PYDOT:

    @compatibility(is_backward_compatible=False)
    class FxGraphDrawer:
        # [CLASS DOC] Builds and stores dot graphs for a graph module and for each of its call_module
        # children that is itself a GraphModule. State: _name (main graph name), dot_graph_shape (node
        # shape, default record), normalize_args, and _dot_graphs (mapping of graph name to dot graph,
        # where child graphs are keyed by main name, underscore and the call target). Query methods return
        # the stored dot graphs. Note that construction mutates the module-level weight template shape.
        """
        Visualize a torch.fx.Graph with graphviz
        Basic usage:
            g = FxGraphDrawer(symbolic_traced, "resnet18")
            g.get_dot_graph().write_svg("a.svg")
        """

        def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool = False,
            ignore_parameters_and_buffers: bool = False,
            skip_node_names_in_args: bool = True,
            parse_stack_trace: bool = False,
            dot_graph_shape: str | None = None,
            normalize_args: bool = False,
        ) -> None:
            # [FUNCTION DOC] Builds the dot graph of the module under the given name and one more for
            # every call_module node whose target is a GraphModule (named by name plus target). Options
            # control skipping get_attr nodes, parameter and buffer nodes, argument name stripping,
            # stack trace display and argument normalization. Side effect: sets the shape entry of the
            # shared module-level weight node template, affecting all drawers. Raises RuntimeError if a
            # call_module target cannot be resolved.
            self._name = name
            self.dot_graph_shape = (
                dot_graph_shape if dot_graph_shape is not None else "record"
            )
            self.normalize_args = normalize_args
            _WEIGHT_TEMPLATE["shape"] = self.dot_graph_shape

            self._dot_graphs = {
                name: self._to_dot(
                    graph_module,
                    name,
                    ignore_getattr,
                    ignore_parameters_and_buffers,
                    skip_node_names_in_args,
                    parse_stack_trace,
                )
            }

            for node in graph_module.graph.nodes:
                if node.op != "call_module":
                    continue

                leaf_node = self._get_leaf_node(graph_module, node)

                if not isinstance(leaf_node, torch.fx.GraphModule):
                    continue

                self._dot_graphs[f"{name}_{node.target}"] = self._to_dot(
                    leaf_node,
                    f"{name}_{node.target}",
                    ignore_getattr,
                    ignore_parameters_and_buffers,
                    skip_node_names_in_args,
                    parse_stack_trace,
                )

        def get_dot_graph(self, submod_name: str | None = None) -> pydot.Dot:
            # [FUNCTION DOC] Returns the main dot graph when no submodule name is given, otherwise the
            # dot graph of the named submodule. Raises a missing key error for unknown names.
            """
            Visualize a torch.fx.Graph with graphviz
            Example:
                >>> # xdoctest: +REQUIRES(module:pydot)
                >>> # xdoctest: +REQUIRES(module:ubelt)
                >>> # define module
                >>> class MyModule(torch.nn.Module):
                >>>     def __init__(self) -> None:
                >>>         super().__init__()
                >>>         self.linear = torch.nn.Linear(4, 5)
                >>>     def forward(self, x):
                >>>         return self.linear(x).clamp(min=0.0, max=1.0)
                >>> module = MyModule()
                >>> # trace the module
                >>> symbolic_traced = torch.fx.symbolic_trace(module)
                >>> # setup output file
                >>> import ubelt as ub
                >>> dpath = ub.Path.appdir("torch/tests/FxGraphDrawer").ensuredir()
                >>> fpath = dpath / "linear.svg"
                >>> # draw the graph
                >>> g = FxGraphDrawer(symbolic_traced, "linear")
                >>> g.get_dot_graph().write_svg(fpath)
            """
            if submod_name is None:
                return self.get_main_dot_graph()
            else:
                return self.get_submod_dot_graph(submod_name)

        def get_main_dot_graph(self) -> pydot.Dot:
            # [FUNCTION DOC] Returns the dot graph built for the top-level module.
            return self._dot_graphs[self._name]

        def get_submod_dot_graph(self, submod_name: str) -> pydot.Dot:
            # [FUNCTION DOC] Returns the dot graph of the submodule with the given call target; raises a
            # missing key error if there is none.
            return self._dot_graphs[f"{self._name}_{submod_name}"]

        def get_all_dot_graphs(self) -> dict[str, pydot.Dot]:
            # [FUNCTION DOC] Returns the internal mapping of graph name to dot graph (not a copy).
            return self._dot_graphs

        def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]:
            # [FUNCTION DOC] HELPER: Returns the dot style attributes for a node (shape, fill color,
            # style, font color). The fill color comes from the op table when the op is listed, otherwise
            # from a stable hash of the printed target name indexing the color palette.
            template = {
                "shape": self.dot_graph_shape,
                "fillcolor": "#CAFFE3",
                "style": '"filled,rounded"',
                "fontcolor": "#000000",
            }
            if node.op in _COLOR_MAP:
                template["fillcolor"] = _COLOR_MAP[node.op]
            else:
                # Use a random color for each node; based on its name so it's stable.
                target_name = node._pretty_print_target(node.target)
                target_hash = int(
                    hashlib.md5(
                        target_name.encode(), usedforsecurity=False
                    ).hexdigest()[:8],
                    16,
                )
                template["fillcolor"] = _HASH_COLOR_MAP[
                    target_hash % len(_HASH_COLOR_MAP)
                ]
            return template

        def _get_leaf_node(
            self, module: torch.nn.Module, node: torch.fx.Node
        ) -> torch.nn.Module:
            # [FUNCTION DOC] HELPER: Resolves a call_module node's dotted string target by following
            # attributes from the given module and returns the resulting object. Raises AssertionError
            # if the target is not a string and RuntimeError if an attribute is missing.
            py_obj = module
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str target, got {type(node.target)}")
            atoms = node.target.split(".")
            for atom in atoms:
                if not hasattr(py_obj, atom):
                    raise RuntimeError(
                        str(py_obj) + " does not have attribute " + atom + "!"
                    )
                py_obj = getattr(py_obj, atom)
            return py_obj

        def _typename(self, target: torch.fx.node.Target | torch.nn.Module) -> str:
            # [FUNCTION DOC] HELPER: Returns a display name: the type name for modules, the string itself
            # for strings, else the qualified function name. Curly braces are escaped for dot labels.
            if isinstance(target, torch.nn.Module):
                ret = torch.typename(target)
            elif isinstance(target, str):
                ret = target
            else:
                ret = _get_qualified_name(target)

            # Escape "{" and "}" to prevent dot files like:
            # https://gist.github.com/SungMinCho/1a017aab662c75d805c5954d62c5aabc
            # which triggers `Error: bad label format (...)` from dot
            return ret.replace("{", r"\{").replace("}", r"\}")

        # shorten path to avoid drawing long boxes
        # for full path = '/home/weif/pytorch/test.py'
        # return short path = 'pytorch/test.py'
        def _shorten_file_name(
            self,
            full_file_name: str,
            truncate_to_last_n: int = 2,
        ) -> str:
            # [FUNCTION DOC] HELPER: Keeps only the last N slash-separated components of a path (default
            # two); shorter paths are returned unchanged.
            splits = full_file_name.split("/")
            if len(splits) >= truncate_to_last_n:
                return "/".join(splits[-truncate_to_last_n:])
            return full_file_name

        def _get_node_label(
            self,
            module: torch.fx.GraphModule,
            node: torch.fx.Node,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
        ) -> str:
            # [FUNCTION DOC] HELPER: Builds the record style label of a node: name and op code; for
            # module calls the module type and its constants; for other nodes the target, formatted args
            # and kwargs (optionally normalized, falling back on failure) and user count. Then appends
            # tensor metadata, buffer metadata (name and origin count) and, if requested and available,
            # the shortened file, line and code from the stack trace. Raises RuntimeError for unsupported
            # tensor metadata types or an unresolvable module target.
            def _get_str_for_args_kwargs(arg: tuple[Any, ...] | dict[str, Any]) -> str:
                # [FUNCTION DOC] HELPER: Formats a tuple of args or a mapping of kwargs as a multi-line
                # label fragment (truncating long lists), optionally dropping entries that mention node
                # names, escaping braces; single entries are put on one line. Returns an empty string
                # for other types or when nothing remains.
                if isinstance(arg, tuple):
                    prefix, suffix = r"|args=(\l", r",\n)\l"
                    arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
                elif isinstance(arg, dict):
                    prefix, suffix = r"|kwargs={\l", r",\n}\l"
                    arg_strs_list = [
                        f"{k}: {_format_arg(v, max_list_len=8)}" for k, v in arg.items()
                    ]
                else:  # Fall back to nothing in unexpected case.
                    return ""

                # Strip out node names if requested.
                if skip_node_names_in_args:
                    arg_strs_list = [a for a in arg_strs_list if "%" not in a]
                if len(arg_strs_list) == 0:
                    return ""
                arg_strs = prefix + r",\n".join(arg_strs_list) + suffix
                if len(arg_strs_list) == 1:
                    arg_strs = arg_strs.replace(r"\l", "").replace(r"\n", "")
                return arg_strs.replace("{", r"\{").replace("}", r"\}")

            label = "{" + f"name=%{node.name}|op_code={node.op}\n"

            if node.op == "call_module":
                leaf_module = self._get_leaf_node(module, node)
                label += r"\n" + self._typename(leaf_module) + r"\n|"
                extra = ""
                if hasattr(leaf_module, "__constants__"):
                    extra = r"\n".join(
                        [
                            f"{c}: {getattr(leaf_module, c)}"
                            for c in leaf_module.__constants__  # type: ignore[union-attr]
                        ]  # type: ignore[union-attr]
                    )
                label += extra + r"\n"
            else:
                label += f"|target={self._typename(node.target)}" + r"\n"
                if self.normalize_args:
                    try:
                        args, kwargs = normalize_function(  # type: ignore[misc]
                            node.target,  # type: ignore[arg-type]
                            node.args,  # type: ignore[arg-type]
                            node.kwargs,
                            normalize_to_only_use_kwargs=True,
                        )
                    except Exception:
                        # Fallback to not normalizing if there's an exception.
                        # Some functions need overloads specified to normalize.
                        args, kwargs = node.args, node.kwargs
                else:
                    args, kwargs = node.args, node.kwargs
                if len(args) > 0:
                    label += _get_str_for_args_kwargs(args)
                if len(kwargs) > 0:
                    label += _get_str_for_args_kwargs(kwargs)
                label += f"|num_users={len(node.users)}" + r"\n"

            tensor_meta = node.meta.get("tensor_meta")
            label += self._tensor_meta_to_label(tensor_meta)

            # for original fx graph
            # print buf=buf0, n_origin=6
            buf_meta = node.meta.get("buf_meta", None)
            if buf_meta is not None:
                label += f"|buf={buf_meta.name}" + r"\n"
                label += f"|n_origin={buf_meta.n_origin}" + r"\n"

            # for original fx graph
            # print file:lineno code
            if parse_stack_trace and node.stack_trace is not None:
                parsed_stack_trace = _parse_stack_trace(node.stack_trace)
                if parsed_stack_trace is not None:
                    fname = self._shorten_file_name(parsed_stack_trace.file)
                    label += (
                        f"|file={fname}:{parsed_stack_trace.lineno} {parsed_stack_trace.code}"
                        + r"\n"
                    )

            return label + "}"

        def _tensor_meta_to_label(self, tm: object) -> str:
            # [FUNCTION DOC] HELPER: Converts tensor metadata to label text: empty for none, formatted
            # fields for a single metadata record, and the concatenation over elements for lists,
            # tuples and mapping values (recursively). Raises RuntimeError for any other type.
            if tm is None:
                return ""
            elif isinstance(tm, TensorMetadata):
                return self._stringify_tensor_meta(tm)
            elif isinstance(tm, list):
                result = ""
                for item in tm:
                    result += self._tensor_meta_to_label(item)
                return result
            elif isinstance(tm, dict):
                result = ""
                for v in tm.values():
                    result += self._tensor_meta_to_label(v)
                return result
            elif isinstance(tm, tuple):
                result = ""
                for item in tm:
                    result += self._tensor_meta_to_label(item)
                return result
            else:
                raise RuntimeError(f"Unsupported tensor meta type {type(tm)}")

        def _stringify_tensor_meta(self, tm: TensorMetadata) -> str:
            # [FUNCTION DOC] HELPER: Renders dtype, shape, requires_grad and stride as label lines, plus
            # quantization parameters (per-tensor scale and zero point, or per-channel scale, zero point
            # and axis) and the scheme when the tensor is quantized. Raises AssertionError if quantization
            # parameters or the scheme are missing and RuntimeError for an unsupported scheme.
            # Side effect: prints the record to standard output when it lacks a dtype attribute.
            # KNOWN ISSUE: the debug print is left in; a record without dtype then fails on the next line.
            result = ""
            if not hasattr(tm, "dtype"):
                print("tm", tm)
            result += "|" + "dtype" + "=" + str(tm.dtype) + r"\n"
            result += "|" + "shape" + "=" + str(tuple(tm.shape)) + r"\n"
            result += "|" + "requires_grad" + "=" + str(tm.requires_grad) + r"\n"
            result += "|" + "stride" + "=" + str(tm.stride) + r"\n"
            if tm.is_quantized:
                if tm.qparams is None:
                    raise AssertionError("qparams is None for quantized tensor")
                if "qscheme" not in tm.qparams:
                    raise AssertionError("qscheme not in qparams")
                qscheme = tm.qparams["qscheme"]
                if qscheme in {
                    torch.per_tensor_affine,
                    torch.per_tensor_symmetric,
                }:
                    result += "|" + "q_scale" + "=" + str(tm.qparams["scale"]) + r"\n"
                    result += (
                        "|"
                        + "q_zero_point"
                        + "="
                        + str(tm.qparams["zero_point"])
                        + r"\n"
                    )
                elif qscheme in {
                    torch.per_channel_affine,
                    torch.per_channel_symmetric,
                    torch.per_channel_affine_float_qparams,
                }:
                    result += (
                        "|"
                        + "q_per_channel_scale"
                        + "="
                        + str(tm.qparams["scale"])
                        + r"\n"
                    )
                    result += (
                        "|"
                        + "q_per_channel_zero_point"
                        + "="
                        + str(tm.qparams["zero_point"])
                        + r"\n"
                    )
                    result += (
                        "|"
                        + "q_per_channel_axis"
                        + "="
                        + str(tm.qparams["axis"])
                        + r"\n"
                    )
                else:
                    raise RuntimeError(f"Unsupported qscheme: {qscheme}")
                result += "|" + "qscheme" + "=" + str(tm.qparams["qscheme"]) + r"\n"
            return result

        def _get_tensor_label(self, t: torch.Tensor) -> str:
            # [FUNCTION DOC] HELPER: Returns dtype followed by the list of dimensions of the tensor,
            # ending with a literal line break escape for dot labels.
            return str(t.dtype) + str(list(t.shape)) + r"\n"

        # when parse_stack_trace=True
        # print file:lineno code
        def _to_dot(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool,
            ignore_parameters_and_buffers: bool,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
        ) -> pydot.Dot:
            # [FUNCTION DOC] HELPER: Creates a top-to-bottom dot graph with one node per FX node (skipping
            # get_attr nodes if requested), styled and labeled; nodes whose buffer metadata has several
            # origins are grouped into colored clusters per buffer. For module calls to non-GraphModule
            # modules, adds weight nodes and edges for each parameter and buffer unless disabled. Finally
            # adds clusters and an edge from each node to each of its users. Returns the dot graph.
            # KNOWN ISSUE: in the weight node label, operator precedence makes buffers lose their name
            # prefix (only the buffer text is used), while parameters get the full prefix.
            """
            Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
            If ignore_parameters_and_buffers is True, the parameters and buffers
            created with the module will not be added as nodes and edges.
            """

            # "TB" means top-to-bottom rank direction in layout
            dot_graph = pydot.Dot(name, rankdir="TB")

            buf_name_to_subgraph: dict[str, pydot.Cluster] = {}

            for node in graph_module.graph.nodes:
                if ignore_getattr and node.op == "get_attr":
                    continue

                style = self._get_node_style(node)
                dot_node = pydot.Node(
                    node.name,
                    label=self._get_node_label(
                        graph_module, node, skip_node_names_in_args, parse_stack_trace
                    ),
                    **style,  # type: ignore[arg-type]
                )

                current_graph = dot_graph

                buf_meta = node.meta.get("buf_meta", None)
                if buf_meta is not None and buf_meta.n_origin > 1:
                    buf_name = buf_meta.name
                    if buf_name not in buf_name_to_subgraph:
                        buf_name_to_subgraph[buf_name] = pydot.Cluster(
                            buf_name, label=buf_name
                        )
                    current_graph = buf_name_to_subgraph.get(buf_name)  # type: ignore[assignment]

                # pyrefly: ignore [missing-attribute]
                current_graph.add_node(dot_node)

                def get_module_params_or_buffers() -> None:
                    # [FUNCTION DOC] HELPER: For the current leaf module, adds a weight node for every
                    # parameter and buffer to the dot graph, each with an edge to the module node.
                    # Reads the enclosing loop variables for the node and module at call time.
                    for pname, ptensor in chain(
                        leaf_module.named_parameters(),
                        # pyrefly: ignore [bad-argument-type]
                        leaf_module.named_buffers(),
                    ):
                        pname1 = node.name + "." + pname
                        label1 = (
                            pname1 + "|op_code=get_" + "parameter"
                            if isinstance(ptensor, torch.nn.Parameter)
                            else "buffer" + r"\l"
                        )
                        dot_w_node = pydot.Node(
                            pname1,
                            label="{" + label1 + self._get_tensor_label(ptensor) + "}",
                            **_WEIGHT_TEMPLATE,  # type: ignore[arg-type]
                        )
                        dot_graph.add_node(dot_w_node)
                        dot_graph.add_edge(pydot.Edge(pname1, node.name))

                if node.op == "call_module":
                    leaf_module = self._get_leaf_node(graph_module, node)

                    if not ignore_parameters_and_buffers and not isinstance(
                        leaf_module, torch.fx.GraphModule
                    ):
                        get_module_params_or_buffers()

            for subgraph in buf_name_to_subgraph.values():
                subgraph.set("color", "royalblue")
                subgraph.set("penwidth", "2")
                dot_graph.add_subgraph(subgraph)  # type: ignore[arg-type]

            for node in graph_module.graph.nodes:
                if ignore_getattr and node.op == "get_attr":
                    continue

                for user in node.users:
                    dot_graph.add_edge(pydot.Edge(node.name, user.name))

            return dot_graph

else:
    if not TYPE_CHECKING:

        @compatibility(is_backward_compatible=False)
        class FxGraphDrawer:
            # [CLASS DOC] Fallback stub used when pydot is not installed; it has no state and cannot be
            # instantiated successfully.
            def __init__(
                self,
                graph_module: torch.fx.GraphModule,
                name: str,
                ignore_getattr: bool = False,
                ignore_parameters_and_buffers: bool = False,
                skip_node_names_in_args: bool = True,
                parse_stack_trace: bool = False,
                dot_graph_shape: str | None = None,
                normalize_args: bool = False,
            ):
                # [FUNCTION DOC] Always raises a runtime error telling the user to install pydot;
                # all arguments are ignored.
                raise RuntimeError(
                    "FXGraphDrawer requires the pydot package to be installed. Please install "
                    "pydot through your favorite Python package manager."
                )
