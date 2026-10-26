# # [MODULE DOC] Applies chunking to a graph whose nodes already carry chunking metadata: moves the
# # chunked region into per-chunk-size subgraphs invoked once per chunk via the invoke_subgraph higher-order
# # op, accumulates or concatenates their outputs, and rebuilds the parent graph module.
# # Main contents: ChunkingApplier, fake_tensor_prop, is_chunking_subgraph_input. Import-time: logger only.
import copy
import logging
import operator
from typing import Any

import torch
from torch import Tensor
from torch._dynamo.utils import detect_fake_mode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from .core import get_chunking_meta, reorder_nodes
from .utils import get_args_of_node_type, get_fake_tensor_from_node_arg, is_tangent_node


log = torch._logging.getArtifactLogger(__name__, "auto_chunker")
aten = torch.ops.aten
prims = torch.ops.prims


def _factory_args(fake_tensor: Tensor) -> dict[str, Any]:
    # # [FUNCTION DOC] HELPER: Returns keyword arguments (device and dtype) for creating a tensor like the
    # # given fake tensor.
    return {
        "device": fake_tensor.device,
        "dtype": fake_tensor.dtype,
    }


def fake_tensor_prop(gm: GraphModule) -> None:
    # # [FUNCTION DOC] Re-runs fake tensor propagation over a graph module so every node gets an up-to-date
    # # value in its metadata. Uses each placeholder's fake tensor when present, else the node itself, and the
    # # fake mode detected from those inputs. Mutates node metadata.
    inputs = []
    for node in gm.graph.find_nodes(op="placeholder", sort=False):
        fake_tensor = get_fake_tensor_from_node_arg(node)
        if fake_tensor is not None:
            inputs.append(fake_tensor)
        else:
            inputs.append(node)

    fake_mode = detect_fake_mode(inputs)
    FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*inputs)


def is_chunking_subgraph_input(node: Node) -> bool:
    # # [FUNCTION DOC] Returns true if the node has chunking meta, is not a tangent, and is either a placeholder
    # # or consumes at least one node lacking chunking meta (so it enters the chunking subgraph from outside).
    meta = get_chunking_meta(node)
    if meta is None or is_tangent_node(node):
        return False
    arg_nodes = get_args_of_node_type(node)
    arg_nodes_no_meta = [node for node in arg_nodes if get_chunking_meta(node) is None]
    return len(arg_nodes_no_meta) > 0 or node.op == "placeholder"


class ChunkingApplier:
    # # [CLASS DOC] Transforms a graph module by running its chunked region as subgraphs. Constructed with a
    # # parent graph module and chunk count. State: reordered copy of the graph (parent_graph), subgraph input /
    # # body / output node lists, tangent overrides (tangent node to all-ones node), chunked input slices,
    # # per-chunk-size sizes and subgraph attribute names, accumulators for outputs that must be summed, and
    # # lists of output chunks to concatenate. Method apply orchestrates the whole transformation.
    """
    A class that chunks the graph assuming chunking metadata has already
    been attached to the nodes in the chunking subgraph.
    """

    def __init__(self, parent_gm: GraphModule, num_chunk: int):
        # # [FUNCTION DOC] Reorders the graph so chunked nodes are contiguous, categorizes subgraph nodes and
        # # initializes bookkeeping: for each subgraph output with a chunk dimension prepares a list of chunks to
        # # concatenate, otherwise reserves an accumulator slot. Raises AssertionError if an output lacks meta or
        # # the body is empty.
        self.gm = parent_gm
        self.parent_graph = reorder_nodes(self.gm.graph)
        # From this point on self.parent_graph is not equal to self.gm.graph
        # due to reordering. We create a new copy of the graph so it's
        # easier to fallback if chunking fails.
        self.num_chunk = num_chunk

        # tangent node to the all-one tensor
        self.overriden_tangent: dict[Node, Node | None] = {}

        self.subgraph_input: list[Node] = []
        self.subgraph_body: list[Node] = []
        self.subgraph_output: list[Node] = []
        self._categorize_subgraph_nodes()

        self.chunk_sizes: list[int] | None = None

        # First index is node index,
        # Second index is chunk index.
        # Node index may be different to the index for self.subgraph_input
        # since not every subgraph_input may be chunked.
        # check self.chunk_subgraph_input for more details
        self.chunked_subgraph_input: list[list[Node]] = []

        self.accumulators: dict[Node, Node | None] = {}
        self.chunks_for_recovering: dict[Node, list[Node]] = {}
        for node in self.subgraph_output:
            meta = get_chunking_meta(node)
            if not meta:
                raise AssertionError("expected chunking meta for subgraph output node")
            if meta.chunk_dim is not None:
                self.chunks_for_recovering[node] = []
            else:
                # the accumulator nodes is created later
                self.accumulators[node] = None

        self.chunk_size_to_gm_attr: dict[int, str] = {}

    def _categorize_subgraph_nodes(self) -> None:
        # # [FUNCTION DOC] HELPER: Walks parent graph nodes with chunking meta. Tangents are recorded as overridden
        # # (value filled later). Other nodes are classified as input (enters from unchunked nodes), output (has users
        # # without meta) or body. Raises AssertionError if the body ends up empty.
        """
        For each chunked node, decide if it's an input/body/output node of the
        chunking subgraph.
        """
        for node in self.parent_graph.nodes:
            meta = get_chunking_meta(node)

            # The node does not have chunking metadata, skip
            if meta is None:
                continue

            # skip tangent nodes since they are overridden as 1 and
            # reapplied later in the bwd graph
            if is_tangent_node(node):
                self.overriden_tangent[node] = None  # will update the value later
                continue

            # To check if a node is a placeholder for the chunking
            # subgraph there are 2 alternatives
            # 1. decide it's a placeholder if it's using any non-chunked node as argument
            # 2. decide it's a placeholder if it's not using any chunked nodes as argument
            # These 2 rules are almost equivalent since we don't mix
            # chunked and non-chunked arguments for a fx.Node.
            # But one subtle difference is for aten.full node.
            # We don't need to pass in aten.full as a placeholder since
            # we can just add the chunked aten.full in the subgraph
            # directly.
            # Alternative 2 does not work for this case, so we implement
            # alternative 1.

            user_nodes = node.users
            user_nodes_no_meta = [
                node for node in user_nodes if get_chunking_meta(node) is None
            ]

            if is_chunking_subgraph_input(node):
                # None of the node's arguments are chunked. It's a placeholder
                self.subgraph_input.append(node)
            elif len(user_nodes_no_meta) > 0:
                self.subgraph_output.append(node)
            else:
                self.subgraph_body.append(node)

        if len(self.subgraph_body) <= 0:
            raise AssertionError("expected non-empty subgraph body")

    def replace_tangent_to_one(self) -> None:
        # # [FUNCTION DOC] For each recorded tangent, creates a full tensor of ones with its shape, dtype and device,
        # # copies metadata, redirects all uses of the tangent to it and stores it in the override map.
        # # KNOWN ISSUE: the name string lacks an f-string prefix, so every ones node gets the literal text with
        # # braces rather than its index; the loop index is otherwise unused.
        for idx, node in enumerate(self.overriden_tangent):
            if not is_tangent_node(node):
                raise AssertionError(f"expected tangent node, got {node}")

            fake_tensor = node.meta["val"]
            one = self.parent_graph.call_function(
                aten.full.default, (fake_tensor.shape, 1), _factory_args(fake_tensor)
            )

            name = "tangent_overriden_as_one_{idx}"
            one._rename(name)
            one.meta = copy.copy(node.meta)
            node.replace_all_uses_with(one)
            self.overriden_tangent[node] = one

    def chunk_subgraph_input(self) -> None:
        # # [FUNCTION DOC] For each subgraph input with a chunk dimension, adds a chunk operation along that
        # # dimension plus one element extraction per chunk, appended to the chunked input list. Inputs without
        # # a chunk dimension are left alone. On the first chunked input computes the list of chunk sizes by
        # # chunking its fake value. AssertionError if an input lacks meta.
        # # KNOWN ISSUE: the chunk operation may yield fewer pieces than num_chunk (for small sizes), but later
        # # code indexes chunk sizes for each of num_chunk chunks.
        """
        Chunk subgraph inputs if necessary. Note that not every subgraph
        input needs to be chunked.
        E.g. the weight for matmul does not need to be chunked.
        """
        for node_idx, subgraph_input in enumerate(self.subgraph_input):
            meta = get_chunking_meta(subgraph_input)
            if meta is None:
                raise AssertionError("expected chunking meta for subgraph input")

            # not chunked
            if meta.chunk_dim is None:
                continue

            chunk_node = self.parent_graph.call_function(
                aten.chunk.default, (subgraph_input, self.num_chunk, meta.chunk_dim)
            )
            chunks = [
                self.parent_graph.call_function(operator.getitem, (chunk_node, i))
                for i in range(self.num_chunk)
            ]

            self.chunked_subgraph_input.append(chunks)

            # let's do a small meta propagation to get the size of each chunk
            if self.chunk_sizes is None:
                input_tensor = subgraph_input.meta["val"]
                tensors = torch.chunk(input_tensor, self.num_chunk)
                self.chunk_sizes = [t.size(0) for t in tensors]

    def create_accumulators(self) -> None:
        # # [FUNCTION DOC] Creates a zero-filled full tensor accumulator for each output that needs summing.
        # # Uses float32, except keeps the original dtype for single-element tensors (avoiding addmm dtype
        # # mismatch). Stores the node in the accumulators map and gives it fake-value metadata.
        for node in self.accumulators:
            # use fp32 for accumulators
            fake_tensor = node.meta["val"]
            if fake_tensor.numel() == 1:
                # TODO(shunting) revisit
                # This tensor may be fused with mm and becomes a addmm
                # if we upcast here, addmm may fail due to incompatible
                # dtypes for input arguments.
                override_dtype = torch.float32
            else:
                override_dtype = fake_tensor.dtype

            kwargs = _factory_args(fake_tensor)
            kwargs["dtype"] = override_dtype
            accum = self.parent_graph.call_function(
                aten.full.default, (fake_tensor.shape, 0), kwargs
            )
            # reuse the meta['val']
            accum.meta = {"val": fake_tensor.to(override_dtype)}
            self.accumulators[node] = accum

    def build_subgraph(self, chunk_size: int) -> GraphModule:
        # # [FUNCTION DOC] Builds a graph module for one chunk size. Creates placeholders for subgraph inputs,
        # # overridden tangents and accumulators (chunked inputs get their fake size reduced along the chunk dim).
        # # Copies body and output nodes, rewriting full, expand and view nodes with a chunk dimension to use the
        # # chunk size in their shape. Then adds each accumulator to its output, outputs all subgraph outputs as a
        # # tuple, removes dead code, lints, wraps in a graph module and runs fake tensor propagation.
        # # Raises AssertionError on unset tangents/accumulators or placeholders inside the body.
        """
        Build a subgraph for the given chunk size.
        The last chunk can be smaller and a new subgraph will be created
        to avoid involving dynamic shapes.
        """
        new_graph = Graph()
        env: dict[Node, Node] = {}

        def _create_placeholder_node(input_node: Node) -> Node:
            # # [FUNCTION DOC] HELPER: Creates a placeholder in the new graph named after the input node, with fake
            # # metadata sliced to the chunk size along the chunk dim when the node is chunked. Returns the placeholder.
            new_node = new_graph.placeholder(input_node.name)
            fake_tensor = input_node.meta["val"]
            chunking_meta = get_chunking_meta(input_node)
            if chunking_meta is not None and chunking_meta.chunk_dim is not None:
                # the node is chunked and we need update the
                # fake tensor
                # TODO any better way to do this?
                new_tensor = aten.slice.Tensor(
                    fake_tensor, chunking_meta.chunk_dim, 0, chunk_size
                )
                fake_tensor = new_tensor
            new_node.meta = {"val": fake_tensor}
            return new_node

        for node_idx, input_node in enumerate(self.subgraph_input):
            env[input_node] = _create_placeholder_node(input_node)

        for overriden_tangent_node in self.overriden_tangent.values():
            if overriden_tangent_node is None:
                raise AssertionError("expected tangent node to be set")
            env[overriden_tangent_node] = _create_placeholder_node(
                overriden_tangent_node
            )

        for accum in self.accumulators.values():
            if accum is None:
                raise AssertionError("expected accumulator to be set")
            env[accum] = _create_placeholder_node(accum)

        for original_node in self.subgraph_body + self.subgraph_output:
            if original_node.op == "placeholder":
                raise AssertionError(
                    f"unexpected placeholder node in subgraph body/output: {original_node}"
                )

            # Chunk aten.full
            if (
                original_node.target == aten.full.default
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[0])  # type: ignore[arg-type]
                # pyrefly: ignore [unsupported-operation]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.full.default,
                    (shape, original_node.args[1]),
                    original_node.kwargs,
                )
                continue
            # Chunk aten.expand: adjust the target shape at the chunk dimension
            if (
                original_node.target == aten.expand.default
                and isinstance(original_node.args[0], torch.fx.Node)
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[1])  # type: ignore[arg-type]
                # pyrefly: ignore [unsupported-operation]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.expand.default,
                    (env.get(original_node.args[0], original_node.args[0]), shape),  # type: ignore[arg-type]
                    original_node.kwargs,
                )
                continue

            # Chunk aten.view: adjust the target shape at the chunk dimension
            if (
                original_node.target == aten.view.default
                and isinstance(original_node.args[0], torch.fx.Node)
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[1])  # type: ignore[arg-type]
                # pyrefly: ignore [unsupported-operation]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.view.default,
                    (env[original_node.args[0]], shape),  # type: ignore[arg-type]
                    original_node.kwargs,
                )
                continue

            # create the node with chunked inputs
            env[original_node] = new_graph.node_copy(original_node, lambda x: env[x])

        # Do the accumulation inside this subgraph
        for node, accum in self.accumulators.items():
            lhs = env[node]
            if accum is None:
                raise AssertionError("expected accumulator to be set")
            rhs = env[accum]

            # add `addend` and `accum`
            add_out = new_graph.call_function(aten.add.Tensor, (lhs, rhs))

            # override the chunk value
            env[node] = add_out

        out_values = []
        for node in self.subgraph_output:
            out_values.append(env[node])

        new_graph.output(tuple(out_values))
        new_graph.eliminate_dead_code()
        new_graph.lint()

        sub_gm = torch.fx._lazy_graph_module._make_graph_module(self.gm, new_graph)
        fake_tensor_prop(sub_gm)
        return sub_gm

    def build_subgraphs(self) -> None:
        # # [FUNCTION DOC] Builds one subgraph per distinct chunk size, registers each on the graph module under a
        # # numbered attribute name, and marks it so the chunker never processes it again. Raises AssertionError if
        # # chunk sizes are not computed.
        if self.chunk_sizes is None:
            raise AssertionError("expected chunk_sizes to be set")
        for chunk_size in self.chunk_sizes:
            if chunk_size in self.chunk_size_to_gm_attr:
                continue

            sub_gm = self.build_subgraph(chunk_size)
            gm_attr = f"chunking_subgraph_{len(self.chunk_size_to_gm_attr)}"
            self.chunk_size_to_gm_attr[chunk_size] = gm_attr
            setattr(self.gm, gm_attr, sub_gm)

            # Mark this sub graph module so we don't recursively chunking
            # it.
            sub_gm.meta["produced_by_chunker"] = True

    def call_subgraph_for_each_chunk(self) -> None:
        # # [FUNCTION DOC] For each chunk, emits a call of the subgraph (picked by that chunk's size) with the chunked
        # # input slice or original input, the overridden tangents and the current accumulators. Extracts each
        # # output, appends it to the chunks to concatenate, and updates accumulators to the latest outputs so
        # # they thread through successive calls. Raises AssertionError on missing sizes, metas or accumulators.
        for chunk_id in range(self.num_chunk):
            if self.chunk_sizes is None:
                raise AssertionError("expected chunk_sizes to be set")
            chunk_size = self.chunk_sizes[chunk_id]
            subgraph_id = self.chunk_size_to_gm_attr[chunk_size]
            sub_gm = self.parent_graph.get_attr(subgraph_id)

            args = []
            chunks_iter = iter(self.chunked_subgraph_input)
            for node in self.subgraph_input:
                chunking_meta = get_chunking_meta(node)
                if chunking_meta is None:
                    raise AssertionError("expected chunking meta for subgraph input")
                if chunking_meta.chunk_dim is not None:
                    args.append(next(chunks_iter)[chunk_id])
                else:
                    # not chunked
                    args.append(node)

            args += list(self.overriden_tangent.values())  # type: ignore[arg-type]

            for accum in self.accumulators.values():
                if accum is None:
                    raise AssertionError("expected accumulator to be set")
                args.append(accum)

            output_node = self.parent_graph.call_function(
                torch.ops.higher_order.invoke_subgraph, (sub_gm, subgraph_id, *args), {}
            )

            output_node_dict = {}
            for i, orig_node in enumerate(self.subgraph_output):
                output_node_dict[orig_node] = self.parent_graph.call_function(
                    operator.getitem, (output_node, i)
                )

            for orig_node, node_list in self.chunks_for_recovering.items():
                chunk = output_node_dict[orig_node]
                node_list.append(chunk)

            for orig_node in self.accumulators:
                self.accumulators[orig_node] = output_node_dict[orig_node]

    def recover_to_unchunked_nodes(self) -> None:
        # # [FUNCTION DOC] For each subgraph output, builds the unchunked replacement: concatenation of chunks along
        # # the chunk dim, or the final accumulator if a sum is needed, else the node itself; then multiplies by
        # # scale_by if set and converts back to the original dtype for summed non-float32 outputs. Redirects all
        # # uses of the original node to the result. AssertionError if meta is missing or nothing was replaced.
        """
        Recover the node from chunks and do the replacement.
        """
        for node in self.subgraph_output:
            meta = get_chunking_meta(node)
            if meta is None:
                raise AssertionError("expected chunking meta for subgraph output")

            recovered: torch.fx.Node = node

            if meta.chunk_dim is not None:
                chunks = self.chunks_for_recovering[node]
                recovered = self.parent_graph.call_function(
                    aten.cat.default, (chunks, meta.chunk_dim)
                )
            elif meta.need_sum:
                recovered = self.accumulators[node]  # type: ignore[assignment]

            # do scaling last
            if meta.scale_by is not None:
                recovered = self.parent_graph.call_function(
                    aten.mul.Tensor, (recovered, meta.scale_by)
                )

            # convert back to the original dtype
            if meta.need_sum:
                original_dtype = node.meta["val"].dtype
                # TODO(shunting): do we always uses a fp32 accumulator?
                if original_dtype != torch.float32:
                    recovered = self.parent_graph.call_function(
                        prims.convert_element_type.default, (recovered, original_dtype)
                    )

            if recovered is node:
                raise AssertionError("expected recovered node to differ from original")
            node.replace_all_uses_with(recovered)

    def erase_original_nodes(self) -> None:
        # # [FUNCTION DOC] Erases the original body and output nodes from the parent graph in reverse order
        # # (users first), skipping placeholders.
        # Traverse reversely to erase user first
        for node in reversed(tuple(self.subgraph_body + self.subgraph_output)):
            if node.op == "placeholder":
                continue

            self.parent_graph.erase_node(node)

    def apply(self) -> GraphModule:
        # # [FUNCTION DOC] Runs the full transformation at the position before the first body node: override tangents,
        # # chunk inputs, create accumulators, build subgraphs, call them per chunk, recover unchunked outputs and
        # # erase originals. Then builds a new graph module from the parent graph, runs fake tensor propagation
        # # (prints readable code in debug mode) and returns it.
        with self.parent_graph.inserting_before(self.subgraph_body[0]):
            self.replace_tangent_to_one()
            self.chunk_subgraph_input()
            self.create_accumulators()
            self.build_subgraphs()
            self.call_subgraph_for_each_chunk()
            self.recover_to_unchunked_nodes()
            self.erase_original_nodes()

        newgm = torch.fx._lazy_graph_module._make_graph_module(
            self.gm,
            self.parent_graph,
        )
        fake_tensor_prop(newgm)
        if log.isEnabledFor(logging.DEBUG):
            print("Graph module after chunking:")
            newgm.print_readable()
        return newgm
