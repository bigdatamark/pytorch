# mypy: allow-untyped-defs
# [MODULE DOC] Intermediate representation for pipeline parallelism. Defines Pipe, an
# nn.Module made of sequential stage submodules (named with the submod_pp prefix) produced by
# splitting a traced/exported model at pipe_split markers, plus SplitPoint, pipeline and
# annotate_split_points for specifying the split, LossWrapper helpers, symbolic backward
# insertion, and FX interpreters/serialization helpers. Import-time side effects: defines and
# registers the custom operator pippy::_pipe_split with a fake implementation and marks it as
# side-effectful in the FX node module so export preserves it.
# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from inspect import Parameter, Signature, signature
from types import MethodType
from typing import Any

import torch
import torch.fx as fx
from torch.distributed import ProcessGroup
from torch.export import ExportedProgram
from torch.export.unflatten import (
    _assign_attr,
    _AttrKind,
    _sink_params,
    InterpreterModule,
)
from torch.fx.node import map_aggregate
from torch.fx.passes.split_module import split_module

from ._backward import _null_coalesce_accumulate, stage_backward
from ._unflatten import _outline_submodules
from ._utils import PipeInfo
from .stage import _PipelineStage


logger = logging.getLogger(__name__)

# TODO:
# 1. investigate gradient sync for shared parameters. how does DDP do it?
# 2. Add parameter movement to split_module


PP_SUBMOD_PREFIX = "submod_pp"


def get_submod_name(stage_idx: int):
    # [FUNCTION DOC] Returns the submodule name for a stage index, the prefix joined with the
    # index by an underscore.
    """Returns the name of the submod for a given stage index.
    For example, "submod_pp_0", "submod_pp_1", etc.
    """
    return "_".join([PP_SUBMOD_PREFIX, str(stage_idx)])


def _find_loss_from_output_and_spec(output_val, spec_val):
    # [FUNCTION DOC] HELPER: Recursively walks an output value and a matching loss specification
    # (True marks the loss, False ignores, tuple/list/dict specs mirror the output structure) and
    # returns the FX node that is the loss. Raises RuntimeError if a True spec points to a non-node,
    # types, lengths or dict keys mismatch, no loss is found in a container, or the spec type is
    # unsupported.
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, fx.Node):
            raise RuntimeError(
                f"Loss spec must specify a dynamic value but got {output_val}"
            )
        return output_val

    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if len(output_val) != len(spec_val):
            raise RuntimeError(
                f"Output value {output_val} must match length of loss specification "
                f"{spec_val}"
            )
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(
                f"Output value {output_val} must match keys of loss specification "
                f"{spec_val}"
            )
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(output_val[k], spec_val[k])
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    raise RuntimeError(f"Unsupported type {type(spec_val)} in loss specification")


def _find_loss_output(mod: torch.nn.Module, g: fx.Graph, output_loss_value_spec):
    # [FUNCTION DOC] HELPER: Locates the loss in a graph's single output node. For a
    # TrivialLossWrapper the whole output is the loss; with no spec, looks for a "loss" key in a dict
    # output; otherwise resolves the given spec. Returns the loss node (or None), the output node,
    # and the spec actually used (or None). Raises AssertionError if the graph does not have exactly
    # one output node.
    output_nodes = [n for n in g.nodes if n.op == "output"]
    if not len(output_nodes) == 1:
        raise AssertionError(f"Expected 1 output node, got {len(output_nodes)}")
    output_node = output_nodes[0]
    output_val = output_node.args[0]
    generated_spec: Any = None

    if isinstance(mod, TrivialLossWrapper):
        # TrivialLossWrapper is pre-defined by PiPPy.
        # It has loss as the only output so we can safely assume the first output arg is the loss.
        if not len(output_node.args) == 1:
            raise AssertionError(f"Expected 1 output arg, got {len(output_node.args)}")
        loss_node = output_val
        generated_spec = TrivialLossWrapper.loss_spec
    elif output_loss_value_spec is None:
        # Use default spec, i.e. search for "loss" in output values
        if isinstance(output_val, dict) and "loss" in output_val:
            loss_node = output_val["loss"]
            generated_spec = {k: k == "loss" for k in output_val}
        else:
            loss_node = None
            generated_spec = None
    else:
        loss_node = _find_loss_from_output_and_spec(output_val, output_loss_value_spec)
        generated_spec = output_loss_value_spec

    return loss_node, output_node, generated_spec


def _insert_stage_symbolic_backward(
    g: fx.Graph,
    loss_node: fx.Node,
    output_node: fx.Node,
):
    # [FUNCTION DOC] HELPER: Appends symbolic backward computation to a stage-split forward graph,
    # in place, just before the output node. Scans for getitem nodes to reconstruct tuple outputs
    # of each stage call, then walks the graph in reverse from the loss; for each live stage module
    # call, inserts a call to stage_backward with its outputs, accumulated output grads, input
    # nodes and live output indices, and records gradients per input (accumulating with a
    # null-tolerant add when an input is used repeatedly, except for placeholders). Raises
    # AssertionError if the forward graph has a non-getitem function call or a malformed getitem.
    # Returns the graph.
    # Collect metadata about tuple output values. TODO: move this to split_module or FX IR
    tuples: dict[fx.Node, tuple] = {}
    for node in reversed(g.nodes):
        if node.op == "call_function":
            # In the forward pass, only emit placeholder, module calls, and
            # getitem calls. If we have a target other than getitem in this
            # (forward-only) code, there is a bug.
            if not node.target == operator.getitem:
                raise AssertionError(
                    "Found non-getitem call in forward pass. Please report a bug to PiPPy"
                )
            if not len(node.args) == 2:
                raise AssertionError(
                    "Found malformed getitem call. Please report a bug to PiPPy"
                )
            indexed_value, node_idx = tuple(node.args)

            # indexed_value is a collection that we are indexing into. It could
            # exist in the tuples map if we've processed another `getitem`
            # already.
            existing_list_size = (
                len(tuples[indexed_value]) if indexed_value in tuples else -1
            )
            new_list_size = max(node_idx + 1, existing_list_size)

            reconstructed_list = [None for _ in range(new_list_size)]

            # Copy over existing elements if present
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val

            # Populate value represented by this node
            reconstructed_list[node_idx] = node

            tuples[indexed_value] = tuple(reconstructed_list)

    # Keep track of nodes that dominate the loss node.
    # We will only emit backward operations for nodes that can contribute
    # to the specified loss value.
    live_nodes = {loss_node: None}
    val_to_grad: dict[fx.Node, fx.Node | None] = {loss_node: None}

    def assign_or_accumulate_grad(forward_node, grad_value):
        # [FUNCTION DOC] HELPER: Records the gradient node for a forward node; if one exists already
        # and the node is not a placeholder, inserts an accumulate call combining them.
        if forward_node in val_to_grad and forward_node.op != "placeholder":
            grad_value = g.call_function(
                _null_coalesce_accumulate,
                (val_to_grad[forward_node], grad_value),
            )
        val_to_grad[forward_node] = grad_value

    with g.inserting_before(output_node):
        for node in reversed(g.nodes):
            if node not in live_nodes:
                continue

            def add_to_live_nodes(n):
                # [FUNCTION DOC] HELPER: Marks a node as live (contributes to the loss) if not already present.
                live_nodes.setdefault(n, None)

            fx.node.map_arg(node.args, add_to_live_nodes)
            fx.node.map_arg(node.kwargs, add_to_live_nodes)
            if node.op == "call_module":
                output_grads: tuple[fx.Node | None, ...] | fx.Node | None
                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple(val_to_grad.get(n) for n in tuples[node])
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes
                    ]
                else:
                    stage_output = (node,)
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]

                output_grads = (
                    (output_grads,)
                    if not isinstance(output_grads, tuple)
                    else output_grads
                )

                grad_call = g.call_function(
                    stage_backward,
                    kwargs={
                        "stage_output": stage_output,
                        "output_grads": output_grads,
                        "input_values": list(node.all_input_nodes),
                        "outputs_with_grads_idxs": outputs_with_grads_idxs,
                    },
                )
                # Insert backward stage debug info
                kwargs_copy = dict(grad_call.kwargs)
                grad_call.kwargs = kwargs_copy

                grad_call_proxy = fx.Proxy(grad_call)
                grads = grad_call_proxy.node

                input_nodes = list(node.all_input_nodes)
                grads_proxy = fx.Proxy(grads)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)  # type: ignore[index]

    return g


def _move_placeholders_to_front(graph: fx.Graph):
    # [FUNCTION DOC] HELPER: Moves non-placeholder nodes that appear before the last placeholder
    # to just after it so placeholders form a contiguous prefix. Mutates the graph in place.
    """Ensure all placeholder nodes form a contiguous prefix of the graph.
    split_module can interleave get_attr nodes with placeholders."""
    last_ph = None
    for node in graph.nodes:
        if node.op == "placeholder":
            last_ph = node
    if last_ph is None:
        return
    nodes_to_move = []
    for node in graph.nodes:
        if node is last_ph:
            break
        if node.op != "placeholder":
            nodes_to_move.append(node)
    insert_point = last_ph
    for node in nodes_to_move:
        insert_point.append(node)
        insert_point = node


class PipeSequential(torch.nn.Sequential):
    # [CLASS DOC] Sequential container whose forward calls pipe_split between consecutive
    # submodules, so each child becomes its own pipeline stage. Create from a plain Sequential with
    # from_sequential, which shallow-copies the children.
    @staticmethod
    def from_sequential(sequential_instance: torch.nn.Sequential):
        # [FUNCTION DOC] Static method. Returns a PipeSequential holding shallow copies of the given
        # Sequential's modules.
        return PipeSequential(*[copy.copy(m) for m in sequential_instance])

    def forward(self, input):
        # [FUNCTION DOC] Runs each child module in order on the input, calling pipe_split after every
        # child except the last. Returns the final output.
        for i, module in enumerate(self):
            input = module(input)
            if i != len(self) - 1:
                pipe_split()
        return input


class LossWrapper(torch.nn.Module):
    # [CLASS DOC] Abstract base module bundling a model (attribute module) and a loss function
    # (attribute loss_fn). Subclasses override forward to define the connectivity from inputs
    # through model and loss to the returned value, which is then backpropagated.
    """
    LossWrapper is a convenient abstract class that allows you to wrap up both
    your model as well as its loss function and specify the connectivity between
    the inputs, model, loss function, and output value. Example::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    The above example defines a connectivity where we expect the forward/loss/backward
    training procedure to take two arguments (x and targets), pass x into the module
    to get the output of the feedforward computation, pass the model output and the
    targets value into the loss function, and get and return the loss value, which will
    be backpropagated by PiPPy. The above class would then be instantiated like::

        model = ...  # instantiate the model
        loss_fn = torch.nn.MSELoss()  # for the sake of demonstration

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """

    def __init__(self, module, loss_fn):
        # [FUNCTION DOC] Stores the wrapped module and the loss function.
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] Abstract; always raises NotImplementedError telling the user to override
        # forward.
        raise NotImplementedError(
            "This instance of LossWrapper does not have an overridden"
            "forward(). Please implement forward() to specify the arguments, "
            "connection between the module and loss, and loss output "
            "value."
        )


class TrivialLossWrapper(LossWrapper):
    # [CLASS DOC] Ready-made LossWrapper whose forward returns the loss of the module on x
    # against targets. Its class attribute loss_spec is True, marking the single output as the loss.
    # pyrefly: ignore [bad-override]
    def forward(self, x, targets):
        # [FUNCTION DOC] Runs the module on x and returns the loss function applied to the output and
        # targets.
        model_out = self.module(x)
        return self.loss_fn(model_out, targets)

    loss_spec = True


# Pipe model representation
#
# Pipe can be thought of as an `nn.Sequential++`. That is to say: it specifies
# a single topological ordering of pipeline "stages" that, when run in series,
# constitutes all of the operations of the program. However, unlike `nn.Sequential`,
# Pipe allows non-local usages of values, so long as those uses still respect
# topological ordering. In particular:
#
# 1. Non-local activations. This type of usage can appear in, for example, skip
#    connections. These values will be directly transmitted from the "def" stage
#    to all stages that use them skipping intermediate stages. During autograd,
#    gradients will be propagated back through this skip connection reverse
#    to how activations propagated in the forward pass.
# 2. Non-local parameter/module invocations. This occurs when a parameter is used
#    in a stage downstream of where it is resident. These values can be carried
#    forward similarly to (1), but in addition one might want to replicate the
#    value on multiple stages. Gradients for these shared parameters will be
#    accumulated separately on each stage, but there will be an additional
#    gradient accumulation before the optimizer step.


# Register `_pipe_split()` as an ATen operator. This is required for Export to
# preserve this marker in the graph.
torch.library.define("pippy::_pipe_split", "() -> ()")


@torch.library.impl("pippy::_pipe_split", "BackendSelect")
def _pipe_split():
    # [FUNCTION DOC] HELPER: Implementation of the custom operator that marks a split point; does
    # nothing and returns None.
    return None


@torch.library.register_fake("pippy::_pipe_split")  # type: ignore[no-redef]
def _pipe_split():
    # [FUNCTION DOC] HELPER: Fake (meta) implementation of the split marker operator; does
    # nothing and returns None.
    return None


# Add an alias for convenience
aten_pipe_split_alias = torch.ops.pippy._pipe_split.default

# Ask Export to preserve the `_pipe_split` op.
# See examples in pytorch/torch/fx/node.py
fx.node._side_effectful_functions.add(aten_pipe_split_alias)


# User facing API
def pipe_split():
    # [FUNCTION DOC] User-facing marker for a stage boundary. Calls the registered custom operator
    # so tracing/export records a split point; it does nothing when the model is run eagerly.
    """
    pipe_split is a special operator that is used to mark the boundary between
    stages in a module. It is used to split the module into stages. It is a
    no-op if your annotated module is run eagerly.

    Example:
        >>> # xdoctest: +SKIP
        >>> def forward(self, x):
        >>>     x = torch.mm(x, self.mm_param)
        >>>     x = torch.relu(x)
        >>>     pipe_split()
        >>>     x = self.lin(x)
        >>>     return x

    The above example will be split into two stages.
    """
    return torch.ops.pippy._pipe_split()


class MultiUseParameterConfig(Enum):
    # [CLASS DOC] Enum of policies for parameters used by multiple stages: TRANSMIT (send the value
    # between stages) or REPLICATE (keep a copy on each stage). MultiUseParamSpec below is a type
    # alias for either one policy or a per-parameter-name mapping of policies.
    TRANSMIT = 1
    REPLICATE = 2


MultiUseParamSpec = MultiUseParameterConfig | dict[str, MultiUseParameterConfig]


class DetachExecutor(fx.Interpreter):
    # [CLASS DOC] FX interpreter used to run the split module locally (testing). It detaches every
    # requires-grad tensor passed to a stage module call, so boundary values become autograd
    # leaves, and remembers the detached copies so the backward calls can be rerouted to them.
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """

    def __init__(self, module, garbage_collect_values=True):
        # [FUNCTION DOC] Initializes the interpreter. The garbage collection flag is always overridden
        # to False so intermediate values stay alive; also creates the empty detached-value map.
        # KNOWN ISSUE: the garbage_collect_values argument is ignored.
        garbage_collect_values = False
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}

    def run(self, *args, initial_env=None):  # type: ignore[override]
        # [FUNCTION DOC] Resets the detached-value map then runs the interpreter on the arguments.
        self.value_remap = {}
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):
        # [FUNCTION DOC] Runs a module call after replacing each requires-grad tensor in args and
        # kwargs by a detached copy that requires grad (cached per original tensor in the map).
        def detach_tensors(a):
            # [FUNCTION DOC] HELPER: Returns the cached detached requires-grad copy of a tensor needing
            # grad, creating it on first use; other values are returned unchanged.
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a

        """
        def dont_traverse_size(a):
            return type(a) is not torch.Size
        """

        args = map_aggregate(
            args,
            detach_tensors,  # dont_traverse_size
        )
        kwargs = map_aggregate(
            kwargs,
            detach_tensors,  # dont_traverse_size
        )

        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        # [FUNCTION DOC] Runs a function call; for stage_backward calls, replaces the input_values
        # with their detached copies so gradients land on the tensors the stage actually used.
        # HACK to reroute saved input tensors to point to the detach()ed version
        if target is stage_backward:
            kwargs = dict(kwargs)
            kwargs["input_values"] = [
                self.value_remap.get(v, v) for v in kwargs["input_values"]
            ]
        return super().call_function(target, args, kwargs)


class _NodeReference:
    # [CLASS DOC] HELPER: Placeholder holding the name of an FX node, used in serialized node args
    # in place of real node objects.
    def __init__(self, name):
        # [FUNCTION DOC] HELPER: Stores the node name.
        self.name = name

    name: str


class _LinearNodeList:
    # [CLASS DOC] HELPER: Picklable linear snapshot of an FX graph's nodes, with node references
    # in args and kwargs replaced by name references. to_graph rebuilds a new graph.
    def __init__(self, node_list):
        # [FUNCTION DOC] HELPER: Copies each node (without a graph) with name-reference args and kwargs
        # and a shallow copy of metadata into a list.
        self.serialize_node_list = []
        for node in node_list:
            node_args = fx.node.map_arg(node.args, lambda n: _NodeReference(n.name))  # type: ignore[arg-type,return-value]
            node_kwargs = fx.node.map_arg(node.kwargs, lambda n: _NodeReference(n.name))  # type: ignore[arg-type,return-value]
            serialize_node = fx.Node(
                graph=None,  # type: ignore[arg-type]
                name=node.name,
                op=node.op,
                target=node.target,
                args=node_args,  # type: ignore[arg-type]
                kwargs=node_kwargs,  # type: ignore[arg-type]
                return_type=node.type,
            )
            serialize_node.meta = copy.copy(node.meta)
            self.serialize_node_list.append(serialize_node)

    def to_graph(self):
        # [FUNCTION DOC] HELPER: Rebuilds and returns a new FX graph by creating nodes in order and
        # resolving name references back to the newly created nodes.
        graph = fx.Graph()

        ref_str_to_node: dict[str, fx.Node] = {}

        def ref_to_node(arg):
            # [FUNCTION DOC] HELPER: Maps a name reference to the rebuilt node; other values are unchanged.
            if isinstance(arg, _NodeReference):
                return ref_str_to_node[arg.name]
            else:
                return arg

        for node in self.serialize_node_list:
            node_args = map_aggregate(node.args, ref_to_node)
            node_kwargs = map_aggregate(node.kwargs, ref_to_node)
            deser_node = graph.create_node(
                op=node.op,
                target=node.target,
                args=node_args,  # type: ignore[arg-type]
                kwargs=node_kwargs,  # type: ignore[arg-type]
                name=node.name,
                type_expr=node.type,
            )
            ref_str_to_node[node.name] = deser_node

        return graph


def _direct_serialization_deserialize(body, nodes):
    # [FUNCTION DOC] HELPER: Reconstruction function used when unpickling a stage GraphModule:
    # builds a dummy module from the saved attribute dictionary and returns a GraphModule over the
    # rebuilt graph. Intended for RPC transmission only, not for persisting to disk.
    """
    Custom `__reduce__` method for serialization.
    DO AS I SAY -- NOT AS I DO. This violates the principle that
    GraphModules serialize via code export & re-tracing. We allow
    for this here because **PIPE STAGES SHOULD NOT BE PERSISTED
    TO DISK -- THIS IS ONLY FOR TRANSMISSION VIA RPC**. Persisting
    these instances to disk will expose internal implementation
    details of `fx.Graph` and related data structures and is
    NOT advised.
    """

    class DummyModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Minimal module whose attribute dictionary is filled from a saved dict.
        def __init__(self, body):
            # [FUNCTION DOC] HELPER: Initializes the module and copies the saved attributes into it.
            super().__init__()
            self.__dict__.update(body)

    dummy = DummyModule(body)

    return fx.GraphModule(dummy, nodes.to_graph())


def _direct_serialization_reduce(self):
    # [FUNCTION DOC] HELPER: Custom reduce for stage GraphModules: returns the deserialize function
    # with the module's attribute dictionary (minus the graph) and a linear node list.
    serialization_dict = dict(self.__dict__)
    serialization_dict.pop("_graph")
    return (
        _direct_serialization_deserialize,
        (serialization_dict, _LinearNodeList(self.graph.nodes)),
    )


def _modify_graph_op_device(
    gm: torch.fx.GraphModule,
    new_device: torch.device,
):
    # [FUNCTION DOC] HELPER: Sets the device keyword of every function call node in the graph that
    # has a different one to the new device, recursing into graph modules and unflattened
    # interpreter submodules (logging a warning for other submodule types), and recompiles if
    # anything changed. Mutates the module in place.
    """
    Modify the device argument of all "call_function" nodes in the graph.  This
    is useful for moving the graph to a different device. In particular for
    generator ops, like torch.ones.
    """
    modified = False
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if "device" in node.kwargs and node.kwargs["device"] != new_device:
                logger.debug(
                    f"Changing device of Node {node.name} from {node.kwargs['device']} to {new_device}"  # noqa: G004
                )
                node.update_kwarg("device", new_device)
                modified = True
        elif node.op == "call_module":
            # Recursively modify "device" in submodules
            submod = gm.get_submodule(node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _modify_graph_op_device(submod, new_device)
            elif isinstance(submod, InterpreterModule):
                # If unflattening has been performed, we need to access its graph module by `.graph_module`
                _modify_graph_op_device(submod.graph_module, new_device)  # type: ignore[arg-type]
            else:
                logger.warning(
                    f"Skipping device modification for submodule {node.target} because it is a {type(submod)}"  # noqa: G004
                )

    if modified:
        gm.recompile()


class Pipe(torch.nn.Module):
    # [CLASS DOC] A pipeline as a module: holds a split graph module whose children are stage
    # submodules, the number of stages, whether loss and backward are included, and the loss spec.
    # It detects parameters shared by several stages (replicated_params). Calling forward runs all
    # stages sequentially on the local process through a DetachExecutor. Key methods: forward,
    # get_stage_module, from_tracing, info, build_stage.
    def __init__(
        self,
        split_gm: fx.GraphModule,
        num_stages: int,
        has_loss_and_backward: bool,
        loss_spec,
    ):
        # [FUNCTION DOC] Stores the split module, a DetachExecutor, stage count, loss flag and loss spec.
        # Checks all graph nodes are of allowed kinds (raises AssertionError otherwise), finds
        # parameters shared by multiple stage submodules and deep-copies them to break aliasing, replaces
        # the split module's forward with one that raises, and installs the direct-serialization reduce on
        # each stage submodule class.
        # TODO: is there a way not to hard wire init?
        torch.nn.Module.__init__(self)
        self.split_gm: fx.GraphModule = split_gm
        self.executor: DetachExecutor = DetachExecutor(self.split_gm)
        self.num_stages: int = num_stages
        self.has_loss_and_backward = has_loss_and_backward
        self.loss_spec = loss_spec

        for node in split_gm.graph.nodes:
            if not (
                node.op in {"call_module", "placeholder", "output"}
                or (node.op, node.target) == ("call_function", operator.getitem)
                or (node.op, node.target) == ("call_method", "backward")
                or (node.op, node.target) == ("call_function", stage_backward)
                or (node.op, node.target)
                == ("call_function", _null_coalesce_accumulate)
            ):
                raise AssertionError(f"Unexpected node: {node}")

        # Detect replicated parameters so we know that we have to do an additional allreduce
        # before applying the optimizer
        #
        # Note that this also handles the case where there were multiple calls to a single
        # module from different stages, regardless of whether that module invocation
        # was handled by the logic above.

        # Map parameter value to a dictionary that maps the user pipeline module
        # to the local qualname within that module
        params_to_users: dict[torch.nn.Parameter, dict[str, str]] = {}

        for m_qualname, mod in self.split_gm.named_children():
            for p_qualname, param in mod.named_parameters():
                params_to_users.setdefault(param, {})
                params_to_users[param][m_qualname] = p_qualname

        self.replicated_params: list[dict[str, str]] = [
            use_mapping
            for use_mapping in params_to_users.values()
            if len(use_mapping) > 1
        ]

        # We must break the aliasing relationship between the replicated parameters for correct
        # numerics in reference runs. If we do not do this, the autograd tape in separate stages
        # will have a reference to the same tensor value and will erroneously apply gradient
        # updates multiple times. Therefore, for each replicated parameter set, we deepcopy the
        # values so that we have separate instances.
        for param_mapping in self.replicated_params:
            for submod_name, param_qualname in param_mapping.items():
                submod = getattr(self.split_gm, submod_name)
                atoms = param_qualname.split(".")
                for atom in atoms[:-1]:
                    submod = getattr(submod, atom)
                setattr(submod, atoms[-1], copy.deepcopy(getattr(submod, atoms[-1])))

        def throw(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement forward for the split module; always raises RuntimeError
            # telling the user to call the Pipe object instead.
            raise RuntimeError(
                "To run pipeline locally, invoke the Pipe object directly, not `split_gm`"
            )

        self.split_gm.forward = throw

        # Make submodules use custom direct-serialized GraphModule
        i = 0
        while True:
            try:
                name = get_submod_name(i)
                submod = getattr(self.split_gm, name)
                submod.__class__.__reduce__ = _direct_serialization_reduce
                i += 1
            except AttributeError:
                break

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the whole pipeline locally. If keyword arguments are given, builds a
        # signature from the graph placeholders (with defaults and star-argument kinds), binds and
        # applies defaults, then executes the executor on the resulting ordered arguments. Returns the
        # result.
        executor_args = args
        if len(kwargs) > 0:
            parameters = []
            for node in self.split_gm.graph.nodes:
                if node.op == "placeholder":
                    if node.args and len(node.args) > 0:
                        parameters.append(
                            Parameter(
                                node.target,
                                Parameter.POSITIONAL_OR_KEYWORD,
                                default=node.args[0],
                            )
                        )
                    else:
                        parameter_kind = Parameter.POSITIONAL_OR_KEYWORD
                        param_name = node.target
                        if node.target.startswith("**"):
                            parameter_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
                            param_name = param_name[2:]
                        elif node.target.startswith("*"):
                            parameter_kind = Parameter.VAR_POSITIONAL  # type: ignore[assignment]
                            param_name = param_name[1:]
                        parameters.append(Parameter(param_name, parameter_kind))
            signature = Signature(parameters)
            ba = signature.bind(*args, **kwargs)
            ba.apply_defaults()
            executor_args = ba.arguments.values()  # type: ignore[assignment]

        res = self.executor.run(*executor_args)

        return res

    def get_stage_module(self, stage_idx: int) -> torch.nn.Module:
        # [FUNCTION DOC] Returns the stage submodule for a stage index. Raises ValueError if the index is
        # out of range.
        """
        Return a stage module corresponding to `stage_idx` of the `pipe`.
        """
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise ValueError(f"Invalid stage index {stage_idx}!")

        submod_name = get_submod_name(stage_idx)
        return getattr(self.split_gm, submod_name)

    @staticmethod
    def _number_and_count_forward_stages(gm: fx.GraphModule):
        # [FUNCTION DOC] HELPER: Static method. Counts call-module nodes whose target starts with the
        # stage prefix and records the stage index in each node's metadata. Returns the count.
        num_stages = 0
        found_idxs: dict[int, None] = {}
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target.startswith(PP_SUBMOD_PREFIX):
                node.meta["stage_idx"] = int(node.target[len(PP_SUBMOD_PREFIX) + 1 :])
                found_idxs.setdefault(node.meta["stage_idx"])
                num_stages += 1

        # this assert will fail if a split point is inserted before the first layer, which creates empty first submodule
        # Update: the following assert may fail against some torch versions >=
        # 2.2.0, as:
        # submod_0, submod_1, submod_2, ...
        # may be named as
        # submod_0, submod_2, submod_4, ...
        # TODO: investigate
        # assert all(i in found_idxs for i in range(num_stages))

        return num_stages

    @staticmethod
    def _from_traced(
        mod: torch.nn.Module,
        exported_program: ExportedProgram,
        multi_use_param_spec: MultiUseParamSpec | None = None,
        output_loss_value_spec=None,
        split_policy: Callable[[torch.fx.GraphModule], torch.fx.GraphModule]
        | None = None,
    ):
        # [FUNCTION DOC] HELPER: Static method that converts an exported program of a model into a
        # Pipe. Steps: get the module from the program (optionally applying a split policy to insert
        # split markers automatically); dedupe get_attr nodes per target; drop consecutive duplicate
        # pipe_split markers; split the graph into numbered stage submodules at the split operator
        # (dropping the marker nodes inside stages); outline each stage into nested submodules; move
        # parameters, buffers and lifted constants from the root into the stage submodules that use them
        # (handling aliased state dict entries), sink them into the graph, and remove unused
        # attributes and the root get_attr nodes; count stages. If output_loss_value_spec is given and a
        # loss node is found, symbolic backward is inserted. Args: mod (original module), exported_program,
        # multi_use_param_spec (see MultiUseParameterConfig), output_loss_value_spec (which output is the
        # loss), split_policy (optional callable to split the module). Returns the Pipe.
        # Raises AssertionError on unexpected graph structure or a non-tensor attribute.
        """
        Additionally, the ``output_loss_value_spec`` value can be specified to disambiguate
        which value in the output of `forward` is the loss value on which PiPPy should apply
        backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
        you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
        a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
        ``output_loss_value_spec={'loss': True, 'model_out': False}``
        """

        traced = exported_program.module(check_guards=False)

        if split_policy is not None:
            logger.info("Auto-splitting model")
            traced = split_policy(traced)  # type: ignore[arg-type]

        logger.debug(traced.print_readable(print_output=False))  # type: ignore[operator]

        # Deduplicate `get_attr` nodes that refer to the same parameter . Downstream code for moving
        # parameters relies on the invariant that parameter accesses happen once. This is not necessarily
        # the case (especially with custom tracers), so fix that up here.
        get_attr_nodes: dict[str, fx.Node] = {}
        for node in traced.graph.nodes:  # type: ignore[union-attr]
            if node.op == "get_attr":
                get_attr_nodes.setdefault(node.target, node)

                if get_attr_nodes[node.target] != node:
                    node.replace_all_uses_with(get_attr_nodes[node.target])
                    traced.graph.erase_node(node)  # type: ignore[operator, union-attr]

        # avoid looking at next node by keeping track of previous pipe_split
        prev_pipe_split_idx = -1
        pipe_split_nodes_to_erase = set()
        for i, node in enumerate(traced.graph.nodes):  # type: ignore[arg-type, union-attr]
            if (node.op, node.target) == ("call_function", pipe_split):
                if prev_pipe_split_idx == i - 1:
                    pipe_split_nodes_to_erase.add(node)
                prev_pipe_split_idx = i

        for node in pipe_split_nodes_to_erase:
            traced.graph.erase_node(node)  # type: ignore[operator, union-attr]

        traced.recompile()  # type: ignore[operator]

        part_idx = 0

        def split_callback(n: fx.Node):
            # [FUNCTION DOC] HELPER: Callback for split_module returning the current partition index; the
            # index is incremented whenever a pipe_split call node is seen (nonlocal counter).
            nonlocal part_idx
            if (n.op, n.target) == (
                "call_function",
                aten_pipe_split_alias,
            ):
                logger.debug(f"Found pipe_split {part_idx}")  # noqa: G004
                part_idx += 1
            return part_idx

        # TODO: what does split do with module invocations? does it move the modules
        # into the submodules?
        split = split_module(traced, mod, split_callback, partition_affix="pp")  # type: ignore[arg-type]
        # a (custom) tracer can produce dead code like orphan get_attr nodes
        split.graph.eliminate_dead_code()

        # peephole to remove pipe_split
        for submodule in split.modules():
            if isinstance(submodule, fx.GraphModule):
                for node in submodule.graph.nodes:
                    if (node.op, node.target) == (
                        "call_function",
                        aten_pipe_split_alias,
                    ):
                        submodule.graph.erase_node(node)
                submodule.recompile()

        for name, submodule in split.named_children():
            if isinstance(submodule, fx.GraphModule):
                _move_placeholders_to_front(submodule.graph)
                new_submod = _outline_submodules(submodule.graph)
                # Replace old submod
                split.register_module(name, new_submod)

        # TODO: backport this into split_module
        def delete_user_reference(node, user):
            # [FUNCTION DOC] HELPER: Removes the one positional argument equal to the given root get_attr
            # node from the submodule call node's args. Raises AssertionError if the user has keyword
            # arguments or the node is not used exactly once.
            """
            Delete reference of `node` from `user`'s arg list.
            Args:
                - node: a `get_attr` node at root.
                - user: a submodule node that uses `node`.
            """
            if not len(user.kwargs) == 0:
                raise AssertionError(
                    f"Expected user.kwargs to be empty, got {len(user.kwargs)}"
                )
            use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
            if not len(use_idxs) == 1:
                raise AssertionError(f"Expected 1 use index, got {len(use_idxs)}")
            args_copy = list(user.args)
            args_copy.pop(use_idxs[0])
            user.args = tuple(args_copy)
            logger.debug(
                f"Deleted {node} from user {user}, arg index = {use_idxs[0]}"  # noqa: G004
            )

        # A list of param referrals for deferred deletion.
        # To be accumulated in `move_param_to_callee`.
        to_delete = []

        def _recursive_getattr_with_parent(mod, fqn):
            # [FUNCTION DOC] HELPER: Walks a dotted attribute path from a module. Returns the pair
            # (parent module, attribute): (None, None) if an intermediate part is missing, (parent, None)
            # if only the last attribute is missing.
            # Returns getattr call given a nested FQN, and the last parent
            atoms = fqn.split(".")
            for atom in atoms[:-1]:
                if not hasattr(mod, atom):
                    return None, None
                mod = getattr(mod, atom)
            if not hasattr(mod, atoms[-1]):
                return mod, None
            attr = getattr(mod, atoms[-1])
            return mod, attr

        def move_param_to_callee(
            root,
            callee_name,
            param_fqn,
        ):
            # [FUNCTION DOC] HELPER: Moves a parameter or buffer (by dotted name) from the root split module
            # into the named stage submodule, preserving persistence, and queues the original attribute for
            # deferred deletion. Raises AssertionError if the value is not a tensor or the callee already has it.
            """
            Move a parameter from the root module to a submodule.
            Args:
                root: The root module.
                callee_name: The name of the submodule to move the parameter to.
                param_fqn: The fully qualified name of the parameter to move.
            """
            # `atoms` is a list of strings representing the path to the
            # parameter in the original model
            atoms = param_fqn.split(".")
            mod_itr, param_val = _recursive_getattr_with_parent(split, param_fqn)
            # Check whether the parameter is a buffer or a parameter
            is_buffer = atoms[-1] in mod_itr._buffers

            # Check whether the parameter is a tensor
            if not isinstance(param_val, torch.Tensor):
                raise AssertionError(
                    f"Expected '{param_fqn}' to be {torch.Tensor} but got {type(param_val)}."
                    + (
                        f" It might happen if module '{param_fqn}' was passed to some 'leaf function'"
                        f"(see https://pytorch.org/docs/stable/fx.html#fx.wrap). Please inspect "
                        f"usages of '{param_fqn}' in the traced graph."
                        if isinstance(param_val, torch.nn.Module)
                        else ""
                    )
                )

            # Get submodule
            callee = root.get_submodule(callee_name)
            if hasattr(callee, param_fqn):
                raise AssertionError(
                    f"Module {callee_name} already has a parameter named {param_fqn}"
                )

            # Assign the parameter to the submodule
            if is_buffer:
                _assign_attr(
                    param_val,
                    callee,
                    param_fqn,
                    attr_kind=_AttrKind.BUFFER,
                    persistent=True,  # TODO: handle non-persistent buffer
                )
            else:
                _assign_attr(
                    param_val,
                    callee,
                    param_fqn,
                    attr_kind=_AttrKind.PARAMETER,
                )
            logger.debug(f"Moved parameter {param_fqn} to {callee_name}")  # noqa: G004

            # Next step is to replace placeholder of submodule with a get_attr.
            # Those placeholders are created by `split_module` inside each
            # submodule.
            # Update: this step is now moved to `_sink_params` because
            # `_sink_params` can do it recursively (i.e. for modules inside
            # submodule)

            to_delete.append((mod_itr, atoms[-1]))

        # Get the list of all parameters in the root module
        attr_nodes = list(filter(lambda n: n.op == "get_attr", split.graph.nodes))
        for node in attr_nodes:
            # Check whether the parameter is used in only one submodule
            if len(node.users) > 1:
                logger.info(
                    f"Parameter {node.target} used in multiple stages: {node.users}."  # noqa: G004
                )
            for user in node.users:
                if not user.op == "call_module":
                    raise AssertionError(
                        f"Expected user.op to be 'call_module', got {user.op}"
                    )
                # Move parameter into submodule
                move_param_to_callee(
                    split,
                    user.target,
                    node.target,
                )

        # [aliasing] store tensor id -> list of FQNs, built from state dict
        # Also assign non-persistent buffers
        id_to_fqns: dict[int, set[str]] = defaultdict(set)
        for fqn, tensor in mod.state_dict(keep_vars=True).items():
            id_to_fqns[id(tensor)].add(fqn)
        for fqn, tensor in mod.named_buffers():
            id_to_fqns[id(tensor)].add(fqn)

        # After moving the params to their corresponding hierarchies, we also
        # need to move the `get_attr` nodes from the root of the graph to those
        # hierarchies.
        # [aliasing] use id -> fqn mapping to list out all valid FQNs
        inputs_to_state: dict[str, list[str]] = {}
        for attr in attr_nodes:
            _, tensor = _recursive_getattr_with_parent(mod, attr.target)
            fqns = list(id_to_fqns[id(tensor)])
            if fqns:
                inputs_to_state[attr.name] = fqns
            elif attr.target in exported_program.constants:  # lifted constants
                inputs_to_state[attr.name] = [attr.target]

        # [aliasing] for each submodule split, assign attributes on FQNs that may be used.
        # We determine this based on whether or not the FQN attribute parent exists.
        # i.e. if the last submodule exists, assign the attribute.
        added_attributes: dict[str, list[str]] = defaultdict(list)
        for fqn, tensor in mod.state_dict(keep_vars=True).items():
            for name, submod in split.named_children():
                if isinstance(submod, fx.GraphModule):
                    parent, child = _recursive_getattr_with_parent(submod, fqn)
                    if (
                        parent and child is None
                    ):  # parent exists, attribute doesn't -> assign
                        added_attributes[name].append(fqn)
                        setattr(parent, fqn.split(".")[-1], tensor)

        # Deferral deletion: Remove the original attributes (to params) from the
        # root GraphModule
        for mod_itr, last_atom in to_delete:
            try:
                delattr(mod_itr, last_atom)
            except AttributeError:
                # This is expected if the parameter is used in multiple stages
                pass

        # This is done by (1) `_sink_params` at each submodule;
        for submod in split.children():
            if isinstance(submod, fx.GraphModule):
                _sink_params(submod, inputs_to_state, [])
                submod.graph.lint()
                submod.recompile()

        # [aliasing] This step is not super necessary, but helps reduce parameter usage/memory.
        # After _sink_params() routine has run, clean up unused attributes that we previously added.
        # Determine this based on the get_attr nodes - if not used, remove it.
        for name, attributes in added_attributes.items():
            submod = getattr(split, name)
            unused_attributes = set(attributes)
            # track used attributes in the submodule, running DFS on subgraph hierarchy
            stack = [("", submod)]  # (scope, submodule)
            while stack:
                scope, _mod = stack.pop()
                if isinstance(_mod, (fx.GraphModule, InterpreterModule)):
                    for node in _mod.graph.nodes:
                        if node.op == "get_attr":
                            # get_attr might get access deeper level attribute
                            fqn = scope + "." + node.target if scope else node.target
                            unused_attributes.discard(fqn)
                for _name, _submod in _mod.named_children():
                    stack.append((scope + "." + _name if scope else _name, _submod))
            # delete unused attributes
            for attr in unused_attributes:
                mod_itr, atoms = submod, attr.split(".")
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                delattr(mod_itr, atoms[-1])

        for node in attr_nodes:
            # And (2): remove `get_attr` node from submod's arg list
            for user in copy.copy(node.users):
                if not user.op == "call_module":
                    raise AssertionError(
                        f"Expected user.op to be 'call_module', got {user.op}"
                    )
                delete_user_reference(node, user)
            # And (3): remove the `get_attr` node from the root graph.
            split.graph.erase_node(node)

        split.delete_all_unused_submodules()
        split.graph.lint()
        split.recompile()

        num_stages = Pipe._number_and_count_forward_stages(split)

        has_loss_and_backward = False
        generated_loss_spec = output_loss_value_spec

        if output_loss_value_spec is not None:
            loss_node, output_node, generated_loss_spec = _find_loss_output(
                mod, split.graph, output_loss_value_spec
            )
            if loss_node is not None:
                _insert_stage_symbolic_backward(
                    split.graph,
                    loss_node,
                    output_node,
                )
                split.recompile()
                has_loss_and_backward = True
                logger.debug("Pipeline is in training mode, backward pass generated")
            else:
                raise RuntimeError(
                    f"Did not find any loss value according to {output_loss_value_spec=}"
                )
        else:
            logger.debug("Pipeline is in inference mode, backward pass not generated")

        logger.debug(f"Full pipe model:\n{split}")  # noqa: G004

        return Pipe(
            split,
            num_stages,
            has_loss_and_backward,
            generated_loss_spec,
        )

    def print_readable(self):
        # [FUNCTION DOC] Prints the root split module and each stage module in readable form.
        """
        Print the pipe in a human-readable format.
        This will print both the root pipe and each stage module.
        """
        self.split_gm.print_readable()

    @staticmethod
    def _trace_with_export(
        mod: torch.nn.Module,
        example_args: tuple[Any, ...],
        example_kwargs: dict[str, Any] | None = None,
    ) -> ExportedProgram:
        # [FUNCTION DOC] HELPER: Static method that exports the module with example positional and
        # keyword inputs and returns the exported program. Raises RuntimeError (chained to the original
        # error) if the model cannot be captured as a full graph. Logs at info level.
        logger.info("Tracing model ...")
        try:
            ep = torch.export.export(mod, example_args, example_kwargs)
        except Exception as e:
            raise RuntimeError(
                "It seems that we cannot capture your model as a full graph. "
                "Typical reasons include graph breaks, data/shape-dependent "
                "control flow, or missing meta kernels for custom operators. "
                "You can use our manual pipeline interfaces, or try to fix the "
                "graph breaks, see https://pytorch.org/docs/stable/export.html"
            ) from e

        return ep

    @staticmethod
    def from_tracing(
        mod: torch.nn.Module,
        example_args: tuple[Any, ...],
        example_kwargs: dict[str, Any] | None = None,
        split_policy: Callable[[fx.GraphModule], fx.GraphModule] | None = None,
    ):
        # [FUNCTION DOC] Static method that traces a module with export and builds a Pipe from it, using
        # the replicate strategy for shared parameters and no loss spec. Optionally applies a split
        # policy. If the first stage takes the same number of arguments as the original model, copies
        # the original input calling convention onto it (so keyword arguments work) and recompiles it;
        # otherwise leaves its signature and logs an info message. Returns the Pipe. Raises RuntimeError
        # if tracing fails.
        # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
        # stages instead of TRANSMIT'ting it
        multi_use_param_spec = MultiUseParameterConfig.REPLICATE

        # Figure out which output is loss from output_chunk_spec
        output_loss_value_spec: Any = None
        # Deprecated
        """
        if output_chunk_spec is not None:
            output_loss_value_spec = map_aggregate(
                output_chunk_spec, lambda v: isinstance(v, _LossReducer)
            )
        """

        # Trace with export
        exported_program = Pipe._trace_with_export(
            mod,
            example_args,
            example_kwargs,
        )

        pipe = Pipe._from_traced(
            mod,
            exported_program,
            multi_use_param_spec,
            output_loss_value_spec=output_loss_value_spec,
            split_policy=split_policy,
        )

        # Users want the first pipeline stage to accept kwargs if the original
        # program does. This is controlled by the `_codegen` field of the graph,
        # so we make a copy here. Note: we only want the input spec and not the
        # output spec, because the output spec is for the last stage. Maybe a
        # TODO? Not sure yet.
        split = pipe.split_gm
        traced = exported_program.module()
        submod0 = next(iter(split.children()))
        submod0_sign = signature(submod0.forward)
        model_sign = signature(traced.forward)
        if len(model_sign.parameters) != len(submod0_sign.parameters):
            # We don't change the signature of the first stage if it takes
            # different number of args than original model
            logger.info(
                f"Original model takes {len(model_sign.parameters)} args but the "  # noqa: G004
                f"first pipeline stage takes {len(submod0_sign.parameters)}. "
                "Please provide args to respective pipeline stages."
            )
        else:
            # Support kwargs for the first stage
            submod0.graph._codegen = copy.deepcopy(traced.graph._codegen)  # type: ignore[union-attr]
            # `_replace` is actually not "private" or internal. based on this doc:
            # To prevent conflicts with field names, the method and attribute names
            # start with an underscore
            submod0.graph._codegen.pytree_info = (  # type: ignore[union-attr]
                submod0.graph._codegen.pytree_info._replace(out_spec=None)  # type: ignore[operator, union-attr]
            )
            submod0.recompile()

        return pipe

    def __str__(self):
        # [FUNCTION DOC] Returns the string form of the split graph module.
        return self.split_gm.__str__()

    def __repr__(self):
        # [FUNCTION DOC] Returns the repr of the split graph module.
        return self.split_gm.__repr__()

    def info(self) -> PipeInfo:
        # [FUNCTION DOC] Returns a PipeInfo holding the split graph, stage count and the loss/backward flag.
        # Deliberately does not reference the Pipe so other stages can be freed.
        """
        Get information about the pipe.

        Returns
        -------
        PipeInfo
            A dataclass containing information about the pipe.
        """
        return PipeInfo(
            graph=self.split_gm.graph,
            num_stages=self.num_stages,
            has_loss_and_backward=self.has_loss_and_backward,
        )

    def build_stage(
        self,
        stage_index: int,
        device: torch.device,
        group: ProcessGroup | None = None,
    ) -> _PipelineStage:
        # [FUNCTION DOC] Creates a pipeline stage object for the given stage index, device and optional
        # process group. Moves hard-coded device keyword arguments in the stage graph to the device
        # (logging a warning if the stage is not a GraphModule), then builds the stage with the pipe info.
        # Raises ValueError if the stage index is out of range.
        """
        Create a `PipelineStage` given a stage index and distributed group.
        The `PipelineStage` can run with `PipelineSchedule`s.
        """
        # Find stage module
        stage_module = self.get_stage_module(stage_index)

        # Move ops argument to device
        # Today PT2 tracer does not treat `x.device` as a symbolic device;
        # instead, the device of tracing time got burned into the generated
        # code.  Here we provide a workaround for users to manually modify the
        # "device" kwarg of operations. Such operation may include:
        # `torch.ones`, `torch.zeros`, `torch.rand`, etc.
        if isinstance(stage_module, torch.fx.GraphModule):
            _modify_graph_op_device(stage_module, device)
        else:
            logger.warning(
                f"Expected a `torch.fx.GraphModule` but got {type(stage_module)}"  # noqa: G004
            )

        # Detach pipe info
        # Note: be careful what's included in `pipe_info`. We don't want to keep
        # a reference to `Pipe` or `Pipe.split_gm` which stops python from
        # recycling them. When python recycles them, other stage modules (which
        # are irrelevant to current rank) can be automatically freed.
        pipe_info = self.info()
        return _PipelineStage(stage_module, stage_index, pipe_info, device, group)


class SplitPoint(Enum):
    # [CLASS DOC] Enum of where to place a split relative to a submodule's forward: BEGINNING
    # (before it runs) or END (after it runs).
    """
    Enum representing the points at which a split can occur in the execution of a submodule.
    Attributes:
        BEGINNING: Represents adding a split point *before* the execution of a certain submodule in the `forward` function.
        END: Represents adding a split point *after* the execution of a certain submodule in the `forward` function.
    """

    BEGINNING = 1
    END = 2


# For backward compatibility, we kept the PipeSplitWrapper class because `class
# SplitPoint` used to be defined in this class.
class PipeSplitWrapper:
    # [CLASS DOC] HELPER: Backward-compatibility holder exposing SplitPoint as a class attribute.
    # Create a class alias for BC
    SplitPoint = SplitPoint


def _split_before_forward(self, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement forward used for splitting before a module: emits a split
    # marker, then calls the original forward saved on the module.
    pipe_split()
    return self._orig_forward(*args, **kwargs)


def _split_after_forward(self, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Replacement forward used for splitting after a module: calls the
    # original forward and always emits a split marker afterwards, even if it raises.
    try:
        return self._orig_forward(*args, **kwargs)
    finally:
        pipe_split()


def annotate_split_points(mod: torch.nn.Module, spec: dict[str, SplitPoint]):
    # [FUNCTION DOC] Annotates the module in place with split points. For each dotted submodule name
    # in the spec, saves its original forward and replaces it with a wrapper that emits the split
    # marker before or after it. Raises AttributeError if a parent module does not exist and
    # ValueError for an unknown split type.
    # TODO: make this implementation out-of-place?
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        predecessor_module = mod
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise AttributeError(
                    f"Specified target {qualname} referenced "
                    f"nonexistent module {'.'.join(atoms[: i + 1])}"
                ) from e

        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        mod_to_wrap._orig_forward = mod_to_wrap.forward
        if split_type == SplitPoint.BEGINNING:
            mod_to_wrap.forward = MethodType(_split_before_forward, mod_to_wrap)
        elif split_type == SplitPoint.END:
            mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)
        else:
            raise ValueError("Unknown split point type.")


def pipeline(
    module: torch.nn.Module,
    mb_args: tuple[Any, ...],
    mb_kwargs: dict[str, Any] | None = None,
    split_spec: dict[str, SplitPoint] | None = None,
    split_policy: Callable[[fx.GraphModule], fx.GraphModule] | None = None,
) -> Pipe:
    # [FUNCTION DOC] Splits a module into pipeline stages and returns a Pipe. With a split spec,
    # annotates the module and traces it; otherwise traces and uses the optional split policy. Args:
    # module, example micro-batch positional and keyword inputs, split spec (submodule name to
    # SplitPoint), split policy. Raises ValueError if both split spec and split policy are given.
    """
    Split a module based on a specification.

    See `Pipe` for more details.

    Arguments
    ---------
    module:
        The module to be split.
    mb_args:
        Example positional inputs, in micro-batch form.
    mb_kwargs:
        Example keyword inputs, in micro-batch form. (default: `None`)
    split_spec:
        A dictionary using submodule names as split marker. (default: `None`)
    split_policy:
        The policy to use for splitting the module. (default: `None`)

    Returns
    -------
    A pipeline representation of class `Pipe`.
    """
    if split_spec is not None and split_policy is not None:
        raise ValueError(
            "Cannot specify both `split_spec` and `split_policy`. Please use only one of them."
        )

    if split_spec is not None:
        # Annotate split points in the module based on user spec
        annotate_split_points(module, split_spec)
        return Pipe.from_tracing(
            mod=module,
            example_args=mb_args,
            example_kwargs=mb_kwargs,
        )
    else:
        # Use split policy
        return Pipe.from_tracing(
            mod=module,
            example_args=mb_args,
            example_kwargs=mb_kwargs,
            split_policy=split_policy,
        )
